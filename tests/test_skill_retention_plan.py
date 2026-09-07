"""Synthetic planning metadata cannot admit real robot behavior."""

from copy import deepcopy
import hashlib
import json
import sys

import pytest

from mjlab_microduck.skill_compatibility import ARTIFACT_FIELDS, DESCRIPTOR_SCHEMA, STATIC_FIELDS
from mjlab_microduck.skill_retention_plan import (
    CASES, SCHEMA, SEMANTIC_GAPS, SKILLS, TRANSITIONS,
    check_plan, empty_plan, load_plan, main,
)


def ref(root, name, payload=b'{"synthetic":true,"accepted":true}'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def filled(root):
    plan = empty_plan()
    shared = ref(root, "synthetic.json")
    for skill in SKILLS:
        descriptor = dict(schema=DESCRIPTOR_SCHEMA, skill_id=skill, training_source="a" * 40,
                          **{field: dict(shared) for field in ARTIFACT_FIELDS})
        plan["skills"][skill] = dict(descriptor=descriptor, original_protocol=dict(shared),
                                     retention_procedure=dict(shared), retention_report=dict(shared))
    for section, names in (("transitions", TRANSITIONS), ("cases", CASES)):
        plan[section] = {name: dict(procedure=dict(shared), report=dict(shared)) for name in names}
    return plan


def assert_no_authority(result):
    for key in ("original_numerical_gates_verified", "descriptor_to_runtime_binding_verified",
                "behavioral_retention_verified", "safe_stop_verified", "policy_acceptance",
                "transition_authorized", "gpu_collection_authorized", "training_authorized",
                "physical_motion_authorized"):
        assert result[key] is False
    assert result["unresolved_semantic_gates"] == list(SEMANTIC_GAPS)


def test_empty_plan_reports_every_unbound_slot(tmp_path):
    result = check_plan(tmp_path, empty_plan())
    assert result["decision"] == "incomplete-reference-coverage"
    assert len(result["missing_references"]) == 4 * 4 + 12 * 2 + 16 * 2 == 72
    assert result["missing_references"][0] == "skills.foundation.descriptor"
    assert all(value["decision"] == "unknown-missing-descriptor"
               for value in result["declared_static_compatibility"].values())
    assert_no_authority(result)


def test_even_complete_synthetic_accepted_references_are_only_coverage(tmp_path):
    plan = filled(tmp_path)
    before = deepcopy(plan)
    result = check_plan(tmp_path, plan)
    assert result["reference_coverage_complete"] is True
    assert result["decision"] == "reference-coverage-only"
    assert result["missing_references"] == []
    assert all(value["decision"] == "declared-spec-match-only"
               for value in result["declared_static_compatibility"].values())
    assert_no_authority(result)
    assert plan == before
    assert result == check_plan(tmp_path, plan)


@pytest.mark.parametrize("name", TRANSITIONS)
def test_every_directed_transition_needed_independently(tmp_path, name):
    plan = filled(tmp_path)
    del plan["transitions"][name]
    result = check_plan(tmp_path, plan)
    assert result["missing_references"] == [f"transitions.{name}.{field}"
                                            for field in ("procedure", "report")]
    assert_no_authority(result)


@pytest.mark.parametrize("name", CASES)
def test_every_interruption_history_and_sensor_case_required(tmp_path, name):
    plan = filled(tmp_path)
    plan["cases"][name]["report"] = None
    result = check_plan(tmp_path, plan)
    assert result["missing_references"] == [f"cases.{name}.report"]


@pytest.mark.parametrize("field", STATIC_FIELDS)
def test_static_mismatch_not_hidden_by_complete_plan(tmp_path, field):
    plan = filled(tmp_path)
    plan["skills"]["hop"]["descriptor"][field] = ref(tmp_path, "different.json", b"different")
    result = check_plan(tmp_path, plan)
    assert result["reference_coverage_complete"] is True
    for name in ("foundation->hop", "hop->foundation"):
        assert result["declared_static_compatibility"][name] == {
            "decision": "declared-spec-mismatch", "mismatches": [field],
        }
    assert_no_authority(result)


def test_original_protocol_cannot_be_replaced(tmp_path):
    plan = filled(tmp_path)
    plan["skills"]["hop"]["original_protocol"] = ref(tmp_path, "weakened.json", b"new")
    with pytest.raises(ValueError, match="original protocol differs"):
        check_plan(tmp_path, plan)


@pytest.mark.parametrize("section,key", [("skills", "foundation"),
    ("transitions", "hop->foundation"), ("cases", "missing-structured-sensor-and-recovery")])
def test_unknown_acceptance_fields_rejected(tmp_path, section, key):
    plan = filled(tmp_path)
    plan[section][key]["accepted"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        check_plan(tmp_path, plan)


@pytest.mark.parametrize("section", ["skills", "transitions", "cases"])
def test_unknown_catalog_entries_rejected(tmp_path, section):
    plan = empty_plan()
    plan[section]["invented-entry"] = {}
    with pytest.raises(ValueError, match="unknown fields"):
        check_plan(tmp_path, plan)


@pytest.mark.parametrize("damage", ["missing", "bytes", "hash", "symlink", "traversal", "boolean"])
def test_supplied_bad_evidence_is_error_not_missing_gap(tmp_path, damage):
    plan = filled(tmp_path)
    item = plan["cases"][CASES[0]]["report"]
    if damage == "missing":
        (tmp_path / item["path"]).unlink()
    elif damage == "bytes":
        item["bytes"] += 1
    elif damage == "hash":
        item["sha256"] = "0" * 64
    elif damage == "symlink":
        (tmp_path / "alias").symlink_to(tmp_path / item["path"])
        item["path"] = "alias"
    elif damage == "traversal":
        item["path"] = "../synthetic.json"
    else:
        plan["cases"][CASES[0]]["report"] = True
    with pytest.raises(ValueError):
        check_plan(tmp_path, plan)


@pytest.mark.parametrize("payload", [b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":Infinity}',
                                     b'{"x":{"a":1,"a":2}}', b"x" * (1024 * 1024 + 1)])
def test_ambiguous_or_oversized_json_rejected(tmp_path, payload):
    path = tmp_path / "plan.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_plan(path)


def test_cli_exclusive_output_and_deterministic_plan_identity(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    output = tmp_path / "audit.json"
    plan.write_text(json.dumps(empty_plan()))
    monkeypatch.setattr(sys, "argv", ["skill_retention_plan", "--root", str(tmp_path),
                                     "--plan", str(plan), "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert result == check_plan(tmp_path, load_plan(plan))
    assert_no_authority(result)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == before


@pytest.mark.parametrize("plan", [[], None, {}, {"schema": SCHEMA, "skills": [], "transitions": {}, "cases": {}}])
def test_invalid_top_level_rejected(tmp_path, plan):
    with pytest.raises(ValueError):
        check_plan(tmp_path, plan)
