"""Synthetic CPU fixtures are metadata checks, never robot skill evidence."""

from copy import deepcopy
import hashlib
import json

import pytest

from mjlab_microduck.skill_compatibility import (
    ARTIFACT_FIELDS, DESCRIPTOR_SCHEMA, STATIC_FIELDS,
    compare_descriptors, verify_descriptor,
)


def reference(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def descriptor(root):
    result = dict(schema=DESCRIPTOR_SCHEMA, skill_id="synthetic-walk", training_source="a" * 40)
    for field in ARTIFACT_FIELDS:
        result[field] = reference(root, field, json.dumps({"synthetic": field}).encode())
    return result


def test_identical_bytes_never_admit_policy_or_transition(tmp_path):
    d = descriptor(tmp_path)
    before = deepcopy(d)
    result = compare_descriptors(tmp_path, d, tmp_path, d)
    assert result["decision"] == "declared-spec-match-only"
    assert result["mismatches"] == []
    assert result["referenced_bytes_verified"] is True
    for field in ("descriptor_to_runtime_binding_verified", "behavioral_retention_verified",
                  "policy_acceptance", "transition_authorized", "physical_motion_authorized"):
        assert result[field] is False
    assert d == before
    assert result == compare_descriptors(tmp_path, d, tmp_path, d)


@pytest.mark.parametrize("field", STATIC_FIELDS)
def test_every_static_axis_is_required_even_with_equal_dimensions(tmp_path, field):
    left = descriptor(tmp_path)
    right = deepcopy(left)
    right[field] = reference(tmp_path, f"changed/{field}", b'{"same_action_dim":14,"changed":true}')
    result = compare_descriptors(tmp_path, left, tmp_path, right)
    assert result["decision"] == "declared-spec-mismatch"
    assert result["mismatches"] == [field]


def test_distinct_weights_and_protocol_are_not_static_mechanics_mismatches(tmp_path):
    left = descriptor(tmp_path)
    right = deepcopy(left)
    right["skill_id"] = "synthetic-hop-rejected"
    right["training_source"] = "b" * 40
    for field in ("checkpoint", "evaluation_protocol", "evaluation_report"):
        right[field] = reference(tmp_path, f"right/{field}", b'{"policy_acceptance":true}')
    result = compare_descriptors(tmp_path, left, tmp_path, right)
    assert result["decision"] == "declared-spec-match-only"
    assert result["policy_acceptance"] is False  # contents are not trusted/interpreted
    assert result["behavioral_retention_verified"] is False


def test_mirror_path_does_not_change_identity(tmp_path):
    left = descriptor(tmp_path / "left")
    right = descriptor(tmp_path / "right")
    for field in ARTIFACT_FIELDS:
        content = (tmp_path / "right" / field).read_bytes()
        right[field] = reference(tmp_path / "right", f"mirror/{field}", content)
    assert compare_descriptors(tmp_path / "left", left, tmp_path / "right", right)["mismatches"] == []


@pytest.mark.parametrize("field", ARTIFACT_FIELDS)
@pytest.mark.parametrize("damage", ["missing", "bytes", "same-size"])
def test_every_artifact_is_verified_before_comparison(tmp_path, field, damage):
    d = descriptor(tmp_path)
    path = tmp_path / d[field]["path"]
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        path.write_bytes(b"longer damaged content than the original reference")
    else:
        path.write_bytes(b"x" * d[field]["bytes"])
    with pytest.raises(ValueError, match=field):
        compare_descriptors(tmp_path, d, tmp_path, d)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a//b", "./checkpoint", ".", "", "a\\b", "a\x00b"])
def test_unsafe_or_noncanonical_path_rejected(tmp_path, path):
    d = descriptor(tmp_path)
    d["checkpoint"]["path"] = path
    with pytest.raises(ValueError, match="checkpoint"):
        verify_descriptor(tmp_path, d)


@pytest.mark.parametrize("link_directory", [False, True])
def test_symlink_rejected_even_if_bytes_match(tmp_path, link_directory):
    d = descriptor(tmp_path)
    if link_directory:
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        d["checkpoint"]["path"] = "alias/checkpoint"
    else:
        (tmp_path / "alias").symlink_to(tmp_path / "checkpoint")
        d["checkpoint"]["path"] = "alias"
    with pytest.raises(ValueError, match="symlink"):
        verify_descriptor(tmp_path, d)


@pytest.mark.parametrize("key,value", [("bytes", True), ("bytes", 0), ("bytes", 1.0),
                                      ("sha256", "unknown"), ("sha256", "A" * 64),
                                      ("path", None)])
def test_reference_schema_fail_closed(tmp_path, key, value):
    d = descriptor(tmp_path)
    d["checkpoint"][key] = value
    with pytest.raises(ValueError):
        verify_descriptor(tmp_path, d)


@pytest.mark.parametrize("mutation", ["extra", "missing", "schema", "source", "id", "reference-extra"])
def test_descriptor_schema_fail_closed(tmp_path, mutation):
    d = descriptor(tmp_path)
    if mutation == "extra":
        d["accepted"] = True
    elif mutation == "missing":
        del d["mechanics"]
    elif mutation == "schema":
        d["schema"] = "skill-descriptor-v2"
    elif mutation == "source":
        d["training_source"] = "2614d09"
    elif mutation == "id":
        d["skill_id"] = "ambiguous skill"
    else:
        d["checkpoint"]["accepted"] = True
    with pytest.raises(ValueError):
        verify_descriptor(tmp_path, d)


def test_all_differences_reported_in_fixed_order(tmp_path):
    left = descriptor(tmp_path)
    right = deepcopy(left)
    for field in reversed(STATIC_FIELDS):
        right[field] = reference(tmp_path, f"new/{field}", b"different")
    assert compare_descriptors(tmp_path, left, tmp_path, right)["mismatches"] == list(STATIC_FIELDS)
