"""Synthetic CPU artifacts for the pure dense-prefix packet audit."""

from copy import deepcopy

import pytest

from mjlab_microduck import stance_friction_prefix_cpu_fixture as fixture
from mjlab_microduck.stance_friction_prefix_packet_audit import (
    PACKET_PROTOCOL,
    PROTOCOL,
    audit_packet,
)


@pytest.fixture(scope="module")
def packet():
    reports = {}
    values = fixture.predeclared_prefix_fixture_values()
    for name, value in values.items():
        reports[name] = fixture.run_prefix_cpu_fixture(
            value["frictionloss"],
            value["qvel"],
            value["invweight"],
            value["solref"],
            value["solimp"],
            value["timestep"],
            value["njmax"],
            value["initial_nefc"],
        )
    helpers = reports["empty"]["prefix_helpers_before"]
    return {
        "protocol": PACKET_PROTOCOL,
        "cases": reports,
        "expected_negative": ["overflow", "maximum"],
        "flags": {
            "full_window_qualified": False,
            "native_qualified": False,
            "physical_acceptance": False,
            "runtime_cause_proven": False,
            "training_authorized": False,
        },
        "caller_binding_before": helpers,
        "caller_binding_after": helpers,
        "held_checker_object_id": helpers["_own_source_binding"]["object_id"],
        "held_checker_code_id": helpers["_own_source_binding"]["code_id"],
        "held_entrypoint_object_id": helpers["run_prefix_cpu_fixture"]["object_id"],
        "held_entrypoint_code_id": helpers["run_prefix_cpu_fixture"]["code_id"],
    }


def _mutate(packet, case="broadcast"):
    changed = deepcopy(packet)
    return changed, changed["cases"][case]


def test_recomputes_ten_case_component_facts_without_admission(packet):
    report = audit_packet(packet)
    assert report["protocol"] == PROTOCOL
    assert report["case_count"] == 10
    assert report["cases"]["broadcast"]["component_exact_without_overflow"] is True
    assert report["cases"]["maximum"]["overflow_negative"] is True
    assert (
        report["cases"]["overflow"]["fixture_decision"]
        == "dense-prefix-negative-or-overflow"
    )
    assert all(value is False for value in report["flags"].values())
    assert report["training_authorized"] is False


def test_signed_zero_is_retained_as_exact_input_and_output_bits(packet):
    report = audit_packet(packet)
    assert report["cases"]["signed-zero"]["component_exact_without_overflow"] is True
    case = packet["cases"]["signed-zero"]
    assert case["input_snapshot"]["qvel"]["u32"][6] == 0x80000000
    assert any(
        row["vel"] == [0x80000000]
        for world in case["worlds"]
        for row in world["candidate0"]["rows"]
    )


def test_accepts_only_the_other_frozen_cpu_platform(packet):
    changed = deepcopy(packet)
    for case in changed["cases"].values():
        for field in ("source_binding_before", "source_binding_after"):
            case[field]["python"] = "3.12.13"
            case[field]["platform"] = {"system": "Linux", "machine": "x86_64"}
    assert audit_packet(changed)["case_count"] == 10
    changed["cases"]["empty"]["source_binding_before"]["python"] = "3.12.14"
    with pytest.raises(ValueError):
        audit_packet(changed)


@pytest.mark.parametrize(
    "edit",
    [
        lambda p, c: c["runs"]["candidate0"]["after"]["nf"]["bits"].__setitem__(0, 99),
        lambda p, c: c["runs"]["candidate0"]["after"]["id"]["bits"].__setitem__(0, 19),
        lambda p, c: c["runs"]["candidate0"]["after"]["J"]["bits"].__setitem__(
            0, 0x3F800000
        ),
        lambda p, c: c["runs"]["candidate0"]["after"]["type"]["bits"].__setitem__(2, 0),
        lambda p, c: c["runs"]["candidate0"]["after"]["pos"]["bits"].__setitem__(
            2, 0x80000000
        ),
        lambda p, c: c["runs"]["candidate0"]["after"]["row_adr"]["bits"].__setitem__(
            0, 1
        ),
        lambda p, c: c["runs"]["candidate0"]["after"]["col_ind"]["bits"].__setitem__(
            0, 1
        ),
        lambda p, c: c["runs"]["candidate0"]["before"]["id"]["bits"].__setitem__(0, 9),
        lambda p, c: c["runs"]["candidate0"]["before"]["J"]["shape"].__setitem__(1, 31),
        lambda p, c: c["runs"]["candidate0"]["after"]["J"]["bits"].pop(),
        lambda p, c: c["input_snapshot"]["qvel"]["u32"].__setitem__(1, 0x3F800000),
    ],
)
def test_rejects_raw_bank_input_or_prefix_mutations(packet, edit):
    changed, case = _mutate(packet)
    edit(changed, case)
    with pytest.raises(ValueError):
        audit_packet(changed)


def test_rejects_changed_replay_and_reported_decisions(packet):
    changed, case = _mutate(packet)
    case["runs"]["candidate1"]["after"]["aref"]["bits"][0] ^= 1
    with pytest.raises(ValueError):
        audit_packet(changed)

    changed, case = _mutate(packet)
    case["candidate_replay_full_bank_bit_identical"] = False
    with pytest.raises(ValueError):
        audit_packet(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["cases"].pop("empty"),
        lambda p: p["cases"].__setitem__("extra", {}),
        lambda p: p["flags"].__setitem__("training_authorized", True),
        lambda p: p["cases"]["broadcast"].__setitem__("training_authorized", True),
        lambda p: p["cases"]["broadcast"].__setitem__(
            "prefix_module_sha256_after", "0" * 64
        ),
        lambda p: p["cases"]["broadcast"]["initial_nefc"]["i32"].__setitem__(0, True),
        lambda p: p["cases"]["broadcast"]["runs"]["original"]["after"][
            "nf"
        ].__setitem__("dtype", "<u4"),
        lambda p: p["cases"]["broadcast"]["worlds"][0].__setitem__(
            "candidate_rows_ascending", "yes"
        ),
        lambda p: p.__setitem__("protocol", "other"),
    ],
)
def test_rejects_schema_pin_and_flag_mutations(packet, mutation):
    changed = deepcopy(packet)
    mutation(changed)
    with pytest.raises(ValueError):
        audit_packet(changed)


@pytest.mark.parametrize(
    "kind",
    ["input-pointer", "caller", "float-id", "platform", "tree-count", "coherent-seed"],
)
def test_rejects_inconsistent_provenance_and_coherent_fixture_changes(packet, kind):
    changed, case = _mutate(packet)
    if kind == "input-pointer":
        for phase in ("inputs_before", "inputs_after"):
            case["runs"]["original"][phase]["frictionloss"]["pointer"] = 7
    elif kind == "caller":
        for phase in ("caller_binding_before", "caller_binding_after"):
            changed[phase]["_as_bits"]["object_id"] += 1
    elif kind == "float-id":
        changed["held_checker_code_id"] = float(changed["held_checker_code_id"])
    elif kind == "platform":
        for phase in ("source_binding_before", "source_binding_after"):
            case[phase]["platform"]["extra"] = "not declared"
    elif kind == "tree-count":
        for phase in ("source_binding_before", "source_binding_after"):
            case[phase]["python_tree_files"] = 69.0
    else:
        for run in case["runs"].values():
            for phase in ("before", "after"):
                run[phase]["row_adr"]["bits"][0] ^= 1
    with pytest.raises(ValueError):
        audit_packet(changed)
