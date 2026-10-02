"""Pure contract checks for the isolated single-substep force fixture."""

import copy
import hashlib
import json

import pytest

from mjlab_microduck import stance_disturbance_contract as contract
from mjlab_microduck import stance_lesson_contract as nominal


SOURCE = "a" * 40
PLANT = {
    "nbody": 4,
    "body_names": ["world", "pelvis", "trunk_base", "left_foot"],
    "body_name": "trunk_base",
    "body_id": 2,
    "selected_fields_sha256": "b" * 64,
}


def test_plan_is_source_and_full_plant_bound_and_non_admitting():
    result = contract.plan(SOURCE, PLANT)
    encoded = json.dumps(PLANT, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert result["protocol"] == "football-b1d-single-substep-force-fixture-v1"
    assert result["plant"] == PLANT
    assert result["plant_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result["worlds"] == 1 and (result["nq"], result["nv"], result["nu"]) == (21, 20, 14)
    assert result["dt"] == 0.002
    assert result["cases"] == [{"name": name, "wrench": list(wrench)}
                                for name, wrench in contract.CASES]
    assert contract.CASES == (
        ("zero-wrench", (0, 0, 0, 0, 0, 0)),
        ("+x", (2, 0, 0, 0, 0, 0)),
        ("-x", (-2, 0, 0, 0, 0, 0)),
        ("+y", (0, 2, 0, 0, 0, 0)),
        ("-y", (0, -2, 0, 0, 0, 0)),
    )
    assert (contract.FRAME, contract.ORDER, contract.APPLICATION) == (
        "world", "force-then-torque", "body-inertial-com")
    assert (result["pre_forward_forced_solves_per_case"], result["euler_steps_per_case"],
            result["post_forward_unforced_captures_per_case"]) == (1, 1, 1)
    assert result["clear_all_applied_forces_after_euler"] is True
    assert result["same_reset_and_prepared_motor_fields_each_case"] is True
    assert result["zero_error_bam_motor_fields"] is result["frozen_ctrl"] is True
    assert result["force_fixture_only"] is True
    assert result["policy_inferences"] == result["optimizer_steps"] == 0
    assert all(result[key] is False for key in ("actor", "actions", "checkpoint", "perception"))
    assert all(result[key] is False for key in contract.NO_ADMISSION)
    assert nominal.PROTOCOL == "football-b1n-nominal-stance-v1"


@pytest.mark.parametrize("source", ["A" * 40, "a" * 39, "g" * 40, True, None])
def test_plan_rejects_invalid_source_revision(source):
    with pytest.raises(ValueError):
        contract.plan(source, PLANT)


@pytest.mark.parametrize("change", [
    {"nbody": True},
    {"nbody": 3},
    {"body_names": ["world", "trunk_base", "trunk_base", "foot"]},
    {"body_names": ["world", "pelvis", "other", "foot"]},
    {"body_name": "pelvis"},
    {"body_id": 0},
    {"body_id": True},
    {"body_id": 4},
    {"extra": float("nan")},
])
def test_plan_rejects_bad_or_ambiguous_compiled_body_binding(change):
    plant = dict(PLANT)
    plant.update(change)
    with pytest.raises(ValueError):
        contract.plan(SOURCE, plant)


@pytest.mark.parametrize("case,wrench", contract.CASES)
def test_expected_wrench_uses_compiled_body_id_and_zero_generalized_force(case, wrench):
    xfrc, qfrc = contract.expected_wrench(case, 4, 2)
    assert len(xfrc) == 1 and len(xfrc[0]) == 4
    assert xfrc[0] == [[0] * 6, [0] * 6, list(wrench), [0] * 6]
    assert qfrc == [[0] * 20]
    assert contract.validate_applied(xfrc, qfrc, case, 4, 2) is True


def test_validate_applied_rejects_wrong_body_or_any_extra_component():
    xfrc, qfrc = contract.expected_wrench("+x", 4, 2)
    variants = []
    wrong_body = copy.deepcopy(xfrc); wrong_body[0][1][0] = 2; variants.append(wrong_body)
    wrong_component = copy.deepcopy(xfrc); wrong_component[0][2][3] = 2; variants.append(wrong_component)
    wrong_torque = copy.deepcopy(xfrc); wrong_torque[0][2][5] = 1; variants.append(wrong_torque)
    for changed in variants:
        with pytest.raises(ValueError):
            contract.validate_applied(changed, qfrc, "+x", 4, 2)
    generalized = copy.deepcopy(qfrc); generalized[0][0] = 1
    with pytest.raises(ValueError, match="qfrc_applied"):
        contract.validate_applied(xfrc, generalized, "+x", 4, 2)


@pytest.mark.parametrize("bad", [
    True, float("nan"), float("inf"), "0", None,
])
def test_validate_applied_rejects_bad_scalar_dtypes_and_nonfinite_values(bad):
    xfrc, qfrc = contract.expected_wrench("zero-wrench", 4, 2)
    xfrc[0][0][0] = bad
    with pytest.raises(ValueError):
        contract.validate_applied(xfrc, qfrc, "zero-wrench", 4, 2)


@pytest.mark.parametrize("damage", ["worlds", "bodies", "components", "qfrc", "case", "binding"])
def test_validate_applied_rejects_malformed_shape_or_binding(damage):
    xfrc, qfrc = contract.expected_wrench("zero-wrench", 4, 2)
    nbody, body_id, case = 4, 2, "zero-wrench"
    if damage == "worlds": xfrc.append(copy.deepcopy(xfrc[0]))
    if damage == "bodies": xfrc[0].pop()
    if damage == "components": xfrc[0][0].pop()
    if damage == "qfrc": qfrc[0].pop()
    if damage == "case": case = "diagonal"
    if damage == "binding": body_id = 0
    with pytest.raises(ValueError):
        contract.validate_applied(xfrc, qfrc, case, nbody, body_id)
