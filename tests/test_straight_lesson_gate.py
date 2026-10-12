import hashlib
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from mjlab_microduck import integrated_training_budget_refinement as foundation
from mjlab_microduck import integrated_joint_limit_refinement as jointlimit
from mjlab_microduck import integrated_straight_lesson as recipe
from mjlab_microduck import straight_lesson_gate as gate


def row(seed, speed, yaw, checkpoint, *, complete=True):
    names = [f"joint_{index}" for index in range(14)]
    names[9] = jointlimit.JOINT
    return dict(seed=seed, speed=speed, yaw=yaw, checkpoint_sha256=checkpoint,
        complete=complete, worlds=8, steps=240,
        terminal_worlds=[] if complete else [3],
        speed_mae=[.03 if speed == 0 else .10] * 8 if complete else None,
        yaw_mae=[.10] * 8 if complete else None,
        torque_p99=.60, rated_speed_exceed_fraction=0.,
        response_diagnostics=dict(protocol="signed-body-response-and-joint-load-v1",
            joint_columns=names, steps=240, hard_stop_margin_rad=.05,
            min_distance_to_hard_stop_rad_by_joint=[-.005] + [.1] * 13,
            hard_stop_proximity_fraction_by_joint=[0.] * 14))


def aggregate():
    hashes = {"control": "a" * 64, "straight": "b" * 64}
    cases = {arm: [row(seed, speed, yaw, hashes[arm])
                   for seed in foundation.base.SEEDS
                   for speed, yaw in foundation.base.CASES]
             for arm in gate.ARMS}
    decisions = {arm: foundation.decision(cases[arm]) for arm in gate.ARMS}
    return dict(protocol=recipe.PROTOCOL, source="f" * 40,
                selection="both final999 only; unchanged held-out protocol",
                cases=cases, checkpoint_sha256=hashes, decisions=decisions,
                policy_acceptance=False, simulator_qualified=False,
                physical_motion_authorized=False)


def refresh_decisions(value):
    value["decisions"] = {arm: foundation.decision(value["cases"][arm]) for arm in gate.ARMS}


def test_exact_thresholds_pass_and_do_not_claim_acceptance_or_authority():
    result = gate.evaluate(aggregate())
    assert result["protocol"] == "straight-lesson-stage-exit-v1"
    assert result["source"] == "f" * 40
    assert result["input_protocol"] == recipe.PROTOCOL
    assert result["selection"] == "both final999 only; unchanged held-out protocol"
    assert result["decision"] == "ready-for-declared-turn-lesson"
    assert result["stage_exit"] and result["arc_safety"] and result["arc_retention"]
    assert result["full_foundation_decisions"]["straight"] == foundation.decision(
        aggregate()["cases"]["straight"])
    assert result["policy_acceptance"] is False
    assert result["simulator_qualified"] is False
    assert result["physical_motion_authorized"] is False


@pytest.mark.parametrize("damage,match", [
    ("missing", "complete ordered evaluation matrix"),
    ("duplicate", "exact seed/case order"),
    ("hash", "distinct control and straight checkpoint hashes"),
    ("row_hash", "checkpoint hash matches aggregate"),
    ("worlds", "fixed world and step shape"),
    ("nan", "finite nonnegative speed_mae coverage"),
])
def test_malformed_incomplete_hash_and_nonfinite_inputs_fail_closed(damage, match):
    value = aggregate()
    if damage == "missing":
        value["cases"]["straight"].pop()
    elif damage == "duplicate":
        value["cases"]["straight"][1] = deepcopy(value["cases"]["straight"][0])
    elif damage == "hash":
        value["checkpoint_sha256"]["straight"] = value["checkpoint_sha256"]["control"]
        for item in value["cases"]["straight"]:
            item["checkpoint_sha256"] = value["checkpoint_sha256"]["straight"]
        refresh_decisions(value)
    elif damage == "row_hash":
        value["cases"]["straight"][0]["checkpoint_sha256"] = "c" * 64
    elif damage == "worlds":
        value["cases"]["straight"][0]["worlds"] = 7
    else:
        value["cases"]["straight"][0]["speed_mae"][0] = float("nan")
        value["decisions"]["straight"] = None
    with pytest.raises((ValueError, TypeError), match=match):
        gate.evaluate(value)


def test_incomplete_case_is_a_canonical_failure_without_speed_none_comparison():
    value = aggregate()
    incomplete = value["cases"]["straight"][0]
    incomplete.update(complete=False, speed_mae=None, yaw_mae=None, terminal_worlds=[3])
    refresh_decisions(value)
    result = gate.evaluate(value)
    assert result["decision"] == "straight-lesson-not-ready"
    assert result["stage_exit_failures"][0]["metric"] == "complete"
    assert result["full_foundation_decisions"]["straight"]["failures"][0][-1] == "first-episode-terminal"


def test_single_world_tracking_failure_has_seed_case_world_metric_evidence():
    value = aggregate()
    value["cases"]["straight"][1]["speed_mae"][6] = .100001
    refresh_decisions(value)
    result = gate.evaluate(value)
    evidence = result["stage_exit_failures"][0]
    assert (evidence["seed"], evidence["case"], evidence["world"], evidence["metric"]) == (
        foundation.base.SEEDS[0], "straight", 6, "speed_mae")


def test_arc_safety_and_exact_paired_world_retention_are_both_required():
    value = aggregate()
    arc_index = 2
    value["cases"]["straight"][arc_index]["torque_p99"] = .600001
    value["cases"]["straight"][arc_index]["yaw_mae"][4] = .100001
    value["cases"]["straight"][arc_index + 1]["speed_mae"][2] = .100001
    refresh_decisions(value)
    result = gate.evaluate(value)
    assert not result["arc_safety"] and not result["arc_retention"]
    assert any(item["metric"] == "torque_p99" for item in result["arc_safety_failures"])
    assert any(item["metric"] == "speed_mae_vs_control" and item["world"] == 2
               for item in result["arc_retention_failures"])


def test_control_foundation_failure_does_not_replace_straight_stage_predicate():
    value = aggregate()
    # Canonical control gate fails in idle; the straight treatment is judged on
    # its own original gates and the two exact arc-retention comparisons.
    value["cases"]["control"][0]["speed_mae"][0] = .030001
    refresh_decisions(value)
    result = gate.evaluate(value)
    assert result["full_foundation_decisions"]["control"]["decision"] == "foundation-not-ready"
    assert result["decision"] == "ready-for-declared-turn-lesson"


def test_retained_arc_tracking_is_not_full_foundation_acceptance():
    value = aggregate()
    for arm in gate.ARMS:
        value["cases"][arm][2]["yaw_mae"] = [.15] * 8
    refresh_decisions(value)
    result = gate.evaluate(value)
    assert result["arc_safety"] and result["arc_retention"] and result["stage_exit"]
    assert result["decision"] == "ready-for-declared-turn-lesson"
    assert result["full_foundation_decisions"]["straight"]["decision"] == "foundation-not-ready"


def test_incomplete_reference_arc_cannot_pass_retention():
    value = aggregate()
    value["cases"]["control"][2].update(complete=False, terminal_worlds=[0],
                                        speed_mae=None, yaw_mae=None)
    refresh_decisions(value)
    result = gate.evaluate(value)
    assert not result["arc_retention"] and result["arc_safety"]
    assert result["arc_retention_failures"][0]["metric"] == "paired_complete"
    assert result["decision"] == "straight-lesson-not-ready"


@pytest.mark.parametrize("field,bad", [("source", "wrong"), ("protocol", "wrong"),
    ("selection", "best checkpoint"), ("policy_acceptance", True),
    ("simulator_qualified", True), ("physical_motion_authorized", True)])
def test_provenance_or_authority_cannot_be_substituted(field, bad):
    value = aggregate()
    value[field] = bad
    with pytest.raises(ValueError): gate.evaluate(value)


@pytest.mark.parametrize("case_index", [1, 2])
def test_named_range_gate_is_retained_for_straight_and_arc(case_index):
    value = aggregate()
    d = value["cases"]["straight"][case_index]["response_diagnostics"]
    idx = d["joint_columns"].index(jointlimit.JOINT)
    d["hard_stop_proximity_fraction_by_joint"][idx] = .010001
    refresh_decisions(value)
    result = gate.evaluate(value)
    evidence = result["stage_exit_failures"] if case_index == 1 else result["arc_safety_failures"]
    assert any(item["metric"] == f"hard_stop_proximity_fraction:{jointlimit.JOINT}" for item in evidence)
    assert result["decision"] == "straight-lesson-not-ready"


def test_cli_hashes_input_and_writes_new_output(tmp_path):
    source = tmp_path / "aggregate.json"
    output = tmp_path / "gate.json"
    source.write_text(json.dumps(aggregate()))
    subprocess.run([sys.executable, "-m", "mjlab_microduck.straight_lesson_gate",
                    "--input", str(source), "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["decision"] == "ready-for-declared-turn-lesson"
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, "-m", "mjlab_microduck.straight_lesson_gate",
                        "--input", str(source), "--output", str(output)], check=True)
