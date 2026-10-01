"""Synthetic walker-only C1-S scorer tests; no policy or physics execution."""

import copy

import pytest

from mjlab_microduck import community_hop_baseline as baseline
from mjlab_microduck import community_hop_rehearsal as c1


@pytest.fixture
def trace():
    samples = []
    delay = c1.OneStepActionDelay()
    raw = applied = [0.0] * 14
    for index in range(c1.TOTAL_SAMPLES):
        t = index * c1.PHYSICS_DT_S
        if index % 4 == 0 and index < c1.TOTAL_SAMPLES - 1:
            applied = delay.submit(raw)
        walking = c1.phase_at(t) in ("walk", "resume_walk")
        samples.append({
            "t_s": t, "command": c1.command_at(t),
            "action_raw": raw.copy(), "action_applied": applied.copy(),
            "base_xy_m": [0.0, 0.0],
            "base_velocity_world_m_s": [0.2 if walking else 0.0, 0.0, 0.0],
            "base_roll_pitch_yaw_rad": [0.0] * 3,
            "base_angular_velocity_rad_s": [0.0] * 3,
            "foot_clearance_m": [0.0, 0.0], "foot_contact": [True, True],
            "foot_normal_force_n": [5.0, 5.0],
            "motor_current_a": [0.3] * 14, "motor_torque_nm": [0.1] * 14,
            "motor_velocity_rad_s": [0.2] * 14,
            "soft_limit_exposed": [False] * 14,
            "body_contact": False, "reset_event": False,
        })
    return {"schema": c1.TRACE_SCHEMA, "samples": samples}


def test_walker_control_passes_without_hop_and_keeps_admission_flags_false(trace):
    before = copy.deepcopy(trace)
    result = baseline.score_walk_only_trace(trace)
    assert trace == before
    assert result["experiment_id"] == "community-hop-c1s-velstand-substitution-v1"
    assert result["measurement_protocol"] == c1.PROTOCOL
    assert result["decision"] == baseline.BASELINE_DECISION
    assert all(result["gates"].values())
    assert result["metrics"]["qualified_airborne_episode_count"] == 0
    assert result["metrics"]["stable_final_zero_command_window_samples"] == 200
    for key, value in result.items():
        if key.endswith("authorized") or key in {
            "impact_limit_calibrated", "thermal_model_verified", "trace_authenticated",
            "plant_runtime_binding_verified", "behavioral_acceptance",
        }:
            assert value is False


def test_qualified_airborne_episode_rejects_walker_control(trace):
    for row in trace["samples"]:
        if 4.5 <= row["t_s"] < 4.6:
            row["foot_contact"] = [False, False]
            row["foot_clearance_m"] = [0.035, 0.035]
    result = baseline.score_walk_only_trace(trace)
    assert result["decision"] == baseline.REJECTED_DECISION
    assert result["metrics"]["qualified_airborne_episode_count"] == 1
    assert not result["gates"]["zero_qualified_airborne_episodes_4_9"]


def test_airborne_episode_after_seven_seconds_still_rejects(trace):
    for row in trace["samples"]:
        if 7.5 <= row["t_s"] < 7.6:
            row["foot_contact"] = [False, False]
            row["foot_clearance_m"] = [0.035, 0.035]
    result = baseline.score_walk_only_trace(trace)
    assert result["metrics"]["qualified_airborne_episode_count"] == 1
    assert result["metrics"]["qualified_flight_t_s"] is None
    assert not result["gates"]["zero_qualified_airborne_episodes_4_9"]


def test_full_final_zero_command_window_must_be_stable(trace):
    trace["samples"][1650]["base_velocity_world_m_s"] = [0.051, 0.0, 0.0]
    result = baseline.score_walk_only_trace(trace)
    assert result["decision"] == baseline.REJECTED_DECISION
    assert not result["gates"]["stable_final_zero_command_window"]
    assert result["first_failure"] == {
        "sample": 1650, "t_s": 8.25, "reason": "stable_final_zero_command_window"
    }


def test_stable_final_window_uses_inclusive_eight_and_exclusive_nine(trace):
    trace["samples"][1600]["base_velocity_world_m_s"] = [0.05, 0.0, 0.0]
    trace["samples"][1799]["base_velocity_world_m_s"] = [0.05, 0.0, 0.0]
    trace["samples"][1800]["base_velocity_world_m_s"] = [0.051, 0.0, 0.0]
    result = baseline.score_walk_only_trace(trace)
    assert result["gates"]["stable_final_zero_command_window"]
    assert result["metrics"]["stable_final_zero_command_window_samples"] == 200


def test_first_fatal_prefix_is_preserved(trace):
    trace["samples"][100]["body_contact"] = True
    trace["samples"][101] = {"invalid_after_failure": True}
    result = baseline.score_walk_only_trace(trace)
    assert result["first_failure"] == {
        "sample": 100, "t_s": 0.5, "reason": "non-foot-body-contact"
    }
    assert result["scored_samples"] == 100
    assert result["ignored_samples"] == c1.TOTAL_SAMPLES - 101
    assert not result["gates"]["complete_first_attempt"]


def test_truncated_trace_fails_completion(trace):
    trace["samples"] = trace["samples"][:100]
    result = baseline.score_walk_only_trace(trace)
    assert result["decision"] == baseline.REJECTED_DECISION
    assert not result["gates"]["complete_first_attempt"]
    assert result["scored_samples"] == 100


def test_overflowing_derived_measurement_fails_closed(trace):
    trace["samples"][930]["motor_torque_nm"] = [1e308] * 14
    trace["samples"][930]["motor_velocity_rad_s"] = [1e308] * 14
    result = baseline.score_walk_only_trace(trace)
    assert result["decision"] == baseline.REJECTED_DECISION
    assert not result["gates"]["finite_derived_peaks"]


@pytest.mark.parametrize("mutate,gate", [
    (lambda rows: rows[100].update(body_contact=True), "no_fatal_event"),
    (lambda rows: rows.__setitem__(slice(400, 600), [
        {**row, "base_velocity_world_m_s": [0.2, 0.051, 0.0]}
        for row in rows[400:600]
    ]), "approach_speed_heading"),
    (lambda rows: rows[799].update(base_velocity_world_m_s=[0.051, 0.0, 0.0]), "stable_entry"),
    (lambda rows: [row.update(base_xy_m=[0.101, 0.0]) for row in rows[801:1000]],
     "hop_settle_drift_100mm"),
    (lambda rows: [row.update(base_velocity_world_m_s=[0.1, 0.0, 0.0])
                   for row in rows[2200:2400]], "resume_speed_heading"),
    (lambda rows: [row.update(motor_current_a=[1.751] * 14) for row in rows[1000:1200]],
     "modeled_current_1p75a"),
    (lambda rows: [row.update(soft_limit_exposed=[True] + [False] * 13)
                   for row in rows[1000:1200]], "no_soft_limit_exposure"),
    (lambda rows: (rows[930].update(motor_torque_nm=[1e308] * 14),
                   rows[930].update(motor_velocity_rad_s=[1e308] * 14)),
     "finite_derived_peaks"),
])
def test_each_shared_gate_failure_is_reported(trace, mutate, gate):
    mutate(trace["samples"])
    result = baseline.score_walk_only_trace(trace)
    assert not result["gates"][gate]
    assert result["decision"] == baseline.REJECTED_DECISION


def test_nonfinite_fatal_row_is_retained_as_first_failure(trace):
    trace["samples"][100]["motor_current_a"] = [float("nan")] * 14
    result = baseline.score_walk_only_trace(trace)
    assert result["first_failure"] == {"sample": 100, "t_s": 0.5, "reason": "nonfinite"}
    assert result["scored_samples"] == 100


def test_invalid_trace_schema_is_rejected(trace):
    trace["extra"] = "not in the declared schema"
    with pytest.raises(ValueError, match="exact C1 trace schema"):
        baseline.score_walk_only_trace(trace)


def test_action_delay_contract_error_is_rejected(trace):
    trace["samples"][4]["action_applied"] = [1.0] * 14
    result = baseline.score_walk_only_trace(trace)
    assert result["first_failure"] == {
        "sample": 4, "t_s": 0.02, "reason": "one-step-action-delay-or-hold"
    }
    assert not result["gates"]["no_fatal_event"]
