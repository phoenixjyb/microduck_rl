"""Synthetic C1 traces only: no policy, physics, or skill acceptance evidence."""

import copy
import math

import pytest

from mjlab_microduck import community_hop_rehearsal as c1


def synthetic_trace():
    samples = []
    delay = c1.OneStepActionDelay()
    for index in range(c1.TOTAL_SAMPLES):
        t = index * c1.PHYSICS_DT_S
        if index % 4 == 0 and index < c1.TOTAL_SAMPLES - 1:
            raw = [index / 10000.0] * 14
            applied = delay.submit(raw)
        flight = 4.5 <= t < 4.6
        walking = c1.phase_at(t) in ("walk", "resume_walk")
        samples.append({
            "t_s": t, "command": c1.command_at(t),
            "action_raw": raw.copy(), "action_applied": applied.copy(),
            "base_xy_m": [0.0, 0.0],
            "base_velocity_world_m_s": [0.2 if walking else 0.0, 0.0, 0.0],
            "base_roll_pitch_yaw_rad": [0.0] * 3,
            "base_angular_velocity_rad_s": [0.0] * 3,
            "foot_clearance_m": [0.035 if flight else 0.0] * 2,
            "foot_contact": [not flight] * 2,
            "foot_normal_force_n": [0.0 if flight else 5.0] * 2,
            "motor_current_a": [0.3] * 14, "motor_torque_nm": [0.1] * 14,
            "motor_velocity_rad_s": [0.2] * 14,
            "soft_limit_exposed": [False] * 14,
            "body_contact": False, "reset_event": False,
        })
    return {"schema": c1.TRACE_SCHEMA, "samples": samples}


@pytest.fixture
def trace():
    return synthetic_trace()


@pytest.mark.parametrize("time,phase", [(0, "walk"), (2.995, "walk"), (3, "entry_settle"),
    (4, "hop"), (7, "landing_settle"), (9, "resume_walk"), (12, "resume_walk")])
def test_fixed_schedule(time, phase):
    assert c1.phase_at(time) == phase
    assert c1.command_at(time) == ([0.2] + [0.0] * 12 if "walk" in phase else [0.0] * 13)


@pytest.mark.parametrize("bad", [-1, 12.1, float("nan"), float("inf"), True, "1"])
def test_schedule_rejects_invalid_time(bad):
    with pytest.raises(ValueError):
        c1.phase_at(bad)


def test_delay_preserves_raw_history_across_policy_handoff_without_filter_or_clipping():
    delay = c1.OneStepActionDelay()
    walk = [2.0] * 14
    hop = [-3.0] * 14
    assert delay.previous_action == [0.0] * 14
    assert delay.submit(walk) == [0.0] * 14
    assert delay.previous_action == walk
    walk[0] = 42  # Caller cannot mutate retained history.
    assert delay.submit(hop) == [2.0] * 14
    assert delay.submit([0.0] * 14) == hop
    copied = delay.previous_action
    copied[0] = 99
    assert delay.previous_action == [0.0] * 14


def test_bad_action_cannot_damage_delay_state():
    delay = c1.OneStepActionDelay()
    delay.submit([1.0] * 14)
    for bad in ([0.0] * 13, [True] * 14, [float("nan")] * 14):
        with pytest.raises(ValueError):
            delay.submit(bad)
        assert delay.previous_action == [1.0] * 14


def observation_args():
    return dict(angular_velocity=[1.0, 2.0, 3.0], projected_gravity=[0.0, 0.0, -1.0],
                servo_position=[1.0] * 14, backlash_position=[0.01] * 14,
                servo_velocity=[2.0] * 14, backlash_velocity=[0.02] * 14,
                home_position=[0.5] * 14, previous_raw_action=[3.0] * 14,
                command=[0.0] * 13)


def test_observation_output_side_order_no_rescale_or_second_normalizer():
    observation = c1.pack_observation(**observation_args())
    assert len(observation) == 61
    assert observation[:6] == [1.0, 2.0, 3.0, 0.0, 0.0, -1.0]
    assert observation[6:20] == pytest.approx([0.51] * 14)
    assert observation[20:34] == pytest.approx([2.02] * 14)
    assert observation[34:48] == [3.0] * 14
    assert observation[48:] == [0.0] * 13


@pytest.mark.parametrize("field,bad", [("projected_gravity", [0, 0, -9.81]),
    ("servo_position", [0] * 13), ("command", [0] * 3),
    ("backlash_velocity", [float("inf")] * 14), ("home_position", [True] * 14)])
def test_observation_rejects_contract_mismatch(field, bad):
    args = observation_args()
    args[field] = bad
    with pytest.raises(ValueError):
        c1.pack_observation(**args)


def test_synthetic_good_trace_is_never_admission(trace):
    before = copy.deepcopy(trace)
    result = c1.score_trace(trace)
    assert trace == before
    assert result["decision"] == "declared-trace-indicators-pass"
    assert all(result["gates"].values())
    assert result["metrics"]["bilateral_clearance_peak_m"] == 0.035
    assert result["metrics"]["landing_100ms_force_sum_peak_n"] == 10.0
    assert result["metrics"]["mechanical_power_abs_sum_peak_w"] == pytest.approx(0.28)
    assert result["metrics"]["landing_100ms_window_complete"] is True
    for flag in ("impact_limit_calibrated", "thermal_model_verified", "trace_authenticated",
                 "plant_runtime_binding_verified", "behavioral_acceptance", "training_authorized",
                 "transition_authorized", "physical_motion_authorized"):
        assert result[flag] is False


@pytest.mark.parametrize("field,value,reason", [
    ("body_contact", True, "non-foot-body-contact"),
    ("reset_event", True, "reset-event"),
    ("base_roll_pitch_yaw_rad", [math.pi / 3, 0.0, 0.0], "fall-tilt"),
    ("motor_current_a", [float("nan")] * 14, "nonfinite"),
    ("foot_clearance_m", [float("inf")] * 2, "nonfinite"),
    ("command", [0.1] + [0.0] * 12, "command-contract"),
])
def test_first_fatal_event_cannot_be_hidden_by_retry(trace, field, value, reason):
    trace["samples"][100][field] = value
    trace["samples"][101] = {"invalid_after_failure": True}
    result = c1.score_trace(trace)
    assert result["first_failure"] == dict(sample=100, t_s=0.5, reason=reason)
    assert result["scored_samples"] == 100
    assert result["ignored_samples"] == c1.TOTAL_SAMPLES - 101
    assert result["decision"] == "declared-trace-rejected"
    assert result["metrics"]["touchdown_t_s"] is None


@pytest.mark.parametrize("index,field", [(800, "action_applied"), (801, "action_raw"),
    (1200, "action_applied"), (c1.TOTAL_SAMPLES - 1, "action_raw")])
def test_delay_carry_and_substep_hold_are_checked_at_handoff(trace, index, field):
    trace["samples"][index][field] = [99.0] * 14
    result = c1.score_trace(trace)
    assert result["first_failure"]["sample"] == index
    assert result["first_failure"]["reason"] == "one-step-action-delay-or-hold"


def test_empty_partial_and_oversized_attempts_fail_closed(trace):
    with pytest.raises(ValueError, match="bounded"):
        c1.score_trace({"schema": c1.TRACE_SCHEMA, "samples": []})
    result = c1.score_trace({"schema": c1.TRACE_SCHEMA, "samples": trace["samples"][:100]})
    assert result["decision"] == "declared-trace-rejected"
    assert not result["gates"]["complete_first_attempt"]
    trace["samples"].append(trace["samples"][-1])
    with pytest.raises(ValueError, match="bounded"):
        c1.score_trace(trace)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(t_s=0.001),
    lambda r: r.update(body_contact="false"),
    lambda r: r.update(foot_contact=[0, 1]),
    lambda r: r.update(motor_current_a=[True] * 14),
    lambda r: r.update(foot_normal_force_n=[-1, 0]),
    lambda r: r.update(extra=1),
    lambda r: r.pop("soft_limit_exposed"),
])
def test_malformed_trace_is_not_scorable(trace, mutate):
    mutate(trace["samples"][0])
    with pytest.raises(ValueError):
        c1.score_trace(trace)


@pytest.mark.parametrize("bad_flight", ["one-foot", "separate-peaks", "too-low", "too-short", "contact"])
def test_flight_requires_simultaneous_mechanical_clearance_without_contact(trace, bad_flight):
    for row in trace["samples"]:
        if 4.5 <= row["t_s"] < 4.6:
            if bad_flight == "one-foot":
                row["foot_contact"][0] = True
            elif bad_flight == "separate-peaks":
                row["foot_clearance_m"] = [0.035, 0.02] if row["t_s"] < 4.55 else [0.02, 0.035]
            elif bad_flight == "too-low":
                row["foot_clearance_m"] = [0.029] * 2
            elif bad_flight == "too-short" and row["t_s"] >= 4.51:
                row["foot_clearance_m"] = [0.0] * 2
            elif bad_flight == "contact":
                row["foot_contact"] = [True] * 2
    result = c1.score_trace(trace)
    assert not result["gates"]["bilateral_30mm_flight_10ms"]
    assert result["metrics"]["touchdown_t_s"] is None


def test_substep_landing_peak_not_only_first_touchdown_is_retained(trace):
    trace["samples"][930]["foot_normal_force_n"] = [77.0, 83.0]  # 4.65 s, after touchdown.
    result = c1.score_trace(trace)
    assert result["metrics"]["landing_100ms_force_sum_peak_n"] == 160.0
    assert result["impact_limit_calibrated"] is False


@pytest.mark.parametrize("touchdown_index", [920, 921, 923, 927, 939])
def test_impact_window_is_twenty_substeps_despite_float_boundaries(trace, touchdown_index):
    for index, row in enumerate(trace["samples"]):
        if 4.5 <= row["t_s"] and index < touchdown_index:
            row["foot_contact"] = [False] * 2
    result = c1.score_trace(trace)
    assert result["metrics"]["touchdown_t_s"] == touchdown_index * 0.005
    assert result["gates"]["landing_100ms_impact_window_complete"] is True


def test_incomplete_impact_window_rejects_explicitly(trace):
    trace["samples"] = trace["samples"][:930]
    result = c1.score_trace(trace)
    assert result["metrics"]["touchdown_t_s"] is not None
    assert not result["gates"]["landing_100ms_impact_window_complete"]


def test_stable_entry_failure_stops_before_hop_no_retry_scoring(trace):
    trace["samples"][799]["base_velocity_world_m_s"] = [0.051, 0.0, 0.0]
    result = c1.score_trace(trace)
    assert result["first_failure"] == dict(sample=800, t_s=4.0, reason="unstable-hop-entry")
    assert result["metrics"]["qualified_flight_t_s"] is None
    assert not result["gates"]["stable_entry"]


@pytest.mark.parametrize("field", ["motor_torque_nm", "foot_normal_force_n"])
def test_finite_inputs_overflowing_derived_peaks_are_not_passes(trace, field):
    trace["samples"][930][field] = [1e308] * len(trace["samples"][930][field])
    trace["samples"][930]["motor_velocity_rad_s"] = [1e308] * 14
    result = c1.score_trace(trace)
    assert not result["gates"]["finite_derived_peaks"]
    assert result["decision"] == "declared-trace-rejected"


@pytest.mark.parametrize("field,value,gate,start,end", [
    ("base_velocity_world_m_s", [0.06, 0, 0], "stable_entry", 3.9, 4),
    ("base_velocity_world_m_s", [0, 0, 0.06], "stable_entry", 3.9, 4),
    ("base_angular_velocity_rad_s", [0.51, 0, 0], "stable_entry", 3.9, 4),
    ("foot_contact", [True, False], "settled_after_landing", 8.9, 9),
    ("base_xy_m", [0.101, 0], "hop_settle_drift_100mm", 4.5, 4.6),
    ("base_velocity_world_m_s", [0.1, 0, 0], "resume_speed_heading", 11, 12),
    ("base_velocity_world_m_s", [0.2, 0.051, 0], "approach_speed_heading", 2, 3),
    ("base_roll_pitch_yaw_rad", [0, 0, math.radians(16)], "resume_speed_heading", 11, 12),
    ("motor_current_a", [1.751] * 14, "modeled_current_1p75a", 5, 6),
    ("soft_limit_exposed", [True] + [False] * 13, "no_soft_limit_exposure", 5, 6),
])
def test_individual_fixed_diagnostic_gates(trace, field, value, gate, start, end):
    for row in trace["samples"]:
        if start <= row["t_s"] < end:
            row[field] = value
    result = c1.score_trace(trace)
    assert not result["gates"][gate]
    assert result["decision"] == "declared-trace-rejected"


def test_no_landing_is_not_success(trace):
    for row in trace["samples"]:
        if row["t_s"] >= 4.6:
            row["foot_contact"] = [False] * 2
    result = c1.score_trace(trace)
    assert not result["gates"]["landed_after_qualified_flight"]
    assert not result["gates"]["settled_after_landing"]
    assert result["metrics"]["landing_100ms_force_sum_peak_n"] is None


def test_second_qualified_hop_is_not_a_one_shot_and_later_impact_is_retained(trace):
    for row in trace["samples"]:
        if 7.5 <= row["t_s"] < 7.6:
            row["foot_contact"] = [False] * 2
            row["foot_clearance_m"] = [0.035] * 2
    trace["samples"][1530]["foot_normal_force_n"] = [70.0, 80.0]  # 7.65 s.
    result = c1.score_trace(trace)
    assert result["metrics"]["qualified_airborne_episode_count"] == 2
    assert not result["gates"]["exactly_one_qualified_hop"]
    assert result["metrics"]["landing_100ms_force_sum_peak_n"] == 10.0
    assert result["metrics"]["hop_settle_force_sum_peak_n"] == 150.0


def test_clearance_threshold_jitter_without_contact_is_not_a_second_episode(trace):
    trace["samples"][910]["foot_clearance_m"] = [0.029] * 2
    result = c1.score_trace(trace)
    assert result["metrics"]["qualified_airborne_episode_count"] == 1
    assert result["gates"]["exactly_one_qualified_hop"]
