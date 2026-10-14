from copy import deepcopy
import math

import pytest

from mjlab_microduck import cycle_heading_diagnosis as diagnosis
from mjlab_microduck import gait_path_qualification as path
from test_gait_path_qualification import make_case_trace, quaternion


def _set_cycle_residual(case, trace, world, values):
    start, end = 70, 110
    assert len(values) == end - start + 1
    start_heading = math.atan2(
        2 * trace["quat_wxyz"][start][world][0] * trace["quat_wxyz"][start][world][3],
        1 - 2 * trace["quat_wxyz"][start][world][3] ** 2)
    start_time = trace["physics_time"][start][world]
    for offset, step in enumerate(range(start, end + 1)):
        elapsed = trace["physics_time"][step][world] - start_time
        trace["quat_wxyz"][step][world] = quaternion(
            start_heading + case["yaw"] * elapsed + values[offset])


def _world_zero_first_cycle(report):
    return report["worlds"][0]["cycles"][0]


def test_perfect_arcs_and_wrapped_start_heading_reconcile_to_route_cycles():
    for speed, yaw, heading in ((.2, .2, 3.12), (.2, -.2, -3.12),
                                (.3, 0.0, math.pi - .01)):
        case, trace = make_case_trace(speed=speed, yaw=yaw, start_heading=heading)
        report = diagnosis.summarize_case(case, trace)
        cycle = _world_zero_first_cycle(report)
        assert cycle["signed_endpoint_residual_rad"] == pytest.approx(0, abs=1e-12)
        assert cycle["max_abs_residual_rad"] == pytest.approx(0, abs=1e-12)
        assert cycle["max_detrended_abs_residual_rad"] == pytest.approx(0, abs=1e-12)
        assert cycle["classification"] == "within-threshold"
        assert report["policy_acceptance"] is False
        assert report["physical_motion_authorized"] is False


def test_linear_bias_is_removed_by_detrending_but_excursion_is_peak_only():
    case, trace = make_case_trace()
    elapsed = [trace["physics_time"][step][0] - trace["physics_time"][70][0]
               for step in range(70, 111)]
    _set_cycle_residual(case, trace, 0, [.11 * t for t in elapsed])
    linear = _world_zero_first_cycle(diagnosis.summarize_case(case, trace))
    assert linear["endpoint_fail"] and linear["max_fail"]
    assert linear["max_detrended_abs_residual_rad"] < 1e-12

    case, trace = make_case_trace()
    residuals = [0.0] * 41
    residuals[10], residuals[20] = .12, -.12
    _set_cycle_residual(case, trace, 0, residuals)
    cycle = _world_zero_first_cycle(diagnosis.summarize_case(case, trace))
    assert not cycle["endpoint_fail"] and cycle["max_fail"] and cycle["peak_only"]
    assert cycle["classification"] == "peak-only"
    assert cycle["peak_step"] == 80  # equal absolute peaks select the earliest sample
    assert cycle["signed_peak_residual_rad"] == pytest.approx(.12)
    assert cycle["max_detrended_abs_residual_rad"] == pytest.approx(.12)


def test_named_head_yaw_peak_values_and_phase_are_reported_without_mutation():
    case, trace = make_case_trace()
    before = deepcopy((case, trace))
    trace["head_position_offset"][80][0][2] = .31
    trace["head_velocity"][80][0][2] = -.42
    residuals = [0.0] * 41
    residuals[10] = .12
    _set_cycle_residual(case, trace, 0, residuals)
    before = deepcopy((case, trace))
    cycle = _world_zero_first_cycle(diagnosis.summarize_case(case, trace))
    assert cycle["head_yaw_offset_at_peak_rad"] == pytest.approx(.31)
    assert cycle["head_yaw_velocity_at_peak_rad_s"] == pytest.approx(-.42)
    assert cycle["peak_interval_phase"] in {
        "double-support", "left-only", "right-only", "flight", "contact-transition"}
    assert (case, trace) == before


def test_idle_has_zero_cycles_and_none_means_and_invalid_trace_is_rejected():
    case, trace = make_case_trace(speed=0.0, yaw=0.0)
    case["yaw_mae"] = [0.0] * 8
    report = diagnosis.summarize_case(case, trace)
    assert report["totals"]["cycle_count"] == 0
    assert report["totals"]["mean_signed_endpoint_residual_rad"] is None
    assert report["totals"]["mean_max_abs_residual_rad"] is None
    assert all(world["cycle_count"] == 0 for world in report["worlds"])
    bad = deepcopy(trace)
    bad["head_joints"] = ["neck_pitch", "head_pitch", "head_roll", "head_yaw"]
    with pytest.raises(ValueError):
        diagnosis.summarize_case(case, bad)
    bad = deepcopy(trace)
    bad["quat_wxyz"][80][0] = [math.sqrt(.5), 0.0, math.sqrt(.5), 0.0]
    with pytest.raises(ValueError, match="projected body-x"):
        diagnosis.summarize_case(case, bad)


def test_threshold_binding_uses_strict_existing_point_one_rate_limit():
    case, trace = make_case_trace()
    elapsed = [trace["physics_time"][step][0] - trace["physics_time"][70][0]
               for step in range(70, 111)]
    _set_cycle_residual(case, trace, 0, [(path.HEADING_ERROR_RAD_S_LIMIT - 1e-9) * t
                                        for t in elapsed])
    cycle = _world_zero_first_cycle(diagnosis.summarize_case(case, trace))
    assert cycle["endpoint_abs_rate_rad_s"] == pytest.approx(.1)
    assert cycle["max_abs_residual_rate_rad_s"] == pytest.approx(.1)
    assert not cycle["endpoint_fail"]
    assert not cycle["max_fail"]
    assert cycle["classification"] == "within-threshold"
    _set_cycle_residual(case, trace, 0, [(path.HEADING_ERROR_RAD_S_LIMIT + 1e-8) * t
                                        for t in elapsed])
    cycle = _world_zero_first_cycle(diagnosis.summarize_case(case, trace))
    assert cycle["endpoint_fail"] and cycle["max_fail"]


def test_actual_per_world_cycle_clock_and_phase_exposure_are_used():
    case, trace = make_case_trace(speed=.3, yaw=0.)
    # Clock origins differ per world; valid increments and physical duration do not.
    for step in range(240):
        for world in range(8):
            trace["physics_time"][step][world] += 10. * world
    report = diagnosis.summarize_case(case, trace)
    for world in report["worlds"]:
        for cycle in world["cycles"]:
            duration = trace["physics_time"][cycle["end_step"]][world["world"]] - trace["physics_time"][cycle["start_step"]][world["world"]]
            assert cycle["duration_s"] == pytest.approx(duration)
            assert sum(cycle["phase_time_s"].values()) == pytest.approx(duration)
