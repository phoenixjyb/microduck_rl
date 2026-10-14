from copy import deepcopy
import math

import pytest

from mjlab_microduck import gait_path_qualification as route
from mjlab_microduck import gait_yaw_trace as gait
from mjlab_microduck import gait_phase_heading_diagnosis as diagnosis


DIGEST = "a" * 64


def quat(heading):
    return [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]


def make_trace(*, start_heading=3.12, speed=.2, yaw=.2, rates=None, clock_dts=None):
    rates = rates or [0.0] * 8
    clock_dts = clock_dts or [.02] * 8
    case = dict(seed=839, speed=speed, yaw=yaw, worlds=8, steps=240, complete=True,
                terminal_worlds=[], checkpoint_sha256=DIGEST, speed_mae=[0.0] * 8,
                yaw_mae=[0.0] * 8, torque_p99=.5, rated_speed_exceed_fraction=0.0,
                response_diagnostics=dict(
                    protocol="signed-body-response-and-joint-load-v1", steps=240,
                    joint_columns=list(route.MOTOR_JOINTS),
                    min_distance_to_hard_stop_rad_by_joint=[.1] * 14,
                    hard_stop_proximity_fraction_by_joint=[0.0] * 14))
    trace = dict(protocol=route.TRACE_PROTOCOL, step_dt=.02, startup_steps=60, steps=240,
                 worlds=8, checkpoint_sha256=DIGEST, timing=gait.TIMING,
                 contact_columns=list(gait.CONTACT_COLUMNS), foot_sites=list(gait.FOOT_SITES),
                 head_joints=list(gait.HEAD_JOINTS), force_timing=gait.FORCE_TIMING,
                 force_frame="world", terrain="stationary-flat", position_frame="world",
                 position_measurement="root-link world position, not integrated velocity",
                 policy_acceptance=False, physical_motion_authorized=False)
    shapes = {"command": 3, "actor_command": 3, "angular_b": 3, "angular_w": 3,
              "quat_wxyz": 4, "episode_length": 1, "physics_time": 1,
              "contact_found": 2, "contact_force_w": (2, 3), "current_air_time": 2,
              "current_contact_time": 2, "foot_velocity_w": (2, 3),
              "head_position_offset": 4, "head_velocity": 4,
              "head_command": 4, "actor_head_command": 4}
    for name in shapes:
        trace[name] = []
    trace["root_position_w"] = []
    for step in range(240):
        times = [step * clock_dts[w] for w in range(8)]
        settled = [max(0, step - 60) * clock_dts[w] for w in range(8)]
        headings = [start_heading + yaw * settled[w] + rates[w] * settled[w]
                    for w in range(8)]
        contacts = []
        for w in range(8):
            # Stable runs exercise all four contact bins; later edits add exact
            # endpoint transitions without depending on simulator-generated data.
            pattern = (step // 4) % 4
            contacts.append(([1, 1], [1, 0], [0, 1], [0, 0])[pattern])
        trace["command"].append([[case["speed"], 0.0, yaw] for _ in range(8)])
        trace["actor_command"].append([[case["speed"], 0.0, yaw] for _ in range(8)])
        trace["angular_b"].append([[0.0, 0.0, yaw] for _ in range(8)])
        trace["angular_w"].append([[0.0, 0.0, yaw] for _ in range(8)])
        trace["quat_wxyz"].append([quat(h) for h in headings])
        trace["episode_length"].append([step] * 8)
        trace["physics_time"].append(times)
        trace["contact_found"].append(contacts)
        trace["contact_force_w"].append([[[0.0, 0.0, 1.0]] * 2 for _ in range(8)])
        trace["current_air_time"].append([[0.0 if x else .02 for x in feet] for feet in contacts])
        trace["current_contact_time"].append([[.02 if x else 0.0 for x in feet] for feet in contacts])
        trace["foot_velocity_w"].append([[[0.0, 0.0, 0.0]] * 2 for _ in range(8)])
        trace["head_position_offset"].append([[0.0] * 4 for _ in range(8)])
        trace["head_velocity"].append([[0.0] * 4 for _ in range(8)])
        trace["head_command"].append([[0.0] * 4 for _ in range(8)])
        trace["actor_head_command"].append([[0.0] * 4 for _ in range(8)])
        trace["root_position_w"].append([[0.0, 0.0, .25] for _ in range(8)])
    return case, trace


def test_phase_partition_transition_and_empty_bins_close_to_route_report():
    case, trace = make_trace(rates=[.1] * 8)
    report = diagnosis.summarize_case(case, trace)
    assert report["interval_steps"] == [60, 239]
    assert report["totals"]["reconciles_full_settled_heading_error"] is True
    assert report["totals"]["pooled_signed_residual_rad"] == pytest.approx(.1 * 3.58 * 8)
    assert report["totals"]["expected_duration_s"] == pytest.approx(3.58)
    for world in report["worlds"]:
        phases = world["phases"]
        assert sum(item["interval_count"] for item in phases.values()) == 179
        assert phases["contact-transition"]["interval_count"] >= 1
        assert all(phases[p]["interval_count"] > 0 for p in
                   ("double-support", "left-only", "right-only", "flight"))
    assert report["policy_acceptance"] is False
    assert report["physical_motion_authorized"] is False


def test_global_yaw_offset_and_wrapping_do_not_change_residuals():
    case, first = make_trace(rates=[.03] * 8)
    _, shifted = make_trace(start_heading=-2.0, rates=[.03] * 8)
    a, b = diagnosis.summarize_case(case, first), diagnosis.summarize_case(case, shifted)
    assert [w["signed_residual_rad"] for w in a["worlds"]] == pytest.approx(
        [w["signed_residual_rad"] for w in b["worlds"]], abs=1e-10)
    assert a["worlds"][0]["signed_residual_rad"] == pytest.approx(.03 * 3.58)


def test_pooled_signed_cancellation_is_distinct_from_mean_world_endpoint_error():
    rates = [.05] * 4 + [-.05] * 4
    case, trace = make_trace(rates=rates)
    report = diagnosis.summarize_case(case, trace)
    assert report["totals"]["pooled_signed_residual_rad"] == pytest.approx(0.0, abs=1e-12)
    assert report["totals"]["mean_per_world_absolute_endpoint_error_rad"] == pytest.approx(.05 * 3.58)
    assert report["totals"]["pooled_absolute_interval_residual_rad"] == pytest.approx(.05 * 3.58 * 8)


def test_unequal_valid_world_clocks_separate_pooled_and_equal_world_rates():
    dts = [.0200009] + [.02] * 7
    case, trace = make_trace(rates=[.5] + [0.0] * 7, clock_dts=dts)
    report = diagnosis.summarize_case(case, trace)
    world_rates = [world["signed_endpoint_rate_rad_s"] for world in report["worlds"]]
    expected_equal = sum(world_rates) / 8
    expected_pooled = (report["totals"]["pooled_signed_residual_rad"] /
                       report["totals"]["pooled_world_duration_s"])
    assert report["timing"]["settled_duration_s_per_world"][0] > 3.58
    assert report["timing"]["settled_duration_s_per_world"][1:] == pytest.approx([3.58] * 7)
    assert report["totals"]["expected_duration_s"] == pytest.approx(3.58)
    assert report["totals"]["equal_world_mean_signed_endpoint_rate_rad_s"] == pytest.approx(
        expected_equal)
    assert report["totals"]["equal_world_mean_absolute_endpoint_rate_rad_s"] == pytest.approx(
        sum(world["absolute_endpoint_rate_rad_s"] for world in report["worlds"]) / 8)
    assert expected_pooled != pytest.approx(expected_equal, abs=1e-7)
    assert all(math.isfinite(value) for world in report["worlds"] for value in
               (world["signed_endpoint_rate_rad_s"], world["absolute_endpoint_rate_rad_s"]))


def test_idle_and_empty_phase_means_are_null_and_inputs_are_unchanged():
    case, trace = make_trace(speed=0.0, yaw=0.0, rates=[0.0] * 8)
    for step in range(60, 240):
        trace["contact_found"][step] = [[0, 0] for _ in range(8)]
    before = deepcopy((case, trace))
    report = diagnosis.summarize_case(case, trace)
    assert trace == before[1] and case == before[0]
    assert report["phases"]["double-support"]["mean_signed_residual_rate_rad_s"] is None
    assert report["phases"]["left-only"]["mean_absolute_residual_rate_rad_s"] is None
    assert all(math.isfinite(value) for world in report["worlds"]
               for value in (world["signed_residual_rad"], world["absolute_residual_rad"]))


@pytest.mark.parametrize("mutation", [
    lambda c, t: t.__setitem__("protocol", "wrong"),
    lambda c, t: t["physics_time"][61].__setitem__(0, t["physics_time"][60][0]),
    lambda c, t: t["quat_wxyz"][70][0].__setitem__(0, float("nan")),
    lambda c, t: c.__setitem__("checkpoint_sha256", "b" * 64),
])
def test_bad_retained_inputs_are_rejected_by_existing_route_validator(mutation):
    case, trace = make_trace()
    mutation(case, trace)
    with pytest.raises((ValueError, TypeError)):
        diagnosis.summarize_case(case, trace)
