from copy import deepcopy
import math

import pytest

from mjlab_microduck import gait_path_qualification as path
from mjlab_microduck import gait_yaw_trace as gait


DIGEST = "a" * 64


def quaternion(heading):
    return [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]


def make_case_trace(*, speed=.2, yaw=.2, start_heading=0.0, seed=839):
    case = dict(seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240,
                complete=True, terminal_worlds=[], checkpoint_sha256=DIGEST,
                speed_mae=[0.0] * 8, yaw_mae=[0.0] * 8, torque_p99=.5,
                rated_speed_exceed_fraction=0.0,
                response_diagnostics=dict(
                    protocol="signed-body-response-and-joint-load-v1", steps=240,
                    joint_columns=list(path.MOTOR_JOINTS),
                    min_distance_to_hard_stop_rad_by_joint=[.1] * 14,
                    hard_stop_proximity_fraction_by_joint=[0.0] * 14))
    trace = dict(
        protocol=path.TRACE_PROTOCOL,
        step_dt=.02, startup_steps=60, steps=240, worlds=8,
        checkpoint_sha256=DIGEST, timing=gait.TIMING,
        contact_columns=list(gait.CONTACT_COLUMNS), foot_sites=list(gait.FOOT_SITES),
        head_joints=list(gait.HEAD_JOINTS), force_timing=gait.FORCE_TIMING,
        force_frame="world", terrain="stationary-flat",
        position_frame="world",
        position_measurement="root-link world position, not integrated velocity",
        policy_acceptance=False, physical_motion_authorized=False,
        command=[], actor_command=[], angular_b=[], angular_w=[], quat_wxyz=[],
        episode_length=[], physics_time=[], contact_found=[], contact_force_w=[],
        current_air_time=[], current_contact_time=[], foot_velocity_w=[],
        head_position_offset=[], head_velocity=[], head_command=[], actor_head_command=[],
        root_position_w=[])
    for step in range(240):
        t = step * .02
        settled_t = t - 60 * .02
        h = start_heading + yaw * settled_t
        local_x, local_y = path._arc_displacement(speed, yaw, settled_t)
        x = math.cos(start_heading) * local_x - math.sin(start_heading) * local_y
        y = math.sin(start_heading) * local_x + math.cos(start_heading) * local_y
        position = [x, y, 0.25]
        if step < 60:
            position = [0.0, 0.0, 0.25]
            h = start_heading
        twist = [speed, 0.0, yaw]
        trace["command"].append([twist[:] for _ in range(8)])
        trace["actor_command"].append([twist[:] for _ in range(8)])
        trace["angular_b"].append([[0.0, 0.0, yaw] for _ in range(8)])
        trace["angular_w"].append([[0.0, 0.0, yaw] for _ in range(8)])
        trace["quat_wxyz"].append([quaternion(h) for _ in range(8)])
        trace["episode_length"].append([step] * 8)
        trace["physics_time"].append([t] * 8)
        found_worlds, air_worlds, contact_time_worlds = [], [], []
        for world in range(8):
            feet = []
            for foot in range(2):
                cycle = 40
                offset = 0 if foot == 0 else 20
                phase = (step - 70 - offset) % cycle
                touching = step >= 70 + offset and phase < 8
                feet.append(1 if touching else 0)
            found_worlds.append(feet)
            air_worlds.append([0.0 if found else .02 for found in feet])
            contact_time_worlds.append([.02 if found else 0.0 for found in feet])
        trace["contact_found"].append(found_worlds)
        trace["contact_force_w"].append([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]] for _ in range(8)])
        trace["current_air_time"].append(air_worlds)
        trace["current_contact_time"].append(contact_time_worlds)
        trace["foot_velocity_w"].append([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in range(8)])
        trace["head_position_offset"].append([[0.0] * 4 for _ in range(8)])
        trace["head_velocity"].append([[0.0] * 4 for _ in range(8)])
        trace["head_command"].append([[0.0] * 4 for _ in range(8)])
        trace["actor_head_command"].append([[0.0] * 4 for _ in range(8)])
        trace["root_position_w"].append([position[:] for _ in range(8)])
    # Preserve the legacy instantaneous-yaw diagnostic while letting it differ
    # from the projected route heading for the yaw-independence test.
    return case, trace


def test_analytic_straight_and_signed_arc_endpoints_with_wrapped_start_heading():
    for speed, yaw, heading in ((.3, 0.0, 0.0), (.2, .2, 3.12), (.2, -.2, -3.12)):
        case, trace = make_case_trace(speed=speed, yaw=yaw, start_heading=heading)
        before = deepcopy(trace)
        report = path.summarize_case(case, trace)
        assert report["status"] == "route-qualified-simulation-only"
        assert all(item["cycle_count"] >= 3 for item in report["worlds"])
        assert all(item["valid_cycle_count"] == item["cycle_count"] for item in report["worlds"])
        assert all(item["full_settled"]["xy_error_mps"] == pytest.approx(0.0, abs=1e-12)
                   for item in report["worlds"])
        assert trace == before


def test_large_oscillatory_instantaneous_yaw_stays_diagnostic_when_path_is_good():
    case, trace = make_case_trace()
    residuals = []
    for step in range(240):
        value = .8 if step % 2 else -.8
        for world in range(8):
            trace["angular_b"][step][world][2] = value
            trace["angular_w"][step][world][2] = value
        if step >= 60:
            residuals.append(abs(value - case["yaw"]))
    case["yaw_mae"] = [sum(residuals) / len(residuals)] * 8
    report = path.summarize_case(case, trace)
    assert report["status"] == "route-qualified-simulation-only"
    assert report["original_yaw_mae_diagnostic"] == case["yaw_mae"]


def test_idle_is_not_counted_as_gait_and_endpoint_drift_still_fails():
    case, trace = make_case_trace(speed=0.0, yaw=0.0)
    case["yaw_mae"] = [0.0] * 8
    report = path.summarize_case(case, trace)
    assert report["status"] == "route-qualified-simulation-only"
    assert {item["cycle_status"] for item in report["worlds"]} == {
        "not-applicable-stationary-command"}
    trace["root_position_w"][239][0][0] += .2
    report = path.summarize_case(case, trace)
    assert report["status"] == "route-not-qualified"
    assert "world-0:full-settled-xy-error-rate" in report["failures"]


def test_cycle_endpoint_and_heading_rate_errors_are_not_hidden_by_full_window():
    case, trace = make_case_trace()
    trace["root_position_w"][110][0][0] += .2
    report = path.summarize_case(case, trace)
    assert report["status"] == "route-not-qualified"
    assert any("cycle-xy-error-rate" in failure for failure in report["failures"])
    case, trace = make_case_trace()
    for step in range(61, 240):
        heading = .2 * (step - 60) * .02 + .5
        trace["quat_wxyz"][step][0] = quaternion(heading)
    report = path.summarize_case(case, trace)
    assert report["status"] == "route-not-qualified"
    assert "world-0:full-settled-heading-error-rate" in report["failures"]


def test_full_turn_cannot_hide_behind_identical_wrapped_heading_endpoint():
    case, trace = make_case_trace()
    for step in range(60, 240):
        settled_t = (step - 60) * .02
        heading = case["yaw"] * settled_t + 2 * math.pi * settled_t / 3.58
        trace["quat_wxyz"][step][0] = quaternion(heading)
    report = path.summarize_case(case, trace)
    full = report["worlds"][0]["full_settled"]
    assert full["heading_delta_wrapped_rad"] == pytest.approx(.2 * 3.58)
    assert full["heading_delta_rad"] == pytest.approx(.2 * 3.58 + 2 * math.pi)
    assert full["max_heading_error_rad_s"] > path.HEADING_ERROR_RAD_S_LIMIT
    assert "world-0:full-settled-heading-error-rate" in report["failures"]


def test_out_and_back_excursion_fails_trajectory_max_with_good_cycle_endpoints():
    case, trace = make_case_trace()
    expected = trace["root_position_w"][90][0][0]
    trace["root_position_w"][90][0][0] += .5
    trace["root_position_w"][91][0][0] = expected
    for step in range(90, 96):
        trace["quat_wxyz"][step][0] = quaternion(case["yaw"] * (step - 60) * .02 + .2)
    report = path.summarize_case(case, trace)
    world = report["worlds"][0]
    assert world["cycles"][0]["xy_error_mps"] == pytest.approx(0.0, abs=1e-12)
    assert world["cycles"][0]["max_xy_error_mps"] > path.MOVING_XY_ERROR_MPS_LIMIT
    assert world["cycles"][0]["heading_error_rad_s"] == pytest.approx(0.0, abs=1e-12)
    assert world["cycles"][0]["max_heading_error_rad_s"] > path.HEADING_ERROR_RAD_S_LIMIT
    assert "world-0:cycle-xy-error-rate-steps-70-110" in report["failures"]


@pytest.mark.parametrize("mutation,reason", [
    ("missing_cycles", "fewer-than-three-complete-left-cycles"),
    ("chatter", "fewer-than-three-complete-left-cycles"),
    ("right_missing", "cycle-must-contain-exactly-one-right-touchdown"),
    ("timer", "contact-timer-disagreement"),
    ("period", "cycle-duration-out-of-range"),
])
def test_unsupported_support_events_reject_without_cherry_picking(mutation, reason):
    case, trace = make_case_trace()
    if mutation == "missing_cycles":
        for step in range(60, 240):
            trace["contact_found"][step][0] = [0, 0]
            trace["current_air_time"][step][0] = [.02, .02]
            trace["current_contact_time"][step][0] = [0.0, 0.0]
    elif mutation == "chatter":
        for step in range(60, 240):
            trace["contact_found"][step][0] = [0, 0]
            trace["current_air_time"][step][0] = [.02, .02]
            trace["current_contact_time"][step][0] = [0.0, 0.0]
        for step in (70, 90, 110, 130, 150, 170, 190, 210):
            trace["contact_found"][step][0][0] = 1
            trace["current_air_time"][step][0][0] = 0.0
            trace["current_contact_time"][step][0][0] = .02
    elif mutation == "right_missing":
        for step in range(60, 240):
            trace["contact_found"][step][0][1] = 0
            trace["current_air_time"][step][0][1] = .02
            trace["current_contact_time"][step][0][1] = 0.0
    elif mutation == "timer":
        trace["current_air_time"][75][0][0] = .02
    else:
        # Remove three expected touches so the remaining event gap is >1.5s.
        for step in (list(range(110, 118)) + list(range(150, 158))
                     + list(range(190, 198))):
            trace["contact_found"][step][0][0] = 0
            trace["current_air_time"][step][0][0] = .02
            trace["current_contact_time"][step][0][0] = 0.0
    report = path.summarize_case(case, trace)
    assert report["status"] == "route-not-qualified"
    assert any(reason in failure for failure in report["failures"])


def test_heading_degeneracy_and_hash_reset_command_identity_are_rejected():
    case, trace = make_case_trace()
    bad = deepcopy(trace)
    bad["quat_wxyz"][80][0] = [math.sqrt(.5), 0.0, math.sqrt(.5), 0.0]
    bad["angular_w"][80][0] = [.2, 0.0, 0.0]
    with pytest.raises(ValueError, match="projected body-x"):
        path.summarize_case(case, bad)
    bad = deepcopy(trace)
    bad["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        path.summarize_case(case, bad)
    bad = deepcopy(trace)
    bad["episode_length"][100][0] = 0
    with pytest.raises(ValueError, match="episode length"):
        path.summarize_case(case, bad)
    bad = deepcopy(trace)
    bad["command"][100][0][0] += .01
    with pytest.raises(ValueError, match="command"):
        path.summarize_case(case, bad)


@pytest.mark.parametrize("key,value,expected", [
    ("torque_p99", .60001, "torque-utilization-p99"),
    ("rated_speed_exceed_fraction", .00001, "rated-speed-exceedance"),
])
def test_motor_gates_reject(key, value, expected):
    case, trace = make_case_trace()
    case[key] = value
    assert expected in path.summarize_case(case, trace)["failures"]


@pytest.mark.parametrize("field,value", [("torque_p99", -0.1), ("speed_mae", -0.01)])
def test_negative_motor_or_speed_metrics_are_invalid(field, value):
    case, trace = make_case_trace()
    if field == "torque_p99":
        case[field] = value
    else:
        case[field][0] = value
    with pytest.raises(ValueError, match="nonnegative"):
        path.summarize_case(case, trace)


@pytest.mark.parametrize("field,index,value,expected", [
    ("min_distance_to_hard_stop_rad_by_joint", 2, -.006, "motor-hinge-margin"),
    ("hard_stop_proximity_fraction_by_joint", 9, .01001, "right-hip-yaw-near-stop"),
])
def test_joint_margin_and_named_right_hip_yaw_gate(field, index, value, expected):
    case, trace = make_case_trace()
    case["response_diagnostics"][field][index] = value
    assert expected in path.summarize_case(case, trace)["failures"]


def test_route_decision_requires_exact_order_and_propagates_nonqualification():
    rows = []
    for seed in gait.EVALUATION_SEEDS:
        for speed, yaw in gait.EVALUATION_CASES:
            case, trace = make_case_trace(speed=speed, yaw=yaw, seed=seed)
            if (speed, yaw) == (0.0, 0.0):
                case["yaw_mae"] = [0.0] * 8
            rows.append(path.summarize_case(case, trace))
    assert path.decision(rows)["decision"] == "route-qualified-simulation-only"
    rows[3]["status"] = "route-not-qualified"
    rows[3]["qualified"] = False
    rows[3]["safety"]["qualified"] = False
    rows[3]["safety"]["failures"].append("injected")
    rows[3]["failures"].append("injected")
    assert path.decision(rows)["decision"] == "route-not-qualified"
    with pytest.raises(ValueError, match="ordered"):
        path.decision(list(reversed(rows)))


def test_decision_rejects_checkpoint_mismatch_threshold_drift_and_spoofed_status():
    rows = []
    for seed in gait.EVALUATION_SEEDS:
        for speed, yaw in gait.EVALUATION_CASES:
            case, trace = make_case_trace(speed=speed, yaw=yaw, seed=seed)
            if (speed, yaw) == (0.0, 0.0):
                case["yaw_mae"] = [0.0] * 8
            rows.append(path.summarize_case(case, trace))
    altered = deepcopy(rows)
    altered[-1]["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="one checkpoint"):
        path.decision(altered)
    altered = deepcopy(rows)
    altered[0]["thresholds"]["heading_error_rad_s"] += .01
    with pytest.raises(ValueError, match="thresholds"):
        path.decision(altered)
    altered = deepcopy(rows)
    altered[0]["status"] = "route-qualified-simulation-only"
    altered[0]["qualified"] = True
    altered[0]["worlds"][0]["qualified"] = False
    with pytest.raises(ValueError, match="flags"):
        path.decision(altered)

