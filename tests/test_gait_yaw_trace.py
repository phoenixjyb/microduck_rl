from copy import deepcopy

import pytest

from mjlab_microduck import gait_yaw_trace as gait


def make_case_trace(*, degenerate=False):
    speed, yaw = .2, -.2
    digest = "a" * 64
    case = dict(seed=839, speed=speed, yaw=yaw, worlds=8, steps=240, complete=True,
                terminal_worlds=[], checkpoint_sha256=digest, yaw_mae=[.01] * 8)
    trace = dict(
        protocol=gait.PROTOCOL,
        step_dt=.02,
        startup_steps=60,
        steps=240,
        worlds=8,
        checkpoint_sha256=digest,
        timing=gait.TIMING,
        contact_columns=list(gait.CONTACT_COLUMNS),
        foot_sites=list(gait.FOOT_SITES),
        head_joints=list(gait.HEAD_JOINTS),
        force_timing=gait.FORCE_TIMING,
        force_frame="world",
        terrain="stationary-flat",
        policy_acceptance=False,
        physical_motion_authorized=False,
        command=[], actor_command=[], angular_b=[], angular_w=[], quat_wxyz=[],
        episode_length=[], physics_time=[],
        contact_found=[], contact_force_w=[], current_air_time=[], current_contact_time=[],
        foot_velocity_w=[], head_position_offset=[], head_velocity=[], head_command=[],
        actor_head_command=[],
    )
    for step in range(240):
        cmd_rows, actor_rows, angular_rows, angular_world_rows, quat_rows = [], [], [], [], []
        episode_rows, time_rows = [], []
        found_rows, force_rows, air_rows, contact_time_rows = [], [], [], []
        foot_velocity_rows, head_offset_rows, head_velocity_rows = [], [], []
        head_command_rows, actor_head_rows = [], []
        residual = 0. if degenerate or step < 60 else (.02 if step % 2 else 0.)
        for world in range(8):
            twist = [speed, 0., yaw]
            cmd_rows.append(twist)
            actor_rows.append(twist[:])
            angular_rows.append([0., 0., yaw + residual])
            angular_world_rows.append([0., 0., yaw + residual])
            quat_rows.append([1., 0., 0., 0.])
            episode_rows.append(step)
            time_rows.append(step * gait.STEP_DT)
            pattern = (step + world) % 3
            found_rows.append(([0, 0], [1, 0], [1, 1])[pattern])
            force_rows.append([[0., 0., 1.], [0., 0., 1.]])
            air_rows.append(([.02, .03], [.0, .03], [.0, .0])[pattern])
            contact_time_rows.append(([0., 0.], [.02, 0.], [.02, .02])[pattern])
            foot_velocity_rows.append([[.1 + world * .01, .0, 0.], [.2, .0, 0.]])
            head_offset_rows.append([0., 0., 0., 0.])
            head_velocity_rows.append([residual, 1., -1., 0.])
            head_command_rows.append([0., 0., 0., 0.])
            actor_head_rows.append([0., 0., 0., 0.])
        for key, rows in (
            ("command", cmd_rows), ("actor_command", actor_rows), ("angular_b", angular_rows),
            ("angular_w", angular_world_rows), ("quat_wxyz", quat_rows),
            ("episode_length", episode_rows), ("physics_time", time_rows),
            ("contact_found", found_rows), ("contact_force_w", force_rows),
            ("current_air_time", air_rows), ("current_contact_time", contact_time_rows),
            ("foot_velocity_w", foot_velocity_rows), ("head_position_offset", head_offset_rows),
            ("head_velocity", head_velocity_rows), ("head_command", head_command_rows),
            ("actor_head_command", actor_head_rows),
        ):
            trace[key].append(rows)
    return case, trace


def test_summary_counts_contacts_flight_slip_and_head_yaw_correlations_read_only():
    case, trace = make_case_trace()
    before = deepcopy(trace)
    summary = gait.summarize_trace(case, trace)
    assert trace == before
    assert summary["protocol"] == gait.SUMMARY_PROTOCOL
    assert summary["contact_pattern_count_per_world"] == {
        "no-contact": [60] * 8,
        "one-contact": [60] * 8,
        "two-contacts": [60] * 8,
    }
    assert summary["contact_pattern_fraction_per_world"] == {
        name: [1 / 3] * 8 for name in ("no-contact", "one-contact", "two-contacts")
    }
    assert summary["double_flight_fraction_per_world"] == [1 / 3] * 8
    assert summary["ambiguous_air_timer_fraction_per_world"] == [0.] * 8
    assert summary["contact_masked_foot_slip_mean_mps_per_world"]["left_foot"] == pytest.approx(
        [0.1 + world * .01 for world in range(8)])
    assert summary["contact_masked_foot_slip_max_mps_per_world"]["right_foot"] == [.2] * 8
    assert summary["yaw_residual_mae_by_contact_pattern_pooled"] == pytest.approx(
        {"no-contact": .01, "one-contact": .01, "two-contacts": .01})
    assert summary["mean_abs_head_velocity_rad_s_pooled"]["neck_pitch"] == pytest.approx(.01)
    assert summary["signed_yaw_residual_head_velocity_pearson_pooled"]["neck_pitch"] == pytest.approx(1.)
    assert summary["signed_yaw_residual_head_velocity_pearson_pooled"]["head_pitch"] is None
    assert summary["signed_yaw_residual_head_velocity_pearson_per_world"]["neck_pitch"] == pytest.approx([1.] * 8)
    assert summary["signed_yaw_residual_head_velocity_pearson_per_world"]["head_pitch"] == [None] * 8
    assert summary["mean_abs_head_velocity_rad_s_per_world"]["neck_pitch"] == pytest.approx([.01] * 8)
    assert summary["yaw_residual_mae_by_contact_pattern_per_world"]["one-contact"] == pytest.approx([.01] * 8)
    assert summary["recomputed_yaw_mae_per_world"] == pytest.approx(case["yaw_mae"])
    assert summary["policy_acceptance"] is False
    assert summary["physical_motion_authorized"] is False


@pytest.mark.parametrize("mutate", [
    lambda case, trace: trace.pop("contact_found"),
    lambda case, trace: trace["foot_velocity_w"][0].pop(),
    lambda case, trace: trace["head_velocity"][0][0].__setitem__(0, float("nan")),
    lambda case, trace: trace["contact_found"][0][0].__setitem__(0, -1),
    lambda case, trace: trace["contact_found"][0][0].__setitem__(0, .5),
    lambda case, trace: trace["contact_found"][0][0].__setitem__(0, True),
    lambda case, trace: trace["contact_force_w"][0][0][0].__setitem__(2, float("inf")),
    lambda case, trace: trace["episode_length"][70].__setitem__(0, 0),
    lambda case, trace: trace["physics_time"][70].__setitem__(0, trace["physics_time"][69][0]),
    lambda case, trace: trace["command"][70][0].__setitem__(2, .2),
    lambda case, trace: trace["actor_head_command"][70][0].__setitem__(1, .1),
    lambda case, trace: trace["quat_wxyz"][70][0].__setitem__(0, 2.),
    lambda case, trace: trace["angular_w"][70][0].__setitem__(1, .01),
    lambda case, trace: case["yaw_mae"].__setitem__(0, .5),
])
def test_rejects_malformed_or_misaligned_trace(mutate):
    case, trace = make_case_trace()
    mutate(case, trace)
    with pytest.raises(ValueError):
        gait.summarize_trace(case, trace)


@pytest.mark.parametrize("field,value", [
    ("protocol", "other"),
    ("checkpoint_sha256", "b" * 64),
    ("contact_columns", ["right_foot_collision", "left_foot_collision"]),
    ("force_frame", "contact"),
    ("terrain", "moving"),
    ("policy_acceptance", True),
])
def test_rejects_wrong_trace_identity_metadata(field, value):
    case, trace = make_case_trace()
    trace[field] = value
    with pytest.raises(ValueError):
        gait.summarize_trace(case, trace)


def test_rejects_incomplete_terminal_or_nonfixed_case():
    case, trace = make_case_trace()
    case["complete"] = False
    with pytest.raises(ValueError):
        gait.summarize_trace(case, trace)
    case, trace = make_case_trace()
    case["terminal_worlds"] = [3]
    with pytest.raises(ValueError):
        gait.summarize_trace(case, trace)
    case, trace = make_case_trace()
    case["worlds"] = 4
    with pytest.raises(ValueError):
        gait.summarize_trace(case, trace)


def test_degenerate_head_correlations_and_empty_contact_bins_are_none():
    case, trace = make_case_trace(degenerate=True)
    for step in range(gait.STARTUP_STEPS, gait.STEPS):
        for world in range(gait.WORLDS):
            trace["head_velocity"][step][world] = [0., 0., 0., 0.]
            trace["contact_found"][step][world] = [1, 1]
            trace["current_air_time"][step][world] = [0., 0.]
    case["yaw_mae"] = [0.] * gait.WORLDS
    summary = gait.summarize_trace(case, trace)
    assert summary["signed_yaw_residual_head_velocity_pearson_pooled"] == dict.fromkeys(gait.HEAD_JOINTS)
    assert summary["double_flight_fraction_per_world"] == [0.] * gait.WORLDS
    assert summary["ambiguous_air_timer_fraction_per_world"] == [0.] * gait.WORLDS
    assert summary["yaw_residual_mae_by_contact_pattern_pooled"] == {
        "no-contact": None, "one-contact": None, "two-contacts": 0.}


def test_ambiguous_air_timer_fraction_is_reported_not_relabelled_as_contact():
    case, trace = make_case_trace()
    trace["current_air_time"][60][0] = [0., 0.]
    summary = gait.summarize_trace(case, trace)
    assert summary["contact_pattern_count_per_world"]["no-contact"][0] == 60
    assert summary["double_flight_fraction_per_world"][0] == pytest.approx(59 / 180)
    assert summary["ambiguous_air_timer_fraction_per_world"][0] == pytest.approx(2 / 360)


def test_contact_found_accepts_integral_float_counts_and_preserves_contact_labels():
    case, trace = make_case_trace()
    trace["contact_found"] = [
        [[float(count) for count in feet] for feet in worlds]
        for worlds in trace["contact_found"]
    ]
    summary = gait.summarize_trace(case, trace)
    assert summary["contact_pattern_fraction_per_world"]["no-contact"] == pytest.approx([1 / 3] * 8)


@pytest.mark.parametrize("field", ["physics_time", "current_air_time", "current_contact_time"])
def test_rejects_negative_physics_or_contact_time(field):
    case, trace = make_case_trace()
    if field == "physics_time":
        trace[field][0][0] = -.001
    else:
        trace[field][0][0][0] = -.001
    with pytest.raises(ValueError, match="nonnegative"):
        gait.summarize_trace(case, trace)


def test_per_world_correlations_avoid_between_world_confounding():
    case, trace = make_case_trace()
    offsets = [-.07, -.05, -.03, -.01, .01, .03, .05, .07]
    case["yaw_mae"] = [abs(offset) for offset in offsets]
    for step in range(gait.STARTUP_STEPS, gait.STEPS):
        oscillation = .001 if step % 2 else -.001
        for world, offset in enumerate(offsets):
            residual = offset + oscillation
            trace["angular_b"][step][world][2] = case["yaw"] + residual
            trace["angular_w"][step][world][2] = case["yaw"] + residual
            trace["head_velocity"][step][world][0] = oscillation - 10 * offset
    summary = gait.summarize_trace(case, trace)
    assert summary["signed_yaw_residual_head_velocity_pearson_pooled"]["neck_pitch"] < -.99
    assert summary["signed_yaw_residual_head_velocity_pearson_per_world"]["neck_pitch"] == pytest.approx([1.] * 8)
    assert summary["mean_abs_head_velocity_rad_s_per_world"]["neck_pitch"] == pytest.approx(
        [10 * abs(offset) for offset in offsets])
