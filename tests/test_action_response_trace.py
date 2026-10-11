"""Pure-stdlib validation and descriptive summaries for action response traces."""

from copy import deepcopy
import math

import pytest

from mjlab_microduck.action_response_trace import JOINTS, summarize_trace


def _fixture():
    steps, worlds, width = 240, 8, 14
    digest = "a" * 64
    speed, yaw = .2, .2
    command = [[[speed, 0.0, yaw] for _ in range(worlds)] for _ in range(steps)]
    actor_command = deepcopy(command)
    quats = [[ [1.0, 0.0, 0.0, 0.0] for _ in range(worlds)] for _ in range(steps)]
    angular_b, angular_w = [], []
    raw_action, previous_actor, previous_raw, qpos, qvel = [], [], [], [], []
    applied_target, offset, bias, limits, constraints = [], [], [], [], []
    episode, physics = [], []
    for step in range(steps):
        step_angular_b, step_angular_w = [], []
        step_raw, step_actor_prev, step_raw_prev = [], [], []
        step_qpos, step_qvel, step_target = [], [], []
        step_offset, step_bias, step_limits, step_constraints = [], [], [], []
        step_episode, step_physics = [], []
        for world in range(worlds):
            yaw_error = .02 * math.sin(step * .1 + world * .2)
            rate = [0.01 * math.sin(step * .07), 0.01 * math.cos(step * .05), yaw + yaw_error]
            step_angular_b.append(rate)
            step_angular_w.append(rate.copy())
            raw = [1.2 + .001 * step + .005 * world + .01 * joint for joint in range(width)]
            off = [.02 * joint for joint in range(width)]
            enc = [.005 * joint for joint in range(width)]
            if step == 0:
                actor_prev = [-.7 - .01 * joint for joint in range(width)]
                raw_prev = actor_prev.copy()
                target = [-10.0 - joint for joint in range(width)]  # first target is restored state
            else:
                actor_prev = raw_action[step - 1][world].copy()
                raw_prev = raw_action[step - 1][world].copy()
                target = [raw_action[step - 1][world][joint] + off[joint] - enc[joint]
                          for joint in range(width)]
            pos = []
            vel = []
            bounds = []
            force = []
            for joint in range(width):
                if joint == 0:
                    pos.append(.97 + .005 * math.sin(step * .05 + world))
                else:
                    pos.append(.1 * math.sin(step * .04 + world * .1 + joint * .2))
                vel.append(.1 * math.cos(step * .06 + world * .1 + joint * .2)
                           + .02 * math.sin(step * .1 + world * .2))
                bounds.append([-1.0, 1.0])
                force.append(.5 * math.sin(step * .08 + world * .2 + joint * .1))
            step_raw.append(raw)
            step_actor_prev.append(actor_prev)
            step_raw_prev.append(raw_prev)
            step_qpos.append(pos)
            step_qvel.append(vel)
            step_target.append(target)
            step_offset.append(off)
            step_bias.append(enc)
            step_limits.append(bounds)
            step_constraints.append(force)
            step_episode.append(1000 + world + step)
            step_physics.append(5.0 + .1 * world + .02 * step)
        angular_b.append(step_angular_b)
        angular_w.append(step_angular_w)
        raw_action.append(step_raw)
        previous_actor.append(step_actor_prev)
        previous_raw.append(step_raw_prev)
        qpos.append(step_qpos)
        qvel.append(step_qvel)
        applied_target.append(step_target)
        offset.append(step_offset)
        bias.append(step_bias)
        limits.append(step_limits)
        constraints.append(step_constraints)
        episode.append(step_episode)
        physics.append(step_physics)

    yaw_mae = [sum(abs(angular_b[step][world][2] - yaw) for step in range(60, 240)) / 180
               for world in range(worlds)]
    case = {
        "complete": True, "steps": steps, "worlds": worlds, "terminal_worlds": [],
        "seed": 839, "speed": speed, "yaw": yaw,
        "checkpoint_sha256": digest, "yaw_mae": yaw_mae,
    }
    trace = {
        "protocol": "microduck-action-response-trace-v1",
        "step_dt": .02, "steps": steps, "worlds": worlds, "startup_steps": 60,
        "checkpoint_sha256": digest,
        "joint_columns": list(JOINTS), "target_ids": list(range(width)),
        "timing": "pre-action state before policy/step",
        "action_timing": "raw policy output after inference before step",
        "target_timing": "previous applied target at pre-action boundary",
        "action_history_timing": (
            "actor_previous_action and raw_previous_action are the pre-policy "
            "previous-step action history; first sample may follow restored/reset state"
        ),
        "constraint_timing": "last-physics-derived generalized constraint at pre-action boundary",
        "action_interpretation": "raw unclipped actor output; magnitude alone is not saturation",
        "constraint_includes_contacts_and_limits": True,
        "hard_stop_force_isolated": False,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
        "simulator_qualified": False,
        "command": command, "actor_command": actor_command,
        "quat_wxyz": quats, "angular_b": angular_b, "angular_w": angular_w,
        "actor_previous_action": previous_actor, "raw_previous_action": previous_raw,
        "raw_action": raw_action, "joint_position": qpos, "joint_velocity": qvel,
        "previous_applied_target": applied_target, "encoder_bias": bias,
        "action_offset": offset, "hard_limits": limits,
        "generalized_constraint_force": constraints,
        "episode_length": episode, "physics_time": physics,
        "common_step_counter": [9000 + step for step in range(steps)],
    }
    return case, trace


def test_valid_full_trace_descriptive_result_and_raw_magnitude_is_not_saturation():
    case, trace = _fixture()
    out = summarize_trace(case, trace)
    assert out["protocol"] == "microduck-action-response-descriptive-audit-v1"
    assert out["status"] == "descriptive-audit-only"
    assert out["policy_acceptance"] is False
    assert out["physical_motion_authorized"] is False
    assert out["raw_action_magnitude_above_one_is_not_saturation"] is True
    assert out["body_yaw_trace_audit"]["policy_acceptance"] is False
    assert out["body_yaw_trace_audit"]["settled_steps"] == 180
    assert len(out["worlds"]) == 8 and len(out["worlds"][0]["joints"]) == 14
    joint = out["worlds"][0]["joints"][0]
    assert joint["joint"] == JOINTS[0]
    assert joint["raw_action_mean"] > 1.0
    assert joint["qpos_within_0_05_rad_hard_stop_margin_fraction"] > .9
    assert joint["hypothetical_joint_limit_proximity_cost_mean"] > 0.1
    assert out["correlations_are_descriptive_not_causal"] is True


@pytest.mark.parametrize("field,value", [
    ("protocol", "wrong"),
    ("checkpoint_sha256", "b" * 64),
    ("timing", "post-step"),
    ("action_timing", "clipped output"),
    ("target_timing", "current target"),
    ("constraint_timing", "current force"),
    ("startup_steps", 59),
    ("steps", 239),
    ("worlds", 7),
    ("simulator_qualified", True),
    ("policy_acceptance", True),
    ("physical_motion_authorized", True),
    ("constraint_includes_contacts_and_limits", False),
    ("hard_stop_force_isolated", True),
])
def test_trace_identity_and_authority_mutations_rejected(field, value):
    case, trace = _fixture()
    trace[field] = value
    with pytest.raises(ValueError):
        summarize_trace(case, trace)


@pytest.mark.parametrize("mutate", [
    lambda case: case.update(complete=False),
    lambda case: case.update(terminal_worlds=[0]),
    lambda case: case.update(seed=1001),
    lambda case: case.update(speed=.5),
    lambda case: case.update(checkpoint_sha256="A" * 64),
    lambda case: case.update(steps=True),
])
def test_case_identity_mutations_rejected(mutate):
    case, trace = _fixture()
    mutate(case)
    with pytest.raises(ValueError):
        summarize_trace(case, trace)


@pytest.mark.parametrize("key", ["actor_previous_action", "raw_previous_action", "raw_action",
                                 "joint_position", "joint_velocity", "previous_applied_target",
                                 "encoder_bias", "action_offset", "hard_limits",
                                 "generalized_constraint_force", "command", "angular_b"])
def test_bool_nonfinite_and_bad_layout_arrays_rejected(key):
    case, trace = _fixture()
    trace[key][60][0][0] = True
    with pytest.raises(ValueError):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace[key][60][0][0] = float("nan")
    with pytest.raises(ValueError):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace[key].pop()
    with pytest.raises(ValueError):
        summarize_trace(case, trace)


def test_ids_joint_columns_and_hash_binding_are_strict():
    case, trace = _fixture()
    trace["target_ids"][1] = trace["target_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["target_ids"][0] = True
    with pytest.raises(ValueError, match="target_ids"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["joint_columns"][0] = "other_joint"
    with pytest.raises(ValueError, match="joint column"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hash"):
        summarize_trace(case, trace)


def test_actor_command_and_held_out_command_must_remain_exact():
    case, trace = _fixture()
    trace["actor_command"][100][2][2] += .01
    with pytest.raises(ValueError, match="actor command"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["command"][100][2][2] += .01
    trace["actor_command"][100][2][2] += .01
    with pytest.raises(ValueError, match="held-out command"):
        summarize_trace(case, trace)


def test_histories_must_be_equal_and_advance_from_previous_raw_action():
    case, trace = _fixture()
    trace["raw_previous_action"][100][0][0] += .1
    with pytest.raises(ValueError, match="histories"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["actor_previous_action"][101][0][0] += .1
    with pytest.raises(ValueError, match="both next-step histories"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["raw_previous_action"][101][0][0] += .1
    with pytest.raises(ValueError, match="both next-step histories"):
        summarize_trace(case, trace)


def test_first_target_is_not_assumed_but_later_target_equation_is_checked():
    case, trace = _fixture()
    trace["previous_applied_target"][0][0][0] = 12345.0
    summarize_trace(case, trace)
    case, trace = _fixture()
    trace["previous_applied_target"][101][0][0] += .01
    with pytest.raises(ValueError, match="target differs"):
        summarize_trace(case, trace)


@pytest.mark.parametrize("field", ["action_offset", "encoder_bias", "hard_limits"])
def test_offsets_biases_and_limits_must_be_constant(field):
    case, trace = _fixture()
    if field == "hard_limits":
        trace[field][100][0][0][0] += .001
    else:
        trace[field][100][0][0] += .001
    with pytest.raises(ValueError, match="unchanged"):
        summarize_trace(case, trace)


def test_integer_clocks_no_reset_and_physics_step_are_required_without_zero_start():
    case, trace = _fixture()
    trace["common_step_counter"][10] += 1
    with pytest.raises(ValueError, match="common-step"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["common_step_counter"][0] = True
    with pytest.raises(ValueError, match="integer"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["episode_length"][10][0] += 1
    with pytest.raises(ValueError, match="episode clock"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["physics_time"][10][0] += .01
    with pytest.raises(ValueError, match="physics clock"):
        summarize_trace(case, trace)


def test_limits_quaternion_frame_and_original_body_yaw_mae_are_validated():
    case, trace = _fixture()
    trace["hard_limits"][100][0][0] = [1.0, -1.0]
    with pytest.raises(ValueError, match="hard limits"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["quat_wxyz"][100][0] = [2.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="unit"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    trace["angular_w"][100][0][2] += .1
    with pytest.raises(ValueError, match="rotation mismatch"):
        summarize_trace(case, trace)
    case, trace = _fixture()
    case["yaw_mae"][0] += .1
    with pytest.raises(ValueError, match="MAE mismatch"):
        summarize_trace(case, trace)


def test_summary_arithmetic_overflow_fails_closed():
    case, trace = _fixture()
    trace["raw_action"][60][0][0] = 1.7e308
    trace["raw_action"][61][0][0] = -1.7e308
    trace["actor_previous_action"][61][0][0] = 1.7e308
    trace["raw_previous_action"][61][0][0] = 1.7e308
    trace["actor_previous_action"][62][0][0] = -1.7e308
    trace["raw_previous_action"][62][0][0] = -1.7e308
    trace["previous_applied_target"][61][0][0] = 1.7e308
    trace["previous_applied_target"][62][0][0] = -1.7e308
    with pytest.raises(ValueError, match="arithmetic"):
        summarize_trace(case, trace)
