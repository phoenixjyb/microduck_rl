"""CPU contract tests for the pre/post projected-heading reward protocol."""

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


def _yaw_quat(angle):
    return torch.tensor([torch.cos(torch.tensor(angle / 2)), 0.0, 0.0,
                         torch.sin(torch.tensor(angle / 2))])


def _yaw_roll_quat(yaw, roll):
    # q_yaw * q_roll: body X keeps the yaw heading while the body is tilted.
    hy, hr = yaw / 2, roll / 2
    return torch.tensor([
        torch.cos(torch.tensor(hy)) * torch.cos(torch.tensor(hr)),
        torch.cos(torch.tensor(hy)) * torch.sin(torch.tensor(hr)),
        torch.sin(torch.tensor(hy)) * torch.sin(torch.tensor(hr)),
        torch.sin(torch.tensor(hy)) * torch.cos(torch.tensor(hr)),
    ])


class _Env:
    def __init__(self, worlds=1, dt=0.02):
        self.num_envs = worlds
        self.device = "cpu"
        self.step_dt = dt
        self.common_step_counter = 0
        self.episode_length_buf = torch.ones(worlds, dtype=torch.long)
        self.command = torch.zeros(worlds, 3)
        self.quat = torch.zeros(worlds, 4)
        self.quat[:, 0] = 1.0
        self.angular = torch.zeros(worlds, 3)
        self.command_manager = SimpleNamespace(get_command=lambda _name: self.command)
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_quat_w=self.quat, root_link_ang_vel_b=self.angular
        ))}

    def pre(self, **kwargs):
        return mdp.prepare_projected_heading_step(self, **kwargs)

    def post(self, angle=None, *, quat=None, **kwargs):
        if quat is not None:
            self.quat.copy_(quat)
        elif angle is not None:
            self.quat.copy_(_yaw_quat(angle).expand_as(self.quat))
        self.common_step_counter += 1
        self.episode_length_buf += 1
        return mdp.projected_heading_error_cost(self, **kwargs)


def _step(env, desired_heading, actual_heading=None, **kwargs):
    env.pre(**kwargs)
    env.post(desired_heading if actual_heading is None else actual_heading, **kwargs)
    return env._microduck_projected_heading_state


@pytest.mark.parametrize("yaw_rate", [0.7, -0.7])
def test_exact_turn_tracks_projected_body_x_when_tilted(yaw_rate):
    env = _Env()
    env.command[0, 2] = yaw_rate
    for i in range(1, 41):
        target = yaw_rate * env.step_dt * i
        env.pre()
        env.post(quat=_yaw_roll_quat(target, 0.63).reshape(1, 4))
    state = env._microduck_projected_heading_state
    assert state["cost"].item() == pytest.approx(0.0, abs=2e-5)
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(yaw_rate * .02 * 40, abs=2e-5)
    assert state["projection_valid"].item()


def test_pi_crossing_unwraps_actual_heading_and_body_rate_is_ignored():
    env = _Env()
    env.pre()
    env.post(angle=3.13)
    env.angular[0, 2] = 1000.0  # Not an input to this reward.
    env.pre()
    cost = env.post(angle=-3.13)
    state = env._microduck_projected_heading_state
    assert state["actual_heading_unwrapped_rad"].item() == pytest.approx(3.1531853, abs=2e-4)
    assert cost.item() == pytest.approx(3.1531853, abs=2e-4)


def test_unwrapped_target_accumulates_past_two_pi_without_wrapping_error():
    env = _Env(dt=0.1)
    env.command[0, 2] = 1.0
    for _ in range(80):
        env.pre()
        env.post(angle=0.0)
    state = env._microduck_projected_heading_state
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(8.0, abs=1e-5)
    assert state["signed_heading_error_rad"].item() == pytest.approx(8.0, abs=1e-5)
    assert state["cost"].item() == pytest.approx(8.0, abs=1e-5)


def test_fresh_same_counter_pre_pose_delta_is_accounted_without_moving_target():
    env = _Env()
    env.pre()
    env.post(angle=0.2)
    target = env._microduck_projected_heading_state["target_heading_unwrapped_rad"].clone()
    # sim.forward refreshes xquat after reward/autoreset; this is not a new step.
    env.quat.copy_(_yaw_quat(0.21).reshape(1, 4))
    env.pre()
    env.post(angle=0.22)
    state = env._microduck_projected_heading_state
    assert state["actual_heading_unwrapped_rad"].item() == pytest.approx(0.22, abs=1e-5)
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(target.item(), abs=1e-5)
    assert state["cost"].item() == pytest.approx(0.22, abs=1e-5)


@pytest.mark.parametrize("yaw_rate", [3.0, -3.0])
def test_long_constant_turn_charges_only_one_substep_staleness_across_pi(yaw_rate):
    env = _Env()
    env.command[0, 2] = yaw_rate
    steps = 500  # More than four complete turns in either direction.
    for n in range(steps):
        # root_link_quat_w refreshed by sim.forward is the accurate pre-action pose.
        env.quat.copy_(_yaw_quat(yaw_rate * n * env.step_dt).reshape(1, 4))
        env.pre()
        # Reward observes a pose one physics substep behind the action boundary.
        env.post(angle=yaw_rate * ((n + 1) * env.step_dt - .005))
    state = env._microduck_projected_heading_state
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(
        yaw_rate * steps * env.step_dt, abs=2e-4
    )
    assert state["actual_heading_unwrapped_rad"].item() == pytest.approx(
        yaw_rate * (steps * env.step_dt - .005), abs=2e-4
    )
    assert state["signed_heading_error_rad"].item() == pytest.approx(yaw_rate * .005, abs=2e-4)
    assert state["cost"].item() == pytest.approx(abs(yaw_rate * .005), abs=2e-4)


def test_full_twist_change_reanchors_only_changed_world_before_action():
    env = _Env(2)
    env.command[:, 2] = 0.2
    env.pre()
    env.post(angle=1.0)
    env.command[0] = torch.tensor([0.5, 0.0, -0.4])
    env.pre()
    env.post(angle=1.4)
    state = env._microduck_projected_heading_state
    assert torch.equal(state["segment_reset"], torch.tensor([True, False]))
    assert state["target_heading_unwrapped_rad"][0].item() == pytest.approx(1.0 - 0.4 * .02, abs=1e-5)
    assert state["target_heading_unwrapped_rad"][1].item() == pytest.approx(.2 * .02 * 2, abs=1e-5)


def test_same_yaw_new_forward_speed_is_a_full_command_reset():
    env = _Env()
    env.command[0, 2] = .2
    env.pre()
    env.post(angle=1.0)
    env.command[0, 0] = .5  # yaw is unchanged; the twist tuple is not.
    env.pre()
    env.post(angle=1.0)
    state = env._microduck_projected_heading_state
    assert state["segment_reset"].item()
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(1.0 + .2 * .02, abs=1e-5)
    assert state["cost"].item() == pytest.approx(.004, abs=1e-5)


def test_episode_reset_reanchors_only_reset_world():
    env = _Env(2)
    env.pre()
    env.post(quat=torch.stack((_yaw_quat(1.0), _yaw_quat(1.0))))
    env.episode_length_buf[0] = 0  # reset was completed after reward
    env.quat[0] = _yaw_quat(-2.0)
    env.pre()
    env.post(quat=torch.stack((_yaw_quat(-1.8), _yaw_quat(1.2))))
    state = env._microduck_projected_heading_state
    assert torch.equal(state["segment_reset"], torch.tensor([True, False]))
    assert state["target_heading_unwrapped_rad"][0].item() == pytest.approx(-2.0, abs=1e-5)
    assert state["actual_heading_unwrapped_rad"][0].item() == pytest.approx(-1.8, abs=1e-5)
    assert state["target_heading_unwrapped_rad"][1].item() == pytest.approx(0.0, abs=1e-5)


def test_reward_metric_repeat_is_idempotent_cloned_and_input_mutation_fails_closed():
    env = _Env()
    env.command[0, 2] = 0.25
    env.pre()
    first = env.post(angle=0.0)
    state = env._microduck_projected_heading_state
    assert state["protocol"] == "projected-heading-error-cost-v1"
    assert state["step"] == 1 and state["command_name"] == "twist"
    assert state["asset_name"] == "robot" and state["reference_time_s"] == 1.0
    assert state["step_dt"] == .02
    assert first.data_ptr() != state["cost"].data_ptr()
    saved = state["cost"].clone()
    first.zero_()
    state["cost"].zero_()
    env.command[0, 2] = 0.0
    with pytest.raises(RuntimeError, match="inputs changed after reward"):
        mdp.projected_heading_error_cost(env)
    env.command[0, 2] = .25
    second = mdp.projected_heading_error_cost(env)
    assert torch.equal(second, saved)
    assert second.data_ptr() != env._microduck_projected_heading_state["cost"].data_ptr()
    assert torch.equal(env._microduck_projected_heading_state["cost"], saved)
    env.episode_length_buf[0] += 1
    with pytest.raises(RuntimeError, match="inputs changed after reward"):
        mdp.projected_heading_error_cost(env)


def test_pre_post_order_skips_counter_rollback_and_same_counter_reprime():
    env = _Env()
    with pytest.raises(RuntimeError, match="must precede"):
        mdp.projected_heading_error_cost(env)
    env.pre()
    with pytest.raises(RuntimeError, match="repeated before post"):
        env.pre()
    env.common_step_counter += 2
    with pytest.raises(RuntimeError, match="skipped forward"):
        mdp.projected_heading_error_cost(env)

    env = _Env()
    env.pre()
    env.post(angle=.4)
    env.common_step_counter = 0
    env.episode_length_buf.zero_()
    env.quat.copy_(_yaw_quat(-.8).reshape(1, 4))
    env.pre()
    env.post(angle=-.7)
    state = env._microduck_projected_heading_state
    assert state["segment_reset"].item()
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(-.8, abs=1e-5)


def test_counter_rollback_discards_pending_pre_action_state():
    env = _Env()
    env.common_step_counter = 4
    env.pre()
    with pytest.raises(RuntimeError, match="repeated before post"):
        env.pre()
    env.common_step_counter = 3
    env.quat.copy_(_yaw_quat(-.7).reshape(1, 4))
    env.pre()  # A genuine rollback starts a new segment, even with a pending old pre.
    env.post(angle=-.6)
    state = env._microduck_projected_heading_state
    assert state["segment_reset"].item()
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(-.7, abs=1e-5)


def test_undefined_projection_uses_finite_fixed_cost_and_recovery_reanchors():
    env = _Env()
    env.pre()
    vertical_x = torch.tensor([[2**-.5, 0.0, 2**-.5, 0.0]])
    out = env.post(quat=vertical_x)
    state = env._microduck_projected_heading_state
    assert out.item() == 1.0
    assert not state["projection_valid"].item()
    assert torch.isfinite(state["actual_heading_unwrapped_rad"]).all()
    env.pre()
    out = env.post(angle=.6)
    state = env._microduck_projected_heading_state
    assert out.item() == pytest.approx(0.0)
    assert state["projection_valid"].item()
    assert state["segment_reset"].item()
    assert state["target_heading_unwrapped_rad"].item() == pytest.approx(.6)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_bad_reference_time_fails_closed(bad):
    with pytest.raises(ValueError, match="reference_time_s"):
        mdp.prepare_projected_heading_step(_Env(), reference_time_s=bad)


def test_malformed_inputs_fail_closed_without_mutating_tensors():
    env = _Env()
    original_command = env.command.clone()
    original_quat = env.quat.clone()
    env.quat[0] *= 2.0
    with pytest.raises(ValueError, match="unit normalized"):
        env.pre()
    env.quat[0] *= 0.5
    env.quat[0, 0] = 1.00002
    with pytest.raises(ValueError, match="unit normalized"):
        env.pre()
    env.quat.copy_(original_quat)
    env.command[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="command"):
        env.pre()
    env.command.copy_(original_command)
    env.quat[0, 1] = float("nan")
    with pytest.raises(FloatingPointError, match="quaternion"):
        env.pre()
    env.quat.copy_(original_quat)
    env.command = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="command must have shape"):
        env.pre()
    env.command = original_command
    env.quat = torch.zeros(1, 3)
    env.scene["robot"].data.root_link_quat_w = env.quat
    with pytest.raises(ValueError, match="must have shape"):
        env.pre()
    env.quat = original_quat
    env.scene["robot"].data.root_link_quat_w = env.quat
    env.step_dt = 0.0
    with pytest.raises(ValueError, match="step_dt"):
        env.pre()
    env.step_dt = .02
    env.common_step_counter = True
    with pytest.raises(ValueError, match="common_step_counter"):
        env.pre()
    env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="episode_length_buf"):
        env.pre()


def test_nonfinite_derived_target_and_input_immutability():
    env = _Env()
    env.command[0, 2] = 2.0
    command_before = env.command.clone()
    quat_before = env.quat.clone()
    env.step_dt = 1.0e308
    with pytest.raises(FloatingPointError, match="target"):
        env.pre()
    assert torch.equal(env.command, command_before)
    assert torch.equal(env.quat, quat_before)

    env.step_dt = .02
    env.pre()
    env.post(angle=.01)
    assert torch.equal(env.command, command_before)


def test_command_change_between_pre_and_post_fails_closed():
    env = _Env()
    env.pre()
    env.command[0, 2] = .1
    env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="changed during step"):
        mdp.projected_heading_error_cost(env)


def test_command_quaternion_dtype_and_reused_layout_drift_fail_closed():
    env = _Env()
    env.command = env.command.double()
    with pytest.raises(ValueError, match="share a floating point dtype"):
        env.pre()
    env.quat = env.quat.double()
    env.scene["robot"].data.root_link_quat_w = env.quat
    env.pre()
    env.post(angle=.01)

    env.num_envs = 2
    env.command = torch.zeros(2, 3, dtype=torch.float64)
    env.quat = torch.zeros(2, 4, dtype=torch.float64)
    env.quat[:, 0] = 1.0
    env.scene["robot"].data.root_link_quat_w = env.quat
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    with pytest.raises(ValueError, match="incompatible config"):
        env.pre()
