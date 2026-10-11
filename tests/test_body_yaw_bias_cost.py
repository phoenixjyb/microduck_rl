"""body_yaw_bias_cost prices only persistent body yaw-rate residual."""

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


class _Env:
    def __init__(self, worlds=2):
        self.num_envs = worlds
        self.device = "cpu"
        self.step_dt = .02
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((worlds,), 10, dtype=torch.long)
        self.command = torch.zeros(worlds, 3)
        self.command_manager = SimpleNamespace(get_command=lambda _name: self.command)
        self.angular = torch.zeros(worlds, 3)
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=self.angular))}


def _step(env, *, command=None, angular=None, episode_length=None):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    if command is not None:
        env.command[:, 2] = command
    if angular is not None:
        env.angular[:, 2] = angular
    if episode_length is not None:
        env.episode_length_buf.copy_(torch.as_tensor(episode_length, dtype=torch.long))
    return mdp.body_yaw_bias_cost(env)


def test_exact_tracking_has_zero_cost_and_expected_state_contract():
    env = _Env()
    env.command[:, 2] = .25
    env.angular[:, 2] = .25
    out = _step(env)
    state = env._microduck_body_yaw_bias_state
    assert torch.equal(out, torch.zeros(2))
    assert state["protocol"] == "body-yaw-bias-cost-v1"
    assert state["step"] == env.common_step_counter
    assert state["command_name"] == "twist" and state["asset_name"] == "robot"
    assert state["tau_s"] == 1.0
    assert state["step_dt"] == .02
    assert torch.equal(state["signed_residual"], torch.zeros(2))
    assert torch.equal(state["signed_ema"], torch.zeros(2))
    assert torch.equal(state["cost"], out)


@pytest.mark.parametrize("bias", [.2, -.2])
def test_persistent_positive_and_negative_bias_have_symmetric_cost(bias):
    env = _Env(1)
    env.command[:, 2] = bias
    for _ in range(180):
        out = _step(env)
    expected = abs(bias) * (1.0 - .98**180)
    assert out.item() == pytest.approx(expected, abs=1e-6)
    assert env._microduck_body_yaw_bias_state["signed_ema"].item() == pytest.approx(
        bias * (1.0 - .98**180), abs=1e-6
    )


def test_alternating_residual_cancels_in_ema_not_instantaneous_l1():
    env = _Env(1)
    out = None
    for index in range(300):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        env.angular[0, 2] = -1.0 if index % 2 == 0 else 1.0
        out = mdp.body_yaw_bias_cost(env)
    state = env._microduck_body_yaw_bias_state
    assert state["signed_residual"].abs().item() == 1.0
    assert out.item() < .011
    assert out.item() < state["signed_residual"].abs().item() * .02


def test_per_world_episode_reset_clears_only_reset_history():
    env = _Env(2)
    env.command[:, 2] = .2
    for _ in range(100):
        out = _step(env)
    before = out.clone()
    out = _step(env, episode_length=[1, 101])
    assert out[0].item() == pytest.approx(.004, abs=1e-7)
    assert out[1].item() > before[1].item() - 1e-6


def test_yaw_command_change_clears_old_target_history():
    env = _Env(1)
    env.command[0, 2] = .2
    for _ in range(100):
        out = _step(env)
    assert out.item() > .15
    # The new command exactly matches the measured body rate. Clearing stale
    # history means it cannot charge the prior command's accumulated bias.
    out = _step(env, command=.4, angular=.4)
    assert out.item() == 0.0
    state = env._microduck_body_yaw_bias_state
    assert state["signed_residual"].item() == 0.0
    assert state["signed_ema"].item() == 0.0


def test_reward_and_post_reward_metric_read_is_idempotent_and_alias_free():
    env = _Env(1)
    env.command[0, 2] = .2
    first = _step(env)
    expected = first.item()
    first.zero_()
    env._microduck_body_yaw_bias_state["signed_ema"].zero_()
    second = mdp.body_yaw_bias_cost(env)  # same common_step_counter
    assert second.item() == pytest.approx(expected)
    assert env._microduck_body_yaw_bias_state["signed_ema"].item() == pytest.approx(expected)
    assert second.data_ptr() != env._microduck_body_yaw_bias_state["cost"].data_ptr()


def test_counter_rollback_clears_full_ema_state():
    env = _Env(1)
    env.command[0, 2] = .2
    for _ in range(100):
        out = _step(env)
    assert out.item() > .15
    env.common_step_counter = 0
    out = mdp.body_yaw_bias_cost(env)
    assert out.item() == pytest.approx(.004, abs=1e-7)


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_tau_fails_closed(tau):
    env = _Env()
    with pytest.raises(ValueError):
        mdp.body_yaw_bias_cost(env, tau_s=tau)


def test_invalid_dt_counter_command_and_layout_fail_closed():
    env = _Env()
    env.step_dt = float("nan")
    with pytest.raises(ValueError, match="step_dt"):
        mdp.body_yaw_bias_cost(env)
    env.step_dt = .02
    env.common_step_counter = True
    with pytest.raises(ValueError, match="common_step_counter"):
        mdp.body_yaw_bias_cost(env)
    env.common_step_counter = 0
    env.command = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="command must have shape"):
        mdp.body_yaw_bias_cost(env)


def test_invalid_angular_layout_episode_buffer_and_world_count_fail_closed():
    env = _Env(2)
    env.angular = torch.zeros(2, 2)
    env.scene["robot"].data.root_link_ang_vel_b = env.angular
    with pytest.raises(ValueError, match="body angular velocity"):
        mdp.body_yaw_bias_cost(env)
    env.angular = torch.zeros(2, 3)
    env.scene["robot"].data.root_link_ang_vel_b = env.angular
    env.episode_length_buf = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="episode_length_buf"):
        mdp.body_yaw_bias_cost(env)
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="episode_length_buf"):
        mdp.body_yaw_bias_cost(env)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.num_envs = 3
    with pytest.raises(ValueError, match="num_envs"):
        mdp.body_yaw_bias_cost(env)


def test_nonfinite_inputs_and_nonfinite_derived_residual_fail_before_cache():
    env = _Env(1)
    env.command[0, 2] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite yaw-bias"):
        mdp.body_yaw_bias_cost(env)
    assert not hasattr(env, "_microduck_body_yaw_bias_internal")
    env.command[0, 2] = 3e38
    env.angular[0, 2] = -3e38
    with pytest.raises(FloatingPointError, match="residual"):
        mdp.body_yaw_bias_cost(env)
    assert not hasattr(env, "_microduck_body_yaw_bias_internal")


def test_config_mismatch_same_step_input_mutation_and_skipped_step_fail():
    env = _Env(1)
    mdp.body_yaw_bias_cost(env)
    with pytest.raises(ValueError, match="incompatible config"):
        mdp.body_yaw_bias_cost(env, tau_s=.5)
    env.common_step_counter += 1
    mdp.body_yaw_bias_cost(env)
    env.command[0, 2] = .1
    with pytest.raises(RuntimeError, match="inputs changed"):
        mdp.body_yaw_bias_cost(env)
    env.command.zero_()
    env.common_step_counter += 2
    with pytest.raises(RuntimeError, match="skipped forward"):
        mdp.body_yaw_bias_cost(env)


def test_step_dt_drift_fails_closed_after_initialization():
    env = _Env(1)
    mdp.body_yaw_bias_cost(env)
    env.step_dt = .01
    with pytest.raises(ValueError, match="incompatible config"):
        mdp.body_yaw_bias_cost(env)


@pytest.mark.parametrize("change", ["command", "asset", "tau"])
def test_fixed_metadata_changes_fail_closed(change):
    env = _Env(1)
    mdp.body_yaw_bias_cost(env)
    kwargs = {}
    if change == "command":
        kwargs["command_name"] = "other"
    elif change == "asset":
        env.scene["other"] = env.scene["robot"]
        kwargs["asset_cfg"] = SimpleNamespace(name="other")
    else:
        kwargs["tau_s"] = .5
    with pytest.raises(ValueError, match="incompatible config"):
        mdp.body_yaw_bias_cost(env, **kwargs)


def test_command_and_robot_inputs_are_never_mutated():
    env = _Env(1)
    env.command[0] = torch.tensor([.1, -.2, .3])
    env.angular[0] = torch.tensor([.4, -.5, .6])
    command_before, angular_before = env.command.clone(), env.angular.clone()
    _step(env)
    assert torch.equal(env.command, command_before)
    assert torch.equal(env.angular, angular_before)

