"""CPU-only contract tests for the persistent planar body-velocity cost."""

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
        self.linear = torch.zeros(worlds, 3)
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=self.linear))}


def _step(env, *, command_xy=None, linear_xy=None, command_yaw=None, episode_length=None):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    if command_xy is not None:
        env.command[:, :2] = torch.as_tensor(command_xy, dtype=env.command.dtype)
    if linear_xy is not None:
        env.linear[:, :2] = torch.as_tensor(linear_xy, dtype=env.linear.dtype)
    if command_yaw is not None:
        env.command[:, 2] = command_yaw
    if episode_length is not None:
        env.episode_length_buf.copy_(torch.as_tensor(episode_length, dtype=torch.long))
    return mdp.body_planar_bias_cost(env)


def test_constant_planar_bias_converges_and_publishes_contract():
    env = _Env(1)
    env.command[0, :2] = torch.tensor([.4, -.2])
    for _ in range(100):
        cost = _step(env)
    state = env._microduck_body_planar_bias_state
    expected = torch.tensor([.4, -.2]) * (1.0 - .96**100)
    assert torch.allclose(state["signed_ema"][0], expected, atol=1e-6)
    assert cost.item() == pytest.approx(expected.abs().sum().item(), abs=1e-6)
    assert state["protocol"] == "body-planar-bias-cost-v1"
    assert state["step"] == env.common_step_counter
    assert state["command_name"] == "twist" and state["asset_name"] == "robot"
    assert state["tau_s"] == .5 and state["step_dt"] == .02
    assert state["command"].shape == (1, 3)
    assert state["signed_residual"].shape == state["signed_ema"].shape == (1, 2)
    assert state["cost"].shape == (1,)
    assert not cost.requires_grad


def test_alternating_planar_residual_cancels_in_ema_vs_raw_l1():
    env = _Env(1)
    out = None
    for index in range(300):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        env.linear[0, 0] = -1.0 if index % 2 == 0 else 1.0
        out = mdp.body_planar_bias_cost(env)
    state = env._microduck_body_planar_bias_state
    assert state["signed_residual"].abs().sum().item() == 1.0
    assert out.item() < .03
    assert out.item() < state["signed_residual"].abs().sum().item() * .03


def test_positive_l1_cost_has_correct_negative_reward_weight_sign():
    env = _Env(1)
    env.command[0, :2] = torch.tensor([.2, -.1])
    cost = _step(env)
    weight = -1.0
    assert cost.item() > 0
    assert (weight * cost).item() < 0


def test_same_step_reads_are_idempotent_alias_free_and_detect_input_changes():
    env = _Env(1)
    env.command[0, 0] = .5
    first = _step(env)
    expected = first.item()
    first.zero_()
    env._microduck_body_planar_bias_state["signed_ema"].zero_()
    second = mdp.body_planar_bias_cost(env)
    assert second.item() == pytest.approx(expected)
    assert env._microduck_body_planar_bias_state["signed_ema"][0, 0].item() == pytest.approx(expected)
    assert second.data_ptr() != env._microduck_body_planar_bias_state["cost"].data_ptr()
    env.command[0, 2] = .1
    with pytest.raises(RuntimeError, match="inputs changed"):
        mdp.body_planar_bias_cost(env)


def test_per_world_reset_and_any_twist_change_clear_history():
    env = _Env(2)
    env.command[:, 0] = .2
    for _ in range(100):
        before = _step(env)
    after = _step(env, episode_length=[1, 101])
    assert after[0].item() == pytest.approx(.008, abs=1e-7)
    assert after[1].item() > before[1].item() - 1e-6
    # Yaw does not enter the planar residual, but changing it invalidates the
    # context used to accumulate the planar bias in that world.
    after = _step(env, command_yaw=.3)
    assert after[0].item() == pytest.approx(.008, abs=1e-7)
    assert after[1].item() == pytest.approx(.008, abs=1e-7)


def test_counter_rollback_resets_all_world_history():
    env = _Env(2)
    env.command[:, 0] = .2
    for _ in range(100):
        out = _step(env)
    assert torch.all(out > .15)
    env.common_step_counter = 0
    out = mdp.body_planar_bias_cost(env)
    assert torch.allclose(out, torch.tensor([.008, .008]), atol=1e-7)


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_tau_fails_closed(tau):
    with pytest.raises(ValueError, match="tau_s"):
        mdp.body_planar_bias_cost(_Env(), tau_s=tau)


def test_skipped_step_and_signature_changes_fail_closed():
    env = _Env(1)
    mdp.body_planar_bias_cost(env)
    env.common_step_counter += 1
    mdp.body_planar_bias_cost(env)
    env.common_step_counter += 2
    with pytest.raises(RuntimeError, match="skipped forward"):
        mdp.body_planar_bias_cost(env)

    env.common_step_counter = 2
    with pytest.raises(ValueError, match="incompatible config"):
        mdp.body_planar_bias_cost(env, tau_s=.25)


@pytest.mark.parametrize("damage", ["dt", "counter", "command_shape", "linear_shape", "episode", "worlds", "device", "dtype"])
def test_invalid_time_type_shapes_devices_and_dtypes_fail_closed(damage):
    env = _Env(2)
    if damage == "dt":
        env.step_dt = float("nan")
        pattern = "step_dt"
    elif damage == "counter":
        env.common_step_counter = True
        pattern = "common_step_counter"
    elif damage == "command_shape":
        env.command = torch.zeros(2, 4)
        pattern = "command must have shape"
    elif damage == "linear_shape":
        env.linear = torch.zeros(2, 2)
        env.scene["robot"].data.root_link_lin_vel_b = env.linear
        pattern = "body linear velocity"
    elif damage == "episode":
        env.episode_length_buf = torch.zeros(2, dtype=torch.float32)
        pattern = "episode_length_buf"
    elif damage == "worlds":
        env.num_envs = 3
        pattern = "num_envs"
    elif damage == "device":
        env.episode_length_buf = torch.zeros(2, dtype=torch.long, device="meta")
        pattern = "share floating dtype and device"
    else:
        env.linear = torch.zeros(2, 3, dtype=torch.float64)
        env.scene["robot"].data.root_link_lin_vel_b = env.linear
        pattern = "share floating dtype and device"
    with pytest.raises(ValueError, match=pattern):
        mdp.body_planar_bias_cost(env)


@pytest.mark.parametrize("target", ["command", "linear"])
def test_nonfinite_inputs_fail_before_cache(target):
    env = _Env(1)
    tensor = env.command if target == "command" else env.linear
    tensor[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite planar-bias"):
        mdp.body_planar_bias_cost(env)
    assert not hasattr(env, "_microduck_body_planar_bias_internal")


def test_config_validation_and_inputs_are_not_mutated():
    env = _Env(1)
    command_before = env.command.clone()
    linear_before = env.linear.clone()
    with pytest.raises(ValueError, match="command_name"):
        mdp.body_planar_bias_cost(env, command_name="")
    _step(env)
    assert torch.equal(env.command, command_before)
    assert torch.equal(env.linear, linear_before)
