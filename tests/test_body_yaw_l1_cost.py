from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


class _Env:
    def __init__(self, worlds=3):
        self.num_envs = worlds
        self.device = "cpu"
        self.command = torch.zeros(worlds, 3)
        self.angular_b = torch.zeros(worlds, 3)
        self.command_manager = SimpleNamespace(get_command=lambda _name: self.command)
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_ang_vel_b=self.angular_b,
            root_link_ang_vel_w=torch.full((worlds, 3), 123.))) }


def test_exact_yaw_match_costs_zero_and_signed_residuals_have_equal_l1_cost():
    env = _Env()
    env.command[:, 2] = torch.tensor([.25, -.25, .4])
    env.angular_b[:, 2] = torch.tensor([.25, .05, -.1])
    cost = mdp.body_yaw_tracking_l1_cost(env)
    assert cost.tolist() == pytest.approx([0., .30, .50])
    assert bool((cost >= 0).all())


def test_cost_uses_only_body_frame_yaw_and_ignores_translation_and_roll_pitch():
    env = _Env(2)
    env.command.copy_(torch.tensor([[.2, 0., .3], [.4, 0., -.2]]))
    env.angular_b.copy_(torch.tensor([[9., -8., -.1], [-7., 6., .1]]))
    before = mdp.body_yaw_tracking_l1_cost(env)
    env.command[:, :2] = torch.tensor([[1000., -2000.], [-3000., 4000.]])
    env.angular_b[:, :2] = torch.tensor([[-5000., 6000.], [7000., -8000.]])
    after = mdp.body_yaw_tracking_l1_cost(env)
    assert after.tolist() == pytest.approx([.4, .3])
    assert torch.equal(before, after)


def test_function_is_read_only_deterministic_and_does_not_advance_rng_or_create_ema():
    env = _Env()
    env.command.copy_(torch.tensor([[.3, 0., .2], [.2, 0., -.2], [.4, 0., 0.]]))
    env.angular_b.copy_(torch.tensor([[2., 3., -.1], [4., 5., .1], [6., 7., .8]]))
    command, angular = env.command.clone(), env.angular_b.clone()
    state = torch.random.get_rng_state().clone()
    first = mdp.body_yaw_tracking_l1_cost(env)
    second = mdp.body_yaw_tracking_l1_cost(env)
    assert torch.equal(first, second)
    assert torch.equal(env.command, command) and torch.equal(env.angular_b, angular)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not any(name.startswith("_microduck_body_yaw") for name in vars(env))


@pytest.mark.parametrize("command,angular", [
    (torch.zeros(2, 2), torch.zeros(2, 3)),
    (torch.zeros(2, 3), torch.zeros(2, 2)),
    (torch.zeros(2, 3, 1), torch.zeros(2, 3, 1)),
    (torch.zeros(2, 3), torch.zeros(1, 3)),
])
def test_invalid_tensor_shapes_fail_closed(command, angular):
    env = _Env(2)
    env.command = command
    env.angular_b = angular
    env.scene["robot"].data.root_link_ang_vel_b = angular
    with pytest.raises(FloatingPointError, match="invalid raw body yaw tracking state"):
        mdp.body_yaw_tracking_l1_cost(env)


@pytest.mark.parametrize("where", ["command", "angular"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_is_rejected_without_sanitizing(where, value):
    env = _Env(1)
    if where == "command":
        env.command[0, 0] = value
    else:
        env.angular_b[0, 1] = value
    with pytest.raises(FloatingPointError, match="invalid raw body yaw tracking state"):
        mdp.body_yaw_tracking_l1_cost(env)


def test_overflowed_yaw_residual_fails_closed():
    env = _Env(1)
    env.command[0, 2] = 3e38
    env.angular_b[0, 2] = -3e38
    with pytest.raises(FloatingPointError, match="nonfinite raw body yaw tracking cost"):
        mdp.body_yaw_tracking_l1_cost(env)
