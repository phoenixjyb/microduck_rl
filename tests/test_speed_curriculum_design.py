"""Verify proposal arithmetic against the installed reward, not a trained gait."""

import math
from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

from mjlab_microduck.speed_response_control import prepare_config


def test_saved_style_speed_reward_is_not_a_tight_tracking_acceptance_gate():
    cfg, _ = prepare_config()
    term = cfg.rewards["track_linear_velocity"]
    assert term.func is track_linear_velocity
    assert term.weight == 4.5 and term.params["std"] == pytest.approx(math.sqrt(.15))
    speeds = torch.tensor([.21, .27, .30], dtype=torch.float64)
    velocity = torch.stack((speeds, torch.zeros_like(speeds), torch.zeros_like(speeds)), -1)
    command = torch.tensor([.3, 0., 0.], dtype=torch.float64).expand(3, 3)
    env = NS(scene={"robot": NS(data=NS(root_link_lin_vel_b=velocity))},
             command_manager=NS(get_command=lambda name: command))
    actual = term.func(env, **term.params)
    expected = torch.exp(-(.3 - speeds).square() / .15)
    torch.testing.assert_close(actual, expected)
    assert actual[0].item() == pytest.approx(.9474321065)
    assert actual[1].item() == pytest.approx(.9940179641)
    assert actual[2].item() == 1.
    assert .3 - speeds[0].item() > .03  # high reward still fails speed tolerance


def test_reward_arithmetic_must_not_ignore_lateral_and_vertical_motion():
    command = torch.tensor([[.3, 0., 0.]])
    actual = torch.tensor([[.21, .1, .1]])
    env = NS(scene={"robot": NS(data=NS(root_link_lin_vel_b=actual))},
             command_manager=NS(get_command=lambda name: command))
    reward = track_linear_velocity(env, math.sqrt(.15), "twist")
    assert reward.item() == pytest.approx(math.exp(-(.09**2 + .1**2 + .1**2) / .15))
    assert reward.item() < math.exp(-.09**2 / .15)
