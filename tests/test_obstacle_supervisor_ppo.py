import pytest
import torch

from mjlab_microduck.hierarchical_obstacle import ObstaclePhase
from mjlab_microduck.obstacle_supervisor_ppo import (
    Hc3PpoCfg,
    generalized_advantage_estimate,
    hc3_reward,
    normalized_command_from_latent,
)


def _reward(*, phase: ObstaclePhase, measured_speed: float) -> torch.Tensor:
    zeros = torch.zeros(1)
    false = torch.zeros(1, dtype=torch.bool)
    return hc3_reward(
        zeros,
        zeros,
        zeros,
        torch.tensor([measured_speed]),
        torch.tensor([0.5]),
        torch.ones(1),
        torch.tensor([int(phase)]),
        false,
        false,
        false,
        false,
        false,
    )


def test_interaction_reward_has_no_nominal_speed_tracking_pressure():
    stopped = _reward(phase=ObstaclePhase.INTERACTION, measured_speed=0.0)
    nominal = _reward(phase=ObstaclePhase.INTERACTION, measured_speed=0.5)
    torch.testing.assert_close(stopped, nominal)


def test_approach_and_recovery_keep_nominal_speed_tracking_pressure():
    for phase in (ObstaclePhase.APPROACH, ObstaclePhase.RECOVERY):
        stopped = _reward(phase=phase, measured_speed=0.0)
        nominal = _reward(phase=phase, measured_speed=0.5)
        assert float(nominal) > float(stopped)


def test_terminal_outcomes_dominate_nonterminal_shaping():
    zeros = torch.zeros(3)
    false = torch.zeros(3, dtype=torch.bool)
    reward = hc3_reward(
        torch.full((3,), 0.1),
        zeros,
        zeros,
        zeros,
        torch.full((3,), 0.5),
        torch.ones(3),
        torch.full((3,), int(ObstaclePhase.INTERACTION)),
        torch.tensor([False, True, False]),
        torch.tensor([True, False, False]),
        torch.tensor([False, False, True]),
        false,
        false,
    )
    assert reward.tolist() == pytest.approx([8.0, -12.0, -4.0])


def test_latent_transform_obeys_normalized_command_bounds():
    command = normalized_command_from_latent(
        torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
    )
    assert torch.all((command[:, 0] >= 0.0) & (command[:, 0] <= 1.0))
    assert torch.all((command[:, 1] >= -1.0) & (command[:, 1] <= 1.0))


def test_gae_stops_bootstrap_across_terminal_transition():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[True], [False]])
    advantage, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        torch.tensor([3.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantage[:, 0], torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(returns, advantage)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"rollout_steps": 63, "minibatches": 4},
        {"learning_rate": 0.0},
        {"gamma": 0.0},
        {"clip_ratio": 1.0},
        {"initial_log_std": (-1.0,)},
    ],
)
def test_hc3_config_rejects_unsafe_values(kwargs):
    with pytest.raises(ValueError):
        Hc3PpoCfg(**kwargs)
