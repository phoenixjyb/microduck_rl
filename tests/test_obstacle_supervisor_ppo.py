import pytest
import torch

from mjlab_microduck.hierarchical_obstacle import ObstaclePhase
from mjlab_microduck.obstacle_supervisor_bc import (
    ObstacleSupervisor,
    interaction_speed_only_command,
)
from mjlab_microduck.obstacle_supervisor_ppo import (
    HC3D_BALANCED_CELLS,
    Hc3PpoCfg,
    balanced_cell_assignment,
    configure_interaction_speed_only_actor,
    frozen_hc2_authority_is_exact,
    generalized_advantage_estimate,
    hc3_reward,
    normalized_command_from_latent,
    restore_frozen_hc2_yaw,
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


def test_hc3e_command_has_only_bounded_interaction_speed_authority():
    observation = torch.zeros(3, 17)
    observation[:, 0] = torch.tensor([0.625, 1.0, 0.625])
    observation[0, -4] = 1.0
    observation[1, -3] = 1.0
    observation[2, -2] = 1.0
    hc2_command = torch.tensor([[0.2, -0.4], [0.3, 0.6], [0.4, -0.2]])
    command = interaction_speed_only_command(
        observation,
        torch.tensor([[-100.0], [-100.0], [100.0]]),
        hc2_command,
    )
    torch.testing.assert_close(command[:, 0], torch.tensor([0.2, 0.375, 0.4]))
    torch.testing.assert_close(command[:, 1], hc2_command[:, 1])


def test_hc3e_interaction_speed_cannot_exceed_nominal():
    observation = torch.zeros(1, 17)
    observation[:, 0] = 0.625
    observation[:, -3] = 1.0
    command = interaction_speed_only_command(
        observation,
        torch.tensor([[100.0]]),
        torch.tensor([[0.4, 0.25]]),
    )
    torch.testing.assert_close(command, torch.tensor([[0.625, 0.25]]))


def test_hc3e_restores_exact_hc2_yaw_head():
    actor = ObstacleSupervisor()
    anchor = ObstacleSupervisor()
    anchor.load_state_dict(actor.state_dict())
    assert frozen_hc2_authority_is_exact(actor, anchor)
    trainable = configure_interaction_speed_only_actor(actor)
    assert set(trainable) == {actor.network[-1].weight, actor.network[-1].bias}
    assert all(
        not parameter.requires_grad
        for parameter in actor.network[:-1].parameters()
    )
    with torch.no_grad():
        actor.network[-1].weight.add_(1.0)
        actor.network[-1].bias.add_(1.0)
    restore_frozen_hc2_yaw(actor, anchor)
    assert frozen_hc2_authority_is_exact(actor, anchor)
    torch.testing.assert_close(
        actor.network[-1].weight[1], anchor.network[-1].weight[1]
    )
    torch.testing.assert_close(actor.network[-1].bias[1], anchor.network[-1].bias[1])
    assert not torch.equal(actor.network[-1].weight[0], anchor.network[-1].weight[0])


def test_hc3d_assignment_balances_all_retained_hc2_cells():
    cell_index, speed, forward = balanced_cell_assignment(
        10, HC3D_BALANCED_CELLS, device="cpu"
    )
    assert torch.bincount(cell_index).tolist() == [3, 3, 2, 2]
    torch.testing.assert_close(
        torch.stack((speed[:4], forward[:4]), dim=-1),
        torch.tensor(HC3D_BALANCED_CELLS),
    )


def test_hc3d_assignment_rejects_missing_cell_coverage():
    with pytest.raises(ValueError, match="cover every"):
        balanced_cell_assignment(3, HC3D_BALANCED_CELLS, device="cpu")


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
