"""Tests for obstacle clearance, collision, and passed instrumentation."""

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Scene:
    def __init__(self, robot_xy, obstacle_xy):
        self.entities = {
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor(
                        [[*xy, 0.1] for xy in robot_xy], dtype=torch.float32
                    )
                )
            ),
            "obstacle": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor(
                        [[*xy, 0.05] for xy in obstacle_xy], dtype=torch.float32
                    )
                )
            ),
        }

    def __getitem__(self, name):
        return self.entities[name]


def _env(robot_xy, obstacle_xy):
    count = len(robot_xy)
    return SimpleNamespace(
        scene=_Scene(robot_xy, obstacle_xy),
        num_envs=count,
        device="cpu",
        extras={"log": {}},
        _obstacle_path_dir_w=torch.tensor([[1.0, 0.0]]).repeat(count, 1),
    )


def test_collision_uses_conservative_planar_envelope():
    env = _env([(0.0, 0.0), (0.0, 0.0)], [(0.22, 0.0), (0.221, 0.0)])
    assert torch.equal(
        microduck_mdp.obstacle_collision(env), torch.tensor([True, False])
    )


def test_clearance_cost_is_zero_at_margin_and_one_at_collision():
    env = _env(
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        [(0.37, 0.0), (0.295, 0.0), (0.22, 0.0)],
    )
    cost = microduck_mdp.obstacle_clearance_cost(env)
    torch.testing.assert_close(cost, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6, rtol=0.0)
    assert "Metrics/obstacle_clearance_mean_m" in env.extras["log"]
    assert "Metrics/obstacle_collision_fraction" in env.extras["log"]


def test_passed_reward_uses_fixed_reset_heading_and_requires_clearance():
    env = _env(
        [(0.23, 0.0), (0.10, 0.0), (0.23, 0.0)],
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.20)],
    )
    # First passed cleanly, second has insufficient progress, third also passed
    # because the fixed +x path direction ignores lateral sign.
    expected = torch.tensor([True, False, True])
    assert torch.equal(microduck_mdp.obstacle_passed(env), expected)
    assert torch.equal(microduck_mdp.obstacle_passed_reward(env), expected.float())
    assert "Metrics/obstacle_passed_fraction" in env.extras["log"]


def test_route_progress_tracks_command_without_rewarding_standing_or_overspeed():
    env = _env(
        [(0.0, 0.0)] * 4,
        [(1.0, 0.0)] * 4,
    )
    env.scene["robot"].data.root_link_lin_vel_w = torch.tensor(
        [[0.25, 0.0, 0.0], [0.10, 0.0, 0.0], [0.50, 0.0, 0.0], [-0.10, 0.0, 0.0]]
    )
    commands = torch.tensor(
        [[0.50, 0.0, 0.0], [0.20, 0.0, 0.0], [0.00, 0.0, 0.0], [0.20, 0.0, 0.0]]
    )
    env.command_manager = SimpleNamespace(get_command=lambda name: commands)
    reward = microduck_mdp.obstacle_route_progress_reward(env)
    torch.testing.assert_close(
        reward,
        torch.tensor([0.5, 0.5, 0.0, 0.0]),
        atol=1e-6,
        rtol=0.0,
    )
    assert "Metrics/obstacle_route_speed_mean_mps" in env.extras["log"]


def test_route_progress_rejects_negative_command_threshold():
    with pytest.raises(ValueError, match="command_threshold"):
        microduck_mdp.obstacle_route_progress_reward(
            _env([(0.0, 0.0)], [(1.0, 0.0)]),
            command_threshold=-0.01,
        )


def test_lateral_excursion_cost_uses_episode_origin_and_soft_corridor():
    env = _env(
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
    )
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    microduck_mdp.obstacle_lateral_excursion_cost(env)

    env.scene["robot"].data.root_link_pos_w[:, 1] = torch.tensor([0.45, 0.60, 0.80])
    env.episode_length_buf.fill_(2)
    cost = microduck_mdp.obstacle_lateral_excursion_cost(env)

    torch.testing.assert_close(
        cost, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6, rtol=0.0
    )
    assert "Metrics/obstacle_lateral_excursion_mean_m" in env.extras["log"]
    assert "Metrics/obstacle_corridor_exceed_fraction" in env.extras["log"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"soft_limit_m": -0.01}, "soft_limit_m"),
        ({"soft_limit_m": 0.45, "hard_limit_m": 0.45}, "hard_limit_m"),
    ],
)
def test_lateral_excursion_cost_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        microduck_mdp.obstacle_lateral_excursion_cost(
            _env([(0.0, 0.0)], [(1.0, 0.0)]), **kwargs
        )


@pytest.mark.parametrize(
    "function, kwargs, message",
    [
        (microduck_mdp.obstacle_collision, {"robot_radius_m": 0.0}, "radii"),
        (microduck_mdp.obstacle_clearance_cost, {"margin_m": 0.0}, "margin_m"),
    ],
)
def test_obstacle_envelope_rejects_invalid_physical_values(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(_env([(0.0, 0.0)], [(1.0, 0.0)]), **kwargs)
