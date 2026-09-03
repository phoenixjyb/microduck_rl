import pytest

from mjlab_microduck.hierarchical_obstacle_rollout import (
    MAX_CASES,
    prepare_rollout_configs,
    validate_rollout_bounds,
)


def test_rollout_config_keeps_obstacle_physics_but_restores_base_observation():
    env_cfg, agent_cfg = prepare_rollout_configs(8, 0.3, 1.15, -0.27)
    assert "obstacle" in env_cfg.scene.entities
    assert "reset_obstacle" in env_cfg.events
    assert "obstacle_collision" in env_cfg.terminations
    assert "obstacle" not in env_cfg.observations["actor"].terms
    assert "obstacle_ground_truth" not in env_cfg.observations["critic"].terms
    assert env_cfg.events["reset_obstacle"].params["forward_range_m"] == (1.15, 1.15)
    assert env_cfg.events["reset_obstacle"].params["lateral_range_m"] == (-0.27, -0.27)
    assert "lateral_abs_range_m" not in env_cfg.events["reset_obstacle"].params
    assert env_cfg.commands["twist"].rel_forward_envs == 0.0
    assert agent_cfg.experiment_name == "run_motor_aware"


def test_rollout_bounds_accept_small_matrix():
    validate_rollout_bounds(64, 400, (0.3,), (1.15,), (-0.27, 0.27), (41,))


@pytest.mark.parametrize(
    "args",
    [
        (0, 10, (0.3,), (1.15,), (0.0,), (41,)),
        (1, 0, (0.3,), (1.15,), (0.0,), (41,)),
        (1, 10, (), (1.15,), (0.0,), (41,)),
        (1, 10, (0.9,), (1.15,), (0.0,), (41,)),
        (1, 10, (0.3,), (), (0.0,), (41,)),
        (1, 10, (0.3,), (0.0,), (0.0,), (41,)),
        (1, 10, (0.3,), (1.15,), (), (41,)),
        (1, 10, (0.3,), (1.15,), (0.0,), ()),
    ],
)
def test_rollout_bounds_reject_invalid_inputs(args):
    with pytest.raises(ValueError):
        validate_rollout_bounds(*args)


def test_rollout_bounds_reject_too_many_cases():
    speeds = tuple(0.1 for _ in range(MAX_CASES + 1))
    with pytest.raises(ValueError, match="case count"):
        validate_rollout_bounds(1, 1, speeds, (1.0,), (0.0,), (1,))
