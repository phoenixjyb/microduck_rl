"""Static bounds and config tests for the obstacle baseline evaluator."""

import pytest

from mjlab_microduck.obstacle_baseline import (
    MAX_BASELINE_ENVS,
    MAX_BASELINE_SEEDS,
    MAX_BASELINE_STEPS,
    prepare_baseline_configs,
    validate_baseline_bounds,
)


def test_baseline_config_uses_deterministic_straight_command():
    env_cfg, agent_cfg = prepare_baseline_configs(8, 0.5)
    twist = env_cfg.commands["twist"]
    assert env_cfg.scene.num_envs == 8
    assert env_cfg.scene.terrain.num_envs == 8
    assert twist.ranges.lin_vel_x == (0.5, 0.5)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert twist.heading_command is False
    assert twist.rel_standing_envs == 0.0
    assert agent_cfg.upload_model is False


@pytest.mark.parametrize(
    "num_envs, steps, seeds",
    [
        (0, 10, (1,)),
        (MAX_BASELINE_ENVS + 1, 10, (1,)),
        (1, 0, (1,)),
        (1, MAX_BASELINE_STEPS + 1, (1,)),
        (1, 10, ()),
        (1, 10, tuple(range(MAX_BASELINE_SEEDS + 1))),
    ],
)
def test_baseline_rejects_unbounded_work(num_envs, steps, seeds):
    with pytest.raises(ValueError):
        validate_baseline_bounds(num_envs, steps, seeds)


def test_baseline_rejects_nonpositive_speed():
    with pytest.raises(ValueError, match="speed_mps"):
        prepare_baseline_configs(8, 0.0)
