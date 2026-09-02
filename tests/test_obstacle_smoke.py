"""Static bounds for the retained obstacle runner smoke."""

import pytest

from mjlab_microduck.obstacle_smoke import (
    MAX_SMOKE_ENVS,
    MAX_SMOKE_ITERATIONS,
    prepare_smoke_configs,
    run_smoke,
)


def test_smoke_configs_are_bounded_and_disable_external_uploads():
    env_cfg, agent_cfg = prepare_smoke_configs(8)
    assert env_cfg.scene.num_envs == 8
    assert env_cfg.scene.terrain.num_envs == 8
    assert agent_cfg.logger == "tensorboard"
    assert agent_cfg.upload_model is False


@pytest.mark.parametrize("num_envs", [0, MAX_SMOKE_ENVS + 1])
def test_smoke_rejects_unbounded_environment_count(num_envs):
    with pytest.raises(ValueError, match="num_envs"):
        prepare_smoke_configs(num_envs)


@pytest.mark.parametrize("iterations", [0, MAX_SMOKE_ITERATIONS + 1])
def test_smoke_rejects_unbounded_iterations(tmp_path, iterations):
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="iterations"):
        run_smoke(checkpoint, tmp_path / "output", iterations=iterations)
