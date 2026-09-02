"""Static bounds for the first obstacle-curriculum training pilot."""

from pathlib import Path

import pytest

from mjlab_microduck.obstacle_pilot import (
    MAX_PILOT_ENVS,
    MAX_PILOT_ITERATIONS,
    prepare_pilot_configs,
    run_pilot,
)


def test_pilot_config_is_local_only_and_uses_requested_shape_and_seed():
    env_cfg, agent_cfg = prepare_pilot_configs(128, 17)
    assert env_cfg.scene.num_envs == 128
    assert env_cfg.scene.terrain.num_envs == 128
    assert agent_cfg.seed == 17
    assert agent_cfg.logger == "tensorboard"
    assert agent_cfg.upload_model is False


@pytest.mark.parametrize("num_envs", [0, MAX_PILOT_ENVS + 1])
def test_pilot_rejects_unbounded_environment_count(num_envs):
    with pytest.raises(ValueError, match="num_envs"):
        prepare_pilot_configs(num_envs, 42)


@pytest.mark.parametrize("iterations", [0, MAX_PILOT_ITERATIONS + 1])
def test_pilot_rejects_unbounded_iteration_count(tmp_path: Path, iterations: int):
    with pytest.raises(ValueError, match="iterations"):
        run_pilot(tmp_path / "missing.pt", tmp_path / "out", iterations=iterations)


def test_pilot_refuses_to_reuse_output_directory(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        run_pilot(checkpoint, output_dir)
