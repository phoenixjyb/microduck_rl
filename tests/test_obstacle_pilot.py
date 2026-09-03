"""Static bounds for the first obstacle-curriculum training pilot."""

from pathlib import Path

import pytest

from mjlab_microduck.obstacle_pilot import (
    MAX_PILOT_ENVS,
    MAX_PILOT_ITERATIONS,
    PILOT_CHECKPOINT_INTERVAL,
    _stamp_intermediate_checkpoints,
    prepare_pilot_configs,
    run_pilot,
)
from mjlab_microduck.obstacle_protocol import OA0_TASK_ID, OA0R_TASK_ID


def test_pilot_config_is_local_only_and_uses_requested_shape_and_seed():
    env_cfg, agent_cfg = prepare_pilot_configs(128, 17)
    assert env_cfg.scene.num_envs == 128
    assert env_cfg.scene.terrain.num_envs == 128
    assert agent_cfg.seed == 17
    assert agent_cfg.logger == "tensorboard"
    assert agent_cfg.upload_model is False
    assert agent_cfg.save_interval == PILOT_CHECKPOINT_INTERVAL


def test_pilot_config_supports_the_assisted_stage():
    env_cfg, agent_cfg = prepare_pilot_configs(16, 23, OA0_TASK_ID)
    assert env_cfg.scene.num_envs == 16
    assert agent_cfg.seed == 23
    assert agent_cfg.experiment_name == "run_obstacle_assisted"


def test_pilot_config_supports_outcome_balanced_stage():
    _, agent_cfg = prepare_pilot_configs(16, 23, OA0R_TASK_ID)
    assert agent_cfg.experiment_name == "run_obstacle_assisted_outcome"


def test_pilot_rejects_unsupported_obstacle_task():
    with pytest.raises(ValueError, match="unsupported obstacle task"):
        prepare_pilot_configs(8, 42, "not-a-task")


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


def test_intermediate_checkpoints_retain_obstacle_migration_metadata(tmp_path: Path):
    import torch

    checkpoint = tmp_path / "model_8016.pt"
    torch.save({"infos": {"env_state": {"step": 18}}, "iter": 8016}, checkpoint)
    metadata = {"actor_dims": [61, 68], "critic_dims": [76, 83]}

    retained = _stamp_intermediate_checkpoints(tmp_path, metadata)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert payload["infos"]["env_state"] == {"step": 18}
    assert payload["infos"]["obstacle_warm_start"] == metadata
    assert retained == {"model_8016.pt": retained["model_8016.pt"]}
    assert len(retained["model_8016.pt"]) == 64
