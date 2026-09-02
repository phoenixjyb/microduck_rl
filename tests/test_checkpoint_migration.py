"""Tests for strict, function-preserving obstacle checkpoint migration."""

from pathlib import Path

import pytest
import torch

from mjlab_microduck.checkpoint_migration import (
    ACTOR_BASE_DIM,
    CRITIC_BASE_DIM,
    OBSTACLE_INPUT_DIM,
    migrate_checkpoint_file,
    migrate_obstacle_checkpoint,
)
from mjlab_microduck.tasks.obstacle_observation import OBSTACLE_OBSERVATION_DIM


def test_migration_dimension_matches_v1_obstacle_contract():
    assert OBSTACLE_INPUT_DIM == OBSTACLE_OBSERVATION_DIM


def _network_state(input_dim, output_dim):
    return {
        "obs_normalizer._mean": torch.randn(1, input_dim),
        "obs_normalizer._var": torch.rand(1, input_dim) + 0.5,
        "obs_normalizer._std": torch.rand(1, input_dim) + 0.5,
        "obs_normalizer.count": torch.tensor(100),
        "mlp.0.weight": torch.randn(8, input_dim),
        "mlp.0.bias": torch.randn(8),
        "mlp.2.weight": torch.randn(output_dim, 8),
        "mlp.2.bias": torch.randn(output_dim),
    }


def _checkpoint():
    actor = _network_state(ACTOR_BASE_DIM, 14)
    critic = _network_state(CRITIC_BASE_DIM, 1)
    optimizer = {
        "state": {
            1: {
                "step": torch.tensor(5.0),
                "exp_avg": torch.randn_like(actor["mlp.0.weight"]),
                "exp_avg_sq": torch.rand_like(actor["mlp.0.weight"]),
            },
            9: {
                "step": torch.tensor(5.0),
                "exp_avg": torch.randn_like(critic["mlp.0.weight"]),
                "exp_avg_sq": torch.rand_like(critic["mlp.0.weight"]),
            },
        },
        "param_groups": [{"params": [1, 9]}],
    }
    return {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": optimizer,
        "iter": 7998,
        "infos": {"env_state": {"example": 1}},
    }


def _first_layer_output(state, observation):
    normalized = (
        observation - state["obs_normalizer._mean"]
    ) / state["obs_normalizer._std"]
    return torch.nn.functional.linear(
        normalized, state["mlp.0.weight"], state["mlp.0.bias"]
    )


@pytest.mark.parametrize(
    "key, old_dim",
    [("actor_state_dict", ACTOR_BASE_DIM), ("critic_state_dict", CRITIC_BASE_DIM)],
)
def test_migration_preserves_function_for_zero_obstacle_input(key, old_dim):
    source = _checkpoint()
    migrated = migrate_obstacle_checkpoint(source, "abc123")
    old_state = source[key]
    new_state = migrated[key]
    observation = torch.randn(4, old_dim)
    expanded_observation = torch.cat(
        (observation, torch.zeros(4, OBSTACLE_OBSERVATION_DIM)), dim=1
    )
    torch.testing.assert_close(
        _first_layer_output(new_state, expanded_observation),
        _first_layer_output(old_state, observation),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(new_state["mlp.0.weight"][:, :old_dim], old_state["mlp.0.weight"])
    assert torch.count_nonzero(new_state["mlp.0.weight"][:, old_dim:]) == 0
    assert torch.count_nonzero(new_state["obs_normalizer._mean"][:, old_dim:]) == 0
    assert torch.all(new_state["obs_normalizer._var"][:, old_dim:] == 1.0)
    assert torch.all(new_state["obs_normalizer._std"][:, old_dim:] == 1.0)


def test_migration_preserves_and_expands_optimizer_moments():
    source = _checkpoint()
    migrated = migrate_obstacle_checkpoint(source, "abc123")
    for parameter_id, old_dim in ((1, ACTOR_BASE_DIM), (9, CRITIC_BASE_DIM)):
        old = source["optimizer_state_dict"]["state"][parameter_id]
        new = migrated["optimizer_state_dict"]["state"][parameter_id]
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(new[name][:, :old_dim], old[name])
            assert torch.count_nonzero(new[name][:, old_dim:]) == 0
    assert migrated["iter"] == 7998
    assert migrated["infos"]["env_state"] == source["infos"]["env_state"]


def test_migration_records_auditable_provenance():
    migrated = migrate_obstacle_checkpoint(_checkpoint(), "abc123")
    metadata = migrated["infos"]["obstacle_warm_start"]
    assert metadata["source_sha256"] == "abc123"
    assert metadata["source_iteration"] == 7998
    assert metadata["actor_dims"] == [61, 68]
    assert metadata["critic_dims"] == [76, 83]


def test_file_migration_refuses_to_overwrite_and_leaves_source_unchanged(tmp_path: Path):
    source = tmp_path / "source.pt"
    destination = tmp_path / "migrated.pt"
    torch.save(_checkpoint(), source)
    source_bytes = source.read_bytes()
    migrate_checkpoint_file(source, destination)
    assert source.read_bytes() == source_bytes
    assert destination.exists()
    with pytest.raises(FileExistsError):
        migrate_checkpoint_file(source, destination)


def test_migration_rejects_unexpected_dimensions():
    source = _checkpoint()
    source["actor_state_dict"]["mlp.0.weight"] = torch.randn(8, 60)
    with pytest.raises(ValueError, match="second dimension 61"):
        migrate_obstacle_checkpoint(source, "abc123")
