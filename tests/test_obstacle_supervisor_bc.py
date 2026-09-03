import torch

from mjlab_microduck.hierarchical_obstacle import SUPERVISOR_OBSERVATION_DIM
from mjlab_microduck.obstacle_supervisor_bc import (
    ObstacleSupervisor,
    split_episode_keys,
)


def test_supervisor_output_is_bounded_normalized_command():
    model = ObstacleSupervisor()
    output = model(torch.randn(8, SUPERVISOR_OBSERVATION_DIM))
    assert output.shape == (8, 2)
    assert torch.all((output[:, 0] >= 0.0) & (output[:, 0] <= 1.0))
    assert torch.all((output[:, 1] >= -1.0) & (output[:, 1] <= 1.0))


def test_episode_split_never_leaks_one_episode_between_partitions():
    keys = torch.tensor([1, 1, 1, 2, 2, 3, 3, 4])
    train, validation = split_episode_keys(keys, 0.25, seed=42)
    assert torch.all(train ^ validation)
    train_keys = set(keys[train].tolist())
    validation_keys = set(keys[validation].tolist())
    assert train_keys.isdisjoint(validation_keys)
