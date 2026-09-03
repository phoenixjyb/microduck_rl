import pytest
import torch

from mjlab_microduck.hierarchical_obstacle import SUPERVISOR_OBSERVATION_DIM
from mjlab_microduck.obstacle_supervisor_bc import (
    HC4R2_STAGE,
    LateralGatedSupervisor,
    ObstacleSupervisor,
    split_episode_keys,
    train_supervisor,
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


def test_train_supervisor_rejects_unknown_stage_before_dataset_access(tmp_path):
    with pytest.raises(ValueError, match="unsupported behavioral-cloning stage"):
        train_supervisor(
            (tmp_path / "missing.pt",),
            tmp_path / "supervisor.pt",
            epochs=1,
            stage="HC4L-unreviewed-stage",
        )


def test_hc4r2_training_requires_teacher_and_student_state_datasets(tmp_path):
    teacher_dataset = tmp_path / "teacher.pt"
    torch.save({"stage": "HC1-successful-teacher-trajectories"}, teacher_dataset)

    with pytest.raises(ValueError, match="requires both HC1 teacher"):
        train_supervisor(
            (teacher_dataset,),
            tmp_path / "supervisor.pt",
            epochs=1,
            stage=HC4R2_STAGE,
        )


def test_lateral_gate_routes_center_shifted_and_invalid_observations():
    class FixedSupervisor(torch.nn.Module):
        def __init__(self, command):
            super().__init__()
            self.register_buffer("command", torch.tensor(command))

        def forward(self, observation):
            return self.command.expand(observation.shape[0], -1)

    supervisor = LateralGatedSupervisor(
        FixedSupervisor((0.25, 0.10)),
        FixedSupervisor((0.50, -0.20)),
        lateral_gate_m=0.06,
    )
    observation = torch.zeros(4, SUPERVISOR_OBSERVATION_DIM)
    observation[:, 3] = 1.0
    observation[:, 7] = 1.0
    observation[1, 8] = 0.12 / 0.75
    observation[2, 8] = -0.12 / 0.75
    observation[3, 8] = 0.12 / 0.75
    observation[3, 7] = 0.0

    command = supervisor(observation)

    torch.testing.assert_close(command[0], torch.tensor((0.25, 0.10)))
    torch.testing.assert_close(command[1], torch.tensor((0.50, -0.20)))
    torch.testing.assert_close(command[2], torch.tensor((0.50, -0.20)))
    torch.testing.assert_close(command[3], torch.tensor((0.25, 0.10)))
