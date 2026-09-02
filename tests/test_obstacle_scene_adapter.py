"""Tests for the simulator-to-obstacle-contract scene adapter."""

import math

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Data:
    def __init__(self, position, velocity, quaternion=None):
        self.root_link_pos_w = torch.tensor(position, dtype=torch.float32)
        self.root_link_lin_vel_w = torch.tensor(velocity, dtype=torch.float32)
        if quaternion is not None:
            self.root_link_quat_w = torch.tensor(quaternion, dtype=torch.float32)


class _Entity:
    def __init__(self, position, velocity, quaternion=None):
        self.data = _Data(position, velocity, quaternion)


class _Scene:
    def __init__(self, robot, obstacle):
        self.entities = {"robot": robot, "obstacle": obstacle}

    def __getitem__(self, name):
        return self.entities[name]


class _Env:
    def __init__(self, robot, obstacle):
        self.scene = _Scene(robot, obstacle)


def _env(obstacle_position, obstacle_velocity=(0.0, 0.0, 0.0)):
    robot = _Entity(
        position=[(0.0, 0.0, 0.0)],
        velocity=[(0.2, 0.0, 0.0)],
        quaternion=[(1.0, 0.0, 0.0, 0.0)],
    )
    obstacle = _Entity(
        position=[obstacle_position],
        velocity=[obstacle_velocity],
    )
    return _Env(robot, obstacle)


def test_scene_adapter_transforms_geometry_and_relative_velocity():
    out = microduck_mdp.obstacle_geometry_observation(
        _env((1.0, 0.0, 0.1)), width_m=0.2, height_m=0.1
    )
    # The static obstacle approaches the robot at its 0.2 m/s forward speed.
    expected = torch.tensor([[0.45, 0.0, 1.0, 0.4, 0.4, 0.1, 1.0]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


def test_scene_adapter_hides_obstacles_outside_range_or_field_of_view():
    too_far = microduck_mdp.obstacle_geometry_observation(
        _env((2.1, 0.0, 0.1)), max_range_m=2.0
    )
    behind = microduck_mdp.obstacle_geometry_observation(
        _env((-0.5, 0.0, 0.1)), horizontal_fov_rad=math.pi
    )
    assert torch.equal(too_far, torch.zeros_like(too_far))
    assert torch.equal(behind, torch.zeros_like(behind))


def test_scene_adapter_keeps_obstacle_at_field_of_view_edge():
    out = microduck_mdp.obstacle_geometry_observation(
        _env((1.0, 1.0, 0.1)), horizontal_fov_rad=math.pi / 2.0
    )
    assert float(out[0, -1]) == 1.0
    torch.testing.assert_close(
        out[0, 1:3],
        torch.tensor([math.sqrt(0.5), math.sqrt(0.5)]),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("horizontal_fov_rad", [0.0, -0.1, 2.0 * math.pi + 0.1])
def test_scene_adapter_rejects_invalid_field_of_view(horizontal_fov_rad):
    with pytest.raises(ValueError, match="horizontal_fov_rad"):
        microduck_mdp.obstacle_geometry_observation(
            _env((1.0, 0.0, 0.1)), horizontal_fov_rad=horizontal_fov_rad
        )
