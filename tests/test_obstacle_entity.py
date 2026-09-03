"""Tests for the first simulated obstacle entity and reset contract."""

from types import SimpleNamespace

import mujoco
import pytest
import torch

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_OBSTACLE_CFG,
    MICRODUCK_OBSTACLE_XML,
    get_obstacle_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_obstacle_xml_compiles_and_cfg_uses_it():
    assert MICRODUCK_OBSTACLE_XML.exists()
    spec = get_obstacle_spec()
    model = spec.compile()
    assert isinstance(model, mujoco.MjModel)
    assert model.nbody == 2
    assert model.ngeom == 1
    assert model.njnt == 1
    assert MICRODUCK_OBSTACLE_CFG.spec_fn is get_obstacle_spec


class _Obstacle:
    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.pose = pose.clone()
        self.pose_env_ids = env_ids.clone()

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.velocity = velocity.clone()
        self.velocity_env_ids = env_ids.clone()


class _Scene:
    def __init__(self, robot, obstacle, num_envs=2):
        self.entities = {"robot": robot, "obstacle": obstacle}
        self.terrain = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))

    def __getitem__(self, name):
        return self.entities[name]


def _reset_env(num_envs=2):
    robot = SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=slice(0, 7)))
    obstacle = _Obstacle()
    # Env 0 at origin facing +x; env 1 at (1, 2) facing +y.
    qpos = torch.zeros(num_envs, 7)
    qpos[:, 2] = 0.12
    qpos[:, 3] = 1.0
    if num_envs >= 2:
        qpos[1] = torch.tensor(
            [1.0, 2.0, 0.12, 2**-0.5, 0.0, 0.0, 2**-0.5]
        )
    env = SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        scene=_Scene(robot, obstacle, num_envs),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
    )
    return env, obstacle


def test_reset_places_obstacle_in_robot_yaw_frame_and_zeros_velocity():
    env, obstacle = _reset_env()
    microduck_mdp.reset_obstacle_ahead(
        env,
        torch.tensor([0, 1]),
        forward_range_m=(0.8, 0.8),
        lateral_range_m=(0.2, 0.2),
        obstacle_height_m=0.1,
    )
    expected_position = torch.tensor([[0.8, 0.2, 0.05], [0.8, 2.8, 0.05]])
    torch.testing.assert_close(obstacle.pose[:, :3], expected_position, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        obstacle.pose[:, 3:],
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]]
        ),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.equal(obstacle.velocity, torch.zeros(2, 6))
    assert torch.equal(obstacle.pose_env_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(
        env._obstacle_path_dir_w,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        env._obstacle_route_origin_w,
        torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_reset_can_sample_both_sides_outside_a_center_exclusion_band():
    env, obstacle = _reset_env(num_envs=256)
    torch.manual_seed(7)
    microduck_mdp.reset_obstacle_ahead(
        env,
        torch.arange(256),
        forward_range_m=(1.15, 1.15),
        lateral_abs_range_m=(0.24, 0.30),
    )
    lateral_dir = torch.stack(
        (-env._obstacle_path_dir_w[:, 1], env._obstacle_path_dir_w[:, 0]), dim=-1
    )
    lateral = (
        (obstacle.pose[:, :2] - env._obstacle_route_origin_w) * lateral_dir
    ).sum(dim=-1)
    assert torch.all(lateral.abs() >= 0.24)
    assert torch.all(lateral.abs() <= 0.30)
    assert torch.any(lateral < 0.0)
    assert torch.any(lateral > 0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"forward_range_m": (-0.1, 0.8)}, "forward_range_m"),
        ({"forward_range_m": (1.0, 0.8)}, "forward_range_m"),
        ({"lateral_range_m": (0.2, -0.2)}, "lateral_range_m"),
        ({"lateral_abs_range_m": (-0.1, 0.2)}, "lateral_abs_range_m"),
        ({"lateral_abs_range_m": (0.3, 0.2)}, "lateral_abs_range_m"),
        ({"obstacle_height_m": 0.0}, "obstacle_height_m"),
    ],
)
def test_reset_rejects_invalid_placement_ranges(kwargs, message):
    env, _ = _reset_env()
    with pytest.raises(ValueError, match=message):
        microduck_mdp.reset_obstacle_ahead(env, torch.tensor([0]), **kwargs)
