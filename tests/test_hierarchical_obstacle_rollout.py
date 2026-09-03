import hashlib
from dataclasses import asdict

import pytest
import torch

from mjlab_microduck.hierarchical_obstacle_rollout import (
    HC1_ATTEMPT_TIMEOUT_S,
    MAX_CASES,
    load_learned_supervisor,
    prepare_rollout_configs,
    validate_rollout_bounds,
)
from mjlab_microduck.obstacle_supervisor_bc import (
    InteractionSpeedOnlySupervisor,
    ObstacleSupervisor,
    SupervisorBcCfg,
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
    assert (
        env_cfg.terminations["obstacle_attempt_timeout"].params[
            "max_attempt_time_s"
        ]
        == HC1_ATTEMPT_TIMEOUT_S
    )
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


def test_load_hc3e_wraps_checkpoint_with_speed_only_authority(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC3E-interaction-speed-PPO",
            "decision": "training-complete-pending-rollout",
            "action_authority": "interaction-speed-only",
            "min_interaction_speed_mps": 0.30,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "anchor_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )
    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )
    assert isinstance(loaded, InteractionSpeedOnlySupervisor)
    observation = torch.zeros(1, 17)
    observation[:, 0] = 0.625
    observation[:, -4] = 1.0
    command = loaded(observation)
    torch.testing.assert_close(command, actor(observation))
