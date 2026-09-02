"""Configuration tests for the single-box obstacle curriculum."""

from types import SimpleNamespace

import torch

from mjlab_microduck.robot.microduck_constants import MICRODUCK_OBSTACLE_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.motor_aware import make_motor_aware_run_variant
from mjlab_microduck.tasks.obstacle_avoidance import (
    OBSTACLE_MIN_FORWARD_SPEED_MPS,
    OBSTACLE_PLACEMENT_STAGES,
    OBSTACLE_RESUME_ITERATION,
    OBSTACLE_SENSOR_STAGES,
    OBSTACLE_VELOCITY_STAGES,
    make_obstacle_avoidance_variant,
)
from mjlab_microduck.tasks.run import make_run_variant


def _cfg(play=False):
    return make_obstacle_avoidance_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_microduck_velocity_env_cfg(play=play))
        ),
        play=play,
    )


def test_obstacle_entity_reset_and_observations_are_registered():
    cfg = _cfg()
    assert cfg.scene.entities["obstacle"] is MICRODUCK_OBSTACLE_CFG
    assert cfg.events["reset_obstacle"].func is microduck_mdp.reset_obstacle_ahead
    actor = cfg.observations["actor"].terms["obstacle"]
    critic = cfg.observations["critic"].terms["obstacle_ground_truth"]
    assert actor.func is microduck_mdp.obstacle_geometry_observation
    assert critic.func is microduck_mdp.obstacle_geometry_observation
    assert actor.delay_min_lag == 0 and actor.delay_max_lag == 1
    assert critic.delay_max_lag == 0


def test_curricula_are_offset_from_stage2_checkpoint_and_bounded_at_point8():
    assert OBSTACLE_PLACEMENT_STAGES[1]["step"] > OBSTACLE_RESUME_ITERATION * 24
    assert OBSTACLE_SENSOR_STAGES[1]["step"] > OBSTACLE_RESUME_ITERATION * 24
    assert OBSTACLE_VELOCITY_STAGES[-1]["lin_vel_range"] == 0.80
    cfg = _cfg()
    assert cfg.curriculum["obstacle_placement"].func is microduck_mdp.event_param_curriculum
    assert cfg.curriculum["obstacle_sensor"].func is microduck_mdp.observation_param_curriculum
    velocity = cfg.curriculum["velocity_command_ranges"].params
    assert velocity["velocity_stages"] == OBSTACLE_VELOCITY_STAGES
    assert velocity["forward_only"] is True
    assert velocity["min_lin_vel"] == OBSTACLE_MIN_FORWARD_SPEED_MPS
    assert velocity["update_lin_vel_y"] is False
    assert velocity["update_ang_vel_z"] is True
    assert all(stage["ang_vel_range"] == 0.0 for stage in OBSTACLE_VELOCITY_STAGES)
    assert cfg.commands["twist"].ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands["twist"].ranges.lin_vel_x == (
        OBSTACLE_MIN_FORWARD_SPEED_MPS,
        0.50,
    )
    assert cfg.commands["twist"].ranges.ang_vel_z == (0.0, 0.0)


def test_velocity_curriculum_preserves_nonstanding_speed_floor():
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
    )
    command_cfg = SimpleNamespace(ranges=ranges)
    env = SimpleNamespace(
        common_step_counter=0,
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(cfg=command_cfg)
        ),
    )
    microduck_mdp.velocity_command_ranges_curriculum(
        env,
        torch.tensor([0]),
        command_name="twist",
        velocity_stages=OBSTACLE_VELOCITY_STAGES,
        forward_only=True,
        min_lin_vel=OBSTACLE_MIN_FORWARD_SPEED_MPS,
    )
    assert ranges.lin_vel_x == (OBSTACLE_MIN_FORWARD_SPEED_MPS, 0.50)


def test_obstacle_reward_and_collision_contract_is_registered():
    cfg = _cfg()
    assert cfg.rewards["obstacle_clearance"].func is microduck_mdp.obstacle_clearance_cost
    assert cfg.rewards["obstacle_collision"].func is microduck_mdp.obstacle_collision
    assert cfg.rewards["obstacle_passed"].func is microduck_mdp.obstacle_passed_reward
    assert cfg.terminations["obstacle_collision"].func is microduck_mdp.obstacle_collision
    assert cfg.terminations["obstacle_collision"].time_out is False


def test_play_cfg_is_deterministic_and_has_no_sensor_curriculum():
    cfg = _cfg(play=True)
    actor = cfg.observations["actor"].terms["obstacle"]
    assert actor.delay_max_lag == 0
    assert "obstacle_sensor" not in cfg.curriculum


def test_observation_param_curriculum_mutates_live_manager_term():
    term = SimpleNamespace(params={"range_noise_m": 99.0})
    manager = SimpleNamespace(get_term_cfg=lambda group, name: term)
    env = SimpleNamespace(
        observation_manager=manager,
        common_step_counter=OBSTACLE_SENSOR_STAGES[1]["step"],
    )
    microduck_mdp.observation_param_curriculum(
        env,
        torch.tensor([0]),
        group_name="actor",
        term_name="obstacle",
        param_stages=OBSTACLE_SENSOR_STAGES,
    )
    assert term.params["range_noise_m"] == 0.01
    assert term.params["dropout_probability"] == 0.01


def test_obstacle_task_is_registered_with_distinct_identity():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks, load_rl_cfg

    task_id = "Mjlab-Run-Obstacle-Flat-MicroDuck"
    assert task_id in list_tasks()
    rl_cfg = load_rl_cfg(task_id)
    assert rl_cfg.experiment_name == "run_obstacle_avoidance"
    assert rl_cfg.run_name == "single_box_curriculum"
