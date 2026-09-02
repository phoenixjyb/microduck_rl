"""Single-box obstacle-avoidance curriculum on the motor-aware Run task.

This config establishes the simulation and observation contract only.  Its
actor input is seven dimensions wider than the retained Stage 2 policy, so a
reviewed warm-start migration of the first actor layer remains a hard gate
before GPU training.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from mjlab_microduck.robot.microduck_constants import MICRODUCK_OBSTACLE_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.motor_aware import MicroduckMotorAwareRunRlCfg


OBSTACLE_WIDTH_M = 0.20
OBSTACLE_HEIGHT_M = 0.10
ROBOT_COLLISION_RADIUS_M = 0.12
OBSTACLE_COLLISION_RADIUS_M = OBSTACLE_WIDTH_M / 2.0
OBSTACLE_CLEARANCE_MARGIN_M = 0.15
OBSTACLE_MIN_FORWARD_SPEED_MPS = 0.25
OBSTACLE_RESUME_ITERATION = 7998


def _resume_step(relative_iteration: int) -> int:
    return (OBSTACLE_RESUME_ITERATION + relative_iteration) * 24


OBSTACLE_PLACEMENT_STAGES = [
    {
        "step": 0,
        "params": {
            "forward_range_m": (1.0, 1.3),
            "lateral_range_m": (-0.35, 0.35),
        },
    },
    {
        "step": _resume_step(500),
        "params": {
            "forward_range_m": (0.8, 1.2),
            "lateral_range_m": (-0.30, 0.30),
        },
    },
    {
        "step": _resume_step(1000),
        "params": {
            "forward_range_m": (0.6, 1.0),
            "lateral_range_m": (-0.25, 0.25),
        },
    },
]

OBSTACLE_SENSOR_STAGES = [
    {
        "step": 0,
        "params": {
            "range_noise_m": 0.0,
            "bearing_noise_rad": 0.0,
            "width_noise_m": 0.0,
            "height_noise_m": 0.0,
            "closing_rate_noise_mps": 0.0,
            "dropout_probability": 0.0,
        },
    },
    {
        "step": _resume_step(500),
        "params": {
            "range_noise_m": 0.01,
            "bearing_noise_rad": math.radians(1.0),
            "width_noise_m": 0.01,
            "height_noise_m": 0.005,
            "closing_rate_noise_mps": 0.05,
            "dropout_probability": 0.01,
        },
    },
    {
        "step": _resume_step(1000),
        "params": {
            "range_noise_m": 0.03,
            "bearing_noise_rad": math.radians(3.0),
            "width_noise_m": 0.02,
            "height_noise_m": 0.01,
            "closing_rate_noise_mps": 0.10,
            "dropout_probability": 0.05,
        },
    },
]

OBSTACLE_VELOCITY_STAGES = [
    {"step": 0, "lin_vel_range": 0.50, "ang_vel_range": 0.0},
    {"step": _resume_step(500), "lin_vel_range": 0.65, "ang_vel_range": 0.0},
    {"step": _resume_step(1000), "lin_vel_range": 0.80, "ang_vel_range": 0.0},
]


def _obstacle_observation_params() -> dict:
    return {
        "asset_name": "obstacle",
        "width_m": OBSTACLE_WIDTH_M,
        "height_m": OBSTACLE_HEIGHT_M,
        "horizontal_fov_rad": math.pi,
        "max_range_m": 2.0,
        **OBSTACLE_SENSOR_STAGES[0]["params"],
    }


def make_obstacle_avoidance_variant(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add one visible obstacle and a staged observation/placement envelope."""
    cfg.scene.entities["obstacle"] = MICRODUCK_OBSTACLE_CFG

    cfg.events["reset_obstacle"] = EventTermCfg(
        func=microduck_mdp.reset_obstacle_ahead,
        mode="reset",
        params={
            **OBSTACLE_PLACEMENT_STAGES[0]["params"],
            "obstacle_height_m": OBSTACLE_HEIGHT_M,
            "asset_name": "obstacle",
        },
    )

    actor_obstacle = ObservationTermCfg(
        func=microduck_mdp.obstacle_geometry_observation,
        params=_obstacle_observation_params(),
        delay_min_lag=0,
        delay_max_lag=0 if play else 1,
        delay_update_period=64,
    )
    cfg.observations["actor"].terms["obstacle"] = actor_obstacle

    critic_params = _obstacle_observation_params()
    cfg.observations["critic"].terms["obstacle_ground_truth"] = ObservationTermCfg(
        func=microduck_mdp.obstacle_geometry_observation,
        params=critic_params,
    )

    envelope_params = {
        "asset_name": "obstacle",
        "robot_radius_m": ROBOT_COLLISION_RADIUS_M,
        "obstacle_radius_m": OBSTACLE_COLLISION_RADIUS_M,
    }
    cfg.rewards["obstacle_clearance"] = RewardTermCfg(
        func=microduck_mdp.obstacle_clearance_cost,
        weight=-2.0,
        params={**envelope_params, "margin_m": OBSTACLE_CLEARANCE_MARGIN_M},
    )
    cfg.rewards["obstacle_collision"] = RewardTermCfg(
        func=microduck_mdp.obstacle_collision,
        weight=-10.0,
        params=dict(envelope_params),
    )
    cfg.rewards["obstacle_passed"] = RewardTermCfg(
        func=microduck_mdp.obstacle_passed_reward,
        weight=10.0,
        params=dict(envelope_params),
    )
    cfg.rewards["obstacle_route_progress"] = RewardTermCfg(
        func=microduck_mdp.obstacle_route_progress_reward,
        weight=1.5,
        params={"command_name": "twist", "command_threshold": 0.01},
    )
    cfg.rewards["obstacle_heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={"std": 0.4},
    )
    cfg.terminations["obstacle_collision"] = TerminationTermCfg(
        func=microduck_mdp.obstacle_collision,
        params=dict(envelope_params),
    )
    cfg.terminations["obstacle_passed"] = TerminationTermCfg(
        func=microduck_mdp.obstacle_passed,
        params=dict(envelope_params),
    )

    cfg.curriculum["obstacle_placement"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "reset_obstacle",
            "param_stages": deepcopy(OBSTACLE_PLACEMENT_STAGES),
        },
    )
    if not play:
        cfg.curriculum["obstacle_sensor"] = CurriculumTermCfg(
            func=microduck_mdp.observation_param_curriculum,
            params={
                "group_name": "actor",
                "term_name": "obstacle",
                "param_stages": deepcopy(OBSTACLE_SENSOR_STAGES),
            },
        )

    velocity = cfg.curriculum["velocity_command_ranges"].params
    velocity["velocity_stages"] = deepcopy(OBSTACLE_VELOCITY_STAGES)
    velocity["forward_only"] = True
    velocity["min_lin_vel"] = OBSTACLE_MIN_FORWARD_SPEED_MPS
    velocity["update_lin_vel_y"] = False
    velocity["update_ang_vel_z"] = True
    # Obstacle episodes train one unambiguous behavior: move along the reset
    # heading and pass the obstacle.  The inherited velocity task otherwise
    # turns 25% of resumed environments into standing cases and gives most of
    # the rest random heading targets, which does not match fixed-speed
    # avoidance evaluation.
    cfg.curriculum.pop("standing_envs", None)
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 1.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = (
        OBSTACLE_MIN_FORWARD_SPEED_MPS,
        0.50,
    )
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    return cfg


MicroduckObstacleAvoidanceRlCfg = replace(
    MicroduckMotorAwareRunRlCfg,
    actor=deepcopy(MicroduckMotorAwareRunRlCfg.actor),
    critic=deepcopy(MicroduckMotorAwareRunRlCfg.critic),
    algorithm=deepcopy(MicroduckMotorAwareRunRlCfg.algorithm),
    experiment_name="run_obstacle_avoidance",
    run_name="single_box_curriculum",
)
