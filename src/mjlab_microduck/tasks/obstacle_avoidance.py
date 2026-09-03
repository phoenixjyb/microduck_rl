"""Centered, exact-geometry O1 task on the motor-aware Run task.

Its actor input is seven dimensions wider than the retained Stage 2 policy, so
a reviewed warm-start migration of the first actor layer remains a hard gate
before each GPU campaign.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from mjlab_microduck.obstacle_protocol import (
    OA0_ATTEMPT_TIMEOUT_S,
    OA0_COMMAND_SPEED_MPS,
    OA0_OBSTACLE_FORWARD_M,
    OA0_OBSTACLE_LATERAL_ABS_RANGE_M,
    OA0_ROUTE_RETURN_TOLERANCE_M,
    OA0R_TERMINAL_OUTCOME_REWARD,
    OA0P_INTERACTION_ENTRY_M,
    OA0P_RECOVERY_ENTRY_M,
    O1_MAX_COMMAND_SPEED_MPS,
    O1_MIN_COMMAND_SPEED_MPS,
    O1_OBSTACLE_FORWARD_M,
    O1_OBSTACLE_LATERAL_M,
    O1_PROTOCOL_NAME,
    o1_evaluation_protocol,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_OBSTACLE_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.motor_aware import MicroduckMotorAwareRunRlCfg


OBSTACLE_WIDTH_M = 0.20
OBSTACLE_HEIGHT_M = 0.10
ROBOT_COLLISION_RADIUS_M = 0.12
OBSTACLE_COLLISION_RADIUS_M = OBSTACLE_WIDTH_M / 2.0
OBSTACLE_CLEARANCE_MARGIN_M = 0.15
OBSTACLE_MIN_FORWARD_SPEED_MPS = O1_MIN_COMMAND_SPEED_MPS


OBSTACLE_PLACEMENT_STAGES = [
    {
        "step": 0,
        "params": {
            "forward_range_m": (O1_OBSTACLE_FORWARD_M, O1_OBSTACLE_FORWARD_M),
            "lateral_range_m": (O1_OBSTACLE_LATERAL_M, O1_OBSTACLE_LATERAL_M),
        },
    },
]

OBSTACLE_SENSOR_STAGES = [
    {
        "step": 0,
        "params": o1_evaluation_protocol()["sensor"],
    },
]

OBSTACLE_VELOCITY_STAGES = [
    {"step": 0, "lin_vel_range": O1_MAX_COMMAND_SPEED_MPS, "ang_vel_range": 0.0},
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
    """Add the centered, exact-geometry O1 obstacle environment."""
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
        delay_max_lag=0,
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
    cfg.rewards["obstacle_lateral_excursion"] = RewardTermCfg(
        func=microduck_mdp.obstacle_lateral_excursion_cost,
        weight=-0.5,
        params={"soft_limit_m": 0.45, "hard_limit_m": 0.75},
    )
    cfg.terminations["obstacle_collision"] = TerminationTermCfg(
        func=microduck_mdp.obstacle_collision,
        params=dict(envelope_params),
    )
    cfg.terminations["obstacle_passed"] = TerminationTermCfg(
        func=microduck_mdp.obstacle_passed,
        params=dict(envelope_params),
    )

    velocity = cfg.curriculum["velocity_command_ranges"].params
    velocity["velocity_stages"] = deepcopy(OBSTACLE_VELOCITY_STAGES)
    velocity["forward_only"] = True
    velocity["min_lin_vel"] = OBSTACLE_MIN_FORWARD_SPEED_MPS
    velocity["update_lin_vel_y"] = False
    velocity["update_ang_vel_z"] = True
    cfg.commands["twist"].ranges.lin_vel_x = (
        OBSTACLE_MIN_FORWARD_SPEED_MPS,
        O1_MAX_COMMAND_SPEED_MPS,
    )
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    return cfg


def make_obstacle_assisted_variant(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build OA0 without weakening the centered O1 benchmark."""
    cfg = make_obstacle_avoidance_variant(cfg, play=play)
    reset_params = cfg.events["reset_obstacle"].params
    reset_params["forward_range_m"] = (
        OA0_OBSTACLE_FORWARD_M,
        OA0_OBSTACLE_FORWARD_M,
    )
    reset_params["lateral_abs_range_m"] = OA0_OBSTACLE_LATERAL_ABS_RANGE_M

    velocity = cfg.curriculum["velocity_command_ranges"].params
    velocity["velocity_stages"] = [
        {
            "step": 0,
            "lin_vel_range": OA0_COMMAND_SPEED_MPS,
            "ang_vel_range": 0.0,
        }
    ]
    velocity["min_lin_vel"] = OA0_COMMAND_SPEED_MPS
    cfg.commands["twist"].ranges.lin_vel_x = (
        OA0_COMMAND_SPEED_MPS,
        OA0_COMMAND_SPEED_MPS,
    )

    envelope_params = {
        "asset_name": "obstacle",
        "robot_radius_m": ROBOT_COLLISION_RADIUS_M,
        "obstacle_radius_m": OBSTACLE_COLLISION_RADIUS_M,
        "return_tolerance_m": OA0_ROUTE_RETURN_TOLERANCE_M,
    }
    cfg.rewards["obstacle_passed"] = RewardTermCfg(
        func=microduck_mdp.obstacle_route_rejoined_reward,
        weight=10.0,
        params=dict(envelope_params),
    )
    cfg.terminations["obstacle_passed"] = TerminationTermCfg(
        func=microduck_mdp.obstacle_route_rejoined,
        params=dict(envelope_params),
    )
    cfg.terminations["obstacle_attempt_timeout"] = TerminationTermCfg(
        func=microduck_mdp.obstacle_attempt_timeout,
        params={"max_attempt_time_s": OA0_ATTEMPT_TIMEOUT_S},
    )
    return cfg


def make_obstacle_assisted_outcome_variant(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add one episode-level outcome impulse to OA0, changing no other axis."""
    cfg = make_obstacle_assisted_variant(cfg, play=play)
    cfg.rewards["obstacle_terminal_outcome"] = RewardTermCfg(
        func=microduck_mdp.obstacle_terminal_outcome_reward,
        weight=OA0R_TERMINAL_OUTCOME_REWARD,
        params={
            "asset_name": "obstacle",
            "robot_radius_m": ROBOT_COLLISION_RADIUS_M,
            "obstacle_radius_m": OBSTACLE_COLLISION_RADIUS_M,
            "return_tolerance_m": OA0_ROUTE_RETURN_TOLERANCE_M,
            "max_attempt_time_s": OA0_ATTEMPT_TIMEOUT_S,
        },
    )
    return cfg


def make_obstacle_assisted_phase_speed_variant(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Gate only speed shaping by obstacle phase on top of OA0R."""
    cfg = make_obstacle_assisted_outcome_variant(cfg, play=play)
    linear = cfg.rewards["track_linear_velocity"]
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=microduck_mdp.obstacle_phase_linear_velocity_reward,
        weight=linear.weight,
        params={
            **linear.params,
            "asset_name": "obstacle",
            "interaction_entry_m": OA0P_INTERACTION_ENTRY_M,
            "recovery_entry_m": OA0P_RECOVERY_ENTRY_M,
        },
    )
    progress = cfg.rewards["obstacle_route_progress"]
    cfg.rewards["obstacle_route_progress"] = RewardTermCfg(
        func=microduck_mdp.obstacle_phase_route_progress_reward,
        weight=progress.weight,
        params={
            **progress.params,
            "asset_name": "obstacle",
            "interaction_entry_m": OA0P_INTERACTION_ENTRY_M,
            "recovery_entry_m": OA0P_RECOVERY_ENTRY_M,
        },
    )
    return cfg


MicroduckObstacleAvoidanceRlCfg = replace(
    MicroduckMotorAwareRunRlCfg,
    actor=deepcopy(MicroduckMotorAwareRunRlCfg.actor),
    critic=deepcopy(MicroduckMotorAwareRunRlCfg.critic),
    algorithm=deepcopy(MicroduckMotorAwareRunRlCfg.algorithm),
    experiment_name="run_obstacle_avoidance",
    run_name="single_box_curriculum",
)

MicroduckObstacleAssistedRlCfg = replace(
    MicroduckObstacleAvoidanceRlCfg,
    actor=deepcopy(MicroduckObstacleAvoidanceRlCfg.actor),
    critic=deepcopy(MicroduckObstacleAvoidanceRlCfg.critic),
    algorithm=deepcopy(MicroduckObstacleAvoidanceRlCfg.algorithm),
    experiment_name="run_obstacle_assisted",
    run_name="oa0_signed_offset",
)

MicroduckObstacleAssistedOutcomeRlCfg = replace(
    MicroduckObstacleAssistedRlCfg,
    actor=deepcopy(MicroduckObstacleAssistedRlCfg.actor),
    critic=deepcopy(MicroduckObstacleAssistedRlCfg.critic),
    algorithm=deepcopy(MicroduckObstacleAssistedRlCfg.algorithm),
    experiment_name="run_obstacle_assisted_outcome",
    run_name="oa0_outcome_balanced",
)

MicroduckObstacleAssistedPhaseSpeedRlCfg = replace(
    MicroduckObstacleAssistedOutcomeRlCfg,
    actor=deepcopy(MicroduckObstacleAssistedOutcomeRlCfg.actor),
    critic=deepcopy(MicroduckObstacleAssistedOutcomeRlCfg.critic),
    algorithm=deepcopy(MicroduckObstacleAssistedOutcomeRlCfg.algorithm),
    experiment_name="run_obstacle_assisted_phase_speed",
    run_name="oa0_phase_aware_speed",
)
