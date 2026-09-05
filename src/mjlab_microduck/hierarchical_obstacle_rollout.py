"""Bounded HC1 rollout of a command teacher over a frozen locomotion actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from mjlab_microduck.evaluation import (
    parse_float_list,
    parse_int_list,
    valid_action_deltas,
)
from mjlab_microduck.hierarchical_obstacle import (
    SUPERVISOR_OBSERVATION_DIM,
    ObstaclePhase,
    ObstacleTeacherCfg,
    advance_obstacle_state,
    apply_bounded_supervisor_command,
    clone_teacher_state,
    make_teacher_state,
    reset_teacher_state,
    supervisor_observation,
    teacher_command,
)
from mjlab_microduck.obstacle_baseline import _resolved_attempt_metrics
from mjlab_microduck.obstacle_protocol import OA0_TASK_ID
from mjlab_microduck.obstacle_supervisor_bc import (
    HC2_STAGE,
    HC4L_STAGE,
    HC4LH_STAGE,
    HC4R2H_STAGE,
    HC4R2L_STAGE,
    HC4R2_STAGE,
    HC4R_STAGE,
    HC4U1_STAGE,
    EpisodeLatchedRangeSpeedSupervisor,
    InteractionSpeedOnlySupervisor,
    LateralGatedSupervisor,
    ObstacleSupervisor,
    RangeSpeedGatedSupervisor,
    SupervisorBcCfg,
)
from mjlab_microduck.tasks.run import (
    MOTOR_NEAR_LIMIT_FRACTION,
    XL330_M288_RATED_NO_LOAD_SPEED_RAD_S,
    XL330_M288_RATED_STALL_TORQUE_NM_6V,
)

BASE_TASK_ID = "Mjlab-Run-MotorAware-Flat-MicroDuck"
MAX_ENVS = 256
MAX_STEPS = 1000
MAX_CASES = 48
HC1_ATTEMPT_TIMEOUT_S = 12.0
HC3E_STAGE = "HC3E-interaction-speed-PPO"
HC3F_STAGE = "HC3F-seed-averaged-speed-head"
HC3G_STAGE = "HC3G-seed-consensus-speed-head"

_RECORDING_STAGE_SLUGS = {
    HC2_STAGE: "hc2",
    HC4L_STAGE: "hc4l",
    HC4LH_STAGE: "hc4lh",
    HC4R_STAGE: "hc4r",
    HC4R2_STAGE: "hc4r2",
    HC4R2H_STAGE: "hc4r2h",
    HC4R2L_STAGE: "hc4r2l",
    "HC3-supervisor-PPO": "hc3",
    HC3E_STAGE: "hc3e",
    HC3F_STAGE: "hc3f",
    HC3G_STAGE: "hc3g",
}


def recording_controller_stage(supervisor_stage: str | None) -> str:
    """Return the filename slug for a teacher or learned supervisor replay."""
    if supervisor_stage is None:
        return "hc1"
    try:
        return _RECORDING_STAGE_SLUGS[supervisor_stage]
    except KeyError as error:
        raise ValueError(
            f"unsupported supervisor stage for recording: {supervisor_stage!r}"
        ) from error


def recording_stem(
    speed: float,
    obstacle_forward: float,
    obstacle_lateral: float,
    *,
    controller_stage: str = "hc1",
) -> str:
    """Return a deterministic replay basename for one geometry cell."""
    if controller_stage not in {"hc1", *_RECORDING_STAGE_SLUGS.values()}:
        raise ValueError("controller_stage is not a supported obstacle curriculum stage")
    return (
        f"microduck-{controller_stage}-{speed:.2f}mps-"
        f"x{obstacle_forward:.2f}m-y{obstacle_lateral:+.2f}m"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_learned_supervisor(
    supervisor_checkpoint: Path,
    base_checkpoint: Path,
    device: str,
) -> torch.nn.Module:
    """Load an eligible supervisor when frozen-gait identity matches."""
    import torch

    payload = torch.load(
        supervisor_checkpoint, map_location=device, weights_only=False
    )
    stage = payload.get("stage")
    decision = payload.get("decision")
    allowed = (
        (
            stage
            in {HC2_STAGE, HC4L_STAGE, HC4R_STAGE, HC4R2_STAGE, HC4U1_STAGE}
            and decision == "offline-imitation-pass"
        )
        or (
            stage in {HC4LH_STAGE, HC4R2H_STAGE, HC4R2L_STAGE}
            and decision
            in {"composition-complete-pending-rollout", "accepted-simulation"}
        )
        or (
            stage == "HC3-supervisor-PPO"
            and decision in {"training-complete-pending-rollout", "accepted-simulation"}
        )
        or (
            stage == HC3E_STAGE
            and decision in {"training-complete-pending-rollout", "accepted-simulation"}
        )
        or (
            stage in {HC3F_STAGE, HC3G_STAGE}
            and decision
            in {"aggregation-complete-pending-rollout", "accepted-simulation"}
        )
    )
    if not allowed:
        raise ValueError("supervisor checkpoint is not eligible for diagnostic rollout")
    if stage == HC4R2L_STAGE and payload.get("selector_state") != (
        "latched-until-explicit-episode-reset"
    ):
        raise ValueError("HC4-R2L checkpoint has invalid selector state")
    if payload.get("source_locomotion_checkpoint_sha256") != _sha256(
        base_checkpoint
    ):
        raise ValueError("supervisor was trained for another locomotion checkpoint")
    model_cfg = dict(payload["model_config"])
    model_cfg["hidden_dims"] = tuple(model_cfg["hidden_dims"])
    model = ObstacleSupervisor(SupervisorBcCfg(**model_cfg)).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    if stage in {HC4LH_STAGE, HC4R2H_STAGE, HC4R2L_STAGE}:
        center_model = ObstacleSupervisor(SupervisorBcCfg(**model_cfg)).to(device)
        center_model.load_state_dict(payload["center_model_state_dict"], strict=True)
        center_model.eval()
        model = LateralGatedSupervisor(
            center_model,
            model,
            lateral_gate_m=payload["lateral_gate_m"],
        ).to(device)
        model.eval()
        if stage in {HC4R2H_STAGE, HC4R2L_STAGE}:
            near_model = ObstacleSupervisor(SupervisorBcCfg(**model_cfg)).to(device)
            near_model.load_state_dict(payload["near_model_state_dict"], strict=True)
            near_model.eval()
            supervisor_type = (
                EpisodeLatchedRangeSpeedSupervisor
                if stage == HC4R2L_STAGE
                else RangeSpeedGatedSupervisor
            )
            model = supervisor_type(
                model,
                near_model,
                near_range_gate_m=payload["near_range_gate_m"],
                max_near_nominal_speed_mps=payload[
                    "max_near_nominal_speed_mps"
                ],
            ).to(device)
            model.eval()
    if stage in {HC3E_STAGE, HC3F_STAGE, HC3G_STAGE}:
        action_authority = payload.get("action_authority")
        if action_authority != "interaction-speed-only":
            raise ValueError(f"{stage} checkpoint has invalid action authority")
        hc2_model = ObstacleSupervisor(SupervisorBcCfg(**model_cfg)).to(device)
        hc2_model.load_state_dict(payload["anchor_model_state_dict"], strict=True)
        hc2_model.eval()
        model = InteractionSpeedOnlySupervisor(
            model,
            hc2_model,
            min_interaction_speed_mps=payload["min_interaction_speed_mps"],
        ).to(device)
        model.eval()
    return model


def validate_rollout_bounds(
    num_envs: int,
    steps: int,
    speeds: tuple[float, ...],
    forward_positions: tuple[float, ...],
    lateral_positions: tuple[float, ...],
    seeds: tuple[int, ...],
) -> None:
    if not 1 <= num_envs <= MAX_ENVS:
        raise ValueError(f"num_envs must be in [1, {MAX_ENVS}]")
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not speeds or any(speed <= 0.0 or speed > 0.8 for speed in speeds):
        raise ValueError("speeds must be non-empty and in (0, 0.8]")
    if not forward_positions or any(position <= 0.0 for position in forward_positions):
        raise ValueError("forward obstacle positions must be positive")
    if not lateral_positions or not seeds:
        raise ValueError("lateral positions and seeds must be non-empty")
    case_count = (
        len(speeds)
        * len(forward_positions)
        * len(lateral_positions)
        * len(seeds)
    )
    if case_count > MAX_CASES:
        raise ValueError(f"case count must not exceed {MAX_CASES}")


def validate_dataset_collection_mode(
    *,
    collect_success_dataset: bool,
    collect_teacher_corrections: bool,
    supervisor_checkpoint: Path | None,
) -> None:
    """Validate mutually exclusive teacher and student-state collection modes."""
    if collect_success_dataset and collect_teacher_corrections:
        raise ValueError("dataset collection modes are mutually exclusive")
    if collect_success_dataset and supervisor_checkpoint is not None:
        raise ValueError("success datasets may be collected only from the teacher")
    if collect_teacher_corrections and supervisor_checkpoint is None:
        raise ValueError(
            "teacher-correction datasets require a student supervisor checkpoint"
        )


def fixed_attempt_metrics(
    *,
    expected_attempts: int,
    completed_attempts: int,
    clean_pass_events: int,
    collision_events: int,
    attempt_timeout_events: int,
    fall_events: int,
    nan_termination_events: int,
    other_terminal_events: int,
) -> dict[str, int | float]:
    """Summarize a fixed-denominator first-attempt evaluation window."""
    counts = {
        "expected_attempts": expected_attempts,
        "completed_attempts": completed_attempts,
        "clean_pass_events": clean_pass_events,
        "collision_events": collision_events,
        "attempt_timeout_events": attempt_timeout_events,
        "fall_events": fall_events,
        "nan_termination_events": nan_termination_events,
        "other_terminal_events": other_terminal_events,
    }
    if expected_attempts <= 0:
        raise ValueError("expected_attempts must be positive")
    if any(value < 0 for value in counts.values()):
        raise ValueError("fixed-attempt counts must be non-negative")
    if completed_attempts > expected_attempts:
        raise ValueError("completed_attempts cannot exceed expected_attempts")
    resolved = clean_pass_events + collision_events + attempt_timeout_events
    if resolved > completed_attempts:
        raise ValueError("resolved outcomes cannot exceed completed_attempts")
    if resolved + other_terminal_events > completed_attempts:
        raise ValueError(
            "resolved and other terminal outcomes cannot exceed completed_attempts"
        )
    return {
        "expected_attempts": expected_attempts,
        "completed_attempts": completed_attempts,
        "unresolved_attempts": expected_attempts - completed_attempts,
        "clean_pass_rate_fixed_denominator": clean_pass_events / expected_attempts,
        "collision_rate_fixed_denominator": collision_events / expected_attempts,
        "attempt_timeout_rate_fixed_denominator": (
            attempt_timeout_events / expected_attempts
        ),
        "hard_failure_events": fall_events + nan_termination_events,
        "other_terminal_events": other_terminal_events,
    }


def advance_first_attempt_window(
    active: torch.Tensor, dones: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Close each environment after its first terminal transition."""
    import torch

    if active.dtype != torch.bool or dones.dtype != torch.bool:
        raise ValueError("first-attempt masks must be boolean")
    if active.shape != dones.shape or active.ndim != 1:
        raise ValueError("first-attempt masks must be matching vectors")
    finished = active & dones
    return active & ~finished, finished


def resolved_correction_samples(
    observations: torch.Tensor,
    teacher_commands: torch.Tensor,
    student_commands: torch.Tensor,
    episode_keys: torch.Tensor,
    episode_outcomes: dict[int, int],
) -> dict[str, torch.Tensor]:
    """Keep labeled samples only from resolved, non-hard-failure episodes."""
    import torch

    sample_count = observations.shape[0]
    if (
        observations.ndim != 2
        or observations.shape[1] != SUPERVISOR_OBSERVATION_DIM
    ):
        raise ValueError(
            f"correction observations must have shape (N, {SUPERVISOR_OBSERVATION_DIM})"
        )
    if teacher_commands.shape != (sample_count, 2):
        raise ValueError("teacher correction commands must have shape (N, 2)")
    if student_commands.shape != (sample_count, 2):
        raise ValueError("student commands must have shape (N, 2)")
    if episode_keys.shape != (sample_count,):
        raise ValueError("episode keys must have shape (N,)")
    if any(code not in {1, 2, 3} for code in episode_outcomes.values()):
        raise ValueError("correction outcome codes must be clean=1/collision=2/timeout=3")
    if not episode_outcomes:
        keep = torch.zeros(sample_count, dtype=torch.bool)
        outcome_codes = torch.empty(0, dtype=torch.int8)
    else:
        resolved_keys = torch.tensor(
            sorted(episode_outcomes), dtype=episode_keys.dtype
        )
        keep = torch.isin(episode_keys.cpu(), resolved_keys)
        outcome_codes = torch.tensor(
            [episode_outcomes[int(key)] for key in episode_keys.cpu()[keep]],
            dtype=torch.int8,
        )
    return {
        "observations": observations[keep],
        "commands": teacher_commands[keep],
        "student_commands": student_commands[keep],
        "episode_keys": episode_keys[keep],
        "outcome_codes": outcome_codes,
    }


def prepare_rollout_configs(
    num_envs: int,
    nominal_speed_mps: float,
    obstacle_forward_m: float,
    obstacle_lateral_m: float,
):
    """Build OA0 physics with the original 61D frozen-policy observation."""
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import mjlab_microduck.tasks  # noqa: F401

    env_cfg = load_env_cfg(OA0_TASK_ID, play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.terrain.num_envs = num_envs
    env_cfg.observations["actor"].terms.pop("obstacle")
    env_cfg.observations["critic"].terms.pop("obstacle_ground_truth")

    reset = env_cfg.events["reset_obstacle"].params
    reset["forward_range_m"] = (obstacle_forward_m, obstacle_forward_m)
    reset["lateral_range_m"] = (obstacle_lateral_m, obstacle_lateral_m)
    reset.pop("lateral_abs_range_m", None)

    twist = env_cfg.commands["twist"]
    twist.resampling_time_range = (1.0e6, 1.0e6)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = (nominal_speed_mps, nominal_speed_mps)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    for name in ("head_pose", "body_pose"):
        command = env_cfg.commands[name]
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.ranges = tuple((0.0, 0.0) for _ in command.ranges)
        command.zero_command_prob = 1.0

    env_cfg.curriculum.clear()
    env_cfg.events.pop("push_robot", None)
    env_cfg.terminations["obstacle_attempt_timeout"].params[
        "max_attempt_time_s"
    ] = HC1_ATTEMPT_TIMEOUT_S
    agent_cfg = load_rl_cfg(BASE_TASK_ID)
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    return env_cfg, agent_cfg


def _route_state(env):
    import torch

    robot = env.scene["robot"]
    path_dir = env._obstacle_path_dir_w
    lateral_dir = torch.stack((-path_dir[:, 1], path_dir[:, 0]), dim=-1)
    route_lateral = (
        (robot.data.root_link_pos_w[:, :2] - env._obstacle_route_origin_w)
        * lateral_dir
    ).sum(dim=-1)
    quat = robot.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    robot_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    route_yaw = torch.atan2(path_dir[:, 1], path_dir[:, 0])
    heading_error = torch.atan2(
        torch.sin(robot_yaw - route_yaw), torch.cos(robot_yaw - route_yaw)
    )
    route_speed = (robot.data.root_link_lin_vel_w[:, :2] * path_dir).sum(dim=-1)
    return route_lateral, heading_error, route_speed


def _run_case(
    checkpoint: Path,
    *,
    num_envs: int,
    steps: int,
    nominal_speed_mps: float,
    obstacle_forward_m: float,
    obstacle_lateral_m: float,
    seed: int,
    collect_success_samples: bool = False,
    collect_teacher_corrections: bool = False,
    first_attempt_only: bool = False,
    case_index: int = 0,
    supervisor_checkpoint: Path | None = None,
) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    from mjlab_microduck.tasks import mdp as microduck_mdp

    env_cfg, agent_cfg = prepare_rollout_configs(
        num_envs,
        nominal_speed_mps,
        obstacle_forward_m,
        obstacle_lateral_m,
    )
    env_cfg.seed = seed
    agent_cfg.seed = seed
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    torch.manual_seed(seed)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(BASE_TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        observations = wrapped.get_observations()

        learned_supervisor = None
        if supervisor_checkpoint is not None:
            learned_supervisor = load_learned_supervisor(
                supervisor_checkpoint, checkpoint, device
            )

        state = make_teacher_state(
            num_envs, device=device, nominal_speed_mps=nominal_speed_mps
        )
        nominal = torch.full((num_envs,), nominal_speed_mps, device=device)
        command = env.command_manager.get_command("twist")
        command[:, 0] = nominal
        command[:, 1:] = 0.0

        collision_events = 0
        clean_pass_events = 0
        attempt_timeout_events = 0
        fall_events = 0
        nan_events = 0
        nonfinite_steps = 0
        phase_speed_sum = torch.zeros(3, device=device)
        phase_samples = torch.zeros(3, dtype=torch.long, device=device)
        lateral_abs_max = torch.zeros(num_envs, device=device)
        pass_lateral_sum = 0.0
        pass_lateral_count = 0
        command_speed_min = math.inf
        command_speed_max = -math.inf
        command_yaw_abs_max = 0.0
        episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        pass_time_sum_s = 0.0
        representative_trace: list[dict] = []
        representative_attempt_done = False
        robot = env.scene["robot"]
        joint_ids, _ = robot.find_joints(r"^(?!passive_).*")
        joint_speed_samples = []
        torque_samples = []
        action_samples = []
        action_rate_samples = []
        previous_actions = env.action_manager.action.detach().clone()
        previous_dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        episode_generation = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        environment_index = torch.arange(num_envs, device=device)
        dataset_observations = []
        dataset_commands = []
        dataset_student_commands = []
        dataset_episode_keys = []
        successful_episode_keys: set[int] = set()
        correction_episode_outcomes: dict[int, int] = {}
        evaluation_active = torch.ones(num_envs, dtype=torch.bool, device=device)
        other_terminal_events = 0
        steps_executed = 0

        with torch.inference_mode():
            for step in range(steps):
                if first_attempt_only and not bool(evaluation_active.any()):
                    break
                active = evaluation_active.clone()
                steps_executed = step + 1
                route_lateral, route_heading, route_speed = _route_state(env)
                lateral_abs_max[active] = torch.maximum(
                    lateral_abs_max[active], route_lateral[active].abs()
                )
                if step % 5 == 0:
                    obstacle_observation = microduck_mdp.obstacle_geometry_observation(
                        env,
                        asset_name="obstacle",
                        width_m=0.20,
                        height_m=0.10,
                        horizontal_fov_rad=2.0 * math.pi,
                        max_range_m=2.0,
                    )
                    previous_supervisor_command = state.previous_command.clone()
                    learned_observation = None
                    teacher_correction = None
                    if learned_supervisor is None:
                        supervisor_command = teacher_command(
                            obstacle_observation,
                            nominal,
                            route_lateral,
                            route_heading,
                            state,
                            cfg=ObstacleTeacherCfg(),
                        )
                    else:
                        advance_obstacle_state(
                            obstacle_observation,
                            route_lateral,
                            route_heading,
                            state,
                        )
                        learned_observation = supervisor_observation(
                            obstacle_observation,
                            nominal,
                            route_lateral,
                            route_heading,
                            route_speed,
                            state,
                            previous_command=previous_supervisor_command,
                        )
                        if collect_teacher_corrections:
                            teacher_correction = teacher_command(
                                obstacle_observation,
                                nominal,
                                route_lateral,
                                route_heading,
                                clone_teacher_state(state),
                                cfg=ObstacleTeacherCfg(),
                            )
                        normalized_command = learned_supervisor(learned_observation)
                        limits = ObstacleTeacherCfg()
                        desired_command = torch.stack(
                            (
                                normalized_command[:, 0]
                                * limits.max_forward_speed_mps,
                                normalized_command[:, 1]
                                * limits.max_yaw_rate_rps,
                            ),
                            dim=-1,
                        )
                        supervisor_command = apply_bounded_supervisor_command(
                            desired_command,
                            obstacle_observation,
                            state,
                        )
                    if collect_success_samples:
                        dataset_observations.append(
                            supervisor_observation(
                                obstacle_observation,
                                nominal,
                                route_lateral,
                                route_heading,
                                route_speed,
                                state,
                                previous_command=previous_supervisor_command,
                            ).cpu()
                        )
                        dataset_commands.append(supervisor_command.cpu())
                        episode_key = (
                            case_index * 1_000_000
                            + episode_generation * num_envs
                            + environment_index
                        )
                        dataset_episode_keys.append(episode_key.cpu())
                    if collect_teacher_corrections:
                        assert learned_observation is not None
                        assert teacher_correction is not None
                        dataset_observations.append(learned_observation.cpu())
                        dataset_commands.append(teacher_correction.cpu())
                        dataset_student_commands.append(supervisor_command.cpu())
                        episode_key = (
                            case_index * 1_000_000
                            + episode_generation * num_envs
                            + environment_index
                        )
                        dataset_episode_keys.append(episode_key.cpu())
                    command[:, 0] = supervisor_command[:, 0]
                    command[:, 1] = 0.0
                    command[:, 2] = supervisor_command[:, 1]

                    if not representative_attempt_done:
                        robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
                        obstacle_xy = env.scene["obstacle"].data.root_link_pos_w[:, :2]
                        path_dir = env._obstacle_path_dir_w
                        obstacle_delta = obstacle_xy - robot_xy
                        route_progress = (
                            (robot_xy - env._obstacle_route_origin_w) * path_dir
                        ).sum(dim=-1)
                        obstacle_ahead = (obstacle_delta * path_dir).sum(dim=-1)
                        center_distance = torch.linalg.vector_norm(
                            obstacle_delta, dim=-1
                        )
                        representative_trace.append(
                            {
                                "time_s": step * env.step_dt,
                                "route_progress_m": float(route_progress[0]),
                                "route_lateral_error_m": float(route_lateral[0]),
                                "route_heading_error_rad": float(route_heading[0]),
                                "route_speed_mps": float(route_speed[0]),
                                "obstacle_ahead_m": float(obstacle_ahead[0]),
                                "obstacle_clearance_m": float(center_distance[0] - 0.22),
                                "phase": ObstaclePhase(int(state.phase[0])).name.lower(),
                                "command_speed_mps": float(command[0, 0]),
                                "command_yaw_rate_rps": float(command[0, 2]),
                            }
                        )

                for phase in ObstaclePhase:
                    mask = (state.phase == int(phase)) & active
                    phase_speed_sum[int(phase)] += torch.nan_to_num(
                        route_speed[mask], nan=0.0
                    ).sum()
                    phase_samples[int(phase)] += mask.sum()
                command_speed_min = min(
                    command_speed_min, float(command[active, 0].min())
                )
                command_speed_max = max(
                    command_speed_max, float(command[active, 0].max())
                )
                command_yaw_abs_max = max(
                    command_yaw_abs_max, float(command[active, 2].abs().max())
                )

                actions = policy(observations)
                observations, rewards, dones, _ = wrapped.step(actions)
                episode_steps[active] += 1
                applied_actions = env.action_manager.action.detach()
                joint_speed = robot.data.joint_vel[:, joint_ids].abs().float()
                torque = robot.data.actuator_force.abs().float()
                if joint_speed.shape != torque.shape:
                    raise RuntimeError(
                        f"joint speed shape {joint_speed.shape} does not match "
                        f"actuator torque shape {torque.shape}"
                    )
                joint_speed_samples.append(joint_speed[active])
                torque_samples.append(torque[active])
                action_samples.append(applied_actions[active].abs().float())
                action_deltas = valid_action_deltas(
                    applied_actions[active],
                    previous_actions[active],
                    previous_dones[active],
                )
                if action_deltas.numel():
                    action_rate_samples.append(action_deltas.float())
                collision = (
                    env.termination_manager.get_term("obstacle_collision").bool()
                    & active
                )
                passed = (
                    env.termination_manager.get_term("obstacle_passed").bool()
                    & active
                )
                attempted_out = env.termination_manager.get_term(
                    "obstacle_attempt_timeout"
                ).bool() & active
                fell = env.termination_manager.get_term("fell_over").bool() & active
                nan_state = (
                    env.termination_manager.get_term("nan_state").bool() & active
                )
                collision_events += int(collision.sum())
                clean_pass_events += int(passed.sum())
                pass_time_sum_s += float(
                    (episode_steps[passed].float() * env.step_dt).sum()
                )
                attempt_timeout_events += int(attempted_out.sum())
                fall_events += int(fell.sum())
                nan_events += int(nan_state.sum())
                pass_lateral_sum += float(lateral_abs_max[passed].sum())
                pass_lateral_count += int(passed.sum())
                if collect_success_samples and bool(passed.any()):
                    episode_key = (
                        case_index * 1_000_000
                        + episode_generation * num_envs
                        + environment_index
                    )
                    successful_episode_keys.update(
                        int(value) for value in episode_key[passed].tolist()
                    )
                if collect_teacher_corrections:
                    terminal_count = (
                        collision.to(torch.int8)
                        + passed.to(torch.int8)
                        + attempted_out.to(torch.int8)
                    )
                    if bool((terminal_count > 1).any()):
                        raise RuntimeError("obstacle terminal outcomes overlap")
                    hard_failure = fell.bool() | nan_state.bool()
                    resolved = (terminal_count == 1) & ~hard_failure
                    if bool(resolved.any()):
                        episode_key = (
                            case_index * 1_000_000
                            + episode_generation * num_envs
                            + environment_index
                        )
                        outcome_code = torch.zeros_like(terminal_count)
                        outcome_code[passed] = 1
                        outcome_code[collision] = 2
                        outcome_code[attempted_out] = 3
                        for key, code in zip(
                            episode_key[resolved].tolist(),
                            outcome_code[resolved].tolist(),
                            strict=True,
                        ):
                            correction_episode_outcomes[int(key)] = int(code)
                finite = torch.isfinite(actions[active]).all()
                finite &= torch.isfinite(rewards[active]).all()
                finite &= all(
                    torch.isfinite(value[active]).all()
                    for value in observations.values()
                )
                nonfinite_steps += int(not bool(finite))
                if not representative_attempt_done and bool(dones[0]):
                    representative_attempt_done = True
                    representative_trace[-1]["terminal"] = {
                        "collision": bool(collision[0]),
                        "clean_pass": bool(passed[0]),
                        "attempt_timeout": bool(attempted_out[0]),
                        "fell": bool(fell[0]),
                        "nan_state": bool(nan_state[0]),
                    }
                reset_teacher_state(
                    state, dones.bool(), nominal_speed_mps=nominal_speed_mps
                )
                if learned_supervisor is not None and hasattr(
                    learned_supervisor, "reset_episodes"
                ):
                    learned_supervisor.reset_episodes(dones.bool())
                if first_attempt_only:
                    evaluation_active, finished = advance_first_attempt_window(
                        active, dones.bool()
                    )
                    classified = collision | passed | attempted_out | fell | nan_state
                    other_terminal_events += int((finished & ~classified).sum())
                lateral_abs_max[dones.bool()] = 0.0
                episode_steps[dones.bool()] = 0
                episode_generation[dones.bool()] += 1
                previous_actions = applied_actions.clone()
                previous_dones = dones.bool().clone()

        joint_speed = torch.cat(joint_speed_samples).flatten()
        torque = torch.cat(torque_samples).flatten()
        actions = torch.cat(action_samples).flatten()
        action_rates = torch.cat(action_rate_samples)
        speed_util = joint_speed / XL330_M288_RATED_NO_LOAD_SPEED_RAD_S
        torque_util = torque / XL330_M288_RATED_STALL_TORQUE_NM_6V

        result = {
            "nominal_speed_mps": nominal_speed_mps,
            "obstacle_forward_m": obstacle_forward_m,
            "obstacle_lateral_m": obstacle_lateral_m,
            "seed": seed,
            "num_envs": num_envs,
            "steps": steps,
            "steps_executed": steps_executed,
            "collision_events": collision_events,
            "clean_pass_events": clean_pass_events,
            "attempt_timeout_events": attempt_timeout_events,
            "fall_events": fall_events,
            "nan_termination_events": nan_events,
            "nonfinite_steps": nonfinite_steps,
            "mean_pass_lateral_excursion_m": (
                pass_lateral_sum / pass_lateral_count if pass_lateral_count else None
            ),
            "mean_passage_time_s": (
                pass_time_sum_s / clean_pass_events if clean_pass_events else None
            ),
            "command_speed_min_mps": command_speed_min,
            "command_speed_max_mps": command_speed_max,
            "command_yaw_abs_max_rps": command_yaw_abs_max,
            "motor_speed_utilization_p99": float(torch.quantile(speed_util, 0.99)),
            "motor_speed_rated_exceed_fraction": float(
                (speed_util > 1.0).float().mean()
            ),
            "motor_torque_utilization_p99": float(
                torch.quantile(torque_util, 0.99)
            ),
            "motor_torque_near_stall_fraction": float(
                (torque_util >= MOTOR_NEAR_LIMIT_FRACTION).float().mean()
            ),
            "motor_thermal_load_proxy_mean": float(torch.square(torque_util).mean()),
            "action_abs_p99": float(torch.quantile(actions, 0.99)),
            "action_rate_abs_p99": float(torch.quantile(action_rates, 0.99)),
            "representative_first_attempt_trace": representative_trace,
        }
        for phase in ObstaclePhase:
            count = int(phase_samples[int(phase)])
            result[f"{phase.name.lower()}_samples"] = count
            result[f"{phase.name.lower()}_route_speed_mps"] = (
                float(phase_speed_sum[int(phase)] / count) if count else None
            )
        result.update(
            _resolved_attempt_metrics(
                collision_events, clean_pass_events, attempt_timeout_events
            )
        )
        if first_attempt_only:
            result["evaluation_window"] = (
                "first-terminal-attempt-per-environment-v1"
            )
            result.update(
                fixed_attempt_metrics(
                    expected_attempts=num_envs,
                    completed_attempts=int((~evaluation_active).sum()),
                    clean_pass_events=clean_pass_events,
                    collision_events=collision_events,
                    attempt_timeout_events=attempt_timeout_events,
                    fall_events=fall_events,
                    nan_termination_events=nan_events,
                    other_terminal_events=other_terminal_events,
                )
            )
        if collect_success_samples:
            all_keys = torch.cat(dataset_episode_keys)
            successful_keys = torch.tensor(
                sorted(successful_episode_keys), dtype=all_keys.dtype
            )
            if successful_keys.numel():
                keep = torch.isin(all_keys, successful_keys)
            else:
                keep = torch.zeros_like(all_keys, dtype=torch.bool)
            result["_success_dataset"] = {
                "observations": torch.cat(dataset_observations)[keep],
                "commands": torch.cat(dataset_commands)[keep],
                "episode_keys": all_keys[keep],
            }
        if collect_teacher_corrections:
            result["_teacher_correction_dataset"] = resolved_correction_samples(
                torch.cat(dataset_observations),
                torch.cat(dataset_commands),
                torch.cat(dataset_student_commands),
                torch.cat(dataset_episode_keys),
                correction_episode_outcomes,
            )
        return result
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_rollout(
    checkpoint: Path,
    output_dir: Path,
    *,
    num_envs: int,
    steps: int,
    speeds: tuple[float, ...],
    forward_positions: tuple[float, ...],
    lateral_positions: tuple[float, ...],
    seeds: tuple[int, ...],
    collect_success_dataset: bool = False,
    collect_teacher_corrections: bool = False,
    first_attempt_only: bool = False,
    supervisor_checkpoint: Path | None = None,
) -> Path:
    validate_rollout_bounds(
        num_envs, steps, speeds, forward_positions, lateral_positions, seeds
    )
    checkpoint = checkpoint.resolve(strict=True)
    validate_dataset_collection_mode(
        collect_success_dataset=collect_success_dataset,
        collect_teacher_corrections=collect_teacher_corrections,
        supervisor_checkpoint=supervisor_checkpoint,
    )
    if first_attempt_only and (
        collect_success_dataset or collect_teacher_corrections
    ):
        raise ValueError("first-attempt evaluation cannot collect training datasets")
    if supervisor_checkpoint is not None:
        supervisor_checkpoint = supervisor_checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    cases = []
    datasets = []
    case_arguments = [
        (speed, forward, lateral, seed)
        for speed in speeds
        for forward in forward_positions
        for lateral in lateral_positions
        for seed in seeds
    ]
    for case_index, (speed, forward, lateral, seed) in enumerate(case_arguments):
        case = _run_case(
            checkpoint,
            num_envs=num_envs,
            steps=steps,
            nominal_speed_mps=speed,
            obstacle_forward_m=forward,
            obstacle_lateral_m=lateral,
            seed=seed,
            collect_success_samples=collect_success_dataset,
            collect_teacher_corrections=collect_teacher_corrections,
            first_attempt_only=first_attempt_only,
            case_index=case_index,
            supervisor_checkpoint=supervisor_checkpoint,
        )
        dataset = case.pop("_success_dataset", None)
        if dataset is not None:
            datasets.append(dataset)
        correction_dataset = case.pop("_teacher_correction_dataset", None)
        if correction_dataset is not None:
            datasets.append(correction_dataset)
        cases.append(case)
    totals = {
        key: sum(case[key] for case in cases)
        for key in (
            "collision_events",
            "clean_pass_events",
            "attempt_timeout_events",
            "fall_events",
            "nan_termination_events",
            "nonfinite_steps",
        )
    }
    totals.update(
        _resolved_attempt_metrics(
            totals["collision_events"],
            totals["clean_pass_events"],
            totals["attempt_timeout_events"],
        )
    )
    if first_attempt_only:
        totals.update(
            fixed_attempt_metrics(
                expected_attempts=sum(case["expected_attempts"] for case in cases),
                completed_attempts=sum(case["completed_attempts"] for case in cases),
                clean_pass_events=totals["clean_pass_events"],
                collision_events=totals["collision_events"],
                attempt_timeout_events=totals["attempt_timeout_events"],
                fall_events=totals["fall_events"],
                nan_termination_events=totals["nan_termination_events"],
                other_terminal_events=sum(
                    case["other_terminal_events"] for case in cases
                ),
            )
        )
    report = {
        "schema_version": 1,
        "stage": "HC1-deterministic-teacher",
        "decision": "diagnostic-only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "base_task_id": BASE_TASK_ID,
        "obstacle_physics_task_id": OA0_TASK_ID,
        "teacher_config": asdict(ObstacleTeacherCfg()),
        "attempt_timeout_s": HC1_ATTEMPT_TIMEOUT_S,
        "perception": "exact structured geometry; no raw camera perception",
        "physical_motion_authorized": False,
        "evaluation_window": (
            "first-terminal-attempt-per-environment-v1"
            if first_attempt_only
            else "fixed-simulator-steps-legacy"
        ),
        "cases": cases,
        "totals": totals,
    }
    if supervisor_checkpoint is not None:
        import torch

        supervisor_payload = torch.load(
            supervisor_checkpoint, map_location="cpu", weights_only=False
        )
        report_stage = {
            HC2_STAGE: "HC2-behavioral-cloning-rollout",
            HC4L_STAGE: "HC4L-lateral-behavioral-cloning-rollout",
            HC4LH_STAGE: "HC4LH-lateral-gated-supervisor-rollout",
            HC4R_STAGE: "HC4R-near-range-behavioral-cloning-rollout",
            HC4R2_STAGE: "HC4R2-student-state-correction-BC-rollout",
            HC4R2H_STAGE: "HC4R2H-range-speed-gated-supervisor-rollout",
            HC4R2L_STAGE: "HC4R2L-episode-latched-supervisor-rollout",
            "HC3-supervisor-PPO": "HC3-supervisor-PPO-rollout",
            HC3E_STAGE: "HC3E-interaction-speed-PPO-rollout",
            HC3F_STAGE: "HC3F-seed-averaged-speed-head-rollout",
            HC3G_STAGE: "HC3G-seed-consensus-speed-head-rollout",
        }
        report["stage"] = report_stage[supervisor_payload.get("stage")]
        report["supervisor_checkpoint"] = str(supervisor_checkpoint)
        report["supervisor_checkpoint_sha256"] = _sha256(supervisor_checkpoint)
    if collect_success_dataset:
        if not datasets or not any(
            dataset["observations"].shape[0] for dataset in datasets
        ):
            raise RuntimeError("HC1 produced no successful teacher samples")
        import torch

        dataset_path = output_dir / "hc1-success-dataset.pt"
        dataset_payload = {
            "schema_version": 1,
            "stage": "HC1-successful-teacher-trajectories",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "teacher_config": asdict(ObstacleTeacherCfg()),
            "observation_dim": SUPERVISOR_OBSERVATION_DIM,
            "command_fields": ["forward_speed_mps", "yaw_rate_rps"],
            "observations": torch.cat(
                [dataset["observations"] for dataset in datasets]
            ),
            "commands": torch.cat([dataset["commands"] for dataset in datasets]),
            "episode_keys": torch.cat(
                [dataset["episode_keys"] for dataset in datasets]
            ),
        }
        torch.save(dataset_payload, dataset_path)
        report["success_dataset"] = str(dataset_path)
        report["success_dataset_samples"] = int(
            dataset_payload["observations"].shape[0]
        )
        report["success_dataset_episodes"] = int(
            torch.unique(dataset_payload["episode_keys"]).numel()
        )
    if collect_teacher_corrections:
        if not datasets or not any(
            dataset["observations"].shape[0] for dataset in datasets
        ):
            raise RuntimeError("student rollout produced no resolved correction samples")
        import torch

        assert supervisor_checkpoint is not None
        correction_path = output_dir / "teacher-correction-dataset.pt"
        correction_payload = {
            "schema_version": 1,
            "stage": "HC4R2-student-state-teacher-corrections",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "student_supervisor_checkpoint": str(supervisor_checkpoint),
            "student_supervisor_checkpoint_sha256": _sha256(supervisor_checkpoint),
            "teacher_config": asdict(ObstacleTeacherCfg()),
            "observation_dim": SUPERVISOR_OBSERVATION_DIM,
            "command_fields": ["forward_speed_mps", "yaw_rate_rps"],
            "outcome_codes": {"clean_pass": 1, "collision": 2, "timeout": 3},
            "observations": torch.cat(
                [dataset["observations"] for dataset in datasets]
            ),
            "commands": torch.cat([dataset["commands"] for dataset in datasets]),
            "student_commands": torch.cat(
                [dataset["student_commands"] for dataset in datasets]
            ),
            "episode_keys": torch.cat(
                [dataset["episode_keys"] for dataset in datasets]
            ),
            "sample_outcome_codes": torch.cat(
                [dataset["outcome_codes"] for dataset in datasets]
            ),
        }
        torch.save(correction_payload, correction_path)
        unique_episode_keys = torch.unique(correction_payload["episode_keys"])
        report["teacher_correction_dataset"] = str(correction_path)
        report["teacher_correction_dataset_samples"] = int(
            correction_payload["observations"].shape[0]
        )
        report["teacher_correction_dataset_episodes"] = int(
            unique_episode_keys.numel()
        )
        report["teacher_correction_episode_outcomes"] = {
            name: int(
                torch.unique(
                    correction_payload["episode_keys"][
                        correction_payload["sample_outcome_codes"] == code
                    ]
                ).numel()
            )
            for name, code in correction_payload["outcome_codes"].items()
        }
    output_path = output_dir / "hierarchical-teacher-evaluation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals, sort_keys=True))
    print(f"[HC1] wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--speeds", default="0.3")
    parser.add_argument("--obstacle-forward", default="1.15")
    parser.add_argument("--obstacle-lateral", default="-0.27,0.27")
    parser.add_argument("--seeds", default="41")
    parser.add_argument("--collect-success-dataset", action="store_true")
    parser.add_argument("--collect-teacher-corrections", action="store_true")
    parser.add_argument("--first-attempt-only", action="store_true")
    parser.add_argument("--supervisor-checkpoint", type=Path)
    args = parser.parse_args()
    run_rollout(
        args.checkpoint,
        args.output_dir,
        num_envs=args.num_envs,
        steps=args.steps,
        speeds=parse_float_list(args.speeds),
        forward_positions=parse_float_list(args.obstacle_forward),
        lateral_positions=parse_float_list(args.obstacle_lateral),
        seeds=parse_int_list(args.seeds),
        collect_success_dataset=args.collect_success_dataset,
        collect_teacher_corrections=args.collect_teacher_corrections,
        first_attempt_only=args.first_attempt_only,
        supervisor_checkpoint=args.supervisor_checkpoint,
    )


if __name__ == "__main__":
    main()
