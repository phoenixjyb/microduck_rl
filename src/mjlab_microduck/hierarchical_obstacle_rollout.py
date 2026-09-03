"""Bounded HC1 rollout of a command teacher over a frozen locomotion actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from mjlab_microduck.evaluation import parse_float_list, parse_int_list
from mjlab_microduck.hierarchical_obstacle import (
    ObstaclePhase,
    ObstacleTeacherCfg,
    make_teacher_state,
    reset_teacher_state,
    teacher_command,
)
from mjlab_microduck.obstacle_baseline import _resolved_attempt_metrics
from mjlab_microduck.obstacle_protocol import OA0_TASK_ID


BASE_TASK_ID = "Mjlab-Run-MotorAware-Flat-MicroDuck"
MAX_ENVS = 256
MAX_STEPS = 1000
MAX_CASES = 48


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def prepare_rollout_configs(
    num_envs: int,
    nominal_speed_mps: float,
    obstacle_forward_m: float,
    obstacle_lateral_m: float,
):
    """Build OA0 physics with the original 61D frozen-policy observation."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

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

        with torch.inference_mode():
            for step in range(steps):
                route_lateral, route_heading, route_speed = _route_state(env)
                lateral_abs_max = torch.maximum(lateral_abs_max, route_lateral.abs())
                if step % 5 == 0:
                    obstacle_observation = microduck_mdp.obstacle_geometry_observation(
                        env,
                        asset_name="obstacle",
                        width_m=0.20,
                        height_m=0.10,
                        horizontal_fov_rad=2.0 * math.pi,
                        max_range_m=2.0,
                    )
                    supervisor_command = teacher_command(
                        obstacle_observation,
                        nominal,
                        route_lateral,
                        route_heading,
                        state,
                        cfg=ObstacleTeacherCfg(),
                    )
                    command[:, 0] = supervisor_command[:, 0]
                    command[:, 1] = 0.0
                    command[:, 2] = supervisor_command[:, 1]

                for phase in ObstaclePhase:
                    mask = state.phase == int(phase)
                    phase_speed_sum[int(phase)] += torch.nan_to_num(
                        route_speed[mask], nan=0.0
                    ).sum()
                    phase_samples[int(phase)] += mask.sum()
                command_speed_min = min(command_speed_min, float(command[:, 0].min()))
                command_speed_max = max(command_speed_max, float(command[:, 0].max()))
                command_yaw_abs_max = max(
                    command_yaw_abs_max, float(command[:, 2].abs().max())
                )

                actions = policy(observations)
                observations, rewards, dones, _ = wrapped.step(actions)
                collision = env.termination_manager.get_term("obstacle_collision")
                passed = env.termination_manager.get_term("obstacle_passed")
                attempted_out = env.termination_manager.get_term(
                    "obstacle_attempt_timeout"
                )
                fell = env.termination_manager.get_term("fell_over")
                nan_state = env.termination_manager.get_term("nan_state")
                collision_events += int(collision.sum())
                clean_pass_events += int(passed.sum())
                attempt_timeout_events += int(attempted_out.sum())
                fall_events += int(fell.sum())
                nan_events += int(nan_state.sum())
                pass_lateral_sum += float(lateral_abs_max[passed].sum())
                pass_lateral_count += int(passed.sum())
                finite = torch.isfinite(actions).all() & torch.isfinite(rewards).all()
                finite &= all(
                    torch.isfinite(value).all() for value in observations.values()
                )
                nonfinite_steps += int(not bool(finite))
                reset_teacher_state(
                    state, dones.bool(), nominal_speed_mps=nominal_speed_mps
                )
                lateral_abs_max[dones.bool()] = 0.0

        result = {
            "nominal_speed_mps": nominal_speed_mps,
            "obstacle_forward_m": obstacle_forward_m,
            "obstacle_lateral_m": obstacle_lateral_m,
            "seed": seed,
            "num_envs": num_envs,
            "steps": steps,
            "collision_events": collision_events,
            "clean_pass_events": clean_pass_events,
            "attempt_timeout_events": attempt_timeout_events,
            "fall_events": fall_events,
            "nan_termination_events": nan_events,
            "nonfinite_steps": nonfinite_steps,
            "mean_pass_lateral_excursion_m": (
                pass_lateral_sum / pass_lateral_count if pass_lateral_count else None
            ),
            "command_speed_min_mps": command_speed_min,
            "command_speed_max_mps": command_speed_max,
            "command_yaw_abs_max_rps": command_yaw_abs_max,
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
) -> Path:
    validate_rollout_bounds(
        num_envs, steps, speeds, forward_positions, lateral_positions, seeds
    )
    checkpoint = checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    cases = [
        _run_case(
            checkpoint,
            num_envs=num_envs,
            steps=steps,
            nominal_speed_mps=speed,
            obstacle_forward_m=forward,
            obstacle_lateral_m=lateral,
            seed=seed,
        )
        for speed in speeds
        for forward in forward_positions
        for lateral in lateral_positions
        for seed in seeds
    ]
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
    report = {
        "schema_version": 1,
        "stage": "HC1-deterministic-teacher",
        "decision": "diagnostic-only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "base_task_id": BASE_TASK_ID,
        "obstacle_physics_task_id": OA0_TASK_ID,
        "teacher_config": asdict(ObstacleTeacherCfg()),
        "perception": "exact structured geometry; no raw camera perception",
        "physical_motion_authorized": False,
        "cases": cases,
        "totals": totals,
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
    )


if __name__ == "__main__":
    main()
