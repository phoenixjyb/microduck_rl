"""Bounded HC1 rollout of a command teacher over a frozen locomotion actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from mjlab_microduck.evaluation import (
    parse_float_list,
    parse_int_list,
    valid_action_deltas,
)
from mjlab_microduck.hierarchical_obstacle import (
    ObstaclePhase,
    ObstacleTeacherCfg,
    advance_obstacle_state,
    apply_bounded_supervisor_command,
    make_teacher_state,
    reset_teacher_state,
    supervisor_observation,
    teacher_command,
)
from mjlab_microduck.obstacle_baseline import _resolved_attempt_metrics
from mjlab_microduck.obstacle_protocol import OA0_TASK_ID
from mjlab_microduck.obstacle_supervisor_bc import (
    ObstacleSupervisor,
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


def recording_stem(
    speed: float, obstacle_forward: float, obstacle_lateral: float
) -> str:
    """Return a deterministic replay basename for one geometry cell."""
    return (
        f"microduck-hc1-{speed:.2f}mps-"
        f"x{obstacle_forward:.2f}m-y{obstacle_lateral:+.2f}m"
    )


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
            supervisor_payload = torch.load(
                supervisor_checkpoint, map_location=device, weights_only=False
            )
            if supervisor_payload.get("decision") != "offline-imitation-pass":
                raise ValueError("supervisor checkpoint did not pass offline imitation")
            if (
                supervisor_payload.get("source_locomotion_checkpoint_sha256")
                != _sha256(checkpoint)
            ):
                raise ValueError("supervisor was trained for another locomotion checkpoint")
            model_cfg = dict(supervisor_payload["model_config"])
            model_cfg["hidden_dims"] = tuple(model_cfg["hidden_dims"])
            learned_supervisor = ObstacleSupervisor(SupervisorBcCfg(**model_cfg)).to(
                device
            )
            learned_supervisor.load_state_dict(
                supervisor_payload["model_state_dict"], strict=True
            )
            learned_supervisor.eval()

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
        dataset_episode_keys = []
        successful_episode_keys: set[int] = set()

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
                    previous_supervisor_command = state.previous_command.clone()
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
                episode_steps += 1
                applied_actions = env.action_manager.action.detach()
                joint_speed = robot.data.joint_vel[:, joint_ids].abs().float()
                torque = robot.data.actuator_force.abs().float()
                if joint_speed.shape != torque.shape:
                    raise RuntimeError(
                        f"joint speed shape {joint_speed.shape} does not match "
                        f"actuator torque shape {torque.shape}"
                    )
                joint_speed_samples.append(joint_speed)
                torque_samples.append(torque)
                action_samples.append(applied_actions.abs().float())
                action_deltas = valid_action_deltas(
                    applied_actions, previous_actions, previous_dones
                )
                if action_deltas.numel():
                    action_rate_samples.append(action_deltas.float())
                collision = env.termination_manager.get_term("obstacle_collision")
                passed = env.termination_manager.get_term("obstacle_passed")
                attempted_out = env.termination_manager.get_term(
                    "obstacle_attempt_timeout"
                )
                fell = env.termination_manager.get_term("fell_over")
                nan_state = env.termination_manager.get_term("nan_state")
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
                finite = torch.isfinite(actions).all() & torch.isfinite(rewards).all()
                finite &= all(
                    torch.isfinite(value).all() for value in observations.values()
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
    supervisor_checkpoint: Path | None = None,
) -> Path:
    validate_rollout_bounds(
        num_envs, steps, speeds, forward_positions, lateral_positions, seeds
    )
    checkpoint = checkpoint.resolve(strict=True)
    if collect_success_dataset and supervisor_checkpoint is not None:
        raise ValueError("success datasets may be collected only from the teacher")
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
            case_index=case_index,
            supervisor_checkpoint=supervisor_checkpoint,
        )
        dataset = case.pop("_success_dataset", None)
        if dataset is not None:
            datasets.append(dataset)
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
    report = {
        "schema_version": 1,
        "stage": (
            "HC1-deterministic-teacher"
            if supervisor_checkpoint is None
            else "HC2-behavioral-cloning-rollout"
        ),
        "decision": "diagnostic-only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "base_task_id": BASE_TASK_ID,
        "obstacle_physics_task_id": OA0_TASK_ID,
        "teacher_config": asdict(ObstacleTeacherCfg()),
        "attempt_timeout_s": HC1_ATTEMPT_TIMEOUT_S,
        "perception": "exact structured geometry; no raw camera perception",
        "physical_motion_authorized": False,
        "cases": cases,
        "totals": totals,
    }
    if supervisor_checkpoint is not None:
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
            "observation_dim": 17,
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
        supervisor_checkpoint=args.supervisor_checkpoint,
    )


if __name__ == "__main__":
    main()
