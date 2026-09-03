"""Evaluate a MicroDuck checkpoint across fixed speeds and random seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.evaluation import (
    aggregate_command_cases,
    aggregate_speed_cases,
    fix_velocity_commands,
    parse_float_list,
    parse_int_list,
    valid_action_deltas,
)
from mjlab_microduck.tasks.run import (
    MOTOR_NEAR_LIMIT_FRACTION,
    XL330_M288_RATED_NO_LOAD_SPEED_RAD_S,
    XL330_M288_RATED_STALL_TORQUE_NM_6V,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item())


def _run_case(args, speed: float, yaw_rate: float, seed: int) -> dict:
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args.num_envs
    fix_velocity_commands(env_cfg, speed, yaw_rate)
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(args.task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        observations = env.get_observations()

        robot = raw_env.scene["robot"]
        joint_ids, _ = robot.find_joints(r"^(?!passive_).*")
        forward_samples = []
        yaw_rate_samples = []
        applied_command_speed_samples = []
        applied_command_yaw_samples = []
        joint_speed_samples = []
        torque_samples = []
        power_samples = []
        action_samples = []
        action_rate_samples = []
        episode_ends = 0
        timeouts = 0
        fall_events = 0
        nan_termination_events = 0
        nonfinite_steps = 0
        previous_actions = raw_env.action_manager.action.detach().clone()
        previous_dones = torch.zeros(
            args.num_envs, dtype=torch.bool, device=args.device
        )

        with torch.inference_mode():
            for step in range(args.warmup_steps + args.steps):
                actions = policy(observations)
                observations, rewards, dones, extras = env.step(actions)
                applied_actions = raw_env.action_manager.action.detach()
                if step < args.warmup_steps:
                    previous_actions = applied_actions.clone()
                    previous_dones = dones.bool().clone()
                    continue

                forward = robot.data.root_link_lin_vel_b[:, 0].float()
                observed_yaw_rate = robot.data.root_link_ang_vel_b[:, 2].float()
                applied_command = raw_env.command_manager.get_command("twist")
                joint_speed = robot.data.joint_vel[:, joint_ids].abs().float()
                torque = robot.data.actuator_force.abs().float()
                if joint_speed.shape != torque.shape:
                    raise RuntimeError(
                        f"joint speed shape {joint_speed.shape} does not match "
                        f"actuator torque shape {torque.shape}"
                    )
                forward_samples.append(forward)
                yaw_rate_samples.append(observed_yaw_rate)
                applied_command_speed_samples.append(applied_command[:, 0].float())
                applied_command_yaw_samples.append(applied_command[:, 2].float())
                joint_speed_samples.append(joint_speed)
                torque_samples.append(torque)
                power_samples.append(torch.sum(joint_speed * torque, dim=1))
                action_samples.append(applied_actions.abs().float())
                action_deltas = valid_action_deltas(
                    applied_actions, previous_actions, previous_dones
                )
                if action_deltas.numel():
                    action_rate_samples.append(action_deltas.float())
                episode_ends += int(dones.sum().item())
                timeout_tensor = extras.get("time_outs")
                if timeout_tensor is not None:
                    timeouts += int(timeout_tensor.sum().item())
                fall_events += int(
                    raw_env.termination_manager.get_term("fell_over").sum().item()
                )
                nan_termination_events += int(
                    raw_env.termination_manager.get_term("nan_state").sum().item()
                )
                finite = torch.isfinite(actions).all()
                finite &= torch.isfinite(applied_actions).all()
                finite &= torch.isfinite(rewards).all()
                finite &= all(
                    torch.isfinite(value).all() for value in observations.values()
                )
                nonfinite_steps += int(not bool(finite))
                previous_actions = applied_actions.clone()
                previous_dones = dones.bool().clone()

        forward = torch.cat(forward_samples)
        observed_yaw_rate = torch.cat(yaw_rate_samples)
        applied_command_speed = torch.cat(applied_command_speed_samples)
        applied_command_yaw = torch.cat(applied_command_yaw_samples)
        joint_speed = torch.cat(joint_speed_samples).flatten()
        torque = torch.cat(torque_samples).flatten()
        power = torch.cat(power_samples)
        actions = torch.cat(action_samples).flatten()
        if not action_rate_samples:
            raise RuntimeError("evaluation produced no valid action-rate samples")
        action_rates = torch.cat(action_rate_samples)
        speed_util = joint_speed / XL330_M288_RATED_NO_LOAD_SPEED_RAD_S
        torque_util = torque / XL330_M288_RATED_STALL_TORQUE_NM_6V
        tracking_error = (forward - speed).abs()
        yaw_tracking_error = (observed_yaw_rate - yaw_rate).abs()

        return {
            "commanded_speed_mps": speed,
            "commanded_yaw_rate_rps": yaw_rate,
            "applied_command_speed_mean_mps": float(
                applied_command_speed.mean().item()
            ),
            "applied_command_yaw_rate_mean_rps": float(
                applied_command_yaw.mean().item()
            ),
            "seed": seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "simulated_seconds_per_env": args.steps * raw_env.step_dt,
            "observed_speed_mean_mps": float(forward.mean().item()),
            "observed_speed_p05_mps": _quantile(forward, 0.05),
            "observed_speed_p95_mps": _quantile(forward, 0.95),
            "tracking_error_mean_mps": float(tracking_error.mean().item()),
            "tracking_error_p95_mps": _quantile(tracking_error, 0.95),
            "observed_yaw_rate_mean_rps": float(observed_yaw_rate.mean().item()),
            "observed_yaw_rate_p05_rps": _quantile(observed_yaw_rate, 0.05),
            "observed_yaw_rate_p95_rps": _quantile(observed_yaw_rate, 0.95),
            "yaw_tracking_error_mean_rps": float(yaw_tracking_error.mean().item()),
            "yaw_tracking_error_p95_rps": _quantile(yaw_tracking_error, 0.95),
            "episode_ends": episode_ends,
            "timeouts": timeouts,
            "non_timeout_ends": episode_ends - timeouts,
            "fall_events": fall_events,
            "nan_termination_events": nan_termination_events,
            "nonfinite_steps": nonfinite_steps,
            "motor_joint_speed_abs_max_rad_s": float(joint_speed.max().item()),
            "motor_joint_speed_abs_p99_rad_s": _quantile(joint_speed, 0.99),
            "motor_speed_utilization_p99": _quantile(speed_util, 0.99),
            "motor_speed_rated_exceed_fraction": float(
                (speed_util > 1.0).float().mean().item()
            ),
            "motor_torque_abs_max_nm": float(torque.max().item()),
            "motor_torque_abs_p99_nm": _quantile(torque, 0.99),
            "motor_torque_utilization_p99": _quantile(torque_util, 0.99),
            "motor_torque_near_stall_fraction": float(
                (torque_util >= MOTOR_NEAR_LIMIT_FRACTION).float().mean().item()
            ),
            "motor_mechanical_power_abs_mean_w": float(power.mean().item()),
            "motor_mechanical_power_abs_p99_w": _quantile(power, 0.99),
            "motor_thermal_load_proxy_mean": float(
                torch.square(torque_util).mean().item()
            ),
            "action_abs_max": float(actions.max().item()),
            "action_abs_p99": _quantile(actions, 0.99),
            "action_rate_abs_max": float(action_rates.max().item()),
            "action_rate_abs_mean": float(action_rates.mean().item()),
            "action_rate_abs_p99": _quantile(action_rates, 0.99),
        }
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="Mjlab-Run-Flat-MicroDuck")
    parser.add_argument("--speeds", default="0.5,0.8,1.0,1.2,1.5")
    parser.add_argument("--yaw-rates", default="0.0")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.speeds = parse_float_list(args.speeds)
    args.yaw_rates = parse_float_list(args.yaw_rates)
    args.seeds = parse_int_list(args.seeds)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.num_envs <= 0 or args.steps <= 0 or args.warmup_steps < 0:
        parser.error("num-envs and steps must be positive; warmup-steps cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    import mjlab_microduck.tasks  # noqa: F401

    configure_torch_backends()
    cases = []
    for speed in args.speeds:
        for yaw_rate in args.yaw_rates:
            for seed in args.seeds:
                print(
                    f"[EVAL] speed={speed:.2f} m/s "
                    f"yaw={yaw_rate:.2f} rad/s seed={seed}",
                    flush=True,
                )
                case = _run_case(args, speed, yaw_rate, seed)
                cases.append(case)
                print(json.dumps(case, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "speeds_mps": list(args.speeds),
        "yaw_rates_rps": list(args.yaw_rates),
        "seeds": list(args.seeds),
        "cases": cases,
        "aggregates": aggregate_speed_cases(cases),
        "command_aggregates": aggregate_command_cases(cases),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "checkpoint-evaluation.json"
    csv_path = args.output_dir / "checkpoint-evaluation-cases.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(f"[EVAL] wrote {json_path}")
    print(f"[EVAL] wrote {csv_path}")


if __name__ == "__main__":
    main()
