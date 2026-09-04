"""Evaluate one periodic-hop checkpoint on the fixed held-out H1 protocol."""

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

from mjlab_microduck.evaluation import parse_int_list
from mjlab_microduck.hop_evaluation import H1_PROTOCOL, h1_decision, summarize_hop_trace
from mjlab_microduck.robot.sprung_foot import SPRING_JOINTS, TRAVEL
from mjlab_microduck.tasks.hop import HOP_PERIOD, SENSOR_NAME
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
    return float(torch.quantile(values.float(), q).item())


def _both_airborne(raw_env: ManagerBasedRlEnv) -> torch.Tensor:
    found = raw_env.scene.sensors[SENSOR_NAME].data.found
    if found is None or found.shape[1] < 2:
        raise RuntimeError(f"{SENSOR_NAME} does not expose two foot contacts")
    found = torch.nan_to_num(found[:, :2].float(), nan=1.0)
    return (found[:, 0] <= 0) & (found[:, 1] <= 0)


def _landing_force(raw_env: ManagerBasedRlEnv) -> torch.Tensor:
    force = raw_env.scene.sensors[SENSOR_NAME].data.force
    if force is None or force.shape[1] < 2:
        raise RuntimeError(f"{SENSOR_NAME} does not expose two foot forces")
    return torch.linalg.vector_norm(torch.nan_to_num(force[:, :2].float()), dim=-1)


def _zero_auxiliary_commands(env_cfg) -> None:
    for name in ("head_pose", "body_pose"):
        command = env_cfg.commands[name]
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.ranges = tuple((0.0, 0.0) for _ in command.ranges)
        command.zero_command_prob = 1.0


def _finite_by_env(
    actions: torch.Tensor,
    applied_actions: torch.Tensor,
    rewards: torch.Tensor,
    observations: dict[str, torch.Tensor],
) -> torch.Tensor:
    def rows_finite(value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)

    finite = rows_finite(actions)
    finite &= rows_finite(applied_actions)
    finite &= rows_finite(rewards)
    for value in observations.values():
        finite &= rows_finite(value)
    return finite


def _run_case(args: argparse.Namespace, seed: int) -> dict:
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)
    env_cfg.commands["twist"].randomize_phase = False
    _zero_auxiliary_commands(env_cfg)
    # Keep the framework timeout beyond this protocol's artificial horizon.
    env_cfg.episode_length_s = (args.cycles + 1) * HOP_PERIOD

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        steps_per_cycle = round(HOP_PERIOD / raw_env.step_dt)
        if not torch.isclose(
            torch.tensor(steps_per_cycle * raw_env.step_dt),
            torch.tensor(HOP_PERIOD),
            atol=1.0e-6,
        ):
            raise RuntimeError("hop period is not an integer number of control steps")
        steps = steps_per_cycle * args.cycles

        runner_cls = load_runner_cls(args.task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint), strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        observations = env.get_observations()

        robot = raw_env.scene["robot"]
        motor_joint_ids, _ = robot.find_joints(r"^(?!passive_).*")
        spring_joint_ids = []
        for name in SPRING_JOINTS:
            try:
                found, _ = robot.find_joints(name)
            except ValueError:
                found = []
            if found:
                spring_joint_ids.append(found[0])

        base_z = [robot.data.root_link_pos_w[:, 2].detach().cpu()]
        base_xy = [robot.data.root_link_pos_w[:, :2].detach().cpu()]
        airborne = [_both_airborne(raw_env).detach().cpu()]
        dones_trace = []
        falls_trace = []
        nan_trace = []
        finite_trace = []
        compression_trace = []
        landing_force_trace = []

        joint_speed_samples = []
        torque_samples = []
        power_samples = []
        action_samples = []
        action_rate_samples = []
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        previous_actions = None

        with torch.inference_mode():
            for _ in range(steps):
                actions = policy(observations)
                observations, rewards, dones, _extras = env.step(actions)
                applied_actions = raw_env.action_manager.action.detach()
                finite = _finite_by_env(actions, applied_actions, rewards, observations)
                falls = raw_env.termination_manager.get_term("fell_over").bool()
                nan_terminations = raw_env.termination_manager.get_term("nan_state").bool()

                base_z.append(robot.data.root_link_pos_w[:, 2].detach().cpu())
                base_xy.append(robot.data.root_link_pos_w[:, :2].detach().cpu())
                airborne.append(_both_airborne(raw_env).detach().cpu())
                dones_trace.append(dones.bool().detach().cpu())
                falls_trace.append(falls.detach().cpu())
                nan_trace.append(nan_terminations.detach().cpu())
                finite_trace.append(finite.detach().cpu())
                landing_force_trace.append(_landing_force(raw_env).detach().cpu())

                if spring_joint_ids:
                    compression = (
                        torch.nan_to_num(
                            robot.data.joint_pos[:, spring_joint_ids].float()
                        ).clamp(min=0.0)
                        / TRAVEL
                    )
                else:
                    compression = torch.zeros(
                        (args.num_envs, 1), device=args.device
                    )
                compression_trace.append(compression.detach().cpu())

                joint_speed = robot.data.joint_vel[:, motor_joint_ids].abs().float()
                torque = robot.data.actuator_force.abs().float()
                if joint_speed.shape != torque.shape:
                    raise RuntimeError(
                        f"joint speed shape {joint_speed.shape} does not match "
                        f"actuator torque shape {torque.shape}"
                    )
                joint_speed_samples.append(joint_speed[active].flatten().cpu())
                torque_samples.append(torque[active].flatten().cpu())
                power_samples.append(
                    torch.sum(joint_speed * torque, dim=1)[active].cpu()
                )
                action_samples.append(applied_actions[active].abs().flatten().cpu())
                if previous_actions is not None:
                    action_rate_samples.append(
                        (applied_actions - previous_actions)[active].abs().flatten().cpu()
                    )
                previous_actions = applied_actions.clone()
                active &= ~dones.bool()

        case = summarize_hop_trace(
            base_z=torch.stack(base_z),
            base_xy=torch.stack(base_xy),
            both_airborne=torch.stack(airborne),
            dones=torch.stack(dones_trace),
            falls=torch.stack(falls_trace),
            nan_terminations=torch.stack(nan_trace),
            finite=torch.stack(finite_trace),
            spring_compression_ratio=torch.stack(compression_trace),
            landing_force=torch.stack(landing_force_trace),
            steps_per_cycle=steps_per_cycle,
            cycles=args.cycles,
        )

        joint_speed = torch.cat(joint_speed_samples)
        torque = torch.cat(torque_samples)
        power = torch.cat(power_samples)
        applied = torch.cat(action_samples)
        action_rate = torch.cat(action_rate_samples)
        speed_util = joint_speed / XL330_M288_RATED_NO_LOAD_SPEED_RAD_S
        torque_util = torque / XL330_M288_RATED_STALL_TORQUE_NM_6V
        case.update(
            {
                "seed": seed,
                "steps_per_cycle": steps_per_cycle,
                "simulated_seconds_per_env": steps * raw_env.step_dt,
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
                "action_abs_max": float(applied.max().item()),
                "action_abs_p99": _quantile(applied, 0.99),
                "action_rate_abs_max": float(action_rate.max().item()),
                "action_rate_abs_mean": float(action_rate.mean().item()),
                "action_rate_abs_p99": _quantile(action_rate, 0.99),
            }
        )
        return case
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task", default="Mjlab-Hop-Flat-Sprung-K3900-MicroDuck"
    )
    parser.add_argument("--seeds", default="211,223,227")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.seeds = parse_int_list(args.seeds)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.num_envs <= 0 or args.cycles <= 0:
        parser.error("num-envs and cycles must be positive")
    return args


def main() -> None:
    args = parse_args()
    import mjlab_microduck.tasks  # noqa: F401

    configure_torch_backends()
    cases = []
    for seed in args.seeds:
        print(f"[H1 EVAL] seed={seed}", flush=True)
        case = _run_case(args, seed)
        cases.append(case)
        print(json.dumps(case, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "protocol": H1_PROTOCOL,
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "seeds": list(args.seeds),
        "num_envs": args.num_envs,
        "cycles": args.cycles,
        "cases": cases,
        **h1_decision(cases),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "hop-checkpoint-evaluation.json"
    csv_path = args.output_dir / "hop-checkpoint-evaluation-cases.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(f"[H1 EVAL] decision={result['decision']}")
    print(f"[H1 EVAL] wrote {json_path}")
    print(f"[H1 EVAL] wrote {csv_path}")


if __name__ == "__main__":
    main()
