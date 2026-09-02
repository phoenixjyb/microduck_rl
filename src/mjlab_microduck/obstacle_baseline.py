"""Retained inference-only baseline for an untrained obstacle warm start."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path


TASK_ID = "Mjlab-Run-Obstacle-Flat-MicroDuck"
MAX_BASELINE_ENVS = 256
MAX_BASELINE_STEPS = 1000
MAX_BASELINE_SEEDS = 5


def validate_baseline_bounds(num_envs: int, steps: int, seeds: tuple[int, ...]) -> None:
    if not 1 <= num_envs <= MAX_BASELINE_ENVS:
        raise ValueError(f"num_envs must be in [1, {MAX_BASELINE_ENVS}]")
    if not 1 <= steps <= MAX_BASELINE_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_BASELINE_STEPS}]")
    if not 1 <= len(seeds) <= MAX_BASELINE_SEEDS:
        raise ValueError(f"seed count must be in [1, {MAX_BASELINE_SEEDS}]")


def prepare_baseline_configs(num_envs: int, speed_mps: float):
    """Build a deterministic play config with one fixed straight command."""
    if speed_mps <= 0.0:
        raise ValueError("speed_mps must be positive")
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.terrain.num_envs = num_envs
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (speed_mps, speed_mps)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 1.0
    twist.resampling_time_range = (1000.0, 1000.0)

    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    return env_cfg, agent_cfg


def _run_case(checkpoint: Path, num_envs: int, steps: int, speed_mps: float, seed: int) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env_cfg, agent_cfg = prepare_baseline_configs(num_envs, speed_mps)
    env_cfg.seed = seed
    agent_cfg.seed = seed
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    torch.manual_seed(seed)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        infos = runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        if infos.get("obstacle_warm_start", {}).get("actor_dims") != [61, 68]:
            raise ValueError("checkpoint lacks obstacle warm-start metadata")
        policy = runner.get_inference_policy(device=device)
        obs = wrapped.get_observations()

        collision_events = 0
        fall_events = 0
        timeout_events = 0
        nan_events = 0
        pass_events = 0
        nonfinite_steps = 0
        clearance_sum = 0.0
        forward_speed_sum = 0.0
        sample_count = 0
        passed_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)

        for _ in range(steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, _ = wrapped.step(actions)

            collision = env.termination_manager.get_term("obstacle_collision")
            fell = env.termination_manager.get_term("fell_over")
            timed_out = env.termination_manager.get_term("time_out")
            nan_state = env.termination_manager.get_term("nan_state")
            collision_events += int(collision.sum())
            fall_events += int(fell.sum())
            timeout_events += int(timed_out.sum())
            nan_events += int(nan_state.sum())

            passed = microduck_mdp.obstacle_passed_reward(env).bool()
            pass_events += int((passed & ~passed_latched).sum())
            passed_latched |= passed
            passed_latched[dones.bool()] = False

            robot = env.scene["robot"]
            obstacle = env.scene["obstacle"]
            center_distance = torch.linalg.vector_norm(
                obstacle.data.root_link_pos_w[:, :2]
                - robot.data.root_link_pos_w[:, :2],
                dim=-1,
            )
            clearance_sum += float((center_distance - 0.22).sum())
            forward_speed_sum += float(
                (
                    robot.data.root_link_lin_vel_w[:, :2]
                    * env._obstacle_path_dir_w
                ).sum(dim=-1).sum()
            )
            sample_count += num_envs
            finite = torch.isfinite(actions).all() & torch.isfinite(rewards).all()
            finite &= all(torch.isfinite(value).all() for value in obs.values())
            nonfinite_steps += int(not bool(finite))

        return {
            "seed": seed,
            "num_envs": num_envs,
            "steps": steps,
            "commanded_speed_mps": speed_mps,
            "collision_events": collision_events,
            "fall_events": fall_events,
            "timeout_events": timeout_events,
            "nan_termination_events": nan_events,
            "clean_pass_events": pass_events,
            "nonfinite_steps": nonfinite_steps,
            "mean_clearance_m": clearance_sum / sample_count,
            "mean_forward_speed_mps": forward_speed_sum / sample_count,
        }
    finally:
        env.close()


def run_baseline(
    checkpoint: Path,
    output_dir: Path,
    *,
    num_envs: int = 64,
    steps: int = 600,
    speed_mps: float = 0.5,
    seeds: tuple[int, ...] = (41, 42, 43),
) -> Path:
    validate_baseline_bounds(num_envs, steps, seeds)
    checkpoint = checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    cases = [
        _run_case(checkpoint, num_envs, steps, speed_mps, seed) for seed in seeds
    ]
    totals = {
        key: sum(case[key] for case in cases)
        for key in (
            "collision_events",
            "fall_events",
            "timeout_events",
            "nan_termination_events",
            "clean_pass_events",
            "nonfinite_steps",
        )
    }
    summary = {
        "task_id": TASK_ID,
        "checkpoint": str(checkpoint),
        "purpose": "untrained obstacle warm-start baseline; not trained-policy evidence",
        "cases": cases,
        "totals": totals,
        "mean_clearance_m": sum(c["mean_clearance_m"] for c in cases) / len(cases),
        "mean_forward_speed_mps": sum(c["mean_forward_speed_mps"] for c in cases)
        / len(cases),
    }
    output_path = output_dir / "obstacle_baseline.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"obstacle_baseline_retained={output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--speed-mps", type=float, default=0.5)
    parser.add_argument("--seeds", default="41,42,43")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    run_baseline(
        args.checkpoint,
        args.output_dir,
        num_envs=args.num_envs,
        steps=args.steps,
        speed_mps=args.speed_mps,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
