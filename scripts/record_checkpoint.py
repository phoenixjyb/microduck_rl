"""Record a bounded MP4 replay of a MicroDuck velocity-family checkpoint."""

from __future__ import annotations

import argparse
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
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fix_commands(env_cfg, speed: float) -> None:
    """Hold a straight-ahead velocity command and neutral pose commands."""
    commands = env_cfg.commands
    twist = commands["twist"]
    twist.resampling_time_range = (1.0e6, 1.0e6)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 1.0
    twist.rel_turn_in_place_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.ranges.lin_vel_x = (speed, speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    for name in ("head_pose", "body_pose"):
        command = commands[name]
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.ranges = tuple((0.0, 0.0) for _ in command.ranges)
        command.zero_command_prob = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default="Mjlab-Run-Flat-MicroDuck")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.frames <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("frames, width, and height must be positive")

    # Import the project task package so its registry entry points are loaded.
    import mjlab_microduck.tasks  # noqa: F401

    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.distance = 1.15
    env_cfg.viewer.azimuth = 115.0
    env_cfg.viewer.elevation = -18.0
    env_cfg.viewer.lookat = (0.0, 0.0, 0.12)
    # MuJoCo Warp's off-screen renderer can leave triangular shadow/reflection
    # artifacts on Blackwell under WSL. These are cosmetic and disabling both
    # produces a clean, deterministic evidence replay without changing physics.
    env_cfg.viewer.enable_reflections = False
    env_cfg.viewer.enable_shadows = False

    # Keep the replay command fixed. Curricula would otherwise overwrite it,
    # and interval pushes make a poor first visual baseline.
    _fix_commands(env_cfg, args.speed)
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = f"microduck-run-{args.speed:.2f}mps"
    expected_video = output_dir / f"{name_prefix}-step-0.mp4"
    if expected_video.exists():
        raise FileExistsError(expected_video)

    raw_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array",
    )
    fps = float(raw_env.metadata.get("render_fps", 30))
    recorder = VideoRecorder(
        raw_env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.frames,
        name_prefix=name_prefix,
    )
    env = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)

    speeds: list[float] = []
    episode_ends = 0
    try:
        runner_cls = load_runner_cls(args.task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

        observations = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.frames):
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
                speed = raw_env.scene["robot"].data.root_link_lin_vel_b[0, 0]
                speeds.append(float(speed.item()))
                episode_ends += int(dones[0].item())
    finally:
        env.close()

    if not expected_video.is_file():
        raise RuntimeError(f"recorder did not produce {expected_video}")

    manifest = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "commanded_forward_speed_mps": args.speed,
        "observed_forward_speed_mps": {
            "mean": sum(speeds) / len(speeds),
            "min": min(speeds),
            "max": max(speeds),
        },
        "episode_ends": episode_ends,
        "frames": args.frames,
        "fps": fps,
        "duration_seconds": args.frames / fps,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "device": args.device,
        "video": str(expected_video),
    }
    manifest_path = expected_video.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
