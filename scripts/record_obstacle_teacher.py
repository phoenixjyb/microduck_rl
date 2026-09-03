"""Record one HC1 duck-and-obstacle replay through the frozen gait actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig
from rsl_rl.runners import OnPolicyRunner

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
from mjlab_microduck.hierarchical_obstacle_rollout import (
    BASE_TASK_ID,
    _route_state,
    load_learned_supervisor,
    prepare_rollout_configs,
    recording_stem,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--obstacle-forward", type=float, required=True)
    parser.add_argument("--obstacle-lateral", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--supervisor-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    supervisor_checkpoint = (
        args.supervisor_checkpoint.expanduser().resolve(strict=True)
        if args.supervisor_checkpoint is not None
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    if not 0.0 < args.speed <= 0.8:
        raise ValueError("speed must be in (0, 0.8]")
    if args.obstacle_forward <= 0.0:
        raise ValueError("obstacle-forward must be positive")
    if min(args.frames, args.width, args.height) <= 0:
        raise ValueError("frames and dimensions must be positive")

    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg, agent_cfg = prepare_rollout_configs(
        1,
        args.speed,
        args.obstacle_forward,
        args.obstacle_lateral,
    )
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    # A replay should show post-obstacle recovery rather than resetting as soon
    # as the numeric success boundary is crossed.
    env_cfg.terminations.pop("obstacle_passed")
    env_cfg.terminations.pop("obstacle_attempt_timeout")
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.distance = 2.0
    env_cfg.viewer.azimuth = 125.0
    env_cfg.viewer.elevation = -22.0
    env_cfg.viewer.lookat = (0.35, 0.0, 0.10)
    env_cfg.viewer.enable_reflections = False
    env_cfg.viewer.enable_shadows = False

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = recording_stem(
        args.speed,
        args.obstacle_forward,
        args.obstacle_lateral,
        controller_stage="hc2" if supervisor_checkpoint is not None else "hc1",
    )
    expected_video = output_dir / f"{stem}-step-0.mp4"
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
        name_prefix=stem,
    )
    env = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)

    collision_events = 0
    fall_events = 0
    nan_events = 0
    nonfinite_steps = 0
    passed_once = False
    first_pass_time_s: float | None = None
    phase_speed_sum = torch.zeros(3, device=args.device)
    phase_samples = torch.zeros(3, dtype=torch.long, device=args.device)
    state = make_teacher_state(1, device=args.device, nominal_speed_mps=args.speed)
    nominal = torch.tensor([args.speed], device=args.device)
    command = raw_env.command_manager.get_command("twist")
    command[:, 0] = nominal
    command[:, 1:] = 0.0

    try:
        runner_cls = load_runner_cls(BASE_TASK_ID) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        learned_supervisor = (
            load_learned_supervisor(
                supervisor_checkpoint, checkpoint, args.device
            )
            if supervisor_checkpoint is not None
            else None
        )
        observations = env.get_observations()

        with torch.inference_mode():
            for step in range(args.frames):
                route_lateral, route_heading, route_speed = _route_state(raw_env)
                if step % 5 == 0:
                    obstacle_observation = microduck_mdp.obstacle_geometry_observation(
                        raw_env,
                        asset_name="obstacle",
                        width_m=0.20,
                        height_m=0.10,
                        horizontal_fov_rad=2.0 * math.pi,
                        max_range_m=2.0,
                    )
                    previous_command = state.previous_command.clone()
                    if learned_supervisor is None:
                        supervisor_command = teacher_command(
                            obstacle_observation,
                            nominal,
                            route_lateral,
                            route_heading,
                            state,
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
                            previous_command=previous_command,
                        )
                        normalized_command = learned_supervisor(
                            learned_observation
                        )
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
                    command[:, 0] = supervisor_command[:, 0]
                    command[:, 1] = 0.0
                    command[:, 2] = supervisor_command[:, 1]

                phase = int(state.phase[0])
                phase_speed_sum[phase] += torch.nan_to_num(route_speed[0], nan=0.0)
                phase_samples[phase] += 1
                actions = policy(observations)
                observations, rewards, dones, _ = env.step(actions)
                collision = raw_env.termination_manager.get_term(
                    "obstacle_collision"
                )
                fell = raw_env.termination_manager.get_term("fell_over")
                nan_state = raw_env.termination_manager.get_term("nan_state")
                collision_events += int(collision.sum())
                fall_events += int(fell.sum())
                nan_events += int(nan_state.sum())
                finite = torch.isfinite(actions).all() & torch.isfinite(rewards).all()
                finite &= all(
                    torch.isfinite(value).all() for value in observations.values()
                )
                nonfinite_steps += int(not bool(finite))
                rejoined = microduck_mdp.obstacle_route_rejoined(
                    raw_env,
                    asset_name="obstacle",
                    robot_radius_m=0.12,
                    obstacle_radius_m=0.10,
                    return_tolerance_m=0.15,
                )
                if bool(rejoined[0]) and not passed_once:
                    passed_once = True
                    first_pass_time_s = (step + 1) * raw_env.step_dt
                reset_teacher_state(
                    state, dones.bool(), nominal_speed_mps=args.speed
                )
    finally:
        env.close()

    if not expected_video.is_file():
        raise RuntimeError(f"recorder did not produce {expected_video}")

    phase_speeds = {}
    for phase in ObstaclePhase:
        count = int(phase_samples[int(phase)])
        phase_speeds[phase.name.lower()] = (
            float(phase_speed_sum[int(phase)] / count) if count else None
        )
    manifest = {
        "stage": (
            "HC1-deterministic-teacher-replay"
            if supervisor_checkpoint is None
            else "HC2-behavioral-cloning-replay"
        ),
        "evidence_status": (
            "representative-success" if passed_once and collision_events == 0 else "diagnostic"
        ),
        "base_task_id": BASE_TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "teacher_config": asdict(ObstacleTeacherCfg()),
        "commanded_nominal_speed_mps": args.speed,
        "obstacle_forward_m": args.obstacle_forward,
        "obstacle_lateral_m": args.obstacle_lateral,
        "phase_route_speed_mean_mps": phase_speeds,
        "passed_once": passed_once,
        "first_pass_time_s": first_pass_time_s,
        "collision_events": collision_events,
        "fall_events": fall_events,
        "nan_termination_events": nan_events,
        "nonfinite_steps": nonfinite_steps,
        "frames": args.frames,
        "fps": fps,
        "duration_seconds": args.frames / fps,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "device": args.device,
        "video": str(expected_video),
        "perception": "exact structured geometry; no raw camera perception",
        "physical_motion_authorized": False,
    }
    if supervisor_checkpoint is not None:
        manifest["supervisor_checkpoint"] = str(supervisor_checkpoint)
        manifest["supervisor_checkpoint_sha256"] = _sha256(
            supervisor_checkpoint
        )
    manifest_path = expected_video.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
