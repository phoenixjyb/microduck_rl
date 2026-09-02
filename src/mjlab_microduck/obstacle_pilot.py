"""Bounded first training pilot for the single-obstacle curriculum."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path


TASK_ID = "Mjlab-Run-Obstacle-Flat-MicroDuck"
MAX_PILOT_ENVS = 2048
MAX_PILOT_ITERATIONS = 256
PILOT_CHECKPOINT_INTERVAL = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_intermediate_checkpoints(
    output_dir: Path, obstacle_warm_start: dict
) -> dict[str, str]:
    """Carry obstacle migration provenance into periodic runner checkpoints."""
    import torch

    retained: dict[str, str] = {}
    for checkpoint in sorted(output_dir.glob("model_[0-9]*.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        infos = payload.setdefault("infos", {})
        existing = infos.get("obstacle_warm_start")
        if existing is not None and existing != obstacle_warm_start:
            raise ValueError(
                f"conflicting obstacle warm-start metadata in {checkpoint.name}"
            )
        infos["obstacle_warm_start"] = dict(obstacle_warm_start)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        retained[checkpoint.name] = _sha256(checkpoint)
    return retained


def prepare_pilot_configs(num_envs: int, seed: int):
    """Load an isolated training config with bounded local-only logging."""
    if not 1 <= num_envs <= MAX_PILOT_ENVS:
        raise ValueError(f"num_envs must be in [1, {MAX_PILOT_ENVS}]")

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.terrain.num_envs = num_envs
    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.save_interval = PILOT_CHECKPOINT_INTERVAL
    return env_cfg, agent_cfg


def run_pilot(
    checkpoint: Path,
    output_dir: Path,
    *,
    num_envs: int = 1024,
    iterations: int = 128,
    seed: int = 42,
) -> Path:
    """Strict-load, train within fixed bounds, and retain a manifest and checkpoint."""
    if not 1 <= iterations <= MAX_PILOT_ITERATIONS:
        raise ValueError(f"iterations must be in [1, {MAX_PILOT_ITERATIONS}]")
    checkpoint = checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    env_cfg, agent_cfg = prepare_pilot_configs(num_envs, seed)
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    configure_torch_backends()
    torch.manual_seed(seed)
    print(
        f"obstacle_pilot task={TASK_ID} device={device} num_envs={num_envs} "
        f"iterations={iterations} seed={seed}"
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(wrapped, asdict(agent_cfg), str(output_dir), device)
        infos = runner.load(str(checkpoint), strict=True, map_location=device)
        metadata = infos.get("obstacle_warm_start", {})
        if metadata.get("actor_dims") != [61, 68]:
            raise ValueError("checkpoint lacks the expected obstacle warm-start metadata")

        start_iteration = runner.current_learning_iteration
        runner.learn(
            num_learning_iterations=iterations,
            init_at_random_ep_len=False,
        )
        intermediate_checkpoints = _stamp_intermediate_checkpoints(
            output_dir, metadata
        )
        retained = output_dir / "model_obstacle_pilot.pt"
        runner.save(str(retained), infos=infos)
        manifest = {
            "task_id": TASK_ID,
            "purpose": "bounded first-stage obstacle curriculum training pilot",
            "input_checkpoint": str(checkpoint),
            "input_checkpoint_sha256": _sha256(checkpoint),
            "retained_checkpoint": str(retained),
            "retained_checkpoint_sha256": _sha256(retained),
            "device": device,
            "num_envs": num_envs,
            "iterations": iterations,
            "checkpoint_interval": PILOT_CHECKPOINT_INTERVAL,
            "intermediate_checkpoints": intermediate_checkpoints,
            "seed": seed,
            "start_iteration": start_iteration,
            "end_iteration": runner.current_learning_iteration,
        }
        manifest_path = output_dir / "pilot_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"obstacle_pilot_retained={retained}")
        print(f"obstacle_pilot_manifest={manifest_path}")
        return retained
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pilot(
        args.checkpoint,
        args.output_dir,
        num_envs=args.num_envs,
        iterations=args.iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
