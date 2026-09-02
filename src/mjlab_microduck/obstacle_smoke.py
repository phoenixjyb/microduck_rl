"""Bounded end-to-end runner smoke for the obstacle warm-start checkpoint."""

import argparse
import os
from dataclasses import asdict
from pathlib import Path


TASK_ID = "Mjlab-Run-Obstacle-Flat-MicroDuck"
MAX_SMOKE_ENVS = 256
MAX_SMOKE_ITERATIONS = 2


def prepare_smoke_configs(num_envs: int):
    """Load isolated configs with external logging/model upload disabled."""
    if not 1 <= num_envs <= MAX_SMOKE_ENVS:
        raise ValueError(f"num_envs must be in [1, {MAX_SMOKE_ENVS}]")

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.terrain.num_envs = num_envs
    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    return env_cfg, agent_cfg


def run_smoke(
    checkpoint: Path,
    output_dir: Path,
    *,
    num_envs: int = 64,
    iterations: int = 1,
) -> Path:
    """Strict-load, learn for at most two iterations, and retain one checkpoint."""
    if not 1 <= iterations <= MAX_SMOKE_ITERATIONS:
        raise ValueError(f"iterations must be in [1, {MAX_SMOKE_ITERATIONS}]")
    checkpoint = checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    env_cfg, agent_cfg = prepare_smoke_configs(num_envs)
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    configure_torch_backends()
    torch.manual_seed(agent_cfg.seed)
    print(
        f"obstacle_smoke task={TASK_ID} device={device} "
        f"num_envs={num_envs} iterations={iterations}"
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(
            wrapped,
            asdict(agent_cfg),
            str(output_dir),
            device,
        )
        infos = runner.load(str(checkpoint), strict=True, map_location=device)
        metadata = infos.get("obstacle_warm_start", {})
        if metadata.get("actor_dims") != [61, 68]:
            raise ValueError("checkpoint lacks the expected obstacle warm-start metadata")
        start_iteration = runner.current_learning_iteration
        runner.learn(
            num_learning_iterations=iterations,
            init_at_random_ep_len=False,
        )
        retained = output_dir / "model_obstacle_smoke.pt"
        runner.save(str(retained), infos=infos)
        print(
            f"obstacle_smoke_passed start_iteration={start_iteration} "
            f"end_iteration={runner.current_learning_iteration} retained={retained}"
        )
        return retained
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    run_smoke(
        args.checkpoint,
        args.output_dir,
        num_envs=args.num_envs,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
