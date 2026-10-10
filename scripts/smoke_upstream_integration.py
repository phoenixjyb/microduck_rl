"""Bounded native100.100 source-integration smoke, not skill acceptance.

Run under a retained user service with a 600 s hard cap and 8 GiB RAM cap.
Uses the existing frozen venv; no downloads, expert policies, or dependency sync.
"""

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time

import torch

from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host

TASKS = ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Run-MotorAware-Flat-MicroDuck")


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(root), *argv), text=True).strip()
    assert git("rev-parse", "HEAD") == args.source
    assert not git("status", "--porcelain"), "exact clean integration source required"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert LOCK.is_file() and not LOCK.is_symlink()
    args.output.mkdir(parents=True, exist_ok=False)
    with LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        assert stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = shared_host()
        runtime = {n: metadata.version(n) for n in (
            "torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "better-actuator-models",
        )}
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.20, device=0)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        rows = []
        for index, task in enumerate(TASKS):
            shared_host()
            cfg, agent = load_env_cfg(task), load_rl_cfg(task)
            cfg.scene.num_envs = cfg.scene.terrain.num_envs = 64
            cfg.seed = agent.seed = 811 + index
            cfg.sim.nan_guard.enabled = True
            agent.logger = "tensorboard"
            agent.save_interval = 1
            log = args.output / f"task-{index}"
            log.mkdir()
            started = time.monotonic()
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            try:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                original_step = wrapped.step
                counters = {"steps": 0, "terminals": 0}
                def checked_step(actions):
                    if counters["steps"] % 24 == 0:
                        shared_host()
                    assert actions.shape == (64, 14) and finite_tree(actions)
                    obs, reward, done, extras = original_step(actions)
                    assert obs["actor"].shape == (64, 61)
                    assert finite_tree((obs, reward)), "nonfinite rollout"
                    counters["steps"] += 1
                    counters["terminals"] += int(done.sum())
                    return obs, reward, done, extras
                wrapped.step = checked_step
                runner = load_runner_cls(task)(wrapped, asdict(agent), str(log), "cuda:0")
                runner.learn(num_learning_iterations=5, init_at_random_ep_len=False)
                assert counters["steps"] == 5 * 24
                checkpoint = log / "model_4.pt"
                assert checkpoint.is_file()
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert finite_tree(payload), "nonfinite learned parameters or optimizer"
                assert payload["optimizer_state_dict"]["state"], "no optimizer state"
                with checkpoint.open("rb") as saved:
                    os.fsync(saved.fileno())
                rows.append({"task": task, **counters, "updates": 5, "worlds": 64,
                    "seconds": time.monotonic() - started,
                    "checkpoint": str(checkpoint),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "checkpoint_and_optimizer_finite": True})
            finally:
                env.close()
        result = {"source": args.source, "runtime": runtime, "initial_host": initial,
            "final_host": shared_host(), "tasks": rows, "status": "integration-smoke-passed",
            "policy_accepted": False, "physical_motion_authorized": False}
        with (args.output / "result.json").open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()
