"""Bounded, stock mjlab training for the registered MicroDuck walking task."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import threading
import time

TASK = "Mjlab-Velocity-Flat-Backlash-MicroDuck"
ROOT = Path(__file__).resolve().parents[1]
PROFILE_WORLDS = {"profile-256": 256, "profile-1024": 1024,
                  "profile-2048": 2048, "profile-4096": 4096}
WORLD_CHOICES = (64, 256, 1024, 2048, 4096)


def phase_settings(phase: str, worlds: int | None = None) -> tuple[int, int]:
    if phase == "smoke":
        if worlds is not None:
            raise ValueError("--worlds is only valid for train")
        return 64, 5
    if phase in PROFILE_WORLDS:
        if worlds is not None:
            raise ValueError("--worlds is only valid for train")
        return PROFILE_WORLDS[phase], 20
    if phase == "train":
        if worlds not in WORLD_CHOICES:
            raise ValueError("train requires --worlds from the approved choices")
        return worlds, 6000
    raise ValueError("unknown phase")


def build_config(phase: str, *, worlds: int | None = None, seed: int = 1019):
    from mjlab.scripts.train import TrainConfig
    from mjlab_microduck.normalized_onnx_reference import validate_actor_contract

    envs, updates = phase_settings(phase, worlds)
    cfg = TrainConfig.from_task(TASK)
    cfg.env.scene.num_envs = envs
    cfg.env.scene.terrain.num_envs = envs
    cfg.env.seed = seed
    cfg.agent.seed = seed
    cfg.agent.max_iterations = updates
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.resume = False
    cfg.agent.save_interval = 250
    # The actor cfg checker is intentionally cfg-only; simulator dimensions are
    # still owned by the native trainer at runtime.
    validate_actor_contract(cfg.env, expected_actuator="BacklashEncoderBamActuatorCfg")
    return cfg


def validate_args(source: str, run_id: str, cap_seconds: int) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("--source must be a lowercase 40-character commit SHA")
    if re.fullmatch(r"stock-walk-[a-z0-9-]+-s[0-9]+-[0-9]{8}", run_id) is None:
        raise ValueError("--run-id must be a dated stock-walk identifier")
    if type(cap_seconds) is not int or not 1 <= cap_seconds <= 10800:
        raise ValueError("--cap-seconds must be between 1 and 10800")


def finite_tree(value, torch) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item, torch) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item, torch) for item in value)
    if isinstance(value, float):
        return value == value and abs(value) != float("inf")
    return True


def tensorboard_summary(log_dir: Path, expected_updates: int, *, timeout: float = 12.0,
                        accumulator_factory=None, sleep_fn=time.sleep):
    if accumulator_factory is None:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator_factory = lambda: EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    required = ("Perf/collection_time", "Perf/learning_time")
    deadline = time.monotonic() + timeout
    while True:
        accumulator = accumulator_factory()
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", ())
        if all(tag in tags and len(accumulator.Scalars(tag)) >= expected_updates
               and accumulator.Scalars(tag)[-1].step == expected_updates - 1 for tag in required):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("TensorBoard performance scalars did not flush the complete run")
        sleep_fn(min(0.25, deadline - time.monotonic()))

    summary = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        values = [float(event.value) for event in events]
        if not values or any(not (value == value and abs(value) != float("inf")) for value in values):
            raise RuntimeError(f"missing or non-finite TensorBoard scalars: {tag}")
        summary[tag] = {"count": len(values), "first": values[0], "last": values[-1],
                        "minimum": min(values), "maximum": max(values),
                        "last_step": events[-1].step}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True,
                        choices=("smoke", *PROFILE_WORLDS, "train"))
    parser.add_argument("--worlds", type=int, choices=WORLD_CHOICES)
    parser.add_argument("--seed", type=int, default=1019)
    parser.add_argument("--cap-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    validate_args(args.source, args.run_id, args.cap_seconds)
    worlds, updates = phase_settings(args.phase, args.worlds)

    def git(*argv):
        return subprocess.check_output(("git", "-C", str(ROOT), *argv), text=True).strip()

    if git("rev-parse", "HEAD") != args.source or git("status", "--porcelain"):
        raise RuntimeError("training requires the exact clean --source checkout")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if os.environ.get("MICRODUCK_RETAINED_DINO_PID") != "2029":
        raise RuntimeError("explicit MICRODUCK_RETAINED_DINO_PID=2029 is required")

    from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host

    if not LOCK.is_file() or LOCK.is_symlink():
        raise RuntimeError("existing regular shared GPU lock is required")
    import torch

    started = time.monotonic()
    stop_monitor = threading.Event()
    lease = LOCK.open("r+")
    try:
        info = os.fstat(lease.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError("GPU lock must be an owned regular file")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_root = ROOT / "artifacts" / "training" / args.run_id
        if run_root.is_symlink():
            raise RuntimeError("run-id directory cannot be a symlink")
        run_root.mkdir(parents=True, exist_ok=True)
        root_info = run_root.stat()
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid():
            raise RuntimeError("run-id path must be an owned directory")
        output = run_root / args.phase
        output.mkdir(exist_ok=False)

        health_path = output / "health.jsonl"

        def health_check():
            sample = shared_host()
            row = {"elapsed_seconds": time.monotonic() - started, **sample}
            with health_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())

        def monitor():
            while True:
                remaining = args.cap_seconds - (time.monotonic() - started)
                if stop_monitor.wait(min(5, max(0, remaining))):
                    return
                try:
                    if time.monotonic() - started >= args.cap_seconds:
                        raise RuntimeError("hard wall-clock cap reached")
                    health_check()
                except BaseException as exc:
                    try:
                        with (output / "monitor-error.txt").open("x", encoding="utf-8") as err:
                            err.write(f"{type(exc).__name__}: {exc}\n")
                    finally:
                        os.kill(os.getpid(), signal.SIGTERM)
                    return

        health_check()
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.20, device=0)
        from mjlab_microduck.official_native_reference import RUNTIME

        runtime = {name: metadata.version(name) for name in (*RUNTIME, "rsl-rl-lib")}
        if any(runtime[name].split("+")[0] != version for name, version in RUNTIME.items()):
            raise RuntimeError("pinned native training runtime mismatch")
        if runtime["rsl-rl-lib"] != "5.0.1":
            raise RuntimeError("pinned rsl-rl-lib runtime mismatch")
        cfg = build_config(args.phase, worlds=args.worlds, seed=args.seed)
        with (output / "launch.json").open("x", encoding="utf-8") as launch:
            json.dump({"source": args.source, "task": TASK, "run_id": args.run_id,
                       "phase": args.phase, "worlds": worlds, "updates": updates,
                       "seed": args.seed, "cap_seconds": args.cap_seconds,
                       "runtime": runtime, "fresh_initialization": True},
                      launch, indent=2, allow_nan=False)
            launch.write("\n")
        thread = threading.Thread(target=monitor, name="gpu-health", daemon=True)
        thread.start()
        try:
            from mjlab.scripts.train import run_train

            run_train(TASK, cfg, output)
            final = output / f"model_{updates - 1}.pt"
            if not final.is_file():
                raise RuntimeError("stock trainer did not save the final expected iteration")
            payload = torch.load(final, map_location="cpu", weights_only=False)
            if payload.get("iter") != updates - 1:
                raise RuntimeError("final checkpoint iteration does not match the requested run")
            if not finite_tree(payload, torch):
                raise RuntimeError("final checkpoint contains non-finite values")
            from mjlab_microduck.integrated_motor_refinement import onnx_check

            onnx_path = output / f"{args.phase}.onnx"
            if not onnx_path.is_file():
                raise RuntimeError("stock runner did not save the expected ONNX export")
            onnx = onnx_check(onnx_path)
        finally:
            stop_monitor.set()
            thread.join()
        health_check()
    finally:
        lease.close()

    hasher = hashlib.sha256()
    with final.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    elapsed = time.monotonic() - started
    result = {
        "source": args.source, "task": TASK, "run_id": args.run_id,
        "phase": args.phase, "worlds": worlds, "updates": updates,
        "transitions": worlds * cfg.agent.num_steps_per_env * updates,
        "elapsed_seconds": elapsed, "final_checkpoint": final.name,
        "checkpoint_sha256": digest, "checkpoint_finite": True, "onnx": onnx,
        "runtime": runtime, "tensorboard_scalars": tensorboard_summary(output, updates),
        "policy_acceptance": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, allow_nan=False)
        result_file.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
