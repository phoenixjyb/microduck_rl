"""Bounded, stock mjlab training for the registered MicroDuck walking task."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import math
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
RESUME_UPDATE_CHOICES = (5, 5500, 4000)
RESUME_STEPS_PER_ENV = 24


def phase_settings(phase: str, worlds: int | None = None, *,
                   additional_updates: int | None = None) -> tuple[int, int]:
    if phase == "resume":
        if worlds != 1024:
            raise ValueError("resume is bound to --worlds 1024")
        if additional_updates not in RESUME_UPDATE_CHOICES:
            raise ValueError("resume requires an approved --additional-updates value")
        return 1024, additional_updates
    if additional_updates is not None:
        raise ValueError("--additional-updates is only valid for resume")
    if phase == "smoke":
        if worlds is not None:
            raise ValueError("--worlds is only valid for train")
        return 64, 5
    if phase in PROFILE_WORLDS:
        if worlds is not None:
            raise ValueError("--worlds is only valid for train")
        return PROFILE_WORLDS[phase], 20
    if phase == "thermal-1024":
        if worlds is not None:
            raise ValueError("--worlds is only valid for train")
        return 1024, 120
    if phase == "train":
        if worlds not in WORLD_CHOICES:
            raise ValueError("train requires --worlds from the approved choices")
        return worlds, 6000
    raise ValueError("unknown phase")


def build_config(phase: str, *, worlds: int | None = None, seed: int = 1019,
                 additional_updates: int | None = None,
                 resume_checkpoint_name: str | None = None):
    from mjlab.scripts.train import TrainConfig
    from mjlab_microduck.normalized_onnx_reference import validate_actor_contract

    envs, updates = phase_settings(phase, worlds, additional_updates=additional_updates)
    cfg = TrainConfig.from_task(TASK)
    cfg.env.scene.num_envs = envs
    cfg.env.scene.terrain.num_envs = envs
    cfg.env.seed = seed
    cfg.agent.seed = seed
    cfg.agent.max_iterations = updates
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.resume = phase == "resume"
    if cfg.agent.resume:
        if (resume_checkpoint_name is None
                or re.fullmatch(r"model_[0-9]+\.pt", resume_checkpoint_name) is None):
            raise ValueError("resume requires the verified parent checkpoint filename")
        cfg.agent.load_run = "resume_parent"
        cfg.agent.load_checkpoint = rf"^{re.escape(resume_checkpoint_name)}$"
    elif resume_checkpoint_name is not None:
        raise ValueError("resume checkpoint filename is only valid for resume")
    cfg.agent.save_interval = 100
    # The actor cfg checker is intentionally cfg-only; simulator dimensions are
    # still owned by the native trainer at runtime.
    validate_actor_contract(cfg.env, expected_actuator="BacklashEncoderBamActuatorCfg")
    return cfg


def validate_cooldown_seconds(cooldown_seconds: float) -> None:
    if (type(cooldown_seconds) not in (int, float)
            or not math.isfinite(cooldown_seconds) or not 0 <= cooldown_seconds <= 1):
        raise ValueError("--cooldown-seconds must be finite and between 0 and 1")


def validate_args(source: str, run_id: str, cap_seconds: int,
                  cooldown_seconds: float = 0.0) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("--source must be a lowercase 40-character commit SHA")
    if re.fullmatch(r"stock-walk-[a-z0-9-]+-s[0-9]+-[0-9]{8}", run_id) is None:
        raise ValueError("--run-id must be a dated stock-walk identifier")
    if type(cap_seconds) is not int or not 1 <= cap_seconds <= 12600:
        raise ValueError("--cap-seconds must be between 1 and 12600")
    validate_cooldown_seconds(cooldown_seconds)


def validate_phase_options(phase: str, *, worlds: int | None, additional_updates: int | None,
                           seed: int, cooldown_seconds: float, max_temperature_c: int,
                           parent_checkpoint: Path | None = None,
                           parent_source: str | None = None,
                           parent_sha256: str | None = None):
    envs, updates = phase_settings(phase, worlds, additional_updates=additional_updates)
    if type(max_temperature_c) is not int or max_temperature_c not in (75, 80):
        raise ValueError("--max-temperature-c must be 75 or 80")
    if phase == "resume":
        if seed != 1019:
            raise ValueError("resume is bound to parent seed 1019")
        if cooldown_seconds != 0.5:
            raise ValueError("resume requires the established 0.5-second update cooldown")
        if parent_checkpoint is None or parent_source is None or parent_sha256 is None:
            raise ValueError("resume requires exact parent checkpoint, source, and SHA256")
        if re.fullmatch(r"[0-9a-f]{40}", parent_source) is None:
            raise ValueError("--parent-source must be a lowercase 40-character commit SHA")
        if re.fullmatch(r"[0-9a-f]{64}", parent_sha256) is None:
            raise ValueError("--parent-sha256 must be a lowercase 64-character SHA256")
    else:
        if any(value is not None for value in (parent_checkpoint, parent_source, parent_sha256)):
            raise ValueError("parent checkpoint arguments are only valid for resume")
        if max_temperature_c != 75:
            raise ValueError("80 C is only approved for the resume phase")
    return envs, updates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume_parent(root: Path, checkpoint: Path, source: str,
                           expected_sha256: str, torch):
    training_root = (root / "artifacts" / "training").resolve()
    if not checkpoint.is_absolute():
        raise RuntimeError("resume parent checkpoint must be an absolute path")
    try:
        resolved = checkpoint.resolve(strict=True)
        relative = resolved.relative_to(training_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError("resume checkpoint must be inside artifacts/training") from exc
    if (len(relative.parts) != 3 or re.fullmatch(r"model_[0-9]+\.pt", relative.name) is None
            or any((training_root / Path(*relative.parts[:index])).is_symlink()
                   for index in range(1, len(relative.parts)) )
            or checkpoint.is_symlink()):
        raise RuntimeError("resume checkpoint must be a direct stock-run phase file")
    parent_dir = resolved.parent
    parent_run_id = relative.parts[0]
    launch_path = parent_dir / "launch.json"
    for path, label in ((training_root / parent_run_id, "parent run"),
                        (parent_dir, "parent phase"),
                        (launch_path, "parent launch"), (resolved, "parent checkpoint")):
        if path.is_symlink() or not path.exists():
            raise RuntimeError(f"resume {label} must exist and cannot be a symlink")
    for directory in (training_root, training_root / parent_run_id, parent_dir):
        info = directory.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError("resume parent directories must be owned directories")
    file_info = resolved.stat()
    if not stat.S_ISREG(file_info.st_mode) or file_info.st_uid != os.getuid():
        raise RuntimeError("resume checkpoint must be an owned regular file")
    launch_info = launch_path.stat()
    if not stat.S_ISREG(launch_info.st_mode) or launch_info.st_uid != os.getuid():
        raise RuntimeError("resume launch provenance must be an owned regular file")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if (launch.get("source") != source or launch.get("task") != TASK
            or launch.get("run_id") != parent_run_id or launch.get("seed") != 1019
            or launch.get("worlds") != 1024
            or re.fullmatch(r"stock-walk-[a-z0-9-]+-s1019-[0-9]{8}", parent_run_id) is None
            or launch.get("phase") not in ("train", "resume")):
        raise RuntimeError("resume parent launch provenance mismatch")
    digest = sha256_file(resolved)
    if digest != expected_sha256:
        raise RuntimeError("resume parent checkpoint SHA256 mismatch")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
    env_state = infos.get("env_state") or {}
    iteration = payload.get("iter")
    step_counter = env_state.get("common_step_counter")
    if launch.get("resume_parent"):
        previous_info = launch["resume_parent"]
        previous = previous_info.get("common_step_counter")
        launch_updates = launch.get("updates")
        expected_counter = (
            previous + launch_updates * RESUME_STEPS_PER_ENV
            if (type(previous) is int and previous > 0 and type(launch_updates) is int
                and launch_updates in RESUME_UPDATE_CHOICES
                and re.fullmatch(r"[0-9a-f]{40}", previous_info.get("source", ""))
                and re.fullmatch(r"[0-9a-f]{64}", previous_info.get("checkpoint_sha256", ""))
                and type(previous_info.get("checkpoint_iteration")) is int)
            else None
        )
    else:
        launch_updates = launch.get("updates")
        expected_counter = (iteration + 1) * RESUME_STEPS_PER_ENV if type(iteration) is int else None
    if (type(iteration) is not int or re.fullmatch(rf"model_{iteration}\.pt", relative.name) is None
            or type(step_counter) is not int or step_counter <= 0
            or step_counter % RESUME_STEPS_PER_ENV != 0
            or type(launch_updates) is not int or launch_updates <= 0
            or step_counter != expected_counter
            or not all(isinstance(payload.get(key), dict) and payload[key]
                       for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
            or not finite_tree(payload, torch)):
        raise RuntimeError("resume parent checkpoint iteration/state/finite validation failed")
    return {"run_id": parent_run_id, "phase": relative.parts[1], "source": source,
            "checkpoint": relative.name, "checkpoint_sha256": digest,
            "checkpoint_iteration": iteration, "common_step_counter": step_counter,
            "completed_updates": step_counter // RESUME_STEPS_PER_ENV,
            "cumulative_transitions": step_counter * 1024}


def ensure_resume_link(run_root: Path, parent_dir: Path) -> Path:
    link = run_root / "resume_parent"
    if link.is_symlink():
        info = link.lstat()
        if info.st_uid != os.getuid():
            raise RuntimeError("resume_parent symlink is not owned by this user")
        if os.readlink(link) != str(parent_dir):
            training_root = (run_root.parent).resolve()
            old_target = Path(os.readlink(link))
            try:
                old_target.resolve(strict=True).relative_to(training_root)
            except (OSError, ValueError) as exc:
                raise RuntimeError("resume_parent points outside artifacts/training") from exc
            raise RuntimeError("resume_parent already points to a different training parent")
    elif link.exists():
        raise RuntimeError("resume_parent exists but is not a symlink")
    else:
        link.symlink_to(parent_dir, target_is_directory=True)
    return link


def output_phase(phase: str, updates: int) -> str:
    if phase != "resume":
        return phase
    return {5: "resume-smoke", 5500: "resume-main", 4000: "resume-later"}[updates]


@contextmanager
def paced_update_logging(cooldown_seconds: float, *, logger_class=None,
                         torch_module=None, sleep_fn=time.sleep):
    validate_cooldown_seconds(cooldown_seconds)
    if cooldown_seconds == 0:
        yield
        return
    if logger_class is None:
        from rsl_rl.utils.logger import Logger as logger_class
    if torch_module is None:
        import torch as torch_module

    original = logger_class.log

    def paced_log(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        torch_module.cuda.synchronize()
        sleep_fn(cooldown_seconds)
        return result

    logger_class.log = paced_log
    try:
        yield
    finally:
        logger_class.log = original


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


def tensorboard_summary(log_dir: Path, expected_updates: int, *, expected_last_step: int | None = None,
                        timeout: float = 12.0,
                        accumulator_factory=None, sleep_fn=time.sleep):
    if accumulator_factory is None:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator_factory = lambda: EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    required = ("Perf/collection_time", "Perf/learning_time")
    expected_last_step = (expected_updates - 1 if expected_last_step is None
                          else expected_last_step)
    deadline = time.monotonic() + timeout
    while True:
        accumulator = accumulator_factory()
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", ())
        if all(tag in tags and len(accumulator.Scalars(tag)) >= expected_updates
               and accumulator.Scalars(tag)[-1].step == expected_last_step for tag in required):
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
                        choices=("smoke", *PROFILE_WORLDS, "thermal-1024", "train", "resume"))
    parser.add_argument("--worlds", type=int, choices=WORLD_CHOICES)
    parser.add_argument("--additional-updates", type=int, choices=RESUME_UPDATE_CHOICES)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-source")
    parser.add_argument("--parent-sha256")
    parser.add_argument("--seed", type=int, default=1019)
    parser.add_argument("--cap-seconds", type=int, required=True)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--max-temperature-c", type=int, choices=(75, 80), default=75)
    args = parser.parse_args(argv)
    validate_args(args.source, args.run_id, args.cap_seconds, args.cooldown_seconds)
    worlds, updates = validate_phase_options(
        args.phase, worlds=args.worlds, additional_updates=args.additional_updates,
        seed=args.seed, cooldown_seconds=args.cooldown_seconds,
        max_temperature_c=args.max_temperature_c, parent_checkpoint=args.parent_checkpoint,
        parent_source=args.parent_source, parent_sha256=args.parent_sha256,
    )
    if args.phase == "resume" and "MICRODUCK_WARM_START" in os.environ:
        raise RuntimeError("resume forbids MICRODUCK_WARM_START; use standard checkpoint resume")

    def git(*argv):
        return subprocess.check_output(("git", "-C", str(ROOT), *argv), text=True).strip()

    status = git("status", "--porcelain")
    if git("rev-parse", "HEAD") != args.source or status:
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
        resume_info = None
        parent_dir = None
        if args.phase == "resume":
            resume_info = validate_resume_parent(
                ROOT, args.parent_checkpoint, args.parent_source, args.parent_sha256, torch)
            parent_dir = args.parent_checkpoint.resolve(strict=True).parent
        run_root = ROOT / "artifacts" / "training" / args.run_id
        if run_root.is_symlink():
            raise RuntimeError("run-id directory cannot be a symlink")
        run_root.mkdir(parents=True, exist_ok=True)
        root_info = run_root.stat()
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid():
            raise RuntimeError("run-id path must be an owned directory")
        if args.phase == "resume":
            ensure_resume_link(run_root, parent_dir)
        output_name = output_phase(args.phase, updates)
        output = run_root / output_name
        output.mkdir(exist_ok=False)

        health_path = output / "health.jsonl"

        def health_check():
            sample = shared_host(max_temperature_c=args.max_temperature_c)
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
        cfg = build_config(args.phase, worlds=args.worlds, seed=args.seed,
                           additional_updates=args.additional_updates,
                           resume_checkpoint_name=(args.parent_checkpoint.name
                                                   if args.phase == "resume" else None))
        iteration_start = (resume_info["checkpoint_iteration"] if resume_info else 0)
        expected_last_iteration = iteration_start + updates - 1
        with (output / "launch.json").open("x", encoding="utf-8") as launch:
            json.dump({"source": args.source, "task": TASK, "run_id": args.run_id,
                       "phase": args.phase, "output_phase": output_name,
                       "worlds": worlds, "updates": updates,
                       "seed": args.seed, "cap_seconds": args.cap_seconds,
                       "cooldown_seconds": args.cooldown_seconds,
                       "max_temperature_c": args.max_temperature_c,
                       "additional_updates": updates if args.phase == "resume" else None,
                       "resume_parent": resume_info,
                       "runtime": runtime, "fresh_initialization": resume_info is None},
                      launch, indent=2, allow_nan=False)
            launch.write("\n")
        thread = threading.Thread(target=monitor, name="gpu-health", daemon=True)
        thread.start()
        try:
            from mjlab.scripts.train import run_train

            with paced_update_logging(args.cooldown_seconds, torch_module=torch):
                run_train(TASK, cfg, output)
            final = output / f"model_{expected_last_iteration}.pt"
            if not final.is_file():
                raise RuntimeError("stock trainer did not save the final expected iteration")
            payload = torch.load(final, map_location="cpu", weights_only=False)
            if payload.get("iter") != expected_last_iteration:
                raise RuntimeError("final checkpoint iteration does not match the requested run")
            if not finite_tree(payload, torch):
                raise RuntimeError("final checkpoint contains non-finite values")
            expected_counter = ((resume_info["common_step_counter"] if resume_info else 0)
                                + updates * cfg.agent.num_steps_per_env)
            actual_counter = (payload.get("infos") or {}).get("env_state", {}).get("common_step_counter")
            if actual_counter != expected_counter:
                raise RuntimeError("final checkpoint did not preserve the expected cumulative curriculum counter")
            from mjlab_microduck.integrated_motor_refinement import onnx_check

            onnx_path = output / f"{output_name}.onnx"
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
    prior_transitions = resume_info["cumulative_transitions"] if resume_info else 0
    local_transitions = worlds * cfg.agent.num_steps_per_env * updates
    result = {
        "source": args.source, "task": TASK, "run_id": args.run_id,
        "phase": args.phase, "output_phase": output_name, "worlds": worlds,
        "updates": updates, "local_updates": updates,
        "prior_completed_updates": resume_info["completed_updates"] if resume_info else 0,
        "cumulative_completed_updates": (resume_info["completed_updates"] if resume_info else 0) + updates,
        "iteration_start": iteration_start, "final_iteration": expected_last_iteration,
        "transitions": local_transitions, "local_transitions": local_transitions,
        "prior_cumulative_transitions": prior_transitions,
        "cumulative_transitions": prior_transitions + local_transitions,
        "resume_parent": resume_info,
        "resume_scope": ("standard model/optimizer checkpoint state restored; RNG and simulator state continuity not claimed"
                         if resume_info else "fresh initialization"),
        "elapsed_seconds": elapsed, "final_checkpoint": final.name,
        "checkpoint_sha256": digest, "checkpoint_finite": True, "onnx": onnx,
        "cooldown_seconds": args.cooldown_seconds,
        "Perf_wall_time_excludes_cooldown": True,
        "runtime": runtime,
        "tensorboard_scalars": tensorboard_summary(
            output, updates, expected_last_step=expected_last_iteration),
        "policy_acceptance": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, allow_nan=False)
        result_file.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
