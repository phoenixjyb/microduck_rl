"""Guarded, bounded task-native walking evaluation and four simulation clips."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import stat
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
TASK = "Mjlab-Velocity-Flat-Backlash-MicroDuck"
CASES = (("idle", 0.0, 0.0), ("straight", 0.3, 0.0),
         ("left", 0.2, 0.2), ("right", 0.2, -0.2))
COMMAND_RESPONSE_CASES = (*CASES,
    ("slow-straight", 0.2, 0.0), ("fast-straight", 0.4, 0.0),
    ("spin-left", 0.0, 0.5), ("spin-right", 0.0, -0.5),
    ("moving-left", 0.3, 0.5), ("moving-right", 0.3, -0.5),
    ("strong-left", 0.3, 0.8), ("strong-right", 0.3, -0.8))


def review_cases(protocol):
    if protocol == "gentle":
        return CASES
    if protocol == "command-response":
        return COMMAND_RESPONSE_CASES
    raise ValueError("unknown stock review protocol")


def script_args(kind, checkpoint, output, speed, yaw):
    args = [str(checkpoint), "--task", TASK, "--output-dir", str(output), "--device", "cuda:0"]
    if kind == "evaluate":
        return args + ["--speeds", str(speed), f"--yaw-rates={yaw}", "--seeds", "42,211,223",
                       "--num-envs", "8", "--steps", "500", "--warmup-steps", "50"]
    if kind == "record":
        return args + ["--speed", str(speed), f"--yaw-rate={yaw}", "--frames", "500", "--seed", "42"]
    raise ValueError("unknown review operation")


@contextmanager
def script_argv(path, args):
    original = sys.argv
    sys.argv = [str(path), *args]
    try:
        yield
    finally:
        sys.argv = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=int, default=1800)
    parser.add_argument("--max-temperature-c", type=int, choices=(75, 80), default=75)
    parser.add_argument("--protocol", choices=("gentle", "command-response"), default="gentle")
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.cap_seconds <= 1800:
        parser.error("review cap must be 1..1800 seconds")
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to(ROOT / "artifacts" / "training"):
        raise ValueError("checkpoint must be retained in this checkout's training artifacts")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    def git(*words):
        return subprocess.check_output(["git", "-C", str(ROOT), *words], text=True).strip()
    if git("rev-parse", "HEAD") != args.source or git("status", "--porcelain"):
        raise ValueError("review requires exact clean source")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("MICRODUCK_RETAINED_DINO_PID") != "2029":
        raise ValueError("exact native CUDA and retained DINO binding required")
    output = args.output_dir.absolute()
    if not output.is_relative_to(ROOT / "artifacts" / "evaluations") or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("owned, non-symlink evaluation artifact path required")
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host
    if LOCK.is_symlink():
        raise ValueError("regular existing GPU lease required")
    stopped = threading.Event()
    started = time.monotonic()
    with LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("owned regular GPU lease required")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def check():
            sample = shared_host(max_temperature_c=args.max_temperature_c)
            with (output / "health.jsonl").open("a") as stream:
                stream.write(json.dumps({"elapsed_seconds": time.monotonic()-started, **sample}, allow_nan=False)+"\n")
                stream.flush(); os.fsync(stream.fileno())
        check()
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.20, device=0)
        def watch():
            while not stopped.wait(5):
                try:
                    if time.monotonic()-started >= args.cap_seconds:
                        raise RuntimeError("review wall-clock cap reached")
                    check()
                except Exception as error:
                    with (output / "monitor-error.txt").open("x") as stream:
                        stream.write(f"{type(error).__name__}: {error}\n")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        evidence = []
        try:
            for kind in (("evaluate",) if args.evaluation_only else ("evaluate", "record")):
                script = ROOT / "scripts" / ("evaluate_checkpoint.py" if kind == "evaluate" else "record_checkpoint.py")
                for name, speed, yaw in review_cases(args.protocol):
                    destination = output / kind / name
                    with script_argv(script, script_args(kind, checkpoint, destination, speed, yaw)):
                        runpy.run_path(str(script), run_name="__main__")
                    if kind == "evaluate":
                        receipt = json.loads((destination / "checkpoint-evaluation.json").read_text())
                        if receipt["checkpoint_sha256"] != args.checkpoint_sha256 or len(receipt["cases"]) != 3:
                            raise ValueError("evaluation provenance/case count mismatch")
                        if any(c["nonfinite_steps"] or c["nan_termination_events"] for c in receipt["cases"]):
                            raise ValueError("non-finite evaluation; do not proceed to videos")
                    else:
                        videos = list(destination.glob("*.mp4"))
                        if len(videos) != 1 or videos[0].stat().st_size < 1000:
                            raise ValueError("missing/nonunique bounded video")
                        video = videos[0]
                        evidence.append({"case": name, "video": str(video.relative_to(output)),
                                         "sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
                    gc.collect(); torch.cuda.empty_cache(); check()
                    time.sleep(1)
        finally:
            stopped.set(); thread.join()
    with (output / "result.json").open("x") as stream:
        json.dump({"source": args.source, "checkpoint_sha256": args.checkpoint_sha256,
                   "task": TASK, "protocol": args.protocol,
                   "evaluation_cases": 3 * len(review_cases(args.protocol)), "videos": evidence,
                   "elapsed_seconds": time.monotonic()-started, "temperature_cutoff_c": args.max_temperature_c,
                   "visual_inspection_pending": not args.evaluation_only,
                   "video_not_requested": args.evaluation_only,
                   "policy_acceptance": False}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
