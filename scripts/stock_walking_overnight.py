"""Finite sequential stock-walking work; no scheduling or GPU service mutations."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def admitted(cap_seconds, now, deadline, *, review_seconds=0):
    # Leave ten minutes for exit, durable receipts and scoped vLLM restoration.
    return cap_seconds > 0 and now + cap_seconds + review_seconds + 600 <= deadline


def continuation_cap(elapsed, updates):
    if not math.isfinite(elapsed) or elapsed <= 0 or updates <= 0:
        raise ValueError("positive finite measured learning exposure required")
    return math.ceil(180 + 1.5 * elapsed / updates * 4000)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-source", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args(argv)
    from train_stock_walking import output_phase
    if re.fullmatch(r"stock-walk-overnight-s1019-[0-9]{8}", args.run_id) is None:
        raise ValueError("predeclared seed1019 overnight run ID required")
    if re.fullmatch(r"[0-9a-f]{40}", args.source) is None:
        raise ValueError("exact source SHA required")
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
    if datetime.fromisoformat(args.deadline_utc).tzinfo is None:
        raise ValueError("deadline must be explicitly timezone-aware")
    output = ROOT / "artifacts" / "training" / args.run_id
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("non-symlink artifact path required")
    output.mkdir(parents=True, exist_ok=False)
    stages = []
    def call(script, words, cap):
        if not admitted(cap, time.time(), deadline):
            raise RuntimeError("insufficient predeclared deadline budget")
        started = time.monotonic()
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, words)]
        subprocess.run(command, cwd=ROOT, check=True, timeout=cap+15)
        stages.append({"script": script, "elapsed_seconds": time.monotonic()-started, "cap_seconds": cap})
        with (output / "stages.json").open("w") as stream:
            json.dump(stages, stream, indent=2, allow_nan=False)
            stream.write("\n")
    def train(run_id, parent, source, digest, updates, cap):
        words = ["--source", args.source, "--run-id", run_id, "--phase", "resume",
                 "--worlds", "1024", "--seed", "1019", "--cap-seconds", cap,
                 "--cooldown-seconds", ".5", "--max-temperature-c", "80",
                 "--parent-checkpoint", parent, "--parent-source", source,
                 "--parent-sha256", digest, "--additional-updates", updates]
        call("train_stock_walking.py", words, cap)
        return json.loads((ROOT / "artifacts" / "training" / run_id / output_phase("resume", updates) / "result.json").read_text())
    def review(checkpoint, digest, name):
        call("review_stock_walking.py", [checkpoint, "--source", args.source,
             "--checkpoint-sha256", digest, "--output-dir", ROOT / "artifacts" / "evaluations" / name,
             "--cap-seconds", "1800", "--max-temperature-c", "80"], 1800)
    if not admitted(360+11000, time.time(), deadline, review_seconds=3600):
        raise RuntimeError("foundation + smoke + two reviews do not fit before deadline")
    # Same plant/seed/checkpoint, five additional standard PPO updates, not a warm start.
    train(args.run_id.replace("-overnight-", "-resume-smoke-"), args.parent_checkpoint,
          args.parent_source, args.parent_sha256, 5, 360)
    review(args.parent_checkpoint, args.parent_sha256, args.run_id+"-partial-review")
    run = args.run_id.replace("-overnight-", "-resume-foundation-")
    result = train(run, args.parent_checkpoint, args.parent_source, args.parent_sha256, 5500, 11000)
    checkpoint = ROOT / "artifacts" / "training" / run / result["output_phase"] / result["final_checkpoint"]
    digest = result["checkpoint_sha256"]
    review(checkpoint, digest, run+"-review")
    stop_reason = "predeclared-two-extension-limit"
    # More exposure in the SAME stock recipe, no reward/seed search or skill promotion.
    for number in (1, 2):
        cap = continuation_cap(result["elapsed_seconds"], result["updates"])
        if cap > 12600 or not admitted(cap, time.time(), deadline, review_seconds=1800):
            stop_reason = "next-4000-update-round-does-not-fit"
            break
        run = args.run_id.replace("-overnight-", f"-extension{number}-")
        result = train(run, checkpoint, args.source, digest, 4000, cap)
        checkpoint = ROOT / "artifacts" / "training" / run / result["output_phase"] / result["final_checkpoint"]
        digest = result["checkpoint_sha256"]
        review(checkpoint, digest, run+"-review")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("final durable checkpoint hash changed")
    with (output / "result.json").open("x") as stream:
        json.dump({"source": args.source, "deadline_utc": args.deadline_utc,
                   "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                   "stop_reason": stop_reason, "stages": stages,
                   "policy_acceptance": False, "visual_inspection_pending": True}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
