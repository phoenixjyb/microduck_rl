"""Bounded opt-in measured route qualification of three predeclared v8 actors."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

import torch

from . import gait_path_qualification as route
from . import integrated_turn_foundation as base
from . import integrated_joint_limit_refinement as legacy
from . import official_native_reference as pinned
from . import v8_actor_learning as learning
from .v8_actor_checkpoint import FrozenV8Checkpoint

PROTOCOL = "v8-gait-aware-route-measurement-v1"
CAUTIOUS_SOURCE = "934bdff47a63d458f571189fe5db155fcd4fd8d0"
CAUTIOUS_RUN = "v8-cautious-lr-934bdff4-20261013"
CAUTIOUS_FINAL_SHA = "820c86e28b3d05d8bd292b63777b78cfa0b4657d69cc028ad17f8afe1ab91738"
ARMS = ("initial", "stock", "cautious")
CAP_SECONDS = 300


def checkpoint(arm):
    base.require(arm in ARMS, "predeclared route checkpoint arm")
    root = base.ROOT / "artifacts/training"
    stock = root / "v8-actor-foundation-722e7a6a-20261013"
    if arm == "initial":
        return stock / "benchmark/initial.pt", "80084814096f99a56a11f78d6e0526c99261b1d0eaae3a57da79517733b61163"
    if arm == "stock":
        return stock / "foundation/model_999.pt", "4631f81d36f7cc846cbed42efdedb9a16e2a40b2d1cadbcbc2834b43e016d7bb"
    result = learning.validate_training_receipt(root / CAUTIOUS_RUN, "foundation",
                                                CAUTIOUS_SOURCE, profile="cautious")
    base.require(result["checkpoint_sha256"] == CAUTIOUS_FINAL_SHA,
                 "complete sealed cautious final999 before route measurement")
    return root / CAUTIOUS_RUN / "foundation/model_999.pt", CAUTIOUS_FINAL_SHA


def evaluate(arm, output, health):
    path, digest = checkpoint(arm)
    actor = FrozenV8Checkpoint(path, digest)
    if arm == "initial": actor.require_initial_seed()
    raw, summaries = [], []
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            health()
            case = base.evaluate_case(path, seed, speed, yaw, health, diagnostics=True,
                yaw_trace=True, gait_trace=True, gait_path_trace=True, frozen_v8_checkpoint=actor)
            base.require(case["checkpoint_sha256"] == digest,
                         "route case bound to exact predeclared actor bytes")
            base.write_new(output / f"s{seed}-c{index}.json", case)
            summary = route.summarize_case(case, case["gait_path_trace"])
            base.write_new(output / f"s{seed}-c{index}-route.json", summary)
            raw.append(case); summaries.append(summary)
    return dict(arm=arm, checkpoint_sha256=digest, cases=summaries,
                route_gate=route.decision(summaries), legacy_gate=legacy.decision(raw),
                inference_backend=actor.provenance(), policy_acceptance=False,
                physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    base.require(re.fullmatch(r"[a-f0-9]{40}", args.source) is not None
                 and args.run_id == f"v8-gait-route-{args.source[:8]}-20261013",
                 "source-bound safe route measurement identity")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()
    base.require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"),
                 "clean exact tested route source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated route GPU")
    runtime = {name: metadata.version(name) for name in (*pinned.RUNTIME, "rsl-rl-lib")}
    base.require(all(runtime[k].split("+")[0] == v for k,v in pinned.RUNTIME.items())
                 and runtime["rsl-rl-lib"] == "5.0.1", "unchanged route runtime")
    base.require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing route GPU lease")
    with base.LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular route lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = base.shared_host()
        path,digest = checkpoint(args.arm)
        base.require(base.sha256(path) == digest, "exact route checkpoint before output")
        root = base.ROOT / "artifacts/evaluations" / args.run_id
        declaration = dict(source=args.source, protocol=PROTOCOL, arms=ARMS,
                           seeds=base.SEEDS, cases=base.CASES, cap_seconds=CAP_SECONDS)
        if args.arm == "initial":
            root.mkdir(parents=True, exist_ok=False)
            base.write_new(root / "campaign.json", declaration)
        else:
            base.require(json.loads((root / "campaign.json").read_text())
                         == json.loads(json.dumps(declaration)), "retained route campaign identity")
            predecessor = "initial" if args.arm == "stock" else "stock"
            prior = json.loads((root / predecessor / "result.json").read_text())
            base.require(prior["source"] == args.source and prior["protocol"] == PROTOCOL
                         and prior["arm"] == predecessor and len(prior["cases"]) == 12,
                         "complete previous route measurement before next arm")
        output = root / args.arm
        output.mkdir(exist_ok=False)
        base.write_new(output / "launch.json", dict(**declaration, arm=args.arm, runtime=runtime,
            host=host, checkpoint_sha256=digest, policy_acceptance=False, physical_motion_authorized=False))
        started = time.monotonic()
        def health():
            base.require(time.monotonic() - started < CAP_SECONDS - 15, "bounded route deadline")
            current = base.shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **current), allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            result = evaluate(args.arm, output, health)
            health()
            result.update(source=args.source, protocol=PROTOCOL, elapsed_s=time.monotonic()-started)
            base.write_new(output / "result.json", result)
        except Exception as error:
            base.write_new(output / "failure.json", dict(source=args.source, arm=args.arm,
                type=type(error).__name__, error=str(error)))
            raise


if __name__ == "__main__": main()
