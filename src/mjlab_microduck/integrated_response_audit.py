"""Bounded replay diagnosis of the completed tracking comparison; no learning."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata as metadata
import json
import math
import os
import re
import stat
import subprocess

import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host

PROTOCOL = "integrated-tracking-response-audit-v1"
INPUT = base.ROOT / "artifacts/training/tracking-refinement-6dbc0613-20261010"
INPUT_SOURCE = "6dbc0613c7fd55f081bf4cebea292b9acec8a61e"
EVALUATION_SHA = "e803f630a02aa6286b4b334334ea6696ada7c458849726f2237af645a29c6af3"
CHECKPOINTS = {"control": "b7a6601c479ef6805d4fc68236deff1805e6997bea4286e5b0f69b7b3255deec",
               "tracking": "5d03c2241d5291ded5786ed87ec85cd70f8262fcd47954837114e1102dfe6fc9"}


def equal_replay(left, right):
    """Compare retained acceptance fields, never silently replace old evidence."""
    if isinstance(right, dict):
        return isinstance(left, dict) and left.keys() == right.keys() and all(
            equal_replay(left[k], value) for k, value in right.items())
    if isinstance(right, list):
        return isinstance(left, list) and len(left) == len(right) and all(
            equal_replay(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(right, float):
        return type(left) in (int, float) and math.isfinite(left) and math.isclose(
            left, right, rel_tol=0., abs_tol=1e-6)
    return type(left) is type(right) and left == right


def replay_differences(left, right, path=""):
    if equal_replay(left, right):
        return []
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return [d for key in right for d in replay_differences(left[key], right[key], f"{path}.{key}".lstrip("."))]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [d for i, (a, b) in enumerate(zip(left, right, strict=True))
                for d in replay_differences(a, b, f"{path}[{i}]")]
    return [dict(field=path, retained=right, observed=left)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-diagnosis", choices=("default", "diagnostic"),
                        help="capture seed839's eight cases without claiming an exact full replay")
    args = parser.parse_args()
    base.require(re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id), "safe unique audit id")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()
    base.require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"), "exact clean audit source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal GPU0 selection")
    base.require(LOCK.is_file() and not LOCK.is_symlink(), "existing regular GPU lease")
    with LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = shared_host()
        evaluation = INPUT / "evaluate/result.json"
        base.require(base.sha256(evaluation) == EVALUATION_SHA, "exact retained evaluation")
        prior = json.loads(evaluation.read_text())
        runtime = {n: metadata.version(n) for n in ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "better-actuator-models")}
        for arm, digest in CHECKPOINTS.items():
            launch = json.loads((INPUT / arm / "launch.json").read_text())
            trained = json.loads((INPUT / arm / "result.json").read_text())
            base.require(launch["runtime"] == runtime and launch["source"] == INPUT_SOURCE
                and trained["source"] == INPUT_SOURCE and trained["updates"] == 1000
                and trained["checkpoint_sha256"] == prior["checkpoint_sha256"][arm] == digest
                and base.sha256(INPUT / arm / "model_999.pt") == digest, "retained parent and identical runtime")
            base.require(base.decision(prior["cases"][arm]) == prior["decisions"][arm], "original deterministic gate")
        output = base.ROOT / "artifacts/evaluations" / args.run_id
        output.mkdir(parents=True, exist_ok=False)
        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source, host=host,
            runtime=runtime, input_source=INPUT_SOURCE, evaluation_sha256=EVALUATION_SHA,
            checkpoint_sha256=CHECKPOINTS, case_diagnosis=args.case_diagnosis,
            policy_acceptance=False, physical_motion_authorized=False))
        def health():
            sample = shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(sample, allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            cases, comparisons = {}, {}
            for arm in CHECKPOINTS:
                rows = []
                declared = prior["cases"][arm][:4] if args.case_diagnosis else prior["cases"][arm]
                for index, old in enumerate(declared):
                    row = base.evaluate_case(INPUT / arm / "model_999.pt", old["seed"], old["speed"], old["yaw"], health,
                                             diagnostics=args.case_diagnosis != "default")
                    acceptance = {k: v for k, v in row.items() if k != "response_diagnostics"}
                    # Preserve numerical failure evidence BEFORE the strict check.
                    base.write_new(output / f"{arm}-s{old['seed']}-c{index%4}.json", row)
                    diff = replay_differences(acceptance, old)
                    comparisons[f"{arm}-{index}"] = diff
                    base.write_new(output / f"{arm}-s{old['seed']}-c{index%4}-comparison.json", dict(differences=diff))
                    if not args.case_diagnosis:
                        base.require(not diff, "diagnostic replay differs from original acceptance evidence")
                    rows.append(row)
                if not args.case_diagnosis:
                    base.require(base.decision(rows) == prior["decisions"][arm], "unchanged replay decision")
                cases[arm] = rows
            base.write_new(output / "result.json", dict(protocol=PROTOCOL, source=args.source,
                evaluation_sha256=EVALUATION_SHA, checkpoint_sha256=CHECKPOINTS, cases=cases,
                original_decisions=prior["decisions"], replay_verified=args.case_diagnosis is None,
                case_diagnosis=args.case_diagnosis, comparisons=comparisons,
                decision="diagnostic-only-not-admission", policy_acceptance=False, physical_motion_authorized=False))
            health()
        except Exception as exc:
            base.write_new(output / "failure.json", dict(type=type(exc).__name__, error=str(exc)))
            raise


if __name__ == "__main__":
    main()
