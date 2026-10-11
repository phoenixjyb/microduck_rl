"""Eight-case diagnostic replay of both rejected turning-range finals; no learning."""

import argparse
import fcntl
import importlib.metadata as metadata
import json
import os
import re
import stat
import subprocess

import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import integrated_turn_range_curriculum as recipe
from mjlab_microduck import integrated_response_audit as replay
from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host

PROTOCOL = "integrated-angular-reward-audit-v1"
INPUT = base.ROOT / "artifacts/training/turn-range-2b75ab02-20261011"
INPUT_SOURCE = "2b75ab0272e6728af6b4124b8e8942b614d57cb5"
EVALUATION_SHA = "71bcfbe46ad8919123222067e2a26f9737967e187950838811efff8b08d5bf94"
CHECKPOINTS = {"control": "11c20ccc4d0682ab1403013dd5ca3308913ea2fd61a86e1c5bc3fb5408c2b8c2",
               "curriculum": "54c1a33894cd0e705d1d23d5e92fb06da015daf604c5d3887a6d68167574eb2f"}
SEED = 839


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
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
            base.require(launch["runtime"] == runtime and launch["source"] == trained["source"] == INPUT_SOURCE
                and trained["protocol"] == recipe.PROTOCOL and trained["updates"] == 1000
                and trained["warm_start"]["parent_sha256"] == recipe.PARENT_SHA
                and trained["checkpoint_sha256"] == prior["checkpoint_sha256"][arm] == digest
                and base.sha256(INPUT / arm / "model_999.pt") == digest, "retained finals and identical runtime")
            base.require(recipe.decision(prior["cases"][arm]) == prior["decisions"][arm], "original deterministic gate")
        output = base.ROOT / "artifacts/evaluations" / args.run_id
        output.mkdir(parents=True, exist_ok=False)
        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source, host=host,
            runtime=runtime, input_source=INPUT_SOURCE, evaluation_sha256=EVALUATION_SHA,
            checkpoint_sha256=CHECKPOINTS, seed=SEED, cases=base.CASES,
            policy_acceptance=False, physical_motion_authorized=False))
        def health():
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(shared_host(), allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            cases, differences = {}, {}
            for arm in CHECKPOINTS:
                rows = []
                for index, old in enumerate(prior["cases"][arm][:4]):
                    base.require((old["seed"], old["speed"], old["yaw"]) == (SEED, *base.CASES[index]), "declared diagnostic cases")
                    row = base.evaluate_case(INPUT / arm / "model_999.pt", SEED, old["speed"], old["yaw"],
                        health, diagnostics=True, angular_diagnostics=True)
                    acceptance = {k: v for k, v in row.items() if k != "angular_reward_diagnostics"}
                    diff = replay.replay_differences(acceptance, old)
                    base.write_new(output / f"{arm}-s{SEED}-c{index}.json", row)
                    differences[f"{arm}-{index}"] = diff
                    rows.append(row)
                cases[arm] = rows
            base.write_new(output / "result.json", dict(protocol=PROTOCOL, source=args.source,
                evaluation_sha256=EVALUATION_SHA, checkpoint_sha256=CHECKPOINTS, cases=cases,
                original_decisions=prior["decisions"], replay_differences=differences,
                replay_verified=not any(differences.values()), decision="diagnostic-only-not-admission",
                policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False))
            health()
        except Exception as exc:
            base.write_new(output / "failure.json", dict(type=type(exc).__name__, error=str(exc)))
            raise


if __name__ == "__main__":
    main()
