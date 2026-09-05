"""Run only the predeclared seed-359 recorder smoke in a bounded user service."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mjlab_microduck.first_attempt_smoke import (
    ACTOR_SHA256, SUPERVISOR_SHA256, PROTOCOL, evaluate_smoke, require, sha256,
)

ROOT = Path("/home/converge/work/microduck_rl-athletics-obstacle-curriculum")
ACTOR = ROOT / "logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt"
SUPERVISOR = ROOT / "artifacts/checkpoints/hc4u4-bc60b2c-s42/supervisor.pt"
OUTPUT = ROOT / "artifacts/evaluations/first-attempt-recorder-s359-v1"
LAST_START = dt.datetime(2026, 9, 5, 22, 25, tzinfo=dt.timezone.utc)


def read(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def check_host():
    for service in ("recomo-ai-mission-vllm.service", "recomo-ai-mission-subject-model-worker.service"):
        require(read("systemctl", "show", service, "-p", "ActiveState", "--value") == "inactive", service)
    require(not read("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"), "GPU has compute workload")
    status = read("nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used", "--format=csv,noheader,nounits")
    utilization, temperature, memory = map(int, status.split(","))
    require(utilization == 0 and temperature < 80 and memory < 100, "idle/cool GPU required")
    return dict(utilization_percent=utilization, temperature_c=temperature, memory_mib=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "exact worktree")
    require(read("git", "branch", "--show-current") == "feat/athletics-obstacle-curriculum", "exact branch")
    require(not read("git", "status", "--porcelain"), "clean source")
    require(read("git", "rev-parse", "HEAD") == args.source and len(args.source) == 40, "exact source")
    require(dt.datetime.now(dt.timezone.utc) < LAST_START, "smoke start cutoff")
    require(sha256(ACTOR) == ACTOR_SHA256 and sha256(SUPERVISOR) == SUPERVISOR_SHA256, "frozen artifacts")
    preflight = check_host()
    OUTPUT.mkdir(exist_ok=False)
    launch = dict(protocol=PROTOCOL, source=args.source, preflight=preflight,
                  actor_sha256=ACTOR_SHA256, supervisor_sha256=SUPERVISOR_SHA256,
                  started_at=dt.datetime.now(dt.timezone.utc).isoformat(), runs=[])
    with (OUTPUT / "launch.json").open("x") as stream:
        stream.write(json.dumps(launch, indent=2) + "\n")
    try:
        for mode in ("disabled", "enabled"):
            if mode == "enabled":
                # GPU utilization sampling can trail process exit by one second.
                time.sleep(2)
                check_host()
            command = [sys.executable, "-m", "mjlab_microduck.hierarchical_obstacle_rollout",
                       str(ACTOR), "--output-dir", str(OUTPUT / mode), "--num-envs", "4",
                       "--steps", "700", "--speeds", "0.4", "--obstacle-forward", "0.9",
                       "--obstacle-lateral", "0.0", "--seeds", "359", "--first-attempt-only",
                       "--supervisor-checkpoint", str(SUPERVISOR)]
            if mode == "enabled":
                command.append("--record-first-attempts")
            started = time.monotonic()
            run = dict(mode=mode, command=command, returncode=None)
            launch["runs"].append(run)
            try:
                with (OUTPUT / f"{mode}.log").open("x") as log:
                    result = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
                                            "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
                                            stdout=log, stderr=subprocess.STDOUT, timeout=240)
                run["returncode"] = result.returncode
            finally:
                run["wall_seconds"] = time.monotonic() - started
            require(result.returncode == 0, f"{mode} process exited {result.returncode}")
        decision = evaluate_smoke(OUTPUT / "disabled/hierarchical-teacher-evaluation.json",
                                  OUTPUT / "enabled/hierarchical-teacher-evaluation.json")
    except Exception as error:
        decision = dict(protocol=PROTOCOL, decision="stop-recorder-diagnosis",
                        may_predeclare_fresh_diagnostic=False, policy_acceptance=False,
                        training_data_admitted=False, physical_motion_authorized=False,
                        errors=[f"{type(error).__name__}: {error}"])
    with (OUTPUT / "runtime.json").open("x") as stream:
        stream.write(json.dumps(launch, indent=2) + "\n")
    with (OUTPUT / "decision.json").open("x") as stream:
        stream.write(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)
    raise SystemExit(0 if decision["may_predeclare_fresh_diagnostic"] else 2)


if __name__ == "__main__":
    main()
