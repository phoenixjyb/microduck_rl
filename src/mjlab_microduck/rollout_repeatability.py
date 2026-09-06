"""One frozen seed-367 recording-disabled/disabled control; never policy admission."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mjlab_microduck.first_attempt_smoke import (
    ACTOR_SHA256, SUPERVISOR_SHA256, OUTCOME, canonical, require, sha256,
)

PROTOCOL = "rollout-repeatability-disabled-s367-v1"
CASE = dict(nominal_speed_mps=0.4, obstacle_forward_m=0.9,
            obstacle_lateral_m=0.0, seed=367, num_envs=4, steps=700)
ROOT = Path("/home/converge/work/microduck_rl-athletics-obstacle-curriculum")
ACTOR = ROOT / "logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt"
SUPERVISOR = ROOT / "artifacts/checkpoints/hc4u4-bc60b2c-s42/supervisor.pt"
OUTPUT = ROOT / "artifacts/evaluations/rollout-repeatability-disabled-s367-v1"
# One newly authorized ten-minute control; not an overnight extension.
LAST_START = dt.datetime(2026, 9, 6, 1, 5, tzinfo=dt.timezone.utc)
COUNT_KEYS = ("collision_events", "clean_pass_events", "attempt_timeout_events",
              "fall_events", "nan_termination_events", "nonfinite_steps",
              "hard_failure_events", "other_terminal_events", "expected_attempts",
              "completed_attempts", "unresolved_attempts")


def validate_report(report):
    canonical(report)
    require(report["stage"] == "HC4U4-near-state-correction-phase-BC-rollout"
            and report["decision"] == "diagnostic-only", "report identity")
    require(report["physical_motion_authorized"] is False, "physical authority")
    require(report["checkpoint_sha256"] == ACTOR_SHA256
            and report["supervisor_checkpoint_sha256"] == SUPERVISOR_SHA256, "artifact identity")
    require(report["terminal_outcome_protocol"] == OUTCOME
            and report["evaluation_window"] == "first-terminal-attempt-per-environment-v1"
            and report["attempt_timeout_s"] == 12.0, "window/protocol")
    require(report["obstacle_sensor_model"] == dict(range_noise_m=0., bearing_noise_rad=0.,
            width_noise_m=0., height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.), "sensor identity")
    require(len(report["cases"]) == 1 and "first_attempt_recording_protocol" not in report
            and not any("dataset" in key for key in report), "unrecorded single case only")
    case = report["cases"][0]
    require(all(case[k] == v for k, v in CASE.items()), "case identity")
    require("first_attempt_recording" not in case and not any("dataset" in key for key in case)
            and case["terminal_outcome_protocol"] == OUTCOME,
            "recording disabled and frozen outcome protocol")
    require(type(case["steps_executed"]) is int and 1 <= case["steps_executed"] <= 700, "step bound")
    require(isinstance(case["representative_first_attempt_trace"], list)
            and case["representative_first_attempt_trace"], "representative trace required")
    for counts in (case, report["totals"]):
        require(all(type(counts[k]) is int and counts[k] >= 0 for k in COUNT_KEYS), "integer counts")
        require(counts["expected_attempts"] == counts["completed_attempts"] == 4
                and counts["unresolved_attempts"] == 0, "four complete first attempts")
        require(all(counts[k] == 0 for k in ("fall_events", "nan_termination_events",
                "nonfinite_steps", "hard_failure_events", "other_terminal_events")), "runtime hard/numerical failure")
        require(sum(counts[k] for k in ("collision_events", "clean_pass_events", "attempt_timeout_events")) == 4,
                "outcome partition")
    require(all(case[k] == report["totals"][k] for k in COUNT_KEYS), "pooled count reconciliation")
    require(case["motor_speed_rated_exceed_fraction"] == 0.
            and 0. <= case["motor_torque_utilization_p99"] <= .60, "runtime motor check")
    return case


def differences(left, right, path=""):
    """All typed JSON leaf/length differences, ordered deterministically."""
    if type(left) is not type(right):
        return [dict(path=path, first=left, second=right)]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(dict(path=path + "/" + key, first=left.get(key), second=right.get(key),
                                   missing="first" if key not in left else "second"))
            else:
                result.extend(differences(left[key], right[key], path + "/" + key))
        return result
    if isinstance(left, list):
        result = [] if len(left) == len(right) else [dict(path=path + "/length", first=len(left), second=len(right))]
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(differences(a, b, path + "/" + str(index)))
        return result
    return [] if canonical(left) == canonical(right) else [dict(path=path, first=left, second=right)]


def evaluate_pair(first_path: Path, second_path: Path):
    result = dict(protocol=PROTOCOL, decision="invalid-control-stop", errors=[],
                  policy_acceptance=False, recorder_validated=False,
                  training_data_admitted=False, physical_motion_authorized=False,
                  further_gpu_job_admitted=False)
    try:
        require(first_path.resolve() != second_path.resolve(), "distinct retained reports required")
        first, second = (json.loads(p.read_text()) for p in (first_path, second_path))
        a, b = validate_report(first), validate_report(second)
        result["report_sha256"] = dict(first=sha256(first_path), second=sha256(second_path))
        result["outcomes"] = [{k: c[k] for k in COUNT_KEYS} for c in (a, b)]
        diff = differences(first, second)
        equal = canonical(first) == canonical(second)
        require(equal == (not diff), "comparison consistency")
        result.update(exact_reports_equal=equal, difference_count=len(diff), differences=diff,
                      decision="same-seed-reports-match-in-control" if equal else
                               "same-seed-reports-diverge-with-recording-disabled")
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def read(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def check_host():
    for service in ("recomo-ai-mission-vllm.service", "recomo-ai-mission-subject-model-worker.service"):
        require(read("systemctl", "show", service, "-p", "ActiveState", "--value") == "inactive", service)
    require(not read("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"), "GPU compute workload")
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
    require(dt.datetime.now(dt.timezone.utc) < LAST_START, "control start cutoff")
    require(sha256(ACTOR) == ACTOR_SHA256 and sha256(SUPERVISOR) == SUPERVISOR_SHA256, "frozen artifacts")
    preflight = check_host()
    child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    launch = dict(protocol=PROTOCOL, source=args.source, preflight=preflight,
                  actor_sha256=ACTOR_SHA256, supervisor_sha256=SUPERVISOR_SHA256,
                  started_at=dt.datetime.now(dt.timezone.utc).isoformat(), runs=[],
                  versions={name: importlib.metadata.version(name) for name in
                            ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab")},
                  numerical_environment={name: child_env.get(name) for name in (
                      "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONHASHSEED",
                      "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_LAUNCH_BLOCKING")})
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / "launch.json").open("x") as stream:
        stream.write(json.dumps(launch, indent=2) + "\n")
    try:
        for mode in ("first", "second"):
            if mode == "second":
                time.sleep(2)
                check_host()
            command = [sys.executable, "-m", "mjlab_microduck.hierarchical_obstacle_rollout",
                       str(ACTOR), "--output-dir", str(OUTPUT / mode), "--num-envs", "4",
                       "--steps", "700", "--speeds", "0.4", "--obstacle-forward", "0.9",
                       "--obstacle-lateral", "0.0", "--seeds", "367", "--first-attempt-only",
                       "--supervisor-checkpoint", str(SUPERVISOR)]
            run = dict(mode=mode, command=command, returncode=None)
            launch["runs"].append(run)
            started = time.monotonic()
            try:
                with (OUTPUT / f"{mode}.log").open("x") as log:
                    child = subprocess.run(command, env=child_env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=240)
                run["returncode"] = child.returncode
            finally:
                run["wall_seconds"] = time.monotonic() - started
            require(child.returncode == 0, f"{mode} process exited {child.returncode}")
            validate_report(json.loads((OUTPUT / mode / "hierarchical-teacher-evaluation.json").read_text()))
        decision = evaluate_pair(OUTPUT / "first/hierarchical-teacher-evaluation.json",
                                 OUTPUT / "second/hierarchical-teacher-evaluation.json")
    except Exception as error:
        decision = dict(protocol=PROTOCOL, decision="invalid-control-stop",
                        policy_acceptance=False, recorder_validated=False, training_data_admitted=False,
                        physical_motion_authorized=False, further_gpu_job_admitted=False,
                        errors=[f"{type(error).__name__}: {error}"])
    for name, value in (("runtime.json", launch), ("decision.json", decision)):
        with (OUTPUT / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in decision.items() if k != "differences"}, sort_keys=True), flush=True)
    raise SystemExit(0 if decision["decision"] == "same-seed-reports-match-in-control" else 2)


if __name__ == "__main__":
    main()
