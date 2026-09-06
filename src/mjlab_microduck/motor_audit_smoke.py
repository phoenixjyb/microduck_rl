"""One seed-373 motor-observer smoke; neither policy nor repeatability admission."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from mjlab_microduck.first_attempt_smoke import (
    ACTOR_SHA256, SUPERVISOR_SHA256, OUTCOME, canonical, require, sha256,
)
from mjlab_microduck.rollout_repeatability import ROOT, ACTOR, SUPERVISOR, COUNT_KEYS, check_host, read

PROTOCOL = "motor-measurement-audit-s373-v1"
AUDIT = "motor-pre-reset-step-audit-v1"
OUTPUT = ROOT / "artifacts/evaluations/motor-measurement-audit-s373-v1"
LAST_START = dt.datetime(2026, 9, 6, 9, 35, tzinfo=dt.timezone.utc)
CASE = dict(nominal_speed_mps=.4, obstacle_forward_m=.9, obstacle_lateral_m=0.,
            seed=373, num_envs=4, steps=700)
JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
          "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
VERSIONS = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
            "mujoco-warp": "3.8.1", "mjlab": "1.3.0"}
DEPENDENCIES = {
    "envs/manager_based_rl_env.py": "a381027e336d6313cd338d541230b36864657edac89708a33e7e60c0c6fb74d2",
    "managers/metrics_manager.py": "c78b9a58d6e5457f841847ee9e48592d2634d294d583585adeedbb030c879d85",
    "entity/data.py": "fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657",
}


def stats(value, count):
    require(type(value["samples"]) is int and value["samples"] == count
            and type(value["nonfinite_samples"]) is int and value["nonfinite_samples"] == 0,
            "sample count/finite coverage")
    for key in ("abs_p99", "abs_max", "rms"):
        require(type(value[key]) in (float, int) and math.isfinite(value[key]) and value[key] >= 0,
                "finite nonnegative statistic")
    require(value["abs_p99"] <= value["abs_max"] and value["rms"] <= value["abs_max"] + 1e-12,
            "statistic ordering")


def validate_report(report):
    canonical(report)
    require(report["stage"] == "HC4U4-near-state-correction-phase-BC-rollout"
            and report["decision"] == "diagnostic-only" and report["physical_motion_authorized"] is False,
            "report authority/identity")
    require(report["checkpoint_sha256"] == ACTOR_SHA256
            and report["supervisor_checkpoint_sha256"] == SUPERVISOR_SHA256, "frozen artifacts")
    require(report["terminal_outcome_protocol"] == OUTCOME
            and report["evaluation_window"] == "first-terminal-attempt-per-environment-v1"
            and report["attempt_timeout_s"] == 12., "first-attempt protocol")
    require(report["obstacle_sensor_model"] == dict(range_noise_m=0., bearing_noise_rad=0.,
            width_noise_m=0., height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.),
            "frozen external obstacle observations")
    require(len(report["cases"]) == 1 and report["motor_measurement_audit_protocol"] == AUDIT,
            "single audited case")
    case = report["cases"][0]
    for obj in (report, case):
        require(not any("dataset" in k or "first_attempt_recording" in k for k in obj),
                "no dataset or trajectory recorder")
    require(all(type(case[k]) is type(v) and case[k] == v for k, v in CASE.items()), "case identity")
    require(case["terminal_outcome_protocol"] == OUTCOME, "case outcome identity")
    steps = case["steps_executed"]
    require(type(steps) is int and 1 <= steps <= 700, "step bound")
    for counts in (case, report["totals"]):
        require(all(type(counts[k]) is int and counts[k] >= 0 for k in COUNT_KEYS), "integer outcomes")
        require(counts["expected_attempts"] == counts["completed_attempts"] == 4
                and counts["unresolved_attempts"] == 0, "four complete first attempts")
        require(sum(counts[k] for k in ("collision_events", "clean_pass_events", "attempt_timeout_events",
                "hard_failure_events")) == 4, "terminal partition")
    require(all(case[k] == report["totals"][k] for k in COUNT_KEYS), "pooled count reconciliation")
    audit = case["motor_measurement_audit"]
    require(audit["protocol"] == AUDIT and audit["decision"] == "diagnostic-only-not-admission", "audit identity")
    require(all(audit[k] is False for k in ("physical_motion_authorized", "policy_acceptance",
            "runtime_equivalence_validated", "training_data_admitted", "legacy_metrics_replaced")), "audit authority")
    require(audit["finite"] is True and audit["joint_columns"] == list(JOINTS)
            and audit["stall_reference_nm"] == .6, "finite named layout/reference")
    for key, expected in {
        "sampling": "post-decimation-metrics-hook-before-reset-and-final-forward",
        "force_timing": "last-physics-substep-derived-force; one-integration lag",
        "speed_timing": "integrated-joint-velocity-at-capture",
        "peak_scope": "control-step samples only; not all physics substeps",
        "summary_precision": "float64; scaled RMS",
    }.items():
        require(audit[key] == expected, "sampling semantics")
    for key, expected in (("steps_captured", steps), ("terminal_environment_steps", 4),
                          ("incomplete_first_attempts", 0)):
        require(type(audit[key]) is int and audit[key] == expected, "capture/terminal reconciliation")
    groups = audit["groups"]
    require(set(groups) == {"all", "approach", "interaction", "recovery"}, "phase layout")
    for group in groups.values():
        n = group["environment_steps"]
        require(type(n) is int and n > 0, "all three phases must be exercised")
        for key in ("force_nm", "speed_rad_s", "stall_reference_utilization"):
            stats(group[key], 14 * n)
        require(set(group["by_joint"]) == set(JOINTS), "named joint coverage")
        for joint in group["by_joint"].values():
            stats(joint, n)
        for key in ("abs_p99", "abs_max", "rms"):
            require(math.isclose(group["force_nm"][key] / .6,
                                 group["stall_reference_utilization"][key], rel_tol=1e-12, abs_tol=1e-12),
                    "force/reference normalization")
    total = groups["all"]["environment_steps"]
    require(steps + 3 <= total <= steps * 4 and total == sum(groups[p]["environment_steps"]
            for p in ("approach", "interaction", "recovery")), "first-attempt phase partition")
    for key in ("terminal_force_nm", "terminal_post_return_force_nm", "terminal_post_return_minus_pre_reset_nm"):
        stats(audit[key], 56)
    require(type(case["motor_torque_utilization_p99"]) in (float, int)
            and case["motor_torque_utilization_p99"] >= 0
            and type(case["motor_speed_rated_exceed_fraction"]) in (float, int)
            and 0 <= case["motor_speed_rated_exceed_fraction"] <= 1, "legacy motor statistic domain")
    return case


def evaluate_report(path):
    result = dict(protocol=PROTOCOL, decision="invalid-audit-stop", errors=[], measurement_structure_validated=False,
                  legacy_runtime_gate_passed=False, policy_acceptance=False, runtime_equivalence_validated=False,
                  training_data_admitted=False, physical_motion_authorized=False, further_gpu_job_admitted=False)
    try:
        result["report_sha256"] = sha256(path)
        case = validate_report(json.loads(path.read_text()))
        legacy = (case["motor_torque_utilization_p99"] <= .60 and case["motor_speed_rated_exceed_fraction"] == 0.
                  and all(case[k] == 0 for k in ("fall_events", "nan_termination_events", "nonfinite_steps",
                                               "hard_failure_events", "other_terminal_events")))
        result.update(measurement_structure_validated=True, legacy_runtime_gate_passed=legacy,
                      outcomes={k: case[k] for k in COUNT_KEYS},
                      legacy_torque_utilization_p99=case["motor_torque_utilization_p99"],
                      pre_reset_stall_reference_utilization_p99=case["motor_measurement_audit"]["groups"]["all"]
                          ["stall_reference_utilization"]["abs_p99"],
                      decision="measurement-smoke-validated-not-admission" if legacy else "legacy-runtime-gate-stop")
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT and read("git", "branch", "--show-current") ==
            "feat/athletics-obstacle-curriculum", "exact worktree/branch")
    require(not read("git", "status", "--porcelain") and read("git", "rev-parse", "HEAD") == args.source
            and len(args.source) == 40, "clean exact source")
    require(dt.datetime.now(dt.timezone.utc) < LAST_START, "single smoke start cutoff")
    require(sha256(ACTOR) == ACTOR_SHA256 and sha256(SUPERVISOR) == SUPERVISOR_SHA256, "artifact hashes")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, "frozen runtime versions")
    import mjlab
    dependencies = {name: sha256(Path(mjlab.__file__).parent / name) for name in DEPENDENCIES}
    require(dependencies == DEPENDENCIES, "audited dependency identity")
    preflight = check_host()
    child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    command = [sys.executable, "-m", "mjlab_microduck.hierarchical_obstacle_rollout", str(ACTOR),
               "--output-dir", str(OUTPUT / "rollout"), "--num-envs", "4", "--steps", "700",
               "--speeds", "0.4", "--obstacle-forward", "0.9", "--obstacle-lateral", "0.0",
               "--seeds", "373", "--first-attempt-only", "--motor-measurement-audit",
               "--supervisor-checkpoint", str(SUPERVISOR)]
    launch = dict(protocol=PROTOCOL, source=args.source, preflight=preflight, versions=versions,
                  dependency_sha256=dependencies, actor_sha256=ACTOR_SHA256, supervisor_sha256=SUPERVISOR_SHA256,
                  command=command, child_timeout_seconds=240, returncode=None,
                  started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  numerical_environment={k: child_env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                    "PYTHONUNBUFFERED", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_LAUNCH_BLOCKING")})
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / "launch.json").open("x") as stream:
        stream.write(json.dumps(launch, indent=2, allow_nan=False) + "\n")
    started = time.monotonic()
    try:
        with (OUTPUT / "rollout.log").open("x") as log:
            child = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        launch["returncode"] = child.returncode
        require(child.returncode == 0, f"rollout exited {child.returncode}")
        decision = evaluate_report(OUTPUT / "rollout/hierarchical-teacher-evaluation.json")
    except Exception as error:
        decision = dict(protocol=PROTOCOL, decision="runtime-failure-stop", errors=[f"{type(error).__name__}: {error}"],
                        measurement_structure_validated=False, legacy_runtime_gate_passed=False,
                        policy_acceptance=False, runtime_equivalence_validated=False, training_data_admitted=False,
                        physical_motion_authorized=False, further_gpu_job_admitted=False)
    launch["wall_seconds"] = time.monotonic() - started
    for name, value in (("runtime.json", launch), ("decision.json", decision)):
        with (OUTPUT / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)
    raise SystemExit(0 if decision["decision"] == "measurement-smoke-validated-not-admission" else 2)


if __name__ == "__main__":
    main()
