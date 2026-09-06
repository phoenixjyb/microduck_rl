"""Single predeclared seed-379 recovery diagnostic; no policy admission."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, OUTCOME, canonical, require, sha256
from mjlab_microduck.hc4u1_gate import NEAR_STAGE, NEAR_SHA256, FAR_STAGE, FAR_SHA256
from mjlab_microduck.hierarchical_obstacle import ObstacleTeacherCfg
from mjlab_microduck.motor_audit_smoke import AUDIT, DEPENDENCIES, JOINTS, VERSIONS, stats
from mjlab_microduck.recovery_control import RecoveryAccelerationCfg, ROLLOUT_STAGE
from mjlab_microduck.rollout_repeatability import ROOT, ACTOR, COUNT_KEYS, check_host, read

PROTOCOL = "recovery-cap-specialist-s379-v1"
OUTPUT = ROOT / "artifacts/evaluations" / PROTOCOL
NEAR = ROOT / "artifacts/checkpoints/hc4r2-bc-796634d-s42/supervisor.pt"
FAR = ROOT / "artifacts/checkpoints/hc4lh-11002cc-center002/supervisor.pt"
DEADLINE = dt.datetime(2026, 9, 6, 23, tzinfo=dt.timezone.utc)
SERVICE_SECONDS, CLOSEOUT_SECONDS, CHILD_SECONDS = 2400, 300, 180
CELLS = ((.3, .9), (.4, .9), (.3, 1.15), (.4, 1.15))
MODES = ("a1", "a2", "b")
PHASES = ("approach", "interaction", "recovery")
STATUSES = ("not-observed", "recovered-in-window", "window-missed", "censored-before-window")
COUNTS = (*COUNT_KEYS, "resolved_attempts")


def number(value, *, nonnegative=True):
    require(type(value) in (float, int) and math.isfinite(value)
            and (not nonnegative or value >= 0), "finite statistic domain")
    return value


def integer(value, low, high):
    require(type(value) is int and low <= value <= high, "integer/count domain")
    return value


def validate_report(report, cell_index, mode):
    """Validate identity/accounting first; numerical gates are separate."""
    canonical(report)  # rejects non-standard NaN/Infinity anywhere, including trace
    speed, forward = CELLS[cell_index]
    stage, supervisor = (NEAR_STAGE, NEAR_SHA256) if forward == .9 else (FAR_STAGE, FAR_SHA256)
    require(report["decision"] == "diagnostic-only" and report["physical_motion_authorized"] is False,
            "diagnostic authority")
    require(report["checkpoint_sha256"] == ACTOR_SHA256
            and report["supervisor_checkpoint_sha256"] == supervisor, "artifact identity")
    require(report["teacher_config"] == asdict(ObstacleTeacherCfg()), "frozen command bounds")
    require(report["perception"] == "exact structured geometry; no raw camera perception"
            and report["obstacle_sensor_model"] == dict(range_noise_m=0., bearing_noise_rad=0.,
                width_noise_m=0., height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.),
            "frozen perception")
    require(report["evaluation_window"] == "first-terminal-attempt-per-environment-v1"
            and report["terminal_outcome_protocol"] == OUTCOME and report["attempt_timeout_s"] == 12.,
            "frozen first attempt protocol")
    require(type(report["cases"]) is list and len(report["cases"]) == 1, "exactly one case")
    case = report["cases"][0]
    for obj in (report, case):
        require(not any("dataset" in k or "first_attempt_recording" in k for k in obj), "no dataset/video")
    expected = dict(nominal_speed_mps=speed, obstacle_forward_m=forward, obstacle_lateral_m=0.,
                    seed=379, num_envs=8, steps=700)
    require(all(type(case[k]) is type(v) and case[k] == v for k, v in expected.items()), "exact case")
    require(case["evaluation_window"] == report["evaluation_window"]
            and case["terminal_outcome_protocol"] == OUTCOME, "case window")
    if mode == "b":
        require(report["stage"] == ROLLOUT_STAGE and report["source_controller_stage"] == stage
                and report["policy_acceptance"] is False, "capped source stage/authority")
        provenance = RecoveryAccelerationCfg(.2).provenance()
        require(report["recovery_control"] == provenance and case["recovery_control"] ==
                {**provenance, "update_dt_s": .1}, "exact recovery execution configuration")
    else:
        require(report["stage"] == stage and "source_controller_stage" not in report
                and "recovery_control" not in report and "recovery_control" not in case, "uncapped baseline")
    steps = integer(case["steps_executed"], 1, 700)
    for obj in (case, report["totals"]):
        for key in COUNTS: integer(obj[key], 0, 8 if key != "nonfinite_steps" else steps)
        require(obj["expected_attempts"] == 8 and obj["completed_attempts"] + obj["unresolved_attempts"] == 8,
                "attempt accounting")
        require(obj["resolved_attempts"] == sum(obj[k] for k in
                ("collision_events", "clean_pass_events", "attempt_timeout_events"))
                and obj["completed_attempts"] == obj["resolved_attempts"] + obj["other_terminal_events"]
                    + obj["hard_failure_events"],
                "terminal partition")
    require(all(case[k] == report["totals"][k] for k in COUNTS), "count reconciliation")
    for phase in PHASES:
        integer(case[f"{phase}_samples"], 1, steps * 8)
        number(case[f"{phase}_route_speed_mps"], nonnegative=False)
    number(case["motor_torque_utilization_p99"])
    require(0 <= number(case["motor_speed_rated_exceed_fraction"]) <= 1, "motor speed domain")
    require(report["motor_measurement_audit_protocol"] == AUDIT, "audit protocol")
    audit = case["motor_measurement_audit"]
    require(audit["protocol"] == AUDIT and audit["decision"] == "diagnostic-only-not-admission"
            and audit["finite"] is True and audit["joint_columns"] == list(JOINTS)
            and audit["stall_reference_nm"] == .6, "audit identity/layout")
    for key in ("physical_motion_authorized", "policy_acceptance", "runtime_equivalence_validated",
                "training_data_admitted", "legacy_metrics_replaced"):
        require(audit[key] is False, "audit authority")
    for key, expected in {
        "sampling": "post-decimation-metrics-hook-before-reset-and-final-forward",
        "force_timing": "last-physics-substep-derived-force; one-integration lag",
        "speed_timing": "integrated-joint-velocity-at-capture",
        "peak_scope": "control-step samples only; not all physics substeps",
        "summary_precision": "float64; scaled RMS",
    }.items(): require(audit[key] == expected, "audited sampling semantics")
    require(integer(audit["steps_captured"], 1, 700) == steps
            and integer(audit["terminal_environment_steps"], 0, 8) == case["completed_attempts"]
            and integer(audit["incomplete_first_attempts"], 0, 8) == case["unresolved_attempts"], "audit coverage")
    groups = audit["groups"]
    require(set(groups) == {"all", *PHASES}, "phase coverage")
    for phase, group in groups.items():
        n = integer(group["environment_steps"], 1, 8 * steps)
        if phase != "all": require(n == case[f"{phase}_samples"], "phase sample reconciliation")
        for key in ("force_nm", "speed_rad_s", "stall_reference_utilization"): stats(group[key], 14 * n)
        require(set(group["by_joint"]) == set(JOINTS), "joint coverage")
        for value in group["by_joint"].values(): stats(value, n)
        for key in ("abs_p99", "abs_max", "rms"):
            require(math.isclose(group["force_nm"][key] / .6,
                    group["stall_reference_utilization"][key], rel_tol=1e-12, abs_tol=1e-12), "normalization")
    require(groups["all"]["environment_steps"] == sum(groups[p]["environment_steps"] for p in PHASES)
            and steps + 7 <= groups["all"]["environment_steps"] <= steps * 8, "audit phase partition")
    for key in ("terminal_force_nm", "terminal_post_return_force_nm", "terminal_post_return_minus_pre_reset_nm"):
        stats(audit[key], case["completed_attempts"] * 14)
    measurement = case["recovery_speed_measurement"]
    require(measurement["protocol"] == "first-attempt-recovery-speed-v1"
            and measurement["sampling"] == "pre-control-step route speed"
            and measurement["nominal_speed_mps"] == speed and measurement["step_dt_s"] == .02
            and measurement["speed_tolerance_mps"] == .03 and measurement["stable_span_s"] == .5
            and measurement["deadline_s"] == 2. and measurement["policy_acceptance"] is False
            and measurement["physical_motion_authorized"] is False, "recovery measurement protocol")
    rows = measurement["environments"]
    require(type(rows) is list and len(rows) == 8, "recovery environment coverage")
    for i, row in enumerate(rows):
        require(type(row["environment"]) is int and row["environment"] == i
                and type(row["terminal"]) is bool, "ordered unique recovery identity")
        entry, span, latency = (row[k] for k in
                               ("first_recovery_step", "sampled_recovery_span_s", "stable_recovery_latency_s"))
        if entry is None:
            require(span is None and latency is None, "unobserved recovery")
            status = "not-observed"
        else:
            integer(entry, 0, steps - 1)
            require(number(span) <= (steps - 1 - entry) * .02 + 1e-12, "recovery time bound")
            if latency is not None:
                require(.5 - 1e-12 <= number(latency) <= span + 1e-12, "stable sampled span")
            status = ("recovered-in-window" if latency is not None and latency <= 2. + 1e-12 else
                      "window-missed" if span >= 2. - 1e-12 else "censored-before-window")
        require(row["status"] == status, "derived recovery classification")
    require(sum(r["terminal"] for r in rows) == case["completed_attempts"], "observer terminal reconciliation")
    require(set(measurement["counts"]) == set(STATUSES), "recovery count keys")
    for status in STATUSES:
        require(integer(measurement["counts"][status], 0, 8) == sum(r["status"] == status for r in rows),
                "recovery counts cannot hide censored rows")
    return case


def absolute_failures(case):
    failures = []
    expected = dict.fromkeys(COUNTS, 0)
    expected.update(expected_attempts=8, completed_attempts=8, resolved_attempts=8, clean_pass_events=8)
    for key, value in expected.items():
        if case[key] != value: failures.append(key)
    if case["motor_torque_utilization_p99"] > .60: failures.append("legacy-torque-p99")
    if case["motor_speed_rated_exceed_fraction"] != 0: failures.append("rated-speed")
    if case["recovery_speed_measurement"]["counts"]["recovered-in-window"] != 8:
        failures.append("recovery-window")
    return failures


def recovery_load(case):
    group = case["motor_measurement_audit"]["groups"]["recovery"]
    return (group["stall_reference_utilization"]["abs_p99"],
            max(value["abs_p99"] for value in group["by_joint"].values()))


def compare_cell(cases):
    failures = []
    if len(cases) >= 2:
        a1, a2 = cases[:2]
        if abs(a2["motor_torque_utilization_p99"] - a1["motor_torque_utilization_p99"]) > .02 + 1e-12:
            failures.append("repeat-torque")
        for phase in ("approach", "recovery"):
            key = f"{phase}_route_speed_mps"
            if abs(a2[key] - a1[key]) > .02 + 1e-12: failures.append(f"repeat-{phase}-speed")
    if len(cases) == 3:
        b = cases[2]
        for index, baseline in enumerate(cases[:2], 1):
            for phase in ("approach", "recovery"):
                key = f"{phase}_route_speed_mps"
                if b[key] < baseline[key] - .03 - 1e-12: failures.append(f"b-v-a{index}-{phase}-speed")
            for metric, (candidate, reference) in enumerate(zip(recovery_load(b), recovery_load(baseline))):
                if candidate > reference + 1e-12: failures.append(f"b-v-a{index}-recovery-load-{metric}")
    return failures


def evaluate_paths(paths):
    result = dict(protocol=PROTOCOL, decision="invalid-evidence-stop", failures=[], reports=[],
                  policy_acceptance=False, physical_motion_authorized=False, training_data_admitted=False,
                  historical_repeatability_validated=False, supports_predeclared_pilot=False)
    try:
        require(1 <= len(paths) <= 12 and len({p.resolve() for p in paths}) == len(paths),
                "distinct ordered prefix of 1..12 reports")
        cases = []
        for index, path in enumerate(paths):
            cell, mode_index = divmod(index, 3)
            case = validate_report(json.loads(path.read_text()), cell, MODES[mode_index])
            cases.append(case)
            result["reports"].append(dict(cell=cell, mode=MODES[mode_index], sha256=sha256(path),
                legacy_torque_p99=case["motor_torque_utilization_p99"], recovery_load=recovery_load(case),
                outcomes={k: case[k] for k in COUNTS},
                recovery_counts=case["recovery_speed_measurement"]["counts"]))
            failures = absolute_failures(case) + compare_cell(cases[cell * 3:])
            if failures:
                require(index == len(paths) - 1, "reports exist after first failed gate")
                result.update(decision="numerical-gate-stop", failures=failures)
                return result
        if len(paths) < 12:
            result["decision"] = "valid-prefix-not-complete"
            return result
        by_mode = [cases[i::3] for i in range(3)]
        def pooled(group, phase):
            count = sum(c[f"{phase}_samples"] for c in group)
            return number(math.fsum(c[f"{phase}_route_speed_mps"] * c[f"{phase}_samples"]
                                    for c in group) / count, nonnegative=False)
        for index, baseline in enumerate(by_mode[:2], 1):
            for phase in ("approach", "recovery"):
                if pooled(by_mode[2], phase) < pooled(baseline, phase) - .01 - 1e-12:
                    result["failures"].append(f"pooled-v-a{index}-{phase}-speed")
            # Arithmetic mean of cell quantiles, explicitly not a pooled quantile.
            if math.fsum(recovery_load(c)[0] for c in by_mode[2]) > .95 * math.fsum(recovery_load(c)[0] for c in baseline) + 1e-12:
                result["failures"].append(f"recovery-cell-mean-v-a{index}-five-percent")
        supported = not result["failures"]
        result.update(decision="recovery-cap-diagnostic-supports-pilot" if supported else "numerical-gate-stop",
                      supports_predeclared_pilot=supported)
    except (ValueError, TypeError, KeyError, OSError, IndexError, OverflowError) as error:
        result.update(decision="invalid-evidence-stop", failures=[f"{type(error).__name__}: {error}"])
    return result


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_source(source):
    require(Path.cwd().resolve() == ROOT and read("git", "branch", "--show-current") ==
            "feat/athletics-obstacle-curriculum", "exact worktree/branch")
    require(len(source) == 40 and all(c in "0123456789abcdef" for c in source)
            and read("git", "rev-parse", "HEAD") == source
            and not read("git", "status", "--porcelain"), "clean exact source")


def command_for(index, destination):
    cell, mode = divmod(index, 3)
    speed, forward = CELLS[cell]
    command = [sys.executable, "-m", "mjlab_microduck.hierarchical_obstacle_rollout", str(ACTOR),
        "--output-dir", str(destination), "--num-envs", "8", "--steps", "700", "--seeds", "379",
        "--speeds", str(speed), "--obstacle-forward", str(forward), "--obstacle-lateral", "0.0",
        "--first-attempt-only", "--motor-measurement-audit", "--supervisor-checkpoint",
        str(NEAR if forward == .9 else FAR)]
    if mode == 2: command += ["--recovery-acceleration-mps2", "0.2"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    verify_source(args.source)
    require(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=SERVICE_SECONDS + CLOSEOUT_SECONDS)
            < DEADLINE, "full service and closeout must fit before deadline")
    hashes = {str(p): sha256(p) for p in (ACTOR, NEAR, FAR)}
    require(list(hashes.values()) == [ACTOR_SHA256, NEAR_SHA256, FAR_SHA256], "frozen model hashes")
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    require(versions == VERSIONS, "frozen versions")
    import mjlab
    dependencies = {k: sha256(Path(mjlab.__file__).parent / k) for k in DEPENDENCIES}
    require(dependencies == DEPENDENCIES, "audited dependencies")
    preflight = check_host()
    OUTPUT.mkdir(exist_ok=False)
    child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    launch = dict(protocol=PROTOCOL, source=args.source, model_sha256=hashes, versions=versions,
                  dependencies=dependencies, initial_preflight=preflight, child_timeout_s=CHILD_SECONDS,
                  deadline=DEADLINE.isoformat(), started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  numerical_environment={k: child_env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                    "PYTHONUNBUFFERED", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                    "CUDA_LAUNCH_BLOCKING")})
    write_new(OUTPUT / "launch.json", launch)
    paths, runs = [], []
    started = time.monotonic()
    decision = None
    try:
        for index in range(12):
            require(time.monotonic() - started + CHILD_SECONDS + 60 < SERVICE_SECONDS,
                    "service closeout budget")
            require(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=CHILD_SECONDS + CLOSEOUT_SECONDS)
                    < DEADLINE, "child deadline")
            verify_source(args.source)
            require({str(p): sha256(p) for p in (ACTOR, NEAR, FAR)} == hashes, "artifacts changed")
            if index: time.sleep(2)  # driver teardown; no other workloads are stopped
            host = check_host()
            destination = OUTPUT / f"{index:02d}-cell{index // 3}-{MODES[index % 3]}"
            command = command_for(index, destination)
            run = dict(index=index, command=command, preflight=host, returncode=None)
            runs.append(run)
            write_new(OUTPUT / f"launch-{index:02d}.json", run)
            child_started = time.monotonic()
            try:
                with (OUTPUT / f"{index:02d}.log").open("x") as log:
                    child = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=CHILD_SECONDS)
                run["returncode"] = child.returncode
            finally:
                run["wall_seconds"] = time.monotonic() - child_started
                write_new(OUTPUT / f"runtime-{index:02d}.json", run)
            require(child.returncode == 0, f"child {index} exit {child.returncode}")
            paths.append(destination / "hierarchical-teacher-evaluation.json")
            decision = evaluate_paths(paths)
            write_new(OUTPUT / f"decision-{index:02d}.json", decision)
            print(json.dumps(decision, sort_keys=True), flush=True)
            if decision["decision"] != "valid-prefix-not-complete": break
    except Exception as error:
        decision = dict(protocol=PROTOCOL, decision="runtime-failure-stop",
                        failures=[f"{type(error).__name__}: {error}"], policy_acceptance=False,
                        physical_motion_authorized=False, supports_predeclared_pilot=False)
    write_new(OUTPUT / "runtime.json", dict(runs=runs, wall_seconds=time.monotonic() - started))
    write_new(OUTPUT / "decision.json", decision)
    raise SystemExit(0 if decision["decision"] == "recovery-cap-diagnostic-supports-pilot" else 2)


if __name__ == "__main__": main()
