"""Frozen, CPU-only validation of the seed-359 recorder equivalence smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PROTOCOL = "first-attempt-recorder-smoke-s359-v1"
RECORDING = "all-first-attempt-pre-step-v1"
OUTCOME = "hard-failure-collision-timeout-pass-v1"
ACTOR_SHA256 = "080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54"
SUPERVISOR_SHA256 = "29855a51df8fe885d6ffed7fedf028093a8449a68b10b4b0e8a4bde7069bcf5b"
CASE = dict(nominal_speed_mps=0.4, obstacle_forward_m=0.9,
            obstacle_lateral_m=0.0, seed=359, num_envs=4, steps=700)
FIELDS = ("route_progress_m", "route_lateral_error_m", "route_heading_error_rad",
          "route_speed_mps", "obstacle_ahead_m", "obstacle_clearance_m", "phase",
          "command_speed_mps", "command_yaw_rate_rps")
RAW = ("collision", "pass", "timeout", "fall", "nan")
COUNT_FIELDS = {"collision": "collision_events", "pass": "clean_pass_events",
                "timeout": "attempt_timeout_events", "other_terminal": "other_terminal_events"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _validate_identity(report: dict) -> dict:
    canonical(report)  # Reject NaN/Infinity anywhere, including legacy metrics.
    require(report["stage"] == "HC4U4-near-state-correction-phase-BC-rollout", "stage")
    require(report["decision"] == "diagnostic-only", "report purpose")
    require(report["physical_motion_authorized"] is False, "physical authority")
    require(report["checkpoint_sha256"] == ACTOR_SHA256, "actor identity")
    require(report["supervisor_checkpoint_sha256"] == SUPERVISOR_SHA256, "supervisor identity")
    require(report["terminal_outcome_protocol"] == OUTCOME, "outcome protocol")
    require(report["evaluation_window"] == "first-terminal-attempt-per-environment-v1", "window")
    require(report["attempt_timeout_s"] == 12.0, "timeout protocol")
    require(report["obstacle_sensor_model"] == dict(range_noise_m=0.0, bearing_noise_rad=0.0,
            width_noise_m=0.0, height_noise_m=0.0, closing_rate_noise_mps=0.0,
            dropout_probability=0.0), "sensor protocol")
    require(len(report["cases"]) == 1, "exactly one smoke case")
    case = report["cases"][0]
    require(all(case[k] == v for k, v in CASE.items()), "case identity")
    require(case["terminal_outcome_protocol"] == OUTCOME, "case outcome protocol")
    require(type(case["steps_executed"]) is int and 1 <= case["steps_executed"] <= 700, "executed steps")
    for counts in (case, report["totals"]):
        require(counts["expected_attempts"] == counts["completed_attempts"] == 4
                and counts["unresolved_attempts"] == 0, "all four first attempts required")
    return case


def validate_sidecar(report: dict, report_path: Path) -> dict:
    case = _validate_identity(report)
    require(report["first_attempt_recording_protocol"] == RECORDING, "recording protocol")
    descriptor = case["first_attempt_recording"]
    path = report_path.parent / "first-attempt-traces" / "case-000.json"
    require(Path(descriptor["path"]).resolve() == path.resolve(), "sidecar path binding")
    require(descriptor["protocol"] == RECORDING and sha256(path) == descriptor["sha256"], "sidecar hash")
    recording = json.loads(path.read_text())
    canonical(recording)
    require(recording["schema_version"] == 1 and recording["protocol"] == RECORDING
            and recording["terminal_outcome_protocol"] == OUTCOME, "sidecar protocols")
    require(recording["purpose"] == "diagnostic-only; not-training-data; not-policy-acceptance"
            and recording["physical_motion_authorized"] is False, "sidecar purpose")
    require(recording["checkpoint_sha256"] == ACTOR_SHA256
            and recording["supervisor_checkpoint_sha256"] == SUPERVISOR_SHA256, "sidecar artifacts")
    require(recording["case_index"] == 0 and recording["case"] == {
        **CASE, "steps_executed": case["steps_executed"]}, "sidecar case")
    require(recording["num_envs"] == recording["completed_attempts"] == 4
            and recording["max_steps"] == 700
            and recording["steps_recorded"] == case["steps_executed"], "sidecar counts")
    require(recording["sample_interval_steps"] == 5
            and recording["terminal_pre_step_always_retained"] is True
            and recording["max_frames_per_environment"] == 141, "sidecar bounds")
    require(recording["clearance_kind"] == "center-distance-minus-0.22m-proxy; not-contact-distance"
            and recording["phase_codes"] == {"0": "approach", "1": "interaction", "2": "recovery"},
            "state semantics")
    attempts = recording["attempts"]
    require(len(attempts) == 4 and [a["environment_id"] for a in attempts] == list(range(4)), "environment identity")
    raw_totals, outcomes = Counter({k: 0 for k in RAW}), Counter()
    overlaps = 0
    final_steps = []
    for attempt in attempts:
        require(attempt["attempt_index"] == 0 and attempt["status"] == "terminal", "first terminal attempt")
        terminal = attempt["terminal"]
        require(set(terminal) == {"after_step", "time_s", "outcome", "raw_flags", "overlap", "state_timing"}, "terminal flags only")
        last = terminal["after_step"]
        require(type(last) is int and 0 <= last < case["steps_executed"], "terminal step")
        final_steps.append(last)
        require(terminal["time_s"] == (last + 1) * .02
                and terminal["state_timing"] == "flags-only-after-step; no-auto-reset-state", "terminal timing")
        flags = terminal["raw_flags"]
        require(set(flags) == set(RAW) and all(type(v) is bool for v in flags.values()), "raw flags")
        expected = ("hard_failure" if flags["fall"] or flags["nan"] else
                    "collision" if flags["collision"] else "timeout" if flags["timeout"] else
                    "pass" if flags["pass"] else "other_terminal")
        require(terminal["outcome"] == expected, "terminal priority")
        overlap = sum(flags.values()) > 1
        require(terminal["overlap"] is overlap, "overlap flag")
        overlaps += overlap
        raw_totals.update({k: int(v) for k, v in flags.items()})
        outcomes[expected] += 1
        frames = attempt["frames"]
        expected_steps = sorted(set(range(0, last + 1, 5)) | {last})
        require([f["step"] for f in frames] == expected_steps, "ordered complete frame coverage")
        require(len(frames) <= 141, "frame ceiling")
        for frame in frames:
            require(set(frame) == {"step", "time_s", "state_timing", "nonfinite_fields", *FIELDS}, "frame schema")
            require(frame["time_s"] == frame["step"] * .02
                    and frame["state_timing"] == "pre-step", "frame timing")
            require(frame["nonfinite_fields"] == [] and all(
                type(frame[k]) in (int, float) and math.isfinite(frame[k]) for k in FIELDS), "finite frame")
            require(frame["phase"] in (0, 1, 2), "phase identity")
    require(max(final_steps) + 1 == case["steps_executed"], "stop after final first attempt")
    require(dict(raw_totals) == case["raw_terminal_events"], "raw outcome reconciliation")
    require(overlaps == case["terminal_overlap_events"], "overlap reconciliation")
    for counts in (case, report["totals"]):
        require(all(counts[key] == outcomes[outcome] for outcome, key in COUNT_FIELDS.items()), "outcome reconciliation")
        require(counts["fall_events"] == raw_totals["fall"]
                and counts["nan_termination_events"] == raw_totals["nan"]
                and counts["hard_failure_events"] == raw_totals["fall"] + raw_totals["nan"], "hard-failure reconciliation")
    return {"path": str(path), "sha256": sha256(path), "attempts": 4,
            "frames": sum(len(a["frames"]) for a in attempts), "outcomes": dict(outcomes)}


def evaluate_smoke(disabled_path: Path, enabled_path: Path) -> dict:
    result = {"protocol": PROTOCOL, "decision": "stop-recorder-diagnosis",
              "policy_acceptance": False, "training_data_admitted": False,
              "physical_motion_authorized": False, "errors": []}
    try:
        off, on = (json.loads(p.read_text()) for p in (disabled_path, enabled_path))
        result["report_sha256"] = {"disabled": sha256(disabled_path), "enabled": sha256(enabled_path)}
        _validate_identity(off)
        _validate_identity(on)
        require("first_attempt_recording_protocol" not in off
                and "first_attempt_recording" not in off["cases"][0], "disabled report changed")
        result["recording"] = validate_sidecar(on, enabled_path)
        stripped = copy.deepcopy(on)
        stripped.pop("first_attempt_recording_protocol")
        stripped["cases"][0].pop("first_attempt_recording")
        require(canonical(off) == canonical(stripped), "legacy reports differ; exact equality required")
        result["decision"] = "recorder-validated"
        case = on["cases"][0]
        result["controller_failures_observed"] = any(case[k] for k in (
            "collision_events", "attempt_timeout_events", "fall_events", "nan_termination_events", "other_terminal_events"))
        result["runtime_numerical_checks_passed"] = all(case[k] == 0 for k in (
            "fall_events", "nan_termination_events", "nonfinite_steps", "hard_failure_events",
            "other_terminal_events", "motor_speed_rated_exceed_fraction")) and case["motor_torque_utilization_p99"] <= .60
        result["may_predeclare_fresh_diagnostic"] = result["runtime_numerical_checks_passed"]
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result["decision"] = "stop-recorder-diagnosis"
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["may_predeclare_fresh_diagnostic"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disabled", type=Path)
    parser.add_argument("enabled", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_smoke(args.disabled, args.enabled)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["may_predeclare_fresh_diagnostic"] else 2)


if __name__ == "__main__":
    main()
