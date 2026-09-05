"""Tests for the deterministic HC4-U1 fixed-attempt pre-screen."""

from __future__ import annotations

import json
from pathlib import Path

from mjlab_microduck.hc4u1_gate import (
    ACTOR_SHA256,
    CANDIDATE_SHA256,
    CANDIDATE_STAGE,
    FAR_SHA256,
    FAR_STAGE,
    NEAR_SHA256,
    NEAR_STAGE,
    PROTOCOL,
    compare_hc4u1_prescreen,
)


def _case(speed: float, forward: float, lateral: float, **overrides):
    case = {
        "seed": 193,
        "num_envs": 64,
        "nominal_speed_mps": speed,
        "obstacle_forward_m": forward,
        "obstacle_lateral_m": lateral,
        "evaluation_window": PROTOCOL,
        "expected_attempts": 64,
        "completed_attempts": 64,
        "resolved_attempts": 64,
        "unresolved_attempts": 0,
        "hard_failure_events": 0,
        "other_terminal_events": 0,
        "fall_events": 0,
        "nan_termination_events": 0,
        "nonfinite_steps": 0,
        "motor_speed_rated_exceed_fraction": 0.0,
        "clean_pass_events": 64,
        "collision_events": 0,
        "attempt_timeout_events": 0,
        "approach_route_speed_mps": 0.20,
        "approach_samples": 100,
        "recovery_route_speed_mps": 0.24,
        "recovery_samples": 100,
        "motor_torque_utilization_p99": 0.55,
    }
    case.update(overrides)
    return case


def _write_report(
    path: Path,
    *,
    stage: str,
    supervisor_sha256: str,
    forward_positions: tuple[float, ...],
    overrides: dict[tuple[float, float, float], dict] | None = None,
) -> Path:
    overrides = overrides or {}
    cases = [
        _case(speed, forward, lateral, **overrides.get((speed, forward, lateral), {}))
        for speed in (0.30, 0.40)
        for forward in forward_positions
        for lateral in (-0.08, 0.00, 0.08)
    ]
    path.write_text(
        json.dumps(
            {
                "evaluation_window": PROTOCOL,
                "stage": stage,
                "checkpoint_sha256": ACTOR_SHA256,
                "supervisor_checkpoint_sha256": supervisor_sha256,
                "physical_motion_authorized": False,
                "perception": "exact structured geometry; no raw camera perception",
                "cases": cases,
            }
        )
        + "\n"
    )
    return path


def _reports(tmp_path: Path, *, candidate_overrides=None):
    candidate = _write_report(
        tmp_path / "candidate.json",
        stage=CANDIDATE_STAGE,
        supervisor_sha256=CANDIDATE_SHA256,
        forward_positions=(0.90, 1.15),
        overrides=candidate_overrides,
    )
    near = _write_report(
        tmp_path / "near.json",
        stage=NEAR_STAGE,
        supervisor_sha256=NEAR_SHA256,
        forward_positions=(0.90,),
    )
    far = _write_report(
        tmp_path / "far.json",
        stage=FAR_STAGE,
        supervisor_sha256=FAR_SHA256,
        forward_positions=(1.15,),
    )
    return candidate, near, far


def test_gate_continues_only_when_every_predeclared_check_passes(tmp_path):
    result = compare_hc4u1_prescreen(*_reports(tmp_path))
    assert result["decision"] == "continue_fresh_seeds"
    assert all(check["status"] == "pass" for check in result["checks"])
    assert result["physical_motion_authorized"] is False


def test_gate_stops_on_one_per_cell_timeout_regression(tmp_path):
    reports = _reports(
        tmp_path,
        candidate_overrides={
            (0.30, 1.15, 0.00): {
                "clean_pass_events": 63,
                "attempt_timeout_events": 1,
            }
        },
    )
    result = compare_hc4u1_prescreen(*reports)
    failures = [check["name"] for check in result["checks"] if check["status"] == "fail"]
    assert result["decision"] == "stop"
    assert failures == [
        "per_cell_timeout_non_regression",
        "per_cell_clean_non_regression",
        "aggregate_outcome_non_regression",
    ]


def test_gate_stops_on_phase_speed_or_motor_regression(tmp_path):
    reports = _reports(
        tmp_path,
        candidate_overrides={
            (0.40, 0.90, 0.08): {
                "recovery_route_speed_mps": 0.20,
                "motor_torque_utilization_p99": 0.61,
            }
        },
    )
    result = compare_hc4u1_prescreen(*reports)
    failures = [check["name"] for check in result["checks"] if check["status"] == "fail"]
    assert failures == [
        "per_cell_recovery_speed_non_regression",
        "motor_torque_p99_at_most_0_60",
    ]
