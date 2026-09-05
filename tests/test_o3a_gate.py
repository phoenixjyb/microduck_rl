"""Tests for the deterministic O3a compact-range-noise pre-screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mjlab_microduck.o3a_gate import (
    ACTOR_SHA256,
    ATTEMPT_PROTOCOL,
    NOISE_PROTOCOL,
    ROLLOUT_STAGE,
    SUPERVISOR_SHA256,
    compare_o3a_prescreen,
)


def _case(lateral: float, *, noisy: bool, **overrides):
    case = {
        "seed": 271,
        "num_envs": 64,
        "nominal_speed_mps": 0.50,
        "obstacle_forward_m": 1.15,
        "obstacle_lateral_m": lateral,
        "evaluation_window": ATTEMPT_PROTOCOL,
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
        "approach_route_speed_mps": 0.45,
        "approach_samples": 100,
        "recovery_route_speed_mps": 0.47,
        "recovery_samples": 100,
        "motor_torque_utilization_p99": 0.55,
        "obstacle_sensor_protocol": {
            "identity": NOISE_PROTOCOL if noisy else "exact-v1",
            "range_noise_bound_m": 0.02 if noisy else 0.0,
            "noise_seed": 3000282 if noisy else None,
            "perturbed_fields": ["range"] if noisy else [],
        },
    }
    case.update(overrides)
    return case


def _write_report(path: Path, *, noisy: bool, overrides=None) -> Path:
    overrides = overrides or {}
    sensor_model = {
        "range_noise_m": 0.02 if noisy else 0.0,
        "bearing_noise_rad": 0.0,
        "width_noise_m": 0.0,
        "height_noise_m": 0.0,
        "closing_rate_noise_mps": 0.0,
        "dropout_probability": 0.0,
    }
    path.write_text(
        json.dumps(
            {
                "evaluation_window": ATTEMPT_PROTOCOL,
                "stage": ROLLOUT_STAGE,
                "checkpoint_sha256": ACTOR_SHA256,
                "supervisor_checkpoint_sha256": SUPERVISOR_SHA256,
                "physical_motion_authorized": False,
                "perception": (
                    "compact structured geometry with bounded range noise; "
                    "no raw camera perception"
                    if noisy
                    else "exact structured geometry; no raw camera perception"
                ),
                "obstacle_sensor_model": sensor_model,
                "cases": [
                    _case(
                        lateral,
                        noisy=noisy,
                        **overrides.get(lateral, {}),
                    )
                    for lateral in (-0.08, 0.00, 0.08)
                ],
            }
        )
        + "\n"
    )
    return path


def _reports(tmp_path: Path, *, noisy_overrides=None):
    return (
        _write_report(tmp_path / "baseline.json", noisy=False),
        _write_report(
            tmp_path / "noisy.json", noisy=True, overrides=noisy_overrides
        ),
    )


def test_gate_continues_when_every_frozen_check_passes(tmp_path):
    result = compare_o3a_prescreen(*_reports(tmp_path))
    assert result["decision"] == "continue_hc4r2_predeclaration"
    assert all(check["status"] == "pass" for check in result["checks"])
    assert result["physical_motion_authorized"] is False


def test_gate_allows_three_clean_passes_of_loss_per_cell(tmp_path):
    result = compare_o3a_prescreen(
        *_reports(
            tmp_path,
            noisy_overrides={
                -0.08: {"clean_pass_events": 61, "attempt_timeout_events": 3},
                0.00: {"clean_pass_events": 61, "attempt_timeout_events": 3},
                0.08: {"clean_pass_events": 61, "attempt_timeout_events": 3},
            },
        )
    )
    assert result["decision"] == "continue_hc4r2_predeclaration"
    assert result["pooled_clean_rate_delta"] == -9 / 192


def test_gate_stops_on_collision_or_excess_clean_loss(tmp_path):
    result = compare_o3a_prescreen(
        *_reports(
            tmp_path,
            noisy_overrides={
                0.00: {
                    "clean_pass_events": 60,
                    "collision_events": 1,
                    "attempt_timeout_events": 3,
                }
            },
        )
    )
    failures = [
        check["name"] for check in result["checks"] if check["status"] == "fail"
    ]
    assert result["decision"] == "stop"
    assert failures == [
        "collision_non_regression",
        "per_cell_clean_loss_at_most_3_of_64",
    ]


def test_gate_stops_on_phase_speed_or_motor_regression(tmp_path):
    result = compare_o3a_prescreen(
        *_reports(
            tmp_path,
            noisy_overrides={
                0.08: {
                    "recovery_route_speed_mps": 0.43,
                    "motor_torque_utilization_p99": 0.61,
                }
            },
        )
    )
    failures = [
        check["name"] for check in result["checks"] if check["status"] == "fail"
    ]
    assert result["decision"] == "stop"
    assert failures == [
        "per_cell_approach_recovery_delta_at_least_minus_0_03_mps",
        "pooled_approach_recovery_delta_at_least_minus_0_01_mps",
        "motor_torque_p99_at_most_0_60",
    ]


def test_gate_rejects_sensor_axis_expansion(tmp_path):
    baseline, noisy = _reports(tmp_path)
    payload = json.loads(noisy.read_text())
    payload["obstacle_sensor_model"]["bearing_noise_rad"] = 0.01
    noisy.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="bearing_noise_rad"):
        compare_o3a_prescreen(baseline, noisy)
