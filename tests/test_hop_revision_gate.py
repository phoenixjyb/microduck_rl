"""Tests for the deterministic H1-S causal comparison gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mjlab_microduck.hop_evaluation import h1_decision
from mjlab_microduck.hop_revision_gate import (
    BASELINE_TASK,
    CANDIDATE_TASK,
    H1T_TASK,
    compare_h1s_to_h1p,
    compare_h1t_to_h1p,
)


def _case(**overrides):
    case = {
        "cycle_success_fraction": 0.97,
        "landing_fraction": 1.0,
        "episode_pass_fraction": 0.0,
        "cycle_rise_p50_m": 0.04,
        "fall_events": 23,
        "nan_termination_events": 0,
        "nonfinite_episodes": 0,
        "cycle_rise_peak_m": 0.08,
        "max_drift_p95_m": 0.59,
        "spring_bottomed_fraction": 0.004,
        "motor_speed_rated_exceed_fraction": 0.0002,
        "motor_torque_utilization_p99": 0.83,
        "motor_torque_near_stall_fraction": 0.0046,
    }
    case.update(overrides)
    return case


def _write_evaluation(path: Path, task: str, case: dict) -> Path:
    cases = [{**case, "seed": seed} for seed in (211, 223, 227)]
    decision = h1_decision(cases)
    path.write_text(
        json.dumps(
            {
                "protocol": "H1-periodic-hop-heldout-v1",
                "task": task,
                "checkpoint": "/retained/model_5999.pt",
                "checkpoint_sha256": "0" * 64,
                "seeds": [211, 223, 227],
                "num_envs": 128,
                "cycles": 6,
                "cases": cases,
                "decision": decision["decision"],
                "physical_motion_authorized": False,
            }
        )
        + "\n"
    )
    return path


def test_gate_advances_only_when_every_predeclared_comparison_passes(tmp_path):
    baseline = _write_evaluation(tmp_path / "baseline.json", BASELINE_TASK, _case())
    candidate = _write_evaluation(
        tmp_path / "candidate.json",
        CANDIDATE_TASK,
        _case(
            episode_pass_fraction=0.1,
            fall_events=22,
            max_drift_p95_m=0.58,
            cycle_success_fraction=0.98,
            spring_bottomed_fraction=0.003,
            motor_speed_rated_exceed_fraction=0.0001,
            motor_torque_utilization_p99=0.82,
            motor_torque_near_stall_fraction=0.004,
        ),
    )
    result = compare_h1s_to_h1p(baseline, candidate)
    assert result["decision"] == "advance_to_multi_seed"
    assert all(check["status"] == "pass" for check in result["comparisons"])
    assert result["physical_motion_authorized"] is False
    assert len(result["baseline"]["sha256"]) == 64


def test_gate_stops_on_a_single_regression(tmp_path):
    baseline = _write_evaluation(tmp_path / "baseline.json", BASELINE_TASK, _case())
    candidate = _write_evaluation(
        tmp_path / "candidate.json",
        CANDIDATE_TASK,
        _case(
            episode_pass_fraction=0.1,
            fall_events=22,
            max_drift_p95_m=0.58,
            cycle_success_fraction=0.96,
        ),
    )
    result = compare_h1s_to_h1p(baseline, candidate)
    failures = [c["name"] for c in result["comparisons"] if c["status"] == "fail"]
    assert result["decision"] == "stop"
    assert failures == ["cycle_success_does_not_regress"]


def test_gate_rejects_wrong_candidate_identity(tmp_path):
    baseline = _write_evaluation(tmp_path / "baseline.json", BASELINE_TASK, _case())
    candidate = _write_evaluation(tmp_path / "candidate.json", BASELINE_TASK, _case())
    with pytest.raises(ValueError, match="unexpected task"):
        compare_h1s_to_h1p(baseline, candidate)


def test_h1t_gate_advances_only_with_behavior_and_envelope_passes(tmp_path):
    baseline = _write_evaluation(tmp_path / "baseline.json", BASELINE_TASK, _case())
    candidate = _write_evaluation(
        tmp_path / "candidate.json",
        H1T_TASK,
        _case(
            cycle_success_fraction=0.95,
            episode_pass_fraction=0.1,
            fall_events=22,
            max_drift_p95_m=0.29,
            spring_bottomed_fraction=0.003,
            motor_speed_rated_exceed_fraction=0.0001,
            motor_torque_utilization_p99=0.82,
            motor_torque_near_stall_fraction=0.004,
        ),
    )
    result = compare_h1t_to_h1p(baseline, candidate)
    assert result["decision"] == "advance_to_multi_seed"
    assert all(check["status"] == "pass" for check in result["comparisons"])
    assert result["protocol"] == "H1-T-vs-H1-P-causal-gate-v1"


def test_h1t_gate_stops_when_drift_threshold_is_missed(tmp_path):
    baseline = _write_evaluation(tmp_path / "baseline.json", BASELINE_TASK, _case())
    candidate = _write_evaluation(
        tmp_path / "candidate.json",
        H1T_TASK,
        _case(
            cycle_success_fraction=0.95,
            episode_pass_fraction=0.1,
            fall_events=22,
            max_drift_p95_m=0.30,
            spring_bottomed_fraction=0.003,
            motor_speed_rated_exceed_fraction=0.0001,
            motor_torque_utilization_p99=0.82,
            motor_torque_near_stall_fraction=0.004,
        ),
    )
    result = compare_h1t_to_h1p(baseline, candidate)
    failures = [c["name"] for c in result["comparisons"] if c["status"] == "fail"]
    assert result["decision"] == "stop"
    assert failures == ["drift_below_0_30_m"]
