"""Hash-bound executed F1 evidence; no policy admission or GPU work."""

import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck.foundation_evaluation import candidate_failures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "f95e8cf16c3606f2c87cdc6bfb1d827ca2c92a62"
DECISION_SHA = "dac6638cb585efc890e2a585a18b2b766a569cc7d86d346f2d8886dce16da284"
REPORT_HASHES = {
    "parent-s397.json": "4596e960c1e37dbd743f0d23509d65aa60ac4c87e5e16d7acc74a22b4b94df47",
    "parent-s401.json": "4d4806b767dd7c6cf603024928772a5052fd9066bbb43faf28ae8ef7b461f551",
    "parent-s409.json": "7348bf084e6c7da73434afae4baa5c368225c911cff6f412ee90dea83e661f32",
    "candidate-s397.json": "a2da30370702ed27c9ab83d6eecadf0b57cdee86cf55de964d70612cc6993f89",
}


def directory(kind):
    remote = ROOT / ("artifacts/evaluations/f1-fixed030-v1" if kind == "evaluation"
                     else "artifacts/training/f1-fixed030-v1-pilot")
    local = ROOT / "artifacts/diagnostics/f1-fixed030-v1" / (
        "evaluation" if kind == "evaluation" else "f1-fixed030-v1-pilot")
    for path in (remote, local):
        if path.is_dir(): return path
    pytest.skip("separately retained F1 artifacts unavailable")


def read_bound(path, digest):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def test_exact_failed_decision_and_first_failed_candidate_stop():
    p = directory("evaluation")
    decision = read_bound(p / "decision.json", DECISION_SHA)
    assert decision["source"] == SOURCE
    assert decision["decision"] == "numerical-gate-stop"
    assert [(x["arm"],x["seed"]) for x in decision["reports"]] == [
        ("parent",397), ("parent",401), ("parent",409), ("candidate",397)]
    reports = {name: read_bound(p/name,digest) for name,digest in REPORT_HASHES.items()}
    for row in decision["reports"]:
        assert row["sha256"] == REPORT_HASHES[Path(row["path"]).name]
    assert not (p/"candidate-s401.json").exists()
    assert not (p/"candidate-s409.json").exists()
    assert candidate_failures(reports["candidate-s397.json"], reports["parent-s397.json"]) == decision["failures"]
    assert decision["failures"] == ["straight-body-mean-outside-band", "heading-drift", "cross-route-motion"]
    assert not any(decision[k] for k in ("policy_acceptance", "f2_admitted", "obstacles_admitted",
                                        "physical_motion_authorized"))


def test_improved_paired_speed_is_not_target_tracking_or_lower_motor_energy():
    p = directory("evaluation")
    parent = read_bound(p/"parent-s397.json",REPORT_HASHES["parent-s397.json"])
    candidate = read_bound(p/"candidate-s397.json",REPORT_HASHES["candidate-s397.json"])
    a,b = parent["groups"]["settled"],candidate["groups"]["settled"]
    assert candidate["safety_failures"] == [] and candidate["terminal_steps"] == []
    assert b["body_forward_mean"] > a["body_forward_mean"]
    assert b["route_forward_mean"] > a["route_forward_mean"]
    assert all(x < .27 for x in b["body_forward_per_env_mean"])
    assert all(x < .27 for x in b["route_forward_per_env_mean"])
    assert candidate["stable_route_response"]["counts"]["window-missed"] == 8
    assert b["legacy_torque_p99"] <= .60
    assert b["pre_reset_squared_utilization_mean"] > a["pre_reset_squared_utilization_mean"]
    assert b["pre_reset_mechanical_abs_power_mean_w"] > a["pre_reset_mechanical_abs_power_mean_w"]
    assert sum(x > .25 for x in b["heading_abs_per_env_max"]) == 4
    assert all(x > .05 for x in b["cross_route_abs_per_env_mean"])


def test_training_budget_and_exploration_are_not_deterministic_safety_evidence():
    p = directory("training")
    result = read_bound(p/"result.json","4e741dd3b9d28b54257fc8afba559151262107fa23bae1d4ac91a1d44439e44f")
    assert result["source"] == SOURCE and result["updates"] == 500
    assert result["status"] == "training-complete-not-accepted" and not result["policy_acceptance"]
    assert result["common_step"] == 204000 and result["parent_step"] == 192000
    assert result["final_sha256"] == "f39a6aeb8547fc0b5a26502d179134efbd0a352a45dfc48f774192f03667f23e"
    raw = (p/"rollout-metrics.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "0ce575d17f8e397e3125580f6b6d030883300fa373f42c45d8174e1fe8343d2e"
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 500
    assert all(r["update"] == i+1 and r["common_step"] == 192000+24*(i+1) for i,r in enumerate(rows))
    assert all(not r["failures"] for r in rows)
    assert sum(r["fall_fraction"] > 0 for r in rows) == 9
    assert max(r["rated_speed_exceed_fraction"] for r in rows) > 0
    assert max(r["pre_reset_torque_p99"] for r in rows) > .60
