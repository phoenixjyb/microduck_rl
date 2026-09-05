"""Tests for the O3a two-specialist campaign aggregator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck.o3a_campaign_gate import (
    EXPECTED_PROTOCOLS,
    compare_o3a_campaign,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(path: Path, protocol: str, decision: str) -> Path:
    baseline = path.with_name(f"{path.stem}-baseline.json")
    noisy = path.with_name(f"{path.stem}-noisy.json")
    baseline.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "approach_route_speed_mps": 0.30,
                        "approach_samples": 100,
                        "recovery_route_speed_mps": 0.35,
                        "recovery_samples": 80,
                    }
                ]
            }
        )
        + "\n"
    )
    noisy.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "approach_route_speed_mps": 0.299,
                        "approach_samples": 100,
                        "recovery_route_speed_mps": 0.351,
                        "recovery_samples": 80,
                    }
                ]
            }
        )
        + "\n"
    )
    path.write_text(
        json.dumps(
            {
                "protocol": protocol,
                "decision": decision,
                "physical_motion_authorized": False,
                "checks": [{"name": "frozen", "status": "pass"}],
                "baseline": {"path": str(baseline), "sha256": _sha256(baseline)},
                "noisy": {"path": str(noisy), "sha256": _sha256(noisy)},
                "baseline_totals": {
                    "clean_pass_events": 10,
                    "collision_events": 1,
                    "attempt_timeout_events": 0,
                },
                "noisy_totals": {
                    "clean_pass_events": 11,
                    "collision_events": 0,
                    "attempt_timeout_events": 0,
                },
                "pooled_clean_rate_delta": 0.01,
                "pooled_phase_speed_deltas_mps": {
                    "approach": -0.001,
                    "recovery": 0.001,
                },
                "noisy_max_torque_utilization_p99": 0.57,
            }
        )
        + "\n"
    )
    return path


def _campaign_decisions(tmp_path: Path) -> tuple[Path, ...]:
    paths = []
    for index, (protocol, (_, _, decision)) in enumerate(
        EXPECTED_PROTOCOLS.items()
    ):
        paths.append(_decision(tmp_path / f"decision-{index}.json", protocol, decision))
    return tuple(paths)


def test_campaign_requires_and_retains_all_six_local_passes(tmp_path):
    result = compare_o3a_campaign(
        _campaign_decisions(tmp_path), "a" * 40
    )
    assert result["decision"] == "simulation_pass_pending_measured_sensor"
    assert len(result["seed_decisions"]) == 6
    assert result["specialist_summaries"]["hc4lh"]["physics_seeds"] == [
        271,
        281,
        283,
    ]
    assert result["specialist_summaries"]["hc4lh"][
        "pooled_phase_speed_deltas_mps"
    ] == pytest.approx({"approach": -0.001, "recovery": 0.001})
    assert result["physical_motion_authorized"] is False


def test_campaign_rejects_a_local_stop(tmp_path):
    paths = list(_campaign_decisions(tmp_path))
    payload = json.loads(paths[-1].read_text())
    payload["decision"] = "stop"
    paths[-1].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="did not pass"):
        compare_o3a_campaign(tuple(paths), "b" * 40)


def test_campaign_rejects_report_hash_drift(tmp_path):
    paths = _campaign_decisions(tmp_path)
    payload = json.loads(paths[0].read_text())
    Path(payload["noisy"]["path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_o3a_campaign(paths, "c" * 40)


def test_campaign_rejects_incomplete_or_invalid_source_identity(tmp_path):
    paths = _campaign_decisions(tmp_path)
    with pytest.raises(ValueError, match="source_commit"):
        compare_o3a_campaign(paths, "short")
    with pytest.raises(ValueError, match="incomplete"):
        compare_o3a_campaign(paths[:-1], "d" * 40)
