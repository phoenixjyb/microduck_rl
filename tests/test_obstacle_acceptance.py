"""Tests for combined O1 obstacle and motor acceptance evidence."""

import json
from pathlib import Path

import pytest

from mjlab_microduck.obstacle_acceptance import (
    render_markdown,
    summarize_obstacle_acceptance,
    write_obstacle_acceptance,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n")
    return path


def _sweep(path: Path, *, selected: int | None = 8032) -> Path:
    gates = [
        {"name": "training_seed_count", "status": "pass"},
        {"name": "common_iteration_obstacle_gates", "status": "pass"},
    ]
    if selected is None:
        gates[1]["status"] = "fail"
    return _write_json(
        path,
        {
            "stage": "O1",
            "selected_checkpoint_iteration": selected,
            "campaign_gates": gates,
            "candidates": [
                {
                    "training_seed": seed,
                    "checkpoint_iteration": 8032,
                    "checkpoint": f"/obstacle/seed-{seed}/model_8032.pt",
                }
                for seed in (42, 43, 44)
            ],
        },
    )


def _motor(path: Path, checkpoint: str, *, action_rate: bool = True) -> Path:
    cases = []
    for seed in (41, 42, 43):
        case = {
            "seed": seed,
            "commanded_speed_mps": 0.5,
            "motor_speed_rated_exceed_fraction": 0.0,
            "motor_torque_utilization_p99": 0.8,
            "motor_torque_near_stall_fraction": 0.001,
            "action_abs_p99": 0.9,
        }
        if action_rate:
            case["action_rate_abs_p99"] = 0.1
        cases.append(case)
    return _write_json(path, {"checkpoint": checkpoint, "cases": cases})


def _manifest(tmp_path: Path, *, review: bool = True) -> dict:
    _sweep(tmp_path / "sweep.json")
    _motor(tmp_path / "parent.json", "/parent/model_7998.pt")
    motor_evaluations = []
    for seed in (42, 43, 44):
        path = tmp_path / f"motor-{seed}.json"
        _motor(path, f"/candidate/seed-{seed}/model_8032.pt")
        motor_evaluations.append(
            {
                "training_seed": seed,
                "checkpoint_iteration": 8032,
                "motor_evaluation": path.name,
            }
        )
    manifest = {
        "schema_version": 1,
        "campaign_id": "o1-acceptance-test",
        "obstacle_sweep": "sweep.json",
        "parent_motor_evaluation": "parent.json",
        "motor_evaluations": motor_evaluations,
    }
    if review:
        manifest["action_regression_review"] = {
            "status": "pass",
            "evidence": "ratios reviewed against the retained parent",
        }
    return manifest


def test_acceptance_requires_all_evidence_and_explicit_action_review(tmp_path: Path):
    summary = summarize_obstacle_acceptance(_manifest(tmp_path), tmp_path)

    assert summary["decision"] == "accepted"
    assert summary["evidence_tier"] == "simulation-only"
    assert summary["physical_motion_authorized"] is False
    assert all(gate["status"] == "pass" for gate in summary["gates"])
    assert len(summary["motor_evidence"]) == 3


def test_missing_action_review_stays_diagnostic_only(tmp_path: Path):
    summary = summarize_obstacle_acceptance(
        _manifest(tmp_path, review=False), tmp_path
    )

    assert summary["decision"] == "diagnostic-only"
    assert summary["gates"][-1]["status"] == "unverified"


def test_failed_obstacle_sweep_is_rejected_without_motor_claims(tmp_path: Path):
    _sweep(tmp_path / "sweep.json", selected=None)
    manifest = {
        "schema_version": 1,
        "campaign_id": "failed-obstacle",
        "obstacle_sweep": "sweep.json",
    }

    summary = summarize_obstacle_acceptance(manifest, tmp_path)

    assert summary["decision"] == "rejected"
    assert summary["gates"][0]["status"] == "fail"
    assert summary["gates"][1]["status"] == "unverified"
    assert summary["motor_evidence"] == []


def test_missing_action_rate_metric_stays_unverified(tmp_path: Path):
    manifest = _manifest(tmp_path)
    _motor(
        tmp_path / "motor-42.json",
        "/candidate/seed-42/model_8032.pt",
        action_rate=False,
    )

    summary = summarize_obstacle_acceptance(manifest, tmp_path)

    assert summary["decision"] == "diagnostic-only"
    seed_42 = next(
        item for item in summary["motor_evidence"] if item["training_seed"] == 42
    )
    assert seed_42["gates"][-1]["status"] == "unverified"


def test_unexpected_motor_training_seed_is_rejected_as_manifest_error(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["motor_evaluations"].append(
        {
            "training_seed": 99,
            "checkpoint_iteration": 8032,
            "motor_evaluation": "motor-42.json",
        }
    )

    with pytest.raises(ValueError, match="unexpected training seeds"):
        summarize_obstacle_acceptance(manifest, tmp_path)


@pytest.mark.parametrize(
    "metric,value,gate_name",
    [
        ("motor_speed_rated_exceed_fraction", 0.001, "rated_speed_exceedance"),
        ("motor_torque_utilization_p99", 0.91, "torque_utilization_p99"),
        ("motor_torque_near_stall_fraction", 0.003, "near_stall_fraction"),
    ],
)
def test_motor_hard_gate_failure_rejects_candidate(
    tmp_path: Path, metric, value, gate_name
):
    manifest = _manifest(tmp_path)
    motor_path = tmp_path / "motor-42.json"
    motor = json.loads(motor_path.read_text())
    motor["cases"][0][metric] = value
    _write_json(motor_path, motor)

    summary = summarize_obstacle_acceptance(manifest, tmp_path)

    assert summary["decision"] == "rejected"
    seed_42 = next(
        item for item in summary["motor_evidence"] if item["training_seed"] == 42
    )
    gate = next(gate for gate in seed_42["gates"] if gate["name"] == gate_name)
    assert gate["status"] == "fail"


def test_write_acceptance_is_retained_and_renders_no_motion(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    output_dir = tmp_path / "acceptance"

    output_path = write_obstacle_acceptance(manifest_path, output_dir)

    summary = json.loads(output_path.read_text())
    markdown = (output_dir / "obstacle_acceptance.md").read_text()
    assert "Physical motion authorized: **no**" in markdown
    assert render_markdown(summary) == markdown
    with pytest.raises(FileExistsError):
        write_obstacle_acceptance(manifest_path, output_dir)
