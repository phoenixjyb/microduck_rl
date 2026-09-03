"""Tests for deterministic O1 checkpoint sweep selection."""

import json
from pathlib import Path

import pytest

from mjlab_microduck.obstacle_checkpoint_sweep import (
    render_markdown,
    summarize_checkpoint_sweep,
    write_checkpoint_sweep,
)
from mjlab_microduck.obstacle_protocol import o1_evaluation_protocol


def _evaluation(
    path: Path,
    iteration: int,
    *,
    passes: int = 80,
    collisions: int = 20,
    pre_speed: float = 0.3,
    lateral: float | None = 0.4,
    passage: float | None = 4.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": f"/retained/model_{iteration}.pt",
                "evaluation_protocol": o1_evaluation_protocol(),
                "cases": [
                    {"seed": seed, "commanded_speed_mps": 0.5}
                    for seed in (41, 42, 43)
                ],
                "totals": {
                    "collision_events": collisions,
                    "fall_events": 0,
                    "nan_termination_events": 0,
                    "clean_pass_events": passes,
                    "nonfinite_steps": 0,
                },
                "pre_obstacle_route_speed_mps": pre_speed,
                "mean_pass_lateral_excursion_m": lateral,
                "mean_passage_time_s": passage,
            }
        )
        + "\n"
    )


def _manifest(tmp_path: Path, rates: dict[tuple[int, int], tuple[int, int]]) -> dict:
    candidates = []
    for (training_seed, iteration), (passes, collisions) in rates.items():
        evaluation = tmp_path / f"seed-{training_seed}-iter-{iteration}.json"
        _evaluation(evaluation, iteration, passes=passes, collisions=collisions)
        candidates.append(
            {
                "training_seed": training_seed,
                "checkpoint_iteration": iteration,
                "obstacle_evaluation": evaluation.name,
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": "test-o1",
        "stage": "O1",
        "candidates": candidates,
    }


def test_sweep_selects_earliest_common_passing_iteration(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        {
            (42, 8016): (80, 20),
            (43, 8016): (69, 31),
            (44, 8016): (80, 20),
            (42, 8032): (90, 10),
            (43, 8032): (90, 10),
            (44, 8032): (90, 10),
        },
    )
    summary = summarize_checkpoint_sweep(manifest, tmp_path)

    assert summary["selected_checkpoint_iteration"] == 8032
    assert summary["promotion_decision"] == "diagnostic-only"
    assert summary["campaign_gates"] == [
        {
            "name": "training_seed_count",
            "status": "pass",
            "observed": 3,
            "criterion": ">= 3",
        },
        {
            "name": "common_iteration_obstacle_gates",
            "status": "pass",
            "observed": 8032,
            "criterion": "earliest exact iteration passing every seed and pooled rate",
        },
    ]
    assert "motor and action-regression evidence" in summary["decision_reason"]


def test_sweep_does_not_select_incomplete_or_last_iteration(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        {
            (42, 8016): (80, 20),
            (43, 8016): (80, 20),
            (44, 8016): (80, 20),
            (42, 8032): (90, 10),
            (43, 8032): (90, 10),
        },
    )
    summary = summarize_checkpoint_sweep(manifest, tmp_path)

    assert summary["selected_checkpoint_iteration"] == 8016
    assert summary["iteration_summaries"][-1]["complete_training_seed_set"] is False


def test_sweep_requires_three_training_seeds_for_campaign_gate(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        {(42, 8032): (80, 20), (43, 8032): (80, 20)},
    )
    summary = summarize_checkpoint_sweep(manifest, tmp_path)

    assert summary["selected_checkpoint_iteration"] == 8032
    assert summary["campaign_gates"][0]["status"] == "fail"
    assert summary["decision_reason"] == "O1 obstacle campaign gates do not pass"


@pytest.mark.parametrize(
    "field,value,failed_gate",
    [
        ("passes", 46, "clean_pass_rate"),
        ("pre_speed", 0.24, "pre_obstacle_route_speed_mps"),
        ("lateral", 0.46, "mean_pass_lateral_excursion_m"),
        ("passage", 4.6, "mean_passage_time_s"),
    ],
)
def test_candidate_reports_failed_o1_gate(tmp_path: Path, field, value, failed_gate):
    evaluation = tmp_path / "candidate.json"
    kwargs = {field: value}
    _evaluation(evaluation, 8032, **kwargs)
    manifest = {
        "schema_version": 1,
        "campaign_id": "failed-gate",
        "stage": "O1",
        "candidates": [
            {
                "training_seed": 42,
                "checkpoint_iteration": 8032,
                "obstacle_evaluation": evaluation.name,
            }
        ],
    }

    candidate = summarize_checkpoint_sweep(manifest, tmp_path)["candidates"][0]
    assert failed_gate in candidate["failed_gates"]
    assert candidate["obstacle_gate_status"] == "fail"


def test_sweep_rejects_checkpoint_iteration_mismatch(tmp_path: Path):
    manifest = _manifest(tmp_path, {(42, 8032): (80, 20)})
    manifest["candidates"][0]["checkpoint_iteration"] = 8048

    with pytest.raises(ValueError, match="iteration mismatch"):
        summarize_checkpoint_sweep(manifest, tmp_path)


def test_legacy_evaluation_without_o1_protocol_is_not_eligible(tmp_path: Path):
    manifest = _manifest(tmp_path, {(42, 8032): (80, 20)})
    evaluation_path = tmp_path / "seed-42-iter-8032.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation.pop("evaluation_protocol")
    evaluation_path.write_text(json.dumps(evaluation) + "\n")

    candidate = summarize_checkpoint_sweep(manifest, tmp_path)["candidates"][0]
    assert candidate["failed_gates"][0] == "evaluation_protocol"


def test_zero_pass_candidate_is_reported_as_failed_not_invalid(tmp_path: Path):
    evaluation = tmp_path / "zero-pass.json"
    _evaluation(
        evaluation,
        8032,
        passes=0,
        collisions=100,
        lateral=None,
        passage=None,
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": "zero-pass",
        "stage": "O1",
        "candidates": [
            {
                "training_seed": 42,
                "checkpoint_iteration": 8032,
                "obstacle_evaluation": evaluation.name,
            }
        ],
    }

    summary = summarize_checkpoint_sweep(manifest, tmp_path)
    candidate = summary["candidates"][0]
    assert candidate["failed_gates"] == [
        "clean_pass_rate",
        "mean_pass_lateral_excursion_m",
        "mean_passage_time_s",
    ]
    assert "| 42 | 8032 | 0.000 | 0.300 | n/a | n/a | fail |" in render_markdown(
        summary
    )


def test_write_sweep_is_retained_and_marked_diagnostic_only(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        {
            (42, 8032): (80, 20),
            (43, 8032): (80, 20),
            (44, 8032): (80, 20),
        },
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    output_dir = tmp_path / "report"

    output_path = write_checkpoint_sweep(manifest_path, output_dir)

    summary = json.loads(output_path.read_text())
    markdown = (output_dir / "obstacle_checkpoint_sweep.md").read_text()
    assert summary["promotion_decision"] == "diagnostic-only"
    assert "Selected obstacle-gate iteration: `8032`" in markdown
    assert render_markdown(summary) == markdown
    with pytest.raises(FileExistsError):
        write_checkpoint_sweep(manifest_path, output_dir)
