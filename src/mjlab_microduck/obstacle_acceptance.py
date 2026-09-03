"""Combine O1 obstacle, motor, and action-review evidence into one decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EVALUATION_SPEED_MPS = 0.5
MAX_TORQUE_UTILIZATION_P99 = 0.90
MAX_NEAR_STALL_FRACTION = 0.0025
MIN_MOTOR_SEEDS = 3


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _resolve(path_value: Any, root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _gate(
    name: str, status: str, observed: Any, criterion: str
) -> dict[str, Any]:
    if status not in {"pass", "fail", "unverified"}:
        raise ValueError(f"invalid gate status: {status}")
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "criterion": criterion,
    }


def _speed_cases(evaluation: dict[str, Any], label: str) -> dict[int, dict[str, Any]]:
    cases = evaluation.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label}.cases must be a list")
    selected = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"{label}.cases[{index}] must be an object")
        speed = _finite_number(
            case.get("commanded_speed_mps"),
            f"{label}.cases[{index}].commanded_speed_mps",
        )
        if not math.isclose(speed, EVALUATION_SPEED_MPS, abs_tol=1e-9):
            continue
        seed = case.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{label}.cases[{index}].seed must be non-negative")
        if seed in selected:
            raise ValueError(f"{label} repeats seed {seed} at 0.5 m/s")
        selected[seed] = case
    return selected


def _optional_metric(case: dict[str, Any], key: str, label: str) -> float | None:
    value = case.get(key)
    return None if value is None else _finite_number(value, f"{label}.{key}")


def _summarize_motor_evidence(
    candidate_path: Path,
    parent_cases: dict[int, dict[str, Any]],
    expected_checkpoint: str,
) -> dict[str, Any]:
    evaluation = _read_json(candidate_path)
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"checkpoint missing in {candidate_path}")
    if Path(checkpoint).name != Path(expected_checkpoint).name:
        raise ValueError(
            f"motor checkpoint mismatch: expected {expected_checkpoint}, got {checkpoint}"
        )
    cases = _speed_cases(evaluation, str(candidate_path))

    rated_exceed = []
    torque_p99 = []
    near_stall = []
    action_comparisons = []
    for seed, case in sorted(cases.items()):
        rated_exceed.append(
            _finite_number(
                case.get("motor_speed_rated_exceed_fraction"),
                f"candidate seed {seed}.motor_speed_rated_exceed_fraction",
            )
        )
        torque_p99.append(
            _finite_number(
                case.get("motor_torque_utilization_p99"),
                f"candidate seed {seed}.motor_torque_utilization_p99",
            )
        )
        near_stall.append(
            _finite_number(
                case.get("motor_torque_near_stall_fraction"),
                f"candidate seed {seed}.motor_torque_near_stall_fraction",
            )
        )
        parent = parent_cases.get(seed)
        candidate_abs = _optional_metric(case, "action_abs_p99", f"candidate seed {seed}")
        candidate_rate = _optional_metric(
            case, "action_rate_abs_p99", f"candidate seed {seed}"
        )
        parent_abs = (
            _optional_metric(parent, "action_abs_p99", f"parent seed {seed}")
            if parent is not None
            else None
        )
        parent_rate = (
            _optional_metric(parent, "action_rate_abs_p99", f"parent seed {seed}")
            if parent is not None
            else None
        )
        action_comparisons.append(
            {
                "seed": seed,
                "candidate_action_abs_p99": candidate_abs,
                "parent_action_abs_p99": parent_abs,
                "action_abs_ratio": (
                    candidate_abs / parent_abs
                    if candidate_abs is not None and parent_abs not in (None, 0.0)
                    else None
                ),
                "candidate_action_rate_abs_p99": candidate_rate,
                "parent_action_rate_abs_p99": parent_rate,
                "action_rate_ratio": (
                    candidate_rate / parent_rate
                    if candidate_rate is not None and parent_rate not in (None, 0.0)
                    else None
                ),
            }
        )

    action_metrics_complete = len(cases) >= MIN_MOTOR_SEEDS and all(
        comparison["action_abs_ratio"] is not None
        and comparison["action_rate_ratio"] is not None
        for comparison in action_comparisons
    )
    gates = [
        _gate(
            "motor_seed_count",
            "pass" if len(cases) >= MIN_MOTOR_SEEDS else "fail",
            len(cases),
            f">= {MIN_MOTOR_SEEDS} at {EVALUATION_SPEED_MPS} m/s",
        ),
        _gate(
            "rated_speed_exceedance",
            "pass" if rated_exceed and max(rated_exceed) == 0.0 else "fail",
            max(rated_exceed) if rated_exceed else None,
            "maximum across seeds == 0",
        ),
        _gate(
            "torque_utilization_p99",
            (
                "pass"
                if torque_p99 and max(torque_p99) <= MAX_TORQUE_UTILIZATION_P99
                else "fail"
            ),
            max(torque_p99) if torque_p99 else None,
            f"maximum across seeds <= {MAX_TORQUE_UTILIZATION_P99}",
        ),
        _gate(
            "near_stall_fraction",
            (
                "pass"
                if near_stall and max(near_stall) <= MAX_NEAR_STALL_FRACTION
                else "fail"
            ),
            max(near_stall) if near_stall else None,
            f"maximum across seeds <= {MAX_NEAR_STALL_FRACTION}",
        ),
        _gate(
            "action_comparison_metrics",
            "pass" if action_metrics_complete else "unverified",
            action_comparisons,
            "candidate and parent action magnitude/rate p99 for every motor seed",
        ),
    ]
    return {
        "motor_evaluation": str(candidate_path),
        "checkpoint": checkpoint,
        "seeds": sorted(cases),
        "action_comparisons": action_comparisons,
        "gates": gates,
    }


def _review_gate(review: Any) -> dict[str, Any]:
    if not isinstance(review, dict):
        return _gate(
            "action_regression_review",
            "unverified",
            None,
            "explicit pass with non-empty evidence after metric review",
        )
    status = review.get("status")
    evidence = review.get("evidence")
    if status not in {"pass", "fail"} or not isinstance(evidence, str) or not evidence:
        raise ValueError(
            "action_regression_review requires pass/fail status and non-empty evidence"
        )
    return _gate(
        "action_regression_review",
        status,
        evidence,
        "explicit pass with non-empty evidence after metric review",
    )


def summarize_obstacle_acceptance(
    manifest: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string")

    sweep_path = _resolve(
        manifest.get("obstacle_sweep"), manifest_dir, "obstacle_sweep"
    )
    sweep = _read_json(sweep_path)
    if sweep.get("stage") != "O1":
        raise ValueError("obstacle sweep must be for O1")
    selected_iteration = sweep.get("selected_checkpoint_iteration")
    if selected_iteration is not None and (
        isinstance(selected_iteration, bool)
        or not isinstance(selected_iteration, int)
        or selected_iteration < 0
    ):
        raise ValueError("selected checkpoint iteration must be non-negative or null")
    sweep_gates = sweep.get("campaign_gates")
    sweep_candidates = sweep.get("candidates")
    if not isinstance(sweep_gates, list) or not sweep_gates:
        raise ValueError("obstacle sweep must contain campaign gates")
    if not isinstance(sweep_candidates, list) or not sweep_candidates:
        raise ValueError("obstacle sweep must contain candidates")
    obstacle_pass = selected_iteration is not None and all(
        isinstance(gate, dict) and gate.get("status") == "pass"
        for gate in sweep_gates
    )
    gates = [
        _gate(
            "obstacle_campaign",
            "pass" if obstacle_pass else "fail",
            selected_iteration,
            "sweep selects an iteration and all obstacle campaign gates pass",
        )
    ]

    motor_summaries = []
    parent_path_value = manifest.get("parent_motor_evaluation")
    motor_entries = manifest.get("motor_evaluations", [])
    if not isinstance(motor_entries, list):
        raise ValueError("motor_evaluations must be a list")
    if obstacle_pass:
        parent_path = _resolve(
            parent_path_value, manifest_dir, "parent_motor_evaluation"
        )
        parent_cases = _speed_cases(_read_json(parent_path), str(parent_path))
        expected = {
            candidate["training_seed"]: candidate
            for candidate in sweep_candidates
            if candidate["checkpoint_iteration"] == selected_iteration
        }
        entries = {}
        for entry in motor_entries:
            if not isinstance(entry, dict):
                raise ValueError("each motor evaluation must be an object")
            seed = entry.get("training_seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("motor evaluation training_seed must be non-negative")
            if seed in entries:
                raise ValueError(f"duplicate motor evaluation for training seed {seed}")
            entries[seed] = entry
        unexpected_seeds = sorted(set(entries) - set(expected))
        if unexpected_seeds:
            raise ValueError(
                f"motor evaluations contain unexpected training seeds: {unexpected_seeds}"
            )
        for training_seed, candidate in sorted(expected.items()):
            entry = entries.get(training_seed)
            if entry is None:
                gates.append(
                    _gate(
                        f"motor_evidence_seed_{training_seed}",
                        "unverified",
                        None,
                        "retained motor evaluation for selected checkpoint",
                    )
                )
                continue
            if entry.get("checkpoint_iteration") != selected_iteration:
                raise ValueError(
                    f"motor iteration mismatch for training seed {training_seed}"
                )
            motor_path = _resolve(
                entry.get("motor_evaluation"),
                manifest_dir,
                f"motor_evaluation seed {training_seed}",
            )
            motor = _summarize_motor_evidence(
                motor_path, parent_cases, candidate["checkpoint"]
            )
            motor["training_seed"] = training_seed
            motor_summaries.append(motor)
            statuses = [gate["status"] for gate in motor["gates"]]
            status = (
                "fail"
                if "fail" in statuses
                else "unverified" if "unverified" in statuses else "pass"
            )
            gates.append(
                _gate(
                    f"motor_evidence_seed_{training_seed}",
                    status,
                    statuses,
                    "all motor gates pass with complete action metrics",
                )
            )
    else:
        gates.append(
            _gate(
                "motor_evidence",
                "unverified",
                None,
                "evaluated only after an obstacle checkpoint survives",
            )
        )

    gates.append(_review_gate(manifest.get("action_regression_review")))
    statuses = [gate["status"] for gate in gates]
    decision = (
        "rejected"
        if "fail" in statuses
        else "diagnostic-only" if "unverified" in statuses else "accepted"
    )
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "stage": "O1",
        "decision": decision,
        "evidence_tier": "simulation-only",
        "physical_motion_authorized": False,
        "selected_checkpoint_iteration": selected_iteration,
        "obstacle_sweep": str(sweep_path),
        "gates": gates,
        "motor_evidence": motor_summaries,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selected_checkpoint_iteration"]
    selected_label = f"`{selected}`" if selected is not None else "none"
    lines = [
        f"# O1 acceptance: {summary['campaign_id']}",
        "",
        f"Decision: **{summary['decision']}** ({summary['evidence_tier']})",
        "",
        f"Selected iteration: {selected_label}",
        "",
        "| Gate | Status | Criterion |",
        "| --- | --- | --- |",
    ]
    for gate in summary["gates"]:
        lines.append(f"| {gate['name']} | {gate['status']} | {gate['criterion']} |")
    lines.extend(
        [
            "",
            "Physical motion authorized: **no**.",
            "",
        ]
    )
    return "\n".join(lines)


def write_obstacle_acceptance(manifest_path: Path, output_dir: Path) -> Path:
    manifest_path = manifest_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    summary = summarize_obstacle_acceptance(
        _read_json(manifest_path), manifest_path.parent
    )
    output_dir.mkdir(parents=True)
    output_path = output_dir / "obstacle_acceptance.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "obstacle_acceptance.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2))
    print(f"obstacle_acceptance_retained={output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_obstacle_acceptance(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
