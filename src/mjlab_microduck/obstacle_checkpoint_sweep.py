"""Deterministic obstacle-stage checkpoint comparison from retained JSON."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from mjlab_microduck.obstacle_protocol import (
    OA0R_TASK_ID,
    oa0_training_protocol,
    obstacle_protocol_for_task,
    o1_evaluation_protocol,
)


MIN_TRAINING_SEEDS = 3
MIN_HELD_OUT_SEEDS = 3
MIN_TRAINING_SEED_PASS_RATE = 0.70
MIN_POOLED_PASS_RATE = 0.75
MIN_PRE_OBSTACLE_SPEED_MPS = 0.25
MAX_PASS_LATERAL_EXCURSION_M = 0.45
MAX_PASSAGE_TIME_S = 4.5

STAGE_PROFILES = {
    "O1": {
        "protocol": o1_evaluation_protocol,
        "commanded_speed_mps": 0.5,
        "min_training_seed_pass_rate": MIN_TRAINING_SEED_PASS_RATE,
        "min_pooled_pass_rate": MIN_POOLED_PASS_RATE,
        "min_pre_obstacle_speed_mps": MIN_PRE_OBSTACLE_SPEED_MPS,
        "max_pass_lateral_excursion_m": MAX_PASS_LATERAL_EXCURSION_M,
        "max_passage_time_s": MAX_PASSAGE_TIME_S,
        "max_success_route_return_error_m": None,
    },
    "OA0": {
        "protocol": oa0_training_protocol,
        "commanded_speed_mps": 0.3,
        "min_training_seed_pass_rate": 0.85,
        "min_pooled_pass_rate": 0.85,
        "min_pre_obstacle_speed_mps": 0.22,
        "max_pass_lateral_excursion_m": 0.45,
        "max_passage_time_s": 7.0,
        "max_success_route_return_error_m": 0.15,
    },
    "OA0R": {
        "protocol": lambda: obstacle_protocol_for_task(OA0R_TASK_ID),
        "commanded_speed_mps": 0.3,
        "min_training_seed_pass_rate": 0.85,
        "min_pooled_pass_rate": 0.85,
        "min_pre_obstacle_speed_mps": 0.22,
        "max_pass_lateral_excursion_m": 0.45,
        "max_passage_time_s": 7.0,
        "max_success_route_return_error_m": 0.15,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _gate(name: str, passed: bool, observed: Any, criterion: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "criterion": criterion,
    }


def _summarize_candidate(
    entry: dict[str, Any], manifest_dir: Path, stage: str
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("each candidate must be an object")
    required = {"training_seed", "checkpoint_iteration", "obstacle_evaluation"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"candidate missing fields: {sorted(missing)}")

    training_seed = _count(entry["training_seed"], "training_seed")
    iteration = _count(entry["checkpoint_iteration"], "checkpoint_iteration")
    evaluation_value = entry["obstacle_evaluation"]
    if not isinstance(evaluation_value, str) or not evaluation_value:
        raise ValueError("obstacle_evaluation must be a non-empty string")
    evaluation_path = Path(evaluation_value)
    if not evaluation_path.is_absolute():
        evaluation_path = manifest_dir / evaluation_path
    evaluation_path = evaluation_path.resolve(strict=True)
    evaluation = _read_json(evaluation_path)

    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"checkpoint must be a non-empty string in {evaluation_path}")
    expected_name = f"model_{iteration}.pt"
    if Path(checkpoint).name != expected_name:
        raise ValueError(
            f"checkpoint iteration mismatch: expected {expected_name}, got {checkpoint}"
        )

    totals = evaluation.get("totals")
    cases = evaluation.get("cases")
    if not isinstance(totals, dict) or not isinstance(cases, list) or not cases:
        raise ValueError(
            f"evaluation must contain non-empty cases and totals: {evaluation_path}"
        )

    profile = STAGE_PROFILES[stage]
    collision_events = _count(totals.get("collision_events"), "collision_events")
    clean_pass_events = _count(totals.get("clean_pass_events"), "clean_pass_events")
    attempt_timeout_events = _count(
        totals.get("attempt_timeout_events", 0), "attempt_timeout_events"
    )
    resolved_attempts = (
        collision_events + clean_pass_events + attempt_timeout_events
    )
    clean_pass_rate = (
        clean_pass_events / resolved_attempts if resolved_attempts else None
    )

    event_counts = {
        name: _count(totals.get(name), name)
        for name in (
            "fall_events",
            "nan_termination_events",
            "nonfinite_steps",
        )
    }
    held_out_seeds = []
    commanded_speeds = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object in {evaluation_path}")
        held_out_seeds.append(_count(case.get("seed"), f"cases[{index}].seed"))
        commanded_speeds.append(
            _number(
                case.get("commanded_speed_mps"),
                f"cases[{index}].commanded_speed_mps",
            )
        )
    if len(set(held_out_seeds)) != len(held_out_seeds):
        raise ValueError(f"held-out seeds must be unique in {evaluation_path}")

    pre_speed = _optional_number(
        evaluation.get("pre_obstacle_route_speed_mps"),
        "pre_obstacle_route_speed_mps",
    )
    pass_lateral = _optional_number(
        evaluation.get("mean_pass_lateral_excursion_m"),
        "mean_pass_lateral_excursion_m",
    )
    passage_time = _optional_number(
        evaluation.get("mean_passage_time_s"), "mean_passage_time_s"
    )
    return_error = _optional_number(
        evaluation.get("mean_success_route_return_error_m"),
        "mean_success_route_return_error_m",
    )
    protocol = evaluation.get("evaluation_protocol")
    expected_protocol = profile["protocol"]()

    gates = [
        _gate(
            "evaluation_protocol",
            protocol == expected_protocol,
            protocol,
            f"exactly matches {expected_protocol['name']}",
        ),
        _gate(
            "held_out_seed_count",
            len(held_out_seeds) >= MIN_HELD_OUT_SEEDS,
            len(held_out_seeds),
            f">= {MIN_HELD_OUT_SEEDS}",
        ),
        _gate(
            "commanded_speed",
            all(
                math.isclose(
                    speed, profile["commanded_speed_mps"], abs_tol=1e-9
                )
                for speed in commanded_speeds
            ),
            sorted(set(commanded_speeds)),
            f"all cases == {profile['commanded_speed_mps']} m/s",
        ),
        _gate(
            "resolved_attempts",
            resolved_attempts > 0,
            resolved_attempts,
            "> 0",
        ),
        _gate(
            "clean_pass_rate",
            clean_pass_rate is not None
            and clean_pass_rate >= profile["min_training_seed_pass_rate"],
            clean_pass_rate,
            f">= {profile['min_training_seed_pass_rate']}",
        ),
        _gate(
            "fall_events",
            event_counts["fall_events"] == 0,
            event_counts["fall_events"],
            "== 0",
        ),
        _gate(
            "nan_termination_events",
            event_counts["nan_termination_events"] == 0,
            event_counts["nan_termination_events"],
            "== 0",
        ),
        _gate(
            "nonfinite_steps",
            event_counts["nonfinite_steps"] == 0,
            event_counts["nonfinite_steps"],
            "== 0",
        ),
        _gate(
            "pre_obstacle_route_speed_mps",
            pre_speed is not None
            and pre_speed >= profile["min_pre_obstacle_speed_mps"],
            pre_speed,
            f">= {profile['min_pre_obstacle_speed_mps']}",
        ),
        _gate(
            "mean_pass_lateral_excursion_m",
            pass_lateral is not None
            and pass_lateral <= profile["max_pass_lateral_excursion_m"],
            pass_lateral,
            f"<= {profile['max_pass_lateral_excursion_m']}",
        ),
        _gate(
            "mean_passage_time_s",
            passage_time is not None
            and passage_time <= profile["max_passage_time_s"],
            passage_time,
            f"<= {profile['max_passage_time_s']}",
        ),
    ]
    max_return_error = profile["max_success_route_return_error_m"]
    if max_return_error is not None:
        gates.append(
            _gate(
                "mean_success_route_return_error_m",
                return_error is not None and return_error <= max_return_error,
                return_error,
                f"<= {max_return_error}",
            )
        )
    failed_gates = [gate["name"] for gate in gates if gate["status"] == "fail"]
    return {
        "training_seed": training_seed,
        "checkpoint_iteration": iteration,
        "checkpoint": checkpoint,
        "obstacle_evaluation": str(evaluation_path),
        "held_out_seeds": sorted(held_out_seeds),
        "collision_events": collision_events,
        "attempt_timeout_events": attempt_timeout_events,
        "clean_pass_events": clean_pass_events,
        "resolved_attempts": resolved_attempts,
        "clean_pass_rate": clean_pass_rate,
        "pre_obstacle_route_speed_mps": pre_speed,
        "mean_pass_lateral_excursion_m": pass_lateral,
        "mean_passage_time_s": passage_time,
        "mean_success_route_return_error_m": return_error,
        "gates": gates,
        "obstacle_gate_status": "pass" if not failed_gates else "fail",
        "failed_gates": failed_gates,
    }


def summarize_checkpoint_sweep(
    manifest: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    """Compare candidate checkpoints and choose the earliest O1 gate survivor."""
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string")
    stage = manifest.get("stage")
    if stage not in STAGE_PROFILES:
        raise ValueError(f"stage must be one of {sorted(STAGE_PROFILES)}")
    entries = manifest.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ValueError("candidates must be a non-empty list")

    candidates = [
        _summarize_candidate(entry, manifest_dir, stage) for entry in entries
    ]
    candidates.sort(
        key=lambda item: (item["checkpoint_iteration"], item["training_seed"])
    )
    identities = [
        (candidate["training_seed"], candidate["checkpoint_iteration"])
        for candidate in candidates
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("training_seed/checkpoint_iteration pairs must be unique")

    training_seeds = sorted({candidate["training_seed"] for candidate in candidates})
    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_iteration[candidate["checkpoint_iteration"]].append(candidate)

    iteration_summaries = []
    selected_iteration = None
    for iteration, group in sorted(by_iteration.items()):
        group_seeds = {candidate["training_seed"] for candidate in group}
        complete_seed_set = group_seeds == set(training_seeds)
        obstacle_gates_pass = complete_seed_set and all(
            candidate["obstacle_gate_status"] == "pass" for candidate in group
        )
        collisions = sum(candidate["collision_events"] for candidate in group)
        attempt_timeouts = sum(
            candidate["attempt_timeout_events"] for candidate in group
        )
        passes = sum(candidate["clean_pass_events"] for candidate in group)
        resolved = collisions + passes + attempt_timeouts
        pooled_rate = passes / resolved if resolved else None
        pooled_gate_pass = (
            pooled_rate is not None
            and pooled_rate >= STAGE_PROFILES[stage]["min_pooled_pass_rate"]
        )
        passes_iteration = obstacle_gates_pass and pooled_gate_pass
        iteration_summaries.append(
            {
                "checkpoint_iteration": iteration,
                "training_seeds": sorted(group_seeds),
                "complete_training_seed_set": complete_seed_set,
                "all_candidate_obstacle_gates_pass": obstacle_gates_pass,
                "pooled_resolved_attempts": resolved,
                "pooled_clean_pass_rate": pooled_rate,
                "pooled_pass_rate_gate": "pass" if pooled_gate_pass else "fail",
                "iteration_obstacle_gate_status": (
                    "pass" if passes_iteration else "fail"
                ),
            }
        )
        if passes_iteration and selected_iteration is None:
            selected_iteration = iteration

    campaign_gates = [
        _gate(
            "training_seed_count",
            len(training_seeds) >= MIN_TRAINING_SEEDS,
            len(training_seeds),
            f">= {MIN_TRAINING_SEEDS}",
        ),
        _gate(
            "common_iteration_obstacle_gates",
            selected_iteration is not None,
            selected_iteration,
            "earliest exact iteration passing every seed and pooled rate",
        ),
    ]
    obstacle_campaign_pass = all(gate["status"] == "pass" for gate in campaign_gates)
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "stage": stage,
        "promotion_decision": "diagnostic-only",
        "decision_reason": (
            "obstacle gates pass; motor and action-regression evidence still required"
            if obstacle_campaign_pass
            else f"{stage} obstacle campaign gates do not pass"
        ),
        "selection_policy": "earliest exact common iteration; never final-by-default",
        "training_seeds": training_seeds,
        "selected_checkpoint_iteration": selected_iteration,
        "campaign_gates": campaign_gates,
        "iteration_summaries": iteration_summaries,
        "candidates": candidates,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selected_checkpoint_iteration"]
    selection_line = (
        f"Selected obstacle-gate iteration: `{selected}`."
        if selected is not None
        else "Selected obstacle-gate iteration: none."
    )
    lines = [
        f"# {summary['stage']} checkpoint sweep: {summary['campaign_id']}",
        "",
        f"Decision: **{summary['promotion_decision']}**",
        "",
        summary["decision_reason"] + ".",
        "",
        f"Selection policy: {summary['selection_policy']}.",
        "",
        selection_line,
        "" if selected is not None else "No iteration passed the obstacle gates.",
        "",
        "| Train seed | Iteration | Pass rate | Pre-speed | Excursion | Passage | Gates |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| {training_seed} | {checkpoint_iteration} | {rate} | "
            "{pre} | {lateral} | {passage} | {status} |".format(
                training_seed=candidate["training_seed"],
                checkpoint_iteration=candidate["checkpoint_iteration"],
                rate=_format_metric(candidate["clean_pass_rate"]),
                pre=_format_metric(candidate["pre_obstacle_route_speed_mps"]),
                lateral=_format_metric(
                    candidate["mean_pass_lateral_excursion_m"]
                ),
                passage=_format_metric(candidate["mean_passage_time_s"]),
                status=candidate["obstacle_gate_status"],
            )
        )
    lines.extend(
        [
            "",
            "This report evaluates obstacle gates only. It cannot promote a checkpoint",
            "until motor-envelope and action-regression evidence are combined separately.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_checkpoint_sweep(manifest_path: Path, output_dir: Path) -> Path:
    manifest_path = manifest_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    summary = summarize_checkpoint_sweep(
        _read_json(manifest_path), manifest_path.parent
    )
    output_dir.mkdir(parents=True)
    output_path = output_dir / "obstacle_checkpoint_sweep.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "obstacle_checkpoint_sweep.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2))
    print(f"obstacle_checkpoint_sweep_retained={output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_checkpoint_sweep(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
