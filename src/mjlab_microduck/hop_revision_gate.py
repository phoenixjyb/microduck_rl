"""Deterministic causal gates for bounded H1 hop revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mjlab_microduck.hop_evaluation import H1_PROTOCOL, h1_decision


BASELINE_TASK = "Mjlab-Hop-H1P-Flat-Sprung-K3900-MicroDuck"
CANDIDATE_TASK = "Mjlab-Hop-H1S-Flat-Sprung-K3900-MicroDuck"
H1T_TASK = "Mjlab-Hop-H1T-Flat-Sprung-K3900-MicroDuck"
HELDOUT_SEEDS = [211, 223, 227]
FINAL_ITERATION = 5999


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_evaluation(path: Path, *, expected_task: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    if value.get("protocol") != H1_PROTOCOL:
        raise ValueError(f"unexpected H1 protocol in {path}")
    if value.get("task") != expected_task:
        raise ValueError(f"unexpected task in {path}: {value.get('task')!r}")
    if value.get("seeds") != HELDOUT_SEEDS:
        raise ValueError(f"unexpected held-out seeds in {path}")
    if value.get("num_envs") != 128 or value.get("cycles") != 6:
        raise ValueError(f"unexpected held-out evaluation shape in {path}")
    if value.get("physical_motion_authorized") is not False:
        raise ValueError(f"evaluation does not retain the no-motion boundary: {path}")
    checkpoint = value.get("checkpoint")
    expected_checkpoint = f"model_{FINAL_ITERATION}.pt"
    if not isinstance(checkpoint, str) or Path(checkpoint).name != expected_checkpoint:
        raise ValueError(f"expected final checkpoint {expected_checkpoint} in {path}")
    checkpoint_sha256 = value.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
    ):
        raise ValueError(f"invalid checkpoint SHA-256 in {path}")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"H1 cases missing in {path}")
    decision = h1_decision(cases)
    if value.get("decision") != decision["decision"]:
        raise ValueError(f"stored H1 decision disagrees with cases in {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "gates": {gate["name"]: gate["observed"] for gate in decision["gates"]},
    }


def compare_h1s_to_h1p(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Apply the predeclared seed-67 H1-S causal gate."""
    baseline = _load_evaluation(baseline_path, expected_task=BASELINE_TASK)
    candidate = _load_evaluation(candidate_path, expected_task=CANDIDATE_TASK)
    checks = [
        ("episode_pass_fraction_improves", ">", "episode_pass_fraction"),
        ("falls_improve", "<", "falls"),
        ("drift_improves", "<", "drift"),
        ("cycle_success_does_not_regress", ">=", "cycle_success_fraction"),
        ("spring_bottoming_does_not_regress", "<=", "spring_bottoming"),
        ("rated_speed_does_not_regress", "<=", "rated_speed_exceedance"),
        ("torque_p99_does_not_regress", "<=", "torque_utilization_p99"),
        ("near_stall_does_not_regress", "<=", "near_stall_fraction"),
    ]
    comparisons = []
    for name, operator, metric in checks:
        baseline_value = float(baseline["gates"][metric])
        candidate_value = float(candidate["gates"][metric])
        passed = {
            ">": candidate_value > baseline_value,
            "<": candidate_value < baseline_value,
            ">=": candidate_value >= baseline_value,
            "<=": candidate_value <= baseline_value,
        }[operator]
        comparisons.append(
            {
                "name": name,
                "metric": metric,
                "operator": operator,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "status": "pass" if passed else "fail",
            }
        )
    accepted = all(item["status"] == "pass" for item in comparisons)
    return {
        "schema_version": 1,
        "protocol": "H1-S-vs-H1-P-causal-gate-v1",
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": comparisons,
        "decision": "advance_to_multi_seed" if accepted else "stop",
        "physical_motion_authorized": False,
    }


def compare_h1t_to_h1p(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Apply H1-T's predeclared absolute and H1-P non-regression gates."""
    baseline = _load_evaluation(baseline_path, expected_task=BASELINE_TASK)
    candidate = _load_evaluation(candidate_path, expected_task=H1T_TASK)
    specifications = [
        (
            "cycle_success_at_least_90_percent",
            "cycle_success_fraction",
            ">=",
            0.90,
            "absolute_threshold",
        ),
        (
            "episode_pass_above_zero",
            "episode_pass_fraction",
            ">",
            0.0,
            "absolute_threshold",
        ),
        ("falls_improve", "falls", "<", float(baseline["gates"]["falls"]), "h1p"),
        (
            "drift_below_0_30_m",
            "drift",
            "<",
            0.30,
            "absolute_threshold",
        ),
        (
            "spring_bottoming_does_not_regress",
            "spring_bottoming",
            "<=",
            float(baseline["gates"]["spring_bottoming"]),
            "h1p",
        ),
        (
            "rated_speed_does_not_regress",
            "rated_speed_exceedance",
            "<=",
            float(baseline["gates"]["rated_speed_exceedance"]),
            "h1p",
        ),
        (
            "torque_p99_does_not_regress",
            "torque_utilization_p99",
            "<=",
            float(baseline["gates"]["torque_utilization_p99"]),
            "h1p",
        ),
        (
            "near_stall_does_not_regress",
            "near_stall_fraction",
            "<=",
            float(baseline["gates"]["near_stall_fraction"]),
            "h1p",
        ),
    ]
    comparisons = []
    for name, metric, operator, reference, reference_kind in specifications:
        candidate_value = float(candidate["gates"][metric])
        passed = {
            ">": candidate_value > reference,
            "<": candidate_value < reference,
            ">=": candidate_value >= reference,
            "<=": candidate_value <= reference,
        }[operator]
        comparisons.append(
            {
                "name": name,
                "metric": metric,
                "operator": operator,
                "reference": reference,
                "reference_kind": reference_kind,
                "candidate": candidate_value,
                "status": "pass" if passed else "fail",
            }
        )
    accepted = all(item["status"] == "pass" for item in comparisons)
    return {
        "schema_version": 1,
        "protocol": "H1-T-vs-H1-P-causal-gate-v1",
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": comparisons,
        "decision": "advance_to_multi_seed" if accepted else "stop",
        "physical_motion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", choices=("h1s", "h1t"), default="h1s")
    args = parser.parse_args()
    comparator = compare_h1s_to_h1p if args.revision == "h1s" else compare_h1t_to_h1p
    result = comparator(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"h1_revision_gate_decision={result['decision']}")
    print(f"h1_revision_gate_retained={args.output}")


if __name__ == "__main__":
    main()
