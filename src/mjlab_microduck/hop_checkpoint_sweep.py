"""Select the earliest all-seed H1 checkpoint from retained evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mjlab_microduck.hop_evaluation import H1_PROTOCOL, h1_decision


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _evaluation_candidate(entry: dict, manifest_dir: Path) -> dict:
    required = {"training_seed", "checkpoint_iteration", "hop_evaluation"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"candidate is missing fields: {sorted(missing)}")
    training_seed = _nonnegative_int(entry["training_seed"], "training_seed")
    iteration = _nonnegative_int(
        entry["checkpoint_iteration"], "checkpoint_iteration"
    )
    evaluation_path = Path(entry["hop_evaluation"])
    if not evaluation_path.is_absolute():
        evaluation_path = manifest_dir / evaluation_path
    evaluation_path = evaluation_path.resolve(strict=True)
    evaluation = _read_object(evaluation_path)
    if evaluation.get("protocol") != H1_PROTOCOL:
        raise ValueError(f"unexpected H1 protocol in {evaluation_path}")
    if evaluation.get("seeds") != [211, 223, 227]:
        raise ValueError(f"unexpected held-out seeds in {evaluation_path}")
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str) or Path(checkpoint).name != f"model_{iteration}.pt":
        raise ValueError(
            f"checkpoint iteration mismatch in {evaluation_path}: {checkpoint!r}"
        )
    cases = evaluation.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"H1 cases missing in {evaluation_path}")
    recomputed = h1_decision(cases)
    if evaluation.get("decision") != recomputed["decision"]:
        raise ValueError(f"stored H1 decision disagrees with gates in {evaluation_path}")
    if evaluation.get("physical_motion_authorized") is not False:
        raise ValueError(f"evaluation does not retain the no-motion boundary: {evaluation_path}")

    return {
        "training_seed": training_seed,
        "checkpoint_iteration": iteration,
        "checkpoint": checkpoint,
        "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
        "hop_evaluation": str(evaluation_path),
        "decision": recomputed["decision"],
        "minimum_episode_pass_fraction": min(
            float(case["episode_pass_fraction"]) for case in cases
        ),
        "minimum_cycle_success_fraction": min(
            float(case["cycle_success_fraction"]) for case in cases
        ),
        "maximum_torque_utilization_p99": max(
            float(case["motor_torque_utilization_p99"]) for case in cases
        ),
        "failed_gates": [
            gate["name"] for gate in recomputed["gates"] if gate["status"] != "pass"
        ],
    }


def summarize_h1_checkpoint_sweep(manifest_path: Path) -> dict:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _read_object(manifest_path)
    expected_seeds = manifest.get("training_seeds")
    if expected_seeds != [47, 53, 59]:
        raise ValueError("training_seeds must be exactly [47, 53, 59]")
    entries = manifest.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ValueError("candidates must be a non-empty list")
    candidates = [
        _evaluation_candidate(entry, manifest_path.parent) for entry in entries
    ]
    keys = [
        (candidate["training_seed"], candidate["checkpoint_iteration"])
        for candidate in candidates
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("training seed/checkpoint iteration pairs must be unique")

    by_iteration: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_iteration.setdefault(candidate["checkpoint_iteration"], []).append(candidate)

    iterations = []
    selected_iteration = None
    selected_candidate = None
    for iteration, group in sorted(by_iteration.items()):
        present_seeds = sorted(candidate["training_seed"] for candidate in group)
        complete = present_seeds == expected_seeds
        accepted = complete and all(
            candidate["decision"] == "accepted" for candidate in group
        )
        iterations.append(
            {
                "checkpoint_iteration": iteration,
                "training_seeds": present_seeds,
                "complete": complete,
                "all_training_seeds_accepted": accepted,
            }
        )
        if accepted and selected_iteration is None:
            selected_iteration = iteration
            selected_candidate = sorted(
                group,
                key=lambda candidate: (
                    -candidate["minimum_episode_pass_fraction"],
                    -candidate["minimum_cycle_success_fraction"],
                    candidate["maximum_torque_utilization_p99"],
                    candidate["training_seed"],
                ),
            )[0]

    return {
        "schema_version": 1,
        "campaign_id": manifest.get("campaign_id"),
        "protocol": H1_PROTOCOL,
        "training_seeds": expected_seeds,
        "candidates": sorted(
            candidates,
            key=lambda candidate: (
                candidate["checkpoint_iteration"],
                candidate["training_seed"],
            ),
        ),
        "iterations": iterations,
        "selected_checkpoint_iteration": selected_iteration,
        "selected_candidate": selected_candidate,
        "decision": "accepted" if selected_candidate is not None else "rejected",
        "selection_rule": (
            "earliest complete all-seed passing iteration; representative by "
            "highest minimum episode pass, highest minimum cycle success, lower "
            "maximum torque p99, then lower training seed"
        ),
        "physical_motion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_h1_checkpoint_sweep(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"h1_checkpoint_sweep_decision={summary['decision']}")
    print(f"h1_checkpoint_sweep_retained={args.output}")


if __name__ == "__main__":
    main()
