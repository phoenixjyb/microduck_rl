"""Apply the unchanged unified-controller numerical gate to the HC4-U3 pilot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mjlab_microduck.hc4u1_gate import compare_hc4u1_prescreen
from mjlab_microduck.hierarchical_obstacle_rollout import FIRST_TERMINAL_OUTCOME_PROTOCOL
from mjlab_microduck.o3a_gate import _reject_nonfinite

HELD_OUT_SEEDS = (293, 307, 311)
STAGE = "HC4U3-phase-separated-BC-rollout"


def compare_hc4u3_prescreen(
    candidate: Path, near_source: Path, far_source: Path,
    *, candidate_sha256: str, seed: int,
) -> dict:
    if seed not in HELD_OUT_SEEDS:
        raise ValueError("seed is outside the predeclared HC4-U3 matrix")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
        raise ValueError("candidate_sha256 must be the retained full checkpoint hash")
    for path in (candidate, near_source, far_source):
        report = json.loads(path.read_text())
        _reject_nonfinite(report)
        if report.get("terminal_outcome_protocol") != FIRST_TERMINAL_OUTCOME_PROTOCOL:
            raise ValueError("HC4-U3 requires failure-priority terminal accounting")
        sensor = report.get("obstacle_sensor_model")
        if not isinstance(sensor, dict) or any(sensor.get(field) != 0.0 for field in (
            "range_noise_m", "bearing_noise_rad", "width_noise_m", "height_noise_m",
            "closing_rate_noise_mps", "dropout_probability",
        )):
            raise ValueError("HC4-U3 requires exact sensor settings")
        for case in report["cases"]:
            if case.get("terminal_outcome_protocol") != FIRST_TERMINAL_OUTCOME_PROTOCOL:
                raise ValueError("HC4-U3 case terminal accounting protocol mismatch")
            outcomes = [case.get(name) for name in (
                "clean_pass_events", "collision_events", "attempt_timeout_events",
            )]
            if any(type(v) is not int or v < 0 for v in outcomes) or sum(outcomes) != 64:
                raise ValueError("terminal outcomes must partition the 64 fixed attempts")
    return compare_hc4u1_prescreen(
        candidate, near_source, far_source, candidate_stage=STAGE,
        candidate_sha256=candidate_sha256, seed=seed,
        protocol=f"HC4-U3-seed-{seed}-fixed-attempt-prescreen-v2",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("near_source", type=Path)
    parser.add_argument("far_source", type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--physics-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_hc4u3_prescreen(
        args.candidate, args.near_source, args.far_source,
        candidate_sha256=args.candidate_sha256, seed=args.physics_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"hc4u3_decision={result['decision']}")


if __name__ == "__main__":
    main()
