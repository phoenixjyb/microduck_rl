"""Deterministic fixed-attempt pre-screen gate for HC4-U1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "first-terminal-attempt-per-environment-v1"
SEED = 193
NUM_ENVS = 64
ACTOR_SHA256 = "080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54"
CANDIDATE_STAGE = "HC4U1-unified-range-lateral-correction-BC-rollout"
CANDIDATE_SHA256 = (
    "2196d2ed2dbc3e182fa0b36edf663d11187330d430cd319ceb368c8a28e9753b"
)
NEAR_STAGE = "HC4R2-student-state-correction-BC-rollout"
NEAR_SHA256 = "c4ba5925de7144373c94145b57b5e7a7ae3e1fc89bc7c2c3203f8724bdebf1b7"
FAR_STAGE = "HC4LH-lateral-gated-supervisor-rollout"
FAR_SHA256 = "0b2608080671c5df85d8c9f900d68b6a6f298ec820eb1c6ba75afc948337505a"
SPEEDS = (0.30, 0.40)
LATERALS = (-0.08, 0.00, 0.08)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_key(case: dict[str, Any]) -> tuple[float, float, float]:
    return (
        round(float(case["nominal_speed_mps"]), 2),
        round(float(case["obstacle_forward_m"]), 2),
        round(float(case["obstacle_lateral_m"]), 2),
    )


def _expected_cells(forward_positions: tuple[float, ...]) -> set[tuple[float, ...]]:
    return {
        (speed, forward, lateral)
        for speed in SPEEDS
        for forward in forward_positions
        for lateral in LATERALS
    }


def _load_report(
    path: Path,
    *,
    expected_stage: str,
    expected_supervisor_sha256: str,
    forward_positions: tuple[float, ...],
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    report = json.loads(path.read_text())
    if report.get("evaluation_window") != PROTOCOL:
        raise ValueError(f"unexpected evaluation protocol in {path}")
    if report.get("stage") != expected_stage:
        raise ValueError(f"unexpected rollout stage in {path}")
    if report.get("checkpoint_sha256") != ACTOR_SHA256:
        raise ValueError(f"unexpected locomotion actor in {path}")
    if report.get("supervisor_checkpoint_sha256") != expected_supervisor_sha256:
        raise ValueError(f"unexpected supervisor checkpoint in {path}")
    if report.get("physical_motion_authorized") is not False:
        raise ValueError(f"report does not retain the no-motion boundary: {path}")
    if report.get("perception") != "exact structured geometry; no raw camera perception":
        raise ValueError(f"unexpected perception authority in {path}")

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"cases are missing in {path}")
    keyed_cases = {_cell_key(case): case for case in cases}
    expected_cells = _expected_cells(forward_positions)
    if len(keyed_cases) != len(cases) or set(keyed_cases) != expected_cells:
        raise ValueError(f"unexpected or duplicate evaluation cells in {path}")
    for case in cases:
        if case.get("seed") != SEED or case.get("num_envs") != NUM_ENVS:
            raise ValueError(f"unexpected seed or environment count in {path}")
        if case.get("evaluation_window") != PROTOCOL:
            raise ValueError(f"case protocol mismatch in {path}")
        for name in ("expected_attempts", "completed_attempts", "resolved_attempts"):
            if case.get(name) != NUM_ENVS:
                raise ValueError(f"{name} does not equal {NUM_ENVS} in {path}")
        for name in (
            "unresolved_attempts",
            "hard_failure_events",
            "other_terminal_events",
            "fall_events",
            "nan_termination_events",
            "nonfinite_steps",
        ):
            if case.get(name) != 0:
                raise ValueError(f"{name} is nonzero in {path}")
        if float(case.get("motor_speed_rated_exceed_fraction", -1.0)) != 0.0:
            raise ValueError(f"rated motor speed was exceeded in {path}")

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "stage": expected_stage,
        "supervisor_checkpoint_sha256": expected_supervisor_sha256,
        "cases": keyed_cases,
    }


def _pooled_phase_speed(cases: list[dict[str, Any]], phase: str) -> float:
    value_name = f"{phase}_route_speed_mps"
    sample_name = f"{phase}_samples"
    samples = sum(int(case[sample_name]) for case in cases)
    if samples <= 0:
        raise ValueError(f"{phase} has no samples")
    return sum(
        float(case[value_name]) * int(case[sample_name]) for case in cases
    ) / samples


def compare_hc4u1_prescreen(
    candidate_path: Path,
    near_source_path: Path,
    far_source_path: Path,
) -> dict[str, Any]:
    """Apply the frozen seed-193 HC4-U1 fixed-attempt pre-screen."""
    candidate = _load_report(
        candidate_path,
        expected_stage=CANDIDATE_STAGE,
        expected_supervisor_sha256=CANDIDATE_SHA256,
        forward_positions=(0.90, 1.15),
    )
    near = _load_report(
        near_source_path,
        expected_stage=NEAR_STAGE,
        expected_supervisor_sha256=NEAR_SHA256,
        forward_positions=(0.90,),
    )
    far = _load_report(
        far_source_path,
        expected_stage=FAR_STAGE,
        expected_supervisor_sha256=FAR_SHA256,
        forward_positions=(1.15,),
    )
    sources = {**near["cases"], **far["cases"]}
    candidate_cases = candidate["cases"]

    cell_deltas = []
    for cell in sorted(candidate_cases):
        candidate_case = candidate_cases[cell]
        source_case = sources[cell]
        cell_deltas.append(
            {
                "speed_mps": cell[0],
                "forward_m": cell[1],
                "lateral_m": cell[2],
                "clean_delta": candidate_case["clean_pass_events"]
                - source_case["clean_pass_events"],
                "collision_delta": candidate_case["collision_events"]
                - source_case["collision_events"],
                "timeout_delta": candidate_case["attempt_timeout_events"]
                - source_case["attempt_timeout_events"],
                "approach_speed_delta_mps": candidate_case[
                    "approach_route_speed_mps"
                ]
                - source_case["approach_route_speed_mps"],
                "recovery_speed_delta_mps": candidate_case[
                    "recovery_route_speed_mps"
                ]
                - source_case["recovery_route_speed_mps"],
            }
        )

    candidate_values = list(candidate_cases.values())
    source_values = list(sources.values())
    candidate_totals = {
        name: sum(int(case[name]) for case in candidate_values)
        for name in ("clean_pass_events", "collision_events", "attempt_timeout_events")
    }
    source_totals = {
        name: sum(int(case[name]) for case in source_values)
        for name in ("clean_pass_events", "collision_events", "attempt_timeout_events")
    }
    pooled_speed_deltas = {
        phase: _pooled_phase_speed(candidate_values, phase)
        - _pooled_phase_speed(source_values, phase)
        for phase in ("approach", "recovery")
    }
    max_torque_p99 = max(
        float(case["motor_torque_utilization_p99"]) for case in candidate_values
    )

    checks = [
        {
            "name": "per_cell_collision_non_regression",
            "status": "pass"
            if all(cell["collision_delta"] <= 0 for cell in cell_deltas)
            else "fail",
            "violations": [
                cell for cell in cell_deltas if cell["collision_delta"] > 0
            ],
        },
        {
            "name": "per_cell_timeout_non_regression",
            "status": "pass"
            if all(cell["timeout_delta"] <= 0 for cell in cell_deltas)
            else "fail",
            "violations": [cell for cell in cell_deltas if cell["timeout_delta"] > 0],
        },
        {
            "name": "per_cell_clean_non_regression",
            "status": "pass"
            if all(cell["clean_delta"] >= 0 for cell in cell_deltas)
            else "fail",
            "violations": [cell for cell in cell_deltas if cell["clean_delta"] < 0],
        },
        {
            "name": "per_cell_approach_speed_non_regression",
            "status": "pass"
            if all(cell["approach_speed_delta_mps"] >= -0.03 for cell in cell_deltas)
            else "fail",
            "violations": [
                cell for cell in cell_deltas if cell["approach_speed_delta_mps"] < -0.03
            ],
        },
        {
            "name": "per_cell_recovery_speed_non_regression",
            "status": "pass"
            if all(cell["recovery_speed_delta_mps"] >= -0.03 for cell in cell_deltas)
            else "fail",
            "violations": [
                cell for cell in cell_deltas if cell["recovery_speed_delta_mps"] < -0.03
            ],
        },
        {
            "name": "aggregate_outcome_non_regression",
            "status": "pass"
            if (
                candidate_totals["collision_events"]
                <= source_totals["collision_events"]
                and candidate_totals["attempt_timeout_events"]
                <= source_totals["attempt_timeout_events"]
                and candidate_totals["clean_pass_events"]
                >= source_totals["clean_pass_events"]
            )
            else "fail",
        },
        {
            "name": "aggregate_phase_speed_non_regression",
            "status": "pass"
            if all(value >= -0.01 for value in pooled_speed_deltas.values())
            else "fail",
            "observed_deltas_mps": pooled_speed_deltas,
        },
        {
            "name": "motor_torque_p99_at_most_0_60",
            "status": "pass" if max_torque_p99 <= 0.60 else "fail",
            "observed": max_torque_p99,
        },
    ]
    accepted = all(check["status"] == "pass" for check in checks)
    return {
        "schema_version": 1,
        "protocol": "HC4-U1-seed-193-fixed-attempt-prescreen-v1",
        "candidate": {k: v for k, v in candidate.items() if k != "cases"},
        "near_source": {k: v for k, v in near.items() if k != "cases"},
        "far_source": {k: v for k, v in far.items() if k != "cases"},
        "candidate_totals": candidate_totals,
        "paired_source_totals": source_totals,
        "cell_deltas": cell_deltas,
        "pooled_phase_speed_deltas_mps": pooled_speed_deltas,
        "candidate_max_torque_utilization_p99": max_torque_p99,
        "checks": checks,
        "decision": "continue_fresh_seeds" if accepted else "stop",
        "physical_motion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("near_source", type=Path)
    parser.add_argument("far_source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_hc4u1_prescreen(
        args.candidate, args.near_source, args.far_source
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"hc4u1_prescreen_decision={result['decision']}")
    print(f"hc4u1_prescreen_retained={args.output}")


if __name__ == "__main__":
    main()
