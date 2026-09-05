"""Deterministic paired gate for the O3a compact-range-noise pre-screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ATTEMPT_PROTOCOL = "first-terminal-attempt-per-environment-v1"
NOISE_PROTOCOL = "compact-range-uniform-v1"
ROLLOUT_STAGE = "HC4LH-lateral-gated-supervisor-rollout"
HC4R2_ROLLOUT_STAGE = "HC4R2-student-state-correction-BC-rollout"
ACTOR_SHA256 = "080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54"
SUPERVISOR_SHA256 = (
    "0b2608080671c5df85d8c9f900d68b6a6f298ec820eb1c6ba75afc948337505a"
)
HC4R2_SUPERVISOR_SHA256 = (
    "c4ba5925de7144373c94145b57b5e7a7ae3e1fc89bc7c2c3203f8724bdebf1b7"
)
PHYSICS_SEED = 271
NOISE_SEED = 3_000_282
NUM_ENVS = 64
SPEED_MPS = 0.50
FORWARD_M = 1.15
LATERALS_M = (-0.08, 0.00, 0.08)
RANGE_NOISE_BOUND_M = 0.02
HC4R2_PHYSICS_SEED = 277
HC4R2_NOISE_SEED = 3_000_288
HC4R2_SPEEDS_MPS = (0.30, 0.40)
HC4R2_FORWARD_M = 0.90


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite(value: Any, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_nonfinite(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, f"{path}[{index}]")


def _cell_key(case: dict[str, Any]) -> tuple[float, float, float]:
    return (
        round(float(case["nominal_speed_mps"]), 2),
        round(float(case["obstacle_forward_m"]), 2),
        round(float(case["obstacle_lateral_m"]), 2),
    )


def _load_report(
    path: Path,
    *,
    noisy: bool,
    expected_stage: str,
    expected_supervisor_sha256: str,
    physics_seed: int,
    noise_seed: int,
    speeds_mps: tuple[float, ...],
    forward_m: float,
    laterals_m: tuple[float, ...],
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    report = json.loads(path.read_text())
    _reject_nonfinite(report)
    if report.get("evaluation_window") != ATTEMPT_PROTOCOL:
        raise ValueError(f"unexpected evaluation protocol in {path}")
    if report.get("stage") != expected_stage:
        raise ValueError(f"unexpected rollout stage in {path}")
    if report.get("checkpoint_sha256") != ACTOR_SHA256:
        raise ValueError(f"unexpected locomotion actor in {path}")
    if report.get("supervisor_checkpoint_sha256") != expected_supervisor_sha256:
        raise ValueError(f"unexpected supervisor checkpoint in {path}")
    if report.get("physical_motion_authorized") is not False:
        raise ValueError(f"report does not retain the no-motion boundary: {path}")

    expected_perception = (
        "compact structured geometry with bounded range noise; "
        "no raw camera perception"
        if noisy
        else "exact structured geometry; no raw camera perception"
    )
    if report.get("perception") != expected_perception:
        raise ValueError(f"unexpected perception authority in {path}")
    sensor_model = report.get("obstacle_sensor_model")
    expected_bound = RANGE_NOISE_BOUND_M if noisy else 0.0
    if not isinstance(sensor_model, dict) or sensor_model.get(
        "range_noise_m"
    ) != expected_bound:
        raise ValueError(f"unexpected range-noise model in {path}")
    for field in (
        "bearing_noise_rad",
        "width_noise_m",
        "height_noise_m",
        "closing_rate_noise_mps",
        "dropout_probability",
    ):
        if sensor_model.get(field) != 0.0:
            raise ValueError(f"O3a changed {field} in {path}")

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"cases are missing in {path}")
    keyed_cases = {_cell_key(case): case for case in cases}
    expected_cells = {
        (speed, forward_m, lateral)
        for speed in speeds_mps
        for lateral in laterals_m
    }
    if len(keyed_cases) != len(cases) or set(keyed_cases) != expected_cells:
        raise ValueError(f"unexpected or duplicate evaluation cells in {path}")

    for case in cases:
        if case.get("seed") != physics_seed or case.get("num_envs") != NUM_ENVS:
            raise ValueError(f"unexpected seed or environment count in {path}")
        if case.get("evaluation_window") != ATTEMPT_PROTOCOL:
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
        resolved = sum(
            int(case[name])
            for name in (
                "clean_pass_events",
                "collision_events",
                "attempt_timeout_events",
            )
        )
        if resolved != NUM_ENVS:
            raise ValueError(f"terminal outcomes do not total {NUM_ENVS} in {path}")
        if float(case.get("motor_speed_rated_exceed_fraction", -1.0)) != 0.0:
            raise ValueError(f"rated motor speed was exceeded in {path}")

        protocol = case.get("obstacle_sensor_protocol")
        expected_identity = NOISE_PROTOCOL if noisy else "exact-v1"
        if not isinstance(protocol, dict) or protocol.get(
            "identity"
        ) != expected_identity:
            raise ValueError(f"unexpected sensor protocol in {path}")
        if protocol.get("range_noise_bound_m") != expected_bound:
            raise ValueError(f"case range-noise bound mismatch in {path}")
        if noisy:
            if protocol.get("noise_seed") != noise_seed:
                raise ValueError(f"unexpected noise seed in {path}")
            if protocol.get("perturbed_fields") != ["range"]:
                raise ValueError(f"O3a perturbed more than range in {path}")
        elif protocol.get("noise_seed") is not None:
            raise ValueError(f"exact baseline carries a noise seed in {path}")

    return {"path": str(path), "sha256": _sha256(path), "cases": keyed_cases}


def _pooled_phase_speed(cases: list[dict[str, Any]], phase: str) -> float:
    value_name = f"{phase}_route_speed_mps"
    sample_name = f"{phase}_samples"
    samples = sum(int(case[sample_name]) for case in cases)
    if samples <= 0:
        raise ValueError(f"{phase} has no samples")
    return sum(
        float(case[value_name]) * int(case[sample_name]) for case in cases
    ) / samples


def _compare_prescreen(
    baseline_path: Path,
    noisy_path: Path,
    *,
    expected_stage: str,
    expected_supervisor_sha256: str,
    physics_seed: int,
    noise_seed: int,
    speeds_mps: tuple[float, ...],
    forward_m: float,
    laterals_m: tuple[float, ...],
    protocol: str,
    continue_decision: str,
) -> dict[str, Any]:
    baseline = _load_report(
        baseline_path,
        noisy=False,
        expected_stage=expected_stage,
        expected_supervisor_sha256=expected_supervisor_sha256,
        physics_seed=physics_seed,
        noise_seed=noise_seed,
        speeds_mps=speeds_mps,
        forward_m=forward_m,
        laterals_m=laterals_m,
    )
    noisy = _load_report(
        noisy_path,
        noisy=True,
        expected_stage=expected_stage,
        expected_supervisor_sha256=expected_supervisor_sha256,
        physics_seed=physics_seed,
        noise_seed=noise_seed,
        speeds_mps=speeds_mps,
        forward_m=forward_m,
        laterals_m=laterals_m,
    )
    baseline_cases = baseline["cases"]
    noisy_cases = noisy["cases"]

    cell_deltas = []
    for cell in sorted(baseline_cases):
        exact_case = baseline_cases[cell]
        noisy_case = noisy_cases[cell]
        cell_deltas.append(
            {
                "speed_mps": cell[0],
                "forward_m": cell[1],
                "lateral_m": cell[2],
                "clean_delta": noisy_case["clean_pass_events"]
                - exact_case["clean_pass_events"],
                "collision_delta": noisy_case["collision_events"]
                - exact_case["collision_events"],
                "timeout_delta": noisy_case["attempt_timeout_events"]
                - exact_case["attempt_timeout_events"],
                "approach_speed_delta_mps": noisy_case[
                    "approach_route_speed_mps"
                ]
                - exact_case["approach_route_speed_mps"],
                "recovery_speed_delta_mps": noisy_case[
                    "recovery_route_speed_mps"
                ]
                - exact_case["recovery_route_speed_mps"],
            }
        )

    exact_values = list(baseline_cases.values())
    noisy_values = list(noisy_cases.values())
    exact_totals = {
        name: sum(int(case[name]) for case in exact_values)
        for name in ("clean_pass_events", "collision_events", "attempt_timeout_events")
    }
    noisy_totals = {
        name: sum(int(case[name]) for case in noisy_values)
        for name in ("clean_pass_events", "collision_events", "attempt_timeout_events")
    }
    pooled_speed_deltas = {
        phase: _pooled_phase_speed(noisy_values, phase)
        - _pooled_phase_speed(exact_values, phase)
        for phase in ("approach", "recovery")
    }
    attempt_count = NUM_ENVS * len(baseline_cases)
    pooled_clean_rate_delta = (
        noisy_totals["clean_pass_events"] - exact_totals["clean_pass_events"]
    ) / attempt_count
    max_torque_p99 = max(
        float(case["motor_torque_utilization_p99"]) for case in noisy_values
    )

    checks = [
        {
            "name": "collision_non_regression",
            "status": "pass"
            if noisy_totals["collision_events"] <= exact_totals["collision_events"]
            and all(cell["collision_delta"] <= 0 for cell in cell_deltas)
            else "fail",
            "violations": [
                cell for cell in cell_deltas if cell["collision_delta"] > 0
            ],
        },
        {
            "name": "per_cell_clean_loss_at_most_3_of_64",
            "status": "pass"
            if all(cell["clean_delta"] >= -3 for cell in cell_deltas)
            else "fail",
            "violations": [cell for cell in cell_deltas if cell["clean_delta"] < -3],
        },
        {
            "name": "pooled_clean_rate_loss_at_most_0_05",
            "status": "pass" if pooled_clean_rate_delta >= -0.05 else "fail",
            "observed_delta": pooled_clean_rate_delta,
        },
        {
            "name": "per_cell_approach_recovery_delta_at_least_minus_0_03_mps",
            "status": "pass"
            if all(
                cell["approach_speed_delta_mps"] >= -0.03
                and cell["recovery_speed_delta_mps"] >= -0.03
                for cell in cell_deltas
            )
            else "fail",
            "violations": [
                cell
                for cell in cell_deltas
                if cell["approach_speed_delta_mps"] < -0.03
                or cell["recovery_speed_delta_mps"] < -0.03
            ],
        },
        {
            "name": "pooled_approach_recovery_delta_at_least_minus_0_01_mps",
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
        "protocol": protocol,
        "baseline": {key: value for key, value in baseline.items() if key != "cases"},
        "noisy": {key: value for key, value in noisy.items() if key != "cases"},
        "baseline_totals": exact_totals,
        "noisy_totals": noisy_totals,
        "cell_deltas": cell_deltas,
        "pooled_clean_rate_delta": pooled_clean_rate_delta,
        "pooled_phase_speed_deltas_mps": pooled_speed_deltas,
        "noisy_max_torque_utilization_p99": max_torque_p99,
        "checks": checks,
        "decision": continue_decision if accepted else "stop",
        "physical_motion_authorized": False,
    }


def compare_o3a_prescreen(
    baseline_path: Path, noisy_path: Path
) -> dict[str, Any]:
    """Apply the frozen HC4-LH seed-271 O3a pre-screen."""
    return _compare_prescreen(
        baseline_path,
        noisy_path,
        expected_stage=ROLLOUT_STAGE,
        expected_supervisor_sha256=SUPERVISOR_SHA256,
        physics_seed=PHYSICS_SEED,
        noise_seed=NOISE_SEED,
        speeds_mps=(SPEED_MPS,),
        forward_m=FORWARD_M,
        laterals_m=LATERALS_M,
        protocol="O3a-HC4LH-seed-271-range-noise-prescreen-v1",
        continue_decision="continue_hc4r2_predeclaration",
    )


def compare_hc4r2_prescreen(
    baseline_path: Path, noisy_path: Path
) -> dict[str, Any]:
    """Apply the frozen HC4-R2 seed-277 O3a pre-screen."""
    return _compare_prescreen(
        baseline_path,
        noisy_path,
        expected_stage=HC4R2_ROLLOUT_STAGE,
        expected_supervisor_sha256=HC4R2_SUPERVISOR_SHA256,
        physics_seed=HC4R2_PHYSICS_SEED,
        noise_seed=HC4R2_NOISE_SEED,
        speeds_mps=HC4R2_SPEEDS_MPS,
        forward_m=HC4R2_FORWARD_M,
        laterals_m=LATERALS_M,
        protocol="O3a-HC4R2-seed-277-range-noise-prescreen-v1",
        continue_decision="continue_multi_seed_predeclaration",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("noisy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specialist", choices=("hc4lh", "hc4r2"), default="hc4lh")
    args = parser.parse_args()
    comparator = (
        compare_o3a_prescreen
        if args.specialist == "hc4lh"
        else compare_hc4r2_prescreen
    )
    result = comparator(args.baseline, args.noisy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"o3a_prescreen_decision={result['decision']}")
    print(f"o3a_prescreen_retained={args.output}")


if __name__ == "__main__":
    main()
