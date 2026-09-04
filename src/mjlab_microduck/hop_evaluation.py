"""Deterministic, reset-safe metrics and gates for periodic-hop rollouts."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


H1_PROTOCOL = "H1-periodic-hop-heldout-v1"
MIN_RISE_M = 0.003
MIN_MEDIAN_RISE_M = 0.020
MAX_RISE_M = 0.100
MAX_DRIFT_M = 0.100
MAX_BOTTOMED_FRACTION = 0.01
MIN_CYCLE_SUCCESS_FRACTION = 0.90
MIN_LANDING_FRACTION = 0.98
MIN_EPISODE_PASS_FRACTION = 0.80
MAX_TORQUE_UTILIZATION_P99 = 0.90
MAX_NEAR_STALL_FRACTION = 0.0025


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q).item())


def _check_trace_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape must be {expected}, got {tuple(value.shape)}")


def summarize_hop_trace(
    *,
    base_z: torch.Tensor,
    base_xy: torch.Tensor,
    both_airborne: torch.Tensor,
    dones: torch.Tensor,
    falls: torch.Tensor,
    nan_terminations: torch.Tensor,
    finite: torch.Tensor,
    spring_compression_ratio: torch.Tensor,
    landing_force: torch.Tensor,
    steps_per_cycle: int,
    cycles: int,
    min_rise_m: float = MIN_RISE_M,
    max_drift_m: float = MAX_DRIFT_M,
) -> dict:
    """Summarize only the first episode of every vectorized environment.

    State traces include the initial state at index zero. Transition flags and
    per-step instrumentation have one fewer row. Automatic resets after a done
    are intentionally ignored so a fall cannot be hidden by a good next episode.
    """
    if steps_per_cycle <= 0 or cycles <= 0:
        raise ValueError("steps_per_cycle and cycles must be positive")
    if min_rise_m <= 0.0 or max_drift_m <= 0.0:
        raise ValueError("min_rise_m and max_drift_m must be positive")
    if base_z.ndim != 2:
        raise ValueError("base_z must have shape [steps + 1, num_envs]")
    steps = steps_per_cycle * cycles
    num_envs = base_z.shape[1]
    _check_trace_shape("base_z", base_z, (steps + 1, num_envs))
    _check_trace_shape("base_xy", base_xy, (steps + 1, num_envs, 2))
    _check_trace_shape("both_airborne", both_airborne, (steps + 1, num_envs))
    for name, value in (
        ("dones", dones),
        ("falls", falls),
        ("nan_terminations", nan_terminations),
        ("finite", finite),
    ):
        _check_trace_shape(name, value, (steps, num_envs))
    if spring_compression_ratio.ndim != 3 or spring_compression_ratio.shape[:2] != (
        steps,
        num_envs,
    ):
        raise ValueError(
            "spring_compression_ratio must have shape [steps, num_envs, joints]"
        )
    if landing_force.ndim != 3 or landing_force.shape[:2] != (steps, num_envs):
        raise ValueError("landing_force must have shape [steps, num_envs, feet]")

    base_z = base_z.float()
    base_xy = base_xy.float()
    airborne = both_airborne.bool()
    dones = dones.bool()
    falls = falls.bool()
    nan_terminations = nan_terminations.bool()
    finite = finite.bool()

    device = base_z.device
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    have_stance = (~airborne[0]) & torch.isfinite(base_z[0])
    stance_z = torch.nan_to_num(base_z[0])
    takeoff_z = stance_z.clone()
    was_airborne = airborne[0] & torch.isfinite(base_z[0])
    flight_cycle = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    flight_qualified = torch.zeros(num_envs, dtype=torch.bool, device=device)

    cycle_rise = torch.zeros((num_envs, cycles), device=device)
    cycle_qualified = torch.zeros((num_envs, cycles), dtype=torch.bool, device=device)
    cycle_landed = torch.zeros((num_envs, cycles), dtype=torch.bool, device=device)
    completed_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)
    max_drift = torch.zeros(num_envs, device=device)
    fall_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    nan_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    nonfinite_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    bottomed_samples = 0
    spring_samples = 0
    landing_samples: list[torch.Tensor] = []

    env_ids = torch.arange(num_envs, device=device)
    for index in range(steps):
        state_index = index + 1
        cycle = min(index // steps_per_cycle, cycles - 1)
        valid = active & finite[index] & torch.isfinite(base_z[state_index])
        current_airborne = airborne[state_index]
        z = torch.nan_to_num(base_z[state_index])

        in_contact = valid & ~current_airborne
        stance_z = torch.where(in_contact, z, stance_z)
        have_stance |= in_contact

        took_off = valid & current_airborne & ~was_airborne
        datum = torch.where(have_stance, stance_z, z)
        takeoff_z = torch.where(took_off, datum, takeoff_z)
        flight_cycle = torch.where(
            took_off, torch.full_like(flight_cycle, cycle), flight_cycle
        )
        flight_qualified = torch.where(
            took_off, torch.zeros_like(flight_qualified), flight_qualified
        )

        rise = torch.where(
            valid & current_airborne,
            torch.clamp(z - takeoff_z, min=0.0),
            torch.zeros_like(z),
        )
        cycle_rise[:, cycle] = torch.maximum(cycle_rise[:, cycle], rise)
        qualified = valid & current_airborne & (rise >= min_rise_m)
        cycle_qualified[:, cycle] |= qualified
        flight_qualified |= qualified

        touchdown = valid & ~current_airborne & was_airborne
        landed = touchdown & flight_qualified
        landed_ids = env_ids[landed]
        landed_cycles = flight_cycle[landed]
        usable = landed_cycles >= 0
        if bool(usable.any()):
            cycle_landed[landed_ids[usable], landed_cycles[usable]] = True

        drift = torch.linalg.vector_norm(
            base_xy[state_index] - base_xy[0], dim=1
        )
        max_drift = torch.where(active, torch.maximum(max_drift, drift), max_drift)

        compression = torch.nan_to_num(
            spring_compression_ratio[index].float()
        ).clamp(min=0.0)
        if compression.numel() > 0:
            selected = compression[active]
            bottomed_samples += int((selected >= 0.95).sum().item())
            spring_samples += selected.numel()
        selected_force = torch.nan_to_num(landing_force[index].float())[touchdown]
        if selected_force.numel() > 0:
            landing_samples.append(selected_force.flatten())

        fall_seen |= active & falls[index]
        nan_seen |= active & nan_terminations[index]
        nonfinite_seen |= active & ~finite[index]
        if (index + 1) % steps_per_cycle == 0:
            completed_cycles += active.long()

        was_airborne = torch.where(valid, current_airborne, was_airborne)
        active &= ~dones[index]

    successful_cycles = cycle_qualified & cycle_landed
    episode_passed = (
        (completed_cycles == cycles)
        & successful_cycles.all(dim=1)
        & ~fall_seen
        & ~nan_seen
        & ~nonfinite_seen
        & (max_drift <= max_drift_m)
    )
    qualified_count = int(cycle_qualified.sum().item())
    landed_count = int(successful_cycles.sum().item())
    planned_cycles = num_envs * cycles
    landing_values = (
        torch.cat(landing_samples)
        if landing_samples
        else torch.zeros(0, device=device)
    )

    return {
        "num_envs": num_envs,
        "cycles_per_env": cycles,
        "planned_cycles": planned_cycles,
        "completed_cycles": int(completed_cycles.sum().item()),
        "qualified_cycles": qualified_count,
        "landed_qualified_cycles": landed_count,
        "cycle_success_fraction": landed_count / planned_cycles,
        "landing_fraction": landed_count / qualified_count if qualified_count else 0.0,
        "cycle_rise_mean_m": float(cycle_rise.mean().item()),
        "cycle_rise_p50_m": _quantile(cycle_rise.flatten(), 0.50),
        "cycle_rise_p95_m": _quantile(cycle_rise.flatten(), 0.95),
        "cycle_rise_peak_m": float(cycle_rise.max().item()),
        "episode_passes": int(episode_passed.sum().item()),
        "episode_pass_fraction": float(episode_passed.float().mean().item()),
        "max_drift_mean_m": float(max_drift.mean().item()),
        "max_drift_p95_m": _quantile(max_drift, 0.95),
        "max_drift_peak_m": float(max_drift.max().item()),
        "fall_events": int(fall_seen.sum().item()),
        "nan_termination_events": int(nan_seen.sum().item()),
        "nonfinite_episodes": int(nonfinite_seen.sum().item()),
        "spring_bottomed_fraction": (
            bottomed_samples / spring_samples if spring_samples else 0.0
        ),
        "landing_force_p95": _quantile(landing_values, 0.95),
        "landing_force_peak": (
            float(landing_values.max().item()) if landing_values.numel() else 0.0
        ),
    }


def _gate(name: str, passed: bool, observed, criterion: str) -> dict:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "criterion": criterion,
    }


def h1_gates(cases: Iterable[dict]) -> list[dict]:
    """Apply the predeclared H1 gates across all held-out seed cases."""
    cases = list(cases)
    if not cases:
        raise ValueError("at least one H1 evaluation case is required")
    for case in cases:
        for key, value in case.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"case metric {key} must be finite")

    def maxima(key: str) -> float:
        return max(float(case[key]) for case in cases)

    def minima(key: str) -> float:
        return min(float(case[key]) for case in cases)

    def totals(key: str) -> int:
        return sum(int(case[key]) for case in cases)
    return [
        _gate(
            "cycle_success_fraction",
            minima("cycle_success_fraction") >= MIN_CYCLE_SUCCESS_FRACTION,
            minima("cycle_success_fraction"),
            f"minimum per seed >= {MIN_CYCLE_SUCCESS_FRACTION}",
        ),
        _gate(
            "landing_fraction",
            minima("landing_fraction") >= MIN_LANDING_FRACTION,
            minima("landing_fraction"),
            f"minimum per seed >= {MIN_LANDING_FRACTION}",
        ),
        _gate(
            "episode_pass_fraction",
            minima("episode_pass_fraction") >= MIN_EPISODE_PASS_FRACTION,
            minima("episode_pass_fraction"),
            f"minimum per seed >= {MIN_EPISODE_PASS_FRACTION}",
        ),
        _gate(
            "median_rise",
            minima("cycle_rise_p50_m") >= MIN_MEDIAN_RISE_M,
            minima("cycle_rise_p50_m"),
            f"minimum per seed >= {MIN_MEDIAN_RISE_M} m",
        ),
        _gate("falls", totals("fall_events") == 0, totals("fall_events"), "total == 0"),
        _gate(
            "nan_terminations",
            totals("nan_termination_events") == 0,
            totals("nan_termination_events"),
            "total == 0",
        ),
        _gate(
            "nonfinite_episodes",
            totals("nonfinite_episodes") == 0,
            totals("nonfinite_episodes"),
            "total == 0",
        ),
        _gate(
            "rise_ceiling",
            maxima("cycle_rise_peak_m") <= MAX_RISE_M,
            maxima("cycle_rise_peak_m"),
            f"maximum across seeds <= {MAX_RISE_M} m",
        ),
        _gate(
            "drift",
            maxima("max_drift_p95_m") <= MAX_DRIFT_M,
            maxima("max_drift_p95_m"),
            f"maximum p95 across seeds <= {MAX_DRIFT_M} m",
        ),
        _gate(
            "spring_bottoming",
            maxima("spring_bottomed_fraction") <= MAX_BOTTOMED_FRACTION,
            maxima("spring_bottomed_fraction"),
            f"maximum across seeds <= {MAX_BOTTOMED_FRACTION}",
        ),
        _gate(
            "rated_speed_exceedance",
            maxima("motor_speed_rated_exceed_fraction") == 0.0,
            maxima("motor_speed_rated_exceed_fraction"),
            "maximum across seeds == 0",
        ),
        _gate(
            "torque_utilization_p99",
            maxima("motor_torque_utilization_p99") <= MAX_TORQUE_UTILIZATION_P99,
            maxima("motor_torque_utilization_p99"),
            f"maximum across seeds <= {MAX_TORQUE_UTILIZATION_P99}",
        ),
        _gate(
            "near_stall_fraction",
            maxima("motor_torque_near_stall_fraction") <= MAX_NEAR_STALL_FRACTION,
            maxima("motor_torque_near_stall_fraction"),
            f"maximum across seeds <= {MAX_NEAR_STALL_FRACTION}",
        ),
    ]


def h1_decision(cases: Iterable[dict]) -> dict:
    cases = list(cases)
    gates = h1_gates(cases)
    return {
        "protocol": H1_PROTOCOL,
        "gates": gates,
        "decision": "accepted" if all(gate["status"] == "pass" for gate in gates) else "rejected",
        "physical_motion_authorized": False,
    }
