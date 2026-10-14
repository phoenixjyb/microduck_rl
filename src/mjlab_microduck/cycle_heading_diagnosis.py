"""Read-only CPU cycle-excursion diagnostics for retained gait traces.

This report exposes within-cycle projected-body-x heading residuals alongside
the existing endpoint and maximum heading-rate gate. It is descriptive only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from mjlab_microduck import gait_path_qualification as route
from mjlab_microduck import gait_phase_heading_diagnosis as phase_diagnosis


PROTOCOL = "microduck-cycle-heading-diagnosis-v1"
HEADING_RATE_LIMIT_RAD_S = route.HEADING_ERROR_RAD_S_LIMIT


def _cycle_report(case: Mapping, trace: Mapping, world: int, cycle: Mapping) -> dict:
    start, end = cycle["start_step"], cycle["end_step"]
    start_time = trace["physics_time"][start][world]
    duration = trace["physics_time"][end][world] - start_time
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("cycle duration must be positive and finite")

    headings = [route._heading(trace["quat_wxyz"][step][world])
                for step in range(start, end + 1)]
    unwrapped = [headings[0]]
    for before, after in zip(headings, headings[1:]):
        unwrapped.append(unwrapped[-1] + route._wrap(after - before))
    start_heading = unwrapped[0]
    yaw = case["yaw"]
    residuals = []
    elapsed = []
    for offset, step in enumerate(range(start, end + 1)):
        t = trace["physics_time"][step][world] - start_time
        elapsed.append(t)
        residuals.append(unwrapped[offset] - start_heading - yaw * t)

    signed_endpoint = residuals[-1]
    max_abs = max(abs(value) for value in residuals)
    max_rate = max_abs / duration
    endpoint_rate = abs(signed_endpoint) / duration
    detrended = [value - signed_endpoint * (t / duration)
                 for value, t in zip(residuals, elapsed)]
    max_detrended = max(abs(value) for value in detrended)
    # max() returns the first index, making equal-magnitude peaks deterministic.
    peak_offset = max(range(len(residuals)), key=lambda i: abs(residuals[i]))
    peak_step = start + peak_offset
    peak_before, peak_after = ((peak_step - 1, peak_step) if peak_step > start
                               else (start, start + 1))
    head_yaw_index = trace["head_joints"].index("head_yaw")
    phase_time_s = {}
    for before in range(start, end):
        phase = phase_diagnosis._phase(trace["contact_found"], world, before, before + 1)
        dt = trace["physics_time"][before + 1][world] - trace["physics_time"][before][world]
        phase_time_s[phase] = phase_time_s.get(phase, 0.0) + dt
    endpoint_fail = endpoint_rate > HEADING_RATE_LIMIT_RAD_S
    max_fail = max_rate > HEADING_RATE_LIMIT_RAD_S
    peak_only = max_fail and not endpoint_fail
    if endpoint_fail and max_fail:
        classification = "endpoint-and-peak-fail"
    elif endpoint_fail:
        classification = "endpoint-fail-only"
    elif peak_only:
        classification = "peak-only"
    else:
        classification = "within-threshold"

    # Keep the new calculations anchored to the established route fields.
    for key, actual in (("heading_error_rad", signed_endpoint),
                        ("max_heading_error_rad", max_abs),
                        ("heading_error_rad_s", endpoint_rate),
                        ("max_heading_error_rad_s", max_rate)):
        if not math.isclose(actual, cycle[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"cycle heading diagnostic does not reconcile: {key}")

    return dict(
        start_step=start, end_step=end, duration_s=duration,
        signed_endpoint_residual_rad=signed_endpoint,
        endpoint_abs_rate_rad_s=endpoint_rate,
        max_abs_residual_rad=max_abs, max_abs_residual_rate_rad_s=max_rate,
        max_detrended_abs_residual_rad=max_detrended,
        max_detrended_abs_residual_rate_rad_s=max_detrended / duration,
        endpoint_fail=endpoint_fail, max_fail=max_fail, peak_only=peak_only,
        classification=classification,
        peak_step=peak_step, peak_elapsed_fraction=elapsed[peak_offset] / duration,
        signed_peak_residual_rad=residuals[peak_offset],
        peak_interval_phase=phase_diagnosis._phase(
            trace["contact_found"], world, peak_before, peak_after),
        phase_time_s=phase_time_s,
        head_yaw_offset_at_peak_rad=trace["head_position_offset"][peak_step][world][head_yaw_index],
        head_yaw_velocity_at_peak_rad_s=trace["head_velocity"][peak_step][world][head_yaw_index],
    )


def summarize_case(case: Mapping, trace: Mapping) -> dict:
    """Summarize actual left-touchdown cycles after strict retained-trace validation."""
    route_report = route.summarize_case(case, trace)
    worlds = []
    all_cycles = []
    for world_report in route_report["worlds"]:
        world = world_report["world"]
        cycles = [_cycle_report(case, trace, world, cycle)
                  for cycle in world_report["cycles"]]
        all_cycles.extend(cycles)
        counts = {name: sum(bool(cycle[name]) for cycle in cycles)
                  for name in ("endpoint_fail", "max_fail", "peak_only")}
        worlds.append(dict(world=world, idle=world_report["idle"],
                           cycle_count=len(cycles), **counts, cycles=cycles,
                           mean_signed_endpoint_residual_rad=(
                               sum(x["signed_endpoint_residual_rad"] for x in cycles) / len(cycles)
                               if cycles else None),
                           mean_max_abs_residual_rad=(
                               sum(x["max_abs_residual_rad"] for x in cycles) / len(cycles)
                               if cycles else None)))

    counts = {name: sum(bool(cycle[name]) for cycle in all_cycles)
              for name in ("endpoint_fail", "max_fail", "peak_only")}
    return dict(
        protocol=PROTOCOL, route_protocol=route.PROTOCOL,
        checkpoint_sha256=case["checkpoint_sha256"], seed=case["seed"],
        speed=case["speed"], yaw=case["yaw"],
        timing=dict(step_dt=trace["step_dt"], startup_steps=trace["startup_steps"],
                    per_world_cycle_durations_s=[
                        [cycle["duration_s"] for cycle in world["cycles"]]
                        for world in worlds]),
        worlds=worlds,
        totals=dict(cycle_count=len(all_cycles), **counts,
                    mean_signed_endpoint_residual_rad=(
                        sum(x["signed_endpoint_residual_rad"] for x in all_cycles) / len(all_cycles)
                        if all_cycles else None),
                    mean_max_abs_residual_rad=(
                        sum(x["max_abs_residual_rad"] for x in all_cycles) / len(all_cycles)
                        if all_cycles else None)),
        interpretation=("descriptive-only; cycle and phase associations do not establish causation"),
        timing_contact_force_lag_caveat=(
            "contact_found is sampled endpoint state; force/contact timing may lag the "
            "kinematic interval, so phase labels are descriptive"),
        policy_acceptance=False, physical_motion_authorized=False,
    )
