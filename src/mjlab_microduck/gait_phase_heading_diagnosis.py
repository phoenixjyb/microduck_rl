"""Offline, descriptive contact-phase heading residuals for retained gait paths."""

from __future__ import annotations

import math
from collections.abc import Mapping

from mjlab_microduck import gait_path_qualification as route


PROTOCOL = "microduck-gait-phase-heading-diagnosis-v1"
PHASES = ("double-support", "left-only", "right-only", "flight", "contact-transition")
START_STEP = 60
END_STEP = 239  # final sample is 239; settled interval starts are steps 60 through 238.


def _new_bin() -> dict:
    return dict(interval_count=0, duration_s=0.0, heading_delta_rad=0.0,
                signed_residual_rad=0.0, absolute_residual_rad=0.0)


def _finish_bin(item: dict) -> dict:
    count, duration = item["interval_count"], item["duration_s"]
    return {**item,
            "mean_signed_residual_rate_rad_s":
                item["signed_residual_rad"] / duration if duration else None,
            "mean_absolute_residual_rate_rad_s":
                item["absolute_residual_rad"] / duration if duration else None,
            "mean_signed_residual_per_interval_rad":
                item["signed_residual_rad"] / count if count else None,
            "mean_absolute_residual_per_interval_rad":
                item["absolute_residual_rad"] / count if count else None}


def _phase(contact: list, world: int, before: int, after: int) -> str:
    a = (contact[before][world][0] > 0.0, contact[before][world][1] > 0.0)
    b = (contact[after][world][0] > 0.0, contact[after][world][1] > 0.0)
    if a != b:
        return "contact-transition"
    return {(True, True): "double-support", (True, False): "left-only",
            (False, True): "right-only", (False, False): "flight"}[a]


def summarize_case(case: Mapping, trace: Mapping) -> dict:
    """Partition every settled interval by stable endpoint contact and heading residual.

    Calling the established route summarizer first provides the retained trace's
    strict fixed-protocol validation and the independent full-settled closure value.
    This function does not modify either input or perform IO.
    """
    route_report = route.summarize_case(case, trace)
    worlds = []
    group_bins = {phase: _new_bin() for phase in PHASES}
    phase_world_items = {phase: [] for phase in PHASES}
    closure_signed = 0.0
    closure_duration = 0.0

    for world in range(route.WORLDS):
        bins = {phase: _new_bin() for phase in PHASES}
        for step in range(START_STEP, END_STEP):
            dt = trace["physics_time"][step + 1][world] - trace["physics_time"][step][world]
            # Strict route validation enforces monotonic physics_time; retain a
            # local guard so no invalid arithmetic can leak into this report.
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("settled interval duration must be positive and finite")
            before = route._heading(trace["quat_wxyz"][step][world])
            after = route._heading(trace["quat_wxyz"][step + 1][world])
            delta = route._wrap(after - before)
            residual = delta - case["yaw"] * dt
            phase = _phase(trace["contact_found"], world, step, step + 1)
            target = bins[phase]
            target["interval_count"] += 1
            target["duration_s"] += dt
            target["heading_delta_rad"] += delta
            target["signed_residual_rad"] += residual
            target["absolute_residual_rad"] += abs(residual)

        phase_reports = {}
        for phase in PHASES:
            finished = _finish_bin(bins[phase])
            phase_reports[phase] = finished
            aggregate = group_bins[phase]
            for key in ("interval_count", "duration_s", "heading_delta_rad",
                        "signed_residual_rad", "absolute_residual_rad"):
                aggregate[key] += finished[key]
            phase_world_items[phase].append(finished)

        full = route_report["worlds"][world]["full_settled"]
        signed = sum(item["signed_residual_rad"] for item in bins.values())
        duration = sum(item["duration_s"] for item in bins.values())
        if not math.isclose(signed, full["heading_error_rad"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("phase residuals do not close to route full-settled heading error")
        if not math.isclose(duration, full["duration_s"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("phase durations do not close to route full-settled duration")
        closure_signed += signed
        closure_duration += duration
        worlds.append(dict(world=world, phases=phase_reports,
                           settled_interval_count=sum(x["interval_count"] for x in bins.values()),
                           settled_duration_s=duration,
                           signed_residual_rad=signed,
                           absolute_residual_rad=sum(x["absolute_residual_rad"] for x in bins.values()),
                           full_settled_heading_error_rad=full["heading_error_rad"],
                           full_settled_duration_s=full["duration_s"],
                           signed_endpoint_rate_rad_s=(
                               full["heading_error_rad"] / full["duration_s"]),
                           absolute_endpoint_rate_rad_s=(
                               abs(full["heading_error_rad"]) / full["duration_s"])))

    group = {}
    for phase in PHASES:
        pooled = _finish_bin(group_bins[phase])
        per_world = phase_world_items[phase]
        # Equal-world means for totals; pooled rates are duration weighted.
        pooled["equal_world_mean_signed_residual_rad"] = sum(
            item["signed_residual_rad"] for item in per_world) / route.WORLDS
        pooled["equal_world_mean_absolute_residual_rad"] = sum(
            item["absolute_residual_rad"] for item in per_world) / route.WORLDS
        group[phase] = pooled

    endpoint_abs_mean = sum(abs(item["full_settled_heading_error_rad"])
                            for item in worlds) / route.WORLDS
    mean_signed_endpoint_rate = sum(item["signed_endpoint_rate_rad_s"]
                                    for item in worlds) / route.WORLDS
    mean_absolute_endpoint_rate = sum(item["absolute_endpoint_rate_rad_s"]
                                      for item in worlds) / route.WORLDS
    nominal_duration = route.STEP_DT * (END_STEP - START_STEP)
    return dict(
        protocol=PROTOCOL, route_protocol=route.PROTOCOL,
        checkpoint_sha256=case["checkpoint_sha256"], seed=case["seed"],
        speed=case["speed"], yaw=case["yaw"],
        interval_steps=[START_STEP, END_STEP], intervals_per_world=END_STEP - START_STEP,
        timing=dict(step_dt=trace["step_dt"], startup_steps=trace["startup_steps"],
                    start_physics_time_s=[trace["physics_time"][START_STEP][w]
                                          for w in range(route.WORLDS)],
                    settled_duration_s_per_world=[
                        item["settled_duration_s"] for item in worlds],
                    pooled_world_duration_s=closure_duration),
        phases=group, worlds=worlds,
        totals=dict(pooled_signed_residual_rad=closure_signed,
                    pooled_absolute_interval_residual_rad=sum(
                        item["absolute_residual_rad"] for item in worlds),
                    mean_per_world_absolute_endpoint_error_rad=endpoint_abs_mean,
                    equal_world_mean_signed_endpoint_rate_rad_s=mean_signed_endpoint_rate,
                    equal_world_mean_absolute_endpoint_rate_rad_s=mean_absolute_endpoint_rate,
                    expected_duration_s=nominal_duration,
                    pooled_world_duration_s=closure_duration,
                    reconciles_full_settled_heading_error=math.isclose(
                        closure_signed, sum(x["full_settled_heading_error_rad"] for x in worlds),
                        rel_tol=1e-9, abs_tol=1e-9)),
        interpretation="descriptive-only; phase association does not establish causation",
        timing_contact_force_lag_caveat=(
            "contact_found is sampled endpoint state; force/contact timing may lag the "
            "kinematic interval, so phase labels are descriptive"),
        policy_acceptance=False, physical_motion_authorized=False)
