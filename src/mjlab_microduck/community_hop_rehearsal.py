"""Preparation and first-attempt scoring for a separate stock Happy Hop probe.

No policy execution, simulator, training, or physical admission lives here.
The fixed C1 protocol is deliberately independent of the sprung-foot H1 gates.
Trace measurements are declarations, not authenticated simulator evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


PROTOCOL = "community-hop-c1-first-attempt-v1"
TRACE_SCHEMA = "community-hop-c1-trace-v1"
CONTROL_DT_S = 0.02
PHYSICS_DT_S = 0.005
FORWARD_COMMAND_M_S = 0.2
# Fixed, predeclared diagnostic schedule, not a controller admission.
PHASES = (
    ("walk", 3.0),
    ("entry_settle", 1.0),
    ("hop", 3.0),
    ("landing_settle", 2.0),
    ("resume_walk", 3.0),
)
TOTAL_TIME_S = sum(duration for _, duration in PHASES)
TOTAL_SAMPLES = round(TOTAL_TIME_S / PHYSICS_DT_S) + 1
VECTOR_FIELDS = {
    "command": 13,
    "action_raw": 14,
    "action_applied": 14,
    "base_xy_m": 2,
    "base_velocity_world_m_s": 3,
    "base_roll_pitch_yaw_rad": 3,
    "base_angular_velocity_rad_s": 3,
    "foot_clearance_m": 2,
    "foot_normal_force_n": 2,
    "motor_current_a": 14,
    "motor_torque_nm": 14,
    "motor_velocity_rad_s": 14,
}
BOOL_VECTOR_FIELDS = {"foot_contact": 2, "soft_limit_exposed": 14}
SAMPLE_FIELDS = {*VECTOR_FIELDS, *BOOL_VECTOR_FIELDS, "t_s", "body_contact", "reset_event"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _numbers(values: Sequence, size: int, name: str, *, finite: bool = True) -> list[float]:
    _require(isinstance(values, (list, tuple)) and len(values) == size,
             f"{name}: expected {size} numbers")
    _require(all(type(value) in (int, float) for value in values),
             f"{name}: numeric values required, not booleans")
    result = [float(value) for value in values]
    if finite:
        _require(all(math.isfinite(value) for value in result), f"{name}: nonfinite value")
    return result


def phase_at(t_s: float) -> str:
    _require(type(t_s) in (int, float) and math.isfinite(t_s)
             and 0.0 <= t_s <= TOTAL_TIME_S, "time outside fixed C1 schedule")
    end = 0.0
    for name, duration in PHASES:
        end += duration
        if t_s < end:
            return name
    return PHASES[-1][0]  # Inclusive terminal state at 12 s; no new action.


def command_at(t_s: float) -> list[float]:
    return [FORWARD_COMMAND_M_S if phase_at(t_s) in ("walk", "resume_walk") else 0.0,
            *([0.0] * 12)]


class OneStepActionDelay:
    """One per-attempt history shared by walk and hop; no switch-time reset.

    The observation sees the last RAW action, and the plant receives that same
    old action while the freshly inferred action is saved for the next tick.
    Starting at HOME means the initial raw/delayed offset is zero. This helper
    proposes a contract; matching the author's unpublished handoff is separate.
    """

    def __init__(self) -> None:
        self._last_raw = [0.0] * 14

    @property
    def previous_action(self) -> list[float]:
        return self._last_raw.copy()

    def submit(self, raw_action: Sequence) -> list[float]:
        current = _numbers(raw_action, 14, "action")
        delayed = self.previous_action
        self._last_raw = current
        return delayed


def pack_observation(*, angular_velocity, projected_gravity, servo_position,
                     backlash_position, servo_velocity, backlash_velocity,
                     home_position, previous_raw_action, command) -> list[float]:
    """Unscaled 61-value proposed encoder view, before a baked normalizer.

    Ordered servo arrays must be bound to the compiled plant separately. The
    firmware must use output-side POSITION feedback but motor-side velocity for
    back-EMF/friction; this observation helper is not a BAM torque controller.
    """
    ang = _numbers(angular_velocity, 3, "angular_velocity")
    gravity = _numbers(projected_gravity, 3, "projected_gravity")
    _require(abs(math.sqrt(sum(x * x for x in gravity)) - 1.0) <= 1e-5,
             "projected_gravity must be a unit vector")
    q = _numbers(servo_position, 14, "servo_position")
    bq = _numbers(backlash_position, 14, "backlash_position")
    dq = _numbers(servo_velocity, 14, "servo_velocity")
    bdq = _numbers(backlash_velocity, 14, "backlash_velocity")
    home = _numbers(home_position, 14, "home_position")
    action = _numbers(previous_raw_action, 14, "previous_raw_action")
    cmd = _numbers(command, 13, "command")
    return (ang + gravity + [x + b - h for x, b, h in zip(q, bq, home)]
            + [x + b for x, b in zip(dq, bdq)] + action + cmd)


def _validate_sample(sample: object, index: int) -> bool:
    _require(type(sample) is dict and set(sample) == SAMPLE_FIELDS,
             f"sample {index}: exact trace fields required")
    t = sample["t_s"]
    _require(type(t) in (int, float) and math.isfinite(t)
             and abs(t - index * PHYSICS_DT_S) <= 1e-9,
             f"sample {index}: contiguous 5 ms timestamps from zero required")
    finite = True
    for name, size in VECTOR_FIELDS.items():
        values = _numbers(sample[name], size, name, finite=False)
        finite &= all(math.isfinite(x) for x in values)
    for name, size in BOOL_VECTOR_FIELDS.items():
        _require(type(sample[name]) is list and len(sample[name]) == size
                 and all(type(x) is bool for x in sample[name]), f"{name}: exact boolean vector")
    for name in ("body_contact", "reset_event"):
        _require(type(sample[name]) is bool, f"{name}: boolean required")
    if finite:
        # Penetration may produce negative clearance, but normal FORCE is a
        # nonnegative magnitude, never a signed or missing flag.
        _require(all(x >= 0.0 for x in sample["foot_normal_force_n"]),
                 "foot_normal_force_n: negative magnitude")
    return finite


def _stable(row: dict) -> bool:
    return (all(row["foot_contact"])
            and math.hypot(*row["base_velocity_world_m_s"][:2]) <= 0.05
            and abs(row["base_velocity_world_m_s"][2]) <= 0.05
            and max(abs(x) for x in row["base_roll_pitch_yaw_rad"][:2]) <= math.radians(10)
            and math.sqrt(sum(x * x for x in row["base_angular_velocity_rad_s"])) <= 0.5)


def _entry_ready(rows: list[dict]) -> bool:
    # Called at the switch boundary on already validated contiguous substeps.
    return (len(rows) >= 200
            and all(r["command"] == [0.0] * 13 for r in rows[-200:])
            and all(map(_stable, rows[-100:])))


def score_trace(trace: dict) -> dict:
    """Score only the first attempt, without creating any admission authority.

    Truncated runs fail completion. After the first fatal event, later samples
    are ignored entirely, preventing a reset/retry from improving the result.
    Substep measurements are required, not 50 Hz rendering snapshots.
    """
    _require(type(trace) is dict and set(trace) == {"schema", "samples"}
             and trace["schema"] == TRACE_SCHEMA, "exact C1 trace schema required")
    samples = trace["samples"]
    _require(type(samples) is list and 0 < len(samples) <= TOTAL_SAMPLES,
             "one bounded nonempty attempt required")
    rows = []
    failure = None
    delay = OneStepActionDelay()
    raw = applied = None
    for index, sample in enumerate(samples):
        finite = _validate_sample(sample, index)
        reason = None
        if not finite:
            reason = "nonfinite"
        elif sample["reset_event"]:
            reason = "reset-event"
        elif sample["body_contact"]:
            reason = "non-foot-body-contact"
        elif max(abs(x) for x in sample["base_roll_pitch_yaw_rad"][:2]) >= math.radians(60):
            reason = "fall-tilt"
        elif sample["command"] != command_at(sample["t_s"]):
            reason = "command-contract"
        elif index == 800 and not _entry_ready(rows):
            reason = "unstable-hop-entry"
        else:
            if index % 4 == 0 and index < TOTAL_SAMPLES - 1:
                raw = sample["action_raw"].copy()
                applied = delay.submit(raw)
            if sample["action_raw"] != raw or sample["action_applied"] != applied:
                reason = "one-step-action-delay-or-hold"
        if reason:
            failure = {"sample": index, "t_s": sample["t_s"], "reason": reason}
            break
        rows.append(sample)

    def window(start: float, end: float) -> list[dict]:
        # Integer-grid selection avoids floating-point boundary drift in a
        # touchdown+100ms window, e.g. 4.605+0.1 rounding either side of 4.705.
        first = math.ceil(start / PHYSICS_DT_S - 1e-8)
        last = math.ceil(end / PHYSICS_DT_S - 1e-8)
        return rows[first:last]

    def stable_window(start: float, end: float) -> bool:
        selected = window(start, end)
        return len(selected) == round((end - start) / PHYSICS_DT_S) and all(map(_stable, selected))

    def walk_metrics(start: float, end: float) -> dict:
        selected = window(start, end)
        result = {"complete": len(selected) == round((end - start) / PHYSICS_DT_S)}
        if not selected:
            return {**result, "forward_mae_m_s": None, "lateral_peak_m_s": None,
                    "heading_peak_rad": None}
        return {**result,
                "forward_mae_m_s": sum(abs(r["base_velocity_world_m_s"][0]
                                            - FORWARD_COMMAND_M_S) for r in selected) / len(selected),
                "lateral_peak_m_s": max(abs(r["base_velocity_world_m_s"][1]) for r in selected),
                "heading_peak_rad": max(abs(r["base_roll_pitch_yaw_rad"][2]) for r in selected)}

    def walk_ok(metrics: dict) -> bool:
        return (metrics["complete"] and metrics["forward_mae_m_s"] <= 0.08
                and metrics["lateral_peak_m_s"] <= 0.05
                and metrics["heading_peak_rad"] <= math.radians(15))

    approach, resume = walk_metrics(2.0, 3.0), walk_metrics(11.0, 12.0)
    hop_rows = window(4.0, 9.0)
    run, flight_end, flight_count, episode_qualified = 0, None, 0, False
    bilateral_peak = None
    for row in hop_rows:
        clearance = min(row["foot_clearance_m"])
        bilateral_peak = clearance if bilateral_peak is None else max(bilateral_peak, clearance)
        if any(row["foot_contact"]):
            run, episode_qualified = 0, False
        elif clearance >= 0.03:
            run += 1
            if run >= 3 and not episode_qualified:  # Three states span at least 10 ms.
                flight_count += 1
                episode_qualified = True
                if flight_end is None and row["t_s"] < 7.0:
                    flight_end = row["t_s"]
        else:
            run = 0
    landing = next((r["t_s"] for r in window(4.0, 9.0)
                    if flight_end is not None and r["t_s"] > flight_end
                    and any(r["foot_contact"])), None)
    impact = window(landing, landing + 0.1) if landing is not None else []
    entry = next((r for r in rows if abs(r["t_s"] - 4.0) <= 1e-9), None)
    negotiation = window(4.0, 9.0)
    drift = (max(math.dist(r["base_xy_m"], entry["base_xy_m"]) for r in negotiation)
             if entry is not None and negotiation else None)
    current_peak = max((abs(x) for r in rows for x in r["motor_current_a"]), default=None)
    soft_exposed = any(any(r["soft_limit_exposed"]) for r in rows)
    impact_peak = max((sum(r["foot_normal_force_n"]) for r in impact), default=None)
    negotiation_force_peak = max((sum(r["foot_normal_force_n"]) for r in negotiation), default=None)
    power_peak = max((sum(abs(t * v) for t, v in zip(r["motor_torque_nm"],
                      r["motor_velocity_rad_s"])) for r in rows), default=None)
    gates = {
        "complete_first_attempt": failure is None and len(rows) == TOTAL_SAMPLES,
        "no_fatal_event": failure is None,
        "approach_speed_heading": walk_ok(approach),
        "stable_entry": stable_window(3.5, 4.0),
        "bilateral_30mm_flight_10ms": flight_end is not None,
        "exactly_one_qualified_hop": flight_count == 1,
        "landed_after_qualified_flight": landing is not None,
        "landing_100ms_impact_window_complete": len(impact) == 20,
        "settled_after_landing": landing is not None and stable_window(8.0, 9.0),
        "hop_settle_drift_100mm": drift is not None and drift <= 0.1,
        "resume_speed_heading": walk_ok(resume),
        "modeled_current_1p75a": current_peak is not None and current_peak <= 1.75 + 1e-6,
        "no_soft_limit_exposure": bool(rows) and not soft_exposed,
        "finite_derived_peaks": all(x is None or math.isfinite(x)
                                    for x in (impact_peak, negotiation_force_peak, power_peak, drift)),
    }
    return {
        "protocol": PROTOCOL,
        "decision": "declared-trace-indicators-pass" if all(gates.values()) else "declared-trace-rejected",
        "gates": gates,
        "first_failure": failure,
        "scored_samples": len(rows),
        "ignored_samples": len(samples) - len(rows) - (1 if failure else 0),
        "metrics": {
            "approach": approach, "resume": resume,
            "bilateral_clearance_peak_m": bilateral_peak,
            "qualified_flight_t_s": flight_end, "touchdown_t_s": landing,
            "qualified_airborne_episode_count": flight_count,
            "hop_settle_drift_peak_m": drift if drift is None or math.isfinite(drift) else None,
            "landing_100ms_force_sum_peak_n": impact_peak if impact_peak is None or math.isfinite(impact_peak) else None,
            "hop_settle_force_sum_peak_n": negotiation_force_peak if negotiation_force_peak is None or math.isfinite(negotiation_force_peak) else None,
            "landing_100ms_window_complete": len(impact) == 20,
            "motor_current_abs_peak_a": current_peak,
            "motor_torque_abs_peak_nm": max((abs(x) for r in rows for x in r["motor_torque_nm"]), default=None),
            "motor_velocity_abs_peak_rad_s": max((abs(x) for r in rows for x in r["motor_velocity_rad_s"]), default=None),
            "mechanical_power_abs_sum_peak_w": power_peak if power_peak is None or math.isfinite(power_peak) else None,
            "soft_limit_exposed_substeps": sum(any(r["soft_limit_exposed"]) for r in rows),
        },
        "impact_limit_calibrated": False,
        "thermal_model_verified": False,
        "trace_authenticated": False,
        "plant_runtime_binding_verified": False,
        "behavioral_acceptance": False,
        "training_authorized": False,
        "transition_authorized": False,
        "physical_motion_authorized": False,
    }
