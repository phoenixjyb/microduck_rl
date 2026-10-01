"""Walker-only control scoring for the predeclared C1-S substitution probe.

This adapter deliberately reuses the frozen C1 trace validator and measurements,
but does not apply the hop scorer's overall decision to a no-hop control.
"""

from __future__ import annotations

from . import community_hop_rehearsal as c1


EXPERIMENT_ID = "community-hop-c1s-velstand-substitution-v1"
MEASUREMENT_PROTOCOL = c1.PROTOCOL
BASELINE_DECISION = "walk-only-indicators-pass"
REJECTED_DECISION = "walk-only-indicators-rejected"

SHARED_GATES = (
    "complete_first_attempt",
    "no_fatal_event",
    "approach_speed_heading",
    "stable_entry",
    "hop_settle_drift_100mm",
    "resume_speed_heading",
    "modeled_current_1p75a",
    "no_soft_limit_exposure",
    "finite_derived_peaks",
)


def score_walk_only_trace(trace: dict) -> dict:
    """Score a walk-only trace using shared C1 measurements and baseline gates."""
    measured = c1.score_trace(trace)
    rows = trace["samples"][:measured["scored_samples"]]
    final_window = [r for r in rows if 8.0 <= r["t_s"] < 9.0]
    final_window_complete = len(final_window) == round(1.0 / c1.PHYSICS_DT_S)
    stable_final = (final_window_complete and all(c1._stable(r) for r in final_window))
    zero_airborne = measured["metrics"]["qualified_airborne_episode_count"] == 0

    gates = {name: measured["gates"][name] for name in SHARED_GATES}
    gates["stable_final_zero_command_window"] = stable_final
    gates["zero_qualified_airborne_episodes_4_9"] = zero_airborne

    # Keep fatal-prefix evidence intact. Otherwise identify the first failing
    # gate in declared order; sample-local gates point to their first bad sample.
    failure = measured["first_failure"]
    if failure is None:
        failed = next((name for name, passed in gates.items() if not passed), None)
        if failed is not None:
            if failed == "stable_final_zero_command_window":
                bad = next((r for r in final_window if not c1._stable(r)), None)
                sample = trace["samples"].index(bad) if bad is not None else None
                t_s = bad["t_s"] if bad is not None else 8.0
            elif failed == "zero_qualified_airborne_episodes_4_9":
                flight = measured["metrics"]["qualified_flight_t_s"]
                bad = next((r for r in rows if flight is not None and r["t_s"] == flight), None)
                sample = trace["samples"].index(bad) if bad is not None else None
                t_s = bad["t_s"] if bad is not None else None
            else:
                sample, t_s = None, None
            failure = {"sample": sample, "t_s": t_s, "reason": failed}

    return {
        "experiment_id": EXPERIMENT_ID,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "decision": BASELINE_DECISION if all(gates.values()) else REJECTED_DECISION,
        "gates": gates,
        "metrics": {
            **measured["metrics"],
            "stable_final_zero_command_window_samples": len(final_window),
        },
        "first_failure": failure,
        "scored_samples": measured["scored_samples"],
        "ignored_samples": measured["ignored_samples"],
        **{name: False for name in (
            "impact_limit_calibrated", "thermal_model_verified", "trace_authenticated",
            "plant_runtime_binding_verified", "behavioral_acceptance", "training_authorized",
            "transition_authorized", "physical_motion_authorized",
        )},
    }
