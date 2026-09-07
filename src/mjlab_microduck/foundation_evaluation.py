"""One ordered parent/candidate F1 diagnostic, without stage promotion."""

import argparse
import datetime as dt
import json

from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, require, sha256, canonical
from mjlab_microduck.foundation_pilot import ROOT, ACTOR, DEADLINE, PROTOCOL, runtime_identity
from mjlab_microduck.recovery_ab import verify_source, write_new
from mjlab_microduck.rollout_repeatability import check_host
from mjlab_microduck.speed_response_control import run_control

SEEDS = (397, 401, 409)
OUTPUT = ROOT / "artifacts/evaluations" / PROTOCOL


def validate_speed_classification(report):
    """Validate redundant speed labels against vectors and first-attempt rows.

    This is consistency checking, not reconstruction of missing raw samples.
    Retained reports still require the separate hash/identity checks.
    """
    require(report["speed_mps"] == .3 and report["step_dt_s"] == .02
            and report["num_envs"] == 8, "fixed F1 speed/timestep/environment contract")
    if report["safety_failures"]:
        require(report["classification"] == "safety-or-coverage-stop", "unsafe report cannot claim tracking")
        return
    require(report["sample_steps"] == 400 and report["startup_steps"] == 100
            and report["settled_steps"] == 300 and report["terminal_steps"] == [], "complete safe response")
    for name, steps in (("all", 400), ("settled", 300)):
        group = report["groups"][name]
        require(group["steps"] == steps, "motor window coverage")
        require(type(group["legacy_torque_p99"]) in (int, float)
                and 0 <= group["legacy_torque_p99"] <= .60, "safe label agrees with absolute torque gate")
        for key in ("legacy_rated_speed_exceed_fraction", "pre_reset_rated_speed_exceed_fraction"):
            require(type(group[key]) in (int, float) and group[key] == 0.,
                    "safe label agrees with zero rated-speed exceedance")
    settled = report["groups"]["settled"]
    flags = []
    for key in ("body_forward_per_env_mean", "route_forward_per_env_mean"):
        values = settled[key]
        require(type(values) is list and len(values) == 8
                and all(type(v) in (int, float) for v in values), "eight numeric speed means")
        flags.append(all(abs(v-.3) <= .03 for v in values))
    measurement = report["stable_route_response"]
    require(measurement["nominal_speed_mps"] == .3 and measurement["step_dt_s"] == .02
            and measurement["speed_tolerance_mps"] == .03 and measurement["stable_span_s"] == .5
            and measurement["deadline_s"] == 2., "unchanged response window")
    rows = measurement["environments"]
    require(type(rows) is list and len(rows) == 8, "eight first-attempt window rows")
    statuses = []
    for i, row in enumerate(rows):
        require(type(row["environment"]) is int and row["environment"] == i
                and row["terminal"] is False and type(row["first_recovery_step"]) is int
                and row["first_recovery_step"] == 100
                and abs(row["sampled_recovery_span_s"]-5.98) < 1e-10, "complete first-attempt identity/span")
        latency = row["stable_recovery_latency_s"]
        require(latency is None or type(latency) in (int, float)
                and .5-1e-12 <= latency <= 5.98+1e-12, "valid sampled stable latency")
        require(latency is None or abs(latency/.02-round(latency/.02)) < 1e-9, "latency on control-step grid")
        status = "recovered-in-window" if latency is not None and latency <= 2.+1e-12 else "window-missed"
        require(row["status"] == status, "window label matches sampled latency")
        statuses.append(status)
    expected_counts = {s: statuses.count(s) for s in
                       ("not-observed", "recovered-in-window", "window-missed", "censored-before-window")}
    require(measurement["counts"] == expected_counts
            and all(type(v) is int for v in measurement["counts"].values()), "window count reconciliation")
    body_ok, route_ok = flags
    stable = expected_counts["recovered-in-window"] == 8
    require(report["body_mean_in_band_all_envs"] is body_ok
            and report["route_mean_in_band_all_envs"] is route_ok
            and report["stable_route_window_all_envs"] is stable, "speed flags agree with measurements")
    derived = ("straight-body-mean-outside-band" if not body_ok else
        "body-route-response-diverge" if not route_ok else
        "mean-tracking-but-instantaneous-window-missed" if not stable else "straight-response-within-both-criteria")
    require(report["classification"] == derived, "speed classification agrees with measurements")


def candidate_failures(report, parent):
    canonical(report); canonical(parent)  # reject NaN/Infinity before comparisons
    require(report["protocol"] == parent["protocol"] == PROTOCOL
            and report["seed"] == parent["seed"] in SEEDS, "paired F1 identity")
    require(report["num_envs"] == parent["num_envs"] == 8, "eight held-out environments")
    validate_speed_classification(report)
    validate_speed_classification(parent)
    failures = list(report["safety_failures"])
    if report["classification"] != "straight-response-within-both-criteria":
        failures.append(report["classification"])
    if "settled" not in report["groups"]: return failures + ["no-settled-evidence"]
    c, p = report["groups"]["settled"], parent["groups"]["settled"]
    if c["heading_abs_max"] > .25: failures.append("heading-drift")
    if any(x > .05 for x in c["cross_route_abs_per_env_mean"]): failures.append("cross-route-motion")
    if c["legacy_torque_p99"] > p["legacy_torque_p99"]+.02: failures.append("torque-nonregression")
    if c["cross_route_abs_mean"] > p["cross_route_abs_mean"]+.01: failures.append("lateral-nonregression")
    if c["heading_abs_max"] > p["heading_abs_max"]+.10: failures.append("heading-nonregression")
    for key in ("body_forward_per_env_mean", "route_forward_per_env_mean"):
        require(len(c[key]) == len(p[key]) == 8, "paired velocity coverage")
        if any(abs(x-.3) > abs(y-.3)+.01 for x, y in zip(c[key], p[key])):
            failures.append(key+"-nonregression")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    verify_source(args.source)
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=6) < DEADLINE, "evaluation closeout budget")
    runtime, host = runtime_identity(), check_host()
    training_dir = ROOT / "artifacts/training" / f"{PROTOCOL}-pilot"
    training = json.loads((training_dir / "result.json").read_text())
    candidate = training_dir / "model_8498.pt"
    require(training["status"] == "training-complete-not-accepted" and training["updates"] == 500
            and training["source"] == args.source and training["final_checkpoint"] == str(candidate),
            "same-source complete fixed candidate")
    require(sha256(candidate) == training["final_sha256"] and sha256(ACTOR) == ACTOR_SHA256, "model hashes")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT / "launch.json", dict(source=args.source, protocol=PROTOCOL,
        runtime=runtime, host=host, parent_sha256=ACTOR_SHA256,
        candidate_sha256=training["final_sha256"], seeds=SEEDS))
    retained, parents, failures = [], {}, []
    try:
        for arm, checkpoint in (("parent", ACTOR), ("candidate", candidate)):
            for seed in SEEDS:
                require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=2) < DEADLINE, "case closeout budget")
                report = run_control(checkpoint=checkpoint, seed=seed, protocol=PROTOCOL)
                path = OUTPUT / f"{arm}-s{seed}.json"
                write_new(path, report)
                retained.append(dict(arm=arm, seed=seed, path=str(path), sha256=sha256(path)))
                failures = report["safety_failures"] if arm == "parent" else candidate_failures(report, parents[seed])
                if failures: break
                if arm == "parent": parents[seed] = report
            if failures: break
        write_new(OUTPUT / "decision.json", dict(protocol=PROTOCOL, source=args.source,
            decision="numerical-gate-stop" if failures else "single-seed-pilot-support-only",
            failures=failures, reports=retained, policy_acceptance=False,
            f2_admitted=False, obstacles_admitted=False, physical_motion_authorized=False))
    except Exception as exc:
        write_new(OUTPUT / "failure.json", dict(type=type(exc).__name__, error=str(exc), reports=retained))
        raise


if __name__ == "__main__":
    main()
