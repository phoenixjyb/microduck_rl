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


def candidate_failures(report, parent):
    canonical(report); canonical(parent)  # reject NaN/Infinity before comparisons
    require(report["protocol"] == parent["protocol"] == PROTOCOL
            and report["seed"] == parent["seed"] in SEEDS, "paired F1 identity")
    require(report["num_envs"] == parent["num_envs"] == 8, "eight held-out environments")
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
