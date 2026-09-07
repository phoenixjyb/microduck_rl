"""One matched F1 reward-width diagnostic; no curriculum promotion or retries."""

import argparse
import datetime as dt
import functools
import json
import math
import os
import subprocess
import sys
import time

from mjlab_microduck import foundation_pilot as training
from mjlab_microduck.foundation_evaluation import candidate_failures, validate_speed_classification
from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, require, sha256, canonical
from mjlab_microduck.recovery_ab import verify_source, write_new
from mjlab_microduck.rollout_repeatability import ROOT, ACTOR, check_host
from mjlab_microduck.speed_response_control import run_control

PROTOCOL = "f1r-width-paired-s421-v1"
OUTPUT = ROOT / "artifacts/experiments" / PROTOCOL
ARMS = ("control", "narrow")
VARIANCES = dict(control=.15, narrow=.05)
SEEDS = (431, 433, 439)
TRAIN_SEED, SMOKE_SEED = 421, 419
MODES = dict(smoke=(10, 180), pilot=(500, 900))
CAMPAIGN_SECONDS = 3150
LAST_START = dt.datetime(2026, 9, 7, 3, tzinfo=dt.timezone.utc)


def prepare_config(mode, *, arm):
    require(arm in ARMS and mode in MODES, "predeclared reward arm/mode")
    cfg, agent = training.prepare_config(mode)
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(VARIANCES[arm])
    cfg.seed = agent.seed = SMOKE_SEED if mode == "smoke" else TRAIN_SEED
    agent.max_iterations = MODES[mode][0]
    agent.experiment_name, agent.run_name = PROTOCOL, f"{arm}-{mode}"
    return cfg, agent


def paired_decision(parent, control, narrow):
    """Ordered first-case gate and descriptive paired effects; never acceptance."""
    canonical([parent, control, narrow])
    for report in (parent, control, narrow):
        require(report["protocol"] == PROTOCOL and report["seed"] == parent["seed"] in SEEDS,
                "matched report identity")
        validate_speed_classification(report)
    require(not parent["safety_failures"] and not control["safety_failures"], "safe paired references")
    control_failures = candidate_failures(control, parent, protocol=PROTOCOL, seeds=SEEDS)
    failures = candidate_failures(narrow, parent, protocol=PROTOCOL, seeds=SEEDS)
    # Retain the historical absolute gates and also compare to matched continuation.
    vs_control = candidate_failures(narrow, control, protocol=PROTOCOL, seeds=SEEDS)
    failures += ["matched-control:"+f for f in vs_control if f.endswith("nonregression")]
    effects = {}
    if "settled" in narrow["groups"]:
        c, n = control["groups"]["settled"], narrow["groups"]["settled"]
        for key in ("body_forward_mean", "route_forward_mean", "legacy_torque_p99",
                    "cross_route_abs_mean", "heading_abs_max",
                    "pre_reset_squared_utilization_mean", "pre_reset_mechanical_abs_power_mean_w"):
            effects[key+"_narrow_minus_control"] = n[key]-c[key]
    return dict(seed=parent["seed"], failures=failures, control_failures=control_failures,
                narrow_vs_control_failures=vs_control, descriptive_effects=effects,
                policy_acceptance=False, independent_training_seeds=1)


def child(kind, arm, seed, source):
    """Each child owns and releases one CUDA context; parent never initializes CUDA."""
    verify_source(source)
    training.runtime_identity()
    check_host()
    require(sha256(ACTOR) == ACTOR_SHA256, "original parent hash")
    if kind in MODES:
        path = OUTPUT / f"{arm}-{kind}"
        path.mkdir(exist_ok=False)
        write_new(path / "launch.json", dict(protocol=PROTOCOL, source=source, arm=arm,
            mode=kind, seed=SMOKE_SEED if kind == "smoke" else TRAIN_SEED,
            variance=VARIANCES[arm], parent_sha256=ACTOR_SHA256))
        result = training.train(kind, path,
            config_factory=functools.partial(prepare_config, arm=arm), protocol=PROTOCOL,
            stop_at=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=MODES[kind][1]),
            max_seconds=MODES[kind][1])
        write_new(path / "result.json", {**result, "source":source, "policy_acceptance":False})
    else:
        require(kind == "evaluate" and seed in SEEDS, "predeclared evaluation")
        checkpoint = ACTOR if arm == "parent" else OUTPUT / f"{arm}-pilot/model_8498.pt"
        if arm != "parent":
            result = json.loads((checkpoint.parent / "result.json").read_text())
            require(result["source"] == source and result["updates"] == 500
                    and result["common_step"] == 204000
                    and result["final_sha256"] == sha256(checkpoint), "complete same-source arm")
        report = run_control(checkpoint=checkpoint, seed=seed, protocol=PROTOCOL, retain_route_trace=True)
        report["source"] = source
        write_new(OUTPUT / f"{arm}-s{seed}.json", report)


def run_child(kind, arm, source, end, *, seed=SEEDS[0]):
    seconds = MODES[kind][1] if kind in MODES else 90
    require(time.monotonic()+seconds+30 < end, "remaining campaign budget")
    check_host()
    command = [sys.executable, "-m", "mjlab_microduck.foundation_reward_experiment", "--source", source,
               "--child", kind, "--arm", arm, "--seed", str(seed)]
    name = f"{arm}-{kind}" if kind in MODES else f"{arm}-s{seed}"
    with (OUTPUT / f"{name}.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            timeout=seconds, env={**os.environ, "CUDA_VISIBLE_DEVICES":"0", "OMP_NUM_THREADS":"1",
                                 "PYTHONUNBUFFERED":"1"})
    require(completed.returncode == 0, f"{name} child exit {completed.returncode}")


def campaign(source):
    verify_source(source)
    require(dt.datetime.now(dt.timezone.utc) < LAST_START, "one-shot start window")
    runtime, host = training.runtime_identity(), check_host()
    require(sha256(ACTOR) == ACTOR_SHA256, "original parent hash")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT / "launch.json", dict(protocol=PROTOCOL, source=source,
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(), runtime=runtime, host=host,
        parent_sha256=ACTOR_SHA256, variances=VARIANCES, training_seed=TRAIN_SEED,
        smoke_seed=SMOKE_SEED, evaluation_seeds=SEEDS, campaign_seconds=CAMPAIGN_SECONDS,
        policy_acceptance=False))
    end = time.monotonic()+CAMPAIGN_SECONDS
    retained, pairs = [], []
    decision = dict(protocol=PROTOCOL, source=source, decision="runtime-failure-stop",
        reports=retained, pairs=pairs, policy_acceptance=False, f2_admitted=False,
        obstacles_admitted=False, physical_motion_authorized=False)
    try:
        for mode in MODES:
            for arm in ARMS:
                if mode == "pilot":
                    smoke = json.loads((OUTPUT / f"{arm}-smoke/result.json").read_text())
                    require(smoke["updates"] == 10 and smoke["source"] == source
                            and smoke["wall_seconds"]*50*1.5 < 870, "measured smoke budget")
                run_child(mode, arm, source, end)
            require(sha256(OUTPUT / f"control-{mode}/initial.pt") ==
                    sha256(OUTPUT / f"narrow-{mode}/initial.pt"), "identical restored paired state")
        parents = {}
        def evaluate(arm, seed):
            run_child("evaluate", arm, source, end, seed=seed)
            path = OUTPUT / f"{arm}-s{seed}.json"
            report = json.loads(path.read_text())
            retained.append(dict(arm=arm, seed=seed, path=str(path), sha256=sha256(path)))
            validate_speed_classification(report)
            return report
        for seed in SEEDS:
            parents[seed] = evaluate("parent", seed)
            if parents[seed]["safety_failures"]:
                decision.update(decision="reference-safety-stop", failures=parents[seed]["safety_failures"])
                break
        else:
            for seed in SEEDS:
                control = evaluate("control", seed)
                if control["safety_failures"]:
                    decision.update(decision="reference-safety-stop", failures=control["safety_failures"])
                    break
                narrow = evaluate("narrow", seed)
                pair = paired_decision(parents[seed], control, narrow)
                pairs.append(pair)
                if pair["failures"]:
                    decision.update(decision="numerical-gate-stop", failures=pair["failures"])
                    break
            else:
                decision.update(decision="single-seed-diagnostic-support-only", failures=[])
    except Exception as error:
        decision.update(decision="runtime-failure-stop", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT / "decision.json", decision)
        entries = {}
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                entries[str(path.relative_to(OUTPUT))] = dict(sha256=sha256(path), bytes=path.stat().st_size)
        write_new(OUTPUT / "manifest.json", dict(protocol=PROTOCOL, source=source, files=entries))
    print(json.dumps(decision, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", choices=(*MODES, "evaluate"))
    parser.add_argument("--arm", choices=(*ARMS, "parent"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    args = parser.parse_args()
    if args.child:
        launch = json.loads((OUTPUT / "launch.json").read_text())
        require(launch["source"] == args.source and not (OUTPUT / "decision.json").exists(),
                "active exact-source campaign; no closed-run replay")
        child(args.child, args.arm, args.seed, args.source)
    else:
        require(args.arm is None, "arm is child-only")
        campaign(args.source)


if __name__ == "__main__": main()
