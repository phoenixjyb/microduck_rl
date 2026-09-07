"""F1-M: one motor-weight contrast, immutable evidence and no skill promotion."""

import argparse
import datetime as dt
import functools
import json
import os
import subprocess
import sys
import time

from mjlab_microduck import foundation_lateral_experiment as lateral
from mjlab_microduck import foundation_pilot as training
from mjlab_microduck.first_attempt_smoke import require, sha256
from mjlab_microduck.gpu_idle_gate import wait_idle
from mjlab_microduck.heading_hold_diagnostic import (
    HeadingHold, MODEL, MODEL_SHA, preserved, validate_controller_trace,
)
from mjlab_microduck.recovery_ab import verify_source, write_new
from mjlab_microduck.rollout_repeatability import ROOT, check_host
from mjlab_microduck.speed_response_control import run_control

PROTOCOL = "f1m-motor-paired-s499-v1"
OUTPUT = ROOT / "artifacts/experiments" / PROTOCOL
ARMS = ("control", "motor")
WEIGHTS = dict(control=-2., motor=-4.)
TRAIN_SEED, SMOKE_SEED = 499, 491
SEEDS = (503, 509, 521)
PARENT_ITERATION, PARENT_STEP = 8498, 204000
MODES = dict(smoke=(10, 180), pilot=(500, 900))
CAMPAIGN_SECONDS = 3300
DEADLINE = dt.datetime(2026, 9, 7, 23, tzinfo=dt.timezone.utc)
LAST_START = DEADLINE - dt.timedelta(hours=1)


def prepare_config(mode, *, arm):
    require(mode in MODES and arm in ARMS, "predeclared arm/mode")
    cfg, agent = lateral.prepare_config(mode, arm="lateral")
    # Change the live initial value AND the completed stage: changing only the
    # reward cfg would silently revert at reset to the old -2 curriculum value.
    cfg.rewards["motor_torque_load"].weight = WEIGHTS[arm]
    cfg.curriculum["motor_torque_load_weight"].params["weight_stages"] = [
        dict(step=0, weight=WEIGHTS[arm])]
    cfg.seed = agent.seed = SMOKE_SEED if mode == "smoke" else TRAIN_SEED
    agent.max_iterations = MODES[mode][0]
    agent.experiment_name, agent.run_name = PROTOCOL, f"{arm}-{mode}"
    return cfg, agent


def paired_decision(parent, control, motor):
    result = lateral.paired_decision(parent, control, motor, protocol=PROTOCOL, seeds=SEEDS)
    # Descriptive causal readout is distinct from ALL unchanged admission gates.
    if "settled" in motor["groups"]:
        c, m = control["groups"]["settled"], motor["groups"]["settled"]
        result["motor_tradeoff"] = {
            key: dict(control=c[key], candidate=m[key], delta=m[key]-c[key])
            for key in ("legacy_torque_p99", "cross_route_abs_mean",
                        "pre_reset_squared_utilization_mean", "pre_reset_mechanical_abs_power_mean_w")}
    return result


def checkpoint(arm, source):
    if arm == "parent":
        require(sha256(MODEL) == MODEL_SHA, "frozen research parent")
        return MODEL, MODEL_SHA
    require(arm in ARMS, "declared evaluation arm")
    path = OUTPUT / f"{arm}-pilot/model_8998.pt"
    result = json.loads((path.parent / "result.json").read_text())
    require(result["status"] == "training-complete-not-accepted"
            and result["source"] == source and result["updates"] == 500
            and result["common_step"] == 216000 and result["parent_step"] == PARENT_STEP
            and result["final_sha256"] == sha256(path), "exact-source complete continuation")
    return path, result["final_sha256"]


def child(kind, arm, seed, source):
    verify_source(source); training.runtime_identity(); check_host(); preserved()
    require(dt.datetime.now(dt.timezone.utc) < DEADLINE, "overnight deadline")
    if kind in MODES:
        require(arm in ARMS, "training arm")
        output = OUTPUT / f"{arm}-{kind}"
        output.mkdir(exist_ok=False)
        write_new(output / "launch.json", dict(protocol=PROTOCOL, source=source, arm=arm, mode=kind,
            parent_sha256=MODEL_SHA, parent_iteration=PARENT_ITERATION, parent_step=PARENT_STEP,
            seed=SMOKE_SEED if kind == "smoke" else TRAIN_SEED,
            lateral_cost_weight=-.5, motor_cost_weight=WEIGHTS[arm]))
        seconds = MODES[kind][1]
        result = training.train(kind, output, config_factory=functools.partial(prepare_config, arm=arm),
            protocol=PROTOCOL, parent=MODEL, parent_iteration=PARENT_ITERATION, parent_step=PARENT_STEP,
            stop_at=min(DEADLINE, dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=seconds)),
            max_seconds=seconds)
        write_new(output / "result.json", {**result, "source": source, "policy_acceptance": False})
    else:
        require(kind == "evaluate" and seed in SEEDS, "declared evaluation")
        path, identity = checkpoint(arm, source)
        report = run_control(checkpoint=path, seed=seed, protocol=PROTOCOL, retain_route_trace=True,
                             command_adapter=HeadingHold(True))
        validate_controller_trace(report, True, protocol=PROTOCOL, seeds=SEEDS, checkpoint_sha=identity)
        report.update(source=source, retained_checkpoint_hashes=preserved())
        write_new(OUTPUT / f"{arm}-s{seed}.json", report)


def run_child(kind, arm, source, end, *, seed=SEEDS[0]):
    seconds = MODES[kind][1] if kind in MODES else 90
    require(time.monotonic()+seconds+40 < end, "remaining campaign budget")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=seconds+40) < DEADLINE,
            "child and closeout fit overnight deadline")
    name = f"{arm}-{kind}" if kind in MODES else f"{arm}-s{seed}"
    # Retain the raw idle evidence before every child, including transitions.
    write_new(OUTPUT / f"{name}-idle.json", wait_idle())
    with (OUTPUT / f"{name}.log").open("x") as log:
        result = subprocess.run([sys.executable, "-m", "mjlab_microduck.foundation_motor_experiment",
            "--source", source, "--child", kind, "--arm", arm, "--seed", str(seed)],
            stdout=log, stderr=subprocess.STDOUT, timeout=seconds,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    require(result.returncode == 0, f"{name} child exit {result.returncode}")


def campaign(source):
    verify_source(source)
    require(dt.datetime.now(dt.timezone.utc) < LAST_START, "one-shot bounded launch window")
    runtime, identities = training.runtime_identity(), preserved()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    end = time.monotonic()+CAMPAIGN_SECONDS
    rows, pairs = [], []
    decision = dict(protocol=PROTOCOL, source=source, decision="runtime-failure-stop", reports=rows, pairs=pairs,
        policy_acceptance=False, obstacles_admitted=False, hop_validated=False,
        stabilization_retention_validated=False, physical_motion_authorized=False)

    def evaluate(arm, seed):
        run_child("evaluate", arm, source, end, seed=seed)
        path = OUTPUT / f"{arm}-s{seed}.json"
        report = json.loads(path.read_text())
        _, identity = checkpoint(arm, source)
        validate_controller_trace(report, True, protocol=PROTOCOL, seeds=SEEDS, checkpoint_sha=identity)
        rows.append(dict(arm=arm, seed=seed, sha256=sha256(path)))
        return report

    try:
        write_new(OUTPUT / "launch.json", dict(protocol=PROTOCOL, source=source,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(), runtime=runtime, idle=wait_idle(),
            retained_checkpoint_hashes=identities, training_seed=TRAIN_SEED, smoke_seed=SMOKE_SEED,
            evaluation_seeds=SEEDS, lateral_cost_weight=-.5, motor_cost_weights=WEIGHTS,
            campaign_seconds=CAMPAIGN_SECONDS, deadline=DEADLINE.isoformat()))
        parents = {}
        for seed in SEEDS:
            parents[seed] = evaluate("parent", seed)
            if parents[seed]["safety_failures"]:
                decision.update(decision="reference-safety-stop", failures=parents[seed]["safety_failures"])
                break
        else:
            for mode in MODES:
                for arm in ARMS:
                    if mode == "pilot":
                        smoke = json.loads((OUTPUT / f"{arm}-smoke/result.json").read_text())
                        require(smoke["status"] == "training-complete-not-accepted"
                                and smoke["source"] == source and smoke["updates"] == 10
                                and smoke["common_step"] == PARENT_STEP+240
                                and smoke["wall_seconds"]*50*1.5 < 870, "complete measured smoke budget")
                    run_child(mode, arm, source, end)
                require(sha256(OUTPUT / f"control-{mode}/initial.pt") ==
                        sha256(OUTPUT / f"motor-{mode}/initial.pt"), "identical paired restored state")
            for seed in SEEDS:
                control = evaluate("control", seed)
                if control["safety_failures"]:
                    decision.update(decision="reference-safety-stop", failures=control["safety_failures"])
                    break
                motor = evaluate("motor", seed)
                pair = paired_decision(parents[seed], control, motor); pairs.append(pair)
                if pair["failures"]:
                    decision.update(decision="numerical-gate-stop", failures=pair["failures"])
                    break
            else:
                decision.update(decision="single-seed-diagnostic-support-only", failures=[])
        decision["retained_checkpoint_hashes"] = preserved()
    except Exception as error:
        decision.update(decision="runtime-failure-stop", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT / "decision.json", decision)
        files = {str(p.relative_to(OUTPUT)): dict(sha256=sha256(p), bytes=p.stat().st_size)
                 for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT / "manifest.json", dict(protocol=PROTOCOL, source=source, files=files))
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", choices=(*MODES, "evaluate"))
    parser.add_argument("--arm", choices=(*ARMS, "parent"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    args = parser.parse_args()
    if args.child:
        launch = json.loads((OUTPUT / "launch.json").read_text())
        require(launch["source"] == args.source and not (OUTPUT / "decision.json").exists(), "open exact-source run")
        child(args.child, args.arm, args.seed, args.source)
    else:
        require(args.arm is None, "arm is child-only")
        campaign(args.source)


if __name__ == "__main__":
    main()
