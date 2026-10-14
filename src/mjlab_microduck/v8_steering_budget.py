"""Matched unchanged-recipe steering continuation: 1000 versus 2000 updates.

The selected-motor penalty was rejected. Both arms retain its unchanged control
recipe; only the declared additional optimization budget differs.
"""
from __future__ import annotations

import hashlib
import argparse
import io
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from . import v8_gait_scale_contrast as driver
from . import v8_motor_headroom as headroom
from . import integrated_turn_foundation as base
from . import v8_actor_lr_profile as profiles

PROTOCOL = "v8-steering-consolidation-budget-v1"
RUN_PREFIX = "v8-steering-budget"
ARMS = ("control", "extended")
MODES = dict(driver.MODES)
STAGE_SPECS = {arm: {**MODES, "foundation": (256, updates, 967)}
               for arm, updates in (("control", 1000), ("extended", 2000))}
CAPS = {**driver.CAPS, "foundation": 3900}
PARENT_RUN = "v8-motor-headroom-06c2b709-20261014"
PARENT = base.ROOT / "artifacts/training" / PARENT_RUN / "control/foundation/model_999.pt"
PARENT_SHA = "dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a"
SUMMARY = Path("/home/converge/.local/state") / (PARENT_RUN + "-handoff/headroom-summary.json")
SUMMARY_SHA = "61391992bf577974883c454155f01367d8faccad916734719aef06a30e574671"
MANIFEST_SHA = "f5e7d7a15b6796172d57af354c022c21ec6f7a7281ce40ac9049b8d304341bbf"
PARENT_ITERATION, PARENT_STEPS = 999, 24000
PARENT_ADAM_STEP, PARENT_CRITIC_COUNT = 60000, 18432000
TAUS = dict.fromkeys(ARMS, .5)
WEIGHT = -1.
KL_PROFILE = "steering-consolidation-budget"


def verify_source(args, *, training):
    return driver.verify_source(args, training=training, run_prefix=RUN_PREFIX)


def _read_parent():
    summary = json.loads(headroom._owned_payload(SUMMARY, SUMMARY_SHA))
    manifest = json.loads(headroom._owned_payload(SUMMARY.with_name("headroom-manifest.json"), MANIFEST_SHA))
    base.require(summary["source"] == manifest["source"] ==
        "06c2b709dc1c536d83edf7f6f2ab5f33422cb99f"
        and summary["status"] == "verified-evidence-not-policy-acceptance"
        and summary["comparison"]["decision"] == "headroom-lesson-rejected"
        and summary["training"]["control"]["foundation"]["checkpoint_sha256"] == PARENT_SHA
        and manifest["files"]["control/foundation/model_999.pt"]["sha256"] == PARENT_SHA,
        "sealed rejected comparison binds unchanged control as provisional parent")
    driver.validate_parent_live_profile({"training": {
        "foundation": summary["training"]["control"]["foundation"]}})
    for name in ("retention_gate", "cautious_retention"):
        base.require(summary["evaluations"]["control"][name]["decision"] == "walking-idle-retained",
                     "provisional parent retains both original walking references")
    from . import projected_heading_retention, steady_steering_retention
    projected_heading_retention.load_baseline()
    steady_steering_retention.load_baseline()
    payload = torch.load(io.BytesIO(headroom._owned_payload(PARENT, PARENT_SHA)),
                         map_location="cpu", weights_only=True)
    driver.validate_parent_payload(payload, iteration=PARENT_ITERATION, steps=PARENT_STEPS,
        adam_step=PARENT_ADAM_STEP, critic_count=PARENT_CRITIC_COUNT)
    from .v8_actor_checkpoint import FrozenV8Checkpoint
    typed = FrozenV8Checkpoint(PARENT, PARENT_SHA)
    base.require(driver._tree_digest(payload["actor_state_dict"]) ==
                 driver._tree_digest(typed._actor_state), "typed exact parent actor")
    return payload


def initialize(runner, env, output):
    return driver.initialize(runner, env, output, experiment=sys.modules[__name__])


def prepare_config(mode, arm):
    base.require(mode in MODES and arm in ARMS, "declared steering budget stage")
    cfg, agent = headroom.prepare_config(mode, "control")
    # Keep the inert selected-load slot and read-only metric so no other config
    # difference from the authenticated control parent is introduced.
    agent.max_iterations = STAGE_SPECS[arm][mode][1]
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def _canonical(mode, arm, *, mask):
    cfg, agent = prepare_config(mode, arm)
    env, algo = asdict(cfg), asdict(agent)
    env["metrics"]["microduck_yaw_bias"]["func"] = (
        "mjlab_microduck.v8_gait_scale_contrast:_treatment_metric")
    env["metrics"]["microduck_selected_motor_load"]["func"] = (
        "mjlab_microduck.v8_motor_headroom:capture_motor_metric")
    if mask:
        algo["max_iterations"] = "<arm-updates>"
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
        sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode):
    left, right = [_canonical(mode, arm, mask=True) for arm in ARMS]
    base.require(left == right, "only declared optimization budget differs")
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    full = [digest(_canonical(mode, arm, mask=False)) for arm in ARMS]
    return dict(protocol=PROTOCOL + ":recipe-v1", mode=mode,
        control_recipe_sha256=digest(left), treatment_recipe_sha256=digest(right),
        comparison_recipe_sha256=digest(left), control_full_recipe_sha256=full[0],
        treatment_full_recipe_sha256=full[1], full_recipe_hashes_distinct=full[0] != full[1],
        comparison_mask=["agent.max_iterations"],
        stage_specs={arm: list(STAGE_SPECS[arm][mode]) for arm in ARMS},
        matched_after_declared_mask=True, policy_acceptance=False, physical_motion_authorized=False)


def frozen_profile_report(source, arm, mode):
    value = headroom.frozen_profile_report(source, "control", mode)
    value.update(protocol=PROTOCOL + ":frozen-profile-v1", arm=arm, parent_sha256=PARENT_SHA,
                 stage_spec=list(STAGE_SPECS[arm][mode]))
    return value


class Observer(headroom.Observer):
    def __init__(self, mode, arm):
        base.require(arm in ARMS, "declared steering budget arm")
        super().__init__(mode, "control")
        self.protocol, self.report_arm, self.receipt_profile = PROTOCOL, arm, KL_PROFILE


def validate_training_receipt(root, arm, mode, source):
    value = driver.validate_training_receipt(root, arm, mode, source,
                                             experiment=sys.modules[__name__])
    rows = [json.loads(line) for line in (Path(root)/arm/mode/"updates.jsonl").read_text().splitlines()]
    base.require(all(row["selected_motor_load_weight"] == 0.
        and row["selected_motor_weighted_reward_mean"] == 0.
        and row["selected_actuator_names"] == list(headroom.ACTUATORS)
        and row["selected_motor_observed_steps"] == 24 for row in rows),
        "rejected penalty remains inactive while read-only motor accounting stays complete")
    return value


def validate_evaluation_receipt(root, arm, source, checkpoint_sha):
    return driver.validate_evaluation_receipt(root, arm, source, checkpoint_sha,
                                              experiment=sys.modules[__name__])


def diagnostic_comparison(control, extended):
    """Predeclared incremental lesson value; never a substitute for route gates."""
    from . import gait_path_qualification as route
    reports = {}
    for arm, rows in zip(ARMS, (control, extended), strict=True):
        expected = [(s,v,w) for s in base.SEEDS for v,w in base.CASES]
        base.require([(r["seed"],r["speed"],r["yaw"]) for r in rows] == expected,
                     "exact ordered steering budget evaluation matrix")
        routes = [route.summarize_case(row,row["gait_path_trace"]) for row in rows]
        groups = []
        for command in base.CASES:
            worlds = [w for r in routes if (r["speed"],r["yaw"]) == command for w in r["worlds"]]
            chosen = [r for r in rows if (r["speed"],r["yaw"]) == command]
            groups.append(dict(command=list(command),
                complete_route_worlds=sum(w["qualified"] for w in worlds),
                speed_passes=sum(v <= (.03 if command==(0.,0.) else .1) for r in chosen for v in r["speed_mae"]),
                full_heading_error_rad_s=math.fsum(w["full_settled"]["heading_error_rad_s"] for w in worlds)/24,
                full_xy_error_mps=math.fsum(w["full_settled"]["xy_error_mps"] for w in worlds)/24))
        reports[arm] = dict(groups=groups,
            complete_moving_route_worlds=sum(g["complete_route_worlds"] for g in groups[1:]),
            moving_speed_passes=sum(g["speed_passes"] for g in groups[1:]),
            safety_passed=all(r["safety"]["qualified"] for r in routes))
    a,b=reports["control"],reports["extended"]
    failures=[]
    if b["complete_moving_route_worlds"] < a["complete_moving_route_worlds"] + 6:
        failures.append("fewer-than-six-additional-complete-moving-worlds")
    if b["moving_speed_passes"] < a["moving_speed_passes"]:
        failures.append("paired-moving-speed-count-regression")
    for x,y in zip(a["groups"][1:],b["groups"][1:],strict=True):
        if y["full_heading_error_rad_s"] > x["full_heading_error_rad_s"] + .01:
            failures.append([y["command"],"paired-heading-endpoint-regression"])
        if y["full_xy_error_mps"] > x["full_xy_error_mps"] + .02:
            failures.append([y["command"],"paired-xy-endpoint-regression"])
    for arm,r in reports.items():
        if not r["safety_passed"]:failures.append([arm,"motor-range-idle-safety-regression"])
    return dict(protocol=PROTOCOL+":comparison-v1",groups=reports,failures=failures,
        decision="budget-lesson-useful-not-qualified" if not failures else "budget-lesson-rejected",
        thresholds=dict(additional_complete_moving_worlds=6,paired_heading_allowance_rad_s=.01,
                        paired_xy_allowance_mps=.02),
        policy_acceptance=False,physical_motion_authorized=False)


def compare_retained(source,run_id):
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_initialized(),
                 "CPU-only steering budget comparison")
    verify_source(argparse.Namespace(source=source,run_id=run_id),training=False)
    root=base.ROOT/"artifacts/training"/run_id
    summary=json.loads(headroom._owned_payload(SUMMARY,SUMMARY_SHA))
    baseline={"source":summary["source"],"evaluations":{"final-eval":summary["evaluations"]["control"]}}
    from . import steady_steering_retention as retention
    rows,receipts,extra={},{},{}
    for arm in ARMS:
        for mode in MODES:validate_training_receipt(root,arm,mode,source)
        cp=driver.foundation_checkpoint(root,sys.modules[__name__],arm)
        receipts[arm],rows[arm]=validate_evaluation_receipt(root,arm,source,base.sha256(cp))
        extra[arm]=retention._compare_sealed_reference(rows[arm],baseline,
            source=summary["source"],checkpoint_sha=PARENT_SHA,summary_sha=SUMMARY_SHA,
            protocol=PROTOCOL+":provisional-parent-retention-v1")
    report=diagnostic_comparison(rows["control"],rows["extended"])
    report["provisional_parent_retention"]=extra
    for arm,r in receipts.items():
        for key,gate in {**{k:r["evaluation"][k] for k in
            ("projected_heading_retention","cautious_retention")},"provisional_parent":extra[arm]}.items():
            if gate["decision"]!="walking-idle-retained":report["failures"].append([arm,key,"walking-idle-retention-failed"])
    if report["failures"]:report["decision"]="budget-lesson-rejected"
    report.update(source=source,run_id=run_id,parent_sha256=PARENT_SHA,
        qualification_gates={arm:{key:r["evaluation"][key] for key in
            ("legacy_gate","route_gate")} for arm,r in receipts.items()},
        evaluated_result_sha256={arm:base.sha256(root/arm/"final-eval/result.json") for arm in ARMS},
        checkpoint_sha256={arm:base.sha256(driver.foundation_checkpoint(root,sys.modules[__name__],arm)) for arm in ARMS})
    base.write_new(root/"budget-decision.json",report)
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return report


def main():
    if len(sys.argv)>1 and sys.argv[1]=="compare":
        parser=argparse.ArgumentParser()
        parser.add_argument("--source",required=True);parser.add_argument("--run-id",required=True)
        args=parser.parse_args(sys.argv[2:])
        return compare_retained(args.source,args.run_id)
    return driver.main(experiment=sys.modules[__name__])


if __name__ == "__main__":
    main()
