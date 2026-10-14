"""Matched yaw-bias reward-gain contrast on the retained MicroDuck parent.

Both arms preserve the same 0.5 s body-yaw EMA and differ only in the
``yaw_bias`` reward weight. The rejected budget comparison is authenticated
before the original motor-headroom control parent is loaded. This protocol is
an exploratory lesson, not policy acceptance or motion authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from . import integrated_turn_foundation as base
from . import v8_gait_scale_contrast as driver
from . import v8_motor_headroom as headroom
from . import v8_steering_budget as budget
from . import v8_actor_lr_profile as profiles

PROTOCOL = "v8-yaw-gain-contrast-v1"
RUN_PREFIX = "v8-yaw-gain"
ARMS = ("control", "reduced")
MODES = dict(budget.MODES)
# The shared driver's explicit matrix mode is reserved for the prior
# 1000-vs-2000 exposure study. Omitting an arm-specific matrix here preserves
# its matched default 64x5 / 256x10 / 256x1000 stages for both arms.
STAGE_SPECS = None
CAPS = dict(driver.CAPS)
PARENT = budget.PARENT
PARENT_SHA = budget.PARENT_SHA
PARENT_ITERATION = budget.PARENT_ITERATION
PARENT_STEPS = budget.PARENT_STEPS
PARENT_ADAM_STEP = budget.PARENT_ADAM_STEP
PARENT_CRITIC_COUNT = budget.PARENT_CRITIC_COUNT
TAUS = dict.fromkeys(ARMS, .5)
WEIGHTS = {"control": -1.0, "reduced": -.5}
# The inherited driver defaults remain available to legacy callers. This
# protocol always supplies its arm-specific value explicitly.
WEIGHT = -1.0
KL_PROFILE = "yaw-gain-contrast"

REJECTED_BUDGET_SOURCE = "c340ff00d8d532fa4a664a298589b7fa401d7467"
REJECTED_BUDGET_RUN = "v8-steering-budget-c340ff00-20261014"
REJECTED_BUDGET_DIR = Path("/home/converge/.local/state") / (REJECTED_BUDGET_RUN + "-handoff")
REJECTED_BUDGET_SUMMARY = REJECTED_BUDGET_DIR / "budget-summary.json"
REJECTED_BUDGET_MANIFEST = REJECTED_BUDGET_DIR / "budget-manifest.json"
REJECTED_BUDGET_SUMMARY_SHA = (
    "bfba1397b4a6657b2411f72eac25afb2f1bbb401b11f1d5443df6bc7f4d58987")
REJECTED_BUDGET_MANIFEST_SHA = (
    "55bfe531793362c938c99eabc3650d6f42426b424c098ca85cf4dcfaa5b8e436")


def verify_source(args, *, training):
    return driver.verify_source(args, training=training, run_prefix=RUN_PREFIX)


def _read_parent():
    """Authenticate the rejected comparison, then load its original parent."""
    summary = json.loads(headroom._owned_payload(
        REJECTED_BUDGET_SUMMARY, REJECTED_BUDGET_SUMMARY_SHA))
    manifest = json.loads(headroom._owned_payload(
        REJECTED_BUDGET_MANIFEST, REJECTED_BUDGET_MANIFEST_SHA))
    comparison = summary.get("comparison", {})
    control = comparison.get("groups", {}).get("control", {})
    extended = comparison.get("groups", {}).get("extended", {})
    positive_control = next((group for group in control.get("groups", [])
                             if group.get("command") == [.2, .2]), {})
    positive_extended = next((group for group in extended.get("groups", [])
                              if group.get("command") == [.2, .2]), {})
    base.require(summary.get("source") == manifest.get("source") == REJECTED_BUDGET_SOURCE
        and summary.get("protocol") == budget.PROTOCOL
        and summary.get("status") == "verified-evidence-not-policy-acceptance"
        and comparison.get("decision") == "budget-lesson-rejected"
        and control.get("complete_moving_route_worlds") == 7
        and extended.get("complete_moving_route_worlds") == 15
        and control.get("moving_speed_passes") == 66
        and extended.get("moving_speed_passes") == 69
        and positive_control.get("full_heading_error_rad_s") == .05288204378424289
        and positive_extended.get("full_heading_error_rad_s") == .07641887840478805
        and summary.get("parent_checkpoint_sha256") == PARENT_SHA
        and manifest.get("source") == REJECTED_BUDGET_SOURCE
        and summary.get("comparison_sha256")
            == manifest.get("files", {}).get("budget-decision.json", {}).get("sha256")
        and manifest.get("total_bytes", 0) > 0,
        "authenticated rejected budget evidence and unchanged original parent")
    return budget._read_parent()


def initialize(runner, env, output):
    return driver.initialize(runner, env, output, experiment=sys.modules[__name__])


def prepare_config(mode, arm):
    base.require(mode in MODES and arm in ARMS, "declared yaw-gain stage")
    cfg, agent = budget.prepare_config(mode, "control")
    agent.max_iterations = MODES[mode][1]
    cfg.rewards["yaw_bias"].weight = WEIGHTS[arm]
    cfg.metrics["microduck_yaw_bias"].func = _capture_metric
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def _capture_metric(env):
    """Read the live configured gain and pass it to the shared verifier."""
    term = env.reward_manager.get_term_cfg(driver.steady.TERM)
    base.require(type(term.weight) in (int, float)
        and term.weight in WEIGHTS.values(),
        "live yaw-bias reward weight is one of the declared gain arms")
    return driver.capture_metric(env, "gait-scale", weight=term.weight)


def _canonical(mode, arm, *, mask):
    cfg, agent = prepare_config(mode, arm)
    env, algo = asdict(cfg), asdict(agent)
    env["metrics"]["microduck_yaw_bias"]["func"] = (
        "mjlab_microduck.v8_yaw_gain:_capture_metric")
    env["metrics"]["microduck_selected_motor_load"]["func"] = (
        "mjlab_microduck.v8_motor_headroom:capture_motor_metric")
    if mask:
        env["rewards"]["yaw_bias"]["weight"] = "<arm-yaw-bias-weight>"
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
        sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode):
    base.require(mode in MODES, "declared yaw-gain recipe mode")
    masked = [_canonical(mode, arm, mask=True) for arm in ARMS]
    base.require(masked[0] == masked[1], "only yaw_bias reward weight differs")
    full = [_canonical(mode, arm, mask=False) for arm in ARMS]
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return dict(protocol=PROTOCOL + ":recipe-v1", mode=mode,
        control_recipe_sha256=digest(masked[0]),
        treatment_recipe_sha256=digest(masked[1]),
        comparison_recipe_sha256=digest(masked[0]),
        control_full_recipe_sha256=digest(full[0]),
        treatment_full_recipe_sha256=digest(full[1]),
        full_recipe_hashes_distinct=digest(full[0]) != digest(full[1]),
        comparison_mask=["reward yaw_bias.weight"], weights=WEIGHTS,
        taus=TAUS,
        stage_specs={arm: list(MODES[mode]) for arm in ARMS},
        matched_after_declared_mask=True, policy_acceptance=False,
        physical_motion_authorized=False)


def frozen_profile_report(source, arm, mode):
    value = budget.frozen_profile_report(source, "control", mode)
    value.update(protocol=PROTOCOL + ":frozen-profile-v1", arm=arm,
        parent_sha256=PARENT_SHA, yaw_bias_weight=WEIGHTS[arm],
        yaw_bias_tau_s=TAUS[arm], stage_spec=list(MODES[mode]))
    return value


class Observer(headroom.Observer):
    """Retain headroom observation while recording this protocol's arm/gain."""
    def __init__(self, mode, arm):
        base.require(arm in ARMS, "declared yaw-gain arm")
        super().__init__(mode, "control", bias_weight=WEIGHTS[arm])
        self.protocol = PROTOCOL
        self.report_arm = arm
        self.receipt_profile = KL_PROFILE
        self.yaw_gain_arm = arm


def validate_training_receipt(root, arm, mode, source):
    value = driver.validate_training_receipt(root, arm, mode, source,
                                             experiment=sys.modules[__name__])
    rows = [json.loads(line) for line in
            (Path(root) / arm / mode / "updates.jsonl").read_text().splitlines()]
    base.require(all(row["selected_motor_load_weight"] == 0.
        and row["selected_motor_weighted_reward_mean"] == 0.
        and row["selected_actuator_names"] == list(headroom.ACTUATORS)
        and row["selected_motor_observed_steps"] == 24 for row in rows),
        "rejected selected-load penalty remains inactive with complete read-only accounting")
    base.require(all(row["yaw_bias_weight"] == WEIGHTS[arm]
        and row["yaw_bias_tau_s"] == .5
        and row["weighted_yaw_bias_reward_mean"] <= 0
        for row in rows), "actual yaw-bias reward follows the declared gain and fixed horizon")
    return value


def validate_evaluation_receipt(root, arm, source, checkpoint_sha):
    return driver.validate_evaluation_receipt(root, arm, source, checkpoint_sha,
                                              experiment=sys.modules[__name__])


def diagnostic_comparison(control, reduced):
    """Reuse unchanged budget thresholds, with this lesson's arm names."""
    report = budget.diagnostic_comparison(control, reduced)
    groups = report["groups"]
    groups["reduced"] = groups.pop("extended")
    failures = []
    for failure in report["failures"]:
        if failure == ["extended", "motor-range-idle-safety-regression"]:
            failures.append(["reduced", "motor-range-idle-safety-failure"])
        elif failure == ["control", "motor-range-idle-safety-regression"]:
            failures.append(["control", "motor-range-idle-safety-failure"])
        else:
            failures.append(failure)
    report.update(protocol=PROTOCOL + ":comparison-v1", groups=groups,
        failures=failures,
        decision="gain-lesson-useful-not-qualified" if not failures else "gain-lesson-rejected",
        weights=WEIGHTS,
        policy_acceptance=False, physical_motion_authorized=False)
    return report


def compare_retained(source, run_id):
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
                 "CPU-only yaw-gain comparison")
    verify_source(argparse.Namespace(source=source, run_id=run_id), training=False)
    root = base.ROOT / "artifacts/training" / run_id
    summary = json.loads(headroom._owned_payload(
        REJECTED_BUDGET_SUMMARY, REJECTED_BUDGET_SUMMARY_SHA))
    # Preserve the original provisional walking reference used by the budget
    # protocol: the authenticated motor-headroom control, not the rejected
    # budget arm. The rejected budget summary above binds this new lesson's
    # lineage; it is not itself a retention baseline.
    parent_summary = json.loads(headroom._owned_payload(
        budget.SUMMARY, budget.SUMMARY_SHA))
    baseline = {"source": parent_summary["source"],
                "evaluations": {"final-eval": parent_summary["evaluations"]["control"]}}
    from . import steady_steering_retention as retention
    rows, receipts, parent_retention = {}, {}, {}
    for arm in ARMS:
        for mode in MODES:
            validate_training_receipt(root, arm, mode, source)
        checkpoint = driver.foundation_checkpoint(root, sys.modules[__name__], arm)
        receipts[arm], rows[arm] = validate_evaluation_receipt(
            root, arm, source, base.sha256(checkpoint))
        parent_retention[arm] = retention._compare_sealed_reference(
            rows[arm], baseline, source=parent_summary["source"],
            checkpoint_sha=PARENT_SHA, summary_sha=budget.SUMMARY_SHA,
            protocol=PROTOCOL + ":original-parent-retention-v1")
    report = diagnostic_comparison(rows["control"], rows["reduced"])
    report["original_parent_retention"] = parent_retention
    for arm, receipt in receipts.items():
        evaluation = receipt["evaluation"]
        for key in ("projected_heading_retention", "cautious_retention"):
            if evaluation[key]["decision"] != "walking-idle-retained":
                report["failures"].append([arm, key, "walking-idle-retention-failure"])
        if parent_retention[arm]["decision"] != "walking-idle-retained":
            report["failures"].append([arm, "original_parent",
                                       "walking-idle-retention-failure"])
    if report["failures"]:
        report["decision"] = "gain-lesson-rejected"
    report.update(source=source, run_id=run_id, parent_sha256=PARENT_SHA,
        rejected_budget_summary_sha256=REJECTED_BUDGET_SUMMARY_SHA,
        qualification_gates={arm: {key: receipt["evaluation"][key] for key in
            ("legacy_gate", "route_gate")} for arm, receipt in receipts.items()},
        evaluated_result_sha256={arm: base.sha256(root / arm / "final-eval" / "result.json")
                                 for arm in ARMS},
        checkpoint_sha256={arm: base.sha256(driver.foundation_checkpoint(
            root, sys.modules[__name__], arm)) for arm in ARMS})
    base.write_new(root / "gain-decision.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", required=True)
        parser.add_argument("--run-id", required=True)
        args = parser.parse_args(sys.argv[2:])
        return compare_retained(args.source, args.run_id)
    return driver.main(experiment=sys.modules[__name__])


if __name__ == "__main__":
    main()
