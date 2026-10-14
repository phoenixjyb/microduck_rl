"""Matched staged head-command ranges during steering practice.

The control and staged arms share the retained motor-headroom continuation
recipe. Their only declared difference is the live head-goal range history;
the historical yaw price, motor-load objective, actor and evaluation gates are
unchanged. This is a curriculum-complexity experiment, not policy acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

from . import integrated_turn_foundation as base
from . import v8_actor_lr_profile as profiles
from . import v8_gait_scale_contrast as driver
from . import v8_head_goal_schedule as schedule
from . import v8_motor_headroom as headroom
from . import v8_steering_budget as budget
from . import v8_yaw_gain as gain

PROTOCOL = schedule.PROTOCOL
RUN_PREFIX = "v8-staged-head-goals"
ARMS = schedule.ARMS
MODES = dict(gain.MODES)
CAPS = dict(gain.CAPS)
PARENT = budget.PARENT
PARENT_SHA = budget.PARENT_SHA
PARENT_ITERATION = budget.PARENT_ITERATION
PARENT_STEPS = budget.PARENT_STEPS
PARENT_ADAM_STEP = budget.PARENT_ADAM_STEP
PARENT_CRITIC_COUNT = budget.PARENT_CRITIC_COUNT
TAUS = dict.fromkeys(ARMS, .5)
WEIGHT = -1.0
KL_PROFILE = "staged-head-goals"
ROOT = base.ROOT

REJECTED_GAIN_SOURCE = "7acacf87313c08973d682e2d701c7ef7b8cabe35"
REJECTED_GAIN_RUN = "v8-yaw-gain-7acacf87-20261014"
REJECTED_GAIN_DIR = Path("/home/converge/.local/state") / (REJECTED_GAIN_RUN + "-handoff")
REJECTED_GAIN_SUMMARY = REJECTED_GAIN_DIR / "gain-summary.json"
REJECTED_GAIN_MANIFEST = REJECTED_GAIN_DIR / "gain-manifest.json"
REJECTED_GAIN_SUMMARY_SHA = "c7f962464669c250cb8c2c4d62085d3614b10085747a9c95085d01591e3d4125"
REJECTED_GAIN_MANIFEST_SHA = "4768d07efd6650ed47bc73f8ac33156a55e78ef1ed017259eacc6fa3d19ffda9"


def verify_source(args, *, training):
    return driver.verify_source(args, training=training, run_prefix=RUN_PREFIX)


def _read_parent():
    """Authenticate rejected gain evidence before resolving the original parent."""
    summary = json.loads(headroom._owned_payload(
        REJECTED_GAIN_SUMMARY, REJECTED_GAIN_SUMMARY_SHA))
    manifest = json.loads(headroom._owned_payload(
        REJECTED_GAIN_MANIFEST, REJECTED_GAIN_MANIFEST_SHA))
    comparison = summary.get("comparison", {})
    base.require(summary.get("source") == REJECTED_GAIN_SOURCE
        and summary.get("protocol") == gain.PROTOCOL
        and summary.get("status") == "verified-evidence-not-policy-acceptance"
        and comparison.get("decision") == "gain-lesson-rejected"
        and summary.get("parent_checkpoint_sha256") == PARENT_SHA
        and manifest.get("source") == REJECTED_GAIN_SOURCE
        and manifest.get("files", {}).get("gain-decision.json", {}).get("sha256")
            == summary.get("comparison_sha256")
        and manifest.get("total_bytes", 0) > 0,
        "authenticated rejected gain summary and inventory bind the original parent")
    # gain._read_parent authenticates the rejected budget evidence and then
    # loads the sealed motor-headroom control checkpoint, never a gain endpoint.
    return gain._read_parent()


def initialize(runner, env, output):
    return driver.initialize(runner, env, output, experiment=sys.modules[__name__])


def _ranges_payload(arm):
    # Receipts are compared after durable JSON encoding. Normalize tuples to
    # lists here so the in-memory recipe has exactly the same shape.
    return json.loads(json.dumps(dict(
        initial=schedule.ranges_at(arm, 0), stages=schedule.stages_for(arm)),
        allow_nan=False))


def prepare_config(mode, arm):
    """Build both arms from the same frozen motor-headroom control recipe."""
    base.require(mode in MODES and arm in ARMS, "declared staged-head-goal stage")
    cfg, agent = gain.prepare_config(mode, "control")
    cfg.commands["head_pose"].ranges = schedule.ranges_at(arm, 0)
    curriculum = cfg.curriculum["head_pose_range"]
    base.require(curriculum.func is not None and isinstance(curriculum.params, dict)
                 and "range_stages" in curriculum.params,
                 "existing head-pose range curriculum is available")
    curriculum.params["range_stages"] = schedule.stages_for(arm)
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def _canonical(mode, arm, *, mask):
    cfg, agent = prepare_config(mode, arm)
    env, algo = asdict(cfg), asdict(agent)
    env["metrics"]["microduck_yaw_bias"]["func"] = (
        "mjlab_microduck.v8_yaw_gain:_capture_metric")
    env["metrics"]["microduck_selected_motor_load"]["func"] = (
        "mjlab_microduck.v8_motor_headroom:capture_motor_metric")
    if mask:
        env["commands"]["head_pose"]["ranges"] = "<declared-head-range-initial>"
        env["curriculum"]["head_pose_range"]["params"]["range_stages"] = (
            "<declared-head-range-history>")
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode):
    base.require(mode in MODES, "declared staged-head-goal recipe mode")
    left, right = [_canonical(mode, arm, mask=True) for arm in ARMS]
    base.require(left == right, "only initial head range and head range stages differ")
    full = [_canonical(mode, arm, mask=False) for arm in ARMS]
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return dict(protocol=PROTOCOL + ":recipe-v1", mode=mode,
        control_recipe_sha256=digest(left), treatment_recipe_sha256=digest(right),
        comparison_recipe_sha256=digest(left), control_full_recipe_sha256=digest(full[0]),
        treatment_full_recipe_sha256=digest(full[1]),
        full_recipe_hashes_distinct=digest(full[0]) != digest(full[1]),
        comparison_mask=["commands.head_pose.ranges",
                         "curriculum.head_pose_range.range_stages"],
        range_history={arm: _ranges_payload(arm) for arm in ARMS},
        matched_after_declared_mask=True, policy_acceptance=False,
        physical_motion_authorized=False)


def frozen_profile_report(source, arm, mode):
    base.require(arm in ARMS and mode in MODES, "declared frozen profile identity")
    value = gain.frozen_profile_report(source, "control", mode)
    value["live_values"]["head_pose_ranges"] = [
        list(pair) for pair in schedule.ranges_at(arm, 0)]
    value["curriculum_stage_counts"]["head_pose_range"] = len(schedule.stages_for(arm))
    value["curriculum_clock_start"] = 0
    value["curriculum_clock_policy"] = "local-common-step-full-training-range-history"
    value.update(protocol=PROTOCOL + ":frozen-profile-v1", arm=arm,
        parent_sha256=PARENT_SHA, head_range_schedule=_ranges_payload(arm),
        yaw_bias_weight=WEIGHT, yaw_bias_tau_s=TAUS[arm], selected_load_weight=0.)
    return json.loads(json.dumps(value))


def validate_head_contract(env):
    """Fail closed if any inherited head objective or command sampler changes."""
    from .tasks import mdp

    tracking = env.reward_manager.get_term_cfg("head_pose_tracking")
    bias = env.reward_manager.get_term_cfg("head_pose_bias")
    command = env.command_manager.get_term("head_pose").cfg
    base.require(tracking.func is mdp.head_pose_tracking
        and tracking.params == {"command_name": "head_pose", "std": .5}
        and tracking.weight == 2.0,
        "head tracking reward function, parameters, and weight remain frozen")
    base.require(bias.func is mdp.head_pose_bias_penalty
        and bias.params == {"command_name": "head_pose", "tau_s": 1.0}
        and bias.weight == 1.0,
        "head bias reward function, horizon, and weight remain frozen")
    base.require(tuple(command.resampling_time_range) == (2.0, 5.0)
        and command.zero_command_prob == 0.0,
        "head command resampling and exact-zero probability remain frozen")
    return True


def capture_head_reward_snapshot(env, expected_step=None):
    """Read already-weighted head reward columns; never call either reward func."""
    import torch

    validate_head_contract(env)
    if expected_step is not None:
        base.require(type(expected_step) is int and expected_step >= 1
            and env.common_step_counter == expected_step,
            "head reward snapshot is from the just-completed local step")
    manager = env.reward_manager
    active = manager.active_terms
    base.require("head_pose_tracking" in active and "head_pose_bias" in active,
                 "both head reward columns are registered and observable")
    values = {}
    for name in ("head_pose_tracking", "head_pose_bias"):
        column = manager._step_reward[:, active.index(name)].detach().clone()
        base.require(isinstance(column, torch.Tensor) and column.shape == (env.num_envs,)
            and bool(torch.isfinite(column).all()), "finite actual weighted head reward column")
        values[name] = column
    base.require(bool((values["head_pose_tracking"] >= 0).all())
        and bool((values["head_pose_bias"] <= 0).all()),
        "actual tracking reward is nonnegative and head-bias penalty is nonpositive")
    result = {"step": env.common_step_counter}
    for name, label in (("head_pose_tracking", "tracking"), ("head_pose_bias", "bias")):
        sample = values[name].double().cpu().tolist()
        result[f"head_pose_{label}_weighted_reward_mean"] = math.fsum(sample) / len(sample)
        result[f"head_pose_{label}_weighted_reward_min"] = min(sample)
        result[f"head_pose_{label}_weighted_reward_max"] = max(sample)
        result[f"head_pose_{label}_weighted_rewards"] = sample
    base.require(all(type(value) is int or type(value) is float and math.isfinite(value)
                     for key, value in result.items() if key.endswith(("_mean", "_min", "_max"))),
                 "finite JSON-safe head reward summary")
    return result


class Observer(gain.Observer):
    """Compose actual scheduled ranges with the inherited read-only observers."""
    def __init__(self, mode, arm):
        base.require(arm in ARMS, "declared staged-head-goal arm")
        # The inherited yaw-gain observer's "control" arm is the unchanged
        # -1.0/tau-.5 parent objective used by both arms in this lesson.
        super().__init__(mode, "control")
        self.protocol, self.report_arm = PROTOCOL, arm
        self.receipt_profile = KL_PROFILE
        self.head_goals = schedule.HeadGoalTelemetry(arm)
        self.head_tracking_rewards, self.head_bias_rewards = [], []

    def expected_head_ranges(self, step):
        return tuple(tuple(pair) for pair in schedule.ranges_at("staged" if self.report_arm == "staged"
                                                                else "control", step))

    def before(self, env):
        self.head_goals.before(env)
        validate_head_contract(env)
        super().before(env)

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        rewards = capture_head_reward_snapshot(env, expected_step=sample.step + 1)
        self.head_tracking_rewards.append(rewards["head_pose_tracking_weighted_reward_mean"])
        self.head_bias_rewards.append(rewards["head_pose_bias_weighted_reward_mean"])
        self.head_goals.consume(env, sample)

    def update(self, env):
        applied = schedule.apply_ranges(env, self.report_arm)
        validate_head_contract(env)
        row = super().update(env)
        base.require(len(self.head_tracking_rewards) == len(self.head_bias_rewards) == 24,
                     "actual head rewards observed on every action step in the update")
        head_telemetry = self.head_goals.drain(expected_steps=schedule.STEPS_PER_UPDATE)
        schedule.validate_window(head_telemetry, self.report_arm, False, env.num_envs,
                                 env.common_step_counter - schedule.STEPS_PER_UPDATE,
                                 schedule.STEPS_PER_UPDATE)
        base.require(applied["ranges"] == schedule.ranges_at(
            self.report_arm, env.common_step_counter),
            "post-rollout range schedule reapplied before inherited update snapshot")
        row["head_goal_telemetry"] = head_telemetry
        row["head_goal_ranges_at_update"] = [list(pair) for pair in applied["ranges"]]
        row["head_pose_tracking_weighted_reward_mean"] = math.fsum(self.head_tracking_rewards) / 24
        row["head_pose_bias_weighted_reward_mean"] = math.fsum(self.head_bias_rewards) / 24
        row["head_pose_reward_observed_steps"] = 24
        base.require(row["head_pose_tracking_weighted_reward_mean"] >= 0
                     and row["head_pose_bias_weighted_reward_mean"] <= 0,
                     "logged actual head reward signs remain valid")
        self.head_tracking_rewards.clear()
        self.head_bias_rewards.clear()
        return row


def validate_training_receipt(root, arm, mode, source):
    value = driver.validate_training_receipt(root, arm, mode, source,
                                             experiment=sys.modules[__name__])
    rows = [json.loads(line) for line in
            (Path(root) / arm / mode / "updates.jsonl").read_text().splitlines()]
    base.require(all(row["selected_motor_load_weight"] == 0.
        and row["selected_motor_weighted_reward_mean"] == 0.
        and row["selected_actuator_names"] == list(headroom.ACTUATORS)
        and row["selected_motor_observed_steps"] == 24
        and row["yaw_bias_weight"] == WEIGHT and row["yaw_bias_tau_s"] == .5
        and row["weighted_yaw_bias_reward_mean"] <= 0
        and row["head_pose_reward_observed_steps"] == 24
        and row["head_pose_tracking_weighted_reward_mean"] >= 0
        and row["head_pose_bias_weighted_reward_mean"] <= 0
        for row in rows),
        "frozen yaw and inactive selected-load terms remain identical in both arms")
    for index, row in enumerate(rows):
        start_step = index * schedule.STEPS_PER_UPDATE
        telemetry = row.get("head_goal_telemetry")
        schedule.validate_window(telemetry, arm, False, value["worlds"], start_step,
                                 schedule.STEPS_PER_UPDATE)
        base.require(row.get("head_goal_ranges_at_update") == [
            list(pair) for pair in schedule.ranges_at(arm, (index + 1) * schedule.STEPS_PER_UPDATE)],
            "actual manager head range at each update follows the declared common-step schedule")
    if mode in ("benchmark", "foundation"):
        from . import v8_head_goal_boundary_probe as probe
        for paired_arm in ARMS:
            probe.validate_receipt(root, paired_arm, source, experiment=sys.modules[__name__])
    return value


def validate_evaluation_receipt(root, arm, source, checkpoint_sha):
    return driver.validate_evaluation_receipt(root, arm, source, checkpoint_sha,
                                              experiment=sys.modules[__name__])


def diagnostic_comparison(control, staged):
    """Reuse the retained gain-lesson paired thresholds with this arm's name."""
    report = gain.diagnostic_comparison(control, staged)
    groups = report["groups"]
    groups["staged"] = groups.pop("reduced")
    failures = []
    for failure in report["failures"]:
        failures.append(["staged" if failure[0] == "reduced" else failure[0], *failure[1:]])
    report.update(protocol=PROTOCOL + ":comparison-v1", groups=groups,
        failures=failures,
        decision="staged-head-goals-lesson-useful-not-qualified" if not failures
                 else "staged-head-goals-lesson-rejected",
        weights=dict.fromkeys(ARMS, WEIGHT),
        policy_acceptance=False, physical_motion_authorized=False)
    return report


def compare_retained(source, run_id):
    """Recompute paired diagnostics and all three retained walking references."""
    import torch

    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
                 "CPU-only staged-head-goal comparison")
    args = argparse.Namespace(source=source, run_id=run_id)
    verify_source(args, training=False)
    root = ROOT / "artifacts/training" / run_id
    # Reauthenticate gain/budget evidence and the original control parent before
    # interpreting any new output as a continuation.
    _read_parent()
    parent_summary = json.loads(headroom._owned_payload(budget.SUMMARY, budget.SUMMARY_SHA))
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

    report = diagnostic_comparison(rows["control"], rows["staged"])
    report["original_parent_retention"] = parent_retention
    for arm, receipt in receipts.items():
        evaluation = receipt["evaluation"]
        for key in ("projected_heading_retention", "cautious_retention"):
            if evaluation[key]["decision"] != "walking-idle-retained":
                report["failures"].append([arm, key, "walking-idle-retention-failure"])
        if parent_retention[arm]["decision"] != "walking-idle-retained":
            report["failures"].append([arm, "original_parent", "walking-idle-retention-failure"])
    if report["failures"]:
        report["decision"] = "staged-head-goals-lesson-rejected"
    report.update(source=source, run_id=run_id, parent_sha256=PARENT_SHA,
        rejected_gain_summary_sha256=REJECTED_GAIN_SUMMARY_SHA,
        qualification_gates={arm: {key: receipt["evaluation"][key] for key in
            ("legacy_gate", "route_gate")} for arm, receipt in receipts.items()},
        evaluated_result_sha256={arm: base.sha256(root / arm / "final-eval" / "result.json")
                                 for arm in ARMS},
        checkpoint_sha256={arm: base.sha256(driver.foundation_checkpoint(
            root, sys.modules[__name__], arm)) for arm in ARMS})
    base.write_new(root / "staged-head-goals-decision.json", report)
    return report


def _boundary_smoke(argv):
    from . import v8_head_goal_boundary_probe as probe

    parser = argparse.ArgumentParser(description="Run the separate bounded GPU head-range boundary probe")
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args(argv)
    verify_source(args, training=True)
    root = ROOT / "artifacts/training" / args.run_id
    for arm in ARMS:
        validate_training_receipt(root, arm, "smoke", args.source)
    return probe.main(argv=argv, experiment=sys.modules[__name__])


def _require_boundary_receipts(root, source):
    from . import v8_head_goal_boundary_probe as probe
    for arm in ARMS:
        probe.validate_receipt(root, arm, source, experiment=sys.modules[__name__])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "boundary-smoke":
        return _boundary_smoke(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = argparse.ArgumentParser(description="Recompute the CPU-only staged-head-goal comparison")
        parser.add_argument("--source", required=True)
        parser.add_argument("--run-id", required=True)
        args = parser.parse_args(sys.argv[2:])
        return compare_retained(args.source, args.run_id)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source")
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=(*MODES, "final-eval"))
    parser.add_argument("--preflight", action="store_true")
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    if args.mode in ("benchmark", "foundation") and not args.preflight:
        base.require(args.run_id is not None and args.source is not None,
                     "boundary admission has source and run identity")
        _require_boundary_receipts(ROOT / "artifacts/training" / args.run_id, args.source)
    return driver.main(experiment=sys.modules[__name__])


if __name__ == "__main__":
    main()
