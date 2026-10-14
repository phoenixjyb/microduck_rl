"""One-factor knee/ankle headroom lesson from the provisional gait-scale actor.

Compose the existing matched continuation driver explicitly. Historical recipe
globals and acceptance gates are never patched or reinterpreted.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import io
import json
import math
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from . import v8_gait_scale_contrast as driver
from . import integrated_turn_foundation as base
from . import v8_actor_lr_profile as profiles

PROTOCOL = "v8-selected-motor-headroom-v1"
RUN_PREFIX = "v8-motor-headroom"
ARMS = ("control", "headroom")
MODES = dict(driver.MODES)
CAPS = dict(driver.CAPS)
PARENT_SHA = "97fac488eae3b8100515dd310a3f956060f7364a55eaa55752602ef4a67ee000"
PARENT_RUN = "v8-gait-scale-steering-1b223f93-20261014"
PARENT = base.ROOT / "artifacts/training" / PARENT_RUN / "gait-scale/foundation/model_999.pt"
PARENT_STATE = Path("/home/converge/.local/state") / (PARENT_RUN + "-handoff")
SUMMARY_SHA = "ac57a9922c9ee84f5386b24a281a49238b3f7e7d7710cdcb453d7e8029c523bf"
MANIFEST_SHA = "a83b604c48cbb1cbddb4bea27bc02149869eec7c0ed5aa14af2541ddc69d1c0e"
PARENT_ITERATION, PARENT_STEPS = 999, 24000
PARENT_ADAM_STEP, PARENT_CRITIC_COUNT = 40000, 12288000
WEIGHT = driver.WEIGHT
TAUS = dict.fromkeys(ARMS, .5)
KL_PROFILE = "selected-motor-headroom"
ACTUATORS = ("left_knee", "left_ankle", "right_knee", "right_ankle")
TERM = "selected_motor_load"
LOAD_WEIGHTS = {"control": 0., "headroom": -.25}
PARAMS = dict(actuator_names=ACTUATORS, rated_stall_torque_nm=.60,
              soft_limit_fraction=.70, over_limit_gain=4.)


def verify_source(args, *, training):
    return driver.verify_source(args, training=training, run_prefix=RUN_PREFIX)


def _owned_payload(path, expected):
    from .community_policy_inspection import _read_payload
    path = Path(path)
    info = path.lstat()
    base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
                 "owned regular sealed lesson artifact")
    raw = _read_payload(path)
    base.require(hashlib.sha256(raw).hexdigest() == expected,
                 "exact sealed lesson artifact bytes")
    return raw


def _read_parent():
    summary = json.loads(_owned_payload(PARENT_STATE / "gait-scale-summary.json", SUMMARY_SHA))
    manifest = json.loads(_owned_payload(PARENT_STATE / "gait-scale-manifest.json", MANIFEST_SHA))
    base.require(summary["source"] == manifest["source"] ==
                 "1b223f93608e14ffcfbfb8be5e819846d032c3e0"
                 and summary["status"] == "verified-evidence-not-policy-acceptance"
                 and summary["training"]["gait-scale"]["foundation"]["checkpoint_sha256"] == PARENT_SHA
                 and manifest["files"]["gait-scale/foundation/model_999.pt"]["sha256"] == PARENT_SHA,
                 "sealed previous run binds the provisional parent")
    driver.validate_parent_live_profile({"training": {
        "foundation": summary["training"]["gait-scale"]["foundation"]}})
    from . import projected_heading_retention, steady_steering_retention
    projected_heading_retention.load_baseline()
    steady_steering_retention.load_baseline()
    payload = torch.load(io.BytesIO(_owned_payload(PARENT, PARENT_SHA)),
                         map_location="cpu", weights_only=True)
    driver.validate_parent_payload(payload, iteration=PARENT_ITERATION, steps=PARENT_STEPS,
                                   adam_step=PARENT_ADAM_STEP, critic_count=PARENT_CRITIC_COUNT)
    from .v8_actor_checkpoint import FrozenV8Checkpoint
    typed = FrozenV8Checkpoint(PARENT, PARENT_SHA)
    base.require(driver._tree_digest(payload["actor_state_dict"]) ==
                 driver._tree_digest(typed._actor_state), "authenticated typed parent actor")
    return payload


def initialize(runner, env, output):
    return driver.initialize(runner, env, output, experiment=sys.modules[__name__])


def capture_motor_metric(env):
    """Capture actual weighted reward before autoreset; control is truly inactive."""
    from .tasks import mdp
    base.require(getattr(env, "_microduck_selected_load_snapshot", None) is None,
                 "previous selected-load snapshot consumed")
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(term.func is mdp.selected_motor_torque_load_cost and term.params == PARAMS
                 and term.weight in LOAD_WEIGHTS.values(), "exact live selected load objective")
    cost = term.func(env, **term.params)
    active = env.reward_manager.active_terms
    base.require(TERM in active, "registered selected motor reward column")
    weighted = env.reward_manager._step_reward[:, active.index(TERM)]
    if term.weight == 0:
        # mjlab calls its registered column list active_terms even when a term
        # has weight0; compute() skips that callable and zeros the column.
        base.require(torch.equal(weighted, torch.zeros_like(cost)),
                     "control selected load reward actually skipped")
    base.require(base.finite_tree((cost, weighted)) and bool((cost >= 0).all())
                 and bool((weighted <= 0).all())
                 and torch.allclose(weighted, term.weight * cost, atol=1e-6, rtol=1e-5),
                 "actual pre-reset selected motor reward accounting")
    env._microduck_selected_load_snapshot = {
        "step": env.common_step_counter, "cost": cost.detach().clone(),
        "weighted": weighted.detach().clone(), "weight": term.weight}
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode, arm):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .tasks import mdp
    base.require(mode in MODES and arm in ARMS, "predeclared motor lesson stage")
    cfg, agent = driver.prepare_config(mode, "gait-scale")
    cfg.rewards[TERM] = RewardTermCfg(func=mdp.selected_motor_torque_load_cost,
                                     weight=LOAD_WEIGHTS[arm], params=copy.deepcopy(PARAMS))
    cfg.metrics["microduck_selected_motor_load"] = MetricsTermCfg(
        func=capture_motor_metric, per_substep=False)
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
        env["rewards"][TERM]["weight"] = "<arm-weight>"
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode):
    left, right = [_canonical(mode, arm, mask=True) for arm in ARMS]
    base.require(left == right, "only selected motor reward weight differs")
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    full = [digest(_canonical(mode, arm, mask=False)) for arm in ARMS]
    return dict(protocol=PROTOCOL + ":recipe-v1", mode=mode,
        control_recipe_sha256=digest(left), treatment_recipe_sha256=digest(right),
        comparison_recipe_sha256=digest(left), control_full_recipe_sha256=full[0],
        treatment_full_recipe_sha256=full[1], full_recipe_hashes_distinct=full[0] != full[1],
        comparison_mask=["reward selected_motor_load.weight"], matched_after_declared_mask=True,
        load_weights=LOAD_WEIGHTS, actuator_names=list(ACTUATORS),
        policy_acceptance=False, physical_motion_authorized=False)


def frozen_profile_report(source, arm, mode):
    value = driver.frozen_profile_report(source, "gait-scale", mode)
    value.update(protocol=PROTOCOL + ":frozen-profile-v1", arm=arm, parent_sha256=PARENT_SHA,
                 selected_load_weight=LOAD_WEIGHTS[arm], selected_load_params=PARAMS)
    return json.loads(json.dumps(value))


class Observer(driver.Observer):
    def __init__(self, mode, arm, *, bias_weight=None):
        base.require(arm in ARMS, "declared motor lesson arm")
        super().__init__(mode, "gait-scale", protocol=PROTOCOL,
                         report_arm=arm, receipt_profile=KL_PROFILE, bias_weight=bias_weight)
        self.selected_arm = arm
        self.motor_costs, self.motor_rewards, self.motor_forces = [], [], []

    def before(self, env):
        super().before(env)
        from .tasks import mdp
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is mdp.selected_motor_torque_load_cost and term.params == PARAMS
                     and term.weight == LOAD_WEIGHTS[self.selected_arm]
                     and getattr(env, "_microduck_selected_load_snapshot", None) is None,
                     "frozen selected motor objective and clean snapshot boundary")

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        cost, weighted, force = consume_motor_snapshot(env, sample, LOAD_WEIGHTS[self.selected_arm])
        self.motor_costs.append(float(cost.mean()))
        self.motor_rewards.append(float(weighted.mean()))
        self.motor_forces.append(force)

    def update(self, env):
        row = super().update(env)
        base.require(len(self.motor_costs) == len(self.motor_rewards) == len(self.motor_forces) == 24,
                     "complete selected motor observation at PPO update")
        forces = torch.stack(self.motor_forces).double().abs().flatten(0, 1)
        row.update(selected_motor_load_weight=LOAD_WEIGHTS[self.selected_arm],
            selected_motor_cost_mean=math.fsum(self.motor_costs)/24,
            selected_motor_weighted_reward_mean=math.fsum(self.motor_rewards)/24,
            selected_motor_observed_steps=24, selected_actuator_names=list(ACTUATORS),
            selected_motor_p99_nm_by_actuator=torch.quantile(forces, .99, dim=0).tolist())
        self.motor_costs.clear(); self.motor_rewards.clear(); self.motor_forces.clear()
        return row


def consume_motor_snapshot(env, sample, weight):
    """mjlab increments its env counter before rewards/metrics, stream IDs are zero-based."""
    snap = getattr(env, "_microduck_selected_load_snapshot", None)
    base.require(isinstance(snap, dict) and snap["step"] == env.common_step_counter
                 and snap["step"] == sample.step + 1 and snap["weight"] == weight,
                 "same-step selected motor snapshot crosses reset exactly once")
    columns = [sample.joint_names.index(name) for name in ACTUATORS]
    expected = sample.joint_cost[:, columns].mean(-1)
    base.require(torch.allclose(snap["cost"].double(), expected, atol=1e-6, rtol=1e-5)
                 and torch.allclose(snap["weighted"].double(), weight * expected,
                                   atol=1e-6, rtol=1e-5),
                 "actual selected reward matches named pre-reset motor stream")
    weighted = snap["weighted"]
    env._microduck_selected_load_snapshot = None
    return expected, weighted, sample.force_nm[:, columns].detach().clone()


def validate_training_receipt(root, arm, mode, source):
    value = driver.validate_training_receipt(root, arm, mode, source,
                                             experiment=sys.modules[__name__])
    rows = [json.loads(line) for line in
            (Path(root) / arm / mode / "updates.jsonl").read_text().splitlines()]
    for row in rows:
        base.require(row["selected_motor_load_weight"] == LOAD_WEIGHTS[arm]
            and row["selected_motor_observed_steps"] == 24
            and row["selected_actuator_names"] == list(ACTUATORS)
            and row["selected_motor_cost_mean"] >= 0
            and row["selected_motor_weighted_reward_mean"] <= 0
            and math.isclose(row["selected_motor_weighted_reward_mean"],
                LOAD_WEIGHTS[arm] * row["selected_motor_cost_mean"], abs_tol=1e-6, rel_tol=1e-5),
            "retained actual selected motor reward accounting")
    base.require(any(row["selected_motor_cost_mean"] > 0 for row in rows),
                 "nonzero selected load in every admitted stage")
    return value


def validate_evaluation_receipt(root, arm, source, checkpoint_sha):
    return driver.validate_evaluation_receipt(root, arm, source, checkpoint_sha,
                                              experiment=sys.modules[__name__])


def comparison(control, treatment):
    """Prospective lesson decision, separate from unchanged policy qualification."""
    from . import gait_path_qualification as route
    expected = [(seed, speed, yaw) for seed in base.SEEDS for speed, yaw in base.CASES]
    for rows in (control, treatment):
        base.require(len(rows) == 12 and [(r["seed"], r["speed"], r["yaw"]) for r in rows] == expected
                     and all(r["complete"] is True and r["worlds"] == 8
                             and r["steps"] == 240 and r["terminal_worlds"] == [] for r in rows),
                     "ordered complete first-episode headroom comparison")
    reports, failures = {}, []
    for arm, rows in zip(ARMS, (control, treatment), strict=True):
        routes = [route.summarize_case(row, row["gait_path_trace"]) for row in rows]
        groups = []
        for speed, yaw in base.CASES:
            selected = [r for r in rows if (r["speed"], r["yaw"]) == (speed, yaw)]
            paths = [r for r in routes if (r["speed"], r["yaw"]) == (speed, yaw)]
            by_actuator = {}
            for name in ACTUATORS:
                values = [r["response_diagnostics"]["abs_force_p99_nm_by_joint"][
                    r["response_diagnostics"]["joint_columns"].index(name)] for r in selected]
                base.require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                 for v in values), "finite nonnegative named motor p99")
                by_actuator[name] = max(values)
            groups.append(dict(command=[speed, yaw], worst_case_p99_nm_by_actuator=by_actuator,
                speed_mae=math.fsum(v for r in selected for v in r["speed_mae"])/24,
                speed_passes=sum(v <= (.03 if speed == 0 else .1)
                                 for r in selected for v in r["speed_mae"]),
                full_heading_error_rad_s=math.fsum(w["full_settled"]["heading_error_rad_s"]
                    for r in paths for w in r["worlds"])/24))
        reports[arm] = dict(groups=groups, worst_moving_selected_p99_nm=max(
            v for g in groups[1:] for v in g["worst_case_p99_nm_by_actuator"].values()))
    a, b = reports["control"], reports["headroom"]
    if b["worst_moving_selected_p99_nm"] > .95 * a["worst_moving_selected_p99_nm"]:
        failures.append("worst-selected-moving-p99-not-reduced-five-percent")
    for ga, gb in zip(a["groups"][1:], b["groups"][1:], strict=True):
        if any(gb["worst_case_p99_nm_by_actuator"][name] >
               1.05 * ga["worst_case_p99_nm_by_actuator"][name] for name in ACTUATORS):
            failures.append([gb["command"], "individual-selected-load-regression-over-five-percent"])
        if gb["speed_mae"] > ga["speed_mae"] + .02:
            failures.append([gb["command"], "paired-speed-mae-regression"])
        if gb["full_heading_error_rad_s"] > ga["full_heading_error_rad_s"] + .01:
            failures.append([gb["command"], "paired-full-heading-regression"])
    if sum(g["speed_passes"] for g in b["groups"][1:]) < sum(g["speed_passes"] for g in a["groups"][1:]):
        failures.append("paired-moving-speed-pass-count-regression")
    return dict(protocol=PROTOCOL + ":comparison-v1", groups=reports, failures=failures,
        decision="headroom-lesson-useful-not-qualified" if not failures else "headroom-lesson-rejected",
        thresholds=dict(worst_selected_p99_reduction=.05, per_command_per_actuator_relative_allowance=.05,
                        paired_speed_mae_allowance_mps=.02, paired_heading_allowance_rad_s=.01),
        policy_acceptance=False, physical_motion_authorized=False)


def compare_retained(source, run_id):
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
                 "comparison is CPU-only")
    verify_source(argparse.Namespace(source=source, run_id=run_id), training=False)
    root = base.ROOT / "artifacts/training" / run_id
    receipts, rows = {}, {}
    for arm in ARMS:
        for mode in MODES:
            validate_training_receipt(root, arm, mode, source)
        checkpoint_sha = base.sha256(root / arm / "foundation/model_999.pt")
        receipts[arm], rows[arm] = validate_evaluation_receipt(root, arm, source, checkpoint_sha)
    report = comparison(rows["control"], rows["headroom"])
    for arm, receipt in receipts.items():
        e = receipt["evaluation"]
        if any(e[key]["decision"] != "walking-idle-retained"
               for key in ("projected_heading_retention", "cautious_retention")):
            report["failures"].append([arm, "sealed-walking-idle-retention-failed"])
        if any(not case["safety"]["qualified"] for case in e["cases"]):
            report["failures"].append([arm, "unchanged-motor-envelope-gate-failed"])
    if report["failures"]:
        report["decision"] = "headroom-lesson-rejected"
    report.update(source=source, run_id=run_id, parent_sha256=PARENT_SHA,
        evaluated_result_sha256={arm: base.sha256(root / arm / "final-eval/result.json") for arm in ARMS},
        checkpoint_sha256={arm: base.sha256(root / arm / "foundation/model_999.pt") for arm in ARMS})
    base.write_new(root / "headroom-decision.json", report)
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
