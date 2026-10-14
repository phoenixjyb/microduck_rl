"""One-factor gait-averaged planar-speed retention experiment.

The rejected head curriculum stays closed. Both arms use the original frozen
parent recipe; only the new planar EMA cost weight differs. No policy acceptance
or physical authority follows from this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from . import integrated_turn_foundation as base
from . import v8_gait_scale_contrast as driver
from . import v8_actor_lr_profile as profiles
from . import v8_motor_headroom as headroom
from . import v8_steering_budget as budget
from . import v8_staged_head_goals as heads

PROTOCOL = "v8-planar-bias-retention-v1"
RUN_PREFIX = "v8-planar-bias-s991"
ARMS = ("control", "planar")
MODES = {"smoke": (64, 5, 947), "benchmark": (256, 10, 953),
         "foundation": (256, 1000, 991)}
STAGE_SPECS = None
CAPS = dict(driver.CAPS)
PARENT = budget.PARENT
PARENT_SHA = budget.PARENT_SHA
PARENT_ITERATION, PARENT_STEPS = budget.PARENT_ITERATION, budget.PARENT_STEPS
PARENT_ADAM_STEP, PARENT_CRITIC_COUNT = budget.PARENT_ADAM_STEP, budget.PARENT_CRITIC_COUNT
TAUS = dict.fromkeys(ARMS, .5)
WEIGHT = -1.0  # The EXISTING yaw-bias price stays unchanged.
KL_PROFILE = "planar-bias-retention"
TERM = "planar_bias"
PLANAR_WEIGHTS = {"control": 0.0, "planar": -1.0}
PARAMS = {"command_name": "twist", "tau_s": .5}
ROOT = base.ROOT
REJECTED_SOURCE = "986c6f4facb44333a3daa76d1f411058c0820ac7"
REJECTED_RUN = "v8-staged-head-replication-s977-986c6f4f-20261014"
REJECTED_STATE = Path("/home/converge/.local/state") / (REJECTED_RUN + "-handoff")
SUMMARY_SHA = "a344c638a7da8f398cc1832d4da41680f8cdeb5c0dc251366b3d8b9f64c82634"
MANIFEST_SHA = "66081b90b72ed98a4ae01cda7198b09085ed83167166d4027e486ea61c8cdfaa"
DECISION_USEFUL = "planar-bias-lesson-useful-not-qualified"
DECISION_REJECTED = "planar-bias-lesson-rejected"
DECISION_FILENAME = "planar-bias-decision.json"


def verify_source(args, *, training):
    return driver.verify_source(args, training=training, run_prefix=RUN_PREFIX)


def _read_parent():
    summary = json.loads(headroom._owned_payload(REJECTED_STATE / "replication-summary.json", SUMMARY_SHA))
    manifest = json.loads(headroom._owned_payload(REJECTED_STATE / "replication-manifest.json", MANIFEST_SHA))
    report = summary.get("comparison", {})
    groups = report.get("groups", {})
    base.require(summary.get("source") == manifest.get("source") == REJECTED_SOURCE
        and summary.get("protocol") == "v8-staged-head-replication-v1"
        and summary.get("status") == "verified-evidence-not-policy-acceptance"
        and summary.get("learning_seed") == report.get("learning_seed") == 977
        and report.get("decision") == "staged-head-replication-rejected"
        and groups.get("control", {}).get("complete_moving_route_worlds") == 15
        and groups.get("staged", {}).get("complete_moving_route_worlds") == 8
        and all(groups.get(arm, {}).get("moving_speed_passes") == 67 for arm in ("control", "staged"))
        and summary.get("parent_checkpoint_sha256") == PARENT_SHA
        and manifest.get("files", {}).get("staged-head-replication-decision.json", {}).get("sha256")
            == summary.get("comparison_sha256")
        and summary.get("comparison_sha256") == "6786ae7b4b4c81c154fb0a5fb24d4d6fa68a5f3bca052fbc7f245fe4b1aa8246"
        and report.get("policy_acceptance") is False and report.get("physical_motion_authorized") is False
        and summary.get("policy_acceptance") is False
        and summary.get("physical_motion_authorized") is False,
        "authenticated rejected seed977 evidence; no rejected endpoint becomes a parent")
    return budget._read_parent()


def prepare_config(mode, arm):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .tasks import mdp
    base.require(mode in MODES and arm in ARMS, "declared planar-bias stage and arm")
    cfg, agent = budget.prepare_config(mode, "control")
    cfg.seed = agent.seed = MODES[mode][2]
    cfg.rewards[TERM] = RewardTermCfg(func=mdp.body_planar_bias_cost,
        weight=PLANAR_WEIGHTS[arm], params=dict(PARAMS))
    cfg.metrics["microduck_planar_bias"] = MetricsTermCfg(func=capture_planar_metric, per_substep=False)
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def _canonical(mode, arm, *, mask):
    cfg, agent = prepare_config(mode, arm)
    env, algo = asdict(cfg), asdict(agent)
    # CLI execution names this callable __main__; imports use its real module.
    # Recipe identity must not depend on how the identical source was invoked.
    env["metrics"]["microduck_planar_bias"]["func"] = (
        "mjlab_microduck.v8_planar_bias:capture_planar_metric")
    if mask:
        env["rewards"][TERM]["weight"] = "<declared-planar-cost-weight>"
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode):
    left, right = [_canonical(mode, arm, mask=True) for arm in ARMS]
    base.require(left == right, "only planar EMA cost weight differs between matched arms")
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    full = [_canonical(mode, arm, mask=False) for arm in ARMS]
    return dict(protocol=PROTOCOL + ":recipe-v1", mode=mode,
        control_recipe_sha256=digest(left), treatment_recipe_sha256=digest(right),
        comparison_recipe_sha256=digest(left),
        control_full_recipe_sha256=digest(full[0]), treatment_full_recipe_sha256=digest(full[1]),
        full_recipe_hashes_distinct=full[0] != full[1],
        comparison_mask=["rewards.planar_bias.weight"], planar_weights=PLANAR_WEIGHTS,
        matched_after_declared_mask=True, policy_acceptance=False, physical_motion_authorized=False)


def frozen_profile_report(source, arm, mode):
    value = budget.frozen_profile_report(source, "control", mode)
    value.update(protocol=PROTOCOL + ":frozen-profile-v1", arm=arm,
        stage_spec=list(MODES[mode]), planar_bias_weight=PLANAR_WEIGHTS[arm],
        planar_bias_params=dict(PARAMS), rejected_replication_summary_sha256=SUMMARY_SHA)
    return json.loads(json.dumps(value))


def initialize(runner, env, output):
    return driver.initialize(runner, env, output, experiment=sys.modules[__name__])


def _planar_term(env):
    from .tasks import mdp
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(term.func is mdp.body_planar_bias_cost and term.params == PARAMS
        and type(term.weight) in (int, float) and term.weight in PLANAR_WEIGHTS.values(),
        "exact live planar bias cost function, horizon and declared weight")
    return term


def capture_planar_metric(env):
    """Capture actual weighted reward before autoreset; zero arm stays inert."""
    base.require(getattr(env, "_microduck_planar_bias_snapshot", None) is None,
                 "previous planar snapshot consumed")
    term = _planar_term(env)
    cost = term.func(env, **term.params)
    state = env._microduck_body_planar_bias_state
    base.require(state["step"] == env.common_step_counter,
                 "planar state belongs to current post-physics reward step")
    weighted = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(TERM)]
    base.require(cost.shape == weighted.shape == (env.num_envs,)
        and base.finite_tree((cost, weighted)) and bool((cost >= 0).all())
        and bool((weighted <= 0).all())
        and torch.allclose(weighted, term.weight * cost, atol=1e-6, rtol=1e-5),
        "actual nonpositive weighted planar reward matches configured cost")
    if term.weight == 0:
        base.require(torch.equal(weighted, torch.zeros_like(weighted)), "control planar reward is exactly zero")
    env._microduck_planar_bias_snapshot = {key: state[key].detach().clone()
        for key in ("command", "signed_residual", "signed_ema", "cost")}
    env._microduck_planar_bias_snapshot.update(step=state["step"], weight=float(term.weight),
                                             weighted_reward=weighted.detach().clone())
    return torch.zeros(env.num_envs, device=env.device)


class Observer(headroom.Observer):
    def __init__(self, mode, arm):
        base.require(arm in ARMS, "declared planar-bias observer arm")
        super().__init__(mode, "control")
        self.protocol, self.report_arm, self.receipt_profile = PROTOCOL, arm, KL_PROFILE
        self.planar_samples = []

    def before(self, env):
        super().before(env)
        base.require(_planar_term(env).weight == PLANAR_WEIGHTS[self.report_arm]
            and getattr(env, "_microduck_planar_bias_snapshot", None) is None,
            "frozen per-arm planar objective and clean snapshot boundary")

    def step(self, env, sample, extras):
        # The inherited yaw observer consumes and clears its command in step().
        # Keep our read-only reference before that consumption, not after it.
        command = self._bias_command
        base.require(isinstance(command, torch.Tensor), "original pre-action command remains available")
        super().step(env, sample, extras)
        snap = getattr(env, "_microduck_planar_bias_snapshot", None)
        base.require(isinstance(snap, dict) and snap.get("step") == sample.step + 1
            and snap["weight"] == PLANAR_WEIGHTS[self.report_arm]
            and snap["command"].shape == (env.num_envs, 3)
            and snap["signed_residual"].shape == snap["signed_ema"].shape == (env.num_envs, 2)
            and base.finite_tree(snap)
            and torch.equal(snap["command"], command)
            and torch.equal(snap["cost"], snap["signed_ema"].abs().sum(-1))
            and torch.allclose(snap["weighted_reward"], snap["weight"] * snap["cost"], atol=1e-6, rtol=1e-5),
            "exact pre-reset planar snapshot and original actor command")
        self.planar_samples.append({key: snap[key].detach().double().cpu()
            for key in ("command", "signed_residual", "signed_ema", "cost", "weighted_reward")})
        env._microduck_planar_bias_snapshot = None

    def update(self, env):
        row = super().update(env)
        base.require(len(self.planar_samples) == 24, "complete planar observation per PPO update")
        tensors = {key: torch.stack([sample[key] for sample in self.planar_samples])
                   for key in self.planar_samples[0]}
        row.update(planar_bias_weight=PLANAR_WEIGHTS[self.report_arm], planar_bias_tau_s=.5,
            planar_bias_observed_steps=24, planar_bias_sample_count=24 * env.num_envs,
            planar_bias_cost_mean=float(tensors["cost"].mean()),
            planar_bias_weighted_reward_mean=float(tensors["weighted_reward"].mean()),
            planar_bias_signed_residual_mean=tensors["signed_residual"].mean((0, 1)).tolist(),
            planar_bias_raw_abs_mean=tensors["signed_residual"].abs().mean((0, 1)).tolist(),
            planar_bias_signed_ema_mean=tensors["signed_ema"].mean((0, 1)).tolist(),
            planar_bias_abs_ema_mean=tensors["signed_ema"].abs().mean((0, 1)).tolist(),
            planar_bias_activity_steps=sum(bool((sample["cost"] > 0).any()) for sample in self.planar_samples))
        self.planar_samples.clear()
        return row


def validate_training_receipt(root, arm, mode, source):
    value = driver.validate_training_receipt(root, arm, mode, source, experiment=sys.modules[__name__])
    rows = [json.loads(line) for line in (Path(root) / arm / mode / "updates.jsonl").read_text().splitlines()]
    for row in rows:
        base.require(row["selected_motor_load_weight"] == row["selected_motor_weighted_reward_mean"] == 0
            and row["selected_motor_observed_steps"] == 24
            and row["selected_actuator_names"] == list(headroom.ACTUATORS)
            and row["planar_bias_weight"] == PLANAR_WEIGHTS[arm] and row["planar_bias_tau_s"] == .5
            and row["planar_bias_observed_steps"] == 24
            and row["planar_bias_sample_count"] == 24 * value["worlds"]
            and 0 <= row["planar_bias_activity_steps"] <= 24
            and row["planar_bias_cost_mean"] >= 0 and row["planar_bias_weighted_reward_mean"] <= 0
            and math.isclose(row["planar_bias_weighted_reward_mean"],
                PLANAR_WEIGHTS[arm] * row["planar_bias_cost_mean"], abs_tol=1e-6, rel_tol=1e-5)
            and len(row["planar_bias_abs_ema_mean"]) == 2
            and math.isclose(sum(row["planar_bias_abs_ema_mean"]), row["planar_bias_cost_mean"], abs_tol=1e-6),
            "actual planar reward/cost accounting; inherited selected-load term stays inactive")
    base.require(any(row["planar_bias_cost_mean"] > 0 for row in rows), "actual nonzero planar errors observed")
    return value


def validate_evaluation_receipt(root, arm, source, checkpoint_sha):
    return driver.validate_evaluation_receipt(root, arm, source, checkpoint_sha, experiment=sys.modules[__name__])


def diagnostic_comparison(control, planar):
    report = heads.diagnostic_comparison(control, planar)
    report["groups"]["planar"] = report["groups"].pop("staged")
    report["failures"] = [failure if isinstance(failure, str) else
        ["planar" if failure[0] == "staged" else failure[0], *failure[1:]] for failure in report["failures"]]
    report.update(protocol=PROTOCOL + ":comparison-v1", planar_weights=PLANAR_WEIGHTS,
        weights=dict.fromkeys(ARMS, WEIGHT),
        decision=DECISION_REJECTED if report["failures"] else DECISION_USEFUL)
    return report


def compare_retained(source, run_id):
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
                 "CPU-only planar-bias comparison")
    verify_source(argparse.Namespace(source=source, run_id=run_id), training=False)
    _read_parent()
    from . import steady_steering_retention as retention
    parent = json.loads(headroom._owned_payload(budget.SUMMARY, budget.SUMMARY_SHA))
    baseline = {"source": parent["source"], "evaluations": {"final-eval": parent["evaluations"]["control"]}}
    root = ROOT / "artifacts/training" / run_id
    receipts, rows, parent_retention = {}, {}, {}
    for arm in ARMS:
        for mode in MODES:
            validate_training_receipt(root, arm, mode, source)
        checkpoint = driver.foundation_checkpoint(root, sys.modules[__name__], arm)
        receipts[arm], rows[arm] = validate_evaluation_receipt(root, arm, source, base.sha256(checkpoint))
        parent_retention[arm] = retention._compare_sealed_reference(rows[arm], baseline,
            source=parent["source"], checkpoint_sha=PARENT_SHA, summary_sha=budget.SUMMARY_SHA,
            protocol=PROTOCOL + ":original-parent-retention-v1")
    report = diagnostic_comparison(rows["control"], rows["planar"])
    for arm in ARMS:
        for key, gate in {"original_parent": parent_retention[arm], **{
            name: receipts[arm]["evaluation"][name] for name in
            ("projected_heading_retention", "cautious_retention")}}.items():
            if gate["decision"] != "walking-idle-retained":
                report["failures"].append([arm, key, "walking-idle-retention-failure"])
    if report["failures"]:
        report["decision"] = DECISION_REJECTED
    report.update(source=source, run_id=run_id, learning_seed=991, parent_sha256=PARENT_SHA,
        original_parent_retention=parent_retention, rejected_replication_summary_sha256=SUMMARY_SHA,
        qualification_gates={arm: {key: receipts[arm]["evaluation"][key] for key in
            ("legacy_gate", "route_gate")} for arm in ARMS},
        checkpoint_sha256={arm: base.sha256(driver.foundation_checkpoint(root, sys.modules[__name__], arm)) for arm in ARMS},
        evaluated_result_sha256={arm: base.sha256(root / arm / "final-eval/result.json") for arm in ARMS})
    base.write_new(root / DECISION_FILENAME, report)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", required=True)
        parser.add_argument("--run-id", required=True)
        args = parser.parse_args(argv[1:])
        return compare_retained(args.source, args.run_id)
    return driver.main(sys.modules[__name__], argv=argv)


if __name__ == "__main__":
    main()
