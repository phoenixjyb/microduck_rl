"""Bounded v8-seeded steady-steering experiment with a fixed yaw-bias cost.

The only recipe delta from cautious v8 learning is one fixed-weight reward
term. The reward EMA is diagnostic/training reward state only; it never edits
observations, actor actions, filters, commands, or the plant.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import integrated_turn_foundation as base
from . import integrated_joint_limit_refinement as limits
from . import integrated_kl_budget_refinement as kl
from . import integrated_training_budget_refinement as exposure
from . import official_native_reference as pinned
from . import v8_actor_learning as learning
from . import v8_actor_lr_profile as profiles
from . import v8_actor_smoke as smoke

PROTOCOL = "v8-steady-steering-v1"
RUN_PREFIX = "v8-steady-steering"
MODES = {
    "smoke": (64, 5, 947),
    "benchmark": (256, 10, 953),
    "foundation": (256, 1000, 967),
}
CAPS = {"smoke": 300, "benchmark": 300, "initial-eval": 600,
        "foundation": 2100, "final-eval": 600}
TERM = "yaw_bias"
TAU_S = 1.0
BIAS_WEIGHT = -1.0
FIRST_KL_MEAN_MAX = .05
FIRST_KL_MINIBATCH_MEAN_MAX = .10


def bias_weight(mode: str) -> float:
    base.require(mode in MODES, "known steady-steering training mode")
    return BIAS_WEIGHT


def prepare_config(mode: str):
    """Start from the cautious recipe and add only the declared yaw-bias term."""
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .tasks import mdp

    base.require(mode in MODES, "known steady-steering training mode")
    cfg, agent = learning.prepare_config(mode, profile="cautious")
    cfg.rewards[TERM] = RewardTermCfg(
        func=mdp.body_yaw_bias_cost, weight=BIAS_WEIGHT,
        params=dict(command_name="twist", tau_s=TAU_S))
    cfg.metrics["microduck_yaw_bias"] = MetricsTermCfg(
        func=capture_bias_metric, per_substep=False)
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def _canonical_config(cfg, agent) -> str:
    """Canonicalize a complete recipe, masking only campaign/run identity."""
    cfg_value, agent_value = asdict(cfg), asdict(agent)
    metric_cfg = cfg.metrics.get("microduck_yaw_bias")
    if metric_cfg is not None:
        base.require(metric_cfg.func is capture_bias_metric and metric_cfg.per_substep is False,
                     "known steady-steering metric callable before canonicalization")
        cfg_value["metrics"]["microduck_yaw_bias"]["func"] = (
            "mjlab_microduck.v8_steady_steering:capture_bias_metric")
    agent_value["experiment_name"] = "<protocol>"
    agent_value["run_name"] = "<mode>"
    value = profiles._canonical({"environment": cfg_value, "agent": agent_value})
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode: str) -> dict:
    """Prove the treatment removes to exactly the cautious config."""
    from .tasks import mdp

    steady_cfg, steady_agent = prepare_config(mode)
    cautious_cfg, cautious_agent = learning.prepare_config(mode, profile="cautious")
    reduced = copy.deepcopy(steady_cfg)
    term = reduced.rewards.pop(TERM)
    metric = reduced.metrics.pop("microduck_yaw_bias")
    base.require(term.func is mdp.body_yaw_bias_cost and term.weight == BIAS_WEIGHT
                 and term.params == dict(command_name="twist", tau_s=TAU_S)
                 and metric.func is capture_bias_metric and metric.per_substep is False,
                 "exact steady-steering reward/metric removal")
    left, right = _canonical_config(reduced, steady_agent), _canonical_config(cautious_cfg, cautious_agent)
    digest = hashlib.sha256(_canonical_config(steady_cfg, steady_agent).encode()).hexdigest()
    base.require(left == right, "steady recipe differs from cautious only by its fixed reward and metric")
    return dict(protocol="v8-steady-steering-recipe-match-v2", mode=mode,
                cautious_recipe_sha256=hashlib.sha256(right.encode()).hexdigest(),
                reduced_steady_recipe_sha256=hashlib.sha256(left.encode()).hexdigest(),
                steady_recipe_sha256=digest, fixed_reward_weight=BIAS_WEIGHT,
                fixed_tau_s=TAU_S, only_recipe_delta="yaw_bias reward and read-only metric",
                matched_after_removing_owned_reward_and_metric=True,
                identity_masked_fields=["experiment_name", "run_name"])


def capture_bias_metric(env):
    """Snapshot cached reward state and actual weighted reward before reset."""
    from .tasks import mdp

    base.require(getattr(env, "_microduck_yaw_bias_snapshot", None) is None,
                 "previous yaw-bias snapshot consumed")
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(term.func is mdp.body_yaw_bias_cost and term.weight == BIAS_WEIGHT
                 and term.params == dict(command_name="twist", tau_s=TAU_S),
                 "live fixed yaw-bias reward weight/function/parameters")
    cost = term.func(env, **term.params)
    state = env._microduck_body_yaw_bias_state
    base.require(state["step"] == env.common_step_counter,
                 "post-reward yaw-bias cache belongs to live common step")
    index = env.reward_manager.active_terms.index(TERM)
    weighted = env.reward_manager._step_reward[:, index]
    base.require(isinstance(weighted, torch.Tensor) and weighted.shape == cost.shape
                 and base.finite_tree(weighted) and bool((weighted <= 0).all())
                 and torch.allclose(weighted, BIAS_WEIGHT * cost, atol=1e-6, rtol=1e-5),
                 "actual nonpositive weighted yaw-bias reward matches cached cost")
    env._microduck_yaw_bias_snapshot = {
        key: state[key].detach().clone()
        for key in ("signed_residual", "signed_ema", "cost")
    }
    env._microduck_yaw_bias_snapshot["weighted_reward"] = weighted.detach().clone()
    base.require(torch.equal(cost, env._microduck_yaw_bias_snapshot["cost"]),
                 "same-step cached cost capture is idempotent")
    return torch.zeros(env.num_envs, device=env.device)


class Observer(learning.Observer):
    """Parent command/KL observer plus consumable, pre-reset yaw telemetry."""
    def __init__(self, mode):
        base.require(mode in MODES, "known steady-steering mode")
        super().__init__()
        self.mode, self.profile = mode, "stock"  # stock curriculum labels only
        self.bias_costs, self.bias_rewards, self.signed_biases = [], [], []
        self.raw_errors, self.signed_raw_errors = [], []
        self.bucket_counts = dict.fromkeys(exposure.EXPOSURE_BINS, 0)
        self.bucket_values = {key: {name: [] for name in
            ("raw_signed_residual", "signed_ema", "cost", "weighted_reward")}
            for key in exposure.EXPOSURE_BINS}
        self.activity_steps = 0
        self.first_update_report = None

    def before(self, env):
        from .tasks import mdp

        super().before(env)
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is mdp.body_yaw_bias_cost and term.weight == BIAS_WEIGHT
                     and term.params == dict(command_name="twist", tau_s=TAU_S),
                     "live steady-steering reward is fixed from step zero")
        base.require(getattr(env, "_microduck_yaw_bias_snapshot", None) is None,
                     "previous pre-reset yaw-bias snapshot consumed")
        self._bias_expected_step = env.common_step_counter
        self._bias_command = env.command_manager.get_command("twist").detach().clone()

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = getattr(env, "_microduck_yaw_bias_snapshot", None)
        expected = self._bias_expected_step
        base.require(isinstance(snapshot, dict)
                     and set(snapshot) == {"signed_residual", "signed_ema", "cost", "weighted_reward"}
                     and all(isinstance(v, torch.Tensor) and v.shape == sample.mean_cost.shape
                             for v in snapshot.values()) and base.finite_tree(snapshot),
                     "finite consumable pre-reset yaw-bias snapshot")
        base.require(env.common_step_counter == expected + 1,
                     "snapshot consumed once at its expected post-step counter")
        cost, ema, raw, weighted = (snapshot[k] for k in
            ("cost", "signed_ema", "signed_residual", "weighted_reward"))
        base.require(bool((cost >= 0).all()) and torch.equal(cost, ema.abs())
                     and bool((weighted <= 0).all())
                     and torch.allclose(weighted, BIAS_WEIGHT * cost, atol=1e-6, rtol=1e-5),
                     "finite nonpositive actual reward matches fixed weight times absolute EMA")
        command = self._bias_command
        bins = exposure.command_exposure(command)
        self.activity_steps += int(bool((cost > 0).any()))
        self.bias_costs.append(float(cost.mean()))
        self.bias_rewards.append(float(weighted.mean()))
        self.signed_biases.append(float(ema.mean()))
        self.raw_errors.append(float(raw.abs().mean()))
        self.signed_raw_errors.append(float(raw.mean()))
        masks = self._yaw_bucket_masks(command)
        for name, mask in masks.items():
            self.bucket_counts[name] += int(mask.sum())
            if bool(mask.any()):
                self.bucket_values[name]["raw_signed_residual"].append(float(raw[mask].sum()))
                self.bucket_values[name]["signed_ema"].append(float(ema[mask].sum()))
                self.bucket_values[name]["cost"].append(float(cost[mask].sum()))
                self.bucket_values[name]["weighted_reward"].append(float(weighted[mask].sum()))
        base.require(sum(bins.values()) == env.num_envs, "complete signed-yaw bucket coverage")
        env._microduck_yaw_bias_snapshot = None
        self._bias_expected_step = None
        self._bias_command = None

    @staticmethod
    def _yaw_bucket_masks(command):
        vx, vy, wz = command.unbind(-1)
        idle = (command == 0).all(-1)
        masks = {"idle": idle, "straight": (wz == 0) & ~idle}
        for sign, label in ((wz < 0, "negative"), (wz > 0, "positive")):
            magnitude = wz.abs()
            masks[label + "-low"] = sign & (magnitude <= .15)
            masks[label + "-target"] = sign & (magnitude > .15) & (magnitude <= .25)
            masks[label + "-high"] = sign & (magnitude > .25)
        return masks

    def update(self, env):
        row = super().update(env)
        base.require(all(len(v) == 24 for v in
                         (self.bias_costs, self.bias_rewards, self.signed_biases,
                          self.raw_errors, self.signed_raw_errors))
                     and sum(self.bucket_counts.values()) == 24 * env.num_envs,
                     "complete 24-step yaw-bias and signed-bucket metrics")
        if self.updates == 1:
            report = first_update_kl_report(self.mode, self.source, row["optimizer"],
                                            steps=env.common_step_counter)
            self.first_update_report = report
            base.write_new(self.output / "first-update-kl.json", report)
            row["first_update_kl"] = report
            if not report["passed"]:
                base.require(self.runner is not None, "runner attached for first-update refusal")
                self.runner.save(str(self.output / "aborted-first-update.pt"), infos={
                    "env_state": {"common_step_counter": env.common_step_counter},
                    "profile": "steady-steering", "mode": self.mode, "source": self.source,
                    "full_budget_completed": False, "policy_acceptance": False,
                    "physical_motion_authorized": False,
                    "first_update_kl_sha256": base.sha256(self.output / "first-update-kl.json"),
                })
                base.require(False, "steady-steering first-update KL guard")
        row.update(yaw_bias_weight=BIAS_WEIGHT, yaw_bias_tau_s=TAU_S,
                   yaw_bias_cost_mean=sum(self.bias_costs)/24,
                   weighted_yaw_bias_reward_mean=sum(self.bias_rewards)/24,
                   signed_yaw_bias_mean=sum(self.signed_biases)/24,
                   raw_yaw_residual_mae=sum(self.raw_errors)/24,
                   signed_raw_yaw_residual_mean=sum(self.signed_raw_errors)/24,
                   yaw_bias_observed_steps=24,
                   yaw_bias_activity_steps=self.activity_steps,
                   yaw_bias_signed_buckets={key: dict(count=self.bucket_counts[key],
                       raw_signed_residual_mean=(sum(value["raw_signed_residual"])/self.bucket_counts[key]
                           if self.bucket_counts[key] else None),
                       signed_ema_mean=(sum(value["signed_ema"])/self.bucket_counts[key]
                           if self.bucket_counts[key] else None),
                       cost_mean=(sum(value["cost"])/self.bucket_counts[key]
                           if self.bucket_counts[key] else None),
                       weighted_reward_mean=(sum(value["weighted_reward"])/self.bucket_counts[key]
                           if self.bucket_counts[key] else None))
                       for key, value in self.bucket_values.items()})
        for values in (self.bias_costs, self.bias_rewards, self.signed_biases,
                       self.raw_errors, self.signed_raw_errors):
            values.clear()
        self.bucket_counts = dict.fromkeys(exposure.EXPOSURE_BINS, 0)
        self.bucket_values = {key: {name: [] for name in
            ("raw_signed_residual", "signed_ema", "cost", "weighted_reward")}
            for key in exposure.EXPOSURE_BINS}
        self.activity_steps = 0
        return row


def first_update_kl_report(mode, source, summary, *, steps):
    mean_kl, max_batch_mean = summary["kl_mean_mean"], summary["kl_mean_max"]
    finite = (type(mean_kl) in (int, float) and math.isfinite(mean_kl)
              and type(max_batch_mean) in (int, float) and math.isfinite(max_batch_mean))
    passed = (steps == 24 and summary["optimizer_minibatches"] == 20 and finite
              and mean_kl <= FIRST_KL_MEAN_MAX and max_batch_mean <= FIRST_KL_MINIBATCH_MEAN_MAX)
    stored_mean = mean_kl if type(mean_kl) in (int, float) and math.isfinite(mean_kl) else None
    stored_max = (max_batch_mean if type(max_batch_mean) in (int, float)
                  and math.isfinite(max_batch_mean) else None)
    return dict(protocol="v8-steady-steering-first-update-kl-v1", profile="steady-steering",
        mode=mode, source=source, update=1, common_step_counter=steps, desired_kl=.01,
        initial_learning_rate=profiles.CAUTIOUS_INITIAL_LR,
        optimizer_minibatches=summary["optimizer_minibatches"], mean_raw_kl=stored_mean,
        maximum_minibatch_mean_raw_kl=stored_max, mean_raw_kl_limit=FIRST_KL_MEAN_MAX,
        maximum_minibatch_mean_raw_kl_limit=FIRST_KL_MINIBATCH_MEAN_MAX, passed=bool(passed),
        fixed_reward_weight=BIAS_WEIGHT, policy_acceptance=False, physical_motion_authorized=False)


def benchmark_projection_seconds(elapsed_s):
    base.require(type(elapsed_s) in (int, float) and math.isfinite(elapsed_s) and elapsed_s > 0,
                 "finite positive measured benchmark time")
    return elapsed_s * 100 * 1.5 + 60


def validate_command_coverage(result, worlds, updates):
    """Validate both schemas; base signed counts use a strict 0.01 threshold.

    Observer bins classify every nonzero yaw by sign, so small signed yaw values
    can also count as base "straight" while appearing in a signed low bin.
    """
    transitions = worlds * updates * 24
    bins = result.get("command_exposure")
    base_counts = {key: result.get(key) for key in base.FULL_COMMAND_COVERAGE}
    base.require(result.get("worlds") == worlds and result.get("updates") == updates
        and result.get("steps") == updates * 24 and result.get("transitions") == transitions,
        "exact world/update/step/transition budget")
    base.require(type(bins) is dict and set(bins) == set(exposure.EXPOSURE_BINS)
        and all(type(count) is int and count >= 0 for count in bins.values())
        and sum(bins.values()) == transitions,
        "exact eight-bin nonnegative actor command dwell totals")
    base.require(all(type(count) is int and count > 0 for count in base_counts.values())
        and base_counts["idle"] == bins["idle"]
        and base_counts["negative_yaw"] <= sum(bins[k] for k in
            ("negative-low", "negative-target", "negative-high"))
        and base_counts["positive_yaw"] <= sum(bins[k] for k in
            ("positive-low", "positive-target", "positive-high"))
        and bins["straight"] <= base_counts["straight"]
        and base_counts["straight"] <= bins["straight"] + bins["negative-low"] + bins["positive-low"],
        "base and signed-bucket command coverage mappings agree")


def validate_training_receipt(root, mode, source):
    base.require(mode in MODES, "declared steady-steering receipt stage")
    output = Path(root) / mode
    result = json.loads((output / "result.json").read_text())
    worlds, updates, seed = MODES[mode]
    checkpoint = output / f"model_{updates-1}.pt"
    base.require(result["protocol"] == PROTOCOL and result["profile"] == "steady-steering"
        and result["source"] == source and result["status"] == "training-complete-not-accepted"
        and (result["worlds"], result["updates"], result["seed"], result["steps"], result["transitions"])
            == (worlds, updates, seed, updates*24, worlds*updates*24)
        and Path(result["checkpoint"]).resolve() == checkpoint.resolve()
        and base.sha256(checkpoint) == result["checkpoint_sha256"]
        and base.sha256(output / "initial.pt") == result["initial_checkpoint_sha256"],
        "exact source-bound complete stage budget and checkpoint bytes")
    actor = json.loads((output / "actor-import.json").read_text())
    base.require(result["actor_import"] == actor and actor["ppo_resume"] is False
        and actor.get("actor_class") == "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
        and actor.get("initial_std") == smoke.INITIAL_STD
        and actor["adam_state_entries"] == 0 and actor["artifact_sha256"] == pinned.V8.sha256
        and actor.get("initial_learning_rate") == profiles.CAUTIOUS_INITIAL_LR
        and actor.get("target_iteration") == actor.get("target_common_step_counter") == 0
        and actor.get("critic_normalizer_count") == 0
        and actor["maximum_actor_error"] <= 2e-5 and actor["maximum_export_forward_error"] <= 2e-5,
        "fresh exact-byte v8 actor-only import")
    match = json.loads((output / "recipe-match.json").read_text())
    expected_match = recipe_match(mode)
    base.require(result["recipe_match"] == match == expected_match and match["mode"] == mode
        and match["protocol"] == "v8-steady-steering-recipe-match-v2"
        and match["fixed_reward_weight"] == BIAS_WEIGHT and match["matched_after_removing_owned_reward_and_metric"],
        "honest steady-versus-cautious recipe match proof")
    matched = json.loads((output / "matched-initialization.json").read_text())
    control = profiles.control_initial(mode)
    base.require(result["matched_initialization"] == matched
        and matched.get("protocol") == "v8-steady-steering-matched-initialization-v1"
        and matched.get("mode") == mode and matched.get("source") == source
        and matched.get("profile") == "steady-steering"
        and matched.get("seed") == MODES[mode][2]
        and matched.get("control_model_tag") == control.tag
        and matched.get("control_source") == control.source
        and matched.get("control_checkpoint_sha256") == control.checkpoint_sha256
        and matched.get("recipe_sha256") == match["steady_recipe_sha256"]
        and "stock_config_sha256" not in matched and "cautious_config_sha256" not in matched
        and matched.get("actor_state_equal") is True and matched.get("critic_state_equal") is True
        and matched.get("optimizer_state_empty") is True
        and matched.get("parameter_groups_match_except_lr") is True
        and matched.get("actor_normalizer_count") == matched.get("critic_normalizer_count") == 0
        and matched.get("tensors_in_report") is False and matched.get("ppo_resume") is False
        and matched.get("candidate_iteration") == matched.get("candidate_common_step_counter") == 0
        and matched.get("control_iteration") == matched.get("control_common_step_counter") == 0
        and matched.get("candidate_learning_rate") == profiles.CAUTIOUS_INITIAL_LR
        and matched.get("control_learning_rate") == profiles.STOCK_INITIAL_LR
        and matched.get("trajectory_identity_claim") is False
        and matched.get("policy_acceptance") is False
        and matched.get("physical_motion_authorized") is False,
        "steady identity adapted from authenticated retained stock initial checkpoint")
    validate_command_coverage(result, worlds, updates)
    base.require(base.finite_tree(result) and result["elapsed_s"] <= CAPS[mode],
                 "finite receipt and bounded duration")
    kl_report = validate_first_update_kl(output, mode, source)
    base.require(result.get("first_update_kl") == kl_report,
                 "training receipt binds its first-update KL guard")
    metric_rows = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    base.require(len(metric_rows) == updates and base.finite_tree(metric_rows)
        and all(row.get("yaw_bias_weight") == BIAS_WEIGHT and row.get("yaw_bias_tau_s") == TAU_S
            and row.get("weighted_yaw_bias_reward_mean", 0) <= 0
            and row.get("yaw_bias_cost_mean", -1) >= 0
            and math.isclose(row.get("weighted_yaw_bias_reward_mean", 1),
                             BIAS_WEIGHT * row.get("yaw_bias_cost_mean", 0), abs_tol=1e-6, rel_tol=1e-5)
            and row.get("yaw_bias_observed_steps") == 24
            and 0 <= row.get("yaw_bias_activity_steps", -1) <= 24
            and row.get("raw_yaw_residual_mae", -1) >= 0
            and set(row.get("yaw_bias_signed_buckets", {})) == set(exposure.EXPOSURE_BINS)
            and len(row.get("yaw_bias_signed_buckets", {})) == len(exposure.EXPOSURE_BINS)
            and sum(bucket["count"] for bucket in row["yaw_bias_signed_buckets"].values()) == worlds*24
            and all(type(bucket["count"]) is int and bucket["count"] >= 0
                    and (bucket["count"] == 0 or math.isclose(bucket["weighted_reward_mean"],
                        BIAS_WEIGHT * bucket["cost_mean"], abs_tol=1e-6, rel_tol=1e-5))
                    for bucket in row["yaw_bias_signed_buckets"].values())
            for row in metric_rows), "finite per-update nonpositive reward and complete signed buckets")
    base.require(any(row["yaw_bias_activity_steps"] > 0 and row["yaw_bias_cost_mean"] > 0
                     for row in metric_rows), "small run demonstrates nonzero yaw-bias activity")
    return result


def validate_first_update_kl(output, mode, source):
    report = json.loads((Path(output) / "first-update-kl.json").read_text())
    summary = dict(optimizer_minibatches=report.get("optimizer_minibatches"),
                   kl_mean_mean=report.get("mean_raw_kl"),
                   kl_mean_max=report.get("maximum_minibatch_mean_raw_kl"))
    expected = first_update_kl_report(mode, source, summary, steps=report.get("common_step_counter"))
    base.require(report == expected and report.get("mode") == mode
                 and report.get("source") == source and expected["passed"],
                 "exact finite steady-steering first-update KL guard")
    return report


def validate_baseline(rows):
    learning.validate_baseline(rows)


def read_raw_cases(output):
    return [json.loads((Path(output) / f"s{seed}-c{index}.json").read_text())
            for seed in base.SEEDS for index in range(len(base.CASES))]


def validate_evaluation_receipt(root, mode, source, checkpoint_sha256):
    base.require(mode in ("initial-eval", "final-eval"), "declared evaluation receipt stage")
    output = Path(root) / mode
    result = json.loads((output / "result.json").read_text())
    expected_files = [dict(file=f"s{seed}-c{index}.json",
        sha256=base.sha256(output / f"s{seed}-c{index}.json"))
        for seed in base.SEEDS for index in range(len(base.CASES))]
    summaries = [json.loads((output / f"s{seed}-c{index}-route.json").read_text())
        for seed in base.SEEDS for index in range(len(base.CASES))]
    raw = read_raw_cases(output)
    base.require(result["source"] == source and result["protocol"] == PROTOCOL
        and result["mode"] == mode and result["profile"] == "steady-steering"
        and result["checkpoint_sha256"] == checkpoint_sha256
        and result["raw_case_files"] == expected_files and result["cases"] == summaries
        and len(raw) == len(summaries) == 12 and "raw_case_records" not in result
        and "raw_decision_inputs" not in result and base.finite_tree(result)
        and isinstance(result.get("route_gate"), dict) and isinstance(result.get("legacy_gate"), dict),
        "source/protocol/checkpoint-bound compact evaluation receipt and raw case hashes")
    identities = [(row["seed"], row["speed"], row["yaw"]) for row in raw]
    base.require(identities == [(seed, speed, yaw) for seed in base.SEEDS
                                for speed, yaw in base.CASES],
                 "ordered complete raw fixed evaluation matrix")
    for row in raw:
        base.require(row["worlds"] == 8 and row["steps"] == 240 and row["complete"] is True
            and row["terminal_worlds"] == [] and row["checkpoint_sha256"] == checkpoint_sha256
            and len(row["speed_mae"]) == len(row["yaw_mae"]) == 8
            and row["response_diagnostics"]["startup_steps"] == 60
            and row["body_yaw_trace"]["startup_steps"] == 60
            and row["body_yaw_trace"]["steps"] == 240
            and row["body_yaw_trace"]["worlds"] == 8
            and row["body_yaw_trace"]["checkpoint_sha256"] == checkpoint_sha256
            and row["gait_path_trace"]["checkpoint_sha256"] == checkpoint_sha256
            and row["gait_yaw_trace"]["checkpoint_sha256"] == checkpoint_sha256,
            "fixed eight-world 240-step evaluation with startup-excluded traces")
    if mode == "initial-eval":
        from .v8_actor_checkpoint import FrozenV8Checkpoint
        FrozenV8Checkpoint(output.parent / "benchmark/initial.pt", checkpoint_sha256).require_initial_seed()
        validate_baseline(raw)
    return result


def load_retention_baseline():
    from . import steady_steering_retention as retention
    return retention.load_baseline()


def retention_gate(raw_rows, baseline=None):
    from . import steady_steering_retention as retention
    selected = load_retention_baseline() if baseline is None else baseline
    return retention.compare(raw_rows, selected)


def evaluate_matrix(checkpoint, output, health, *, initial=False):
    from .v8_actor_checkpoint import FrozenV8Checkpoint
    from . import gait_path_qualification as route

    typed = FrozenV8Checkpoint(checkpoint, base.sha256(checkpoint))
    if initial:
        typed.require_initial_seed()
    raw, summaries, raw_hashes = [], [], []
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            health()
            case = base.evaluate_case(checkpoint, seed, speed, yaw, health, diagnostics=True,
                yaw_trace=True, gait_trace=True, gait_path_trace=True, frozen_v8_checkpoint=typed)
            base.require(case["checkpoint_sha256"] == typed.sha256
                         and (case["seed"], case["speed"], case["yaw"]) == (seed, speed, yaw),
                         "fixed exact-byte v8 case identity")
            base.write_new(output / f"s{seed}-c{index}.json", case)
            raw_hashes.append(dict(file=f"s{seed}-c{index}.json",
                                   sha256=base.sha256(output / f"s{seed}-c{index}.json")))
            summary = route.summarize_case(case, case["gait_path_trace"])
            base.write_new(output / f"s{seed}-c{index}-route.json", summary)
            raw.append(case)
            summaries.append(summary)
    return dict(checkpoint_sha256=typed.sha256, cases=summaries, raw_case_files=raw_hashes,
        raw_decision_inputs=raw,
        route_gate=route.decision(summaries), legacy_gate=limits.decision(raw),
        inference_backend=typed.provenance(), policy_acceptance=False,
        physical_motion_authorized=False)


def assemble_receipt(source, mode, result):
    reserved = {"source", "protocol", "mode", "profile"}
    value = {key: item for key, item in result.items() if key not in reserved}
    value.update(source=source, protocol=PROTOCOL, mode=mode, profile="steady-steering")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=(*MODES, "initial-eval", "final-eval"))
    args = parser.parse_args()
    base.require(re.fullmatch(r"[a-f0-9]{40}", args.source) is not None
                 and args.run_id == f"{RUN_PREFIX}-{args.source[:8]}-20261013",
                 "exact source-bound steady-steering run ID")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()
    base.require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"),
                 "exact clean tested native source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated GPU selection")
    base.require(base.sha256(pinned.V8.path) == pinned.V8.sha256
                 and base.sha256(pinned.V8.path.with_name("manifest.json")) == pinned.V8.manifest_sha256,
                 "exact pinned v8 policy/manifest bytes")
    runtime = {p: metadata.version(p) for p in (*pinned.RUNTIME, "rsl-rl-lib", "onnx", "onnxruntime")}
    base.require(all(runtime[k].split("+")[0] == v for k, v in pinned.RUNTIME.items())
                 and runtime["rsl-rl-lib"] == "5.0.1", "unchanged installed runtime")
    base.require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing regular GPU lease")
    with base.LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = base.shared_host()
        root = base.ROOT / "artifacts/training" / args.run_id
        binding = dict(source=args.source, protocol=PROTOCOL, profile="steady-steering",
            runtime=runtime, artifact_sha256=pinned.V8.sha256,
            manifest_sha256=pinned.V8.manifest_sha256, modes=MODES, caps=CAPS,
            heldout_seeds=base.SEEDS, heldout_cases=base.CASES,
            recipe_sha256=recipe_match("smoke")["steady_recipe_sha256"])
        if args.mode == "smoke":
            root.mkdir(parents=True, exist_ok=False)
            base.write_new(root / "campaign.json", binding)
        else:
            retention_baseline = None
            retained = json.loads((root / "campaign.json").read_text())
            base.require(retained == json.loads(json.dumps(binding)), "exact retained campaign declaration")
            validate_training_receipt(root, "smoke", args.source)
            if args.mode in ("initial-eval", "foundation", "final-eval"):
                validate_training_receipt(root, "benchmark", args.source)
            if args.mode in ("foundation", "final-eval"):
                benchmark = validate_training_receipt(root, "benchmark", args.source)
                base.require(benchmark_projection_seconds(benchmark["elapsed_s"]) < CAPS["foundation"] - 30,
                             "conservative benchmark projection fits foundation cap")
                baseline = json.loads((root / "initial-eval/result.json").read_text())
                base.require(baseline["source"] == args.source
                             and baseline["checkpoint_sha256"] == benchmark["initial_checkpoint_sha256"],
                             "initial evaluation binds benchmark initial actor")
                validate_evaluation_receipt(root, "initial-eval", args.source,
                                            benchmark["initial_checkpoint_sha256"])
                validate_baseline(read_raw_cases(root / "initial-eval"))
                retention_baseline = load_retention_baseline()
            if args.mode == "final-eval":
                validate_training_receipt(root, "foundation", args.source)
        output = root / args.mode
        output.mkdir(exist_ok=False)
        launch = dict(source=args.source, protocol=PROTOCOL, mode=args.mode, host=host,
            cap_seconds=CAPS[args.mode], runtime=runtime,
            recipe=asdict(prepare_config(args.mode)[1]) if args.mode in MODES else None,
            policy_acceptance=False, physical_motion_authorized=False)
        base.write_new(output / "launch.json", launch)
        started = time.monotonic()
        def health():
            base.require(time.monotonic() - started < CAPS[args.mode] - 15, "bounded stage deadline")
            sample = base.shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False) + "\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            if args.mode in MODES:
                observer, imported = Observer(args.mode), {}
                observer.output, observer.source = output, args.source
                recipe = prepare_config(args.mode)
                proof = recipe_match(args.mode)
                base.write_new(output / "recipe-match.json", proof)
                def initialize(runner, env):
                    imported.update(smoke.initialize(runner, env, output))
                    observer.runner = runner
                    matched = profiles.matched_initialization(runner, env, args.mode, args.source)
                    matched.pop("stock_config_sha256", None)
                    matched.pop("cautious_config_sha256", None)
                    matched.update(protocol="v8-steady-steering-matched-initialization-v1",
                        profile="steady-steering", recipe_match=proof,
                        recipe_sha256=proof["steady_recipe_sha256"])
                    base.write_new(output / "matched-initialization.json", matched)
                    kl.install_optimizer_capture(runner, env, initial_lr=profiles.CAUTIOUS_INITIAL_LR)
                result = base.training(output, args.mode, health, recipe=recipe,
                    initialize=initialize, before_action=observer.before,
                    after_step=observer.step, after_update=observer.update)
                base.require(sum(observer.dwell.values()) == result["transitions"],
                             "complete actual command dwell")
                from .integrated_motor_refinement import onnx_check
                result.update(profile="steady-steering", actor_import=imported,
                    command_exposure=observer.dwell,
                    initial_checkpoint_sha256=base.sha256(output / "initial.pt"),
                    recipe_match=proof,
                    matched_initialization=json.loads((output / "matched-initialization.json").read_text()),
                    first_update_kl=observer.first_update_report,
                    onnx=onnx_check(output / f"{output.name}.onnx"),
                    policy_acceptance=False, physical_motion_authorized=False)
            else:
                mode = "benchmark" if args.mode == "initial-eval" else "foundation"
                checkpoint = root / ("benchmark/initial.pt" if args.mode == "initial-eval"
                                     else "foundation/model_999.pt")
                if args.mode == "final-eval":
                    validate_training_receipt(root, "foundation", args.source)
                result = evaluate_matrix(checkpoint, output, health, initial=args.mode == "initial-eval")
                result["elapsed_s"] = time.monotonic() - started
                raw_rows = result.pop("raw_decision_inputs")
                if args.mode == "initial-eval":
                    validate_baseline(raw_rows)
                if args.mode == "final-eval":
                    result["paired_descriptive_deltas"] = learning.paired_comparison(
                        read_raw_cases(root / "initial-eval"), raw_rows)
                    result["retention_gate"] = retention_gate(raw_rows, retention_baseline)
            health()
            base.write_new(output / "result.json", assemble_receipt(args.source, args.mode, result))
            if args.mode in ("initial-eval", "final-eval"):
                validate_evaluation_receipt(root, args.mode, args.source, base.sha256(checkpoint))
        except Exception as error:
            base.write_new(output / "failure.json", dict(source=args.source,
                type=type(error).__name__, error=str(error)))
            raise


if __name__ == "__main__":
    main()
