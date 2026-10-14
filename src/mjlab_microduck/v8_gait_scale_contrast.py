"""Matched optimizer-preserving steady yaw-bias continuation contrast.

The control keeps the authenticated parent yaw-bias time constant (1.0 s); the
treatment changes only it to 0.5 s. Both start from the same live steady PPO
checkpoint and retain fresh episode/runner clocks. This is a bounded research
protocol, not policy acceptance or motion authorization.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata as metadata
import io
import json
import math
import os
import re
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from . import integrated_turn_foundation as base
from . import v8_steady_steering as steady
from . import v8_actor_lr_profile as profiles
from . import integrated_kl_budget_refinement as kl
from . import integrated_training_budget_refinement as exposure
from . import v8_actor_learning as learning

PROTOCOL = "v8-gait-scale-steering-contrast-v1"
RUN_PREFIX = "v8-gait-scale-steering"
ARMS = ("control", "gait-scale")
MODES = {
    "smoke": (64, 5, 947),
    "benchmark": (256, 10, 953),
    "foundation": (256, 1000, 967),
}
CAPS = {"smoke": 300, "benchmark": 300, "foundation": 2100, "final-eval": 600}
PARENT_SHA = "d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f"
PARENT = base.ROOT / "artifacts/training/v8-steady-steering-7f55ded7-20261013/foundation/model_999.pt"
PARENT_ITERATION, PARENT_STEPS = 999, 24000
PARENT_ADAM_STEP, PARENT_CRITIC_COUNT = 20000, 6144000
TAUS = {"control": 1.0, "gait-scale": 0.5}
WEIGHT = -1.0
FIRST_KL_MEAN_MAX, FIRST_KL_MINIBATCH_MEAN_MAX = .05, .10
_ACTIVE_STAGE = None


def _require(ok: bool, message: str) -> None:
    base.require(ok, message)


def verify_source(args, *, training):
    _require(re.fullmatch(r"[a-f0-9]{40}", args.source) is not None
        and args.run_id == f"{RUN_PREFIX}-{args.source[:8]}-20261014",
        "exact source-bound 20261014 run identity")
    git = lambda *argv: subprocess.check_output(
        ("git", "-C", str(base.ROOT), *argv), text=True).strip()
    _require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"),
             "exact clean tested native source")
    if training:
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated GPU selection")
    from . import official_native_reference as pinned
    _require(base.sha256(pinned.V8.path) == pinned.V8.sha256
        and base.sha256(pinned.V8.path.with_name("manifest.json")) == pinned.V8.manifest_sha256,
        "exact pinned v8 policy and manifest bytes")
    runtime = {name: metadata.version(name) for name in
        (*pinned.RUNTIME, "rsl-rl-lib", "onnx", "onnxruntime")}
    _require(all(runtime[key].split("+")[0] == expected
        for key, expected in pinned.RUNTIME.items()) and runtime["rsl-rl-lib"] == "5.0.1",
        "unchanged native runtime pins")
    return runtime


def _control_metric(env):
    return capture_metric(env, "control")


def _treatment_metric(env):
    return capture_metric(env, "gait-scale")


def capture_metric(env, arm: str):
    from .tasks import mdp

    _require(arm in ARMS and getattr(env, "_microduck_yaw_bias_snapshot", None) is None,
             "known arm and previous yaw-bias snapshot consumed")
    tau = TAUS[arm]
    term = env.reward_manager.get_term_cfg(steady.TERM)
    _require(term.func is mdp.body_yaw_bias_cost and term.weight == WEIGHT
             and term.params == {"command_name": "twist", "tau_s": tau},
             "live contrast reward function, weight, and arm time constant")
    cost = term.func(env, **term.params)
    state = env._microduck_body_yaw_bias_state
    _require(state["step"] == env.common_step_counter, "same-step cached gait-scale reward state")
    index = env.reward_manager.active_terms.index(steady.TERM)
    weighted = env.reward_manager._step_reward[:, index]
    _require(isinstance(weighted, torch.Tensor) and weighted.shape == cost.shape
             and base.finite_tree(weighted) and bool((weighted <= 0).all())
             and torch.allclose(weighted, WEIGHT * cost, atol=1e-6, rtol=1e-5),
             "actual weighted reward matches the configured nonpositive cost")
    env._microduck_yaw_bias_snapshot = {
        key: state[key].detach().clone() for key in ("signed_residual", "signed_ema", "cost")}
    env._microduck_yaw_bias_snapshot["weighted_reward"] = weighted.detach().clone()
    _require(torch.equal(cost, env._microduck_yaw_bias_snapshot["cost"]),
             "same-step cached reward capture is idempotent")
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode: str, arm: str):
    _require(mode in MODES and arm in ARMS, "declared contrast mode and arm")
    cfg, agent = steady.prepare_config(mode)
    term = cfg.rewards[steady.TERM]
    term.params["tau_s"] = TAUS[arm]
    cfg.metrics["microduck_yaw_bias"].func = (
        _control_metric if arm == "control" else _treatment_metric)
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    freeze_live_profile(cfg)
    return cfg, agent


def freeze_live_profile(cfg):
    """Replace every known scheduled curriculum by the authenticated live value."""
    from .tasks import mdp

    expected = {"action_rate_weight", "standing_envs", "head_pose_range",
                "body_pose_range", "com_range", "head_com_range", "head_pose_bias_weight"}
    _require(set(cfg.curriculum) == expected, "exact supported parent curriculum set")
    values = {
        "action_rate_weight": (mdp.reward_weight, "weight_stages", {"step": 0, "weight": -.4}),
        "standing_envs": (mdp.standing_envs_curriculum, "standing_stages",
                          {"step": 0, "rel_standing_envs": .10}),
        "head_pose_range": (mdp.pose_command_range_curriculum, "range_stages", {"step": 0,
            "ranges": ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))}),
        "body_pose_range": (mdp.pose_command_range_curriculum, "range_stages", {"step": 0,
            "ranges": ((-.005, .005), (-.005, .005), (-.005, .005),
                       (-.05, .05), (-.05, .05), (-.05, .05))}),
        "com_range": (mdp.com_range_curriculum, "range_stages", {"step": 0, "range": .005}),
        "head_com_range": (mdp.com_range_curriculum, "range_stages", {"step": 0, "range": .005}),
        "head_pose_bias_weight": (mdp.reward_weight, "weight_stages", {"step": 0, "weight": 1.0}),
    }
    for name, (func, stage_key, stage) in values.items():
        term = cfg.curriculum[name]
        _require(term.func is func and isinstance(term.params, dict)
                 and stage_key in term.params, f"known {name} curriculum definition")
        term.params[stage_key] = [copy.deepcopy(stage)]
    _require(cfg.commands["twist"].rel_standing_envs == .02,
             "known initial standing command before frozen live snapshot")
    cfg.commands["twist"].rel_standing_envs = .10
    cfg.commands["head_pose"].ranges = values["head_pose_range"][2]["ranges"]
    cfg.commands["body_pose"].ranges = values["body_pose_range"][2]["ranges"]
    cfg.rewards["action_rate_l2"].weight = -.4
    cfg.rewards["head_pose_bias"].weight = 1.0
    for name in ("randomize_com", "randomize_head_com"):
        _require(name in cfg.events and cfg.events[name].params.get("ranges") == (-.003, .003),
                 f"known {name} reset distribution")
        cfg.events[name].params["ranges"] = (-.005, .005)
    return cfg


def _canonical(cfg, agent):
    env, algo = asdict(cfg), asdict(agent)
    metric = env["metrics"]["microduck_yaw_bias"]
    _require(metric["func"] in (_control_metric, _treatment_metric)
             and metric["per_substep"] is False,
             "known contrast observer callable before recipe canonicalization")
    metric["func"] = "mjlab_microduck.v8_gait_scale_contrast:arm_metric"
    # Remove only metric callable identity and the declared tau comparison.
    _require(algo.get("experiment_name") == PROTOCOL and algo.get("run_name") in MODES,
             "common contrast experiment/run identity before comparison")
    env["rewards"][steady.TERM]["params"]["tau_s"] = "<arm-tau>"
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def _full_canonical(cfg, agent):
    env, algo = asdict(cfg), asdict(agent)
    metric = env["metrics"]["microduck_yaw_bias"]
    _require(metric["func"] in (_control_metric, _treatment_metric),
             "known arm metric callable before full recipe hash")
    metric["func"] = "mjlab_microduck.v8_gait_scale_contrast:arm_metric"
    return json.dumps(profiles._canonical({"environment": env, "agent": algo}),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_match(mode):
    left_cfg, left_agent = prepare_config(mode, "control")
    right_cfg, right_agent = prepare_config(mode, "gait-scale")
    left, right = _canonical(left_cfg, left_agent), _canonical(right_cfg, right_agent)
    _require(left == right, "paired recipes differ only by yaw-bias tau and arm identity")
    full_left, full_right = _full_canonical(left_cfg, left_agent), _full_canonical(right_cfg, right_agent)
    return {"protocol": PROTOCOL + ":recipe-v1", "mode": mode,
            "control_tau_s": TAUS["control"], "treatment_tau_s": TAUS["gait-scale"],
            "control_full_recipe_sha256": hashlib.sha256(full_left.encode()).hexdigest(),
            "treatment_full_recipe_sha256": hashlib.sha256(full_right.encode()).hexdigest(),
            "control_recipe_sha256": hashlib.sha256(left.encode()).hexdigest(),
            "treatment_recipe_sha256": hashlib.sha256(right.encode()).hexdigest(),
            "comparison_recipe_sha256": hashlib.sha256(left.encode()).hexdigest(),
            "comparison_mask": ["reward yaw_bias.params.tau_s", "arm identity metric callable"],
            "full_recipe_hashes_distinct": hashlib.sha256(full_left.encode()).hexdigest()
                != hashlib.sha256(full_right.encode()).hexdigest(),
            "matched_after_declared_mask": True,
            "policy_acceptance": False, "physical_motion_authorized": False}


def frozen_profile_report(source, arm, mode):
    return {"protocol": PROTOCOL + ":frozen-profile-v1", "source": source,
        "arm": arm, "mode": mode, "parent_sha256": PARENT_SHA,
        "live_values": {"action_rate_weight": -.4, "head_pose_bias_weight": 1.0,
            "rel_standing_envs": .10,
            "head_pose_ranges": [[-.39,.39],[-.39,.39],[-.49,.49],[-.11,.11]],
            "body_pose_ranges": [[-.005,.005],[-.005,.005],[-.005,.005],
                [-.05,.05],[-.05,.05],[-.05,.05]],
            "randomize_com": {"ranges": [-.005,.005], "operation": "add"},
            "randomize_head_com": {"ranges": [-.005,.005], "operation": "add"}},
        "curriculum_stage_counts": {name: 1 for name in (
            "action_rate_weight", "standing_envs", "head_pose_range", "body_pose_range",
            "com_range", "head_com_range", "head_pose_bias_weight")},
        "curriculum_clock_start": 0,
        "curriculum_clock_policy": "single-step-zero-stage",
        "policy_acceptance": False, "physical_motion_authorized": False}


def _tree_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()+b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode()+b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()+b"\0"); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode()+b"\0")
    visit(value)
    return digest.hexdigest()


def _read_parent(path=PARENT, expected_sha=PARENT_SHA):
    from . import projected_heading_retention as retention
    # Authenticate both sealed baselines before authorizing any training stage.
    summary = retention.load_baseline()
    _require(summary["source"] == "7f55ded7c615e0adda4f859e6e05dde40c447e66"
             and summary["evaluations"]["final-eval"]["checkpoint_sha256"] == expected_sha,
             "sealed steady summary identifies exact parent")
    validate_parent_live_profile(summary)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("parent checkpoint could not be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and info.st_size <= 2**31, "owned bounded regular parent checkpoint")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(info.st_size + 1)
        _require(len(raw) == info.st_size, "parent checkpoint changed while being read")
    finally:
        os.close(descriptor)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha, "exact authenticated parent checkpoint bytes")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("parent must be a weights-only checkpoint") from exc
    validate_parent_payload(payload)
    from .v8_actor_checkpoint import FrozenV8Checkpoint
    typed = FrozenV8Checkpoint(path, expected_sha)
    actor_state = payload["actor_state_dict"]
    _require(set(actor_state) == set(typed._actor_state)
        and all(torch.equal(actor_state[name], typed._actor_state[name]) for name in actor_state),
        "parent actor matches the authenticated frozen v8 checkpoint contract")
    return payload


def validate_parent_live_profile(summary):
    live = summary.get("training", {}).get("foundation", {}).get("final_live_curriculum")
    expected = {"common_step_counter": PARENT_STEPS, "action_rate_weight": -.4,
        "head_pose_bias_weight": 1.0, "rel_standing_envs": .10,
        "head_pose_ranges": [[-.39,.39],[-.39,.39],[-.49,.49],[-.11,.11]],
        "randomize_com": {"ranges": [-.005,.005], "operation": "add"},
        "randomize_head_com": {"ranges": [-.005,.005], "operation": "add"}}
    _require(isinstance(live, Mapping) and all(live.get(key) == value
             for key, value in expected.items()),
             "sealed parent final_live_curriculum matches exact mixed manager snapshot")
    return live


def validate_parent_payload(payload):
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
    _require(isinstance(payload, Mapping) and set(payload) == required,
             "exact PPO checkpoint fields required for continuation")
    _require(type(payload["iter"]) is int and payload["iter"] == PARENT_ITERATION
             and isinstance(payload["infos"], Mapping)
             and isinstance(payload["infos"].get("env_state"), Mapping)
             and type(payload["infos"]["env_state"].get("common_step_counter")) is int
             and payload["infos"]["env_state"]["common_step_counter"] == PARENT_STEPS,
             "exact historical parent clocks")
    _require(base.finite_tree(payload["actor_state_dict"]) and base.finite_tree(payload["critic_state_dict"])
             and base.finite_tree(payload["optimizer_state_dict"]), "finite parent PPO tensors")
    optimizer = payload["optimizer_state_dict"]
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, Mapping) else []
    states = optimizer.get("state", {}) if isinstance(optimizer, Mapping) else {}
    param_ids = [param for group in groups for param in group.get("params", [])]
    _require(isinstance(optimizer, Mapping) and len(groups) == 1
             and len(param_ids) == len(set(param_ids)) == 17
             and set(param_ids) == set(states) and len(states) == 17
             and all(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                 and isinstance(state["step"], torch.Tensor) and state["step"].ndim == 0
                 and int(state["step"].item()) == PARENT_ADAM_STEP
                 and isinstance(state["exp_avg"], torch.Tensor)
                 and isinstance(state["exp_avg_sq"], torch.Tensor)
                 and state["exp_avg"].shape == state["exp_avg_sq"].shape
                 and base.finite_tree(state)
                 for state in states.values()),
             "exact inherited 17-entry Adam state at step 20000")
    critic_count = payload["critic_state_dict"].get("obs_normalizer.count")
    _require(isinstance(critic_count, torch.Tensor) and critic_count.item() == PARENT_CRITIC_COUNT,
             "exact inherited critic normalizer count")
    return payload


def initialize(runner, env, output):
    """Load authenticated model/critic/Adam tensors without importing runner clocks."""
    from .integrated_fresh_foundation import normalizer_count
    from .v8_actor_seed import FrozenV8Actor

    payload = _read_parent()
    alg = runner.alg
    _require(type(alg.actor) is FrozenV8Actor and type(alg.optimizer) is torch.optim.Adam
             and runner.current_learning_iteration == env.common_step_counter == 0,
             "fresh FrozenV8Actor runner and zero local clocks")
    _require(normalizer_count(alg.actor) == normalizer_count(alg.critic) == 0,
             "fresh target actor/critic normalizer clocks")
    actor_ids = tuple(id(p) for p in alg.actor.parameters())
    critic_ids = tuple(id(p) for p in alg.critic.parameters())
    optimizer_ownership = tuple(id(p) for g in alg.optimizer.param_groups for p in g["params"])
    actor_before = alg.actor.state_dict()
    critic_before = alg.critic.state_dict()
    _require(set(actor_before) == set(payload["actor_state_dict"])
             and set(critic_before) == set(payload["critic_state_dict"]),
             "parent actor/critic tensor layouts exactly match fresh runner")
    alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    alg.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
    rates = {group["lr"] for group in alg.optimizer.param_groups}
    _require(len(rates) == 1 and type(next(iter(rates))) in (float, int)
             and math.isfinite(next(iter(rates))) and next(iter(rates)) > 0,
             "one finite inherited optimizer learning rate")
    inherited_lr = next(iter(rates))
    alg.learning_rate = inherited_lr
    _require(tuple(id(p) for p in alg.actor.parameters()) == actor_ids
             and tuple(id(p) for p in alg.critic.parameters()) == critic_ids
             and tuple(id(p) for g in alg.optimizer.param_groups for p in g["params"])
                 == optimizer_ownership,
             "actor, critic, and optimizer parameter ownership remains local")
    restored = alg.save()
    _require(_tree_digest(restored["actor_state_dict"]) == _tree_digest(payload["actor_state_dict"])
             and _tree_digest(restored["critic_state_dict"]) == _tree_digest(payload["critic_state_dict"])
             and _tree_digest(restored["optimizer_state_dict"]) == _tree_digest(payload["optimizer_state_dict"]),
             "exactly restored parent actor, critic, normalizers, and Adam state")
    adam_steps = {int(state["step"].item()) for state in alg.optimizer.state.values()}
    critic_count = int(alg.critic.obs_normalizer.count.item())
    _require(len(alg.optimizer.state) == 17 and adam_steps == {PARENT_ADAM_STEP}
             and critic_count == PARENT_CRITIC_COUNT,
             "parent Adam and critic normalizer historical counts")
    _require(runner.current_learning_iteration == env.common_step_counter == 0,
             "local runner/environment clocks remain zero after tensor load")
    report = {"protocol": PROTOCOL + ":initialization-v1", "parent_sha256": PARENT_SHA,
        "historical_parent_iteration": PARENT_ITERATION,
        "historical_parent_common_step_counter": PARENT_STEPS,
        "local_iteration": runner.current_learning_iteration,
        "local_common_step_counter": env.common_step_counter,
        "parent_actor_sha256": _tree_digest(payload["actor_state_dict"]),
        "parent_critic_sha256": _tree_digest(payload["critic_state_dict"]),
        "parent_optimizer_sha256": _tree_digest(payload["optimizer_state_dict"]),
        "parent_optimizer_state_sha256": _tree_digest(payload["optimizer_state_dict"]["state"]),
        "parent_optimizer_groups_sha256": _tree_digest(payload["optimizer_state_dict"]["param_groups"]),
        "inherited_learning_rate": inherited_lr, "adam_state_entries": len(alg.optimizer.state),
        "adam_step_min": min(adam_steps), "adam_step_max": max(adam_steps),
        "critic_normalizer_count": critic_count, "actor_normalizer_count": normalizer_count(alg.actor),
        "tensor_ownership_preserved": True, "runner_load_used": False,
        "episode_state_imported": False, "rng_state_imported": False,
        "bitwise_resume": False, "trajectory_identity_claim": False,
        "policy_acceptance": False, "physical_motion_authorized": False}
    base.write_new(Path(output) / "initialization.json", report)
    return report


def _optimizer_steps(optimizer):
    steps = {int(v["step"].item()) for v in optimizer.state.values() if "step" in v}
    return sorted(steps)


class Observer(learning.Observer):
    """Read-only command, frozen-curriculum, and pre-reset reward observer."""
    def __init__(self, mode, arm):
        _require(mode in MODES and arm in ARMS, "known observer stage and arm")
        super().__init__()
        self.mode, self.arm = mode, arm
        self.profile = "gait-scale-contrast"
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
        self._assert_live_profile(env)
        live = env.reward_manager.get_term_cfg(steady.TERM)
        _require(live.func is mdp.body_yaw_bias_cost and live.weight == WEIGHT
                 and live.params == {"command_name": "twist", "tau_s": TAUS[self.arm]},
                 "frozen arm-specific reward remains live throughout training")
        _require(getattr(env, "_microduck_yaw_bias_snapshot", None) is None,
                 "prior pre-reset reward snapshot consumed")
        self._bias_expected_step = env.common_step_counter
        self._bias_command = env.command_manager.get_command("twist").detach().clone()

    def _assert_live_profile(self, env):
        from .tasks import mdp
        reward = env.reward_manager
        _require(reward.get_term_cfg("action_rate_l2").weight == -.4
                 and reward.get_term_cfg("head_pose_bias").weight == 1.0,
                 "parent live reward curriculum snapshot remains frozen")
        twist = env.command_manager.get_term("twist").cfg
        _require(twist.rel_standing_envs == .10, "parent standing command fraction remains frozen")
        for name, expected in (("head_pose", ((-.39, .39), (-.39, .39),
                                                (-.49, .49), (-.11, .11))),
                               ("body_pose", ((-.005, .005), (-.005, .005),
                                               (-.005, .005), (-.05, .05),
                                               (-.05, .05), (-.05, .05)))):
            command = env.command_manager.get_term(name).cfg
            _require(tuple(command.ranges) == expected,
                     f"parent live {name} command ranges remain frozen")
        for name, value in (("randomize_com", .005), ("randomize_head_com", .005)):
            term = env.event_manager.get_term_cfg(name)
            _require(term.params.get("ranges") == (-value, value),
                     f"parent live {name} distribution remains frozen")
        _require(env.common_step_counter >= 0, "valid local counter boundary")

    @staticmethod
    def _bucket_masks(command):
        _, _, wz = command.unbind(-1)
        idle = (command == 0).all(-1)
        masks = {"idle": idle, "straight": (wz == 0) & ~idle}
        for sign, label in ((wz < 0, "negative"), (wz > 0, "positive")):
            magnitude = wz.abs()
            masks[label + "-low"] = sign & (magnitude <= .15)
            masks[label + "-target"] = sign & (magnitude > .15) & (magnitude <= .25)
            masks[label + "-high"] = sign & (magnitude > .25)
        return masks

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snap = getattr(env, "_microduck_yaw_bias_snapshot", None)
        _require(isinstance(snap, dict) and set(snap) ==
                 {"signed_residual", "signed_ema", "cost", "weighted_reward"}
                 and all(isinstance(v, torch.Tensor) and v.shape == sample.mean_cost.shape
                         for v in snap.values()) and base.finite_tree(snap),
                 "finite consumable pre-reset gait-scale snapshot")
        _require(env.common_step_counter == self._bias_expected_step + 1,
                 "reward snapshot consumed at exactly the next local counter")
        raw, ema, cost, weighted = (snap[k] for k in
            ("signed_residual", "signed_ema", "cost", "weighted_reward"))
        _require(bool((cost >= 0).all()) and torch.equal(cost, ema.abs())
                 and bool((weighted <= 0).all())
                 and torch.allclose(weighted, WEIGHT * cost, atol=1e-6, rtol=1e-5),
                 "actual pre-reset reward matches cost and fixed weight")
        command = self._bias_command
        self.activity_steps += int(bool((cost > 0).any()))
        self.bias_costs.append(float(cost.mean())); self.bias_rewards.append(float(weighted.mean()))
        self.signed_biases.append(float(ema.mean())); self.raw_errors.append(float(raw.abs().mean()))
        self.signed_raw_errors.append(float(raw.mean()))
        for name, mask in self._bucket_masks(command).items():
            self.bucket_counts[name] += int(mask.sum())
            if bool(mask.any()):
                for key, tensor in (("raw_signed_residual", raw), ("signed_ema", ema),
                                    ("cost", cost), ("weighted_reward", weighted)):
                    self.bucket_values[name][key].append(float(tensor[mask].sum()))
        _require(sum(self.bucket_counts.values()) == len(self.bias_costs) * env.num_envs,
                 "per-update command bucket accounting boundary")
        env._microduck_yaw_bias_snapshot = None
        self._bias_expected_step = self._bias_command = None

    def update(self, env):
        row = super().update(env)
        profile = row.get("target_curriculum", {})
        _require(profile.get("action_rate_weight") == -.4
                 and profile.get("head_pose_bias_weight") == 1.0
                 and profile.get("rel_standing_envs") == .10
                 and profile.get("head_pose_ranges") == [[-.39,.39],[-.39,.39],[-.49,.49],[-.11,.11]]
                 and profile.get("randomize_com", {}).get("ranges") == [-.005,.005]
                 and profile.get("randomize_head_com", {}).get("ranges") == [-.005,.005],
                 "manager-applied parent live curriculum snapshot remains exact")
        _require(len(self.bias_costs) == 24 and sum(self.bucket_counts.values()) ==
                 24 * env.num_envs, "complete 24-step reward and bucket observation")
        if self.updates == 1:
            report = steady.first_update_kl_report(
                self.mode, self.source, row["optimizer"], steps=env.common_step_counter)
            report["protocol"] = PROTOCOL + ":first-update-kl-v1"
            report["inherited_parent_lr"] = getattr(self, "initial_lr", None)
            report["initial_learning_rate"] = self.initial_lr
            report["arm"] = self.arm
            report["profile"] = "gait-scale-contrast"
            self.first_update_report = report
            base.write_new(self.output / "first-update-kl.json", report)
            row["first_update_kl"] = report
            if not report["passed"]:
                _require(self.runner is not None, "runner attached for durable KL refusal")
                self.runner.save(str(self.output / "aborted-first-update.pt"), infos={
                    "env_state": {"common_step_counter": env.common_step_counter},
                    "profile": "gait-scale-contrast", "arm": self.arm, "mode": self.mode,
                    "full_budget_completed": False, "policy_acceptance": False,
                    "physical_motion_authorized": False,
                    "first_update_kl_sha256": base.sha256(self.output / "first-update-kl.json")})
                _require(False, "gait-scale-contrast first-update KL guard")
        row.update(arm=self.arm, yaw_bias_weight=WEIGHT, yaw_bias_tau_s=TAUS[self.arm],
            yaw_bias_cost_mean=sum(self.bias_costs)/24,
            weighted_yaw_bias_reward_mean=sum(self.bias_rewards)/24,
            signed_yaw_bias_mean=sum(self.signed_biases)/24,
            raw_yaw_residual_mae=sum(self.raw_errors)/24,
            signed_raw_yaw_residual_mean=sum(self.signed_raw_errors)/24,
            yaw_bias_observed_steps=24, yaw_bias_activity_steps=self.activity_steps,
            optimizer_adam_steps=_optimizer_steps(self.runner.alg.optimizer),
            local_common_step_counter=env.common_step_counter,
            yaw_bias_signed_buckets={key: {"count": self.bucket_counts[key],
                **{f"{field}_mean": (sum(values)/self.bucket_counts[key]
                    if self.bucket_counts[key] else None)
                   for field, values in self.bucket_values[key].items()}}
                for key in exposure.EXPOSURE_BINS})
        for values in (self.bias_costs, self.bias_rewards, self.signed_biases,
                       self.raw_errors, self.signed_raw_errors): values.clear()
        self.bucket_counts = dict.fromkeys(exposure.EXPOSURE_BINS, 0)
        self.bucket_values = {key: {name: [] for name in
            ("raw_signed_residual", "signed_ema", "cost", "weighted_reward")}
            for key in exposure.EXPOSURE_BINS}
        self.activity_steps = 0
        return row


def validate_training_receipt(root, arm, mode, source):
    _require(arm in ARMS and mode in MODES, "declared contrast receipt")
    output = Path(root) / arm / mode
    result = json.loads((output / "result.json").read_text())
    worlds, updates, seed = MODES[mode]
    checkpoint = output / f"model_{updates-1}.pt"
    init = json.loads((output / "initialization.json").read_text())
    recipe = recipe_match(mode)
    launch = json.loads((output / "launch.json").read_text())
    frozen = json.loads((output / "frozen-profile.json").read_text())
    from . import official_native_reference as pinned
    runtime = launch.get("runtime", {})
    runtime_names = (*pinned.RUNTIME, "rsl-rl-lib", "onnx", "onnxruntime")
    _require(result["protocol"] == PROTOCOL and result["source"] == source
        and result["arm"] == arm and result["mode"] == mode
        and result["status"] == "training-complete-not-accepted"
        and (result["worlds"], result["updates"], result["seed"], result["steps"],
             result["transitions"]) == (worlds, updates, seed, updates*24, worlds*updates*24)
        and result["checkpoint_sha256"] == base.sha256(checkpoint)
        and result["initial_checkpoint_sha256"] == base.sha256(output / "initial.pt"),
        "source-bound exact complete contrast budget and checkpoint bytes")
    _require(launch["protocol"] == PROTOCOL and launch["source"] == source
        and launch["arm"] == arm and launch["mode"] == mode
        and launch["parent_sha256"] == PARENT_SHA and launch["recipe_match"] == recipe
        and launch["frozen_profile_sha256"] == base.sha256(output / "frozen-profile.json")
        and all(runtime.get(key, "").split("+")[0] == version
                for key, version in pinned.RUNTIME.items())
        and runtime.get("rsl-rl-lib") == "5.0.1"
        and all(runtime.get(name) == metadata.version(name) for name in runtime_names)
        and frozen == frozen_profile_report(source, arm, mode)
        and result.get("frozen_profile") == frozen,
        "source/runtime/parent/frozen-curriculum/recipe-bound training receipt")
    parent = _read_parent()
    expected_init = {"protocol": PROTOCOL + ":initialization-v1", "parent_sha256": PARENT_SHA,
        "historical_parent_iteration": PARENT_ITERATION,
        "historical_parent_common_step_counter": PARENT_STEPS,
        "local_iteration": 0, "local_common_step_counter": 0,
        "parent_actor_sha256": _tree_digest(parent["actor_state_dict"]),
        "parent_critic_sha256": _tree_digest(parent["critic_state_dict"]),
        "parent_optimizer_sha256": _tree_digest(parent["optimizer_state_dict"]),
        "parent_optimizer_state_sha256": _tree_digest(parent["optimizer_state_dict"]["state"]),
        "parent_optimizer_groups_sha256": _tree_digest(parent["optimizer_state_dict"]["param_groups"]),
        "inherited_learning_rate": float(parent["optimizer_state_dict"]["param_groups"][0]["lr"]),
        "adam_state_entries": 17, "adam_step_min": PARENT_ADAM_STEP,
        "adam_step_max": PARENT_ADAM_STEP, "critic_normalizer_count": PARENT_CRITIC_COUNT,
        "actor_normalizer_count": 0, "tensor_ownership_preserved": True,
        "runner_load_used": False, "episode_state_imported": False,
        "rng_state_imported": False, "bitwise_resume": False,
        "trajectory_identity_claim": False, "policy_acceptance": False,
        "physical_motion_authorized": False}
    _require(init == expected_init and result.get("initialization") == init,
        "full initialization report exactly binds all parent tensors/groups and local clocks")
    _require(init["parent_sha256"] == PARENT_SHA and init["local_iteration"] == 0
        and init["local_common_step_counter"] == 0 and init["runner_load_used"] is False
        and init["bitwise_resume"] is False and init["tensor_ownership_preserved"] is True
        and init["parent_optimizer_sha256"] == _tree_digest(parent["optimizer_state_dict"]),
        "authenticated optimizer-preserving local-zero initialization")
    _require(result["recipe_match"] == recipe and recipe["matched_after_declared_mask"]
        and recipe["full_recipe_hashes_distinct"]
        and recipe["control_recipe_sha256"] == recipe["treatment_recipe_sha256"],
        "paired recipe hash proof")
    onnx = result.get("onnx", {})
    _require(onnx.get("input") == [1, 61] and onnx.get("output") == [1, 14]
        and onnx.get("finite") is True
        and onnx.get("sha256") == base.sha256(output / f"{mode}.onnx"),
        "complete finite CPU ONNX export receipt")
    rows = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    expected_adam = PARENT_ADAM_STEP + 20 * updates
    expected_critic = PARENT_CRITIC_COUNT + worlds * 24 * updates
    _require(len(rows) == updates and base.finite_tree(rows)
        and all(row["arm"] == arm and row["yaw_bias_tau_s"] == TAUS[arm]
            and row["yaw_bias_weight"] == WEIGHT
            and row["weighted_yaw_bias_reward_mean"] <= 0
            and math.isclose(row["weighted_yaw_bias_reward_mean"],
                WEIGHT * row["yaw_bias_cost_mean"], abs_tol=1e-6, rel_tol=1e-5)
            and row["yaw_bias_observed_steps"] == 24
            and 0 <= row["yaw_bias_activity_steps"] <= 24
            and row["optimizer_adam_steps"] == [PARENT_ADAM_STEP + 20 * (index+1)]
            and row["local_common_step_counter"] == (index+1)*24
            and set(row["yaw_bias_signed_buckets"]) == set(exposure.EXPOSURE_BINS)
            and sum(bucket["count"] for bucket in row["yaw_bias_signed_buckets"].values()) == worlds*24
            and all(type(bucket["count"]) is int and bucket["count"] >= 0
                and all((bucket[field+"_mean"] is None) if bucket["count"] == 0 else
                    type(bucket[field+"_mean"]) in (int,float)
                    and math.isfinite(bucket[field+"_mean"])
                    for field in ("raw_signed_residual", "signed_ema", "cost", "weighted_reward"))
                for bucket in row["yaw_bias_signed_buckets"].values())
            for index, row in enumerate(rows)),
        "finite optimizer continuation and actual pre-reset reward activity")
    _require(all(result.get(key, 0) > 0 for key in base.FULL_COMMAND_COVERAGE)
        and set(result.get("command_exposure", {})) == set(exposure.EXPOSURE_BINS)
        and all(type(count) is int and count >= 0 for count in result["command_exposure"].values())
        and sum(result["command_exposure"].values()) == worlds*24*updates,
        "complete four-bin source coverage and eight-bin signed command exposure")
    _require(any(row["yaw_bias_activity_steps"] > 0 and row["yaw_bias_cost_mean"] > 0
                 for row in rows), "nonzero actual reward activity in admitted short run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    final_optimizer = payload["optimizer_state_dict"]
    final_groups = final_optimizer.get("param_groups", [])
    final_states = final_optimizer.get("state", {})
    _require(base.finite_tree(payload) and len(final_groups) == 1
        and len(final_states) == 17
        and type(final_groups[0].get("lr")) in (float,int)
        and math.isfinite(final_groups[0]["lr"])
        and all(set(state) == {"step", "exp_avg", "exp_avg_sq"}
            and state["exp_avg"].shape == state["exp_avg_sq"].shape
            and base.finite_tree(state) for state in final_states.values())
        and payload["iter"] == updates-1
        and payload["infos"]["env_state"]["common_step_counter"] == updates*24
        and int(payload["critic_state_dict"]["obs_normalizer.count"].item()) == expected_critic
        and int(payload["actor_state_dict"]["obs_normalizer.count"].item()) == 0
        and {int(s["step"].item()) for s in final_states.values()}
            == {expected_adam}, "final checkpoint has expected local/Adam/critic clocks")
    for key in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var", "obs_normalizer.count"):
        _require(torch.equal(payload["actor_state_dict"][key], parent["actor_state_dict"][key]),
                 "frozen actor affine remains byte-identical after optimizer updates")
    initial = torch.load(output / "initial.pt", map_location="cpu", weights_only=True)
    parent = _read_parent()
    _require(initial["iter"] == 0
        and initial["infos"]["env_state"]["common_step_counter"] == 0
        and _tree_digest(initial["actor_state_dict"]) == _tree_digest(parent["actor_state_dict"])
        and _tree_digest(initial["critic_state_dict"]) == _tree_digest(parent["critic_state_dict"])
        and _tree_digest(initial["optimizer_state_dict"]) == _tree_digest(parent["optimizer_state_dict"]),
        "durable local-zero initial checkpoint exactly binds all inherited PPO tensors")
    kl_report = json.loads((output / "first-update-kl.json").read_text())
    expected_kl = steady.first_update_kl_report(mode, source, rows[0]["optimizer"], steps=24)
    expected_kl.update(protocol=PROTOCOL + ":first-update-kl-v1",
        inherited_parent_lr=expected_init["inherited_learning_rate"],
        initial_learning_rate=expected_init["inherited_learning_rate"], arm=arm,
        profile="gait-scale-contrast")
    _require(kl_report == expected_kl and result.get("first_update_kl") == kl_report
        and kl_report["passed"] is True and kl_report["source"] == source
        and kl_report["mode"] == mode and kl_report["arm"] == arm
        and kl_report["profile"] == "gait-scale-contrast"
        and kl_report["initial_learning_rate"] == expected_init["inherited_learning_rate"]
        and kl_report["common_step_counter"] == 24
        and kl_report["mean_raw_kl"] <= FIRST_KL_MEAN_MAX
        and kl_report["maximum_minibatch_mean_raw_kl"] <= FIRST_KL_MINIBATCH_MEAN_MAX,
        "passed first-update KL guard")
    _require(result["elapsed_s"] <= CAPS[mode], "bounded contrast stage duration")
    return result


def projection_seconds(elapsed_s):
    _require(type(elapsed_s) in (int, float) and math.isfinite(elapsed_s) and elapsed_s > 0,
             "finite positive benchmark timing")
    return elapsed_s * 100 * 1.5 + 60


def validate_evaluation_receipt(root, arm, source, checkpoint_sha):
    _require(arm in ARMS, "declared final-evaluation arm")
    output = Path(root) / arm / "final-eval"
    receipt = json.loads((output / "result.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    from . import official_native_reference as pinned
    runtime_names = (*pinned.RUNTIME, "rsl-rl-lib", "onnx", "onnxruntime")
    expected = [f"s{seed}-c{index}.json" for seed in base.SEEDS
        for index in range(len(base.CASES))]
    evaluation = receipt.get("evaluation", {})
    raw_files, route_files = evaluation.get("raw_case_files"), evaluation.get("route_case_files")
    _require(receipt.get("protocol") == PROTOCOL and receipt.get("source") == source
        and receipt.get("arm") == arm and receipt.get("mode") == "final-eval"
        and launch.get("protocol") == PROTOCOL and launch.get("source") == source
        and launch.get("arm") == arm and launch.get("mode") == "final-eval"
        and launch.get("foundation_iteration") == 999
        and launch.get("cap_seconds") == CAPS["final-eval"]
        and isinstance(launch.get("host"), dict)
        and all(launch.get("runtime", {}).get(key, "").split("+")[0] == version
                for key, version in pinned.RUNTIME.items())
        and launch.get("runtime", {}).get("rsl-rl-lib") == "5.0.1"
        and all(launch.get("runtime", {}).get(name) == metadata.version(name)
                for name in runtime_names)
        and evaluation.get("checkpoint_sha256") == checkpoint_sha
        and [item.get("file") for item in raw_files] == expected
        and len(route_files) == 12
        and [item.get("file") for item in route_files] == [name[:-5]+"-route.json" for name in expected],
        "source/arm/checkpoint-bound complete ordered 12-case evaluation receipt")
    health = [json.loads(line) for line in (output / "health.jsonl").read_text().splitlines()]
    _require(len(health) == 36 and all(base.finite_tree(row) and "utc" in row for row in health),
             "three finite shared-host health samples per held-out case")
    for item in raw_files + route_files:
        _require(base.sha256(output / item["file"]) == item["sha256"],
                 "evaluation receipt exact raw and route file hash")
    route_rows = [json.loads((output / item["file"]).read_text()) for item in route_files]
    _require(route_rows == evaluation.get("cases"),
             "compact route summaries exactly match their separately hashed files")
    raw = steady.read_raw_cases(output)
    _require([(row["seed"], row["speed"], row["yaw"]) for row in raw]
        == [(seed, speed, yaw) for seed in base.SEEDS for speed, yaw in base.CASES]
        and all(row.get("checkpoint_sha256") == checkpoint_sha and row.get("complete") is True
                and row.get("worlds") == 8 and row.get("steps") == 240
                and row.get("terminal_worlds") == []
                and len(row.get("speed_mae", [])) == len(row.get("yaw_mae", [])) == 8
                and row.get("response_diagnostics", {}).get("startup_steps") == 60
                and row.get("body_yaw_trace", {}).get("startup_steps") == 60
                and row.get("body_yaw_trace", {}).get("steps") == 240
                and row.get("body_yaw_trace", {}).get("worlds") == 8
                and row.get("body_yaw_trace", {}).get("checkpoint_sha256") == checkpoint_sha
                and row.get("gait_yaw_trace", {}).get("checkpoint_sha256") == checkpoint_sha
                and row.get("gait_path_trace", {}).get("checkpoint_sha256") == checkpoint_sha
                for row in raw), "fixed held-out seed/matrix, checkpoint, and startup trace contracts")
    from . import gait_path_qualification as route
    from . import integrated_joint_limit_refinement as limits
    _require(evaluation.get("legacy_gate") == limits.decision(raw)
        and evaluation.get("route_gate") == route.decision(route_rows),
        "compact evaluation gates recompute exactly from authenticated raw/routes")
    from . import projected_heading_retention as retention
    from . import steady_steering_retention as cautious_retention
    steady_reference = retention.load_baseline()
    cautious_reference = cautious_retention.load_baseline()
    _require(evaluation.get("projected_heading_retention") == retention.compare(raw, steady_reference)
        and evaluation.get("cautious_retention") == cautious_retention.compare(raw, cautious_reference),
        "both sealed prospective retention reports recompute from authenticated raw cases")
    _require(receipt.get("steady_reference_summary_sha256") == retention.BASELINE_SUMMARY_SHA
        and receipt.get("cautious_reference_summary_sha256") == cautious_retention.BASELINE_SUMMARY_SHA
        and receipt.get("cautious_reference_loaded") is True,
        "both sealed baseline summary identities are bound")
    _require(receipt.get("policy_acceptance") is False
        and receipt.get("physical_motion_authorized") is False
        and evaluation.get("policy_acceptance") is False
        and evaluation.get("physical_motion_authorized") is False
        and type(receipt.get("elapsed_s")) in (float, int)
        and math.isfinite(receipt["elapsed_s"]) and receipt["elapsed_s"] <= CAPS["final-eval"],
        "bounded non-acceptance evaluation receipt")
    if arm == "gait-scale":
        control_sha = base.sha256(Path(root) / "control/foundation/model_999.pt")
        _, control_rows = validate_evaluation_receipt(root, "control", source, control_sha)
        _require(receipt.get("paired_descriptive_deltas") ==
                 learning.paired_comparison(control_rows, raw),
                 "paired descriptive deltas recompute against exact control raw cases")
    else:
        _require("paired_descriptive_deltas" not in receipt,
                 "control receipt does not claim an incomplete pair")
    return receipt, raw


def _main_impl():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--mode", choices=(*MODES, "final-eval"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    _require(args.preflight or (args.arm is not None and args.mode is not None),
             "training requires one declared arm and mode")
    runtime = verify_source(args, training=not args.preflight)
    if args.preflight:
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
            and not torch.cuda.is_initialized(),
            "CPU preflight requires hidden CUDA and an uninitialized CUDA runtime")
    # CPU-only admission authenticates the parent, both sealed references, and
    # both complete recipes without acquiring the GPU lease or creating output.
    payload = _read_parent()
    recipes = {mode: recipe_match(mode) for mode in MODES}
    if args.preflight:
        report = {"protocol": PROTOCOL, "source": args.source, "parent_sha256": PARENT_SHA,
                "parent_actor_sha256": _tree_digest(payload["actor_state_dict"]),
                "parent_critic_sha256": _tree_digest(payload["critic_state_dict"]),
                "parent_optimizer_sha256": _tree_digest(payload["optimizer_state_dict"]),
                "recipes": recipes, "runtime": runtime, "policy_acceptance": False,
                "physical_motion_authorized": False}
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return report

    _require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing regular GPU lease")
    lease = base.LOCK.open("r+")
    lease_info = os.fstat(lease.fileno())
    _require(stat.S_ISREG(lease_info.st_mode) and lease_info.st_uid == os.getuid(),
             "owned regular GPU lease")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    host = base.shared_host()

    if args.mode == "final-eval":
        _require(args.arm in ARMS, "final evaluation requires one declared arm")
    else:
        _require(args.arm in ARMS, "training arm required")
    output_root = base.ROOT / "artifacts/training" / args.run_id
    if args.mode == "foundation":
        # Both arms' smoke and benchmark receipts are required before either
        # foundation run is admitted.
        for arm in ARMS:
            validate_training_receipt(output_root, arm, "smoke", args.source)
            benchmark = validate_training_receipt(output_root, arm, "benchmark", args.source)
            _require(projection_seconds(benchmark["elapsed_s"]) < CAPS["foundation"] - 30,
                     "both paired benchmarks project inside the foundation cap")

    if args.mode == "final-eval":
        for arm in ARMS:
            validate_training_receipt(output_root, arm, "foundation", args.source)
        arm = args.arm
        mode_dir = output_root / arm / "final-eval"
        mode_dir.mkdir(parents=True, exist_ok=False)
        global _ACTIVE_STAGE
        _ACTIVE_STAGE = (mode_dir, {"source": args.source, "arm": arm, "mode": "final-eval"})
        base.write_new(mode_dir / "launch.json", {"protocol": PROTOCOL, "source": args.source,
            "mode": "final-eval", "arm": arm, "host": host, "runtime": runtime,
            "cap_seconds": CAPS["final-eval"], "foundation_iteration": 999,
            "policy_acceptance": False, "physical_motion_authorized": False})
        started = time.monotonic()
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        checkpoint = output_root / arm / "foundation" / "model_999.pt"
        def eval_health():
            _require(time.monotonic()-started < CAPS["final-eval"]-15,
                     "bounded final held-out evaluation")
            sample = base.shared_host()
            with (mode_dir / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        eval_result = steady.evaluate_matrix(checkpoint, mode_dir,
            eval_health)
        # evaluate_matrix owns writing both raw rows and their route summaries;
        # its in-memory raw rows are needed only for retention comparisons.
        raw = eval_result.pop("raw_decision_inputs")
        _require(eval_result.get("checkpoint_sha256") == base.sha256(checkpoint)
            and len(raw) == 12 and eval_result.get("policy_acceptance") is False
            and eval_result.get("physical_motion_authorized") is False,
            "exact-checkpoint, non-authorizing 12-case final evaluation result")
        summaries = eval_result["cases"]
        _require(len(summaries) == 12 and len(eval_result["raw_case_files"]) == 12,
                 "steady evaluator returned complete raw and route receipts")
        report = {"checkpoint_sha256": base.sha256(checkpoint),
            "raw_case_files": eval_result["raw_case_files"],
            "route_case_files": [{"file": f"s{seed}-c{index}-route.json",
                "sha256": base.sha256(mode_dir / f"s{seed}-c{index}-route.json")}
                for seed in base.SEEDS for index in range(len(base.CASES))],
            "cases": summaries, "legacy_gate": eval_result["legacy_gate"],
            "route_gate": eval_result["route_gate"],
            "inference_backend": eval_result["inference_backend"],
            "policy_acceptance": False, "physical_motion_authorized": False}
        from . import projected_heading_retention as retention
        cautious = retention.load_baseline()
        from . import steady_steering_retention as cautious_retention
        cautious_summary = cautious_retention.load_baseline()
        report["projected_heading_retention"] = retention.compare(raw, cautious)
        report["cautious_retention"] = cautious_retention.compare(raw, cautious_summary)
        result = {"protocol": PROTOCOL, "source": args.source, "mode": "final-eval",
            "arm": arm, "evaluation": report,
            "steady_reference_summary_sha256": retention.BASELINE_SUMMARY_SHA,
            "cautious_reference_loaded": bool(cautious_summary),
            "cautious_reference_summary_sha256": cautious_retention.BASELINE_SUMMARY_SHA,
            "elapsed_s": time.monotonic()-started,
            "policy_acceptance": False, "physical_motion_authorized": False}
        if arm == "gait-scale":
            control_sha = base.sha256(output_root / "control/foundation/model_999.pt")
            _, control_rows = validate_evaluation_receipt(output_root, "control", args.source, control_sha)
            result["paired_descriptive_deltas"] = learning.paired_comparison(control_rows, raw)
        base.write_new(mode_dir / "result.json", result)
        validate_evaluation_receipt(output_root, arm, args.source, base.sha256(checkpoint))
        return result

    arm, mode = args.arm, args.mode
    stage_root = output_root / arm
    if mode == "smoke":
        stage_root.mkdir(parents=True, exist_ok=False)
    else:
        required_prior = ("smoke",) if mode == "benchmark" else ("smoke", "benchmark")
        for prior in required_prior:
            validate_training_receipt(output_root, arm, prior, args.source)
    output = stage_root / mode
    output.mkdir(parents=True, exist_ok=False)
    _ACTIVE_STAGE = (output, {"source": args.source, "arm": arm, "mode": mode})
    cfg, agent = prepare_config(mode, arm)
    observer = Observer(mode, arm)
    observer.output, observer.source, observer.runner = output, args.source, None
    recipe = recipes[mode]
    frozen = frozen_profile_report(args.source, arm, mode)
    base.write_new(output / "frozen-profile.json", frozen)
    base.write_new(output / "recipe-match.json", recipe)
    base.write_new(output / "launch.json", {"protocol": PROTOCOL, "source": args.source,
        "mode": mode, "arm": arm, "host": host, "runtime": runtime,
        "cap_seconds": CAPS[mode], "worlds": MODES[mode][0],
        "updates": MODES[mode][1], "seed": MODES[mode][2],
        "parent_sha256": PARENT_SHA, "recipe_match": recipe,
        "frozen_profile_sha256": base.sha256(output / "frozen-profile.json"),
        "policy_acceptance": False, "physical_motion_authorized": False})
    parent_lr = float(next(iter({group["lr"] for group in payload["optimizer_state_dict"]["param_groups"]})))
    observer.initial_lr = parent_lr
    started = time.monotonic()
    def health():
        _require(time.monotonic()-started < CAPS[mode]-15, "bounded contrast stage deadline")
        sample = base.shared_host()
        with (output / "health.jsonl").open("a") as log:
            log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False)+"\n")
            log.flush(); os.fsync(log.fileno())
    try:
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        def init(runner, env):
            observer.runner = runner
            result = initialize(runner, env, output)
            observer.initial_lr = result["inherited_learning_rate"]
            kl.install_optimizer_capture(runner, env,
                initial_lr=result["inherited_learning_rate"])
            return result
        result = base.training(output, mode, health, recipe=(cfg, agent), initialize=init,
            before_action=observer.before, after_step=observer.step, after_update=observer.update)
        _require(sum(observer.dwell.values()) == result["transitions"],
                 "complete actual command dwell coverage")
        from .integrated_motor_refinement import onnx_check
        result.update(protocol=PROTOCOL, source=args.source, arm=arm, mode=mode,
            initial_checkpoint_sha256=base.sha256(output / "initial.pt"),
            initialization=json.loads((output / "initialization.json").read_text()),
            recipe_match=recipe, command_exposure=dict(observer.dwell),
            frozen_profile=frozen,
            first_update_kl=observer.first_update_report,
            onnx=onnx_check(output / f"{mode}.onnx"),
            policy_acceptance=False, physical_motion_authorized=False)
        base.write_new(output / "result.json", result)
        validate_training_receipt(output_root, arm, mode, args.source)
        return result
    except Exception as exc:
        base.write_new(output / "failure.json", dict(source=args.source, arm=arm, mode=mode,
            type=type(exc).__name__, error=str(exc), policy_acceptance=False,
            physical_motion_authorized=False))
        raise


def main():
    global _ACTIVE_STAGE
    _ACTIVE_STAGE = None
    try:
        return _main_impl()
    except Exception as exc:
        if _ACTIVE_STAGE is not None:
            output, identity = _ACTIVE_STAGE
            failure = output / "failure.json"
            if not failure.exists():
                base.write_new(failure, {**identity, "type": type(exc).__name__,
                    "error": str(exc), "policy_acceptance": False,
                    "physical_motion_authorized": False})
        raise


if __name__ == "__main__":
    main()


