"""Matched motor-cost continuation of the unaccepted integrated foundation."""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host

ROOT = base.ROOT
PROTOCOL = "integrated-motor-continuation-v1"
PARENT = ROOT / "artifacts/training/turn-foundation-791f27c4-20261010/foundation/model_1499.pt"
PARENT_SHA = "b4c51e072b4b21c865ab4108c477ed4c27f26a129577e7ad17a14b1e6250dd90"
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 859), ("benchmark", 256, 10, 861))
         for arm in ("control", "motor")}
MODES.update(control=(256, 1000, 863), motor=(256, 1000, 863))
CAP_SECONDS = 2100
MOTOR_STAGES = ((0, 0.), (100, -.25), (300, -.50), (600, -1.))


def freeze_source(cfg):
    """Materialize the last consolidated source stage, not its unused endpoint."""
    expected = {"action_rate_weight", "standing_envs", "head_pose_range", "body_pose_range",
                "com_range", "head_com_range", "head_pose_bias_weight"}
    base.require(set(cfg.curriculum) == expected, "known source curricula only")
    for term in cfg.curriculum.values():
        params = term.params
        key = next(k for k in params if k.endswith("_stages"))
        stage = [s for s in params[key] if s["step"] < 36000][-1]
        if key == "weight_stages":
            cfg.rewards[params["reward_name"]].weight = stage["weight"]
        elif key == "standing_stages":
            cfg.commands[params["command_name"]].rel_standing_envs = stage["rel_standing_envs"]
        elif "command_name" in params:
            cfg.commands[params["command_name"]].ranges = copy.deepcopy(stage["ranges"])
        else:
            cfg.events[params["event_name"]].params["ranges"] = (-stage["range"], stage["range"])
    cfg.curriculum.clear()


def stages(mode):
    if mode.endswith("control") or mode == "control":
        return ((0, 0.),)
    if mode.startswith("smoke"):
        return ((0, 0.), (1, -.25), (2, -.50), (3, -1.))
    if mode.startswith("benchmark"):
        return ((0, -1.),)
    return MOTOR_STAGES


def motor_weight(step, schedule):
    weight = schedule[0][1]
    for iteration, value in schedule:
        if step > iteration * 24:
            weight = value
    return weight


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab_microduck.tasks import mdp

    worlds, updates, seed = MODES[mode]
    cfg, agent = base.prepare_config("foundation")
    freeze_source(cfg)
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    # Both arms have the identical term; its sole difference is the weight.
    cfg.rewards["motor_load"] = RewardTermCfg(func=mdp.motor_torque_load_cost,
        weight=stages(mode)[0][1], params=dict(rated_stall_torque_nm=.60,
                                             soft_limit_fraction=.70, over_limit_gain=4.))
    agent.max_iterations, agent.save_interval = updates, 250
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def warm_load(runner, env, parent=PARENT, *, parent_sha=PARENT_SHA,
              parent_iteration=1499, parent_steps=36000):
    base.require(base.sha256(parent) == parent_sha, "exact new-plant parent bytes")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    base.require(base.finite_tree(payload) and payload["iter"] == parent_iteration and
                 payload["infos"]["env_state"]["common_step_counter"] == parent_steps, "finite parent state")
    previous = os.environ.get("MICRODUCK_WARM_START")
    os.environ["MICRODUCK_WARM_START"] = "1"
    try:
        runner.load(str(parent), strict=True, map_location=env.device)
    finally:
        if previous is None:
            os.environ.pop("MICRODUCK_WARM_START", None)
        else:
            os.environ["MICRODUCK_WARM_START"] = previous
    base.require(runner.current_learning_iteration == env.common_step_counter == 0, "warm counters reset")
    state = runner.alg.save()
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        base.require(equal_tree(state[key], payload[key]), "exact restored " + key)
    rates = {g["lr"] for g in state["optimizer_state_dict"]["param_groups"]}
    base.require(len(rates) == 1 and all(0 < r < 1 for r in rates), "single valid parent Adam LR")
    runner.alg.learning_rate = rates.pop()
    # Fresh episodes, not resumed simulator state. Config is pinned before init.
    env.reset()
    base.require(env.common_step_counter == 0, "fresh warm-start episodes")
    return dict(parent_sha256=parent_sha, restored_models_normalizers_adam=True,
                iteration=0, common_step_counter=0, learning_rate=runner.alg.learning_rate)


class MotorObserver:
    def __init__(self, mode):
        self.schedule = stages(mode)
        self.weights, self.costs, self.weighted = [], [], []
        self.weight = None

    def before(self, env):
        self.weight = motor_weight(env.common_step_counter, self.schedule)
        env.reward_manager.get_term_cfg("motor_load").weight = self.weight

    def step(self, env, sample, extras):
        index = env.reward_manager.active_terms.index("motor_load")
        weighted = env.reward_manager._step_reward[:, index]
        base.require(base.finite_tree(weighted) and bool((weighted <= 0).all()), "nonpositive motor reward")
        # These are the same pre-reset forces. Validate actual reward activity,
        # including terminal worlds, rather than trusting a configured weight.
        utilization = (sample.force_nm.abs().float() / .60).clamp(max=2.)
        expected = (utilization.square() + 4.*(utilization-.70).clamp_min(0).square()).mean(1) * self.weight
        base.require(torch.allclose(weighted, expected, atol=1e-6, rtol=1e-5), "motor reward matches pre-reset cost")
        self.weights.append(self.weight)
        self.costs.append(float(sample.mean_cost.mean()))
        self.weighted.append(float(weighted.mean()))

    def update(self, env):
        row = dict(motor_weight_min=min(self.weights), motor_weight_max=max(self.weights),
                   motor_cost_mean=sum(self.costs)/24, weighted_motor_reward_mean=sum(self.weighted)/24)
        self.weights.clear(); self.costs.clear(); self.weighted.clear()
        return row


def onnx_check(path):
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    value = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
    base.require(value.shape == (1, 14) and np.isfinite(value).all(), "official normalized ONNX CPU smoke")
    return dict(path=str(path), sha256=base.sha256(path), input=[1, 61], output=[1, 14], finite=True)


def paired_differences(control, motor, *, label="motor_minus_control"):
    base.require(len(control) == len(motor) == len(base.SEEDS)*len(base.CASES), "complete paired matrix")
    differences = []
    for left, right in zip(control, motor, strict=True):
        identity = (left["seed"], left["speed"], left["yaw"])
        base.require(identity == (right["seed"], right["speed"], right["yaw"]), "matched case identity")
        delta = {}
        for key in ("torque_p99", "soft_limit_fraction", "rated_speed_exceed_fraction",
                    "mean_abs_mechanical_power_w", "thermal_load_proxy"):
            delta[key] = right[key] - left[key]
        for key in ("speed_mae", "yaw_mae"):
            # A partial/fallen case is not made comparable by resampling it.
            delta[key+"_mean"] = (sum(right[key])-sum(left[key]))/8 if left["complete"] and right["complete"] else None
        differences.append(dict(seed=identity[0], speed=identity[1], yaw=identity[2],
                                both_complete=left["complete"] and right["complete"], **{label: delta}))
    return differences


def main(experiment=None):
    # Share the bounded two-arm harness; the original recipe is the default.
    recipe = sys.modules[__name__] if experiment is None else experiment
    arms = getattr(recipe, "ARMS", ("control", "motor"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*recipe.MODES, "evaluate"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    base.require(re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id), "safe unique run id")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(ROOT), *argv), text=True).strip()
    base.require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"), "exact clean source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal GPU0 selection")
    base.require(LOCK.is_file() and not LOCK.is_symlink(), "existing regular GPU lease")
    root = ROOT / "artifacts/training" / args.run_id
    output = root / args.mode
    with LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = shared_host()
        if args.mode.startswith("benchmark"):
            arm = args.mode.split("-")[1]
            receipt = json.loads((root / f"smoke-{arm}/result.json").read_text())
            base.require(receipt["source"] == args.source and receipt["updates"] == 5 and
                         receipt["worlds"] == 64 and receipt["protocol"] == recipe.PROTOCOL
                         and receipt["status"] == "training-complete-not-accepted"
                         and receipt["onnx"]["finite"] and
                         base.sha256(receipt["onnx"]["path"]) == receipt["onnx"]["sha256"], "same-arm ONNX smoke before benchmark")
        if args.mode in arms:
            for arm in arms:
                for phase, worlds, updates in (("smoke", 64, 5), ("benchmark", 256, 10)):
                    receipt = json.loads((root / f"{phase}-{arm}/result.json").read_text())
                    base.require(receipt["source"] == args.source and receipt["worlds"] == worlds and
                                 receipt["protocol"] == recipe.PROTOCOL and receipt["updates"] == updates
                                 and receipt["warm_start"]["parent_sha256"] == recipe.PARENT_SHA
                                 and receipt["status"] == "training-complete-not-accepted", "both same-source preflights")
                    if phase == "smoke":
                        base.require(receipt["onnx"]["finite"] and base.sha256(receipt["onnx"]["path"]) == receipt["onnx"]["sha256"], "durable ONNX smoke")
                    else:
                        base.require(receipt["elapsed_s"]*100*1.5 < recipe.CAP_SECONDS-60, "measured continuation fits cap")
            if args.mode == arms[1]:
                control = json.loads((root / "control/result.json").read_text())
                base.require(control["source"] == args.source and control["updates"] == 1000 and
                             control["protocol"] == recipe.PROTOCOL and
                             control["warm_start"]["parent_sha256"] == recipe.PARENT_SHA and
                             control["status"] == "training-complete-not-accepted" and
                             base.sha256(root / "control/model_999.pt") == control["checkpoint_sha256"], "control completes before motor arm")
        output.mkdir(parents=True, exist_ok=False)
        base.write_new(output / "launch.json", dict(protocol=recipe.PROTOCOL, source=args.source, mode=args.mode,
            host=host, parent=str(recipe.PARENT), parent_sha256=recipe.PARENT_SHA, runtime={n: metadata.version(n) for n in
                ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "better-actuator-models")},
            policy_acceptance=False, physical_motion_authorized=False))
        def health():
            sample = shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            if args.mode == "evaluate":
                cases, decisions, hashes = {}, {}, {}
                for arm in arms:
                    trained = json.loads((root / arm / "result.json").read_text())
                    checkpoint = root / arm / "model_999.pt"
                    base.require(trained["source"] == args.source and trained["updates"] == 1000 and
                                 trained["protocol"] == recipe.PROTOCOL and
                                 trained["status"] == "training-complete-not-accepted" and
                                 trained["warm_start"]["parent_sha256"] == recipe.PARENT_SHA and
                                 base.sha256(checkpoint) == trained["checkpoint_sha256"], "complete matched arm")
                    rows = []
                    for seed in base.SEEDS:
                        for index, (speed, yaw) in enumerate(base.CASES):
                            row = getattr(recipe, "evaluate_case", base.evaluate_case)(checkpoint, seed, speed, yaw, health)
                            base.write_new(output / f"{arm}-s{seed}-c{index}.json", row)
                            rows.append(row)
                    cases[arm], decisions[arm] = rows, getattr(recipe, "decision", base.decision)(rows)
                    hashes[arm] = base.sha256(checkpoint)
                result = dict(cases=cases, decisions=decisions, checkpoint_sha256=hashes,
                    paired_differences=paired_differences(cases[arms[0]], cases[arms[1]],
                        label=arms[1]+"_minus_control"),
                    selection="both final999 only; unchanged held-out protocol", policy_acceptance=False,
                    simulator_qualified=False, physical_motion_authorized=False)
            else:
                observer, restored = recipe.MotorObserver(args.mode), {}
                def initialize(runner, env):
                    restored.update(recipe.warm_load(runner, env))
                    base.write_new(output / "warm-start.json", restored)
                result = base.training(output, args.mode, health, recipe=recipe.prepare_config(args.mode),
                    initialize=initialize, before_action=observer.before,
                    after_step=observer.step, after_update=observer.update)
                result["warm_start"] = restored
                if args.mode.startswith("smoke"):
                    result["onnx"] = onnx_check(output / f"{args.mode}.onnx")
            base.write_new(output / "result.json", dict(protocol=recipe.PROTOCOL, source=args.source, **result))
            health()
        except Exception as exc:
            base.write_new(output / "failure.json", dict(type=type(exc).__name__, error=str(exc), source=args.source))
            raise


if __name__ == "__main__":
    main()
