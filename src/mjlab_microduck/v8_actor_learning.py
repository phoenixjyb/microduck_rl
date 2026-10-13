"""Bounded v8-seeded foundation learning with unchanged held-out gates.

Benchmark and foundation each start from the same pinned deterministic actor,
not from the previous stage's PPO state. The benchmark initial actor is the
fixed inference baseline. No checkpoint search, automatic promotion or motion.
"""
from __future__ import annotations

import argparse
import fcntl
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
from . import v8_actor_smoke as smoke
from . import v8_actor_lr_profile as learning_profiles

PROTOCOL = "v8-actor-only-foundation-learning-v1"
MODES = {"benchmark": (256, 10, 953), "foundation": (256, 1000, 967)}
CAPS = {"benchmark": 300, "initial-eval": 600, "foundation": 2100, "final-eval": 600}


def prepare_config(mode, *, profile="stock"):
    selected = learning_profiles.profile(profile)
    modes = MODES if profile == "stock" else selected.modes
    base.require(mode in modes, "predeclared v8 learning mode")
    cfg, agent = smoke.prepare_config()
    worlds, updates, seed = modes[mode]
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.seed = agent.seed = seed
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = selected.protocol, mode
    if profile == "cautious":
        agent.algorithm.learning_rate = selected.initial_lr
    return cfg, agent


def benchmark_projection_seconds(elapsed_s):
    base.require(type(elapsed_s) in (int, float) and math.isfinite(elapsed_s) and elapsed_s > 0,
                 "finite positive measured benchmark time")
    # Include measured setup/export costs in the deliberately conservative
    # multiplier; do not fit away overhead after seeing an inconvenient result.
    return elapsed_s * 100 * 1.5 + 60


def validate_training_receipt(root, mode, source, *, profile="stock"):
    selected = learning_profiles.profile(profile)
    modes = MODES if profile == "stock" else selected.modes
    base.require(mode in modes, "declared training receipt mode")
    output = Path(root) / mode
    result = json.loads((output / "result.json").read_text())
    worlds, updates, seed = modes[mode]
    base.require(result["protocol"] == selected.protocol and result.get("profile", "stock") == profile
                 and result["source"] == source
                 and result["status"] == "training-complete-not-accepted"
                 and (result["worlds"], result["updates"], result["seed"], result["steps"], result["transitions"])
                 == (worlds, updates, seed, updates*24, worlds*updates*24),
                 "exact-source complete declared learning budget")
    checkpoint = output / f"model_{updates-1}.pt"
    base.require(Path(result["checkpoint"]).resolve() == checkpoint.resolve()
                 and base.sha256(checkpoint) == result["checkpoint_sha256"]
                 and base.sha256(output/"initial.pt") == result["initial_checkpoint_sha256"],
                 "exact durable initial/final checkpoint bytes")
    base.require(result["actor_import"] == json.loads((output/"actor-import.json").read_text())
                 and result["actor_import"]["ppo_resume"] is False
                 and result["actor_import"]["adam_state_entries"] == 0
                 and result["actor_import"].get("initial_learning_rate", learning_profiles.STOCK_INITIAL_LR)
                     == selected.initial_lr
                 and result["actor_import"]["artifact_sha256"] == pinned.V8.sha256
                 and result["actor_import"]["maximum_actor_error"] <= 2e-5
                 and result["actor_import"]["maximum_export_forward_error"] <= 2e-5,
                 "bound fresh actor-only initialization")
    if profile == "cautious":
        matched_path = output / "matched-initialization.json"
        matched = json.loads(matched_path.read_text())
        base.require(matched["protocol"] == "v8-matched-initialization-v1"
                     and matched["profile"] == "cautious" and matched["mode"] == mode
                     and matched["source"] == source and matched["seed"] == seed
                     and matched["control_iteration"] == matched["candidate_iteration"] == 0
                     and matched["control_common_step_counter"] == matched["candidate_common_step_counter"] == 0
                     and matched["control_learning_rate"] == learning_profiles.STOCK_INITIAL_LR
                     and matched["candidate_learning_rate"] == learning_profiles.CAUTIOUS_INITIAL_LR
                     and matched["ppo_resume"] is False and matched["tensors_in_report"] is False
                     and matched["actor_state_equal"] is True
                     and matched["critic_state_equal"] is True
                     and matched["optimizer_state_empty"] is True
                     and matched["parameter_groups_match_except_lr"] is True
                     and matched["trajectory_identity_claim"] is False
                     and matched["policy_acceptance"] is False
                     and matched["physical_motion_authorized"] is False,
                     "source/profile-bound matched initial state")
        base.require(result["initial_learning_rate"] == learning_profiles.CAUTIOUS_INITIAL_LR
                     and result["matched_initialization"] == matched,
                     "cautious receipt binds its initial LR and matched-start report")
        validate_first = learning_profiles.validate_first_update_kl(root, mode, source)
        base.require(result["first_update_kl"] == validate_first,
                     "cautious receipt binds its first-update KL guard")
    else:
        base.require(result.get("initial_learning_rate", learning_profiles.STOCK_INITIAL_LR)
                     == learning_profiles.STOCK_INITIAL_LR,
                     "stock receipt retains its declared initial learning rate")
    base.require(all(result[k] > 0 for k in base.FULL_COMMAND_COVERAGE)
                 and base.finite_tree(result), "finite result and full command coverage")
    base.require(result["elapsed_s"] <= selected.caps[mode], "completed within declared process cap")
    return result


def validate_baseline(rows):
    """Tracking is the learning target; catastrophic baseline failure aborts."""
    expected = [(s,v,w) for s in base.SEEDS for v,w in base.CASES]
    base.require(len(rows) == 12 and base.finite_tree(rows)
                 and [(r["seed"],r["speed"],r["yaw"]) for r in rows] == expected,
                 "complete ordered fixed baseline matrix")
    for row in rows:
        d = row["response_diagnostics"]
        base.require(row["complete"] and row["steps"] == 240 and row["worlds"] == 8
                     and not row["terminal_worlds"] and row["rated_speed_exceed_fraction"] == 0
                     and row["torque_p99"] <= .6
                     and min(d["min_distance_to_hard_stop_rad_by_joint"]) >= -.005
                     and d["hard_stop_proximity_fraction_by_joint"][d["joint_columns"].index("right_hip_yaw")] <= .01,
                     "baseline complete first episodes and unchanged motor/range safety gates")


def evaluate_matrix(checkpoint, output, health, *, initial=False):
    from .v8_actor_checkpoint import FrozenV8Checkpoint
    typed = FrozenV8Checkpoint(checkpoint, base.sha256(checkpoint))
    if initial:
        typed.require_initial_seed()
    rows = []
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            health()
            row = base.evaluate_case(checkpoint, seed, speed, yaw, health,
                diagnostics=True, yaw_trace=True, gait_trace=True, frozen_v8_checkpoint=typed)
            base.require(row["checkpoint_sha256"] == typed.sha256
                         and (row["seed"],row["speed"],row["yaw"]) == (seed,speed,yaw),
                         "bound exact checkpoint and ordered held-out cases")
            base.write_new(output / f"s{seed}-c{index}.json", row)
            rows.append(row)
    return dict(checkpoint_sha256=typed.sha256, cases=rows,
                numerical_gate=limits.decision(rows), inference_backend=typed.provenance(),
                policy_acceptance=False, physical_motion_authorized=False)


def paired_comparison(baseline, candidate):
    base.require(len(baseline) == len(candidate) == 12, "complete paired twelve-case matrices")
    rows = []
    for left, right in zip(baseline, candidate, strict=True):
        identity = (left["seed"],left["speed"],left["yaw"])
        base.require(identity == (right["seed"],right["speed"],right["yaw"]), "paired held-out identity")
        row = dict(seed=identity[0], speed=identity[1], yaw=identity[2],
                   complete_pair=left["complete"] and right["complete"])
        for key in ("speed_mae","yaw_mae"):
            row[key+"_mean_delta"] = ((sum(right[key])-sum(left[key]))/8 if row["complete_pair"] else None)
        for key in ("torque_p99","abs_force_p99_nm","soft_limit_fraction",
                    "rated_speed_exceed_fraction","mean_abs_mechanical_power_w","thermal_load_proxy"):
            row[key+"_delta"] = right[key]-left[key]
        rows.append(row)
    return rows


def assemble_result_receipt(source, protocol, mode, result, *, profile="stock"):
    """Merge result fields without allowing protocol identity key collisions."""
    learning_profiles.profile(profile)
    reserved = {"source", "protocol", "mode", "profile"}
    receipt = {key: value for key, value in result.items() if key not in reserved}
    receipt.update(source=source, protocol=protocol, mode=mode)
    if profile == "cautious":
        receipt["profile"] = profile
    return receipt


class Observer:
    """Read-only target curriculum and adaptive-KL telemetry each update."""
    def __init__(self):
        self.dwell = dict.fromkeys(exposure.EXPOSURE_BINS, 0)
        self.pending = None
        self.runner = None
        self.output = None
        self.mode = None
        self.source = None
        self.profile = "stock"
        self.updates = 0

    def before(self, env):
        base.require(self.pending is None, "previous action's command consumed")
        self.pending = (env.common_step_counter,
                        exposure.command_exposure(env.command_manager.get_command("twist")))

    def step(self, env, sample, extras):
        del sample, extras
        base.require(self.pending is not None and env.common_step_counter == self.pending[0]+1,
                     "one actual command per physics step")
        for key,value in self.pending[1].items():
            self.dwell[key] += value
        self.pending = None

    def update(self, env):
        from .integrated_fresh_foundation import FreshObserver
        base.require(self.pending is None, "command consumed before optimizer update")
        summary = kl.optimizer_summary(env, .01)
        self.updates += 1
        row = dict(optimizer=summary,
                   target_curriculum=FreshObserver.curriculum_profile(env),
                   command_exposure_cumulative=dict(self.dwell))
        if self.profile == "cautious" and self.updates == 1:
            report = learning_profiles.first_update_kl_report(
                self.mode, self.source, summary, steps=env.common_step_counter)
            base.write_new(self.output / "first-update-kl.json", report)
            row["first_update_kl"] = report
            if not report["passed"]:
                base.require(self.runner is not None, "runner attached for cautious early abort")
                self.runner.save(str(self.output / "aborted-first-update.pt"), infos={
                    "profile": "cautious", "mode": self.mode, "source": self.source,
                    "common_step_counter": env.common_step_counter,
                    "first_update_kl_sha256": base.sha256(self.output / "first-update-kl.json"),
                    "full_budget_completed": False, "policy_acceptance": False,
                    "physical_motion_authorized": False,
                })
                base.require(False, "cautious first-update KL guard")
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("stock", "cautious"), default="stock")
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    selected = learning_profiles.profile(args.profile)
    modes = MODES if args.profile == "stock" else selected.modes
    caps = CAPS if args.profile == "stock" else selected.caps
    base.require(args.mode in caps, "predeclared profile stage")
    expected_run_id = (f"v8-actor-foundation-{args.source[:8]}-20261013"
                       if args.profile == "stock"
                       else f"{selected.run_prefix}-{args.source[:8]}-20261013")
    base.require(re.fullmatch(r"[a-f0-9]{40}", args.source) is not None
                 and args.run_id == expected_run_id,
                 "exact source-bound safe campaign run ID")
    def git(*argv):
        return subprocess.check_output(("git","-C",str(base.ROOT),*argv), text=True).strip()
    base.require(git("rev-parse","HEAD") == args.source and not git("status","--porcelain"),
                 "clean exact tested native source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated GPU selection")
    base.require(base.sha256(pinned.V8.path) == pinned.V8.sha256
                 and base.sha256(pinned.V8.path.with_name("manifest.json")) == pinned.V8.manifest_sha256,
                 "exact pinned v8 policy/manifest bytes")
    runtime = {p:metadata.version(p) for p in (*pinned.RUNTIME,"rsl-rl-lib","onnx","onnxruntime")}
    base.require(all(runtime[k].split("+")[0] == v for k,v in pinned.RUNTIME.items())
                 and runtime["rsl-rl-lib"] == "5.0.1", "unchanged installed runtime")
    base.require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing regular lease")
    with base.LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = base.shared_host()
        root = base.ROOT / "artifacts/training" / args.run_id
        binding = dict(source=args.source, protocol=selected.protocol, runtime=runtime,
            artifact_sha256=pinned.V8.sha256, manifest_sha256=pinned.V8.manifest_sha256,
            modes=modes, caps=caps, heldout_seeds=base.SEEDS, heldout_cases=base.CASES)
        if args.profile == "cautious":
            binding["profile"] = args.profile
        first_stage = "benchmark" if args.profile == "stock" else "smoke"
        if args.mode == first_stage:
            root.mkdir(parents=True, exist_ok=False)
            base.write_new(root/"campaign.json", binding)
        else:
            # JSON tuples become lists; compare deterministic serialized forms.
            retained = json.loads((root/"campaign.json").read_text())
            base.require(retained == json.loads(json.dumps(binding)), "exact retained campaign declaration")
            prior_stage = "benchmark" if args.profile == "stock" else "smoke"
            validate_training_receipt(root, prior_stage, args.source, profile=args.profile)
            if args.profile == "cautious" and args.mode == "benchmark":
                learning_profiles.validate_first_update_kl(root, "smoke", args.source)
            if args.profile == "cautious" and args.mode == "initial-eval":
                validate_training_receipt(root, "benchmark", args.source, profile=args.profile)
                learning_profiles.validate_first_update_kl(root, "benchmark", args.source)
        if args.mode in ("foundation","final-eval"):
            benchmark = validate_training_receipt(root, "benchmark", args.source, profile=args.profile)
            base.require(benchmark_projection_seconds(benchmark["elapsed_s"]) < caps["foundation"]-30,
                         "conservative measured benchmark projection fits declared cap")
            baseline = json.loads((root/"initial-eval/result.json").read_text())
            base.require(baseline["source"] == args.source
                         and baseline["checkpoint_sha256"] == benchmark["initial_checkpoint_sha256"],
                         "exact retained initial actor baseline")
            validate_baseline(baseline["cases"])
        output = root/args.mode
        output.mkdir(exist_ok=False)
        launch = dict(source=args.source, protocol=selected.protocol,
            mode=args.mode, host=host, cap_seconds=caps[args.mode],
            recipe=asdict(prepare_config(args.mode, profile=args.profile)[1]) if args.mode in modes else None,
            policy_acceptance=False, physical_motion_authorized=False)
        if args.profile == "cautious":
            launch["profile"] = args.profile
        base.write_new(output/"launch.json", launch)
        started = time.monotonic()
        def health():
            base.require(time.monotonic()-started < caps[args.mode]-15, "bounded stage deadline")
            sample = base.shared_host()
            with (output/"health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(),**sample),allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20,device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            if args.mode in modes:
                observer, imported = Observer(), {}
                observer.output, observer.mode = output, args.mode
                observer.source, observer.profile = args.source, args.profile
                def initialize(runner, env):
                    imported.update(smoke.initialize(runner,env,output))
                    if args.profile == "cautious":
                        observer.runner = runner
                        matched = learning_profiles.matched_initialization(
                            runner, env, args.mode, args.source)
                        base.write_new(output/"matched-initialization.json", matched)
                    kl.install_optimizer_capture(runner,env,initial_lr=selected.initial_lr)
                result = base.training(output,args.mode,health,
                    recipe=prepare_config(args.mode, profile=args.profile),
                    initialize=initialize,before_action=observer.before,
                    after_step=observer.step,after_update=observer.update)
                base.require(sum(observer.dwell.values()) == result["transitions"], "complete actual command dwell")
                from .integrated_motor_refinement import onnx_check
                result.update(actor_import=imported,command_exposure=observer.dwell,
                    initial_checkpoint_sha256=base.sha256(output/"initial.pt"),
                    onnx=onnx_check(output/f"{output.name}.onnx"),policy_acceptance=False,
                    physical_motion_authorized=False)
                if args.profile == "cautious":
                    result.update(profile=args.profile,
                        initial_learning_rate=selected.initial_lr,
                        matched_initialization=json.loads((output/"matched-initialization.json").read_text()),
                        first_update_kl=json.loads((output/"first-update-kl.json").read_text()))
            else:
                checkpoint = root/("benchmark/initial.pt" if args.mode == "initial-eval" else "foundation/model_999.pt")
                if args.mode == "final-eval":
                    validate_training_receipt(root,"foundation",args.source,profile=args.profile)
                result = evaluate_matrix(checkpoint,output,health,initial=args.mode == "initial-eval")
                if args.mode == "final-eval":
                    result["paired_descriptive_deltas"] = paired_comparison(baseline["cases"],result["cases"])
                result["elapsed_s"] = time.monotonic()-started
            health()
            base.write_new(output/"result.json", assemble_result_receipt(
                args.source, selected.protocol, args.mode, result, profile=args.profile))
        except Exception as error:
            base.write_new(output/"failure.json",dict(source=args.source,type=type(error).__name__,error=str(error)))
            raise


if __name__ == "__main__":
    main()
