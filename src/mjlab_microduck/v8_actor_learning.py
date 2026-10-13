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

PROTOCOL = "v8-actor-only-foundation-learning-v1"
MODES = {"benchmark": (256, 10, 953), "foundation": (256, 1000, 967)}
CAPS = {"benchmark": 300, "initial-eval": 600, "foundation": 2100, "final-eval": 600}


def prepare_config(mode):
    base.require(mode in MODES, "predeclared v8 learning mode")
    cfg, agent = smoke.prepare_config()
    worlds, updates, seed = MODES[mode]
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.seed = agent.seed = seed
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def benchmark_projection_seconds(elapsed_s):
    base.require(type(elapsed_s) in (int, float) and math.isfinite(elapsed_s) and elapsed_s > 0,
                 "finite positive measured benchmark time")
    # Include measured setup/export costs in the deliberately conservative
    # multiplier; do not fit away overhead after seeing an inconvenient result.
    return elapsed_s * 100 * 1.5 + 60


def validate_training_receipt(root, mode, source):
    base.require(mode in MODES, "declared training receipt mode")
    output = Path(root) / mode
    result = json.loads((output / "result.json").read_text())
    worlds, updates, seed = MODES[mode]
    base.require(result["protocol"] == PROTOCOL and result["source"] == source
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
                 and result["actor_import"]["artifact_sha256"] == pinned.V8.sha256
                 and result["actor_import"]["maximum_actor_error"] <= 2e-5
                 and result["actor_import"]["maximum_export_forward_error"] <= 2e-5,
                 "bound fresh actor-only initialization")
    base.require(all(result[k] > 0 for k in base.FULL_COMMAND_COVERAGE)
                 and base.finite_tree(result), "finite result and full command coverage")
    base.require(result["elapsed_s"] <= CAPS[mode], "completed within declared process cap")
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


class Observer:
    """Read-only target curriculum and adaptive-KL telemetry each update."""
    def __init__(self):
        self.dwell = dict.fromkeys(exposure.EXPOSURE_BINS, 0)
        self.pending = None

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
        return dict(optimizer=kl.optimizer_summary(env, .01),
                    target_curriculum=FreshObserver.curriculum_profile(env),
                    command_exposure_cumulative=dict(self.dwell))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=tuple(CAPS), required=True)
    args = parser.parse_args()
    base.require(re.fullmatch(r"[a-f0-9]{40}", args.source) is not None
                 and args.run_id == f"v8-actor-foundation-{args.source[:8]}-20261013",
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
        binding = dict(source=args.source, protocol=PROTOCOL, runtime=runtime,
            artifact_sha256=pinned.V8.sha256, manifest_sha256=pinned.V8.manifest_sha256,
            modes=MODES, caps=CAPS, heldout_seeds=base.SEEDS, heldout_cases=base.CASES)
        if args.mode == "benchmark":
            root.mkdir(parents=True, exist_ok=False)
            base.write_new(root/"campaign.json", binding)
        else:
            # JSON tuples become lists; compare deterministic serialized forms.
            retained = json.loads((root/"campaign.json").read_text())
            base.require(retained == json.loads(json.dumps(binding)), "exact retained campaign declaration")
            validate_training_receipt(root, "benchmark", args.source)
        if args.mode in ("foundation","final-eval"):
            benchmark = validate_training_receipt(root, "benchmark", args.source)
            base.require(benchmark_projection_seconds(benchmark["elapsed_s"]) < CAPS["foundation"]-30,
                         "conservative measured benchmark projection fits declared cap")
            baseline = json.loads((root/"initial-eval/result.json").read_text())
            base.require(baseline["source"] == args.source
                         and baseline["checkpoint_sha256"] == benchmark["initial_checkpoint_sha256"],
                         "exact retained initial actor baseline")
            validate_baseline(baseline["cases"])
        output = root/args.mode
        output.mkdir(exist_ok=False)
        base.write_new(output/"launch.json", dict(source=args.source,protocol=PROTOCOL,
            mode=args.mode, host=host, cap_seconds=CAPS[args.mode],
            recipe=asdict(prepare_config(args.mode)[1]) if args.mode in MODES else None,
            policy_acceptance=False, physical_motion_authorized=False))
        started = time.monotonic()
        def health():
            base.require(time.monotonic()-started < CAPS[args.mode]-15, "bounded stage deadline")
            sample = base.shared_host()
            with (output/"health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(),**sample),allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20,device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            if args.mode in MODES:
                observer, imported = Observer(), {}
                def initialize(runner, env):
                    imported.update(smoke.initialize(runner,env,output))
                    kl.install_optimizer_capture(runner,env,initial_lr=1e-3)
                result = base.training(output,args.mode,health,recipe=prepare_config(args.mode),
                    initialize=initialize,before_action=observer.before,
                    after_step=observer.step,after_update=observer.update)
                base.require(sum(observer.dwell.values()) == result["transitions"], "complete actual command dwell")
                from .integrated_motor_refinement import onnx_check
                result.update(actor_import=imported,command_exposure=observer.dwell,
                    initial_checkpoint_sha256=base.sha256(output/"initial.pt"),
                    onnx=onnx_check(output/f"{output.name}.onnx"),policy_acceptance=False,
                    physical_motion_authorized=False)
            else:
                checkpoint = root/("benchmark/initial.pt" if args.mode == "initial-eval" else "foundation/model_999.pt")
                if args.mode == "final-eval":
                    validate_training_receipt(root,"foundation",args.source)
                result = evaluate_matrix(checkpoint,output,health,initial=args.mode == "initial-eval")
                if args.mode == "final-eval":
                    result["paired_descriptive_deltas"] = paired_comparison(baseline["cases"],result["cases"])
                result["elapsed_s"] = time.monotonic()-started
            health()
            base.write_new(output/"result.json",dict(source=args.source,protocol=PROTOCOL,mode=args.mode,**{
                k:v for k,v in result.items() if k not in ("source","protocol","mode")}))
        except Exception as error:
            base.write_new(output/"failure.json",dict(source=args.source,type=type(error).__name__,error=str(error)))
            raise


if __name__ == "__main__":
    main()
