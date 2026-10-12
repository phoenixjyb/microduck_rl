"""Fresh-policy training on the integrated plant with stock curricula."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time

import torch

from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_straight_lesson as straight
from mjlab_microduck import integrated_training_budget_refinement as budget
from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host

PROTOCOL = "integrated-fresh-curriculum-foundation-v1"
MODES = {"smoke": (64, 5, 821), "benchmark": (256, 10, 823),
         "foundation": (256, 6000, 1327)}
FOUNDATION_CAP_SECONDS = 9000
FOUNDATION_PROJECTION_LIMIT_SECONDS = 8940
INITIAL_LEARNING_RATE = 1e-3
REQUIRED_COVERAGE = base.FULL_COMMAND_COVERAGE
EVAL_SEEDS = base.SEEDS
EVAL_CASES = base.CASES
FINAL_CHECKPOINT = 5999
evaluate_case = straight.evaluate_case
decision = budget.decision


def require(condition, message):
    base.require(condition, message)


def prepare_config(mode):
    require(mode in MODES, "declared fresh foundation mode")
    cfg, agent = base.prepare_config("foundation")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations, agent.save_interval = updates, 250
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def benchmark_projection_seconds(benchmark_elapsed_s):
    require(type(benchmark_elapsed_s) in (int, float)
            and math.isfinite(benchmark_elapsed_s) and benchmark_elapsed_s > 0,
            "finite positive benchmark elapsed time")
    return benchmark_elapsed_s * (MODES["foundation"][1] / MODES["benchmark"][1]) * 1.5


def projection_admitted(benchmark_elapsed_s):
    return benchmark_projection_seconds(benchmark_elapsed_s) < FOUNDATION_PROJECTION_LIMIT_SECONDS


def normalizer_count(model):
    """Find the single empirical observation normalizer on an actor or critic."""
    matches = []
    for module in model.modules():
        count = getattr(module, "count", None)
        if (isinstance(count, torch.Tensor) and count.ndim == 0
                and all(isinstance(getattr(module, name, None), torch.Tensor)
                        for name in ("_mean", "_var", "_std"))):
            matches.append(count)
    require(len(matches) == 1, "exactly one empirical observation normalizer")
    count = matches[0]
    require(count.dtype == torch.int64 and count.item() == 0,
            "fresh empty integer observation normalizer")
    return 0


def validate_fresh_runner(runner, env, *, install_capture=True):
    """Reject restored/advanced learners before the first checkpoint is saved."""
    alg = runner.alg
    require(type(runner.current_learning_iteration) is int
            and runner.current_learning_iteration == 0, "fresh learner iteration clock")
    require(type(env.common_step_counter) is int and env.common_step_counter == 0,
            "fresh environment curriculum clock")
    require(alg.schedule == "adaptive" and alg.desired_kl == .01
            and alg.num_learning_epochs == 5 and alg.num_mini_batches == 4,
            "stock adaptive PPO schedule")
    require(type(alg.learning_rate) in (int, float)
            and math.isfinite(alg.learning_rate)
            and alg.learning_rate == INITIAL_LEARNING_RATE,
            "stock initial Python learning rate")
    rates = [group.get("lr") for group in alg.optimizer.param_groups]
    require(rates and all(type(rate) in (int, float) and math.isfinite(rate)
                          and rate == INITIAL_LEARNING_RATE for rate in rates),
            "fresh Adam groups use stock initial learning rate")
    require(isinstance(alg.optimizer.state, dict) and len(alg.optimizer.state) == 0,
            "fresh Adam has no moments")
    require(type(alg.optimizer) is torch.optim.Adam, "canonical fresh Adam optimizer")
    model_parameters = [parameter for model in (alg.actor, alg.critic)
                        for parameter in model.parameters()]
    optimizer_parameters = [parameter for group in alg.optimizer.param_groups
                            for parameter in group["params"]]
    require(model_parameters
            and len({id(parameter) for parameter in model_parameters}) == len(model_parameters)
            and {id(parameter) for parameter in optimizer_parameters}
            == {id(parameter) for parameter in model_parameters}
            and len(optimizer_parameters) == len(model_parameters),
            "fresh Adam owns exactly unique actor and critic parameters")
    require(base.finite_tree(alg.actor.state_dict())
            and base.finite_tree(alg.critic.state_dict()), "finite fresh actor and critic")
    actor_count = normalizer_count(alg.actor)
    critic_count = normalizer_count(alg.critic)
    if install_capture:
        kl.install_optimizer_capture(runner, env, initial_lr=INITIAL_LEARNING_RATE)
    return dict(protocol="fresh-policy-initialization-v1", initialized_from_checkpoint=False,
                parent_checkpoint=None, learner_iteration=0, common_step_counter=0,
                adam_state_entries=0, actor_normalizer_count=actor_count,
                critic_normalizer_count=critic_count, initial_learning_rate=INITIAL_LEARNING_RATE,
                learning_rate_schedule="adaptive", desired_kl=.01,
                optimizer_capture="read-only-raw-kl-telemetry")


class FreshObserver:
    """Read-only actor-command dwell and curriculum/optimizer telemetry."""

    def __init__(self, mode):
        require(mode in MODES, "declared fresh observer mode")
        self.mode = mode
        self.pending = None
        self.exposure_steps = 0
        self.exposure_counts = dict.fromkeys(budget.EXPOSURE_BINS, 0)
        self.cumulative_exposure = dict.fromkeys(budget.EXPOSURE_BINS, 0)

    def before_action(self, env):
        require(self.pending is None, "previous actor command consumed")
        command = env.command_manager.get_command("twist")
        require(command.shape == (env.num_envs, 3), "actor twist world layout")
        self.pending = (env.common_step_counter, budget.command_exposure(command))

    def after_step(self, env, sample, extras):
        del sample, extras
        require(self.pending is not None
                and env.common_step_counter == self.pending[0] + 1,
                "one actor command consumed by one environment step")
        for name, count in self.pending[1].items():
            self.exposure_counts[name] += count
            self.cumulative_exposure[name] += count
        self.exposure_steps += 1
        self.pending = None

    @staticmethod
    def _value(value):
        if isinstance(value, torch.Tensor):
            require(value.numel() == 1 and base.finite_tree(value), "finite scalar curriculum value")
            return value.item()
        if isinstance(value, dict):
            return {str(key): FreshObserver._value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FreshObserver._value(item) for item in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            require(math.isfinite(value), "finite curriculum value")
            return value
        return value

    @classmethod
    def curriculum_profile(cls, env):
        reward_manager = env.reward_manager
        command = env.command_manager.get_term("twist").cfg
        head = env.command_manager.get_term("head_pose").cfg
        profile = dict(
            common_step_counter=env.common_step_counter,
            action_rate_weight=reward_manager.get_term_cfg("action_rate_l2").weight,
            head_pose_bias_weight=reward_manager.get_term_cfg("head_pose_bias").weight,
            rel_standing_envs=command.rel_standing_envs,
            head_pose_ranges=head.ranges,
        )
        for event_name in ("randomize_com", "randomize_head_com"):
            try:
                term = env.event_manager.get_term_cfg(event_name)
            except (KeyError, ValueError):
                profile[event_name] = None
            else:
                require(isinstance(term.params, dict) and "ranges" in term.params,
                        "live numeric CoM curriculum range")
                profile[event_name] = dict(ranges=term.params["ranges"],
                                           operation=term.params.get("operation"))
        return {key: cls._value(value) for key, value in profile.items()}

    def after_update(self, env):
        worlds, updates, _ = MODES[self.mode]
        require(self.pending is None and self.exposure_steps == 24
                and sum(self.exposure_counts.values()) == 24 * worlds,
                "complete 24-step command exposure")
        state = getattr(env, "_microduck_optimizer_audit", None)
        require(state is not None, "installed fresh adaptive-KL capture")
        optimizer = kl.optimizer_summary(env, .01)
        row = dict(command_exposure_protocol="actor-command-dwell-v1",
                   command_exposure_counts=dict(self.exposure_counts),
                   command_exposure_cumulative=dict(self.cumulative_exposure),
                   command_exposure_steps=self.exposure_steps,
                   optimizer=optimizer, curriculum=self.curriculum_profile(env))
        self.exposure_steps = 0
        self.exposure_counts = dict.fromkeys(budget.EXPOSURE_BINS, 0)
        return row

    def receipt(self, result):
        worlds, updates, _ = MODES[self.mode]
        require(result["updates"] == updates and result["worlds"] == worlds,
                "observer result matches declared mode")
        total = updates * 24 * worlds
        require(sum(self.cumulative_exposure.values()) == total,
                "complete cumulative action-step command exposure")
        coverage = dict(idle=self.cumulative_exposure["idle"],
                        straight=self.cumulative_exposure["straight"],
                        negative_yaw=sum(self.cumulative_exposure[key] for key in
                                         ("negative-low", "negative-target", "negative-high")),
                        positive_yaw=sum(self.cumulative_exposure[key] for key in
                                         ("positive-low", "positive-target", "positive-high")))
        require(all(coverage[key] > 0 for key in REQUIRED_COVERAGE),
                "actual idle/straight/both-yaw training command coverage")
        require(all(result[key] > 0 for key in REQUIRED_COVERAGE),
                "training harness independently observed all four command buckets")
        return dict(protocol="actor-command-dwell-v1", steps=updates * 24,
                    transitions=total, counts=dict(self.cumulative_exposure),
                    required_counts=coverage,
                    required_coverage=list(REQUIRED_COVERAGE))


def validate_training_receipt(root, mode, source, receipt):
    require(mode in MODES and isinstance(receipt, dict), "declared training receipt")
    worlds, updates, seed = MODES[mode]
    output = Path(root) / mode
    checkpoint = output / f"model_{updates-1}.pt"
    require(receipt.get("protocol") == PROTOCOL and receipt.get("source") == source
            and receipt.get("status") == "training-complete-not-accepted"
            and receipt.get("mode") == mode and receipt.get("worlds") == worlds
            and receipt.get("seed") == seed and receipt.get("updates") == updates
            and receipt.get("steps") == updates * 24
            and receipt.get("transitions") == worlds * updates * 24,
            "exact-source complete fresh training budget")
    require(receipt.get("required_command_coverage") == list(REQUIRED_COVERAGE)
            and all(type(receipt.get(key)) is int and receipt[key] > 0
                    for key in REQUIRED_COVERAGE), "complete required command coverage")
    exposure = receipt.get("command_exposure")
    require(isinstance(exposure, dict)
            and exposure.get("protocol") == "actor-command-dwell-v1"
            and exposure.get("steps") == updates * 24
            and exposure.get("transitions") == worlds * updates * 24
            and exposure.get("required_coverage") == list(REQUIRED_COVERAGE)
            and isinstance(exposure.get("counts"), dict)
            and set(exposure["counts"]) == set(budget.EXPOSURE_BINS)
            and all(type(value) is int and value >= 0 for value in exposure["counts"].values())
            and sum(exposure["counts"].values()) == worlds * updates * 24
            and isinstance(exposure.get("required_counts"), dict)
            and set(exposure["required_counts"]) == set(REQUIRED_COVERAGE)
            and all(type(exposure["required_counts"][key]) is int
                    and exposure["required_counts"][key] > 0 for key in REQUIRED_COVERAGE),
            "complete consumed actor-command dwell receipt")
    dwell = exposure["counts"]
    required_counts = exposure["required_counts"]
    require(required_counts["idle"] == dwell["idle"]
            and required_counts["straight"] == dwell["straight"]
            and required_counts["negative_yaw"] == sum(dwell[key] for key in
                ("negative-low", "negative-target", "negative-high"))
            and required_counts["positive_yaw"] == sum(dwell[key] for key in
                ("positive-low", "positive-target", "positive-high")),
            "consumed command coverage bins reconcile")
    require(Path(receipt.get("checkpoint", "")).resolve() == checkpoint.resolve()
            and checkpoint.is_file() and receipt.get("checkpoint_sha256") == base.sha256(checkpoint),
            "exact durable final checkpoint bytes")
    initialization_path = output / "fresh-initialization.json"
    require(initialization_path.is_file(), "durable fresh initialization receipt")
    initialization = json.loads(initialization_path.read_text())
    require(receipt.get("fresh_initialization") == initialization
            and initialization.get("source") == source
            and initialization.get("protocol") == "fresh-policy-initialization-v1"
            and initialization.get("initialized_from_checkpoint") is False
            and initialization.get("parent_checkpoint") is None
            and initialization.get("learner_iteration") == 0
            and initialization.get("common_step_counter") == 0
            and initialization.get("adam_state_entries") == 0
            and initialization.get("actor_normalizer_count") == 0
            and initialization.get("critic_normalizer_count") == 0
            and initialization.get("initial_learning_rate") == INITIAL_LEARNING_RATE
            and initialization.get("learning_rate_schedule") == "adaptive"
            and initialization.get("desired_kl") == .01
            and initialization.get("optimizer_capture") == "read-only-raw-kl-telemetry",
            "result bound to exact fresh, empty learner initialization")
    initial_checkpoint = output / "initial.pt"
    require(initialization.get("initial_checkpoint") == str(initial_checkpoint)
            and initial_checkpoint.is_file()
            and initialization.get("initial_checkpoint_sha256") == base.sha256(initial_checkpoint),
            "exact durable initial checkpoint binding")
    elapsed = receipt.get("elapsed_s")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
            "finite positive training elapsed time")
    if mode == "smoke":
        onnx = receipt.get("onnx")
        onnx_path = output / "smoke.onnx"
        require(isinstance(onnx, dict) and onnx.get("finite") is True
                and onnx.get("input") == [1, 61] and onnx.get("output") == [1, 14]
                and Path(onnx.get("path", "")).resolve() == onnx_path.resolve()
                and onnx_path.is_file()
                and base.sha256(onnx["path"]) == onnx.get("sha256"),
                "durable normalized ONNX 61-to-14 CPU smoke")
    if mode == "foundation":
        require(elapsed <= FOUNDATION_CAP_SECONDS, "foundation completed within declared cap")
    return dict(mode=mode, worlds=worlds, updates=updates, seed=seed,
                steps=updates * 24, transitions=worlds * updates * 24,
                checkpoint=str(checkpoint), checkpoint_sha256=receipt["checkpoint_sha256"],
                elapsed_s=elapsed)


def validate_evaluation_receipt(root, source, receipt):
    validate_training_receipt(root, "foundation", source, receipt)
    final = Path(root) / "foundation" / "model_5999.pt"
    require(receipt.get("checkpoint") == str(final)
            and final.is_file() and receipt.get("checkpoint_sha256") == base.sha256(final),
            "exact fresh final5999 checkpoint bytes")
    return final


def onnx_check(path):
    return motor.onnx_check(path)


def evaluate(root, output, source, health):
    root = Path(root)
    trained = json.loads((root / "foundation" / "result.json").read_text())
    checkpoint = validate_evaluation_receipt(root, source, trained)
    rows = []
    for seed in EVAL_SEEDS:
        for index, (speed, yaw) in enumerate(EVAL_CASES):
            row = evaluate_case(checkpoint, seed, speed, yaw, health)
            base.write_new(output / f"s{seed}-c{index}.json", row)
            rows.append(row)
    result = decision(rows)
    result.update(cases=rows, checkpoint=str(checkpoint),
                  checkpoint_sha256=base.sha256(checkpoint), final_iteration=FINAL_CHECKPOINT,
                  selection="final5999 only; no checkpoint fishing",
                  policy_acceptance=False, simulator_qualified=False,
                  physical_motion_authorized=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*MODES, "evaluate"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id), "safe unique run id")

    def git(*argv):
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()

    require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"),
            "exact clean source")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated device selection")
    require(LOCK.is_file() and not LOCK.is_symlink(), "existing nonsymlink GPU lease")
    root = base.ROOT / "artifacts/training" / args.run_id
    output = root / args.mode
    with LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = shared_host()
        if args.mode in ("benchmark", "foundation"):
            smoke = json.loads((root / "smoke" / "result.json").read_text())
            validate_training_receipt(root, "smoke", args.source, smoke)
        if args.mode == "foundation":
            bench = json.loads((root / "benchmark" / "result.json").read_text())
            validate_training_receipt(root, "benchmark", args.source, bench)
            projected = benchmark_projection_seconds(bench["elapsed_s"])
            require(projection_admitted(bench["elapsed_s"]),
                    "measured benchmark projection fits 9000-second cap with margin")
        if args.mode == "evaluate":
            trained = json.loads((root / "foundation" / "result.json").read_text())
            validate_evaluation_receipt(root, args.source, trained)
        output.mkdir(parents=True, exist_ok=False)
        runtime = {name: metadata.version(name) for name in
                   ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "better-actuator-models")}
        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source,
            mode=args.mode, runtime=runtime, host=host, parent_checkpoint=None,
            initialization="fresh actor/critic/normalizers/Adam", policy_acceptance=False,
            physical_motion_authorized=False, foundation_cap_seconds=FOUNDATION_CAP_SECONDS
            if args.mode == "foundation" else None))

        def health():
            sample = shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
            return sample

        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            if args.mode == "evaluate":
                result = evaluate(root, output, args.source, health)
            else:
                observer = FreshObserver(args.mode)
                initialization = {}

                def initialize(runner, env):
                    initialization.update(validate_fresh_runner(runner, env))

                result = base.training(output, args.mode, health,
                    recipe=prepare_config(args.mode), initialize=initialize,
                    before_action=observer.before_action, after_step=observer.after_step,
                    after_update=observer.after_update, required_command_coverage=REQUIRED_COVERAGE)
                result["command_exposure"] = observer.receipt(result)
                initial_checkpoint = output / "initial.pt"
                require(initial_checkpoint.is_file(), "durable initial fresh checkpoint")
                initialization["initial_checkpoint"] = str(initial_checkpoint)
                initialization["initial_checkpoint_sha256"] = base.sha256(initial_checkpoint)
                initialization["source"] = args.source
                base.write_new(output / "fresh-initialization.json", initialization)
                result["fresh_initialization"] = initialization
                if args.mode == "smoke":
                    result["onnx"] = onnx_check(output / f"{args.mode}.onnx")
            result.update(protocol=PROTOCOL, source=args.source)
            if args.mode == "foundation":
                result["cap_seconds"] = FOUNDATION_CAP_SECONDS
                result["benchmark_projection_seconds"] = projected
            base.write_new(output / "result.json", result)
            health()
        except Exception as exc:
            base.write_new(output / "failure.json", dict(type=type(exc).__name__,
                error=str(exc), source=args.source, protocol=PROTOCOL))
            raise


if __name__ == "__main__":
    main()
