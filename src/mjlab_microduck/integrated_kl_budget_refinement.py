"""Matched adaptive-KL budget test; rewards, restored Adam and gates stay fixed."""

import importlib.metadata as metadata
import math
import sys
from types import SimpleNamespace

import torch

from mjlab_microduck import integrated_yaw_precision_refinement as precision

base, motor = precision.base, precision.motor
PROTOCOL = "integrated-adaptive-kl-budget-v1"
ARMS = ("control", "klbudget")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 991), ("benchmark", 256, 10, 997))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1009), klbudget=(256, 1000, 1009))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/yaw-precision-4447fe64-20261011/control/model_999.pt"
PARENT_SHA = "3e0b01bce31f8b73a535fa44bbfe92ce522908c03916269ad6ae203432b6249a"


def target(mode):
    return .01 if mode.endswith("control") else .02


def prepare_config(mode):
    cfg, agent = precision.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    base.require(agent.algorithm.desired_kl == .01 and agent.algorithm.schedule == "adaptive"
                 and agent.algorithm.num_mini_batches == 4 and agent.algorithm.num_learning_epochs == 5,
                 "known adaptive PPO configuration")
    agent.algorithm.desired_kl = target(mode)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def check_optimizer(alg, expected):
    rates = [g["lr"] for g in alg.optimizer.param_groups]
    base.require(not alg.is_multi_gpu and alg.schedule == "adaptive" and alg.desired_kl == expected
                 and alg.num_mini_batches == 4 and alg.num_learning_epochs == 5,
                 "fixed live adaptive KL configuration")
    base.require(rates and all(math.isfinite(v) and v == alg.learning_rate for v in rates)
                 and 1e-5 <= alg.learning_rate <= 1e-2, "consistent finite algorithm and Adam LR")
    return alg.learning_rate


def install_optimizer_capture(runner, env, *, initial_lr=1e-5):
    base.require(metadata.version("rsl-rl-lib") == "5.0.1", "reviewed PPO runtime")
    base.require(not hasattr(env, "_microduck_optimizer_audit"), "unique optimizer audit")
    alg = runner.alg
    expected = alg.desired_kl
    base.require(expected in (.01, .02), "predeclared KL target")
    base.require(isinstance(initial_lr, (int, float)) and not isinstance(initial_lr, bool)
                 and math.isfinite(initial_lr) and 1e-5 <= initial_lr <= 1e-2,
                 "finite predeclared initial parent LR")
    base.require(check_optimizer(alg, expected) == initial_lr, "known restored initial parent LR")
    original = alg.actor.get_kl_divergence
    state = SimpleNamespace(alg=alg, expected=expected, initial_lr=initial_lr, samples=[])
    env._microduck_optimizer_audit = state

    def measured(*args, **kwargs):
        rate = check_optimizer(alg, expected)
        value = original(*args, **kwargs)
        base.require(isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() > 0,
                     "raw per-sample KL layout")
        # Read-only detached summaries; transfer after update, not per minibatch.
        state.samples.append((torch.stack((value.mean(), value.min(), value.max())).detach().clone(), rate))
        return value  # exact original tensor, not clipped or reconstructed

    alg.actor.get_kl_divergence = measured


def warm_load(runner, env):
    restored = motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                               parent_iteration=999, parent_steps=24000)
    install_optimizer_capture(runner, env)
    return restored


def optimizer_summary(env, expected):
    state = env._microduck_optimizer_audit
    rate = check_optimizer(state.alg, expected)
    base.require(state.expected == expected and len(state.samples) == 20, "twenty audited optimizer minibatches")
    values = torch.stack([v for v, _ in state.samples]).cpu()
    rates = [v for _, v in state.samples]
    base.require(base.finite_tree(values) and all(math.isfinite(v) and 1e-5 <= v <= 1e-2 for v in rates),
                 "finite raw KL and LR telemetry")
    means = values[:, 0]
    base.require(bool(((values[:, 1] <= means) & (means <= values[:, 2])).all()), "ordered raw KL summaries")
    result = dict(desired_kl=expected, optimizer_minibatches=20,
        kl_mean_min=float(means.min()), kl_mean_max=float(means.max()), kl_mean_mean=float(means.mean()),
        kl_sample_min=float(values[:, 1].min()), kl_sample_max=float(values[:, 2].max()),
        kl_high_count=int((means > expected*2).sum()),
        kl_low_positive_count=int(((means > 0) & (means < expected/2)).sum()),
        lr_before_min=min(rates), lr_before_max=max(rates), lr_after_update=rate,
        minibatch_kl_means=means.tolist(), minibatch_lr_before=rates,
        minibatch_lr_applied_inferred=rates[1:]+[rate],
        lr_timing="before KL adaptation; applied LR inferred from next pre-LR or final post-update LR")
    state.samples.clear()
    return result


class MotorObserver(precision.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.expected_kl = target(mode)

    def before(self, env):
        super().before(env)
        state = env._microduck_optimizer_audit
        check_optimizer(state.alg, self.expected_kl)
        base.require(not state.samples, "previous optimizer samples consumed")

    def update(self, env):
        row = super().update(env)
        row.update(optimizer_summary(env, self.expected_kl))
        return row


evaluate_case, decision = precision.evaluate_case, precision.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
