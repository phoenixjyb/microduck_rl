"""Matched angular reward-form experiment with decoupled error-group kernels."""

from collections import Counter
import sys

import torch

from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_yaw_precision_refinement as precision

base = precision.base
PROTOCOL = "integrated-angular-decoupling-v1"
ARMS = ("control", "decoupled")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1013),
                                              ("benchmark", 256, 10, 1017))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1019), decoupled=(256, 1000, 1019))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/kl-budget-a9a70f0e-20261011/control/model_999.pt"
PARENT_SHA = "3ee173730a78fae605e561a3d40024b8e66d3b0a0be09262a6322fa819fe3df4"
TERM = precision.TERM
VARIANCE = .5
WEIGHT = 2.
AGGREGATIONS = {"control": "multiplicative_product", "decoupled": "additive_decoupled"}


def prepare_config(mode):
    from mjlab_microduck.tasks import mdp

    base.require(mode in MODES, "known angular decoupling mode")
    cfg, agent = precision.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    term = cfg.rewards[TERM]
    base.require(term.func is mdp.factorized_body_angular_tracking and term.weight == WEIGHT
                 and term.params == dict(command_name="twist", yaw_variance=VARIANCE,
                                         roll_pitch_variance=VARIANCE),
                 "known product-form angular control")
    base.require(agent.algorithm.desired_kl == .01 and agent.algorithm.schedule == "adaptive",
                 "known matched adaptive PPO configuration")
    if mode.endswith("decoupled"):
        term.func = mdp.decoupled_body_angular_tracking
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    restored = motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                               parent_iteration=999, parent_steps=24000)
    base.require(restored["learning_rate"] == 1e-5, "exact inherited initial PPO learning rate")
    kl.install_optimizer_capture(runner, env)
    return restored


class MotorObserver(forward.MotorObserver):
    def __init__(self, mode):
        base.require(mode in MODES, "known angular decoupling mode")
        super().__init__("control")
        from mjlab_microduck.tasks import mdp

        self.mode = mode
        self.expected_func = (mdp.factorized_body_angular_tracking if mode.endswith("control")
                              else mdp.decoupled_body_angular_tracking)
        self.aggregation = AGGREGATIONS["control" if mode.endswith("control") else "decoupled"]
        self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards = [], [], [], []
        self.negative_samples = 0
        self.total_samples = 0
        self.cumulative_counts = Counter()

    def before(self, env):
        super().before(env)
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is self.expected_func and term.weight == WEIGHT
                     and term.params == dict(command_name="twist", yaw_variance=VARIANCE,
                                             roll_pitch_variance=VARIANCE),
                     "fixed live angular reward configuration")
        base.require(getattr(env, "_microduck_precision_snapshot", None) is None,
                     "previous angular snapshot consumed before action")
        state = getattr(env, "_microduck_optimizer_audit", None)
        base.require(state is not None, "installed adaptive-KL capture")
        kl.check_optimizer(state.alg, .01)
        base.require(not state.samples, "previous optimizer minibatches consumed")

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = getattr(env, "_microduck_precision_snapshot", None)
        base.require(isinstance(snapshot, tuple) and len(snapshot) == 4,
                     "pre-reset angular reward snapshot")
        yaw_sq, xy_sq, actual, variance = snapshot
        base.require(all(isinstance(value, torch.Tensor) and value.shape == sample.mean_cost.shape
                         for value in (yaw_sq, xy_sq, actual))
                     and base.finite_tree(snapshot) and variance == VARIANCE
                     and bool((yaw_sq >= 0).all() & (xy_sq >= 0).all()),
                     "finite angular errors and captured reward")
        if self.mode.endswith("control"):
            expected = WEIGHT * torch.exp(-yaw_sq / VARIANCE - xy_sq / VARIANCE)
            base.require(bool(((actual >= 0.) & (actual <= 2.)).all()),
                         "control angular reward remains within [0,2]")
        else:
            expected = WEIGHT * (torch.exp(-yaw_sq / VARIANCE)
                                 + torch.exp(-xy_sq / VARIANCE) - 1.)
            base.require(bool(((actual >= -2.) & (actual <= 2.)).all()),
                         "decoupled angular reward remains within [-2,2]")
        base.require(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
                     "actual weighted angular reward matches live aggregation")
        self.variances.append(variance)
        self.cumulative_counts[str(variance)] += 1
        self.yaw_costs.append(float(yaw_sq.mean()))
        self.xy_costs.append(float(xy_sq.mean()))
        self.angular_rewards.append(float(actual.mean()))
        self.negative_samples += int((actual < 0.).sum())
        self.total_samples += actual.numel()
        env._microduck_precision_snapshot = None

    def update(self, env):
        row = super().update(env)
        base.require(all(len(values) == 24 for values in
                         (self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards))
                     and self.total_samples > 0,
                     "full 24-step angular aggregation observation")
        row.update(
            angular_aggregation=self.aggregation,
            yaw_variance_min=min(self.variances), yaw_variance_max=max(self.variances),
            roll_pitch_variance=VARIANCE, angular_tracking_weight=WEIGHT,
            yaw_error_sq_mean=sum(self.yaw_costs) / 24,
            roll_pitch_rate_sq_mean=sum(self.xy_costs) / 24,
            weighted_angular_reward_mean=sum(self.angular_rewards) / 24,
            yaw_variance_step_counts=dict(Counter(map(str, self.variances))),
            yaw_variance_cumulative_step_counts=dict(self.cumulative_counts),
            negative_reward_sample_count=self.negative_samples,
            angular_reward_sample_count=self.total_samples,
            negative_reward_fraction=self.negative_samples / self.total_samples,
        )
        state = env._microduck_optimizer_audit
        kl.check_optimizer(state.alg, .01)
        row.update(kl.optimizer_summary(env, .01))
        for values in (self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards):
            values.clear()
        self.negative_samples = self.total_samples = 0
        return row


evaluate_case = precision.evaluate_case
decision = precision.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
