"""Matched yaw-precision continuation of the signed decoupled angular reward."""

from collections import Counter
import sys

import torch

from mjlab_microduck import integrated_angular_decoupling_refinement as decoupling
from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_yaw_precision_refinement as precision


base = decoupling.base
PROTOCOL = "integrated-decoupled-yaw-precision-v1"
ARMS = ("control", "precision")
MODES = {
    f"{phase}-{arm}": (worlds, updates, seed)
    for phase, worlds, updates, seed in (
        ("smoke", 64, 5, 1021),
        ("benchmark", 256, 10, 1023),
    )
    for arm in ARMS
}
MODES.update(control=(256, 1000, 1027), precision=(256, 1000, 1027))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/angular-decoupling-ea24a14a-20261011/decoupled/model_999.pt"
PARENT_SHA = "c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae"
PARENT_LEARNING_RATE = 5.062500000000001e-05
TERM = precision.TERM
VARIANCE = .5
ROLL_PITCH_VARIANCE = .5
WEIGHT = 2.
YAW_STAGES = ((0, .5), (250, .25), (500, .1))
AGGREGATION = "additive_decoupled"


def stages(mode):
    base.require(mode in MODES, "known decoupled yaw precision mode")
    if mode.endswith("control"):
        return ((0, VARIANCE),)
    if mode.startswith("smoke"):
        return ((0, .5), (1, .25), (2, .1))
    if mode.startswith("benchmark"):
        return ((0, .1),)
    return YAW_STAGES


def prepare_config(mode):
    from mjlab_microduck.tasks import mdp

    base.require(mode in MODES, "known decoupled yaw precision mode")
    # Start both arms from the same signed, additive angular reward. Only the
    # precision arm's yaw variance schedule changes from the .5 control value.
    cfg, agent = decoupling.prepare_config("decoupled")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    term = cfg.rewards[TERM]
    base.require(term.func is mdp.decoupled_body_angular_tracking
                 and term.weight == WEIGHT
                 and term.params == dict(command_name="twist", yaw_variance=VARIANCE,
                                         roll_pitch_variance=ROLL_PITCH_VARIANCE),
                 "known signed decoupled angular reward")
    base.require(agent.algorithm.desired_kl == .01 and agent.algorithm.schedule == "adaptive"
                 and agent.algorithm.num_mini_batches == 4
                 and agent.algorithm.num_learning_epochs == 5,
                 "known matched adaptive-KL learner")
    term.params["yaw_variance"] = stages(mode)[0][1]
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    restored = motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                               parent_iteration=999, parent_steps=24000)
    base.require(restored["learning_rate"] == PARENT_LEARNING_RATE,
                 "exact inherited parent PPO learning rate; do not reset or rebudget")
    kl.install_optimizer_capture(runner, env, initial_lr=PARENT_LEARNING_RATE)
    return restored


class MotorObserver(forward.MotorObserver):
    """Observe scheduled reward and KL telemetry without changing PPO behavior."""

    def __init__(self, mode):
        base.require(mode in MODES, "known decoupled yaw precision mode")
        super().__init__("control")
        self.mode = mode
        self.yaw_schedule = stages(mode)
        self.previous_variance = self.yaw_schedule[0][1]
        self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards = [], [], [], []
        self.cumulative_counts = Counter()
        self.negative_samples = 0
        self.total_samples = 0

    def before(self, env):
        super().before(env)
        from mjlab_microduck.tasks import mdp

        variance = motor.motor_weight(env.common_step_counter, self.yaw_schedule)
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is mdp.decoupled_body_angular_tracking and term.weight == WEIGHT
                     and term.params == dict(command_name="twist", yaw_variance=self.previous_variance,
                                             roll_pitch_variance=ROLL_PITCH_VARIANCE)
                     and self.yaw_schedule == stages(self.mode)
                     and .1 <= variance <= self.previous_variance <= VARIANCE,
                     "known monotone live decoupled yaw precision")
        base.require(getattr(env, "_microduck_precision_snapshot", None) is None,
                     "previous angular snapshot consumed before action")
        state = getattr(env, "_microduck_optimizer_audit", None)
        base.require(state is not None, "installed adaptive-KL capture")
        kl.check_optimizer(state.alg, .01)
        base.require(not state.samples, "previous optimizer minibatches consumed")
        term.params["yaw_variance"] = variance
        self.previous_variance = variance

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = getattr(env, "_microduck_precision_snapshot", None)
        base.require(isinstance(snapshot, tuple) and len(snapshot) == 4,
                     "pre-reset angular reward snapshot")
        yaw_sq, xy_sq, actual, variance = snapshot
        base.require(all(isinstance(value, torch.Tensor) and value.shape == sample.mean_cost.shape
                         for value in (yaw_sq, xy_sq, actual))
                     and base.finite_tree(snapshot) and variance == self.previous_variance
                     and bool((yaw_sq >= 0).all() & (xy_sq >= 0).all()),
                     "finite captured angular state and live variance")
        expected = WEIGHT * (torch.exp(-yaw_sq / variance)
                             + torch.exp(-xy_sq / ROLL_PITCH_VARIANCE) - 1.)
        base.require(bool(((actual >= -WEIGHT) & (actual <= WEIGHT)).all())
                     and torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
                     "actual signed pre-reset reward matches decoupled aggregation")
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
                     and self.total_samples == 24 * env.num_envs,
                     "complete 24-step angular observation including terminal pre-reset samples")
        row.update(
            angular_aggregation=AGGREGATION,
            yaw_variance_min=min(self.variances),
            yaw_variance_max=max(self.variances),
            roll_pitch_variance=ROLL_PITCH_VARIANCE,
            angular_tracking_weight=WEIGHT,
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


evaluate_case = decoupling.evaluate_case
decision = decoupling.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
