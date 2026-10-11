"""Matched yaw-width continuation; keep roll/pitch kernel and all gates fixed."""

from collections import Counter
import math
import sys

import torch

from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_motor_refinement as motor

base = forward.base
PROTOCOL = "integrated-independent-yaw-precision-v1"
ARMS = ("control", "precision")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 971), ("benchmark", 256, 10, 973))
         for arm in ARMS}
MODES.update(control=(256, 1000, 977), precision=(256, 1000, 977))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/turn-range-2b75ab02-20261011/control/model_999.pt"
PARENT_SHA = "11c20ccc4d0682ab1403013dd5ca3308913ea2fd61a86e1c5bc3fb5408c2b8c2"
YAW_STAGES = ((0, .5), (250, .25), (500, .1))
TERM = "track_angular_velocity"


def stages(mode):
    if mode.endswith("control"):
        return ((0, .5),)
    if mode.startswith("smoke"):
        return ((0, .5), (1, .25), (2, .1))
    if mode.startswith("benchmark"):
        return ((0, .1),)
    return YAW_STAGES


def capture_angular_metric(env):
    from mjlab_microduck.tasks.mdp import body_angular_tracking_errors
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(getattr(env, "_microduck_precision_snapshot", None) is None, "previous angular snapshot consumed")
    yaw_sq, xy_sq = body_angular_tracking_errors(env, term.params["command_name"])
    actual = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(TERM)]
    env._microduck_precision_snapshot = (yaw_sq.detach().clone(), xy_sq.detach().clone(),
        actual.detach().clone(), term.params["yaw_variance"])
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp
    cfg, agent = forward.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    original = cfg.rewards[TERM]
    base.require(original.weight == 2. and original.params == dict(std=math.sqrt(.5), command_name="twist"),
                 "known original angular reward config")
    cfg.rewards[TERM] = RewardTermCfg(func=mdp.factorized_body_angular_tracking, weight=2.,
        params=dict(command_name="twist", yaw_variance=stages(mode)[0][1], roll_pitch_variance=.5))
    cfg.metrics["microduck_precision_reward"] = MetricsTermCfg(func=capture_angular_metric, per_substep=False)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    return motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                           parent_iteration=999, parent_steps=24000)


class MotorObserver(forward.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.yaw_schedule = stages(mode)
        self.previous_variance = self.yaw_schedule[0][1]
        self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards = [], [], [], []
        self.cumulative_counts = Counter()

    def before(self, env):
        super().before(env)
        from mjlab_microduck.tasks.mdp import factorized_body_angular_tracking
        term = env.reward_manager.get_term_cfg(TERM)
        variance = motor.motor_weight(env.common_step_counter, self.yaw_schedule)
        base.require(term.func is factorized_body_angular_tracking and term.weight == 2.
            and term.params == dict(command_name="twist", yaw_variance=self.previous_variance, roll_pitch_variance=.5)
            and .1 <= variance <= self.previous_variance <= .5, "known monotonic live yaw precision")
        base.require(getattr(env, "_microduck_precision_snapshot", None) is None, "angular snapshot consumed before action")
        term.params["yaw_variance"] = variance
        self.previous_variance = variance

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = getattr(env, "_microduck_precision_snapshot", None)
        base.require(isinstance(snapshot, tuple) and len(snapshot) == 4, "pre-reset angular snapshot")
        yaw_sq, xy_sq, actual, variance = snapshot
        base.require(all(isinstance(v, torch.Tensor) and v.shape == sample.mean_cost.shape for v in (yaw_sq, xy_sq, actual))
            and base.finite_tree(snapshot) and variance == self.previous_variance
            and bool((yaw_sq >= 0).all() & (xy_sq >= 0).all()), "finite raw angular components")
        expected = 2.*torch.exp(-yaw_sq/variance-xy_sq/.5)
        base.require(bool(((actual >= 0) & (actual <= 2.)).all())
            and torch.allclose(actual, expected, atol=1e-6, rtol=1e-5), "actual positive pre-reset factorized reward")
        self.variances.append(variance); self.cumulative_counts[str(variance)] += 1
        self.yaw_costs.append(float(yaw_sq.mean())); self.xy_costs.append(float(xy_sq.mean()))
        self.angular_rewards.append(float(actual.mean()))
        env._microduck_precision_snapshot = None

    def update(self, env):
        row = super().update(env)
        base.require(all(len(v) == 24 for v in (self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards)),
                     "full angular precision update")
        row.update(yaw_variance_min=min(self.variances), yaw_variance_max=max(self.variances),
            roll_pitch_variance=.5, angular_tracking_weight=2., yaw_error_sq_mean=sum(self.yaw_costs)/24,
            roll_pitch_rate_sq_mean=sum(self.xy_costs)/24, weighted_angular_reward_mean=sum(self.angular_rewards)/24,
            yaw_variance_step_counts=dict(Counter(map(str, self.variances))),
            yaw_variance_cumulative_step_counts=dict(self.cumulative_counts))
        for values in (self.variances, self.yaw_costs, self.xy_costs, self.angular_rewards): values.clear()
        return row


evaluate_case = forward.evaluate_case
decision = forward.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
