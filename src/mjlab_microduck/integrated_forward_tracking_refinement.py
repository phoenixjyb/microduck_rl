"""Matched forward-error escalation pilot; retain range/motor/turn shaping.

The parent is not an accepted policy. This tests one schedule, not three
independent doses, and does not authorize obstacles or physical motion.
"""

from __future__ import annotations

from collections import Counter
import sys

import torch

from mjlab_microduck import integrated_safe_tracking_refinement as safe
from mjlab_microduck import integrated_motor_refinement as motor

base = safe.base
PROTOCOL = "integrated-forward-retaining-turn-continuation-v1"
ARMS = ("control", "forward")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 929), ("benchmark", 256, 10, 931))
         for arm in ARMS}
MODES.update(control=(256, 1000, 937), forward=(256, 1000, 937))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/safe-tracking-be487638-20261011/tracking/model_999.pt"
PARENT_SHA = "503fb02f0bbddbd3fc886453b20a2c878a91a8c1c985eeddfeec768020c53b39"
FORWARD_STAGES = ((0, -.25), (250, -.75), (500, -1.5))


def stages(mode):
    if mode.endswith("control"):
        return ((0, 0.),)
    if mode.startswith("smoke"):
        return ((0, -.25), (1, -.75), (2, -1.5))
    if mode.startswith("benchmark"):
        return ((0, -1.5),)
    return FORWARD_STAGES


def capture_forward_metric(env):
    from mjlab_microduck.tasks.mdp import body_forward_tracking_cost
    term = env.reward_manager.get_term_cfg("forward_error")
    # Capture in the same post-reward/pre-reset hook, including control's
    # skipped zero-weight term and worlds that terminate this step.
    env._microduck_forward_cost = body_forward_tracking_cost(env, **term.params).detach().clone()
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    cfg, agent = safe.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards["command_error"].weight = -.5
    cfg.rewards["forward_error"] = RewardTermCfg(func=mdp.body_forward_tracking_cost,
        weight=stages(mode)[0][1], params={"command_name": "twist"})
    cfg.metrics["microduck_forward_cost"] = MetricsTermCfg(func=capture_forward_metric, per_substep=False)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    return motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                           parent_iteration=999, parent_steps=24000)


class MotorObserver(safe.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.tracking_schedule = ((0, -.5),)
        self.forward_schedule = stages(mode)
        self.forward_weights, self.forward_costs, self.forward_rewards = [], [], []

    def before(self, env):
        super().before(env)
        self.forward_weight = motor.motor_weight(env.common_step_counter, self.forward_schedule)
        env.reward_manager.get_term_cfg("forward_error").weight = self.forward_weight
        env._microduck_forward_cost = None

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        cost = env._microduck_forward_cost
        base.require(isinstance(cost, torch.Tensor) and cost.shape == sample.mean_cost.shape
                     and base.finite_tree(cost) and bool((cost >= 0).all()), "finite pre-reset forward cost")
        index = env.reward_manager.active_terms.index("forward_error")
        weighted = env.reward_manager._step_reward[:, index]
        base.require(base.finite_tree(weighted) and bool((weighted <= 0).all())
                     and torch.allclose(weighted, cost*self.forward_weight, atol=1e-6, rtol=1e-5),
                     "forward reward matches pre-reset cost")
        self.forward_weights.append(self.forward_weight)
        self.forward_costs.append(float(cost.mean()))
        self.forward_rewards.append(float(weighted.mean()))
        env._microduck_forward_cost = None

    def update(self, env):
        row = super().update(env)
        row.update(forward_weight_min=min(self.forward_weights), forward_weight_max=max(self.forward_weights),
                   forward_cost_mean=sum(self.forward_costs)/24,
                   weighted_forward_reward_mean=sum(self.forward_rewards)/24,
                   forward_weight_step_counts=dict(Counter(map(str, self.forward_weights))))
        self.forward_weights.clear(); self.forward_costs.clear(); self.forward_rewards.clear()
        return row


evaluate_case = safe.evaluate_case
decision = safe.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
