"""Matched modest command-error shaping, retaining the learned motor cost."""

from __future__ import annotations

import sys

import torch

from mjlab_microduck import integrated_motor_refinement as motor

base = motor.base
PROTOCOL = "integrated-command-tracking-continuation-v1"
ARMS = ("control", "tracking")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 877), ("benchmark", 256, 10, 879))
         for arm in ARMS}
MODES.update(control=(256, 1000, 881), tracking=(256, 1000, 881))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/motor-refinement-4da92898-20261010/motor/model_999.pt"
PARENT_SHA = "ee6d0f69e34fd73c34f3b1f307a2d4bde85f73d1ed81a53cba07ca5899091fc1"


def tracking_weight(mode):
    return -.5 if mode.split("-")[-1] == "tracking" else 0.


def capture_tracking_metric(env):
    from mjlab_microduck.tasks.mdp import body_twist_tracking_cost
    # Metrics run after rewards but before automatic reset or command resampling.
    # Capture even in the control arm, whose zero-weight reward is skipped.
    env._microduck_tracking_cost = body_twist_tracking_cost(env).detach().clone()
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    cfg, agent = motor.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards["motor_load"].weight = -1.
    cfg.rewards["command_error"] = RewardTermCfg(func=mdp.body_twist_tracking_cost,
        weight=tracking_weight(mode), params={"command_name": "twist"})
    cfg.metrics["microduck_tracking_cost"] = MetricsTermCfg(func=capture_tracking_metric, per_substep=False)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    return motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                           parent_iteration=999, parent_steps=24000)


class MotorObserver(motor.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.schedule = ((0, -1.),)
        self.tracking_weight = tracking_weight(mode)
        self.tracking_costs, self.tracking_rewards, self.tracking_maxima = [], [], []

    def before(self, env):
        super().before(env)
        env.reward_manager.get_term_cfg("command_error").weight = self.tracking_weight
        env._microduck_tracking_cost = None

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        cost = env._microduck_tracking_cost
        base.require(isinstance(cost, torch.Tensor) and cost.shape == sample.mean_cost.shape
                     and base.finite_tree(cost) and bool((cost >= 0).all()), "finite pre-reset command cost")
        index = env.reward_manager.active_terms.index("command_error")
        weighted = env.reward_manager._step_reward[:, index]
        base.require(base.finite_tree(weighted) and bool((weighted <= 0).all())
                     and torch.allclose(weighted, cost*self.tracking_weight, atol=1e-6, rtol=1e-5),
                     "command reward matches raw pre-reset cost")
        self.tracking_costs.append(float(cost.mean()))
        self.tracking_rewards.append(float(weighted.mean()))
        self.tracking_maxima.append(float(cost.max()))
        env._microduck_tracking_cost = None

    def update(self, env):
        row = super().update(env)
        row.update(command_error_weight=self.tracking_weight,
                   command_cost_mean=sum(self.tracking_costs)/24,
                   command_cost_max=max(self.tracking_maxima),
                   weighted_command_reward_mean=sum(self.tracking_rewards)/24)
        self.tracking_costs.clear(); self.tracking_rewards.clear(); self.tracking_maxima.clear()
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
