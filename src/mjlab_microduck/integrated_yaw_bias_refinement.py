"""Matched persistent yaw-bias pressure; no action or observation filtering.

The EMA is training-reward state only. The held-out instantaneous body-yaw
gate is unchanged, and neither the inherited parent nor this experiment is
an accepted policy.
"""

import sys

import torch

from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor


base = previous.base
PROTOCOL = "integrated-body-yaw-bias-continuation-v1"
ARMS = ("control", "bias")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1031),
                                              ("benchmark", 256, 10, 1033))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1039), bias=(256, 1000, 1039))
CAP_SECONDS = motor.CAP_SECONDS
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
TERM = "yaw_bias"
TAU_S = 1.
BIAS_WEIGHT = -.1


def bias_weight(mode):
    base.require(mode in MODES, "known yaw-bias mode")
    return BIAS_WEIGHT if mode.split("-")[-1] == "bias" else 0.


def capture_bias_metric(env):
    """Read the reward-state cache before reset; control computes it once here."""
    from mjlab_microduck.tasks import mdp

    base.require(getattr(env, "_microduck_yaw_bias_snapshot", None) is None,
                 "previous yaw-bias snapshot consumed")
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(term.func is mdp.body_yaw_bias_cost
                 and term.params == dict(command_name="twist", tau_s=TAU_S),
                 "known yaw-bias metric/reward parameters")
    cost = term.func(env, **term.params)
    state = env._microduck_body_yaw_bias_state
    base.require(state["step"] == env.common_step_counter,
                 "same-step yaw-bias reward cache")
    env._microduck_yaw_bias_snapshot = {
        key: state[key].detach().clone()
        for key in ("signed_residual", "signed_ema", "cost")
    }
    base.require(torch.equal(cost, env._microduck_yaw_bias_snapshot["cost"]),
                 "captured yaw-bias cost identity")
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    base.require(mode in MODES, "known yaw-bias mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards[TERM] = RewardTermCfg(func=mdp.body_yaw_bias_cost,
        weight=bias_weight(mode), params=dict(command_name="twist", tau_s=TAU_S))
    cfg.metrics["microduck_yaw_bias"] = MetricsTermCfg(func=capture_bias_metric, per_substep=False)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load


class MotorObserver(previous.MotorObserver):
    def __init__(self, mode):
        base.require(mode in MODES, "known yaw-bias mode")
        # Keep the prior observer's control schedule: yaw/roll-pitch width .5.
        super().__init__("control")
        self.bias_weight = bias_weight(mode)
        self.bias_costs, self.bias_rewards, self.signed_biases, self.raw_errors = [], [], [], []

    def before(self, env):
        from mjlab_microduck.tasks import mdp

        super().before(env)
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is mdp.body_yaw_bias_cost and term.weight == self.bias_weight
                     and term.params == dict(command_name="twist", tau_s=TAU_S),
                     "unchanged declared yaw-bias reward")
        base.require(getattr(env, "_microduck_yaw_bias_snapshot", None) is None,
                     "previous yaw-bias snapshot consumed")

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = getattr(env, "_microduck_yaw_bias_snapshot", None)
        base.require(isinstance(snapshot, dict) and set(snapshot) == {"signed_residual", "signed_ema", "cost"}
                     and all(isinstance(v, torch.Tensor) and v.shape == sample.mean_cost.shape
                             for v in snapshot.values()) and base.finite_tree(snapshot),
                     "finite consumable pre-reset yaw-bias snapshot")
        cost = snapshot["cost"]
        base.require(bool((cost >= 0).all()) and torch.equal(cost, snapshot["signed_ema"].abs()),
                     "yaw-bias cost is absolute signed EMA")
        weighted = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(TERM)]
        base.require(base.finite_tree(weighted) and bool((weighted <= 0).all())
                     and torch.allclose(weighted, self.bias_weight * cost, atol=1e-6, rtol=1e-5),
                     "actual nonpositive yaw-bias reward matches pre-reset cost")
        self.bias_costs.append(float(cost.mean()))
        self.bias_rewards.append(float(weighted.mean()))
        self.signed_biases.append(float(snapshot["signed_ema"].mean()))
        self.raw_errors.append(float(snapshot["signed_residual"].abs().mean()))
        env._microduck_yaw_bias_snapshot = None

    def update(self, env):
        row = super().update(env)
        base.require(all(len(v) == 24 for v in
                         (self.bias_costs, self.bias_rewards, self.signed_biases, self.raw_errors)),
                     "full 24-step yaw-bias observation including terminal snapshots")
        row.update(yaw_bias_weight=self.bias_weight, yaw_bias_tau_s=TAU_S,
                   yaw_bias_cost_mean=sum(self.bias_costs)/24,
                   weighted_yaw_bias_reward_mean=sum(self.bias_rewards)/24,
                   signed_yaw_bias_mean=sum(self.signed_biases)/24,
                   raw_yaw_residual_mae=sum(self.raw_errors)/24)
        for values in (self.bias_costs, self.bias_rewards, self.signed_biases, self.raw_errors):
            values.clear()
        return row


evaluate_case = previous.evaluate_case
decision = previous.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
