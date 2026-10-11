"""Matched instantaneous yaw-L1 pressure; asymmetric plant/safety unchanged.

This is a one-weight exploratory continuation, not a policy admission. Keep
the original instantaneous held-out gates and all non-yaw shaping unchanged.
"""

import sys

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-body-yaw-l1-continuation-v1"
ARMS = ("control", "yawl1")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1091),
                                              ("benchmark", 256, 10, 1093))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1097), yawl1=(256, 1000, 1097))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {arm: CAP_SECONDS for arm in ARMS}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
EXPOSURE_BINS = previous.EXPOSURE_BINS
TERM = "yaw_l1_error"
WEIGHT = -.5


def weight(mode):
    base.require(type(mode) is str and mode in MODES, "known yaw-L1 mode")
    return WEIGHT if mode.split("-")[-1] == "yawl1" else 0.


def capture_yaw_l1_metric(env):
    """Copy actual post-reward/pre-reset state once, including terminal worlds."""
    from mjlab_microduck.tasks import mdp

    base.require(getattr(env, "_microduck_yaw_l1_snapshot", None) is None,
                 "previous yaw-L1 snapshot consumed")
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(term.func is mdp.body_yaw_tracking_l1_cost
                 and term.params == dict(command_name="twist"),
                 "known yaw-L1 metric/reward parameters")
    cost = term.func(env, **term.params)
    signed = (env.scene["robot"].data.root_link_ang_vel_b[:, 2]
              - env.command_manager.get_command("twist")[:, 2])
    actual = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(TERM)]
    env._microduck_yaw_l1_snapshot = (cost.detach().clone(), signed.detach().clone(),
                                    actual.detach().clone(), env.common_step_counter)
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    expected_weight = weight(mode)
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    base.require(agent.algorithm.symmetry_cfg is None,
                 "asymmetric recipe must not enable mirror loss or augmentation")
    cfg.rewards[TERM] = RewardTermCfg(func=mdp.body_yaw_tracking_l1_cost,
        weight=expected_weight, params=dict(command_name="twist"))
    cfg.metrics["microduck_yaw_l1"] = MetricsTermCfg(func=capture_yaw_l1_metric,
                                                   per_substep=False)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
evaluate_case = previous.evaluate_case
decision = previous.decision


class MotorObserver(previous.MotorObserver):
    def __init__(self, mode):
        self.yaw_l1_weight = weight(mode)
        super().__init__("control")
        self.yaw_l1_costs, self.yaw_l1_rewards, self.yaw_l1_signed_errors = [], [], []

    def before(self, env):
        from mjlab_microduck.tasks import mdp

        super().before(env)
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is mdp.body_yaw_tracking_l1_cost
                     and term.weight == self.yaw_l1_weight
                     and term.params == dict(command_name="twist"),
                     "unchanged declared yaw-L1 reward")
        base.require(getattr(env, "_microduck_yaw_l1_snapshot", None) is None,
                     "previous yaw-L1 snapshot consumed")

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = getattr(env, "_microduck_yaw_l1_snapshot", None)
        base.require(isinstance(snapshot, tuple) and len(snapshot) == 4
                     and type(snapshot[3]) is int and snapshot[3] == env.common_step_counter,
                     "same-step pre-reset yaw-L1 snapshot")
        cost, signed, actual, _ = snapshot
        base.require(all(isinstance(v, torch.Tensor) and v.shape == sample.mean_cost.shape
                         for v in (cost, signed, actual)) and base.finite_tree(snapshot)
                     and bool((cost >= 0).all()) and torch.equal(cost, signed.abs()),
                     "finite instantaneous absolute yaw residual")
        base.require(bool((actual <= 0).all())
                     and torch.allclose(actual, self.yaw_l1_weight * cost, atol=1e-6, rtol=1e-5),
                     "actual nonpositive yaw-L1 reward matches pre-reset cost")
        self.yaw_l1_costs.append(float(cost.mean()))
        self.yaw_l1_rewards.append(float(actual.mean()))
        self.yaw_l1_signed_errors.append(float(signed.mean()))
        env._microduck_yaw_l1_snapshot = None

    def update(self, env):
        row = super().update(env)
        base.require(all(len(v) == 24 for v in
                         (self.yaw_l1_costs, self.yaw_l1_rewards, self.yaw_l1_signed_errors)),
                     "full 24-step yaw-L1 observation including terminal snapshots")
        row.update(yaw_l1_weight=self.yaw_l1_weight,
                   yaw_l1_cost_mean=sum(self.yaw_l1_costs)/24,
                   weighted_yaw_l1_reward_mean=sum(self.yaw_l1_rewards)/24,
                   yaw_l1_signed_residual_mean=sum(self.yaw_l1_signed_errors)/24)
        for values in (self.yaw_l1_costs, self.yaw_l1_rewards, self.yaw_l1_signed_errors):
            values.clear()
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
