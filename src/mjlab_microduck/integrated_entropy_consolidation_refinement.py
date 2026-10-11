"""Matched PPO entropy-pressure consolidation, with unchanged policy interfaces.

Zero entropy coefficient does not freeze learned action variance. Both arms
restore the original models, observation normalizers and Adam state.
"""

import math
import sys

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-entropy-consolidation-continuation-v1"
ARMS = ("control", "consolidate")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1123),
                                              ("benchmark", 256, 10, 1129))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1151), consolidate=(256, 1000, 1151))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {arm: CAP_SECONDS for arm in ARMS}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
EXPOSURE_BINS = previous.EXPOSURE_BINS


def entropy_coefficient(mode):
    base.require(type(mode) is str and mode in MODES, "known entropy-consolidation mode")
    return 0. if mode.split("-")[-1] == "consolidate" else .01


def prepare_config(mode):
    coefficient = entropy_coefficient(mode)
    cfg, agent = previous.prepare_config("control")
    base.require(agent.algorithm.entropy_coef == .01, "unchanged inherited entropy coefficient")
    agent.algorithm.entropy_coef = coefficient
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
evaluate_case = previous.evaluate_case
decision = previous.decision


def action_std(env, expected):
    """Read the actual scalar Gaussian parameter, without updating or sampling."""
    from rsl_rl.modules.distribution import GaussianDistribution

    alg = env._microduck_optimizer_audit.alg
    distribution = alg.actor.distribution
    base.require(type(alg.entropy_coef) is float and alg.entropy_coef == expected,
                 "unchanged live entropy coefficient")
    base.require(type(distribution) is GaussianDistribution and distribution.std_type == "scalar",
                 "reviewed direct scalar Gaussian action distribution")
    std = distribution.std_param
    base.require(isinstance(std, torch.nn.Parameter) and std.requires_grad and std.shape == (14,)
                 and std.is_floating_point() and base.finite_tree(std)
                 and bool((std > 0).all()), "finite strictly positive 14D action standard deviations")
    return std.detach().clone()


class MotorObserver(previous.MotorObserver):
    def __init__(self, mode):
        self.entropy_expected = entropy_coefficient(mode)
        super().__init__("control")
        self.entropy_before = None
        self.entropy_steps = 0
        self.entropy_action_names = None

    def before(self, env):
        from mjlab_microduck import action_response_capture

        super().before(env)
        _, term, _ = action_response_capture._bind_action_layout(
            env, env.scene["robot"], env._microduck_motor_step_stream)
        names = tuple(term.target_names)
        base.require(self.entropy_action_names in (None, names), "unchanged named action-std columns")
        self.entropy_action_names = names
        std = action_std(env, self.entropy_expected)
        if self.entropy_before is None:
            self.entropy_before = std
        else:
            base.require(torch.equal(std, self.entropy_before),
                         "action variance unchanged during the 24-step collection")
        self.entropy_steps += 1
        base.require(self.entropy_steps <= 24, "bounded action variance observations")

    def update(self, env):
        base.require(self.entropy_steps == 24 and self.entropy_before is not None,
                     "complete 24-step action variance observations")
        after = action_std(env, self.entropy_expected)
        row = super().update(env)
        before = self.entropy_before
        row.update(entropy_coefficient=self.entropy_expected,
                   action_std_protocol="direct-positive-scalar-gaussian-parameter-v1",
                   action_std_names=list(self.entropy_action_names),
                   action_std_before_update=before.cpu().tolist(),
                   action_std_after_update=after.cpu().tolist(),
                   action_distribution_entropy_before=float((before.log() + .5*math.log(2*math.pi*math.e)).sum()),
                   action_distribution_entropy_after=float((after.log() + .5*math.log(2*math.pi*math.e)).sum()))
        self.entropy_before = None
        self.entropy_steps = 0
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
