"""Matched low-yaw rehearsal; unchanged objectives and held-out acceptance.

Only the training yaw support narrows. Wider-turn capability is not admitted
by this probe, nor are either arm's weights promoted automatically.
"""

import sys
from collections.abc import Mapping

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-low-yaw-rehearsal-continuation-v1"
ARMS = ("control", "rehearsal")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1163),
                                              ("benchmark", 256, 10, 1171))
         for arm in ARMS}
MODES.update(control=(256, 500, 1181), rehearsal=(256, 500, 1181))
CAP_SECONDS = 900
TRAINING_CAP_SECONDS = {arm: CAP_SECONDS for arm in ARMS}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
EXPOSURE_BINS = previous.EXPOSURE_BINS


def yaw_support(mode):
    base.require(type(mode) is str and mode in MODES, "known low-yaw rehearsal mode")
    return (-.25, .25) if mode.split("-")[-1] == "rehearsal" else (-.5, .5)


def prepare_config(mode):
    support = yaw_support(mode)
    cfg, agent = previous.prepare_config("control")
    cfg.commands["twist"].ranges.ang_vel_z = support
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
evaluate_case = previous.evaluate_case
decision = previous.decision


class MotorObserver(previous.MotorObserver):
    """Validate actual cached commands; retain identical exclusive dwell bins."""

    def __init__(self, mode):
        self.yaw_support_expected = yaw_support(mode)
        super().__init__("control")

    def before(self, env):
        base.require(env._microduck_optimizer_audit.alg.entropy_coef == .01,
                     "unchanged live PPO entropy coefficient")
        term = env.command_manager.get_term("twist")
        base.require(term.cfg.ranges.ang_vel_z == self.yaw_support_expected
                     and term.cfg.ranges.lin_vel_x == (.2, .4)
                     and term.cfg.ranges.lin_vel_y == (0., 0.)
                     and term.cfg.resampling_time_range == (3., 5.)
                     and term.cfg.rel_standing_envs == .15
                     and term.cfg.rel_forward_envs == .25
                     and term.cfg.rel_turn_in_place_envs == 0.,
                     "unchanged declared narrow-yaw sampler")
        command = env.command_manager.get_command("twist")
        buffer = getattr(env.observation_manager, "_obs_buffer", None)
        cached = buffer.get("actor") if isinstance(buffer, Mapping) else None
        base.require(isinstance(cached, torch.Tensor) and cached.shape == (env.num_envs, 61)
                     and command.shape == (env.num_envs, 3)
                     and base.finite_tree(cached)
                     and torch.equal(cached[:, 48:51], command)
                     and bool((command[:, 2] >= self.yaw_support_expected[0]).all())
                     and bool((command[:, 2] <= self.yaw_support_expected[1]).all()),
                     "raw cached actor command within declared yaw support")
        super().before(env)

    def update(self, env):
        row = super().update(env)
        if self.yaw_support_expected == (-.25, .25):
            base.require(row["command_exposure_counts"]["negative-high"] == 0
                         and row["command_exposure_counts"]["positive-high"] == 0,
                         "no high-yaw exposure in narrow rehearsal")
        row.update(training_yaw_support=list(self.yaw_support_expected))
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
