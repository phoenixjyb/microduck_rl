"""Matched roll/pitch-width continuation of the signed angular reward."""

import math
import sys

from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as angular
from mjlab_microduck import integrated_straight_lesson as straight
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp


base = previous.base
PROTOCOL = "integrated-roll-pitch-allowance-continuation-v1"
ARMS = ("control", "allowance")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1283),
                                             ("benchmark", 256, 10, 1289))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1291), allowance=(256, 1000, 1291))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = dict(control=2100, allowance=2100)
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
TERM = angular.TERM
CONTROL_VARIANCE = .5
ALLOWANCE_VARIANCE = 1.
WEIGHT = angular.WEIGHT
AGGREGATION = angular.AGGREGATION
warm_load = previous.warm_load
decision = previous.decision


def expected_roll_pitch_variance(mode):
    base.require(mode in MODES, "known roll/pitch allowance mode")
    return ALLOWANCE_VARIANCE if mode.split("-")[-1] == "allowance" else CONTROL_VARIANCE


def prepare_config(mode):
    base.require(mode in MODES, "known roll/pitch allowance mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    term = cfg.rewards[TERM]
    base.require(term.func is mdp.decoupled_body_angular_tracking
                 and term.weight == WEIGHT
                 and term.params == dict(command_name="twist", yaw_variance=.5,
                                         roll_pitch_variance=CONTROL_VARIANCE),
                 "known signed additive decoupled angular reward")
    base.require(agent.algorithm.desired_kl == .01 and agent.algorithm.schedule == "adaptive"
                 and agent.algorithm.num_mini_batches == 4
                 and agent.algorithm.num_learning_epochs == 5,
                 "known matched adaptive-KL learner")
    if mode.endswith("allowance"):
        term.params["roll_pitch_variance"] = ALLOWANCE_VARIANCE
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


class MotorObserver(previous.MotorObserver):
    """Measure the declared roll/pitch kernel while inheriting all reward/KL checks."""

    def __init__(self, mode):
        base.require(mode in MODES, "known roll/pitch allowance observer mode")
        super().__init__("control")
        self.allowance_mode = mode
        self.roll_pitch_variance = expected_roll_pitch_variance(mode)

    def _check_allowance(self):
        base.require(self.allowance_mode in MODES
                     and type(self.roll_pitch_variance) is float
                     and math.isfinite(self.roll_pitch_variance)
                     and self.roll_pitch_variance == expected_roll_pitch_variance(self.allowance_mode),
                     "unchanged declared roll/pitch allowance")

    def before(self, env):
        self._check_allowance()
        super().before(env)

    def step(self, env, sample, extras):
        self._check_allowance()
        super().step(env, sample, extras)

    def update(self, env):
        self._check_allowance()
        return super().update(env)


# Held-out case acceptance stays the training-budget contract; gait telemetry
# is the read-only straight-target wrapper used by the completed lesson.
evaluate_case = straight.evaluate_case


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
