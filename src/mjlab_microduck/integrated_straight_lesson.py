"""Matched straight/idle sampler diagnostic, with turning progression closed.

Only twist sampling differs. Rewards, stock delays, model and optimizer are
inherited unchanged. Passing this lesson is not full foundation acceptance.
"""

import sys

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-straight-lesson-continuation-v1"
ARMS = ("control", "straight")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1259),
                                              ("benchmark", 256, 10, 1277))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1279), straight=(256, 1000, 1279))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = dict(control=2100, straight=2100)
COMMAND_COVERAGE = dict(control=base.FULL_COMMAND_COVERAGE,
                        straight=base.STRAIGHT_COMMAND_COVERAGE)
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
warm_load, decision = previous.warm_load, previous.decision


def prepare_config(mode):
    base.require(mode in MODES, "declared straight lesson mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    twist = cfg.commands["twist"]
    base.require(twist.ranges.lin_vel_x == (.2, .4)
                 and twist.ranges.lin_vel_y == (0., 0.)
                 and twist.ranges.ang_vel_z == (-.5, .5)
                 and twist.rel_forward_envs == .25 and twist.rel_standing_envs == .15
                 and twist.rel_turn_in_place_envs == 0.
                 and twist.resampling_time_range == (3., 5.) and not cfg.curriculum,
                 "unchanged inherited sampler before declared lesson")
    if mode == "straight" or mode.endswith("-straight"):
        twist.ranges.lin_vel_x = (.3, .3)
        twist.ranges.ang_vel_z = (0., 0.)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


class MotorObserver(previous.MotorObserver):
    def __init__(self, mode):
        base.require(mode in MODES, "declared straight lesson observer mode")
        super().__init__("control")
        self.lesson_mode = mode

    def before(self, env):
        if self.lesson_mode == "straight" or self.lesson_mode.endswith("-straight"):
            command = env.command_manager.get_command("twist")
            idle = (command == 0.).all(-1)
            base.require(command.shape == (env.num_envs, 3) and base.finite_tree(command)
                         and bool((command[:, 1:] == 0.).all())
                         and bool((idle | torch.isclose(command[:, 0],
                             command.new_tensor(.3), atol=1e-7, rtol=0.)).all()),
                         "pure idle or .3m/s straight consumed commands")
            cfg = env.command_manager.get_term("twist").cfg
            base.require(cfg.ranges.lin_vel_x == (.3, .3)
                         and cfg.ranges.lin_vel_y == (0., 0.)
                         and cfg.ranges.ang_vel_z == (0., 0.)
                         and cfg.rel_standing_envs == .15 and cfg.rel_forward_envs == .25
                         and cfg.rel_turn_in_place_envs == 0.
                         and cfg.resampling_time_range == (3., 5.),
                         "fixed live straight lesson sampler")
        super().before(env)


def evaluate_case(checkpoint, seed, speed, yaw, health):
    # Descriptive gait capture only at the straight target; no observation,
    # command, RNG, delay or admission-predicate modification.
    return base.evaluate_case(checkpoint, seed, speed, yaw, health,
                              diagnostics=True, yaw_trace=speed == .3 and yaw == 0.,
                              gait_trace=speed == .3 and yaw == 0.)


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
