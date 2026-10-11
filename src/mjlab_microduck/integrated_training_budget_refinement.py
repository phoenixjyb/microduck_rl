"""Matched 1000/4000 additional updates of the unchanged decoupled recipe.

Duration, not reward/sampler/plant, is the sole treatment. Command-dwell
telemetry is identical in both arms and never modifies commands or RNG.
"""

import sys

import torch

from mjlab_microduck import integrated_yaw_bias_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-training-duration-continuation-v1"
ARMS = ("control", "budget")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1049),
                                              ("benchmark", 256, 10, 1051))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1061), budget=(256, 4000, 1061))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {"control": 2100, "budget": 7200}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
EXPOSURE_BINS = ("idle", "straight", "negative-low", "negative-target", "negative-high",
                 "positive-low", "positive-target", "positive-high")


def command_exposure(command):
    """Exclusive action-step counts; target band is (.15,.25] rad/s magnitude."""
    base.require(isinstance(command, torch.Tensor) and command.ndim == 2
                 and command.shape[0] > 0 and command.shape[1] == 3
                 and command.is_floating_point() and base.finite_tree(command),
                 "finite nonempty Nx3 command exposure")
    vx, vy, wz = command.unbind(-1)
    idle = (command == 0).all(-1)
    base.require(bool((vy == 0).all()) and bool((wz.abs() <= .5).all())
                 and bool((idle | ((vx >= .2) & (vx <= .4))).all()),
                 "unchanged effective twist support")
    magnitude = wz.abs()
    masks = [idle, (wz == 0) & ~idle]
    for sign in (wz < 0, wz > 0):
        masks.extend((sign & (magnitude <= .15),
                      sign & (magnitude > .15) & (magnitude <= .25),
                      sign & (magnitude > .25)))
    counts = dict(zip(EXPOSURE_BINS, (int(mask.sum()) for mask in masks), strict=True))
    base.require(sum(counts.values()) == command.shape[0], "exclusive complete command bins")
    return counts


def prepare_config(mode):
    base.require(mode in MODES, "known training-duration mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    twist = cfg.commands["twist"]
    base.require(twist.ranges.lin_vel_x == (.2, .4)
                 and twist.ranges.lin_vel_y == (0., 0.)
                 and twist.ranges.ang_vel_z == (-.5, .5)
                 and twist.resampling_time_range == (3., 5.)
                 and twist.rel_standing_envs == .15 and twist.rel_forward_envs == .25
                 and twist.rel_turn_in_place_envs == 0. and not cfg.curriculum,
                 "fixed inherited sampler and collapsed curricula")
    base.require(cfg.rewards[previous.TERM].weight == 0., "no yaw-bias treatment")
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load


class MotorObserver(previous.MotorObserver):
    def __init__(self, mode):
        base.require(mode in MODES, "known training-duration mode")
        super().__init__("control")
        self.pending_exposure = None
        self.exposure_steps = 0
        self.exposure_counts = dict.fromkeys(EXPOSURE_BINS, 0)
        self.cumulative_exposure = dict.fromkeys(EXPOSURE_BINS, 0)

    def before(self, env):
        super().before(env)
        base.require(self.pending_exposure is None, "previous command exposure consumed")
        command = env.command_manager.get_command("twist")
        base.require(command.shape == (env.num_envs, 3), "command exposure world layout")
        self.pending_exposure = (env.common_step_counter, command_exposure(command))

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        base.require(self.pending_exposure is not None
                     and env.common_step_counter == self.pending_exposure[0] + 1,
                     "one complete action step per command exposure")
        for key, count in self.pending_exposure[1].items():
            self.exposure_counts[key] += count
            self.cumulative_exposure[key] += count
        self.exposure_steps += 1
        self.pending_exposure = None

    def update(self, env):
        row = super().update(env)
        base.require(self.pending_exposure is None and self.exposure_steps == 24
                     and sum(self.exposure_counts.values()) == 24 * env.num_envs
                     and sum(self.cumulative_exposure.values()) == env.common_step_counter * env.num_envs,
                     "complete per-update and cumulative command dwell")
        row.update(command_exposure_protocol="actor-command-dwell-v1",
                   command_exposure_counts=dict(self.exposure_counts),
                   command_exposure_cumulative=dict(self.cumulative_exposure))
        self.exposure_steps = 0
        self.exposure_counts = dict.fromkeys(EXPOSURE_BINS, 0)
        return row


evaluate_case = previous.evaluate_case
decision = previous.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
