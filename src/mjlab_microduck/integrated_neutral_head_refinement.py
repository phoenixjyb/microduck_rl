"""Matched explicit HOME-head exposure, without freezing head motion or rewards.

The zero-command atom changes the sampler and its RNG consumption. Equal seeds
do not imply identical simulated trajectories. Original held-out gates apply.
"""

import sys
from collections.abc import Mapping

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-neutral-head-exposure-continuation-v1"
ARMS = ("control", "neutral")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1103),
                                              ("benchmark", 256, 10, 1109))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1117), neutral=(256, 1000, 1117))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {arm: CAP_SECONDS for arm in ARMS}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
EXPOSURE_BINS = previous.EXPOSURE_BINS
HEAD_RANGES = ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))
HEAD_RESAMPLING = (2., 5.)
ZERO_PROBABILITY = .25


def zero_probability(mode):
    base.require(type(mode) is str and mode in MODES, "known neutral-head mode")
    return ZERO_PROBABILITY if mode.split("-")[-1] == "neutral" else 0.


def prepare_config(mode):
    probability = zero_probability(mode)
    cfg, agent = previous.prepare_config("control")
    head = cfg.commands["head_pose"]
    base.require(tuple(head.ranges) == HEAD_RANGES
                 and head.resampling_time_range == HEAD_RESAMPLING
                 and head.zero_command_prob == 0., "unchanged inherited head sampler")
    head.zero_command_prob = probability
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
    """Count the actor-visible commands without recomputing noisy observations.

    Step exposure is measured immediately before each action, then consumed
    once after it regardless of termination/resampling. Segment counts use
    the public sampler counter plus episode reset, including zero-to-zero
    resamples. They count sampled segments, not Bernoulli draws or probabilities.
    """

    def __init__(self, mode):
        self.neutral_head_probability = zero_probability(mode)
        super().__init__("control")
        self.neutral_head_pending = None
        self.neutral_head_previous_counter = None
        self.neutral_head_counts = self._empty_counts()
        self.neutral_head_cumulative = self._empty_counts()
        self.neutral_head_steps = 0
        self.neutral_head_segments = dict(zero=0, nonzero=0)
        self.neutral_head_segments_cumulative = dict(zero=0, nonzero=0)

    @staticmethod
    def _empty_counts():
        return {f"{name}:{state}": 0 for name in EXPOSURE_BINS
                for state in ("zero", "nonzero")}

    def before(self, env):
        super().before(env)
        base.require(self.neutral_head_pending is None, "previous neutral-head exposure consumed")
        term = env.command_manager.get_term("head_pose")
        base.require(term.cfg.zero_command_prob == self.neutral_head_probability
                     and tuple(term.cfg.ranges) == HEAD_RANGES
                     and term.cfg.resampling_time_range == HEAD_RESAMPLING,
                     "unchanged declared head sampler")
        head = env.command_manager.get_command("head_pose")
        body = env.command_manager.get_command("body_pose")
        twist = env.command_manager.get_command("twist")
        buffer = getattr(env.observation_manager, "_obs_buffer", None)
        base.require(isinstance(buffer, Mapping), "existing raw observation cache")
        cached = buffer.get("actor")
        base.require(isinstance(cached, torch.Tensor) and cached.shape == (env.num_envs, 61)
                     and head.shape == (env.num_envs, 4) and body.shape == (env.num_envs, 6)
                     and base.finite_tree((cached, head, body))
                     and torch.equal(cached[:, 48:51], twist)
                     and torch.equal(cached[:, 51:55], head)
                     and torch.equal(cached[:, 55:61], body),
                     "cached raw actor sees unchanged twist/head/body command slots")
        zero = (head == 0).all(-1)
        counter = term.command_counter
        base.require(isinstance(counter, torch.Tensor) and counter.shape == (env.num_envs,)
                     and counter.dtype == torch.long and bool((counter > 0).all()),
                     "initialized public head command counters")
        new = (torch.ones_like(zero) if self.neutral_head_previous_counter is None else
               ((counter != self.neutral_head_previous_counter) | (env.episode_length_buf == 0)))
        self.neutral_head_previous_counter = counter.detach().clone()
        counts = self._empty_counts()
        for state, mask in (("zero", zero), ("nonzero", ~zero)):
            if bool(mask.any()):
                for name, count in previous.command_exposure(twist[mask]).items():
                    counts[f"{name}:{state}"] = count
        segments = dict(zero=int((new & zero).sum()), nonzero=int((new & ~zero).sum()))
        base.require(sum(counts.values()) == env.num_envs, "complete joint head/twist exposure")
        self.neutral_head_pending = (env.common_step_counter, counts, segments)

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        pending = self.neutral_head_pending
        base.require(pending is not None and env.common_step_counter == pending[0] + 1,
                     "one complete action step per neutral-head exposure")
        for name, count in pending[1].items():
            self.neutral_head_counts[name] += count
            self.neutral_head_cumulative[name] += count
        for name, count in pending[2].items():
            self.neutral_head_segments[name] += count
            self.neutral_head_segments_cumulative[name] += count
        self.neutral_head_steps += 1
        self.neutral_head_pending = None

    def update(self, env):
        row = super().update(env)
        base.require(self.neutral_head_pending is None and self.neutral_head_steps == 24
                     and sum(self.neutral_head_counts.values()) == 24 * env.num_envs
                     and sum(self.neutral_head_cumulative.values()) == env.common_step_counter * env.num_envs,
                     "complete actor-visible neutral-head dwell")
        row.update(neutral_head_zero_probability=self.neutral_head_probability,
                   neutral_head_exposure_protocol="cached-actor-head-twist-dwell-v1",
                   neutral_head_counts=dict(self.neutral_head_counts),
                   neutral_head_cumulative=dict(self.neutral_head_cumulative),
                   neutral_head_segments=dict(self.neutral_head_segments),
                   neutral_head_segments_cumulative=dict(self.neutral_head_segments_cumulative))
        self.neutral_head_counts = self._empty_counts()
        self.neutral_head_segments = dict(zero=0, nonzero=0)
        self.neutral_head_steps = 0
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
