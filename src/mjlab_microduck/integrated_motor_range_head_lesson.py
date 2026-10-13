"""Locomotion-first lesson: restrict only secondary head-command amplitude."""
from __future__ import annotations

import math
import sys

import torch

from mjlab_microduck import integrated_motor_range_linear_boost as linear
from mjlab_microduck import integrated_motor_refinement as motor

base, previous = linear.base, linear.previous
PROTOCOL = "integrated-motor-range-head-lesson-v1"
ARMS = ("control", "quiethead")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1553), ("benchmark", 256, 10, 1559))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1567), quiethead=(256, 1000, 1567))
CAP_SECONDS = 2100
PARENT, PARENT_SHA = linear.PARENT, linear.PARENT_SHA
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = linear.PARENT_NORMALIZER_COUNT, linear.PARENT_ADAM_STEP
TERM, VARIANCE, WEIGHT = linear.TERM, linear.VARIANCE, linear.WEIGHT
warm_load, evaluate_case, decision = linear.warm_load, linear.evaluate_case, linear.decision
HEAD_TERM = "head_pose"
HEAD_REWARD_WEIGHTS = {"head_pose_tracking": 2., "head_pose_bias": 3.}
FULL_RANGES = ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-.31, .31))
# Exact existing velocity-recipe beginner ranges, not zero padding or a new scale.
BEGINNER_RANGES = ((-.05, .05), (-.05, .05), (-.07, .07), (-.015, .015))


def head_ranges(mode):
    base.require(mode in MODES, "declared head-lesson mode")
    return BEGINNER_RANGES if mode == "quiethead" or mode.endswith("-quiethead") else FULL_RANGES


def reference_mode(mode):
    head_ranges(mode)
    return ("smoke-control" if mode.startswith("smoke-") else
            "benchmark-control" if mode.startswith("benchmark-") else "control")


def prepare_config(mode):
    # First materialize every final parent curriculum, then change this one
    # declared lesson axis before managers build/deepcopy their live configs.
    cfg, agent = linear.prepare_config(reference_mode(mode))
    worlds, updates, seed = MODES[mode]
    cfg.commands[HEAD_TERM].ranges = head_ranges(mode)
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations, agent.save_interval = updates, 250
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def validate_action_row(row, mode, worlds, update):
    linear.validate_action_row(row, reference_mode(mode), worlds, update)


def validate_linear_row(row, mode, worlds, update):
    linear.validate_linear_row(row, reference_mode(mode), worlds, update)


def validate_head_row(row, mode, worlds, update):
    ranges = head_ranges(mode)
    base.require(type(update) is int and 1 <= update <= MODES[mode][1]
                 and type(worlds) is int and worlds == MODES[mode][0], "declared head-lesson row budget")
    count = worlds * 24
    base.require(row["head_command_ranges"] == row["curriculum"]["head_pose_ranges"] == [list(r) for r in ranges]
                 and row["head_command_steps"] == 24
                 and type(row["head_command_sample_count"]) is int
                 and row["head_command_sample_count"] == count, "fixed head ranges and complete actor-command dwell")
    rewards = row["head_reward_means"]
    base.require(row["head_reward_weights"] == HEAD_REWARD_WEIGHTS
                 and type(row["head_reward_sample_count"]) is int and row["head_reward_sample_count"] == count
                 and set(rewards) == set(HEAD_REWARD_WEIGHTS) and base.finite_tree(rewards)
                 and 0 <= rewards["head_pose_tracking"] <= 2. and rewards["head_pose_bias"] <= 0,
                 "finite actual signed head rewards")
    stats = [row[k] for k in ("head_command_min", "head_command_max", "head_command_abs_mean", "head_command_nonzero_count")]
    base.require(all(isinstance(v, list) and len(v) == 4 for v in stats)
                 and base.finite_tree(stats), "finite four-slot head-command statistics")
    for (lo, hi), minimum, maximum, mean, nonzero in zip(ranges, *stats, strict=True):
        base.require(lo-1e-7 <= minimum <= maximum <= hi+1e-7
                     and 0 <= mean <= max(abs(minimum), abs(maximum))+1e-7
                     and type(nonzero) is int and 0 <= nonzero <= count,
                     "bounded sampled head-command statistics")
        base.require((nonzero > 0 and mean > 0) or (nonzero == 0 and minimum == maximum == mean == 0),
                     "head nonzero count agrees with absolute mean")


class MotorObserver(linear.MotorObserver):
    def __init__(self, mode):
        self.head_mode = mode
        super().__init__(reference_mode(mode))
        cfg, _ = prepare_config(mode)
        self.expected_head_cfg = cfg.commands[HEAD_TERM]
        self.expected_head_rewards = {name: cfg.rewards[name] for name in HEAD_REWARD_WEIGHTS}
        self.head_pending = None
        self.head_samples = []
        self.head_rewards = []

    def before(self, env):
        term = env.command_manager.get_term(HEAD_TERM)
        base.require(term.cfg == self.expected_head_cfg
                     and tuple(term.cfg.ranges) == head_ranges(self.head_mode), "unchanged live head-command configuration")
        base.require(self.head_pending is None, "previous actor head command consumed")
        for name, expected in self.expected_head_rewards.items():
            live = env.reward_manager.get_term_cfg(name)
            base.require(live.func is expected.func and live.params == expected.params
                         and live.weight == expected.weight == HEAD_REWARD_WEIGHTS[name],
                         "unchanged live head reward contract")
        head = env.command_manager.get_command(HEAD_TERM).detach()
        base.require(isinstance(head, torch.Tensor) and head.shape == (env.num_envs, 4)
                     and base.finite_tree(head), "finite four-slot actor head command")
        bounds = head.new_tensor(head_ranges(self.head_mode))
        base.require(bool(((head >= bounds[:, 0]-1e-7)&(head <= bounds[:, 1]+1e-7)).all()),
                     "actor head command inside live lesson ranges")
        super().before(env)
        # Capture exactly what the actor consumes, before stepping/resampling or
        # autoreset. No new RNG, reward, filtering, clipping or physics call.
        self.head_pending = (env.common_step_counter, head.double().amin(0).tolist(),
                             head.double().amax(0).tolist(), head.double().abs().mean(0).tolist(),
                             (head != 0).sum(0).tolist())

    def step(self, env, sample, extras):
        base.require(self.head_pending is not None and env.common_step_counter == self.head_pending[0]+1,
                     "same-step pre-action head command consumed")
        super().step(env, sample, extras)
        # The reward manager's completed-step matrix survives autoreset (as in
        # inherited motor/range captures). Read it, do not call the stateful EMA
        # reward again: doing so would change training. No inferred raw cost.
        rewards = {}
        for name, weight in HEAD_REWARD_WEIGHTS.items():
            actual = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(name)]
            base.require(actual.shape == (env.num_envs,) and base.finite_tree(actual)
                         and (bool(((actual >= 0)&(actual <= weight)).all()) if name == "head_pose_tracking"
                              else bool((actual <= 0).all())), "actual signed completed-step head reward")
            rewards[name] = float(actual.double().mean())
        self.head_rewards.append(rewards)
        self.head_samples.append(self.head_pending[1:])
        self.head_pending = None

    def update(self, env):
        base.require(self.head_pending is None and len(self.head_samples) == len(self.head_rewards) == 24,
                     "complete head-command update telemetry")
        row = super().update(env)
        row.update(head_command_ranges=[list(r) for r in head_ranges(self.head_mode)],
                   head_reward_weights=dict(HEAD_REWARD_WEIGHTS), head_reward_sample_count=env.num_envs*24,
                   head_reward_means={name: math.fsum(s[name] for s in self.head_rewards)/24
                                      for name in HEAD_REWARD_WEIGHTS},
                   head_command_steps=24, head_command_sample_count=env.num_envs*24,
                   head_command_min=[min(s[0][i] for s in self.head_samples) for i in range(4)],
                   head_command_max=[max(s[1][i] for s in self.head_samples) for i in range(4)],
                   head_command_abs_mean=[math.fsum(s[2][i] for s in self.head_samples)/24 for i in range(4)],
                   head_command_nonzero_count=[sum(s[3][i] for s in self.head_samples) for i in range(4)])
        validate_head_row(row, self.head_mode, env.num_envs, env.common_step_counter//24)
        self.head_samples.clear()
        self.head_rewards.clear()
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
