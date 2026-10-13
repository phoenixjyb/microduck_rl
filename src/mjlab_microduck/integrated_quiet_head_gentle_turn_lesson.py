"""Matched gentle-turn command-support lesson inside the quiet-head recipe."""
from __future__ import annotations

import sys
from collections.abc import Mapping

import torch

from mjlab_microduck import integrated_motor_range_head_lesson as head
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_training_budget_refinement as exposure

base, previous = head.base, head.previous
PROTOCOL = "integrated-quiet-head-gentle-turn-v1"
ARMS = ("control", "gentleturn")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1673),
                                              ("benchmark", 256, 10, 1693))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1697), gentleturn=(256, 1000, 1697))
CAP_SECONDS = 2100
SAVE_INTERVAL = 250
PARENT, PARENT_SHA = head.PARENT, head.PARENT_SHA
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = head.PARENT_NORMALIZER_COUNT, head.PARENT_ADAM_STEP
TERM, VARIANCE, WEIGHT = head.TERM, head.VARIANCE, head.WEIGHT
warm_load, decision = head.warm_load, head.decision
EXPOSURE_BINS = exposure.EXPOSURE_BINS
YAW_SUPPORT = {"control": (-.5, .5), "gentleturn": (-.25, .25)}


def gentle_turn(mode):
    base.require(mode in MODES, "declared gentle-turn lesson mode")
    return mode == "gentleturn" or mode.endswith("-gentleturn")


def yaw_support(mode):
    return YAW_SUPPORT["gentleturn" if gentle_turn(mode) else "control"]


def yaw_variance(mode):
    gentle_turn(mode)
    return VARIANCE


def reference_mode(mode):
    gentle_turn(mode)
    return ("smoke-quiethead" if mode.startswith("smoke-") else
            "benchmark-quiethead" if mode.startswith("benchmark-") else "quiethead")


def head_ranges(mode):
    return head.head_ranges(reference_mode(mode))


def prepare_config(mode):
    base.require(mode in MODES, "known gentle-turn lesson mode")
    cfg, agent = head.prepare_config(reference_mode(mode))
    worlds, updates, seed = MODES[mode]
    cfg.commands["twist"].ranges.ang_vel_z = yaw_support(mode)
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations, agent.save_interval = updates, SAVE_INTERVAL
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def validate_action_row(row, mode, worlds, update):
    head.validate_action_row(row, reference_mode(mode), worlds, update)


def validate_linear_row(row, mode, worlds, update):
    head.validate_linear_row(row, reference_mode(mode), worlds, update)


def validate_head_row(row, mode, worlds, update):
    head.validate_head_row(row, reference_mode(mode), worlds, update)


def validate_yaw_row(row, mode, worlds, update):
    support = yaw_support(mode)
    base.require(type(update) is int and 1 <= update <= MODES[mode][1]
                 and type(worlds) is int and worlds == MODES[mode][0],
                 "declared gentle-turn row budget")
    base.require(row["yaw_variance"] == VARIANCE and row["roll_pitch_variance"] == VARIANCE
                 and row["angular_tracking_weight"] == WEIGHT
                 and row["angular_aggregation"] == "stock_multiplicative_product"
                 and row["roll_pitch_tracking_included"] is True
                 and row["angular_reward_sample_count"] == worlds * 24
                 and row["negative_reward_sample_count"] == row["negative_reward_fraction"] == 0
                 and base.finite_tree([row[k] for k in ("weighted_angular_reward_mean",
                     "yaw_error_sq_mean", "roll_pitch_rate_sq_mean")])
                 and 0 <= row["weighted_angular_reward_mean"] <= WEIGHT
                 and row["yaw_error_sq_mean"] >= 0 and row["roll_pitch_rate_sq_mean"] >= 0
                 and row["body_ang_vel_weight"] == -.05
                 and row["angular_momentum_weight"] == -.02,
                 "unchanged stock angular tracking and separate stability terms")
    base.require(row["training_yaw_support"] == list(support),
                 "fixed declared training yaw support")


def validate_support_row(row, mode, worlds, update):
    support = yaw_support(mode)
    base.require(type(update) is int and 1 <= update <= MODES[mode][1]
                 and type(worlds) is int and worlds == MODES[mode][0],
                 "declared gentle-turn row budget")
    curriculum = row.get("curriculum")
    bins = row["command_exposure_counts"]
    base.require(row["command_exposure_protocol"] == "actor-command-dwell-v1"
                 and row["training_yaw_support"] == list(support)
                 and type(row["command_exposure_steps"]) is int and row["command_exposure_steps"] == 24
                 and isinstance(curriculum, dict) and curriculum.get("rel_standing_envs") == .25
                 and isinstance(bins, dict) and set(bins) == set(EXPOSURE_BINS)
                 and all(type(v) is int and v >= 0 for v in bins.values())
                 and sum(bins.values()) == worlds * 24,
                 "complete exclusive per-update command exposure")
    if support == (-.25, .25):
        base.require(bins["negative-high"] == bins["positive-high"] == 0,
                     "narrow support excludes high-yaw exposure")
    cumulative = row["command_exposure_cumulative"]
    expected = worlds * update * 24
    base.require(isinstance(cumulative, dict) and set(cumulative) == set(EXPOSURE_BINS)
                 and all(type(v) is int and v >= 0 for v in cumulative.values())
                 and sum(cumulative.values()) == expected
                 and all(cumulative[key] >= bins[key] for key in EXPOSURE_BINS),
                 "complete cumulative command exposure")
    if support == (-.25, .25):
        base.require(cumulative["negative-high"] == cumulative["positive-high"] == 0,
                     "cumulative narrow support excludes high-yaw exposure")


def evaluate_case(checkpoint, seed, speed, yaw, health):
    return base.evaluate_case(checkpoint, seed, speed, yaw, health,
                              diagnostics=True, yaw_trace=True, gait_trace=True)


class MotorObserver(head.MotorObserver):
    def __init__(self, mode):
        self.lesson_mode = mode
        super().__init__(reference_mode(mode))
        cfg, _ = prepare_config(mode)
        self.expected_twist_cfg = cfg.commands["twist"]

    def before(self, env):
        live = env.command_manager.get_term("twist").cfg
        base.require(live == self.expected_twist_cfg,
                     "unchanged complete live twist sampler configuration")
        command = env.command_manager.get_command("twist")
        buffer = getattr(env.observation_manager, "_obs_buffer", None)
        actor = buffer.get("actor") if isinstance(buffer, Mapping) else None
        base.require(isinstance(command, torch.Tensor) and command.shape == (env.num_envs, 3)
                     and command.is_floating_point() and base.finite_tree(command),
                     "finite live Nx3 twist command")
        low, high = yaw_support(self.lesson_mode)
        base.require(bool(((command[:, 2] >= low) & (command[:, 2] <= high)).all()),
                     "live twist command within declared yaw support")
        base.require(isinstance(actor, torch.Tensor) and actor.shape == (env.num_envs, 61)
                     and base.finite_tree(actor) and torch.equal(actor[:, 48:51], command),
                     "raw cached actor command equals live twist manager")
        # The inherited FreshObserver validates and captures the same live
        # actor command in its normal before_action hook. This lesson only
        # adds independent support/config guards; it must not count a second
        # exposure stream alongside that observer.
        super().before(env)

    def update(self, env):
        row = super().update(env)
        row.update(training_yaw_support=list(yaw_support(self.lesson_mode)))
        update = env.common_step_counter // 24
        validate_support_row(row, self.lesson_mode, env.num_envs, update)
        validate_yaw_row(row, self.lesson_mode, env.num_envs, update)
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
