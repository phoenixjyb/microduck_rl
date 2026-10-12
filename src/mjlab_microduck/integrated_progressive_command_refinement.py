"""Matched progressive command distribution continuation.

The only treatment is the live twist sampler's yaw ceiling and forward
command mixture. Existing commands are never rewritten when a stage widens.
"""

import sys
from collections.abc import Mapping

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor


base = previous.base
PROTOCOL = "integrated-progressive-command-continuation-v1"
ARMS = ("control", "progressive")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1231),
                                               ("benchmark", 256, 10, 1237))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1249), progressive=(256, 1000, 1249))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {"control": 2100, "progressive": 2100}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE

# Smoke reaches all four settings at updates 0, 1, 2, and 3. The .05 first
# ceiling keeps signed-yaw practice available during a five-update smoke run,
# whose horizon is shorter than the inherited 3-5 second resampling interval.
FULL_STAGES = ((0, .05, .65), (250, .10, .50),
               (500, .25, .35), (750, .50, .25))
SMOKE_STAGES = ((0, .05, .65), (1, .10, .50),
                (2, .25, .35), (3, .50, .25))
BENCHMARK_STAGE = ((0, .50, .25),)
CONTROL_STAGE = ((0, .50, .25),)
_BASE_YAW_RANGES = (-.5, .5)
_BASE_FORWARD_MIX = .25


def stages(mode):
    base.require(type(mode) is str and mode in MODES, "known progressive command mode")
    if mode.endswith("control") or mode == "control":
        return CONTROL_STAGE
    if mode.startswith("smoke-"):
        return SMOKE_STAGES
    if mode.startswith("benchmark-"):
        return BENCHMARK_STAGE
    return FULL_STAGES


def stage_at_step(common_step, schedule):
    """Select a stage at its inclusive pre-action common-step boundary."""
    base.require(type(common_step) is int and common_step >= 0,
                 "nonnegative integer common-step clock")
    base.require(isinstance(schedule, tuple) and bool(schedule), "nonempty immutable stage schedule")
    base.require(isinstance(schedule[0], tuple) and len(schedule[0]) == 3
                 and schedule[0][0] == 0, "stage schedule starts at update zero")
    previous_step = -1
    previous_yaw = -1.
    previous_forward = 2.
    for row in schedule:
        base.require(isinstance(row, tuple) and len(row) == 3
                     and type(row[0]) is int and row[0] >= 0
                     and type(row[1]) in (float, int) and type(row[2]) in (float, int),
                     "well-formed command stage")
        update, yaw_cap, forward_mix = row
        base.require(update > previous_step and 0. < yaw_cap <= .5
                     and 0. <= forward_mix <= 1.
                     and yaw_cap >= previous_yaw and forward_mix <= previous_forward,
                     "strictly ordered monotonic command stages")
        previous_step, previous_yaw, previous_forward = update, yaw_cap, forward_mix
    selected = schedule[0]
    for row in schedule:
        if common_step >= row[0] * 24:
            selected = row
        else:
            break
    return selected


def prepare_config(mode):
    base.require(mode in MODES, "known progressive command mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    twist = cfg.commands["twist"]
    base.require(twist.ranges.lin_vel_x == (.2, .4)
                 and twist.ranges.lin_vel_y == (0., 0.)
                 and twist.ranges.ang_vel_z == _BASE_YAW_RANGES
                 and twist.resampling_time_range == (3., 5.)
                 and twist.rel_standing_envs == .15
                 and twist.rel_forward_envs == _BASE_FORWARD_MIX
                 and twist.rel_turn_in_place_envs == 0.
                 and not cfg.curriculum,
                 "unchanged inherited twist sampler and collapsed curricula")
    _, yaw_cap, forward_mix = stages(mode)[0]
    twist.ranges.ang_vel_z = (-yaw_cap, yaw_cap)
    twist.rel_forward_envs = forward_mix
    base.require(cfg.rewards[previous.previous.TERM].weight == 0.,
                 "unchanged inherited zero yaw-bias treatment")
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
evaluate_case = previous.evaluate_case
decision = previous.decision


def _live_twist(env):
    term = env.command_manager.get_term("twist")
    cfg = term.cfg
    return term, cfg


class MotorObserver(previous.MotorObserver):
    """Apply sampler stages before action while preserving inherited telemetry."""

    def __init__(self, mode):
        base.require(type(mode) is str and mode in MODES,
                     "known progressive command observer mode")
        super().__init__("control")
        self.command_mode = mode
        self.schedule = stages(mode)
        self.pending_clock = None
        self.pending_stage = None
        self.last_post_clock = None
        self.update_stage_counts = dict.fromkeys(range(len(self.schedule)), 0)
        self.cumulative_stage_counts = dict.fromkeys(range(len(self.schedule)), 0)
        self.update_action_steps = 0
        self.cumulative_action_steps = 0
        self.observed_max_yaw = 0.
        self.configured_forward_stage = None

    def _validate_sampler(self, env, cfg, expected):
        _, yaw_cap, forward_mix = expected
        base.require(cfg.ranges.ang_vel_z == (-yaw_cap, yaw_cap)
                     and cfg.ranges.lin_vel_x == (.2, .4)
                     and cfg.ranges.lin_vel_y == (0., 0.)
                     and cfg.resampling_time_range == (3., 5.)
                     and cfg.rel_standing_envs == .15
                     and cfg.rel_forward_envs == forward_mix
                     and cfg.rel_turn_in_place_envs == 0.,
                     "live twist cfg equals the expected prior command stage")

    def before(self, env):
        base.require(self.schedule == stages(self.command_mode), "unchanged declared command schedule")
        base.require(self.pending_clock is None and self.pending_stage is None,
                     "previous progressive command action consumed")
        base.require(type(env.common_step_counter) is int and env.common_step_counter >= 0,
                     "nonnegative live common-step clock")
        expected_clock = 0 if self.last_post_clock is None else self.last_post_clock
        base.require(env.common_step_counter == expected_clock,
                     "continuous nonreversing one-step command clock")
        stage = stage_at_step(env.common_step_counter, self.schedule)
        term, cfg = _live_twist(env)
        old_stage = stage_at_step(max(0, env.common_step_counter - 1), self.schedule)
        if self.last_post_clock is None:
            old_stage = stage
        self._validate_sampler(env, cfg, old_stage)

        command = env.command_manager.get_command("twist")
        manager = env.observation_manager
        buffer = getattr(manager, "_obs_buffer", None)
        actor = buffer.get("actor") if isinstance(buffer, Mapping) else None
        base.require(isinstance(command, torch.Tensor) and command.shape == (env.num_envs, 3)
                     and base.finite_tree(command)
                     and isinstance(actor, torch.Tensor) and actor.shape == (env.num_envs, 61)
                     and base.finite_tree(actor) and torch.equal(actor[:, 48:51], command),
                     "cached raw 61D actor command matches live manager command")
        idle = (command == 0.).all(-1)
        base.require(bool((command[:, 1] == 0.).all())
                     and bool((idle | ((command[:, 0] >= .2)
                                       & (command[:, 0] <= .4))).all())
                     and bool((command[:, 2] >= -old_stage[1]).all())
                     and bool((command[:, 2] <= old_stage[1]).all()),
                     "cached pre-action commands fit the old stage support")

        # The inherited observer snapshots actual pre-action dwell before the
        # config is widened. It does not resample or mutate command tensors.
        super().before(env)
        if stage != old_stage:
            cfg.ranges.ang_vel_z = (-stage[1], stage[1])
            cfg.rel_forward_envs = stage[2]
        self._validate_sampler(env, cfg, stage)

        self.pending_clock = env.common_step_counter
        self.pending_stage = stage
        stage_index = self.schedule.index(stage)
        self.update_stage_counts[stage_index] += env.num_envs
        self.cumulative_stage_counts[stage_index] += env.num_envs
        self.update_action_steps += 1
        self.cumulative_action_steps += 1
        self.observed_max_yaw = max(self.observed_max_yaw,
                                    float(command[:, 2].abs().max()))
        self.configured_forward_stage = stage[2]

    def step(self, env, sample, extras):
        base.require(self.pending_clock is not None and self.pending_stage is not None
                     and env.common_step_counter == self.pending_clock + 1,
                     "one completed action step per progressive command clock")
        super().step(env, sample, extras)
        self.last_post_clock = env.common_step_counter
        self.pending_clock = self.pending_stage = None

    def update(self, env):
        row = super().update(env)
        base.require(self.pending_clock is None and self.pending_stage is None
                     and self.update_action_steps == 24
                     and sum(self.update_stage_counts.values()) == 24 * env.num_envs
                     and sum(self.cumulative_stage_counts.values())
                     == env.common_step_counter * env.num_envs
                     and self.cumulative_action_steps == env.common_step_counter,
                     "complete 24-step per-update and cumulative stage census")
        row.update(command_curriculum_protocol="pre-action-command-stage-census-v1",
                   command_mode=self.command_mode,
                   command_curriculum_stage=list(stage_at_step(env.common_step_counter - 1,
                                                                self.schedule)),
                   command_curriculum_stage_counts={str(k): v for k, v in
                                                    self.update_stage_counts.items()},
                   command_curriculum_cumulative_counts={str(k): v for k, v in
                                                         self.cumulative_stage_counts.items()},
                   command_curriculum_action_steps=self.update_action_steps,
                   command_curriculum_cumulative_action_steps=self.cumulative_action_steps,
                   command_curriculum_observed_max_abs_yaw=self.observed_max_yaw,
                   command_curriculum_configured_forward_fraction=self.configured_forward_stage)
        self.update_action_steps = 0
        self.update_stage_counts = dict.fromkeys(range(len(self.schedule)), 0)
        self.observed_max_yaw = 0.
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
