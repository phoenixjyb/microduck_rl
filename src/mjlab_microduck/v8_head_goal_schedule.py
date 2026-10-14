"""Bounded head-goal range history and read-only temporal telemetry.

The learner and boundary probe share this module.  It applies the existing
pose-range curriculum to the live CommandManager term before each action and
captures the already-computed 61D actor command without resampling or mutation.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping

import torch


PROTOCOL = "v8-staged-head-goals-v1"
ARMS = ("control", "staged")
HEAD_TERM = "head_pose"
ACTOR_WIDTH = 61
HEAD_SLOT = slice(51, 55)
STEPS_PER_UPDATE = 24
PROBE_SEED = 971
PROBE_WORLDS = 64
PROBE_STEPS = 320
BOUND_TOLERANCE = 1e-7

BEGINNER_RANGES = ((-.05, .05), (-.05, .05), (-.07, .07), (-.015, .015))
INTERMEDIATE_RANGES = ((-.17, .17), (-.17, .17), (-.21, .21), (-.047, .047))
PARENT_RANGES = ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))

# The schedule is keyed by the local environment common-step counter, not by
# the inherited parent counter or by per-world episode/reset clocks.
FULL_SCHEDULES = {
    "control": ((0, PARENT_RANGES),),
    "staged": ((0, BEGINNER_RANGES), (12_000, INTERMEDIATE_RANGES),
               (18_000, PARENT_RANGES)),
}
PROBE_SCHEDULES = {
    "control": ((0, PARENT_RANGES),),
    "staged": ((0, BEGINNER_RANGES), (80, INTERMEDIATE_RANGES),
               (160, PARENT_RANGES)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _step_value(step: int) -> int:
    _require(type(step) is int and step >= 0,
             "local common step must be a nonnegative integer")
    return step


def _schedule(arm: str, probe: bool) -> tuple[tuple[int, tuple[tuple[float, float], ...]], ...]:
    _require(type(arm) is str and arm in ARMS, "known head-goal schedule arm")
    _require(type(probe) is bool, "probe selector must be boolean")
    return (PROBE_SCHEDULES if probe else FULL_SCHEDULES)[arm]


def stage_index(arm: str, step: int, *, probe: bool = False) -> int:
    """Return the active predeclared stage at a local common-step counter."""
    step = _step_value(step)
    schedule = _schedule(arm, probe)
    index = 0
    for candidate, (boundary, _ranges) in enumerate(schedule):
        if step >= boundary:
            index = candidate
    return index


def ranges_at(arm: str, step: int, probe: bool = False) -> list[tuple[float, float]]:
    """Return an isolated copy of the four live inclusive command bounds."""
    step = _step_value(step)
    schedule = _schedule(arm, probe)
    index = stage_index(arm, step, probe=probe)
    selected = schedule[index][1]
    # Return mutable outer/nested data only as private copies, never schedule
    # constants; callers may safely hand this to configuration machinery.
    return copy.deepcopy([tuple(pair) for pair in selected])


def stages_for(arm: str, *, probe: bool = False) -> list[dict]:
    """Return a detached stage list suitable for the existing MDP callback."""
    return [dict(step=step, ranges=copy.deepcopy([tuple(pair) for pair in ranges]))
            for step, ranges in _schedule(arm, probe)]


def apply_ranges(env, arm: str, *, probe: bool = False) -> dict:
    """Apply the shared schedule through the existing live-manager MDP helper."""
    from .tasks import mdp

    _require(type(getattr(env, "common_step_counter", None)) is int
             and env.common_step_counter >= 0, "valid local environment common-step counter")
    step = env.common_step_counter
    ranges = ranges_at(arm, step, probe)
    mdp.pose_command_range_curriculum(
        env, None, command_name=HEAD_TERM, range_stages=stages_for(arm, probe=probe))
    term = env.command_manager.get_term(HEAD_TERM)
    _require(term is not None and tuple(tuple(pair) for pair in term.cfg.ranges)
             == tuple(ranges), "scheduled ranges applied to live head command manager")
    return dict(local_common_step=step, stage=stage_index(arm, step, probe=probe),
                ranges=copy.deepcopy(ranges))


def _tensor_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


class HeadGoalTelemetry:
    """One-step pre-action snapshots and truthful per-step range exposure."""

    def __init__(self, arm: str, *, probe: bool = False, expected_worlds: int | None = None):
        _schedule(arm, probe)
        _require(expected_worlds is None or (type(expected_worlds) is int and expected_worlds > 0),
                 "expected world count must be a positive integer")
        self.arm = arm
        self.probe = probe
        self.expected_worlds = expected_worlds
        self.pending: dict | None = None
        self.rows: list[dict] = []
        self.worlds: int | None = None
        self.first_step: int | None = None
        self.last_completed_step: int | None = None
        self.row_limit = PROBE_STEPS if probe else STEPS_PER_UPDATE

    def before(self, env, actor_obs: torch.Tensor | None = None) -> dict:
        """Apply schedule, then snapshot manager and cached actor head commands."""
        _require(self.pending is None, "previous head command snapshot is unconsumed")
        applied = apply_ranges(env, self.arm, probe=self.probe)
        step = applied["local_common_step"]
        _require(self.last_completed_step is None or step == self.last_completed_step,
                 "consecutive local common-step head telemetry; reject rollback or jump")
        worlds = getattr(env, "num_envs", None)
        _require(type(worlds) is int and worlds > 0
                 and self.expected_worlds in (None, worlds)
                 and self.worlds in (None, worlds), "stable declared head telemetry world count")

        manager = env.command_manager
        term = manager.get_term(HEAD_TERM)
        _require(term is not None and tuple(tuple(pair) for pair in term.cfg.ranges)
                 == tuple(tuple(pair) for pair in applied["ranges"]),
                 "live manager head ranges equal applied schedule")
        manager_command = manager.get_command(HEAD_TERM)
        _require(isinstance(manager_command, torch.Tensor)
                 and tuple(manager_command.shape) == (worlds, 4)
                 and _tensor_finite(manager_command), "finite four-slot manager head command")

        observation_manager = getattr(env, "observation_manager", None)
        buffer = getattr(observation_manager, "_obs_buffer", None)
        if actor_obs is None:
            actor_obs = buffer.get("actor") if isinstance(buffer, Mapping) else None
        _require(isinstance(actor_obs, torch.Tensor)
                 and tuple(actor_obs.shape) == (worlds, ACTOR_WIDTH)
                 and _tensor_finite(actor_obs), "finite cached raw 61D actor observation")
        actor_head = actor_obs[:, HEAD_SLOT]
        _require(torch.equal(actor_head, manager_command),
                 "cached actor head slots exactly match manager command")

        ranges = applied["ranges"]
        bounds = manager_command.new_tensor(ranges)
        _require(bool(((manager_command >= bounds[:, 0] - BOUND_TOLERANCE)
                       & (manager_command <= bounds[:, 1] + BOUND_TOLERANCE)).all()),
                 "sampled manager and actor head command lies inside live ranges")
        _require(all(lo < 0.0 < hi for lo, hi in ranges),
                 "all four head command slots remain nonzero")

        command = manager_command.detach().clone()
        previous_ranges = None
        stage = applied["stage"]
        if stage > 0:
            previous_ranges = ranges_at(self.arm, _schedule(self.arm, self.probe)[stage][0] - 1,
                                        probe=self.probe)
        self.pending = dict(
            before_step=step,
            stage=stage,
            ranges=copy.deepcopy(ranges),
            previous_ranges=copy.deepcopy(previous_ranges),
            manager_command=command,
            actor_command=actor_head.detach().clone(),
            worlds=worlds,
        )
        if self.first_step is None:
            self.first_step = step
        self.worlds = worlds
        return {key: copy.deepcopy(value) for key, value in self.pending.items()
                if key not in ("manager_command", "actor_command")}

    def consume(self, env, sample=None) -> dict:
        """Consume exactly the pre-action command after one completed env step."""
        del sample
        _require(self.pending is not None, "head telemetry step requires one pre-action snapshot")
        _require(len(self.rows) < self.row_limit,
                 "drain bounded head telemetry window before capturing more steps")
        pending = self.pending
        completed = getattr(env, "common_step_counter", None)
        _require(type(completed) is int and completed == pending["before_step"] + 1,
                 "exactly one local environment step consumed per head snapshot")
        command = pending["manager_command"]
        minimum = command.amin(dim=0).double().tolist()
        maximum = command.amax(dim=0).double().tolist()
        abs_mean = command.abs().double().mean(dim=0).tolist()
        nonzero = (command != 0).sum(dim=0).to(torch.int64).tolist()
        outside_previous = [0, 0, 0, 0]
        if pending["previous_ranges"] is not None:
            previous = command.new_tensor(pending["previous_ranges"])
            outside_previous = (
                (command < previous[:, 0] - BOUND_TOLERANCE)
                | (command > previous[:, 1] + BOUND_TOLERANCE)
            ).sum(dim=0).to(torch.int64).tolist()
        row = dict(
            before_step=pending["before_step"], completed_step=completed,
            schedule_kind="boundary-probe" if self.probe else "full-training",
            arm=self.arm, stage=pending["stage"], ranges=copy.deepcopy(pending["ranges"]),
            previous_ranges=copy.deepcopy(pending["previous_ranges"]),
            worlds=pending["worlds"], command_min=minimum, command_max=maximum,
            command_abs_mean=abs_mean, command_nonzero_count=nonzero,
            command_outside_previous_range_count=outside_previous,
        )
        self.rows.append(row)
        self.last_completed_step = completed
        self.pending = None
        return copy.deepcopy(row)

    def step(self, env, sample=None, extras=None) -> dict:
        """Callback-compatible alias for :meth:`consume`."""
        del extras
        return self.consume(env, sample)

    def drain(self, expected_steps: int | None = None) -> dict:
        """Aggregate and clear the bounded row window, preserving actual exposure."""
        _require(self.pending is None, "cannot summarize with an unconsumed head snapshot")
        if expected_steps is not None:
            _require(type(expected_steps) is int and expected_steps > 0
                     and expected_steps <= self.row_limit,
                     "expected drain size is a positive bounded integer")
            _require(len(self.rows) == expected_steps, "drain requires exact retained sample count")
        rows = self.rows
        _require(rows, "head telemetry summary requires actual consumed samples")
        worlds = rows[0]["worlds"]
        _require(all(row["worlds"] == worlds for row in rows), "stable worlds in head summary")
        count = len(rows) * worlds
        stage_counts = Counter(row["stage"] for row in rows)
        result = dict(
            arm=self.arm, probe=self.probe,
            schedule_kind="boundary-probe" if self.probe else "full-training",
            steps=len(rows), worlds=worlds,
            first_before_step=rows[0]["before_step"], last_completed_step=rows[-1]["completed_step"],
            stage_action_steps={str(stage): stage_counts.get(stage, 0)
                                for stage in range(len(_schedule(self.arm, self.probe)))},
            ranges_by_stage={str(row["stage"]): [list(pair) for pair in row["ranges"]]
                             for row in rows},
            command_min=[min(row["command_min"][i] for row in rows) for i in range(4)],
            command_max=[max(row["command_max"][i] for row in rows) for i in range(4)],
            command_abs_mean=[math.fsum(row["command_abs_mean"][i] for row in rows) / len(rows)
                              for i in range(4)],
            command_nonzero_count=[sum(row["command_nonzero_count"][i] for row in rows)
                                   for i in range(4)],
            command_outside_previous_range_count=[
                sum(row["command_outside_previous_range_count"][i] for row in rows)
                for i in range(4)],
            command_sample_count=count,
            # These are bounded records (24 per learner update, 320 per probe),
            # enabling independent closed-row reconciliation without retaining
            # per-world trajectories or a full learner history.
            step_records=copy.deepcopy(rows),
        )
        self.rows.clear()
        return result

    def summary(self, *, start: int = 0, stop: int | None = None) -> dict:
        """Compatibility read of a selected retained window (without clearing it)."""
        _require(self.pending is None, "cannot summarize with an unconsumed head snapshot")
        _require(type(start) is int and start >= 0, "summary start must be nonnegative integer")
        if stop is None:
            stop = len(self.rows)
        _require(type(stop) is int and stop >= start and stop <= len(self.rows),
                 "summary stop must select retained rows")
        selected = self.rows
        self.rows = selected[start:stop]
        try:
            return self.drain()
        finally:
            self.rows = selected


def _finite_vector(value, *, integer: bool = False) -> bool:
    return (isinstance(value, list) and len(value) == 4
            and all(type(item) is int if integer else
                    type(item) in (int, float) and math.isfinite(item)
                    for item in value))


def _same_ranges(value, expected) -> bool:
    return (isinstance(value, list) and len(value) == 4
            and all(isinstance(pair, (list, tuple)) and len(pair) == 2
                    and tuple(pair) == tuple(expected[index])
                    for index, pair in enumerate(value)))


def validate_window(summary: Mapping, arm: str, probe: bool, expected_worlds: int,
                    start_step: int, expected_steps: int) -> dict:
    """Validate a closed learner/probe window and recompute its aggregates.

    This validates the JSON-shaped rows emitted by :meth:`HeadGoalTelemetry.drain`;
    it does not infer commands or exposure that were not captured before action.
    """
    schedule = _schedule(arm, probe)
    _require(isinstance(summary, Mapping), "head telemetry summary mapping")
    _require(type(expected_worlds) is int and expected_worlds > 0,
             "expected head telemetry world count")
    _require(type(start_step) is int and start_step >= 0,
             "head telemetry window start step")
    _require(type(expected_steps) is int and expected_steps > 0
             and expected_steps <= (PROBE_STEPS if probe else STEPS_PER_UPDATE),
             "head telemetry window size within schedule-kind limit")
    kind = "boundary-probe" if probe else "full-training"
    _require(summary.get("arm") == arm and summary.get("probe") is probe
             and summary.get("schedule_kind") == kind,
             "head telemetry schedule kind and arm identity")
    _require(summary.get("steps") == expected_steps
             and summary.get("worlds") == expected_worlds
             and summary.get("command_sample_count") == expected_steps * expected_worlds,
             "head telemetry exact step/world sample count")
    _require(summary.get("first_before_step") == start_step
             and summary.get("last_completed_step") == start_step + expected_steps,
             "head telemetry window counters")
    rows = summary.get("step_records")
    _require(isinstance(rows, list) and len(rows) == expected_steps,
             "head telemetry bounded per-step records")

    stage_counts: Counter = Counter()
    min_rows, max_rows, mean_rows = [], [], []
    nonzero_rows, outside_rows = [], []
    ranges_by_stage = {}
    for offset, row in enumerate(rows):
        expected_step = start_step + offset
        expected_stage = stage_index(arm, expected_step, probe=probe)
        expected_ranges = ranges_at(arm, expected_step, probe)
        _require(isinstance(row, Mapping)
                 and row.get("schedule_kind") == kind and row.get("arm") == arm,
                 "step record schedule kind and arm")
        _require(row.get("before_step") == expected_step
                 and row.get("completed_step") == expected_step + 1
                 and row.get("stage") == expected_stage,
                 "contiguous actual pre-action step counters and stage")
        _require(row.get("worlds") == expected_worlds
                 and _same_ranges(row.get("ranges"), expected_ranges),
                 "actual applied manager ranges match declared stage")
        expected_previous = (None if expected_stage == 0 else
                             ranges_at(arm, schedule[expected_stage][0] - 1, probe=probe))
        _require((row.get("previous_ranges") is None and expected_previous is None)
                 or _same_ranges(row.get("previous_ranges"), expected_previous),
                 "previous-stage range metadata")
        _require(_finite_vector(row.get("command_min"))
                 and _finite_vector(row.get("command_max"))
                 and _finite_vector(row.get("command_abs_mean"))
                 and _finite_vector(row.get("command_nonzero_count"), integer=True)
                 and _finite_vector(row.get("command_outside_previous_range_count"), integer=True),
                 "finite four-component command summaries")
        for i, (lo, hi) in enumerate(expected_ranges):
            lower, upper = row["command_min"][i], row["command_max"][i]
            mean = row["command_abs_mean"][i]
            nonzero = row["command_nonzero_count"][i]
            outside = row["command_outside_previous_range_count"][i]
            _require(lo - BOUND_TOLERANCE <= lower <= upper <= hi + BOUND_TOLERANCE
                     and 0 <= mean <= max(abs(lower), abs(upper)) + BOUND_TOLERANCE
                     and 0 <= nonzero <= expected_worlds and 0 <= outside <= nonzero,
                     "sampled command statistics inside actual live range")
            _require((nonzero == 0 and lower == upper == mean == 0.0)
                     or (nonzero > 0 and mean > 0.0),
                     "command nonzero counts agree with captured absolute means")
            if expected_previous is None:
                _require(outside == 0, "no previous-stage exposure count at first stage")
            else:
                prev_lo, prev_hi = expected_previous[i]
                extremes_cross_previous = (
                    lower < prev_lo - BOUND_TOLERANCE
                    or upper > prev_hi + BOUND_TOLERANCE
                )
                _require((outside >= 1) if extremes_cross_previous else (outside == 0),
                         "previous-stage exposure count agrees with captured command extrema")
        stage_counts[expected_stage] += 1
        ranges_by_stage[str(expected_stage)] = [list(pair) for pair in expected_ranges]
        min_rows.append(row["command_min"])
        max_rows.append(row["command_max"])
        mean_rows.append(row["command_abs_mean"])
        nonzero_rows.append(row["command_nonzero_count"])
        outside_rows.append(row["command_outside_previous_range_count"])

    expected_stage_counts = {str(stage): stage_counts.get(stage, 0)
                             for stage in range(len(schedule))}
    _require(summary.get("stage_action_steps") == expected_stage_counts
             and summary.get("ranges_by_stage") == ranges_by_stage,
             "summary stage exposure recomputes from actual step rows")
    expected_min = [min(row[i] for row in min_rows) for i in range(4)]
    expected_max = [max(row[i] for row in max_rows) for i in range(4)]
    expected_mean = [math.fsum(row[i] for row in mean_rows) / expected_steps for i in range(4)]
    expected_nonzero = [sum(row[i] for row in nonzero_rows) for i in range(4)]
    expected_outside = [sum(row[i] for row in outside_rows) for i in range(4)]
    _require(summary.get("command_min") == expected_min
             and summary.get("command_max") == expected_max
             and summary.get("command_abs_mean") == expected_mean
             and summary.get("command_nonzero_count") == expected_nonzero
             and summary.get("command_outside_previous_range_count") == expected_outside,
             "window aggregates recompute from actual step rows")

    if probe and arm == "staged":
        for stage in (1, 2):
            stage_outside = [sum(row["command_outside_previous_range_count"][i]
                                 for row in rows if row["stage"] == stage)
                             for i in range(4)]
            _require(all(count >= 1 for count in stage_outside),
                     "probe requires actual per-component exposure beyond each previous range")
    return dict(valid=True, schedule_kind=kind, arm=arm, steps=expected_steps,
                worlds=expected_worlds, start_step=start_step,
                end_step=start_step + expected_steps)


__all__ = ["ARMS", "PROTOCOL", "BEGINNER_RANGES", "INTERMEDIATE_RANGES", "PARENT_RANGES",
           "FULL_SCHEDULES", "PROBE_SCHEDULES", "PROBE_SEED", "PROBE_WORLDS", "PROBE_STEPS",
           "stage_index", "ranges_at", "stages_for", "apply_ranges", "HeadGoalTelemetry",
           "validate_window"]
