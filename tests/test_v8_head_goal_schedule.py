from __future__ import annotations

import copy
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import v8_head_goal_schedule as schedule


class _Manager:
    def __init__(self, ranges, command):
        self.term = NS(cfg=NS(ranges=copy.deepcopy(ranges)))
        self.command = command

    def get_term(self, name):
        assert name == schedule.HEAD_TERM
        return self.term

    def get_command(self, name):
        assert name == schedule.HEAD_TERM
        return self.command


class _Env:
    def __init__(self, step=0, worlds=4, command=None):
        if command is None:
            command = torch.zeros(worlds, 4)
        self.common_step_counter = step
        self.num_envs = worlds
        self.command_manager = _Manager(schedule.PARENT_RANGES, command)
        actor = torch.zeros(worlds, schedule.ACTOR_WIDTH)
        actor[:, schedule.HEAD_SLOT] = command
        self.observation_manager = NS(_obs_buffer={"actor": actor})

    def advance(self, *, command=None):
        self.common_step_counter += 1
        if command is not None:
            self.command_manager.command = command
            self.observation_manager._obs_buffer["actor"][:, schedule.HEAD_SLOT] = command


def _assert_ranges(actual, expected):
    assert tuple(tuple(pair) for pair in actual) == expected


@pytest.mark.parametrize("arm,step,expected_stage,expected", [
    ("control", 0, 0, schedule.PARENT_RANGES),
    ("control", 23999, 0, schedule.PARENT_RANGES),
    ("staged", 0, 0, schedule.BEGINNER_RANGES),
    ("staged", 11999, 0, schedule.BEGINNER_RANGES),
    ("staged", 12000, 1, schedule.INTERMEDIATE_RANGES),
    ("staged", 17999, 1, schedule.INTERMEDIATE_RANGES),
    ("staged", 18000, 2, schedule.PARENT_RANGES),
    ("staged", 23999, 2, schedule.PARENT_RANGES),
])
def test_full_schedule_boundaries(arm, step, expected_stage, expected):
    assert schedule.stage_index(arm, step) == expected_stage
    _assert_ranges(schedule.ranges_at(arm, step), expected)


@pytest.mark.parametrize("step,expected_stage,expected", [
    (0, 0, schedule.BEGINNER_RANGES),
    (79, 0, schedule.BEGINNER_RANGES),
    (80, 1, schedule.INTERMEDIATE_RANGES),
    (159, 1, schedule.INTERMEDIATE_RANGES),
    (160, 2, schedule.PARENT_RANGES),
    (319, 2, schedule.PARENT_RANGES),
])
def test_probe_schedule_has_only_its_accelerated_boundaries(step, expected_stage, expected):
    assert schedule.stage_index("staged", step, probe=True) == expected_stage
    _assert_ranges(schedule.ranges_at("staged", step, probe=True), expected)
    # The same counter in the real learner is still at the beginner stage.
    assert schedule.stage_index("staged", step, probe=False) == 0


def test_control_probe_never_accidentally_uses_staged_probe_schedule():
    for step in (0, 80, 160, 319):
        _assert_ranges(schedule.ranges_at("control", step, probe=True), schedule.PARENT_RANGES)
        _assert_ranges(schedule.ranges_at("control", step, probe=False), schedule.PARENT_RANGES)
    with pytest.raises(ValueError, match="known head-goal schedule arm"):
        schedule.ranges_at("quiethead", 0)
    for bad in (-1, True, 0.0, "0"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            schedule.ranges_at("control", bad)
    with pytest.raises(ValueError, match="probe selector"):
        schedule.ranges_at("control", 0, probe=1)


def test_ranges_and_stage_payloads_are_detached_from_schedule_constants():
    first = schedule.ranges_at("staged", 0)
    first[0] = (-9.0, 9.0)
    assert schedule.ranges_at("staged", 0)[0] == (-.05, .05)
    stages = schedule.stages_for("staged")
    stages[0]["ranges"][0] = (-8.0, 8.0)
    assert schedule.stages_for("staged")[0]["ranges"][0] == (-.05, .05)


@pytest.mark.parametrize("arm,step,expected", [
    ("control", 0, schedule.PARENT_RANGES),
    ("staged", 0, schedule.BEGINNER_RANGES),
    ("staged", 12000, schedule.INTERMEDIATE_RANGES),
    ("staged", 18000, schedule.PARENT_RANGES),
])
def test_before_reapplies_existing_curriculum_to_live_manager(arm, step, expected):
    env = _Env(step=step)
    env.command_manager.term.cfg.ranges = schedule.PARENT_RANGES  # simulate another callback
    collector = schedule.HeadGoalTelemetry(arm)
    snap = collector.before(env)
    _assert_ranges(env.command_manager.term.cfg.ranges, expected)
    _assert_ranges(snap["ranges"], expected)
    assert snap["before_step"] == step


def test_exact_cached_actor_slots_nonzero_counts_and_pre_reset_snapshot_survive():
    command = torch.tensor([
        [.02, 0., 0., 0.], [-.02, .03, 0., 0.],
        [0., -.03, .04, 0.], [0., 0., -.04, .005],
    ])
    env = _Env(worlds=4, command=command)
    collector = schedule.HeadGoalTelemetry("staged", expected_worlds=4)
    snap = collector.before(env)
    _assert_ranges(snap["ranges"], schedule.BEGINNER_RANGES)
    # Simulated autoreset/resample occurs after the saved actor command was used.
    reset = torch.full_like(command, 9.)
    env.advance(command=reset)
    row = collector.step(env)
    assert row["command_min"] == pytest.approx([-.02, -.03, -.04, 0.])
    assert row["command_max"] == pytest.approx([.02, .03, .04, .005])
    assert row["command_nonzero_count"] == [2, 2, 2, 1]
    assert row["command_outside_previous_range_count"] == [0, 0, 0, 0]
    assert torch.all(env.command_manager.command == 9.)  # not rewritten by telemetry
    summary = collector.summary()
    assert summary["steps"] == 1 and summary["command_sample_count"] == 4
    assert summary["stage_action_steps"] == {"0": 1, "1": 0, "2": 0}


def test_probe_outside_previous_range_counts_only_real_samples():
    collector = schedule.HeadGoalTelemetry("staged", probe=True, expected_worlds=4)
    # Existing beginner-range samples persist through a widening unless naturally
    # resampled; no extra RNG or fabricated exposure is introduced.
    stage1_command = torch.zeros(4, 4)
    stage1_command[0] = torch.tensor([.08, .08, .10, .02])
    env = _Env(step=80, worlds=4, command=stage1_command)
    collector.before(env)
    env.advance()
    first = collector.step(env)
    assert first["stage"] == 1
    assert first["command_outside_previous_range_count"] == [1, 1, 1, 1]

    env.command_manager.command = torch.tensor([
        [.08, .08, .10, .02], [-.08, -.08, -.10, -.02],
        [.16, 0., 0., 0.], [0., -.16, 0., 0.],
    ])
    env.observation_manager._obs_buffer["actor"][:, schedule.HEAD_SLOT] = env.command_manager.command
    collector.before(env)
    env.advance()
    second = collector.step(env)
    assert second["command_nonzero_count"] == [3, 3, 2, 2]
    assert second["command_outside_previous_range_count"] == [3, 3, 2, 2]


def test_step_requires_exactly_one_action_and_rejects_pending_or_global_counter_jumps():
    env = _Env(worlds=2)
    collector = schedule.HeadGoalTelemetry("control", expected_worlds=2)
    with pytest.raises(ValueError, match="requires one pre-action snapshot"):
        collector.step(env)
    collector.before(env)
    with pytest.raises(ValueError, match="snapshot is unconsumed"):
        collector.before(env)
    env.common_step_counter += 2
    with pytest.raises(ValueError, match="exactly one local environment step"):
        collector.step(env)
    collector.pending = None
    env.common_step_counter = 0
    collector.before(env)
    env.advance()
    collector.step(env)
    env.common_step_counter = 0  # global rollback is rejected, unlike per-world reset
    with pytest.raises(ValueError, match="rollback or jump"):
        collector.before(env)


@pytest.mark.parametrize("fault", ["shape", "nan", "bounds", "actor_mismatch", "bad_actor"])
def test_before_rejects_empty_or_invalid_command_evidence(fault):
    env = _Env(worlds=2)
    actor = env.observation_manager._obs_buffer["actor"]
    if fault == "shape":
        env.command_manager.command = torch.zeros(2, 0)
    elif fault == "nan":
        env.command_manager.command[0, 0] = float("nan")
    elif fault == "bounds":
        env.command_manager.command[0, 2] = .08
        actor[:, schedule.HEAD_SLOT] = env.command_manager.command
    elif fault == "actor_mismatch":
        actor[0, 51] = .01
    elif fault == "bad_actor":
        env.observation_manager._obs_buffer["actor"] = torch.zeros(2, 60)
    collector = schedule.HeadGoalTelemetry("staged", expected_worlds=2)
    expected_message = {
        "shape": "finite four-slot manager head command",
        "nan": "finite four-slot manager head command",
        "bounds": "sampled manager and actor head command lies inside live ranges",
        "actor_mismatch": "cached actor head slots exactly match manager command",
        "bad_actor": "finite cached raw 61D actor observation",
    }[fault]
    with pytest.raises(ValueError, match=expected_message):
        collector.before(env)


def test_per_world_reset_does_not_change_global_stage_and_no_sample_is_fabricated():
    env = _Env(step=80, worlds=2)
    collector = schedule.HeadGoalTelemetry("staged", probe=True, expected_worlds=2)
    # Two individual worlds may reset without advancing or rolling back global time.
    snap = collector.before(env)
    assert snap["stage"] == 1
    env.advance(command=torch.zeros(2, 4))
    collector.step(env)
    env.command_manager.command.zero_()
    env.observation_manager._obs_buffer["actor"][:, schedule.HEAD_SLOT] = 0.
    snap = collector.before(env)
    assert snap["stage"] == 1
    env.advance()
    collector.step(env)
    result = collector.drain(expected_steps=2)
    assert result["steps"] == 2
    assert result["command_sample_count"] == 4
    assert result["command_nonzero_count"] == [0, 0, 0, 0]
    assert result["command_outside_previous_range_count"] == [0, 0, 0, 0]


def test_full_probe_retains_exact_80_80_160_actual_step_exposure():
    env = _Env(step=0, worlds=schedule.PROBE_WORLDS)
    collector = schedule.HeadGoalTelemetry(
        "staged", probe=True, expected_worlds=schedule.PROBE_WORLDS)
    for _ in range(schedule.PROBE_STEPS):
        if env.common_step_counter in (80, 160):
            command = torch.zeros(schedule.PROBE_WORLDS, 4)
            command[0] = torch.tensor([.08, .08, .10, .02]
                                      if env.common_step_counter == 80
                                      else [.20, .20, .25, .06])
            env.command_manager.command = command
            env.observation_manager._obs_buffer["actor"][:, schedule.HEAD_SLOT] = command
        collector.before(env)
        env.advance()
        collector.consume(env, sample=None)
    assert len(collector.rows) == schedule.PROBE_STEPS
    result = collector.drain(expected_steps=schedule.PROBE_STEPS)
    assert result["stage_action_steps"] == {"0": 80, "1": 80, "2": 160}
    assert result["first_before_step"] == 0
    assert result["last_completed_step"] == 320
    assert result["command_sample_count"] == 64 * 320
    assert len(result["step_records"]) == 320
    assert [result["step_records"][step]["stage"] for step in (0, 79, 80, 159, 160, 319)] == [
        0, 0, 1, 1, 2, 2]
    assert [result["step_records"][step]["before_step"] for step in (0, 79, 80, 159, 160, 319)] == [
        0, 79, 80, 159, 160, 319]
    assert collector.rows == []
    assert schedule.validate_window(result, "staged", True, 64, 0, 320)["valid"] is True


def test_validate_window_recomputes_full_update_and_rejects_schedule_or_aggregate_tampering():
    env = _Env(step=0, worlds=2)
    collector = schedule.HeadGoalTelemetry("control", expected_worlds=2)
    for _ in range(24):
        collector.before(env)
        env.advance()
        collector.consume(env)
    report = collector.drain(expected_steps=24)
    assert schedule.validate_window(report, "control", False, 2, 0, 24)["valid"] is True

    bad_kind = copy.deepcopy(report)
    bad_kind["schedule_kind"] = "boundary-probe"
    with pytest.raises(ValueError, match="schedule kind and arm identity"):
        schedule.validate_window(bad_kind, "control", False, 2, 0, 24)
    bad_aggregate = copy.deepcopy(report)
    bad_aggregate["command_sample_count"] -= 1
    with pytest.raises(ValueError, match="exact step/world sample count"):
        schedule.validate_window(bad_aggregate, "control", False, 2, 0, 24)
    bad_counter = copy.deepcopy(report)
    bad_counter["step_records"][3]["before_step"] += 1
    with pytest.raises(ValueError, match="contiguous actual pre-action"):
        schedule.validate_window(bad_counter, "control", False, 2, 0, 24)


@pytest.mark.parametrize("sample_value,forged_count", [(.02, 1), (.08, 0)])
def test_validate_window_rejects_forged_previous_range_counts_even_when_aggregate_is_recomputed(
        sample_value, forged_count):
    # This acts like a closed row whose aggregate was recomputed after tampering:
    # the validator must bind exposure to the captured command extrema, not only
    # trust a self-consistent per-step/summary count pair.
    command = torch.zeros(4, 4)
    command[0, 0] = sample_value
    env = _Env(step=12000, worlds=4, command=command)
    collector = schedule.HeadGoalTelemetry("staged", expected_worlds=4)
    collector.before(env)
    env.advance()
    collector.consume(env)
    report = collector.drain(expected_steps=1)
    assert schedule.validate_window(report, "staged", False, 4, 12000, 1)["valid"] is True

    forged = copy.deepcopy(report)
    forged["step_records"][0]["command_outside_previous_range_count"][0] = forged_count
    forged["command_outside_previous_range_count"][0] = forged_count
    with pytest.raises(ValueError, match="previous-stage exposure count agrees"):
        schedule.validate_window(forged, "staged", False, 4, 12000, 1)

