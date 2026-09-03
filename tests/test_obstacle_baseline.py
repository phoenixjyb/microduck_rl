"""Static bounds and config tests for the obstacle baseline evaluator."""

import json
from pathlib import Path

import pytest

from mjlab_microduck.obstacle_baseline import (
    MAX_BASELINE_ENVS,
    MAX_BASELINE_SEEDS,
    MAX_BASELINE_STEPS,
    _mean_or_none,
    _resolved_attempt_metrics,
    _weighted_case_mean,
    prepare_baseline_configs,
    run_baseline,
    validate_baseline_bounds,
)
from mjlab_microduck.obstacle_protocol import OA0_TASK_ID


def test_baseline_config_uses_deterministic_straight_command():
    env_cfg, agent_cfg = prepare_baseline_configs(8, 0.5)
    twist = env_cfg.commands["twist"]
    assert env_cfg.scene.num_envs == 8
    assert env_cfg.scene.terrain.num_envs == 8
    assert twist.ranges.lin_vel_x == (0.5, 0.5)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert twist.heading_command is False
    assert twist.ranges.heading is None
    assert twist.rel_standing_envs == 0.0
    assert "push_robot" not in env_cfg.events
    assert env_cfg.curriculum == {}
    assert agent_cfg.upload_model is False


def test_assisted_baseline_uses_oa0_task_and_keeps_attempt_termination():
    env_cfg, agent_cfg = prepare_baseline_configs(8, 0.3, OA0_TASK_ID)
    assert env_cfg.commands["twist"].ranges.lin_vel_x == (0.3, 0.3)
    assert "obstacle_attempt_timeout" in env_cfg.terminations
    assert agent_cfg.experiment_name == "run_obstacle_assisted"


def test_baseline_retains_exact_o1_protocol(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        "mjlab_microduck.obstacle_baseline._run_case",
        lambda *_args: {
            "collision_events": 1,
            "fall_events": 0,
            "timeout_events": 0,
            "attempt_timeout_events": 0,
            "nan_termination_events": 0,
            "clean_pass_events": 1,
            "nonfinite_steps": 0,
            "mean_clearance_m": 0.1,
            "mean_forward_speed_mps": 0.3,
            "pre_obstacle_samples": 1,
            "pre_obstacle_route_speed_mps": 0.3,
            "mean_passage_time_s": 4.0,
            "mean_collision_time_s": 2.0,
            "mean_pass_lateral_excursion_m": 0.2,
            "mean_collision_lateral_excursion_m": 0.1,
            "mean_success_route_return_error_m": 0.1,
            "success_route_return_events": 1,
        },
    )

    output = run_baseline(checkpoint, tmp_path / "output", seeds=(41,))
    summary = json.loads(output.read_text())

    assert summary["evaluation_protocol"]["name"] == "O1-centered-exact-v1"
    assert summary["evaluation_protocol"]["obstacle_lateral_range_m"] == [0.0, 0.0]


def test_baseline_retains_exact_oa0_protocol(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        "mjlab_microduck.obstacle_baseline._run_case",
        lambda *_args: {
            "collision_events": 0,
            "fall_events": 0,
            "timeout_events": 0,
            "attempt_timeout_events": 1,
            "nan_termination_events": 0,
            "clean_pass_events": 1,
            "nonfinite_steps": 0,
            "mean_clearance_m": 0.2,
            "mean_forward_speed_mps": 0.3,
            "pre_obstacle_samples": 1,
            "pre_obstacle_route_speed_mps": 0.3,
            "mean_passage_time_s": 5.0,
            "mean_collision_time_s": None,
            "mean_pass_lateral_excursion_m": 0.3,
            "mean_collision_lateral_excursion_m": None,
            "mean_success_route_return_error_m": 0.1,
            "success_route_return_events": 1,
        },
    )

    output = run_baseline(
        checkpoint,
        tmp_path / "output",
        seeds=(41,),
        speed_mps=0.3,
        task_id=OA0_TASK_ID,
    )
    summary = json.loads(output.read_text())

    assert summary["task_id"] == OA0_TASK_ID
    assert summary["evaluation_protocol"]["name"] == (
        "OA0-offset-assisted-exact-v1"
    )
    assert summary["totals"]["attempt_timeout_events"] == 1
    assert summary["mean_success_route_return_error_m"] == 0.1


def test_baseline_rejects_unsupported_obstacle_task():
    with pytest.raises(ValueError, match="unsupported obstacle task"):
        prepare_baseline_configs(8, 0.3, "not-a-task")


@pytest.mark.parametrize(
    "num_envs, steps, seeds",
    [
        (0, 10, (1,)),
        (MAX_BASELINE_ENVS + 1, 10, (1,)),
        (1, 0, (1,)),
        (1, MAX_BASELINE_STEPS + 1, (1,)),
        (1, 10, ()),
        (1, 10, tuple(range(MAX_BASELINE_SEEDS + 1))),
    ],
)
def test_baseline_rejects_unbounded_work(num_envs, steps, seeds):
    with pytest.raises(ValueError):
        validate_baseline_bounds(num_envs, steps, seeds)


def test_baseline_rejects_nonpositive_speed():
    with pytest.raises(ValueError, match="speed_mps"):
        prepare_baseline_configs(8, 0.0)


def test_baseline_rejects_empty_purpose_before_resolving_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="purpose"):
        run_baseline(tmp_path / "missing.pt", tmp_path / "out", purpose="  ")


def test_diagnostic_means_are_count_weighted_and_empty_safe():
    cases = [
        {"events": 2, "time": 1.0},
        {"events": 1, "time": 4.0},
    ]
    assert _mean_or_none(3.0, 2) == 1.5
    assert _mean_or_none(0.0, 0) is None
    assert _weighted_case_mean(cases, "time", "events") == 2.0
    assert _weighted_case_mean(
        [{"events": 0, "time": None}], "time", "events"
    ) is None


def test_resolved_attempt_metrics_report_pass_and_collision_rates():
    assert _resolved_attempt_metrics(3, 7) == {
        "resolved_attempts": 10,
        "clean_pass_rate": 0.7,
        "collision_rate": 0.3,
        "attempt_timeout_rate": 0.0,
    }


def test_resolved_attempt_metrics_are_empty_safe():
    assert _resolved_attempt_metrics(0, 0) == {
        "resolved_attempts": 0,
        "clean_pass_rate": None,
        "collision_rate": None,
        "attempt_timeout_rate": None,
    }


def test_resolved_attempt_metrics_count_attempt_timeouts_as_failures():
    assert _resolved_attempt_metrics(1, 7, 2) == {
        "resolved_attempts": 10,
        "clean_pass_rate": 0.7,
        "collision_rate": 0.1,
        "attempt_timeout_rate": 0.2,
    }


@pytest.mark.parametrize(
    "collision_events, clean_pass_events, attempt_timeout_events",
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
)
def test_resolved_attempt_metrics_reject_negative_counts(
    collision_events, clean_pass_events, attempt_timeout_events
):
    with pytest.raises(ValueError, match="non-negative"):
        _resolved_attempt_metrics(
            collision_events, clean_pass_events, attempt_timeout_events
        )
