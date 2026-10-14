"""Contracts for the staged head-goal continuation lesson."""
from __future__ import annotations

import json
import copy
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import v8_head_goal_schedule as schedule
from mjlab_microduck import v8_motor_headroom as headroom
from mjlab_microduck import v8_staged_head_goals as lesson
from mjlab_microduck import v8_yaw_gain as gain
from mjlab_microduck import v8_head_goal_boundary_probe as boundary_probe
from mjlab_microduck.tasks import mdp


@pytest.mark.parametrize("mode", lesson.MODES)
def test_recipe_masks_only_initial_head_ranges_and_stage_history(mode):
    proof = lesson.recipe_match(mode)
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert proof["full_recipe_hashes_distinct"] is True
    assert proof["comparison_mask"] == [
        "commands.head_pose.ranges", "curriculum.head_pose_range.range_stages"]
    assert proof["range_history"]["control"]["initial"] == list(schedule.PARENT_RANGES)
    assert proof["range_history"]["staged"]["initial"] == list(schedule.BEGINNER_RANGES)
    assert proof["range_history"]["staged"]["stages"] == schedule.stages_for("staged")

    left = json.loads(lesson._canonical(mode, "control", mask=False))
    right = json.loads(lesson._canonical(mode, "staged", mask=False))
    left["environment"]["commands"]["head_pose"]["ranges"] = "<declared>"
    right["environment"]["commands"]["head_pose"]["ranges"] = "<declared>"
    left["environment"]["curriculum"]["head_pose_range"]["params"]["range_stages"] = "<declared>"
    right["environment"]["curriculum"]["head_pose_range"]["params"]["range_stages"] = "<declared>"
    assert left == right


def test_both_arms_keep_parent_nonhead_profile_and_frozen_yaw_terms():
    control, control_agent = lesson.prepare_config("foundation", "control")
    staged, staged_agent = lesson.prepare_config("foundation", "staged")
    assert control_agent.max_iterations == staged_agent.max_iterations == 1000
    for cfg, expected in ((control, schedule.PARENT_RANGES),
                          (staged, schedule.BEGINNER_RANGES)):
        assert tuple(tuple(pair) for pair in cfg.commands["head_pose"].ranges) == expected
        assert cfg.rewards["yaw_bias"].weight == -1.0
        assert cfg.rewards["yaw_bias"].params["tau_s"] == .5
        assert cfg.rewards["selected_motor_load"].weight == 0.
        assert cfg.rewards["head_pose_bias"].weight == 1.0
        assert cfg.rewards["head_pose_tracking"].weight == 2.0
        assert cfg.rewards["head_pose_tracking"].params["std"] == .5
        assert cfg.curriculum["head_pose_range"].params["range_stages"] == schedule.stages_for(
            "control" if cfg is control else "staged")
    for arm in lesson.ARMS:
        report = lesson.frozen_profile_report("a" * 40, arm, "foundation")
        assert report["live_values"]["head_pose_ranges"] == [
            list(pair) for pair in schedule.ranges_at(arm, 0)]
        assert report["curriculum_stage_counts"]["head_pose_range"] == len(
            schedule.stages_for(arm))
        assert report["curriculum_clock_start"] == 0
        assert report["curriculum_clock_policy"] == (
            "local-common-step-full-training-range-history")


@pytest.mark.parametrize("step,expected", [
    (0, schedule.BEGINNER_RANGES),
    (11_999, schedule.BEGINNER_RANGES),
    (12_000, schedule.INTERMEDIATE_RANGES),
    (17_999, schedule.INTERMEDIATE_RANGES),
    (18_000, schedule.PARENT_RANGES),
    (23_999, schedule.PARENT_RANGES),
])
def test_observer_uses_tuple_ranges_at_the_full_learner_boundaries(step, expected):
    observer = lesson.Observer("foundation", "staged")
    assert observer.expected_head_ranges(step) == expected
    assert isinstance(observer.expected_head_ranges(step), tuple)
    assert all(isinstance(pair, tuple) for pair in observer.expected_head_ranges(step))


def test_control_observer_keeps_original_parent_ranges_for_all_steps():
    observer = lesson.Observer("foundation", "control")
    for step in (0, 11_999, 12_000, 17_999, 18_000, 23_999):
        assert observer.expected_head_ranges(step) == schedule.PARENT_RANGES


def test_comparison_relabels_staged_arm_and_preserves_paired_thresholds(monkeypatch):
    source = {"groups": {"control": {"complete_moving_route_worlds": 9},
                         "reduced": {"complete_moving_route_worlds": 16}},
        "failures": [["reduced", "heading-endpoint-growth"]],
        "thresholds": {"additional_complete_moving_worlds": 6,
                       "paired_heading_allowance_rad_s": .01,
                       "paired_xy_allowance_mps": .02},
        "weights": {"control": -1.0, "reduced": -.5},
        "decision": "gain-lesson-rejected", "policy_acceptance": False,
        "physical_motion_authorized": False}
    monkeypatch.setattr(gain, "diagnostic_comparison",
                        lambda _control, _staged: copy.deepcopy(source))
    report = lesson.diagnostic_comparison([], [])
    assert report["protocol"] == lesson.PROTOCOL + ":comparison-v1"
    assert report["groups"]["staged"]["complete_moving_route_worlds"] == 16
    assert "reduced" not in report["groups"]
    assert report["failures"] == [["staged", "heading-endpoint-growth"]]
    assert report["thresholds"] == source["thresholds"]
    assert report["weights"] == {"control": -1.0, "staged": -1.0}
    assert report["decision"] == "staged-head-goals-lesson-rejected"
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False


@pytest.mark.parametrize("mode", ["benchmark", "foundation"])
def test_cli_requires_both_boundary_receipts_before_later_training_modes(monkeypatch, mode):
    events = []
    monkeypatch.setattr(boundary_probe, "validate_receipt", lambda root, arm, source, **kwargs:
                        events.append(("boundary", arm, source)))
    monkeypatch.setattr(lesson.driver, "main", lambda **kwargs:
                        events.append(("driver", kwargs["experiment"])) or "delegated")
    monkeypatch.setattr(sys, "argv", ["v8_staged_head_goals", "--source", "a"*40,
        "--run-id", "run", "--arm", "control", "--mode", mode])
    assert lesson.main() == "delegated"
    assert events == [("boundary", "control", "a"*40),
                      ("boundary", "staged", "a"*40), ("driver", lesson)]


@pytest.mark.parametrize("argv", [
    ["v8_staged_head_goals", "--source", "a"*40, "--run-id", "run", "--preflight"],
    ["v8_staged_head_goals", "--source", "a"*40, "--run-id", "run",
     "--arm", "control", "--mode", "smoke"],
])
def test_preflight_and_ordinary_smoke_do_not_require_boundary_receipts(monkeypatch, argv):
    events = []
    monkeypatch.setattr(boundary_probe, "validate_receipt", lambda *args, **kwargs:
                        events.append(("boundary", args)))
    monkeypatch.setattr(lesson.driver, "main", lambda **kwargs:
                        events.append(("driver", kwargs["experiment"])) or "delegated")
    monkeypatch.setattr(sys, "argv", argv)
    assert lesson.main() == "delegated"
    assert events == [("driver", lesson)]


def test_boundary_smoke_cli_validates_both_standard_smokes_then_delegates(monkeypatch):
    events = []
    args = ["--source", "a"*40, "--run-id", "run", "--arm", "staged"]
    monkeypatch.setattr(lesson, "verify_source", lambda parsed, *, training:
                        events.append(("source", parsed.run_id, training)))
    monkeypatch.setattr(lesson, "validate_training_receipt", lambda root, arm, mode, source:
                        events.append(("smoke", arm, mode, source)) or {"arm": arm})
    monkeypatch.setattr(boundary_probe, "main", lambda *, argv, experiment:
                        events.append(("probe", argv, experiment)) or "probe-run")
    monkeypatch.setattr(sys, "argv", ["v8_staged_head_goals", "boundary-smoke", *args])
    assert lesson.main() == "probe-run"
    assert events == [
        ("source", "run", True),
        ("smoke", "control", "smoke", "a"*40),
        ("smoke", "staged", "smoke", "a"*40),
        ("probe", args, lesson),
    ]


def test_parent_reader_authenticates_gain_summary_and_inventory_before_parent(monkeypatch):
    summary = {"source": lesson.REJECTED_GAIN_SOURCE, "protocol": gain.PROTOCOL,
        "status": "verified-evidence-not-policy-acceptance",
        "comparison": {"decision": "gain-lesson-rejected"},
        "parent_checkpoint_sha256": lesson.PARENT_SHA, "comparison_sha256": "decision-seal"}
    manifest = {"source": lesson.REJECTED_GAIN_SOURCE, "total_bytes": 123,
        "files": {"gain-decision.json": {"sha256": "decision-seal"},
                   "control/foundation/model_999.pt": {"sha256": lesson.PARENT_SHA}}}
    calls = []

    def owned_payload(path, expected_sha):
        calls.append((path, expected_sha))
        if path == lesson.REJECTED_GAIN_SUMMARY:
            return json.dumps(summary).encode()
        if path == lesson.REJECTED_GAIN_MANIFEST:
            return json.dumps(manifest).encode()
        raise AssertionError(f"unexpected sealed input {path}")

    parent = {"source": "original-headroom-parent"}
    monkeypatch.setattr(headroom, "_owned_payload", owned_payload)
    monkeypatch.setattr(gain, "_read_parent", lambda: parent)
    assert lesson._read_parent() is parent
    assert calls == [
        (lesson.REJECTED_GAIN_SUMMARY, lesson.REJECTED_GAIN_SUMMARY_SHA),
        (lesson.REJECTED_GAIN_MANIFEST, lesson.REJECTED_GAIN_MANIFEST_SHA),
    ]


def test_parent_reader_fails_closed_before_gain_parent_on_bad_manifest(monkeypatch):
    summary = {"source": lesson.REJECTED_GAIN_SOURCE, "protocol": gain.PROTOCOL,
        "status": "verified-evidence-not-policy-acceptance",
        "comparison": {"decision": "gain-lesson-rejected"},
        "parent_checkpoint_sha256": lesson.PARENT_SHA, "comparison_sha256": "decision-seal"}
    manifest = {"source": lesson.REJECTED_GAIN_SOURCE, "total_bytes": 123,
        "files": {"gain-decision.json": {"sha256": "wrong"},
                   "control/foundation/model_999.pt": {"sha256": lesson.PARENT_SHA}}}
    payloads = {lesson.REJECTED_GAIN_SUMMARY: json.dumps(summary).encode(),
                lesson.REJECTED_GAIN_MANIFEST: json.dumps(manifest).encode()}
    monkeypatch.setattr(headroom, "_owned_payload", lambda path, _sha: payloads[path])
    parent_calls = []
    monkeypatch.setattr(gain, "_read_parent", lambda: parent_calls.append(True))
    with pytest.raises(ValueError, match="authenticated rejected gain summary"):
        lesson._read_parent()
    assert parent_calls == []


def _head_contract_env():
    tracking = NS(func=mdp.head_pose_tracking,
                  params={"command_name": "head_pose", "std": .5}, weight=2.0)
    bias = NS(func=mdp.head_pose_bias_penalty,
              params={"command_name": "head_pose", "tau_s": 1.0}, weight=1.0)
    command = NS(resampling_time_range=(2.0, 5.0), zero_command_prob=0.0)
    reward = NS(get_term_cfg=lambda name: {
        "head_pose_tracking": tracking, "head_pose_bias": bias}[name],
        active_terms=["head_pose_tracking", "head_pose_bias"],
        _step_reward=torch.tensor([[1.5, -.2], [1.0, -.1]]))
    commands = NS(get_term=lambda name: NS(cfg=command))
    return NS(reward_manager=reward, command_manager=commands, common_step_counter=7,
              num_envs=2), tracking, bias, command


def test_head_contract_and_snapshot_read_only_actual_weighted_columns():
    env, _tracking, _bias, _command = _head_contract_env()
    assert lesson.validate_head_contract(env) is True
    result = lesson.capture_head_reward_snapshot(env, expected_step=7)
    assert result["step"] == 7
    assert result["head_pose_tracking_weighted_reward_mean"] == 1.25
    assert result["head_pose_bias_weighted_reward_mean"] == pytest.approx(-.15)
    assert result["head_pose_tracking_weighted_rewards"] == [1.5, 1.0]
    assert result["head_pose_bias_weighted_rewards"] == pytest.approx([-.2, -.1])


@pytest.mark.parametrize("fault", ["tracking_func", "tracking_params", "tracking_weight",
                                     "bias_func", "bias_params", "bias_weight",
                                     "resampling", "zero_probability"])
def test_head_contract_rejects_inherited_reward_or_sampler_drift(fault):
    env, tracking, bias, command = _head_contract_env()
    if fault == "tracking_func":
        tracking.func = lambda *_a, **_kw: None
    elif fault == "tracking_params":
        tracking.params["std"] = .6
    elif fault == "tracking_weight":
        tracking.weight = 2.1
    elif fault == "bias_func":
        bias.func = lambda *_a, **_kw: None
    elif fault == "bias_params":
        bias.params["tau_s"] = .9
    elif fault == "bias_weight":
        bias.weight = .9
    elif fault == "resampling":
        command.resampling_time_range = (1., 5.)
    else:
        command.zero_command_prob = .01
    with pytest.raises(ValueError, match="remain frozen"):
        lesson.validate_head_contract(env)


def test_observer_callbacks_apply_then_snapshot_and_consume_after_parent(monkeypatch):
    events = []

    class _Telemetry:
        def __init__(self, arm):
            self.arm = arm

        def before(self, env):
            events.append(("telemetry-before", env.common_step_counter))

        def consume(self, env, sample):
            events.append(("telemetry-consume", env.common_step_counter, sample))

        def drain(self, *, expected_steps):
            events.append(("drain", expected_steps))
            return {"arm": self.arm, "steps": expected_steps}

    monkeypatch.setattr(schedule, "HeadGoalTelemetry", _Telemetry)
    monkeypatch.setattr(gain.Observer, "before",
                        lambda self, env: events.append(("parent-before", env.common_step_counter)))
    monkeypatch.setattr(gain.Observer, "step",
                        lambda self, env, sample, extras: events.append(("parent-step", sample)))
    monkeypatch.setattr(gain.Observer, "update",
                        lambda self, env: events.append(("parent-update", env.common_step_counter)) or {})
    monkeypatch.setattr(schedule, "validate_window", lambda *args, **kwargs:
                        events.append(("validate-window", args[1], args[4])))
    monkeypatch.setattr(lesson, "validate_head_contract", lambda env:
                        events.append(("validate-head-contract", env.common_step_counter)))
    monkeypatch.setattr(lesson, "capture_head_reward_snapshot", lambda env, expected_step:
                        events.append(("capture-head-rewards", expected_step)) or {
                            "head_pose_tracking_weighted_reward_mean": 1.0,
                            "head_pose_bias_weighted_reward_mean": -.1})
    monkeypatch.setattr(schedule, "apply_ranges", lambda env, arm: {
        "local_common_step": env.common_step_counter, "stage": 1,
        "ranges": schedule.ranges_at(arm, env.common_step_counter)})

    observer = lesson.Observer("foundation", "staged")
    observer.head_tracking_rewards[:] = [1.0] * 23
    observer.head_bias_rewards[:] = [-.1] * 23
    env = NS(common_step_counter=24, num_envs=64)
    observer.before(env)
    observer.step(env, NS(step=23), "extras")
    row = observer.update(env)
    assert events[:4] == [
        ("telemetry-before", 24), ("validate-head-contract", 24),
        ("parent-before", 24), ("parent-step", NS(step=23))]
    assert events[4:] == [
        ("capture-head-rewards", 24), ("telemetry-consume", 24, NS(step=23)),
        ("validate-head-contract", 24), ("parent-update", 24),
        ("drain", 24),
        ("validate-window", "staged", 0)]
    assert row["head_goal_telemetry"] == {"arm": "staged", "steps": 24}
    assert row["head_pose_tracking_weighted_reward_mean"] == 1.0
    assert row["head_pose_bias_weighted_reward_mean"] == pytest.approx(-.1)
    assert row["head_goal_ranges_at_update"] == [list(pair) for pair in schedule.BEGINNER_RANGES]

