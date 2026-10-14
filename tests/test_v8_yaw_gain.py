"""Single-factor yaw-bias gain lesson contracts."""
import copy
import json

import pytest
import torch

from mjlab_microduck import v8_gait_scale_contrast as driver
from mjlab_microduck import v8_motor_headroom as headroom
from mjlab_microduck import v8_steering_budget as budget
from mjlab_microduck import v8_yaw_gain as lesson
from test_v8_gait_scale_contrast import _FakeFrozenActor, _payload, _runner


@pytest.mark.parametrize("mode", lesson.MODES)
def test_only_reward_gain_is_masked_and_full_recipes_remain_distinct(mode):
    proof = lesson.recipe_match(mode)
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert proof["full_recipe_hashes_distinct"] is True
    assert proof["comparison_mask"] == ["reward yaw_bias.weight"]
    assert proof["weights"] == {"control": -1.0, "reduced": -.5}
    assert proof["taus"] == {"control": .5, "reduced": .5}
    assert proof["stage_specs"]["control"] == proof["stage_specs"]["reduced"]
    assert driver.stage_spec(lesson, mode, "control") == lesson.MODES[mode]
    assert driver.stage_spec(lesson, mode, "reduced") == lesson.MODES[mode]
    assert driver.bias_weight(lesson, "control") == -1.0
    assert driver.bias_weight(lesson, "reduced") == -.5

    left = json.loads(lesson._canonical(mode, "control", mask=False))
    right = json.loads(lesson._canonical(mode, "reduced", mask=False))
    assert left["environment"]["rewards"]["yaw_bias"]["weight"] == -1.0
    assert right["environment"]["rewards"]["yaw_bias"]["weight"] == -.5
    left["environment"]["rewards"]["yaw_bias"]["weight"] = "<declared-gain>"
    right["environment"]["rewards"]["yaw_bias"]["weight"] = "<declared-gain>"
    assert left == right

    cfgs = [lesson.prepare_config(mode, arm) for arm in lesson.ARMS]
    (control_cfg, control_agent), (reduced_cfg, reduced_agent) = cfgs
    assert control_agent.max_iterations == reduced_agent.max_iterations == lesson.MODES[mode][1]
    assert control_cfg.rewards["yaw_bias"].params["tau_s"] == .5
    assert reduced_cfg.rewards["yaw_bias"].params["tau_s"] == .5
    assert control_cfg.rewards["yaw_bias"].weight == -1.0
    assert reduced_cfg.rewards["yaw_bias"].weight == -.5
    assert control_cfg.rewards["selected_motor_load"].weight == 0
    assert reduced_cfg.rewards["selected_motor_load"].weight == 0
    assert driver.WEIGHT == budget.WEIGHT == -1.0


def test_shared_metric_uses_live_declared_gain_without_mutating_driver(monkeypatch):
    seen = []

    def capture(env, arm, *, weight=driver.WEIGHT):
        seen.append((env, arm, weight))
        return "captured"

    monkeypatch.setattr(driver, "capture_metric", capture)
    from types import SimpleNamespace

    def env_for(weight):
        term = SimpleNamespace(weight=weight)
        return SimpleNamespace(reward_manager=SimpleNamespace(
            get_term_cfg=lambda _name: term))

    control, reduced = env_for(-1.0), env_for(-.5)
    assert lesson._capture_metric(control) == lesson._capture_metric(reduced) == "captured"
    assert seen == [(control, "gait-scale", -1.0), (reduced, "gait-scale", -.5)]
    assert driver.WEIGHT == -1.0
    with pytest.raises(ValueError, match="declared gain arms"):
        lesson._capture_metric(env_for(-.25))


def test_observer_uses_explicit_gain_and_retains_the_zero_load_arm():
    control = lesson.Observer("foundation", "control")
    reduced = lesson.Observer("foundation", "reduced")
    assert control.bias_weight == -1.0
    assert reduced.bias_weight == -.5
    assert control.selected_arm == reduced.selected_arm == "control"
    assert control.report_arm == "control" and reduced.report_arm == "reduced"
    assert control.protocol == reduced.protocol == lesson.PROTOCOL
    assert control.receipt_profile == reduced.receipt_profile == lesson.KL_PROFILE
    assert control.profile == reduced.profile == "gait-scale-contrast"
    assert lesson.CAPS["foundation"] == 2100
    assert str(lesson.REJECTED_BUDGET_DIR).startswith("/home/converge/.local/state/")


def test_reduced_first_update_kl_report_binds_reduced_weight(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
            "kl_mean_mean": .01, "kl_mean_max": .02},
            "target_curriculum": {"action_rate_weight": -.4,
                "head_pose_bias_weight": 1.0, "rel_standing_envs": .10,
                "head_pose_ranges": [[-.39,.39],[-.39,.39],[-.49,.49],[-.11,.11]],
                "randomize_com": {"ranges": [-.005,.005]},
                "randomize_head_com": {"ranges": [-.005,.005]}}}

    monkeypatch.setattr(driver.learning.Observer, "update", parent_update)
    observer = lesson.Observer("smoke", "reduced")
    observer.output, observer.source, observer.initial_lr = tmp_path, "a" * 40, .0001
    observer.runner = SimpleNamespace(alg=SimpleNamespace(optimizer=SimpleNamespace(state={})))
    observer.bias_costs = [0.] * 24
    observer.bias_rewards = [0.] * 24
    observer.signed_biases = [0.] * 24
    observer.raw_errors = [0.] * 24
    observer.signed_raw_errors = [0.] * 24
    observer.motor_costs = [0.] * 24
    observer.motor_rewards = [0.] * 24
    observer.motor_forces = [torch.zeros((8, len(headroom.ACTUATORS)))] * 24
    observer.bucket_counts = dict.fromkeys(driver.exposure.EXPOSURE_BINS, 0)
    observer.bucket_counts["idle"] = 24 * 8
    env = SimpleNamespace(common_step_counter=24, num_envs=8)

    row = observer.update(env)
    report = json.loads((tmp_path / "first-update-kl.json").read_text())
    assert report["passed"] is True
    assert report["fixed_reward_weight"] == -.5
    assert row["yaw_bias_weight"] == -.5


def test_parent_reader_authenticates_rejected_budget_evidence_before_original_parent(monkeypatch):
    comparison = {"decision": "budget-lesson-rejected", "groups": {
        "control": {"complete_moving_route_worlds": 7, "moving_speed_passes": 66,
                    "groups": [{"command": [.2, .2],
                                "full_heading_error_rad_s": .05288204378424289}]},
        "extended": {"complete_moving_route_worlds": 15, "moving_speed_passes": 69,
                     "groups": [{"command": [.2, .2],
                                 "full_heading_error_rad_s": .07641887840478805}]}}}
    summary = {"source": lesson.REJECTED_BUDGET_SOURCE,
        "protocol": budget.PROTOCOL,
        "status": "verified-evidence-not-policy-acceptance",
        "comparison": comparison,
        "parent_checkpoint_sha256": lesson.PARENT_SHA,
        "comparison_sha256": "decision-hash"}
    manifest = {"source": lesson.REJECTED_BUDGET_SOURCE,
        "total_bytes": 123,
        "files": {"budget-decision.json": {"sha256": "decision-hash"}}}
    calls = []

    def owned_payload(path, sha):
        calls.append((path, sha))
        if path == lesson.REJECTED_BUDGET_SUMMARY:
            return json.dumps(summary).encode()
        if path == lesson.REJECTED_BUDGET_MANIFEST:
            return json.dumps(manifest).encode()
        raise AssertionError(f"unexpected evidence path: {path}")

    payload = {"sealed": "original-parent"}
    monkeypatch.setattr(headroom, "_owned_payload", owned_payload)
    monkeypatch.setattr(budget, "_read_parent", lambda: payload)
    assert lesson._read_parent() is payload
    assert calls == [
        (lesson.REJECTED_BUDGET_SUMMARY, lesson.REJECTED_BUDGET_SUMMARY_SHA),
        (lesson.REJECTED_BUDGET_MANIFEST, lesson.REJECTED_BUDGET_MANIFEST_SHA),
    ]


def test_initializer_preserves_original_parent_optimizer_and_critic_counts(monkeypatch, tmp_path):
    from mjlab_microduck import v8_actor_seed

    payload = _payload()
    for state in payload["optimizer_state_dict"]["state"].values():
        state["step"].fill_(60000)
    payload["critic_state_dict"]["obs_normalizer.count"].fill_(18432000)
    payload["optimizer_state_dict"]["param_groups"][0]["lr"] = .00011390625000000005
    monkeypatch.setattr(lesson, "_read_parent", lambda: payload)
    monkeypatch.setattr(v8_actor_seed, "FrozenV8Actor", _FakeFrozenActor)
    runner, env = _runner()
    report = lesson.initialize(runner, env, tmp_path)
    assert report["parent_sha256"] == lesson.PARENT_SHA
    assert report["adam_step_min"] == report["adam_step_max"] == 60000
    assert report["critic_normalizer_count"] == 18432000
    assert report["local_iteration"] == report["local_common_step_counter"] == 0
    assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"]
    assert runner.alg.learning_rate == .00011390625000000005


def test_comparison_relabels_budget_diagnostic_without_changing_thresholds(monkeypatch):
    source = {"protocol": budget.PROTOCOL + ":comparison-v1",
        "groups": {"control": {"complete_moving_route_worlds": 10},
                   "extended": {"complete_moving_route_worlds": 17}},
        "failures": [], "decision": "budget-lesson-useful-not-qualified",
        "thresholds": {"additional_complete_moving_worlds": 6,
                       "paired_heading_allowance_rad_s": .01,
                       "paired_xy_allowance_mps": .02},
        "policy_acceptance": False, "physical_motion_authorized": False}
    monkeypatch.setattr(budget, "diagnostic_comparison", lambda _a, _b: copy.deepcopy(source))
    report = lesson.diagnostic_comparison([], [])
    assert report["protocol"] == lesson.PROTOCOL + ":comparison-v1"
    assert "extended" not in report["groups"]
    assert report["groups"]["reduced"]["complete_moving_route_worlds"] == 17
    assert report["decision"] == "gain-lesson-useful-not-qualified"
    assert report["thresholds"] == source["thresholds"]
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False


def test_safety_regression_gets_gain_protocol_label(monkeypatch):
    source = {"groups": {"control": {}, "extended": {}},
        "failures": [["extended", "motor-range-idle-safety-regression"]],
        "thresholds": {}, "decision": "budget-lesson-rejected",
        "policy_acceptance": False, "physical_motion_authorized": False}
    monkeypatch.setattr(budget, "diagnostic_comparison", lambda _a, _b: copy.deepcopy(source))
    report = lesson.diagnostic_comparison([], [])
    assert report["failures"] == [["reduced", "motor-range-idle-safety-failure"]]
    assert report["decision"] == "gain-lesson-rejected"
