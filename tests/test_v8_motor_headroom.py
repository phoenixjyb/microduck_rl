"""CPU recipe and exact-continuation contracts; not GPU/policy acceptance."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import v8_gait_scale_contrast as driver
from mjlab_microduck import v8_motor_headroom as lesson
from mjlab_microduck.tasks import mdp
from test_v8_gait_scale_contrast import _payload, _runner, _FakeFrozenActor


@pytest.mark.parametrize("mode", lesson.MODES)
def test_only_weight_changes_and_parent_profile_is_preserved(mode):
    control, ca = lesson.prepare_config(mode, "control")
    treatment, ta = lesson.prepare_config(mode, "headroom")
    parent, pa = driver.prepare_config(mode, "gait-scale")
    assert control.rewards[lesson.TERM].weight == 0
    assert treatment.rewards[lesson.TERM].weight == -.25
    assert control.rewards[lesson.TERM].params == lesson.PARAMS
    assert control.rewards[lesson.TERM].func is mdp.selected_motor_torque_load_cost
    assert control.rewards["yaw_bias"].params["tau_s"] == .5
    assert treatment.rewards["yaw_bias"].params["tau_s"] == .5
    control.rewards.pop(lesson.TERM)
    control.metrics.pop("microduck_selected_motor_load")
    ca.experiment_name = pa.experiment_name
    assert driver._full_canonical(control, ca) == driver._full_canonical(parent, pa)
    proof = lesson.recipe_match(mode)
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert proof["control_full_recipe_sha256"] != proof["treatment_full_recipe_sha256"]
    assert proof["comparison_mask"] == ["reward selected_motor_load.weight"]
    assert json.loads(json.dumps(proof)) == proof
    frozen = lesson.frozen_profile_report("a"*40, "headroom", mode)
    assert json.loads(json.dumps(frozen)) == frozen
    assert frozen["parent_sha256"] == lesson.PARENT_SHA


def new_payload():
    value = _payload()
    for state in value["optimizer_state_dict"]["state"].values():
        state["step"].fill_(40000)
    value["critic_state_dict"]["obs_normalizer.count"].fill_(12288000)
    value["optimizer_state_dict"]["param_groups"][0]["lr"] = 2.25e-5
    return value


def test_explicit_initializer_preserves_new_parent_and_historical_globals(monkeypatch, tmp_path):
    from mjlab_microduck import v8_actor_seed
    value = new_payload()
    driver.validate_parent_payload(value, adam_step=40000, critic_count=12288000)
    monkeypatch.setattr(lesson, "_read_parent", lambda: value)
    monkeypatch.setattr(v8_actor_seed, "FrozenV8Actor", _FakeFrozenActor)
    old = (driver.PROTOCOL, driver.PARENT_SHA, driver.PARENT_ADAM_STEP,
           driver.PARENT_CRITIC_COUNT, copy.deepcopy(driver.TAUS))
    runner, env = _runner()
    report = lesson.initialize(runner, env, tmp_path)
    assert report["protocol"] == lesson.PROTOCOL + ":initialization-v1"
    assert report["parent_sha256"] == lesson.PARENT_SHA
    assert report["adam_step_min"] == report["adam_step_max"] == 40000
    assert report["critic_normalizer_count"] == 12288000
    assert report["local_iteration"] == report["local_common_step_counter"] == 0
    assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"] == 2.25e-5
    assert old == (driver.PROTOCOL, driver.PARENT_SHA, driver.PARENT_ADAM_STEP,
                   driver.PARENT_CRITIC_COUNT, driver.TAUS)
    with pytest.raises(ValueError, match="Adam state"):
        driver.validate_parent_payload(value)  # Old protocol still expects step20000.


@pytest.mark.parametrize("mutation", ("sha", "owner", "symlink"))
def test_sealed_artifact_refuses_wrong_hash_owner_and_symlink(tmp_path, monkeypatch, mutation):
    from hashlib import sha256
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    expected = sha256(path.read_bytes()).hexdigest()
    if mutation == "sha": expected = "0"*64
    if mutation == "owner": monkeypatch.setattr(lesson.os, "getuid", lambda: -1)
    if mutation == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        lesson._owned_payload(path, expected)


@pytest.mark.parametrize("arm", lesson.ARMS)
def test_metric_captures_actual_pre_reset_reward_and_control_is_inactive(arm):
    weight = lesson.LOAD_WEIGHTS[arm]
    force = torch.tensor([[.3, .1, .4, .5]])
    asset = SimpleNamespace(data=SimpleNamespace(actuator_force=force),
        find_actuators=lambda names, preserve_order: ([0, 1, 2, 3], list(names)))
    env = SimpleNamespace(scene={"robot": asset}, num_envs=1, device="cpu", common_step_counter=3)
    cost = mdp.selected_motor_torque_load_cost(env, **lesson.PARAMS)
    term = SimpleNamespace(func=mdp.selected_motor_torque_load_cost, params=lesson.PARAMS, weight=weight)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: term,
        active_terms=[lesson.TERM], _step_reward=(weight*cost)[:, None])
    assert torch.equal(lesson.capture_motor_metric(env), torch.zeros(1))
    assert env._microduck_selected_load_snapshot["step"] == 3
    assert torch.equal(env._microduck_selected_load_snapshot["cost"], cost)
    assert torch.equal(env._microduck_selected_load_snapshot["weighted"], weight*cost)
    with pytest.raises(ValueError, match="snapshot consumed"):
        lesson.capture_motor_metric(env)


def test_control_reward_cannot_silently_be_active():
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=0,
        scene={"robot": SimpleNamespace(data=SimpleNamespace(actuator_force=torch.zeros(1,4)),
            find_actuators=lambda names, preserve_order: ([0,1,2,3], list(names)))})
    env.reward_manager = SimpleNamespace(active_terms=[lesson.TERM],
        _step_reward=torch.ones(1,1), get_term_cfg=lambda name:
        SimpleNamespace(func=mdp.selected_motor_torque_load_cost, params=lesson.PARAMS, weight=0))
    with pytest.raises(ValueError, match="actually skipped"):
        lesson.capture_motor_metric(env)


def test_real_manager_keeps_registered_zero_weight_column_but_skips_callable():
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager
    def forbidden(env):
        raise AssertionError("zero-weight reward must not execute")
    env = SimpleNamespace(num_envs=1, device="cpu")
    manager = RewardManager({lesson.TERM: RewardTermCfg(func=forbidden, weight=0.)}, env)
    assert lesson.TERM in manager.active_terms
    assert torch.equal(manager.compute(.02), torch.zeros(1))
    assert torch.equal(manager._step_reward, torch.zeros(1,1))


def test_recipe_injection_is_explicit_not_global_mutation():
    assert driver._bindings(None, "PROTOCOL PARENT_SHA") == (driver.PROTOCOL, driver.PARENT_SHA)
    assert driver._bindings(lesson, "PROTOCOL PARENT_SHA") == (lesson.PROTOCOL, lesson.PARENT_SHA)
    with pytest.raises(AttributeError):
        driver._bindings(SimpleNamespace(), "PROTOCOL")
    assert lesson.Observer("smoke", "control").receipt_profile == lesson.KL_PROFILE
    assert lesson.Observer("smoke", "headroom").report_arm == "headroom"
    assert driver.Observer("smoke", "gait-scale").protocol == driver.PROTOCOL


@pytest.fixture(scope="module")
def comparison_rows():
    from test_gait_path_qualification import make_case_trace
    rows = []
    for seed in driver.base.SEEDS:
        for speed, yaw in driver.base.CASES:
            row, trace = make_case_trace(speed=speed, yaw=yaw, seed=seed)
            row["gait_path_trace"] = trace
            row["response_diagnostics"]["abs_force_p99_nm_by_joint"] = [.4]*14
            rows.append(row)
    return rows


def test_motor_comparison_five_percent_gate_is_deterministic_not_policy_acceptance(comparison_rows):
    treatment = copy.deepcopy(comparison_rows)
    for row in treatment:
        row["response_diagnostics"]["abs_force_p99_nm_by_joint"] = [.37]*14
    report = lesson.comparison(comparison_rows, treatment)
    assert report["decision"] == "headroom-lesson-useful-not-qualified"
    assert report["failures"] == []
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False
    assert lesson.comparison(comparison_rows, comparison_rows)["failures"] == [
        "worst-selected-moving-p99-not-reduced-five-percent"]


def test_motor_comparison_cannot_trade_speed_for_headroom(comparison_rows):
    treatment = copy.deepcopy(comparison_rows)
    for row in treatment:
        row["response_diagnostics"]["abs_force_p99_nm_by_joint"] = [.37]*14
        if row["speed"] != 0:
            row["speed_mae"] = [.12]*8
    result = lesson.comparison(comparison_rows, treatment)
    assert result["decision"] == "headroom-lesson-rejected"
    assert "paired-moving-speed-pass-count-regression" in result["failures"]
    assert sum(f[-1] == "paired-speed-mae-regression" for f in result["failures"]
               if isinstance(f, list)) == 3


def test_motor_comparison_rejects_matrix_and_invalid_named_force(comparison_rows):
    with pytest.raises(ValueError, match="ordered complete"):
        lesson.comparison(comparison_rows[::-1], comparison_rows)
    bad = copy.deepcopy(comparison_rows)
    bad[1]["response_diagnostics"]["abs_force_p99_nm_by_joint"][3] = float("nan")
    with pytest.raises(ValueError, match="finite nonnegative"):
        lesson.comparison(comparison_rows, bad)
