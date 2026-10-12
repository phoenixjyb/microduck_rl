import copy
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_fresh_foundation as fresh
from mjlab_microduck import integrated_fresh_motor_range as recipe
from mjlab_microduck import integrated_motor_refinement as motor


@pytest.mark.parametrize("mode", tuple(recipe.MODES))
def test_only_two_costs_and_read_only_metric_are_added_to_fresh_stock_recipe(mode):
    stock, agent = fresh.prepare_config(mode)
    cfg, revised = recipe.prepare_config(mode)
    clone = copy.deepcopy(cfg)
    assert clone.rewards.pop("motor_load").params == dict(
        rated_stall_torque_nm=.6, soft_limit_fraction=.7, over_limit_gain=4.)
    limit = clone.rewards.pop("joint_limit")
    assert limit.params["margin"] == .15
    assert limit.params["asset_cfg"].joint_names == ("^(left|right)_hip_yaw$",)
    metric = clone.metrics.pop("fresh_bilateral_limit_cost")
    assert metric.func is recipe.capture_limit_metric and not metric.per_substep
    assert clone == stock
    revised.experiment_name = agent.experiment_name
    assert revised == agent
    assert cfg.actions["joint_pos"].clip is None


@pytest.mark.parametrize("backlash", [False, True])
def test_bilateral_selector_resolves_named_joints_with_passive_interleaving(backlash):
    from mjlab.entity import Entity
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    cfg, _ = recipe.prepare_config("foundation")
    robot = Entity(MICRODUCK_WALK_BACKLASH_ROBOT_CFG if backlash else cfg.scene.entities["robot"])
    ids, names = robot.find_joints(cfg.rewards["joint_limit"].params["asset_cfg"].joint_names)
    assert len(ids) == 2 and set(names) == set(recipe.JOINTS)
    assert [robot.joint_names[index] for index in ids] == names
    if backlash:
        assert ids != [0, 9]


def test_frozen_budget_matrix_and_strict_schedules():
    assert recipe.MODES == fresh.MODES
    assert recipe.FINAL_CHECKPOINT == 5999
    assert recipe.evaluate_case is fresh.evaluate_case
    assert recipe.decision is fresh.decision
    assert recipe.EVAL_SEEDS == fresh.EVAL_SEEDS and recipe.EVAL_CASES == fresh.EVAL_CASES
    assert recipe.REQUIRED_COVERAGE == fresh.REQUIRED_COVERAGE
    for schedule in recipe.schedules("foundation"):
        for i, (iteration, weight) in enumerate(schedule):
            assert motor.motor_weight(iteration*24, schedule) == schedule[max(0, i-1)][1]
            assert motor.motor_weight(iteration*24+1, schedule) == weight
    assert recipe.schedules("smoke")[0][-1] == (3, -1.)
    assert recipe.schedules("benchmark") == (((0, -1.),), ((0, -4.),))
    assert recipe.projection_admitted(11.)
    assert not recipe.projection_admitted(12.)
    assert recipe.FOUNDATION_CAP_SECONDS == 10800
    for invalid in (True, 0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            recipe.projection_admitted(invalid)


def fake_env():
    # Reordered named columns with an unselected preceding passive joint.
    q = torch.zeros(64, 3)
    q[0, 2] = .5
    q[1, 1] = .55
    bounds = torch.tensor([[[-.5, .5]]*3]).repeat(64, 1, 1)
    selector = NS(name="robot", joint_ids=[1, 2])
    terms = {name: NS(weight=0.) for name in ("motor_load", "joint_limit", "action_rate_l2", "head_pose_bias")}
    terms["joint_limit"].params = dict(asset_cfg=selector, margin=.15)
    cfg = NS(rel_standing_envs=.02, ranges=((-.05, .05),))
    command = torch.zeros(64, 3)
    command[1:, 0] = .3
    command[2, 2], command[3, 2] = -.2, .2
    env = NS(num_envs=64, device="cpu", common_step_counter=0,
        scene={"robot": NS(joint_names=["passive_dummy", "right_hip_yaw", "left_hip_yaw"],
            data=NS(joint_pos=q, joint_pos_limits=bounds))},
        command_manager=NS(get_command=lambda name: command, get_term=lambda name: NS(cfg=cfg)),
        reward_manager=NS(active_terms=list(terms), get_term_cfg=lambda name: terms[name],
            _step_reward=torch.zeros(64, 4)),
        event_manager=NS(get_term_cfg=lambda name: NS(params=dict(ranges=(-.003, .003)))),
        _microduck_optimizer_audit=object())
    return env


def test_nonpositive_pre_reset_reward_and_bilateral_columns_survive_autoreset(monkeypatch):
    env, observer = fake_env(), recipe.FreshObserver("smoke")
    monkeypatch.setattr(fresh.kl, "optimizer_summary", lambda *args: {"optimizer_minibatches": 20})
    original = env.scene["robot"].data.joint_pos.clone()
    sample = NS(force_nm=torch.full((64, 14), .6), mean_cost=torch.full((64,), 1.36))
    for _ in range(24):
        observer.before_action(env)
        recipe.capture_limit_metric(env)
        cost = env._microduck_fresh_limit_snapshot[0]
        env.reward_manager._step_reward[:, 0] = 1.36*observer.motor.weight
        env.reward_manager._step_reward[:, 1] = cost*observer.limit_weight
        env.scene["robot"].data.joint_pos.zero_()  # terminal reset cannot launder cost
        env.common_step_counter += 1
        observer.after_step(env, sample, {})
        assert env._microduck_fresh_limit_snapshot is None
        env.scene["robot"].data.joint_pos.copy_(original)
    row = observer.after_update(env)
    assert row["limit_cost_mean"] == pytest.approx(.35/64)
    assert row["weighted_limit_reward_mean"] == pytest.approx(-.25*.35/64)
    assert row["motor_weight_min"] == row["motor_weight_max"] == 0
    assert row["motor_cost_mean"] == pytest.approx(1.36)  # visible even at zero weight
    assert row["weighted_motor_reward_mean"] == 0
    assert row["hip_limit_cost_mean"] == pytest.approx(dict(left_hip_yaw=.15/64, right_hip_yaw=.2/64))
    assert row["hip_stop_exposure_mean"] == dict.fromkeys(recipe.JOINTS, 1/64)
    assert row["hip_max_range_overrun_rad"] == pytest.approx(dict(left_hip_yaw=0, right_hip_yaw=.05))
    assert not observer.costs and not observer.column_costs


def test_reported_motor_cost_uses_reward_cap_not_uncapped_stream_cost():
    env, observer = fake_env(), recipe.FreshObserver("benchmark")
    observer.before_action(env)
    recipe.capture_limit_metric(env)
    sample = NS(force_nm=torch.full((64, 14), 1.8), mean_cost=torch.full((64,), 30.16))
    env.reward_manager._step_reward[:, 0] = -10.76  # u=2:4+4*(2-.7)^2
    env.reward_manager._step_reward[:, 1] = env._microduck_fresh_limit_snapshot[0]*observer.limit_weight
    env.common_step_counter += 1
    observer.after_step(env, sample, {})
    assert observer.motor.costs == pytest.approx([10.76])
    assert observer.motor.weighted == pytest.approx([-10.76])


@pytest.mark.parametrize("failure", ["missing", "positive", "nan", "wrong-names"])
def test_bad_metric_or_weighted_reward_fails_closed(failure):
    env, observer = fake_env(), recipe.FreshObserver("smoke")
    observer.before_action(env)
    sample = NS(force_nm=torch.zeros(64, 14), mean_cost=torch.zeros(64))
    if failure != "missing":
        if failure == "wrong-names":
            env.scene["robot"].joint_names[1] = "right_knee"
            with pytest.raises(ValueError, match="bilateral"):
                recipe.capture_limit_metric(env)
            return
        recipe.capture_limit_metric(env)
        cost = env._microduck_fresh_limit_snapshot[0]
        env.reward_manager._step_reward[:, 1] = cost*observer.limit_weight
        env.reward_manager._step_reward[0, 1] = 1 if failure == "positive" else float("nan")
    env.common_step_counter += 1
    with pytest.raises(ValueError, match="bilateral"):
        observer.after_step(env, sample, {})


def test_dynamic_evaluator_uses_declared_receipt_but_keeps_all_twelve_cases(tmp_path, monkeypatch):
    import json
    (tmp_path/"foundation").mkdir()
    (tmp_path/"foundation/result.json").write_text(json.dumps({"marker": 1}))
    calls = []
    checkpoint = tmp_path/"foundation/model_5999.pt"
    checkpoint.write_bytes(b"final")
    monkeypatch.setattr(recipe, "validate_evaluation_receipt", lambda *args: checkpoint)
    def evaluate(path, seed, speed, yaw, health):
        calls.append((seed, speed, yaw))
        return dict(seed=seed, speed=speed, yaw=yaw)
    monkeypatch.setattr(recipe, "evaluate_case", evaluate)
    monkeypatch.setattr(recipe, "decision", lambda rows: dict(decision="test-gate"))
    output = tmp_path/"evaluate"
    output.mkdir()
    result = fresh.evaluate(tmp_path, output, "a"*40, lambda: None, experiment=recipe)
    assert calls == [(s, v, w) for s in fresh.EVAL_SEEDS for v, w in fresh.EVAL_CASES]
    assert result["final_iteration"] == 5999 and len(result["cases"]) == 12
    assert result["selection"] == "final5999 only; no checkpoint fishing"
    assert result["policy_acceptance"] is result["physical_motion_authorized"] is False


def test_revised_receipt_keeps_protocol_separation_and_cap(monkeypatch):
    calls = []
    monkeypatch.setattr(fresh, "validate_training_receipt", lambda *args, **kwargs: calls.append((args, kwargs)))
    recipe.validate_training_receipt("root", "foundation", "a"*40, {"marker": 1})
    assert calls[0][1] == dict(protocol=recipe.PROTOCOL, cap_seconds=10800)
