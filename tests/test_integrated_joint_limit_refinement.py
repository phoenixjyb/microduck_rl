from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_joint_limit_refinement as repair
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_tracking_refinement as tracking
from mjlab_microduck.tasks.mdp import joint_pos_limit_proximity


def test_only_named_qpos_weight_differs_and_no_target_clipping_is_added():
    a, aa = repair.prepare_config("control")
    b, ba = repair.prepare_config("limit")
    assert a.rewards["joint_limit"].weight == 0 and b.rewards["joint_limit"].weight == -1
    b.rewards["joint_limit"].weight = 0.
    assert repr(a) == repr(b)
    ba.run_name = aa.run_name
    assert aa == ba and aa.seed == a.seed == 907 and aa.max_iterations == 1000
    assert a.actions["joint_pos"].clip is None
    assert a.rewards["command_error"].weight == 0 and a.rewards["motor_load"].weight == -1
    assert a.curriculum == {} and a.rewards["joint_limit"].params["margin"] == .15


@pytest.mark.parametrize("backlash", [False, True])
def test_selector_resolves_actual_entity_joint_even_when_passive_joints_interleave(backlash):
    from mjlab.entity import Entity
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    cfg, _ = repair.prepare_config("limit")
    robot = Entity(MICRODUCK_WALK_BACKLASH_ROBOT_CFG if backlash else cfg.scene.entities["robot"])
    selector = cfg.rewards["joint_limit"].params["asset_cfg"]
    ids, names = robot.find_joints(selector.joint_names)
    assert names == ["right_hip_yaw"] and len(ids) == 1
    assert robot.joint_names[ids[0]] == repair.JOINT
    if backlash: assert ids != [9]


def fake_env():
    # A passive-like preceding column makes hardcoded index0 incorrect.
    q = torch.tensor([[0., 0.], [0., .4363323], [0., .4863323], [0., -.5735988]])
    bounds = torch.tensor([[[-1., 1.], [-.5235988, .4363323]]]).repeat(4, 1, 1)
    data = NS(joint_pos=q, joint_pos_limits=bounds,
              root_link_lin_vel_b=torch.zeros(4, 3), root_link_ang_vel_b=torch.zeros(4, 3))
    selector = NS(name="robot", joint_ids=[1])
    terms = {k:NS(weight=0.) for k in ("motor_load", "command_error", "joint_limit")}
    terms["joint_limit"].params = dict(asset_cfg=selector, margin=.15)
    return NS(scene={"robot":NS(data=data)}, num_envs=4, device="cpu", common_step_counter=0,
        command_manager=NS(get_command=lambda n: torch.zeros(4, 3)),
        reward_manager=NS(active_terms=list(terms), get_term_cfg=lambda n:terms[n], _step_reward=torch.zeros(4, 3)))


def test_asymmetric_limit_cost_and_pre_reset_live_weight_are_not_command_penalties():
    env = fake_env()
    term = env.reward_manager.get_term_cfg("joint_limit")
    assert joint_pos_limit_proximity(env, **term.params).tolist() == pytest.approx([0., .15, .20, .20])
    observer = repair.MotorObserver("limit")
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    original_q = env.scene["robot"].data.joint_pos.clone()
    for _ in range(24):
        observer.before(env)
        tracking.capture_tracking_metric(env)
        repair.capture_limit_metric(env)
        env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*observer.limit_weight
        env.scene["robot"].data.joint_pos.zero_()  # simulate terminal reset
        observer.step(env, sample, {})
        assert env._microduck_limit_snapshot is None
        env.scene["robot"].data.joint_pos.copy_(original_q)
    row = observer.update(env)
    assert row["limit_cost_mean"] == pytest.approx(.1375)
    assert row["weighted_limit_reward_mean"] == pytest.approx(-.1375)
    assert row["right_hip_stop_exposure_mean"] == pytest.approx(.75)
    assert row["right_hip_max_range_overrun_rad"] == pytest.approx(.05)
    assert not observer.limit_costs and not observer.limit_overruns


def test_stages_smoke_covers_ramp_and_parent_is_control_final():
    assert repair.stages("control") == ((0, 0.),)
    assert repair.stages("smoke-limit") == ((0, -1.), (1, -2.), (2, -4.))
    assert repair.stages("benchmark-limit") == ((0, -4.),)
    assert motor.motor_weight(250*24, repair.LIMIT_STAGES) == -1
    assert motor.motor_weight(250*24+1, repair.LIMIT_STAGES) == -2
    assert motor.motor_weight(500*24+1, repair.LIMIT_STAGES) == -4
    assert repair.PARENT.name == "model_999.pt" and repair.PARENT.parent.name == "control"
    assert repair.PARENT_SHA == "b7a6601c479ef6805d4fc68236deff1805e6997bea4286e5b0f69b7b3255deec"


def test_joint_guard_adds_to_original_gate_never_replaces_it():
    names = [f"j{i}" for i in range(13)] + [repair.JOINT]
    rows = [dict(seed=seed, speed=speed, yaw=yaw, complete=True, worlds=8, steps=240,
        terminal_worlds=[], speed_mae=[.02]*8, yaw_mae=[.09]*8, torque_p99=.5,
        rated_speed_exceed_fraction=0., response_diagnostics=dict(
            protocol="signed-body-response-and-joint-load-v1", joint_columns=names,
            steps=240, hard_stop_margin_rad=.05, min_distance_to_hard_stop_rad_by_joint=[.1]*14,
            hard_stop_proximity_fraction_by_joint=[0.]*14))
        for seed in repair.base.SEEDS for speed, yaw in repair.base.CASES]
    assert repair.decision(rows)["decision"] == "foundation-ready-for-next-experiment"
    rows[0]["response_diagnostics"]["min_distance_to_hard_stop_rad_by_joint"][-1] = -.006
    assert repair.decision(rows)["failures"][-1][-1] == "joint-range-or-stop-exposure"
    rows[0]["response_diagnostics"]["min_distance_to_hard_stop_rad_by_joint"][-1] = .1
    rows[0]["yaw_mae"][0] = .11
    assert repair.decision(rows)["failures"][0][-1] == "speed-or-yaw-tracking"
    assert repair.decision(rows)["decision"] == "foundation-not-ready"
    rows[0]["response_diagnostics"]["hard_stop_margin_rad"] = .01
    with pytest.raises(ValueError, match="position coverage"): repair.decision(rows)
    rows[0]["response_diagnostics"]["hard_stop_margin_rad"] = .05
    rows[0]["response_diagnostics"]["hard_stop_proximity_fraction_by_joint"] = []
    with pytest.raises(ValueError, match="position coverage"): repair.decision(rows)


def test_missing_or_wrong_signed_actual_penalty_fails_closed():
    env = fake_env()
    observer = repair.MotorObserver("limit")
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    observer.before(env)
    tracking.capture_tracking_metric(env)
    with pytest.raises(ValueError, match="limit snapshot"): observer.step(env, sample, {})
    observer.before(env)
    tracking.capture_tracking_metric(env)
    repair.capture_limit_metric(env)
    env.reward_manager._step_reward[:, 2] = 1
    with pytest.raises(ValueError, match="limit reward"): observer.step(env, sample, {})
