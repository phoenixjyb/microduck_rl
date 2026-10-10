from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_safe_tracking_refinement as safe
from mjlab_microduck import integrated_joint_limit_refinement as repair
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_tracking_refinement as tracking


def test_matched_recipe_keeps_joint_and_motor_repair_while_only_tracking_differs():
    control, ca = safe.prepare_config("control")
    candidate, ta = safe.prepare_config("tracking")
    assert control.rewards["command_error"].weight == 0.
    assert candidate.rewards["command_error"].weight == -.125
    candidate.rewards["command_error"].weight = 0.
    assert repr(control) == repr(candidate)
    ta.run_name = ca.run_name
    assert ca == ta and ca.seed == control.seed == 919
    assert control.rewards["motor_load"].weight == -1.
    assert control.rewards["joint_limit"].weight == -4.
    assert control.rewards["joint_limit"].params["margin"] == .15
    assert control.curriculum == {} and control.actions["joint_pos"].clip is None
    assert ca.max_iterations == 1000 and ca.save_interval == 250
    assert control.commands["twist"].ranges.lin_vel_x == (.2, .4)
    assert control.commands["twist"].ranges.ang_vel_z == (-.5, .5)


def test_parent_is_exact_range_repaired_checkpoint_and_warm_clocks_are_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(motor, "warm_load", lambda *a, **k: calls.append((a, k)) or {"ok": True})
    assert safe.warm_load("runner", "env") == {"ok": True}
    assert calls == [(("runner", "env", safe.PARENT), dict(parent_sha=safe.PARENT_SHA,
                     parent_iteration=999, parent_steps=24000))]
    assert safe.PARENT.parent.name == "limit"
    assert safe.PARENT_SHA == "212a567984014f90054cbd7137e7560d8c76a00496a3ca15958407b9de746e6b"
    assert safe.evaluate_case is repair.evaluate_case and safe.decision is repair.decision


def test_all_preflight_modes_and_step_boundaries_match_declared_schedule():
    for mode, (worlds, updates, seed) in safe.MODES.items():
        cfg, agent = safe.prepare_config(mode)
        assert cfg.scene.num_envs == worlds and agent.max_iterations == updates and agent.seed == seed
        obs = safe.MotorObserver(mode)
        assert obs.limit_schedule == ((0, -4.),) and obs.schedule == ((0, -1.),)
        assert cfg.rewards["command_error"].weight == safe.stages(mode)[0][1]
    assert safe.stages("smoke-tracking") == ((0, -.125), (1, -.25), (2, -.5))
    assert safe.stages("benchmark-tracking") == ((0, -.5),)
    assert motor.motor_weight(250*24, safe.TRACKING_STAGES) == -.125
    assert motor.motor_weight(250*24+1, safe.TRACKING_STAGES) == -.25
    assert motor.motor_weight(500*24+1, safe.TRACKING_STAGES) == -.5


@pytest.mark.parametrize("mode,weight", [("control", 0.), ("tracking", -.5)])
def test_actual_pre_reset_command_reward_keeps_fixed_limit_penalty(mode, weight):
    from test_integrated_joint_limit_refinement import fake_env
    env = fake_env()
    env.common_step_counter = 500*24+1
    observer = safe.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    q = env.scene["robot"].data.joint_pos.clone()
    env.command_manager.get_command = lambda n: torch.tensor([[.2, 0., .2]]).repeat(4, 1)
    for _ in range(24):
        observer.before(env)
        assert env.reward_manager.get_term_cfg("joint_limit").weight == -4.
        assert env.reward_manager.get_term_cfg("motor_load").weight == -1.
        assert env.reward_manager.get_term_cfg("command_error").weight == weight
        tracking.capture_tracking_metric(env)
        repair.capture_limit_metric(env)
        env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*weight
        env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
        env.scene["robot"].data.joint_pos.zero_()
        observer.step(env, sample, {})
        env.scene["robot"].data.joint_pos.copy_(q)
    row = observer.update(env)
    assert row["weighted_command_reward_mean"] == pytest.approx(.08*weight)
    assert row["weighted_limit_reward_mean"] == pytest.approx(-.55)
    assert row["limit_weight_min"] == row["limit_weight_max"] == -4.


@pytest.mark.parametrize("mode", ["control", "tracking"])
def test_live_observer_transitions_keep_fixed_motor_and_qpos_terms(mode):
    from test_integrated_joint_limit_refinement import fake_env
    env = fake_env()
    observer = safe.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    env.command_manager.get_command = lambda n: torch.tensor([[.2, 0., .2]]).repeat(4, 1)
    for step, staged in ((0, -.125), (6000, -.125), (6001, -.25), (12000, -.25), (12001, -.5)):
        env.common_step_counter = step
        observer.before(env)
        expected = 0. if mode == "control" else staged
        assert env.reward_manager.get_term_cfg("command_error").weight == expected
        assert observer.tracking_weight == expected and observer.limit_weight == -4.
        assert env.reward_manager.get_term_cfg("motor_load").weight == -1.
        tracking.capture_tracking_metric(env)
        repair.capture_limit_metric(env)
        env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*expected
        env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
        observer.step(env, sample, {})
        assert env._microduck_tracking_cost is env._microduck_limit_snapshot is None
