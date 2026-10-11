from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_safe_tracking_refinement as safe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_tracking_refinement as tracking
from mjlab_microduck import integrated_joint_limit_refinement as repair
from mjlab_microduck.tasks.mdp import body_forward_tracking_cost


def test_forward_residual_is_not_a_motion_tax_and_ignores_other_axes():
    from test_integrated_tracking_refinement import fake_env
    env = fake_env([[.3, 9., 10.]]*4,
                   [[.3, 7., 8.], [0., 7., 8.], [.6, 7., 8.], [-.3, 7., 8.]],
                   [[5., 6., 7.]]*4)
    assert body_forward_tracking_cost(env).tolist() == pytest.approx([0., .09, .09, .36])
    env.command_manager.get_command = lambda _: torch.zeros(4, 3)
    env.scene["robot"].data.root_link_lin_vel_b.zero_()
    assert body_forward_tracking_cost(env).tolist() == [0.]*4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("where", ["command", "velocity"])
def test_raw_nonfinite_forward_residual_fails_closed(value, where):
    from test_integrated_tracking_refinement import fake_env
    env = fake_env([[value if where == "command" else .3, 0., .2]],
                   [[value if where == "velocity" else .3, 0., 0.]], [[0., 0., 0.]])
    with pytest.raises(FloatingPointError, match="raw body forward"):
        body_forward_tracking_cost(env)


def test_matched_configs_and_exact_parent_retain_existing_repairs(monkeypatch):
    cfg, agent = forward.prepare_config("control")
    other, oa = forward.prepare_config("forward")
    assert other.rewards["forward_error"].weight == -.25
    other.rewards["forward_error"].weight = cfg.rewards["forward_error"].weight
    assert repr(cfg) == repr(other)
    oa.run_name = agent.run_name
    assert agent == oa and cfg.seed == agent.seed == 937
    assert cfg.rewards["joint_limit"].weight == -4.
    assert cfg.rewards["motor_load"].weight == -1.
    assert cfg.rewards["command_error"].weight == -.5
    assert cfg.curriculum == {} and cfg.actions["joint_pos"].clip is None
    assert agent.max_iterations == 1000 and agent.save_interval == 250
    assert forward.evaluate_case is safe.evaluate_case and forward.decision is safe.decision
    calls = []
    monkeypatch.setattr(motor, "warm_load", lambda *a, **k: calls.append((a, k)))
    forward.warm_load("runner", "env")
    assert calls == [(("runner", "env", forward.PARENT), dict(parent_sha=forward.PARENT_SHA,
                     parent_iteration=999, parent_steps=24000))]
    assert forward.PARENT_SHA == "503fb02f0bbddbd3fc886453b20a2c878a91a8c1c985eeddfeec768020c53b39"


def test_all_modes_and_strict_control_step_boundaries():
    for mode, (worlds, updates, seed) in forward.MODES.items():
        cfg, agent = forward.prepare_config(mode)
        assert cfg.scene.num_envs == worlds and agent.max_iterations == updates and agent.seed == seed
        assert cfg.rewards["command_error"].weight == -.5
        assert cfg.rewards["forward_error"].weight == forward.stages(mode)[0][1]
    assert forward.stages("smoke-forward") == ((0, -.25), (1, -.75), (2, -1.5))
    assert forward.stages("benchmark-forward") == ((0, -1.5),)
    for step, expected in ((0, -.25), (6000, -.25), (6001, -.75), (12000, -.75), (12001, -1.5)):
        assert motor.motor_weight(step, forward.FORWARD_STAGES) == expected


@pytest.mark.parametrize("mode", ["control", "forward"])
@pytest.mark.parametrize("start,counts", [(5976, {"-0.25": 24}),
    (6000, {"-0.25": 1, "-0.75": 23}), (11976, {"-0.75": 24}),
    (12000, {"-0.75": 1, "-1.5": 23})])
def test_advancing_control_clock_splits_boundary_update_as_declared(mode, start, counts):
    from collections import Counter
    from test_integrated_joint_limit_refinement import fake_env
    env = fake_env()
    original = env.reward_manager.get_term_cfg
    new_term = NS(weight=0.)
    env.reward_manager.get_term_cfg = lambda name: new_term if name == "forward_error" else original(name)
    env.common_step_counter = start
    observer = forward.MotorObserver(mode)
    observed = []
    for _ in range(24):
        observer.before(env)
        observed.append(str(new_term.weight))
        env.common_step_counter += 1  # actual vector-step clock, not worlds*steps
    assert dict(Counter(observed)) == ({"0.0": 24} if mode == "control" else counts)


@pytest.mark.parametrize("mode", ["control", "forward"])
@pytest.mark.parametrize("fault", [None, "missing", "wrong-reward"])
def test_pre_reset_live_forward_reward_and_fixed_terms(mode, fault):
    from test_integrated_joint_limit_refinement import fake_env
    env = fake_env()
    env.reward_manager.active_terms.append("forward_error")
    env.reward_manager._step_reward = torch.zeros(4, 4)
    terms = {name: env.reward_manager.get_term_cfg(name) for name in env.reward_manager.active_terms[:-1]}
    terms["forward_error"] = NS(weight=0., params={"command_name": "twist"})
    env.reward_manager.get_term_cfg = terms.__getitem__
    env.command_manager.get_command = lambda _: torch.tensor([[.2, 0., .2]]).repeat(4, 1)
    observer = forward.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    q = env.scene["robot"].data.joint_pos.clone()
    for step, staged in ((0, -.25), (6000, -.25), (6001, -.75), (12000, -.75), (12001, -1.5)):
        env.common_step_counter = step
        for _ in range(24):
            env.scene["robot"].data.joint_pos.copy_(q)
            env.scene["robot"].data.root_link_lin_vel_b.zero_()
            observer.before(env)
            expected = 0. if mode == "control" else staged
            assert terms["motor_load"].weight == -1. and terms["joint_limit"].weight == -4.
            assert terms["command_error"].weight == -.5 and terms["forward_error"].weight == expected
            tracking.capture_tracking_metric(env); repair.capture_limit_metric(env)
            forward.capture_forward_metric(env)
            env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*-.5
            env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
            env.reward_manager._step_reward[:, 3] = env._microduck_forward_cost*expected
            # Emulate auto-reset: all reward checks must use snapshots, not this state.
            env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 9.
            env.scene["robot"].data.joint_pos.zero_()
            if fault == "missing": env._microduck_forward_cost = None
            if fault == "wrong-reward": env.reward_manager._step_reward[:, 3] = .1
            if fault:
                with pytest.raises(ValueError, match="forward"):
                    observer.step(env, sample, {})
                return
            observer.step(env, sample, {})
            assert env._microduck_forward_cost is None
        row = observer.update(env)
        assert row["forward_cost_mean"] == pytest.approx(.04)
        assert row["weighted_forward_reward_mean"] == pytest.approx(.04*expected)
        assert row["forward_weight_step_counts"] == {str(expected): 24}
        assert row["forward_weight_min"] == row["forward_weight_max"] == expected
