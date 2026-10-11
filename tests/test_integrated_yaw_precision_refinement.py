import math
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_yaw_precision_refinement as precision
from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_tracking_refinement as tracking
from mjlab_microduck import integrated_joint_limit_refinement as repair
from mjlab_microduck.tasks import mdp


def fake_env():
    from test_integrated_joint_limit_refinement import fake_env as original
    env = original()
    env.scene["robot"].data.root_link_ang_vel_b = torch.tensor([[0., 0., .2], [.3, -.4, .2],
                                                            [0., 0., -.2], [0., 0., 0.]])
    env.command_manager.get_command = lambda _: torch.tensor([[.3, 0., .2]]*4)
    terms = {k: env.reward_manager.get_term_cfg(k) for k in env.reward_manager.active_terms}
    terms["forward_error"] = NS(weight=0., params={"command_name": "twist"})
    terms[precision.TERM] = NS(func=mdp.factorized_body_angular_tracking, weight=2.,
        params=dict(command_name="twist", yaw_variance=.5, roll_pitch_variance=.5))
    env.reward_manager.active_terms = list(terms)
    env.reward_manager.get_term_cfg = terms.__getitem__
    env.reward_manager._step_reward = torch.zeros(4, 5)
    return env


def test_control_reproduces_stock_and_yaw_sharpening_does_not_sharpen_xy():
    from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity
    env = fake_env()
    # Include a broad finite range, signed yaw and independent roll/pitch.
    torch.manual_seed(19)
    env.num_envs = 1000
    env.scene["robot"].data.root_link_ang_vel_b = torch.randn(1000, 3)*3
    command = torch.randn(1000, 3)
    env.command_manager.get_command = lambda _: command
    stock = track_angular_velocity(env, math.sqrt(.5), "twist")
    control = mdp.factorized_body_angular_tracking(env)
    assert torch.allclose(stock, control, atol=1e-6, rtol=1e-5)
    env = fake_env()
    a = mdp.factorized_body_angular_tracking(env)
    b = mdp.factorized_body_angular_tracking(env, yaw_variance=.1)
    assert a.tolist() == pytest.approx([1., math.exp(-.5), math.exp(-.32), math.exp(-.08)])
    assert b.tolist() == pytest.approx([1., math.exp(-.5), math.exp(-1.6), math.exp(-.4)])
    assert b[1] == a[1]  # nonzero roll/pitch, matching yaw: same kernel
    env.command_manager.get_command = lambda _: torch.zeros(4, 3)
    env.scene["robot"].data.root_link_ang_vel_b.zero_()
    assert mdp.factorized_body_angular_tracking(env).tolist() == [1.]*4


@pytest.mark.parametrize("where", ["command", "angular", "overflow"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_raw_nonfinite_is_not_sanitized(where, value):
    env = fake_env()
    if where == "command":
        env.command_manager.get_command = lambda _: torch.full((4, 3), value)
    else:
        env.scene["robot"].data.root_link_ang_vel_b[0, 0] = 1e30 if where == "overflow" else value
    with pytest.raises(FloatingPointError): mdp.factorized_body_angular_tracking(env)


@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf")])
@pytest.mark.parametrize("key", ["yaw_variance", "roll_pitch_variance"])
def test_invalid_variance_fails_closed(key, value):
    with pytest.raises(ValueError): mdp.factorized_body_angular_tracking(fake_env(), **{key: value})


def test_matched_configs_exact_parent_and_unchanged_commands_gates(monkeypatch):
    cfg, agent = precision.prepare_config("control")
    other, oa = precision.prepare_config("precision")
    assert repr(cfg) == repr(other)
    oa.run_name = agent.run_name
    assert agent == oa and cfg.seed == agent.seed == 977
    orig, _ = forward.prepare_config("control")
    cfg.rewards[precision.TERM] = orig.rewards[precision.TERM]
    cfg.metrics.pop("microduck_precision_reward")
    cfg.seed = orig.seed
    assert repr(cfg) == repr(orig)  # no other plant, command, DR, action/reward differences
    assert cfg.rewards["motor_load"].weight == -1. and cfg.rewards["joint_limit"].weight == -4.
    assert cfg.rewards["command_error"].weight == -.5 and cfg.rewards["forward_error"].weight == 0.
    assert cfg.rewards["action_rate_l2"].weight == -.8 and cfg.curriculum == {}
    assert cfg.actions["joint_pos"].clip is None and agent.max_iterations == 1000 and agent.save_interval == 250
    assert precision.evaluate_case is forward.evaluate_case and precision.decision is forward.decision
    calls = []
    monkeypatch.setattr(motor, "warm_load", lambda *a, **k: calls.append((a, k)))
    precision.warm_load("runner", "env")
    assert calls == [(("runner", "env", precision.PARENT), dict(parent_sha=precision.PARENT_SHA,
                     parent_iteration=999, parent_steps=24000))]
    assert precision.PARENT.parent.name == "control"
    assert precision.PARENT_SHA == "11c20ccc4d0682ab1403013dd5ca3308913ea2fd61a86e1c5bc3fb5408c2b8c2"


def test_modes_and_strict_clock_boundaries():
    for mode, (worlds, updates, seed) in precision.MODES.items():
        cfg, agent = precision.prepare_config(mode)
        assert cfg.scene.num_envs == worlds and agent.seed == cfg.seed == seed and agent.max_iterations == updates
        assert cfg.rewards[precision.TERM].params["yaw_variance"] == precision.stages(mode)[0][1]
    for step, expected in ((0, .5), (6000, .5), (6001, .25), (12000, .25), (12001, .1)):
        assert motor.motor_weight(step, precision.YAW_STAGES) == expected
    assert precision.stages("benchmark-precision") == ((0, .1),)


@pytest.mark.parametrize("mode", ["control", "precision"])
def test_live_reward_stages_and_snapshots_survive_auto_reset(mode):
    env = fake_env(); observer = precision.MotorObserver(mode)
    data = env.scene["robot"].data
    angular, q = data.root_link_ang_vel_b.clone(), data.joint_pos.clone()
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    cumulative = {}
    for start in (0, 6000, 12000):
        env.common_step_counter = start
        expected_counts = {}
        for _ in range(24):
            data.root_link_ang_vel_b.copy_(angular); data.joint_pos.copy_(q)
            observer.before(env)
            term = env.reward_manager.get_term_cfg(precision.TERM)
            variance = term.params["yaw_variance"]
            assert term.params["roll_pitch_variance"] == .5 and term.weight == 2.
            expected_counts[str(variance)] = expected_counts.get(str(variance), 0)+1
            tracking.capture_tracking_metric(env); repair.capture_limit_metric(env); forward.capture_forward_metric(env)
            env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*-.5
            env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
            env.reward_manager._step_reward[:, 4] = 2*term.func(env, **term.params)
            precision.capture_angular_metric(env)
            data.root_link_ang_vel_b.zero_(); data.joint_pos.zero_()  # emulate reset
            observer.step(env, sample, {})
            assert env._microduck_precision_snapshot is None
            env.common_step_counter += 1
        row = observer.update(env)
        for k,n in expected_counts.items(): cumulative[k] = cumulative.get(k, 0)+n
        assert row["yaw_variance_step_counts"] == expected_counts
        assert row["yaw_variance_cumulative_step_counts"] == cumulative
        assert row["roll_pitch_variance"] == .5 and row["angular_tracking_weight"] == 2.
        assert row["roll_pitch_rate_sq_mean"] == pytest.approx(.0625)
        assert row["yaw_error_sq_mean"] == pytest.approx(.05)
        if mode == "precision" and start == 6000: assert expected_counts == {"0.5": 1, "0.25": 23}
        if mode == "precision" and start == 12000: assert expected_counts == {"0.25": 1, "0.1": 23}


@pytest.mark.parametrize("fault", ["wrong-xy", "wrong-weight", "reverse-clock", "missing", "wrong-reward", "stale"])
def test_observer_failures_do_not_trust_config_or_skip_capture(fault):
    env = fake_env(); observer = precision.MotorObserver("precision")
    term = env.reward_manager.get_term_cfg(precision.TERM)
    if fault == "wrong-xy": term.params["roll_pitch_variance"] = .1
    if fault == "wrong-weight": term.weight = -2.
    if fault == "reverse-clock":
        env.common_step_counter = 6001; observer.before(env); env.common_step_counter = 0
    if fault in ("wrong-xy", "wrong-weight", "reverse-clock"):
        with pytest.raises(ValueError): observer.before(env)
        return
    observer.before(env)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    tracking.capture_tracking_metric(env); repair.capture_limit_metric(env); forward.capture_forward_metric(env)
    env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*-.5
    env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
    env.reward_manager._step_reward[:, 4] = 2*term.func(env, **term.params)
    if fault != "missing": precision.capture_angular_metric(env)
    if fault == "wrong-reward": env._microduck_precision_snapshot[2].zero_()
    if fault == "stale":
        with pytest.raises(ValueError): precision.capture_angular_metric(env)
        return
    with pytest.raises(ValueError): observer.step(env, sample, {})
