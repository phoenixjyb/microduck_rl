from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_angular_decoupling_refinement as decoupling
from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_yaw_precision_refinement as precision
from mjlab_microduck.tasks import mdp


def fake_env():
    from test_integrated_yaw_precision_refinement import fake_env as original
    env = original()
    term = env.reward_manager.get_term_cfg(precision.TERM)
    term.func = mdp.factorized_body_angular_tracking
    term.weight = 2.
    term.params = dict(command_name="twist", yaw_variance=.5, roll_pitch_variance=.5)
    return env


def fake_kl_runner(monkeypatch, env):
    from test_integrated_kl_budget_refinement import fake_optimizer
    runner, _, _, _ = fake_optimizer(monkeypatch, .01)
    kl.install_optimizer_capture(runner, env)
    return runner


def configure_reward(env, mode):
    term = env.reward_manager.get_term_cfg(precision.TERM)
    term.func = (mdp.factorized_body_angular_tracking if mode == "control"
                 else mdp.decoupled_body_angular_tracking)
    term.weight = 2.
    term.params = dict(command_name="twist", yaw_variance=.5, roll_pitch_variance=.5)
    return term


def test_only_reward_callable_changes_and_all_modes_are_fixed():
    control, control_agent = decoupling.prepare_config("control")
    treatment, treatment_agent = decoupling.prepare_config("decoupled")
    cterm, tterm = control.rewards[decoupling.TERM], treatment.rewards[decoupling.TERM]
    assert cterm.func is mdp.factorized_body_angular_tracking
    assert tterm.func is mdp.decoupled_body_angular_tracking
    assert cterm.weight == tterm.weight == 2.
    assert cterm.params == tterm.params == dict(
        command_name="twist", yaw_variance=.5, roll_pitch_variance=.5)
    tterm.func = cterm.func
    treatment_agent.run_name = control_agent.run_name
    assert repr(control) == repr(treatment)
    assert control_agent == treatment_agent and control.seed == control_agent.seed == 1019

    original, original_agent = precision.prepare_config("control")
    control.seed = original.seed
    control_agent.seed = original_agent.seed
    control_agent.experiment_name = original_agent.experiment_name
    control_agent.run_name = original_agent.run_name
    assert repr(control) == repr(original)
    assert control_agent == original_agent
    assert control.curriculum == {}
    assert control.rewards["motor_load"].weight == -1.
    assert control.rewards["joint_limit"].weight == -4.
    assert control.rewards["command_error"].weight == -.5
    assert control.rewards["forward_error"].weight == 0.
    assert control.rewards["action_rate_l2"].weight == -.8
    assert control.actions["joint_pos"].clip is None
    assert control_agent.algorithm.desired_kl == .01
    assert control_agent.algorithm.schedule == "adaptive"
    assert decoupling.evaluate_case is precision.evaluate_case
    assert decoupling.decision is precision.decision

    for mode, (worlds, updates, seed) in decoupling.MODES.items():
        cfg, agent = decoupling.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates
        assert agent.algorithm.desired_kl == .01
        assert cfg.rewards[decoupling.TERM].params["yaw_variance"] == .5
        assert cfg.rewards[decoupling.TERM].params["roll_pitch_variance"] == .5


def test_exact_parent_warm_load_then_optimizer_capture(monkeypatch):
    env = fake_env()
    from test_integrated_kl_budget_refinement import fake_optimizer
    runner, _, _, _ = fake_optimizer(monkeypatch, .01)
    calls = []
    receipt = {"learning_rate": 1e-5, "parent_sha256": decoupling.PARENT_SHA}

    def restore(*args, **kwargs):
        calls.append((args, kwargs))
        return receipt

    monkeypatch.setattr(motor, "warm_load", restore)
    result = decoupling.warm_load(runner, env)
    assert result is receipt
    assert calls == [((runner, env, decoupling.PARENT), dict(
        parent_sha=decoupling.PARENT_SHA, parent_iteration=999, parent_steps=24000))]
    assert env._microduck_optimizer_audit.alg is runner.alg
    assert runner.alg.learning_rate == 1e-5
    assert "kl-budget-a9a70f0e-20261011/control/model_999.pt" in str(decoupling.PARENT)
    assert decoupling.PARENT_SHA == "3ee173730a78fae605e561a3d40024b8e66d3b0a0be09262a6322fa819fe3df4"

    runner2, _, _, _ = fake_optimizer(monkeypatch, .01)
    receipt_bad = dict(receipt, learning_rate=.001)
    monkeypatch.setattr(motor, "warm_load", lambda *args, **kwargs: receipt_bad)
    with pytest.raises(ValueError, match="inherited initial PPO learning rate"):
        decoupling.warm_load(runner2, fake_env())


def test_analytic_control_parity_axis_zero_identity_and_negative_range():
    env = fake_env()
    command = torch.zeros(4, 3)
    angular = torch.tensor([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [1., 0., 1.]])
    env.command_manager.get_command = lambda _: command
    env.scene["robot"].data.root_link_ang_vel_b = angular
    yaw_sq, xy_sq = mdp.body_angular_tracking_errors(env)
    a, b = torch.exp(-yaw_sq / .5), torch.exp(-xy_sq / .5)
    control = mdp.factorized_body_angular_tracking(env)
    treatment = mdp.decoupled_body_angular_tracking(env)
    assert torch.allclose(control, a * b)
    assert torch.allclose(treatment, a + b - 1.)
    assert torch.allclose(2. * control, 2. * a * b)
    assert torch.allclose(2. * treatment, 2. * (a + b - 1.))
    assert torch.allclose(control[:3], treatment[:3])
    assert control[0] == treatment[0] == 1.
    assert treatment.min() >= -1. and treatment.max() <= 1.
    assert 2. * treatment[3] < 0.
    assert torch.allclose(treatment - control, -(1. - a) * (1. - b), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("key", ["yaw_variance", "roll_pitch_variance"])
@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf")])
def test_invalid_variance_fails_closed(key, value):
    with pytest.raises(ValueError):
        mdp.decoupled_body_angular_tracking(fake_env(), **{key: value})


@pytest.mark.parametrize("where,value", [
    ("command", float("nan")), ("command", float("inf")),
    ("angular", float("nan")), ("angular", -float("inf")),
    ("angular-overflow", 1e30),
])
def test_nonfinite_raw_state_and_overflow_fail_closed(where, value):
    env = fake_env()
    if where == "command":
        env.command_manager.get_command = lambda _: torch.full((4, 3), value)
    else:
        env.scene["robot"].data.root_link_ang_vel_b[0, 0] = value
    with pytest.raises(FloatingPointError):
        mdp.decoupled_body_angular_tracking(env)


def test_finite_variance_can_still_overflow_combined_exponent():
    env = fake_env()
    env.command_manager.get_command = lambda _: torch.zeros(4, 3)
    env.scene["robot"].data.root_link_ang_vel_b.zero_()
    env.scene["robot"].data.root_link_ang_vel_b[0, 2] = 1.
    with pytest.raises(FloatingPointError, match="exponent"):
        mdp.decoupled_body_angular_tracking(env, yaw_variance=1e-309)


@pytest.mark.parametrize("mode", ["control", "decoupled"])
def test_live_24_step_reward_snapshot_auto_reset_and_20_kl_minibatches(monkeypatch, mode):
    env = fake_env()
    term = configure_reward(env, mode)
    runner = fake_kl_runner(monkeypatch, env)
    observer = decoupling.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    from mjlab_microduck import integrated_tracking_refinement as tracking
    from mjlab_microduck import integrated_joint_limit_refinement as repair
    data = env.scene["robot"].data
    original_q = data.joint_pos.clone()
    negative_worlds = 0

    for _ in range(24):
        data.joint_pos.copy_(original_q)
        data.root_link_ang_vel_b.copy_(torch.tensor([[1., 0., 1.]] * 4))
        observer.before(env)
        tracking.capture_tracking_metric(env)
        repair.capture_limit_metric(env)
        forward.capture_forward_metric(env)
        env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost * -.5
        env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0] * -4.
        env.reward_manager._step_reward[:, 4] = term.weight * term.func(env, **term.params)
        expected_pre_reset = env.reward_manager._step_reward[:, 4].clone()
        negative_worlds = int((expected_pre_reset < 0.).sum())
        precision.capture_angular_metric(env)
        saved = env._microduck_precision_snapshot
        assert torch.equal(saved[2], expected_pre_reset)
        data.root_link_ang_vel_b.zero_()  # emulate auto-reset after metric capture
        observer.step(env, sample, {})
        assert env._microduck_precision_snapshot is None
        env.common_step_counter += 1

    for _ in range(20):
        runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["optimizer_minibatches"] == 20
    assert len(row["minibatch_kl_means"]) == len(row["minibatch_lr_before"]) == 20
    assert row["desired_kl"] == .01 and row["lr_before_min"] == 1e-5
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["roll_pitch_variance"] == .5 and row["angular_tracking_weight"] == 2.
    assert row["yaw_error_sq_mean"] == pytest.approx(.64)
    assert row["roll_pitch_rate_sq_mean"] == pytest.approx(1.)
    assert row["yaw_variance_step_counts"] == {"0.5": 24}
    assert row["yaw_variance_cumulative_step_counts"] == {"0.5": 24}
    assert row["negative_reward_sample_count"] == (negative_worlds * 24 if mode == "decoupled" else 0)
    assert row["angular_reward_sample_count"] == 4 * 24
    assert row["negative_reward_fraction"] == pytest.approx(negative_worlds / 4 if mode == "decoupled" else 0.)
    assert row["angular_aggregation"] == decoupling.AGGREGATIONS[mode]
    assert not env._microduck_optimizer_audit.samples
    observer.before(env)
    runner.alg.actor.get_kl_divergence(None)
    with pytest.raises(ValueError, match="previous optimizer minibatches consumed"):
        observer.before(env)


@pytest.mark.parametrize("mode", ["control", "decoupled"])
@pytest.mark.parametrize("fault", ["wrong-func", "wrong-weight", "wrong-params"])
def test_before_action_rejects_reward_drift(monkeypatch, mode, fault):
    env = fake_env()
    term = configure_reward(env, mode)
    runner = fake_kl_runner(monkeypatch, env)
    observer = decoupling.MotorObserver(mode)
    if fault == "wrong-func":
        term.func = mdp.decoupled_body_angular_tracking if mode == "control" else mdp.factorized_body_angular_tracking
    elif fault == "wrong-weight":
        term.weight = -2.
    else:
        term.params["roll_pitch_variance"] = .25
    with pytest.raises(ValueError, match="fixed live angular reward"):
        observer.before(env)
    assert runner.alg.learning_rate == 1e-5


@pytest.mark.parametrize("fault", ["missing", "mismatch", "nonfinite", "stale"])
def test_step_requires_consumable_pre_reset_snapshot(monkeypatch, fault):
    mode = "decoupled"
    env = fake_env()
    term = configure_reward(env, mode)
    fake_kl_runner(monkeypatch, env)
    observer = decoupling.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    observer.before(env)
    from mjlab_microduck import integrated_tracking_refinement as tracking
    from mjlab_microduck import integrated_joint_limit_refinement as repair
    tracking.capture_tracking_metric(env)
    repair.capture_limit_metric(env)
    forward.capture_forward_metric(env)
    env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost * -.5
    env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0] * -4.
    env.reward_manager._step_reward[:, 4] = term.weight * term.func(env, **term.params)
    if fault != "missing":
        precision.capture_angular_metric(env)
    if fault == "mismatch":
        env._microduck_precision_snapshot[2].zero_()
    elif fault == "nonfinite":
        env._microduck_precision_snapshot[0][0] = float("nan")
    elif fault == "stale":
        with pytest.raises(ValueError, match="previous angular snapshot"):
            precision.capture_angular_metric(env)
        return
    with pytest.raises(ValueError):
        observer.step(env, sample, {})

