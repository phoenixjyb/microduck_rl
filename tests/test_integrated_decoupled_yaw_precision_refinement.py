from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_angular_decoupling_refinement as decoupling
from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as precision
from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_yaw_precision_refinement as previous
from mjlab_microduck.tasks import mdp


PARENT_LR = 5.062500000000001e-05


def fake_env():
    from test_integrated_yaw_precision_refinement import fake_env as original

    env = original()
    term = env.reward_manager.get_term_cfg(previous.TERM)
    term.func = mdp.decoupled_body_angular_tracking
    term.weight = 2.
    term.params = dict(command_name="twist", yaw_variance=.5, roll_pitch_variance=.5)
    return env


def fake_optimizer(monkeypatch, env, *, initial_lr=PARENT_LR):
    from test_integrated_kl_budget_refinement import fake_optimizer as original

    runner, _, value, calls = original(monkeypatch, .01)
    runner.alg.learning_rate = initial_lr
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = initial_lr
    kl.install_optimizer_capture(runner, env, initial_lr=initial_lr)
    return runner, value, calls


def set_turn_state(env):
    env.command_manager.get_command = lambda _: torch.tensor([[.3, 0., .2]] * 4)
    env.scene["robot"].data.root_link_ang_vel_b.copy_(torch.tensor([
        [0., 0., .2], [1., 0., 1.2], [0., 0., .2], [1., 0., 1.2],
    ]))


def capture_other_pre_reset_terms(env, term):
    from mjlab_microduck import integrated_joint_limit_refinement as repair
    from mjlab_microduck import integrated_tracking_refinement as tracking

    tracking.capture_tracking_metric(env)
    repair.capture_limit_metric(env)
    forward.capture_forward_metric(env)
    env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost * -.5
    env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0] * -4.
    index = env.reward_manager.active_terms.index(previous.TERM)
    env.reward_manager._step_reward[:, index] = term.weight * term.func(env, **term.params)


def test_modes_and_configs_keep_same_signed_reward_and_change_only_yaw_schedule():
    assert precision.MODES == {
        "smoke-control": (64, 5, 1021),
        "smoke-precision": (64, 5, 1021),
        "benchmark-control": (256, 10, 1023),
        "benchmark-precision": (256, 10, 1023),
        "control": (256, 1000, 1027),
        "precision": (256, 1000, 1027),
    }
    for phase in ("smoke", "benchmark", "full"):
        control_mode = "control" if phase == "full" else f"{phase}-control"
        treatment_mode = "precision" if phase == "full" else f"{phase}-precision"
        control, control_agent = precision.prepare_config(control_mode)
        treatment, treatment_agent = precision.prepare_config(treatment_mode)
        cterm, tterm = control.rewards[precision.TERM], treatment.rewards[precision.TERM]
        assert cterm.func is tterm.func is mdp.decoupled_body_angular_tracking
        assert cterm.weight == tterm.weight == precision.WEIGHT
        assert cterm.params["roll_pitch_variance"] == tterm.params["roll_pitch_variance"] == .5
        assert cterm.params["yaw_variance"] == .5
        assert tterm.params["yaw_variance"] == precision.stages(treatment_mode)[0][1]
        assert control_agent.algorithm.desired_kl == treatment_agent.algorithm.desired_kl == .01
        assert control_agent.algorithm.schedule == treatment_agent.algorithm.schedule == "adaptive"
        assert control_agent.algorithm.num_mini_batches == treatment_agent.algorithm.num_mini_batches == 4
        assert control_agent.algorithm.num_learning_epochs == treatment_agent.algorithm.num_learning_epochs == 5
        assert control.seed == treatment.seed == control_agent.seed == treatment_agent.seed
        assert control.scene.num_envs == treatment.scene.num_envs
        assert control.actions["joint_pos"].scale == treatment.actions["joint_pos"].scale == 1.
        assert control.actions["joint_pos"].clip is treatment.actions["joint_pos"].clip is None
        cterm.params["yaw_variance"] = tterm.params["yaw_variance"] = .5
        treatment_agent.run_name = control_agent.run_name
        assert repr(control) == repr(treatment)
        assert control_agent == treatment_agent

    assert precision.PROTOCOL == "integrated-decoupled-yaw-precision-v1"
    assert precision.ARMS == ("control", "precision")
    assert precision.CAP_SECONDS == motor.CAP_SECONDS == 2100
    assert precision.evaluate_case is decoupling.evaluate_case
    assert precision.decision is decoupling.decision
    for mode, (worlds, updates, seed) in precision.MODES.items():
        cfg, agent = precision.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates


def test_parent_warm_load_restores_exact_lr_before_capture_without_reset(monkeypatch):
    env = fake_env()
    from test_integrated_kl_budget_refinement import fake_optimizer as original

    runner, _, value, _ = original(monkeypatch, .01)
    runner.alg.learning_rate = PARENT_LR
    runner.alg.optimizer.param_groups[0]["lr"] = PARENT_LR
    calls = []
    receipt = {"learning_rate": PARENT_LR, "parent_sha256": precision.PARENT_SHA}

    def restore(*args, **kwargs):
        calls.append((args, kwargs))
        return receipt

    monkeypatch.setattr(motor, "warm_load", restore)
    result = precision.warm_load(runner, env)
    assert result is receipt
    assert calls == [((runner, env, precision.PARENT), dict(
        parent_sha=precision.PARENT_SHA, parent_iteration=999, parent_steps=24000))]
    assert precision.PARENT_SHA == "c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae"
    assert str(precision.PARENT).endswith(
        "angular-decoupling-ea24a14a-20261011/decoupled/model_999.pt")
    assert runner.alg.learning_rate == PARENT_LR
    assert runner.alg.optimizer.param_groups[0]["lr"] == PARENT_LR
    assert env._microduck_optimizer_audit.alg is runner.alg
    assert not env._microduck_optimizer_audit.samples
    assert value.requires_grad and value.grad is None

    runner_bad, _, _, _ = original(monkeypatch, .01)
    runner_bad.alg.learning_rate = runner_bad.alg.optimizer.param_groups[0]["lr"] = .001
    receipt_bad = {"learning_rate": .001}
    monkeypatch.setattr(motor, "warm_load", lambda *args, **kwargs: receipt_bad)
    with pytest.raises(ValueError, match="exact inherited parent PPO learning rate"):
        precision.warm_load(runner_bad, fake_env())


def test_stage_thresholds_match_required_exposure_counts():
    assert precision.stages("control") == ((0, .5),)
    assert precision.stages("precision") == ((0, .5), (250, .25), (500, .1))
    assert precision.stages("smoke-precision") == ((0, .5), (1, .25), (2, .1))
    assert precision.stages("benchmark-precision") == ((0, .1),)

    smoke = precision.stages("smoke-precision")
    assert [motor.motor_weight(step, smoke) for step in (24, 25, 48, 49)] == [.5, .25, .25, .1]
    smoke_counts = {variance: sum(motor.motor_weight(step, smoke) == variance for step in range(120))
                    for variance in (.5, .25, .1)}
    assert smoke_counts == {.5: 25, .25: 24, .1: 71}

    full = precision.stages("precision")
    assert [motor.motor_weight(step, full) for step in (6000, 6001, 12000, 12001)] == [
        .5, .25, .25, .1]
    full_counts = {variance: sum(motor.motor_weight(step, full) == variance for step in range(24000))
                   for variance in (.5, .25, .1)}
    assert full_counts == {.5: 6001, .25: 6000, .1: 11999}
    benchmark = precision.stages("benchmark-precision")
    assert {motor.motor_weight(step, benchmark) for step in range(24000)} == {.1}


def test_before_action_applies_live_schedule_only_to_yaw_variance(monkeypatch):
    env = fake_env()
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = precision.MotorObserver("precision")
    term = env.reward_manager.get_term_cfg(precision.TERM)

    for step, expected in ((6000, .5), (6001, .25), (12000, .25), (12001, .1)):
        env.common_step_counter = step
        observer.before(env)
        assert term.params == dict(command_name="twist", yaw_variance=expected,
                                   roll_pitch_variance=.5)
        assert term.func is mdp.decoupled_body_angular_tracking
        assert term.weight == 2.
        assert observer.previous_variance == expected
        assert not env._microduck_optimizer_audit.samples

    assert runner.alg.learning_rate == PARENT_LR
    assert runner.alg.optimizer.param_groups[0]["lr"] == PARENT_LR


def test_signed_pre_reset_snapshot_counts_negative_rewards_and_captures_kl(monkeypatch):
    env = fake_env()
    runner, value, calls = fake_optimizer(monkeypatch, env)
    term = env.reward_manager.get_term_cfg(precision.TERM)
    observer = precision.MotorObserver("smoke-precision")
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    data = env.scene["robot"].data
    initial_position = data.joint_pos.clone()
    captured_rewards = []

    for step in range(24):
        data.joint_pos.copy_(initial_position)
        set_turn_state(env)
        observer.before(env)
        capture_other_pre_reset_terms(env, term)
        previous.capture_angular_metric(env)
        snapshot = env._microduck_precision_snapshot
        captured_rewards.append(snapshot[2].clone())
        assert snapshot[3] == .5
        assert bool((snapshot[2] < 0.).any())
        data.root_link_ang_vel_b.zero_()  # simulate auto-reset after the metric hook
        observer.step(env, sample, {})
        assert env._microduck_precision_snapshot is None
        env.common_step_counter += 1

    assert all(torch.equal(value, captured_rewards[0]) for value in captured_rewards)
    assert int((captured_rewards[0] < 0.).sum()) == 2
    for _ in range(20):
        assert runner.alg.actor.get_kl_divergence("old", new="new") is value
    assert calls == [(("old",), {"new": "new"})] * 20

    row = observer.update(env)
    assert row["angular_aggregation"] == "additive_decoupled"
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["roll_pitch_variance"] == .5 and row["angular_tracking_weight"] == 2.
    assert row["yaw_variance_step_counts"] == {"0.5": 24}
    assert row["yaw_variance_cumulative_step_counts"] == {"0.5": 24}
    assert row["negative_reward_sample_count"] == 48
    assert row["angular_reward_sample_count"] == 4 * 24
    assert row["negative_reward_fraction"] == pytest.approx(.5)
    assert row["optimizer_minibatches"] == 20 and row["desired_kl"] == .01
    assert len(row["minibatch_kl_means"]) == len(row["minibatch_lr_before"]) == 20
    assert row["lr_before_min"] == PARENT_LR
    assert row["lr_after_update"] == PARENT_LR
    assert not env._microduck_optimizer_audit.samples
    assert observer.total_samples == observer.negative_samples == 0
    observer.before(env)
    runner.alg.actor.get_kl_divergence(None)
    with pytest.raises(ValueError, match="previous optimizer minibatches consumed"):
        observer.before(env)


@pytest.mark.parametrize("fault", ["wrong-func", "wrong-weight", "wrong-params", "stale-snapshot"])
def test_before_action_rejects_reward_drift_or_stale_snapshot(monkeypatch, fault):
    env = fake_env()
    fake_optimizer(monkeypatch, env)
    term = env.reward_manager.get_term_cfg(precision.TERM)
    observer = precision.MotorObserver("precision")
    if fault == "wrong-func": term.func = mdp.factorized_body_angular_tracking
    elif fault == "wrong-weight": term.weight = -2.
    elif fault == "wrong-params": term.params["roll_pitch_variance"] = .25
    else: env._microduck_precision_snapshot = (torch.zeros(4),)
    with pytest.raises(ValueError):
        observer.before(env)


@pytest.mark.parametrize("fault", ["missing", "wrong-reward", "nonfinite", "wrong-variance", "stale"])
def test_step_requires_matching_consumable_pre_reset_snapshot(monkeypatch, fault):
    env = fake_env()
    fake_optimizer(monkeypatch, env)
    term = env.reward_manager.get_term_cfg(precision.TERM)
    observer = precision.MotorObserver("precision")
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    set_turn_state(env)
    observer.before(env)
    capture_other_pre_reset_terms(env, term)
    if fault != "missing": previous.capture_angular_metric(env)
    if fault == "wrong-reward": env._microduck_precision_snapshot[2].zero_()
    elif fault == "nonfinite": env._microduck_precision_snapshot[0][0] = float("nan")
    elif fault == "wrong-variance":
        snapshot = env._microduck_precision_snapshot
        env._microduck_precision_snapshot = (*snapshot[:3], .25)
    elif fault == "stale":
        with pytest.raises(ValueError, match="previous angular snapshot"):
            previous.capture_angular_metric(env)
        return
    with pytest.raises(ValueError):
        observer.step(env, sample, {})


def test_live_compressed_schedule_signed_rewards_and_terminal_snapshots(monkeypatch):
    env = fake_env()
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = precision.MotorObserver("smoke-precision")
    term = env.reward_manager.get_term_cfg(precision.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    data = env.scene["robot"].data
    initial_q = data.joint_pos.clone()
    rows = []
    for update in range(5):
        for _ in range(24):
            data.joint_pos.copy_(initial_q)
            set_turn_state(env)
            observer.before(env)
            capture_other_pre_reset_terms(env, term)
            previous.capture_angular_metric(env)
            yaw_sq, xy_sq, actual, variance = env._microduck_precision_snapshot
            expected = 2. * (torch.exp(-yaw_sq / variance) + torch.exp(-xy_sq / .5) - 1.)
            assert torch.allclose(actual, expected)
            data.root_link_ang_vel_b.zero_()
            observer.step(env, sample, {})
            env.common_step_counter += 1
        for _ in range(20): runner.alg.actor.get_kl_divergence(None)
        row = observer.update(env)
        assert row["angular_reward_sample_count"] == 96
        assert row["negative_reward_sample_count"] == 48
        assert row["negative_reward_fraction"] == .5
        rows.append(row)
    assert [row["yaw_variance_step_counts"] for row in rows] == [
        {"0.5": 24}, {"0.5": 1, "0.25": 23}, {"0.25": 1, "0.1": 23},
        {"0.1": 24}, {"0.1": 24},
    ]
    assert rows[-1]["yaw_variance_cumulative_step_counts"] == {"0.5": 25, "0.25": 24, "0.1": 71}


@pytest.mark.parametrize("fault", ["changed-schedule", "reverse-stage", "learner"])
def test_unexpected_schedule_or_learner_drift_fails_closed(monkeypatch, fault):
    env = fake_env()
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = precision.MotorObserver("precision")
    if fault == "changed-schedule": observer.yaw_schedule = ((0, .05),)
    elif fault == "learner": runner.alg.num_mini_batches = 5
    else:
        env.common_step_counter = 12001
        observer.before(env)
        env.common_step_counter = 6001
    with pytest.raises(ValueError): observer.before(env)
