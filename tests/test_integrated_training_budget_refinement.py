from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_training_budget_refinement as budget
from mjlab_microduck import integrated_yaw_bias_refinement as previous


def test_only_duration_differs_and_parent_evaluator_unchanged():
    assert budget.PARENT == previous.PARENT and budget.PARENT_SHA == previous.PARENT_SHA
    assert budget.warm_load is previous.warm_load
    assert budget.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE
    assert budget.evaluate_case is previous.evaluate_case and budget.decision is previous.decision
    assert budget.TRAINING_CAP_SECONDS == dict(control=2100, budget=7200)
    assert budget.MODES == {
        "smoke-control": (64, 5, 1049), "smoke-budget": (64, 5, 1049),
        "benchmark-control": (256, 10, 1051), "benchmark-budget": (256, 10, 1051),
        "control": (256, 1000, 1061), "budget": (256, 4000, 1061),
    }
    for prefix in ("smoke-", "benchmark-", ""):
        cfg, agent = budget.prepare_config(prefix+"control")
        treatment, ta = budget.prepare_config(prefix+"budget")
        assert repr(cfg) == repr(treatment)
        assert agent.max_iterations == (5 if prefix == "smoke-" else 10 if prefix else 1000)
        assert ta.max_iterations == (5 if prefix == "smoke-" else 10 if prefix else 4000)
        ta.run_name, ta.max_iterations = agent.run_name, agent.max_iterations
        assert agent == ta and agent.save_interval == 250
        assert cfg.rewards[previous.TERM].weight == 0.
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["command_error"].weight == -.5
        assert cfg.actions["joint_pos"].clip is None
        assert not cfg.curriculum


def test_command_bins_exclusive_boundary_complete_and_read_only():
    values = torch.tensor([[0., 0., 0.], [.3, 0., 0.],
                           [.2, 0., -.15], [.4, 0., -.25], [.3, 0., -.5],
                           [.2, 0., .15], [.4, 0., .25], [.3, 0., .5]])
    original = values.clone()
    state = torch.random.get_rng_state().clone()
    assert budget.command_exposure(values) == dict.fromkeys(budget.EXPOSURE_BINS, 1)
    assert torch.equal(values, original) and torch.equal(state, torch.random.get_rng_state())
    assert budget.command_exposure(torch.tensor([[.3, 0., .2]]))["positive-target"] == 1
    assert budget.command_exposure(torch.tensor([[.3, 0., -.2]]))["negative-target"] == 1


@pytest.mark.parametrize("bad", [torch.empty(0, 3), torch.zeros(2, 2), torch.zeros(3),
    torch.zeros(2, 3, dtype=torch.long), torch.tensor([[float("nan"), 0., 0.]]),
    torch.tensor([[.3, .1, 0.]]), torch.tensor([[.3, 0., .501]]),
    torch.tensor([[.1, 0., .2]]), torch.tensor([[0., 0., .2]]),
    torch.tensor([[.41, 0., 0.]])])
def test_command_bins_fail_closed(bad):
    with pytest.raises(ValueError): budget.command_exposure(bad)


def test_observer_counts_pre_action_commands_not_post_reset(monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "update", lambda *args: {})
    command = torch.tensor([[.3, 0., .2], [.2, 0., -.2]])
    env = NS(num_envs=2, common_step_counter=0,
             command_manager=NS(get_command=lambda name: command))
    observer = budget.MotorObserver("budget")
    for update in range(2):
        for _ in range(24):
            command[:, 0] = torch.tensor([.3, .2])
            command[:, 2] = torch.tensor([.2, -.2])
            observer.before(env)
            command.zero_()  # final obs/reset must not rewrite the consumed action count
            env.common_step_counter += 1
            observer.step(env, None, {})
        row = observer.update(env)
        assert row["command_exposure_counts"]["positive-target"] == 24
        assert row["command_exposure_counts"]["negative-target"] == 24
        assert row["command_exposure_counts"]["idle"] == 0
        assert row["command_exposure_cumulative"]["positive-target"] == 24*(update+1)
        assert observer.exposure_steps == 0 and observer.pending_exposure is None
        assert not any(observer.exposure_counts.values())
        command[:, 0] = torch.tensor([.3, .2])


def test_missing_duplicate_skipped_exposure_fails_closed(monkeypatch):
    for name in ("before", "step"):
        monkeypatch.setattr(previous.MotorObserver, name, lambda *args: None)
    env = NS(num_envs=1, common_step_counter=0,
             command_manager=NS(get_command=lambda name: torch.tensor([[.3, 0., .2]])))
    observer = budget.MotorObserver("control")
    with pytest.raises(ValueError): observer.step(env, None, {})
    observer.before(env)
    with pytest.raises(ValueError): observer.before(env)
    env.common_step_counter += 2
    with pytest.raises(ValueError): observer.step(env, None, {})


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-budget", "control", "budget"])
def test_full_observer_preserves_reward_and_optimizer_checks(monkeypatch, mode):
    from test_integrated_yaw_bias_refinement import fake_env
    from test_integrated_decoupled_yaw_precision_refinement import (
        fake_optimizer, set_turn_state, capture_other_pre_reset_terms,
    )
    from mjlab_microduck import integrated_yaw_precision_refinement as angular

    env = fake_env("control")
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = budget.MotorObserver(mode)
    term = env.reward_manager.get_term_cfg(previous.previous.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    q = env.scene["robot"].data.joint_pos.clone()
    for _ in range(24):
        env.scene["robot"].data.joint_pos.copy_(q)
        set_turn_state(env)
        observer.before(env)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, term)
        angular.capture_angular_metric(env)
        previous.capture_bias_metric(env)
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        observer.step(env, sample, {})
    for _ in range(20): runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["yaw_bias_weight"] == row["weighted_yaw_bias_reward_mean"] == 0.
    assert row["raw_yaw_residual_mae"] == .5
    assert row["optimizer_minibatches"] == 20
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["command_exposure_counts"]["positive-target"] == 96
    assert sum(row["command_exposure_counts"].values()) == 96
    assert row["command_exposure_cumulative"] == row["command_exposure_counts"]


@pytest.mark.parametrize("call", [budget.prepare_config, budget.MotorObserver])
def test_unknown_mode_fails_closed(call):
    with pytest.raises(ValueError): call("undeclared")
