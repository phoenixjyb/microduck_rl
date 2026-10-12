from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as angular
from mjlab_microduck import integrated_roll_pitch_allowance as allowance
from mjlab_microduck import integrated_straight_lesson as straight
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck.tasks import mdp


def test_modes_configs_and_lineage_change_only_roll_pitch_variance():
    assert allowance.MODES == {
        "smoke-control": (64, 5, 1283),
        "smoke-allowance": (64, 5, 1283),
        "benchmark-control": (256, 10, 1289),
        "benchmark-allowance": (256, 10, 1289),
        "control": (256, 1000, 1291),
        "allowance": (256, 1000, 1291),
    }
    assert allowance.ARMS == ("control", "allowance")
    assert allowance.PROTOCOL == "integrated-roll-pitch-allowance-continuation-v1"
    assert allowance.PARENT == previous.PARENT and allowance.PARENT_SHA == previous.PARENT_SHA
    assert allowance.PARENT_SHA == "c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae"
    assert allowance.PARENT.as_posix().endswith(
        "artifacts/training/angular-decoupling-ea24a14a-20261011/decoupled/model_999.pt")
    assert allowance.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE == 5.062500000000001e-05
    assert allowance.warm_load is previous.warm_load
    assert allowance.decision is previous.decision
    assert allowance.evaluate_case is straight.evaluate_case
    assert not hasattr(allowance, "COMMAND_COVERAGE")
    assert allowance.CAP_SECONDS == 2100
    assert allowance.TRAINING_CAP_SECONDS == dict(control=2100, allowance=2100)

    for phase in ("smoke", "benchmark", "full"):
        control_mode = "control" if phase == "full" else f"{phase}-control"
        treatment_mode = "allowance" if phase == "full" else f"{phase}-allowance"
        control, control_agent = allowance.prepare_config(control_mode)
        treatment, treatment_agent = allowance.prepare_config(treatment_mode)
        cterm, tterm = control.rewards[allowance.TERM], treatment.rewards[allowance.TERM]
        assert cterm.func is tterm.func is mdp.decoupled_body_angular_tracking
        assert cterm.weight == tterm.weight == 2.
        assert cterm.params == dict(command_name="twist", yaw_variance=.5,
                                    roll_pitch_variance=.5)
        assert tterm.params == dict(command_name="twist", yaw_variance=.5,
                                    roll_pitch_variance=1.)
        assert control.seed == treatment.seed == control_agent.seed == treatment_agent.seed
        assert control.scene.num_envs == treatment.scene.num_envs
        assert control.actions["joint_pos"].scale == treatment.actions["joint_pos"].scale == 1.
        assert control.actions["joint_pos"].clip is treatment.actions["joint_pos"].clip is None
        assert control_agent.algorithm.desired_kl == treatment_agent.algorithm.desired_kl == .01
        assert control_agent.algorithm.schedule == treatment_agent.algorithm.schedule == "adaptive"
        assert control_agent.algorithm.num_mini_batches == treatment_agent.algorithm.num_mini_batches == 4
        assert control_agent.algorithm.num_learning_epochs == treatment_agent.algorithm.num_learning_epochs == 5
        assert control_agent.actor.obs_normalization is treatment_agent.actor.obs_normalization is True
        assert control_agent.critic.obs_normalization is treatment_agent.critic.obs_normalization is True
        treatment_term_variance = tterm.params["roll_pitch_variance"]
        tterm.params["roll_pitch_variance"] = cterm.params["roll_pitch_variance"]
        treatment_agent.run_name = control_agent.run_name
        assert repr(control) == repr(treatment)
        assert control_agent == treatment_agent
        assert treatment_term_variance == allowance.expected_roll_pitch_variance(treatment_mode)
        assert not control.curriculum and not treatment.curriculum

    for mode, (worlds, updates, seed) in allowance.MODES.items():
        cfg, agent = allowance.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates


def test_default_decoupled_observer_keeps_half_variance():
    observer = angular.MotorObserver("control")
    assert observer.roll_pitch_variance == .5


def _fake_env_and_optimizer(monkeypatch, mode):
    from test_integrated_decoupled_yaw_precision_refinement import fake_optimizer
    from test_integrated_yaw_bias_refinement import fake_env

    env = fake_env("control")
    term = env.reward_manager.get_term_cfg(allowance.TERM)
    term.params["roll_pitch_variance"] = allowance.expected_roll_pitch_variance(mode)
    runner, value, calls = fake_optimizer(monkeypatch, env)
    return env, term, runner, value, calls


@pytest.mark.parametrize("mode,variance", [("control", .5), ("allowance", 1.)])
def test_both_arms_inherit_pre_reset_reward_and_optimizer_callbacks(monkeypatch, mode, variance):
    from test_integrated_decoupled_yaw_precision_refinement import (
        capture_other_pre_reset_terms,
        set_turn_state,
    )
    from mjlab_microduck import integrated_yaw_bias_refinement as bias
    from mjlab_microduck import integrated_yaw_precision_refinement as precision

    env, term, runner, value, calls = _fake_env_and_optimizer(monkeypatch, mode)
    observer = allowance.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    data = env.scene["robot"].data
    initial_position = data.joint_pos.clone()
    observed_rewards = []

    for _ in range(24):
        data.joint_pos.copy_(initial_position)
        set_turn_state(env)
        observer.before(env)
        assert term.params == dict(command_name="twist", yaw_variance=.5,
                                   roll_pitch_variance=variance)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, term)
        precision.capture_angular_metric(env)
        yaw_sq, xy_sq, actual, yaw_variance = env._microduck_precision_snapshot
        expected = 2. * (torch.exp(-yaw_sq / yaw_variance)
                         + torch.exp(-xy_sq / variance) - 1.)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        observed_rewards.append(actual.clone())
        bias.capture_bias_metric(env)
        assert env._microduck_yaw_bias_snapshot["cost"].shape == sample.mean_cost.shape
        data.root_link_ang_vel_b.zero_()
        observer.step(env, sample, {})
        assert env._microduck_precision_snapshot is None

    assert len(observed_rewards) == 24
    for _ in range(20):
        assert runner.alg.actor.get_kl_divergence("old", new="new") is value
    assert calls == [(('old',), {"new": "new"})] * 20

    row = observer.update(env)
    assert row["angular_aggregation"] == "additive_decoupled"
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["roll_pitch_variance"] == variance
    assert row["angular_tracking_weight"] == 2.
    assert row["angular_reward_sample_count"] == 4 * 24
    assert row["yaw_bias_weight"] == row["weighted_yaw_bias_reward_mean"] == 0.
    assert row["raw_yaw_residual_mae"] == .5
    assert row["optimizer_minibatches"] == 20 and row["desired_kl"] == .01
    assert len(row["minibatch_kl_means"]) == len(row["minibatch_lr_before"]) == 20
    assert sum(row["command_exposure_counts"].values()) == 4 * 24
    assert row["command_exposure_counts"]["positive-target"] == 4 * 24
    assert row["command_exposure_cumulative"] == row["command_exposure_counts"]
    assert observer.exposure_steps == 0 and observer.pending_exposure is None


@pytest.mark.parametrize("mode", ["bad", "smoke-bad", "benchmark-bad"])
def test_unknown_mode_fails_closed(mode):
    with pytest.raises(ValueError):
        allowance.prepare_config(mode)
    with pytest.raises(ValueError):
        allowance.MotorObserver(mode)


@pytest.mark.parametrize("fault", ["wrong-mode", "wrong-variance", "self-drift",
                                   "bool-variance", "nan-variance", "unsupported-variance"])
def test_before_rejects_mode_or_roll_pitch_variance_drift(monkeypatch, fault):
    from test_integrated_decoupled_yaw_precision_refinement import fake_env
    from test_integrated_decoupled_yaw_precision_refinement import fake_optimizer

    env = fake_env()
    fake_optimizer(monkeypatch, env)
    observer = allowance.MotorObserver("allowance")
    term = env.reward_manager.get_term_cfg(allowance.TERM)
    term.params["roll_pitch_variance"] = 1.
    if fault == "wrong-mode":
        observer.allowance_mode = "control"
    elif fault == "wrong-variance":
        term.params["roll_pitch_variance"] = .5
    elif fault == "self-drift":
        observer.roll_pitch_variance = .5
    elif fault == "bool-variance":
        observer.roll_pitch_variance = True
    elif fault == "nan-variance":
        observer.roll_pitch_variance = float("nan")
    else:
        observer.roll_pitch_variance = .75
    with pytest.raises(ValueError):
        observer.before(env)
