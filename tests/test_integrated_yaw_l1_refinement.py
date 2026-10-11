from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as decoupled
from mjlab_microduck import integrated_yaw_l1_refinement as yaw_l1
from mjlab_microduck.tasks import mdp


def fake_env(mode):
    from test_integrated_yaw_bias_refinement import fake_env as prior_fake_env

    env = prior_fake_env("control")
    terms = {key: env.reward_manager.get_term_cfg(key)
             for key in env.reward_manager.active_terms}
    terms[yaw_l1.TERM] = NS(func=mdp.body_yaw_tracking_l1_cost,
                            weight=yaw_l1.weight(mode), params={"command_name": "twist"})
    env.reward_manager.active_terms = list(terms)
    env.reward_manager.get_term_cfg = terms.__getitem__
    env.reward_manager._step_reward = torch.zeros(env.num_envs, len(terms))
    env._microduck_yaw_l1_snapshot = None
    return env


def test_recipe_changes_only_the_declared_yaw_weight_and_preserves_inherited_contract():
    assert yaw_l1.PROTOCOL == "integrated-body-yaw-l1-continuation-v1"
    assert yaw_l1.ARMS == ("control", "yawl1")
    assert yaw_l1.WEIGHT == -.5
    assert yaw_l1.TERM == "yaw_l1_error"
    assert yaw_l1.PARENT == previous.PARENT and yaw_l1.PARENT_SHA == previous.PARENT_SHA
    assert yaw_l1.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE
    assert yaw_l1.warm_load is previous.warm_load
    assert yaw_l1.evaluate_case is previous.evaluate_case
    assert yaw_l1.decision is previous.decision
    assert yaw_l1.EXPOSURE_BINS is previous.EXPOSURE_BINS

    assert yaw_l1.MODES == {
        "smoke-control": (64, 5, 1091), "smoke-yawl1": (64, 5, 1091),
        "benchmark-control": (256, 10, 1093), "benchmark-yawl1": (256, 10, 1093),
        "control": (256, 1000, 1097), "yawl1": (256, 1000, 1097),
    }
    for control_mode, treatment_mode in (
            ("smoke-control", "smoke-yawl1"),
            ("benchmark-control", "benchmark-yawl1"),
            ("control", "yawl1")):
        cfg, agent = yaw_l1.prepare_config(control_mode)
        treatment, treatment_agent = yaw_l1.prepare_config(treatment_mode)
        cterm, tterm = cfg.rewards[yaw_l1.TERM], treatment.rewards[yaw_l1.TERM]
        assert cterm.func is tterm.func is mdp.body_yaw_tracking_l1_cost
        assert cterm.weight == 0. and tterm.weight == -.5
        assert cterm.params == tterm.params == {"command_name": "twist"}
        assert cfg.metrics["microduck_yaw_l1"].func is yaw_l1.capture_yaw_l1_metric
        assert treatment.metrics["microduck_yaw_l1"].func is yaw_l1.capture_yaw_l1_metric
        assert agent.algorithm.symmetry_cfg is treatment_agent.algorithm.symmetry_cfg is None
        assert cfg.seed == treatment.seed == agent.seed == treatment_agent.seed
        assert cfg.scene.num_envs == treatment.scene.num_envs
        tterm.weight = cterm.weight
        treatment_agent.run_name = agent.run_name
        assert repr(cfg) == repr(treatment)
        assert agent == treatment_agent

        inherited, inherited_agent = previous.prepare_config("control")
        cfg.rewards.pop(yaw_l1.TERM)
        cfg.metrics.pop("microduck_yaw_l1")
        cfg.seed = inherited.seed
        cfg.scene.num_envs = cfg.scene.terrain.num_envs = inherited.scene.num_envs
        assert repr(cfg) == repr(inherited)
        for key in ("seed", "max_iterations", "experiment_name", "run_name"):
            setattr(agent, key, getattr(inherited_agent, key))
        assert agent == inherited_agent
        assert cfg.actions["joint_pos"].clip is None
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["command_error"].weight == -.5
        assert not cfg.curriculum

    assert yaw_l1.CAP_SECONDS == previous.CAP_SECONDS == 2100
    assert yaw_l1.TRAINING_CAP_SECONDS == {"control": 2100, "yawl1": 2100}
    for mode, (worlds, updates, seed) in yaw_l1.MODES.items():
        cfg, agent = yaw_l1.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert cfg.rewards[yaw_l1.TERM].weight == yaw_l1.weight(mode)
        assert cfg.rewards[decoupled.TERM].weight == decoupled.WEIGHT
        assert cfg.rewards["yaw_l1_error"].params == {"command_name": "twist"}
        assert cfg.commands["twist"].resampling_time_range == (3., 5.)
        assert agent.algorithm.symmetry_cfg is None


@pytest.mark.parametrize("mode", ["control", "yawl1", "smoke-control", "smoke-yawl1"])
def test_observer_checks_actual_pre_reset_l1_reward_cost_and_dwell(monkeypatch, mode):
    from test_integrated_decoupled_yaw_precision_refinement import (
        capture_other_pre_reset_terms, fake_optimizer, set_turn_state,
    )
    from mjlab_microduck import integrated_yaw_bias_refinement as bias
    from mjlab_microduck import integrated_yaw_precision_refinement as angular

    env = fake_env(mode)
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = yaw_l1.MotorObserver(mode)
    precision_term = env.reward_manager.get_term_cfg(angular.TERM)
    l1_term = env.reward_manager.get_term_cfg(yaw_l1.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))

    for _ in range(24):
        set_turn_state(env)
        observer.before(env)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, precision_term)
        angular.capture_angular_metric(env)
        bias.capture_bias_metric(env)
        index = env.reward_manager.active_terms.index(yaw_l1.TERM)
        cost = l1_term.func(env, **l1_term.params)
        env.reward_manager._step_reward[:, index] = yaw_l1.weight(mode) * cost
        yaw_l1.capture_yaw_l1_metric(env)
        snapshot = env._microduck_yaw_l1_snapshot
        assert snapshot[3] == env.common_step_counter
        assert torch.equal(snapshot[0], cost) and torch.equal(snapshot[1].abs(), cost)
        assert bool((snapshot[0] > 0).any())
        expected_reward = yaw_l1.weight(mode) * cost
        assert torch.allclose(snapshot[2], expected_reward)
        if mode.endswith("control"):
            assert torch.equal(snapshot[2], torch.zeros_like(snapshot[2]))
        else:
            assert bool((snapshot[2] <= 0).all()) and bool((snapshot[2] < 0).any())

        # Simulate auto-reset after the metric hook: the observer must retain the
        # same-step pre-reset residual and reward until it consumes the snapshot.
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        env.episode_length_buf.zero_()
        observer.step(env, sample, {})
        assert env._microduck_yaw_l1_snapshot is None

    for _ in range(20):
        runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["yaw_l1_weight"] == yaw_l1.weight(mode)
    assert row["yaw_l1_cost_mean"] > 0.
    assert row["weighted_yaw_l1_reward_mean"] == pytest.approx(
        yaw_l1.weight(mode) * row["yaw_l1_cost_mean"])
    assert row["yaw_l1_signed_residual_mean"] == pytest.approx(.5)
    assert row["optimizer_minibatches"] == 20
    assert row["command_exposure_counts"]["positive-target"] == 96
    assert sum(row["command_exposure_counts"].values()) == 96
    assert row["command_exposure_cumulative"] == row["command_exposure_counts"]
    assert (not observer.yaw_l1_costs and not observer.yaw_l1_rewards
            and not observer.yaw_l1_signed_errors)


@pytest.mark.parametrize("fault", ["function", "weight", "params", "stale"])
def test_before_fails_closed_on_yaw_l1_contract_drift(monkeypatch, fault):
    from test_integrated_decoupled_yaw_precision_refinement import fake_optimizer

    env = fake_env("yawl1")
    fake_optimizer(monkeypatch, env)
    term = env.reward_manager.get_term_cfg(yaw_l1.TERM)
    if fault == "function":
        term.func = mdp.body_forward_tracking_cost
    elif fault == "weight":
        term.weight = .5
    elif fault == "params":
        term.params["command_name"] = "other"
    else:
        env._microduck_yaw_l1_snapshot = (torch.ones(4), torch.ones(4), torch.ones(4), 0)
    with pytest.raises(ValueError):
        yaw_l1.MotorObserver("yawl1").before(env)


def test_metric_rejects_duplicate_snapshot_and_step_rejects_missing_stale_or_positive_reward(
        monkeypatch):
    from test_integrated_decoupled_yaw_precision_refinement import fake_optimizer

    env = fake_env("yawl1")
    fake_optimizer(monkeypatch, env)
    term = env.reward_manager.get_term_cfg(yaw_l1.TERM)
    index = env.reward_manager.active_terms.index(yaw_l1.TERM)
    env.command_manager.get_command = lambda _: torch.tensor([[.3, 0., .2]] * 4)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor([.2, 1.2, .2, 1.2])
    env.common_step_counter += 1
    cost = term.func(env, **term.params)
    env.reward_manager._step_reward[:, index] = yaw_l1.WEIGHT * cost
    yaw_l1.capture_yaw_l1_metric(env)
    with pytest.raises(ValueError, match="previous yaw-L1 snapshot"):
        yaw_l1.capture_yaw_l1_metric(env)

    # Isolate the new observer's snapshot checks from inherited telemetry checks.
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    observer = yaw_l1.MotorObserver("yawl1")
    sample = NS(mean_cost=torch.zeros(4))
    valid = (cost.clone(), torch.tensor([0., -1., 0., -1.]),
             (yaw_l1.WEIGHT * cost).clone(), env.common_step_counter)
    env._microduck_yaw_l1_snapshot = None
    with pytest.raises(ValueError, match="same-step pre-reset"):
        observer.step(env, sample, {})

    env._microduck_yaw_l1_snapshot = (valid[0], valid[1], valid[2], valid[3] - 1)
    with pytest.raises(ValueError, match="same-step pre-reset"):
        observer.step(env, sample, {})

    env._microduck_yaw_l1_snapshot = (valid[0], valid[1], torch.full((4,), .1), valid[3])
    with pytest.raises(ValueError, match="actual nonpositive"):
        observer.step(env, sample, {})


@pytest.mark.parametrize("call", [yaw_l1.weight, yaw_l1.prepare_config, yaw_l1.MotorObserver])
def test_unknown_modes_fail_closed(call):
    with pytest.raises(ValueError):
        call("unknown-mode")
