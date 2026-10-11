from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_yaw_bias_refinement as bias
from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as previous
from mjlab_microduck.tasks import mdp


def fake_env(mode="bias"):
    from test_integrated_decoupled_yaw_precision_refinement import fake_env as original

    env = original()
    env.num_envs, env.device, env.step_dt = 4, "cpu", .02
    env.episode_length_buf = torch.full((4,), 10, dtype=torch.long)
    terms = {k: env.reward_manager.get_term_cfg(k) for k in env.reward_manager.active_terms}
    terms[bias.TERM] = NS(func=mdp.body_yaw_bias_cost, weight=bias.bias_weight(mode),
                          params=dict(command_name="twist", tau_s=1.))
    env.reward_manager.get_term_cfg = terms.__getitem__
    env.reward_manager.active_terms = list(terms)
    env.reward_manager._step_reward = torch.zeros(4, len(terms))
    return env


def test_configs_change_only_declared_bias_reward_and_no_policy_or_plant_contract():
    assert bias.PROTOCOL == "integrated-body-yaw-bias-continuation-v1"
    assert bias.ARMS == ("control", "bias") and bias.CAP_SECONDS == 2100
    assert bias.PARENT_SHA == previous.PARENT_SHA
    assert bias.PARENT == previous.PARENT and bias.warm_load is previous.warm_load
    assert bias.PARENT_LEARNING_RATE == 5.062500000000001e-05
    assert bias.evaluate_case is previous.evaluate_case and bias.decision is previous.decision
    for prefix in ("smoke-", "benchmark-", ""):
        cfg, agent = bias.prepare_config(prefix+"control")
        treatment, ta = bias.prepare_config(prefix+"bias")
        assert cfg.rewards[bias.TERM].weight == 0.
        assert treatment.rewards[bias.TERM].weight == -.1
        assert treatment.rewards[bias.TERM].func is mdp.body_yaw_bias_cost
        assert treatment.rewards[bias.TERM].params == dict(command_name="twist", tau_s=1.)
        treatment.rewards[bias.TERM].weight = 0.
        ta.run_name = agent.run_name
        assert repr(cfg) == repr(treatment) and agent == ta
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["command_error"].weight == -.5
        assert cfg.rewards["action_rate_l2"].weight == -.8
        assert cfg.actions["joint_pos"].clip is None
        original, oa = previous.prepare_config("control")
        cfg.rewards.pop(bias.TERM)
        cfg.metrics.pop("microduck_yaw_bias")
        cfg.seed = original.seed
        cfg.scene.num_envs = cfg.scene.terrain.num_envs = original.scene.num_envs
        assert repr(cfg) == repr(original)
        for key in ("seed", "max_iterations", "experiment_name", "run_name"):
            setattr(agent, key, getattr(oa, key))
        assert agent == oa
    assert bias.MODES == {
        "smoke-control": (64, 5, 1031), "smoke-bias": (64, 5, 1031),
        "benchmark-control": (256, 10, 1033), "benchmark-bias": (256, 10, 1033),
        "control": (256, 1000, 1039), "bias": (256, 1000, 1039),
    }
    for mode, (worlds, updates, seed) in bias.MODES.items():
        cfg, agent = bias.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed and agent.max_iterations == updates
        assert agent.save_interval == 250


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-bias", "control", "bias"])
def test_actual_pre_reset_reward_activity_and_optimizer_capture(monkeypatch, mode):
    from test_integrated_decoupled_yaw_precision_refinement import (
        fake_optimizer, set_turn_state, capture_other_pre_reset_terms,
    )
    from mjlab_microduck import integrated_yaw_precision_refinement as angular

    env = fake_env(mode)
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = bias.MotorObserver(mode)
    term = env.reward_manager.get_term_cfg(previous.TERM)
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
        bt = env.reward_manager.get_term_cfg(bias.TERM)
        index = env.reward_manager.active_terms.index(bias.TERM)
        if bt.weight != 0:
            env.reward_manager._step_reward[:, index] = bt.weight * bt.func(env, **bt.params)
            before_metric = env._microduck_body_yaw_bias_state["signed_ema"].clone()
        bias.capture_bias_metric(env)
        if bt.weight != 0:
            assert torch.equal(before_metric, env._microduck_body_yaw_bias_state["signed_ema"])
        snapshot = env._microduck_yaw_bias_snapshot
        assert bool((snapshot["cost"] > 0).any())
        assert bool((snapshot["signed_residual"] != 0).any())
        env.scene["robot"].data.root_link_ang_vel_b.zero_()  # terminal auto-reset after hook
        observer.step(env, sample, {})
        assert env._microduck_yaw_bias_snapshot is None
    for _ in range(20):
        runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["yaw_bias_weight"] == bias.bias_weight(mode)
    assert row["yaw_bias_tau_s"] == 1.
    assert row["yaw_bias_cost_mean"] > 0.
    assert row["weighted_yaw_bias_reward_mean"] == pytest.approx(
        bias.bias_weight(mode) * row["yaw_bias_cost_mean"])
    assert row["raw_yaw_residual_mae"] == .5
    assert row["optimizer_minibatches"] == 20
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert not observer.bias_costs and not observer.raw_errors


@pytest.mark.parametrize("fault", ["func", "weight", "params", "stale"])
def test_before_rejects_contract_drift(monkeypatch, fault):
    from test_integrated_decoupled_yaw_precision_refinement import fake_optimizer
    env = fake_env(); fake_optimizer(monkeypatch, env)
    term = env.reward_manager.get_term_cfg(bias.TERM)
    if fault == "func": term.func = mdp.body_forward_tracking_cost
    elif fault == "weight": term.weight = .1
    elif fault == "params": term.params["tau_s"] = 2.
    else: env._microduck_yaw_bias_snapshot = {}
    with pytest.raises(ValueError): bias.MotorObserver("bias").before(env)


def test_missing_or_stale_bias_metric_fails_closed():
    env = fake_env()
    bias.capture_bias_metric(env)
    with pytest.raises(ValueError, match="previous yaw-bias snapshot"):
        bias.capture_bias_metric(env)
    snapshot = env._microduck_yaw_bias_snapshot
    assert snapshot is not env._microduck_body_yaw_bias_state
    env._microduck_body_yaw_bias_state["cost"].zero_()
    assert bool((snapshot["cost"] > 0).any())


@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite", "not-absolute", "wrong-reward"])
def test_step_rejects_invalid_pre_reset_snapshot(monkeypatch, fault):
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    env = fake_env()
    observer = bias.MotorObserver("bias")
    cost = torch.full((4,), .2)
    env._microduck_yaw_bias_snapshot = dict(
        signed_residual=cost.clone(), signed_ema=cost.clone(), cost=cost.clone())
    index = env.reward_manager.active_terms.index(bias.TERM)
    env.reward_manager._step_reward[:, index] = -.1 * cost
    if fault == "missing": env._microduck_yaw_bias_snapshot = None
    elif fault == "shape": env._microduck_yaw_bias_snapshot["cost"] = torch.ones(1)
    elif fault == "nonfinite": env._microduck_yaw_bias_snapshot["cost"][0] = float("nan")
    elif fault == "not-absolute": env._microduck_yaw_bias_snapshot["cost"][0] = -.2
    else: env.reward_manager._step_reward[:, index] = 0.
    with pytest.raises(ValueError):
        observer.step(env, NS(mean_cost=torch.zeros(4)), {})


def test_unknown_modes_fail_closed():
    for call in (bias.bias_weight, bias.prepare_config, bias.MotorObserver):
        with pytest.raises(ValueError): call("not-declared")
