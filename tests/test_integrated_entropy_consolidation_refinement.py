from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_entropy_consolidation_refinement as entropy
from mjlab_microduck import integrated_training_budget_refinement as previous


def gaussian(std=None):
    from rsl_rl.modules.distribution import GaussianDistribution

    result = GaussianDistribution(14, init_std=.35, std_type="scalar")
    if std is not None:
        with torch.no_grad():
            result.std_param.copy_(torch.as_tensor(std, dtype=result.std_param.dtype))
    return result


def action_runtime():
    from test_action_response_capture import runtime as capture_runtime

    env, robot, actor_obs, stream, _command = capture_runtime.__wrapped__()
    env._microduck_motor_step_stream = stream
    return env, robot, actor_obs, stream


def install_optimizer_audit(env, *, coefficient=.01, std=None):
    distribution = gaussian(std)
    alg = NS(entropy_coef=coefficient, actor=NS(distribution=distribution))
    env._microduck_optimizer_audit = NS(alg=alg)
    return distribution


def test_recipe_is_entropy_only_matched_to_the_inherited_c81_training_contract():
    assert entropy.PROTOCOL == "integrated-entropy-consolidation-continuation-v1"
    assert entropy.ARMS == ("control", "consolidate")
    assert entropy.MODES == {
        "smoke-control": (64, 5, 1123), "smoke-consolidate": (64, 5, 1123),
        "benchmark-control": (256, 10, 1129), "benchmark-consolidate": (256, 10, 1129),
        "control": (256, 1000, 1151), "consolidate": (256, 1000, 1151),
    }
    assert entropy.CAP_SECONDS == 2100
    assert entropy.TRAINING_CAP_SECONDS == {"control": 2100, "consolidate": 2100}
    assert entropy.PARENT == previous.PARENT and entropy.PARENT_SHA == previous.PARENT_SHA
    assert entropy.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE
    assert entropy.warm_load is previous.warm_load
    assert entropy.evaluate_case is previous.evaluate_case
    assert entropy.decision is previous.decision
    assert entropy.EXPOSURE_BINS == previous.EXPOSURE_BINS

    for control_mode, consolidate_mode in (
        ("smoke-control", "smoke-consolidate"),
        ("benchmark-control", "benchmark-consolidate"),
        ("control", "consolidate"),
    ):
        control, control_agent = entropy.prepare_config(control_mode)
        treatment, treatment_agent = entropy.prepare_config(consolidate_mode)
        assert control_agent.algorithm.entropy_coef == .01
        assert treatment_agent.algorithm.entropy_coef == 0.
        assert control.seed == treatment.seed == control_agent.seed == treatment_agent.seed
        assert control.scene.num_envs == treatment.scene.num_envs
        assert set(control.rewards) == set(treatment.rewards)
        assert control.curriculum == treatment.curriculum == {}
        assert control.commands["head_pose"] == treatment.commands["head_pose"]
        assert control.commands["twist"] == treatment.commands["twist"]
        assert control.actions["joint_pos"] == treatment.actions["joint_pos"]
        assert control.actions["joint_pos"].clip is None
        assert control_agent.algorithm.desired_kl == treatment_agent.algorithm.desired_kl == .01
        assert control_agent.algorithm.schedule == treatment_agent.algorithm.schedule == "adaptive"
        assert control_agent.algorithm.num_mini_batches == treatment_agent.algorithm.num_mini_batches == 4
        assert control_agent.algorithm.num_learning_epochs == treatment_agent.algorithm.num_learning_epochs == 5
        assert control_agent.algorithm.symmetry_cfg is treatment_agent.algorithm.symmetry_cfg is None

        treatment_agent.algorithm.entropy_coef = .01
        treatment_agent.run_name = control_agent.run_name
        assert repr(control) == repr(treatment)
        assert control_agent == treatment_agent

        inherited, inherited_agent = previous.prepare_config("control")
        treatment.seed = inherited.seed
        treatment.scene.num_envs = treatment.scene.terrain.num_envs = inherited.scene.num_envs
        assert repr(treatment) == repr(inherited)
        for key in ("seed", "max_iterations", "experiment_name", "run_name"):
            setattr(treatment_agent, key, getattr(inherited_agent, key))
        assert treatment_agent == inherited_agent

    for mode, (worlds, updates, seed) in entropy.MODES.items():
        cfg, agent = entropy.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed and agent.max_iterations == updates
        assert agent.save_interval == 250
        assert agent.algorithm.entropy_coef == entropy.entropy_coefficient(mode)


@pytest.mark.parametrize("bad", ["entropy", "distribution", "shape", "frozen", "zero", "negative", "nan", "inf"])
def test_action_std_requires_live_declared_entropy_and_actual_positive_scalar_parameter(bad):
    env = NS()
    distribution = install_optimizer_audit(env)
    if bad == "entropy":
        env._microduck_optimizer_audit.alg.entropy_coef = .02
    elif bad == "distribution":
        env._microduck_optimizer_audit.alg.actor.distribution = NS(
            std_type="scalar", std_param=torch.nn.Parameter(torch.ones(14)))
    elif bad == "shape":
        distribution.std_param = torch.nn.Parameter(torch.ones(1))
    elif bad == "frozen":
        distribution.std_param.requires_grad_(False)
    elif bad == "zero":
        with torch.no_grad(): distribution.std_param[0] = 0.
    elif bad == "negative":
        with torch.no_grad(): distribution.std_param[0] = -.1
    elif bad == "nan":
        with torch.no_grad(): distribution.std_param[0] = float("nan")
    elif bad == "inf":
        with torch.no_grad(): distribution.std_param[0] = float("inf")
    with pytest.raises(ValueError):
        entropy.action_std(env, .01)


def test_observer_uses_named_action_binding_and_reads_std_without_policy_sampling(monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env, robot, _actor_obs, stream = action_runtime()
    distribution = install_optimizer_audit(env)
    calls = []
    # These methods would draw/use a policy distribution; telemetry must only
    # read the raw learned parameter.
    for method in ("sample", "update", "forward"):
        if hasattr(distribution, method):
            monkeypatch.setattr(distribution, method,
                                lambda *args, _method=method, **kwargs: calls.append(_method))
    observer = entropy.MotorObserver("control")
    rng = torch.random.get_rng_state().clone()
    before = distribution.std_param.detach().clone()
    observer.before(env)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(observer.entropy_before, before)
    assert observer.entropy_steps == 1
    assert observer.entropy_action_names == tuple(stream.names)
    assert observer.entropy_action_names == tuple(env.action_manager.get_term("joint_pos").target_names)
    assert calls == []

    term = env.action_manager.get_term("joint_pos")
    stream.names = tuple(reversed(stream.names))
    with pytest.raises(ValueError):
        observer.before(env)
    stream.names = tuple(term.target_names)
    assert calls == []


@pytest.mark.parametrize("fault", ["active-terms", "order", "ids", "clip", "scale"])
def test_observer_fails_closed_on_action_column_contract_drift(monkeypatch, fault):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env, _robot, _actor_obs, stream = action_runtime()
    install_optimizer_audit(env)
    term = env.action_manager.get_term("joint_pos")
    if fault == "active-terms":
        env.action_manager._term_names = ["other"]
    elif fault == "order":
        stream.names = tuple(reversed(stream.names))
    elif fault == "ids":
        stream.joint_ids = tuple(reversed(stream.joint_ids))
    elif fault == "clip":
        term.cfg.clip = {".*": (-1., 1.)}
    else:
        term.cfg.scale = 2.
    with pytest.raises(ValueError):
        entropy.MotorObserver("control").before(env)


def test_24_step_entropy_snapshot_is_stable_during_collection_and_reports_post_update_std(monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "update", lambda *args: {})
    env, _robot, _actor_obs, _stream = action_runtime()
    distribution = install_optimizer_audit(env, std=torch.full((14,), .35))
    observer = entropy.MotorObserver("control")
    rng = torch.random.get_rng_state().clone()
    for _ in range(24):
        observer.before(env)
    assert observer.entropy_steps == 24
    assert torch.equal(observer.entropy_before, torch.full((14,), .35))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(distribution.std_param, observer.entropy_before)

    with torch.no_grad():
        distribution.std_param.copy_(torch.linspace(.2, .4, 14))
    after = distribution.std_param.detach().clone()
    row = observer.update(env)
    expected_before = torch.full((14,), .35)
    expected_entropy_before = float((expected_before.log() + .5 * torch.log(torch.tensor(2 * torch.pi * torch.e))).sum())
    expected_entropy_after = float((after.log() + .5 * torch.log(torch.tensor(2 * torch.pi * torch.e))).sum())
    assert row["entropy_coefficient"] == .01
    assert row["action_std_protocol"] == "direct-positive-scalar-gaussian-parameter-v1"
    assert row["action_std_names"] == list(env.action_manager.get_term("joint_pos").target_names)
    assert row["action_std_before_update"] == pytest.approx(expected_before.tolist())
    assert row["action_std_after_update"] == pytest.approx(after.tolist())
    assert row["action_distribution_entropy_before"] == pytest.approx(expected_entropy_before)
    assert row["action_distribution_entropy_after"] == pytest.approx(expected_entropy_after)
    assert observer.entropy_before is None and observer.entropy_steps == 0


@pytest.mark.parametrize("fault", ["std-drift", "entropy-drift", "extra-step", "incomplete-update"])
def test_observer_rejects_rollout_drift_or_incomplete_action_std_observation(monkeypatch, fault):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "update", lambda *args: {})
    env, _robot, _actor_obs, _stream = action_runtime()
    distribution = install_optimizer_audit(env)
    observer = entropy.MotorObserver("control")
    for _ in range(24):
        observer.before(env)
    if fault == "std-drift":
        with torch.no_grad(): distribution.std_param[0] += .1
        with pytest.raises(ValueError, match="unchanged during"):
            observer.before(env)
    elif fault == "entropy-drift":
        env._microduck_optimizer_audit.alg.entropy_coef = 0.
        with pytest.raises(ValueError, match="entropy coefficient"):
            observer.update(env)
    elif fault == "extra-step":
        with pytest.raises(ValueError, match="bounded action variance"):
            observer.before(env)
    else:
        observer.entropy_steps = 23
        with pytest.raises(ValueError, match="complete 24-step"):
            observer.update(env)


@pytest.mark.parametrize("mode", ["control", "consolidate"])
def test_full_inherited_reward_and_20_minibatch_update_path_keeps_entropy_capture(monkeypatch, mode):
    from test_integrated_yaw_bias_refinement import fake_env
    from test_integrated_decoupled_yaw_precision_refinement import (
        capture_other_pre_reset_terms, fake_optimizer, set_turn_state,
    )
    from mjlab_microduck import integrated_yaw_precision_refinement as angular
    from mjlab_microduck import action_response_capture

    env = fake_env("control")
    runner, _value, _calls = fake_optimizer(monkeypatch, env)
    runner.alg.entropy_coef = entropy.entropy_coefficient(mode)
    runner.alg.actor.distribution = gaussian(std=torch.full((14,), .35))
    names = tuple(action_response_capture.JOINTS)
    env._microduck_motor_step_stream = NS(names=names, joint_ids=tuple(range(14)))
    # The named binding is independently exercised with a real ActionManager
    # above; this parent-reward fixture focuses on inherited metric/optimizer
    # state and supplies the same checked layout shape to the observer.
    monkeypatch.setattr(action_response_capture, "_bind_action_layout",
                        lambda _env, _robot, stream: (None, NS(target_names=stream.names), tuple(stream.joint_ids)))
    observer = entropy.MotorObserver(mode)
    term = env.reward_manager.get_term_cfg(angular.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    for _ in range(24):
        set_turn_state(env)
        observer.before(env)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, term)
        angular.capture_angular_metric(env)
        # The parent yaw-bias observer is part of the unchanged training stack.
        from mjlab_microduck import integrated_yaw_bias_refinement as bias
        bias.capture_bias_metric(env)
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        observer.step(env, sample, {})
    for _ in range(20):
        runner.alg.actor.get_kl_divergence(None)
    with torch.no_grad():
        runner.alg.actor.distribution.std_param.mul_(.9)
    row = observer.update(env)
    assert row["optimizer_minibatches"] == 20
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["yaw_bias_weight"] == row["weighted_yaw_bias_reward_mean"] == 0.
    assert row["command_exposure_counts"]["positive-target"] == 96
    assert sum(row["command_exposure_counts"].values()) == 96
    assert row["entropy_coefficient"] == entropy.entropy_coefficient(mode)
    assert row["action_std_before_update"] == pytest.approx([.35] * 14)
    assert row["action_std_after_update"] == pytest.approx([.315] * 14)
