from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_yaw_precision_refinement as precision


def fake_optimizer(monkeypatch, target=.01):
    monkeypatch.setattr(kl.metadata, "version", lambda _: "5.0.1")
    calls = []
    value = torch.tensor([0., .01, .03], requires_grad=True)
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return value
    alg = NS(actor=NS(get_kl_divergence=original), is_multi_gpu=False, schedule="adaptive",
             desired_kl=target, num_mini_batches=4, num_learning_epochs=5,
             learning_rate=1e-5, optimizer=NS(param_groups=[{"lr": 1e-5}]))
    runner, env = NS(alg=alg), NS()
    return runner, env, value, calls


def test_only_kl_target_changes_between_matched_arms():
    cfg, a = kl.prepare_config("control")
    other, b = kl.prepare_config("klbudget")
    assert repr(cfg) == repr(other) and a.algorithm.desired_kl == .01 and b.algorithm.desired_kl == .02
    b.algorithm.desired_kl = .01; b.run_name = a.run_name
    assert a == b and a.seed == cfg.seed == 1009
    orig, oa = precision.prepare_config("control")
    cfg.seed = orig.seed
    assert repr(cfg) == repr(orig)
    a.seed, a.experiment_name, a.run_name = oa.seed, oa.experiment_name, oa.run_name
    assert a == oa
    for mode, (worlds, updates, seed) in kl.MODES.items():
        c, agent = kl.prepare_config(mode)
        assert c.scene.num_envs == c.scene.terrain.num_envs == worlds
        assert c.seed == agent.seed == seed and agent.max_iterations == updates
        assert agent.algorithm.desired_kl == kl.target(mode)
        assert c.rewards[precision.TERM].params["yaw_variance"] == .5
    assert kl.evaluate_case is precision.evaluate_case and kl.decision is precision.decision


def test_exact_parent_and_warm_counters(monkeypatch):
    runner, env, _, _ = fake_optimizer(monkeypatch)
    calls = []
    receipt = {"learning_rate": 1e-5}
    def load(*a, **kw):
        calls.append((a, kw)); return receipt
    monkeypatch.setattr(kl.motor, "warm_load", load)
    assert kl.warm_load(runner, env) is receipt
    assert calls == [((runner, env, kl.PARENT), dict(parent_sha=kl.PARENT_SHA, parent_iteration=999, parent_steps=24000))]
    assert kl.PARENT.parent.name == "control" and "yaw-precision-4447fe64" in str(kl.PARENT)
    assert kl.PARENT_SHA == "3e0b01bce31f8b73a535fa44bbfe92ce522908c03916269ad6ae203432b6249a"


@pytest.mark.parametrize("target", [.01, .02])
def test_capture_returns_exact_tensor_preserves_rng_gradients_and_summarizes(monkeypatch, target):
    runner, env, value, calls = fake_optimizer(monkeypatch, target)
    original = runner.alg.actor.get_kl_divergence
    rng = torch.random.get_rng_state().clone()
    kl.install_optimizer_capture(runner, env)
    for i in range(20):
        rate = 1e-5 * (1.5 if i%2 else 1.)
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"] = rate
        assert runner.alg.actor.get_kl_divergence("old", new="new") is value
    assert calls == [(("old",), {"new": "new"})]*20
    assert torch.equal(rng, torch.random.get_rng_state())
    assert value.requires_grad and value.grad is None
    row = kl.optimizer_summary(env, target)
    assert row["optimizer_minibatches"] == 20 and row["desired_kl"] == target
    assert row["kl_mean_mean"] == pytest.approx(.04/3)
    assert row["kl_high_count"] == 0 and row["kl_low_positive_count"] == 0
    assert row["lr_before_min"] == 1e-5 and row["lr_before_max"] == row["lr_after_update"]
    assert row["lr_after_update"] == pytest.approx(1.5e-5)
    assert len(row["minibatch_kl_means"]) == len(row["minibatch_lr_before"]) == 20
    assert row["minibatch_lr_applied_inferred"] == row["minibatch_lr_before"][1:]+[row["lr_after_update"]]
    assert not env._microduck_optimizer_audit.samples
    value.sum().backward(); assert value.grad.tolist() == [1.]*3
    # The wrapper is instance-only, not a global algorithm/class replacement.
    assert original("again") is value


@pytest.mark.parametrize("change", ["schedule", "target", "group_lr", "epochs", "multi_gpu"])
def test_live_optimizer_mismatch_fails_closed(monkeypatch, change):
    runner, env, _, _ = fake_optimizer(monkeypatch)
    kl.install_optimizer_capture(runner, env)
    if change == "schedule": runner.alg.schedule = "fixed"
    if change == "target": runner.alg.desired_kl = .04
    if change == "group_lr": runner.alg.optimizer.param_groups[0]["lr"] = .001
    if change == "epochs": runner.alg.num_learning_epochs = 4
    if change == "multi_gpu": runner.alg.is_multi_gpu = True
    with pytest.raises(ValueError): runner.alg.actor.get_kl_divergence(None)


@pytest.mark.parametrize("count", [0, 19, 21])
def test_incomplete_or_extra_minibatches_rejected(monkeypatch, count):
    runner, env, _, _ = fake_optimizer(monkeypatch)
    kl.install_optimizer_capture(runner, env)
    for _ in range(count): runner.alg.actor.get_kl_divergence(None)
    with pytest.raises(ValueError): kl.optimizer_summary(env, .01)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_raw_nonfinite_kl_is_not_sanitized(monkeypatch, value):
    runner, env, raw, _ = fake_optimizer(monkeypatch)
    raw.data[0] = value
    kl.install_optimizer_capture(runner, env)
    for _ in range(20): assert runner.alg.actor.get_kl_divergence(None) is raw
    with pytest.raises(ValueError): kl.optimizer_summary(env, .01)


def test_runtime_and_double_install_guard(monkeypatch):
    runner, env, _, _ = fake_optimizer(monkeypatch)
    monkeypatch.setattr(kl.metadata, "version", lambda _: "other")
    with pytest.raises(ValueError): kl.install_optimizer_capture(runner, env)
    monkeypatch.setattr(kl.metadata, "version", lambda _: "5.0.1")
    kl.install_optimizer_capture(runner, env)
    with pytest.raises(ValueError): kl.install_optimizer_capture(runner, env)


def test_nonparent_initial_lr_rejected(monkeypatch):
    runner, env, _, _ = fake_optimizer(monkeypatch)
    runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"] = .001
    with pytest.raises(ValueError, match="initial parent LR"): kl.install_optimizer_capture(runner, env)


def test_explicit_parent_lr_is_checked_not_reset(monkeypatch):
    runner, env, value, _ = fake_optimizer(monkeypatch)
    inherited = 5.062500000000001e-05
    runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"] = inherited
    kl.install_optimizer_capture(runner, env, initial_lr=inherited)
    assert env._microduck_optimizer_audit.initial_lr == inherited
    assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"] == inherited
    assert runner.alg.actor.get_kl_divergence(None) is value
    assert env._microduck_optimizer_audit.samples[0][1] == inherited


@pytest.mark.parametrize("initial", [True, None, 1e-6, .1, float("nan"), float("inf")])
def test_invalid_predeclared_parent_lr_fails_before_capture(monkeypatch, initial):
    runner, env, _, _ = fake_optimizer(monkeypatch)
    with pytest.raises(ValueError):
        kl.install_optimizer_capture(runner, env, initial_lr=initial)
    assert not hasattr(env, "_microduck_optimizer_audit")
    assert runner.alg.learning_rate == 1e-5


def test_exact_adaptive_threshold_counts(monkeypatch):
    runner, env, raw, _ = fake_optimizer(monkeypatch)
    kl.install_optimizer_capture(runner, env)
    for i in range(20):
        raw.data.fill_((0., .005, .02, .001, .021)[i%5])
        runner.alg.actor.get_kl_divergence(None)
    row = kl.optimizer_summary(env, .01)
    assert row["kl_high_count"] == 4 and row["kl_low_positive_count"] == 4


@pytest.mark.parametrize("mode", ["control", "klbudget"])
def test_complete_live_rollout_and_optimizer_update(monkeypatch, mode):
    from test_integrated_yaw_precision_refinement import fake_env
    from mjlab_microduck import integrated_tracking_refinement as tracking
    from mjlab_microduck import integrated_joint_limit_refinement as repair
    from mjlab_microduck import integrated_forward_tracking_refinement as forward
    env = fake_env()
    runner, _, _, _ = fake_optimizer(monkeypatch, kl.target(mode))
    kl.install_optimizer_capture(runner, env)
    observer = kl.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    for _ in range(24):
        observer.before(env)
        tracking.capture_tracking_metric(env); repair.capture_limit_metric(env); forward.capture_forward_metric(env)
        term = env.reward_manager.get_term_cfg(precision.TERM)
        env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*-.5
        env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
        env.reward_manager._step_reward[:, 4] = 2*term.func(env, **term.params)
        precision.capture_angular_metric(env)
        observer.step(env, sample, {})
        env.common_step_counter += 1
    for _ in range(20): runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["desired_kl"] == kl.target(mode) and row["optimizer_minibatches"] == 20
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["motor_weight_min"] == -1. and row["limit_weight_min"] == -4.
    assert not env._microduck_optimizer_audit.samples
    observer.before(env)
    runner.alg.actor.get_kl_divergence(None)
    with pytest.raises(ValueError, match="samples consumed"): observer.before(env)
