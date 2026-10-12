from types import SimpleNamespace as NS
import inspect

import pytest
import torch

from mjlab_microduck import integrated_straight_lesson as lesson
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor


@pytest.mark.parametrize("prefix", ["smoke-", "benchmark-", ""])
def test_only_declared_twist_support_differs(prefix):
    cfg, agent = lesson.prepare_config(prefix + "control")
    treatment, ta = lesson.prepare_config(prefix + "straight")
    assert treatment.commands["twist"].ranges.lin_vel_x == (.3, .3)
    assert treatment.commands["twist"].ranges.ang_vel_z == (0., 0.)
    assert treatment.commands["twist"].rel_forward_envs == .25
    assert treatment.commands["twist"].rel_standing_envs == .15
    treatment.commands["twist"].ranges.lin_vel_x = cfg.commands["twist"].ranges.lin_vel_x
    treatment.commands["twist"].ranges.ang_vel_z = cfg.commands["twist"].ranges.ang_vel_z
    assert repr(cfg) == repr(treatment)
    ta.run_name = agent.run_name
    assert ta == agent
    assert not cfg.curriculum and cfg.actions["joint_pos"].clip is None
    assert lesson.PARENT == previous.PARENT and lesson.PARENT_SHA == previous.PARENT_SHA
    assert lesson.warm_load is previous.warm_load and lesson.decision is previous.decision
    assert cfg.rewards["motor_load"].weight == -1.
    assert cfg.rewards["joint_limit"].weight == -4.


def test_coverage_default_and_explicit_complete_declaration():
    assert inspect.signature(lesson.base.training).parameters["required_command_coverage"].default == lesson.base.FULL_COMMAND_COVERAGE
    for mode in previous.MODES:
        assert motor.command_coverage(previous, mode) == lesson.base.FULL_COMMAND_COVERAGE
    for mode in lesson.MODES:
        expected = lesson.base.STRAIGHT_COMMAND_COVERAGE if mode.endswith("straight") else lesson.base.FULL_COMMAND_COVERAGE
        assert motor.command_coverage(lesson, mode) == expected
    for declaration in ({}, {"control": ("idle",), "straight": ("idle", "straight")},
                        {"control": lesson.base.FULL_COMMAND_COVERAGE, "straight": ["idle", "straight"]},
                        {**lesson.COMMAND_COVERAGE, "extra": ("idle",)}):
        bad = NS(MODES=lesson.MODES, ARMS=lesson.ARMS, COMMAND_COVERAGE=declaration)
        with pytest.raises(ValueError): motor.command_coverage(bad, "straight")
    with pytest.raises(ValueError): motor.command_coverage(lesson, "unknown")


@pytest.mark.parametrize("coverage", [None, [], ["idle", "straight"], ("idle",),
                                      lesson.base.FULL_COMMAND_COVERAGE])
def test_straight_receipt_cannot_omit_or_substitute_coverage(monkeypatch, coverage):
    contract = motor.training_contract(lesson, "straight")
    receipt = dict(source="source", protocol=lesson.PROTOCOL, mode="straight",
        status="training-complete-not-accepted", updates=1000, worlds=256, seed=1279,
        steps=24000, transitions=6144000, checkpoint_sha256="weights",
        warm_start=dict(parent_sha256=lesson.PARENT_SHA))
    monkeypatch.setattr(lesson.base, "sha256", lambda _: "weights")
    if coverage is not None:
        receipt["required_command_coverage"] = coverage
    if coverage == ["idle", "straight"]:
        motor.validate_training_receipt(receipt, lesson, "source", contract, "checkpoint")
    else:
        with pytest.raises(ValueError):
            motor.validate_training_receipt(receipt, lesson, "source", contract, "checkpoint")


@pytest.mark.parametrize("mode", tuple(lesson.MODES))
def test_inherited_motor_callback_pair_schedule_unchanged(mode):
    observer = lesson.MotorObserver(mode)
    term = NS(weight=0.)
    env = NS(common_step_counter=0, reward_manager=NS(get_term_cfg=lambda _: term))
    for clock in (0, 24, 6000, 12000, 18000, 23999):
        env.common_step_counter = clock
        motor.MotorObserver.before(observer, env)
        assert observer.mode == "control" and observer.schedule == ((0, -1.),)
        assert observer.weight == term.weight == -1.


def test_pure_straight_observer_full_inherited_callbacks(monkeypatch):
    from test_integrated_yaw_bias_refinement import fake_env
    from test_integrated_decoupled_yaw_precision_refinement import (
        fake_optimizer, set_turn_state, capture_other_pre_reset_terms)
    from mjlab_microduck import integrated_yaw_precision_refinement as angular
    from mjlab_microduck import integrated_yaw_bias_refinement as bias

    env = fake_env("control")
    runner, _, _ = fake_optimizer(monkeypatch, env)
    command = env.command_manager.get_command("twist")
    cfg, _ = lesson.prepare_config("straight")
    original_get_term = getattr(env.command_manager, "get_term", None)
    env.command_manager.get_term = lambda name: NS(cfg=cfg.commands["twist"]) if name == "twist" else original_get_term(name)
    observer = lesson.MotorObserver("straight")
    term = env.reward_manager.get_term_cfg(bias.previous.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    q = env.scene["robot"].data.joint_pos.clone()
    for _ in range(24):
        env.scene["robot"].data.joint_pos.copy_(q)
        set_turn_state(env)
        env.command_manager.get_command = lambda _: command
        command[:, 0] = .3
        command[:, 2] = 0.
        command[0] = 0.
        state = torch.random.get_rng_state().clone()
        before = command.clone()
        observer.before(env)
        assert torch.equal(command, before) and torch.equal(state, torch.random.get_rng_state())
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, term)
        angular.capture_angular_metric(env)
        bias.capture_bias_metric(env)
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        observer.step(env, sample, {})
    for _ in range(20): runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["optimizer_minibatches"] == 20
    assert row["yaw_bias_weight"] == 0.
    assert row["command_exposure_counts"]["idle"] == 24
    assert row["command_exposure_counts"]["straight"] == 72
    assert sum(row["command_exposure_counts"].values()) == 96


@pytest.mark.parametrize("yaw,speed", [(.001, .3), (-.02, .3), (0., .2), (0., .4)])
def test_wrong_consumed_commands_fail_before_inherited_callback(yaw, speed):
    cfg, _ = lesson.prepare_config("straight")
    command = torch.tensor([[speed, 0., yaw]])
    env = NS(num_envs=1, command_manager=NS(get_command=lambda _: command,
        get_term=lambda _: NS(cfg=cfg.commands["twist"])))
    with pytest.raises(ValueError): lesson.MotorObserver("straight").before(env)


def test_evaluator_adds_only_read_only_straight_gait_trace(monkeypatch):
    calls = []
    monkeypatch.setattr(lesson.base, "evaluate_case", lambda *args, **kwargs: calls.append((args, kwargs)))
    for speed, yaw in lesson.base.CASES:
        lesson.evaluate_case("checkpoint", 839, speed, yaw, None)
        expected = speed == .3 and yaw == 0.
        assert calls[-1][1] == dict(diagnostics=True, yaw_trace=expected, gait_trace=expected)


@pytest.mark.parametrize("call", [lesson.prepare_config, lesson.MotorObserver])
def test_unknown_mode_closed(call):
    with pytest.raises(ValueError): call("unknown")
