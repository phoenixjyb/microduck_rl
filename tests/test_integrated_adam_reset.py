from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_adam_reset as reset
from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as angular
from mjlab_microduck import integrated_straight_lesson as straight
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_yaw_bias_refinement as bias


def _models_and_optimizer(*, amsgrad=False):
    actor = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.BatchNorm1d(5),
                                torch.nn.Tanh(), torch.nn.Linear(5, 2))
    critic = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.BatchNorm1d(5),
                                 torch.nn.Tanh(), torch.nn.Linear(5, 1))
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=reset.PARENT_LEARNING_RATE,
                                 amsgrad=amsgrad)
    return actor, critic, optimizer


def _populate(actor, critic, optimizer):
    torch.manual_seed(41)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(8, 3)
        (actor(x).square().mean() + critic(x).square().mean()).backward()
        optimizer.step()


def _runner(actor, critic, optimizer):
    return NS(alg=NS(actor=actor, critic=critic, optimizer=optimizer,
                     learning_rate=reset.PARENT_LEARNING_RATE))


def _warm_load_stub(monkeypatch):
    calls = []

    def warm(runner, env):
        calls.append((runner, env))
        return dict(parent_sha256=reset.PARENT_SHA,
                    restored_models_normalizers_adam=True,
                    learning_rate=reset.PARENT_LEARNING_RATE)

    monkeypatch.setattr(reset.previous, "warm_load", warm)
    return calls


def _optimizer_copy(optimizer):
    return {
        parameter: {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in slots.items()}
        for parameter, slots in optimizer.state.items()
    }


def _assert_tree_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            _assert_tree_equal(left[key], right[key])
        elif isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]) or torch.allclose(
                left[key], right[key], equal_nan=True)
        else:
            assert left[key] == right[key]


def test_protocol_modes_parent_caps_evaluator_and_exactly_matched_configs():
    assert reset.PROTOCOL == "integrated-adam-reset-continuation-v1"
    assert reset.ARMS == ("control", "reset")
    assert reset.MODES == {
        "smoke-control": (64, 5, 1301), "smoke-reset": (64, 5, 1301),
        "benchmark-control": (256, 10, 1303), "benchmark-reset": (256, 10, 1303),
        "control": (256, 1000, 1307), "reset": (256, 1000, 1307),
    }
    assert reset.CAP_SECONDS == 2100
    assert reset.TRAINING_CAP_SECONDS == dict(control=2100, reset=2100)
    assert reset.PARENT == previous.PARENT and reset.PARENT_SHA == previous.PARENT_SHA
    assert reset.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE == 5.062500000000001e-05
    assert reset.warm_load is previous.warm_load
    assert reset.evaluate_case is straight.evaluate_case
    assert reset.decision is previous.decision

    for mode, (worlds, updates, seed) in reset.MODES.items():
        cfg, agent = reset.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates

    for prefix in ("smoke-", "benchmark-", ""):
        control_name = prefix + "control"
        reset_name = prefix + "reset"
        control_cfg, control_agent = reset.prepare_config(control_name)
        reset_cfg, reset_agent = reset.prepare_config(reset_name)
        assert repr(control_cfg) == repr(reset_cfg)
        assert control_agent.run_name == control_name and reset_agent.run_name == reset_name
        reset_agent.run_name = control_agent.run_name
        assert reset_agent == control_agent

    cfg, agent = reset.prepare_config("control")
    source_cfg, source_agent = previous.prepare_config("control")
    assert cfg.rewards[angular.TERM].params["roll_pitch_variance"] == .5
    source_cfg.seed = cfg.seed
    assert repr(cfg) == repr(source_cfg)
    source_agent.seed = agent.seed
    source_agent.max_iterations = agent.max_iterations
    source_agent.experiment_name = agent.experiment_name
    source_agent.run_name = agent.run_name
    assert agent == source_agent


@pytest.mark.parametrize("arm", ["control", "reset"])
def test_parent_restore_precedes_mode_and_preserves_models_groups_lr_and_rng(monkeypatch, arm):
    actor, critic, optimizer = _models_and_optimizer(amsgrad=True)
    _populate(actor, critic, optimizer)
    runner, env = _runner(actor, critic, optimizer), object()
    warm_calls = _warm_load_stub(monkeypatch)
    actor_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    critic_before = {key: value.detach().clone() for key, value in critic.state_dict().items()}
    adam_before = _optimizer_copy(optimizer)
    groups_object = optimizer.param_groups
    group_objects = list(optimizer.param_groups)
    group_options = [{key: value for key, value in group.items() if key != "params"}
                     for group in optimizer.param_groups]
    parameter_ids = [[id(parameter) for parameter in group["params"]]
                     for group in optimizer.param_groups]
    defaults = dict(optimizer.defaults)
    state_mapping = optimizer.state
    lr = runner.alg.learning_rate
    cpu_rng = torch.random.get_rng_state().clone()

    receipt = reset.warm_load_for_mode(runner, env, arm)

    assert warm_calls == [(runner, env)]
    assert receipt["parent_sha256"] == reset.PARENT_SHA
    assert receipt["parent_adam_restoration_verified"] is True
    assert receipt["restored_models_normalizers_adam"] is (arm == "control")
    assert receipt["optimizer_initialization_kind"] == (
        "restored_parent_adam_state" if arm == "control"
        else "restored_parent_adam_then_state_reset")
    assert receipt["adam_state_parameter_entries_before"] == len(adam_before)
    assert receipt["adam_state_parameter_entries_after"] == (len(adam_before) if arm == "control" else 0)
    expected_slot_counts = {"step": len(adam_before), "exp_avg": len(adam_before),
                            "exp_avg_sq": len(adam_before),
                            "max_exp_avg_sq": len(adam_before)}
    assert receipt["adam_slot_counts_before"] == expected_slot_counts
    assert receipt["adam_slot_counts_after"] == (expected_slot_counts if arm == "control" else {})
    assert receipt["adam_groups_preserved"] and receipt["adam_parameter_identities_preserved"]
    assert receipt["actor_critic_models_and_normalizers_preserved"]
    assert receipt["torch_rng_preserved"] and receipt["original_learning_rate"] == reset.PARENT_LEARNING_RATE
    assert runner.alg.learning_rate == lr == reset.PARENT_LEARNING_RATE
    assert optimizer.param_groups is groups_object
    assert all(left is right for left, right in zip(optimizer.param_groups, group_objects, strict=True))
    assert [[id(parameter) for parameter in group["params"]]
            for group in optimizer.param_groups] == parameter_ids
    assert [{key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups] == group_options
    assert optimizer.defaults == defaults
    assert optimizer.state is state_mapping
    assert all(torch.equal(actor_before[key], value) for key, value in actor.state_dict().items())
    assert all(torch.equal(critic_before[key], value) for key, value in critic.state_dict().items())
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if arm == "control":
        _assert_tree_equal(adam_before, optimizer.state)
    else:
        assert len(optimizer.state) == 0


@pytest.mark.parametrize("fault", ["parent-hash", "parent-lr", "group-lr", "missing-model-parameter"])
def test_mode_initializer_rejects_wrong_parent_or_optimizer_contract_before_reset(monkeypatch, fault):
    actor, critic, optimizer = _models_and_optimizer()
    _populate(actor, critic, optimizer)
    if fault == "group-lr":
        optimizer.param_groups[0]["lr"] *= 2
    elif fault == "missing-model-parameter":
        optimizer.param_groups[0]["params"].pop()
    runner = _runner(actor, critic, optimizer)
    before = _optimizer_copy(optimizer)

    def warm(*args):
        return dict(parent_sha256=("0" * 64 if fault == "parent-hash" else reset.PARENT_SHA),
                    restored_models_normalizers_adam=True,
                    learning_rate=(0.1 if fault == "parent-lr"
                                   else reset.PARENT_LEARNING_RATE))

    monkeypatch.setattr(reset.previous, "warm_load", warm)
    with pytest.raises(ValueError):
        reset.warm_load_for_mode(runner, object(), "reset")
    _assert_tree_equal(before, optimizer.state)


def test_reset_then_first_real_adam_update_starts_every_parameter_at_step_one(monkeypatch):
    actor, critic, optimizer = _models_and_optimizer()
    _populate(actor, critic, optimizer)
    runner = _runner(actor, critic, optimizer)
    _warm_load_stub(monkeypatch)
    reset.warm_load_for_mode(runner, object(), "reset")
    assert not optimizer.state

    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(8, 3)
    (actor(x).square().mean() + critic(x).square().mean()).backward()
    optimizer.step()
    assert len(optimizer.state) == len(optimizer.param_groups[0]["params"])
    assert all(slots["step"].item() == 1 for slots in optimizer.state.values())


def test_reset_helper_validates_and_clears_only_populated_adam_state_in_place():
    actor, critic, optimizer = _models_and_optimizer(amsgrad=True)
    _populate(actor, critic, optimizer)
    assert reset.validate_populated_adam(optimizer) == len(optimizer.param_groups[0]["params"])
    state_mapping = optimizer.state
    result = reset.reset_adam_state(optimizer)
    assert result == dict(before=len(optimizer.param_groups[0]["params"]), after=0)
    assert optimizer.state is state_mapping and not optimizer.state


@pytest.mark.parametrize("fault", ["wrong-optimizer", "empty", "partial", "bad-step",
                                   "wrong-step-dtype", "fractional-step", "wrong-shape",
                                   "wrong-dtype", "nan-moment", "negative-square-moment", "unexpected-slot",
                                   "duplicate-parameter"])
def test_reset_helper_fails_closed_on_malformed_optimizer_or_slots(fault):
    actor, critic, optimizer = _models_and_optimizer()
    if fault != "empty":
        _populate(actor, critic, optimizer)
    if fault == "wrong-optimizer":
        optimizer = torch.optim.SGD(list(actor.parameters()), lr=.1)
    elif fault == "partial":
        optimizer.state.pop(next(iter(optimizer.state)))
    elif fault == "bad-step":
        next(iter(optimizer.state.values()))["step"].fill_(0)
    elif fault == "wrong-step-dtype":
        slots = next(iter(optimizer.state.values()))
        slots["step"] = slots["step"].double()
    elif fault == "fractional-step":
        next(iter(optimizer.state.values()))["step"].fill_(1.5)
    elif fault == "wrong-shape":
        slots = next(iter(optimizer.state.values()))
        slots["exp_avg"] = torch.zeros(1)
    elif fault == "wrong-dtype":
        slots = next(iter(optimizer.state.values()))
        slots["exp_avg"] = slots["exp_avg"].double()
    elif fault == "nan-moment":
        next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
    elif fault == "negative-square-moment":
        next(iter(optimizer.state.values()))["exp_avg_sq"].fill_(-1.)
    elif fault == "unexpected-slot":
        next(iter(optimizer.state.values()))["max_exp_avg_sq"] = torch.zeros_like(
            next(iter(optimizer.state.values()))["exp_avg_sq"])
    elif fault == "duplicate-parameter":
        optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])

    state_before = _optimizer_copy(optimizer) if type(optimizer) is torch.optim.Adam else {}
    with pytest.raises((ValueError, TypeError)):
        reset.reset_adam_state(optimizer)
    if type(optimizer) is torch.optim.Adam:
        _assert_tree_equal(state_before, optimizer.state)


@pytest.mark.parametrize("mode", ["control", "reset"])
def test_observer_keeps_inherited_reward_kl_and_command_census_callbacks(monkeypatch, mode):
    from test_integrated_decoupled_yaw_precision_refinement import (
        capture_other_pre_reset_terms, fake_optimizer, set_turn_state,
    )
    from test_integrated_yaw_bias_refinement import fake_env
    from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as precision
    from mjlab_microduck import integrated_yaw_precision_refinement as angular

    env = fake_env("control")
    runner, value, calls = fake_optimizer(monkeypatch, env)
    observer = reset.MotorObserver(mode)
    assert observer.reset_mode == mode
    term = env.reward_manager.get_term_cfg(precision.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    initial_position = env.scene["robot"].data.joint_pos.clone()
    for _ in range(24):
        env.scene["robot"].data.joint_pos.copy_(initial_position)
        set_turn_state(env)
        observer.before(env)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, term)
        angular.capture_angular_metric(env)
        bias.capture_bias_metric(env)
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        observer.step(env, sample, {})
    for _ in range(20):
        assert runner.alg.actor.get_kl_divergence("old", new="new") is value
    assert calls == [(('old',), {"new": "new"})] * 20

    row = observer.update(env)
    assert row["optimizer_minibatches"] == 20
    assert row["desired_kl"] == .01
    assert row["yaw_bias_weight"] == row["weighted_yaw_bias_reward_mean"] == 0.
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["roll_pitch_variance"] == .5
    assert row["command_exposure_counts"]["positive-target"] == 4 * 24
    assert sum(row["command_exposure_counts"].values()) == 4 * 24
    assert row["command_exposure_cumulative"] == row["command_exposure_counts"]


@pytest.mark.parametrize("mode", ["bad", "smoke-bad", "benchmark-bad"])
def test_unknown_mode_fails_closed(mode):
    with pytest.raises(ValueError):
        reset.prepare_config(mode)
    with pytest.raises(ValueError):
        reset.MotorObserver(mode)
    with pytest.raises(ValueError):
        reset.warm_load_for_mode(NS(), NS(), mode)
