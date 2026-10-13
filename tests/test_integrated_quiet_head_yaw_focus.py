import copy
import math

import pytest
import torch

from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity
from mjlab_microduck import integrated_motor_range_head_lesson as head
from mjlab_microduck import integrated_motor_range_yaw_only as yaw_only
from mjlab_microduck import integrated_quiet_head_yaw_focus as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp


def _paired_configs(control_mode, treatment_mode):
    control, control_agent = recipe.prepare_config(control_mode)
    treatment, treatment_agent = recipe.prepare_config(treatment_mode)
    normalized_control, normalized_treatment = copy.deepcopy(control), copy.deepcopy(treatment)
    normalized_control_agent = copy.deepcopy(control_agent)
    normalized_treatment_agent = copy.deepcopy(treatment_agent)
    cterm = normalized_control.rewards[recipe.TERM]
    tterm = normalized_treatment.rewards[recipe.TERM]

    assert cterm.func is track_angular_velocity
    assert tterm.func is mdp.factorized_body_angular_tracking
    assert cterm.weight == tterm.weight == recipe.WEIGHT == 2.
    assert cterm.params == dict(std=math.sqrt(.5), command_name="twist")
    assert tterm.params == dict(command_name="twist", yaw_variance=.1, roll_pitch_variance=.5)

    # Normalize both representations to the exact same stock factorized
    # contract; the only remaining treatment change is yaw variance .5 -> .1.
    cterm.func = tterm.func = mdp.factorized_body_angular_tracking
    cterm.params = tterm.params = dict(command_name="twist", yaw_variance=.5,
                                       roll_pitch_variance=.5)
    normalized_treatment_agent.run_name = normalized_control_agent.run_name
    assert repr(normalized_control) == repr(normalized_treatment)
    assert normalized_control_agent == normalized_treatment_agent
    return control, treatment


def test_modes_parent_protocol_and_matched_yaw_variance_only():
    assert recipe.PROTOCOL == "integrated-quiet-head-yaw-focus-v1"
    assert recipe.ARMS == ("control", "yawfocus")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1637), "smoke-yawfocus": (64, 5, 1637),
        "benchmark-control": (256, 10, 1657), "benchmark-yawfocus": (256, 10, 1657),
        "control": (256, 1000, 1663), "yawfocus": (256, 1000, 1663),
    }
    assert recipe.CAP_SECONDS == 2100
    assert recipe.PARENT == head.PARENT == yaw_only.PARENT
    assert recipe.PARENT.parent.name == "control" and recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == head.PARENT_SHA == yaw_only.PARENT_SHA == (
        "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    )
    assert recipe.PARENT_NORMALIZER_COUNT == 43_008_000
    assert recipe.PARENT_ADAM_STEP == 140_000
    assert recipe.warm_load is head.warm_load
    assert recipe.decision is head.decision
    assert recipe.evaluate_case is not head.evaluate_case
    assert recipe.evaluate_case.__module__ == recipe.__name__

    for control_mode, treatment_mode in (
        ("smoke-control", "smoke-yawfocus"),
        ("benchmark-control", "benchmark-yawfocus"),
        ("control", "yawfocus"),
    ):
        control, treatment = _paired_configs(control_mode, treatment_mode)
        reference_mode = recipe.reference_mode(control_mode)
        parent_cfg, parent_agent = head.prepare_config(reference_mode)
        for cfg in (control, treatment):
            assert cfg.commands == parent_cfg.commands
            assert cfg.observations == parent_cfg.observations
            assert cfg.actions == parent_cfg.actions
            assert cfg.events == parent_cfg.events
            assert cfg.curriculum == parent_cfg.curriculum == {}
            for name in cfg.rewards:
                if name != recipe.TERM:
                    assert cfg.rewards[name] == parent_cfg.rewards[name]
            assert cfg.commands[head.HEAD_TERM].ranges == head.BEGINNER_RANGES
        assert control.rewards[recipe.TERM].func is track_angular_velocity
        assert treatment.rewards[recipe.TERM].func is mdp.factorized_body_angular_tracking
        assert parent_agent.actor == recipe.prepare_config(control_mode)[1].actor

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent = recipe.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert agent.experiment_name == recipe.PROTOCOL and agent.run_name == mode
        assert recipe.yaw_variance(mode) == (.1 if mode.endswith("yawfocus") else .5)
        assert recipe.head_ranges(mode) == head.BEGINNER_RANGES
        assert cfg.observations["actor"].terms["head_command"].params["command_name"] == "head_pose"
        assert cfg.rewards["body_ang_vel"].weight == -.05
        assert cfg.rewards["angular_momentum"].weight == -.02
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["joint_limit"].params["margin"] == .15
        assert cfg.rewards["forward_error"].weight == 0.


def test_stock_parity_yaw_precision_and_xy_factor_are_independent():
    rates = torch.tensor([
        [0., 0., .2], [1., 0., .2], [0., 0., -.1], [1., 0., -.1],
    ])
    command = torch.zeros(4, 3)
    command[:, 2] = torch.tensor([.2, .2, .2, .2])
    env = type("FakeEnv", (), {
        "scene": {"robot": type("Robot", (), {"data": type("Data", (), {
            "root_link_ang_vel_b": rates,
        })()})()},
        "command_manager": type("Commands", (), {"get_command": lambda _self, _name: command})(),
    })()

    stock = track_angular_velocity(env, math.sqrt(.5), "twist")
    control = mdp.factorized_body_angular_tracking(
        env, yaw_variance=.5, roll_pitch_variance=.5, command_name="twist")
    treatment = mdp.factorized_body_angular_tracking(
        env, yaw_variance=.1, roll_pitch_variance=.5, command_name="twist")
    yaw_sq, xy_sq = mdp.body_angular_tracking_errors(env, "twist")
    expected_control = torch.exp(-(yaw_sq + xy_sq) / .5)
    expected_treatment = torch.exp(-yaw_sq / .1 - xy_sq / .5)
    assert torch.allclose(control, stock, atol=1e-6, rtol=1e-5)
    assert torch.allclose(control, expected_control)
    assert torch.allclose(treatment, expected_treatment)
    assert treatment[0] != treatment[1]  # XY precision is retained at .5.
    assert treatment[2] < treatment[0]  # Same XY error, larger yaw error is penalized.
    assert bool(((treatment > 0.) & (treatment <= 2.)).all())


def test_parent_action_linear_and_head_row_validators_keep_quiet_head_reference_modes(monkeypatch):
    calls = {"action": [], "linear": [], "head": []}
    monkeypatch.setattr(head, "validate_action_row",
                        lambda *args: calls["action"].append(args))
    monkeypatch.setattr(head, "validate_linear_row",
                        lambda *args: calls["linear"].append(args))
    monkeypatch.setattr(head, "validate_head_row",
                        lambda *args: calls["head"].append(args))
    for mode, (worlds, updates, _seed) in recipe.MODES.items():
        expected = recipe.reference_mode(mode)
        for validator, name in ((recipe.validate_action_row, "action"),
                                (recipe.validate_linear_row, "linear"),
                                (recipe.validate_head_row, "head")):
            validator({"mode": mode}, mode, worlds, updates)
            assert calls[name][-1] == ({"mode": mode}, expected, worlds, updates)


def _observer_env(mode):
    from test_integrated_motor_range_head_lesson import _head_observer_env

    env = _head_observer_env(recipe.reference_mode(mode))
    cfg, _ = recipe.prepare_config(mode)
    term_cfg = cfg.rewards[recipe.TERM]
    terms = {name: env.reward_manager.get_term_cfg(name)
             for name in env.reward_manager.active_terms}
    terms[recipe.TERM].func = term_cfg.func
    terms[recipe.TERM].params = dict(term_cfg.params)
    terms[recipe.TERM].weight = term_cfg.weight
    env.reward_manager._step_reward = torch.zeros(
        env.num_envs, len(env.reward_manager.active_terms))
    env.scene["robot"].data.root_link_ang_vel_b = torch.tensor(
        [[.2, 0., .5]]).repeat(env.num_envs, 1)
    original_get_command = env.command_manager.get_command
    env.command_manager.get_command = lambda name: (
        torch.zeros(env.num_envs, 3) if name == "twist"
        else original_get_command(name)
    )
    env.reward_manager.get_term_cfg = terms.__getitem__
    env.cfg = cfg
    return env


def _prepare_yaw_head_step(env):
    from test_integrated_motor_range_action_release import _prepare_parent_metrics
    from mjlab_microduck import integrated_motor_range_action_release as action
    from mjlab_microduck import integrated_motor_range_linear_boost as linear

    action_term = env.reward_manager.get_term_cfg(action.ACTION_TERM)
    _prepare_parent_metrics(env, action_term, torch.full((env.num_envs, 14), .1))
    mdp.capture_action_rate_training_metric(env)
    linear_term = env.reward_manager.get_term_cfg(linear.LINEAR_TERM)
    twist = env.command_manager.get_command("twist")
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = twist[:, :2] - torch.tensor([.04, .02])
    raw_linear = linear_term.func(env, **linear_term.params)
    env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(linear.LINEAR_TERM)] = (
        raw_linear * linear_term.weight)
    mdp.capture_linear_tracking_training_metric(env)

    tracking = torch.full((env.num_envs,), .8)
    bias = torch.full((env.num_envs,), -.15)
    reward = env.reward_manager._step_reward
    reward[:, env.reward_manager.active_terms.index("head_pose_tracking")] = tracking
    reward[:, env.reward_manager.active_terms.index("head_pose_bias")] = bias
    term = env.reward_manager.get_term_cfg(recipe.TERM)
    reward[:, env.reward_manager.active_terms.index(recipe.TERM)] = term.weight * term.func(
        env, **term.params)
    return tracking, bias


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-yawfocus"])
def test_head_yaw_observer_snapshots_actual_pre_reset_product_for_24_steps(monkeypatch, mode):
    from test_integrated_motor_range_head_lesson import _head_values

    env = _observer_env(mode)
    observer = recipe.MotorObserver(mode)
    sample = type("Sample", (), {
        "force_nm": torch.zeros(env.num_envs, 14), "mean_cost": torch.zeros(env.num_envs),
    })()
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))
    yaw_means, xy_means, reward_means = [], [], []

    for step in range(24):
        env._head_command_holder["head"] = _head_values(step, env.num_envs)
        observer.before(env)
        _prepare_yaw_head_step(env)
        angular_snapshot = env._microduck_yaw_aggregation_snapshot
        assert angular_snapshot[3] == env.common_step_counter
        yaw, xy, actual, _ = angular_snapshot
        rates = env.scene["robot"].data.root_link_ang_vel_b
        command = env.command_manager.get_command("twist")
        expected_yaw = (command[:, 2] - rates[:, 2]).square()
        expected_xy = (command[:, :2] - rates[:, :2]).square().sum(-1)
        assert torch.allclose(yaw, expected_yaw)
        assert torch.allclose(xy, expected_xy)
        if mode.endswith("yawfocus"):
            expected = 2. * torch.exp(-yaw / .1 - xy / .5)
        else:
            expected = 2. * torch.exp(-(yaw + xy) / .5)
        assert actual.requires_grad is False
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        assert bool(((actual > 0.) & (actual <= 2.)).all())
        yaw_means.append(float(yaw.mean()))
        xy_means.append(float(xy.mean()))
        reward_means.append(float(actual.mean()))

        env._head_command_holder["head"] = torch.full((env.num_envs, 4), 9.)
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        observer.step(env, sample, {})
        assert env._microduck_yaw_aggregation_snapshot is None

    row = observer.update(env)
    assert row["yaw_variance"] == recipe.yaw_variance(mode)
    assert row["roll_pitch_variance"] == .5
    assert row["roll_pitch_tracking_included"] is True
    assert row["angular_tracking_weight"] == 2.
    assert row["angular_aggregation"] == (
        "axis_precision_multiplicative_product" if mode.endswith("yawfocus")
        else "stock_multiplicative_product")
    assert row["yaw_error_sq_mean"] == pytest.approx(sum(yaw_means) / 24)
    assert row["roll_pitch_rate_sq_mean"] == pytest.approx(sum(xy_means) / 24)
    assert row["weighted_angular_reward_mean"] == pytest.approx(sum(reward_means) / 24)
    assert row["angular_reward_sample_count"] == env.num_envs * 24
    assert row["negative_reward_sample_count"] == row["negative_reward_fraction"] == 0
    assert row["head_command_steps"] == 24 and row["head_reward_sample_count"] == env.num_envs * 24


@pytest.mark.parametrize("fault", ["func", "params", "weight"])
def test_head_yaw_observer_rejects_live_angular_contract_drift(fault):
    env = _observer_env("smoke-yawfocus")
    observer = recipe.MotorObserver("smoke-yawfocus")
    term = env.reward_manager.get_term_cfg(recipe.TERM)
    if fault == "func":
        term.func = track_angular_velocity
    elif fault == "params":
        term.params["yaw_variance"] = .5
    else:
        term.weight = 1.
    with pytest.raises(ValueError, match="fixed live angular aggregation"):
        observer.before(env)


def test_head_yaw_observer_rejects_reward_snapshot_not_matching_actual_product(monkeypatch):
    from test_integrated_motor_range_head_lesson import _head_values

    mode = "smoke-yawfocus"
    env = _observer_env(mode)
    observer = recipe.MotorObserver(mode)
    sample = type("Sample", (), {
        "force_nm": torch.zeros(env.num_envs, 14), "mean_cost": torch.zeros(env.num_envs),
    })()
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))
    env._head_command_holder["head"] = _head_values(0, env.num_envs)
    observer.before(env)
    _prepare_yaw_head_step(env)
    snapshot = env._microduck_yaw_aggregation_snapshot
    env._microduck_yaw_aggregation_snapshot = (
        snapshot[0], snapshot[1], snapshot[2] + .2, snapshot[3])
    with pytest.raises(ValueError, match="actual signed angular reward matches live aggregation"):
        observer.step(env, sample, {})


def _valid_yaw_row(mode):
    from test_integrated_motor_range_head_lesson import _valid_head_row

    row = _valid_head_row(recipe.reference_mode(mode))
    variance = recipe.yaw_variance(mode)
    row.update(
        yaw_variance=variance,
        roll_pitch_variance=.5,
        angular_tracking_weight=2.,
        angular_aggregation=("axis_precision_multiplicative_product" if variance == .1
                             else "stock_multiplicative_product"),
        roll_pitch_tracking_included=True,
        angular_reward_sample_count=recipe.MODES[mode][0] * 24,
        negative_reward_sample_count=0,
        negative_reward_fraction=0,
        weighted_angular_reward_mean=1.2,
        yaw_error_sq_mean=.1,
        roll_pitch_rate_sq_mean=.2,
        body_ang_vel_weight=-.05,
        angular_momentum_weight=-.02,
    )
    return row


@pytest.mark.parametrize("mode", list(recipe.MODES))
def test_validate_yaw_row_accepts_fixed_positive_product_contract(mode):
    worlds, updates, _ = recipe.MODES[mode]
    recipe.validate_yaw_row(_valid_yaw_row(mode), mode, worlds, updates)


@pytest.mark.parametrize("fault", ["variance", "roll-variance", "weight", "aggregation",
                                    "included", "count", "negative", "reward-high",
                                    "yaw-nan", "xy-negative", "stability"])
def test_validate_yaw_row_rejects_changed_or_invalid_angular_evidence(fault):
    row = _valid_yaw_row("yawfocus")
    if fault == "variance": row["yaw_variance"] = .5
    elif fault == "roll-variance": row["roll_pitch_variance"] = .1
    elif fault == "weight": row["angular_tracking_weight"] = 1.
    elif fault == "aggregation": row["angular_aggregation"] = "additive_decoupled"
    elif fault == "included": row["roll_pitch_tracking_included"] = False
    elif fault == "count": row["angular_reward_sample_count"] -= 1
    elif fault == "negative": row["negative_reward_sample_count"] = 1
    elif fault == "reward-high": row["weighted_angular_reward_mean"] = 2.1
    elif fault == "yaw-nan": row["yaw_error_sq_mean"] = float("nan")
    elif fault == "xy-negative": row["roll_pitch_rate_sq_mean"] = -.1
    else: row["body_ang_vel_weight"] = -.1
    with pytest.raises(ValueError, match="actual fixed positive yaw-axis precision contract"):
        recipe.validate_yaw_row(row, "yawfocus", 256, 1)


def test_yaw_row_rejects_wrong_mode_budget_and_update():
    row = _valid_yaw_row("yawfocus")
    with pytest.raises(ValueError, match="declared yaw-focus row budget"):
        recipe.validate_yaw_row(row, "yawfocus", 64, 1)
    with pytest.raises(ValueError, match="declared yaw-focus row budget"):
        recipe.validate_yaw_row(row, "yawfocus", 256, 1001)
    with pytest.raises((ValueError, KeyError)):
        recipe.yaw_variance("unlisted")


def test_evaluator_forwards_read_only_trace_flags_to_unchanged_gate(monkeypatch):
    calls = []
    result = {"steps": 240, "score": .7}
    monkeypatch.setattr(recipe.base, "evaluate_case",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    got = recipe.evaluate_case("checkpoint", 839, .3, -.1, "health")
    assert got is result
    assert calls == [(("checkpoint", 839, .3, -.1, "health"),
                      dict(diagnostics=True, yaw_trace=True, gait_trace=True))]
    assert recipe.decision is head.decision
    assert recipe.base.SEEDS == head.base.SEEDS and recipe.base.CASES == head.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "yawfocus": 999,
    }
