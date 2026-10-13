from types import SimpleNamespace as NS
import math

import pytest
import torch

from mjlab_microduck import integrated_motor_range_yaw as angular
from mjlab_microduck import integrated_motor_range_yaw_only as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity


def _angular_env(angular_rates, commands=None):
    if commands is None:
        commands = torch.zeros_like(angular_rates)
    return NS(
        scene={"robot": NS(data=NS(root_link_ang_vel_b=angular_rates))},
        command_manager=NS(get_command=lambda _name: commands),
    )


def test_yaw_only_kernel_matches_positive_gaussian_and_ignores_xy_rates():
    command = torch.tensor([
        [0., 0., .2], [0., 0., .2], [0., 0., -.4], [0., 0., .7],
    ])
    angular_rates = torch.tensor([
        [0., 0., .2], [12., -8., .2], [-3., 7., -.1], [.5, -.4, -.3],
    ])
    env = _angular_env(angular_rates, command)

    actual = mdp.yaw_only_body_angular_tracking(env, math.sqrt(.5), "twist")
    yaw_sq = (command[:, 2] - angular_rates[:, 2]).square()
    expected = torch.exp(-yaw_sq / .5)
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual[:2], torch.ones(2))
    assert actual[0] == actual[1]  # changing roll/pitch rates cannot change this kernel
    assert bool((actual > 0.).all() & (actual <= 1.).all())

    stock = track_angular_velocity(env, math.sqrt(.5), "twist")
    assert actual[0] == stock[0]  # at zero XY error, the legacy control is unchanged
    assert actual[1] != stock[1]  # nonzero XY error is omitted only from this reward


@pytest.mark.parametrize("std", [True, False, None, "0.5", 0., -1., float("nan"), float("inf"),
                                 -float("inf"), 1e-300, 1e300])
def test_yaw_only_kernel_rejects_invalid_std(std):
    env = _angular_env(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="yaw tracking std must be finite and positive"):
        mdp.yaw_only_body_angular_tracking(env, std)


@pytest.mark.parametrize("fault", ["nan-xy", "nan-yaw", "nan-command", "shape", "yaw-overflow", "xy-overflow"])
def test_yaw_only_still_validates_all_raw_angular_state(fault):
    rates = torch.zeros(2, 3)
    command = torch.zeros(2, 3)
    if fault == "nan-xy":
        rates[0, 0] = float("nan")
    elif fault == "nan-yaw":
        rates[0, 2] = float("nan")
    elif fault == "nan-command":
        command[0, 1] = float("nan")
    elif fault == "shape":
        command = torch.zeros(2, 2)
    elif fault == "yaw-overflow":
        rates[0, 2] = 1e30
    else:
        rates[0, 0] = 1e30
    with pytest.raises(FloatingPointError):
        mdp.yaw_only_body_angular_tracking(_angular_env(rates, command), math.sqrt(.5))


def test_yaw_only_rejects_nonfinite_scaled_exponent():
    env = _angular_env(torch.tensor([[0., 0., .5]]))
    with pytest.raises(FloatingPointError, match="yaw-only angular exponent"):
        mdp.yaw_only_body_angular_tracking(env, 1e-20)


def _paired_configs(control_mode, treatment_mode):
    control, control_agent = recipe.prepare_config(control_mode)
    treatment, treatment_agent = recipe.prepare_config(treatment_mode)
    cterm = control.rewards[recipe.TERM]
    tterm = treatment.rewards[recipe.TERM]

    assert cterm.func is track_angular_velocity
    assert tterm.func is mdp.yaw_only_body_angular_tracking
    assert cterm.weight == tterm.weight == recipe.WEIGHT == 2.
    assert cterm.params == tterm.params == dict(std=math.sqrt(.5), command_name="twist")

    # The reward callable is the only semantic treatment difference; its
    # argument contract and every other environment/learner setting match.
    tterm.func = cterm.func
    treatment_agent.run_name = control_agent.run_name
    assert repr(control) == repr(treatment)
    assert control_agent == treatment_agent
    return control, treatment


def test_modes_parent_and_matched_config_change_only_tracking_callable():
    assert recipe.PROTOCOL == "integrated-motor-range-yaw-only-v1"
    assert recipe.ARMS == ("control", "yawonly")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1433), "smoke-yawonly": (64, 5, 1433),
        "benchmark-control": (256, 10, 1439), "benchmark-yawonly": (256, 10, 1439),
        "control": (256, 1000, 1447), "yawonly": (256, 1000, 1447),
    }
    assert recipe.CAP_SECONDS == 2100
    assert recipe.PARENT == angular.PARENT
    assert recipe.PARENT.parent.name == "control" and recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    assert recipe.PARENT_NORMALIZER_COUNT == 43008000
    assert recipe.PARENT_ADAM_STEP == 140000
    assert recipe.warm_load is angular.warm_load
    assert recipe.evaluate_case is angular.evaluate_case
    assert recipe.decision is angular.decision

    for control_mode, treatment_mode in (
        ("smoke-control", "smoke-yawonly"),
        ("benchmark-control", "benchmark-yawonly"),
        ("control", "yawonly"),
    ):
        _paired_configs(control_mode, treatment_mode)

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent = recipe.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert recipe.yaw_only(mode) == (mode == "yawonly" or mode.endswith("-yawonly"))


def test_parent_stability_motor_range_observation_and_raw_action_contracts_remain_fixed():
    cfg, agent = recipe.prepare_config("benchmark-control")
    treatment, treatment_agent = recipe.prepare_config("benchmark-yawonly")
    parent, parent_agent = angular.prepare_config("control")

    # Both arms retain the parent's full 61D actor layout, including padded
    # command slots, and use the same unbounded position-action interface.
    assert cfg.observations == treatment.observations == parent.observations
    assert cfg.commands == treatment.commands == parent.commands
    assert cfg.events == treatment.events == parent.events
    actor_terms = cfg.observations["actor"].terms
    assert "head_command" in actor_terms and "body_command" in actor_terms
    assert "base_lin_vel" not in actor_terms and "height_scan" not in actor_terms
    assert actor_terms["head_command"].params["command_name"] == "head_pose"
    assert actor_terms["body_command"].params["command_name"] == "body_pose"
    assert tuple(actor_terms["joint_pos"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert tuple(actor_terms["joint_vel"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert cfg.commands["head_pose"].ranges and len(cfg.commands["head_pose"].ranges) == 4
    assert cfg.commands["body_pose"].ranges and len(cfg.commands["body_pose"].ranges) == 6
    # 48 proprioceptive/action values plus the 3+4+6 command block.
    assert 14 + 14 + 3 + 3 + 14 == 48
    assert 48 + 3 + len(cfg.commands["head_pose"].ranges) + len(cfg.commands["body_pose"].ranges) == 61
    assert cfg.actions == treatment.actions == parent.actions
    action = cfg.actions["joint_pos"]
    assert action.scale == 1. and action.clip is None
    assert tuple(action.actuator_names) == (".*",)
    assert agent.clip_actions is treatment_agent.clip_actions is parent_agent.clip_actions is None
    assert agent.actor == treatment_agent.actor == parent_agent.actor
    assert agent.actor.obs_normalization is True
    assert sum((14, 14, 3, 3, 14, 3, 4, 6)) == 61

    for candidate in (cfg, treatment):
        assert candidate.curriculum == {}
        assert candidate.rewards["track_angular_velocity"].weight == 2.
        assert candidate.rewards["body_ang_vel"].weight == -.05
        assert candidate.rewards["angular_momentum"].weight == -.02
        assert candidate.rewards["motor_load"].weight == -1.
        assert candidate.rewards["motor_load"].params == dict(
            rated_stall_torque_nm=.60, soft_limit_fraction=.70, over_limit_gain=4.,
        )
        assert candidate.rewards["joint_limit"].weight == -4.
        assert candidate.rewards["joint_limit"].params["margin"] == .15
        assert candidate.rewards["joint_limit"].params["asset_cfg"].joint_names == (
            "^(left|right)_hip_yaw$",
        )
        assert candidate.rewards["forward_error"].weight == 0.


def test_legacy_control_reward_and_final999_evaluation_are_preserved():
    cfg, agent = recipe.prepare_config("control")
    legacy, legacy_agent = angular.prepare_config("control")
    assert cfg.rewards[recipe.TERM].func is legacy.rewards[recipe.TERM].func is track_angular_velocity
    assert cfg.rewards[recipe.TERM].params == legacy.rewards[recipe.TERM].params
    assert cfg.rewards[recipe.TERM].weight == legacy.rewards[recipe.TERM].weight == 2.
    assert cfg.rewards["body_ang_vel"] == legacy.rewards["body_ang_vel"]
    assert cfg.rewards["angular_momentum"] == legacy.rewards["angular_momentum"]
    assert cfg.rewards["motor_load"] == legacy.rewards["motor_load"]
    assert cfg.rewards["joint_limit"] == legacy.rewards["joint_limit"]
    assert cfg.observations == legacy.observations and cfg.actions == legacy.actions
    assert cfg.scene.entities == legacy.scene.entities
    assert cfg.seed == agent.seed == 1447
    assert legacy.seed == legacy_agent.seed == 1429

    assert recipe.evaluate_case is angular.evaluate_case
    assert recipe.decision is angular.decision
    assert recipe.base.SEEDS == angular.base.SEEDS
    assert recipe.base.CASES == angular.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "yawonly": 999,
    }


def _observer_env(mode):
    from test_integrated_motor_range_forward import _fake_observer_env

    env = _fake_observer_env()
    cfg, _ = recipe.prepare_config(mode)
    term_cfg = cfg.rewards[recipe.TERM]
    original_get_term_cfg = env.reward_manager.get_term_cfg
    terms = {name: original_get_term_cfg(name) for name in env.reward_manager.active_terms}
    terms[recipe.TERM] = NS(func=term_cfg.func, weight=term_cfg.weight,
                            params=dict(term_cfg.params))
    terms["body_ang_vel"] = NS(weight=-.05)
    terms["angular_momentum"] = NS(weight=-.02)
    env.reward_manager.active_terms.extend((recipe.TERM, "body_ang_vel", "angular_momentum"))
    env.reward_manager.get_term_cfg = lambda name: (
        terms[name] if name in terms else original_get_term_cfg(name)
    )
    env.reward_manager._step_reward = torch.zeros(env.num_envs, len(env.reward_manager.active_terms))
    env.scene["robot"].data.root_link_ang_vel_b = torch.tensor(
        [[1., 0., .5]]
    ).repeat(env.num_envs, 1)
    env.command_manager.get_command = lambda _name: torch.zeros(env.num_envs, 3)
    return env


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-yawonly"])
def test_observer_validates_pre_reset_reward_and_reports_axis_inclusion(monkeypatch, mode):
    env = _observer_env(mode)
    observer = recipe.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.angular.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))
    rates = env.scene["robot"].data.root_link_ang_vel_b.clone()
    qpos = env.scene["robot"].data.joint_pos.clone()
    snapshots = []

    for _ in range(24):
        env.scene["robot"].data.root_link_ang_vel_b.copy_(rates)
        env.scene["robot"].data.joint_pos.copy_(qpos)
        observer.before(env)
        # The environment advances the step counter before its metric callback.
        env.common_step_counter += 1
        recipe.angular.previous.safety.capture_limit_metric(env)
        recipe.angular.previous.forward.capture_forward_metric(env)
        env.reward_manager._step_reward[:, 0] = 0.
        env.reward_manager._step_reward[:, 1] = env._microduck_fresh_limit_snapshot[0] * -4.
        env.reward_manager._step_reward[:, 2] = 0.
        term = env.reward_manager.get_term_cfg(recipe.TERM)
        index = env.reward_manager.active_terms.index(recipe.TERM)
        env.reward_manager._step_reward[:, index] = term.weight * term.func(env, **term.params)
        recipe.angular.capture_angular_metric(env)
        yaw_sq, xy_sq, actual, step = env._microduck_yaw_aggregation_snapshot
        assert step == env.common_step_counter
        assert torch.equal(yaw_sq, torch.full((env.num_envs,), .25))
        assert torch.equal(xy_sq, torch.ones(env.num_envs))
        assert actual.requires_grad is False
        expected = (2. * torch.exp(-yaw_sq / .5) if mode.endswith("yawonly")
                    else 2. * torch.exp(-(yaw_sq + xy_sq) / .5))
        assert bool(((actual >= 0.) & (actual <= 2.)).all())
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        snapshots.append(actual.clone())
        # Autoreset after the metric callback cannot change the saved values.
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        env.scene["robot"].data.joint_pos.zero_()
        observer.step(env, sample, {})
        assert env._microduck_yaw_aggregation_snapshot is None

    row = observer.update(env)
    assert row["angular_tracking_weight"] == 2.
    assert row["body_ang_vel_weight"] == -.05
    assert row["angular_momentum_weight"] == -.02
    assert row["roll_pitch_rate_sq_mean"] == pytest.approx(1.)
    assert row["angular_reward_sample_count"] == env.num_envs * 24
    assert all(torch.equal(value, snapshots[0]) for value in snapshots)
    assert row["angular_aggregation"] == (
        "stock_multiplicative_product" if mode.endswith("control") else "yaw_only_tracking"
    )
    assert row["roll_pitch_tracking_included"] is mode.endswith("control")
    assert row["roll_pitch_variance"] == (.5 if mode.endswith("control") else None)
    if mode.endswith("control"):
        assert bool((snapshots[0] > 0.).all())
        assert row["negative_reward_sample_count"] == 0
    else:
        assert bool((snapshots[0] > 0.).all())
        assert row["negative_reward_sample_count"] == 0
        assert row["negative_reward_fraction"] == 0.


@pytest.mark.parametrize("fault", ["tracking-func", "tracking-params", "tracking-weight",
                                    "body-ang-vel", "angular-momentum"])
def test_observer_rejects_tracking_or_separate_stability_drift(fault):
    env = _observer_env("smoke-yawonly")
    observer = recipe.MotorObserver("smoke-yawonly")
    if fault == "tracking-func":
        env.reward_manager.get_term_cfg(recipe.TERM).func = track_angular_velocity
    elif fault == "tracking-params":
        env.reward_manager.get_term_cfg(recipe.TERM).params["std"] = .25
    elif fault == "tracking-weight":
        env.reward_manager.get_term_cfg(recipe.TERM).weight = 1.
    elif fault == "body-ang-vel":
        env.reward_manager.get_term_cfg("body_ang_vel").weight = -.1
    else:
        env.reward_manager.get_term_cfg("angular_momentum").weight = -.04

    expected = ("fixed live angular aggregation" if fault.startswith("tracking-")
                else "fixed live separate angular stability weights")
    with pytest.raises(ValueError, match=expected):
        observer.before(env)


@pytest.mark.parametrize("fault", ["missing", "wrong-reward", "negative", "stale"])
def test_yaw_only_rejects_bad_pre_reset_snapshots(monkeypatch, fault):
    env = _observer_env("smoke-yawonly")
    observer = recipe.MotorObserver("smoke-yawonly")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))
    observer.before(env)
    env.common_step_counter += 1
    recipe.angular.previous.safety.capture_limit_metric(env)
    recipe.angular.previous.forward.capture_forward_metric(env)
    env.reward_manager._step_reward[:, 1] = env._microduck_fresh_limit_snapshot[0] * -4.
    term = env.reward_manager.get_term_cfg(recipe.TERM)
    index = env.reward_manager.active_terms.index(recipe.TERM)
    env.reward_manager._step_reward[:, index] = term.weight * term.func(env, **term.params)
    if fault != "missing":
        recipe.angular.capture_angular_metric(env)
    if fault == "wrong-reward":
        env._microduck_yaw_aggregation_snapshot[2].zero_()
    elif fault == "negative":
        env._microduck_yaw_aggregation_snapshot[2].fill_(-.5)
    elif fault == "stale":
        snapshot = env._microduck_yaw_aggregation_snapshot
        env._microduck_yaw_aggregation_snapshot = (*snapshot[:3], snapshot[3]-1)
    with pytest.raises(ValueError, match="same-step pre-reset angular snapshot" if fault in (
            "missing", "stale") else "actual signed angular reward matches live aggregation"):
        observer.step(env, sample, {})

