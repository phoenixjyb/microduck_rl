import math
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_motor_range_action_release as action
from mjlab_microduck import integrated_motor_range_linear_boost as recipe
from mjlab_microduck import integrated_motor_range_yaw_only as parent
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity, track_linear_velocity


def _paired_configs(control_mode, boost_mode):
    control, control_agent = recipe.prepare_config(control_mode)
    boost, boost_agent = recipe.prepare_config(boost_mode)
    control_term = control.rewards[recipe.LINEAR_TERM]
    boost_term = boost.rewards[recipe.LINEAR_TERM]
    assert control_term.func is boost_term.func is track_linear_velocity
    assert control_term.params == boost_term.params == dict(std=math.sqrt(.1), command_name="twist")
    assert control_term.weight == 2.
    assert boost_term.weight == 2.5
    boost_term.weight = control_term.weight
    boost_agent.run_name = control_agent.run_name
    assert repr(control) == repr(boost)
    assert control_agent == boost_agent
    return control, boost, control_agent, boost_agent


def test_modes_parent_and_single_linear_weight_factor_are_fixed():
    assert recipe.PROTOCOL == "integrated-motor-range-linear-boost-v1"
    assert recipe.ARMS == ("control", "linearboost")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1523), "smoke-linearboost": (64, 5, 1523),
        "benchmark-control": (256, 10, 1531), "benchmark-linearboost": (256, 10, 1531),
        "control": (256, 1000, 1543), "linearboost": (256, 1000, 1543),
    }
    assert recipe.CAP_SECONDS == action.CAP_SECONDS == 2100
    assert recipe.PARENT == action.PARENT == parent.PARENT
    assert recipe.PARENT.parent.name == "control" and recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == action.PARENT_SHA == parent.PARENT_SHA == (
        "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    )
    assert recipe.PARENT_NORMALIZER_COUNT == 43_008_000
    assert recipe.PARENT_ADAM_STEP == 140_000
    assert recipe.warm_load is action.warm_load
    assert recipe.evaluate_case is action.evaluate_case and recipe.decision is action.decision

    for control_mode, boost_mode in (
        ("smoke-control", "smoke-linearboost"),
        ("benchmark-control", "benchmark-linearboost"),
        ("control", "linearboost"),
    ):
        control, boost, control_agent, boost_agent = _paired_configs(control_mode, boost_mode)
        assert control.seed == boost.seed == recipe.MODES[control_mode][2]
        assert control.scene.num_envs == boost.scene.num_envs == recipe.MODES[control_mode][0]
        assert control_agent.max_iterations == boost_agent.max_iterations == recipe.MODES[control_mode][1]
        assert control_agent.save_interval == boost_agent.save_interval == 250
        assert control.observations == boost.observations
        assert control.actions == boost.actions
        assert control.commands == boost.commands
        assert control.events == boost.events
        assert control.curriculum == boost.curriculum == {}
        assert control.scene.entities == boost.scene.entities
        assert control_agent.actor == boost_agent.actor
        assert control_agent.clip_actions is boost_agent.clip_actions is None

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent = recipe.prepare_config(mode)
        action_cfg, _ = action.prepare_config(recipe.action_mode(mode))
        parent_cfg, parent_agent = parent.prepare_config("control")
        assert cfg.seed == agent.seed == seed
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert agent.experiment_name == recipe.PROTOCOL and agent.run_name == mode
        assert cfg.rewards[recipe.LINEAR_TERM].func is track_linear_velocity
        assert cfg.rewards[recipe.LINEAR_TERM].params == dict(std=math.sqrt(.1), command_name="twist")
        assert cfg.rewards[recipe.LINEAR_TERM].weight == recipe.linear_weight(mode)
        assert cfg.rewards["track_angular_velocity"].func is track_angular_velocity
        assert cfg.rewards["track_angular_velocity"].weight == parent_cfg.rewards["track_angular_velocity"].weight == 2.
        assert cfg.rewards["track_angular_velocity"].params == parent_cfg.rewards["track_angular_velocity"].params
        assert cfg.rewards["action_rate_l2"].weight == action_cfg.rewards["action_rate_l2"].weight == -1.
        assert cfg.rewards["action_rate_l2"].func is action_cfg.rewards["action_rate_l2"].func
        assert cfg.rewards["action_rate_l2"].params == action_cfg.rewards["action_rate_l2"].params == {}
        assert cfg.rewards["body_ang_vel"].weight == -.05
        assert cfg.rewards["angular_momentum"].weight == -.02
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["joint_limit"].params["margin"] == .15
        assert cfg.rewards["forward_error"].weight == 0.
        assert cfg.observations == action_cfg.observations
        assert cfg.actions == action_cfg.actions
        assert cfg.commands == action_cfg.commands
        assert cfg.events == action_cfg.events
        assert cfg.curriculum == {}
        assert agent.actor == parent_agent.actor and agent.clip_actions is None
        assert {key: value for key, value in cfg.metrics.items()
                if key not in ("linear_tracking_training_snapshot",)} == action_cfg.metrics
        assert cfg.metrics["linear_tracking_training_snapshot"].func is mdp.capture_linear_tracking_training_metric
        assert cfg.metrics["linear_tracking_training_snapshot"].per_substep is False

    cfg, agent = recipe.prepare_config("linearboost")
    actor_terms = cfg.observations["actor"].terms
    assert "head_command" in actor_terms and "body_command" in actor_terms
    assert tuple(actor_terms["joint_pos"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert tuple(actor_terms["joint_vel"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert 14 + 14 + 3 + 3 + 14 + 3 + 4 + 6 == 61
    assert cfg.actions["joint_pos"].scale == 1. and cfg.actions["joint_pos"].clip is None
    assert tuple(cfg.actions["joint_pos"].actuator_names) == (".*",)
    assert agent.actor.obs_normalization is True
    assert recipe.base.SEEDS == parent.base.SEEDS and recipe.base.CASES == parent.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "linearboost": 999,
    }


def _linear_metric_env(weight=2.):
    command = torch.tensor([[.3, 0., 0.], [0., 0., .2], [.2, 0., -.2], [0., 0., 0.]])
    velocity = torch.tensor([[.28, .01, 0.], [.03, -.02, 0.], [.17, .02, 0.], [0., 0., 0.]])
    term = NS(func=track_linear_velocity, params=dict(std=math.sqrt(.1), command_name="twist"), weight=weight)
    env = NS(
        num_envs=4, device="cpu", common_step_counter=77,
        scene={"robot": NS(data=NS(root_link_lin_vel_b=velocity))},
        command_manager=NS(get_command=lambda name: command),
        reward_manager=NS(get_term_cfg=lambda _name: term, active_terms=[recipe.LINEAR_TERM],
                          _step_reward=torch.zeros(4, 1)),
    )
    raw = term.func(env, **term.params)
    env.reward_manager._step_reward[:, 0] = weight * raw
    return env, term, raw


@pytest.mark.parametrize("weight", [2., 2.5])
def test_linear_metric_captures_positive_official_raw_kernel_before_reset(weight):
    env, term, expected_raw = _linear_metric_env(weight)
    initial_velocity = env.scene["robot"].data.root_link_lin_vel_b.clone()
    zero = mdp.capture_linear_tracking_training_metric(env)
    raw, weighted, actual_weight, step = env._microduck_linear_tracking_snapshot
    assert torch.allclose(raw, expected_raw)
    assert torch.allclose(weighted, raw * weight)
    assert actual_weight == term.weight == weight
    assert step == env.common_step_counter
    assert raw.shape == weighted.shape == (env.num_envs,)
    assert bool(((raw > 0.) & (raw <= 1.) & (weighted > 0.) & (weighted <= weight)).all())
    assert raw.requires_grad is False and weighted.requires_grad is False
    assert torch.equal(zero, torch.zeros(env.num_envs))
    env.scene["robot"].data.root_link_lin_vel_b.zero_()
    assert torch.equal(env._microduck_linear_tracking_snapshot[0], raw)
    assert torch.equal(env._microduck_linear_tracking_snapshot[1], weighted)
    assert not torch.equal(initial_velocity, env.scene["robot"].data.root_link_lin_vel_b)
    with pytest.raises(ValueError, match="previous linear tracking snapshot consumed"):
        mdp.capture_linear_tracking_training_metric(env)


@pytest.mark.parametrize("fault,exc,match", [
    ("func", ValueError, "fixed official linear tracking callable parameters and positive weight"),
    ("std", ValueError, "fixed official linear tracking callable parameters and positive weight"),
    ("command", ValueError, "fixed official linear tracking callable parameters and positive weight"),
    ("weight", ValueError, "fixed official linear tracking callable parameters and positive weight"),
    ("weight-bool", ValueError, "fixed official linear tracking callable parameters and positive weight"),
    ("actual", FloatingPointError, "actual positive linear tracking reward matches finite raw kernel"),
    ("nonfinite", FloatingPointError, "actual positive linear tracking reward matches finite raw kernel"),
])
def test_linear_metric_rejects_bad_callable_params_weight_or_actual(fault, exc, match):
    env, term, _ = _linear_metric_env()
    if fault == "func":
        term.func = lambda _env, **_params: torch.ones(env.num_envs)
    elif fault == "std":
        term.params["std"] = math.sqrt(.2)
    elif fault == "command":
        term.params["command_name"] = "other"
    elif fault == "weight":
        term.weight = 3.
    elif fault == "weight-bool":
        term.weight = True
    elif fault == "actual":
        env.reward_manager._step_reward[:, 0] -= .2
    else:
        env.scene["robot"].data.root_link_lin_vel_b[0, 0] = float("nan")
    with pytest.raises(exc, match=match):
        mdp.capture_linear_tracking_training_metric(env)


def _linear_observer_env(mode="smoke-linearboost"):
    from test_integrated_motor_range_action_release import _action_observer_env

    env = _action_observer_env(recipe.action_mode(mode))
    cfg, _ = recipe.prepare_config(mode)
    original_get = env.reward_manager.get_term_cfg
    terms = {name: original_get(name) for name in env.reward_manager.active_terms}
    cfg_term = cfg.rewards[recipe.LINEAR_TERM]
    terms[recipe.LINEAR_TERM] = NS(func=cfg_term.func, params=dict(cfg_term.params), weight=cfg_term.weight)
    env.reward_manager.active_terms.append(recipe.LINEAR_TERM)
    env.reward_manager.get_term_cfg = lambda name: terms[name] if name in terms else original_get(name)
    env.reward_manager._step_reward = torch.zeros(env.num_envs, len(env.reward_manager.active_terms))
    env.scene["robot"].data.root_link_lin_vel_b = torch.zeros(env.num_envs, 3)
    env.cfg = cfg
    return env


@pytest.mark.parametrize("fault", ["func", "params", "weight", "cached-weight"])
def test_linear_observer_before_rejects_live_and_cached_contract_drift(fault):
    env = _linear_observer_env()
    observer = recipe.MotorObserver("smoke-linearboost")
    term = env.reward_manager.get_term_cfg(recipe.LINEAR_TERM)
    if fault == "func":
        term.func = lambda _env, **_params: torch.ones(env.num_envs)
    elif fault == "params":
        term.params["std"] = math.sqrt(.2)
    elif fault == "weight":
        term.weight = 2.
    else:
        observer.linear_weight = 2.
    with pytest.raises(ValueError, match="fixed live linear tracking contract"):
        observer.before(env)


@pytest.mark.parametrize("fault", [None, "missing", "stale", "malformed", "wrong-reward"])
def test_linear_observer_rejects_bad_snapshots_and_consumes_valid_snapshots(monkeypatch, fault):
    # The step integration is covered below; this focused table verifies each
    # pre-reset snapshot fault after the inherited angular/action observer has
    # received its normal matching snapshots.
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _linear_observer_env("smoke-linearboost")
    observer = recipe.MotorObserver("smoke-linearboost")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    observer.before(env)
    action_term = env.reward_manager.get_term_cfg(action.ACTION_TERM)
    delta = torch.full((env.num_envs, 14), .1)
    _prepare_parent_metrics(env, action_term, delta)
    mdp.capture_action_rate_training_metric(env)
    linear_term = env.reward_manager.get_term_cfg(recipe.LINEAR_TERM)
    command = env.command_manager.get_command("twist")
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = command[:, :2] - torch.tensor([.04, .02])
    index = env.reward_manager.active_terms.index(recipe.LINEAR_TERM)
    raw_expected = linear_term.func(env, **linear_term.params)
    env.reward_manager._step_reward[:, index] = raw_expected * linear_term.weight
    if fault != "missing":
        mdp.capture_linear_tracking_training_metric(env)
    if fault == "stale":
        snap = env._microduck_linear_tracking_snapshot
        env._microduck_linear_tracking_snapshot = (*snap[:3], snap[3] - 1)
    elif fault == "malformed":
        env._microduck_linear_tracking_snapshot = (raw_expected, env.reward_manager._step_reward[:, index], 2.5)
    elif fault == "wrong-reward":
        env._microduck_linear_tracking_snapshot[1].zero_()
    match = ("same-step pre-reset linear tracking snapshot" if fault in ("missing", "stale", "malformed")
             else "actual positive linear tracking matches fixed weight")
    if fault is not None:
        with pytest.raises(ValueError, match=match):
            observer.step(env, sample, {})
        return
    saved = env._microduck_linear_tracking_snapshot
    assert saved is not None and saved[3] == env.common_step_counter
    assert torch.allclose(saved[0], raw_expected)
    assert torch.allclose(saved[1], raw_expected * 2.5)
    env.scene["robot"].data.root_link_lin_vel_b.zero_()
    observer.step(env, sample, {})
    assert env._microduck_linear_tracking_snapshot is None


def test_24_step_linear_observer_consumes_snapshots_and_aggregates_actual_strata(monkeypatch):
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _linear_observer_env("smoke-linearboost")
    observer = recipe.MotorObserver("smoke-linearboost")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    raw_means, weighted_means = [], []
    for step in range(24):
        observer.before(env)
        action_term = env.reward_manager.get_term_cfg(action.ACTION_TERM)
        delta = torch.full((env.num_envs, 14), (step + 1) / 100.)
        _prepare_parent_metrics(env, action_term, delta)
        mdp.capture_action_rate_training_metric(env)
        linear_term = env.reward_manager.get_term_cfg(recipe.LINEAR_TERM)
        command = env.command_manager.get_command("twist")
        env.scene["robot"].data.root_link_lin_vel_b[:, :2] = command[:, :2] - torch.tensor([.04 + step/1000, .02])
        index = env.reward_manager.active_terms.index(recipe.LINEAR_TERM)
        raw = linear_term.func(env, **linear_term.params)
        actual = raw * linear_term.weight
        env.reward_manager._step_reward[:, index] = actual
        mdp.capture_linear_tracking_training_metric(env)
        snapshot = env._microduck_linear_tracking_snapshot
        assert snapshot[3] == env.common_step_counter
        assert torch.allclose(snapshot[0], raw)
        assert torch.allclose(snapshot[1], actual)
        raw_means.append(float(raw.double().mean()))
        weighted_means.append(float(actual.double().mean()))
        # The post-reward snapshot must survive reset-side velocity changes.
        env.scene["robot"].data.root_link_lin_vel_b.zero_()
        observer.step(env, sample, {})
        assert env._microduck_linear_tracking_snapshot is None

    row = observer.update(env)
    assert row["linear_tracking_weight"] == 2.5
    assert row["linear_tracking_variance"] == .1
    assert row["linear_tracking_sample_count"] == env.num_envs * 24
    assert row["linear_tracking_raw_mean"] == pytest.approx(sum(raw_means) / 24)
    assert row["weighted_linear_tracking_reward_mean"] == pytest.approx(sum(weighted_means) / 24)
    strata = row["linear_tracking_command_strata"]
    assert set(strata) == set(recipe.GROUPS)
    assert {name: values["sample_count"] for name, values in strata.items()} == {
        "idle": 24, "straight": 61 * 24, "negative_yaw": 24, "positive_yaw": 24,
    }
    assert sum(values["sample_count"] for values in strata.values()) == env.num_envs * 24
    for values in strata.values():
        assert values["raw_mean"] == pytest.approx(values["raw_sum"] / values["sample_count"])
        assert values["weighted_reward_mean"] == pytest.approx(
            values["weighted_reward_sum"] / values["sample_count"]
        )


def _valid_linear_row(mode, *, worlds=None):
    worlds = recipe.MODES[mode][0] if worlds is None else worlds
    weight = recipe.linear_weight(mode)
    total = worlds * 24
    counts = dict(idle=total, straight=0,
                  **{"negative-low": 0, "negative-target": 0, "negative-high": 0,
                     "positive-low": 0, "positive-target": 0, "positive-high": 0})
    raw_sum = total * .5
    weighted_sum = weight * raw_sum
    groups = {
        name: dict(sample_count=count, raw_sum=(raw_sum if name == "idle" else 0.),
                   weighted_reward_sum=(weighted_sum if name == "idle" else 0.),
                   raw_mean=(.5 if count else None),
                   weighted_reward_mean=(weighted_sum/count if count else None))
        for name, count in (("idle", total), ("straight", 0),
                            ("negative_yaw", 0), ("positive_yaw", 0))
    }
    return dict(linear_tracking_weight=weight, linear_tracking_variance=.1,
                linear_tracking_sample_count=total, command_exposure_counts=counts,
                linear_tracking_command_strata=groups,
                linear_tracking_raw_mean=.5,
                weighted_linear_tracking_reward_mean=weighted_sum/total)


@pytest.mark.parametrize("mode", [
    "control", "linearboost", "smoke-control", "smoke-linearboost",
    "benchmark-control", "benchmark-linearboost",
])
def test_validate_linear_row_accepts_control_and_boost_strata(mode):
    row = _valid_linear_row(mode)
    recipe.validate_linear_row(row, mode, recipe.MODES[mode][0], 1)
    for empty in ("straight", "negative_yaw", "positive_yaw"):
        assert row["linear_tracking_command_strata"][empty]["sample_count"] == 0
        assert row["linear_tracking_command_strata"][empty]["raw_mean"] is None
        assert row["linear_tracking_command_strata"][empty]["weighted_reward_mean"] is None


@pytest.mark.parametrize("fault", ["weight", "variance", "count", "strata", "raw-bound",
                                    "negative-reward", "reward-bound", "reward-mismatch",
                                    "group-mean", "full-mean"])
def test_validate_linear_row_rejects_summary_drift(fault):
    row = _valid_linear_row("linearboost")
    groups = row["linear_tracking_command_strata"]
    if fault == "weight":
        row["linear_tracking_weight"] = 2.
    elif fault == "variance":
        row["linear_tracking_variance"] = .2
    elif fault == "count":
        row["linear_tracking_sample_count"] -= 1
    elif fault == "strata":
        groups["idle"]["sample_count"] -= 1
    elif fault == "raw-bound":
        groups["idle"]["raw_sum"] = groups["idle"]["sample_count"] + 1
    elif fault == "negative-reward":
        groups["idle"]["weighted_reward_sum"] = -1.
    elif fault == "reward-bound":
        groups["idle"]["weighted_reward_sum"] = groups["idle"]["sample_count"] * 2.5 + 1
    elif fault == "reward-mismatch":
        groups["idle"]["weighted_reward_sum"] += 1.
    elif fault == "group-mean":
        groups["idle"]["raw_mean"] += .01
    else:
        row["weighted_linear_tracking_reward_mean"] += .01
    with pytest.raises(ValueError):
        recipe.validate_linear_row(row, "linearboost", 64, 1)


def test_validate_action_row_delegates_to_fixed_action_control_mode(monkeypatch):
    captured = {}
    monkeypatch.setattr(action, "validate_action_row",
                        lambda row, mode, worlds, update: captured.update(
                            row=row, mode=mode, worlds=worlds, update=update))
    row = {"sentinel": 1}
    for mode, expected_action_mode in (
        ("smoke-control", "smoke-control"),
        ("smoke-linearboost", "smoke-control"),
        ("benchmark-control", "benchmark-control"),
        ("benchmark-linearboost", "benchmark-control"),
        ("control", "control"),
        ("linearboost", "control"),
    ):
        recipe.validate_action_row(row, mode, recipe.MODES[mode][0], 1)
        assert captured == dict(row=row, mode=expected_action_mode,
                                worlds=recipe.MODES[mode][0], update=1)

