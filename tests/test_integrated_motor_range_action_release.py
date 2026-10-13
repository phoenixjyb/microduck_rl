from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_motor_range_action_release as recipe
from mjlab_microduck import integrated_motor_range_yaw_only as original
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity
from mjlab.envs.mdp.rewards import action_rate_l2


def _same_config(mode):
    cfg, agent = recipe.prepare_config(mode)
    parent, parent_agent = original.prepare_config("control")
    assert cfg.seed == agent.seed == recipe.MODES[mode][2]
    assert agent.run_name == mode
    assert agent.experiment_name == recipe.PROTOCOL
    assert agent.max_iterations == recipe.MODES[mode][1]
    assert agent.save_interval == 250
    assert cfg.rewards[recipe.ACTION_TERM].func is parent.rewards[recipe.ACTION_TERM].func is action_rate_l2
    assert cfg.rewards[recipe.ACTION_TERM].params == parent.rewards[recipe.ACTION_TERM].params == {}
    assert cfg.rewards[recipe.ACTION_TERM].weight == recipe.schedule(mode)[0][1]
    assert {key: value for key, value in cfg.metrics.items() if key == "action_rate_training_snapshot"}
    return cfg, agent, parent, parent_agent


def test_modes_parent_and_config_match_except_action_coefficient_and_labels():
    assert recipe.PROTOCOL == "integrated-motor-range-action-release-v1"
    assert recipe.ARMS == ("control", "release")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1501), "smoke-release": (64, 5, 1501),
        "benchmark-control": (256, 10, 1511), "benchmark-release": (256, 10, 1511),
        "control": (256, 1000, 1517), "release": (256, 1000, 1517),
    }
    assert recipe.CAP_SECONDS == 2100
    assert recipe.PARENT == original.PARENT
    assert recipe.PARENT.parent.name == "control" and recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    assert recipe.PARENT_NORMALIZER_COUNT == original.PARENT_NORMALIZER_COUNT == 43_008_000
    assert recipe.PARENT_ADAM_STEP == original.PARENT_ADAM_STEP == 140_000
    assert recipe.warm_load is original.warm_load
    assert recipe.evaluate_case is original.evaluate_case and recipe.decision is original.decision

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent, parent, parent_agent = _same_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert agent.max_iterations == updates and cfg.seed == agent.seed == seed
        assert cfg.observations == parent.observations
        assert cfg.commands == parent.commands
        assert cfg.events == parent.events
        assert cfg.actions == parent.actions
        assert cfg.scene.entities == parent.scene.entities
        assert cfg.curriculum == parent.curriculum == {}
        assert agent.actor == parent_agent.actor
        assert agent.clip_actions is parent_agent.clip_actions is None
        assert cfg.rewards["track_angular_velocity"].weight == parent.rewards["track_angular_velocity"].weight == 2.
        assert cfg.rewards["track_angular_velocity"].params == parent.rewards["track_angular_velocity"].params
        assert cfg.rewards["track_angular_velocity"].func is parent.rewards["track_angular_velocity"].func is track_angular_velocity
        assert cfg.rewards["body_ang_vel"].weight == parent.rewards["body_ang_vel"].weight == -.05
        assert cfg.rewards["angular_momentum"].weight == parent.rewards["angular_momentum"].weight == -.02
        assert cfg.rewards["motor_load"].weight == parent.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == parent.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["joint_limit"].params["margin"] == .15
        assert cfg.rewards["joint_limit"].params["asset_cfg"].joint_names == ("^(left|right)_hip_yaw$",)
        assert cfg.rewards["forward_error"].weight == parent.rewards["forward_error"].weight == 0.
        assert {key: value for key, value in cfg.metrics.items() if key != "action_rate_training_snapshot"} == parent.metrics
        assert cfg.rewards[recipe.ACTION_TERM].weight == (-1. if mode.endswith("control") or mode.startswith("benchmark") else -.1)

    for control_mode, release_mode in (
        ("smoke-control", "smoke-release"),
        ("benchmark-control", "benchmark-release"),
        ("control", "release"),
    ):
        control, control_agent, _, _ = _same_config(control_mode)
        release, release_agent, _, _ = _same_config(release_mode)
        assert control.rewards[recipe.ACTION_TERM].weight == -1.
        assert release.rewards[recipe.ACTION_TERM].weight == recipe.schedule(release_mode)[0][1]
        release.rewards[recipe.ACTION_TERM].weight = control.rewards[recipe.ACTION_TERM].weight
        release_agent.run_name = control_agent.run_name
        assert repr(control) == repr(release)
        assert control_agent == release_agent

    cfg, agent, parent, parent_agent = _same_config("release")
    terms = cfg.observations["actor"].terms
    assert "head_command" in terms and "body_command" in terms
    assert tuple(terms["joint_pos"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert tuple(terms["joint_vel"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert 14 + 14 + 3 + 3 + 14 + 3 + 4 + 6 == 61
    assert cfg.actions["joint_pos"].scale == 1. and cfg.actions["joint_pos"].clip is None
    assert tuple(cfg.actions["joint_pos"].actuator_names) == (".*",)
    assert cfg.actions == parent.actions and agent.actor == parent_agent.actor
    assert recipe.base.SEEDS == original.base.SEEDS and recipe.base.CASES == original.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "release": 999,
    }

    with pytest.raises(ValueError):
        recipe.prepare_config("unlisted-mode")
    with pytest.raises(ValueError):
        recipe.schedule("unlisted-mode")


def test_schedule_boundaries_and_parent_final_weight_restoration():
    smoke = recipe.schedule("smoke-release")
    training = recipe.schedule("release")
    assert smoke == ((0, -.1), (1, -.4), (3, -1.))
    assert training == ((0, -.1), (250, -.4), (500, -1.))
    assert recipe.schedule("smoke-control") == ((0, -1.),)
    assert recipe.schedule("control") == ((0, -1.),)
    assert recipe.schedule("benchmark-release") == ((0, -1.),)
    assert recipe.schedule("benchmark-control") == ((0, -1.),)

    for stages in (smoke, training):
        for index, (iteration, value) in enumerate(stages):
            boundary = iteration * 24
            prior_value = stages[max(0,index-1)][1]
            assert motor.motor_weight(boundary - 1, stages) == prior_value
            assert motor.motor_weight(boundary, stages) == prior_value
            assert motor.motor_weight(boundary + 1, stages) == value
        assert motor.motor_weight(stages[-1][0] * 24 + 1, stages) == -1.
    assert motor.motor_weight(999 * 24, training) == -1.


def _valid_action_row(mode, update):
    worlds = recipe.MODES[mode][0]
    stages = recipe.schedule(mode)
    weights = [motor.motor_weight(step, stages)
               for step in range((update - 1) * 24, update * 24)]
    total = worlds * 24
    # Synthetic command exposure is all idle. This intentionally exercises
    # zero-count group means as well as the populated stratum.
    counts = dict(idle=total, straight=0,
                  **{"negative-low": 0, "negative-target": 0, "negative-high": 0,
                     "positive-low": 0, "positive-target": 0, "positive-high": 0})
    weighted_sum = worlds * sum(weights)
    groups = {
        name: dict(
            sample_count=count,
            cost_sum=float(count),
            weighted_reward_sum=(float(weighted_sum) if name == "idle" else 0.),
            cost_mean=(1. if count else None),
            weighted_reward_mean=(weighted_sum / count if count else None),
        )
        for name, count in (
            ("idle", total), ("straight", 0), ("negative_yaw", 0), ("positive_yaw", 0),
        )
    }
    return dict(
        action_rate_schedule=[list(stage) for stage in stages],
        action_rate_weight_min=min(weights),
        action_rate_weight_max=max(weights),
        curriculum={"action_rate_weight": weights[-1]},
        action_rate_command_strata=groups,
        command_exposure_counts=counts,
        action_rate_sample_count=total,
        action_rate_cost_mean=1.,
        weighted_action_rate_reward_mean=weighted_sum / total,
    )


@pytest.mark.parametrize("mode,update", [
    ("control", 1), ("release", 1), ("smoke-control", 5), ("smoke-release", 5),
    ("benchmark-control", 10), ("benchmark-release", 10),
    ("release", 250), ("release", 251), ("release", 500), ("release", 501),
    ("release", 1000),
])
def test_validate_action_row_accepts_aligned_modes_stages_and_empty_strata(mode, update):
    row = _valid_action_row(mode, update)
    recipe.validate_action_row(row, mode, recipe.MODES[mode][0], update)
    assert row["action_rate_command_strata"]["idle"]["sample_count"] == recipe.MODES[mode][0] * 24
    for empty in ("straight", "negative_yaw", "positive_yaw"):
        assert row["action_rate_command_strata"][empty]["sample_count"] == 0
        assert row["action_rate_command_strata"][empty]["cost_mean"] is None
        assert row["action_rate_command_strata"][empty]["weighted_reward_mean"] is None


def test_validate_action_row_observes_strict_training_stage_boundaries():
    boundary_cases = (
        (250, -.1, -.1, -.1),
        (251, -.4, -.1, -.4),
        (500, -.4, -.4, -.4),
        (501, -1., -.4, -1.),
        (1000, -1., -1., -1.),
    )
    for update, minimum, maximum, final_weight in boundary_cases:
        row = _valid_action_row("release", update)
        assert row["action_rate_weight_min"] == minimum
        assert row["action_rate_weight_max"] == maximum
        assert row["curriculum"]["action_rate_weight"] == final_weight
        recipe.validate_action_row(row, "release", 256, update)


@pytest.mark.parametrize("fault,match", [
    ("schedule", "actual action-rate stage weights"),
    ("weight", "actual action-rate stage weights"),
    ("count", "complete action-rate command strata"),
    ("sample-count", "complete action-rate command strata"),
    ("mean", "action-rate group and full means agree"),
    ("stratum-mean", "action-rate stratum means"),
    ("signed-cost", "finite signed action-rate stratum"),
    ("signed-reward", "finite signed action-rate stratum"),
])
def test_validate_action_row_rejects_schedule_count_mean_and_sign_drift(fault, match):
    row = _valid_action_row("release", 251)
    if fault == "schedule":
        row["action_rate_schedule"][0][1] = -.2
    elif fault == "weight":
        row["action_rate_weight_min"] += .01
    elif fault == "count":
        row["command_exposure_counts"]["idle"] -= 1
    elif fault == "sample-count":
        row["action_rate_sample_count"] -= 1
    elif fault == "mean":
        row["weighted_action_rate_reward_mean"] -= .01
    elif fault == "stratum-mean":
        row["action_rate_command_strata"]["idle"]["cost_mean"] += .01
    elif fault == "signed-cost":
        row["action_rate_command_strata"]["idle"]["cost_sum"] = -1.
    else:
        row["action_rate_command_strata"]["idle"]["weighted_reward_sum"] = 1.
    with pytest.raises(ValueError, match=match):
        recipe.validate_action_row(row, "release", 256, 251)


def test_validate_action_row_rejects_update_or_world_budget_mismatch():
    row = _valid_action_row("release", 251)
    with pytest.raises(ValueError, match="declared action-rate row budget"):
        recipe.validate_action_row(row, "release", 64, 251)
    with pytest.raises(ValueError, match="declared action-rate row budget"):
        recipe.validate_action_row(row, "release", 256, 1001)


def _metric_env(*, weight=-.4, func=action_rate_l2, params=None):
    action = torch.arange(42, dtype=torch.float32).reshape(3, 14).requires_grad_() / 10
    prior = torch.zeros_like(action)
    term = NS(func=func, params={} if params is None else params, weight=weight)
    env = NS(
        num_envs=3,
        device="cpu",
        common_step_counter=96,
        action_manager=NS(action=action, prev_action=prior),
        reward_manager=NS(
            get_term_cfg=lambda _name: term,
            active_terms=["action_rate_l2"],
            _step_reward=torch.zeros(3, 1),
        ),
    )
    delta = action.detach() - prior
    cost = delta.square().sum(dim=-1)
    env.reward_manager._step_reward[:, 0] = weight * cost
    return env, term, cost


def test_action_metric_snapshots_exact_raw_14d_cost_read_only_and_pre_reset():
    env, term, cost = _metric_env()
    original_action = env.action_manager.action.detach().clone()
    result = mdp.capture_action_rate_training_metric(env)
    snapshot_cost, actual, weight, step = env._microduck_action_rate_snapshot
    assert torch.equal(snapshot_cost, cost)
    assert torch.equal(actual, cost * term.weight)
    assert torch.equal(result, torch.zeros(env.num_envs))
    assert weight == term.weight == -.4 and step == env.common_step_counter
    assert snapshot_cost.shape == actual.shape == (3,)
    assert snapshot_cost.requires_grad is False and actual.requires_grad is False
    # The captured tensors do not alias policy output; autoreset/mutation is safe.
    env.action_manager.action.detach().zero_()
    assert torch.equal(env._microduck_action_rate_snapshot[0], cost)
    assert torch.equal(env._microduck_action_rate_snapshot[1], cost * -.4)
    assert not torch.equal(env.action_manager.action.detach(), original_action)
    with pytest.raises(ValueError, match="previous action-rate snapshot consumed"):
        mdp.capture_action_rate_training_metric(env)


@pytest.mark.parametrize("fault,match", [
    ("func", "unchanged action-rate callable and bounded negative weight"),
    ("params", "unchanged action-rate callable and bounded negative weight"),
    ("weight-positive", "unchanged action-rate callable and bounded negative weight"),
    ("weight-small", "unchanged action-rate callable and bounded negative weight"),
    ("weight-bool", "unchanged action-rate callable and bounded negative weight"),
    ("action-shape", "finite raw fourteen-action history"),
    ("prev-shape", "finite raw fourteen-action history"),
    ("action-nan", "finite raw fourteen-action history"),
    ("prev-inf", "finite raw fourteen-action history"),
    ("actual-drift", "actual action-rate reward matches finite raw cost"),
])
def test_action_metric_fails_closed_on_bad_live_contract(fault, match):
    env, term, _ = _metric_env()
    if fault == "func":
        term.func = lambda _env: torch.zeros(env.num_envs)
    elif fault == "params":
        term.params = {"ignored": True}
    elif fault == "weight-positive":
        term.weight = .1
    elif fault == "weight-small":
        term.weight = -1.1
    elif fault == "weight-bool":
        term.weight = True
    elif fault == "action-shape":
        env.action_manager.action = torch.zeros(env.num_envs, 13)
    elif fault == "prev-shape":
        env.action_manager.prev_action = torch.zeros(env.num_envs, 14, 1)
    elif fault == "action-nan":
        env.action_manager.action[0, 0] = float("nan")
    elif fault == "prev-inf":
        env.action_manager.prev_action[1, 3] = float("inf")
    else:
        env.reward_manager._step_reward[:, 0] += .25
    with pytest.raises((ValueError, FloatingPointError), match=match):
        mdp.capture_action_rate_training_metric(env)


def _action_observer_env(mode="smoke-release"):
    from test_integrated_motor_range_yaw_only import _observer_env

    env = _observer_env("smoke-control")
    cfg, _ = recipe.prepare_config(mode)
    terms = {name: env.reward_manager.get_term_cfg(name) for name in env.reward_manager.active_terms}
    action_cfg = cfg.rewards[recipe.ACTION_TERM]
    terms[recipe.ACTION_TERM] = NS(func=action_cfg.func, params=dict(action_cfg.params), weight=action_cfg.weight)
    env.reward_manager.active_terms.append(recipe.ACTION_TERM)
    original_get = env.reward_manager.get_term_cfg
    env.reward_manager.get_term_cfg = lambda name: terms[name] if name in terms else original_get(name)
    env.reward_manager._step_reward = torch.zeros(env.num_envs, len(env.reward_manager.active_terms))
    env.action_manager = NS(action=torch.zeros(env.num_envs, 14), prev_action=torch.zeros(env.num_envs, 14))
    command = torch.zeros(env.num_envs,3)
    command[1:,0] = .3
    command[2,2], command[3,2] = -.2,.2
    env.command_manager.get_command = lambda _name: command
    env.cfg = cfg
    return env


@pytest.mark.parametrize("fault", ["func", "params", "weight", "schedule"])
def test_action_observer_before_rejects_live_drift(fault):
    env = _action_observer_env()
    observer = recipe.MotorObserver("smoke-release")
    term = env.reward_manager.get_term_cfg(recipe.ACTION_TERM)
    if fault == "func":
        term.func = lambda _env: torch.zeros(env.num_envs)
    elif fault == "params":
        term.params["other"] = 1
    elif fault == "weight":
        term.weight = -.25
    else:
        observer.action_schedule = ((0, -.25),)
    match = "unchanged action-release schedule" if fault == "schedule" else "unchanged live raw action-rate function"
    with pytest.raises(ValueError, match=match):
        observer.before(env)


@pytest.mark.parametrize("counter,expected", [(25, -.4), (73, -1.)])
def test_observer_applies_live_schedule_without_mutating_cfg(counter, expected):
    env = _action_observer_env("smoke-release")
    observer = recipe.MotorObserver("smoke-release")
    initial_cfg_weight = env.cfg.rewards[recipe.ACTION_TERM].weight
    env.common_step_counter = counter
    observer.before(env)
    assert env.reward_manager.get_term_cfg(recipe.ACTION_TERM).weight == expected
    assert env.cfg.rewards[recipe.ACTION_TERM].weight == initial_cfg_weight == -.1


def _prepare_parent_metrics(env, term, action_delta, *, counter_step=1):
    # Match the existing 24-step yaw observer fixture while adding a real raw
    # action delta and reward column for the new instrumentation.
    env.common_step_counter += counter_step
    angular = env.reward_manager.get_term_cfg(original.TERM)
    angular_index = env.reward_manager.active_terms.index(original.TERM)
    original.previous.safety.capture_limit_metric(env)
    original.previous.forward.capture_forward_metric(env)
    reward = env.reward_manager._step_reward
    reward[:, env.reward_manager.active_terms.index("motor_load")] = 0.
    limit_cost = env._microduck_fresh_limit_snapshot[0]
    limit_term = env.reward_manager.get_term_cfg("joint_limit")
    reward[:, env.reward_manager.active_terms.index("joint_limit")] = limit_cost * limit_term.weight
    reward[:, env.reward_manager.active_terms.index("forward_error")] = 0.
    reward[:, angular_index] = angular.weight * angular.func(env, **angular.params)
    original.angular.capture_angular_metric(env)
    action_index = env.reward_manager.active_terms.index(recipe.ACTION_TERM)
    env.action_manager.prev_action.zero_()
    env.action_manager.action.copy_(action_delta)
    cost = action_delta.square().sum(dim=-1)
    env.reward_manager._step_reward[:, action_index] = term.weight * cost
    return cost


def test_24_step_observer_consumes_snapshots_and_aligns_schedule_profiles_and_command_strata(monkeypatch):
    env = _action_observer_env("smoke-release")
    # Start at the first compressed boundary so the 24-step telemetry window
    # includes both the initial and first released coefficient.
    env.common_step_counter = 23
    observer = recipe.MotorObserver("smoke-release")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.original.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    weights, costs = [], []
    for index in range(24):
        observer.before(env)
        term = env.reward_manager.get_term_cfg(recipe.ACTION_TERM)
        delta = torch.full((env.num_envs, 14), (index + 1) / 100.)
        cost = _prepare_parent_metrics(env, term, delta)
        mdp.capture_action_rate_training_metric(env)
        saved_cost, saved_reward, saved_weight, saved_step = env._microduck_action_rate_snapshot
        assert saved_step == env.common_step_counter
        assert saved_weight == term.weight
        assert torch.allclose(saved_cost, cost)
        assert torch.allclose(saved_reward, cost * saved_weight)
        costs.append(float(cost.double().mean()))
        weights.append(saved_weight)
        # Simulate autoreset after the snapshot callback but before telemetry.
        env.action_manager.action.zero_()
        env.action_manager.prev_action.fill_(9.)
        observer.step(env, sample, {})
        assert env._microduck_action_rate_snapshot is None
        assert observer.pending_action_masks is None

    row = observer.update(env)
    assert row["action_rate_schedule"] == [list(stage) for stage in recipe.SMOKE_STAGES]
    assert row["action_rate_weight_min"] == min(weights) == -.4
    assert row["action_rate_weight_max"] == max(weights) == -.1
    assert row["action_rate_cost_mean"] == pytest.approx(sum(costs) / 24)
    assert row["action_rate_sample_count"] == env.num_envs * 24
    assert row["weighted_action_rate_reward_mean"] == pytest.approx(
        sum(cost * weight for cost, weight in zip(costs, weights, strict=True)) / 24
    )
    strata = row["action_rate_command_strata"]
    assert set(strata) == set(recipe.GROUPS)
    assert {name: value["sample_count"] for name, value in strata.items()} == {
        "idle": 24, "straight": 61 * 24, "negative_yaw": 24, "positive_yaw": 24,
    }
    assert sum(value["sample_count"] for value in strata.values()) == env.num_envs * 24
    assert all(value["cost_mean"] is not None and value["weighted_reward_mean"] is not None
               for value in strata.values())


@pytest.mark.parametrize("fault", ["missing", "stale", "malformed", "wrong-reward"])
def test_observer_rejects_missing_or_invalid_action_snapshots(monkeypatch, fault):
    from mjlab_microduck import integrated_motor_range_yaw_only as angular

    env = _action_observer_env()
    observer = recipe.MotorObserver("smoke-release")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.original.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    observer.before(env)
    term = env.reward_manager.get_term_cfg(recipe.ACTION_TERM)
    cost = _prepare_parent_metrics(env, term, torch.full((env.num_envs, 14), .2))
    action_index = env.reward_manager.active_terms.index(recipe.ACTION_TERM)
    if fault != "missing":
        mdp.capture_action_rate_training_metric(env)
    if fault == "stale":
        snap = env._microduck_action_rate_snapshot
        env._microduck_action_rate_snapshot = (*snap[:3], snap[3] - 1)
    elif fault == "malformed":
        env._microduck_action_rate_snapshot = (cost, env.reward_manager._step_reward[:, action_index], -.4)
    elif fault == "wrong-reward":
        env._microduck_action_rate_snapshot[1].zero_()
    match = "same-step pre-reset action-rate snapshot" if fault in ("missing", "stale", "malformed") else "actual negative action-rate reward matches live schedule"
    with pytest.raises(ValueError, match=match):
        observer.step(env, sample, {})

