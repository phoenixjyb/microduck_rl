import copy
import math
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_motor_range_action_release as action
from mjlab_microduck import integrated_motor_range_head_lesson as recipe
from mjlab_microduck import integrated_motor_range_linear_boost as linear
from mjlab_microduck import integrated_motor_range_yaw_only as parent
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity, track_linear_velocity


def _paired_configs(control_mode, quiet_mode):
    control, control_agent = recipe.prepare_config(control_mode)
    quiet, quiet_agent = recipe.prepare_config(quiet_mode)
    assert control.commands[recipe.HEAD_TERM].ranges == recipe.FULL_RANGES
    assert quiet.commands[recipe.HEAD_TERM].ranges == recipe.BEGINNER_RANGES
    quiet.commands[recipe.HEAD_TERM].ranges = control.commands[recipe.HEAD_TERM].ranges
    quiet_agent.run_name = control_agent.run_name
    assert repr(control) == repr(quiet)
    assert control_agent == quiet_agent
    return control, quiet


def test_modes_parent_and_head_ranges_are_the_only_environment_change():
    assert recipe.PROTOCOL == "integrated-motor-range-head-lesson-v1"
    assert recipe.ARMS == ("control", "quiethead")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1553), "smoke-quiethead": (64, 5, 1553),
        "benchmark-control": (256, 10, 1559), "benchmark-quiethead": (256, 10, 1559),
        "control": (256, 1000, 1567), "quiethead": (256, 1000, 1567),
    }
    assert recipe.CAP_SECONDS == linear.CAP_SECONDS == 2100
    assert recipe.PARENT == linear.PARENT == parent.PARENT
    assert recipe.PARENT.parent.name == "control" and recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == linear.PARENT_SHA == parent.PARENT_SHA == (
        "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    )
    assert recipe.PARENT_NORMALIZER_COUNT == 43_008_000
    assert recipe.PARENT_ADAM_STEP == 140_000
    assert recipe.warm_load is linear.warm_load
    assert recipe.evaluate_case is linear.evaluate_case and recipe.decision is linear.decision
    assert recipe.FULL_RANGES == ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-.31, .31))
    assert recipe.BEGINNER_RANGES == ((-.05, .05), (-.05, .05), (-.07, .07), (-.015, .015))

    for control_mode, quiet_mode in (
        ("smoke-control", "smoke-quiethead"),
        ("benchmark-control", "benchmark-quiethead"),
        ("control", "quiethead"),
    ):
        control, quiet = _paired_configs(control_mode, quiet_mode)
        assert control.seed == quiet.seed == recipe.MODES[control_mode][2]
        assert control.scene.num_envs == quiet.scene.num_envs == recipe.MODES[control_mode][0]
        assert control.rewards == quiet.rewards
        assert control.observations == quiet.observations
        assert control.actions == quiet.actions
        assert control.events == quiet.events
        assert control.curriculum == quiet.curriculum == {}
        assert control.commands.keys() == quiet.commands.keys()
        for name in control.commands:
            if name != recipe.HEAD_TERM:
                assert control.commands[name] == quiet.commands[name]

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent = recipe.prepare_config(mode)
        baseline, baseline_agent = linear.prepare_config(recipe.reference_mode(mode))
        assert cfg.seed == agent.seed == seed
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert agent.experiment_name == recipe.PROTOCOL and agent.run_name == mode
        assert cfg.commands[recipe.HEAD_TERM].ranges == recipe.head_ranges(mode)
        assert cfg.rewards == baseline.rewards
        assert cfg.observations == baseline.observations
        assert cfg.actions == baseline.actions
        assert cfg.events == baseline.events
        assert cfg.scene.entities == baseline.scene.entities
        assert cfg.curriculum == baseline.curriculum == {}
        assert cfg.commands.keys() == baseline.commands.keys()
        for name in cfg.commands:
            if name != recipe.HEAD_TERM:
                assert cfg.commands[name] == baseline.commands[name]
        assert agent.actor == baseline_agent.actor and agent.clip_actions is None
        assert agent.actor.obs_normalization is True
        assert cfg.observations["actor"].terms["head_command"].params["command_name"] == "head_pose"
        assert cfg.commands[recipe.HEAD_TERM].ranges != ((0., 0.),) * 4
        assert cfg.rewards["track_linear_velocity"].func is track_linear_velocity
        assert cfg.rewards["track_linear_velocity"].weight == baseline.rewards["track_linear_velocity"].weight
        assert cfg.rewards["track_linear_velocity"].params == dict(std=math.sqrt(.1), command_name="twist")
        assert cfg.rewards["action_rate_l2"].weight == -1.
        assert cfg.rewards["track_angular_velocity"].func is track_angular_velocity
        assert cfg.rewards["track_angular_velocity"].weight == 2.
        assert cfg.rewards["body_ang_vel"].weight == -.05
        assert cfg.rewards["angular_momentum"].weight == -.02
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["joint_limit"].params["margin"] == .15
        assert cfg.rewards["forward_error"].weight == 0.
        assert cfg.rewards["head_pose_tracking"].weight == 2.
        assert cfg.rewards["head_pose_bias"].weight == 3.
        assert cfg.rewards["head_pose_tracking"].params == baseline.rewards["head_pose_tracking"].params
        assert cfg.rewards["head_pose_bias"].params == baseline.rewards["head_pose_bias"].params

    cfg, _ = recipe.prepare_config("quiethead")
    actor_terms = cfg.observations["actor"].terms
    assert "head_command" in actor_terms and "body_command" in actor_terms
    assert tuple(actor_terms["joint_pos"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert tuple(actor_terms["joint_vel"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert 14 + 14 + 3 + 3 + 14 + 3 + 4 + 6 == 61
    assert cfg.actions["joint_pos"].scale == 1. and cfg.actions["joint_pos"].clip is None
    assert tuple(cfg.actions["joint_pos"].actuator_names) == (".*",)
    assert recipe.base.SEEDS == parent.base.SEEDS and recipe.base.CASES == parent.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "quiethead": 999,
    }


def test_head_ranges_match_existing_beginner_lesson_and_leave_linear_recipe_unchanged():
    assert recipe.head_ranges("control") == recipe.FULL_RANGES
    assert recipe.head_ranges("quiethead") == recipe.BEGINNER_RANGES
    assert recipe.head_ranges("smoke-quiethead") == recipe.BEGINNER_RANGES
    assert recipe.head_ranges("benchmark-quiethead") == recipe.BEGINNER_RANGES
    assert recipe.reference_mode("smoke-quiethead") == "smoke-control"
    assert recipe.reference_mode("benchmark-quiethead") == "benchmark-control"
    assert recipe.reference_mode("quiethead") == "control"

    baseline, _ = linear.prepare_config("control")
    old_ranges = copy.deepcopy(baseline.commands[recipe.HEAD_TERM].ranges)
    assert old_ranges == recipe.FULL_RANGES
    lesson, _ = recipe.prepare_config("quiethead")
    assert lesson.commands[recipe.HEAD_TERM].ranges == recipe.BEGINNER_RANGES
    assert baseline.commands[recipe.HEAD_TERM].ranges == old_ranges
    with pytest.raises(ValueError):
        recipe.head_ranges("unlisted-mode")
    with pytest.raises(ValueError):
        recipe.prepare_config("unlisted-mode")


def _head_observer_env(mode="smoke-quiethead"):
    from test_integrated_motor_range_linear_boost import _linear_observer_env

    cfg, _ = recipe.prepare_config(mode)
    env = _linear_observer_env(recipe.reference_mode(mode))
    original_get_reward_term = env.reward_manager.get_term_cfg
    terms = {name: original_get_reward_term(name) for name in env.reward_manager.active_terms}
    for name in recipe.HEAD_REWARD_WEIGHTS:
        cfg_term = cfg.rewards[name]
        terms[name] = NS(func=cfg_term.func, params=dict(cfg_term.params), weight=cfg_term.weight)
        env.reward_manager.active_terms.append(name)
    env.reward_manager.get_term_cfg = lambda name: (
        terms[name] if name in terms else original_get_reward_term(name)
    )
    env.reward_manager._step_reward = torch.zeros(env.num_envs, len(env.reward_manager.active_terms))

    original_get_command = env.command_manager.get_command
    head = torch.zeros(env.num_envs, 4)
    commands = {"head": head}
    env.command_manager.get_command = lambda name: commands["head"] if name == recipe.HEAD_TERM else original_get_command(name)
    original_get_command_term = env.command_manager.get_term
    live_head_cfg = copy.deepcopy(cfg.commands[recipe.HEAD_TERM])
    env.command_manager.get_term = lambda name: (
        NS(cfg=live_head_cfg) if name == recipe.HEAD_TERM else original_get_command_term(name)
    )
    env._head_command_holder = commands
    env._head_live_cfg = live_head_cfg
    env.cfg = cfg
    return env


def _head_values(step, worlds=64):
    sign = 1. if step % 2 == 0 else -1.
    values = torch.zeros(worlds, 4)
    values[:worlds // 2, 0] = sign * .02
    values[worlds // 2:3 * worlds // 4, 1] = sign * .03
    values[3 * worlds // 4:7 * worlds // 8, 2] = sign * .04
    values[7 * worlds // 8:, 3] = sign * .005
    return values


@pytest.mark.parametrize("fault", ["cfg", "tracking-func", "tracking-params", "tracking-weight",
                                    "bias-func", "bias-params", "bias-weight", "shape", "nan", "bounds"])
def test_head_observer_before_guards_live_config_rewards_and_actor_slots(fault):
    env = _head_observer_env()
    observer = recipe.MotorObserver("smoke-quiethead")
    env._head_command_holder["head"] = _head_values(0)
    if fault == "cfg":
        env._head_live_cfg.ranges = recipe.FULL_RANGES
    elif fault == "tracking-func":
        env.reward_manager.get_term_cfg("head_pose_tracking").func = lambda _env, **_params: torch.ones(env.num_envs)
    elif fault == "tracking-params":
        env.reward_manager.get_term_cfg("head_pose_tracking").params["std"] = .1
    elif fault == "tracking-weight":
        env.reward_manager.get_term_cfg("head_pose_tracking").weight = 1.
    elif fault == "bias-func":
        env.reward_manager.get_term_cfg("head_pose_bias").func = lambda _env, **_params: torch.zeros(env.num_envs)
    elif fault == "bias-params":
        env.reward_manager.get_term_cfg("head_pose_bias").params["tau_s"] = .2
    elif fault == "bias-weight":
        env.reward_manager.get_term_cfg("head_pose_bias").weight = 2.
    elif fault == "shape":
        env._head_command_holder["head"] = torch.zeros(env.num_envs, 3)
    elif fault == "nan":
        env._head_command_holder["head"][0, 1] = float("nan")
    else:
        env._head_command_holder["head"][0, 2] = .08
    message = {
        "cfg": "unchanged live head-command configuration",
        "tracking-func": "unchanged live head reward contract",
        "tracking-params": "unchanged live head reward contract",
        "tracking-weight": "unchanged live head reward contract",
        "bias-func": "unchanged live head reward contract",
        "bias-params": "unchanged live head reward contract",
        "bias-weight": "unchanged live head reward contract",
        "shape": "finite four-slot actor head command",
        "nan": "finite four-slot actor head command",
        "bounds": "actor head command inside live lesson ranges",
    }[fault]
    with pytest.raises(ValueError, match=message):
        observer.before(env)


def test_head_observer_snapshots_pre_action_command_and_reads_completed_reward_rows(monkeypatch):
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _head_observer_env("smoke-quiethead")
    observer = recipe.MotorObserver("smoke-quiethead")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    env._head_command_holder["head"] = _head_values(0)

    observer.before(env)
    pending = observer.head_pending
    assert pending[0] == env.common_step_counter
    assert pending[1] == pytest.approx([0., 0., 0., 0.])
    assert pending[2] == pytest.approx([.02, .03, .04, .005])
    assert pending[3] == pytest.approx([.01, .0075, .005, .000625])
    assert pending[4] == [32, 16, 8, 8]

    action_term = env.reward_manager.get_term_cfg(action.ACTION_TERM)
    _prepare_parent_metrics(env, action_term, torch.full((env.num_envs, 14), .1))
    mdp.capture_action_rate_training_metric(env)
    linear_term = env.reward_manager.get_term_cfg(linear.LINEAR_TERM)
    twist = env.command_manager.get_command("twist")
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = twist[:, :2] - torch.tensor([.04, .02])
    raw_linear = linear_term.func(env, **linear_term.params)
    linear_index = env.reward_manager.active_terms.index(linear.LINEAR_TERM)
    env.reward_manager._step_reward[:, linear_index] = raw_linear * linear_term.weight
    mdp.capture_linear_tracking_training_metric(env)

    tracking_values = torch.linspace(.6, 1.8, env.num_envs)
    bias_values = torch.linspace(-.1, -.4, env.num_envs)
    reward = env.reward_manager._step_reward
    reward[:, env.reward_manager.active_terms.index("head_pose_tracking")] = tracking_values
    reward[:, env.reward_manager.active_terms.index("head_pose_bias")] = bias_values
    # Simulate command resampling/autoreset after the actor's input and reward
    # rows have been recorded. The observer must retain the exact prior values.
    env._head_command_holder["head"] = torch.full((env.num_envs, 4), 9.)
    env.scene["robot"].data.joint_pos.zero_()
    assert observer.head_pending is pending
    assert torch.equal(reward[:, env.reward_manager.active_terms.index("head_pose_tracking")], tracking_values)
    observer.step(env, sample, {})
    assert observer.head_pending is None
    assert observer.head_samples[0] == pending[1:]
    assert observer.head_rewards[0] == pytest.approx({
        "head_pose_tracking": float(tracking_values.mean()),
        "head_pose_bias": float(bias_values.mean()),
    })


def test_head_observer_rejects_missing_stale_reward_and_incomplete_step_records(monkeypatch):
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _head_observer_env()
    observer = recipe.MotorObserver("smoke-quiethead")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))

    with pytest.raises(ValueError, match="same-step pre-action head command consumed"):
        observer.step(env, sample, {})
    observer.before(env)
    with pytest.raises(ValueError, match="previous actor head command consumed"):
        observer.before(env)

    action_term = env.reward_manager.get_term_cfg(action.ACTION_TERM)
    _prepare_parent_metrics(env, action_term, torch.full((env.num_envs, 14), .1))
    mdp.capture_action_rate_training_metric(env)
    linear_term = env.reward_manager.get_term_cfg(linear.LINEAR_TERM)
    twist = env.command_manager.get_command("twist")
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = twist[:, :2]
    raw = linear_term.func(env, **linear_term.params)
    env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(linear.LINEAR_TERM)] = raw * linear_term.weight
    mdp.capture_linear_tracking_training_metric(env)
    env.common_step_counter += 1
    with pytest.raises(ValueError, match="same-step pre-action head command consumed"):
        observer.step(env, sample, {})

    incomplete = recipe.MotorObserver("smoke-quiethead")
    fresh_env = _head_observer_env()
    with pytest.raises(ValueError, match="complete head-command update telemetry"):
        incomplete.update(fresh_env)


@pytest.mark.parametrize("fault", ["tracking-high", "bias-positive"])
def test_head_observer_rejects_invalid_completed_reward_sign_or_range(monkeypatch, fault):
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _head_observer_env()
    observer = recipe.MotorObserver("smoke-quiethead")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    env._head_command_holder["head"] = _head_values(0)
    observer.before(env)
    _prepare_parent_metrics(env, env.reward_manager.get_term_cfg(action.ACTION_TERM),
                            torch.full((env.num_envs, 14), .1))
    mdp.capture_action_rate_training_metric(env)
    linear_term = env.reward_manager.get_term_cfg(linear.LINEAR_TERM)
    twist = env.command_manager.get_command("twist")
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = twist[:, :2]
    raw = linear_term.func(env, **linear_term.params)
    env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(linear.LINEAR_TERM)] = raw * linear_term.weight
    mdp.capture_linear_tracking_training_metric(env)
    track_index = env.reward_manager.active_terms.index("head_pose_tracking")
    bias_index = env.reward_manager.active_terms.index("head_pose_bias")
    env.reward_manager._step_reward[:, track_index] = 3. if fault == "tracking-high" else 1.
    env.reward_manager._step_reward[:, bias_index] = -.2 if fault == "tracking-high" else .1
    with pytest.raises(ValueError, match="actual signed completed-step head reward"):
        observer.step(env, sample, {})


def test_head_observer_aggregates_24_completed_steps_including_zero_command_slots(monkeypatch):
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _head_observer_env("smoke-quiethead")
    observer = recipe.MotorObserver("smoke-quiethead")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost", lambda _env, **_params: torch.zeros(env.num_envs))
    tracking_means, bias_means = [], []
    for step in range(24):
        env._head_command_holder["head"] = _head_values(step)
        observer.before(env)
        action_term = env.reward_manager.get_term_cfg(action.ACTION_TERM)
        _prepare_parent_metrics(env, action_term, torch.full((env.num_envs, 14), .1))
        mdp.capture_action_rate_training_metric(env)
        linear_term = env.reward_manager.get_term_cfg(linear.LINEAR_TERM)
        twist = env.command_manager.get_command("twist")
        env.scene["robot"].data.root_link_lin_vel_b[:, :2] = twist[:, :2] - torch.tensor([.04, .02])
        raw = linear_term.func(env, **linear_term.params)
        env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(linear.LINEAR_TERM)] = raw * linear_term.weight
        mdp.capture_linear_tracking_training_metric(env)
        tracking = torch.full((env.num_envs,), .8 + step * .01)
        bias = torch.full((env.num_envs,), -.15 - step * .002)
        reward = env.reward_manager._step_reward
        reward[:, env.reward_manager.active_terms.index("head_pose_tracking")] = tracking
        reward[:, env.reward_manager.active_terms.index("head_pose_bias")] = bias
        tracking_means.append(float(tracking.mean()))
        bias_means.append(float(bias.mean()))
        # Change/reset the actor command and joint state after the pre-action
        # capture; telemetry must use pending command + completed reward rows.
        env._head_command_holder["head"] = torch.zeros(env.num_envs, 4)
        env.scene["robot"].data.joint_pos.zero_()
        observer.step(env, sample, {})
        assert observer.head_pending is None

    row = observer.update(env)
    assert row["head_command_ranges"] == [list(r) for r in recipe.BEGINNER_RANGES]
    assert row["curriculum"]["head_pose_ranges"] == [list(r) for r in recipe.BEGINNER_RANGES]
    assert row["head_command_steps"] == 24
    assert row["head_command_sample_count"] == env.num_envs * 24
    assert row["head_reward_weights"] == recipe.HEAD_REWARD_WEIGHTS
    assert row["head_reward_sample_count"] == env.num_envs * 24
    assert row["head_reward_means"] == pytest.approx({
        "head_pose_tracking": sum(tracking_means) / 24,
        "head_pose_bias": sum(bias_means) / 24,
    })
    assert row["head_command_min"] == pytest.approx([-.02, -.03, -.04, -.005])
    assert row["head_command_max"] == pytest.approx([.02, .03, .04, .005])
    assert row["head_command_abs_mean"] == pytest.approx([.01, .0075, .005, .000625])
    assert row["head_command_nonzero_count"] == [32 * 24, 16 * 24, 8 * 24, 8 * 24]


def _valid_head_row(mode):
    worlds = recipe.MODES[mode][0]
    ranges = recipe.head_ranges(mode)
    count = worlds * 24
    fractions = (.5, .25, .125, .125)
    magnitudes = (.02, .03, .04, .005)
    return dict(
        head_command_ranges=[list(r) for r in ranges],
        curriculum={"head_pose_ranges": [list(r) for r in ranges]},
        head_command_steps=24,
        head_command_sample_count=count,
        head_reward_weights=dict(recipe.HEAD_REWARD_WEIGHTS),
        head_reward_sample_count=count,
        head_reward_means={"head_pose_tracking": 1.1, "head_pose_bias": -.25},
        head_command_min=[-v for v in magnitudes],
        head_command_max=list(magnitudes),
        head_command_abs_mean=[m*f for m, f in zip(magnitudes, fractions, strict=True)],
        head_command_nonzero_count=[int(count*f) for f in fractions],
    )


@pytest.mark.parametrize("mode", ["control", "quiethead", "smoke-control", "smoke-quiethead",
                                   "benchmark-control", "benchmark-quiethead"])
def test_validate_head_row_accepts_declared_ranges_rewards_and_slot_statistics(mode):
    row = _valid_head_row(mode)
    recipe.validate_head_row(row, mode, recipe.MODES[mode][0], 1)


@pytest.mark.parametrize("fault", ["ranges", "curriculum", "steps", "command-count", "reward-weights",
                                    "reward-count", "tracking-sign", "tracking-range", "bias-sign",
                                    "nonfinite", "stat-length", "bounds", "zero-count-mean"])
def test_validate_head_row_rejects_invalid_ranges_rewards_and_statistics(fault):
    row = _valid_head_row("quiethead")
    if fault == "ranges":
        row["head_command_ranges"][0] = list(recipe.FULL_RANGES[0])
    elif fault == "curriculum":
        row["curriculum"]["head_pose_ranges"][0] = list(recipe.FULL_RANGES[0])
    elif fault == "steps":
        row["head_command_steps"] = 23
    elif fault == "command-count":
        row["head_command_sample_count"] -= 1
    elif fault == "reward-weights":
        row["head_reward_weights"]["head_pose_bias"] = 2.
    elif fault == "reward-count":
        row["head_reward_sample_count"] -= 1
    elif fault == "tracking-sign":
        row["head_reward_means"]["head_pose_tracking"] = -.1
    elif fault == "tracking-range":
        row["head_reward_means"]["head_pose_tracking"] = 2.1
    elif fault == "bias-sign":
        row["head_reward_means"]["head_pose_bias"] = .1
    elif fault == "nonfinite":
        row["head_command_abs_mean"][1] = float("nan")
    elif fault == "stat-length":
        row["head_command_max"] = row["head_command_max"][:3]
    elif fault == "bounds":
        row["head_command_max"][2] = .08
    else:
        row["head_command_nonzero_count"][0] = 0
    with pytest.raises(ValueError):
        recipe.validate_head_row(row, "quiethead", 256, 1)


def test_validate_head_row_rejects_world_or_update_budget_mismatch():
    row = _valid_head_row("quiethead")
    with pytest.raises(ValueError, match="declared head-lesson row budget"):
        recipe.validate_head_row(row, "quiethead", 64, 1)
    with pytest.raises(ValueError, match="declared head-lesson row budget"):
        recipe.validate_head_row(row, "quiethead", 256, 1001)
