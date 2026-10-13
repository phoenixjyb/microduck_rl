import copy
from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity

from mjlab_microduck import integrated_motor_range_head_lesson as head
from mjlab_microduck import integrated_motor_range_action_release as action
from mjlab_microduck import integrated_motor_range_linear_boost as linear
from mjlab_microduck import integrated_motor_range_yaw_only as yaw_only
from mjlab_microduck import integrated_quiet_head_gentle_turn_lesson as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp


def _paired_configs(control_mode, gentle_mode):
    control, control_agent = recipe.prepare_config(control_mode)
    gentle, gentle_agent = recipe.prepare_config(gentle_mode)
    normalized_control, normalized_gentle = copy.deepcopy(control), copy.deepcopy(gentle)
    normalized_control_agent, normalized_gentle_agent = copy.deepcopy(control_agent), copy.deepcopy(gentle_agent)
    assert normalized_control.commands["twist"].ranges.ang_vel_z == (-.5, .5)
    assert normalized_gentle.commands["twist"].ranges.ang_vel_z == (-.25, .25)
    normalized_gentle.commands["twist"].ranges.ang_vel_z = (-.5, .5)
    normalized_gentle_agent.run_name = normalized_control_agent.run_name
    assert repr(normalized_control) == repr(normalized_gentle)
    assert normalized_control_agent == normalized_gentle_agent
    return control, gentle


def test_modes_parent_protocol_and_yaw_support_are_the_only_environment_change():
    assert recipe.PROTOCOL == "integrated-quiet-head-gentle-turn-v1"
    assert recipe.ARMS == ("control", "gentleturn")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1673), "smoke-gentleturn": (64, 5, 1673),
        "benchmark-control": (256, 10, 1693), "benchmark-gentleturn": (256, 10, 1693),
        "control": (256, 1000, 1697), "gentleturn": (256, 1000, 1697),
    }
    assert recipe.CAP_SECONDS == 2100 and recipe.SAVE_INTERVAL == 250
    assert recipe.PARENT == head.PARENT == yaw_only.PARENT
    assert recipe.PARENT.parent.name == "control" and recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == head.PARENT_SHA == yaw_only.PARENT_SHA == (
        "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    )
    assert recipe.PARENT_NORMALIZER_COUNT == 43_008_000
    assert recipe.PARENT_ADAM_STEP == 140_000
    assert recipe.warm_load is head.warm_load and recipe.decision is head.decision
    assert recipe.evaluate_case.__module__ == recipe.__name__

    for control_mode, gentle_mode in (
        ("smoke-control", "smoke-gentleturn"),
        ("benchmark-control", "benchmark-gentleturn"),
        ("control", "gentleturn"),
    ):
        control, gentle = _paired_configs(control_mode, gentle_mode)
        assert control.seed == gentle.seed == recipe.MODES[control_mode][2]
        assert control.rewards == gentle.rewards
        assert control.observations == gentle.observations
        assert control.actions == gentle.actions
        assert control.events == gentle.events
        assert control.curriculum == gentle.curriculum == {}
        assert control.commands.keys() == gentle.commands.keys()
        for name in control.commands:
            if name != "twist":
                assert control.commands[name] == gentle.commands[name]
        assert control.commands["head_pose"].ranges == gentle.commands["head_pose"].ranges == head.BEGINNER_RANGES

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent = recipe.prepare_config(mode)
        reference, reference_agent = head.prepare_config(recipe.reference_mode(mode))
        support = (-.25, .25) if "gentleturn" in mode else (-.5, .5)
        assert cfg.seed == agent.seed == seed
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert agent.experiment_name == recipe.PROTOCOL and agent.run_name == mode
        assert cfg.commands["twist"].ranges.ang_vel_z == support
        assert cfg.commands["twist"].ranges.lin_vel_x == (.2, .4)
        assert cfg.commands["twist"].ranges.lin_vel_y == (0., 0.)
        assert cfg.commands["twist"].rel_standing_envs == .25
        assert cfg.commands["twist"].rel_forward_envs == .25
        assert cfg.commands["twist"].rel_turn_in_place_envs == 0.
        assert cfg.commands["twist"].resampling_time_range == (3., 5.)
        assert cfg.commands["head_pose"].ranges == head.BEGINNER_RANGES
        assert recipe.head_ranges(mode) == head.BEGINNER_RANGES
        assert cfg.rewards == reference.rewards
        assert cfg.observations == reference.observations
        assert cfg.actions == reference.actions
        assert cfg.events == reference.events
        assert cfg.scene.entities == reference.scene.entities
        assert cfg.curriculum == reference.curriculum == {}
        assert cfg.commands.keys() == reference.commands.keys()
        assert agent.actor == reference_agent.actor and agent.clip_actions is None
        assert agent.actor.obs_normalization is True
        assert sum((14, 14, 3, 3, 14, 3, 4, 6)) == 61
        assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
        assert cfg.rewards["track_angular_velocity"].func is track_angular_velocity
        assert cfg.rewards["track_angular_velocity"].weight == 2.
        assert cfg.rewards["track_angular_velocity"].params == {"std": 2 ** -.5, "command_name": "twist"}
        assert cfg.rewards["track_linear_velocity"].weight == 2.
        assert cfg.rewards["action_rate_l2"].weight == -1.
        assert cfg.rewards["body_ang_vel"].weight == -.05
        assert cfg.rewards["angular_momentum"].weight == -.02
        assert cfg.rewards["motor_load"].weight == -1.
        assert cfg.rewards["joint_limit"].weight == -4.
        assert cfg.rewards["joint_limit"].params["margin"] == .15
        assert cfg.rewards["forward_error"].weight == 0.
        assert cfg.rewards["head_pose_tracking"].weight == 2.
        assert cfg.rewards["head_pose_bias"].weight == 3.
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "gentleturn": 999,
    }


def test_reference_modes_validators_and_eval_use_the_unchanged_trace_contract(monkeypatch):
    assert recipe.reference_mode("smoke-control") == "smoke-quiethead"
    assert recipe.reference_mode("benchmark-gentleturn") == "benchmark-quiethead"
    assert recipe.reference_mode("gentleturn") == "quiethead"
    for fn in (recipe.yaw_support, recipe.yaw_variance, recipe.reference_mode, recipe.head_ranges,
               recipe.prepare_config):
        with pytest.raises(ValueError):
            fn("unlisted-mode")
    assert all(recipe.yaw_variance(mode) == .5 for mode in recipe.MODES)

    calls = {name: [] for name in ("action", "linear", "head")}
    monkeypatch.setattr(head, "validate_action_row", lambda *args: calls["action"].append(args))
    monkeypatch.setattr(head, "validate_linear_row", lambda *args: calls["linear"].append(args))
    monkeypatch.setattr(head, "validate_head_row", lambda *args: calls["head"].append(args))
    for mode, (worlds, updates, _seed) in recipe.MODES.items():
        expected = recipe.reference_mode(mode)
        for fn, key in ((recipe.validate_action_row, "action"),
                        (recipe.validate_linear_row, "linear"),
                        (recipe.validate_head_row, "head")):
            row = {"mode": mode}
            fn(row, mode, worlds, updates)
            assert calls[key][-1] == (row, expected, worlds, updates)

    forwarded = []
    monkeypatch.setattr(recipe.base, "evaluate_case", lambda *args, **kwargs: forwarded.append((args, kwargs)) or {"ok": True})
    result = recipe.evaluate_case("checkpoint", 3, .3, .1, "healthy")
    assert result == {"ok": True}
    assert forwarded[0][0] == ("checkpoint", 3, .3, .1, "healthy")
    assert forwarded[0][1] == {"diagnostics": True, "yaw_trace": True, "gait_trace": True}


def _observer_env(mode="smoke-gentleturn"):
    from test_integrated_motor_range_head_lesson import _head_observer_env

    env = _head_observer_env(recipe.reference_mode(mode))
    cfg, _ = recipe.prepare_config(mode)
    twist_cfg = copy.deepcopy(cfg.commands["twist"])
    command = torch.zeros(env.num_envs, 3)
    command[:, 0] = .3
    command[0] = 0
    command[1, 2] = -.1
    command[2, 2] = -.2
    command[3, 2] = .1
    command[4, 2] = .2
    actor = torch.zeros(env.num_envs, 61)
    actor[:, 48:51] = command
    original_get_term = env.command_manager.get_term
    original_get_command = env.command_manager.get_command
    env.command_manager.get_term = lambda name: NS(cfg=twist_cfg) if name == "twist" else original_get_term(name)
    env.command_manager.get_command = lambda name: command if name == "twist" else original_get_command(name)
    env.observation_manager = NS(_obs_buffer={"actor": actor})
    env._gentle_command, env._gentle_actor, env._gentle_cfg = command, actor, twist_cfg
    return env


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-gentleturn"])
def test_full_observer_chain_records_one_command_exposure_stream_for_24_steps(monkeypatch, mode):
    from test_integrated_motor_range_head_lesson import _head_values
    from test_integrated_motor_range_action_release import _prepare_parent_metrics

    env = _observer_env(mode)
    observer = recipe.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(env.num_envs, 14), mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *_args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))
    for step in range(24):
        env._head_command_holder["head"] = _head_values(step, env.num_envs)
        observer.before(env)
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
        rewards = env.reward_manager._step_reward
        rewards[:, env.reward_manager.active_terms.index("head_pose_tracking")] = .8
        rewards[:, env.reward_manager.active_terms.index("head_pose_bias")] = -.15
        observer.step(env, sample, {})
        assert observer.exposure_steps == step + 1
    row = observer.update(env)
    assert row["training_yaw_support"] == list(recipe.yaw_support(mode))
    assert row["command_exposure_protocol"] == "actor-command-dwell-v1"
    assert row["command_exposure_steps"] == 24
    assert row["curriculum"]["rel_standing_envs"] == .25
    assert sum(row["command_exposure_counts"].values()) == 1536
    assert sum(row["command_exposure_cumulative"].values()) == 1536
    assert row["command_exposure_counts"] == row["command_exposure_cumulative"]
    assert observer.exposure_steps == 0
    assert row["command_exposure_counts"]["negative-low"] == 24
    assert row["command_exposure_counts"]["negative-target"] == 24
    assert row["command_exposure_counts"]["positive-low"] == 24
    assert row["command_exposure_counts"]["positive-target"] == 24
    if mode.endswith("gentleturn"):
        assert row["command_exposure_counts"]["negative-high"] == 0
        assert row["command_exposure_counts"]["positive-high"] == 0


@pytest.mark.parametrize("fault", ["yaw", "standing", "heading", "heading-range", "rel-heading",
                                    "rel-world", "rel-forward", "resampling", "turn-in-place", "actor", "command"])
def test_observer_before_rejects_live_twist_or_cached_actor_drift(fault):
    env = _observer_env()
    observer = recipe.MotorObserver("smoke-gentleturn")
    if fault == "yaw":
        env._gentle_cfg.ranges.ang_vel_z = (-.5, .5)
    elif fault == "standing":
        env._gentle_cfg.rel_standing_envs = .15
    elif fault == "heading":
        env._gentle_cfg.heading_command = True
    elif fault == "heading-range":
        env._gentle_cfg.ranges.heading = (-3.14, 3.14)
    elif fault == "rel-heading":
        env._gentle_cfg.rel_heading_envs = .1
    elif fault == "rel-world":
        env._gentle_cfg.rel_world_envs = .1
    elif fault == "rel-forward":
        env._gentle_cfg.rel_forward_envs = .5
    elif fault == "resampling":
        env._gentle_cfg.resampling_time_range = (2., 4.)
    elif fault == "turn-in-place":
        env._gentle_cfg.rel_turn_in_place_envs = .1
    elif fault == "actor":
        env._gentle_actor[0, 48] = .1
    else:
        env._gentle_command[0, 2] = .3
    with pytest.raises(ValueError):
        observer.before(env)
    assert observer.pending is None
    assert observer.exposure_steps == 0
    assert not any(observer.exposure_counts.values())
    assert not any(observer.cumulative_exposure.values())


def test_support_and_stock_angular_row_validators_fail_closed():
    mode, worlds, update = "smoke-gentleturn", 64, 1
    row = dict(training_yaw_support=[-.25, .25], command_exposure_protocol="actor-command-dwell-v1",
               command_exposure_steps=24, curriculum={"rel_standing_envs": .25},
               command_exposure_counts=dict.fromkeys(recipe.EXPOSURE_BINS, 0),
               command_exposure_cumulative=dict.fromkeys(recipe.EXPOSURE_BINS, 0))
    row["command_exposure_counts"]["idle"] = worlds * 24
    row["command_exposure_cumulative"]["idle"] = worlds * 24
    recipe.validate_support_row(row, mode, worlds, update)
    row["command_exposure_counts"]["negative-high"] = 1
    row["command_exposure_counts"]["idle"] -= 1
    with pytest.raises(ValueError, match="narrow support excludes high-yaw"):
        recipe.validate_support_row(row, mode, worlds, update)
    row["command_exposure_counts"]["negative-high"] = 0
    row["command_exposure_counts"]["idle"] += 1
    row["command_exposure_cumulative"]["idle"] -= 1
    row["command_exposure_cumulative"]["straight"] += 1
    with pytest.raises(ValueError, match="complete cumulative command exposure"):
        recipe.validate_support_row(row, mode, worlds, update)
    row["command_exposure_cumulative"]["idle"] += 1
    row["command_exposure_cumulative"]["straight"] -= 1
    row["command_exposure_cumulative"]["idle"] -= 1
    with pytest.raises(ValueError, match="complete cumulative command exposure"):
        recipe.validate_support_row(row, mode, worlds, update)
    row["command_exposure_cumulative"]["idle"] += 1
    row["command_exposure_protocol"] = "tampered"
    with pytest.raises(ValueError, match="complete exclusive per-update command exposure"):
        recipe.validate_support_row(row, mode, worlds, update)

    angular = dict(yaw_variance=.5, roll_pitch_variance=.5, angular_tracking_weight=2.,
                   angular_aggregation="stock_multiplicative_product", roll_pitch_tracking_included=True,
                   angular_reward_sample_count=worlds * 24, negative_reward_sample_count=0,
                   negative_reward_fraction=0., weighted_angular_reward_mean=1.1,
                   yaw_error_sq_mean=.2, roll_pitch_rate_sq_mean=.3, body_ang_vel_weight=-.05,
                   angular_momentum_weight=-.02, training_yaw_support=[-.25, .25])
    recipe.validate_yaw_row(angular, mode, worlds, update)
    angular["roll_pitch_tracking_included"] = False
    with pytest.raises(ValueError, match="unchanged stock angular"):
        recipe.validate_yaw_row(angular, mode, worlds, update)


def test_control_support_allows_high_bins_but_still_requires_complete_integer_counts():
    worlds, update = 64, 1
    bins = dict.fromkeys(recipe.EXPOSURE_BINS, 0)
    bins["negative-high"], bins["positive-high"] = 11, 13
    bins["straight"] = worlds * 24 - 24
    cumulative = dict(bins)
    row = dict(training_yaw_support=[-.5, .5], command_exposure_protocol="actor-command-dwell-v1",
               command_exposure_steps=24, curriculum={"rel_standing_envs": .25},
               command_exposure_counts=bins, command_exposure_cumulative=cumulative)
    recipe.validate_support_row(row, "smoke-control", worlds, update)
    row["curriculum"]["rel_standing_envs"] = .15
    with pytest.raises(ValueError, match="complete exclusive per-update command exposure"):
        recipe.validate_support_row(row, "smoke-control", worlds, update)
    row["curriculum"]["rel_standing_envs"] = .25
    row["command_exposure_counts"]["idle"] = True
    with pytest.raises(ValueError, match="complete exclusive per-update command exposure"):
        recipe.validate_support_row(row, "smoke-control", worlds, update)
