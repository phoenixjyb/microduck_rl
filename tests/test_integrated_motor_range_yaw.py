from types import SimpleNamespace as NS
import math

import pytest
import torch

from mjlab_microduck import integrated_motor_range_forward as previous
from mjlab_microduck import integrated_motor_range_yaw as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity


def _paired_configs(control_mode, treatment_mode):
    control, control_agent = recipe.prepare_config(control_mode)
    treatment, treatment_agent = recipe.prepare_config(treatment_mode)
    cterm = control.rewards[recipe.TERM]
    tterm = treatment.rewards[recipe.TERM]
    assert cterm.func is track_angular_velocity
    assert cterm.weight == tterm.weight == recipe.WEIGHT == 2.
    assert cterm.params == dict(command_name="twist", std=math.sqrt(.5))
    assert tterm.func is mdp.decoupled_body_angular_tracking
    assert tterm.params == dict(command_name="twist", yaw_variance=.5,
                                roll_pitch_variance=.5)
    assert control.metrics["motor_range_angular_snapshot"].func is recipe.capture_angular_metric
    assert treatment.metrics["motor_range_angular_snapshot"].func is recipe.capture_angular_metric
    assert control.metrics["motor_range_angular_snapshot"].per_substep is False
    assert treatment.metrics["motor_range_angular_snapshot"].per_substep is False

    # After replacing only the treatment callable and its parameterization,
    # the full environment and learner configs must match.
    tterm.func, tterm.params = cterm.func, dict(cterm.params)
    treatment_agent.run_name = control_agent.run_name
    assert repr(control) == repr(treatment)
    assert control_agent == treatment_agent
    return control, treatment


def test_six_modes_are_matched_and_only_angular_aggregation_changes():
    assert tuple(recipe.MODES) == (
        "smoke-control", "smoke-decoupled", "benchmark-control", "benchmark-decoupled",
        "control", "decoupled",
    )
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1423), "smoke-decoupled": (64, 5, 1423),
        "benchmark-control": (256, 10, 1427), "benchmark-decoupled": (256, 10, 1427),
        "control": (256, 1000, 1429), "decoupled": (256, 1000, 1429),
    }
    for control_mode, treatment_mode in (
        ("smoke-control", "smoke-decoupled"),
        ("benchmark-control", "benchmark-decoupled"),
        ("control", "decoupled"),
    ):
        _paired_configs(control_mode, treatment_mode)

    for mode, (worlds, updates, seed) in recipe.MODES.items():
        cfg, agent = recipe.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert recipe.decoupled(mode) == (mode == "decoupled" or mode.endswith("-decoupled"))


def test_parent_final_curriculum_safety_terms_interface_and_read_only_forward_metric():
    cfg, _ = recipe.prepare_config("benchmark-control")
    parent, _ = previous.prepare_config("benchmark-control")

    assert cfg.curriculum == {}
    assert cfg.rewards["action_rate_l2"].weight == -1.
    assert cfg.rewards["head_pose_bias"].weight == 3.
    assert cfg.commands["twist"].rel_standing_envs == .25
    assert cfg.commands["head_pose"].ranges == (
        (-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-.31, .31),
    )
    assert cfg.commands["body_pose"].ranges == ((-.005, .005),) * 3 + ((-.05, .05),) * 3
    assert cfg.events["randomize_com"].params["ranges"] == (-.015, .015)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-.01, .01)

    assert cfg.rewards["motor_load"].weight == -1.
    assert cfg.rewards["motor_load"].params == dict(
        rated_stall_torque_nm=.60, soft_limit_fraction=.70, over_limit_gain=4.,
    )
    assert cfg.rewards["joint_limit"].weight == -4.
    assert cfg.rewards["joint_limit"].params["margin"] == .15
    assert cfg.rewards["joint_limit"].params["asset_cfg"].joint_names == (
        "^(left|right)_hip_yaw$",
    )

    # Preserve the parent's inactive forward term and read-only metric as-is.
    assert cfg.rewards["forward_error"].weight == 0.
    assert cfg.rewards["forward_error"].func is parent.rewards["forward_error"].func
    assert cfg.rewards["forward_error"].params == parent.rewards["forward_error"].params
    assert cfg.metrics["motor_range_forward_cost"].func is parent.metrics["motor_range_forward_cost"].func
    assert cfg.metrics["motor_range_forward_cost"].per_substep is False

    assert cfg.observations == parent.observations
    assert cfg.actions == parent.actions
    assert cfg.actions["joint_pos"].clip is None
    assert cfg.commands == parent.commands
    assert cfg.events == parent.events


def test_final999_evaluation_contract_and_exact_parent_identity():
    assert recipe.PARENT == recipe.base.ROOT / (
        "artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt"
    )
    assert recipe.PARENT_SHA == "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    assert recipe.PARENT_NORMALIZER_COUNT == 43008000
    assert recipe.PARENT_ADAM_STEP == 140000
    assert recipe.evaluate_case is previous.evaluate_case
    assert recipe.decision is previous.decision
    assert recipe.base.SEEDS == previous.base.SEEDS
    assert recipe.base.CASES == previous.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12

    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "decoupled": 999,
    }
    assert all(item["worlds"] == 256 and item["updates"] == 1000
               for item in plan["contracts"].values())


def test_warm_load_uses_parent_counters_and_carries_parent_lr_into_kl(monkeypatch):
    from test_integrated_motor_range_forward import fake_runner

    runner, env = fake_runner(actor_count=recipe.PARENT_NORMALIZER_COUNT,
                              critic_count=recipe.PARENT_NORMALIZER_COUNT,
                              adam_step=recipe.PARENT_ADAM_STEP), NS(device="cpu")
    warm_calls, kl_calls = [], []
    receipt = {"learning_rate": 3.25e-5, "parent_sha256": recipe.PARENT_SHA}

    def warm(*args, **kwargs):
        warm_calls.append((args, kwargs))
        return dict(receipt)

    monkeypatch.setattr(recipe.motor, "warm_load", warm)
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "install_optimizer_capture",
                        lambda *args, **kwargs: kl_calls.append((args, kwargs)))
    result = recipe.warm_load(runner, env)

    assert warm_calls == [((runner, env, recipe.PARENT), dict(
        parent_sha=recipe.PARENT_SHA, parent_iteration=999, parent_steps=24000,
    ))]
    assert result == {
        **receipt,
        "actor_normalizer_count": 43008000,
        "critic_normalizer_count": 43008000,
        "adam_state_entries": 17,
        "adam_step": 140000,
    }
    assert kl_calls == [((runner, env), {"initial_lr": receipt["learning_rate"]})]


def _angular_observer_env(mode):
    from test_integrated_motor_range_forward import _fake_observer_env

    env = _fake_observer_env()
    cfg, _ = recipe.prepare_config(mode)
    term = cfg.rewards[recipe.TERM]
    original_get_term_cfg = env.reward_manager.get_term_cfg
    terms = {name: env.reward_manager.get_term_cfg(name)
             for name in env.reward_manager.active_terms}
    terms[recipe.TERM] = NS(func=term.func, weight=term.weight, params=dict(term.params))
    env.reward_manager.active_terms.append(recipe.TERM)
    env.reward_manager.get_term_cfg = lambda name: (
        terms[name] if name in terms else original_get_term_cfg(name)
    )
    env.reward_manager._step_reward = torch.zeros(env.num_envs, len(env.reward_manager.active_terms))
    env.scene["robot"].data.root_link_ang_vel_b = torch.tensor([[1., 0., .5]]).repeat(env.num_envs, 1)
    env.command_manager.get_command = lambda _name: torch.zeros(env.num_envs, 3)
    env._microduck_yaw_aggregation_snapshot = None
    return env


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-decoupled"])
def test_24step_angular_observer_captures_actual_pre_reset_reward_and_negative_samples(
        monkeypatch, mode):
    env = _angular_observer_env(mode)
    observer = recipe.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(env.num_envs, 14),
                mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(recipe.previous.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))

    data = env.scene["robot"].data
    angular = data.root_link_ang_vel_b.clone()
    qpos = data.joint_pos.clone()
    rewards = []
    for _ in range(24):
        data.root_link_ang_vel_b.copy_(angular)
        data.joint_pos.copy_(qpos)
        observer.before(env)
        # Env.step advances its counter before its metric callback and autoreset.
        env.common_step_counter += 1
        recipe.previous.safety.capture_limit_metric(env)
        recipe.previous.forward.capture_forward_metric(env)
        env.reward_manager._step_reward[:, 0] = 0.
        limit = env._microduck_fresh_limit_snapshot[0]
        env.reward_manager._step_reward[:, 1] = limit * -4.
        env.reward_manager._step_reward[:, 2] = 0.
        term = env.reward_manager.get_term_cfg(recipe.TERM)
        index = env.reward_manager.active_terms.index(recipe.TERM)
        env.reward_manager._step_reward[:, index] = term.weight * term.func(env, **term.params)
        recipe.capture_angular_metric(env)
        rewards.append(env._microduck_yaw_aggregation_snapshot[2].clone())
        # Simulate autoreset after the reward and metric snapshots were made.
        data.root_link_ang_vel_b.zero_()
        data.joint_pos.zero_()
        observer.step(env, sample, {})
        assert env._microduck_yaw_aggregation_snapshot is None

    row = observer.update(env)
    assert row["angular_aggregation"] == (
        "stock_multiplicative_product" if mode.endswith("control") else "additive_decoupled"
    )
    assert row["yaw_variance"] == row["roll_pitch_variance"] == .5
    assert row["angular_tracking_weight"] == 2.
    assert row["yaw_error_sq_mean"] == pytest.approx(.25)
    assert row["roll_pitch_rate_sq_mean"] == pytest.approx(1.)
    assert row["angular_reward_sample_count"] == env.num_envs * 24
    assert all(torch.allclose(value, rewards[0]) for value in rewards)
    if mode.endswith("control"):
        assert bool((torch.stack(rewards) > 0.).all())
        assert row["negative_reward_sample_count"] == 0
    else:
        assert bool((torch.stack(rewards) < 0.).all())
        assert row["negative_reward_sample_count"] == env.num_envs * 24
        assert row["negative_reward_fraction"] == 1.


@pytest.mark.parametrize("fault", ["missing", "wrong-reward", "stale"])
def test_step_rejects_missing_wrong_or_stale_angular_snapshot(monkeypatch, fault):
    env = _angular_observer_env("smoke-decoupled")
    observer = recipe.MotorObserver("smoke-decoupled")
    sample = NS(force_nm=torch.zeros(env.num_envs, 14),
                mean_cost=torch.zeros(env.num_envs))
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda _env, **_params: torch.zeros(env.num_envs))
    observer.before(env)
    env.common_step_counter += 1
    recipe.previous.safety.capture_limit_metric(env)
    recipe.previous.forward.capture_forward_metric(env)
    env.reward_manager._step_reward[:, 0] = 0.
    env.reward_manager._step_reward[:, 1] = env._microduck_fresh_limit_snapshot[0] * -4.
    env.reward_manager._step_reward[:, 2] = 0.
    term = env.reward_manager.get_term_cfg(recipe.TERM)
    index = env.reward_manager.active_terms.index(recipe.TERM)
    env.reward_manager._step_reward[:, index] = term.weight * term.func(env, **term.params)
    if fault != "missing":
        recipe.capture_angular_metric(env)
    if fault == "wrong-reward":
        env._microduck_yaw_aggregation_snapshot[2].zero_()
    elif fault == "stale":
        snapshot = env._microduck_yaw_aggregation_snapshot
        env._microduck_yaw_aggregation_snapshot = (*snapshot[:3], snapshot[3] - 1)

    with pytest.raises(ValueError, match=(
        "same-step pre-reset angular snapshot" if fault == "stale"
        else "actual signed angular reward matches live aggregation" if fault == "wrong-reward"
        else "same-step pre-reset angular snapshot"
    )):
        observer.step(env, sample, {})


@pytest.mark.parametrize("fault", ["wrong-func", "wrong-weight", "wrong-params"])
def test_before_rejects_angular_reward_drift(fault):
    env = _angular_observer_env("smoke-decoupled")
    observer = recipe.MotorObserver("smoke-decoupled")
    term = env.reward_manager.get_term_cfg(recipe.TERM)
    if fault == "wrong-func":
        term.func = mdp.factorized_body_angular_tracking
    elif fault == "wrong-weight":
        term.weight = -2.
    else:
        term.params["yaw_variance"] = .25
    with pytest.raises(ValueError, match="fixed live angular aggregation"):
        observer.before(env)
