from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from mjlab_microduck import integrated_fresh_motor_range as safety
from mjlab_microduck import integrated_motor_range_forward as recipe
from mjlab_microduck import integrated_motor_refinement as motor


MODES = tuple(recipe.MODES)


def test_six_declared_modes_and_only_forward_arm_weight_differs():
    assert MODES == (
        "smoke-control", "smoke-forward", "benchmark-control", "benchmark-forward",
        "control", "forward",
    )
    assert recipe.MODES["smoke-control"] == (64, 5, 1381)
    assert recipe.MODES["smoke-forward"] == recipe.MODES["smoke-control"]
    assert recipe.MODES["benchmark-control"] == (256, 10, 1399)
    assert recipe.MODES["benchmark-forward"] == recipe.MODES["benchmark-control"]
    assert recipe.MODES["control"] == recipe.MODES["forward"] == (256, 1000, 1409)

    for control_mode, forward_mode in (
        ("smoke-control", "smoke-forward"),
        ("benchmark-control", "benchmark-forward"),
        ("control", "forward"),
    ):
        control, control_agent = recipe.prepare_config(control_mode)
        forward, forward_agent = recipe.prepare_config(forward_mode)
        assert recipe.forward_weight(control_mode) == 0.
        assert recipe.forward_weight(forward_mode) == -.25
        assert control.rewards["forward_error"].weight == 0.
        assert forward.rewards["forward_error"].weight == -.25
        control.rewards["forward_error"].weight = forward.rewards["forward_error"].weight
        assert repr(control) == repr(forward)
        control_agent.run_name = forward_agent.run_name
        assert control_agent == forward_agent


@pytest.mark.parametrize("mode", MODES)
def test_each_config_pins_final_parent_objectives_and_preserves_policy_interface(mode):
    cfg, agent = recipe.prepare_config(mode)
    stock, _ = safety.prepare_config("benchmark")

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
    assert "forward_error" in cfg.rewards
    assert cfg.rewards["forward_error"].weight == recipe.forward_weight(mode)
    assert "command_error" not in cfg.rewards
    assert cfg.observations == stock.observations
    assert cfg.actions == stock.actions
    assert cfg.actions["joint_pos"].clip is None

    metric = cfg.metrics["motor_range_forward_cost"]
    assert metric.func is recipe.forward.capture_forward_metric
    assert not metric.per_substep
    worlds, updates, seed = recipe.MODES[mode]
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
    assert cfg.seed == agent.seed == seed
    assert agent.max_iterations == updates and agent.save_interval == 250


def test_final_checkpoint_and_unchanged_twelve_case_gate_are_reused():
    assert recipe.evaluate_case is safety.evaluate_case
    assert recipe.decision is safety.decision
    assert recipe.base.SEEDS == safety.EVAL_SEEDS
    assert recipe.base.CASES == safety.EVAL_CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: contract["final_iteration"] for arm, contract in plan["contracts"].items()} == {
        "control": 999, "forward": 999,
    }
    assert all(contract["worlds"] == 256 and contract["updates"] == 1000
               for contract in plan["contracts"].values())


def test_parent_curriculum_freeze_rejects_unknown_or_unreached_stages():
    cfg, _ = safety.prepare_config("benchmark")
    cfg.curriculum["unknown"] = next(iter(cfg.curriculum.values()))
    with pytest.raises(ValueError, match="known seven parent curricula"):
        recipe.freeze_parent_curricula(cfg)

    cfg, _ = safety.prepare_config("benchmark")
    standing = cfg.curriculum["standing_envs"]
    standing.params["standing_stages"].append(
        {"step": recipe.PARENT_STEPS, "rel_standing_envs": .3},
    )
    with pytest.raises(ValueError, match="parent reached every ordered source stage"):
        recipe.freeze_parent_curricula(cfg)


class _Normalizer(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.register_buffer("count", torch.tensor(count, dtype=torch.int64))
        self.register_buffer("_mean", torch.zeros(61))
        self.register_buffer("_var", torch.ones(61))
        self.register_buffer("_std", torch.ones(61))


class _NormalizedModel(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.normalizer = _Normalizer(count)


def fake_runner(*, actor_count=recipe.PARENT_NORMALIZER_COUNT,
                critic_count=recipe.PARENT_NORMALIZER_COUNT,
                adam_count=17, adam_step=recipe.PARENT_ADAM_STEP):
    states = {
        index: {"step": torch.tensor(float(adam_step))}
        for index in range(adam_count)
    }
    return NS(alg=NS(
        actor=_NormalizedModel(actor_count),
        critic=_NormalizedModel(critic_count),
        optimizer=NS(state=states),
    ))


def test_parent_warm_load_pins_iteration_step_restored_counts_and_kl(monkeypatch):
    runner, env = fake_runner(), NS(device="cpu")
    warm_calls, kl_calls = [], []
    receipt = {"learning_rate": 3.25e-5, "parent_sha256": recipe.PARENT_SHA}

    def warm(*args, **kwargs):
        warm_calls.append((args, kwargs))
        return dict(receipt)

    monkeypatch.setattr(recipe.motor, "warm_load", warm)
    monkeypatch.setattr(recipe.safety.fresh.kl, "install_optimizer_capture",
                        lambda *args, **kwargs: kl_calls.append((args, kwargs)))
    result = motor.initialize_continuation(recipe, runner, env, "control")

    assert warm_calls == [((runner, env, recipe.PARENT), dict(
        parent_sha=recipe.PARENT_SHA,
        parent_iteration=5999, parent_steps=144000,
    ))]
    assert result == {
        **receipt,
        "actor_normalizer_count": 36864000,
        "critic_normalizer_count": 36864000,
        "adam_state_entries": 17,
        "adam_step": 120000,
    }
    assert kl_calls == [((runner, env), {"initial_lr": receipt["learning_rate"]})]


@pytest.mark.parametrize(
    "runner,expected",
    [
        (fake_runner(actor_count=36863999), "exact restored actor normalizer count"),
        (fake_runner(critic_count=36863999), "exact restored critic normalizer count"),
        (fake_runner(adam_count=16), "exact restored seventeen Adam counters"),
        (fake_runner(adam_step=119999), "exact restored seventeen Adam counters"),
    ],
)
def test_restored_parent_counters_fail_closed(runner, expected):
    with pytest.raises(ValueError, match=expected):
        recipe.restored_counters(runner)


def _fake_observer_env():
    qpos = torch.zeros(64, 3)
    qpos[0, 2] = .5
    qpos[1, 1] = .55
    limits = torch.tensor([[[-.5, .5]] * 3]).repeat(64, 1, 1)
    selector = NS(name="robot", joint_ids=[1, 2])
    terms = {name: NS(weight=0.) for name in (
        "motor_load", "joint_limit", "forward_error", "action_rate_l2", "head_pose_bias",
    )}
    terms["joint_limit"].params = dict(asset_cfg=selector, margin=.15)
    terms["forward_error"].params = {"command_name": "twist"}
    head = NS(ranges=((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-.31, .31)))
    command_cfg = NS(rel_standing_envs=.25)
    command = torch.zeros(64, 3)
    command[1:, 0] = .3
    command[2, 2], command[3, 2] = -.2, .2
    robot = NS(joint_names=["passive_dummy", "right_hip_yaw", "left_hip_yaw"],
               data=NS(joint_pos=qpos, joint_pos_limits=limits))
    env = NS(
        num_envs=64, device="cpu", common_step_counter=0,
        scene={"robot": robot},
        command_manager=NS(
            get_command=lambda name: command,
            get_term=lambda name: NS(cfg=head if name == "head_pose" else command_cfg),
        ),
        reward_manager=NS(
            active_terms=["motor_load", "joint_limit", "forward_error"],
            get_term_cfg=lambda name: terms[name],
            _step_reward=torch.zeros(64, 3),
        ),
        event_manager=NS(get_term_cfg=lambda name: NS(params={"ranges": (-.01, .01)})),
        _microduck_optimizer_audit=object(),
    )
    return env


@pytest.mark.parametrize("mode,expected_forward_weight", [
    ("smoke-control", 0.), ("smoke-forward", -.25),
])
def test_observer_reconciles_prefreeze_costs_and_logs_capped_motor_reward(
        mode, expected_forward_weight, monkeypatch):
    from mjlab_microduck.tasks import mdp

    env = _fake_observer_env()
    observer = recipe.MotorObserver(mode)
    monkeypatch.setattr(recipe.safety.fresh.kl, "optimizer_summary",
                        lambda *args: {"optimizer_minibatches": 20})
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda env, **params: torch.full((64,), .4))

    original_qpos = env.scene["robot"].data.joint_pos.clone()
    sample = NS(force_nm=torch.full((64, 14), 1.8),
                mean_cost=torch.full((64,), 30.16))
    for _ in range(24):
        observer.before(env)
        safety.capture_limit_metric(env)
        recipe.forward.capture_forward_metric(env)
        limit_cost = env._microduck_fresh_limit_snapshot[0]
        env.reward_manager._step_reward[:, 0] = -10.76
        env.reward_manager._step_reward[:, 1] = limit_cost * -4.
        env.reward_manager._step_reward[:, 2] = .4 * expected_forward_weight
        # Simulate autoreset after both metric callbacks and before telemetry.
        env.scene["robot"].data.joint_pos.zero_()
        env.common_step_counter += 1
        observer.step(env, sample, {})
        assert env._microduck_fresh_limit_snapshot is None
        assert env._microduck_forward_cost is None
        env.scene["robot"].data.joint_pos.copy_(original_qpos)

    row = observer.update(env)
    assert row["motor_weight_min"] == row["motor_weight_max"] == -1.
    assert row["limit_weight_min"] == row["limit_weight_max"] == -4.
    assert row["forward_weight_min"] == row["forward_weight_max"] == expected_forward_weight
    assert row["motor_cost_mean"] == pytest.approx(10.76)
    assert row["weighted_motor_reward_mean"] == pytest.approx(-10.76)
    assert row["limit_cost_mean"] == pytest.approx(.35 / 64)
    assert row["weighted_limit_reward_mean"] == pytest.approx(-4. * .35 / 64)
    assert row["hip_limit_cost_mean"] == pytest.approx(
        dict(left_hip_yaw=.15 / 64, right_hip_yaw=.2 / 64),
    )
    assert row["forward_cost_mean"] == pytest.approx(.4)
    assert row["weighted_forward_reward_mean"] == pytest.approx(.4 * expected_forward_weight)
    assert not observer.forward_costs and not observer.forward_rewards


@pytest.mark.parametrize("fault,message", [
    ("missing", "finite pre-reset forward cost"),
    ("wrong-weight", "forward reward matches pre-reset cost"),
])
def test_observer_rejects_missing_or_mismatched_forward_snapshot(fault, message, monkeypatch):
    from mjlab_microduck.tasks import mdp

    env = _fake_observer_env()
    observer = recipe.MotorObserver("smoke-forward")
    monkeypatch.setattr(mdp, "body_forward_tracking_cost",
                        lambda env, **params: torch.full((64,), .4))
    observer.before(env)
    safety.capture_limit_metric(env)
    limit_cost = env._microduck_fresh_limit_snapshot[0]
    env.reward_manager._step_reward[:, 0] = -10.76
    env.reward_manager._step_reward[:, 1] = limit_cost * -4.
    if fault == "wrong-weight":
        recipe.forward.capture_forward_metric(env)
    env.reward_manager._step_reward[:, 2] = 0.
    env.common_step_counter += 1
    sample = NS(force_nm=torch.full((64, 14), 1.8),
                mean_cost=torch.full((64,), 30.16))
    with pytest.raises(ValueError, match=message):
        observer.step(env, sample, {})
