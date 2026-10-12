import copy
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from mjlab_microduck import integrated_fresh_foundation as fresh
from mjlab_microduck import integrated_kl_budget_refinement as kl
from mjlab_microduck import integrated_turn_foundation as original
from mjlab.tasks.registry import load_env_cfg


def _curriculum_boundaries(cfg):
    return {
        name: {key: tuple(stage["step"] for stage in value)
               for key, value in term.params.items()
               if key.endswith("_stages") and isinstance(value, list)
               and value and all(isinstance(stage, dict) and "step" in stage for stage in value)}
        for name, term in cfg.curriculum.items()
    }


def test_fresh_recipe_changes_only_declared_world_budget_seed_and_run_labels():
    stock, stock_agent = original.prepare_config("foundation")
    for mode, (worlds, updates, seed) in fresh.MODES.items():
        cfg, agent = fresh.prepare_config(mode)
        normalized = copy.deepcopy(cfg)
        normalized.seed = stock.seed
        normalized.scene.num_envs = normalized.scene.terrain.num_envs = stock.scene.num_envs
        agent_normalized = copy.deepcopy(agent)
        agent_normalized.seed = stock_agent.seed
        agent_normalized.max_iterations = stock_agent.max_iterations
        agent_normalized.experiment_name = stock_agent.experiment_name
        agent_normalized.run_name = stock_agent.run_name
        assert normalized == stock
        assert agent_normalized == stock_agent
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed
        assert agent.max_iterations == updates and agent.save_interval == 250
        assert agent.experiment_name == fresh.PROTOCOL and agent.run_name == mode
        assert cfg.commands["twist"].rel_turn_in_place_envs == 0.
        assert cfg.commands["twist"].ranges.lin_vel_x == (.20, .40)
        assert cfg.commands["twist"].ranges.lin_vel_y == (0., 0.)
        assert cfg.commands["twist"].ranges.ang_vel_z == (-.50, .50)
        assert cfg.commands["twist"].rel_standing_envs == .02
        assert cfg.actions["joint_pos"].clip is None
        assert not any(name in cfg.rewards for name in ("motor_load", "joint_limit"))


def test_full_budget_crosses_every_unchanged_stock_curriculum_endpoint():
    original_cfg = load_env_cfg(original.TASK)
    cfg, _ = fresh.prepare_config("foundation")
    expected = _curriculum_boundaries(original_cfg)
    assert _curriculum_boundaries(cfg) == expected
    assert expected["action_rate_weight"]["weight_stages"][-1] == 36_000
    assert expected["standing_envs"]["standing_stages"][-1] == 48_000
    assert expected["head_pose_range"]["range_stages"][-1] == 48_000
    assert expected["com_range"]["range_stages"][-1] == 36_000
    assert expected["head_com_range"]["range_stages"][-1] == 24_000
    assert expected["head_pose_bias_weight"]["weight_stages"][-1] == 36_000
    assert fresh.MODES["foundation"][1] * 24 == 144_000


def test_declared_single_fresh_mode_budget_eval_and_projection_cap():
    assert fresh.evaluate_case is fresh.straight.evaluate_case
    assert fresh.decision is fresh.budget.decision
    assert fresh.MODES == {"smoke": (64, 5, 821), "benchmark": (256, 10, 823),
                           "foundation": (256, 6000, 1327)}
    assert fresh.FOUNDATION_CAP_SECONDS == 9000
    assert fresh.FOUNDATION_PROJECTION_LIMIT_SECONDS == 8940
    assert fresh.REQUIRED_COVERAGE == ("idle", "straight", "negative_yaw", "positive_yaw")
    assert fresh.EVAL_SEEDS == original.SEEDS == (839, 853, 857)
    assert fresh.EVAL_CASES == original.CASES
    assert fresh.FINAL_CHECKPOINT == 5999
    assert fresh.benchmark_projection_seconds(9.698) == pytest.approx(8728.2)
    assert fresh.projection_admitted(9.698)
    assert not fresh.projection_admitted(10.)
    for invalid in (True, float("nan"), float("inf"), 0., -1., "9.698"):
        with pytest.raises(ValueError):
            fresh.projection_admitted(invalid)


class FakeNorm(nn.Module):
    def __init__(self, count=0):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, 2))
        self.register_buffer("_var", torch.ones(1, 2))
        self.register_buffer("_std", torch.ones(1, 2))
        self.register_buffer("count", torch.tensor(count, dtype=torch.long))


class FakeModel(nn.Module):
    def __init__(self, count=0):
        super().__init__()
        self.obs_normalizer = FakeNorm(count)
        self.linear = nn.Linear(2, 2)


def fake_runner():
    actor, critic = FakeModel(), FakeModel()
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    alg = NS(actor=actor, critic=critic, optimizer=optimizer, learning_rate=1e-3,
             schedule="adaptive", desired_kl=.01, num_learning_epochs=5, num_mini_batches=4)
    return NS(alg=alg, current_learning_iteration=0)


def test_fresh_cpu_runner_requires_empty_clocks_normalizers_adam_and_stock_lr(monkeypatch):
    calls = []
    monkeypatch.setattr(kl, "install_optimizer_capture",
                        lambda runner, env, *, initial_lr: calls.append(initial_lr))
    runner, env = fake_runner(), NS(common_step_counter=0)
    result = fresh.validate_fresh_runner(runner, env)
    assert result["initialized_from_checkpoint"] is False
    assert result["parent_checkpoint"] is None
    assert result["adam_state_entries"] == 0
    assert result["actor_normalizer_count"] == result["critic_normalizer_count"] == 0
    assert result["initial_learning_rate"] == .001
    assert calls == [.001]

    runner.current_learning_iteration = 1
    with pytest.raises(ValueError, match="iteration clock"):
        fresh.validate_fresh_runner(runner, env, install_capture=False)
    runner.current_learning_iteration = 0
    runner.alg.actor.obs_normalizer.count.fill_(1)
    with pytest.raises(ValueError, match="normalizer"):
        fresh.validate_fresh_runner(runner, env, install_capture=False)
    runner.alg.actor.obs_normalizer.count.zero_()
    runner.alg.actor.obs_normalizer.count = torch.tensor(.5)
    with pytest.raises(ValueError, match="integer observation normalizer"):
        fresh.validate_fresh_runner(runner, env, install_capture=False)
    runner.alg.actor.obs_normalizer.count = torch.tensor(0, dtype=torch.int64)
    runner.alg.optimizer.state[next(iter(runner.alg.optimizer.param_groups[0]["params"]))] = {}
    with pytest.raises(ValueError, match="Adam has no moments"):
        fresh.validate_fresh_runner(runner, env, install_capture=False)


def test_fresh_cpu_runner_rejects_nonzero_environment_clock_and_wrong_lr():
    with pytest.raises(ValueError, match="environment curriculum clock"):
        fresh.validate_fresh_runner(fake_runner(), NS(common_step_counter=1), install_capture=False)
    runner = fake_runner()
    runner.alg.optimizer.param_groups[0]["lr"] = .0005
    with pytest.raises(ValueError, match="stock initial learning rate"):
        fresh.validate_fresh_runner(runner, NS(common_step_counter=0), install_capture=False)


def test_installed_empirical_normalizer_has_zero_initial_count():
    from rsl_rl.modules import EmpiricalNormalization

    model = nn.Sequential(EmpiricalNormalization(2))
    assert fresh.normalizer_count(model) == 0
    model[0].update(torch.ones(3, 2))
    with pytest.raises(ValueError, match="fresh empty integer observation normalizer"):
        fresh.normalizer_count(model)


def test_command_exposure_observer_counts_consumed_commands_and_live_curricula(monkeypatch):
    observer = fresh.FreshObserver("smoke")
    command = torch.zeros(64, 3)
    command[1:, 0] = .3
    command[2, 2] = -.2
    command[3, 2] = .2
    consumed = command.clone()
    env = NS(num_envs=64, common_step_counter=0,
             command_manager=NS(get_command=lambda name: command,
                                get_term=lambda name: NS(cfg=NS(
                                    rel_standing_envs=.02, ranges=((-.05, .05),)))) )
    for _ in range(24):
        observer.before_action(env)
        command.zero_()  # A later mutation cannot rewrite what the actor consumed.
        env.common_step_counter += 1
        observer.after_step(env, None, {})
        command.copy_(consumed)
    env.reward_manager = NS(get_term_cfg=lambda name: NS(weight=-.1 if name == "action_rate_l2" else 0.))
    env.command_manager.get_term = lambda name: NS(cfg=NS(
        rel_standing_envs=.02, ranges=((-.05, .05),)))
    env.event_manager = NS(get_term_cfg=lambda name: NS(params={
        "ranges": (-.003, .003), "asset_cfg": object()}))
    env._microduck_optimizer_audit = object()
    monkeypatch.setattr(kl, "optimizer_summary", lambda env, expected: {"optimizer_minibatches": 20})
    row = observer.after_update(env)
    assert row["command_exposure_steps"] == 24
    assert row["command_exposure_counts"]["idle"] == 24
    assert row["command_exposure_counts"]["straight"] == 61 * 24
    assert row["command_exposure_counts"]["negative-target"] == 24
    assert row["command_exposure_counts"]["positive-target"] == 24
    assert row["curriculum"]["action_rate_weight"] == -.1
    assert row["curriculum"]["rel_standing_envs"] == .02
    assert row["optimizer"]["optimizer_minibatches"] == 20
    assert observer.cumulative_exposure["idle"] == 24
    assert observer.cumulative_exposure["straight"] == 61 * 24


def test_curriculum_profile_serializes_live_stock_cfg_without_opaque_selectors():
    cfg = load_env_cfg(original.TASK)
    env = NS(common_step_counter=0,
        reward_manager=NS(get_term_cfg=lambda name: cfg.rewards[name]),
        command_manager=NS(get_term=lambda name: NS(cfg=cfg.commands[name])),
        event_manager=NS(get_term_cfg=lambda name: cfg.events[name]))
    profile = fresh.FreshObserver.curriculum_profile(env)
    import json
    encoded = json.dumps(profile, allow_nan=False)
    assert '"randomize_com": {"ranges": [-0.003, 0.003]' in encoded
    assert '"randomize_head_com": {"ranges": [-0.003, 0.003]' in encoded


def _receipt(tmp_path, mode="smoke"):
    worlds, updates, seed = fresh.MODES[mode]
    output = tmp_path / mode
    output.mkdir()
    checkpoint = output / f"model_{updates-1}.pt"
    checkpoint.write_bytes(b"durable final checkpoint")
    initial_checkpoint = output / "initial.pt"
    initial_checkpoint.write_bytes(b"fresh initial checkpoint")
    initialization = dict(protocol="fresh-policy-initialization-v1",
        initialized_from_checkpoint=False, parent_checkpoint=None,
        learner_iteration=0, common_step_counter=0, adam_state_entries=0,
        actor_normalizer_count=0, critic_normalizer_count=0,
        initial_learning_rate=.001, learning_rate_schedule="adaptive", desired_kl=.01,
        optimizer_capture="read-only-raw-kl-telemetry",
        initial_checkpoint=str(initial_checkpoint),
        initial_checkpoint_sha256=original.sha256(initial_checkpoint), source="a" * 40)
    (output / "fresh-initialization.json").write_text(__import__("json").dumps(initialization))
    dwell = dict.fromkeys(fresh.budget.EXPOSURE_BINS, 0)
    dwell.update(idle=1, straight=1, **{"negative-target": 1, "positive-target": 1})
    dwell["positive-high"] += worlds * updates * 24 - 4
    exposure = dict(protocol="actor-command-dwell-v1", steps=updates * 24,
        transitions=worlds * updates * 24, counts=dwell,
        required_counts=dict(idle=1, straight=1, negative_yaw=1,
                             positive_yaw=dwell["positive-target"] + dwell["positive-high"]),
        required_coverage=list(fresh.REQUIRED_COVERAGE))
    receipt = dict(protocol=fresh.PROTOCOL, source="a" * 40,
        status="training-complete-not-accepted", mode=mode, worlds=worlds, seed=seed,
        updates=updates, steps=updates * 24, transitions=worlds * updates * 24,
        required_command_coverage=list(fresh.REQUIRED_COVERAGE),
        idle=1, straight=1, negative_yaw=1,
        positive_yaw=exposure["required_counts"]["positive_yaw"],
        elapsed_s=9.698, checkpoint=str(checkpoint),
        checkpoint_sha256=original.sha256(checkpoint), fresh_initialization=initialization,
        command_exposure=exposure)
    if mode == "smoke":
        onnx = output / "smoke.onnx"
        onnx.write_bytes(b"normalized onnx")
        receipt["onnx"] = dict(path=str(onnx), sha256=original.sha256(onnx),
                               input=[1, 61], output=[1, 14], finite=True)
    return receipt


def test_training_receipt_validates_exact_source_budget_checkpoint_and_smoke_onnx(tmp_path):
    receipt = _receipt(tmp_path)
    contract = fresh.validate_training_receipt(tmp_path, "smoke", "a" * 40, receipt)
    assert contract["transitions"] == 64 * 5 * 24
    for key, value in (("source", "b" * 40), ("updates", 6), ("seed", 999),
                       ("checkpoint_sha256", "0" * 64), ("idle", 0)):
        changed = dict(receipt)
        changed[key] = value
        with pytest.raises(ValueError):
            fresh.validate_training_receipt(tmp_path, "smoke", "a" * 40, changed)
    changed = dict(receipt)
    changed["onnx"] = dict(receipt["onnx"], output=[1, 61])
    with pytest.raises(ValueError):
        fresh.validate_training_receipt(tmp_path, "smoke", "a" * 40, changed)


def test_foundation_evaluation_receipt_admits_only_final5999(tmp_path):
    receipt = _receipt(tmp_path, "foundation")
    final = tmp_path / "foundation" / "model_5999.pt"
    assert fresh.validate_evaluation_receipt(tmp_path, "a" * 40, receipt) == final
    for key, value in (("updates", 1500), ("seed", 827),
                       ("checkpoint", str(tmp_path / "foundation" / "model_750.pt")),
                       ("transitions", 9_216_000)):
        changed = dict(receipt)
        changed[key] = value
        with pytest.raises(ValueError):
            fresh.validate_evaluation_receipt(tmp_path, "a" * 40, changed)


def test_evaluation_updates_gate_result_without_duplicate_keyword_error(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path, "foundation")
    root_result = tmp_path / "foundation" / "result.json"
    root_result.write_text(__import__("json").dumps(receipt))
    output = tmp_path / "evaluate"
    output.mkdir()
    rows = [dict(seed=seed, speed=speed, yaw=yaw)
            for seed in fresh.EVAL_SEEDS for speed, yaw in fresh.EVAL_CASES]
    calls = iter(rows)
    monkeypatch.setattr(fresh, "evaluate_case", lambda *args: next(calls))
    monkeypatch.setattr(fresh, "decision", lambda cases: {
        "decision": "foundation-not-ready", "policy_acceptance": False,
        "simulator_qualified": False, "physical_motion_authorized": False})
    result = fresh.evaluate(tmp_path, output, "a" * 40, lambda: {})
    assert result["decision"] == "foundation-not-ready"
    assert len(result["cases"]) == 12
    assert result["final_iteration"] == 5999
    assert result["selection"] == "final5999 only; no checkpoint fishing"
