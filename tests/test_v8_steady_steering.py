"""CPU-only contract tests for the isolated steady-steering protocol."""
from __future__ import annotations

import json
import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_actor_learning as learning
from mjlab_microduck import v8_steady_steering as steady


SOURCE = "a" * 40


@pytest.mark.parametrize("mode,worlds,updates,seed", [
    ("smoke", 64, 5, 947), ("benchmark", 256, 10, 953),
    ("foundation", 256, 1000, 967),
])
def test_recipe_is_only_fixed_yaw_bias_term_beyond_cautious(mode, worlds, updates, seed):
    cfg, agent = steady.prepare_config(mode)
    cautious_cfg, cautious_agent = learning.prepare_config(mode, profile="cautious")
    proof = steady.recipe_match(mode)
    assert cfg.seed == agent.seed == seed
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
    assert agent.max_iterations == updates
    assert cfg.rewards[steady.TERM].weight == -1.0
    assert cfg.rewards[steady.TERM].params == {"command_name": "twist", "tau_s": 1.0}
    assert "microduck_yaw_bias" in cfg.metrics
    assert proof["matched_after_removing_owned_reward_and_metric"] is True
    assert proof["identity_masked_fields"] == ["experiment_name", "run_name"]
    assert proof["protocol"] == "v8-steady-steering-recipe-match-v2"
    assert proof["steady_recipe_sha256"] != proof["cautious_recipe_sha256"]
    assert cfg.commands == cautious_cfg.commands and cfg.curriculum == cautious_cfg.curriculum
    assert agent.algorithm.learning_rate == cautious_agent.algorithm.learning_rate == 1e-4


def test_metric_callable_recipe_identity_is_stable_across_import_context(monkeypatch):
    cfg, agent = steady.prepare_config("smoke")
    expected = hashlib.sha256(steady._canonical_config(cfg, agent).encode()).hexdigest()

    monkeypatch.setattr(steady.capture_bias_metric, "__module__", "__main__")
    changed_context = hashlib.sha256(steady._canonical_config(cfg, agent).encode()).hexdigest()
    assert changed_context == expected

    mismatched = copy.deepcopy(cfg)
    mismatched.metrics["microduck_yaw_bias"].func = lambda _env: None
    with pytest.raises(ValueError, match="known steady-steering metric callable"):
        steady._canonical_config(mismatched, agent)


class _RewardManager:
    def __init__(self, term, weighted):
        self.term = term
        self.active_terms = [steady.TERM]
        self._step_reward = weighted[:, None]

    def get_term_cfg(self, name):
        assert name == steady.TERM
        return self.term


class _Env:
    def __init__(self, command, step=4):
        self.common_step_counter = step
        self.num_envs = len(command)
        self.device = "cpu"
        self.command_manager = SimpleNamespace(get_command=lambda _name: command)


def test_metric_captures_actual_weighted_cost_without_advancing_cached_ema(monkeypatch):
    from mjlab_microduck.tasks import mdp
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager

    cost = torch.tensor([.1, .2, .3])
    ema_calls, advances = [], []
    def cached_cost(env, **_params):
        ema_calls.append(env.common_step_counter)
        state = getattr(env, "_microduck_body_yaw_bias_state", None)
        if state is None or state["step"] != env.common_step_counter:
            advances.append(env.common_step_counter)
        env._microduck_body_yaw_bias_state = {
            "step": env.common_step_counter, "signed_residual": torch.tensor([1., -2., 3.]),
            "signed_ema": torch.tensor([.1, -.2, .3]), "cost": cost.clone(),
        }
        return cost.clone()
    env = _Env(torch.zeros((3, 3)))
    env.max_episode_length_s = 20.
    monkeypatch.setattr(mdp, "body_yaw_bias_cost", cached_cost)
    manager = RewardManager({steady.TERM: RewardTermCfg(func=cached_cost, weight=-1.0,
        params={"command_name": "twist", "tau_s": 1.0})}, env, scale_by_dt=False)
    env.reward_manager = manager
    manager.compute(.02)

    metric = steady.capture_bias_metric(env)

    assert torch.equal(metric, torch.zeros(3))
    assert ema_calls == [4, 4]
    assert advances == [4]
    assert torch.equal(env._microduck_yaw_bias_snapshot["weighted_reward"], -cost)
    assert torch.equal(env._microduck_yaw_bias_snapshot["signed_residual"], torch.tensor([1., -2., 3.]))
    manager.reset(torch.arange(3))
    assert torch.equal(env._microduck_yaw_bias_snapshot["cost"], cost)


@pytest.mark.parametrize("weight,weighted", [(1.0, torch.tensor([.2])), (-1.0, torch.tensor([-.1]))])
def test_metric_refuses_wrong_live_weight_or_actual_reward(weight, weighted):
    from mjlab_microduck.tasks import mdp
    cost = torch.tensor([.1])
    env = _Env(torch.zeros((1, 3)))
    term = SimpleNamespace(func=mdp.body_yaw_bias_cost, weight=weight,
                           params={"command_name": "twist", "tau_s": 1.0})
    env.reward_manager = _RewardManager(term, weighted)
    env._microduck_body_yaw_bias_state = {"step": 4, "signed_residual": torch.zeros(1),
        "signed_ema": cost, "cost": cost}
    with pytest.raises(ValueError):
        steady.capture_bias_metric(env)


def _snapshot(residual, ema, weight=-1.0):
    ema = torch.tensor(ema, dtype=torch.float32)
    return {"signed_residual": torch.tensor(residual, dtype=torch.float32),
            "signed_ema": ema, "cost": ema.abs(), "weighted_reward": weight * ema.abs()}


def test_observer_consumes_exact_pre_reset_snapshot_once_and_covers_signed_buckets(tmp_path):
    from mjlab_microduck.tasks import mdp
    command = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [.3, 0., -.1],
        [.3, 0., -.2], [.3, 0., -.3], [.3, 0., .1], [.3, 0., .2], [.3, 0., .3]])
    env = _Env(command, step=10)
    env.reward_manager = _RewardManager(SimpleNamespace(func=mdp.body_yaw_bias_cost,
        weight=-1.0, params={"command_name": "twist", "tau_s": 1.0}), torch.zeros(8))
    observer = steady.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    observer.before(env)
    env._microduck_yaw_bias_snapshot = _snapshot([1.] * 8, [.1, .2, -.1, -.2, -.3, .1, .2, .3])
    env.common_step_counter = 11
    sample = SimpleNamespace(mean_cost=torch.zeros(8))

    observer.step(env, sample, {})

    assert env._microduck_yaw_bias_snapshot is None
    assert observer.pending is None
    assert sum(observer.bucket_counts.values()) == 8
    assert all(observer.bucket_counts[k] == 1 for k in steady.exposure.EXPOSURE_BINS)
    with pytest.raises(ValueError, match="one actual command per physics step"):
        observer.step(env, sample, {})


def test_failed_first_update_guard_writes_report_then_durable_abort_with_nested_clock(tmp_path, monkeypatch):
    events = []
    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
                "kl_mean_mean": .051, "kl_mean_max": .05}}
    monkeypatch.setattr(learning.Observer, "update", parent_update)
    observer = steady.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    observer.runner = SimpleNamespace(save=lambda path, infos: events.append((path, infos)))
    observer.dwell = dict.fromkeys(learning.exposure.EXPOSURE_BINS, 1)
    observer.bias_costs = [0.] * 24
    observer.bias_rewards = [0.] * 24
    observer.signed_biases = [0.] * 24
    observer.raw_errors = [0.] * 24
    observer.signed_raw_errors = [0.] * 24
    observer.bucket_counts = dict.fromkeys(learning.exposure.EXPOSURE_BINS, 24)
    observer.bucket_values = {key: {name: [] for name in
        ("raw_signed_residual", "signed_ema", "cost", "weighted_reward")}
        for key in learning.exposure.EXPOSURE_BINS}
    env = SimpleNamespace(common_step_counter=24, num_envs=8)

    with pytest.raises(ValueError, match="steady-steering first-update KL guard"):
        observer.update(env)

    report = json.loads((tmp_path / "first-update-kl.json").read_text())
    assert report["passed"] is False
    assert len(events) == 1
    assert events[0][1]["env_state"] == {"common_step_counter": 24}
    assert events[0][1]["first_update_kl_sha256"] == base.sha256(tmp_path / "first-update-kl.json")


def test_protocol_receipt_reserves_identity_fields():
    result = steady.assemble_receipt(SOURCE, "smoke", {
        "source": "bad", "protocol": "bad", "mode": "bad", "profile": "bad", "x": 1})
    assert result == {"source": SOURCE, "protocol": steady.PROTOCOL,
                      "mode": "smoke", "profile": "steady-steering", "x": 1}
    assert steady.assemble_receipt(SOURCE, "smoke", {"first_update_kl": {"passed": True}})[
        "first_update_kl"] == {"passed": True}


def test_evaluation_receipt_binds_compact_summary_and_raw_case_hashes(tmp_path):
    root = tmp_path
    output = root / "final-eval"
    output.mkdir()
    raw_files, summaries = [], []
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            name = f"s{seed}-c{index}.json"
            row = {"seed": seed, "speed": speed, "yaw": yaw, "complete": True,
                "worlds": 8, "steps": 240, "terminal_worlds": [], "checkpoint_sha256": "b"*64,
                "speed_mae": [0.] * 8, "yaw_mae": [0.] * 8,
                "response_diagnostics": {"startup_steps": 60},
                "body_yaw_trace": {"startup_steps": 60, "steps": 240, "worlds": 8,
                                    "checkpoint_sha256": "b"*64},
                "gait_yaw_trace": {"checkpoint_sha256": "b"*64},
                "gait_path_trace": {"checkpoint_sha256": "b"*64}}
            (output / name).write_text(json.dumps(row))
            raw_files.append({"file": name, "sha256": base.sha256(output / name)})
            summary = {"seed": seed, "speed": speed, "yaw": yaw, "compact": True}
            (output / f"s{seed}-c{index}-route.json").write_text(json.dumps(summary))
            summaries.append(summary)
    digest = "b" * 64
    receipt = steady.assemble_receipt(SOURCE, "final-eval", {
        "checkpoint_sha256": digest, "raw_case_files": raw_files, "cases": summaries,
        "route_gate": {"decision": "recorded"}, "legacy_gate": {"decision": "recorded"},
    })
    (output / "result.json").write_text(json.dumps(receipt))

    assert steady.validate_evaluation_receipt(root, "final-eval", SOURCE, digest) == receipt
    (output / "s853-c1.json").write_text("{}")
    with pytest.raises(ValueError, match="raw case hashes"):
        steady.validate_evaluation_receipt(root, "final-eval", SOURCE, digest)


def test_retention_gate_uses_the_sealed_helper_without_promoting(monkeypatch):
    from mjlab_microduck import steady_steering_retention as retention
    rows, baseline = [dict(seed=1)], {"baseline": "sealed"}
    events = []
    monkeypatch.setattr(retention, "load_baseline", lambda: (events.append("load") or baseline))
    monkeypatch.setattr(retention, "compare", lambda observed, control: (
        events.append((observed, control)) or {"decision": "walking-idle-retained",
            "policy_acceptance": False, "physical_motion_authorized": False}))

    report = steady.retention_gate(rows)

    assert events == ["load", (rows, baseline)]
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False


def test_signed_bucket_means_are_transition_weighted_for_uneven_populations(tmp_path, monkeypatch):
    from mjlab_microduck.tasks import mdp
    command = torch.zeros((8, 3))
    env = _Env(command, step=0)
    env.reward_manager = _RewardManager(SimpleNamespace(func=mdp.body_yaw_bias_cost,
        weight=-1.0, params={"command_name": "twist", "tau_s": 1.0}), torch.zeros(8))
    env.command_manager.get_command = lambda _name: env.command
    observer = steady.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    expected_raw = []
    expected_count = 0
    for step in range(24):
        env.common_step_counter = step
        env.command = torch.zeros((8, 3))
        env.command[:, 0] = .3
        env.command[:1 if step < 12 else 3, 2] = -.1
        observer.before(env)
        raw = torch.arange(1, 9, dtype=torch.float32) + step
        ema = raw * .1
        env._microduck_yaw_bias_snapshot = {"signed_residual": raw, "signed_ema": ema,
            "cost": ema.abs(), "weighted_reward": -ema.abs()}
        mask = env.command[:, 2] < 0
        expected_raw.extend(raw[mask].tolist())
        expected_count += int(mask.sum())
        env.common_step_counter = step + 1
        observer.step(env, SimpleNamespace(mean_cost=torch.zeros(8)), {})

    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
                "kl_mean_mean": .01, "kl_mean_max": .02}}
    monkeypatch.setattr(learning.Observer, "update", parent_update)
    row = observer.update(env)

    bucket = row["yaw_bias_signed_buckets"]["negative-low"]
    assert bucket["count"] == expected_count == 48
    assert bucket["raw_signed_residual_mean"] == pytest.approx(sum(expected_raw) / len(expected_raw))


def test_first_update_guard_limits_and_nonfinite_values():
    report = steady.first_update_kl_report("smoke", SOURCE,
        {"optimizer_minibatches": 20, "kl_mean_mean": .05, "kl_mean_max": .10}, steps=24)
    assert report["passed"] is True
    for summary in (
        {"optimizer_minibatches": 20, "kl_mean_mean": .05001, "kl_mean_max": .01},
        {"optimizer_minibatches": 20, "kl_mean_mean": .01, "kl_mean_max": .10001},
        {"optimizer_minibatches": 20, "kl_mean_mean": float("nan"), "kl_mean_max": .01},
        {"optimizer_minibatches": 19, "kl_mean_mean": .01, "kl_mean_max": .01},
    ):
        assert steady.first_update_kl_report("smoke", SOURCE, summary, steps=24)["passed"] is False


def _native_smoke_coverage_receipt():
    return {"worlds": 64, "updates": 5, "steps": 120, "transitions": 7680,
        "idle": 480, "straight": 1680, "negative_yaw": 2760, "positive_yaw": 2760,
        "command_exposure": {"idle": 480, "straight": 1560,
            "negative-low": 960, "negative-target": 480, "negative-high": 1320,
            "positive-low": 480, "positive-target": 480, "positive-high": 1920}}


def test_native_smoke_receipt_uses_distinct_base_and_eight_bin_coverage_schemas():
    steady.validate_command_coverage(_native_smoke_coverage_receipt(), worlds=64, updates=5)


@pytest.mark.parametrize("damage", ["missing-bin", "bad-bin-total", "bad-bin-type",
                                     "bad-negative-map", "budget"])
def test_command_coverage_receipt_rejects_missing_bins_mapping_and_budget(damage):
    receipt = _native_smoke_coverage_receipt()
    if damage == "missing-bin":
        del receipt["command_exposure"]["positive-high"]
    elif damage == "bad-bin-total":
        receipt["command_exposure"]["positive-high"] -= 1
    elif damage == "bad-bin-type":
        receipt["command_exposure"]["positive-high"] = True
    elif damage == "bad-negative-map":
        receipt["negative_yaw"] = 2761
    elif damage == "budget":
        receipt["transitions"] = 7681
    with pytest.raises(ValueError):
        steady.validate_command_coverage(receipt, worlds=64, updates=5)


def test_evaluator_uses_fixed_frozen_v8_diagnostics_and_saves_full_case_first(tmp_path, monkeypatch):
    from mjlab_microduck import gait_path_qualification as route
    from mjlab_microduck import v8_actor_checkpoint

    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = base.sha256(checkpoint)
    calls, saved = [], []
    class Frozen:
        def __init__(self, path, sha):
            assert path == checkpoint and sha == digest
            self.sha256 = sha
        def require_initial_seed(self):
            calls.append("initial-seed")
        def provenance(self):
            return {"kind": "frozen"}
    monkeypatch.setattr(v8_actor_checkpoint, "FrozenV8Checkpoint", Frozen)
    def evaluate(path, seed, speed, yaw, health, **kwargs):
        calls.append(kwargs)
        return {"checkpoint_sha256": digest, "seed": seed, "speed": speed, "yaw": yaw,
                "gait_path_trace": {"trace": seed}, "complete": True}
    monkeypatch.setattr(base, "evaluate_case", evaluate)
    def summarize(case, trace):
        raw_name = f"s{case['seed']}-c{base.CASES.index((case['speed'], case['yaw']))}.json"
        assert (tmp_path / raw_name).is_file()
        saved.append(raw_name)
        return {"seed": case["seed"], "speed": case["speed"], "yaw": case["yaw"]}
    monkeypatch.setattr(route, "summarize_case", summarize)
    monkeypatch.setattr(route, "decision", lambda rows: {"route": "kept"})
    monkeypatch.setattr(steady.limits, "decision", lambda rows: {"legacy": "kept"})

    report = steady.evaluate_matrix(checkpoint, tmp_path, lambda: None, initial=True)

    assert calls[0] == "initial-seed"
    eval_calls = calls[1:]
    assert len(eval_calls) == 12
    assert all(call["diagnostics"] and call["yaw_trace"] and call["gait_trace"]
               and call["gait_path_trace"] and type(call["frozen_v8_checkpoint"]) is Frozen
               for call in eval_calls)
    assert len(report["raw_case_files"]) == len(report["cases"]) == 12
    assert report["route_gate"] == {"route": "kept"}
    assert report["legacy_gate"] == {"legacy": "kept"}
