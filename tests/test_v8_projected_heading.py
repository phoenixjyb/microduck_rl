"""CPU contracts for the matched projected-heading protocol."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_actor_learning as learning
from mjlab_microduck import v8_projected_heading as heading


SOURCE = "a" * 40


@pytest.mark.parametrize("mode,worlds,updates,seed", [
    ("smoke", 64, 5, 947), ("benchmark", 256, 10, 953),
    ("foundation", 256, 1000, 967),
])
def test_config_is_one_heading_reward_and_metric_replacement(mode, worlds, updates, seed):
    from mjlab_microduck import v8_steady_steering as steady

    cfg, agent = heading.prepare_config(mode)
    steady_cfg, steady_agent = steady.prepare_config(mode)
    proof = heading.recipe_match(mode)
    assert cfg.seed == agent.seed == seed
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
    assert agent.max_iterations == updates
    assert cfg.rewards[heading.TERM].weight == -1.0
    assert cfg.rewards[heading.TERM].params == {
        "command_name": "twist", "reference_time_s": 1.0}
    assert cfg.metrics["microduck_projected_heading"].func is heading.capture_heading_metric
    assert set(cfg.rewards) - {heading.TERM} == set(steady_cfg.rewards) - {steady.TERM}
    assert set(cfg.metrics) - {"microduck_projected_heading"} == set(steady_cfg.metrics) - {
        "microduck_yaw_bias"}
    assert cfg.commands == steady_cfg.commands and cfg.curriculum == steady_cfg.curriculum
    assert agent.algorithm.learning_rate == steady_agent.algorithm.learning_rate == 1e-4
    assert proof["matched_after_removing_owned_reward_and_metric"] is True
    assert proof["full_recipe_match"] is False
    assert proof["steady_recipe_sha256"] != proof["projected_heading_recipe_sha256"]
    assert proof["reduced_steady_recipe_sha256"] == proof["reduced_projected_heading_recipe_sha256"]


def test_metric_callable_recipe_identity_is_stable_across_import_context(monkeypatch):
    cfg, agent = heading.prepare_config("smoke")
    expected = hashlib.sha256(heading._canonical_config(cfg, agent).encode()).hexdigest()
    monkeypatch.setattr(heading.capture_heading_metric, "__module__", "__main__")
    assert hashlib.sha256(heading._canonical_config(cfg, agent).encode()).hexdigest() == expected
    mismatched = copy.deepcopy(cfg)
    mismatched.metrics["microduck_projected_heading"].func = lambda _env: None
    with pytest.raises(ValueError, match="known projected-heading metric callable"):
        heading._canonical_config(mismatched, agent)


class _RewardManager:
    def __init__(self, term, weighted):
        self.term = term
        self.active_terms = [heading.TERM]
        self._step_reward = weighted[:, None]

    def get_term_cfg(self, name):
        assert name == heading.TERM
        return self.term


class _Env:
    def __init__(self, command, step=4):
        self.common_step_counter = step
        self.num_envs = len(command)
        self.device = "cpu"
        self.command = command
        self.command_manager = SimpleNamespace(get_command=lambda _name: self.command)


def _state(cost, error, valid, *, reset=None, step=4):
    n = len(cost)
    return {
        "protocol": "projected-heading-error-cost-v1", "step": step, "reference_time_s": 1.0,
        "command_name": "twist", "asset_name": "robot", "step_dt": .02,
        "cost": torch.as_tensor(cost, dtype=torch.float32).clone(),
        "signed_heading_error_rad": torch.tensor(error, dtype=torch.float32),
        "projection_valid": torch.tensor(valid, dtype=torch.bool),
        "segment_reset": torch.tensor(reset if reset is not None else [False] * n, dtype=torch.bool),
        "actual_heading_unwrapped_rad": torch.linspace(0, .1, n),
        "target_heading_unwrapped_rad": torch.linspace(.1, .2, n),
    }


def test_metric_captures_actual_weighted_heading_cost_without_advancing_state(monkeypatch):
    from mjlab_microduck.tasks import mdp
    cost = torch.tensor([.1, .2, 1.0])
    state = _state(cost, [-.1, .2, 5.], [True, True, False])
    calls = []

    def cached_cost(env, **_params):
        calls.append(env.common_step_counter)
        env._microduck_projected_heading_state = state
        return state["cost"].clone()

    monkeypatch.setattr(mdp, "projected_heading_error_cost", cached_cost)
    env = _Env(torch.zeros((3, 3)))
    env._microduck_projected_heading_state = state
    env.reward_manager = _RewardManager(SimpleNamespace(func=cached_cost, weight=-1.0,
        params={"command_name": "twist", "reference_time_s": 1.0}), -cost)

    result = heading.capture_heading_metric(env)
    assert torch.equal(result, torch.zeros(3))
    assert calls == [4]
    assert torch.equal(env._microduck_projected_heading_snapshot["weighted_reward"], -cost)
    assert torch.equal(env._microduck_projected_heading_snapshot["signed_heading_error_rad"],
                       torch.tensor([-.1, .2, 5.]))
    state["cost"].zero_()
    assert torch.equal(env._microduck_projected_heading_snapshot["cost"], cost)


def test_real_reward_metric_observer_flow_handles_invalid_and_terminal_worlds(tmp_path):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager
    from mjlab_microduck.tasks import mdp

    command = torch.tensor([[.3, 0., .2], [.3, 0., .2]])
    env = _Env(command, step=0)
    env.num_envs = 2
    env.step_dt = .02
    env.episode_length_buf = torch.tensor([1, 1], dtype=torch.int64)
    # World 0 begins level and turns slightly; world 1 has an undefined XY
    # projection (body X points vertically) and must receive finite cost 1.
    root = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([
        [1., 0., 0., 0.], [.70710677, 0., .70710677, 0.]])))
    env.scene = {"robot": root}
    env.reward_manager = RewardManager({heading.TERM: RewardTermCfg(
        func=mdp.projected_heading_error_cost, weight=-1.,
        params={"command_name": "twist", "reference_time_s": 1.0})}, env, scale_by_dt=False)
    observer = heading.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE

    observer.before(env)  # primes at counter 0, before action/physics
    root.data.root_link_quat_w = torch.tensor([
        [.9998875, 0., 0., .0149994], [.70710677, 0., .70710677, 0.]])
    env.common_step_counter = 1
    env.reward_manager.compute(.02)
    captured = heading.capture_heading_metric(env)
    assert torch.equal(captured, torch.zeros(2))
    snapshot = env._microduck_projected_heading_snapshot
    assert snapshot["projection_valid"].tolist() == [True, False]
    assert snapshot["cost"][0].item() == pytest.approx(.026, abs=2e-4)
    assert snapshot["cost"][1].item() == pytest.approx(1.)
    assert torch.equal(snapshot["weighted_reward"], env.reward_manager._step_reward[:, 0])

    # Autoreset replaces pose after metric capture; the observer must consume
    # the saved terminal snapshot rather than reading the now-reset root.
    root.data.root_link_quat_w = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
    observer.step(env, SimpleNamespace(mean_cost=torch.zeros(2)),
                  {"dones": torch.tensor([True, False])})
    assert observer.invalid_projection_counts == [1]
    assert env._microduck_projected_heading_snapshot is None


def test_metric_rejects_positive_weight_or_wrong_actual_reward(monkeypatch):
    from mjlab_microduck.tasks import mdp
    cost = torch.tensor([.2])
    def cached_cost(env, **_params):
        env._microduck_projected_heading_state = _state([.2], [.2], [True])
        return cost
    monkeypatch.setattr(mdp, "projected_heading_error_cost", cached_cost)
    for weight, weighted in ((1., cost), (-1., torch.tensor([-.1]))):
        env = _Env(torch.zeros((1, 3)))
        env._microduck_projected_heading_state = _state([.2], [.2], [True])
        env.reward_manager = _RewardManager(SimpleNamespace(func=cached_cost, weight=weight,
            params={"command_name": "twist", "reference_time_s": 1.0}), weighted)
        with pytest.raises(ValueError):
            heading.capture_heading_metric(env)


def test_observer_primes_at_pre_action_counter_then_consumes_terminal_snapshot(monkeypatch, tmp_path):
    from mjlab_microduck.tasks import mdp
    command = torch.tensor([[.3, 0., .1], [0., 0., 0.]])
    env = _Env(command, step=10)
    term = SimpleNamespace(func=mdp.projected_heading_error_cost, weight=-1.0,
        params={"command_name": "twist", "reference_time_s": 1.0})
    env.reward_manager = _RewardManager(term, torch.zeros(2))
    calls = []
    monkeypatch.setattr(mdp, "prepare_projected_heading_step",
        lambda e, **kw: calls.append((e.common_step_counter, kw)))
    observer = heading.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    observer.before(env)
    assert calls == [(10, {"command_name": "twist", "reference_time_s": 1.0})]

    env._microduck_projected_heading_snapshot = {
        "cost": torch.tensor([.2, 1.]),
        "signed_heading_error_rad": torch.tensor([-.2, .7]),
        "projection_valid": torch.tensor([True, False]),
        "segment_reset": torch.tensor([True, False]),
        "actual_heading_unwrapped_rad": torch.tensor([.1, .2]),
        "target_heading_unwrapped_rad": torch.tensor([-.1, .3]),
        "weighted_reward": torch.tensor([-.2, -1.]),
    }
    env.common_step_counter = 11
    observer.step(env, SimpleNamespace(mean_cost=torch.zeros(2)), {"dones": torch.tensor([True, False])})
    assert env._microduck_projected_heading_snapshot is None
    assert observer.invalid_projection_counts == [1]
    assert observer.segment_reset_counts == [1]
    assert observer.bucket_counts["positive-low"] == 1
    assert observer.bucket_counts["idle"] == 1
    with pytest.raises(ValueError, match="one actual command per physics step"):
        observer.step(env, SimpleNamespace(mean_cost=torch.zeros(2)), {})


def test_update_reports_signed_error_invalid_and_reset_counts_with_transition_weighted_buckets(
        monkeypatch, tmp_path):
    from mjlab_microduck.tasks import mdp
    monkeypatch.setattr(mdp, "prepare_projected_heading_step", lambda *_args, **_kwargs: None)
    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
            "kl_mean_mean": .01, "kl_mean_max": .02}}
    monkeypatch.setattr(learning.Observer, "update", parent_update)
    observer = heading.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    command = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [.3, 0., -.1],
        [.3, 0., -.2], [.3, 0., -.3], [.3, 0., .1], [.3, 0., .2], [.3, 0., .3]])
    env = _Env(command, step=0)
    env.num_envs = 8
    env.reward_manager = _RewardManager(SimpleNamespace(func=mdp.projected_heading_error_cost,
        weight=-1.0, params={"command_name": "twist", "reference_time_s": 1.0}), torch.zeros(8))
    signed_samples, invalid_total, reset_total = [], 0, 0
    for step in range(24):
        env.common_step_counter = step
        env.command = command
        observer.before(env)
        error = torch.arange(1, 9, dtype=torch.float32) * (step + 1) / 100
        valid = torch.ones(8, dtype=torch.bool)
        valid[step % 8] = False
        resets = torch.zeros(8, dtype=torch.bool)
        resets[(step + 1) % 8] = True
        cost = torch.where(valid, error.abs(), torch.ones_like(error))
        env._microduck_projected_heading_snapshot = {
            "cost": cost, "signed_heading_error_rad": error,
            "projection_valid": valid, "segment_reset": resets,
            "actual_heading_unwrapped_rad": torch.zeros(8),
            "target_heading_unwrapped_rad": error,
            "weighted_reward": -cost,
        }
        signed_samples.extend(error[valid].tolist())
        invalid_total += int((~valid).sum())
        reset_total += int(resets.sum())
        env.common_step_counter = step + 1
        observer.step(env, SimpleNamespace(mean_cost=torch.zeros(8)), {})

    row = observer.update(env)
    assert row["mean_signed_valid_heading_error_rad"] == pytest.approx(
        sum(signed_samples) / len(signed_samples))
    assert row["mean_abs_valid_heading_error_rad"] == pytest.approx(
        sum(map(abs, signed_samples)) / len(signed_samples))
    assert row["valid_projection_samples"] == len(signed_samples) == 168
    assert row["invalid_projection_samples"] == invalid_total == 24
    assert row["segment_reset_samples"] == reset_total == 24
    assert row["valid_projection_samples"] + row["invalid_projection_samples"] == 192
    assert sum(bucket["count"] for bucket in row["projected_heading_command_buckets"].values()) == 192
    assert row["projected_heading_command_buckets"]["negative-low"]["count"] == 24
    assert row["projected_heading_command_buckets"]["negative-low"]["valid_projection_count"] == 21
    assert row["projected_heading_command_buckets"]["negative-low"]["invalid_projection_count"] == 3
    assert "signed_ema" not in json.dumps(row)
    assert "raw_error_rate" not in json.dumps(row)


def test_first_update_kl_refusal_persists_report_and_explicit_aborted_state(tmp_path, monkeypatch):
    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
            "kl_mean_mean": .051, "kl_mean_max": .05}}
    monkeypatch.setattr(learning.Observer, "update", parent_update)
    observer = heading.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    saves = []
    observer.runner = SimpleNamespace(save=lambda path, infos: saves.append((path, infos)))
    observer.dwell = dict.fromkeys(learning.exposure.EXPOSURE_BINS, 0)
    for name in ("idle", "straight", "negative-low", "negative-target", "negative-high",
                 "positive-low", "positive-target", "positive-high"):
        observer.bucket_counts[name] = 24
    for values in (observer.heading_costs, observer.heading_rewards, observer.valid_heading_error_sums,
                   observer.valid_projection_counts, observer.invalid_projection_counts,
                   observer.segment_reset_counts):
        values.extend([0.] * 24)
    observer.bucket_values = {key: {name: [] for name in
        ("valid_signed_heading_error_rad", "valid_abs_heading_error_rad",
         "valid_count", "invalid_count", "cost", "weighted_reward")}
        for key in learning.exposure.EXPOSURE_BINS}
    for value in observer.bucket_values.values():
        value["cost"] = [0.]
        value["weighted_reward"] = [0.]
    with pytest.raises(ValueError, match="projected-heading first-update KL guard"):
        observer.update(SimpleNamespace(common_step_counter=24, num_envs=8))
    report = json.loads((tmp_path / "first-update-kl.json").read_text())
    assert report["passed"] is False
    assert saves[0][1]["env_state"] == {"common_step_counter": 24}
    assert saves[0][1]["full_budget_completed"] is False


def test_first_update_kl_limits_and_nonfinite_values():
    report = heading.first_update_kl_report("smoke", SOURCE,
        {"optimizer_minibatches": 20, "kl_mean_mean": .05, "kl_mean_max": .10}, steps=24)
    assert report["passed"] is True
    for summary in (
        {"optimizer_minibatches": 20, "kl_mean_mean": .05001, "kl_mean_max": .01},
        {"optimizer_minibatches": 20, "kl_mean_mean": .01, "kl_mean_max": .10001},
        {"optimizer_minibatches": 20, "kl_mean_mean": float("nan"), "kl_mean_max": .01},
        {"optimizer_minibatches": 19, "kl_mean_mean": .01, "kl_mean_max": .01},
    ):
        assert not heading.first_update_kl_report("smoke", SOURCE, summary, steps=24)["passed"]


def _coverage_receipt():
    return {"worlds": 64, "updates": 5, "steps": 120, "transitions": 7680,
        "idle": 480, "straight": 1680, "negative_yaw": 2760, "positive_yaw": 2760,
        "command_exposure": {"idle": 480, "straight": 1560,
            "negative-low": 960, "negative-target": 480, "negative-high": 1320,
            "positive-low": 480, "positive-target": 480, "positive-high": 1920}}


def test_native_coverage_keeps_both_exact_command_schemas():
    heading.validate_command_coverage(_coverage_receipt(), worlds=64, updates=5)
    bad = _coverage_receipt()
    bad["command_exposure"].pop("positive-high")
    with pytest.raises(ValueError):
        heading.validate_command_coverage(bad, worlds=64, updates=5)


def test_duration_and_receipt_identity_helpers():
    assert heading.benchmark_projection_seconds(10.) == 1560.
    with pytest.raises(ValueError):
        heading.benchmark_projection_seconds(float("nan"))
    receipt = heading.assemble_receipt(SOURCE, "smoke", {
        "source": "bad", "protocol": "bad", "mode": "bad", "profile": "bad", "x": 1})
    assert receipt == {"source": SOURCE, "protocol": heading.PROTOCOL,
        "mode": "smoke", "profile": "projected-heading", "x": 1}


def test_training_receipt_binds_initial_actor_and_honest_stock_control_match(
        tmp_path, monkeypatch):
    output = tmp_path / "smoke"
    output.mkdir()
    checkpoint = output / "model_4.pt"
    checkpoint.write_bytes(b"final checkpoint")
    initial = output / "initial.pt"
    initial.write_bytes(b"initial checkpoint")
    control = SimpleNamespace(tag="retained-stock-control", source="c" * 40,
        checkpoint_sha256="d" * 64)
    monkeypatch.setattr(heading.profiles, "control_initial", lambda _mode: control)
    monkeypatch.setattr(heading, "validate_command_coverage", lambda *_args: None)
    first_kl = {"passed": True, "mode": "smoke", "source": SOURCE}
    monkeypatch.setattr(heading, "validate_first_update_kl", lambda *_args: first_kl)
    match = {"mode": "smoke", "protocol": "v8-projected-heading-recipe-match-v2",
        "fixed_reward_weight": -1.0, "matched_after_removing_owned_reward_and_metric": True,
        "full_recipe_match": False, "projected_heading_recipe_sha256": "e" * 64}
    monkeypatch.setattr(heading, "recipe_match", lambda _mode: match)
    (output / "recipe-match.json").write_text(json.dumps(match))
    actor = {"ppo_resume": False, "actor_class": "mjlab_microduck.v8_actor_seed:FrozenV8Actor",
        "initial_std": heading.smoke.INITIAL_STD, "adam_state_entries": 0,
        "artifact_sha256": heading.pinned.V8.sha256,
        "initial_learning_rate": heading.profiles.CAUTIOUS_INITIAL_LR,
        "target_iteration": 0, "target_common_step_counter": 0,
        "critic_normalizer_count": 0, "maximum_actor_error": 0.,
        "maximum_export_forward_error": 0.}
    (output / "actor-import.json").write_text(json.dumps(actor))
    matched = {"protocol": "v8-projected-heading-matched-initialization-v1",
        "mode": "smoke", "source": SOURCE, "profile": "projected-heading", "seed": 947,
        "control_model_tag": control.tag, "control_source": control.source,
        "control_checkpoint_sha256": control.checkpoint_sha256,
        "recipe_sha256": match["projected_heading_recipe_sha256"],
        "actor_state_equal": True, "critic_state_equal": True,
        "optimizer_state_empty": True, "parameter_groups_match_except_lr": True,
        "actor_normalizer_count": 0, "critic_normalizer_count": 0,
        "tensors_in_report": False, "ppo_resume": False,
        "candidate_iteration": 0, "candidate_common_step_counter": 0,
        "control_iteration": 0, "control_common_step_counter": 0,
        "candidate_learning_rate": heading.profiles.CAUTIOUS_INITIAL_LR,
        "control_learning_rate": heading.profiles.STOCK_INITIAL_LR,
        "trajectory_identity_claim": False, "policy_acceptance": False,
        "physical_motion_authorized": False}
    (output / "matched-initialization.json").write_text(json.dumps(matched))
    (output / "first-update-kl.json").write_text(json.dumps(first_kl))
    bins = {name: 64 * 5 * 24 // 8 for name in heading.exposure.EXPOSURE_BINS}
    metric = {"projected_heading_weight": -1.0,
        "projected_heading_reference_time_s": 1.0,
        "weighted_projected_heading_reward_mean": -.2,
        "projected_heading_cost_mean": .2,
        "projected_heading_observed_steps": 24,
        "projected_heading_activity_steps": 1,
        "mean_abs_valid_heading_error_rad": .2,
        "mean_signed_valid_heading_error_rad": -.1,
        "valid_projection_samples": 1536,
        "invalid_projection_samples": 0, "segment_reset_samples": 0,
        "projected_heading_command_buckets": {name: {"count": 64 * 24 // 8,
            "valid_projection_count": 64 * 24 // 8,
            "invalid_projection_count": 0,
            "signed_valid_heading_error_rad_mean": 0.,
            "abs_valid_heading_error_rad_mean": .2, "cost_mean": .2,
            "weighted_reward_mean": -.2} for name in heading.exposure.EXPOSURE_BINS}}
    (output / "updates.jsonl").write_text("\n".join(json.dumps(metric) for _ in range(5)))
    result = {"protocol": heading.PROTOCOL, "profile": "projected-heading",
        "source": SOURCE, "status": "training-complete-not-accepted", "worlds": 64,
        "updates": 5, "seed": 947, "steps": 120, "transitions": 7680,
        "checkpoint": str(checkpoint), "checkpoint_sha256": base.sha256(checkpoint),
        "initial_checkpoint_sha256": base.sha256(initial), "actor_import": actor,
        "recipe_match": match, "matched_initialization": matched,
        "command_exposure": bins, "idle": bins["idle"], "straight": bins["straight"] + 100,
        "negative_yaw": 1000, "positive_yaw": 1000, "elapsed_s": 3.,
        "first_update_kl": first_kl, "policy_acceptance": False,
        "physical_motion_authorized": False}
    (output / "result.json").write_text(json.dumps(result))

    assert heading.validate_training_receipt(tmp_path, "smoke", SOURCE) == result
    matched["trajectory_identity_claim"] = True
    (output / "matched-initialization.json").write_text(json.dumps(matched))
    (output / "result.json").write_text(json.dumps({**result, "matched_initialization": matched}))
    with pytest.raises(ValueError):
        heading.validate_training_receipt(tmp_path, "smoke", SOURCE)


def test_evaluation_receipt_binds_raw_hashes_compact_summaries_and_fixed_matrix(tmp_path):
    output = tmp_path / "final-eval"
    output.mkdir()
    raw_hashes, summaries = [], []
    digest = "b" * 64
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            name = f"s{seed}-c{index}.json"
            row = {"seed": seed, "speed": speed, "yaw": yaw, "complete": True,
                "worlds": 8, "steps": 240, "terminal_worlds": [], "checkpoint_sha256": digest,
                "speed_mae": [0.] * 8, "yaw_mae": [0.] * 8,
                "response_diagnostics": {"startup_steps": 60},
                "body_yaw_trace": {"startup_steps": 60, "steps": 240, "worlds": 8,
                                    "checkpoint_sha256": digest},
                "gait_yaw_trace": {"checkpoint_sha256": digest},
                "gait_path_trace": {"checkpoint_sha256": digest}}
            (output / name).write_text(json.dumps(row))
            raw_hashes.append({"file": name, "sha256": base.sha256(output / name)})
            summary = {"seed": seed, "speed": speed, "yaw": yaw, "compact": True}
            (output / f"s{seed}-c{index}-route.json").write_text(json.dumps(summary))
            summaries.append(summary)
    receipt = heading.assemble_receipt(SOURCE, "final-eval", {
        "checkpoint_sha256": digest, "raw_case_files": raw_hashes, "cases": summaries,
        "route_gate": {"decision": "recorded"}, "legacy_gate": {"decision": "recorded"},
        "policy_acceptance": False, "physical_motion_authorized": False})
    (output / "result.json").write_text(json.dumps(receipt))
    assert heading.validate_evaluation_receipt(tmp_path, "final-eval", SOURCE, digest) == receipt
    (output / "s853-c1.json").write_text("{}")
    with pytest.raises(ValueError, match="raw case hashes"):
        heading.validate_evaluation_receipt(tmp_path, "final-eval", SOURCE, digest)
