"""CPU contracts for the matched projected-yaw-bias protocol."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_actor_learning as learning
from mjlab_microduck import v8_projected_yaw_bias as projected


SOURCE = "a" * 40


@pytest.mark.parametrize("mode,worlds,updates,seed", [
    ("smoke", 64, 5, 947), ("benchmark", 256, 10, 953),
    ("foundation", 256, 1000, 967),
])
def test_recipe_changes_only_owned_reward_and_metric(mode, worlds, updates, seed):
    from mjlab_microduck import v8_steady_steering as steady
    from mjlab_microduck.tasks import mdp

    cfg, agent = projected.prepare_config(mode)
    steady_cfg, steady_agent = steady.prepare_config(mode)
    proof = projected.recipe_match(mode)
    term = cfg.rewards[projected.TERM]
    assert cfg.seed == agent.seed == seed
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
    assert agent.max_iterations == updates
    assert agent.save_interval == 250
    assert term.func is mdp.projected_yaw_bias_cost and term.weight == -1.0
    assert term.params == {"command_name": "twist", "tau_s": 1.0}
    assert cfg.metrics["microduck_projected_yaw_bias"].func is projected.capture_yaw_bias_metric
    assert set(cfg.rewards) - {projected.TERM} == set(steady_cfg.rewards) - {steady.TERM}
    assert set(cfg.metrics) - {"microduck_projected_yaw_bias"} == set(steady_cfg.metrics) - {
        "microduck_yaw_bias"}
    assert cfg.commands == steady_cfg.commands and cfg.curriculum == steady_cfg.curriculum
    assert agent.algorithm.learning_rate == steady_agent.algorithm.learning_rate == 1e-4
    assert agent.algorithm.desired_kl == steady_agent.algorithm.desired_kl == .01
    assert proof["matched_after_removing_owned_reward_and_metric"] is True
    assert proof["full_recipe_match"] is False
    assert proof["steady_recipe_sha256"] != proof["projected_yaw_bias_recipe_sha256"]
    assert proof["reduced_steady_recipe_sha256"] == proof["reduced_projected_yaw_bias_recipe_sha256"]


def test_metric_callable_identity_is_stable_and_rejects_substitution(monkeypatch):
    cfg, agent = projected.prepare_config("smoke")
    expected = hashlib.sha256(projected._canonical_config(cfg, agent).encode()).hexdigest()
    monkeypatch.setattr(projected.capture_yaw_bias_metric, "__module__", "__main__")
    assert hashlib.sha256(projected._canonical_config(cfg, agent).encode()).hexdigest() == expected
    mismatched = copy.deepcopy(cfg)
    mismatched.metrics["microduck_projected_yaw_bias"].func = lambda _env: None
    with pytest.raises(ValueError, match="known projected-yaw-bias metric callable"):
        projected._canonical_config(mismatched, agent)


class _Env:
    def __init__(self, command, step=0):
        self.common_step_counter = step
        self.num_envs = len(command)
        self.device = "cpu"
        self.step_dt = .02
        self.command = command
        self.command_manager = SimpleNamespace(get_command=lambda _name: self.command)


def test_real_mdp_reward_manager_metric_terminal_snapshot_observer_flow(tmp_path):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager
    from mjlab_microduck.tasks import mdp

    command = torch.tensor([[.3, 0., .2], [.3, 0., -.1]])
    env = _Env(command)
    env.episode_length_buf = torch.tensor([2, 2], dtype=torch.int64)
    env.max_episode_length_s = 20.
    # World 1 has body X vertical, so projected heading rate is undefined.
    root = SimpleNamespace(data=SimpleNamespace(
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.], [.70710677, 0., .70710677, 0.]]),
        root_link_ang_vel_b=torch.tensor([[0., 0., .1], [0., 0., 0.]])))
    env.scene = {"robot": root}
    env.reward_manager = RewardManager({projected.TERM: RewardTermCfg(
        func=mdp.projected_yaw_bias_cost, weight=-1.,
        params={"command_name": "twist", "tau_s": 1.0})}, env, scale_by_dt=False)
    observer = projected.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE

    observer.before(env)  # records the command; no EMA priming call
    root.data.root_link_quat_w = torch.tensor([
        [.9998875, 0., 0., .0149994], [.70710677, 0., .70710677, 0.]])
    env.common_step_counter = 1
    env.reward_manager.compute(.02)
    captured = projected.capture_yaw_bias_metric(env)
    assert torch.equal(captured, torch.zeros(2))
    snapshot = env._microduck_projected_yaw_bias_snapshot
    assert snapshot["projection_valid"].tolist() == [True, False]
    assert snapshot["signed_residual"][0].item() == pytest.approx(.1, abs=2e-4)
    assert snapshot["signed_ema"][0].item() == pytest.approx(.002, abs=5e-5)
    assert snapshot["signed_residual"][1].item() == snapshot["signed_ema"][1].item() == 0.
    assert snapshot["projected_yaw_rate_rad_s"][1].item() == 0.
    assert snapshot["cost"][1].item() == 1.
    assert torch.equal(snapshot["weighted_reward"], env.reward_manager._step_reward[:, 0])

    # Simulate autoreset replacing terminal pose before observer consumption.
    env.reward_manager.reset(torch.tensor([0, 1]))
    root.data.root_link_quat_w = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
    observer.step(env, SimpleNamespace(mean_cost=torch.zeros(2)),
                  {"dones": torch.tensor([True, False])})
    assert observer.invalid_counts == [1]
    assert observer.valid_counts == [1]
    assert env._microduck_projected_yaw_bias_snapshot is None
    with pytest.raises(ValueError, match="one actual command per physics step"):
        observer.step(env, SimpleNamespace(mean_cost=torch.zeros(2)), {})


def test_update_metrics_are_validity_weighted_and_keep_invalid_bucket_values_null(
        monkeypatch, tmp_path):
    from mjlab_microduck.tasks import mdp

    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
            "kl_mean_mean": .01, "kl_mean_max": .02}}
    monkeypatch.setattr(learning.Observer, "update", parent_update)
    command = torch.tensor([[0., 0., 0.], [.3, 0., .2], [.3, 0., -.2]])
    env = _Env(command)
    env.num_envs = len(command)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(
        func=mdp.projected_yaw_bias_cost, weight=-1., params={"command_name": "twist", "tau_s": 1.0}))
    observer = projected.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    valid_ema, valid_residual, valid_rate = [], [], []
    for step in range(24):
        env.common_step_counter = step
        observer.before(env)
        valid = torch.tensor([False, True, True])
        ema = torch.tensor([0., .04, -.08])
        residual = torch.tensor([0., .2, -.4])
        rate = torch.tensor([0., .1, .2])
        cost = torch.where(valid, ema.abs(), torch.ones_like(ema))
        env._microduck_projected_yaw_bias_snapshot = {
            "cost": cost, "signed_residual": residual, "signed_ema": ema,
            "projection_valid": valid, "projected_yaw_rate_rad_s": rate,
            "weighted_reward": -cost,
        }
        valid_ema.extend(ema[valid].tolist())
        valid_residual.extend(residual[valid].tolist())
        valid_rate.extend(rate[valid].tolist())
        env.common_step_counter = step + 1
        observer.step(env, SimpleNamespace(mean_cost=torch.zeros(3)), {})

    row = observer.update(env)
    assert row["valid_projection_samples"] == 48
    assert row["invalid_projection_samples"] == 24
    assert row["mean_signed_valid_yaw_ema"] == pytest.approx(sum(valid_ema) / len(valid_ema))
    assert row["mean_signed_valid_residual"] == pytest.approx(sum(valid_residual) / len(valid_residual))
    assert row["mean_projected_yaw_rate_rad_s"] == pytest.approx(sum(valid_rate) / len(valid_rate))
    buckets = row["projected_yaw_bias_command_buckets"]
    assert buckets["idle"]["valid_projection_count"] == 0
    assert buckets["idle"]["invalid_projection_count"] == 24
    assert buckets["idle"]["signed_ema_mean"] is None
    assert buckets["idle"]["signed_residual_mean"] is None
    assert buckets["idle"]["projected_yaw_rate_rad_s_mean"] is None
    assert buckets["idle"]["cost_mean"] == 1.
    assert buckets["idle"]["weighted_reward_mean"] == -1.
    assert sum(bucket["count"] for bucket in buckets.values()) == 72


def test_first_update_kl_guard_persists_refusal(monkeypatch, tmp_path):
    events = []
    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
            "kl_mean_mean": .051, "kl_mean_max": .05}}
    monkeypatch.setattr(learning.Observer, "update", parent_update)
    observer = projected.Observer("smoke")
    observer.output, observer.source = tmp_path, SOURCE
    observer.runner = SimpleNamespace(save=lambda path, infos: events.append((path, infos)))
    for values in (observer.yaw_bias_cost_sums, observer.yaw_bias_reward_sums,
                   observer.valid_counts, observer.invalid_counts, observer.valid_ema_sums,
                   observer.valid_residual_sums, observer.valid_rate_sums):
        values.extend([0.] * 24)
    observer.bucket_counts = dict.fromkeys(projected.exposure.EXPOSURE_BINS, 0)
    observer.bucket_counts["idle"] = 24
    env = _Env(torch.zeros((1, 3)), step=24)
    with pytest.raises(ValueError, match="first-update KL guard"):
        observer.update(env)
    report = __import__("json").loads((tmp_path / "first-update-kl.json").read_text())
    assert report["passed"] is False and report["fixed_reward_weight"] == -1.
    assert len(events) == 1 and events[0][1]["physical_motion_authorized"] is False


def test_training_receipt_refuses_heading_protocol_or_old_hash(tmp_path):
    stage = tmp_path / "smoke"
    stage.mkdir()
    (stage / "result.json").write_text('{"protocol":"v8-projected-heading-v1"}')
    with pytest.raises((ValueError, KeyError)):
        projected.validate_training_receipt(tmp_path, "smoke", SOURCE)


def test_training_receipt_binds_initialization_recipe_checkpoint_schema_and_duration(tmp_path):
    from mjlab_microduck import official_native_reference as pinned
    from mjlab_microduck import v8_actor_lr_profile as profiles
    from mjlab_microduck import v8_actor_smoke as smoke

    output = tmp_path / "smoke"
    output.mkdir()
    checkpoint = output / "model_4.pt"
    initial = output / "initial.pt"
    checkpoint.write_bytes(b"final checkpoint bytes")
    initial.write_bytes(b"initial checkpoint bytes")
    proof = projected.recipe_match("smoke")
    actor = dict(ppo_resume=False, actor_class="mjlab_microduck.v8_actor_seed:FrozenV8Actor",
        initial_std=smoke.INITIAL_STD, adam_state_entries=0, artifact_sha256=pinned.V8.sha256,
        initial_learning_rate=profiles.CAUTIOUS_INITIAL_LR, target_iteration=0,
        target_common_step_counter=0, critic_normalizer_count=0,
        maximum_actor_error=0., maximum_export_forward_error=0.)
    control = profiles.control_initial("smoke")
    matched = dict(protocol="v8-projected-yaw-bias-matched-initialization-v1",
        mode="smoke", source=SOURCE, profile="projected-yaw-bias", seed=947,
        control_model_tag=control.tag, control_source=control.source,
        control_checkpoint_sha256=control.checkpoint_sha256,
        recipe_sha256=proof["projected_yaw_bias_recipe_sha256"],
        actor_state_equal=True, critic_state_equal=True, optimizer_state_empty=True,
        parameter_groups_match_except_lr=True, actor_normalizer_count=0,
        critic_normalizer_count=0, tensors_in_report=False, ppo_resume=False,
        candidate_iteration=0, candidate_common_step_counter=0,
        control_iteration=0, control_common_step_counter=0,
        candidate_learning_rate=profiles.CAUTIOUS_INITIAL_LR,
        control_learning_rate=profiles.STOCK_INITIAL_LR,
        trajectory_identity_claim=False, policy_acceptance=False,
        physical_motion_authorized=False)
    (output / "actor-import.json").write_text(json.dumps(actor))
    (output / "recipe-match.json").write_text(json.dumps(proof))
    (output / "matched-initialization.json").write_text(json.dumps(matched))
    kl_report = projected.first_update_kl_report("smoke", SOURCE,
        dict(optimizer_minibatches=20, kl_mean_mean=.01, kl_mean_max=.02), steps=24)
    (output / "first-update-kl.json").write_text(json.dumps(kl_report))
    bins = dict.fromkeys(projected.exposure.EXPOSURE_BINS, 960)
    metric_bucket_counts = dict.fromkeys(projected.exposure.EXPOSURE_BINS, 0)
    metric_bucket_counts["idle"] = 1536
    metric_buckets = {key: dict(count=count, valid_projection_count=count,
        invalid_projection_count=0, signed_ema_mean=.01 if count else None,
        signed_residual_mean=.1 if count else None,
        projected_yaw_rate_rad_s_mean=.1 if count else None,
        cost_mean=.01 if count else None, weighted_reward_mean=-.01 if count else None)
        for key, count in metric_bucket_counts.items()}
    updates = []
    for update in range(1, 6):
        updates.append(dict(update=update, projected_yaw_bias_weight=-1.,
            projected_yaw_bias_tau_s=1., weighted_projected_yaw_bias_reward_mean=-.01,
            projected_yaw_bias_cost_mean=.01, projected_yaw_bias_observed_steps=24,
            projected_yaw_bias_activity_steps=1, valid_projection_samples=1536,
            invalid_projection_samples=0, mean_signed_valid_yaw_ema=.01,
            mean_signed_valid_residual=.1, mean_projected_yaw_rate_rad_s=.1,
            projected_yaw_bias_command_buckets=metric_buckets))
    (output / "updates.jsonl").write_text("\n".join(json.dumps(row) for row in updates) + "\n")
    result = dict(protocol=projected.PROTOCOL, profile="projected-yaw-bias", source=SOURCE,
        status="training-complete-not-accepted", policy_acceptance=False,
        physical_motion_authorized=False, worlds=64, updates=5, seed=947, steps=120,
        transitions=7680, checkpoint=str(checkpoint), checkpoint_sha256=projected.base.sha256(checkpoint),
        initial_checkpoint_sha256=projected.base.sha256(initial), actor_import=actor,
        recipe_match=proof, matched_initialization=matched, command_exposure=bins,
        idle=960, straight=1536, negative_yaw=1536, positive_yaw=1536,
        forward=1, lateral=1, reverse=1, elapsed_s=10., first_update_kl=kl_report)
    (output / "result.json").write_text(json.dumps(result))

    assert projected.validate_training_receipt(tmp_path, "smoke", SOURCE) == result
    result["elapsed_s"] = projected.CAPS["smoke"] + 1
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="bounded duration"):
        projected.validate_training_receipt(tmp_path, "smoke", SOURCE)
    result["elapsed_s"] = 10.
    (output / "result.json").write_text(json.dumps(result))
    checkpoint.write_bytes(b"altered checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        projected.validate_training_receipt(tmp_path, "smoke", SOURCE)


def test_final_evaluation_receipt_binds_case_identity_hashes_and_duration(tmp_path):
    output = tmp_path / "final-eval"
    output.mkdir()
    checkpoint_sha = "b" * 64
    rows, summaries, raw_hashes = [], [], []
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            raw = dict(seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240,
                complete=True, terminal_worlds=[], checkpoint_sha256=checkpoint_sha,
                speed_mae=[.05 if speed else .002] * 8, yaw_mae=[.01] * 8,
                response_diagnostics={"startup_steps": 60},
                body_yaw_trace={"startup_steps": 60, "steps": 240, "worlds": 8,
                    "checkpoint_sha256": checkpoint_sha},
                gait_path_trace={"checkpoint_sha256": checkpoint_sha},
                gait_yaw_trace={"checkpoint_sha256": checkpoint_sha})
            raw_path = output / f"s{seed}-c{index}.json"
            raw_path.write_text(json.dumps(raw, sort_keys=True))
            raw_hashes.append({"file": raw_path.name, "sha256": projected.base.sha256(raw_path)})
            summary = {"seed": seed, "speed": speed, "yaw": yaw}
            (output / f"s{seed}-c{index}-route.json").write_text(json.dumps(summary))
            rows.append(raw)
            summaries.append(summary)
    result = dict(source=SOURCE, protocol=projected.PROTOCOL, mode="final-eval",
        profile="projected-yaw-bias", checkpoint_sha256=checkpoint_sha,
        raw_case_files=raw_hashes, cases=summaries, route_gate={}, legacy_gate={}, elapsed_s=2.5,
        policy_acceptance=False, physical_motion_authorized=False)
    (output / "result.json").write_text(json.dumps(result))

    assert projected.validate_evaluation_receipt(tmp_path, "final-eval", SOURCE, checkpoint_sha) == result
    result["elapsed_s"] = projected.CAPS["final-eval"] + 1
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="bounded evaluation duration"):
        projected.validate_evaluation_receipt(tmp_path, "final-eval", SOURCE, checkpoint_sha)

    result["elapsed_s"] = 2.5
    result["source"] = "c" * 40
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        projected.validate_evaluation_receipt(tmp_path, "final-eval", SOURCE, checkpoint_sha)

    result["source"] = SOURCE
    result["raw_case_files"][0]["sha256"] = "0" * 64
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        projected.validate_evaluation_receipt(tmp_path, "final-eval", SOURCE, checkpoint_sha)


def test_command_coverage_checks_base_and_signed_schemas():
    worlds, updates = 64, 5
    transitions = worlds * updates * 24
    bins = dict.fromkeys(projected.exposure.EXPOSURE_BINS, 1)
    bins["idle"] = transitions - len(bins) + 1
    result = dict(worlds=worlds, updates=updates, steps=updates * 24, transitions=transitions,
        command_exposure=bins, idle=bins["idle"], straight=1, negative_yaw=1, positive_yaw=1,
        forward=1, lateral=1, reverse=1)
    projected.validate_command_coverage(result, worlds, updates)
    result["straight"] = 0
    with pytest.raises(ValueError):
        projected.validate_command_coverage(result, worlds, updates)
