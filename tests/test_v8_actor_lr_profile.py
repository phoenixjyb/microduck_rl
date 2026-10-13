"""CPU-only profile, exact-start and first-update KL guard tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_actor_learning as learning
from mjlab_microduck import v8_actor_lr_profile as profiles


SOURCE = "b" * 40


def test_stock_default_and_cautious_config_differ_only_in_initial_learning_rate():
    assert learning.PROTOCOL == "v8-actor-only-foundation-learning-v1"
    assert learning.MODES == {"benchmark": (256, 10, 953), "foundation": (256, 1000, 967)}
    assert learning.CAPS == {"benchmark": 300, "initial-eval": 600,
                             "foundation": 2100, "final-eval": 600}
    assert learning.prepare_config("benchmark")[1].algorithm.learning_rate == .001
    assert profiles.profile("stock").protocol == learning.PROTOCOL
    assert profiles.profile("cautious").protocol == "v8-cautious-initial-lr-v1"
    assert profiles.profile("cautious").run_prefix == "v8-cautious-lr"
    assert profiles.CAUTIOUS_MODES == {"smoke": (64, 5, 947),
                                       "benchmark": (256, 10, 953),
                                       "foundation": (256, 1000, 967)}
    assert profiles.CAUTIOUS_CAPS == {"smoke": 300, "benchmark": 300,
                                      "initial-eval": 600, "foundation": 2100,
                                      "final-eval": 600}
    for mode in profiles.CAUTIOUS_MODES:
        assert profiles.require_stock_recipe_match(mode)["profile_config_match"] is True
        cfg, agent = learning.prepare_config(mode, profile="cautious")
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == profiles.CAUTIOUS_MODES[mode][0]
        assert cfg.seed == agent.seed == profiles.CAUTIOUS_MODES[mode][2]
        assert agent.max_iterations == profiles.CAUTIOUS_MODES[mode][1]
        assert agent.algorithm.learning_rate == profiles.CAUTIOUS_INITIAL_LR
        assert agent.algorithm.desired_kl == .01 and agent.algorithm.schedule == "adaptive"
        assert agent.algorithm.num_learning_epochs == 5 and agent.algorithm.num_mini_batches == 4


def test_stock_control_source_pins_use_full_commit_shas():
    assert profiles.control_initial("smoke").source == "c178fbcc53c9d7e16d9bd09b5b21896fecf18645"
    assert profiles.control_initial("benchmark").source == "722e7a6ac6f9cf2995cbab37b7b13aa14e7c218a"
    assert profiles.control_initial("foundation").source == "722e7a6ac6f9cf2995cbab37b7b13aa14e7c218a"


@pytest.mark.parametrize("name", ["", "Cautious", "resume", "stock-cautious", None])
def test_invalid_profile_names_refuse(name):
    with pytest.raises(ValueError, match="profile must be exactly"):
        profiles.profile(name)


def test_first_update_kl_thresholds_and_atomic_evidence(tmp_path):
    passed = profiles.first_update_kl_report(
        "smoke", SOURCE,
        {"optimizer_minibatches": 20, "kl_mean_mean": .05, "kl_mean_max": .10},
        steps=24)
    assert passed["passed"] is True
    assert passed["mean_raw_kl_limit"] == .05
    assert passed["maximum_minibatch_mean_raw_kl_limit"] == .10
    assert passed["policy_acceptance"] is False

    failed = profiles.first_update_kl_report(
        "benchmark", SOURCE,
        {"optimizer_minibatches": 20, "kl_mean_mean": .050001, "kl_mean_max": .02},
        steps=24)
    assert failed["passed"] is False
    stage = tmp_path / "smoke"
    stage.mkdir()
    base.write_new(stage / "first-update-kl.json", passed)
    with pytest.raises(FileExistsError):
        base.write_new(stage / "first-update-kl.json", failed)
    assert profiles.validate_first_update_kl(tmp_path, "smoke", SOURCE) == passed

    damaged = copy.deepcopy(passed)
    damaged["source"] = "c" * 40
    (stage / "first-update-kl.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="source-bound cautious first-update KL guard"):
        profiles.validate_first_update_kl(tmp_path, "smoke", SOURCE)


def test_observer_persists_failed_first_kl_and_aborted_update_before_stopping(tmp_path, monkeypatch):
    from mjlab_microduck import integrated_fresh_foundation as fresh

    stage = tmp_path / "smoke"
    stage.mkdir()
    saved = {}
    observer = learning.Observer()
    observer.profile, observer.mode, observer.source, observer.output = "cautious", "smoke", SOURCE, stage
    observer.updates = 0
    class Runner:
        def save(self, path, infos=None):
            saved["first_report_exists"] = (stage / "first-update-kl.json").is_file()
            saved["infos"] = infos
            Path(path).write_bytes(b"aborted first-update checkpoint")
    observer.runner = Runner()
    monkeypatch.setattr(learning.kl, "optimizer_summary", lambda env, desired: {
        "optimizer_minibatches": 20, "kl_mean_mean": .051, "kl_mean_max": .02,
    })
    monkeypatch.setattr(fresh.FreshObserver, "curriculum_profile", staticmethod(lambda env: {}))
    env = type("Env", (), {"common_step_counter": 24})()

    with pytest.raises(ValueError, match="cautious first-update KL guard"):
        observer.update(env)

    report = json.loads((stage / "first-update-kl.json").read_text())
    assert report["passed"] is False
    assert saved["first_report_exists"] is True
    assert saved["infos"]["common_step_counter"] == 24
    assert saved["infos"]["full_budget_completed"] is False
    assert (stage / "aborted-first-update.pt").is_file()


def test_result_receipt_assembly_deduplicates_profile_and_preserves_stock_shape():
    fields = {"source": "spoof", "protocol": "spoof", "mode": "spoof",
              "profile": "cautious", "updates": 5}
    stock = learning.assemble_result_receipt(SOURCE, learning.PROTOCOL, "benchmark", fields)
    cautious = learning.assemble_result_receipt(
        SOURCE, profiles.CAUTIOUS_PROTOCOL, "smoke", fields, profile="cautious")
    assert stock == {"source": SOURCE, "protocol": learning.PROTOCOL,
                     "mode": "benchmark", "updates": 5}
    assert cautious == {"source": SOURCE, "protocol": profiles.CAUTIOUS_PROTOCOL,
                        "mode": "smoke", "profile": "cautious", "updates": 5}


class _Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_mean", torch.zeros(3))
        self.register_buffer("_var", torch.ones(3))
        self.register_buffer("_std", torch.ones(3))
        self.eps = 0.0
        self.until = 0


class FrozenV8Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("v8_seed_digest", torch.tensor(
            list(bytes.fromhex(profiles.pinned.V8.sha256)), dtype=torch.uint8))
        self.obs_normalizer = _Normalizer()
        self.params = nn.ParameterList([nn.Parameter(torch.full((2,), float(i))) for i in range(8)])


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_normalizer = _Normalizer()
        self.params = nn.ParameterList([nn.Parameter(torch.full((2,), float(i + 10))) for i in range(9)])


class _Runner:
    def __init__(self, *, damage=None):
        self.alg = type("Algorithm", (), {})()
        self.alg.actor, self.alg.critic = FrozenV8Actor(), _Critic()
        self.alg.optimizer = torch.optim.Adam(
            list(self.alg.actor.parameters()) + list(self.alg.critic.parameters()),
            lr=profiles.CAUTIOUS_INITIAL_LR)
        self.alg.learning_rate = profiles.CAUTIOUS_INITIAL_LR
        self.alg.schedule, self.alg.desired_kl = "adaptive", .01
        self.alg.num_learning_epochs, self.alg.num_mini_batches = 5, 4
        self.current_learning_iteration = 0


def _matched_fixture(tmp_path, monkeypatch, *, damage=None):
    runner = _Runner(damage=damage)
    actor = {key: value.detach().cpu().clone() for key, value in runner.alg.actor.state_dict().items()}
    critic = {key: value.detach().cpu().clone() for key, value in runner.alg.critic.state_dict().items()}
    if damage == "control-actor":
        actor["params.0"].add_(1)
    if damage == "control-critic":
        critic["params.0"].add_(1)
    if damage == "control-normalizer-affine":
        actor["obs_normalizer._mean"][0].add_(1)
    optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
    optimizer["param_groups"][0]["lr"] = profiles.STOCK_INITIAL_LR
    if damage == "control-adam":
        optimizer["state"] = {0: {"step": torch.tensor(1.)}}
    if damage == "control-group-field":
        optimizer["param_groups"][0]["weight_decay"] = .1
    if damage == "control-param-order":
        optimizer["param_groups"][0]["params"].reverse()
    saved = {"actor_state_dict": actor, "critic_state_dict": critic,
             "optimizer_state_dict": optimizer, "iter": 0,
             "infos": {"env_state": {"common_step_counter": 0}}}
    if damage == "control-clock":
        saved["iter"] = 1
    if damage == "control-common-step-counter":
        saved["infos"]["env_state"]["common_step_counter"] = 1
    raw_stream = io.BytesIO()
    torch.save(saved, raw_stream)
    raw = raw_stream.getvalue()
    path = tmp_path / "initial.pt"
    path.write_bytes(raw)
    control = profiles.ControlInitial(
        "fixture-stock-control", "fixture-source", hashlib.sha256(raw).hexdigest(), path, 947)
    (tmp_path / "actor-import.json").write_text(json.dumps({
        "artifact_sha256": profiles.pinned.V8.sha256, "ppo_resume": False,
        "actor_class": "mjlab_microduck.v8_actor_seed:FrozenV8Actor",
        "initial_learning_rate": profiles.STOCK_INITIAL_LR,
        "target_iteration": 0, "target_common_step_counter": 0,
        "adam_state_entries": 0, "critic_normalizer_count": 0,
    }))
    monkeypatch.setattr(profiles, "control_initial", lambda mode: control)
    monkeypatch.setattr(profiles, "require_stock_recipe_match", lambda mode: {
        "stock_config_sha256": "1" * 64,
        "cautious_config_sha256": "1" * 64,
        "profile_config_match": True,
    })
    if damage == "wrong-hash":
        monkeypatch.setattr(profiles, "control_initial", lambda mode: profiles.ControlInitial(
            control.tag, control.source, "0" * 64, control.path, control.seed))
    if damage == "actor":
        with torch.no_grad():
            runner.alg.actor.params[0].add_(1)
    if damage == "critic":
        with torch.no_grad():
            runner.alg.critic.params[0].add_(1)
    if damage == "candidate-normalizer-count":
        runner.alg.actor.obs_normalizer.count.fill_(1)
    if damage == "adam":
        runner.alg.optimizer.state[runner.alg.actor.params[0]]["step"] = torch.tensor(1.)
    if damage == "candidate-group-field":
        runner.alg.optimizer.param_groups[0]["weight_decay"] = .1
    if damage == "candidate-param-order":
        original_state_dict = runner.alg.optimizer.state_dict
        def reordered_state_dict():
            state = original_state_dict()
            state["param_groups"][0]["params"].reverse()
            return state
        runner.alg.optimizer.state_dict = reordered_state_dict
    return runner


@pytest.mark.parametrize("damage", ["wrong-hash", "actor", "critic", "adam", "control-adam",
                                     "control-clock", "control-common-step-counter",
                                     "candidate-group-field", "control-group-field",
                                     "candidate-param-order", "control-param-order",
                                     "control-normalizer-affine", "candidate-normalizer-count"])
def test_matched_initialization_rejects_any_difference_beyond_lr(tmp_path, monkeypatch, damage):
    runner = _matched_fixture(tmp_path, monkeypatch, damage=damage)
    with pytest.raises(ValueError):
        profiles.matched_initialization(runner, type("Env", (), {"common_step_counter": 0})(),
                                        "smoke", SOURCE)


def test_matched_initialization_accepts_exact_actor_critic_and_optimizer_except_lr(tmp_path, monkeypatch):
    runner = _matched_fixture(tmp_path, monkeypatch)
    report = profiles.matched_initialization(runner, type("Env", (), {"common_step_counter": 0})(),
                                             "smoke", SOURCE)
    assert report["actor_state_equal"] is report["critic_state_equal"] is True
    assert report["control_learning_rate"] == .001
    assert report["candidate_learning_rate"] == .0001
    assert report["parameter_groups_match_except_lr"] is True
    assert report["trajectory_identity_claim"] is False
    assert report["tensors_in_report"] is False
    assert report["source"] == SOURCE and report["mode"] == "smoke"


def test_matched_initialization_requires_owned_control_checkpoint(tmp_path, monkeypatch):
    runner = _matched_fixture(tmp_path, monkeypatch)
    owner = profiles.control_initial("smoke")
    monkeypatch.setattr(profiles, "control_initial", lambda mode: owner)
    monkeypatch.setattr(profiles.os, "getuid", lambda: owner.path.stat().st_uid + 1)
    with pytest.raises(ValueError, match="owned regular file"):
        profiles.matched_initialization(runner, type("Env", (), {"common_step_counter": 0})(),
                                        "smoke", SOURCE)


def test_state_match_hash_rejects_signed_zero_byte_difference():
    candidate = {"weight": torch.tensor([0.0], dtype=torch.float32)}
    control = {"weight": torch.tensor([-0.0], dtype=torch.float32)}
    assert torch.equal(candidate["weight"], control["weight"])
    with pytest.raises(ValueError, match="byte-identical actor state hash"):
        profiles._assert_state_equal(candidate, control, "actor")


def _cautious_receipt(tmp_path, mode="smoke"):
    worlds, updates, seed = profiles.CAUTIOUS_MODES[mode]
    output = tmp_path / mode
    output.mkdir()
    (output / "initial.pt").write_bytes(b"cautious initial")
    checkpoint = output / f"model_{updates - 1}.pt"
    checkpoint.write_bytes(b"cautious final")
    actor_import = {"ppo_resume": False, "adam_state_entries": 0,
                    "artifact_sha256": profiles.pinned.V8.sha256,
                    "maximum_actor_error": 1e-6, "maximum_export_forward_error": 1e-6,
                    "initial_learning_rate": profiles.CAUTIOUS_INITIAL_LR}
    (output / "actor-import.json").write_text(json.dumps(actor_import))
    matched = {"profile": "cautious", "mode": mode, "source": SOURCE, "seed": seed,
               "protocol": "v8-matched-initialization-v1",
               "control_iteration": 0, "candidate_iteration": 0,
               "control_common_step_counter": 0, "candidate_common_step_counter": 0,
               "control_learning_rate": profiles.STOCK_INITIAL_LR,
               "candidate_learning_rate": profiles.CAUTIOUS_INITIAL_LR,
               "ppo_resume": False, "tensors_in_report": False,
               "actor_state_equal": True, "critic_state_equal": True,
               "optimizer_state_empty": True, "parameter_groups_match_except_lr": True,
               "trajectory_identity_claim": False, "policy_acceptance": False,
               "physical_motion_authorized": False}
    (output / "matched-initialization.json").write_text(json.dumps(matched))
    first_kl = profiles.first_update_kl_report(
        mode, SOURCE, {"optimizer_minibatches": 20, "kl_mean_mean": .01,
                       "kl_mean_max": .02}, steps=24)
    (output / "first-update-kl.json").write_text(json.dumps(first_kl))
    result = {"protocol": profiles.CAUTIOUS_PROTOCOL, "profile": "cautious", "source": SOURCE,
              "status": "training-complete-not-accepted", "worlds": worlds,
              "updates": updates, "seed": seed, "steps": updates * 24,
              "transitions": worlds * updates * 24, "checkpoint": str(checkpoint),
              "checkpoint_sha256": base.sha256(checkpoint),
              "initial_checkpoint_sha256": base.sha256(output / "initial.pt"),
              "actor_import": actor_import, "initial_learning_rate": profiles.CAUTIOUS_INITIAL_LR,
              "matched_initialization": matched, "first_update_kl": first_kl,
              "elapsed_s": 60., **dict.fromkeys(base.FULL_COMMAND_COVERAGE, 1)}
    (output / "result.json").write_text(json.dumps(result))
    return output, result


def test_cautious_training_receipt_binds_profile_initialization_and_kl(tmp_path):
    output, expected = _cautious_receipt(tmp_path)
    assert learning.validate_training_receipt(tmp_path, "smoke", SOURCE, profile="cautious") == expected

    damaged = copy.deepcopy(expected)
    damaged["first_update_kl"]["source"] = "c" * 40
    (output / "result.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError):
        learning.validate_training_receipt(tmp_path, "smoke", SOURCE, profile="cautious")

