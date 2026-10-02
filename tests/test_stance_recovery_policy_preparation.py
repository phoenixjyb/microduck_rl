"""Real local PPO/storage wiring checks with explicitly synthetic parent weights.

These tests do not authenticate or load the historical parent archive and are
not evidence of training, runtime, GPU, or job admission.
"""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_policy_preparation as preparation


SYNTHETIC_RAW = b"synthetic parent bytes; not the selected historical archive"


def _false_flags():
    return dict(contract.FALSE_FLAGS)


def _synthetic_pair(*, nontrainable=False, mutate=False):
    """Build models of the exact architecture; weights are synthetic fixtures."""
    identity = parent.expected_identity()
    actor, critic = checkpoint.validate_identity(identity, evaluation="lean-replication")
    if mutate:
        with torch.no_grad():
            next(actor.parameters()).add_(0.125)
    actor.train().requires_grad_(True)
    critic.train().requires_grad_(True)
    if nontrainable:
        next(actor.parameters()).requires_grad_(False)
    state_sha = checkpoint.state_hash(checkpoint.states_of(actor, critic))
    receipt = dict(protocol=parent.PROTOCOL, parent_checkpoint_sha256="synthetic-only",
        parent_identity=identity, parent_state_sha256=state_sha,
        cpu_math_profile={"test_only_synthetic_profile_receipt": True},
        strict_actor_restore=True, strict_critic_restore=True, models_trainable=True,
        weights_only=True, fresh_optimizer_required=True, optimizer_created=False,
        optimizer_restored=False, simulator_restored=False, storage_restored=False,
        rng_restored=False, normalization_restored=False, historical_archive_authenticated_by_loader=False,
        installed_in_learner=False, execution_admitted=False, **_false_flags())
    return actor, critic, state_sha, receipt


def _install_parent(monkeypatch, *, nontrainable=False, mutate=False):
    actor, critic, state_sha, receipt = _synthetic_pair(nontrainable=nontrainable, mutate=mutate)
    monkeypatch.setattr(parent, "PARENT_STATE_SHA256", state_sha)
    calls = []
    def load(raw):
        calls.append(raw)
        return {"actor": actor, "critic": critic, "receipt": deepcopy(receipt)}
    monkeypatch.setattr(preparation.parent, "load_parent", load)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(preparation.torch.cuda, "is_initialized", lambda: False)
    return actor, critic, state_sha, receipt, calls


@pytest.mark.parametrize("seed", [653, 659])
@pytest.mark.parametrize("worlds", [2, 64])
def test_prepare_policy_wires_real_models_storage_and_fresh_adam_without_rng_mutation(
        monkeypatch, seed, worlds):
    actor, critic, state_sha, parent_receipt, calls = _install_parent(monkeypatch)
    before_rng = torch.random.get_rng_state().clone()
    result = preparation.prepare_policy(SYNTHETIC_RAW, seed=seed, worlds=worlds)
    assert calls == [SYNTHETIC_RAW]
    assert result["actor"] is actor and result["critic"] is critic
    assert result["algorithm"].actor is actor and result["algorithm"].critic is critic
    assert result["algorithm"].storage is result["storage"]
    assert type(result["algorithm"]) is preparation.FinitePPO
    assert type(result["storage"]) is preparation.RolloutStorage
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert torch.equal(result["learner_rng_state"], torch.Generator(device="cpu").manual_seed(seed).get_state())

    storage = result["storage"]
    assert storage.device == "cpu" and storage.num_envs == worlds
    assert storage.num_transitions_per_env == preparation.STEPS == 24
    assert storage.actions_shape == (10,) and storage.step == 0
    assert storage.distribution_params is None
    assert all(torch.count_nonzero(tensor) == 0 for tensor in storage.observations.values())
    for name in ("rewards", "actions", "dones", "values", "actions_log_prob", "returns", "advantages"):
        assert torch.count_nonzero(getattr(storage, name)) == 0

    optimizer = result["algorithm"].optimizer
    assert type(optimizer) is torch.optim.Adam and optimizer.state == {}
    actor_ids = [id(parameter) for parameter in actor.parameters()]
    critic_ids = [id(parameter) for parameter in critic.parameters()]
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert set(actor_ids).isdisjoint(critic_ids)
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == set(actor_ids + critic_ids)
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 and parameter.requires_grad
               for parameter in (*actor.parameters(), *critic.parameters()))

    receipt = result["receipt"]
    assert receipt["protocol"] == preparation.PROTOCOL
    assert receipt["learner_seed"] == seed and receipt["worlds"] == worlds
    assert receipt["initial_state_sha256"] == state_sha == parent_receipt["parent_state_sha256"]
    assert receipt["parent_preparation"] == parent_receipt
    assert receipt["storage_step"] == 0 and receipt["optimizer_state_entries"] == 0
    assert receipt["optimizer_steps"] == 0 and receipt["cuda_initialized"] is False
    for flag in ("policy_objects_wired", "exact_optimizer_parameter_binding", "global_cpu_rng_unchanged"):
        assert receipt[flag] is True
    for flag in ("simulator_created", "schedule_installed", "transition_bridge_qualified",
        "learner_rng_connected_to_sampler",
        "finite_optimizer_step_qualified", "student_export_available", "training_job_predeclared",
        "execution_admitted", *contract.FALSE_FLAGS):
        assert receipt[flag] is False


def test_prepare_calls_parent_loader_once_and_each_call_gets_distinct_fresh_policy_storage(monkeypatch):
    pairs = [_synthetic_pair(), _synthetic_pair()]
    monkeypatch.setattr(parent, "PARENT_STATE_SHA256", pairs[0][2])
    calls = []
    def load(raw):
        index = len(calls); calls.append(raw)
        actor, critic, state_sha, receipt = pairs[index]
        assert state_sha == pairs[0][2]
        return {"actor": actor, "critic": critic, "receipt": deepcopy(receipt)}
    monkeypatch.setattr(preparation.parent, "load_parent", load)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(preparation.torch.cuda, "is_initialized", lambda: False)
    first = preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)
    second = preparation.prepare_policy(SYNTHETIC_RAW, seed=659, worlds=2)
    assert calls == [SYNTHETIC_RAW, SYNTHETIC_RAW]
    assert first["actor"] is not second["actor"] and first["critic"] is not second["critic"]
    assert first["storage"] is not second["storage"] and first["algorithm"].optimizer is not second["algorithm"].optimizer
    assert first["learner_rng_state"].equal(second["learner_rng_state"]) is False


@pytest.mark.parametrize("seed,worlds", [
    (True, 2), (653.0, 2), (521, 2), (577, 2), (660, 2),
    (653, True), (653, 2.0), (653, 3), (653, 8), (653, 512),
])
def test_invalid_seed_or_world_refuses_before_parent_load(monkeypatch, seed, worlds):
    calls = []
    monkeypatch.setattr(preparation.parent, "load_parent", lambda _raw: calls.append(True))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(preparation.torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(ValueError):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=seed, worlds=worlds)
    assert calls == []


@pytest.mark.parametrize("visible,initialized", [("0", False), ("", True)])
def test_cuda_gate_refuses_before_parent_loader(monkeypatch, visible, initialized):
    calls = []
    monkeypatch.setattr(preparation.parent, "load_parent", lambda _raw: calls.append(True))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(preparation.torch.cuda, "is_initialized", lambda: initialized)
    with pytest.raises(ValueError, match="CUDA-hidden CPU recovery policy preparation"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)
    assert calls == []


def test_pinned_ppo_storage_source_hashes_match_local_installed_files():
    root = preparation.distribution("rsl-rl-lib").locate_file("rsl_rl")
    assert {name: sha256((root / name).read_bytes()).hexdigest() for name in preparation.PINS} == preparation.PINS


def test_source_pin_refusal_precedes_real_storage_and_ppo_construction(monkeypatch, tmp_path):
    _install_parent(monkeypatch)
    (tmp_path / "algorithms").mkdir()
    (tmp_path / "storage").mkdir()
    (tmp_path / "algorithms/ppo.py").write_bytes(b"changed synthetic PPO source")
    (tmp_path / "storage/rollout_storage.py").write_bytes(b"changed synthetic storage source")
    monkeypatch.setattr(preparation, "distribution", lambda _name: SimpleNamespace(locate_file=lambda _p: tmp_path))
    ppo_calls, storage_calls = [], []
    monkeypatch.setattr(preparation, "FinitePPO", lambda *args, **kwargs: ppo_calls.append(True))
    original_storage = preparation.RolloutStorage
    monkeypatch.setattr(preparation, "RolloutStorage", lambda *args, **kwargs: (
        storage_calls.append(True), original_storage(*args, **kwargs))[1])
    with pytest.raises(ValueError, match="unchanged reviewed PPO and storage source bytes"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)
    assert ppo_calls == [] and storage_calls == []


def test_optimizer_parameter_id_mismatch_is_rejected(monkeypatch):
    _install_parent(monkeypatch)
    original = preparation.FinitePPO
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.optimizer.param_groups[0]["params"].pop()
        return result
    monkeypatch.setattr(preparation, "FinitePPO", corrupt)
    with pytest.raises(ValueError, match="exact unique actor/critic optimizer parameter binding"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)


def test_nonempty_optimizer_state_is_rejected(monkeypatch):
    _install_parent(monkeypatch)
    original = preparation.FinitePPO
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        parameter = result.optimizer.param_groups[0]["params"][0]
        result.optimizer.state[parameter]["test_marker"] = torch.tensor(1.)
        return result
    monkeypatch.setattr(preparation, "FinitePPO", corrupt)
    with pytest.raises(ValueError, match="fresh empty declared Adam without restored moments"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)


def test_nonempty_rollout_storage_is_rejected(monkeypatch):
    _install_parent(monkeypatch)
    original = preparation.RolloutStorage
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.step = 1
        return result
    monkeypatch.setattr(preparation, "RolloutStorage", corrupt)
    with pytest.raises(ValueError, match="fresh empty CPU rollout storage"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)


def test_parent_hash_mismatch_and_nontrainable_model_are_rejected(monkeypatch):
    _install_parent(monkeypatch, mutate=True)
    monkeypatch.setattr(parent, "PARENT_STATE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="actual policy starts at the exact reviewed parent weights"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)

    _install_parent(monkeypatch, nontrainable=True)
    with pytest.raises(ValueError, match="trainable float32 CPU policy groups"):
        preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)


def test_detects_constructor_global_rng_mutation_and_restores_test_rng(monkeypatch):
    _actor, _critic, _state, _receipt, _calls = _install_parent(monkeypatch)
    original_rng = torch.random.get_rng_state().clone()
    original_loader = preparation.parent.load_parent
    def corrupt_loader(raw):
        result = original_loader(raw)
        torch.random.default_generator.manual_seed(123456)
        return result
    monkeypatch.setattr(preparation.parent, "load_parent", corrupt_loader)
    try:
        with pytest.raises(ValueError, match="preparation preserves caller CPU RNG"):
            preparation.prepare_policy(SYNTHETIC_RAW, seed=653, worlds=2)
    finally:
        torch.random.set_rng_state(original_rng)


def test_preparation_does_not_expand_historical_fresh_initializer_or_checkpoint_allowlists():
    assert checkpoint.FRESH_SEEDS == (521, 523, 563, 571, 577, 587, 593)
    assert checkpoint.PURPOSES == ("pilot", "smoke", "eager-learning", "lean-lesson", "lean-replication")
    assert preparation.lesson.TRAINING_SEEDS == (653, 659)
    assert set(preparation.lesson.TRAINING_SEEDS).isdisjoint(checkpoint.FRESH_SEEDS)
    assert parent.expected_identity()["training_seed"] == 577
