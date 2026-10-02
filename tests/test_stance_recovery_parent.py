"""CPU-hidden parent-loader checks with explicitly synthetic pinned bytes only."""

from copy import deepcopy
from hashlib import sha256
import io

import pytest
import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_parent as parent


@pytest.fixture(scope="module")
def synthetic_export():
    """Construct schema-valid test bytes; this is not the historical archive."""
    identity = parent.expected_identity()
    actor, critic = checkpoint.validate_identity(identity, evaluation="lean-replication")
    states = checkpoint.states_of(actor, critic)
    payload = {"identity": identity, "states": states}
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    return raw, identity, checkpoint.state_hash(states)


def pin_test_only(monkeypatch, raw, state_hash):
    """Pin synthetic content only inside one isolated test."""
    digest = sha256(raw).hexdigest()
    monkeypatch.setattr(parent.baseline, "CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(parent, "PARENT_STATE_SHA256", state_hash)
    monkeypatch.setattr(parent.profile, "checked_receipt", lambda: {
        "test_only_synthetic_profile_receipt": True})
    return digest


def roundtrip_mutation(raw, mutate):
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    mutate(payload)
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def test_expected_identity_is_exact_pinned_model255_and_returns_owned_architecture_copy():
    identity = parent.expected_identity()
    assert identity == dict(protocol="football-b1n-evaluation-checkpoint-v1",
        source="be2d59661af293b0d67ae20d2e16db50514cce14", purpose="lean-replication",
        training_seed=577, worlds=64, iteration=255,
        runtime_sha256="ef01ea829b5b69069cf732b6e252ee0e7c205e92e5f17f028753ab9ed08682cb",
        training_launch_sha256="8236806aa44c1d424b77e74942d1e01b6530702a55216186165856a870f48455",
        initial_state_sha256="0ca246873f1143c540ecdebb6e6bc80cb826f86b558dda9c3cff8c5694263144",
        parent_checkpoint_sha256="46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93",
        architecture=dict(actor_dim=44, critic_dim=50, action_dim=10,
            hidden_dims=[128, 128, 64], activation="elu", obs_normalization=False,
            gaussian_std_type="scalar", initial_std=.3, evaluation_output="deterministic-mean"))
    changed = parent.expected_identity()
    changed["architecture"]["hidden_dims"][0] = 1
    assert parent.expected_identity()["architecture"]["hidden_dims"] == [128, 128, 64]


def test_parent_load_builds_two_exact_trainable_cpu_groups_without_resume_state(monkeypatch, synthetic_export):
    raw, _identity, state_hash = synthetic_export
    pin_test_only(monkeypatch, raw, state_hash)
    result = parent.load_parent(raw)
    actor, critic, receipt = result["actor"], result["critic"], result["receipt"]
    assert actor is not critic
    assert actor.obs_dim == 44 and critic.obs_dim == 50
    assert actor.distribution.std_param.shape == (10,)
    assert torch.isfinite(actor.distribution.std_param).all() and (actor.distribution.std_param > 0).all()
    for model in (actor, critic):
        assert model.training is True
        assert all(param.device.type == "cpu" and param.dtype == torch.float32
                   and param.requires_grad and torch.isfinite(param).all()
                   for param in model.parameters())
    assert receipt["parent_identity"] == parent.expected_identity()
    assert receipt["parent_state_sha256"] == state_hash
    assert receipt["strict_actor_restore"] is True and receipt["strict_critic_restore"] is True
    assert receipt["models_trainable"] is True and receipt["weights_only"] is True
    assert receipt["fresh_optimizer_required"] is True
    for flag in ("optimizer_created", "optimizer_restored", "simulator_restored", "storage_restored",
                 "rng_restored", "normalization_restored", "historical_archive_authenticated_by_loader",
                 "installed_in_learner", "execution_admitted"):
        assert receipt[flag] is False
    assert all(receipt[flag] is False for flag in parent.baseline.FALSE_FLAGS)
    assert receipt["cpu_math_profile"] == {"test_only_synthetic_profile_receipt": True}


def test_hash_gate_precedes_profile_deserialization_and_model_construction(monkeypatch, synthetic_export):
    raw, _identity, _state_hash = synthetic_export
    monkeypatch.setattr(parent.baseline, "CHECKPOINT_SHA256", "0" * 64)
    events = []
    monkeypatch.setattr(parent.profile, "checked_receipt", lambda: events.append("profile"))
    monkeypatch.setattr(parent.checkpoint, "validate_identity", lambda *_a, **_k: events.append("models"))
    monkeypatch.setattr(parent.torch, "load", lambda *_a, **_k: events.append("deserialize"))
    with pytest.raises(ValueError, match="exact D1 parent byte hash"):
        parent.load_parent(raw)
    assert events == []


@pytest.mark.parametrize("cuda_visible,initialized", [("0", False), ("", True)])
def test_hidden_uninitialized_cuda_gate_precedes_profile_and_model_work(
        monkeypatch, synthetic_export, cuda_visible, initialized):
    raw, _identity, _state_hash = synthetic_export
    events = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible)
    monkeypatch.setattr(parent.torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(parent.profile, "checked_receipt", lambda: events.append("profile"))
    monkeypatch.setattr(parent.checkpoint, "validate_identity", lambda *_a, **_k: events.append("models"))
    monkeypatch.setattr(parent.torch, "load", lambda *_a, **_k: events.append("deserialize"))
    with pytest.raises(ValueError, match="CUDA-hidden uninitialized CPU parent preparation"):
        parent.load_parent(raw)
    assert events == []


@pytest.mark.parametrize("raw", [b"", "not bytes", bytearray(b"bytes")])
def test_parent_loader_requires_nonempty_exact_bytes(monkeypatch, synthetic_export, raw):
    good_raw, _identity, _state_hash = synthetic_export
    pin_test_only(monkeypatch, good_raw, "f" * 64)
    with pytest.raises(ValueError, match="bounded parent export bytes"):
        parent.load_parent(raw)


def test_parent_size_limit_precedes_hash_profile_or_deserialization(monkeypatch, synthetic_export):
    raw, _identity, _state_hash = synthetic_export
    pin_test_only(monkeypatch, raw, "f" * 64)
    monkeypatch.setattr(parent.checkpoint, "LIMIT", len(raw)-1)
    events = []
    monkeypatch.setattr(parent.profile, "checked_receipt", lambda: events.append("profile"))
    monkeypatch.setattr(parent.torch, "load", lambda *_a, **_k: events.append("deserialize"))
    with pytest.raises(ValueError, match="bounded parent export bytes"):
        parent.load_parent(raw)
    assert events == []


def test_actual_profile_receipt_gate_precedes_model_construction_and_load(monkeypatch, synthetic_export):
    raw, _identity, state_hash = synthetic_export
    pin_test_only(monkeypatch, raw, state_hash)
    events = []
    load_kwargs = []
    original_validate = parent.checkpoint.validate_identity
    original_load = parent.torch.load
    def checked_profile():
        events.append("profile")
        return {"explicit_test_profile_receipt": True}
    def validate(*args, **kwargs):
        events.append("models")
        return original_validate(*args, **kwargs)
    def load(*args, **kwargs):
        events.append("deserialize")
        load_kwargs.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(parent.profile, "checked_receipt", checked_profile)
    monkeypatch.setattr(parent.checkpoint, "validate_identity", validate)
    monkeypatch.setattr(parent.torch, "load", load)
    result = parent.load_parent(raw)
    assert events == ["profile", "models", "deserialize"]
    assert load_kwargs == [{"map_location": "cpu", "weights_only": True}]
    assert result["receipt"]["cpu_math_profile"] == {"explicit_test_profile_receipt": True}


def test_profile_failure_stops_before_model_creation_or_deserialization(monkeypatch, synthetic_export):
    raw, _identity, _state_hash = synthetic_export
    monkeypatch.setattr(parent.baseline, "CHECKPOINT_SHA256", sha256(raw).hexdigest())
    events = []
    def profile_failure():
        events.append("profile")
        raise ValueError("profile receipt unavailable")
    monkeypatch.setattr(parent.profile, "checked_receipt", profile_failure)
    monkeypatch.setattr(parent.checkpoint, "validate_identity", lambda *_a, **_k: events.append("models"))
    monkeypatch.setattr(parent.torch, "load", lambda *_a, **_k: events.append("deserialize"))
    with pytest.raises(ValueError, match="profile receipt unavailable"):
        parent.load_parent(raw)
    assert events == ["profile"]


@pytest.mark.parametrize("damage", ["metadata", "missing-group", "extra-group", "missing-weight",
    "actor-shape", "critic-shape", "dtype", "nonfinite", "std-zero", "state-hash", "optimizer"])
def test_rehashed_test_only_malformed_parent_exports_fail_closed(monkeypatch, synthetic_export, damage):
    raw, _identity, state_hash = synthetic_export
    def mutate(value):
        actor, critic = value["states"]["actor"], value["states"]["critic"]
        if damage == "metadata": value["identity"]["iteration"] = 256
        elif damage == "missing-group": value["states"].pop("critic")
        elif damage == "extra-group": value["states"]["optimizer"] = {}
        elif damage == "missing-weight": actor.pop("mlp.0.bias")
        elif damage == "actor-shape": actor["mlp.0.weight"] = torch.zeros(128, 61)
        elif damage == "critic-shape": critic["mlp.0.weight"] = torch.zeros(128, 44)
        elif damage == "dtype": actor["mlp.0.bias"] = actor["mlp.0.bias"].double()
        elif damage == "nonfinite": critic["mlp.0.bias"][0] = float("nan")
        elif damage == "std-zero": actor["distribution.std_param"][0] = 0.
        elif damage == "state-hash": actor["mlp.0.bias"][0] += 1.
        elif damage == "optimizer": value["optimizer"] = {"state": {}}
    altered = roundtrip_mutation(raw, mutate)
    # Rehash only for this test so deeper schema checks can be reached; this is
    # not a caller-controlled production mechanism or historical authenticity.
    pin_test_only(monkeypatch, altered, state_hash)
    with pytest.raises(ValueError):
        parent.load_parent(altered)


def test_loader_does_not_mutate_existing_models_or_cpu_rng_and_never_resumes(monkeypatch, synthetic_export):
    raw, _identity, state_hash = synthetic_export
    pin_test_only(monkeypatch, raw, state_hash)
    existing_actor, existing_critic = checkpoint.fresh_models(577)
    before_existing = checkpoint.state_hash(checkpoint.states_of(existing_actor, existing_critic))
    before_rng = torch.random.get_rng_state().clone()
    result = parent.load_parent(raw)
    after_rng = torch.random.get_rng_state().clone()
    after_existing = checkpoint.state_hash(checkpoint.states_of(existing_actor, existing_critic))
    assert before_existing == after_existing
    assert torch.equal(before_rng, after_rng)
    assert result["actor"] is not existing_actor and result["critic"] is not existing_critic
    assert result["receipt"]["optimizer_created"] is False
    assert result["receipt"]["optimizer_restored"] is False
    assert result["receipt"]["storage_restored"] is False
    assert result["receipt"]["simulator_restored"] is False
    assert result["receipt"]["rng_restored"] is False
    assert result["receipt"]["normalization_restored"] is False


def test_recovery_parent_loader_preserves_historical_checkpoint_allowlists():
    from mjlab_microduck import stance_attempt_trace as trace
    assert trace.CHECKPOINTS == (128, 256, 384, 511)
    assert trace.LEAN_CHECKPOINTS == (64, 128, 192, 255)
    assert trace.LEAN_REPLICATION_CHECKPOINTS == trace.LEAN_CHECKPOINTS
    assert checkpoint.EVALUABLE == {
        "pilot": trace.CHECKPOINTS,
        "lean-lesson": trace.LEAN_CHECKPOINTS,
        "lean-replication": trace.LEAN_REPLICATION_CHECKPOINTS,
    }
