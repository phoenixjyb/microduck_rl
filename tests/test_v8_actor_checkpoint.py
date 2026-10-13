"""Exact-byte actor-only checkpoint admission; CPU fixtures only."""

from __future__ import annotations

import hashlib
import io
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import normalized_onnx_reference as reference
from mjlab_microduck import v8_actor_checkpoint as checkpoint_module
from mjlab_microduck.v8_actor_seed import FrozenV8Actor


SHA = "a" * 64


def _affine():
    return {
        "obs_normalizer._mean": torch.linspace(-.4, .8, 61, dtype=torch.float32)[None, :],
        "onnx::Div_24": torch.linspace(.3, 1.7, 61, dtype=torch.float32)[None, :],
    }


def _actor_state(monkeypatch):
    affine = _affine()
    monkeypatch.setattr(reference, "OFFICIAL_V8_SHA256", SHA)
    monkeypatch.setattr(checkpoint_module, "_pinned_affine", lambda: affine)
    obs = TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1])
    actor = FrozenV8Actor(
        obs, {"actor": ("actor",)}, "actor", 14,
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": .3, "std_type": "scalar",
        },
    )
    with torch.no_grad():
        actor.obs_normalizer._mean.copy_(affine["obs_normalizer._mean"])
        actor.obs_normalizer._std.copy_(affine["onnx::Div_24"])
        actor.obs_normalizer._var.copy_(affine["onnx::Div_24"].square())
        actor.obs_normalizer.count.zero_()
        actor.v8_seed_digest.copy_(torch.tensor(list(bytes.fromhex(SHA)), dtype=torch.uint8))
    return {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}


def _write_checkpoint(tmp_path, state):
    buffer = io.BytesIO()
    torch.save({"actor_state_dict": state}, buffer)
    raw = buffer.getvalue()
    path = tmp_path / "frozen-v8.pt"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _checkpoint(monkeypatch, tmp_path, state=None):
    state = _actor_state(monkeypatch) if state is None else state
    path, digest = _write_checkpoint(tmp_path, state)
    return checkpoint_module.FrozenV8Checkpoint(path, digest)


def test_exact_fixture_affine_and_seeded_actor_state_are_admitted(monkeypatch, tmp_path):
    state = _actor_state(monkeypatch)
    path, digest = _write_checkpoint(tmp_path, state)
    admitted = checkpoint_module.FrozenV8Checkpoint(path, digest)

    assert admitted.path == path.resolve()
    assert admitted.sha256 == digest
    assert admitted.provenance()["backend"] == "frozen-v8-ppo-actor-only"
    assert admitted.provenance()["ppo_resume"] is False
    assert admitted.provenance()["optimizer_state_imported"] is False
    assert admitted.provenance()["physical_motion_authorized"] is False
    path.write_bytes(b"replacement bytes after admission")
    assert not admitted.matches_positional_checkpoint(path)


def test_construct_actor_loads_only_verified_actor_and_keeps_empty_frozen_normalizer(
        monkeypatch, tmp_path):
    state = _actor_state(monkeypatch)
    state["distribution.std_param"].fill_(.42)
    path, digest = _write_checkpoint(tmp_path, state)
    admitted = checkpoint_module.FrozenV8Checkpoint(path, digest)
    actor_cfg = SimpleNamespace(hidden_dims=(512, 256, 128), activation="elu",
                                 obs_normalization=True, class_name="stock",
                                 distribution_cfg=None)
    agent = SimpleNamespace(actor=actor_cfg, obs_groups={"actor": ("actor",)})
    observations = TensorDict({"actor": torch.zeros(2, 61)}, batch_size=[2])

    actor = admitted.construct_actor(observations, agent)
    output = actor(observations)

    assert type(actor) is FrozenV8Actor and not actor.training
    assert actor_cfg.class_name == "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
    assert actor.distribution.std_type == "scalar"
    assert torch.allclose(actor.distribution.std_param,
                          torch.full((14,), .42), atol=0., rtol=0.)
    assert actor.obs_normalizer.count.item() == 0
    assert actor.obs_normalizer.eps == 0. and actor.obs_normalizer.until == 0
    assert output.shape == (2, 14) and torch.isfinite(output).all()


@pytest.mark.parametrize("damage", [
    "wrong-seed", "nonfinite", "shape", "normalizer-count", "affine", "float-count",
])
def test_actor_state_damage_is_rejected_atomically(monkeypatch, tmp_path, damage):
    state = _actor_state(monkeypatch)
    damaged = {name: value.clone() for name, value in state.items()}
    if damage == "wrong-seed":
        damaged["v8_seed_digest"].zero_()
    elif damage == "nonfinite":
        damaged["mlp.0.weight"][0, 0] = float("nan")
    elif damage == "shape":
        damaged["mlp.6.bias"] = damaged["mlp.6.bias"][:-1]
    elif damage == "normalizer-count":
        damaged["obs_normalizer.count"].fill_(1.)
    elif damage == "affine":
        damaged["obs_normalizer._std"][0, 0] += .01
    elif damage == "float-count":
        damaged["obs_normalizer.count"] = torch.tensor(0., dtype=torch.float32)
    path, digest = _write_checkpoint(tmp_path, damaged)

    with pytest.raises(ValueError):
        checkpoint_module.FrozenV8Checkpoint(path, digest)


def test_wrong_checkpoint_hash_and_symlink_are_rejected(monkeypatch, tmp_path):
    state = _actor_state(monkeypatch)
    path, digest = _write_checkpoint(tmp_path, state)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checkpoint_module.FrozenV8Checkpoint(path, "0" * 64)

    link = tmp_path / "linked.pt"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        checkpoint_module.FrozenV8Checkpoint(link, digest)


def test_evaluator_rejects_frozen_onnx_and_task_override_before_env_setup(monkeypatch, tmp_path):
    import mjlab.envs

    path = tmp_path / "frozen-v8.pt"
    path.write_bytes(b"unused by mutual exclusion check")
    typed = object.__new__(checkpoint_module.FrozenV8Checkpoint)
    object.__setattr__(typed, "path", path.resolve())
    object.__setattr__(typed, "sha256", "b" * 64)
    object.__setattr__(typed, "_payload", b"")
    object.__setattr__(typed, "_actor_state", MappingProxyType({}))
    object.__setattr__(typed, "_pinned_sha256", reference.OFFICIAL_V8_SHA256)
    constructed = []
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv",
                        lambda *_args, **_kwargs: constructed.append(True))

    with pytest.raises(ValueError, match="exclusive with ONNX and task overrides"):
        base.evaluate_case(path, 839, .3, 0., lambda: None,
                           onnx_reference=object(),
                           evaluation_task_override="Mjlab-Velocity-Flat-Backlash-MicroDuck",
                           frozen_v8_checkpoint=typed)
    assert constructed == []


def test_wrong_frozen_checkpoint_type_is_rejected_before_env_setup(monkeypatch, tmp_path):
    import mjlab.envs

    constructed = []
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv",
                        lambda *_args, **_kwargs: constructed.append(True))
    with pytest.raises(ValueError, match="typed FrozenV8Checkpoint"):
        base.evaluate_case(tmp_path / "unused.pt", 839, .3, 0., lambda: None,
                           frozen_v8_checkpoint=object())
    assert constructed == []


def test_initial_baseline_requires_original_mean_weights_not_a_trained_actor(monkeypatch, tmp_path):
    state = _actor_state(monkeypatch)
    pinned = _affine() | {k:v.clone() for k,v in state.items() if k.startswith("mlp.")}
    monkeypatch.setattr(checkpoint_module, "_pinned_affine", lambda: pinned)
    admitted = _checkpoint(monkeypatch, tmp_path, state=state)
    admitted.require_initial_seed()
    changed = {k:v.clone() for k,v in state.items()}
    changed["mlp.6.bias"][0] += .01
    path, digest = _write_checkpoint(tmp_path, changed)
    trained = checkpoint_module.FrozenV8Checkpoint(path, digest)
    with pytest.raises(ValueError, match="initial baseline"):
        trained.require_initial_seed()
