"""CPU-only contracts for pinned v8 actor import; no simulator or training run."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from mjlab_microduck import normalized_onnx_reference as reference
from mjlab_microduck.v8_actor_seed import FrozenV8Actor, import_v8_actor


DIST = {
    "class_name": "rsl_rl.modules:GaussianDistribution",
    "init_std": 0.35,
    "std_type": "scalar",
}
SHAPES = {
    "mlp.0.weight": (512, 61), "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512), "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256), "mlp.4.bias": (128,),
    "mlp.6.weight": (14, 128), "mlp.6.bias": (14,),
}


def _model():
    rng = np.random.default_rng(718)
    initializers = [
        numpy_helper.from_array(np.linspace(-0.7, 0.9, 61, dtype=np.float32)[None, :],
                                "obs_normalizer._mean"),
        numpy_helper.from_array(np.linspace(0.35, 1.8, 61, dtype=np.float32)[None, :],
                                "onnx::Div_24"),
    ]
    for name, shape in SHAPES.items():
        if name.endswith("weight"):
            value = rng.normal(0, 0.025, size=shape).astype(np.float32)
        else:
            value = rng.normal(0, 0.01, size=shape).astype(np.float32)
        initializers.append(numpy_helper.from_array(value, name))

    nodes = [helper.make_node("Sub", ["obs", "obs_normalizer._mean"], ["centered"]),
             helper.make_node("Div", ["centered", "onnx::Div_24"], ["normalized"])]
    previous = "normalized"
    for index, (weight, bias, width) in enumerate((
        ("mlp.0.weight", "mlp.0.bias", 512),
        ("mlp.2.weight", "mlp.2.bias", 256),
        ("mlp.4.weight", "mlp.4.bias", 128),
        ("mlp.6.weight", "mlp.6.bias", 14),
    )):
        output = "actions" if index == 3 else f"linear_{index}"
        nodes.append(helper.make_node("Gemm", [previous, weight, bias], [output], transB=1))
        if index < 3:
            activated = f"elu_{index}"
            nodes.append(helper.make_node("Elu", [output], [activated], alpha=1.0))
            previous = activated

    graph = helper.make_graph(
        nodes, "synthetic-pinned-v8-actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializers,
    )
    model = helper.make_model(graph, ir_version=8,
                              opset_imports=[helper.make_opsetid("", 18)])
    helper.set_model_props(model, {
        "observation_names": reference.OBSERVATION_NAMES_CSV,
        "command_names": reference.COMMAND_NAMES_CSV,
        "joint_names": reference.JOINT_NAMES_CSV,
        "action_scale": "1.0",
        "default_joint_pos": "0.000,-0.087,-0.458,-0.005,0.453,0.349,0.349,0.000,0.000,0.000,0.087,0.458,0.005,-0.453",
    })
    return model


def _payload(model=None):
    return (model or _model()).SerializeToString()


def _write_payload(tmp_path, payload, name="actor.onnx"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _actor(*, obs=None, groups=None, obs_set="actor", output_dim=14,
           hidden_dims=(512, 256, 128), activation="elu",
           obs_normalization=True, distribution_cfg=None):
    observations = obs or TensorDict({"actor": torch.zeros((2, 61))}, batch_size=[2])
    return FrozenV8Actor(
        observations, groups or {"actor": ["actor"]}, obs_set, output_dim,
        hidden_dims=hidden_dims, activation=activation,
        obs_normalization=obs_normalization,
        distribution_cfg=copy.deepcopy(DIST if distribution_cfg is None else distribution_cfg),
    )


def _import(monkeypatch, tmp_path, payload=None, actor=None, probe=None):
    payload = _payload() if payload is None else payload
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(reference, "OFFICIAL_V8_SHA256", digest)
    path = _write_payload(tmp_path, payload)
    actor = actor or _actor()
    probe = probe if probe is not None else torch.linspace(-2.5, 3.1, 4 * 61).reshape(4, 61)
    report = import_v8_actor(actor, path, probe)
    return actor, report, payload


def _state_snapshot(actor):
    return {name: value.detach().clone() for name, value in actor.state_dict().items()}


def _assert_state_unchanged(actor, before):
    assert set(actor.state_dict()) == set(before)
    assert all(torch.equal(value, before[name]) for name, value in actor.state_dict().items())


def test_import_matches_raw_onnx_and_stock_rsl_export_on_nonunit_affine(monkeypatch, tmp_path):
    actor, report, payload = _import(monkeypatch, tmp_path)
    probe = torch.linspace(-2.5, 3.1, 4 * 61).reshape(4, 61)
    session = __import__("onnxruntime").InferenceSession(
        payload, providers=["CPUExecutionProvider"])
    expected = torch.from_numpy(np.concatenate([
        session.run(["actions"], {"obs": row[None, :].numpy()})[0]
        for row in probe
    ]))
    exported = actor.as_onnx(False)(probe)

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert actor.obs_normalizer._mean.abs().max() > 0.1
    assert actor.obs_normalizer._std.min() == pytest.approx(0.35)
    assert not torch.allclose(actor.obs_normalizer._std, torch.ones_like(actor.obs_normalizer._std))
    assert torch.allclose(actor(TensorDict({"actor": probe}, batch_size=[4])), expected,
                           atol=2e-5, rtol=0)
    assert torch.allclose(exported, expected, atol=2e-5, rtol=0)
    assert report["maximum_actor_error"] <= 2e-5
    assert report["maximum_export_forward_error"] <= 2e-5
    assert report["normalizer_count"] == 0
    assert report["gaussian_state_imported"] is False


def test_import_is_one_shot_and_unseeded_inference_and_export_are_blocked(monkeypatch, tmp_path):
    actor = _actor()
    raw = TensorDict({"actor": torch.zeros((2, 61))}, batch_size=[2])
    with pytest.raises(ValueError, match="not seeded"):
        actor(raw)
    with pytest.raises(ValueError, match="not seeded"):
        actor.as_onnx(False)
    with pytest.raises(ValueError, match="not seeded"):
        actor.as_jit()

    actor, _, _ = _import(monkeypatch, tmp_path, actor=actor)
    seeded_state = _state_snapshot(actor)
    payload = _payload()
    path = _write_payload(tmp_path, payload, "second.onnx")
    with pytest.raises(ValueError, match="one-shot"):
        import_v8_actor(actor, path, torch.zeros((1, 61)))
    _assert_state_unchanged(actor, seeded_state)


def test_frozen_normalizer_uses_exported_denominator_and_stays_empty_in_training(monkeypatch, tmp_path):
    actor, _, payload = _import(monkeypatch, tmp_path)
    model = onnx.load_model_from_string(payload)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    assert torch.equal(actor.obs_normalizer._std,
                       torch.from_numpy(constants["onnx::Div_24"].copy()))
    assert torch.equal(actor.obs_normalizer._var,
                       torch.from_numpy((constants["onnx::Div_24"] ** 2).copy()))
    assert actor.obs_normalizer.eps == 0.0 and actor.obs_normalizer.until == 0
    before = _state_snapshot(actor)

    actor.train()
    actor.update_normalization(TensorDict({"actor": torch.randn(32, 61) * 9 + 14}, batch_size=[32]))
    assert actor.obs_normalizer.count.item() == 0
    _assert_state_unchanged(actor, before)


def test_custom_class_checkpoint_roundtrip_keeps_frozen_affine_without_source_count(monkeypatch, tmp_path):
    actor, _, _ = _import(monkeypatch, tmp_path)
    state = copy.deepcopy(actor.state_dict())
    restored = _actor()
    assert restored.load_state_dict(state, strict=True).missing_keys == []
    probe = torch.randn(3, 61)
    observations = TensorDict({"actor": probe}, batch_size=[3])

    assert torch.equal(actor.v8_seed_digest, restored.v8_seed_digest)
    assert restored.obs_normalizer.count.item() == 0
    assert restored.obs_normalizer.eps == 0.0 and restored.obs_normalizer.until == 0
    assert torch.equal(actor(observations), restored(observations))
    restored.train()
    restored.update_normalization(observations)
    assert restored.obs_normalizer.count.item() == 0


@pytest.mark.parametrize("change", [
    lambda model: model.graph.node[3].CopyFrom(helper.make_node("Relu", ["linear_0"], ["elu_0"])),
    lambda model: model.graph.node[2].input.__setitem__(0, "obs"),
    lambda model: _replace_initializer(model, "onnx::Div_24", np.full((1, 61), -1, np.float32)),
    lambda model: _replace_initializer(model, "onnx::Div_24", np.full((1, 61), np.inf, np.float32)),
    lambda model: _replace_initializer(model, "onnx::Div_24", np.ones((1, 60), np.float32)),
])
def test_rejected_graphs_leave_all_caller_state_unchanged(monkeypatch, tmp_path, change):
    model = _model()
    change(model)
    payload = _payload(model)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(reference, "OFFICIAL_V8_SHA256", digest)
    path = _write_payload(tmp_path, payload)
    actor = _actor()
    before = _state_snapshot(actor)
    param_ids = tuple(id(param) for param in actor.parameters())
    probe = torch.zeros((2, 61))

    with pytest.raises((ValueError, RuntimeError)):
        import_v8_actor(actor, path, probe)

    _assert_state_unchanged(actor, before)
    assert tuple(id(param) for param in actor.parameters()) == param_ids


def _replace_initializer(model, name, value):
    from onnx import numpy_helper
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name))
            return
    raise AssertionError(f"fixture initializer missing: {name}")


def test_wrong_digest_and_missing_file_are_atomic(monkeypatch, tmp_path):
    payload = _payload()
    monkeypatch.setattr(reference, "OFFICIAL_V8_SHA256", hashlib.sha256(payload).hexdigest())
    actor = _actor()
    before = _state_snapshot(actor)
    path = _write_payload(tmp_path, payload + b"tampered")
    with pytest.raises(ValueError, match="pinned v8 bytes"):
        import_v8_actor(actor, path, torch.zeros((1, 61)))
    _assert_state_unchanged(actor, before)
    with pytest.raises(ValueError, match="could not be opened safely"):
        import_v8_actor(actor, tmp_path / "missing.onnx", torch.zeros((1, 61)))
    _assert_state_unchanged(actor, before)


@pytest.mark.parametrize("probe", [
    torch.zeros((0, 61)), torch.zeros((1, 60)), torch.zeros((1, 513, 61)),
    torch.zeros((1, 61), dtype=torch.float64),
    torch.full((1, 61), float("nan")),
])
def test_bad_raw_probe_is_refused_atomically(monkeypatch, tmp_path, probe):
    payload = _payload()
    monkeypatch.setattr(reference, "OFFICIAL_V8_SHA256", hashlib.sha256(payload).hexdigest())
    path = _write_payload(tmp_path, payload)
    actor = _actor()
    before = _state_snapshot(actor)

    with pytest.raises(ValueError, match="raw float32 CPU probe"):
        import_v8_actor(actor, path, probe)

    _assert_state_unchanged(actor, before)


@pytest.mark.parametrize("kwargs", [
    {"obs_set": "critic"},
    {"groups": {"actor": ["critic"]}},
    {"obs": TensorDict({"actor": torch.zeros(2, 60)}, batch_size=[2])},
    {"output_dim": 13}, {"hidden_dims": (512, 256, 127)},
    {"activation": "relu"}, {"obs_normalization": False},
    {"distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.2, "std_type": "scalar"}},
    {"distribution_cfg": {"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0, "std_type": "scalar"}},
    {"distribution_cfg": {"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": float("nan"), "std_type": "scalar"}},
    {"distribution_cfg": {"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0.2, "std_type": "vector"}},
])
def test_constructor_rejects_noncontract_configs(kwargs):
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        _actor(**kwargs)


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_constructor_accepts_stock_gaussian_std_parameterizations(std_type):
    cfg = {"class_name": "rsl_rl.modules:GaussianDistribution",
           "init_std": 0.2, "std_type": std_type}
    actor = _actor(distribution_cfg=cfg)
    assert isinstance(actor.distribution, GaussianDistribution)
    assert actor.distribution.std_type == std_type


def test_import_preserves_optimizer_ownership_critic_and_allows_gradients(monkeypatch, tmp_path):
    actor = _actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    optimizer_params = tuple(p for group in optimizer.param_groups for p in group["params"])
    ownership = tuple(id(param) for param in optimizer_params)
    critic = torch.nn.Linear(61, 4)
    critic_before = _state_snapshot(critic)
    actor, _, _ = _import(monkeypatch, tmp_path, actor=actor)

    assert tuple(id(param) for group in optimizer.param_groups for param in group["params"]) == ownership
    assert tuple(id(param) for param in actor.parameters()) == ownership
    _assert_state_unchanged(critic, critic_before)
    raw = TensorDict({"actor": torch.randn(5, 61)}, batch_size=[5])
    sampled = actor(raw, stochastic_output=True)
    loss = -actor.get_output_log_prob(sampled).mean() + actor.output_entropy.mean() * 0.01
    loss.backward()
    assert actor.mlp[0].weight.grad is not None
    assert torch.isfinite(actor.mlp[0].weight.grad).all()
    assert isinstance(actor.distribution, GaussianDistribution)
    assert actor.distribution.std_param.grad is not None
    assert torch.isfinite(actor.distribution.std_param.grad).all()


def test_retained_actual_v8_matches_torch_reload_and_regular_runner_export(tmp_path):
    """Optional retained-byte test; fixtures alone do not prove actual parity."""
    from types import SimpleNamespace
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab_microduck.community_policy_normalizer import inspect_affine_prefix

    path = Path(__file__).resolve().parents[1] / "artifacts/community/intake-20261013/official-velstand-v8/policy.onnx"
    if not path.is_file():
        pytest.skip("retained official v8 intake is not in this checkout")
    payload = path.read_bytes()
    values = inspect_affine_prefix(payload, reference.OFFICIAL_V8_SHA256)["affine_values"]
    generator = torch.Generator().manual_seed(991)
    raw = (torch.tensor(values["mean"])[None]
           + torch.tensor(values["denominator"])[None]
           * torch.randn(128, 61, generator=generator))
    raw[0] = 0
    raw[1] = 0
    raw[1, 5] = -1
    raw[1, 48] = .3
    actor = _actor()
    report = import_v8_actor(actor, path, raw)
    assert report["maximum_actor_error"] <= 2e-5
    restored = _actor()
    restored.load_state_dict(actor.state_dict(), strict=True)
    observations = TensorDict({"actor": raw}, batch_size=[128])
    assert torch.equal(restored(observations), actor(observations))
    actor.train()
    actor.update_normalization(observations)
    assert actor.obs_normalizer.count.item() == 0

    # Exercise the same installed normalized exporter used by the registered
    # runner; no env metadata, publication or deployment claim in this CPU test.
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor))
    MjlabOnPolicyRunner.export_policy_to_onnx(runner, str(tmp_path), "cpu-parity.onnx")
    import onnxruntime as ort
    session = ort.InferenceSession(str(tmp_path / "cpu-parity.onnx"),
                                   providers=["CPUExecutionProvider"])
    source = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    for row in raw:
        feed = {"obs": row[None].numpy()}
        output = session.run(["actions"], feed)[0]
        expected = source.run(["actions"], feed)[0]
        assert np.isfinite(output).all()
        assert np.max(np.abs(output - expected)) <= 2e-5
