"""CPU-only adapter contract tests; no simulator, collector, or training."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck import normalized_onnx_reference as ref


def _model(*, metadata_overrides=None, extra_input=False, input_shape=(1, 61),
           output_shape=(1, 14)):
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "denominator"], ["normalized"]),
        helper.make_node("Gemm", ["normalized", "w1", "b1"], ["hidden"], transB=1),
        helper.make_node("Tanh", ["hidden"], ["activated"]),
        helper.make_node("Gemm", ["activated", "w2", "b2"], ["actions"], transB=1),
    ]
    inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, list(input_shape))]
    if extra_input:
        inputs.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1, 1]))
    graph = helper.make_graph(
        nodes,
        "normalized-reference-test",
        inputs,
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, list(output_shape))],
        [
            numpy_helper.from_array(np.zeros((1, 61), dtype=np.float32), "mean"),
            numpy_helper.from_array(np.ones((1, 61), dtype=np.float32), "denominator"),
            numpy_helper.from_array(np.ones((4, 61), dtype=np.float32), "w1"),
            numpy_helper.from_array(np.zeros(4, dtype=np.float32), "b1"),
            numpy_helper.from_array(np.ones((14, 4), dtype=np.float32), "w2"),
            numpy_helper.from_array(np.zeros(14, dtype=np.float32), "b2"),
        ],
    )
    model = helper.make_model(
        graph, ir_version=8, opset_imports=[helper.make_opsetid("", 18)]
    )
    metadata = {
        "observation_names": ref.OBSERVATION_NAMES_CSV,
        "command_names": ref.COMMAND_NAMES_CSV,
        "joint_names": ref.JOINT_NAMES_CSV,
        "action_scale": "1.0",
        "default_joint_pos": "0.000,-0.087,-0.458,-0.005,0.453,0.349,0.349,0.000,0.000,0.000,0.087,0.458,0.005,-0.453",
    }
    metadata.update(metadata_overrides or {})
    helper.set_model_props(model, metadata)
    return model


def _write_model(tmp_path, model):
    payload = model.SerializeToString()
    path = tmp_path / "reference.onnx"
    path.write_bytes(payload)
    return path, payload, hashlib.sha256(payload).hexdigest()


class _Arg:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.type = dtype
        self.shape = shape


class _FakeSession:
    def __init__(self, *, inputs=None, outputs=None, providers=None, return_fn=None):
        self._inputs = ([_Arg("obs", "tensor(float)", [1, 61])]
                        if inputs is None else inputs)
        self._outputs = ([_Arg("actions", "tensor(float)", [1, 14])]
                         if outputs is None else outputs)
        self._providers = (["CPUExecutionProvider"] if providers is None else providers)
        self._return_fn = return_fn or (lambda row: row[:, :14].copy())
        self.calls = []

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, outputs, feed):
        self.calls.append((outputs, feed))
        return [self._return_fn(feed["obs"])]


def _fake_ort(monkeypatch, *, session=None):
    state = {"payload": None, "options": None, "providers": None}
    active_session = session or _FakeSession()

    class SessionOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None
        execution_mode = None

    class FakeOrt:
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")

        @staticmethod
        def InferenceSession(payload, *, sess_options, providers):
            state["payload"] = payload
            state["options"] = sess_options
            state["providers"] = providers
            return active_session

    FakeOrt.SessionOptions = SessionOptions

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    return state, active_session


def _adapter(monkeypatch, tmp_path, model=None, *, session=None):
    path, payload, digest = _write_model(tmp_path, model or _model())
    monkeypatch.setattr(ref, "OFFICIAL_VELSTAND_SHA256", digest)
    state, fake_session = _fake_ort(monkeypatch, session=session)
    return ref.load_normalized_onnx_reference(path, digest), state, fake_session, payload


def test_exact_verified_bytes_cpu_options_and_rowwise_raw_inference(monkeypatch, tmp_path):
    adapter, state, session, payload = _adapter(monkeypatch, tmp_path)
    input_rows = torch.arange(3 * 61, dtype=torch.float32).reshape(3, 61)
    observations = TensorDict({"actor": input_rows.clone()}, batch_size=[3])

    actions = adapter(observations)

    assert state["payload"] == payload
    assert state["providers"] == ["CPUExecutionProvider"]
    assert state["options"].intra_op_num_threads == 1
    assert state["options"].inter_op_num_threads == 1
    assert state["options"].execution_mode == "sequential"
    assert adapter.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert adapter.artifact_bytes == len(payload)
    assert adapter.provider == "CPUExecutionProvider"
    assert adapter.normalization == "baked-in-onnx-only"
    assert actions.dtype == torch.float32 and actions.device == input_rows.device
    assert torch.equal(actions, input_rows[:, :14])
    assert len(session.calls) == 3
    assert all(call[0] == ["actions"] for call in session.calls)
    assert all(call[1]["obs"].shape == (1, 61) for call in session.calls)
    assert all(call[1]["obs"].dtype == np.float32 for call in session.calls)
    assert np.array_equal(session.calls[1][1]["obs"], input_rows[1:2].numpy())


def test_hash_is_checked_before_ort_and_only_pinned_artifact_is_allowed(monkeypatch, tmp_path):
    path, payload, digest = _write_model(tmp_path, _model())
    calls = []

    class NeverOrt:
        @staticmethod
        def InferenceSession(*args, **kwargs):
            calls.append((args, kwargs))
            pytest.fail("ORT must not be constructed before artifact checks")

    monkeypatch.setitem(sys.modules, "onnxruntime", NeverOrt)
    monkeypatch.setattr(ref, "OFFICIAL_VELSTAND_SHA256", digest)
    path.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ref.load_normalized_onnx_reference(path, digest)
    with pytest.raises(ValueError, match="only the pinned"):
        ref.load_normalized_onnx_reference(path, "a" * 64)
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        ref.load_normalized_onnx_reference(path, "A" * 64)
    assert calls == []


def test_unregistered_digest_is_rejected_before_payload_or_runtime_access(monkeypatch, tmp_path):
    payload_calls = []
    runtime_calls = []

    def forbidden_payload(path):
        payload_calls.append(path)
        pytest.fail("unregistered digest must be rejected before artifact reads")

    class NeverOrt:
        @staticmethod
        def InferenceSession(*args, **kwargs):
            runtime_calls.append((args, kwargs))
            pytest.fail("unregistered digest must be rejected before ORT construction")

    monkeypatch.setattr(ref.policy_inspection, "_read_payload", forbidden_payload)
    monkeypatch.setitem(sys.modules, "onnxruntime", NeverOrt)
    path = tmp_path / "does-not-exist.onnx"
    with pytest.raises(ValueError, match="only the pinned official VelStand artifacts"):
        ref.load_normalized_onnx_reference(path, "0" * 64)

    assert payload_calls == []
    assert runtime_calls == []


def test_two_registered_policy_hashes_remain_explicit():
    assert ref.OFFICIAL_VELSTAND_SHA256 == (
        "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45")
    assert ref.OFFICIAL_V8_SHA256 == (
        "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492")


@pytest.mark.parametrize(
    ("inputs", "outputs", "providers", "message"),
    [
        ([_Arg("obs", "tensor(double)", [1, 61])], None, None, "runtime input"),
        ([_Arg("obs", "tensor(float)", [None, 61])], None, None, "runtime input"),
        ([_Arg("obs", "tensor(float)", [1, 60])], None, None, "runtime input"),
        ([_Arg("obs", "tensor(float)", [1, 61]), _Arg("extra", "tensor(float)", [1, 1])],
         None, None, "exactly one"),
        (None, [_Arg("actions", "tensor(double)", [1, 14])], None, "runtime output"),
        (None, [_Arg("actions", "tensor(float)", [1, 13])], None, "runtime output"),
        (None, [_Arg("actions", "tensor(float)", [1, 14]), _Arg("more", "tensor(float)", [1, 1])],
         None, "exactly one"),
        (None, None, ["CUDAExecutionProvider", "CPUExecutionProvider"], "sole active provider"),
    ],
)
def test_runtime_signature_and_provider_fail_closed(
    monkeypatch, tmp_path, inputs, outputs, providers, message
):
    session = _FakeSession(inputs=inputs, outputs=outputs, providers=providers)
    with pytest.raises(ValueError, match=message):
        _adapter(monkeypatch, tmp_path, session=session)


@pytest.mark.parametrize(
    "actor",
    [
        torch.zeros((1, 60), dtype=torch.float32),
        torch.zeros((9, 61), dtype=torch.float32),
        torch.zeros((1, 61), dtype=torch.float64),
        torch.full((1, 61), float("nan"), dtype=torch.float32),
        torch.full((1, 61), float("inf"), dtype=torch.float32),
    ],
)
def test_bad_actor_tensor_rejected(monkeypatch, tmp_path, actor):
    adapter, _, session, _ = _adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        adapter(TensorDict({"actor": actor}, batch_size=[actor.shape[0]]))
    assert session.calls == []


def test_missing_actor_and_non_tensordict_rejected(monkeypatch, tmp_path):
    adapter, _, session, _ = _adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="TensorDict"):
        adapter({"actor": torch.zeros((1, 61), dtype=torch.float32)})
    with pytest.raises(ValueError, match="actor tensor"):
        adapter(TensorDict({"critic": torch.zeros((1, 61))}, batch_size=[1]))
    assert session.calls == []


@pytest.mark.parametrize("bad_output", [
    np.zeros((1, 13), dtype=np.float32),
    np.zeros((1, 14), dtype=np.float64),
    np.full((1, 14), np.nan, dtype=np.float32),
    np.full((1, 14), np.inf, dtype=np.float32),
])
def test_bad_runtime_action_rejected(monkeypatch, tmp_path, bad_output):
    session = _FakeSession(return_fn=lambda _row: bad_output)
    adapter, _, _, _ = _adapter(monkeypatch, tmp_path, session=session)
    with pytest.raises(ValueError):
        adapter(TensorDict({"actor": torch.zeros((1, 61))}, batch_size=[1]))


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"observation_names": "joint_pos,joint_vel,base_ang_vel,projected_gravity,actions,command,head_command,body_command"}, "API1"),
        ({"command_names": "twist"}, "metadata mismatch"),
        ({"joint_names": "wrong,order"}, "metadata mismatch"),
        ({"action_scale": "0.5"}, "metadata mismatch"),
    ],
)
def test_model_metadata_mismatch_rejected_before_runtime(monkeypatch, tmp_path, metadata_overrides, message):
    path, _, digest = _write_model(tmp_path, _model(metadata_overrides=metadata_overrides))
    monkeypatch.setattr(ref, "OFFICIAL_VELSTAND_SHA256", digest)

    class NeverOrt:
        @staticmethod
        def InferenceSession(*args, **kwargs):
            pytest.fail("bad metadata must be rejected before runtime construction")

    monkeypatch.setitem(sys.modules, "onnxruntime", NeverOrt)
    with pytest.raises(ValueError, match=message):
        ref.load_normalized_onnx_reference(path, digest)


def test_actor_cfg_contract_checks_order_scale_history_and_concat():
    class JointPositionActionCfg:
        def __init__(self, scale=1.):
            self.scale = scale

    class FrictionDRBamActuatorCfg:
        pass

    class BacklashEncoderBamActuatorCfg(FrictionDRBamActuatorCfg):
        pass

    def make_cfg(actor, *, scale=1., actuator_type=FrictionDRBamActuatorCfg):
        return SimpleNamespace(
            observations={"actor": actor},
            actions={"joint_pos": JointPositionActionCfg(scale)},
            scene=SimpleNamespace(entities={"robot": SimpleNamespace(
                articulation=SimpleNamespace(actuators=(actuator_type(),)))}),
        )

    terms = {name: SimpleNamespace(scale=None, history_length=0)
             for name in ref.ACTOR_TERM_ORDER}
    valid = make_cfg(SimpleNamespace(terms=terms, concatenate_terms=True, history_length=None))
    assert ref.validate_actor_contract(valid) == ref.ACTOR_TERM_ORDER

    reordered_terms = dict(reversed(list(terms.items())))
    reordered = make_cfg(SimpleNamespace(
        terms=reordered_terms, concatenate_terms=True, history_length=None))
    with pytest.raises(ValueError, match="exact 8-term"):
        ref.validate_actor_contract(reordered)

    scaled_terms = dict(terms)
    scaled_terms["joint_pos"] = SimpleNamespace(scale=1.0, history_length=0)
    scaled = make_cfg(SimpleNamespace(
        terms=scaled_terms, concatenate_terms=True, history_length=None))
    with pytest.raises(ValueError, match="external scale"):
        ref.validate_actor_contract(scaled)

    history_terms = dict(terms)
    history_terms["actions"] = SimpleNamespace(scale=None, history_length=2)
    history = make_cfg(SimpleNamespace(
        terms=history_terms, concatenate_terms=True, history_length=None))
    with pytest.raises(ValueError, match="must not add history"):
        ref.validate_actor_contract(history)

    bad_scale = make_cfg(SimpleNamespace(
        terms=terms, concatenate_terms=True, history_length=None), scale=.5)
    with pytest.raises(ValueError, match="action scale"):
        ref.validate_actor_contract(bad_scale)

    bad_actuator = make_cfg(SimpleNamespace(
        terms=terms, concatenate_terms=True, history_length=None))
    bad_actuator.scene.entities["robot"].articulation.actuators = (SimpleNamespace(),)
    with pytest.raises(ValueError, match="FrictionDRBam"):
        ref.validate_actor_contract(bad_actuator)

    backlash = make_cfg(
        SimpleNamespace(terms=terms, concatenate_terms=True, history_length=None),
        actuator_type=BacklashEncoderBamActuatorCfg)
    with pytest.raises(ValueError, match="FrictionDRBamActuatorCfg"):
        ref.validate_actor_contract(backlash)
    assert ref.validate_actor_contract(
        backlash, expected_actuator="BacklashEncoderBamActuatorCfg") == ref.ACTOR_TERM_ORDER

    with pytest.raises(ValueError, match="explicitly supported BAM actuator profiles"):
        ref.validate_actor_contract(valid, expected_actuator="AnyBamActuatorCfg")
    with pytest.raises(ValueError, match="BacklashEncoderBamActuatorCfg"):
        ref.validate_actor_contract(valid, expected_actuator="BacklashEncoderBamActuatorCfg")
    with pytest.raises(ValueError, match="FrictionDRBamActuatorCfg"):
        ref.validate_actor_contract(backlash, expected_actuator="FrictionDRBamActuatorCfg")
    backlash_bad_scale = make_cfg(
        SimpleNamespace(terms=terms, concatenate_terms=True, history_length=None),
        scale=.5, actuator_type=BacklashEncoderBamActuatorCfg)
    with pytest.raises(ValueError, match="action scale"):
        ref.validate_actor_contract(backlash_bad_scale,
                                    expected_actuator="BacklashEncoderBamActuatorCfg")
    backlash_bad_history = make_cfg(
        SimpleNamespace(terms=history_terms, concatenate_terms=True, history_length=None),
        actuator_type=BacklashEncoderBamActuatorCfg)
    with pytest.raises(ValueError, match="must not add history"):
        ref.validate_actor_contract(backlash_bad_history,
                                    expected_actuator="BacklashEncoderBamActuatorCfg")


def test_registered_backlash_task_cpu_cfg_matches_explicit_profile():
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Backlash-MicroDuck")

    assert ref.validate_actor_contract(
        cfg, expected_actuator="BacklashEncoderBamActuatorCfg") == ref.ACTOR_TERM_ORDER
    actor = cfg.observations["actor"]
    assert tuple(actor.terms) == ref.ACTOR_TERM_ORDER
    assert actor.history_length is None
    assert all(term.scale is None and term.history_length == 0 for term in actor.terms.values())
    assert cfg.actions["joint_pos"].scale == 1.0
    actuators = cfg.scene.entities["robot"].articulation.actuators
    assert len(actuators) == 1 and type(actuators[0]).__name__ == "BacklashEncoderBamActuatorCfg"


def test_robot_binding_and_provenance(monkeypatch, tmp_path):
    adapter, _, _, _ = _adapter(monkeypatch, tmp_path)
    home = torch.tensor([0.000, -0.087, -0.458, -0.005, 0.453, 0.349, 0.349,
                         0.000, 0.000, 0.000, 0.087, 0.458, 0.005, -0.453])
    robot = SimpleNamespace(
        joint_names=list(ref.JOINT_NAMES),
        data=SimpleNamespace(default_joint_pos=home.repeat(2, 1)),
    )
    adapter.validate_robot(robot, tuple(range(14)), ref.JOINT_NAMES)
    proof = adapter.provenance()
    assert proof["ppo_checkpoint"] is False
    assert proof["source_policy_lineage_claim"] is False
    assert proof["normalization"] == "baked-in-onnx-only"
    assert proof["action_output"] == "raw-unfiltered"

    with pytest.raises(ValueError, match="names"):
        adapter.validate_robot(robot, tuple(range(14)), tuple(reversed(ref.JOINT_NAMES)))
    bad_home = SimpleNamespace(
        joint_names=list(ref.JOINT_NAMES),
        data=SimpleNamespace(default_joint_pos=(home + .001).repeat(1, 1)),
    )
    with pytest.raises(ValueError, match="HOME"):
        adapter.validate_robot(bad_home, tuple(range(14)), ref.JOINT_NAMES)


def test_optional_archived_velstand_cpu_smoke():
    path = (Path(__file__).resolve().parents[1]
            / "artifacts/community/intake-20261001/official-velstand/policy.onnx")
    if not path.is_file():
        pytest.skip("ignored archived VelStand artifact is not present")
    pytest.importorskip("onnxruntime")
    adapter = ref.load_normalized_onnx_reference(path, ref.OFFICIAL_VELSTAND_SHA256)
    actor = TensorDict({"actor": torch.zeros((1, 61), dtype=torch.float32)}, batch_size=[1])
    actions = adapter(actor)
    assert actions.shape == (1, 14)
    assert actions.dtype == torch.float32
    assert bool(torch.isfinite(actions).all())


def test_optional_official_v8_cpu_smoke_batch_eight():
    path = (Path(__file__).resolve().parents[1]
            / "artifacts/community/intake-20261013/official-velstand-v8/policy.onnx")
    if not path.is_file():
        pytest.skip("pinned official v8 artifact is not present")
    assert path.stat().st_size == 793705
    pytest.importorskip("onnxruntime")
    adapter = ref.load_normalized_onnx_reference(path, ref.OFFICIAL_V8_SHA256)
    actor = TensorDict({"actor": torch.zeros((8, 61), dtype=torch.float32)}, batch_size=[8])

    actions = adapter(actor)

    assert adapter.artifact_sha256 == ref.OFFICIAL_V8_SHA256
    assert adapter.provider == "CPUExecutionProvider"
    assert actions.shape == (8, 14)
    assert actions.dtype == torch.float32 and actions.device.type == "cpu"
    assert bool(torch.isfinite(actions).all())
