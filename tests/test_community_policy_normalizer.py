"""Synthetic graph inspection only; no policy inference, training or physics."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import pytest

from mjlab_microduck import community_policy_normalizer as n


def fixture():
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["centered"]),
             helper.make_node("Div", ["centered", "denominator"], ["normalized"]),
             helper.make_node("Gemm", ["normalized", "weight", "bias"], ["actions"], transB=1)]
    initializers = [numpy_helper.from_array(np.zeros((1, 61), dtype=np.float32), "mean"),
                    numpy_helper.from_array(np.full((1, 61), 2., dtype=np.float32), "denominator"),
                    numpy_helper.from_array(np.ones((14, 61), dtype=np.float32), "weight"),
                    numpy_helper.from_array(np.zeros(14, dtype=np.float32), "bias")]
    graph = helper.make_graph(nodes, "not-a-public-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])], initializers)
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 18)])
    helper.set_model_props(model, {"observation_names": n.OBSERVATION_NAMES})
    return model


def inspect(model):
    payload = model.SerializeToString()
    return n.inspect_affine_prefix(payload, hashlib.sha256(payload).hexdigest())


def replace(model, name, values):
    item = next(x for x in model.graph.initializer if x.name == name)
    item.CopyFrom(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))


def test_recognized_prefix_retains_only_static_claims():
    report = inspect(fixture())
    assert report["decision"] == "recognized-static-affine-prefix-only"
    assert report["hidden_width"] == 14
    assert report["command_features"][0] == dict(index=48, mean=0., denominator=2., first_layer_nonzero_count=14)
    assert report["terms"][-1]["start"] == 55
    assert report["policy_inferences"] == 0
    assert all(value is False for key, value in report.items() if key.endswith(("verified", "authorized", "executed", "acceptance")))


def test_zero_first_layer_command_column_is_described_not_rehabilitated():
    model = fixture()
    weights = np.ones((14, 61), dtype=np.float32)
    weights[:, 48] = 0
    replace(model, "weight", weights)
    assert inspect(model)["command_features"][0]["first_layer_nonzero_count"] == 0


def test_all_affine_constants_are_retained_exactly_not_only_command_slots():
    model = fixture()
    mean = np.arange(61, dtype=np.float32).reshape(1, 61) / 16
    denominator = (np.arange(61, dtype=np.float32).reshape(1, 61) + 1) / 8
    replace(model, "mean", mean)
    replace(model, "denominator", denominator)
    assert inspect(model)["affine_values"] == dict(
        mean=mean.ravel().tolist(), denominator=denominator.ravel().tolist())


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_bad_denominator_fails_closed(value):
    model = fixture()
    denominator = np.ones((1, 61), dtype=np.float32)
    denominator[0, 48] = value
    replace(model, "denominator", denominator)
    with pytest.raises(ValueError, match="positive|nonfinite"):
        inspect(model)


def test_initializer_override_cannot_be_treated_as_constant():
    model = fixture()
    model.graph.input.append(helper.make_tensor_value_info("mean", TensorProto.FLOAT, [1, 61]))
    with pytest.raises(ValueError, match="overridable"):
        inspect(model)


def test_missing_term_order_does_not_guess_command_semantics():
    model = fixture()
    del model.metadata_props[:]
    with pytest.raises(ValueError, match="term order"):
        inspect(model)


def test_bypass_of_normalized_input_is_not_recognized():
    model = fixture()
    model.graph.node.append(helper.make_node("Identity", ["obs"], ["unused-raw-bypass"]))
    with pytest.raises(ValueError, match="bypass"):
        inspect(model)


def test_subtraction_direction_cannot_be_swapped():
    model = fixture()
    model.graph.node[0].input[:] = ["mean", "obs"]
    with pytest.raises(ValueError, match="wiring"):
        inspect(model)


def test_unsupported_gemm_scale_is_not_silently_assumed():
    model = fixture()
    model.graph.node[2].attribute.append(helper.make_attribute("alpha", 2.))
    with pytest.raises(ValueError, match="Gemm convention"):
        inspect(model)


def test_nonfinite_first_layer_weight_rejected():
    model = fixture()
    replace(model, "weight", np.full((14, 61), float("nan"), dtype=np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        inspect(model)


def test_model_hash_mismatch_fails_before_inspection():
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        n.inspect_affine_prefix(fixture().SerializeToString(), "0" * 64)


def test_cli_reads_same_bytes_and_never_executes_actor(tmp_path, monkeypatch, capsys):
    path = tmp_path / "fixture.onnx"
    payload = fixture().SerializeToString()
    path.write_bytes(payload)
    monkeypatch.setattr(sys, "argv", ["normalizer", str(path), "--sha256", hashlib.sha256(payload).hexdigest()])
    n.main()
    assert json.loads(capsys.readouterr().out)["actor_forward_executed"] is False


def test_isolated_import_loads_no_native_modules():
    command = "import sys; import mjlab_microduck.community_policy_normalizer; print(','.join(sorted(set(sys.modules) & {'numpy','onnx','onnxruntime','mujoco','torch','warp','bam'})))"
    result = subprocess.run([sys.executable, "-c", command], text=True, capture_output=True, check=True)
    assert not result.stdout.strip()
