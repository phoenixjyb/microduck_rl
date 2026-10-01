"""Read-only ONNX structure inspection and strict interface classification."""

import hashlib
import json
import os
import signal

import onnx
import pytest
from onnx import TensorProto, helper

from mjlab_microduck import community_policy_inspection as inspection


FLAGS = (
    "manifest_runtime_binding_verified", "normalizer_verified", "behavioral_acceptance",
    "training_authorized", "transition_authorized", "physical_motion_authorized",
)


def model_bytes(inputs, outputs, *, nodes=None, opsets=None):
    # Identity wiring only makes these fixtures checker-valid; they test declared
    # metadata and are not executable/action-correct policy examples.
    if nodes is None:
        source = inputs[0].name
        nodes = [helper.make_node("Identity", [source], [output.name]) for output in outputs]
    graph = helper.make_graph(nodes, "synthetic", inputs, outputs)
    model = helper.make_model(graph, opset_imports=opsets or [helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model.SerializeToString()


def value(name, shape, dtype=TensorProto.FLOAT, symbolic=False):
    if symbolic:
        info = helper.make_tensor_value_info(name, dtype, None)
        dim = info.type.tensor_type.shape.dim.add()
        dim.dim_param = "batch"
        for size in shape[1:]:
            info.type.tensor_type.shape.dim.add().dim_value = size
        return info
    return helper.make_tensor_value_info(name, dtype, shape)


def save_model(tmp_path, payload, name="candidate.onnx"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def feedforward_payload(inputs=None, outputs=None, **kwargs):
    return model_bytes(inputs or [value("obs", [1, 61])],
                       outputs or [value("actions", [1, 14])], **kwargs)


def test_feedforward_signature_is_reported_without_admission(tmp_path):
    path, digest = save_model(tmp_path, feedforward_payload())
    report = inspection.inspect_policy(path, digest)
    assert report["interface_classification"] == "feedforward-microduck-api1"
    assert report["runtime_inputs"] == [dict(name="obs", dtype="float32", shape=[1, 61])]
    assert report["outputs"] == [dict(name="actions", dtype="float32", shape=[1, 14])]
    assert report["artifact"] == dict(sha256=digest, bytes=path.stat().st_size)
    assert all(report[name] is False for name in FLAGS)


def test_cli_prints_json_to_stdout(tmp_path, monkeypatch, capsys):
    path, digest = save_model(tmp_path, feedforward_payload())
    monkeypatch.setattr("sys.argv", ["community_policy_inspection", str(path), "--sha256", digest])
    inspection.main()
    assert json.loads(capsys.readouterr().out)["interface_classification"] == "feedforward-microduck-api1"


def test_cli_prints_json_error_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    path, _ = save_model(tmp_path, feedforward_payload())
    monkeypatch.setattr("sys.argv", ["community_policy_inspection", str(path), "--sha256", "0" * 64])
    with pytest.raises(SystemExit, match="2"):
        inspection.main()
    assert "SHA256 mismatch" in json.loads(capsys.readouterr().out)["error"]


def test_lstm_signature_is_order_agnostic_but_preserves_graph_order(tmp_path):
    inputs = [value("c_in", [1, 1, 256]), value("obs", [1, 61]), value("h_in", [1, 1, 256])]
    outputs = [value("c_out", [1, 1, 256]), value("actions", [1, 14]),
               value("h_out", [1, 1, 256])]
    path, digest = save_model(tmp_path, model_bytes(inputs, outputs))
    report = inspection.inspect_policy(path, digest)
    assert report["interface_classification"] == "lstm-microduck-api2"
    assert [x["name"] for x in report["runtime_inputs"]] == ["c_in", "obs", "h_in"]
    assert [x["name"] for x in report["outputs"]] == ["c_out", "actions", "h_out"]


@pytest.mark.parametrize("inputs,outputs", [
    ([value("obs", [1, 61]), value("extra", [1])], [value("actions", [1, 14])]),
    ([value("obs", [1, 60])], [value("actions", [1, 14])]),
    ([value("observation", [1, 61])], [value("actions", [1, 14])]),
    ([value("obs", [1, 61], TensorProto.DOUBLE)], [value("actions", [1, 14])]),
    ([value("obs", [1, 61])], [value("actions", [1, 14]), value("extra", [1])]),
    ([value("obs", [1, 61], symbolic=True)], [value("actions", [1, 14])]),
])
def test_non_exact_interface_is_unreviewed(tmp_path, inputs, outputs):
    path, digest = save_model(tmp_path, model_bytes(inputs, outputs))
    report = inspection.inspect_policy(path, digest)
    assert report["interface_classification"] == "other-interface-unreviewed"
    assert all(report[name] is False for name in FLAGS)


def test_initializers_are_not_runtime_inputs(tmp_path):
    obs = value("obs", [1, 61])
    actions = value("actions", [1, 14])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    path, digest = save_model(tmp_path, model_bytes(
        [obs, value("weight", [1])], [actions],
    ))
    # Include a graph initializer explicitly; its matching graph input is excluded.
    model = onnx.load_model_from_string(path.read_bytes())
    model.graph.initializer.append(weight)
    payload = model.SerializeToString()
    path.write_bytes(payload)
    report = inspection.inspect_policy(path, hashlib.sha256(payload).hexdigest())
    assert [x["name"] for x in report["runtime_inputs"]] == ["obs"]
    assert report["interface_classification"] == "feedforward-microduck-api1"


def test_sparse_initializer_values_name_is_not_a_runtime_input(tmp_path):
    obs = value("obs", [1, 61])
    actions = value("actions", [1, 14])
    values = helper.make_tensor("weight_values", TensorProto.FLOAT, [1], [1.0])
    indices = helper.make_tensor("weight_indices", TensorProto.INT64, [1], [0])
    sparse = helper.make_sparse_tensor(values, indices, [1])
    extra_inputs = [value("weight_values", [1])]
    payload = model_bytes([obs, *extra_inputs], [actions])
    model = onnx.load_model_from_string(payload)
    model.graph.sparse_initializer.append(sparse)
    payload = model.SerializeToString()
    path, digest = save_model(tmp_path, payload)
    report = inspection.inspect_policy(path, digest)
    assert [x["name"] for x in report["runtime_inputs"]] == ["obs"]
    assert report["interface_classification"] == "feedforward-microduck-api1"


def test_invalid_hash_and_mismatched_payload_rejected(tmp_path):
    path, digest = save_model(tmp_path, feedforward_payload())
    with pytest.raises(ValueError, match="64 lowercase"):
        inspection.inspect_policy(path, "A" * 64)
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspection.inspect_policy(path, digest)


def test_sparse_storage_name_does_not_hide_extra_runtime_input(tmp_path):
    model = onnx.load_model_from_string(model_bytes(
        [value("obs", [1, 61]), value("extra", [1], TensorProto.INT64)],
        [value("actions", [1, 14])],
    ))
    values = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    indices = helper.make_tensor("extra", TensorProto.INT64, [1], [0])
    model.graph.sparse_initializer.append(helper.make_sparse_tensor(values, indices, [1]))
    path, digest = save_model(tmp_path, model.SerializeToString())
    report = inspection.inspect_policy(path, digest)
    assert [x["name"] for x in report["runtime_inputs"]] == ["obs", "extra"]
    assert report["interface_classification"] == "other-interface-unreviewed"


def test_correctly_hashed_malformed_onnx_rejected(tmp_path):
    path, digest = save_model(tmp_path, b"not protobuf model")
    with pytest.raises(ValueError, match="invalid or unsupported ONNX"):
        inspection.inspect_policy(path, digest)


def test_symlink_and_oversized_file_rejected(tmp_path):
    target = tmp_path / "target.onnx"
    target.write_bytes(feedforward_payload())
    link = tmp_path / "link.onnx"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        inspection.inspect_policy(link, hashlib.sha256(target.read_bytes()).hexdigest())

    oversized = tmp_path / "oversized.onnx"
    oversized.write_bytes(b"x" * (inspection.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="32 MiB"):
        inspection.inspect_policy(oversized, "0" * 64)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_fifo_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "candidate.fifo"
    os.mkfifo(fifo)

    def timeout(*_):
        raise TimeoutError("FIFO open blocked")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(2)
    try:
        with pytest.raises(ValueError, match="regular file"):
            inspection.inspect_policy(fifo, "0" * 64)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_custom_operator_domain_rejected(tmp_path):
    node = helper.make_node("Identity", ["obs"], ["actions"], domain="community.custom")
    payload = model_bytes([value("obs", [1, 61])], [value("actions", [1, 14])],
                          nodes=[node], opsets=[helper.make_opsetid("", 18),
                                               helper.make_opsetid("community.custom", 1)])
    path, digest = save_model(tmp_path, payload)
    with pytest.raises(ValueError, match="custom operator domain"):
        inspection.inspect_policy(path, digest)


def test_unused_custom_function_domain_rejected(tmp_path):
    payload = feedforward_payload()
    model = onnx.load_model_from_string(payload)
    function = helper.make_function(
        "community.custom", "Unused", ["x"], ["y"],
        [helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_opsetid("", 18)],
    )
    model.functions.append(function)
    path, digest = save_model(tmp_path, model.SerializeToString())
    with pytest.raises(ValueError, match="custom operator domain"):
        inspection.inspect_policy(path, digest)


def test_custom_domain_inside_nested_subgraph_rejected(tmp_path):
    nested = helper.make_graph(
        [helper.make_node("Identity", ["nested_in"], ["nested_out"], domain="community.custom")],
        "nested", [value("nested_in", [1])], [value("nested_out", [1])],
    )
    node = helper.make_node("Identity", ["obs"], ["actions"], nested_graph=nested)
    payload = model_bytes([value("obs", [1, 61])], [value("actions", [1, 14])], nodes=[node])
    path, digest = save_model(tmp_path, payload)
    with pytest.raises(ValueError, match="custom operator domain"):
        inspection.inspect_policy(path, digest)


def test_nested_external_tensor_in_attribute_subgraph_rejected(tmp_path):
    tensor = helper.make_tensor("nested", TensorProto.FLOAT, [1], [0.0])
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="weights.bin")
    nested_graph = helper.make_graph([], "nested", [], [], initializer=[tensor])
    node = helper.make_node("Identity", ["obs"], ["actions"], nested_graph=nested_graph)
    payload = model_bytes([value("obs", [1, 61])], [value("actions", [1, 14])], nodes=[node])
    path, digest = save_model(tmp_path, payload)
    with pytest.raises(ValueError, match="external tensor data"):
        inspection.inspect_policy(path, digest)


def test_external_sparse_initializer_nested_in_subgraph_rejected(tmp_path):
    values = helper.make_tensor("values", TensorProto.FLOAT, [1], [0.0])
    values.data_location = TensorProto.EXTERNAL
    values.external_data.add(key="location", value="sparse.bin")
    indices = helper.make_tensor("indices", TensorProto.INT64, [1], [0])
    sparse = helper.make_sparse_tensor(values, indices, [1])
    nested_graph = helper.make_graph([], "nested", [], [], sparse_initializer=[sparse])
    node = helper.make_node("Identity", ["obs"], ["actions"], nested_graph=nested_graph)
    path, digest = save_model(tmp_path, model_bytes(
        [value("obs", [1, 61])], [value("actions", [1, 14])], nodes=[node],
    ))
    with pytest.raises(ValueError, match="external tensor data"):
        inspection.inspect_policy(path, digest)
