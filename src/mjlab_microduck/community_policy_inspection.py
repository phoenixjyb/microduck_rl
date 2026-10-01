"""Read-only structural inspection of community ONNX policy artifacts.

This module parses and validates a self-contained ONNX graph. It never runs
the graph and its interface classifications are not policy admissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import onnx
from google.protobuf.message import Message


MAX_BYTES = 32 * 1024 * 1024
ALLOWED_DOMAINS = {"", "ai.onnx"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_payload(path: Path) -> bytes:
    """Open one regular non-symlink file and read a bounded immutable payload."""
    path = Path(path)
    _require(not path.is_symlink(), "policy path must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("policy file could not be opened safely") from exc
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode), "policy path must be a regular file")
        _require(info.st_size <= MAX_BYTES, "policy file exceeds 32 MiB")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(MAX_BYTES + 1)
        _require(len(payload) <= MAX_BYTES, "policy file exceeds 32 MiB")
        _require(len(payload) == info.st_size, "policy file changed while being read")
        return payload
    finally:
        os.close(fd)


def _messages(message: Message):
    """Walk populated protobuf message fields, including nested graphs/attrs."""
    yield message
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for child in value:
                yield from _messages(child)
        else:
            yield from _messages(value)


def _reject_external_tensors(model: onnx.ModelProto) -> None:
    tensor_name = onnx.TensorProto.DESCRIPTOR.full_name
    for message in _messages(model):
        if message.DESCRIPTOR.full_name != tensor_name:
            continue
        tensor = message
        _require(tensor.data_location != onnx.TensorProto.EXTERNAL
                 and not tensor.external_data,
                 "external tensor data is not allowed")


def _reject_custom_domains(model: onnx.ModelProto) -> None:
    for message in _messages(model):
        if message.DESCRIPTOR.full_name in (
            onnx.NodeProto.DESCRIPTOR.full_name,
            onnx.FunctionProto.DESCRIPTOR.full_name,
        ):
            _require(message.domain in ALLOWED_DOMAINS,
                     "custom operator domain is not allowed: " + message.domain)
        elif message.DESCRIPTOR.full_name == onnx.OperatorSetIdProto.DESCRIPTOR.full_name:
            _require(message.domain in ALLOWED_DOMAINS,
                     "custom operator domain is not allowed: " + message.domain)


def _dtype(elem_type: int) -> str:
    name = onnx.TensorProto.DataType.Name(elem_type)
    if name == "FLOAT":
        return "float32"
    return name.lower()


def _value_info(value: onnx.ValueInfoProto) -> dict:
    tensor_type = value.type.tensor_type
    shape = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append({"symbol": dim.dim_param})
            else:
                shape.append(None)
    else:
        shape = None
    return {"name": value.name, "dtype": _dtype(tensor_type.elem_type), "shape": shape}


def _signature_matches(values: list[dict], expected: list[tuple[str, list[int]]]) -> bool:
    return (len(values) == len(expected)
            and all(value["name"] == name and value["dtype"] == "float32"
                    and value["shape"] == shape
                    for value, (name, shape) in zip(values, expected)))


def _classify(inputs: list[dict], outputs: list[dict]) -> str:
    if (_signature_matches(inputs, [("obs", [1, 61])])
            and _signature_matches(outputs, [("actions", [1, 14])])):
        return "feedforward-microduck-api1"
    # API2 matching is name-based so original ONNX ordering remains report-only.
    by_input = {item["name"]: item for item in inputs}
    by_output = {item["name"]: item for item in outputs}
    input_names = {"obs", "h_in", "c_in"}
    output_names = {"actions", "h_out", "c_out"}
    if (set(by_input) == input_names and len(inputs) == 3
            and set(by_output) == output_names and len(outputs) == 3
            and all(by_input[name]["dtype"] == "float32"
                    and by_input[name]["shape"] == shape
                    for name, shape in (("obs", [1, 61]), ("h_in", [1, 1, 256]),
                                        ("c_in", [1, 1, 256])))
            and all(by_output[name]["dtype"] == "float32"
                    and by_output[name]["shape"] == shape
                    for name, shape in (("actions", [1, 14]), ("h_out", [1, 1, 256]),
                                        ("c_out", [1, 1, 256])))):
        return "lstm-microduck-api2"
    return "other-interface-unreviewed"


def inspect_policy(path: Path, expected_sha256: str) -> dict:
    """Inspect one exact local ONNX payload without executing it."""
    _require(type(expected_sha256) is str
             and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "expected SHA256 must be 64 lowercase hexadecimal characters")
    payload = _read_payload(Path(path))
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == expected_sha256, "policy SHA256 mismatch")
    try:
        model = onnx.load_model_from_string(payload)
        _reject_external_tensors(model)
        _reject_custom_domains(model)
        onnx.checker.check_model(model)
    except ValueError:
        raise
    except Exception as exc:
        # ONNX/protobuf versions expose malformed graph failures through several
        # exception classes. Normalize them at this untrusted-file boundary.
        raise ValueError("invalid or unsupported ONNX model") from exc

    initializer_names = {item.name for item in model.graph.initializer}
    for sparse in model.graph.sparse_initializer:
        # ONNX names the sparse initializer by values.name; indices.name is
        # storage metadata, not another initialized graph input.
        initializer_names.add(sparse.values.name)
    inputs = [_value_info(value) for value in model.graph.input
              if value.name not in initializer_names]
    outputs = [_value_info(value) for value in model.graph.output]
    return dict(
        artifact=dict(sha256=digest, bytes=len(payload)),
        onnx=dict(ir_version=int(model.ir_version), opsets=[
            dict(domain=opset.domain, version=int(opset.version))
            for opset in model.opset_import
        ]),
        runtime_inputs=inputs,
        outputs=outputs,
        interface_classification=_classify(inputs, outputs),
        manifest_runtime_binding_verified=False,
        normalizer_verified=False,
        behavioral_acceptance=False,
        training_authorized=False,
        transition_authorized=False,
        physical_motion_authorized=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        result = inspect_policy(args.path, args.sha256)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
