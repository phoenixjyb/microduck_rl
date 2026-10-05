"""CPU-only byte-contract tests for the bounded CRB launch trace archive."""

from copy import deepcopy
from collections.abc import Mapping
import os
import struct

import pytest
import torch
import test_stance_crb_launch_trace as launch_tests

from test_stance_crb_launch_trace import (
    SOURCE,
    _cpu_runtime,
    _tree_equal,
    smooth,
    wp,
)
from mjlab_microduck import stance_crb_launch_trace as crb
from mjlab_microduck import stance_crb_trace_archive as archive
from mjlab_microduck import stance_recovery_schedule as schedule

actual_case = launch_tests.actual_case


@pytest.fixture(scope="module")
def _module_cuda_hidden():
    prior = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior


def _decode(manifest, inertia_raw, stage_raw, projected, declaration, record):
    return archive.decode(
        manifest, inertia_raw, stage_raw, projected, declaration, record
    )


def _authenticated_stage_variant(manifest, stage_raw, change):
    header, tensor_bytes = archive._parse_stage_payload(stage_raw)
    changed = deepcopy(header)
    change(changed)
    encoded = archive.canonical(changed).encode("utf-8")
    raw = archive.MAGIC + struct.pack("<I", len(encoded)) + encoded + tensor_bytes
    new_manifest = deepcopy(manifest)
    new_manifest["stage_bytes"] = len(raw)
    new_manifest["stage_sha256"] = archive._sha(raw)
    return new_manifest, raw


@pytest.fixture(scope="module")
def encoded_case(actual_case):
    case = actual_case
    projected = crb._project(case["trace"])
    manifest, inertia_raw, stage_raw = archive.encode(
        case["trace"], case["declaration"], case["record"]
    )
    return {
        **case,
        "projected": projected,
        "manifest": manifest,
        "inertia_raw": inertia_raw,
        "stage_raw": stage_raw,
    }


@pytest.fixture(scope="module")
def actual_64_case(_module_cuda_hidden):
    """One genuine 64-world CPU capture for exact maximum-layout accounting."""
    assert not torch.cuda.is_initialized()
    cells = ["zero-wrench"] * 32 + ["+x-2n-20steps-t250"] * 32
    declaration = schedule.declaration(SOURCE, "dose", "training", cells)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        runtime = _cpu_runtime(declaration)
    observer = crb.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    try:
        with observer:
            result = runtime.step_with_schedule(
                torch.zeros(64, 10), capture_control=True
            )
        trace = observer.capture()
    finally:
        assert observer.faulted is False
    assert not torch.cuda.is_initialized()
    record = {"runtime_result_before_reset": result}
    projected = crb._project(trace)
    manifest, inertia_raw, stage_raw = archive.encode(trace, declaration, record)
    return {
        "trace": trace,
        "declaration": declaration,
        "record": record,
        "projected": projected,
        "manifest": manifest,
        "inertia_raw": inertia_raw,
        "stage_raw": stage_raw,
    }


def test_two_world_roundtrip_preserves_old_projection_and_owned_stage_bytes(
    encoded_case,
):
    case = encoded_case
    projection_input = deepcopy(case["projected"])
    trace, score = _decode(
        case["manifest"],
        case["inertia_raw"],
        case["stage_raw"],
        projection_input,
        case["declaration"],
        case["record"],
    )
    assert _tree_equal(trace, case["trace"])
    assert _tree_equal(crb._project(trace), case["projected"])
    assert score["protocol"] == crb.PROTOCOL + ":score"
    assert score["worlds"] == 2 and score["events"] == 6
    assert case["manifest"]["protocol"] == archive.PROTOCOL
    assert case["manifest"]["source"] == SOURCE
    assert all(value is False for value in crb.FLAGS.values())
    assert all(case["manifest"][key] is False for key in crb.FLAGS)
    assert trace["partial_crb_accumulation"] is None
    for event_index, event in enumerate(trace["events"]):
        for stage_index, stage in enumerate(event["crb_accumulation_stages"]):
            for field in archive.FIELDS:
                tensor = stage[field]
                assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
                assert tensor.is_contiguous()
                assert tensor.untyped_storage().data_ptr() != 0
                assert (
                    tensor.data_ptr()
                    != case["trace"]["events"][event_index]["crb_accumulation_stages"][
                        stage_index
                    ][field].data_ptr()
                )
    decoded_input = trace["events"][0]["inputs"]["xfrc_applied"]
    projected_input = projection_input["events"][0]["inputs"]["xfrc_applied"]
    assert decoded_input.data_ptr() != projected_input.data_ptr()
    before = decoded_input.clone()
    projected_input.fill_(123.0)
    assert _tree_equal(decoded_input, before)


def test_decode_never_uses_torch_load(encoded_case, monkeypatch):
    case = encoded_case

    def forbidden(*_args, **_kwargs):
        raise AssertionError("archive decoding must not deserialize torch payloads")

    monkeypatch.setattr(archive.torch, "load", forbidden)
    decoded, score = _decode(
        case["manifest"],
        case["inertia_raw"],
        case["stage_raw"],
        case["projected"],
        case["declaration"],
        case["record"],
    )
    assert score["worlds"] == 2
    assert _tree_equal(decoded, case["trace"])


def test_archive_codec_preserves_signed_zero_bits(encoded_case):
    case = encoded_case
    trace = deepcopy(case["trace"])
    first = trace["events"][0]["crb_accumulation_stages"][0]
    zero = torch.nonzero(first["before_crb"] == 0, as_tuple=False)
    assert len(zero) > 0
    index = tuple(zero[0].tolist())
    for event in trace["events"]:
        for stage in event["crb_accumulation_stages"]:
            for name in archive.FIELDS:
                value = stage[name]
                if value[index].item() == 0.0:
                    value[index] = -0.0
        persistent = event["persistent"]
        for name in ("crb", "cinert"):
            value = persistent[name]
            if value[index].item() == 0.0:
                value[index] = -0.0
    projected = crb._project(trace)
    manifest, inertia_raw, stage_raw = archive.encode(
        trace, case["declaration"], case["record"]
    )
    decoded, _score = _decode(
        manifest, inertia_raw, stage_raw, projected, case["declaration"], case["record"]
    )
    assert _tree_equal(decoded, trace)
    decoded_value = decoded["events"][0]["crb_accumulation_stages"][0]["before_crb"][
        index
    ]
    assert torch.signbit(decoded_value)


@pytest.mark.parametrize(
    "damage",
    [
        "inertia-hash",
        "stage-hash",
        "inertia-size",
        "stage-size",
        "inertia-bytes",
        "stage-bytes",
        "type",
        "cap",
        "extra",
        "missing",
        "flag",
        "hash-bomb",
        "inertia-cap",
    ],
)
def test_manifest_and_whole_byte_failures_precede_header_parse(
    encoded_case, monkeypatch, damage
):
    case = encoded_case
    manifest = deepcopy(case["manifest"])
    inertia_raw, stage_raw = case["inertia_raw"], case["stage_raw"]
    if damage == "inertia-hash":
        manifest["inertia_sha256"] = "0" * 64
    elif damage == "stage-hash":
        manifest["stage_sha256"] = "0" * 64
    elif damage == "inertia-size":
        manifest["inertia_bytes"] += 1
    elif damage == "stage-size":
        manifest["stage_bytes"] += 1
    elif damage == "inertia-bytes":
        inertia_raw = bytes([inertia_raw[0] ^ 1]) + inertia_raw[1:]
    elif damage == "stage-bytes":
        stage_raw = bytes([stage_raw[0] ^ 1]) + stage_raw[1:]
    elif damage == "type":
        stage_raw = bytearray(stage_raw)
    elif damage == "cap":
        manifest["stage_bytes"] = archive.MAX_ARTIFACT_BYTES + 1
    elif damage == "inertia-cap":
        manifest["inertia_bytes"] = archive.MAX_INERTIA_BYTES + 1
    elif damage == "extra":
        manifest["unexpected"] = None
    elif damage == "missing":
        del manifest["stage_sha256"]
    elif damage == "hash-bomb":

        class HashBomb:
            armed = False

            def __hash__(self):
                if self.armed:
                    raise AssertionError("manifest key hashed before size check")
                return 17

        hostile = HashBomb()
        manifest[hostile] = None
        hostile.armed = True
    else:
        manifest[next(iter(crb.FLAGS))] = True

    def forbidden(*_args, **_kwargs):
        raise AssertionError("header parser ran before complete authentication")

    monkeypatch.setattr(archive, "_parse_stage_payload", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _decode(
            manifest,
            inertia_raw,
            stage_raw,
            case["projected"],
            case["declaration"],
            case["record"],
        )


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate-key",
        "noncanonical",
        "nonfinite",
        "extra-header",
        "missing-header",
        "event-count",
        "stage-count",
        "dtype",
        "wrong-magic",
        "bad-header-length",
        "phase",
        "source",
        "topology",
        "stage-order",
        "shape-layout",
        "trailing-payload",
        "wrong-protocol",
    ],
)
def test_authenticated_malformed_headers_and_fixed_layout_are_rejected(
    encoded_case, damage
):
    case = encoded_case
    manifest, raw = _authenticated_stage_variant(
        case["manifest"],
        case["stage_raw"],
        lambda header: _damage_header(header, damage),
    )
    if damage in {
        "duplicate-key",
        "noncanonical",
        "nonfinite",
        "trailing-payload",
        "wrong-magic",
        "bad-header-length",
    }:
        # Construct deliberately invalid framing while authenticating the exact
        # bytes in the supplied manifest.
        _header, tensor_bytes = archive._parse_stage_payload(case["stage_raw"])
        header, _ = archive._parse_stage_payload(case["stage_raw"])
        header_raw = archive.canonical(header).encode()
        if damage == "duplicate-key":
            header_raw = header_raw[:-1] + b',"protocol":"duplicate"}'
        elif damage == "noncanonical":
            header_raw = b" " + header_raw
        elif damage == "nonfinite":
            header_raw = header_raw[:-1] + b',"nonfinite":NaN}'
        else:
            tensor_bytes += b"x"
        raw = (
            archive.MAGIC
            + struct.pack("<I", len(header_raw))
            + header_raw
            + tensor_bytes
        )
        if damage == "wrong-magic":
            raw = b"BADMAGIC" + raw[len(archive.MAGIC) :]
        elif damage == "bad-header-length":
            raw = (
                raw[: len(archive.MAGIC)]
                + struct.pack("<I", 0)
                + raw[archive.PREFIX_BYTES :]
            )
        manifest["stage_bytes"] = len(raw)
        manifest["stage_sha256"] = archive._sha(raw)
    with pytest.raises((TypeError, ValueError)):
        _decode(
            manifest,
            case["inertia_raw"],
            raw,
            case["projected"],
            case["declaration"],
            case["record"],
        )


def _damage_header(header, damage):
    if damage == "extra-header":
        header["unexpected"] = None
    elif damage == "missing-header":
        del header["source"]
    elif damage == "event-count":
        header["events"].append(deepcopy(header["events"][-1]))
    elif damage == "stage-count":
        header["events"][0]["stages"].pop()
    elif damage == "dtype":
        header["dtype"] = "float64"
    elif damage == "phase":
        header["events"][0]["phase"] = "unforced-post"
    elif damage == "source":
        header["source"] = "f" * 40
    elif damage == "topology":
        header["compiled_topology"]["body_parentid"][1] = 999
    elif damage == "stage-order":
        header["events"][0]["stages"].reverse()
    elif damage == "shape-layout":
        header["events"][0]["stages"][0]["dim"][0] += 1
    elif damage == "wrong-protocol":
        header["protocol"] = "wrong"


def test_64_world_capture_has_exact_bounded_stage_tensor_layout(actual_64_case):
    case = actual_64_case
    assert not torch.cuda.is_initialized()
    assert case["trace"]["worlds"] == 64
    assert case["manifest"]["stage_payload_tensor_bytes"] == 6_881_280
    assert case["manifest"]["stage_bytes"] <= archive.MAX_ARTIFACT_BYTES
    assert case["manifest"]["inertia_bytes"] <= archive.MAX_INERTIA_BYTES
    decoded, score = _decode(
        case["manifest"],
        case["inertia_raw"],
        case["stage_raw"],
        case["projected"],
        case["declaration"],
        case["record"],
    )
    assert score["worlds"] == 64 and score["events"] == 6
    assert _tree_equal(decoded, case["trace"])
    assert all(value is False for key, value in score.items() if key in crb.FLAGS)


def test_fresh_topology_mismatch_rejects_before_stage_tensor_rebuild(
    encoded_case, monkeypatch
):
    case = encoded_case
    changed_sha = {}

    def alter(header):
        parents = header["compiled_topology"]["body_parentid"]
        parents[1] = (parents[1] + 2) % 16
        changed_sha["value"] = archive._sha(
            archive.canonical(header["compiled_topology"]).encode("utf-8")
        )
        header["compiled_topology_sha256"] = changed_sha["value"]

    manifest, raw = _authenticated_stage_variant(
        case["manifest"], case["stage_raw"], alter
    )
    manifest["compiled_topology_sha256"] = changed_sha["value"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("stage reconstruction preceded fresh topology binding")

    monkeypatch.setattr(archive, "_rebuild_stages", forbidden)
    with pytest.raises(ValueError):
        _decode(
            manifest,
            case["inertia_raw"],
            raw,
            case["projected"],
            case["declaration"],
            case["record"],
        )


@pytest.mark.parametrize("scalar", [1 << 100, "x" * 4097, "界" * 1400])
def test_excessive_inherited_scalar_is_rejected_before_encoding(monkeypatch, scalar):
    """Synthetic seam verifies serializer ordering, not physical evidence."""
    monkeypatch.setattr(
        crb, "check", lambda *_args, **_kwargs: {"source": SOURCE, "worlds": 2}
    )
    monkeypatch.setattr(crb, "_project", lambda _value: {"oversized": scalar})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inherited serializer ran before scalar budget check")

    monkeypatch.setattr(archive.evidence, "encode", forbidden)
    with pytest.raises(ValueError):
        archive.encode(None, None, None)


def test_decode_caps_python_tree_before_inherited_reader(encoded_case, monkeypatch):
    case = encoded_case

    class OversizedMapping(Mapping):
        def __len__(self):
            return archive.evidence.NODE_BUDGET + 1

        def __iter__(self):
            raise AssertionError("oversized keys traversed before container limit")

        def __getitem__(self, _key):
            raise AssertionError("oversized mapping accessed before container limit")

    projected = deepcopy(case["projected"])
    projected["events"][0]["inputs"] = OversizedMapping()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inherited reader ran before new tree preflight")

    monkeypatch.setattr(crb.inertia, "check", forbidden)
    with pytest.raises(ValueError, match="mapping size"):
        _decode(
            case["manifest"],
            case["inertia_raw"],
            case["stage_raw"],
            projected,
            case["declaration"],
            case["record"],
        )


@pytest.mark.parametrize(
    "header_raw",
    [b"[" * 1100 + b"0" + b"]" * 1100, b'"\\ud800"'],
    ids=["nested-root", "surrogate-root"],
)
def test_authenticated_recursive_or_invalid_unicode_header_fails_closed(
    encoded_case, header_raw, monkeypatch
):
    case = encoded_case
    _, tensors = archive._parse_stage_payload(case["stage_raw"])
    raw = archive.MAGIC + struct.pack("<I", len(header_raw)) + header_raw + tensors
    manifest = deepcopy(case["manifest"])
    manifest.update(stage_bytes=len(raw), stage_sha256=archive._sha(raw))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid header reached tensor reconstruction")

    monkeypatch.setattr(archive, "_rebuild_stages", forbidden)
    with pytest.raises(ValueError):
        _decode(
            manifest,
            case["inertia_raw"],
            raw,
            case["projected"],
            case["declaration"],
            case["record"],
        )


def test_header_canonicalization_recursion_error_fails_closed(
    encoded_case, monkeypatch
):
    def forbidden(_value):
        raise RecursionError("synthetic bounded canonicalization error")

    monkeypatch.setattr(archive, "canonical", forbidden)
    with pytest.raises(ValueError, match="invalid CRB stage archive JSON header"):
        archive._parse_stage_payload(encoded_case["stage_raw"])


@pytest.mark.parametrize("fault", ["status", "partial"])
def test_encode_refuses_faulted_or_partial_capture(encoded_case, fault):
    case = encoded_case
    changed = deepcopy(case["trace"])
    if fault == "status":
        changed["status"] = "faulted"
    else:
        changed["partial_crb_accumulation"] = {"phase": "scheduled-pre"}
    with pytest.raises((TypeError, ValueError)):
        archive.encode(changed, case["declaration"], case["record"])
