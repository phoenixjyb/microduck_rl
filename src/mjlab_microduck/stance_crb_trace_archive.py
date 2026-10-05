"""Bounded split-byte archive for completed scoped CRB traces.

The inherited persistent trace and the additional per-launch snapshots remain
separate payloads, each under its existing 8 MiB boundary. This is a codec,
not a signature or provenance authenticator: callers still bind both payloads
to their independently retained declaration and first record.
"""

import json
from hashlib import sha256
import struct
from collections.abc import Mapping
import math

import numpy as np
import torch

from mjlab_microduck import stance_crb_launch_trace as crb
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-crb-trace-archive-v1"
HEADER_PROTOCOL = "football-b1d-crb-trace-stage-payload-v1"
MAGIC = b"MDCRB1\x00\x00"
PREFIX_BYTES = len(MAGIC) + 4
MAX_HEADER_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = crb.MAX_STAGE_PAYLOAD_BYTES
MAX_INERTIA_BYTES = evidence.base.RAW_LIMIT
FIELDS = ("before_crb", "before_cinert", "after_crb", "after_cinert")
TENSOR_SHAPE = (16, 10)
_MANIFEST_KEYS = {
    "protocol",
    "source",
    "worlds",
    "compiled_topology_sha256",
    "inertia_bytes",
    "inertia_sha256",
    "stage_bytes",
    "stage_sha256",
    "stage_payload_tensor_bytes",
    "stage_payload_limit_bytes",
    *crb.FLAGS,
}
_HEADER_KEYS = {
    "protocol",
    "trace_protocol",
    "source",
    "worlds",
    "launch_trace_boundary",
    "smooth_module",
    "smooth_source_sha256",
    "accumulation_kernel",
    "compiled_topology",
    "compiled_topology_sha256",
    "stage_payload_tensor_bytes",
    "stage_payload_limit_bytes",
    "partial_crb_accumulation",
    "events",
}
_TOPOLOGY_KEYS = {
    "nbody",
    "nq",
    "nv",
    "nu",
    "worlds",
    "body_parentid",
    "reversed_body_tree_ids",
}
_EVENT_KEYS = {"phase", "step", "stages"}
_STAGE_KEYS = {"stage_index", "body_tree_ids", "dim"}


def _sha(raw):
    return sha256(raw).hexdigest()


def _is_int(value):
    return type(value) is int


def _exact_keys(value, expected):
    if type(value) is not dict or len(value) != len(expected):
        return False
    # Do not hash arbitrary caller-controlled keys during schema preflight.
    if any(type(key) is not str or len(key) > 128 for key in value):
        return False
    return set(value) == expected


def _int_list(value, maximum):
    return (
        type(value) is list
        and len(value) <= maximum
        and all(_is_int(item) for item in value)
    )


def _preflight_inertia_tree(value):
    """Bound old-tree tensors and serialization overhead before encoding."""
    budget = {"nodes": 0, "tensor_bytes": 0, "tensors": 0, "text_bytes": 0}

    def add_text(item, label):
        require(type(item) is str, "string " + label)
        try:
            encoded = item.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("valid UTF-8 " + label) from error
        require(len(encoded) <= 4096, "bounded UTF-8 " + label)
        budget["text_bytes"] += len(encoded)
        require(budget["text_bytes"] <= 64 * 1024, "bounded inertia text bytes")

    def visit(item, depth=0):
        budget["nodes"] += 1
        require(
            depth <= 32 and budget["nodes"] <= evidence.NODE_BUDGET,
            "bounded inherited inertia serialization tree",
        )
        if torch.is_tensor(item):
            budget["tensors"] += 1
            budget["tensor_bytes"] += item.numel() * item.element_size()
            require(
                budget["tensor_bytes"] <= MAX_INERTIA_BYTES,
                "bounded inherited inertia tensor bytes before encoding",
            )
        elif isinstance(item, Mapping):
            require(
                len(item) <= evidence.NODE_BUDGET,
                "bounded inherited inertia mapping size",
            )
            require(
                all(type(key) is str and len(key) <= 4096 for key in item),
                "bounded string inherited inertia mapping keys",
            )
            for key in item:
                add_text(key, "inherited inertia mapping key")
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) in (list, tuple):
            require(
                len(item) <= evidence.NODE_BUDGET,
                "bounded inherited inertia sequence size",
            )
            for child in item:
                visit(child, depth + 1)
        else:
            require(
                item is None or type(item) in (str, bool, int, float),
                "safe inherited inertia serialization leaf",
            )
            if type(item) is str:
                add_text(item, "inherited inertia string")
            elif type(item) is int:
                require(
                    item.bit_length() <= 64,
                    "bounded inherited inertia integer width",
                )
            require(
                type(item) is not float or math.isfinite(item),
                "finite inherited inertia serialization scalar",
            )

    visit(value)
    # ZIP headers/alignment, pickle metadata, and storage descriptors are
    # estimated with a conservative heuristic; this is not a proof of encoder
    # overhead. The post-encode whole-byte cap remains authoritative.
    estimated = (
        budget["tensor_bytes"]
        + budget["tensors"] * 256
        + budget["nodes"] * 128
        + budget["text_bytes"]
        + 64 * 1024
    )
    require(
        estimated <= MAX_INERTIA_BYTES,
        "inherited inertia serialization budget before encoding",
    )
    evidence._owned_tree(value, clone=False)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate CRB archive header key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("non-finite CRB archive header value")


def _header(value):
    topology = value["compiled_topology"]
    events = []
    for event in value["events"]:
        events.append(
            {
                "phase": event["phase"],
                "step": event["step"],
                "stages": [
                    {
                        "stage_index": stage["stage_index"],
                        "body_tree_ids": list(stage["body_tree_ids"]),
                        "dim": list(stage["dim"]),
                    }
                    for stage in event["crb_accumulation_stages"]
                ],
            }
        )
    return {
        "protocol": HEADER_PROTOCOL,
        "trace_protocol": crb.PROTOCOL,
        "source": value["source"],
        "worlds": value["worlds"],
        "launch_trace_boundary": value["launch_trace_boundary"],
        "smooth_module": value["smooth_module"],
        "smooth_source_sha256": value["smooth_source_sha256"],
        "accumulation_kernel": value["accumulation_kernel"],
        "compiled_topology": topology,
        "compiled_topology_sha256": value["compiled_topology_sha256"],
        "stage_payload_tensor_bytes": value["stage_payload_tensor_bytes"],
        "stage_payload_limit_bytes": value["stage_payload_limit_bytes"],
        "partial_crb_accumulation": value["partial_crb_accumulation"],
        "events": events,
    }


def _tensor_raw(value):
    require(
        torch.is_tensor(value)
        and value.ndim == 3
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and value.shape[0] in (2, 64)
        and tuple(value.shape[1:]) == TENSOR_SHAPE
        and value.is_contiguous()
        and bool(torch.isfinite(value).all()),
        "owned finite contiguous CPU float32 CRB stage tensor",
    )
    # Little-endian IEEE float32 gives the byte payload a stable wire format;
    # conversion preserves signed zero and every finite float32 bit pattern.
    array = value.detach().numpy().astype("<f4", copy=False)
    return array.tobytes(order="C")


def _pack_stage_payload(value):
    header = _header(value)
    header_raw = canonical(header).encode("utf-8")
    require(
        0 < len(header_raw) <= MAX_HEADER_BYTES, "bounded canonical CRB stage header"
    )
    chunks = []
    for event in value["events"]:
        for stage in event["crb_accumulation_stages"]:
            for field in FIELDS:
                chunks.append(_tensor_raw(stage[field]))
    tensor_raw = b"".join(chunks)
    require(
        len(tensor_raw) == value["stage_payload_tensor_bytes"]
        and len(tensor_raw) <= crb.MAX_STAGE_PAYLOAD_BYTES,
        "exact bounded CRB stage tensor bytes",
    )
    framed = MAGIC + struct.pack("<I", len(header_raw)) + header_raw + tensor_raw
    require(0 < len(framed) <= MAX_ARTIFACT_BYTES, "bounded framed CRB stage artifact")
    return framed, header


def _validate_manifest(manifest):
    require(
        _exact_keys(manifest, _MANIFEST_KEYS)
        and type(manifest["protocol"]) is str
        and manifest["protocol"] == PROTOCOL
        and type(manifest["source"]) is str
        and len(manifest["source"]) == 40
        and all(char in "0123456789abcdef" for char in manifest["source"])
        and _is_int(manifest["worlds"])
        and manifest["worlds"] in (2, 64)
        and type(manifest["compiled_topology_sha256"]) is str
        and len(manifest["compiled_topology_sha256"]) == 64
        and all(
            char in "0123456789abcdef" for char in manifest["compiled_topology_sha256"]
        )
        and _is_int(manifest["inertia_bytes"])
        and 0 < manifest["inertia_bytes"] <= MAX_INERTIA_BYTES
        and _is_int(manifest["stage_bytes"])
        and PREFIX_BYTES < manifest["stage_bytes"] <= MAX_ARTIFACT_BYTES
        and _is_int(manifest["stage_payload_tensor_bytes"])
        and 0 < manifest["stage_payload_tensor_bytes"] <= MAX_ARTIFACT_BYTES
        and _is_int(manifest["stage_payload_limit_bytes"])
        and manifest["stage_payload_limit_bytes"] == crb.MAX_STAGE_PAYLOAD_BYTES
        and all(manifest[name] is False for name in crb.FLAGS),
        "exact bounded non-admitting CRB archive manifest",
    )
    for name in ("inertia_sha256", "stage_sha256"):
        digest = manifest[name]
        require(
            type(digest) is str
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "exact CRB archive " + name,
        )


def _parse_stage_payload(raw):
    require(
        type(raw) is bytes and PREFIX_BYTES < len(raw) <= MAX_ARTIFACT_BYTES,
        "bounded CRB stage archive bytes",
    )
    require(raw[: len(MAGIC)] == MAGIC, "exact CRB stage archive magic")
    header_len = struct.unpack("<I", raw[len(MAGIC) : PREFIX_BYTES])[0]
    require(
        0 < header_len <= MAX_HEADER_BYTES and PREFIX_BYTES + header_len < len(raw),
        "bounded CRB stage header framing",
    )
    header_raw = raw[PREFIX_BYTES : PREFIX_BYTES + header_len]
    try:
        header_text = header_raw.decode("utf-8")
        header = json.loads(
            header_text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
        canonical_header = canonical(header).encode("utf-8")
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid CRB stage archive JSON header") from error
    require(
        canonical_header == header_raw,
        "canonical CRB stage archive JSON header",
    )
    tensor_raw = raw[PREFIX_BYTES + header_len :]
    return header, tensor_raw


def _validate_topology(topology, worlds, topology_sha):
    require(
        _exact_keys(topology, _TOPOLOGY_KEYS)
        and all(
            _is_int(topology[name]) for name in ("nbody", "nq", "nv", "nu", "worlds")
        )
        and topology["nbody"] == 16
        and topology["nq"] == 21
        and topology["nv"] == 20
        and topology["nu"] == 14
        and topology["worlds"] == worlds
        and _int_list(topology["body_parentid"], 16)
        and len(topology["body_parentid"]) == 16
        and all(0 <= item < 16 for item in topology["body_parentid"])
        and type(topology["reversed_body_tree_ids"]) is list
        and 0 < len(topology["reversed_body_tree_ids"]) <= 16
        and all(
            _int_list(level, 16) and 0 < len(level) <= 16
            for level in topology["reversed_body_tree_ids"]
        )
        and sorted(
            body for level in topology["reversed_body_tree_ids"] for body in level
        )
        == list(range(16))
        and _sha(canonical(topology).encode("utf-8")) == topology_sha,
        "exact bounded CRB archive compiled topology",
    )
    return topology["reversed_body_tree_ids"]


def _validate_header(header, manifest):
    require(
        _exact_keys(header, _HEADER_KEYS)
        and header["protocol"] == HEADER_PROTOCOL
        and header["trace_protocol"] == crb.PROTOCOL
        and header["source"] == manifest["source"]
        and _is_int(header["worlds"])
        and header["worlds"] == manifest["worlds"]
        and header["launch_trace_boundary"] == crb.BOUNDARY
        and header["smooth_module"] == crb.SMOOTH_MODULE
        and header["smooth_source_sha256"] == crb.SMOOTH_SOURCE_SHA256
        and header["accumulation_kernel"] == "mujoco_warp._src.smooth._crb_accumulate"
        and header["compiled_topology_sha256"] == manifest["compiled_topology_sha256"]
        and _is_int(header["stage_payload_tensor_bytes"])
        and header["stage_payload_tensor_bytes"]
        == manifest["stage_payload_tensor_bytes"]
        and _is_int(header["stage_payload_limit_bytes"])
        and header["stage_payload_limit_bytes"] == crb.MAX_STAGE_PAYLOAD_BYTES
        and header["partial_crb_accumulation"] is None,
        "exact bounded CRB stage header metadata",
    )
    levels = _validate_topology(
        header["compiled_topology"],
        manifest["worlds"],
        manifest["compiled_topology_sha256"],
    )
    events = header["events"]
    require(
        type(events) is list and len(events) == 2 * 3,
        "exact six archived CRB events",
    )
    for event_index, event in enumerate(events):
        require(
            _exact_keys(event, _EVENT_KEYS)
            and type(event["phase"]) is str
            and _is_int(event["step"])
            and event["phase"]
            == ("scheduled-pre" if event_index % 2 == 0 else "unforced-post")
            and event["step"] == event_index // 2
            and type(event["stages"]) is list
            and len(event["stages"]) == len(levels),
            "exact ordered archived CRB event metadata",
        )
        for stage_index, stage in enumerate(event["stages"]):
            require(
                _exact_keys(stage, _STAGE_KEYS)
                and _is_int(stage["stage_index"])
                and stage["stage_index"] == stage_index
                and _int_list(stage["body_tree_ids"], 16)
                and stage["body_tree_ids"] == levels[stage_index]
                and _int_list(stage["dim"], 2)
                and len(stage["dim"]) == 2
                and stage["dim"] == [manifest["worlds"], len(levels[stage_index])],
                "exact ordered archived CRB stage metadata",
            )
    expected_tensor_bytes = (
        len(events)
        * len(levels)
        * manifest["worlds"]
        * TENSOR_SHAPE[0]
        * TENSOR_SHAPE[1]
        * 4
        * len(FIELDS)
    )
    require(
        expected_tensor_bytes == manifest["stage_payload_tensor_bytes"]
        and expected_tensor_bytes <= MAX_ARTIFACT_BYTES,
        "exact fixed CRB stage tensor layout",
    )
    return levels, expected_tensor_bytes


def _rebuild_stages(header, tensor_raw):
    worlds = header["worlds"]
    elements = worlds * TENSOR_SHAPE[0] * TENSOR_SHAPE[1]
    bytes_per_tensor = elements * 4
    offset = 0
    events = []
    for event in header["events"]:
        stages = []
        for stage in event["stages"]:
            rebuilt = dict(stage)
            for field in FIELDS:
                end = offset + bytes_per_tensor
                raw = tensor_raw[offset:end]
                require(len(raw) == bytes_per_tensor, "complete CRB stage tensor bytes")
                array = (
                    np.frombuffer(raw, dtype="<f4")
                    .copy()
                    .reshape((worlds, *TENSOR_SHAPE))
                )
                # Convert explicitly to native-endian float32 for torch; clone
                # owns the resulting storage independently of archive buffers.
                tensor = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
                require(bool(torch.isfinite(tensor).all()), "finite decoded CRB stage")
                rebuilt[field] = tensor.contiguous().clone()
                offset = end
            stages.append(rebuilt)
        events.append(stages)
    require(offset == len(tensor_raw), "no trailing CRB stage tensor bytes")
    return events


def encode(value, declaration, first_record):
    """Encode a fully checked trace into manifest plus two bounded blobs."""
    crb.inertia._hidden()
    projected = crb._project(value)
    _preflight_inertia_tree(projected)
    score = crb.check(value, declaration, first_record)
    inertia_raw = evidence.encode(projected)
    require(
        type(inertia_raw) is bytes and 0 < len(inertia_raw) <= MAX_INERTIA_BYTES,
        "bounded inherited inertia payload bytes",
    )
    stage_raw, header = _pack_stage_payload(value)
    manifest = {
        "protocol": PROTOCOL,
        "source": score["source"],
        "worlds": score["worlds"],
        "compiled_topology_sha256": value["compiled_topology_sha256"],
        "inertia_bytes": len(inertia_raw),
        "inertia_sha256": _sha(inertia_raw),
        "stage_bytes": len(stage_raw),
        "stage_sha256": _sha(stage_raw),
        "stage_payload_tensor_bytes": header["stage_payload_tensor_bytes"],
        "stage_payload_limit_bytes": crb.MAX_STAGE_PAYLOAD_BYTES,
        **crb.FLAGS,
    }
    _validate_manifest(manifest)
    return manifest, inertia_raw, stage_raw


def decode(
    manifest, inertia_raw, stage_raw, projected_inertia, declaration, first_record
):
    """Authenticate exact blobs, reconstruct, and re-run the unchanged reader.

    `projected_inertia` is the independently decoded old-protocol tree. This
    boundary deliberately performs no new pickle/torch deserialization.
    """
    crb.inertia._hidden()
    _validate_manifest(manifest)
    require(
        type(inertia_raw) is bytes
        and len(inertia_raw) == manifest["inertia_bytes"]
        and 0 < len(inertia_raw) <= MAX_INERTIA_BYTES
        and type(stage_raw) is bytes
        and len(stage_raw) == manifest["stage_bytes"]
        and PREFIX_BYTES < len(stage_raw) <= MAX_ARTIFACT_BYTES,
        "exact bounded CRB archive payload byte types and sizes",
    )
    require(
        _sha(inertia_raw) == manifest["inertia_sha256"]
        and _sha(stage_raw) == manifest["stage_sha256"],
        "whole CRB archive payload hashes before parsing",
    )
    header, tensor_raw = _parse_stage_payload(stage_raw)
    levels, expected_tensor_bytes = _validate_header(header, manifest)
    require(
        len(tensor_raw) == expected_tensor_bytes,
        "exact CRB stage payload bytes before allocation",
    )
    fresh_topology = crb._fresh_topology(manifest["worlds"])
    require(
        header["compiled_topology"] == fresh_topology
        and levels == fresh_topology["reversed_body_tree_ids"]
        and _sha(canonical(fresh_topology).encode("utf-8"))
        == manifest["compiled_topology_sha256"],
        "archive topology binds fresh compiled CPU topology before tensor allocation",
    )
    require(type(projected_inertia) is dict, "typed projected inherited inertia tree")
    _preflight_inertia_tree(projected_inertia)
    inherited_score = crb.inertia.check(projected_inertia, declaration, first_record)
    require(
        inherited_score["source"] == manifest["source"]
        and inherited_score["worlds"] == manifest["worlds"],
        "inherited inertia source/world binding",
    )
    require(
        evidence.encode(projected_inertia) == inertia_raw,
        "projected inertia tree binds exact archived whole bytes",
    )
    owned_projected = evidence._owned_tree(projected_inertia, clone=True)
    decoded_events = _rebuild_stages(header, tensor_raw)
    trace = dict(owned_projected)
    trace["protocol"] = crb.PROTOCOL
    trace["events"] = [
        {
            **event,
            "crb_accumulation_stages": stages,
        }
        for event, stages in zip(owned_projected["events"], decoded_events, strict=True)
    ]
    trace.update(
        launch_trace_boundary=header["launch_trace_boundary"],
        smooth_module=header["smooth_module"],
        smooth_source_sha256=header["smooth_source_sha256"],
        accumulation_kernel=header["accumulation_kernel"],
        compiled_topology=header["compiled_topology"],
        compiled_topology_sha256=header["compiled_topology_sha256"],
        stage_payload_tensor_bytes=header["stage_payload_tensor_bytes"],
        stage_payload_limit_bytes=header["stage_payload_limit_bytes"],
        partial_crb_accumulation=None,
        **crb.FLAGS,
    )
    score = crb.check(trace, declaration, first_record)
    return trace, score
