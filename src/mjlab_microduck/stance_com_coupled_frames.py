"""Byte-authenticated NumPy comparisons for paired coupled-forward frames.

This helper compares saved evidence only. Its result does not qualify a CUDA
run, establish runtime cause, or authorize training or physical acceptance.
"""

from hashlib import sha256
from itertools import combinations
import re

import numpy as np


CASES = ("original0", "original1", "serial0", "serial1")
BOUNDARIES = ("constructor", "eager_forward")
WORLDS = 64
FRAME_FIELDS = (
    ("subtree_com", (16, 3)),
    ("cinert", (16, 10)),
    ("cdof", (20, 6)),
    ("crb", (16, 10)),
    ("qM", (20, 20)),
    ("qLD", (20, 20)),
    ("cvel", (16, 6)),
    ("cdof_dot", (20, 6)),
    ("qfrc_bias", (20,)),
    ("qfrc_smooth", (20,)),
    ("qpos", (21,)),
    ("qvel", (20,)),
    ("time", ()),
    ("qacc_warmstart", (20,)),
    ("ctrl", (14,)),
    ("xfrc_applied", (16, 6)),
    ("qfrc_applied", (20,)),
)
BODY_FIELDS = frozenset(("subtree_com", "cinert", "crb", "cvel", "xfrc_applied"))
FIXED_INPUT_FIELDS = (
    "qpos",
    "qvel",
    "time",
    "qacc_warmstart",
    "ctrl",
    "xfrc_applied",
    "qfrc_applied",
)
FALSE_FLAGS = {
    "original_run_entry_captured": False,
    "original_pair_accepted": False,
    "actual_kernel_order_observed": False,
    "runtime_cause_proven": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
}

_FRAME_FLOATS = sum(
    WORLDS * int(np.prod(shape, dtype=np.int64)) if shape else WORLDS
    for _, shape in FRAME_FIELDS
)
FRAME_BYTES = _FRAME_FLOATS * 4
PACKET_BYTES = FRAME_BYTES * len(BOUNDARIES)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_U32 = np.dtype("<u4")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _authenticate(packets, hashes):
    _require(
        type(packets) is dict and packets.keys() == set(CASES), "exact packet cases"
    )
    _require(type(hashes) is dict and hashes.keys() == set(CASES), "exact hash cases")

    observed = {}
    for case in CASES:
        packet = packets[case]
        digest = hashes[case]
        _require(type(packet) is bytes, case + " packet must be immutable bytes")
        _require(
            len(packet) == PACKET_BYTES,
            case + " packet must contain exactly two frames",
        )
        _require(
            type(digest) is str and _SHA256.fullmatch(digest) is not None,
            case + " external SHA-256 must be 64 lowercase hex characters",
        )
        observed[case] = sha256(packet).hexdigest()

    _require(observed == hashes, "whole packet SHA-256 authentication failed")


def _decode_packet(packet, case):
    frames = []
    for boundary_index, boundary in enumerate(BOUNDARIES):
        frame_start = boundary_index * FRAME_BYTES
        frame_end = frame_start + FRAME_BYTES
        frame_bytes = memoryview(packet)[frame_start:frame_end]
        offset = 0
        fields = {}
        for name, shape in FRAME_FIELDS:
            count = WORLDS * int(np.prod(shape, dtype=np.int64)) if shape else WORLDS
            byte_count = count * 4
            values = np.frombuffer(
                frame_bytes,
                dtype=np.dtype("<f4"),
                count=count,
                offset=offset,
            )
            expected_shape = (WORLDS, *shape) if shape else (WORLDS,)
            values = values.reshape(expected_shape)
            _require(
                bool(np.isfinite(values).all()),
                case + " " + boundary + " " + name + " must be finite",
            )
            fields[name] = values
            offset += byte_count
        _require(offset == FRAME_BYTES, "literal frame schema byte count")
        frames.append(fields)
    return frames


def _field_row(left, right, name):
    left_bytes = left.tobytes(order="C")
    right_bytes = right.tobytes(order="C")
    mismatch = left.view(_U32) != right.view(_U32)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    result = {
        "left_sha256": sha256(left_bytes).hexdigest(),
        "right_sha256": sha256(right_bytes).hexdigest(),
        "scalars": int(left.size),
        "bit_mismatch_scalars": int(mismatch.sum()),
        "max_abs_delta": float(delta.max()),
        "exact_raw_bits": not bool(mismatch.any()),
    }
    if name in BODY_FIELDS:
        per_body = np.moveaxis(mismatch, 1, 0).reshape(16, -1).sum(axis=1)
        result["body_mismatch_scalars"] = {
            "body0": int(per_body[0]),
            "body1": int(per_body[1]),
            "others": int(per_body[2:].sum()),
        }
    return result


def compare(packets, hashes):
    """Authenticate and compare four complete two-frame packet byte strings."""
    _authenticate(packets, hashes)

    # Decode only after every packet length and every whole-packet digest passed.
    decoded = {case: _decode_packet(packets[case], case) for case in CASES}
    comparisons = []
    for left_case, right_case in combinations(CASES, 2):
        for boundary_index, boundary in enumerate(BOUNDARIES):
            left = decoded[left_case][boundary_index]
            right = decoded[right_case][boundary_index]
            comparisons.append(
                {
                    "left": left_case,
                    "right": right_case,
                    "boundary": boundary,
                    "fields": {
                        name: _field_row(left[name], right[name], name)
                        for name, _ in FRAME_FIELDS
                    },
                }
            )

    def repeat_exact(left_case, right_case):
        return all(
            row["fields"][name]["exact_raw_bits"]
            for row in comparisons
            if row["left"] == left_case and row["right"] == right_case
            for name, _ in FRAME_FIELDS
        )

    fixed_inputs_exact = all(
        np.array_equal(
            decoded[CASES[0]][0][name].view(_U32),
            decoded[case][boundary_index][name].view(_U32),
        )
        for boundary_index in range(len(BOUNDARIES))
        for case in CASES[1:]
        for name in FIXED_INPUT_FIELDS
    )
    original_repeat_exact = repeat_exact("original0", "original1")
    serial_repeat_exact = repeat_exact("serial0", "serial1")
    decision = (
        "coupled-serial-repeat-exact"
        if serial_repeat_exact and fixed_inputs_exact
        else "coupled-serial-repeat-negative"
    )
    return {
        "protocol": "microduck-com-coupled-frames-oct7-v1",
        "cases": list(CASES),
        "boundaries": list(BOUNDARIES),
        "field_order": [name for name, _ in FRAME_FIELDS],
        "worlds": WORLDS,
        "frame_bytes": FRAME_BYTES,
        "packet_bytes": PACKET_BYTES,
        "authenticated_packet_sha256": {case: hashes[case] for case in CASES},
        "comparisons": comparisons,
        "original_repeat_exact": original_repeat_exact,
        "serial_repeat_exact": serial_repeat_exact,
        "all_cases_fixed_inputs_exact": fixed_inputs_exact,
        "decision": decision,
        "flags": dict(FALSE_FLAGS),
    }
