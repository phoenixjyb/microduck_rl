"""Pure raw-carrier comparison; this is not runtime or admission evidence."""

import hashlib
import math
import re
import struct


PROTOCOL = "microduck-raw-carrier-delta-oct8-v1"
MAX_BYTES = 16 * 1024**2
MAX_DIMENSION = 1 << 20
FIELDS = frozenset({"raw", "shape", "dtype", "bytes", "sha256"})
FLAGS = (
    "runtime_cause_proven",
    "native_qualified",
    "full_window_qualified",
    "training_authorized",
    "physical_acceptance",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _authenticate(value, label):
    """Validate immutable bytes and their external metadata without decoding."""
    _need(type(value) is dict and set(value) == FIELDS, "exact carrier schema " + label)
    shape = value["shape"]
    _need(type(shape) is list and len(shape) <= 4, "bounded plain shape " + label)
    for dimension in shape:
        _need(
            type(dimension) is int and 0 <= dimension <= MAX_DIMENSION,
            "plain bounded shape dimension " + label,
        )
    dtype = value["dtype"]
    _need(
        type(dtype) is str and dtype in ("<f4", "<i4"),
        "literal little-endian dtype " + label,
    )
    size = value["bytes"]
    _need(
        type(size) is int and 0 <= size <= MAX_BYTES,
        "bounded plain byte count " + label,
    )
    raw = value["raw"]
    _need(type(raw) is bytes, "immutable bytes carrier " + label)
    expected_size = _product(shape) * 4
    _need(
        size == expected_size and len(raw) == expected_size,
        "exact carrier byte length " + label,
    )
    expected_sha = value["sha256"]
    _need(
        type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
        "literal SHA256 metadata " + label,
    )
    actual_sha = hashlib.sha256(raw).hexdigest()
    _need(actual_sha == expected_sha, "whole carrier SHA256 " + label)
    return {"raw": raw, "shape": tuple(shape), "dtype": dtype, "sha256": actual_sha}


def _coordinates(index, shape):
    if not shape:
        return []
    coordinates = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        dimension = shape[axis]
        coordinates[axis] = index % dimension
        index //= dimension
    return coordinates


def _decoded(word, dtype, label):
    if dtype == "<f4":
        value = struct.unpack("<f", word)[0]
        _need(math.isfinite(value), "finite float32 word " + label)
        return value
    return struct.unpack("<i", word)[0]


def compare_carriers(left, right):
    """Compare two fully authenticated little-endian raw carriers bit for bit.

    This reports only local array differences. It carries no source identity,
    runtime cause, readiness, qualification, or promotion claim.
    """
    left_checked = _authenticate(left, "left")
    right_checked = _authenticate(right, "right")
    _need(left_checked["shape"] == right_checked["shape"], "identical carrier shapes")
    _need(left_checked["dtype"] == right_checked["dtype"], "identical carrier dtypes")

    dtype = left_checked["dtype"]
    left_raw, right_raw = left_checked["raw"], right_checked["raw"]
    word_count = len(left_raw) // 4
    differing = 0
    first_difference = None
    for index in range(word_count):
        start = index * 4
        left_word, right_word = (
            left_raw[start : start + 4],
            right_raw[start : start + 4],
        )
        left_value = _decoded(left_word, dtype, "left")
        right_value = _decoded(right_word, dtype, "right")
        if left_word != right_word:
            differing += 1
            if first_difference is None:
                first_difference = {
                    "word_index": index,
                    "coordinates": _coordinates(index, left_checked["shape"]),
                    "left_word_le_hex": left_word.hex(),
                    "right_word_le_hex": right_word.hex(),
                    "left_value": left_value,
                    "right_value": right_value,
                }

    return {
        "protocol": PROTOCOL,
        "equal": left_raw == right_raw,
        "shape": list(left_checked["shape"]),
        "dtype": dtype,
        "words_compared": word_count,
        "differing_32bit_word_count": differing,
        "first_difference": first_difference,
        "input_sha256": {
            "left": left_checked["sha256"],
            "right": right_checked["sha256"],
        },
        "flags": {name: False for name in FLAGS},
    }
