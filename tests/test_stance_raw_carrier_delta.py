"""Explicit CPU fixtures for the raw-carrier comparison draft."""

import hashlib
import struct

import pytest


from mjlab_microduck import stance_raw_carrier_delta as _MODULE


def _carrier(raw, shape, dtype="<f4", *, sha=None, size=None):
    return {
        "raw": raw,
        "shape": shape,
        "dtype": dtype,
        "bytes": len(raw) if size is None else size,
        "sha256": hashlib.sha256(raw).hexdigest() if sha is None else sha,
    }


def test_equal_float_carriers_return_hashes_and_literal_false_flags_without_bytes():
    raw = struct.pack("<4f", 1.0, -2.0, 0.0, 4.5)
    carrier = _carrier(raw, [2, 2])
    result = _MODULE.compare_carriers(carrier, carrier.copy())
    assert result["protocol"] == "microduck-raw-carrier-delta-oct8-v1"
    assert result["equal"] is True
    assert result["differing_32bit_word_count"] == 0
    assert result["first_difference"] is None
    assert result["words_compared"] == 4
    assert result["input_sha256"] == {
        "left": carrier["sha256"],
        "right": carrier["sha256"],
    }
    assert result["flags"] == {name: False for name in _MODULE.FLAGS}
    assert "raw" not in result and raw not in result.values()
    assert result == _MODULE.compare_carriers(carrier, carrier.copy())


def test_differences_report_exact_words_coordinates_and_values():
    left = _carrier(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0), [2, 2])
    right = _carrier(struct.pack("<4f", 1.0, 2.25, 3.0, 4.5), [2, 2])
    result = _MODULE.compare_carriers(left, right)
    assert result["equal"] is False
    assert result["differing_32bit_word_count"] == 2
    assert result["first_difference"] == {
        "word_index": 1,
        "coordinates": [0, 1],
        "left_word_le_hex": struct.pack("<f", 2.0).hex(),
        "right_word_le_hex": struct.pack("<f", 2.25).hex(),
        "left_value": 2.0,
        "right_value": 2.25,
    }


def test_signed_zero_is_a_raw_bit_difference_not_normalized():
    left = _carrier(struct.pack("<f", 0.0), [1])
    right = _carrier(struct.pack("<f", -0.0), [1])
    result = _MODULE.compare_carriers(left, right)
    assert result["equal"] is False
    assert result["differing_32bit_word_count"] == 1
    assert result["first_difference"]["left_value"] == 0.0
    assert result["first_difference"]["right_value"] == 0.0
    assert (
        result["first_difference"]["left_word_le_hex"]
        != result["first_difference"]["right_word_le_hex"]
    )


def test_integer_carriers_decode_signed_little_endian_values():
    left = _carrier(struct.pack("<3i", -1, 2, 3), [3], "<i4")
    right = _carrier(struct.pack("<3i", -1, 7, 3), [3], "<i4")
    result = _MODULE.compare_carriers(left, right)
    assert result["first_difference"]["coordinates"] == [1]
    assert result["first_difference"]["left_value"] == 2
    assert result["first_difference"]["right_value"] == 7


def test_extreme_signed_int32_values_decode_exactly():
    left = _carrier(struct.pack("<i", -(1 << 31)), [1], "<i4")
    right = _carrier(struct.pack("<i", (1 << 31) - 1), [1], "<i4")
    result = _MODULE.compare_carriers(left, right)
    assert result["differing_32bit_word_count"] == 1
    assert result["first_difference"]["left_value"] == -(1 << 31)
    assert result["first_difference"]["right_value"] == (1 << 31) - 1


def test_four_dimensional_coordinates_decode_in_row_major_order():
    left = _carrier(struct.pack("<8i", *([0] * 8)), [1, 2, 2, 2], "<i4")
    values = [0] * 8
    values[6] = -17
    right = _carrier(struct.pack("<8i", *values), [1, 2, 2, 2], "<i4")
    result = _MODULE.compare_carriers(left, right)
    assert result["first_difference"]["word_index"] == 6
    assert result["first_difference"]["coordinates"] == [0, 1, 1, 0]


def test_scalar_and_empty_shapes_have_deterministic_coordinates_and_counts():
    scalar = _carrier(struct.pack("<i", 4), [], "<i4")
    changed_scalar = _carrier(struct.pack("<i", 5), [], "<i4")
    scalar_result = _MODULE.compare_carriers(scalar, changed_scalar)
    assert scalar_result["first_difference"]["coordinates"] == []
    empty = _carrier(b"", [3, 0, 2])
    empty_result = _MODULE.compare_carriers(empty, empty.copy())
    assert empty_result["equal"] is True
    assert empty_result["words_compared"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "mutable",
        "tuple-shape",
        "negative-dimension",
        "dtype",
        "size-bool",
        "wrong-length",
        "hash",
        "uppercase-hash",
        "dimension",
        "rank",
        "cap",
    ],
)
def test_carrier_authentication_refuses_malformed_schema_types_and_bounds(
    mutation, monkeypatch
):
    raw = struct.pack("<2f", 1.0, 2.0)
    carrier = _carrier(raw, [2])
    if mutation == "schema":
        carrier["extra"] = False
    elif mutation == "mutable":
        carrier["raw"] = bytearray(raw)
    elif mutation == "tuple-shape":
        carrier["shape"] = (2,)
    elif mutation == "negative-dimension":
        carrier["shape"] = [-2]
    elif mutation == "shape":
        carrier["shape"] = [True]
    elif mutation == "dtype":
        carrier["dtype"] = " <f4"
    elif mutation == "size-bool":
        carrier["bytes"] = True
    elif mutation == "wrong-length":
        carrier["bytes"] += 4
    elif mutation == "hash":
        carrier["sha256"] = "0" * 64
    elif mutation == "uppercase-hash":
        carrier["sha256"] = carrier["sha256"].upper()
    elif mutation == "dimension":
        carrier["shape"] = [(1 << 20) + 1]
    elif mutation == "rank":
        carrier["shape"] = [1, 1, 1, 1, 1]
    elif mutation == "cap":
        monkeypatch.setattr(_MODULE, "MAX_BYTES", 4)
    with pytest.raises(ValueError):
        _MODULE.compare_carriers(carrier, carrier.copy())


def test_both_inputs_are_authenticated_before_any_word_decoding(monkeypatch):
    left = _carrier(struct.pack("<f", 1.0), [1])
    right = _carrier(struct.pack("<f", 2.0), [1], sha="0" * 64)
    called = []
    original = _MODULE.struct.unpack

    def tracked_unpack(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_MODULE.struct, "unpack", tracked_unpack)
    with pytest.raises(ValueError, match="whole carrier SHA256 right"):
        _MODULE.compare_carriers(left, right)
    assert called == []


def test_right_only_nonfinite_word_is_refused_after_valid_left_decoding():
    left = _carrier(struct.pack("<f", 1.25), [1])
    right = _carrier(struct.pack("<f", float("nan")), [1])
    with pytest.raises(ValueError, match="finite float32 word right"):
        _MODULE.compare_carriers(left, right)


@pytest.mark.parametrize(
    "left,right",
    [
        (
            _carrier(struct.pack("<f", 1.0), [1]),
            _carrier(struct.pack("<f", 1.0), [1], "<i4"),
        ),
        (
            _carrier(struct.pack("<2f", 1.0, 2.0), [2]),
            _carrier(struct.pack("<2f", 1.0, 2.0), [1, 2]),
        ),
    ],
)
def test_carriers_must_have_identical_shape_and_dtype(left, right):
    with pytest.raises(ValueError, match="identical carrier"):
        _MODULE.compare_carriers(left, right)


@pytest.mark.parametrize(
    "raw", [struct.pack("<f", float("nan")), struct.pack("<f", float("inf"))]
)
def test_float_carriers_refuse_nonfinite_values(raw):
    carrier = _carrier(raw, [1])
    with pytest.raises(ValueError, match="finite float32"):
        _MODULE.compare_carriers(carrier, carrier.copy())
