"""Per-case constructor-to-forward analysis of complete coupled frame packets.

This pure NumPy post-run report preserves raw packet hashes and all scalar
mismatches. It does not accept the original control, qualify runtime cause, or
authorize training or physical acceptance.
"""

from hashlib import sha256

import numpy as np

from mjlab_microduck import stance_com_coupled_frames as frames


PROTOCOL = "microduck-com-coupled-idempotence-oct7-v1"
MAX_NEGATIVE_COORDINATES = 1024
FALSE_FLAGS = {
    "original_run_entry_captured": False,
    "original_pair_accepted": False,
    "actual_kernel_order_observed": False,
    "runtime_cause_proven": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
}
_U32 = np.dtype("<u4")
_SIGN = 0x80000000
_U32_MASK = 0xFFFFFFFF


def _ordered_bits(bits):
    """Map IEEE binary32 bits to an integer ordered by numeric value."""
    return (~bits & _U32_MASK) if bits & _SIGN else (bits | _SIGN)


def _coordinate(left, right, index):
    left_bits = int(left.view(_U32)[index])
    right_bits = int(right.view(_U32)[index])
    left_value = np.float32(left[index])
    right_value = np.float32(right[index])
    return {
        "indices": list(index),
        "world": int(index[0]),
        "field_indices": list(index[1:]),
        "left_f32": float(left_value),
        "right_f32": float(right_value),
        "left_u32": left_bits,
        "right_u32": right_bits,
        "ordered_bit_distance": abs(
            _ordered_bits(left_bits) - _ordered_bits(right_bits)
        ),
    }


def analyze(packets, hashes):
    """Report complete per-case temporal changes across two authenticated frames.

    All four packet lengths and external whole-packet SHA-256 values are checked
    before the first NumPy decode. If the complete report would contain more
    than ``MAX_NEGATIVE_COORDINATES`` mismatch coordinates, analysis refuses to
    materialize any coordinate rows rather than truncating them.
    """
    frames._authenticate(packets, hashes)

    # Authentication is complete for all four immutable packets before decode.
    decoded = {
        case: frames._decode_packet(packets[case], case) for case in frames.CASES
    }
    field_rows = []
    masks = []
    total_negative_coordinates = 0

    for case in frames.CASES:
        constructor, eager_forward = decoded[case]
        fields = {}
        for name, _shape in frames.FRAME_FIELDS:
            left, right = constructor[name], eager_forward[name]
            mismatch = left.view(_U32) != right.view(_U32)
            count = int(np.count_nonzero(mismatch))
            total_negative_coordinates += count
            summary = frames._field_row(left, right, name)
            summary["world_mismatch_scalars"] = [
                int(value) for value in mismatch.reshape(frames.WORLDS, -1).sum(axis=1)
            ]
            fields[name] = summary
            masks.append((case, name, left, right, mismatch, count))
        field_rows.append(
            {
                "case": case,
                "from_boundary": frames.BOUNDARIES[0],
                "to_boundary": frames.BOUNDARIES[1],
                "fields": fields,
            }
        )

    if total_negative_coordinates > MAX_NEGATIVE_COORDINATES:
        raise ValueError(
            "complete mismatch coordinate count exceeds limit: "
            f"{total_negative_coordinates} > {MAX_NEGATIVE_COORDINATES}"
        )

    negative_coordinates = []
    for case, name, left, right, mismatch, count in masks:
        if count == 0:
            continue
        coordinates = np.argwhere(mismatch)
        # The full count was checked above; this guards an impossible mask drift.
        frames._require(len(coordinates) == count, "stable complete mismatch mask")
        negative_coordinates.extend(
            {
                "case": case,
                "field": name,
                **_coordinate(left, right, tuple(int(value) for value in index)),
            }
            for index in coordinates
        )

    temporal_exact = {
        row["case"]: all(field["exact_raw_bits"] for field in row["fields"].values())
        for row in field_rows
    }
    serial_temporal_exact = all(temporal_exact[case] for case in ("serial0", "serial1"))
    original_temporal_exact = all(
        temporal_exact[case] for case in ("original0", "original1")
    )
    all_cases_fixed_inputs_temporal_exact = all(
        row["fields"][name]["exact_raw_bits"]
        for row in field_rows
        for name in frames.FIXED_INPUT_FIELDS
    )

    return {
        "protocol": PROTOCOL,
        "cases": list(frames.CASES),
        "boundaries": list(frames.BOUNDARIES),
        "field_order": [name for name, _ in frames.FRAME_FIELDS],
        "worlds": frames.WORLDS,
        "frame_bytes": frames.FRAME_BYTES,
        "packet_bytes": frames.PACKET_BYTES,
        "authenticated_packet_bytes": {
            case: len(packets[case]) for case in frames.CASES
        },
        "authenticated_packet_sha256": {
            case: sha256(packets[case]).hexdigest() for case in frames.CASES
        },
        "temporal_comparisons": field_rows,
        "negative_coordinate_count": total_negative_coordinates,
        "negative_coordinates": negative_coordinates,
        "serial_temporal_exact": serial_temporal_exact,
        "original_temporal_exact": original_temporal_exact,
        "all_cases_fixed_inputs_temporal_exact": all_cases_fixed_inputs_temporal_exact,
        "decision": (
            "temporal-serial-exact"
            if serial_temporal_exact
            else "temporal-serial-negative"
        ),
        "flags": dict(FALSE_FLAGS),
    }
