"""Synthetic tests for authenticated constructor-to-forward frame analysis."""

from hashlib import sha256

import numpy as np
import pytest

from mjlab_microduck import stance_com_coupled_frames as frames
from mjlab_microduck import stance_com_coupled_idempotence as analysis


def packet(overrides=()):
    overrides = {
        (boundary, name, index): value for boundary, name, index, value in overrides
    }
    chunks = []
    for boundary_index in range(2):
        for field_index, (name, suffix) in enumerate(frames.FRAME_FIELDS):
            shape = (frames.WORLDS, *suffix) if suffix else (frames.WORLDS,)
            count = int(np.prod(shape))
            values = (
                np.arange(count, dtype=np.float32).reshape(shape) / count
                + np.float32(field_index * 2)
            ).astype("<f4")
            for (which_boundary, which_name, index), value in overrides.items():
                if which_boundary == boundary_index and which_name == name:
                    values[index] = value
            chunks.append(values.tobytes(order="C"))
    return b"".join(chunks)


def packet_hashes(packets):
    return {case: sha256(packets[case]).hexdigest() for case in frames.CASES}


@pytest.fixture
def exact_packets():
    raw = packet()
    return {case: raw for case in frames.CASES}


def test_identical_packets_produce_all_68_full_field_rows(exact_packets):
    result = analysis.analyze(exact_packets, packet_hashes(exact_packets))
    assert result["protocol"] == analysis.PROTOCOL
    assert result["authenticated_packet_bytes"] == {
        case: frames.PACKET_BYTES for case in frames.CASES
    }
    assert result["authenticated_packet_sha256"] == packet_hashes(exact_packets)
    assert len(result["temporal_comparisons"]) == 4
    assert sum(len(row["fields"]) for row in result["temporal_comparisons"]) == 68
    assert all(
        value["exact_raw_bits"]
        and value["bit_mismatch_scalars"] == 0
        and value["max_abs_delta"] == 0.0
        and value["world_mismatch_scalars"] == [0] * 64
        for row in result["temporal_comparisons"]
        for value in row["fields"].values()
    )
    assert result["negative_coordinate_count"] == 0
    assert result["negative_coordinates"] == []
    assert result["serial_temporal_exact"] is True
    assert result["original_temporal_exact"] is True
    assert result["all_cases_fixed_inputs_temporal_exact"] is True
    assert result["decision"] == "temporal-serial-exact"
    assert all(value is False for value in result["flags"].values())


def test_three_ulp_bias_and_smooth_differences_are_retained_without_waiver():
    before = np.float32(-0.004055817145854235)
    after = np.float32(-0.004055818542838097)
    negative_before, negative_after = -before, -after
    # Constructor and forward hold the same values in the baseline packet.
    baseline = packet(
        (
            (0, "qfrc_bias", (53, 4), before),
            (1, "qfrc_bias", (53, 4), before),
            (0, "qfrc_smooth", (53, 4), negative_before),
            (1, "qfrc_smooth", (53, 4), negative_before),
        )
    )
    changed = packet(
        (
            (0, "qfrc_bias", (53, 4), before),
            (1, "qfrc_bias", (53, 4), after),
            (0, "qfrc_smooth", (53, 4), negative_before),
            (1, "qfrc_smooth", (53, 4), negative_after),
        )
    )
    values = {case: baseline for case in frames.CASES}
    values["serial0"] = changed
    values["serial1"] = changed

    result = analysis.analyze(values, packet_hashes(values))
    assert result["original_temporal_exact"] is True
    assert result["serial_temporal_exact"] is False
    assert result["decision"] == "temporal-serial-negative"
    assert result["negative_coordinate_count"] == 4
    assert {
        (
            row["case"],
            row["field"],
            row["world"],
            tuple(row["field_indices"]),
        )
        for row in result["negative_coordinates"]
    } == {
        (case, field, 53, (4,))
        for case in ("serial0", "serial1")
        for field in ("qfrc_bias", "qfrc_smooth")
    }
    for row in result["negative_coordinates"]:
        assert row["ordered_bit_distance"] == 3
        assert row["world"] == 53
        assert row["field_indices"] == [4]
        assert row["left_u32"] != row["right_u32"]
        assert row["left_f32"] != row["right_f32"]
    serial_row = next(
        row for row in result["temporal_comparisons"] if row["case"] == "serial0"
    )
    for field in ("qfrc_bias", "qfrc_smooth"):
        summary = serial_row["fields"][field]
        assert summary["bit_mismatch_scalars"] == 1
        assert summary["world_mismatch_scalars"][53] == 1
        assert sum(summary["world_mismatch_scalars"]) == 1


def test_signed_zero_is_retained_with_one_ordered_bit_step():
    positive_zero = packet(
        (
            (0, "qfrc_bias", (53, 4), 0.0),
            (1, "qfrc_bias", (53, 4), 0.0),
        )
    )
    negative_zero = packet(
        (
            (0, "qfrc_bias", (53, 4), 0.0),
            (1, "qfrc_bias", (53, 4), -0.0),
        )
    )
    values = {case: positive_zero for case in frames.CASES}
    values["serial0"] = negative_zero
    result = analysis.analyze(values, packet_hashes(values))
    row = next(
        row
        for row in result["negative_coordinates"]
        if row["case"] == "serial0" and row["field"] == "qfrc_bias"
    )
    assert row["ordered_bit_distance"] == 1
    assert row["left_f32"] == row["right_f32"] == 0.0
    assert row["left_u32"] != row["right_u32"]
    assert result["serial_temporal_exact"] is False


def test_constructor_to_forward_fixed_input_mutation_is_negative():
    baseline = packet()
    changed = packet(((1, "qpos", (53, 4), 999.0),))
    values = {case: baseline for case in frames.CASES}
    values["serial1"] = changed
    result = analysis.analyze(values, packet_hashes(values))
    assert result["serial_temporal_exact"] is False
    assert result["all_cases_fixed_inputs_temporal_exact"] is False
    assert result["decision"] == "temporal-serial-negative"
    assert any(
        row["case"] == "serial1"
        and row["field"] == "qpos"
        and row["world"] == 53
        and row["field_indices"] == [4]
        for row in result["negative_coordinates"]
    )


def test_external_whole_packet_hash_is_checked_before_numpy_decode(
    exact_packets, monkeypatch
):
    hashes = packet_hashes(exact_packets)
    hashes["serial1"] = "0" * 64

    def forbidden(*_args, **_kwargs):
        raise AssertionError("packet decoded before all four hashes passed")

    monkeypatch.setattr(frames.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="authentication"):
        analysis.analyze(exact_packets, hashes)


def test_mutable_and_wrong_size_packets_are_rejected(exact_packets):
    hashes = packet_hashes(exact_packets)
    mutable = dict(exact_packets)
    mutable["serial1"] = bytearray(mutable["serial1"])
    with pytest.raises(ValueError, match="immutable bytes"):
        analysis.analyze(mutable, hashes)
    short = dict(exact_packets)
    short["serial1"] = short["serial1"][:-4]
    with pytest.raises(ValueError, match="exactly two frames"):
        analysis.analyze(short, packet_hashes(short))


def test_authenticated_nonfinite_packet_is_rejected():
    nonfinite = packet(((1, "qfrc_smooth", (53, 4), np.nan),))
    values = {case: packet() for case in frames.CASES}
    values["serial1"] = nonfinite
    with pytest.raises(ValueError, match="must be finite"):
        analysis.analyze(values, packet_hashes(values))


def test_refuses_more_than_1024_total_coordinates_before_argwhere(
    exact_packets, monkeypatch
):
    changed = packet(
        (1, "qfrc_bias", (world, dof), 999.0)
        for world in range(64)
        for dof in range(20)
    )
    values = dict(exact_packets)
    values["serial1"] = changed

    monkeypatch.setattr(
        analysis.np,
        "argwhere",
        lambda *_args, **_kwargs: pytest.fail("oversized coordinates were allocated"),
    )
    with pytest.raises(ValueError, match="1280 > 1024"):
        analysis.analyze(values, packet_hashes(values))
