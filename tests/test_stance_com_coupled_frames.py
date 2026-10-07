"""Pure NumPy tests for complete paired coupled-forward frame packets."""

from hashlib import sha256

import numpy as np
import pytest

from mjlab_microduck import stance_com_coupled_frames as frames


EXPECTED_FIELDS = (
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


def packet(overrides=()):
    """Build both frames from field-major, little-endian synthetic arrays."""
    overrides = {
        (boundary, name, index): value for boundary, name, index, value in overrides
    }
    chunks = []
    for boundary_index, _ in enumerate(frames.BOUNDARIES):
        for field_index, (name, suffix) in enumerate(EXPECTED_FIELDS):
            shape = (frames.WORLDS, *suffix) if suffix else (frames.WORLDS,)
            count = int(np.prod(shape))
            values = (
                np.arange(count, dtype=np.float32).reshape(shape) / count
                + np.float32(
                    field_index * 2
                    + (
                        boundary_index * 40
                        if name not in frames.FIXED_INPUT_FIELDS
                        else 0
                    )
                )
            ).astype("<f4")
            for (override_boundary, override_name, index), value in overrides.items():
                if override_boundary == boundary_index and override_name == name:
                    values[index] = value
            chunks.append(values.tobytes(order="C"))
    return b"".join(chunks)


def authenticated(packet_bytes_by_case):
    return {
        case: sha256(packet_bytes_by_case[case]).hexdigest() for case in frames.CASES
    }


@pytest.fixture
def identical_packets():
    raw = packet()
    return {case: raw for case in frames.CASES}


def test_literal_schema_and_all_full_comparisons(identical_packets):
    assert frames.CASES == ("original0", "original1", "serial0", "serial1")
    assert frames.BOUNDARIES == ("constructor", "eager_forward")
    assert frames.FRAME_FIELDS == EXPECTED_FIELDS
    result = frames.compare(identical_packets, authenticated(identical_packets))
    assert result["frame_bytes"] == frames.FRAME_BYTES
    assert result["packet_bytes"] == len(next(iter(identical_packets.values())))
    assert len(result["comparisons"]) == 12
    assert sum(len(row["fields"]) for row in result["comparisons"]) == 204
    assert all(
        tuple(row["fields"]) == tuple(n for n, _ in EXPECTED_FIELDS)
        for row in result["comparisons"]
    )
    assert all(
        field["scalars"] == 64 * int(np.prod(shape) if shape else 1)
        and field["bit_mismatch_scalars"] == 0
        and field["max_abs_delta"] == 0.0
        and field["exact_raw_bits"] is True
        for row in result["comparisons"]
        for (name, shape), field in zip(
            EXPECTED_FIELDS, row["fields"].values(), strict=True
        )
    )
    assert result["original_repeat_exact"] is True
    assert result["serial_repeat_exact"] is True
    assert result["all_cases_fixed_inputs_exact"] is True
    assert result["decision"] == "coupled-serial-repeat-exact"
    assert result["flags"] == {
        "original_run_entry_captured": False,
        "original_pair_accepted": False,
        "actual_kernel_order_observed": False,
        "runtime_cause_proven": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
    }


@pytest.mark.parametrize(
    ("coordinate", "body_counts"),
    [
        ((0, 0, 0), {"body0": 1, "body1": 0, "others": 0}),
        ((7, 1, 2), {"body0": 0, "body1": 1, "others": 0}),
        ((63, 9, 1), {"body0": 0, "body1": 0, "others": 1}),
    ],
    ids=("world0-root0", "trunk1", "other-body"),
)
def test_body_mismatch_summary_keeps_world_and_body_axes(coordinate, body_counts):
    baseline = packet()
    changed = packet(((0, "subtree_com", coordinate, 1000.0),))
    values = {case: baseline for case in frames.CASES}
    values["serial1"] = changed
    result = frames.compare(values, authenticated(values))
    row = next(
        row
        for row in result["comparisons"]
        if row["left"] == "serial0"
        and row["right"] == "serial1"
        and row["boundary"] == "constructor"
    )
    field = row["fields"]["subtree_com"]
    assert field["bit_mismatch_scalars"] == 1
    assert field["body_mismatch_scalars"] == body_counts
    assert result["serial_repeat_exact"] is False
    assert result["decision"] == "coupled-serial-repeat-negative"


def test_constant_wrong_field_reports_every_scalar_and_delta():
    baseline = packet()
    wrong = packet(
        (0, "qfrc_bias", (world, joint), 100.0)
        for world in range(64)
        for joint in range(20)
    )
    values = {case: baseline for case in frames.CASES}
    values["serial1"] = wrong
    result = frames.compare(values, authenticated(values))
    row = next(
        row
        for row in result["comparisons"]
        if row["left"] == "serial0"
        and row["right"] == "serial1"
        and row["boundary"] == "constructor"
    )
    assert row["fields"]["qfrc_bias"]["bit_mismatch_scalars"] == 64 * 20
    assert row["fields"]["qfrc_bias"]["max_abs_delta"] > 0.0


def test_signed_zero_is_a_raw_bit_mismatch_with_zero_numeric_delta():
    plus_zero = packet(((0, "qvel", (0, 0), 0.0),))
    minus_zero = packet(((0, "qvel", (0, 0), -0.0),))
    values = {case: plus_zero for case in frames.CASES}
    values["serial1"] = minus_zero
    result = frames.compare(values, authenticated(values))
    row = next(
        row
        for row in result["comparisons"]
        if row["left"] == "serial0"
        and row["right"] == "serial1"
        and row["boundary"] == "constructor"
    )
    assert row["fields"]["qvel"]["bit_mismatch_scalars"] == 1
    assert row["fields"]["qvel"]["max_abs_delta"] == 0.0
    assert row["fields"]["qvel"]["exact_raw_bits"] is False


def test_fixed_inputs_must_match_all_cases_and_both_boundaries():
    baseline = packet()
    changed = packet(((1, "ctrl", (12, 4), 999.0),))
    values = {case: baseline for case in frames.CASES}
    values["original1"] = changed
    result = frames.compare(values, authenticated(values))
    assert result["all_cases_fixed_inputs_exact"] is False
    assert result["decision"] == "coupled-serial-repeat-negative"
    assert result["serial_repeat_exact"] is True


def test_fixed_inputs_compare_all_later_frames_to_original0_constructor():
    baseline = packet()
    changed_later = packet(
        (1, "qpos", (world, coordinate), 999.0)
        for world in range(64)
        for coordinate in range(21)
    )
    values = {case: baseline for case in frames.CASES}
    for case in frames.CASES:
        values[case] = changed_later
    result = frames.compare(values, authenticated(values))
    assert result["all_cases_fixed_inputs_exact"] is False
    assert result["decision"] == "coupled-serial-repeat-negative"
    assert result["serial_repeat_exact"] is True


def test_wrong_hash_fails_before_any_numpy_decode(monkeypatch, identical_packets):
    digests = authenticated(identical_packets)
    digests["serial0"] = "0" * 64

    def forbidden(*_args, **_kwargs):
        raise AssertionError("decode started before all packet digests authenticated")

    monkeypatch.setattr(frames.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="authentication"):
        frames.compare(identical_packets, digests)


@pytest.mark.parametrize("bad_hash", ["0" * 63, "G" * 64, 12, None])
def test_external_hash_must_be_literal_lowercase_sha256(bad_hash, identical_packets):
    digests = authenticated(identical_packets)
    digests["original0"] = bad_hash
    with pytest.raises(ValueError, match="SHA-256"):
        frames.compare(identical_packets, digests)


@pytest.mark.parametrize("container", ["packets", "hashes"])
def test_case_maps_require_exact_keys(container, identical_packets):
    digests = authenticated(identical_packets)
    maps = {"packets": dict(identical_packets), "hashes": dict(digests)}
    del maps[container]["original0"]
    with pytest.raises(ValueError, match="exact .* cases"):
        frames.compare(maps["packets"], maps["hashes"])
    maps = {"packets": dict(identical_packets), "hashes": dict(digests)}
    maps[container]["extra"] = b"" if container == "packets" else "0" * 64
    with pytest.raises(ValueError, match="exact .* cases"):
        frames.compare(maps["packets"], maps["hashes"])


def test_packet_must_be_immutable_bytes_of_exact_length(identical_packets):
    values = dict(identical_packets)
    values["serial1"] = bytearray(values["serial1"])
    with pytest.raises(ValueError, match="immutable bytes"):
        frames.compare(
            values, authenticated({case: bytes(raw) for case, raw in values.items()})
        )
    values["serial1"] = b"short"
    digests = authenticated(
        {case: raw for case, raw in values.items() if type(raw) is bytes}
    )
    digests["serial1"] = sha256(values["serial1"]).hexdigest()
    with pytest.raises(ValueError, match="exactly two frames"):
        frames.compare(values, digests)


def test_nonfinite_value_is_rejected_after_valid_whole_packet_authentication():
    nonfinite = packet(((0, "qfrc_bias", (0, 0), np.inf),))
    values = {case: packet() for case in frames.CASES}
    values["serial1"] = nonfinite
    with pytest.raises(ValueError, match="must be finite"):
        frames.compare(values, authenticated(values))


def test_module_has_no_torch_import_or_partial_comparison_surface():
    import sys

    assert "torch" not in frames.__dict__
    result = frames.compare(
        {case: packet() for case in frames.CASES},
        authenticated({case: packet() for case in frames.CASES}),
    )
    assert len(result["comparisons"]) == 12
    assert all(len(row["fields"]) == 17 for row in result["comparisons"])
    assert not any(name.startswith("subset") for name in result)
    assert sys.modules.get("torch") is None or "torch" not in frames.__dict__
