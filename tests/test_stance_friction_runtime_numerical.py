"""Mutation tests for the non-admitting raw-carrier numerical audit."""

import copy
import json
import struct

import pytest

from mjlab_microduck.stance_friction_runtime_numerical import audit_entries


WORLDS = 2
CAPACITY = 8
FORWARDS = 3
BANK_LAYOUT = {
    "nf": ("<i4", (WORLDS,)),
    "nefc": ("<i4", (WORLDS,)),
    "efc_nnz": ("<i4", (WORLDS,)),
    "type": ("<i4", (WORLDS, CAPACITY)),
    "id": ("<i4", (WORLDS, CAPACITY)),
    "row_nnz": ("<i4", (WORLDS, 0)),
    "row_adr": ("<i4", (WORLDS, 0)),
    "col_ind": ("<i4", (WORLDS, 0, 0)),
    "J": ("<f4", (WORLDS, CAPACITY, 20)),
    **{
        name: ("<f4", (WORLDS, CAPACITY))
        for name in ("pos", "margin", "D", "vel", "aref", "frictionloss", "force")
    },
}


def _pack(value, dtype):
    if dtype == "<i4":
        return struct.pack("<i", value)
    return struct.pack("<f", value)


def _carrier(raw, shape, dtype):
    return {"shape": list(shape), "dtype": dtype, "raw": raw}


def _input_carriers():
    values = {
        "frictionloss": ((WORLDS, 20), [0.0] * (WORLDS * 20)),
        "qvel": ((WORLDS, 20), [0.25] * (WORLDS * 20)),
        "invweight": ((1, 20), [1.0] * 20),
        "solref": ((1, 20, 2), [1.0] * 40),
        "solimp": ((1, 20, 5), [1.0] * 100),
        "timestep": ((1,), [0.002]),
    }
    values["frictionloss"][1][2] = 0.75
    values["frictionloss"][1][4] = 1.25
    values["frictionloss"][1][22] = 0.75
    values["frictionloss"][1][24] = 1.25
    return {
        name: _carrier(b"".join(_pack(item, "<f4") for item in data), shape, "<f4")
        for name, (shape, data) in values.items()
    }


def _bank():
    bank = {}
    for name, (dtype, shape) in BANK_LAYOUT.items():
        count = 1
        for dimension in shape:
            count *= dimension
        raw = b"".join(_pack(0, dtype) for _ in range(count))
        bank[name] = _carrier(raw, shape, dtype)
    for world in range(WORLDS):
        _set_scalar(bank["nf"], world, 1, "<i4")
        _set_scalar(bank["nefc"], world, 1, "<i4")
        _set_scalar(bank["type"], world * CAPACITY, 99, "<i4")
        _set_scalar(bank["id"], world * CAPACITY, 77, "<i4")
        for name in ("J", "pos", "margin", "D", "vel", "aref", "frictionloss"):
            _set_scalar(
                bank[name], world * CAPACITY * (20 if name == "J" else 1), 9.0, "<f4"
            )
        _set_scalar(bank["force"], world * CAPACITY, 3.0, "<f4")
    return bank


def _set_scalar(carrier, index, value, dtype):
    offset = index * 4
    raw = bytearray(carrier["raw"])
    raw[offset : offset + 4] = _pack(value, dtype)
    carrier["raw"] = bytes(raw)


def _set_i32(carrier, index, value):
    _set_scalar(carrier, index, value, "<i4")


def _write_row(bank, world, row, dof, inputs):
    _set_scalar(bank["type"], world * CAPACITY + row, 1, "<i4")
    _set_scalar(bank["id"], world * CAPACITY + row, dof, "<i4")
    for column in range(20):
        _set_scalar(
            bank["J"],
            (world * CAPACITY + row) * 20 + column,
            float(column == dof),
            "<f4",
        )
    _set_scalar(bank["D"], world * CAPACITY + row, 1.0, "<f4")
    qvel = inputs["qvel"]["raw"]
    _set_scalar(
        bank["vel"],
        world * CAPACITY + row,
        struct.unpack_from("<f", qvel, (world * 20 + dof) * 4)[0],
        "<f4",
    )
    loss = inputs["frictionloss"]["raw"]
    _set_scalar(
        bank["frictionloss"],
        world * CAPACITY + row,
        struct.unpack_from("<f", loss, dof * 4)[0],
        "<f4",
    )


def _entry():
    inputs = _input_carriers()
    before = _bank()
    after = copy.deepcopy(before)
    active = (2, 4)
    for world in range(WORLDS):
        for offset, dof in enumerate(active, start=1):
            _write_row(after, world, offset, dof, inputs)
        _set_i32(after["nf"], world, 3)
        _set_i32(after["nefc"], world, 3)
    return {
        "inputs": inputs,
        "inputs_after": copy.deepcopy(inputs),
        "before": before,
        "after": after,
        "scalars": {
            "nv": 20,
            "disableflags": 0,
            "is_sparse": False,
            "njmax": CAPACITY,
            "njmax_nnz": CAPACITY * 20,
        },
        "dim": [WORLDS, 20],
    }


def _entries():
    base = _entry()
    return {
        arm: [copy.deepcopy(base) for _ in range(FORWARDS)]
        for arm in ("original", "candidate0", "candidate1")
    }


def _audit(entries):
    return audit_entries(entries, worlds=WORLDS, capacity=CAPACITY, forwards=FORWARDS)


def test_exact_small_capture_is_json_reportable_and_never_admits_runtime():
    result = _audit(_entries())
    assert result["decision"] == "dense-friction-entries-exact-no-admission"
    assert result["component_exact_without_overflow"] is True
    assert result["candidate_replay"] == {"exact": True, "first_delta": None}
    assert result["original_candidate_comparison"]["matched_entries"] == FORWARDS
    assert (
        result["original_candidate_comparison"]["unmatched_later_trajectory_entries"]
        == 0
    )
    assert all(value is False for value in result["flags"].values())
    assert json.loads(json.dumps(result))["qualification"] is False
    assert len(json.dumps(result)) < 100_000


@pytest.mark.parametrize("carrier", ["nf", "nefc"])
def test_counter_delta_is_reported_and_rejected(carrier):
    entries = _entries()
    _set_i32(entries["candidate0"][0]["after"][carrier], 0, 2)
    result = _audit(entries)
    assert result["arms"]["candidate0"]["calls"][0]["worlds"][0]["first_delta"] == {
        "world": 0,
        "field": "counters",
    }


@pytest.mark.parametrize(
    "field,index", [("force", 0), ("efc_nnz", 0), ("type", 0), ("type", 7)]
)
def test_force_prefix_and_suffix_mutations_are_detected(field, index):
    entries = _entries()
    data = entries["candidate0"][0]["after"][field]
    value = 8 if data["dtype"] == "<i4" else 2.0
    _set_scalar(data, index, value, data["dtype"])
    result = _audit(entries)
    assert (
        result["arms"]["candidate0"]["calls"][0]["worlds"][0]["structurally_exact"]
        is False
    )


def test_changed_inputs_break_replay_and_start_unmatched_later_trajectory():
    entries = _entries()
    changed = entries["candidate0"][1]["inputs"]["qvel"]
    _set_scalar(changed, 0, 0.5, "<f4")
    entries["candidate0"][1]["inputs_after"]["qvel"]["raw"] = changed["raw"]
    result = _audit(entries)
    comparison = result["original_candidate_comparison"]
    assert comparison["matched_entries"] == 1
    assert comparison["unmatched_entries"] == 2
    assert comparison["unmatched_later_trajectory_entries"] == 1
    assert comparison["entries"][2]["unmatched_reason"] == "later-unmatched-trajectory"
    assert result["candidate_replay"]["exact"] is False


def test_candidate_rows_must_be_ascending():
    entries = _entries()
    for field in (
        "type",
        "id",
        "J",
        "pos",
        "margin",
        "D",
        "vel",
        "aref",
        "frictionloss",
    ):
        carrier = entries["candidate0"][0]["after"][field]
        width = 20 if field == "J" else 1
        raw = bytearray(carrier["raw"])
        first = (0 * CAPACITY + 1) * width * 4
        second = (0 * CAPACITY + 2) * width * 4
        size = width * 4
        raw[first : first + size], raw[second : second + size] = (
            raw[second : second + size],
            raw[first : first + size],
        )
        carrier["raw"] = bytes(raw)
    result = _audit(entries)
    world = result["arms"]["candidate0"]["calls"][0]["worlds"][0]
    assert world["structurally_exact"] is False
    assert world["first_delta"]["field"] in {"addresses", "appended-row", "ascending"}


def test_overflow_is_negative_but_counters_still_include_all_proposals():
    entries = _entries()
    for arm in entries:
        for call in entries[arm]:
            for world in range(WORLDS):
                _set_i32(call["before"]["nefc"], world, CAPACITY - 1)
                _set_i32(call["before"]["nf"], world, CAPACITY - 1)
                _set_i32(call["after"]["nefc"], world, CAPACITY + 1)
                _set_i32(call["after"]["nf"], world, CAPACITY + 1)
                # One row fits, the second proposal still increments both counters.
                _write_row(call["after"], world, CAPACITY - 1, 2, call["inputs"])
    result = _audit(entries)
    assert result["overflow_negative"] is True
    assert result["component_exact_without_overflow"] is False
    world = result["arms"]["candidate0"]["calls"][0]["worlds"][0]
    assert world["expected_counts"] == {"nf": CAPACITY + 1, "nefc": CAPACITY + 1}
    assert world["actual_counts"] == world["expected_counts"]


def test_truncated_carriers_and_bool_integers_are_rejected():
    entries = _entries()
    entries["candidate1"][0]["after"]["J"]["raw"] = b"short"
    with pytest.raises(ValueError, match="raw carrier byte length"):
        _audit(entries)
    entries = _entries()
    entries["candidate1"][0]["scalars"]["nv"] = True
    with pytest.raises(ValueError, match="literal nv"):
        _audit(entries)


def test_input_carrier_shape_metadata_is_part_of_exact_candidate_replay():
    entries = _entries()
    for key in ("inputs", "inputs_after"):
        carrier = entries["candidate0"][0][key]["invweight"]
        carrier["shape"] = [WORLDS, 20]
        carrier["raw"] = carrier["raw"] * WORLDS
    result = _audit(entries)
    assert result["candidate_replay"]["exact"] is False
    assert result["candidate_replay"]["first_delta"]["field"] == "inputs"
