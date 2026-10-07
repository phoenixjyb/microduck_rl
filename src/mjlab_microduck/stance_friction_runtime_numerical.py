"""Pure numerical checks for captured dense DOF-friction kernel entries.

This module consumes caller-authenticated raw carriers only. It does not prove
where they came from and never qualifies a runtime, training run, or hardware.
"""

import math
import struct


PROTOCOL = "microduck-friction-runtime-numerical-oct8-v1"
ARMS = ("original", "candidate0", "candidate1")
INPUT_NAMES = ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep")
BANK_NAMES = (
    "nf",
    "nefc",
    "efc_nnz",
    "type",
    "id",
    "row_nnz",
    "row_adr",
    "col_ind",
    "J",
    "pos",
    "margin",
    "D",
    "vel",
    "aref",
    "frictionloss",
    "force",
)
ROW_FIELDS = ("type", "id", "J", "pos", "margin", "D", "vel", "aref", "frictionloss")
FLOAT_ROW_FIELDS = ("J", "pos", "margin", "D", "vel", "aref", "frictionloss")
INPUT_RECORD_FIELDS = frozenset({"shape", "dtype", "raw"})
ENTRY_FIELDS = frozenset(
    {"inputs", "inputs_after", "before", "after", "scalars", "dim"}
)
SCALAR_FIELDS = frozenset({"nv", "disableflags", "is_sparse", "njmax", "njmax_nnz"})
I32_MAX = (1 << 31) - 1
ZERO_F32 = b"\x00\x00\x00\x00"
ONE_F32 = struct.pack("<f", 1.0)
TOP_FLAGS = (
    "runtime_cause_proven",
    "native_qualified",
    "full_window_qualified",
    "training_authorized",
    "physical_acceptance",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _plain_int(value, label, minimum, maximum):
    _need(type(value) is int and minimum <= value <= maximum, label)
    return value


def _product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _carrier(value, shape, dtype, label):
    _need(
        type(value) is dict and set(value) == INPUT_RECORD_FIELDS,
        "exact raw carrier schema " + label,
    )
    actual_shape = value["shape"]
    _need(type(actual_shape) is list, "plain carrier shape " + label)
    for dimension in actual_shape:
        _plain_int(dimension, "plain carrier dimension " + label, 0, 1 << 20)
    _need(actual_shape == list(shape), "carrier shape " + label)
    _need(
        type(value["dtype"]) is str and value["dtype"] == dtype,
        "carrier dtype " + label,
    )
    raw = value["raw"]
    _need(type(raw) is bytes, "raw bytes carrier " + label)
    _need(len(raw) == _product(shape) * 4, "raw carrier byte length " + label)
    return raw


def _finite_f32(raw, label):
    for (value,) in struct.iter_unpack("<f", raw):
        _need(math.isfinite(value), "finite float32 input " + label)


def _input_shapes(worlds):
    return {
        "frictionloss": (worlds, 20),
        "qvel": (worlds, 20),
        "invweight": (None, 20),
        "solref": (None, 20, 2),
        "solimp": (None, 20, 5),
        "timestep": (None,),
    }


def _inputs(value, worlds, label):
    _need(
        type(value) is dict and set(value) == set(INPUT_NAMES),
        "exact input fields " + label,
    )
    result = {}
    for name in INPUT_NAMES:
        shape = _input_shapes(worlds)[name]
        if shape[0] is None:
            actual = value[name].get("shape") if type(value[name]) is dict else None
            _need(
                type(actual) is list
                and len(actual) == len(shape)
                and type(actual[0]) is int
                and actual[0] in (1, worlds),
                "broadcast row count " + label + "." + name,
            )
            shape = (actual[0],) + shape[1:]
        raw = _carrier(value[name], shape, "<f4", label + "." + name)
        _finite_f32(raw, label + "." + name)
        result[name] = raw
    return result


def _input_metadata(value):
    return {
        name: {"shape": tuple(value[name]["shape"]), "dtype": value[name]["dtype"]}
        for name in INPUT_NAMES
    }


def _bank_shapes(worlds, capacity):
    return {
        "nf": ("<i4", (worlds,)),
        "nefc": ("<i4", (worlds,)),
        "efc_nnz": ("<i4", (worlds,)),
        "type": ("<i4", (worlds, capacity)),
        "id": ("<i4", (worlds, capacity)),
        "row_nnz": ("<i4", (worlds, 0)),
        "row_adr": ("<i4", (worlds, 0)),
        "col_ind": ("<i4", (worlds, 0, 0)),
        "J": ("<f4", (worlds, capacity, 20)),
        "pos": ("<f4", (worlds, capacity)),
        "margin": ("<f4", (worlds, capacity)),
        "D": ("<f4", (worlds, capacity)),
        "vel": ("<f4", (worlds, capacity)),
        "aref": ("<f4", (worlds, capacity)),
        "frictionloss": ("<f4", (worlds, capacity)),
        "force": ("<f4", (worlds, capacity)),
    }


def _bank(value, worlds, capacity, label):
    _need(
        type(value) is dict and set(value) == set(BANK_NAMES),
        "exact bank fields " + label,
    )
    result = {}
    for name, (dtype, shape) in _bank_shapes(worlds, capacity).items():
        result[name] = _carrier(value[name], shape, dtype, label + "." + name)
    return result


def _validate_scalars(value, capacity, label):
    _need(
        type(value) is dict and set(value) == SCALAR_FIELDS,
        "exact scalar fields " + label,
    )
    _need(type(value["nv"]) is int and value["nv"] == 20, "literal nv " + label)
    _need(
        type(value["disableflags"]) is int and value["disableflags"] == 0,
        "literal disableflags " + label,
    )
    _need(value["is_sparse"] is False, "dense invocation only " + label)
    _need(
        type(value["njmax"]) is int and value["njmax"] == capacity,
        "literal njmax " + label,
    )
    _need(
        type(value["njmax_nnz"]) is int and value["njmax_nnz"] == capacity * 20,
        "literal dense njmax_nnz " + label,
    )


def _entry(value, worlds, capacity, index, arm):
    label = arm + ".forward" + str(index)
    _need(
        type(value) is dict and set(value) == ENTRY_FIELDS,
        "exact entry fields " + label,
    )
    before_inputs = _inputs(value["inputs"], worlds, label + ".inputs")
    after_inputs = _inputs(value["inputs_after"], worlds, label + ".inputs_after")
    _need(before_inputs == after_inputs, "input mutation " + label)
    before_input_metadata = _input_metadata(value["inputs"])
    after_input_metadata = _input_metadata(value["inputs_after"])
    _need(
        before_input_metadata == after_input_metadata,
        "input shape or dtype mutation " + label,
    )
    before = _bank(value["before"], worlds, capacity, label + ".before")
    after = _bank(value["after"], worlds, capacity, label + ".after")
    _validate_scalars(value["scalars"], capacity, label)
    dim = value["dim"]
    _need(
        type(dim) is list
        and len(dim) == 2
        and all(type(item) is int for item in dim)
        and dim == [worlds, 20],
        "exact two-dimensional invocation " + label,
    )
    counters = []
    for world in range(worlds):
        before_nf = _i32(before["nf"], world, label + ".before.nf")
        before_nefc = _i32(before["nefc"], world, label + ".before.nefc")
        after_nf = _i32(after["nf"], world, label + ".after.nf")
        after_nefc = _i32(after["nefc"], world, label + ".after.nefc")
        _need(
            min(before_nf, before_nefc, after_nf, after_nefc) >= 0,
            "nonnegative counters " + label,
        )
        counters.append((before_nf, before_nefc, after_nf, after_nefc))
    return {
        "inputs": before_inputs,
        "inputs_after": after_inputs,
        "input_metadata": before_input_metadata,
        "inputs_after_metadata": after_input_metadata,
        "before": before,
        "after": after,
        "scalars": value["scalars"],
        "dim": dim,
        "counters": counters,
    }


def _i32(raw, index, label):
    value = struct.unpack_from("<i", raw, index * 4)[0]
    _need(value >= 0, "nonnegative int32 " + label)
    return value


def _row_bytes(bank, field, world, row, capacity):
    width = 20 if field == "J" else 1
    start = (world * capacity + row) * width * 4
    return bank[field][start : start + width * 4]


def _array_bytes(bank, field, world, capacity):
    width = 20 if field == "J" else 1
    start = world * capacity * width * 4
    end = start + capacity * width * 4
    return bank[field][start:end]


def _input_scalar(raw, row, dof, width):
    return raw[
        (row * 20 * width + dof * width) * 4 : (row * 20 * width + (dof + 1) * width)
        * 4
    ]


def _active_signature(bank, world, row, capacity):
    dof = struct.unpack_from("<i", bank["id"], (world * capacity + row) * 4)[0]
    return {
        "type": _row_bytes(bank, "type", world, row, capacity),
        **{
            field: _row_bytes(bank, field, world, row, capacity)
            for field in FLOAT_ROW_FIELDS
        },
    }, dof


def _first_row_delta(left, right):
    for key in ("type",) + FLOAT_ROW_FIELDS:
        if left.get(key) != right.get(key):
            return key
    return None


def _world_result(entry, worlds, capacity, arm, forward):
    inputs, before, after = entry["inputs"], entry["before"], entry["after"]
    world_results = []
    all_structural = True
    any_overflow = False
    for world in range(worlds):
        loss_row = world % (len(inputs["frictionloss"]) // (20 * 4))
        active_dofs = [
            dof
            for dof in range(20)
            if struct.unpack_from(
                "<f", inputs["frictionloss"], (loss_row * 20 + dof) * 4
            )[0]
            > 0.0
        ]
        before_nf, before_nefc, after_nf, after_nefc = entry["counters"][world]
        expected_nf = before_nf + len(active_dofs)
        expected_nefc = before_nefc + len(active_dofs)
        _need(
            expected_nf <= I32_MAX and expected_nefc <= I32_MAX,
            "friction counter arithmetic remains int32",
        )
        counts_match = after_nf == expected_nf and after_nefc == expected_nefc
        start = min(before_nefc, capacity)
        visible = min(len(active_dofs), max(capacity - before_nefc, 0))
        end = start + visible
        overflow = before_nefc + len(active_dofs) > capacity
        any_overflow |= overflow

        prefix_preserved = all(
            _array_bytes(before, field, world, capacity)[
                : start * (20 if field == "J" else 1) * 4
            ]
            == _array_bytes(after, field, world, capacity)[
                : start * (20 if field == "J" else 1) * 4
            ]
            for field in ROW_FIELDS
        )
        suffix_preserved = all(
            _array_bytes(before, field, world, capacity)[
                end * (20 if field == "J" else 1) * 4 :
            ]
            == _array_bytes(after, field, world, capacity)[
                end * (20 if field == "J" else 1) * 4 :
            ]
            for field in ROW_FIELDS
        )
        scratch_unchanged = all(
            before[field] == after[field]
            for field in ("row_nnz", "row_adr", "col_ind", "efc_nnz")
        )
        force_unchanged = before["force"] == after["force"]

        addresses = []
        active_valid = True
        first_row_delta = None
        for offset in range(visible):
            row = before_nefc + offset
            signature, dof = _active_signature(after, world, row, capacity)
            addresses.append(dof)
            valid = dof in active_dofs and dof not in addresses[:-1]
            valid &= signature["type"] == struct.pack("<i", 1)
            jacobian = signature["J"]
            expected_j = bytearray(20 * 4)
            if 0 <= dof < 20:
                expected_j[dof * 4 : (dof + 1) * 4] = ONE_F32
            valid &= jacobian == bytes(expected_j)
            valid &= signature["pos"] == ZERO_F32 and signature["margin"] == ZERO_F32
            valid &= (
                signature["vel"] == _input_scalar(inputs["qvel"], world, dof, 1)
                if 0 <= dof < 20
                else False
            )
            flrow = world % (len(inputs["frictionloss"]) // 80)
            valid &= (
                signature["frictionloss"]
                == _input_scalar(inputs["frictionloss"], flrow, dof, 1)
                if 0 <= dof < 20
                else False
            )
            finite = all(
                math.isfinite(value)
                for field in FLOAT_ROW_FIELDS
                for (value,) in struct.iter_unpack("<f", signature[field])
            )
            valid &= finite
            active_valid &= valid
            if not valid and first_row_delta is None:
                first_row_delta = {"row": row, "dof": dof, "field": "appended-row"}
        ascending = addresses == sorted(addresses)
        addresses_complete = (
            len(addresses) == visible and len(set(addresses)) == visible
        )
        if arm == "original":
            if not overflow:
                addresses_complete &= set(addresses) == set(active_dofs)
        else:
            addresses_complete &= addresses == active_dofs[:visible]
        structurally_exact = all(
            (
                counts_match,
                prefix_preserved,
                suffix_preserved,
                scratch_unchanged,
                force_unchanged,
                active_valid,
                addresses_complete,
                arm == "original" or ascending,
            )
        )
        all_structural &= structurally_exact
        first_delta = None
        for name, passed in (
            ("counters", counts_match),
            ("prefix", prefix_preserved),
            ("suffix", suffix_preserved),
            ("scratch", scratch_unchanged),
            ("force", force_unchanged),
            ("appended-row", active_valid),
            ("addresses", addresses_complete),
            ("ascending", arm == "original" or ascending),
        ):
            if not passed:
                first_delta = {"world": world, "field": name}
                if name == "appended-row":
                    first_delta.update(first_row_delta or {})
                break
        world_results.append(
            {
                "world": world,
                "active_dof_count": len(active_dofs),
                "expected_counts": {"nf": expected_nf, "nefc": expected_nefc},
                "actual_counts": {"nf": after_nf, "nefc": after_nefc},
                "stored_rows": visible,
                "overflow": overflow,
                "first_delta": first_delta,
                "structurally_exact": bool(structurally_exact),
                "decision": "exact"
                if structurally_exact and not overflow
                else "negative",
            }
        )
    return world_results, bool(all_structural), bool(any_overflow)


def _bank_delta(left, right):
    for name in BANK_NAMES:
        if left[name] != right[name]:
            return name
    return None


def _entry_match(original, candidate, worlds, capacity):
    if (
        original["inputs"] != candidate["inputs"]
        or original["inputs_after"] != candidate["inputs_after"]
        or original["input_metadata"] != candidate["input_metadata"]
        or original["inputs_after_metadata"] != candidate["inputs_after_metadata"]
        or original["scalars"] != candidate["scalars"]
        or original["dim"] != candidate["dim"]
    ):
        return False, "input-or-invocation"
    for world in range(worlds):
        if original["counters"][world] != candidate["counters"][world]:
            return False, "counters"
        stop = min(original["counters"][world][1], capacity)
        for field in ROW_FIELDS:
            if (
                _array_bytes(original["before"], field, world, capacity)[
                    : stop * (20 if field == "J" else 1) * 4
                ]
                != _array_bytes(candidate["before"], field, world, capacity)[
                    : stop * (20 if field == "J" else 1) * 4
                ]
            ):
                return False, "opaque-prefix"
    return True, None


def _addressed_delta(original, candidate, worlds, capacity):
    for world in range(worlds):
        left_start = min(original["counters"][world][1], capacity)
        right_start = min(candidate["counters"][world][1], capacity)
        left_end = min(original["counters"][world][3], capacity)
        right_end = min(candidate["counters"][world][3], capacity)
        left_rows = {}
        right_rows = {}
        for row in range(left_start, left_end):
            signature, dof = _active_signature(original["after"], world, row, capacity)
            left_rows[dof] = signature
        for row in range(right_start, right_end):
            signature, dof = _active_signature(candidate["after"], world, row, capacity)
            right_rows[dof] = signature
        if set(left_rows) != set(right_rows):
            return {"world": world, "field": "addresses"}
        for dof in sorted(left_rows):
            field = _first_row_delta(left_rows[dof], right_rows[dof])
            if field is not None:
                return {"world": world, "dof": dof, "field": field}
    return None


def audit_entries(entries, *, worlds=64, capacity=512, forwards=21):
    """Audit captured dense-friction entries without asserting provenance.

    ``entries`` must contain three literal arms, each with exactly ``forwards``
    calls. All runtime identity, source, device, stream, and capture claims
    remain the caller's responsibility.
    """
    _plain_int(worlds, "world bound", 1, 512)
    _plain_int(capacity, "capacity bound", 1, 512)
    _plain_int(forwards, "forward-count bound", 1, 21)
    _need(
        type(entries) is dict and set(entries) == set(ARMS), "exact three runtime arms"
    )
    validated = {}
    reports = {}
    for arm in ARMS:
        calls = entries[arm]
        _need(
            type(calls) is list and len(calls) == forwards,
            "exact arm forward count " + arm,
        )
        validated[arm] = [
            _entry(call, worlds, capacity, index, arm)
            for index, call in enumerate(calls)
        ]
        arm_calls = []
        arm_structural = True
        arm_overflow = False
        for index, call in enumerate(validated[arm]):
            per_world, structural, overflow = _world_result(
                call, worlds, capacity, arm, index
            )
            arm_structural &= structural
            arm_overflow |= overflow
            first_world_delta = next(
                (world["first_delta"] for world in per_world if world["first_delta"]),
                None,
            )
            arm_calls.append(
                {
                    "forward": index,
                    "worlds": per_world,
                    "structurally_exact": structural,
                    "overflow": overflow,
                    "first_delta": first_world_delta,
                    "component_exact_without_overflow": bool(
                        structural and not overflow
                    ),
                    "decision": "exact" if structural and not overflow else "negative",
                }
            )
        reports[arm] = {
            "entry_count": len(arm_calls),
            "calls": arm_calls,
            "structurally_exact": bool(arm_structural),
            "overflow": bool(arm_overflow),
            "first_delta": next(
                (
                    call["first_delta"]
                    for call in arm_calls
                    if call["first_delta"] is not None
                ),
                None,
            ),
            "component_exact_without_overflow": bool(
                arm_structural and not arm_overflow
            ),
            "decision": "exact" if arm_structural and not arm_overflow else "negative",
        }

    replay_delta = None
    original_matches = []
    first_matched_delta = None
    first_unmatched_forward = None
    for index, (left, right, baseline) in enumerate(
        zip(
            validated["candidate0"],
            validated["candidate1"],
            validated["original"],
            strict=True,
        )
    ):
        if replay_delta is None:
            for field in (
                "inputs",
                "inputs_after",
                "input_metadata",
                "inputs_after_metadata",
                "scalars",
                "dim",
            ):
                if left[field] != right[field]:
                    replay_delta = {"forward": index, "field": field}
                    break
            if replay_delta is None:
                field = _bank_delta(left["before"], right["before"])
                if field is not None:
                    replay_delta = {"forward": index, "field": "before." + field}
            if replay_delta is None:
                field = _bank_delta(left["after"], right["after"])
                if field is not None:
                    replay_delta = {"forward": index, "field": "after." + field}
        if first_unmatched_forward is None:
            matched, reason = _entry_match(baseline, left, worlds, capacity)
        else:
            matched, reason = False, "later-unmatched-trajectory"
        if not matched and first_unmatched_forward is None:
            first_unmatched_forward = index
        row_delta = (
            _addressed_delta(baseline, left, worlds, capacity) if matched else None
        )
        if matched and row_delta is not None and first_matched_delta is None:
            first_matched_delta = {"forward": index, **row_delta}
        original_matches.append(
            {
                "forward": index,
                "matched": matched,
                "unmatched_reason": reason,
                "addressed_exact": bool(matched and row_delta is None),
                "delta": row_delta if matched else None,
            }
        )
    replay_exact = replay_delta is None
    all_matched = all(row["matched"] for row in original_matches)
    original_exact = all(row["addressed_exact"] for row in original_matches)
    any_overflow = any(report["overflow"] for report in reports.values())
    exact_without_overflow = (
        all(report["structurally_exact"] for report in reports.values())
        and replay_exact
        and all_matched
        and original_exact
        and not any_overflow
    )
    return {
        "protocol": PROTOCOL,
        "counts": {
            "arms": 3,
            "forwards_per_arm": forwards,
            "worlds": worlds,
            "capacity": capacity,
        },
        "arms": reports,
        "candidate_replay": {"exact": replay_exact, "first_delta": replay_delta},
        "original_candidate_comparison": {
            "matched_entries": sum(row["matched"] for row in original_matches),
            "unmatched_entries": sum(not row["matched"] for row in original_matches),
            "unmatched_later_trajectory_entries": sum(
                row["unmatched_reason"] == "later-unmatched-trajectory"
                for row in original_matches
            ),
            "first_unmatched": next(
                (row for row in original_matches if not row["matched"]), None
            ),
            "first_matched_delta": first_matched_delta,
            "entries": original_matches,
        },
        "overflow_negative": bool(any_overflow),
        "component_exact_without_overflow": bool(exact_without_overflow),
        "decision": "dense-friction-entries-exact-no-admission"
        if exact_without_overflow
        else "dense-friction-entries-negative-or-unmatched",
        "qualification": False,
        "flags": {name: False for name in TOP_FLAGS},
        **{name: False for name in TOP_FLAGS},
    }
