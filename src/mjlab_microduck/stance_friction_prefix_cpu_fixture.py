"""CPU-only dense-friction prefix preservation component fixture.

This composes the pinned original and ascending-DOF kernels from
``stance_friction_row_cpu_fixture``. It is a standalone component check, not a
runtime controller or simulator path.
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import warp as wp

from mjlab_microduck import stance_friction_row_cpu_fixture as base

PROTOCOL = "microduck-dense-friction-prefix-cpu-fixture-oct7-v1"
BASE_MODULE_SHA256 = "8e96c5077f248aad06cacab4c3a45f17cd6b3c35c09c7a46457281916e8046d9"
OUTPUT_CAP = 4 * 1024 * 1024
_F32 = np.dtype("<f4")
_I32 = np.dtype("<i4")

_BANK_DTYPES = {
    "nf": "i32",
    "nefc": "i32",
    "type": "i32",
    "id": "i32",
    "row_nnz": "i32",
    "row_adr": "i32",
    "col_ind": "i32",
    "J": "u32",
    "pos": "u32",
    "margin": "u32",
    "D": "u32",
    "vel": "u32",
    "aref": "u32",
    "frictionloss": "u32",
    "efc_nnz": "i32",
}
_ROW_FIELDS = ("type", "id", "J", "pos", "margin", "D", "vel", "aref", "frictionloss")
_FLOAT_FIELDS = ("J", "pos", "margin", "D", "vel", "aref", "frictionloss")
_BASE_HELPERS = {
    name: (getattr(base, name), getattr(base, name).__code__)
    for name in (
        "_source_binding",
        "_validate_fixture",
        "_copies",
        "_wp_inputs",
        "_float_bits",
    )
}
_INPUT_NAMES = ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _module_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _base_source_binding():
    _check_base_helpers()
    _need(
        sha256(Path(base.__file__).read_bytes()).hexdigest() == BASE_MODULE_SHA256,
        "exact pinned dense-row fixture module",
    )
    return _BASE_HELPERS["_source_binding"][0]()


def _check_base_helpers():
    for name, (function, code) in _BASE_HELPERS.items():
        current = getattr(base, name)
        _need(
            current is function and current.__code__ is code,
            "pinned borrowed dense-row helper " + name,
        )


def _validate_prefix(initial_nefc, nworld, njmax):
    _need(
        type(initial_nefc) is np.ndarray
        and initial_nefc.dtype == _I32
        and initial_nefc.shape == (nworld,)
        and initial_nefc.flags.c_contiguous,
        "initial_nefc literal contiguous int32 per-world vector",
    )
    _need(
        bool(((initial_nefc >= 0) & (initial_nefc <= njmax)).all()),
        "initial_nefc bounded by row capacity",
    )


def _as_bits(array, kind):
    if kind == "u32":
        return np.asarray(array, dtype=_F32).view("<u4").reshape(-1).tolist()
    return np.asarray(array, dtype=_I32).view("<i4").reshape(-1).tolist()


def _bank_record(values):
    result = {}
    for name, value in values.items():
        kind = _BANK_DTYPES[name]
        result[name] = {
            "shape": list(value.shape),
            "dtype": "<u4" if kind == "u32" else "<i4",
            "bits": _as_bits(value, kind),
        }
    return result


def _allocation_record(arrays):
    result = {}
    for name, value in arrays.items():
        _need(
            str(value.device) == "cpu" and value.is_contiguous,
            "owned contiguous CPU allocation",
        )
        result[name] = {
            "object_id": id(value),
            "pointer": int(value.ptr),
            "device": str(value.device),
            "warp_dtype": str(value.dtype),
            "warp_shape": list(value.shape),
            "warp_strides": list(value.strides),
            "bytes": int(value.numpy().nbytes),
        }
    return result


def _input_record(arrays, expected_values):
    result = {}
    for name, value, expected in zip(
        _INPUT_NAMES, arrays, expected_values, strict=True
    ):
        actual = value.numpy()
        _need(
            str(value.device) == "cpu"
            and value.is_contiguous
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and np.array_equal(actual.view("<u4"), expected.view("<u4")),
            "actual Warp input equals owned NumPy snapshot: " + name,
        )
        result[name] = {
            "shape": list(actual.shape),
            "dtype": "<f4",
            "device": str(value.device),
            "warp_dtype": str(value.dtype),
            "warp_shape": list(value.shape),
            "pointer": int(value.ptr),
            "u32": _BASE_HELPERS["_float_bits"][0](actual),
        }
    return result


def _prefix_values(nworld, njmax, initial_nefc):
    """Build deterministic prefix rows and bit-distinct poisoned suffixes."""

    def ints(shape, fill):
        return np.full(shape, fill, dtype=_I32)

    def floats(shape, bits):
        return np.full(shape, bits, dtype="<u4").view(_F32)

    values = {
        "nf": np.zeros((nworld,), dtype=_I32),
        "nefc": np.array(initial_nefc, dtype=_I32, copy=True),
        "type": ints((nworld, njmax), -101),
        "id": ints((nworld, njmax), -202),
        "row_nnz": ints((nworld, njmax), -303),
        "row_adr": ints((nworld, njmax), -404),
        "col_ind": ints((nworld, 1, njmax * base.NV), -505),
        "J": floats((nworld, njmax, base.NV), 0x7FC01234),
        "pos": floats((nworld, njmax), 0x80000000),
        "margin": floats((nworld, njmax), 0x7FC05678),
        "D": floats((nworld, njmax), 0x7FC09ABC),
        "vel": floats((nworld, njmax), 0x80000000),
        "aref": floats((nworld, njmax), 0x7FC0DEF0),
        "frictionloss": floats((nworld, njmax), 0x7FC01111),
        "efc_nnz": np.zeros((nworld,), dtype=_I32),
    }
    for world, prefix_count in enumerate(initial_nefc.tolist()):
        values["efc_nnz"][world] = prefix_count * base.NV
        for row in range(prefix_count):
            values["type"][world, row] = 0
            values["id"][world, row] = 1000 + world * njmax + row
            values["row_nnz"][world, row] = base.NV
            values["row_adr"][world, row] = row * base.NV
            values["col_ind"][world, 0, row * base.NV : (row + 1) * base.NV] = (
                np.arange(base.NV, dtype=_I32)
            )
            values["J"][world, row] = np.arange(base.NV, dtype="<f4") + 0.25 + world
            values["pos"][world, row] = np.float32(10 + world + row / 16)
            values["margin"][world, row] = np.float32(20 + world + row / 16)
            values["D"][world, row] = np.float32(30 + world + row / 16)
            values["vel"][world, row] = np.float32(40 + world + row / 16)
            values["aref"][world, row] = np.float32(50 + world + row / 16)
            values["frictionloss"][world, row] = np.float32(60 + world + row / 16)
    return values


def _new_wp_bank(values):
    output = {}
    for name, value in values.items():
        dtype = wp.int32 if value.dtype.kind == "i" else wp.float32
        output[name] = wp.array(
            np.array(value, copy=True, order="C"), dtype=dtype, device="cpu"
        )
    return output


def _numpy_bank(wp_bank):
    return {name: value.numpy().copy() for name, value in wp_bank.items()}


def _launch_original(inputs, bank, nworld, njmax):
    frictionloss, qvel, invweight, solref, solimp, timestep = inputs
    wp.launch(
        base._ORIGINAL_KERNEL,
        dim=(nworld, base.NV),
        inputs=[
            base.NV,
            timestep,
            0,
            solref,
            solimp,
            frictionloss,
            invweight,
            False,
            qvel,
            njmax,
            njmax * base.NV,
        ],
        outputs=[
            bank["nf"],
            bank["nefc"],
            bank["type"],
            bank["id"],
            bank["row_nnz"],
            bank["row_adr"],
            bank["col_ind"],
            bank["J"],
            bank["pos"],
            bank["margin"],
            bank["D"],
            bank["vel"],
            bank["aref"],
            bank["frictionloss"],
            bank["efc_nnz"],
        ],
        device="cpu",
    )


def _launch_candidate(inputs, bank, nworld, njmax):
    frictionloss, qvel, invweight, solref, solimp, timestep = inputs
    wp.launch(
        base._CANDIDATE_KERNEL,
        dim=nworld,
        inputs=[
            base.NV,
            timestep,
            0,
            solref,
            solimp,
            frictionloss,
            invweight,
            qvel,
            njmax,
        ],
        outputs=[
            bank["nf"],
            bank["nefc"],
            bank["type"],
            bank["id"],
            bank["J"],
            bank["pos"],
            bank["margin"],
            bank["D"],
            bank["vel"],
            bank["aref"],
            bank["frictionloss"],
        ],
        device="cpu",
    )


def _wp_inputs(values):
    return base._wp_inputs(wp, *base._copies(values))


def _capture_active(arrays, nworld, njmax, prefix):
    rows = []
    counts = []
    for world in range(nworld):
        nf, nefc = int(arrays["nf"][world]), int(arrays["nefc"][world])
        _need(
            0 <= nf <= base.NV and nefc == int(prefix[world]) + nf,
            "bounded prefix/friction counters",
        )
        visible_end = min(nefc, njmax)
        active = []
        for row_index in range(prefix[world], visible_end):
            _need(
                int(arrays["type"][world, row_index]) == 1,
                "active friction row has literal FRICTION_DOF type",
            )
            dof = int(arrays["id"][world, row_index])
            _need(0 <= dof < base.NV, "active friction row DOF address")
            expected_J = np.zeros(base.NV, dtype="<u4")
            expected_J[dof] = 0x3F800000
            _need(
                np.array_equal(arrays["J"][world, row_index].view("<u4"), expected_J),
                "literal dense one-hot addressed Jacobian",
            )
            _need(
                all(
                    np.isfinite(arrays[name][world, row_index]).all()
                    for name in _FLOAT_FIELDS
                ),
                "finite complete active friction row",
            )
            active.append(
                {
                    "row": row_index,
                    "dof": dof,
                    **{
                        name: _as_bits(arrays[name][world, row_index], "u32")
                        for name in _FLOAT_FIELDS
                    },
                }
            )
        rows.append(active)
        counts.append({"nf": nf, "nefc": nefc, "stored_rows": visible_end})
    return arrays, counts, rows


def _addressed_rows(rows):
    return sorted(
        (
            row["dof"],
            tuple(
                (name, tuple(row[name]) if isinstance(row[name], list) else row[name])
                for name in _FLOAT_FIELDS
            ),
        )
        for row in rows
    )


def _unchanged(before, after, world, start, end, names):
    return all(
        before[name][world, start:end].tobytes()
        == after[name][world, start:end].tobytes()
        for name in names
    )


def run_prefix_cpu_fixture(
    frictionloss, qvel, invweight, solref, solimp, timestep, njmax, initial_nefc
):
    """Compare original and two independent candidate launches over a prefix."""
    entry, entry_code = _OWN_HELPERS["run_prefix_cpu_fixture"]
    binding_function, binding_code = _OWN_HELPERS["_own_source_binding"]
    need_function, need_code = _OWN_HELPERS["_need"]
    if not (
        _need is need_function
        and need_function.__code__ is need_code
        and run_prefix_cpu_fixture is entry
        and entry.__code__ is entry_code
        and _own_source_binding is binding_function
        and binding_function.__code__ is binding_code
    ):
        raise ValueError("pinned prefix orchestration and binding checker")
    own_before = binding_function()
    source_before = _base_source_binding()
    nworld, _ = _BASE_HELPERS["_validate_fixture"][0](
        frictionloss, qvel, invweight, solref, solimp, timestep, njmax
    )
    _validate_prefix(initial_nefc, nworld, njmax)
    module_before = _module_sha256()
    input_values = _BASE_HELPERS["_copies"][0](
        (frictionloss, qvel, invweight, solref, solimp, timestep)
    )
    prefix = initial_nefc.copy()
    _BASE_HELPERS["_validate_fixture"][0](*input_values, njmax)
    _validate_prefix(prefix, nworld, njmax)
    before_rows = _prefix_values(nworld, njmax, prefix)
    run_specs = (
        ("original", _launch_original),
        ("candidate0", _launch_candidate),
        ("candidate1", _launch_candidate),
    )
    run_data = {}
    all_before = None
    with wp.ScopedDevice("cpu"):
        for name, launch in run_specs:
            inputs = _wp_inputs(input_values)
            bank = _new_wp_bank(before_rows)
            before = _numpy_bank(bank)
            before_record = _bank_record(before)
            input_before = _input_record(inputs, input_values)
            allocations = {
                "inputs": _allocation_record(
                    dict(zip(_INPUT_NAMES, inputs, strict=True))
                ),
                "outputs": _allocation_record(bank),
            }
            if all_before is None:
                all_before = before_record
            else:
                _need(
                    before_record == all_before,
                    "independent runs start from identical full banks",
                )
            launch(inputs, bank, nworld, njmax)
            after = _numpy_bank(bank)
            input_after = _input_record(inputs, input_values)
            _need(
                input_before == input_after,
                "Warp inputs remain byte-identical through launch",
            )
            arrays, counts, rows = _capture_active(after, nworld, njmax, prefix)
            run_data[name] = {
                "before": before_record,
                "after": _bank_record(after),
                "input_before": input_before,
                "input_after": input_after,
                "allocations": allocations,
                "arrays": arrays,
                "wp_bank": bank,
                "wp_inputs": inputs,
                "counts": counts,
                "rows": rows,
            }
    source_after = _base_source_binding()
    own_after = binding_function()
    module_after = _module_sha256()
    _check_base_helpers()
    _need(source_before == source_after, "base kernel/source pins unchanged")
    _need(module_before == module_after, "prefix fixture module unchanged")
    _need(own_before == own_after, "prefix helper/control identity and code unchanged")

    expected_worlds = []
    all_prefix_preserved = True
    all_suffix_preserved = True
    all_sparse_scratch_unchanged = True
    all_counts_complete = True
    all_candidate_ascending = True
    all_original_candidate_exact = True
    all_candidate_repeat_exact = True
    any_overflow = False
    for world in range(nworld):
        expected_dofs = (
            np.flatnonzero(input_values[0][world % input_values[0].shape[0]] > 0)
            .astype(int)
            .tolist()
        )
        start = int(prefix[world])
        visible_end = min(start + len(expected_dofs), njmax)
        world_overflow = start + len(expected_dofs) > njmax
        any_overflow |= world_overflow
        expected_counts = {
            "nf": len(expected_dofs),
            "nefc": start + len(expected_dofs),
            "stored_rows": visible_end,
        }
        side_results = {}
        prefix_preserved = True
        suffix_preserved = True
        scratch_unchanged = True
        for side, info in run_data.items():
            before = {
                name: np.asarray(
                    info["before"][name]["bits"],
                    dtype="<u4" if _BANK_DTYPES[name] == "u32" else "<i4",
                ).reshape(info["before"][name]["shape"])
                for name in _BANK_DTYPES
            }
            after = info["arrays"]
            side_prefix = _unchanged(before, after, world, 0, start, _ROW_FIELDS)
            side_suffix = _unchanged(
                before, after, world, visible_end, njmax, _ROW_FIELDS
            )
            side_scratch = _unchanged(
                before, after, world, 0, njmax, ("row_nnz", "row_adr")
            )
            side_scratch &= (
                before["col_ind"][world].tobytes() == after["col_ind"][world].tobytes()
            )
            side_scratch &= (
                before["efc_nnz"][world].tobytes() == after["efc_nnz"][world].tobytes()
            )
            _need(
                info["counts"][world] == expected_counts,
                "complete initial-prefix and friction proposal counts",
            )
            prefix_preserved &= side_prefix
            suffix_preserved &= side_suffix
            scratch_unchanged &= side_scratch
            side_results[side] = {
                "rows": info["rows"][world],
                "counts": info["counts"][world],
                "prefix_preserved": side_prefix,
                "suffix_preserved": side_suffix,
                "sparse_scratch_unchanged": side_scratch,
            }
        all_prefix_preserved &= prefix_preserved
        all_suffix_preserved &= suffix_preserved
        all_sparse_scratch_unchanged &= scratch_unchanged
        _need(all_prefix_preserved, "all non-friction leading prefix rows unchanged")
        _need(all_suffix_preserved, "all inactive suffix poison unchanged")
        _need(all_sparse_scratch_unchanged, "dense-mode sparse scratch unchanged")
        expected_candidate = expected_dofs[: max(0, njmax - start)]
        candidate0 = run_data["candidate0"]["rows"][world]
        candidate1 = run_data["candidate1"]["rows"][world]
        original = run_data["original"]["rows"][world]
        candidate_order = all(
            [row["dof"] for row in rows] == expected_candidate
            for rows in (candidate0, candidate1)
        )
        candidate_replay_exact = _addressed_rows(candidate0) == _addressed_rows(
            candidate1
        )
        original_exact = _addressed_rows(original) == _addressed_rows(candidate0)
        counts_complete = all(
            run_data[name]["counts"][world] == expected_counts
            for name in ("original", "candidate0", "candidate1")
        )
        original_ids = [row["dof"] for row in original]
        counts_complete &= (
            len(original_ids) == len(set(original_ids))
            and all(dof in expected_dofs for dof in original_ids)
            and candidate_order
        )
        all_candidate_ascending &= candidate_order
        all_candidate_repeat_exact &= candidate_replay_exact
        all_original_candidate_exact &= original_exact
        all_counts_complete &= counts_complete
        expected_worlds.append(
            {
                "world": world,
                "initial_nefc": start,
                "expected_active_dofs": expected_dofs,
                "expected_counts": expected_counts,
                "overflow": world_overflow,
                "original": side_results["original"],
                "candidate0": side_results["candidate0"],
                "candidate1": side_results["candidate1"],
                "candidate_rows_ascending": candidate_order,
                "candidate_replay_addressed_exact": candidate_replay_exact,
                "original_candidate_addressed_exact": original_exact,
            }
        )
    full_candidate_bank_exact = (
        run_data["candidate0"]["after"] == run_data["candidate1"]["after"]
    )
    all_candidate_repeat_exact &= full_candidate_bank_exact
    all_allocations = [
        record
        for info in run_data.values()
        for group in (info["allocations"]["inputs"], info["allocations"]["outputs"])
        for record in group.values()
    ]
    pointers = [record["pointer"] for record in all_allocations]
    intervals = sorted(
        (record["pointer"], record["pointer"] + record["bytes"])
        for record in all_allocations
    )
    _need(
        len(pointers) == len(set(pointers))
        and all(pointer > 0 for pointer in pointers),
        "all three runs use independent live input and output allocations",
    )
    _need(
        all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
        "non-overlapping complete CPU allocation ranges",
    )
    _need(
        all(info["input_before"] == info["input_after"] for info in run_data.values()),
        "all actual CPU input arrays unchanged",
    )
    all_prefix_preserved &= all(
        run_data[side]["arrays"]["nf"].tolist()
        == [
            len(np.flatnonzero(input_values[0][w % input_values[0].shape[0]] > 0))
            for w in range(nworld)
        ]
        for side in run_data
    )
    exact_without_overflow = (
        all_prefix_preserved
        and all_suffix_preserved
        and all_sparse_scratch_unchanged
        and all_counts_complete
        and all_candidate_ascending
        and all_candidate_repeat_exact
        and all_original_candidate_exact
        and not any_overflow
    )
    report = {
        "protocol": PROTOCOL,
        "device": "cpu",
        "source_binding_before": source_before,
        "source_binding_after": source_after,
        "prefix_module_sha256_before": module_before,
        "prefix_module_sha256_after": module_after,
        "prefix_helpers_before": own_before,
        "prefix_helpers_after": own_after,
        "old_fixture_module_sha256": BASE_MODULE_SHA256,
        "input_snapshot": {
            name: {
                "shape": list(value.shape),
                "dtype": "<f4",
                "u32": base._float_bits(value),
            }
            for name, value in zip(
                ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep"),
                input_values,
                strict=True,
            )
        },
        "initial_nefc": {
            "shape": list(prefix.shape),
            "dtype": "<i4",
            "i32": prefix.tolist(),
        },
        "initial_nf": {"shape": [nworld], "dtype": "<i4", "i32": [0] * nworld},
        "runs": {
            name: {
                "inputs_before": info["input_before"],
                "inputs_after": info["input_after"],
                "allocations": info["allocations"],
                "before": info["before"],
                "after": info["after"],
            }
            for name, info in run_data.items()
        },
        "all_actual_inputs_unchanged": True,
        "all_run_buffers_independent": True,
        "worlds": expected_worlds,
        "all_prefix_rows_preserved": all_prefix_preserved,
        "all_inactive_suffix_poison_preserved": all_suffix_preserved,
        "all_dense_sparse_scratch_unchanged": all_sparse_scratch_unchanged,
        "counts_and_addresses_complete": all_counts_complete,
        "candidate_rows_ascending": all_candidate_ascending,
        "candidate_replay_full_bank_bit_identical": full_candidate_bank_exact,
        "candidate_replay_addressed_exact": all_candidate_repeat_exact,
        "original_candidate_addressed_exact": all_original_candidate_exact,
        "overflow_negative": any_overflow,
        "overflow_decision": "overflow-negative-no-qualification"
        if any_overflow
        else "no-overflow",
        "component_exact_without_overflow": exact_without_overflow,
        "fixture_decision": "dense-prefix-by-address-exact"
        if exact_without_overflow
        else "dense-prefix-negative-or-overflow",
        "qualification": False,
        "runtime_cause_proven": False,
        "native_qualified": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
        "flags": {
            "runtime_cause_proven": False,
            "native_qualified": False,
            "full_window_qualified": False,
            "training_authorized": False,
            "physical_acceptance": False,
        },
    }
    canonical = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    _need(len(canonical) <= OUTPUT_CAP, "complete bounded prefix fixture report")
    return report


_OWN_HELPERS = {
    name: (globals()[name], globals()[name].__code__)
    for name in (
        "_need",
        "_module_sha256",
        "_check_base_helpers",
        "_base_source_binding",
        "_validate_prefix",
        "_as_bits",
        "_bank_record",
        "_allocation_record",
        "_input_record",
        "_prefix_values",
        "_new_wp_bank",
        "_numpy_bank",
        "_launch_original",
        "_launch_candidate",
        "_wp_inputs",
        "_capture_active",
        "_addressed_rows",
        "_unchanged",
        "run_prefix_cpu_fixture",
    )
}


def _own_source_binding():
    need_function, need_code = _OWN_HELPERS["_need"]
    if _need is not need_function or need_function.__code__ is not need_code:
        raise ValueError("pinned prefix helper _need")
    _need(
        base.NV == 20
        and base.OUTPUT_CAP == 1024 * 1024
        and OUTPUT_CAP == 4 * 1024 * 1024,
        "literal CPU component bounds",
    )
    result = {}
    for name, (function, code) in _OWN_HELPERS.items():
        current = globals()[name]
        _need(
            current is function and current.__code__ is code,
            "pinned prefix helper " + name,
        )
        result[name] = {
            "object_id": id(function),
            "code_id": id(code),
            "code_name": code.co_qualname,
        }
    return result


def predeclared_prefix_fixture_values():
    """Ten synthetic CPU cases, including append boundaries and overflow."""
    cases = base.predeclared_fixture_values()
    prefixes = {
        "empty": [32],
        "broadcast": [0, 2, 5],
        "per-world": [0, 2, 5, 8],
        "mixed-broadcast": [1, 2, 3, 4],
        "all-dofs": [0],
        "signed-zero": [2],
        "direct-solref": [4, 8],
        "overflow": [1, 2],
    }
    for name, values in cases.items():
        values["initial_nefc"] = np.array(prefixes[name], dtype="<i4")
    append = base.predeclared_fixture_values()["broadcast"]
    append["njmax"] = 5
    append["initial_nefc"] = np.full(3, 2, dtype="<i4")
    cases["append-exact-fill"] = append
    maximum = base.predeclared_fixture_values()["per-world"]
    maximum["njmax"] = 32
    maximum["frictionloss"][:] = 0.5
    maximum["initial_nefc"] = np.array([0, 1, 16, 32], dtype="<i4")
    cases["maximum"] = maximum
    return cases


_OWN_HELPERS["_own_source_binding"] = (
    _own_source_binding,
    _own_source_binding.__code__,
)
