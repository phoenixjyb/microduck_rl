"""Numerical-only audit for raw CUDA dense-friction prefix samples.

This checker intentionally makes no claim about how a sample was produced.  It
accepts plain values only and reports whether the captured banks match the
pinned numerical component fixture.
"""

from mjlab_microduck import stance_friction_prefix_packet_audit as pinned
import math


PROTOCOL = "microduck-dense-friction-prefix-cuda-numerical-oct8-v1"
_INPUT_NAMES = pinned._INPUT_NAMES
_ROW_FIELDS = pinned._ROW_FIELDS
_FLOAT_FIELDS = pinned._FLOAT_FIELDS
_TOP_FLAGS = (
    "full_window_qualified",
    "native_qualified",
    "physical_acceptance",
    "runtime_cause_proven",
    "training_authorized",
)


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _case_inputs(case, worlds, name):
    inputs = pinned._input_snapshot(case, worlds)
    input_hash = next(row[3] for key, row in pinned._CASES.items() if key == name)
    _need(
        pinned._digest(
            {
                "input_snapshot": case["input_snapshot"],
                "initial_nefc": case["initial_nefc"],
                "initial_nf": case["initial_nf"],
            }
        )
        == input_hash,
        "pinned CUDA numerical inputs " + name,
    )
    return inputs


def _initial_record(value, worlds, expected, label):
    _need(
        type(value) is dict
        and set(value) == {"shape", "dtype", "i32"}
        and value["shape"] == [worlds]
        and value["dtype"] == "<i4"
        and type(value["i32"]) is list
        and len(value["i32"]) == worlds,
        "initial prefix/counter schema " + label,
    )
    for item in value["i32"]:
        pinned._plain_int(item, -(1 << 31), (1 << 31) - 1, label + " int32")
    _need(value["i32"] == list(expected), "pinned initial prefix/counter " + label)


def _input_record(record, expected, label):
    _need(
        type(record) is dict and set(record) == set(_INPUT_NAMES),
        "input record fields " + label,
    )
    for name in _INPUT_NAMES:
        row = record[name]
        _need(
            type(row) is dict and set(row) == {"shape", "dtype", "u32"},
            "numerical input record schema " + label + "." + name,
        )
        bits = row["u32"]
        _need(
            type(bits) is list and all(type(bit) is int for bit in bits),
            "numerical input bit payload " + label + "." + name,
        )
        _need(
            row
            == {
                "shape": expected[name]["shape"],
                "dtype": "<f4",
                "u32": expected[name]["bits"],
            },
            "input record differs from pinned snapshot " + label + "." + name,
        )


def _row_signature(bank, world, row, cap):
    dof = pinned._at(bank, "id", world, row, cap)
    values = tuple(
        (
            field,
            tuple(pinned._at(bank, field, world, row, cap, col) for col in range(20))
            if field == "J"
            else (pinned._at(bank, field, world, row, cap),),
        )
        for field in _FLOAT_FIELDS
    )
    return dof, values


def _active_rows(bank, world, start, end, cap, dofs, inputs):
    rows = []
    valid = True
    for row in range(start, end):
        dof = pinned._at(bank, "id", world, row, cap)
        dof_ok = type(dof) is int and 0 <= dof < 20 and dof in dofs
        valid &= dof_ok
        if not dof_ok:
            continue
        valid &= pinned._at(bank, "type", world, row, cap) == 1
        jac = [pinned._at(bank, "J", world, row, cap, col) for col in range(20)]
        valid &= jac == [0x3F800000 if col == dof else 0 for col in range(20)]
        for field in _FLOAT_FIELDS:
            values = (
                [pinned._at(bank, field, world, row, cap, col) for col in range(20)]
                if field == "J"
                else [pinned._at(bank, field, world, row, cap)]
            )
            valid &= all(math.isfinite(pinned._float(bit)) for bit in values)
        flrow = world % inputs["frictionloss"]["shape"][0]
        valid &= (
            pinned._at(bank, "frictionloss", world, row, cap)
            == inputs["frictionloss"]["bits"][flrow * 20 + dof]
        )
        valid &= (
            pinned._at(bank, "vel", world, row, cap)
            == inputs["qvel"]["bits"][world * 20 + dof]
        )
        valid &= pinned._at(bank, "pos", world, row, cap) == 0
        valid &= pinned._at(bank, "margin", world, row, cap) == 0
        rows.append(_row_signature(bank, world, row, cap))
    addresses = [row[0] for row in rows]
    unique = len(addresses) == len(set(addresses))
    return rows, bool(valid and unique), addresses


def audit_cases(cases):
    """Audit ten CUDA-shaped raw numerical samples without runtime admission."""
    _need(
        type(cases) is dict and set(cases) == set(pinned._CASES),
        "exact ten CUDA numerical cases",
    )
    result_cases = {}
    for name, (worlds, cap, prefix, input_sha, bank_sha) in pinned._CASES.items():
        del input_sha
        case = cases[name]
        _need(
            type(case) is dict
            and set(case)
            == {"device", "input_snapshot", "initial_nefc", "initial_nf", "runs"},
            "CUDA numerical case fields " + name,
        )
        _need(case["device"] == "cuda:0", "literal CUDA numerical device " + name)
        inputs = _case_inputs(case, worlds, name)
        _initial_record(case["initial_nefc"], worlds, prefix, name + ".initial_nefc")
        _initial_record(case["initial_nf"], worlds, [0] * worlds, name + ".initial_nf")
        runs = case["runs"]
        _need(
            type(runs) is dict
            and set(runs) == {"original", "candidate0", "candidate1"},
            "three numerical runs " + name,
        )
        banks = {}
        for side in ("original", "candidate0", "candidate1"):
            run = runs[side]
            _need(
                type(run) is dict
                and set(run) == {"before", "after", "inputs_before", "inputs_after"},
                "CUDA numerical run fields " + name + "." + side,
            )
            _input_record(run["inputs_before"], inputs, name + "." + side + ".before")
            _input_record(run["inputs_after"], inputs, name + "." + side + ".after")
            _need(
                run["inputs_before"] == run["inputs_after"],
                "numerical input mutation " + name + "." + side,
            )
            before_record, before = pinned._bank(run, "before", worlds, cap)
            after_record, after = pinned._bank(run, "after", worlds, cap)
            banks[side] = (before_record, before, after_record, after)
        expected_before = banks["original"][0]
        _need(
            pinned._digest(expected_before) == bank_sha
            and all(banks[side][0] == expected_before for side in banks),
            "identical pinned initial numerical banks " + name,
        )
        positive = []
        for world in range(worlds):
            flrow = world % inputs["frictionloss"]["shape"][0]
            positive.append(
                [
                    dof
                    for dof in range(20)
                    if pinned._float(inputs["frictionloss"]["bits"][flrow * 20 + dof])
                    > 0.0
                ]
            )

        prefix_ok = suffix_ok = scratch_ok = counts_ok = ascending_ok = True
        replay_addressed_ok = replay_fullbank_ok = original_addressed_ok = True
        any_overflow = False
        world_results = []
        for world, dofs in enumerate(positive):
            start = prefix[world]
            stored = min(len(dofs), cap - start)
            end = start + stored
            overflow = start + len(dofs) > cap
            any_overflow |= overflow
            expected_counts = {
                "nf": len(dofs),
                "nefc": start + len(dofs),
                "stored_rows": end,
            }
            per_run = {}
            for side in ("original", "candidate0", "candidate1"):
                _, before, _, after = banks[side]
                count_ok = before["nf"][world] == 0 and before["nefc"][world] == start
                count_ok &= after["nf"][world] == len(dofs) and after["nefc"][
                    world
                ] == start + len(dofs)
                counts_ok &= count_ok
                prefix_good = suffix_good = True
                for field in _ROW_FIELDS:
                    width = 20 if field == "J" else 1
                    offset = world * cap * width
                    prefix_good &= (
                        before[field][offset : offset + start * width]
                        == after[field][offset : offset + start * width]
                    )
                    suffix_good &= (
                        before[field][offset + end * width : offset + cap * width]
                        == after[field][offset + end * width : offset + cap * width]
                    )
                scratch_good = all(
                    before[field] == after[field]
                    for field in ("row_nnz", "row_adr", "col_ind", "efc_nnz")
                )
                prefix_ok &= prefix_good
                suffix_ok &= suffix_good
                scratch_ok &= scratch_good
                rows, rows_valid, addresses = _active_rows(
                    after, world, start, end, cap, dofs, inputs
                )
                per_run[side] = {
                    "rows": rows,
                    "rows_valid": rows_valid,
                    "addresses": addresses,
                    "counts_match": bool(count_ok),
                    "prefix_preserved": bool(prefix_good),
                    "suffix_preserved": bool(suffix_good),
                    "scratch_unchanged": bool(scratch_good),
                }
            candidate_expected = dofs[:stored]
            ascending = all(
                per_run[side]["addresses"] == candidate_expected
                for side in ("candidate0", "candidate1")
            )
            ascending &= all(
                per_run[side]["rows_valid"] for side in ("candidate0", "candidate1")
            )
            ascending_ok &= ascending
            cand0 = {dof: fields for dof, fields in per_run["candidate0"]["rows"]}
            cand1 = {dof: fields for dof, fields in per_run["candidate1"]["rows"]}
            replay_exact = cand0 == cand1 and len(cand0) == len(
                per_run["candidate0"]["rows"]
            )
            replay_addressed_ok &= replay_exact
            orig = {dof: fields for dof, fields in per_run["original"]["rows"]}
            original_exact = overflow or (
                orig == cand0 and len(orig) == len(per_run["original"]["rows"])
            )
            original_exact &= per_run["original"]["rows_valid"]
            if not overflow:
                original_exact &= (
                    sorted(per_run["original"]["addresses"]) == candidate_expected
                )
            original_addressed_ok &= original_exact
            world_results.append(
                {
                    "world": world,
                    "positive_dofs": dofs,
                    "expected_counts": expected_counts,
                    "overflow": overflow,
                    "candidate_rows_ascending": bool(ascending),
                    "candidate_replay_addressed_exact": bool(replay_exact),
                    "original_candidate_addressed_exact": bool(original_exact),
                    "runs": per_run,
                }
            )
        replay_fullbank_ok = banks["candidate0"][2] == banks["candidate1"][2]
        comp_exact = (
            all(
                (
                    prefix_ok,
                    suffix_ok,
                    scratch_ok,
                    counts_ok,
                    ascending_ok,
                    replay_addressed_ok,
                    replay_fullbank_ok,
                    original_addressed_ok,
                )
            )
            and not any_overflow
        )
        expected_overflow_case = name in {"overflow", "maximum"}
        decision = (
            "dense-prefix-by-address-exact"
            if comp_exact
            else "dense-prefix-negative-or-overflow"
        )
        result_cases[name] = {
            "worlds": world_results,
            "all_prefix_rows_preserved": bool(prefix_ok),
            "all_inactive_suffix_preserved": bool(suffix_ok),
            "all_sparse_scratch_unchanged": bool(scratch_ok),
            "counts_and_addresses_complete": bool(counts_ok),
            "candidate_rows_ascending": bool(ascending_ok),
            "candidate_replay_addressed_exact": bool(replay_addressed_ok),
            "candidate_replay_full_bank_bit_identical": bool(replay_fullbank_ok),
            "original_candidate_addressed_exact": bool(original_addressed_ok),
            "overflow_negative": bool(any_overflow),
            "overflow_cases_match_pinned_matrix": bool(
                any_overflow == expected_overflow_case
            ),
            "component_exact_without_overflow": bool(comp_exact),
            "fixture_decision": decision,
        }
    return {
        "protocol": PROTOCOL,
        "case_count": len(result_cases),
        "cases": result_cases,
        "qualification": False,
        "runtime_cause_proven": False,
        "native_qualified": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
        "flags": {flag: False for flag in _TOP_FLAGS},
    }
