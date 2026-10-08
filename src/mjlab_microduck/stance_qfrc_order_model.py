"""Exact binary32 arithmetic hypotheses over retained active qfrc terms.

Integer dyadics, RN-even, gradual underflow; neither host float nor GPU physics
is an arithmetic oracle here. No captured simulation bytes are modified.
"""

from pathlib import Path
import os
import struct
import subprocess

PROTOCOL = "microduck-qfrc-order-model-oct8-v1"
BASE = "5456225b0b7ce14a0c42fbfeb3a87c493a56d9cd"
OWN = {
    "src/mjlab_microduck/stance_qfrc_order_model.py",
    "tests/test_stance_qfrc_order_model.py",
    "docs/experiments/2026-10-08-qfrc-order-model.md",
}
MODES = ("multiply-then-add", "fused-multiply-add")
FLAGS = {
    "native_qualified": False,
    "full_window_qualified": False,
    "runtime_cause_proven": False,
    "training_authorized": False,
    "physical_acceptance": False,
}


def need(value, message):
    if not value:
        raise ValueError(message)


def finite_parts(bits):
    """Exact signed integer coefficient and power of two for a finite f32."""
    need(type(bits) is int and 0 <= bits <= 0xFFFFFFFF, "literal binary32 word")
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    need(exponent != 255, "nonfinite binary32 input refused")
    coefficient = fraction if exponent == 0 else (1 << 23) | fraction
    if bits >> 31:
        coefficient = -coefficient
    return coefficient, -149 if exponent == 0 else exponent - 150


def _rounded_shift(n, shift):
    if shift <= 0:
        return n << -shift
    quotient, remainder = divmod(n, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + int(remainder > half or (remainder == half and quotient & 1))


def round_dyadic(coefficient, exponent, *, zero_sign=0):
    """Round exact coefficient*2**exponent to finite binary32, ties-to-even.

    Overflow is explicitly rejected, including any unfused intermediate.
    Negative tiny values round to negative zero. Exact cancellation uses the
    operation's explicit zero sign; under RN-even nonzero cancellation is +0.
    """
    need(
        type(coefficient) is int
        and type(exponent) is int
        and -1024 <= exponent <= 1024
        and coefficient.bit_length() <= 2048
        and type(zero_sign) is int
        and zero_sign in (0, 1),
        "bounded exact dyadic",
    )
    if coefficient == 0:
        return zero_sign << 31
    sign = int(coefficient < 0)
    n = abs(coefficient)
    floor_exponent = n.bit_length() - 1 + exponent
    if floor_exponent > 127:
        raise OverflowError("binary32 model overflow")
    quantum = max(-149, floor_exponent - 23)
    mantissa = _rounded_shift(n, quantum - exponent)
    if mantissa == 0:
        return sign << 31
    if mantissa >= 1 << 24:
        need(mantissa == 1 << 24, "one-bit rounding carry")
        mantissa >>= 1
        quantum += 1
    if mantissa < 1 << 23:
        need(quantum == -149, "subnormal quantum")
        return (sign << 31) | mantissa
    exponent_field = quantum + 150
    if exponent_field >= 255:
        raise OverflowError("binary32 model overflow")
    need(1 <= exponent_field <= 254, "normal rounded exponent")
    return (sign << 31) | (exponent_field << 23) | (mantissa - (1 << 23))


def _exact_add(a, ea, b, eb):
    exponent = min(ea, eb)
    return (a << (ea - exponent)) + (b << (eb - exponent)), exponent


def multiply_bits(a, b):
    ca, ea = finite_parts(a)
    cb, eb = finite_parts(b)
    return round_dyadic(ca * cb, ea + eb, zero_sign=(a ^ b) >> 31)


def add_bits(a, b):
    ca, ea = finite_parts(a)
    cb, eb = finite_parts(b)
    c, exponent = _exact_add(ca, ea, cb, eb)
    negative_zero = int(ca == cb == 0 and (a >> 31) and (b >> 31))
    return round_dyadic(c, exponent, zero_sign=negative_zero)


def fma_bits(a, b, accumulator):
    ca, ea = finite_parts(a)
    cb, eb = finite_parts(b)
    cc, ec = finite_parts(accumulator)
    product = ca * cb
    c, exponent = _exact_add(product, ea + eb, cc, ec)
    negative_zero = int(product == cc == 0 and ((a ^ b) >> 31) and (accumulator >> 31))
    return round_dyadic(c, exponent, zero_sign=negative_zero)


def left_fold(terms, mode):
    """Kernel-shaped +0 initial accumulator and one declared operation/term."""
    need(
        type(terms) in (list, tuple) and len(terms) <= 512 and mode in MODES,
        "bounded explicit row fold",
    )
    accumulator = 0
    for term in terms:
        need(type(term) in (list, tuple) and len(term) == 2, "exact J/force term")
        a, b = term
        accumulator = (
            fma_bits(a, b, accumulator)
            if mode == MODES[1]
            else add_bits(accumulator, multiply_bits(a, b))
        )
    return accumulator


def _words(raw):
    need(type(raw) is bytes and len(raw) % 4 == 0, "whole four-byte carrier")
    return struct.unpack("<" + str(len(raw) // 4) + "I", raw)


def _comparison(left, right):
    from hashlib import sha256

    need(len(left) == len(right), "same complete model extent")
    indices = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    index = indices[0] if indices else None
    return {
        "compared_words": len(left),
        "differing_words": len(indices),
        "exact": not indices if left else None,
        "first_difference": None
        if index is None
        else {
            "world": index // 20,
            "dof": index % 20,
            "word_index": index,
            "left_word_le_hex": struct.pack("<I", left[index]).hex(),
            "right_word_le_hex": struct.pack("<I", right[index]).hex(),
        },
        "left_sha256": sha256(
            struct.pack("<" + str(len(left)) + "I", *left)
        ).hexdigest(),
        "right_sha256": sha256(
            struct.pack("<" + str(len(right)) + "I", *right)
        ).hexdigest(),
    }


def compare_qfrc(left, right):
    """Active term/order hypotheses, not compiled device arithmetic proof."""
    from mjlab_microduck import stance_solver_init_row_view as rows

    receiver = rows.receiver
    for bank in (left, right):
        rows.checked_fields(bank)
    initialized = rows.compare_initialized(left, right)
    need(
        initialized["full_active_row_coverage"] is True
        and not initialized["excluded_done_worlds"],
        "complete active non-done row coverage",
    )
    need(
        all(
            total["exact"] is True
            for kind in initialized["row_comparisons"].values()
            for total in kind.values()
            if total["compared_words"]
        ),
        "paired initialized row-local equality",
    )
    contacts = [
        {name: bank[name] for name in receiver.CONTACT_PACKET_ORDER}
        for bank in (left, right)
    ]
    complete = [
        {name: bank[name] for name in receiver.COMPLETE_ORDER} for bank in (left, right)
    ]
    linked = receiver.compare_linked_contact_construction(
        contacts[0], complete[0], contacts[1], complete[1]
    )
    construction = receiver.compare_all_observed_construction(
        contacts[0], complete[0], contacts[1], complete[1]
    )
    need(
        construction["full_active_row_coverage"] is True
        and all(
            total["exact"] is True
            for kind in ("contact_field_comparisons", "noncontact_field_comparisons")
            for total in construction[kind].values()
            if total["compared_words"]
        ),
        "paired captured construction equality",
    )
    counts = receiver._int_words(left["data.nefc"])
    mapping = {}
    for link in linked["compared_payload_links"]:
        for a, b in zip(link["left_rows"], link["right_rows"]):
            need((link["world"], a) not in mapping, "unique row map")
            mapping[link["world"], a] = b
    kinds = [receiver._int_words(bank["efc.type"]) for bank in (left, right)]
    for world, count in enumerate(counts):
        for row in range(count):
            if (world, row) in mapping:
                continue
            i = world * 512 + row
            need(
                kinds[0][i] in range(5)
                and kinds[1][i] in range(5)
                and all(
                    left[name]["raw"][4 * i : 4 * i + 4]
                    == right[name]["raw"][4 * i : 4 * i + 4]
                    for name in ("efc.id", "efc.type")
                ),
                "same-offset noncontact map only",
            )
            mapping[world, row] = row
        need(
            {mapping[world, row] for row in range(count)} == set(range(count)),
            "complete one-to-one virtual row sequence",
        )
    banks = [
        {
            name: _words(bank[name]["raw"])
            for name in ("efc.J", "efc.force", "data.qfrc_constraint")
        }
        for bank in (left, right)
    ]
    for bank in banks:
        for bits in bank["data.qfrc_constraint"]:
            finite_parts(bits)
    term_count = 0
    for world, count in enumerate(counts):
        for row in range(count):
            mapped = mapping[world, row]
            a, b = world * 512 + row, world * 512 + mapped
            force = [bank["efc.force"][i] for bank, i in zip(banks, (a, b))]
            for bits in force:
                finite_parts(bits)
            need(force[0] == force[1], "paired exact initialized force term")
            for dof in range(20):
                j = [bank["efc.J"][i * 20 + dof] for bank, i in zip(banks, (a, b))]
                for bits in j:
                    finite_parts(bits)
                need(j[0] == j[1], "paired exact J term")
                term_count += 1
    observed = [bank["data.qfrc_constraint"] for bank in banks]
    result = {}
    for mode in MODES:
        stored = [[], []]
        virtual_right = []
        for world, count in enumerate(counts):
            for dof in range(20):
                for arm, bank in enumerate(banks):
                    terms = [
                        (
                            bank["efc.J"][(world * 512 + row) * 20 + dof],
                            bank["efc.force"][world * 512 + row],
                        )
                        for row in range(count)
                    ]
                    stored[arm].append(left_fold(terms, mode))
                terms = [
                    (
                        banks[1]["efc.J"][
                            (world * 512 + mapping[world, row]) * 20 + dof
                        ],
                        banks[1]["efc.force"][world * 512 + mapping[world, row]],
                    )
                    for row in range(count)
                ]
                virtual_right.append(left_fold(terms, mode))
        result[mode] = {
            "candidate0_stored_vs_observed": _comparison(stored[0], observed[0]),
            "candidate1_stored_vs_observed": _comparison(stored[1], observed[1]),
            "stored_candidate_repeat": _comparison(stored[0], stored[1]),
            "virtual_common_order_repeat": _comparison(stored[0], virtual_right),
            "candidate1_order_sensitivity": _comparison(stored[1], virtual_right),
            "stored_delta_mask_matches_observed": [a != b for a, b in zip(*stored)]
            == [a != b for a, b in zip(*observed)],
            "virtual_common_equality_expected_by_construction": True,
        }
    return {
        "protocol": PROTOCOL,
        "forward": 4,
        "phase": "initialized-before-search",
        "arithmetic": {
            "rounding": "nearest-ties-to-even",
            "underflow": "gradual",
            "overflow": "refuse including unfused intermediates",
            "initial_accumulator_word_le_hex": "00000000",
            "compiled_fma_ftz_or_schedule_proven": False,
        },
        "active_rows": sum(counts),
        "matched_term_pairs": term_count,
        "changed_contact_row_offsets": sum(
            row != mapped for (_, row), mapped in mapping.items()
        ),
        "virtual_order": "left original active row order, right existing payload/backlink mapping; analysis only",
        "observed_candidate_repeat": _comparison(*observed),
        "models": result,
        "initialized_row_view": initialized,
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": "mathematical consistency and order sensitivity only; not compiled instructions, atomic schedule, runtime cause, solver equivalence or training/physical qualification",
    }


def analysis_binding(root, source):
    from mjlab_microduck import stance_solver_init_row_view as rows

    need(
        root in rows.ROOTS
        and Path.cwd().resolve() == root
        and Path(__file__).resolve().parents[2] == root,
        "exact arithmetic worktree",
    )
    need(
        rows._git(root, "rev-parse", "HEAD").decode().strip() == source
        and rows._git(root, "branch", "--show-current").decode().strip() == rows.BRANCH
        and not rows._git(root, "status", "--porcelain"),
        "exact clean arithmetic source",
    )
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE, source],
        check=True,
        timeout=15,
    )
    need(
        set(rows._git(root, "diff", "--name-only", BASE, source).decode().splitlines())
        == OWN,
        "exact three-path arithmetic source fence",
    )
    return rows.git_binding(root, source, worktree=True)


def analyze_retained(root, source):
    from mjlab_microduck import stance_solver_init_row_view as rows

    binding = analysis_binding(root, source)
    short = rows.CAPTURE_SOURCE[:12]
    closeout = root / "artifacts/tools" / ("solver-init-tick-closeout-" + short)
    raw_root = root / "artifacts/evaluations" / ("solver-init-tick-run-" + short)
    inventory = rows.receiver.decode(
        rows.anchored(closeout / "inventory.json", 128 * 1024, rows.INVENTORY_SHA)
    )
    need(
        len(inventory) == rows.RAW_FILES
        and sum(x["bytes"] for x in inventory.values()) == rows.RAW_BYTES,
        "exact whole retained capture",
    )
    capture_binding = rows.git_binding(root, rows.CAPTURE_SOURCE, worktree=False)
    report = rows.receiver.verify_run(
        raw_root,
        inventory,
        expected_source=capture_binding,
        expected_tests_sha=rows.TESTS_SHA,
        historical_root=root,
    )
    need(
        rows.receiver.canonical(report)
        == rows.anchored(closeout / "receiver.json", 8 * 1024**2, rows.RECEIVER_SHA),
        "whole historical receiver reproduced",
    )
    raw = rows.receiver.authenticate(raw_root, inventory)
    child = rows.receiver.json_packet(raw["child.json"])
    fields = [
        rows.unpack_initialized(child["arms"][arm]["solver_init"]["snapshot"], raw)
        for arm in ("candidate0", "candidate1")
    ]
    result = compare_qfrc(*fields)
    need(binding == analysis_binding(root, source), "arithmetic source unchanged")
    return {
        "protocol": PROTOCOL,
        "analysis_source_binding": binding,
        "capture_source": rows.CAPTURE_SOURCE,
        "raw_inventory_sha256": rows.INVENTORY_SHA,
        "historical_receiver_sha256": rows.RECEIVER_SHA,
        "whole_receiver_recomputed": True,
        "raw_files": rows.RAW_FILES,
        "raw_bytes": rows.RAW_BYTES,
        "qfrc_order_model": result,
        "flags": dict(FLAGS),
        "decision": "qfrc-arithmetic-order-model-only",
    }


def main():
    import argparse
    import sys
    from mjlab_microduck import stance_solver_init_row_view as rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not sys.flags.optimize,
        "CUDA-hidden unoptimized arithmetic model",
    )
    print(
        rows.receiver.canonical(
            analyze_retained(Path.cwd().resolve(), args.source)
        ).decode()
    )


if __name__ == "__main__":
    main()
