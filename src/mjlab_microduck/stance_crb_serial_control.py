"""Diagnostic checker for a declared serial CRB accumulation schedule.

The selected schedule is fixed as body 2, then 7, then 11. Results report
whether all repeated output roots exactly match that float32 candidate; they do
not establish an original kernel cause or qualify training or a full window.
"""

from hashlib import sha256

import numpy as np

from mjlab_microduck import stance_crb_kernel_repeat as repeat

PROTOCOL = "football-b1d-crb-serial-control-v1"
REPEATS = repeat.REPEATS
WORLDS = repeat.WORLDS
BODIES = repeat.BODIES
COMPONENTS = repeat.COMPONENTS
RAW_BYTES = repeat.RAW_BYTES
SERIAL_ORDER = (2, 7, 11)
SERIAL_SCHEDULE = ((2,), (7,), (11,))
SELECTED_CANDIDATE_ORDER_ENUM = 0
KERNEL_CALLS_PER_REPEAT = 3
MISMATCH_SAMPLE_LIMIT = 32
DECISION_STABLE = "stable-exact-candidate"
DECISION_NEGATIVE = "not-stable-or-not-candidate"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(rawbytes: bytes, fixture_value: dict) -> dict:
    """Check every repeated root against the fixed serial float32 candidate.

    The frozen repeat analyzer authenticates the fixture structure, output byte
    count, finiteness, and unchanged non-root bodies before this function reads
    root bits. Candidate mismatches are retained as evidence in the result.
    """
    # Run the independent whole-output validator first. Its returned summary is
    # retained so this control report does not hide any repeat-level evidence.
    repeat_summary = repeat.analyze(rawbytes, fixture_value)
    outputs = np.frombuffer(rawbytes, dtype="<f4").reshape(
        REPEATS, WORLDS, BODIES, COMPONENTS
    )
    baseline = fixture_value["baseline"]

    # Recompute in the one declared left-to-right float32 order. In-place
    # additions force a float32 rounding point after each serial kernel call.
    candidate = baseline[:, repeat.ROOT_BODY, :].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for body in SERIAL_ORDER:
            np.add(candidate, baseline[:, body, :], out=candidate)
    candidate_finite = bool(np.isfinite(candidate).all())
    candidate_bits = candidate.view("<u4")
    metadata = fixture_value["metadata"]
    _need(
        metadata["candidate_orders"][SELECTED_CANDIDATE_ORDER_ENUM]
        == list(SERIAL_ORDER),
        "selected candidate order is predeclared",
    )
    selected_bits = np.asarray(
        [
            [int(cell, 16) for cell in row]
            for row in metadata["candidate_root_uint32_bits"][
                SELECTED_CANDIDATE_ORDER_ENUM
            ]
        ],
        dtype="<u4",
    )
    _need(
        np.array_equal(candidate_bits, selected_bits),
        "independent serial CPU calculation binds the selected oracle candidate",
    )
    observed = outputs[:, :, repeat.ROOT_BODY, :]
    observed_bits = observed.view("<u4")
    mismatch_mask = observed_bits != candidate_bits[None, :, :]

    by_repeat = []
    mismatch_samples = []
    root_hash_counts = {}
    for index in range(REPEATS):
        root_raw = observed_bits[index].tobytes(order="C")
        digest = sha256(root_raw).hexdigest()
        root_hash_counts[digest] = root_hash_counts.get(digest, 0) + 1
        mismatch_count = int(np.count_nonzero(mismatch_mask[index]))
        row = {
            "repeat": index,
            "mismatched_root_scalars": mismatch_count,
            "matches_selected_candidate": mismatch_count == 0,
            "root_uint32_sha256": digest,
        }
        if candidate_finite:
            delta = np.abs(
                observed[index].astype(np.float64) - candidate.astype(np.float64)
            )
            row["max_abs_delta_from_candidate"] = float(delta.max())
        else:
            row["max_abs_delta_from_candidate"] = None
        by_repeat.append(row)

        if mismatch_count and len(mismatch_samples) < MISMATCH_SAMPLE_LIMIT:
            for world, component in np.argwhere(mismatch_mask[index]):
                if len(mismatch_samples) >= MISMATCH_SAMPLE_LIMIT:
                    break
                mismatch_samples.append(
                    {
                        "repeat": index,
                        "world": int(world),
                        "component": int(component),
                        "observed_uint32": f"0x{int(observed_bits[index, world, component]):08x}",
                        "candidate_uint32": f"0x{int(candidate_bits[world, component]):08x}",
                    }
                )

    mismatch_repeats = sum(row["mismatched_root_scalars"] > 0 for row in by_repeat)
    mismatch_scalar_count = int(np.count_nonzero(mismatch_mask))
    mismatch_cell_count = int(np.count_nonzero(mismatch_mask.any(axis=0)))
    matches_all = candidate_finite and mismatch_scalar_count == 0
    return {
        "protocol": PROTOCOL,
        "decision": DECISION_STABLE if matches_all else DECISION_NEGATIVE,
        "decision_basis": (
            "all repeated root uint32 matrices exactly match the selected CPU candidate"
            if matches_all
            else "at least one repeated root differs from the selected CPU candidate or the candidate is nonfinite"
        ),
        "serial_schedule": [list(level) for level in SERIAL_SCHEDULE],
        "serial_order": list(SERIAL_ORDER),
        "kernel_calls_per_repeat": KERNEL_CALLS_PER_REPEAT,
        "selected_cpu_candidate": {
            "name": "root+body2+body7+body11-left-to-right-float32",
            "order_enum": SELECTED_CANDIDATE_ORDER_ENUM,
            "order": list(SERIAL_ORDER),
            "finite": candidate_finite,
            "root_uint32_sha256": sha256(candidate_bits.tobytes(order="C")).hexdigest(),
        },
        "candidate_match_every_repeat": matches_all,
        "candidate_mismatch_repeat_count": mismatch_repeats,
        "candidate_mismatch_scalar_count": mismatch_scalar_count,
        "candidate_mismatch_cell_count": mismatch_cell_count,
        "candidate_mismatch_by_repeat": by_repeat,
        "candidate_mismatch_samples": mismatch_samples,
        "candidate_mismatch_samples_truncated": mismatch_scalar_count
        > len(mismatch_samples),
        "unique_root_snapshots": len(root_hash_counts),
        "root_snapshot_hash_counts": [
            {"sha256": digest, "count": count}
            for digest, count in sorted(root_hash_counts.items())
        ],
        "max_abs_delta_between_root_repeats": repeat_summary[
            "max_abs_delta_between_root_repeats"
        ],
        "repeat_analysis": repeat_summary,
        "serial_schedule_is_cause_proof": False,
        "serial_schedule_qualifies_training": False,
        "serial_schedule_qualifies_full_window": False,
        **repeat.FLAGS,
    }
