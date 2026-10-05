"""Whole-bank CPU comparison; supplied hashes never establish entry provenance."""

from hashlib import sha256
import re

import numpy as np

from mjlab_microduck import stance_forward_entry_reference as reference

PROTOCOL = "football-b1d-forward-entry-repeat-analysis-v1"
REPEATS = 32
MAX_OUTPUT_BYTES = 786432
SAMPLE_LIMIT = 8
_COMPONENTS = {"subtree_com": 3, "rne_backward": 6}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _bind(raw, digest, size, label):
    _need(type(raw) is bytes and len(raw) == size, f"{label} plain exact-sized bytes")
    _need(
        type(digest) is str
        and len(digest) == 64
        and _SHA256.fullmatch(digest) is not None,
        f"{label} lowercase SHA-256",
    )
    _need(sha256(raw).hexdigest() == digest, f"{label} complete byte hash binding")


def analyze(
    input_raw,
    input_sha256,
    output_raw,
    output_sha256,
    topology,
    topology_sha256,
    reduction,
):
    """Retain all finite negatives; a matching CPU candidate is not admission."""
    _need(
        type(reduction) is str
        and 0 < len(reduction) <= 32
        and reduction in _COMPONENTS,
        "fixed entry-repeat reduction",
    )
    components = _COMPONENTS[reduction]
    snapshot_bytes = 64 * 16 * components * 4
    _need(REPEATS * snapshot_bytes <= MAX_OUTPUT_BYTES, "fixed output bank cap")
    # Authenticate BOTH complete byte vectors before the reference decodes one.
    _bind(input_raw, input_sha256, snapshot_bytes, "input")
    _bind(output_raw, output_sha256, REPEATS * snapshot_bytes, "output bank")
    candidate = reference.reference(
        input_raw,
        input_sha256,
        topology,
        topology_sha256,
        reduction,
    )
    outputs = np.frombuffer(output_raw, dtype="<f4").reshape(32, 64, 16, components)
    _need(bool(np.isfinite(outputs).all()), "finite output-bank float32 values")
    expected = np.frombuffer(candidate["expected_raw"], dtype="<f4").reshape(
        64, 16, components
    )
    output_bits, expected_bits = outputs.view("<u4"), expected.view("<u4")
    mismatch = output_bits != expected_bits[None, ...]
    variability = np.any(output_bits != output_bits[0], axis=0)
    different_repeats = np.any(mismatch, axis=(1, 2, 3))
    numeric = outputs.astype(np.float64)
    delta = np.abs(numeric - expected.astype(np.float64)[None, ...])
    unique_count = len(
        {
            output_raw[index * snapshot_bytes : (index + 1) * snapshot_bytes]
            for index in range(REPEATS)
        }
    )
    samples = []
    # Only bound samples, never mismatch accounting. flatnonzero is bounded by
    # the fixed <=196608-scalar bank, even for all-negative output.
    for flat_index in np.flatnonzero(mismatch)[:SAMPLE_LIMIT]:
        repeat, world, body, component = (
            int(value) for value in np.unravel_index(int(flat_index), mismatch.shape)
        )
        samples.append(
            {
                "repeat": repeat,
                "world": world,
                "body": body,
                "component": component,
                "expected_uint32": f"0x{int(expected_bits[world, body, component]):08x}",
                "observed_uint32": f"0x{int(output_bits[repeat, world, body, component]):08x}",
            }
        )
    body_rows = []
    for label, bodies in (
        ("world_body_0", [0]),
        ("branch_root_1", [1]),
        ("other_bodies", list(range(2, 16))),
    ):
        body_rows.append(
            {
                "label": label,
                "body_ids": bodies,
                "mismatched_scalars": int(np.count_nonzero(mismatch[:, :, bodies, :])),
                "varying_cells": int(np.count_nonzero(variability[:, bodies, :])),
                "max_abs_delta_from_reference": float(delta[:, :, bodies, :].max()),
            }
        )
    all_exact = not bool(mismatch.any())
    return {
        "protocol": PROTOCOL,
        "decision": "exact-reference-and-stable"
        if all_exact and unique_count == 1
        else "numerically-negative",
        "decision_is_native_or_training_admission": False,
        "reduction": reduction,
        "input_role": candidate["metadata"]["input_role"],
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "reference_sha256": candidate["metadata"]["expected_sha256"],
        "shape": [32, 64, 16, components],
        "scalars_compared": int(outputs.size),
        "mismatched_scalars": int(np.count_nonzero(mismatch)),
        "mismatched_repeats": int(np.count_nonzero(different_repeats)),
        "varying_cells": int(np.count_nonzero(variability)),
        "full_output_unique_snapshot_count": unique_count,
        "all_exact_raw_bits": all_exact,
        "max_abs_delta_from_reference": float(delta.max()),
        "max_pairwise_same_cell_repeat_delta": float(
            (numeric.max(axis=0) - numeric.min(axis=0)).max()
        ),
        "body_groups": body_rows,
        "mismatch_samples": samples,
        "flags": dict(candidate["flags"]),
    }
