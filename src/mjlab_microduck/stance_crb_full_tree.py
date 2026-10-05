"""Pure NumPy fixture and checker for full-tree CRB accumulation outputs.

The CPU predictor applies the pinned level plan to completed-forward inputs.
It does not authenticate source trace files, launch a kernel, or establish an
actual device reduction order or training/acceptance result.
"""

from copy import deepcopy
from hashlib import sha256
import json
import re

import numpy as np

from mjlab_microduck import stance_crb_level_plan as level_plan

PROTOCOL = "football-b1d-crb-full-tree-fixture-v1"
ANALYSIS_PROTOCOL = "football-b1d-crb-full-tree-analysis-v1"
WORLDS = 64
BODIES = 16
COMPONENTS = 10
SCALARS_PER_SNAPSHOT = WORLDS * BODIES * COMPONENTS
SNAPSHOT_BYTES = SCALARS_PER_SNAPSHOT * 4
REPEATS = 32
OUTPUT_BYTES = REPEATS * SNAPSHOT_BYTES
SCALARS_PER_REPEAT = SCALARS_PER_SNAPSHOT
OUTPUT_SCALARS = REPEATS * SCALARS_PER_REPEAT
ROOT_BODY = 1
NONROOT_SCALARS = WORLDS * (BODIES - 1) * COMPONENTS
SAMPLE_LIMIT = 32
SCHEDULES = frozenset({"concurrent", "serial"})
FLAGS = {
    "actual_launch_inputs_captured": False,
    "input_trace_files_authenticated_here": False,
    "actual_kernel_order_observed": False,
    "original_pair_accepted": False,
    "cause_proven": False,
    "full_window_passed": False,
    "training_authorized": False,
    "physical_result_accepted": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(raw):
    return sha256(raw).hexdigest()


def _canonical_object(value):
    try:
        return json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical fixture object encoding") from exc


def _bound_plan(value):
    """Cap caller-owned plan structure before canonical comparison."""
    _need(type(value) is dict and len(value) == 7, "bounded plan object")
    _need(
        set(value)
        == {
            "protocol",
            "topology_sha256",
            "worlds",
            "levels",
            "launch_groups",
            "witnesses",
            "flags",
        },
        "exact plan fields",
    )
    stack = [(value, 0)]
    visited = 0
    while stack:
        item, depth = stack.pop()
        visited += 1
        _need(visited <= 2048 and depth <= 8, "bounded plan structure")
        if type(item) is dict:
            _need(len(item) <= 16, "bounded plan object members")
            for key, child in item.items():
                _need(type(key) is str and len(key) <= 128, "bounded plan key")
                stack.append((child, depth + 1))
        elif type(item) is list:
            _need(len(item) <= 16, "bounded plan list")
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            _need(len(item) <= 128, "bounded plan string")
        elif type(item) is int:
            _need(item.bit_length() <= 32, "bounded plan integer")
        else:
            _need(type(item) is bool, "plain JSON plan scalar")


def _raw_array(raw, label, byte_count):
    _need(type(raw) is bytes, f"{label} must be plain bytes")
    _need(len(raw) == byte_count, f"{label} exact byte length")
    values = np.frombuffer(raw, dtype="<f4")
    _need(bool(np.isfinite(values).all()), f"{label} finite float32 values")
    return values.reshape(WORLDS, BODIES, COMPONENTS).copy()


def _predict(baseline, parents, plan):
    expected = baseline.copy(order="C")
    with np.errstate(over="ignore", invalid="ignore"):
        for group in plan["launch_groups"]:
            for body, parent in zip(
                group["body_ids"], group["parent_ids"], strict=True
            ):
                _need(
                    parent == parents[body],
                    "planned parent ID bound to exact parent vector",
                )
                if parent == 0:
                    # Preserve the two pinned parent-zero launch groups as
                    # explicit no-ops; the kernel itself returns for these.
                    continue
                np.add(
                    expected[:, parent, :],
                    expected[:, body, :],
                    out=expected[:, parent, :],
                )
    _need(bool(np.isfinite(expected).all()), "finite full-tree CPU prediction")
    return expected


def _root_comparison(expected_root, observed_root):
    expected_bits = expected_root.view("<u4")
    observed_bits = observed_root.view("<u4")
    different = expected_bits != observed_bits
    delta = np.abs(expected_root.astype(np.float64) - observed_root.astype(np.float64))
    return {
        "scalar_count": WORLDS * COMPONENTS,
        "mismatched_scalars": int(np.count_nonzero(different)),
        "max_abs_delta": float(delta.max()),
    }


def _validate_flags(value, label):
    _need(
        type(value) is dict
        and len(value) == len(FLAGS)
        and set(value) == set(FLAGS)
        and all(type(flag) is bool and flag is False for flag in value.values()),
        label,
    )


def _validate_body_checks(value):
    expected_ids = [body for body in range(BODIES) if body != ROOT_BODY]
    _need(
        type(value) is list and len(value) == len(expected_ids),
        "fixture non-root body checks",
    )
    for row, body_id in zip(value, expected_ids, strict=True):
        _need(
            type(row) is dict
            and len(row) == 5
            and set(row)
            == {
                "body_id",
                "scalar_count",
                "capture_mismatched_scalars",
                "replay_mismatched_scalars",
                "matches_both",
            },
            "fixture non-root body row schema",
        )
        _need(
            type(row["body_id"]) is int and row["body_id"] == body_id,
            "fixture non-root body ID",
        )
        _need(
            type(row["scalar_count"]) is int
            and row["scalar_count"] == WORLDS * COMPONENTS,
            "fixture non-root scalar count",
        )
        for key in ("capture_mismatched_scalars", "replay_mismatched_scalars"):
            _need(
                type(row[key]) is int and row[key] == 0,
                "fixture non-root mismatch scalar count",
            )
        _need(
            type(row["matches_both"]) is bool and row["matches_both"] is True,
            "fixture non-root match flag",
        )


def _validate_root_comparisons(value):
    _need(
        type(value) is dict and len(value) == 2 and set(value) == {"capture", "replay"},
        "fixture original root comparison sides",
    )
    for side in ("capture", "replay"):
        row = value[side]
        _need(
            type(row) is dict
            and len(row) == 3
            and set(row) == {"scalar_count", "mismatched_scalars", "max_abs_delta"},
            "fixture original root comparison schema",
        )
        _need(
            type(row["scalar_count"]) is int
            and row["scalar_count"] == WORLDS * COMPONENTS,
            "fixture original root scalar count",
        )
        _need(
            type(row["mismatched_scalars"]) is int
            and 0 <= row["mismatched_scalars"] <= WORLDS * COMPONENTS,
            "fixture original root mismatch count",
        )
        _need(
            type(row["max_abs_delta"]) in (int, float)
            and np.isfinite(row["max_abs_delta"])
            and row["max_abs_delta"] >= 0,
            "fixture original root maximum delta",
        )


def _nonroot_checks(expected, capture_crb, replay_crb):
    checks = []
    capture_total = 0
    replay_total = 0
    for body in range(BODIES):
        if body == ROOT_BODY:
            continue
        expected_bits = expected[:, body, :].view("<u4")
        capture_bits = capture_crb[:, body, :].view("<u4")
        replay_bits = replay_crb[:, body, :].view("<u4")
        capture_mismatches = int(np.count_nonzero(expected_bits != capture_bits))
        replay_mismatches = int(np.count_nonzero(expected_bits != replay_bits))
        capture_total += capture_mismatches
        replay_total += replay_mismatches
        checks.append(
            {
                "body_id": body,
                "scalar_count": WORLDS * COMPONENTS,
                "capture_mismatched_scalars": capture_mismatches,
                "replay_mismatched_scalars": replay_mismatches,
                "matches_both": capture_mismatches == 0 and replay_mismatches == 0,
            }
        )
    return checks, capture_total, replay_total


def fixture(
    capture_cinert_raw: bytes,
    replay_cinert_raw: bytes,
    capture_crb_raw: bytes,
    replay_crb_raw: bytes,
    topology: dict,
    topology_sha256: str,
) -> dict:
    """Build an owned full-tree prediction from four exact float32 snapshots."""
    _need(type(capture_cinert_raw) is bytes, "capture cinert must be plain bytes")
    _need(type(replay_cinert_raw) is bytes, "replay cinert must be plain bytes")
    _need(type(capture_crb_raw) is bytes, "capture CRB must be plain bytes")
    _need(type(replay_crb_raw) is bytes, "replay CRB must be plain bytes")
    _need(len(capture_cinert_raw) == SNAPSHOT_BYTES, "capture cinert exact byte length")
    _need(len(replay_cinert_raw) == SNAPSHOT_BYTES, "replay cinert exact byte length")
    _need(len(capture_crb_raw) == SNAPSHOT_BYTES, "capture CRB exact byte length")
    _need(len(replay_crb_raw) == SNAPSHOT_BYTES, "replay CRB exact byte length")
    _need(
        capture_cinert_raw == replay_cinert_raw,
        "capture and replay cinert full bytes identical",
    )
    _need(
        type(topology_sha256) is str and _SHA256.fullmatch(topology_sha256) is not None,
        "topology lowercase SHA-256",
    )
    plan = level_plan.plan(topology, topology_sha256)
    parents = np.asarray(topology["body_parentid"], dtype="<i4").copy()
    baseline = _raw_array(capture_cinert_raw, "capture cinert", SNAPSHOT_BYTES)
    capture_crb = _raw_array(capture_crb_raw, "capture CRB", SNAPSHOT_BYTES)
    replay_crb = _raw_array(replay_crb_raw, "replay CRB", SNAPSHOT_BYTES)
    expected = _predict(baseline, parents, plan)

    nonroot_checks, capture_nonroot_mismatches, replay_nonroot_mismatches = (
        _nonroot_checks(expected, capture_crb, replay_crb)
    )
    _need(
        capture_nonroot_mismatches == 0 and replay_nonroot_mismatches == 0,
        "all 15 non-root body rows match capture and replay CRB bits",
    )
    root_checks = {
        "capture": _root_comparison(
            expected[:, ROOT_BODY, :], capture_crb[:, ROOT_BODY, :]
        ),
        "replay": _root_comparison(
            expected[:, ROOT_BODY, :], replay_crb[:, ROOT_BODY, :]
        ),
    }
    baseline_raw = baseline.astype("<f4", copy=False).tobytes(order="C")
    capture_owned_raw = capture_crb.astype("<f4", copy=False).tobytes(order="C")
    replay_owned_raw = replay_crb.astype("<f4", copy=False).tobytes(order="C")
    expected_raw = expected.astype("<f4", copy=False).tobytes(order="C")
    metadata = {
        "protocol": PROTOCOL,
        "fixture_kind": "derived-completed-forward-full-cinert",
        "topology_sha256": plan["topology_sha256"],
        "input_sha256": {
            "capture_cinert": _digest(capture_cinert_raw),
            "replay_cinert": _digest(replay_cinert_raw),
            "capture_crb": _digest(capture_owned_raw),
            "replay_crb": _digest(replay_owned_raw),
        },
        "baseline_sha256": _digest(baseline_raw),
        "expected_complete_sha256": _digest(expected_raw),
        "nonroot_checked_scalars_per_run": NONROOT_SCALARS,
        "nonroot_mismatched_scalars": {"capture": 0, "replay": 0},
        "nonroot_body_checks": nonroot_checks,
        "original_root_comparisons": root_checks,
        "flags": dict(FLAGS),
    }
    return {
        "baseline": baseline,
        "expected": expected,
        "parents": parents,
        "topology": deepcopy(topology),
        "plan": deepcopy(plan),
        "capture_crb": capture_crb,
        "replay_crb": replay_crb,
        "metadata": metadata,
        "flags": dict(FLAGS),
    }


def _validate_fixture(value):
    _need(
        type(value) is dict
        and len(value) == 9
        and set(value)
        == {
            "baseline",
            "expected",
            "parents",
            "topology",
            "plan",
            "capture_crb",
            "replay_crb",
            "metadata",
            "flags",
        },
        "exact full-tree fixture fields",
    )
    metadata = value["metadata"]
    _need(type(metadata) is dict and len(metadata) == 11, "fixture metadata object")
    _need(
        set(metadata)
        == {
            "protocol",
            "fixture_kind",
            "topology_sha256",
            "input_sha256",
            "baseline_sha256",
            "expected_complete_sha256",
            "nonroot_checked_scalars_per_run",
            "nonroot_mismatched_scalars",
            "nonroot_body_checks",
            "original_root_comparisons",
            "flags",
        },
        "exact full-tree metadata fields",
    )
    _need(
        metadata["protocol"] == PROTOCOL
        and metadata["fixture_kind"] == "derived-completed-forward-full-cinert",
        "fixture protocol and provenance kind",
    )
    _validate_flags(metadata["flags"], "fixture metadata flags")
    _validate_flags(value["flags"], "fixture top-level flags")

    arrays = {}
    for key in ("baseline", "expected", "capture_crb", "replay_crb"):
        array = value[key]
        _need(
            type(array) is np.ndarray
            and array.dtype == np.dtype("<f4")
            and array.shape == (WORLDS, BODIES, COMPONENTS)
            and array.flags.c_contiguous
            and bool(np.isfinite(array).all()),
            f"fixture {key} owned finite float32 matrix",
        )
        arrays[key] = array
    parents = value["parents"]
    _need(
        type(parents) is np.ndarray
        and parents.dtype == np.dtype("<i4")
        and parents.shape == (BODIES,)
        and parents.flags.c_contiguous,
        "fixture parent vector",
    )
    topology = value["topology"]
    _need(type(topology) is dict, "fixture topology object")
    _bound_plan(value["plan"])
    topology_sha = metadata["topology_sha256"]
    _need(
        type(topology_sha) is str and _SHA256.fullmatch(topology_sha) is not None,
        "fixture topology digest",
    )
    expected_plan = level_plan.plan(topology, topology_sha)
    _need(
        _canonical_object(value["plan"]) == _canonical_object(expected_plan),
        "fixture detached topology plan recomputed",
    )
    _need(
        np.array_equal(parents, np.asarray(topology["body_parentid"], dtype="<i4")),
        "fixture parent vector binds topology",
    )

    baseline_raw = arrays["baseline"].astype("<f4", copy=False).tobytes(order="C")
    input_hashes = metadata["input_sha256"]
    _need(
        type(input_hashes) is dict
        and len(input_hashes) == 4
        and set(input_hashes)
        == {"capture_cinert", "replay_cinert", "capture_crb", "replay_crb"}
        and all(
            type(digest) is str and _SHA256.fullmatch(digest)
            for digest in input_hashes.values()
        ),
        "fixture input digest schema",
    )
    _need(
        input_hashes["capture_cinert"] == _digest(baseline_raw)
        and input_hashes["replay_cinert"] == _digest(baseline_raw),
        "fixture capture/replay cinert digest binding",
    )
    for key in ("capture_crb", "replay_crb"):
        raw = arrays[key].astype("<f4", copy=False).tobytes(order="C")
        _need(input_hashes[key] == _digest(raw), f"fixture {key} digest binding")
    _need(
        type(metadata["baseline_sha256"]) is str
        and _SHA256.fullmatch(metadata["baseline_sha256"]) is not None
        and metadata["baseline_sha256"] == _digest(baseline_raw),
        "fixture baseline digest",
    )

    predicted = _predict(arrays["baseline"], parents, expected_plan)
    predicted_raw = predicted.astype("<f4", copy=False).tobytes(order="C")
    _need(
        metadata["expected_complete_sha256"] == _digest(predicted_raw),
        "fixture prediction digest",
    )
    _need(
        np.array_equal(predicted.view("<u4"), arrays["expected"].view("<u4")),
        "fixture complete prediction recomputed bitwise",
    )
    checks, capture_mismatches, replay_mismatches = _nonroot_checks(
        predicted, arrays["capture_crb"], arrays["replay_crb"]
    )
    _need(
        capture_mismatches == 0 and replay_mismatches == 0,
        "fixture non-root comparisons remain exact",
    )
    _need(
        type(metadata["nonroot_checked_scalars_per_run"]) is int
        and metadata["nonroot_checked_scalars_per_run"] == NONROOT_SCALARS,
        "fixture non-root comparison count",
    )
    nonroot_counts = metadata["nonroot_mismatched_scalars"]
    _need(
        type(nonroot_counts) is dict
        and len(nonroot_counts) == 2
        and set(nonroot_counts) == {"capture", "replay"}
        and all(type(count) is int and count == 0 for count in nonroot_counts.values()),
        "fixture non-root mismatch counts",
    )
    _validate_body_checks(metadata["nonroot_body_checks"])
    _need(
        metadata["nonroot_mismatched_scalars"] == {"capture": 0, "replay": 0}
        and metadata["nonroot_body_checks"] == checks,
        "fixture non-root witness recomputed",
    )
    root_checks = {
        "capture": _root_comparison(
            predicted[:, ROOT_BODY, :], arrays["capture_crb"][:, ROOT_BODY, :]
        ),
        "replay": _root_comparison(
            predicted[:, ROOT_BODY, :], arrays["replay_crb"][:, ROOT_BODY, :]
        ),
    }
    _validate_root_comparisons(metadata["original_root_comparisons"])
    _need(
        metadata["original_root_comparisons"] == root_checks,
        "fixture original root mismatches recomputed",
    )
    return predicted


def analyze(outputs_raw: bytes, fixture_value: dict, *, schedule: str) -> dict:
    """Analyze 32 complete snapshots against the revalidated full-tree result."""
    _need(
        type(schedule) is str and schedule in SCHEDULES,
        "declared concurrent or serial schedule",
    )
    _need(type(outputs_raw) is bytes, "outputs must be plain bytes")
    _need(len(outputs_raw) == OUTPUT_BYTES, "exact 32-snapshot output byte length")
    expected = _validate_fixture(fixture_value)
    outputs = np.frombuffer(outputs_raw, dtype="<f4").reshape(
        REPEATS, WORLDS, BODIES, COMPONENTS
    )
    _need(bool(np.isfinite(outputs).all()), "finite complete output matrix")
    output_bits = outputs.view("<u4")
    expected_bits = expected.view("<u4")

    nonroot_differences = output_bits[:, :, :, :] != expected_bits[None, :, :, :]
    nonroot_differences[:, :, ROOT_BODY, :] = False
    _need(
        not bool(nonroot_differences.any()),
        "all output non-root bodies match recomputed full-tree prediction",
    )

    observed_root = outputs[:, :, ROOT_BODY, :]
    observed_root_bits = output_bits[:, :, ROOT_BODY, :]
    expected_root = expected[:, ROOT_BODY, :]
    expected_root_bits = expected_bits[:, ROOT_BODY, :]
    mismatch_mask = observed_root_bits != expected_root_bits[None, :, :]
    mismatch_scalar_count = int(np.count_nonzero(mismatch_mask))
    mismatch_cell_count = int(np.count_nonzero(mismatch_mask.any(axis=0)))
    root_cell_unique_counts = np.empty((WORLDS, COMPONENTS), dtype="<i4")
    for world in range(WORLDS):
        for component in range(COMPONENTS):
            root_cell_unique_counts[world, component] = len(
                set(int(item) for item in observed_root_bits[:, world, component])
            )
    root_variability_cell_count = int(np.count_nonzero(root_cell_unique_counts > 1))

    pairwise_ranges = observed_root.astype(np.float64).max(
        axis=0
    ) - observed_root.astype(np.float64).min(axis=0)
    max_pairwise_same_cell_delta = float(pairwise_ranges.max())
    max_abs_delta_from_prediction = np.abs(
        observed_root.astype(np.float64) - expected_root[None, :, :].astype(np.float64)
    )
    signed_zero_mismatch_count = int(
        np.count_nonzero(mismatch_mask & (observed_root == expected_root[None, :, :]))
    )

    per_repeat = []
    mismatch_samples = []
    full_hash_counts = {}
    for index in range(REPEATS):
        full_digest = _digest(outputs[index].tobytes(order="C"))
        full_hash_counts[full_digest] = full_hash_counts.get(full_digest, 0) + 1
        count = int(np.count_nonzero(mismatch_mask[index]))
        per_repeat.append(
            {
                "repeat": index,
                "fixed_prediction_mismatched_root_scalars": count,
                "matches_fixed_prediction": count == 0,
                "max_abs_delta_from_prediction": float(
                    max_abs_delta_from_prediction[index].max()
                ),
                "full_output_sha256": full_digest,
            }
        )
        if count and len(mismatch_samples) < SAMPLE_LIMIT:
            for world, component in np.argwhere(mismatch_mask[index]):
                if len(mismatch_samples) >= SAMPLE_LIMIT:
                    break
                mismatch_samples.append(
                    {
                        "repeat": index,
                        "world": int(world),
                        "component": int(component),
                        "observed_uint32": f"0x{int(observed_root_bits[index, world, component]):08x}",
                        "predicted_uint32": f"0x{int(expected_root_bits[world, component]):08x}",
                    }
                )

    unique_full_outputs = len(full_hash_counts)
    stable_exact = mismatch_scalar_count == 0 and unique_full_outputs == 1
    if schedule == "serial":
        decision = (
            "stable-exact-candidate" if stable_exact else "not-stable-or-not-candidate"
        )
        decision_scope = (
            "all 32 complete outputs exactly match the CPU prediction and each other"
            if stable_exact
            else "at least one root snapshot differs from the CPU prediction or outputs vary"
        )
    else:
        decision = "diagnostic-only-concurrent"
        decision_scope = (
            "concurrent diagnostic retains root differences without candidate selection"
        )
    return {
        "protocol": ANALYSIS_PROTOCOL,
        "schedule": schedule,
        "decision": decision,
        "decision_scope": decision_scope,
        "output_bytes": OUTPUT_BYTES,
        "repeats": REPEATS,
        "output_checked_scalars": OUTPUT_SCALARS,
        "scalars_per_repeat": SCALARS_PER_REPEAT,
        "fixed_prediction_mismatched_output_scalars": mismatch_scalar_count,
        "root_mismatched_scalars_from_prediction": mismatch_scalar_count,
        "root_mismatched_cells_from_prediction": mismatch_cell_count,
        "root_mismatch_repeats_from_prediction": sum(
            row["fixed_prediction_mismatched_root_scalars"] > 0 for row in per_repeat
        ),
        "root_variability_cell_count": root_variability_cell_count,
        "signed_zero_mismatch_scalar_count": signed_zero_mismatch_count,
        "max_abs_delta_from_prediction": float(max_abs_delta_from_prediction.max()),
        "max_pairwise_same_cell_repeat_delta": max_pairwise_same_cell_delta,
        "full_output_unique_snapshot_count": unique_full_outputs,
        "full_output_hash_counts": [
            {"sha256": digest, "count": count}
            for digest, count in sorted(full_hash_counts.items())
        ],
        "per_repeat": per_repeat,
        "mismatch_samples": mismatch_samples,
        "mismatch_samples_truncated": mismatch_scalar_count > len(mismatch_samples),
        "repeat_analysis": {
            "baseline_sha256": fixture_value["metadata"]["baseline_sha256"],
            "expected_complete_sha256": fixture_value["metadata"][
                "expected_complete_sha256"
            ],
            "topology_sha256": fixture_value["metadata"]["topology_sha256"],
        },
        "flags": dict(FLAGS),
    }
