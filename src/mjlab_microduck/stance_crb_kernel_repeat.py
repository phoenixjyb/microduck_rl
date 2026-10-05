"""Pure NumPy fixture and raw-byte analyzer for repeated CRB kernel samples.

This is a synthetic fixture derived from authenticated completed-forward data.
It does not capture launch-stage inputs and cannot establish kernel determinism,
execution order, or a physical result.
"""

from hashlib import sha256
import json
import math
import re
from itertools import permutations

import numpy as np

PROTOCOL = "football-b1d-crb-kernel-repeat-fixture-v1"
ORACLE_SOURCE = "65abe930caa16476e4d8ece44f937492da47ce2e"
ORACLE_SHA256 = "7731633338071aeb1435c0dd613f1216cc1b528b15edc8e658cbcad8691fa45c"
TOPOLOGY_SHA256 = "5b445215f52d61d10bc892a14b0d0b15e3041ff3a5dfb6c1964f06211d0cb43a"
REPEATS = 32
WORLDS = 64
BODIES = 16
COMPONENTS = 10
RAW_BYTES = REPEATS * WORLDS * BODIES * COMPONENTS * 4
JSON_LIMIT = 2 * 1024 * 1024
SMOOTH_LIMIT = 1024 * 1024
KERNEL_NAME = "_crb_accumulate"
ROOT_BODY = 1
CHILDREN = ((2, "yaw2roll"), (7, "neck"), (11, "bearing_roll"))
EXPECTED_LEVELS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2, 7, 11], [1], [0]]
EXPECTED_PARENTS = [0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14]
FLAGS = {
    "actual_launch_inputs_captured": False,
    "original_pair_accepted": False,
    "cause_proven": False,
    "full_window_passed": False,
    "training_authorized": False,
    "physical_result_accepted": False,
    "component_map_qualified": False,
    "reduction_order_cause_proven": False,
    "addition_order_qualified": False,
    "addition_order_cause_proven": False,
    "actual_kernel_order_observed": False,
    "order_membership_claims_actual_execution": False,
    "order_membership_is_not_selection": True,
}
_HEX32 = re.compile(r"0x[0-9a-f]{8}\Z")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_matrix(value, label):
    _need(type(value) is list and len(value) == WORLDS, f"{label} world rows")
    for row in value:
        _need(type(row) is list and len(row) == COMPONENTS, f"{label} components")
        for item in row:
            _need(
                type(item) in (int, float) and math.isfinite(item),
                f"{label} finite numeric scalar",
            )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype="<f4")
    _need(bool(np.isfinite(result).all()), f"{label} finite float32 conversion")
    return result


def _bit_matrix(value, label):
    _need(type(value) is list and len(value) == WORLDS, f"{label} world rows")
    for row in value:
        _need(type(row) is list and len(row) == COMPONENTS, f"{label} components")
        for item in row:
            _need(
                type(item) is str and _HEX32.fullmatch(item) is not None,
                f"{label} uint32 hex",
            )
    return np.asarray([[int(item, 16) for item in row] for row in value], dtype="<u4")


def _validate_topology(topology):
    _need(type(topology) is dict, "topology object")
    _need(
        set(topology)
        == {
            "nbody",
            "nq",
            "nv",
            "nu",
            "worlds",
            "body_parentid",
            "reversed_body_tree_ids",
        },
        "exact topology fields",
    )
    dimensions = ("nbody", "nq", "nv", "nu")
    expected = (BODIES, 21, 20, 14)
    for key, value in zip(dimensions, expected, strict=True):
        _need(
            type(topology.get(key)) is int and topology[key] == value, f"topology {key}"
        )
    _need(
        type(topology.get("worlds")) is int and topology["worlds"] == WORLDS,
        "topology worlds",
    )
    parents = topology.get("body_parentid")
    _need(type(parents) is list and len(parents) == BODIES, "topology parents length")
    _need(
        all(type(parent) is int and 0 <= parent < BODIES for parent in parents),
        "topology safe parent indices",
    )
    _need(parents == EXPECTED_PARENTS, "exact compiled body parent vector")
    _need(
        all(parents[index] == ROOT_BODY for index, _ in CHILDREN),
        "authenticated children parent root",
    )
    levels = topology.get("reversed_body_tree_ids")
    _need(
        type(levels) is list
        and all(
            type(level) is list and all(type(body) is int for body in level)
            for level in levels
        )
        and levels == EXPECTED_LEVELS,
        "exact reversed kernel levels",
    )
    flattened = [index for row in levels for index in row]
    _need(
        sorted(flattened) == list(range(BODIES)),
        "kernel levels cover each body exactly once",
    )
    return np.asarray(parents, dtype="<i4")


def _compare_schema(comparison):
    _need(type(comparison) is dict, "oracle comparison JSON object")
    _need(
        comparison.get("protocol")
        == "football-b1d-stance-inertia-order-oracle-v1:comparison",
        "oracle protocol",
    )
    _need(comparison.get("source") == ORACLE_SOURCE, "oracle source revision")
    _need(
        type(comparison.get("worlds")) is int and comparison["worlds"] == WORLDS,
        "oracle world count",
    )
    _need(
        comparison.get("compiled_topology_sha256") == TOPOLOGY_SHA256,
        "oracle compiled topology digest",
    )
    event = comparison.get("event")
    _need(
        type(event) is dict and type(event.get("index")) is int and event["index"] == 0,
        "oracle event index",
    )
    _need(
        event.get("phase") == "scheduled-pre"
        and type(event.get("step")) is int
        and event["step"] == 0,
        "oracle event phase and step",
    )
    _need(
        comparison.get("actual_kernel_order_observed") is False,
        "oracle does not claim observed kernel order",
    )
    _need(
        comparison.get("order_membership_is_not_selection") is True,
        "oracle membership limitation",
    )
    _need(
        comparison.get("order_membership_claims_actual_execution") is False,
        "oracle execution limitation",
    )
    _need(
        comparison.get("addition_order_qualified") is False
        and comparison.get("addition_order_cause_proven") is False,
        "oracle qualification flags",
    )
    _need(
        comparison.get("reduction_order_cause_proven") is False
        and comparison.get("component_map_qualified") is False,
        "component map flags",
    )
    node = comparison.get("node_binding")
    _need(type(node) is dict, "oracle node binding")
    root = node.get("root")
    _need(
        type(root) is dict
        and type(root.get("index")) is int
        and root["index"] == ROOT_BODY
        and root.get("name") == "trunk_base",
        "oracle root binding",
    )
    children = node.get("children")
    _need(
        type(children) is list and len(children) == len(CHILDREN),
        "oracle child bindings",
    )
    for row, (index, name) in zip(children, CHILDREN, strict=True):
        _need(
            type(row) is dict
            and type(row.get("index")) is int
            and row["index"] == index
            and row.get("name") == name,
            "oracle child binding",
        )

    root_values = _finite_matrix(
        comparison.get("root_cinert_values"), "root cinert values"
    )
    root_bits = _bit_matrix(
        comparison.get("root_cinert_uint32_bits"), "root cinert bits"
    )
    _need(
        np.array_equal(root_values.view("<u4"), root_bits),
        "root cinert raw-bit binding",
    )

    child_rows = comparison.get("child_crb")
    _need(
        type(child_rows) is list and len(child_rows) == len(CHILDREN),
        "oracle child CRB rows",
    )
    children_by_body = {}
    for row, (body, name) in zip(child_rows, CHILDREN, strict=True):
        _need(
            type(row) is dict
            and type(row.get("body_index")) is int
            and row["body_index"] == body,
            "child CRB body index",
        )
        _need(row.get("body_name") == name, "child CRB body name")
        values = _finite_matrix(row.get("values"), f"child {body} values")
        bits = _bit_matrix(row.get("uint32_bits"), f"child {body} bits")
        _need(np.array_equal(values.view("<u4"), bits), f"child {body} raw-bit binding")
        children_by_body[body] = values
    observed = {}
    for label in ("capture_observed_root_crb", "replay_observed_root_crb"):
        row = comparison.get(label)
        _need(type(row) is dict, f"oracle {label} object")
        values = _finite_matrix(row.get("values"), f"oracle {label} values")
        bits = _bit_matrix(row.get("uint32_bits"), f"oracle {label} bits")
        _need(
            np.array_equal(values.view("<u4"), bits), f"oracle {label} raw-bit binding"
        )
        observed[label] = (values, bits)
    return root_values, root_bits, children_by_body, observed


def fixture(oracle_raw: bytes, topology: dict) -> dict:
    """Build a bounded synthetic CPU fixture from authenticated oracle bytes."""
    _need(type(oracle_raw) is bytes, "oracle input must be bytes")
    _need(len(oracle_raw) <= JSON_LIMIT, "oracle JSON size cap")
    _need(
        sha256(oracle_raw).hexdigest() == ORACLE_SHA256,
        "authenticated oracle whole-file SHA-256",
    )
    parents = _validate_topology(topology)
    try:
        comparison = json.loads(oracle_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("oracle JSON decoding") from exc
    root, root_bits, child_inputs, observed = _compare_schema(comparison)

    baseline = np.zeros((WORLDS, BODIES, COMPONENTS), dtype="<f4")
    baseline[:, ROOT_BODY, :] = root
    for body, child_values in child_inputs.items():
        baseline[:, body, :] = child_values
    candidate_bits = []
    candidate_labels = []
    for order in permutations(tuple(body for body, _ in CHILDREN)):
        candidate = root.copy()
        for body in order:
            np.add(candidate, child_inputs[body], out=candidate)
        candidate_bits.append(candidate.view("<u4"))
        candidate_labels.append(list(order))
    metadata = {
        "protocol": PROTOCOL,
        "fixture_kind": "derived-complete-forward-reduction-fixture",
        "oracle_sha256": ORACLE_SHA256,
        "oracle_source": ORACLE_SOURCE,
        "oracle_topology_sha256": TOPOLOGY_SHA256,
        **FLAGS,
        "root_cinert_uint32_bits": [
            [f"0x{int(value):08x}" for value in row] for row in root_bits
        ],
        "capture_observed_root_crb_uint32_bits": [
            [f"0x{int(value):08x}" for value in row]
            for row in observed["capture_observed_root_crb"][1]
        ],
        "replay_observed_root_crb_uint32_bits": [
            [f"0x{int(value):08x}" for value in row]
            for row in observed["replay_observed_root_crb"][1]
        ],
        "child_crb_uint32_bits": {
            str(body): [
                [f"0x{int(value):08x}" for value in row]
                for row in child_inputs[body].view("<u4")
            ]
            for body, _ in CHILDREN
        },
        "candidate_orders": candidate_labels,
        "candidate_root_uint32_bits": [
            [[f"0x{int(value):08x}" for value in row] for row in bits]
            for bits in candidate_bits
        ],
        "synthetic_unchanged_bodies": [
            index
            for index in range(BODIES)
            if index not in {ROOT_BODY, *(body for body, _ in CHILDREN)}
        ],
    }
    return {
        "baseline": baseline,
        "parents": parents,
        "level": np.asarray([2, 7, 11], dtype="<i4"),
        "metadata": metadata,
    }


def _validate_fixture(value):
    _need(type(value) is dict, "fixture object")
    metadata = value.get("metadata")
    _need(type(metadata) is dict, "fixture metadata")
    expected_flags = FLAGS
    _need(
        metadata.get("protocol") == PROTOCOL
        and metadata.get("fixture_kind")
        == "derived-complete-forward-reduction-fixture",
        "fixture provenance",
    )
    _need(
        metadata.get("oracle_sha256") == ORACLE_SHA256
        and metadata.get("oracle_source") == ORACLE_SOURCE
        and metadata.get("oracle_topology_sha256") == TOPOLOGY_SHA256,
        "fixture oracle binding",
    )
    _need(
        all(
            type(metadata.get(key)) is bool and metadata[key] is expected
            for key, expected in expected_flags.items()
        ),
        "fixture limitation flags",
    )
    baseline = value.get("baseline")
    _need(
        type(baseline) is np.ndarray
        and baseline.dtype == np.dtype("<f4")
        and baseline.shape == (WORLDS, BODIES, COMPONENTS),
        "fixture baseline shape and dtype",
    )
    _need(
        baseline.flags.c_contiguous and bool(np.isfinite(baseline).all()),
        "contiguous finite fixture baseline",
    )
    root_bits = _bit_matrix(
        metadata.get("root_cinert_uint32_bits"), "fixture root bits"
    )
    _need(
        np.array_equal(baseline[:, ROOT_BODY, :].view("<u4"), root_bits),
        "fixture root baseline bits",
    )
    parents = value.get("parents")
    _need(
        type(parents) is np.ndarray
        and parents.dtype == np.dtype("<i4")
        and parents.shape == (BODIES,),
        "fixture parents shape and dtype",
    )
    _need(
        parents.flags.c_contiguous
        and np.array_equal(parents, np.asarray(EXPECTED_PARENTS, dtype="<i4")),
        "fixture exact parent vector",
    )
    level = value.get("level")
    _need(
        type(level) is np.ndarray
        and level.dtype == np.dtype("<i4")
        and level.flags.c_contiguous
        and level.shape == (3,)
        and np.array_equal(level, np.asarray([2, 7, 11], dtype="<i4")),
        "fixture kernel level",
    )
    bits_by_body = metadata.get("child_crb_uint32_bits")
    _need(
        type(bits_by_body) is dict
        and set(bits_by_body) == {str(body) for body, _ in CHILDREN},
        "fixture child input metadata",
    )
    for body, _ in CHILDREN:
        child_bits = _bit_matrix(bits_by_body[str(body)], f"fixture child {body} bits")
        _need(
            np.array_equal(baseline[:, body, :].view("<u4"), child_bits),
            f"fixture child {body} bits",
        )
    unchanged = metadata.get("synthetic_unchanged_bodies")
    expected_unchanged = [
        index
        for index in range(BODIES)
        if index not in {ROOT_BODY, *(body for body, _ in CHILDREN)}
    ]
    _need(unchanged == expected_unchanged, "fixture synthetic body declaration")
    for body in expected_unchanged:
        _need(
            bool(np.all(baseline[:, body, :].view("<u4") == 0)),
            f"synthetic zero body {body}",
        )
    for label in (
        "capture_observed_root_crb_uint32_bits",
        "replay_observed_root_crb_uint32_bits",
    ):
        _bit_matrix(metadata.get(label), f"fixture {label}")
    candidate_orders = metadata.get("candidate_orders")
    _need(
        candidate_orders
        == [list(order) for order in permutations(tuple(body for body, _ in CHILDREN))],
        "fixture candidate order enumeration",
    )
    candidate_matrices = metadata.get("candidate_root_uint32_bits")
    _need(
        type(candidate_matrices) is list and len(candidate_matrices) == 6,
        "fixture candidate matrix count",
    )
    root = baseline[:, ROOT_BODY, :]
    child_matrices = {
        body: _bit_matrix(bits_by_body[str(body)], f"fixture child {body} bits")
        for body, _ in CHILDREN
    }
    expected_candidates = []
    for order in candidate_orders:
        candidate = root.copy()
        for body in order:
            np.add(candidate, child_matrices[body].view("<f4"), out=candidate)
        expected_candidates.append(candidate.view("<u4"))
    for index, expected_candidate in enumerate(expected_candidates):
        _need(
            np.array_equal(
                _bit_matrix(candidate_matrices[index], "fixture candidate bits"),
                expected_candidate,
            ),
            "fixture candidate raw-bit consistency",
        )
    return baseline


def analyze(output_raw: bytes, fixture_value: dict) -> dict:
    """Analyze complete little-endian float32 snapshots without tensor loading."""
    _need(type(output_raw) is bytes, "output must be raw bytes")
    _need(len(output_raw) == RAW_BYTES, "exact output raw byte length")
    baseline = _validate_fixture(fixture_value)
    outputs = np.frombuffer(output_raw, dtype="<f4").reshape(
        REPEATS, WORLDS, BODIES, COMPONENTS
    )
    _need(bool(np.isfinite(outputs).all()), "finite output matrix")
    output_bits = outputs.view("<u4")
    baseline_bits = baseline.view("<u4")
    for body in range(BODIES):
        if body != ROOT_BODY:
            _need(
                bool(
                    np.all(
                        output_bits[:, :, body, :] == baseline_bits[None, :, body, :]
                    )
                ),
                f"non-root body {body} unchanged bitwise",
            )

    root = outputs[:, :, ROOT_BODY, :]
    root_bits = output_bits[:, :, ROOT_BODY, :]
    cell_values = []
    cell_counts = []
    all_bits = set()
    for world in range(WORLDS):
        values_row = []
        counts_row = []
        for component in range(COMPONENTS):
            unique = sorted({int(value) for value in root_bits[:, world, component]})
            all_bits.update(unique)
            values_row.append([f"0x{value:08x}" for value in unique])
            counts_row.append(len(unique))
        cell_values.append(values_row)
        cell_counts.append(counts_row)

    snapshot_counts = {}
    for repeat in range(REPEATS):
        digest = sha256(outputs[repeat].tobytes(order="C")).hexdigest()
        snapshot_counts[digest] = snapshot_counts.get(digest, 0) + 1
    deltas = np.abs(
        root.astype(np.float64) - baseline[None, :, ROOT_BODY, :].astype(np.float64)
    )
    root64 = root.astype(np.float64)
    repeat_deltas = root64.max(axis=0) - root64.min(axis=0)
    candidate_bits = [
        _bit_matrix(rows, "fixture candidate bits")
        for rows in fixture_value["metadata"]["candidate_root_uint32_bits"]
    ]
    membership_counts = np.zeros((WORLDS, COMPONENTS), dtype="<i4")
    unmatched = []
    unmatched_cells = 0
    unmatched_observations = 0
    unmatched_unique_values = set()
    for world in range(WORLDS):
        for component in range(COMPONENTS):
            observed_rows = [int(value) for value in root_bits[:, world, component]]
            observed = set(observed_rows)
            candidates = {int(matrix[world, component]) for matrix in candidate_bits}
            membership_counts[world, component] = len(observed & candidates)
            difference = observed - candidates
            if difference:
                unmatched_cells += 1
                unmatched_observations += sum(
                    value not in candidates for value in observed_rows
                )
                unmatched_unique_values.update(difference)
                if len(unmatched) < 32:
                    unmatched.append(
                        {
                            "world": world,
                            "component": component,
                            "unmatched_uint32_values": [
                                f"0x{value:08x}" for value in sorted(difference)
                            ],
                        }
                    )
    capture_bits = _bit_matrix(
        fixture_value["metadata"]["capture_observed_root_crb_uint32_bits"],
        "fixture capture root bits",
    )
    replay_bits = _bit_matrix(
        fixture_value["metadata"]["replay_observed_root_crb_uint32_bits"],
        "fixture replay root bits",
    )
    world53 = root[:, 53, :].view("<u4")
    capture53 = capture_bits[53]
    replay53 = replay_bits[53]
    return {
        "protocol": PROTOCOL + ":analysis",
        "fixture_kind": "derived-complete-forward-reduction-fixture",
        "oracle_sha256": ORACLE_SHA256,
        "repeats": REPEATS,
        "worlds": WORLDS,
        "bodies": BODIES,
        "components": COMPONENTS,
        "root_body": ROOT_BODY,
        "per_cell_unique_root_uint32_values": cell_values,
        "per_cell_unique_root_uint32_counts": cell_counts,
        "unique_root_uint32_value_count": len(all_bits),
        "whole_result_unique_snapshots": len(snapshot_counts),
        "whole_result_snapshot_counts": [
            {"sha256": digest, "count": count}
            for digest, count in sorted(snapshot_counts.items())
        ],
        "per_cell_candidate_membership_count": membership_counts.tolist(),
        "candidate_unmatched_cell_count": unmatched_cells,
        "candidate_unmatched_observation_count": unmatched_observations,
        "candidate_unmatched_unique_value_count": len(unmatched_unique_values),
        "candidate_unmatched_samples": unmatched,
        "candidate_unmatched_samples_truncated": unmatched_cells > len(unmatched),
        "world_53_root_whole_capture_bit_match_repeats": int(
            np.count_nonzero(np.all(world53 == capture53[None, :], axis=1))
        ),
        "world_53_root_whole_replay_bit_match_repeats": int(
            np.count_nonzero(np.all(world53 == replay53[None, :], axis=1))
        ),
        "world_53_root_scalar_capture_bit_match_count": int(
            np.count_nonzero(world53 == capture53[None, :])
        ),
        "world_53_root_scalar_replay_bit_match_count": int(
            np.count_nonzero(world53 == replay53[None, :])
        ),
        "max_abs_delta_from_fixture_root": float(deltas.max()),
        "max_abs_delta_between_root_repeats": float(repeat_deltas.max()),
        "bit_variation_observed": any(
            count > 1 for row in cell_counts for count in row
        ),
        "bit_variation_is_determinism_proof": False,
        **FLAGS,
    }
