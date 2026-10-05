"""Synthetic CPU coverage for the bounded whole-byte CRB repeat reader."""

import copy
import json

import numpy as np
import pytest

from mjlab_microduck import stance_crb_kernel_repeat as repeat


def _bits(matrix):
    return [[f"0x{int(value):08x}" for value in row] for row in matrix.view("<u4")]


def _comparison():
    root = np.zeros((repeat.WORLDS, repeat.COMPONENTS), dtype="<f4")
    root[53, 0] = np.float32(-0.0)
    root[4, 3] = np.float32(1.25)
    children = []
    for body, name in repeat.CHILDREN:
        values = np.zeros_like(root)
        values[4, 3] = np.float32(body / 16)
        values[53, 0] = np.float32(-0.0 if body == 7 else 0.0)
        children.append(
            {
                "body_index": body,
                "body_name": name,
                "values": values.tolist(),
                "uint32_bits": _bits(values),
            }
        )
    observed_capture = root.copy()
    for row in children:
        observed_capture = np.add(
            observed_capture, np.asarray(row["values"], dtype="<f4")
        )
    observed_replay = observed_capture.copy()
    return {
        "protocol": "football-b1d-stance-inertia-order-oracle-v1:comparison",
        "source": repeat.ORACLE_SOURCE,
        "worlds": repeat.WORLDS,
        "event": {"index": 0, "phase": "scheduled-pre", "step": 0},
        "node_binding": {
            "root": {"index": 1, "name": "trunk_base"},
            "children": [
                {"index": body, "name": name} for body, name in repeat.CHILDREN
            ],
        },
        "compiled_topology_sha256": repeat.TOPOLOGY_SHA256,
        "root_cinert_values": root.tolist(),
        "root_cinert_uint32_bits": _bits(root),
        "child_crb": children,
        "capture_observed_root_crb": {
            "values": observed_capture.tolist(),
            "uint32_bits": _bits(observed_capture),
        },
        "replay_observed_root_crb": {
            "values": observed_replay.tolist(),
            "uint32_bits": _bits(observed_replay),
        },
        "actual_kernel_order_observed": False,
        "order_membership_is_not_selection": True,
        "order_membership_claims_actual_execution": False,
        "addition_order_qualified": False,
        "addition_order_cause_proven": False,
        "reduction_order_cause_proven": False,
        "component_map_qualified": False,
    }


def _topology():
    return {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": copy.deepcopy(repeat.EXPECTED_PARENTS),
        "reversed_body_tree_ids": copy.deepcopy(repeat.EXPECTED_LEVELS),
    }


@pytest.fixture
def built(monkeypatch):
    raw = json.dumps(_comparison(), separators=(",", ":"), allow_nan=False).encode()
    import hashlib

    monkeypatch.setattr(repeat, "ORACLE_SHA256", hashlib.sha256(raw).hexdigest())
    value = repeat.fixture(raw, _topology())
    return raw, value


def _raw(value):
    return (
        np.broadcast_to(
            value["baseline"],
            (repeat.REPEATS, repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS),
        )
        .copy()
        .astype("<f4")
        .tobytes()
    )


def test_fixture_builds_authenticated_bit_bound_synthetic_baseline(built):
    _, value = built
    assert value["baseline"].shape == (64, 16, 10)
    assert value["baseline"].dtype == np.dtype("<f4")
    assert value["parents"].dtype == np.dtype("<i4")
    assert value["level"].tolist() == [2, 7, 11]
    assert value["baseline"][53, 1, 0].view("<u4") == np.uint32(0x80000000)
    assert all(
        not np.any(value["baseline"][:, body].view("<u4"))
        for body in value["metadata"]["synthetic_unchanged_bodies"]
    )
    assert (
        value["metadata"]["fixture_kind"]
        == "derived-complete-forward-reduction-fixture"
    )
    assert value["metadata"]["actual_launch_inputs_captured"] is False
    assert len(value["metadata"]["candidate_root_uint32_bits"]) == 6


def test_analyzer_reports_constant_samples_without_claiming_determinism(built):
    _, value = built
    result = repeat.analyze(_raw(value), value)
    assert result["repeats"] == 32 and result["worlds"] == 64
    assert result["bit_variation_observed"] is False
    assert result["bit_variation_is_determinism_proof"] is False
    assert result["whole_result_unique_snapshots"] == 1
    assert result["world_53_root_whole_capture_bit_match_repeats"] == 0
    assert result["world_53_root_scalar_capture_bit_match_count"] == 288


def test_analyzer_preserves_signed_zero_bits_and_reports_variation(built):
    _, value = built
    matrix = (
        np.frombuffer(_raw(value), dtype="<f4")
        .copy()
        .reshape(repeat.REPEATS, repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
    )
    matrix[0, 0, 1, 0] = np.float32(-0.0)
    result = repeat.analyze(matrix.tobytes(), value)
    assert result["bit_variation_observed"] is True
    assert result["per_cell_unique_root_uint32_values"][0][0] == [
        "0x00000000",
        "0x80000000",
    ]
    assert result["whole_result_unique_snapshots"] == 2
    assert result["bit_variation_is_determinism_proof"] is False


def test_analyzer_counts_a_mixed_candidate_and_unmatched_cell(built):
    _, value = built
    baseline_result = repeat.analyze(_raw(value), value)
    matrix = (
        np.frombuffer(_raw(value), dtype="<f4")
        .copy()
        .reshape(repeat.REPEATS, repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
    )
    matrix[0, 0, 1, 0] = np.float32(1.0)
    result = repeat.analyze(matrix.tobytes(), value)
    assert result["per_cell_candidate_membership_count"][0][0] == 1
    assert (
        result["candidate_unmatched_cell_count"]
        == baseline_result["candidate_unmatched_cell_count"] + 1
    )
    assert (
        result["candidate_unmatched_observation_count"]
        == baseline_result["candidate_unmatched_observation_count"] + 1
    )
    assert result["candidate_unmatched_samples_truncated"] is False
    assert result["max_abs_delta_between_root_repeats"] == 1.0


def test_between_repeat_delta_includes_extremes_not_only_first_sample(built):
    _, value = built
    matrix = (
        np.frombuffer(_raw(value), dtype="<f4")
        .copy()
        .reshape(repeat.REPEATS, repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
    )
    matrix[1, 0, 1, 0] = np.float32(-2.0)
    matrix[2, 0, 1, 0] = np.float32(3.0)
    result = repeat.analyze(matrix.tobytes(), value)
    assert result["max_abs_delta_between_root_repeats"] == 5.0


@pytest.mark.parametrize(
    "damage",
    [
        "length",
        "trailing",
        "nonfinite",
        "nonroot",
        "fixture_source",
        "fixture_flag",
        "fixture_root_bits",
        "fixture_level",
        "fixture_parents",
    ],
)
def test_analyzer_rejects_bad_raw_bytes_and_inconsistent_fixture(built, damage):
    _, value = built
    raw = _raw(value)
    if damage == "length":
        raw = raw[:-1]
    elif damage == "trailing":
        raw += b"\0"
    elif damage == "nonfinite":
        matrix = np.frombuffer(raw, dtype="<f4").copy()
        matrix[0] = np.float32(np.inf)
        raw = matrix.tobytes()
    elif damage == "nonroot":
        matrix = (
            np.frombuffer(raw, dtype="<f4")
            .copy()
            .reshape(repeat.REPEATS, repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
        )
        matrix[0, 0, 2, 0] = np.float32(-0.0)
        raw = matrix.tobytes()
    else:
        value = copy.deepcopy(value)
        if damage == "fixture_source":
            value["metadata"]["oracle_source"] = "f" * 40
        elif damage == "fixture_flag":
            value["metadata"]["cause_proven"] = True
        elif damage == "fixture_root_bits":
            value["metadata"]["root_cinert_uint32_bits"][0][0] = "0x00000001"
        elif damage == "fixture_level":
            value["level"][0] = 7
        elif damage == "fixture_parents":
            value["parents"][2] = 16
    with pytest.raises(ValueError):
        repeat.analyze(raw, value)


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "size",
        "json",
        "source",
        "worlds",
        "bits",
        "topology",
        "topology_bool",
        "children",
        "flags",
    ],
)
def test_fixture_rejects_bad_oracle_schema_hash_bindings_and_topology(
    built, monkeypatch, damage
):
    raw, _ = built
    obj = _comparison()
    topology = _topology()
    if damage == "hash":
        bad_raw = raw + b" "
        with pytest.raises(ValueError, match="SHA-256"):
            repeat.fixture(bad_raw, topology)
        return
    if damage == "size":
        with pytest.raises(ValueError, match="size cap"):
            repeat.fixture(b" " * (repeat.JSON_LIMIT + 1), topology)
        return
    if damage == "json":
        bad_raw = b"not json"
    else:
        if damage == "source":
            obj["source"] = "f" * 40
        elif damage == "worlds":
            obj["worlds"] = True
        elif damage == "bits":
            obj["root_cinert_uint32_bits"][0][0] = "0x00000001"
        elif damage == "topology":
            topology["reversed_body_tree_ids"][0].reverse()
        elif damage == "topology_bool":
            topology["reversed_body_tree_ids"][0][0] = True
        elif damage == "children":
            obj["child_crb"][0]["body_index"] = True
        elif damage == "flags":
            obj["actual_kernel_order_observed"] = 0
        bad_raw = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()
    import hashlib

    monkeypatch.setattr(repeat, "ORACLE_SHA256", hashlib.sha256(bad_raw).hexdigest())
    with pytest.raises(ValueError):
        repeat.fixture(bad_raw, topology)
