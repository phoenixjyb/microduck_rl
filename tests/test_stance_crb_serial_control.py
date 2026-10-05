"""CPU-only tests for the fixed serial CRB accumulation checker."""

import copy
import hashlib
import json

import numpy as np
import pytest

from mjlab_microduck import stance_crb_kernel_repeat as repeat
from mjlab_microduck import stance_crb_serial_control as serial


def _bits(values):
    return [[f"0x{int(value):08x}" for value in row] for row in values.view("<u4")]


def _oracle(signed_zero=False):
    root = np.zeros((repeat.WORLDS, repeat.COMPONENTS), dtype="<f4")
    if signed_zero:
        root[0, 0] = np.float32(-0.0)
    child_rows = []
    for body, name in repeat.CHILDREN:
        child = np.zeros_like(root)
        child[2, 1] = np.float32(body / 32)
        if signed_zero:
            child[0, 0] = np.float32(-0.0)
        child_rows.append(
            {
                "body_index": body,
                "body_name": name,
                "values": child.tolist(),
                "uint32_bits": _bits(child),
            }
        )
    candidate = root.copy()
    for row in child_rows:
        np.add(candidate, np.asarray(row["values"], dtype="<f4"), out=candidate)
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
        "child_crb": child_rows,
        "capture_observed_root_crb": {
            "values": candidate.tolist(),
            "uint32_bits": _bits(candidate),
        },
        "replay_observed_root_crb": {
            "values": candidate.tolist(),
            "uint32_bits": _bits(candidate),
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
    raw = json.dumps(_oracle(), separators=(",", ":"), allow_nan=False).encode()
    monkeypatch.setattr(repeat, "ORACLE_SHA256", hashlib.sha256(raw).hexdigest())
    fixture_value = repeat.fixture(raw, _topology())
    return fixture_value


def _candidate(fixture_value):
    baseline = fixture_value["baseline"]
    candidate = baseline[:, repeat.ROOT_BODY, :].copy()
    for body in serial.SERIAL_ORDER:
        np.add(candidate, baseline[:, body, :], out=candidate)
    return candidate


def _output(fixture_value, root_values):
    matrix = np.broadcast_to(
        fixture_value["baseline"],
        (serial.REPEATS, serial.WORLDS, serial.BODIES, serial.COMPONENTS),
    ).copy()
    matrix[:, :, repeat.ROOT_BODY, :] = root_values
    return matrix.astype("<f4", copy=False).tobytes()


def test_declared_serial_schedule_matches_every_repeat(built):
    raw = _output(built, _candidate(built))
    result = serial.analyze(raw, built)
    assert result["decision"] == serial.DECISION_STABLE
    assert result["candidate_match_every_repeat"] is True
    assert result["serial_schedule"] == [[2], [7], [11]]
    assert result["selected_cpu_candidate"]["order_enum"] == 0
    assert result["selected_cpu_candidate"]["order"] == [2, 7, 11]
    assert result["kernel_calls_per_repeat"] == 3
    assert result["candidate_mismatch_scalar_count"] == 0
    assert result["candidate_mismatch_repeat_count"] == 0
    assert result["candidate_mismatch_samples"] == []
    assert result["serial_schedule_is_cause_proof"] is False
    assert result["serial_schedule_qualifies_training"] is False
    assert result["serial_schedule_qualifies_full_window"] is False


def test_another_valid_addition_order_does_not_pass_the_fixed_serial_order(monkeypatch):
    oracle = _oracle()
    root = np.asarray(oracle["root_cinert_values"], dtype="<f4")
    root[0, 0] = np.float32(2**24)
    oracle["root_cinert_values"], oracle["root_cinert_uint32_bits"] = (
        root.tolist(),
        _bits(root),
    )
    for row, scalar in zip(oracle["child_crb"], (-(2**24), 1, 1), strict=True):
        matrix = np.asarray(row["values"], dtype="<f4")
        matrix[0, 0] = np.float32(scalar)
        row["values"], row["uint32_bits"] = matrix.tolist(), _bits(matrix)
    raw = json.dumps(oracle, separators=(",", ":"), allow_nan=False).encode()
    monkeypatch.setattr(repeat, "ORACLE_SHA256", hashlib.sha256(raw).hexdigest())
    value = repeat.fixture(raw, _topology())
    assert _candidate(value)[0, 0] == np.float32(2)
    other = value["baseline"][:, repeat.ROOT_BODY].copy()
    for body in (7, 2, 11):
        np.add(other, value["baseline"][:, body], out=other)
    assert other[0, 0] == np.float32(1)
    result = serial.analyze(_output(value, other), value)
    assert result["repeat_analysis"]["candidate_unmatched_cell_count"] == 0
    assert result["decision"] == serial.DECISION_NEGATIVE
    assert result["candidate_mismatch_scalar_count"] == serial.REPEATS


def test_between_repeat_extrema_reuse_the_full_pairwise_metric(built):
    roots = np.broadcast_to(
        _candidate(built), (serial.REPEATS, serial.WORLDS, serial.COMPONENTS)
    ).copy()
    roots[1, 0, 0], roots[2, 0, 0] = np.float32(-2), np.float32(3)
    result = serial.analyze(_output(built, roots), built)
    assert result["max_abs_delta_between_root_repeats"] == 5.0


@pytest.mark.parametrize("damage", ["variation", "constant-mismatch"])
def test_variation_or_constant_mismatch_is_retained_as_negative(built, damage):
    root = np.broadcast_to(
        _candidate(built), (serial.REPEATS, serial.WORLDS, serial.COMPONENTS)
    ).copy()
    if damage == "variation":
        root[0, 0, 0] = np.float32(4.0)
    else:
        root[:, 0, 0] = np.float32(4.0)
    result = serial.analyze(_output(built, root), built)
    assert result["decision"] == serial.DECISION_NEGATIVE
    assert result["candidate_match_every_repeat"] is False
    assert result["candidate_mismatch_cell_count"] == 1
    assert result["candidate_mismatch_scalar_count"] == (
        1 if damage == "variation" else serial.REPEATS
    )
    assert result["candidate_mismatch_samples"]
    assert result["candidate_mismatch_samples_truncated"] is False
    assert (
        result["candidate_mismatch_by_repeat"][0]["matches_selected_candidate"] is False
    )


def test_selected_comparison_uses_uint32_signed_zero_bits(built, monkeypatch):
    raw = json.dumps(
        _oracle(signed_zero=True), separators=(",", ":"), allow_nan=False
    ).encode()
    monkeypatch.setattr(repeat, "ORACLE_SHA256", hashlib.sha256(raw).hexdigest())
    fixture_value = repeat.fixture(raw, _topology())
    candidate = _candidate(fixture_value)
    assert candidate[0, 0].view("<u4") == np.uint32(0x80000000)
    root = np.broadcast_to(
        candidate, (serial.REPEATS, serial.WORLDS, serial.COMPONENTS)
    ).copy()
    root[0, 0, 0] = np.float32(0.0)
    result = serial.analyze(_output(fixture_value, root), fixture_value)
    assert result["decision"] == serial.DECISION_NEGATIVE
    assert result["candidate_mismatch_scalar_count"] == 1
    assert result["candidate_mismatch_samples"][0]["observed_uint32"] == "0x00000000"
    assert result["candidate_mismatch_samples"][0]["candidate_uint32"] == "0x80000000"


@pytest.mark.parametrize("damage", ["trailing", "short", "fixture-metadata"])
def test_malformed_output_or_fixture_is_rejected(built, damage):
    raw = _output(built, _candidate(built))
    fixture_value = built
    if damage == "trailing":
        raw += b"\0"
    elif damage == "short":
        raw = raw[:-1]
    else:
        fixture_value = copy.deepcopy(built)
        fixture_value["metadata"]["oracle_source"] = "f" * 40
    with pytest.raises(ValueError):
        serial.analyze(raw, fixture_value)
