"""Synthetic whole-bank checks, never authentic GPU-stage evidence."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from mjlab_microduck import stance_forward_entry_repeat_analysis as analysis

PARENTS = [0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14]
LEVELS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2, 7, 11], [1], [0]]
REDUCTIONS = [("subtree_com", 3), ("rne_backward", 6)]


def topology():
    return {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": list(PARENTS),
        "reversed_body_tree_ids": deepcopy(LEVELS),
    }


def run(inputs, outputs, reduction, **kwargs):
    input_raw = inputs.astype("<f4", copy=False).tobytes()
    output_raw = outputs.astype("<f4", copy=False).tobytes()
    return analysis.analyze(
        kwargs.get("input_raw", input_raw),
        kwargs.get("input_digest", sha256(input_raw).hexdigest()),
        kwargs.get("output_raw", output_raw),
        kwargs.get("output_digest", sha256(output_raw).hexdigest()),
        kwargs.get("topology", topology()),
        kwargs.get("topology_digest", analysis.reference.planner.TOPOLOGY_SHA256),
        reduction,
    )


def zeros(components):
    return (
        np.zeros((64, 16, components), dtype="<f4"),
        np.zeros((32, 64, 16, components), dtype="<f4"),
    )


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_independent_integer_sums_match_complete_bank_without_admission(
    reduction, components
):
    inputs = np.broadcast_to(
        np.arange(16, dtype="<f4")[None, :, None], (64, 16, components)
    ).copy()
    expected = inputs.astype(np.float64)
    for child in range(1, 16):
        ancestor = PARENTS[child]
        while True:
            expected[:, ancestor, :] += inputs[:, child, :]
            if ancestor == 0:
                break
            ancestor = PARENTS[ancestor]
    outputs = np.broadcast_to(expected.astype("<f4"), (32, 64, 16, components)).copy()
    result = run(inputs, outputs, reduction)
    assert result["decision"] == "exact-reference-and-stable"
    assert result["all_exact_raw_bits"] is True
    assert result["full_output_unique_snapshot_count"] == 1
    assert result["scalars_compared"] == 32 * 64 * 16 * components
    assert (
        result["mismatched_scalars"]
        == result["mismatched_repeats"]
        == result["varying_cells"]
        == 0
    )
    assert result["mismatch_samples"] == []
    assert result["decision_is_native_or_training_admission"] is False
    assert all(value is False for value in result["flags"].values())


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
@pytest.mark.parametrize(
    "body,label", [(0, "world_body_0"), (1, "branch_root_1"), (15, "other_bodies")]
)
def test_constant_wrong_bank_is_negative_in_every_body_group(
    reduction, components, body, label
):
    inputs, outputs = zeros(components)
    outputs[:, 2, body, 0] = np.float32(0.5)
    result = run(inputs, outputs, reduction)
    assert result["decision"] == "numerically-negative"
    assert result["full_output_unique_snapshot_count"] == 1
    assert result["varying_cells"] == 0
    assert result["mismatched_scalars"] == result["mismatched_repeats"] == 32
    assert result["max_abs_delta_from_reference"] == 0.5
    assert result["max_pairwise_same_cell_repeat_delta"] == 0
    assert (
        next(row for row in result["body_groups"] if row["label"] == label)[
            "mismatched_scalars"
        ]
        == 32
    )
    assert sum(row["mismatched_scalars"] for row in result["body_groups"]) == 32


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_root_zero_and_root_one_variation_counted_separately_and_globally(
    reduction, components
):
    inputs, outputs = zeros(components)
    outputs[1, 3, 0, 0] = np.float32(0.25)
    outputs[2, 3, 1, 0] = np.float32(-0.5)
    result = run(inputs, outputs, reduction)
    assert (
        result["mismatched_scalars"]
        == result["mismatched_repeats"]
        == result["varying_cells"]
        == 2
    )
    assert result["full_output_unique_snapshot_count"] == 3
    assert [row["mismatched_scalars"] for row in result["body_groups"]] == [1, 1, 0]
    assert [row["varying_cells"] for row in result["body_groups"]] == [1, 1, 0]
    assert result["max_pairwise_same_cell_repeat_delta"] == 0.5
    assert result["decision"] == "numerically-negative"


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_signed_zero_variation_has_zero_numeric_delta_but_is_negative(
    reduction, components
):
    inputs, outputs = zeros(components)
    outputs[7, 4, 0, 1] = np.float32(-0.0)
    result = run(inputs, outputs, reduction)
    assert result["mismatched_scalars"] == result["varying_cells"] == 1
    assert result["full_output_unique_snapshot_count"] == 2
    assert (
        result["max_abs_delta_from_reference"]
        == result["max_pairwise_same_cell_repeat_delta"]
        == 0
    )
    assert result["mismatch_samples"] == [
        {
            "repeat": 7,
            "world": 4,
            "body": 0,
            "component": 1,
            "expected_uint32": "0x00000000",
            "observed_uint32": "0x80000000",
        }
    ]
    assert result["decision"] == "numerically-negative"


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_all_negative_accounting_is_not_truncated_with_samples(reduction, components):
    inputs, outputs = zeros(components)
    outputs[:] = np.float32(1)
    result = run(inputs, outputs, reduction)
    assert result["mismatched_scalars"] == outputs.size
    assert (
        sum(row["mismatched_scalars"] for row in result["body_groups"]) == outputs.size
    )
    assert result["mismatched_repeats"] == 32
    assert len(result["mismatch_samples"]) == 8
    assert result["mismatch_samples"][0]["repeat"] == 0
    assert result["mismatch_samples"][0]["body"] == 0
    assert result["decision"] == "numerically-negative"


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_inputs_outputs_and_result_are_detached(reduction, components):
    inputs, outputs = zeros(components)
    before_inputs, before_outputs = inputs.tobytes(), outputs.tobytes()
    result = run(inputs, outputs, reduction)
    assert inputs.tobytes() == before_inputs and outputs.tobytes() == before_outputs
    result["flags"]["training_authorized"] = True
    result["body_groups"][0]["body_ids"].append(1)
    again = run(inputs, outputs, reduction)
    assert again["flags"]["training_authorized"] is False
    assert again["body_groups"][0]["body_ids"] == [0]


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_hash_bound_nonfinite_outputs_are_schema_errors(reduction, components, value):
    inputs, outputs = zeros(components)
    outputs[31, 63, 15, -1] = value
    with pytest.raises(ValueError, match="finite output-bank"):
        run(inputs, outputs, reduction)


@pytest.mark.parametrize("bad", [None, True, 0, [], {}, "crb", "rne", "", "x" * 10000])
def test_invalid_reduction_rejected_before_any_hash_or_decode(bad, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid reduction must not hash or decode")

    monkeypatch.setattr(analysis, "sha256", forbidden)
    monkeypatch.setattr(analysis.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="fixed entry-repeat reduction"):
        analysis.analyze(None, None, None, None, None, None, bad)


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
@pytest.mark.parametrize(
    "kind", ["short", "long", "one-repeat", "mutable", "wrong-width"]
)
def test_output_bank_exact_size_and_type_before_decode(
    reduction, components, kind, monkeypatch
):
    inputs, outputs = zeros(components)
    raw = outputs.tobytes()
    if kind == "short":
        raw = raw[:-1]
    elif kind == "long":
        raw += b"\0"
    elif kind == "one-repeat":
        raw = raw[: 64 * 16 * components * 4]
    elif kind == "mutable":
        raw = bytearray(raw)
    else:
        raw *= 2

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bad bank must not decode either vector")

    monkeypatch.setattr(analysis.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="output bank plain exact-sized"):
        run(inputs, outputs, reduction, output_raw=raw)


@pytest.mark.parametrize(
    "digest", [None, True, 0, "A" * 64, "a" * 63, "a" * 65, "a" * 64]
)
def test_output_hash_authenticated_before_input_decode(digest, monkeypatch):
    inputs, outputs = zeros(3)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must authenticate output before input decode")

    monkeypatch.setattr(analysis.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="output bank.*SHA-256|output bank.*binding"):
        run(inputs, outputs, "subtree_com", output_digest=digest)


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_input_hash_and_topology_rejected_before_output_decode(
    reduction, components, monkeypatch
):
    inputs, outputs = zeros(components)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bad input binding/topology must not decode")

    monkeypatch.setattr(analysis.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="input.*binding"):
        run(inputs, outputs, reduction, input_digest="a" * 64)
    with pytest.raises(ValueError):
        run(inputs, outputs, reduction, topology_digest="a" * 64)
