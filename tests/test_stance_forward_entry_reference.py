"""Synthetic CPU recurrence tests; no captured-entry or native claims."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from mjlab_microduck import stance_forward_entry_reference as ref

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


def run(array, reduction, **kwargs):
    raw = array.astype("<f4", copy=False).tobytes(order="C")
    return ref.reference(
        kwargs.get("raw", raw),
        kwargs.get("digest", sha256(raw).hexdigest()),
        kwargs.get("topology", topology()),
        kwargs.get("topology_digest", ref.planner.TOPOLOGY_SHA256),
        reduction,
    )


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_independent_descendant_sum_including_root(reduction, components):
    values = (
        np.arange(64, dtype="<f4")[:, None, None] * 64
        + np.arange(16, dtype="<f4")[None, :, None] * 8
        + np.arange(components, dtype="<f4")[None, None, :]
    )
    original = values.copy()
    result = run(values, reduction)
    actual = np.frombuffer(result["expected_raw"], dtype="<f4").reshape(values.shape)
    # Independent ancestor traversal, not the production grouping algorithm.
    sums = values.astype(np.float64)
    for child in range(1, 16):
        ancestor = PARENTS[child]
        while True:
            sums[:, ancestor, :] += values[:, child, :].astype(np.float64)
            if ancestor == 0:
                break
            ancestor = PARENTS[ancestor]
    assert np.array_equal(actual.view("<u4"), sums.astype("<f4").view("<u4"))
    assert np.array_equal(values.view("<u4"), original.view("<u4"))
    assert not np.array_equal(actual[:, 0, :], original[:, 0, :])
    assert (
        sha256(result["expected_raw"]).hexdigest()
        == result["metadata"]["expected_sha256"]
    )
    assert result["metadata"]["shape"] == [64, 16, components]
    assert (
        result["metadata"]["input_role"]
        == "caller-supplied-unqualified-entry-candidate"
    )
    assert all(value is False for value in result["flags"].values())
    assert all(value is False for value in result["plan"]["flags"].values())


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_per_addition_float32_rounding_not_float64_final_round(reduction, components):
    values = np.zeros((64, 16, components), dtype="<f4")
    values[:, 2, :] = np.float32(2**24)
    values[:, 7, :] = np.float32(1)
    values[:, 11, :] = np.float32(-(2**24))
    actual = np.frombuffer(run(values, reduction)["expected_raw"], dtype="<f4").reshape(
        values.shape
    )
    assert np.all(actual[:, 1, :] == 0)
    assert np.all(actual[:, 0, :] == 0)
    assert np.all(values.astype(np.float64).sum(axis=1) == 1)


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_signed_zero_retains_immutable_bytes_and_detached_metadata(
    reduction, components
):
    values = np.full((64, 16, components), np.float32(-0.0), dtype="<f4")
    result = run(values, reduction)
    assert type(result["expected_raw"]) is bytes
    assert result["expected_raw"] == values.tobytes()
    assert result["metadata"]["bytes_per_snapshot"] == 64 * 16 * components * 4
    result["metadata"]["shape"][0] = 1
    result["plan"]["launch_groups"][0]["body_ids"][0] = 99
    again = run(values, reduction)
    assert again["metadata"]["shape"] == [64, 16, components]
    assert again["plan"]["launch_groups"][0]["body_ids"] == [6, 15]


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_hash_bound_nonfinite_inputs_rejected(reduction, components, bad):
    values = np.zeros((64, 16, components), dtype="<f4")
    values[0, 15, 0] = bad
    with pytest.raises(ValueError, match="finite entry-candidate"):
        run(values, reduction)


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_intermediate_overflow_rejected_even_with_later_cancellation(
    reduction, components
):
    values = np.zeros((64, 16, components), dtype="<f4")
    maximum = np.finfo(np.float32).max
    values[:, 1, :] = maximum
    values[:, 2, :] = maximum
    values[:, 7, :] = -maximum
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="finite intermediate"):
        run(values, reduction)


@pytest.mark.parametrize("bad", [None, True, 0, [], {}, "crb", "rne", "", "x" * 10000])
def test_invalid_reduction_rejected_before_any_input_hash_or_decode(bad, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid name must not hash or decode")

    monkeypatch.setattr(ref, "sha256", forbidden)
    monkeypatch.setattr(ref.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="fixed entry-reference reduction"):
        ref.reference(None, None, None, None, bad)


@pytest.mark.parametrize(
    "kind", ["mutable", "short", "long", "wrong-width", "subclass"]
)
@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_plain_exact_byte_bound_checked_before_hash(
    kind, reduction, components, monkeypatch
):
    raw = bytes(64 * 16 * components * 4)
    if kind == "mutable":
        raw = bytearray(raw)
    elif kind == "short":
        raw = raw[:-1]
    elif kind == "long":
        raw += b"\0"
    elif kind == "wrong-width":
        raw *= 2
    else:

        class Derived(bytes):
            pass

        raw = Derived(raw)

    def forbidden(*_args):
        raise AssertionError("invalid raw must not hash")

    monkeypatch.setattr(ref, "sha256", forbidden)
    with pytest.raises(ValueError, match="plain exact-sized"):
        ref.reference(raw, "0" * 64, None, None, reduction)


@pytest.mark.parametrize(
    "digest", [None, True, 0, "A" * 64, "a" * 63, "a" * 65, "a" * 64]
)
def test_wrong_or_malformed_hash_rejected_before_decode(digest, monkeypatch):
    raw = bytes(64 * 16 * 3 * 4)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unbound input must not decode")

    monkeypatch.setattr(ref.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="SHA-256|whole input"):
        ref.reference(
            raw, digest, topology(), ref.planner.TOPOLOGY_SHA256, "subtree_com"
        )


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_wrong_topology_or_digest_rejected_before_decode(
    reduction, components, monkeypatch
):
    values = np.zeros((64, 16, components), dtype="<f4")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bad topology must not decode")

    monkeypatch.setattr(ref.np, "frombuffer", forbidden)
    bad_topology = topology()
    bad_topology["body_parentid"][1] = 1
    with pytest.raises(ValueError):
        run(values, reduction, topology=bad_topology)
    with pytest.raises(ValueError):
        run(values, reduction, topology_digest="a" * 64)


@pytest.mark.parametrize("reduction,components", REDUCTIONS)
def test_correct_hash_of_arbitrary_stage_does_not_admit_stage_provenance(
    reduction, components
):
    # A synthetic normalized-looking CoM or final sensor force can be byte-bound
    # but the helper must never upgrade it to an actual captured entry fixture.
    values = np.full((64, 16, components), np.float32(0.125), dtype="<f4")
    result = run(values, reduction)
    assert result["metadata"]["whole_input_bytes_bound"] is True
    assert result["flags"]["input_stage_provenance_authenticated"] is False
    assert result["flags"]["actual_launch_inputs_captured"] is False
    assert result["flags"]["training_authorized"] is False
