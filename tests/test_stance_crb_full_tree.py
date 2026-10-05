"""CUDA-hidden tests for full-tree prediction and repeated-byte analysis."""

import copy

import numpy as np
import pytest

from mjlab_microduck import stance_crb_full_tree as full_tree
from mjlab_microduck import stance_crb_level_plan as level_plan


def _topology():
    return {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": list(level_plan.EXPECTED_PARENTS),
        "reversed_body_tree_ids": [list(level) for level in level_plan.EXPECTED_LEVELS],
    }


def _source_values(signed_zero=False):
    baseline = np.empty((64, 16, 10), dtype="<f4")
    for body in range(16):
        baseline[:, body, :] = np.float32(body + 1)
    if signed_zero:
        baseline[:, :, 0] = np.float32(-0.0)
    return baseline


def _predict(baseline, topology):
    planned = level_plan.plan(topology, level_plan.TOPOLOGY_SHA256)
    expected = baseline.copy()
    parents = topology["body_parentid"]
    for group in planned["launch_groups"]:
        for body in group["body_ids"]:
            parent = parents[body]
            if parent != 0:
                np.add(
                    expected[:, parent, :],
                    expected[:, body, :],
                    out=expected[:, parent, :],
                )
    return expected


def _bytes(array):
    return array.astype("<f4", copy=False).tobytes(order="C")


def _built(signed_zero=False):
    topology = _topology()
    baseline = _source_values(signed_zero)
    expected = _predict(baseline, topology)
    capture_crb = expected.copy()
    replay_crb = expected.copy()
    if not signed_zero:
        capture_crb[0, 1, 0] += np.float32(1.0)
        replay_crb[0, 1, 0] -= np.float32(1.0)
    return full_tree.fixture(
        _bytes(baseline),
        _bytes(baseline.copy()),
        _bytes(capture_crb),
        _bytes(replay_crb),
        topology,
        level_plan.TOPOLOGY_SHA256,
    )


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _outputs(fixture_value, root=None):
    expected = fixture_value["expected"]
    snapshot = expected.copy()
    if root is not None:
        snapshot[:, 1, :] = root
    return np.broadcast_to(snapshot, (32, 64, 16, 10)).copy().astype("<f4").tobytes()


def test_full_tree_fixture_evolves_ancestors_and_retains_root_mismatches():
    fixture_value = _built()
    baseline = fixture_value["baseline"]
    expected = fixture_value["expected"]
    assert fixture_value["plan"]["witnesses"]["expected_launch_group_count"] is True
    assert np.all(expected[:, 5, :] == np.float32(13.0))
    assert np.all(expected[:, 4, :] == np.float32(18.0))
    assert np.all(expected[:, 3, :] == np.float32(22.0))
    assert np.all(expected[:, 2, :] == np.float32(25.0))
    assert np.all(expected[:, 0, :] == baseline[:, 0, :])
    assert not np.array_equal(
        expected[:, 2, :].view("<u4"), baseline[:, 2, :].view("<u4")
    )
    assert fixture_value["metadata"]["nonroot_checked_scalars_per_run"] == 9600
    assert fixture_value["metadata"]["nonroot_mismatched_scalars"] == {
        "capture": 0,
        "replay": 0,
    }
    assert (
        fixture_value["metadata"]["original_root_comparisons"]["capture"][
            "mismatched_scalars"
        ]
        == 1
    )
    assert (
        fixture_value["metadata"]["original_root_comparisons"]["replay"][
            "mismatched_scalars"
        ]
        == 1
    )
    assert all(value is False for value in fixture_value["flags"].values())


def test_serial_analyzer_accepts_exact_stable_complete_prediction():
    fixture_value = _built()
    result = full_tree.analyze(
        _outputs(fixture_value), fixture_value, schedule="serial"
    )
    assert result["decision"] == "stable-exact-candidate"
    assert result["root_mismatched_scalars_from_prediction"] == 0
    assert result["root_variability_cell_count"] == 0
    assert result["full_output_unique_snapshot_count"] == 1
    assert result["max_pairwise_same_cell_repeat_delta"] == 0.0
    assert all(value is False for value in result["flags"].values())


def test_concurrent_analyzer_retains_root_variation_without_rejecting_it():
    fixture_value = _built()
    expected_root = fixture_value["expected"][:, 1, :]
    roots = np.broadcast_to(expected_root, (32, 64, 10)).copy()
    roots[:, 0, 0] = np.float32(0.0)
    roots[1, 0, 0] = np.float32(1.0)
    roots[2, 0, 0] = np.float32(-1.0)
    raw = _outputs(fixture_value)
    outputs = np.frombuffer(raw, dtype="<f4").copy().reshape(32, 64, 16, 10)
    outputs[:, :, 1, :] = roots
    result = full_tree.analyze(outputs.tobytes(), fixture_value, schedule="concurrent")
    assert result["decision"] == "diagnostic-only-concurrent"
    assert result["root_variability_cell_count"] == 1
    assert result["max_pairwise_same_cell_repeat_delta"] == 2.0
    assert result["root_mismatched_scalars_from_prediction"] == 32
    assert result["root_mismatch_repeats_from_prediction"] == 32
    assert result["full_output_unique_snapshot_count"] == 3
    assert result["mismatch_samples"]


def test_serial_constant_noncandidate_is_not_stable_exact_candidate():
    fixture_value = _built()
    roots = np.broadcast_to(fixture_value["expected"][:, 1, :], (32, 64, 10)).copy()
    roots[:, 0, 0] = np.float32(1.0)
    outputs = np.broadcast_to(fixture_value["expected"], (32, 64, 16, 10)).copy()
    outputs[:, :, 1, :] = roots
    result = full_tree.analyze(outputs.tobytes(), fixture_value, schedule="serial")
    assert result["decision"] == "not-stable-or-not-candidate"
    assert result["full_output_unique_snapshot_count"] == 1
    assert result["root_mismatched_scalars_from_prediction"] == 32


def test_analyzer_compares_signed_zero_by_raw_bits():
    fixture_value = _built(signed_zero=True)
    expected = fixture_value["expected"]
    assert np.all(expected[:, 1, 0].view("<u4") == np.uint32(0x80000000))
    outputs = np.broadcast_to(expected, (32, 64, 16, 10)).copy()
    outputs[0, 0, 1, 0] = np.float32(0.0)
    result = full_tree.analyze(outputs.tobytes(), fixture_value, schedule="concurrent")
    assert result["root_mismatched_scalars_from_prediction"] == 1
    assert result["signed_zero_mismatch_scalar_count"] == 1
    assert result["mismatch_samples"][0]["observed_uint32"] == "0x00000000"
    assert result["mismatch_samples"][0]["predicted_uint32"] == "0x80000000"


@pytest.mark.parametrize(
    "damage",
    [
        "short",
        "trailing",
        "nonfinite",
        "nonbytes",
        "cinert-different",
        "bad-topology-hash",
        "nonroot-source",
    ],
)
def test_fixture_rejects_invalid_raw_inputs_and_nonroot_disagreement(damage):
    topology = _topology()
    baseline = _source_values()
    expected = _predict(baseline, topology)
    capture = expected.copy()
    replay = expected.copy()
    args = [
        _bytes(baseline),
        _bytes(baseline.copy()),
        _bytes(capture),
        _bytes(replay),
        topology,
        level_plan.TOPOLOGY_SHA256,
    ]
    if damage == "short":
        args[0] = args[0][:-1]
    elif damage == "trailing":
        args[2] += b"\0"
    elif damage == "nonfinite":
        bad = baseline.copy()
        bad[0, 0, 0] = np.float32(np.nan)
        args[0] = _bytes(bad)
        args[1] = _bytes(bad)
    elif damage == "nonbytes":
        args[0] = bytearray(args[0])
    elif damage == "cinert-different":
        changed = baseline.copy()
        changed[0, 0, 0] = np.float32(-0.0)
        args[1] = _bytes(changed)
    elif damage == "bad-topology-hash":
        args[5] = "0" * 64
    elif damage == "nonroot-source":
        capture[0, 2, 0] += np.float32(1.0)
        args[2] = _bytes(capture)
    with pytest.raises(ValueError):
        full_tree.fixture(*args)


@pytest.mark.parametrize(
    "damage",
    [
        "short",
        "trailing",
        "nonfinite",
        "nonbytes",
        "schedule",
        "nonroot-corruption",
        "forged-expected",
        "forged-hash",
        "forged-plan",
        "forged-flags",
        "forged-count",
        "forged-body-check",
    ],
)
def test_analyzer_rejects_malformed_output_or_forged_fixture(damage):
    fixture_value = _built()
    raw = _outputs(fixture_value)
    schedule = "serial"
    if damage == "short":
        raw = raw[:-1]
    elif damage == "trailing":
        raw += b"\0"
    elif damage == "nonfinite":
        outputs = np.frombuffer(raw, dtype="<f4").copy()
        outputs[0] = np.float32(np.inf)
        raw = outputs.tobytes()
    elif damage == "nonbytes":
        raw = bytearray(raw)
    elif damage == "schedule":
        schedule = "other"
    elif damage == "nonroot-corruption":
        outputs = np.frombuffer(raw, dtype="<f4").copy().reshape(32, 64, 16, 10)
        outputs[0, 0, 2, 0] += np.float32(1.0)
        raw = outputs.tobytes()
    elif damage == "forged-expected":
        fixture_value = copy.deepcopy(fixture_value)
        fixture_value["expected"][0, 1, 0] += np.float32(1.0)
    elif damage == "forged-hash":
        fixture_value = copy.deepcopy(fixture_value)
        fixture_value["metadata"]["expected_complete_sha256"] = "0" * 64
    elif damage == "forged-plan":
        fixture_value = copy.deepcopy(fixture_value)
        fixture_value["plan"]["launch_groups"][0]["body_ids"][0] = 15
    elif damage == "forged-flags":
        fixture_value = copy.deepcopy(fixture_value)
        fixture_value["flags"]["cause_proven"] = 0
    elif damage == "forged-count":
        fixture_value = copy.deepcopy(fixture_value)
        fixture_value["metadata"]["nonroot_checked_scalars_per_run"] = True
    elif damage == "forged-body-check":
        fixture_value = copy.deepcopy(fixture_value)
        fixture_value["metadata"]["nonroot_body_checks"][0]["body_id"] = True
    with pytest.raises(ValueError):
        full_tree.analyze(raw, fixture_value, schedule=schedule)


def test_fixture_owns_arrays_and_topology_plan_are_detached():
    topology = _topology()
    baseline = _source_values()
    expected = _predict(baseline, topology)
    capture = expected.copy()
    replay = expected.copy()
    fixture_value = full_tree.fixture(
        _bytes(baseline),
        _bytes(baseline.copy()),
        _bytes(capture),
        _bytes(replay),
        topology,
        level_plan.TOPOLOGY_SHA256,
    )
    topology["reversed_body_tree_ids"][0][0] = 99
    assert fixture_value["topology"]["reversed_body_tree_ids"][0] == [6, 15]
    assert not np.shares_memory(fixture_value["baseline"], fixture_value["expected"])
    assert not np.shares_memory(fixture_value["expected"], fixture_value["capture_crb"])
    fixture_value["expected"][0, 1, 0] += np.float32(1.0)
    with pytest.raises(ValueError):
        full_tree.analyze(_outputs(fixture_value), fixture_value, schedule="serial")


def test_analyzer_bounds_forged_plan_before_canonical_serialization(monkeypatch):
    fixture_value = _built()
    fixture_value["plan"]["launch_groups"].extend([{}] * 100_000)

    def forbidden_serialization(_value):
        raise AssertionError("oversized plan reached canonical serialization")

    monkeypatch.setattr(full_tree, "_canonical_object", forbidden_serialization)
    with pytest.raises(ValueError, match="bounded plan list"):
        full_tree.analyze(_outputs(fixture_value), fixture_value, schedule="concurrent")


def test_analyzer_rejects_deep_plan_before_serialization(monkeypatch):
    fixture_value = _built()
    nested = []
    for _ in range(100):
        nested = [nested]
    fixture_value["plan"]["levels"] = nested

    def forbidden_serialization(_value):
        raise AssertionError("deep plan reached canonical serialization")

    monkeypatch.setattr(full_tree, "_canonical_object", forbidden_serialization)
    with pytest.raises(ValueError, match="bounded plan structure"):
        full_tree.analyze(_outputs(fixture_value), fixture_value, schedule="serial")


def test_analyzer_checks_output_bound_before_fixture(monkeypatch):
    def forbidden_validation(_value):
        raise AssertionError("invalid output reached fixture validation")

    monkeypatch.setattr(full_tree, "_validate_fixture", forbidden_validation)
    with pytest.raises(ValueError, match="exact 32-snapshot output byte length"):
        full_tree.analyze(b"", {}, schedule="serial")


def test_analyzer_checks_fixture_length_before_key_scan():
    fixture_value = _built()
    fixture_value.update({str(index): None for index in range(100_000)})
    with pytest.raises(ValueError, match="exact full-tree fixture fields"):
        full_tree.analyze(_outputs(fixture_value), fixture_value, schedule="serial")
