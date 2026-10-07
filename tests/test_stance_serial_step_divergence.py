"""Synthetic contracts plus the retained one-tick closeout replay."""

from copy import deepcopy
import shutil

import numpy as np
import pytest

from mjlab_microduck import stance_serial_step_divergence as divergence
from mjlab_microduck import stance_serial_step_receiver as receiver

RUN = (
    divergence.Path(__file__).resolve().parents[1]
    / "artifacts/evaluations/serial-step-run-26e2ee8244b7"
)
CLOSEOUT = (
    divergence.Path(__file__).resolve().parents[1]
    / "artifacts/tools/serial-step-run-closeout-26e2ee8244b7"
)


def test_retained_closeout_replays_exactly_and_reports_observed_negative():
    result = divergence.analyze_closeout(RUN, CLOSEOUT)
    assert result["paired_receiver_decision"] == "fresh-one-step-serial-repeat-negative"
    assert result["first_observed_divergence"] is not None
    assert result["serialized_bytes"] <= divergence.OUTPUT_CAP
    assert result["serialized_bytes"] == len(receiver.old.canonical(result))
    assert result["flags"] == divergence.FLAGS
    assert result["runtime_cause_proven"] is False
    assert result["tolerance_applied"] is False
    assert len(result["comparisons"]) > 300
    first = result["first_observed_divergence"]
    assert (first["stream"], first["index"], first["field"]) == (
        "motor",
        2,
        "committed.friction",
    )
    assert first["mismatch_count"] == 345
    assert first["max_abs_delta"] == 3.725290298461914e-09
    assert result["first_observed_divergence_by_stream"]["com.weighted"] is None
    assert all(
        len(row["stats"].get("world_mismatch_scalars", ())) == 64
        for row in result["comparisons"]
        if "world_mismatch_scalars" in row["stats"]
    )


def test_signed_zero_is_a_literal_float32_mismatch():
    left = np.zeros((64,), dtype="<f4")
    right = left.copy()
    right[0] = np.float32(-0.0)
    result = divergence._float_stats(left, right, label="signed-zero fixture")
    assert result["exact_raw_bits"] is False
    assert result["bit_mismatch_scalars"] == 1
    assert result["world_mismatch_scalars"][0] == 1
    assert result["max_ordered_bit_distance"] == 1
    assert result["coordinate_samples"][0]["left_u32"] == 0
    assert result["coordinate_samples"][0]["right_u32"] == 0x80000000


def test_float_stats_keep_full_counts_and_cap_coordinates():
    left = np.zeros((64, 3), dtype="<f4")
    right = left.copy()
    right[:3, :] = 1
    result = divergence._float_stats(left, right, label="bounded samples")
    assert result["bit_mismatch_scalars"] == 9
    assert result["world_mismatch_scalars"][:3] == [3, 3, 3]
    assert len(result["coordinate_samples"]) == divergence.SAMPLE_LIMIT


def test_nonfinite_or_wrong_dtype_is_rejected():
    left = np.zeros((64,), dtype="<f4")
    nonfinite = left.copy()
    nonfinite[17] = np.nan
    with pytest.raises(ValueError, match="finite"):
        divergence._float_stats(left, nonfinite)
    with pytest.raises(ValueError, match="float32"):
        divergence._float_stats(left.astype("<f8"), left.astype("<f8"))


def test_first_divergence_uses_supplied_observation_order():
    first = {
        "stream": "rne.entries",
        "index": 4,
        "field": "rne.entries",
        "stats": {
            "bit_mismatch_scalars": 2,
            "max_abs_delta": 0.25,
            "coordinate_samples": [{"world": 1}],
        },
    }
    later = {
        "stream": "com.entries",
        "index": 2,
        "field": "com.entries",
        "stats": {
            "bit_mismatch_scalars": 1,
            "max_abs_delta": 0.5,
            "coordinate_samples": [],
        },
    }
    assert divergence._first_event([first, later])["stream"] == "rne.entries"
    assert divergence._first_event([later, first])["stream"] == "com.entries"


def test_first_divergence_preserves_scalar_mask_mismatch():
    row = {
        "stream": "masks",
        "index": 2,
        "field": "accepted",
        "stats": {
            "mismatch_scalars": 1,
            "coordinate_samples": [{"world": 3, "left": 1, "right": 0}],
        },
    }
    first = divergence._first_event([row])
    assert first["stream"] == "masks"
    assert first["mismatch_count"] == 1
    assert first["mismatch_unit"] == "mask-scalars"


def test_closeout_anchor_change_fails_before_numeric_decode(monkeypatch):
    anchors = deepcopy(divergence.CLOSEOUT_ANCHORS)
    anchors["receiver.json"]["sha256"] = "0" * 64

    def forbidden(*_args, **_kwargs):
        pytest.fail("NumPy decode ran before closeout authentication")

    monkeypatch.setattr(divergence.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="whole closeout SHA-256"):
        divergence.analyze_closeout(RUN, CLOSEOUT, closeout_anchors=anchors)


def test_wrong_source_tree_or_leaf_binding_is_rejected():
    with pytest.raises(ValueError, match="closeout consistency"):
        divergence.analyze_closeout(RUN, CLOSEOUT, source="0" * 40)
    with pytest.raises(ValueError, match="closeout consistency"):
        divergence.analyze_closeout(RUN, CLOSEOUT, expected_tree="0" * 40)
    with pytest.raises(ValueError, match="closeout consistency"):
        divergence.analyze_closeout(RUN, CLOSEOUT, expected_leaves_sha256="0" * 64)


def test_bad_terminal_anchor_is_rejected():
    anchors = deepcopy(divergence.CLOSEOUT_ANCHORS)
    anchors["terminal.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="whole closeout SHA-256 terminal.json"):
        divergence.analyze_closeout(RUN, CLOSEOUT, closeout_anchors=anchors)


def test_terminal_must_match_the_separate_external_digest(monkeypatch):
    monkeypatch.setattr(divergence, "TERMINAL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="external terminal SHA-256"):
        divergence.analyze_closeout(RUN, CLOSEOUT)


def test_bad_raw_artifact_fails_before_numpy_decode(tmp_path, monkeypatch):
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    target = copied / "serial0.com.entries.bin"
    raw = target.read_bytes()
    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])

    def forbidden(*_args, **_kwargs):
        pytest.fail("NumPy decode ran before all raw hashes passed")

    monkeypatch.setattr(divergence.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="SHA-256"):
        divergence.analyze_closeout(copied, CLOSEOUT)


def test_independent_receiver_must_reproduce_closed_receiver_bytes(monkeypatch):
    monkeypatch.setattr(
        receiver, "verify_directory", lambda *_args, **_kwargs: {"tampered": True}
    )
    with pytest.raises(ValueError, match="does not reproduce closed receiver bytes"):
        divergence.analyze_closeout(RUN, CLOSEOUT)


def test_output_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(divergence, "OUTPUT_CAP", 1)
    with pytest.raises(ValueError, match="exceeds 2 MiB"):
        divergence.analyze_closeout(RUN, CLOSEOUT)
