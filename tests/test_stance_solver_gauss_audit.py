"""CPU arithmetic hypotheses and authentication refusal tests; no CUDA."""

from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_solver_gauss_audit as audit


def fields():
    result = {name: struct.pack("<1280f", *([1.0] * 1280)) for name in audit.NAMES}
    for name in ("data.qfrc_smooth", "data.qacc_smooth"):
        result[name] = bytes(5120)
    result["context.gauss"] = struct.pack("<64f", *([10.0] * 64))
    result["context.done"] = bytes(64)
    return result


def test_exact_synthetic_result():
    result = audit.analyze(fields())
    assert result["decision"] == "retained-gauss-arithmetic-consistent-only"
    assert result["outside_bound"] == 0
    assert result["separate_bit_mismatches"] == result["contracted_bit_mismatches"] == 0
    assert not any(result["qualification"].values())
    assert all(row["mathematical"] == row["captured"] == 10.0 for row in result["rows"])
    assert all(row["absolute_error"] == 0 for row in result["rows"])


@pytest.mark.parametrize("value,expected", [
    (Fraction(0), 0.0), (Fraction(1), 1.0), (Fraction(-1), -1.0),
    (Fraction(1) + Fraction(1, 1 << 24), 1.0),
    (Fraction(1) + Fraction(3, 1 << 24), 1.0 + 2**-22),
    (Fraction(1, 1 << 150), 0.0),
    (Fraction(3, 1 << 150), 2**-148),
    (Fraction(1, 1 << 126) - Fraction(1, 1 << 150), 2**-126),
    (Fraction(1 << 127), float(1 << 127)),
])
def test_rne_rounding(value, expected):
    assert audit.round32(value) == expected


def test_negative_subnormal_underflow_preserves_sign():
    assert struct.pack("<f", audit.round32(-Fraction(1, 1 << 150))) == bytes.fromhex("00000080")


def test_overflow_refuses():
    with pytest.raises(ValueError, match="finite binary32"):
        audit.round32(Fraction(1 << 128))


def test_non_rational_rounding_refuses():
    with pytest.raises(ValueError, match="exact rational"):
        audit.round32(1.0)


def test_random_exact_representable_dyadics_match_struct():
    rng = random.Random(211)
    for _ in range(400):
        value = Fraction(rng.randrange(-2**40, 2**40), 2**rng.randrange(0, 120))
        expected = struct.unpack("<f", struct.pack("<f", float(value)))[0]
        assert struct.pack("<f", audit.round32(value)) == struct.pack("<f", expected)


def test_fma_cancellation_remains_a_distinct_hypothesis():
    pad = (0.0,) * 18
    vectors = ((-1.0, 1.0 + 2**-23) + pad, (0.0,) * 20,
               (1.0, 1.0 - 2**-23) + pad, (0.0,) * 20)
    exact, separate, contracted, bound = audit.gauss_world(vectors)
    assert exact == Fraction(-1, 1 << 47)
    assert separate == 0.0
    assert contracted == -2**-47
    assert abs(Fraction(separate) - exact) <= bound


def test_subnormal_error_floor():
    pad = (0.0,) * 19
    exact, separate, _, bound = audit.gauss_world(((2**-149,) + pad, (0.0,) * 20,
                                                  (0.5,) + pad, (0.0,) * 20))
    assert exact == Fraction(1, 1 << 151)
    assert separate == 0.0 and bound > abs(exact)


def test_large_output_error_is_not_accepted():
    payload = fields()
    payload["context.gauss"] = struct.pack("<64f", 11.0, *([10.0] * 63))
    result = audit.analyze(payload)
    assert result["decision"] == "retained-gauss-arithmetic-inconsistent"
    assert result["outside_bound"] == 1
    assert not any(result["qualification"].values())


@pytest.mark.parametrize("name", (*audit.NAMES, "context.gauss", "context.done"))
@pytest.mark.parametrize("mutation", ("missing", "extra_byte", "mutable"))
def test_closed_byte_schema(name, mutation):
    payload = fields()
    if mutation == "missing":
        del payload[name]
    elif mutation == "extra_byte":
        payload[name] += b"\0"
    else:
        payload[name] = bytearray(payload[name])
    with pytest.raises(ValueError):
        audit.analyze(payload)


def test_extra_field_refuses():
    payload = fields()
    payload["context.cost"] = bytes(256)
    with pytest.raises(ValueError, match="closed Gauss"):
        audit.analyze(payload)


@pytest.mark.parametrize("value", (1, 2, 255))
def test_done_boundary_not_generalized(value):
    payload = fields()
    payload["context.done"] = bytes([value]) + bytes(63)
    with pytest.raises(ValueError, match="all-active"):
        audit.analyze(payload)


@pytest.mark.parametrize("name", (*audit.NAMES, "context.gauss"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_refuses(name, value):
    payload = fields()
    payload[name] = struct.pack("<f", value) + payload[name][4:]
    with pytest.raises(ValueError, match="finite decoded"):
        audit.analyze(payload)


def authentication_fixture(monkeypatch):
    from mjlab_microduck import stance_solver_replay_probe as replay
    from mjlab_microduck import stance_solver_replay_receiver as receiver
    from mjlab_microduck import stance_solver_scratch as scratch
    from mjlab_microduck import stance_solver_target_binding as target
    root, solver = Path("/synthetic"), Path("/solver.py")
    raw = b"fake-packet-for-auth-order-only"
    meta = json.dumps({"arms": {"original": {"solver_init": {"snapshot": {}, "recipe": {}}}}}).encode()
    inv = json.dumps({"child.json": {"bytes": len(meta), "sha256": sha256(meta).hexdigest()},
                      replay.RAW_NAME: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}}).encode()
    anchors = dict(inventory_sha256=sha256(inv).hexdigest(), metadata_sha256=sha256(meta).hexdigest(),
                   raw_sha256=sha256(raw).hexdigest(), raw_bytes=len(raw))
    blobs = {solver: b"pinned-solver", root / replay.INVENTORY: inv,
             root / replay.HISTORY / "child.json": meta,
             root / replay.HISTORY / replay.RAW_NAME: raw}
    events = []
    monkeypatch.setattr(replay, "INPUT", anchors)
    monkeypatch.setattr(replay, "read_plain", lambda path, *args: blobs[path])
    monkeypatch.setattr(target, "verify_source", lambda raw: events.append("source"))
    parse = receiver._json
    monkeypatch.setattr(receiver, "_json", lambda raw, label: (events.append("parse"), parse(raw, label))[1])

    def decode(raw, snapshot, digest, recipe, elliptic_type):
        assert digest == anchors["raw_sha256"] and elliptic_type == 7
        events.append("decode")
        return SimpleNamespace(fields=fields())

    monkeypatch.setattr(scratch, "decode_packet", decode)
    return root, solver, blobs, events, replay


@pytest.mark.parametrize("which", ("inventory", "metadata", "packet"))
def test_authenticate_all_anchors_before_json(monkeypatch, which):
    root, solver, blobs, events, replay = authentication_fixture(monkeypatch)
    path = {"inventory": root / replay.INVENTORY,
            "metadata": root / replay.HISTORY / "child.json",
            "packet": root / replay.HISTORY / replay.RAW_NAME}[which]
    blobs[path] += b"x"
    with pytest.raises(ValueError, match="whole historical anchors"):
        audit.audit_retained(root, solver)
    assert events == ["source"]


def test_successful_auth_order_and_boundaries(monkeypatch):
    root, solver, _, events, _ = authentication_fixture(monkeypatch)
    result = audit.audit_retained(root, solver)
    assert events == ["source", "parse", "parse", "decode"]
    assert result["boundary"]["active_elliptic_rows"] == 0
    assert result["boundary"]["contact_friction_restored"] is False
    assert result["boundary"]["total_cost_checked"] is False
    assert result["boundary"]["gpu_execution_checked"] is False


def test_frozen_source_refuses_before_reading_history(monkeypatch):
    root, solver, _, events, _ = authentication_fixture(monkeypatch)
    from mjlab_microduck import stance_solver_target_binding as target

    def refuse(_):
        raise ValueError("whole frozen solver source hash")

    monkeypatch.setattr(target, "verify_source", refuse)
    with pytest.raises(ValueError, match="frozen solver source"):
        audit.audit_retained(root, solver)
    assert events == []


def test_import_is_inert():
    code = """
import sys
from mjlab_microduck import stance_solver_gauss_audit
assert not any(n.split('.')[0] in {'warp', 'torch', 'mujoco', 'mujoco_warp', 'numpy'} for n in sys.modules)
"""
    subprocess.run((sys.executable, "-c", code), check=True, timeout=15,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
