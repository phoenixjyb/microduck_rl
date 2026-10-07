"""First-observed divergence analysis for a retained one-tick serial proof.

This is a bounded read-only byte analysis. It does not identify runtime cause,
reexecute physics, waive numerical differences, or admit training or motion.
"""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

from mjlab_microduck import stance_com_coupled_frames as frames
from mjlab_microduck import stance_serial_step_receiver as receiver

PROTOCOL = "microduck-serial-one-step-divergence-oct7-v1"
SOURCE = "26e2ee8244b7bd05fb3290d655113e538bd1a989"
TREE = "e3f919145e145281b49168d352b964523a3297bb"
LEAVES_SHA256 = "a0677ebf4facdc92cebb255faed309c1b6101b651dd95ace9306e8244d14bd50"
CASE_ORDER = ("serial0", "serial1")
CALLS = 21
SUBSTEPS = 10
WORLDS = 64
SAMPLE_LIMIT = 4
OUTPUT_CAP = 2 * 1024**2

CLOSEOUT_CAPS = {
    "inventory.json": 2500,
    "terminal.json": 390,
    "verification.json": 1801,
    "receiver.json": 61159,
    "mac-receiver.json": 61159,
}
CLOSEOUT_ANCHORS = {
    "inventory.json": {
        "bytes": 2500,
        "sha256": "60138c8381f91fb2c31541c51fc8e5dcfe2eaa1a6a3b46c8b59715403c3dce2c",
    },
    "terminal.json": {
        "bytes": 390,
        "sha256": "232b94280d51a431c4027db38b97d911d303f7d0d9bea117a1182f0af094c721",
    },
    "verification.json": {
        "bytes": 1801,
        "sha256": "046293cc74747690f041da7f9aac5912a83c1a8127bf90cc67ca969384e14c7e",
    },
    "receiver.json": {
        "bytes": 61159,
        "sha256": "d8c38bf63780f40aaa0fc40492bd5a43b7e40c3290195ec59555b781727eee71",
    },
    "mac-receiver.json": {
        "bytes": 61159,
        "sha256": "d8c38bf63780f40aaa0fc40492bd5a43b7e40c3290195ec59555b781727eee71",
    },
}
TERMINAL_SHA256 = CLOSEOUT_ANCHORS["terminal.json"]["sha256"]
FLAGS = {
    "original_run_entry_captured": False,
    "original_pair_accepted": False,
    "actual_kernel_order_observed": False,
    "runtime_cause_proven": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
}

MOTOR_FIELDS = receiver.MOTOR_FIELDS
MOTOR_BYTES = receiver.MOTOR_BYTES
MASK_BYTES = receiver.MASK_BYTES
RNE_BYTES = receiver.RNE_BYTES
COM_BYTES = receiver.COM_BYTES
LEDGER_FIELDS = (
    "reward",
    "upright",
    "stillness",
    "height",
    "support",
    "motor",
    "joint_speed",
    "correction",
    "correction_change",
)
MASK_FIELDS = ("before_steps", "live", "accepted", "rejected")
STREAMS = (
    "com.entries",
    "com.weighted",
    "rne.entries",
    "rne.outputs",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_U32 = np.dtype("<u4")
_F32 = np.dtype("<f4")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _read_regular(root, name, limit):
    path = Path(root) / name
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _need(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            "bounded regular file " + name,
        )
        parts = []
        total = 0
        while total <= limit:
            block = os.read(descriptor, min(65536, limit + 1 - total))
            if not block:
                break
            parts.append(block)
            total += len(block)
        after = os.fstat(descriptor)
        _need(
            total == before.st_size
            and (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "stable whole file " + name,
        )
    finally:
        os.close(descriptor)
    return b"".join(parts)


def _read_closeout(root, anchors):
    root = Path(root)
    _need(root.is_dir() and not root.is_symlink(), "regular closeout directory")
    _need(
        type(anchors) is dict
        and set(anchors) == set(CLOSEOUT_CAPS)
        and {p.name for p in root.iterdir()} == set(CLOSEOUT_CAPS),
        "exact five-file closeout inventory",
    )
    raw = {}
    # Authenticate every closeout byte string before decoding any JSON.
    for name in sorted(CLOSEOUT_CAPS):
        anchor = anchors[name]
        _need(
            type(anchor) is dict
            and set(anchor) == {"bytes", "sha256"}
            and type(anchor["bytes"]) is int
            and anchor["bytes"] == CLOSEOUT_CAPS[name]
            and type(anchor["sha256"]) is str
            and _SHA256.fullmatch(anchor["sha256"]) is not None,
            "external whole closeout anchor " + name,
        )
        value = _read_regular(root, name, CLOSEOUT_CAPS[name])
        _need(
            len(value) == anchor["bytes"]
            and sha256(value).hexdigest() == anchor["sha256"],
            "whole closeout SHA-256 " + name,
        )
        raw[name] = value
    return raw


def _ordered_bits(values):
    bits = values.view(_U32).astype(np.uint64)
    sign = bits & np.uint64(0x80000000)
    return np.where(
        sign != 0, np.bitwise_not(bits) & np.uint64(0xFFFFFFFF), bits ^ 0x80000000
    )


def _float_stats(left, right, *, label="field"):
    """Compare same-shaped finite f32 arrays with bounded coordinate examples."""
    _need(
        isinstance(left, np.ndarray)
        and isinstance(right, np.ndarray)
        and left.shape == right.shape
        and left.ndim >= 1
        and left.shape[0] == WORLDS
        and left.dtype == _F32
        and right.dtype == _F32,
        "literal complete float32 comparison " + label,
    )
    _need(
        bool(np.isfinite(left).all()) and bool(np.isfinite(right).all()),
        "finite complete float32 comparison " + label,
    )
    lbits, rbits = left.view(_U32), right.view(_U32)
    mismatch = lbits != rbits
    per_world = mismatch.reshape(WORLDS, -1).sum(axis=1)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    distance = np.abs(
        _ordered_bits(left).astype(np.int64) - _ordered_bits(right).astype(np.int64)
    )
    coords = np.argwhere(mismatch)[:SAMPLE_LIMIT]
    samples = [
        {
            "world": int(coord[0]),
            "index": [int(v) for v in coord[1:]],
            "left": float(left[tuple(coord)]),
            "right": float(right[tuple(coord)]),
            "left_u32": int(lbits[tuple(coord)]),
            "right_u32": int(rbits[tuple(coord)]),
            "ordered_bit_distance": int(distance[tuple(coord)]),
        }
        for coord in coords
    ]
    return {
        "left_sha256": sha256(left.tobytes(order="C")).hexdigest(),
        "right_sha256": sha256(right.tobytes(order="C")).hexdigest(),
        "scalars": int(left.size),
        "bit_mismatch_scalars": int(mismatch.sum()),
        "world_mismatch_scalars": [int(v) for v in per_world],
        "max_abs_delta": float(delta.max(initial=0.0)),
        "max_ordered_bit_distance": int(distance.max(initial=0)),
        "exact_raw_bits": not bool(mismatch.any()),
        "coordinate_samples": samples,
    }


def _mask_stats(left, right, *, label):
    _need(
        isinstance(left, np.ndarray)
        and isinstance(right, np.ndarray)
        and left.shape == right.shape == (WORLDS,)
        and left.dtype == right.dtype
        and left.dtype in (np.dtype("<i8"), np.dtype("u1")),
        "complete literal scalar mask comparison " + label,
    )
    mismatch = left != right
    indices = np.flatnonzero(mismatch)[:SAMPLE_LIMIT]
    left_raw, right_raw = left.tobytes(order="C"), right.tobytes(order="C")
    return {
        "left_sha256": sha256(left_raw).hexdigest(),
        "right_sha256": sha256(right_raw).hexdigest(),
        "scalars": WORLDS,
        "mismatch_scalars": int(mismatch.sum()),
        "world_mismatch_scalars": [int(v) for v in mismatch],
        "exact_raw_bits": left_raw == right_raw,
        "coordinate_samples": [
            {
                "world": int(index),
                "left": int(left[index]),
                "right": int(right[index]),
            }
            for index in indices
        ],
    }


def _float_packet(raw, per_snapshot_bytes, shape, count, label):
    _need(
        type(raw) is bytes and len(raw) == per_snapshot_bytes * count,
        "complete 21-snapshot bank " + label,
    )
    values = np.frombuffer(raw, dtype=_F32)
    _need(bool(np.isfinite(values).all()), "finite full snapshot bank " + label)
    return values.reshape((count, *shape))


def _motor_views(raw):
    _need(type(raw) is bytes and len(raw) == MOTOR_BYTES, "full ten-step motor packet")
    per_step = MOTOR_BYTES // SUBSTEPS
    rows = []
    for step in range(SUBSTEPS):
        offset = step * per_step
        fields = {}
        for name, shape in MOTOR_FIELDS.items():
            count = WORLDS * math.prod(shape)
            size = count * 4
            values = np.frombuffer(raw, dtype=_F32, count=count, offset=offset)
            _need(bool(np.isfinite(values).all()), "finite motor field " + name)
            fields[name] = values.reshape((WORLDS, *shape))
            offset += size
        _need(offset == (step + 1) * per_step, "exact motor field-major frame length")
        rows.append(fields)
    return rows


def _mask_views(raw):
    _need(type(raw) is bytes and len(raw) == MASK_BYTES, "full ten-step masks packet")
    rows = []
    offset = 0
    for _step in range(SUBSTEPS):
        steps = np.frombuffer(raw, dtype="<i8", count=WORLDS, offset=offset)
        offset += WORLDS * 8
        fields = {"before_steps": steps}
        for name in ("live", "accepted", "rejected"):
            fields[name] = np.frombuffer(
                raw, dtype=np.uint8, count=WORLDS, offset=offset
            )
            offset += WORLDS
        rows.append(fields)
    _need(offset == len(raw), "complete mask field-major packet")
    return rows


def _ledger_fields(child, case):
    case_data = child["cases"][case]
    ledger = case_data["ledger"]
    _need(
        type(ledger) is dict
        and set(ledger)
        == {
            "reward",
            "terminated",
            "timed_out",
            "episode_steps",
            "executed_steps",
            "live",
            "term_sums",
        }
        and type(ledger["term_sums"]) is dict
        and set(ledger["term_sums"]) == set(LEDGER_FIELDS[1:]),
        "complete nine-field one-tick reward ledger",
    )
    result = {}
    for name in LEDGER_FIELDS:
        values = ledger["reward"] if name == "reward" else ledger["term_sums"][name]
        _need(
            type(values) is list and len(values) == WORLDS,
            "complete reward field " + name,
        )
        array = np.asarray(values, dtype=_F32)
        _need(bool(np.isfinite(array).all()), "finite reward field " + name)
        result[name] = array
    return result


def _event(kind, index, name, stats, boundary=None):
    row = {"stream": kind, "index": index, "field": name, "stats": stats}
    if boundary is not None:
        row["boundary"] = boundary
    return row


def _first_event(events):
    return next(
        (
            {
                "stream": row["stream"],
                "index": row.get("index"),
                "field": row["field"],
                "boundary": row.get("boundary"),
                "mismatch_count": row["stats"].get(
                    "bit_mismatch_scalars", row["stats"].get("mismatch_scalars", 0)
                ),
                "mismatch_unit": "f32-scalars"
                if "bit_mismatch_scalars" in row["stats"]
                else "mask-scalars",
                "max_abs_delta": row["stats"].get("max_abs_delta"),
                "max_ordered_bit_distance": row["stats"].get(
                    "max_ordered_bit_distance"
                ),
                "coordinate_samples": row["stats"]["coordinate_samples"],
            }
            for row in events
            if row["stats"].get(
                "bit_mismatch_scalars", row["stats"].get("mismatch_scalars", 0)
            )
        ),
        None,
    )


def _analyze_authenticated(raw, child, source):
    events = []
    per_stream = {name: [] for name in (*STREAMS, "motor", "masks", "frames", "ledger")}

    def retain(row):
        per_stream[row["stream"]].append(row)
        return row

    forward_rows = [[] for _ in range(CALLS)]
    for name, size, shape in (
        ("com.entries", COM_BYTES, (WORLDS, 16, 3)),
        ("com.weighted", COM_BYTES, (WORLDS, 16, 3)),
        ("rne.entries", RNE_BYTES, (WORLDS, 16, 6)),
        ("rne.outputs", RNE_BYTES, (WORLDS, 16, 6)),
    ):
        banks = {
            case: _float_packet(raw[f"{case}.{name}.bin"], size, shape, CALLS, name)
            for case in CASE_ORDER
        }
        for call in range(CALLS):
            stats = _float_stats(
                banks["serial0"][call], banks["serial1"][call], label=name
            )
            phase = (
                "constructor" if call == 0 else "step-pre" if call % 2 else "step-post"
            )
            forward_rows[call].append(retain(_event(name, call, name, stats, phase)))

    motor_banks = {case: _motor_views(raw[f"{case}.motor.bin"]) for case in CASE_ORDER}
    mask_banks = {case: _mask_views(raw[f"{case}.masks.bin"]) for case in CASE_ORDER}
    motor_rows = [[] for _ in range(SUBSTEPS)]
    mask_rows = [[] for _ in range(SUBSTEPS)]
    for step in range(SUBSTEPS):
        for name in MOTOR_FIELDS:
            stats = _float_stats(
                motor_banks["serial0"][step][name],
                motor_banks["serial1"][step][name],
                label=name,
            )
            motor_rows[step].append(
                retain(_event("motor", step, name, stats, "before-step-pre-forward"))
            )
        for name in MASK_FIELDS:
            stats = _mask_stats(
                mask_banks["serial0"][step][name],
                mask_banks["serial1"][step][name],
                label=name,
            )
            row = {
                "stream": "masks",
                "index": step,
                "field": name,
                "stats": stats,
                "boundary": "before-step-pre-forward",
            }
            mask_rows[step].append(retain(row))

    frame_banks = {
        case: frames._decode_packet(raw[f"{case}.frames.bin"], case)
        for case in CASE_ORDER
    }
    frame_rows = [[], []]
    for boundary_index, boundary in enumerate(("constructor", "after-one-tick")):
        for name, _shape in frames.FRAME_FIELDS:
            stats = _float_stats(
                frame_banks["serial0"][boundary_index][name],
                frame_banks["serial1"][boundary_index][name],
                label=name,
            )
            frame_rows[boundary_index].append(
                retain(_event("frames", boundary_index, name, stats, boundary))
            )

    ledgers = {case: _ledger_fields(child, case) for case in CASE_ORDER}
    ledger_rows = []
    for name in LEDGER_FIELDS:
        stats = _float_stats(
            ledgers["serial0"][name], ledgers["serial1"][name], label=name
        )
        ledger_rows.append(retain(_event("ledger", 0, name, stats, "after-one-tick")))

    # Preserve observation order, not an inferred kernel-execution or causal order.
    events.extend(forward_rows[0])
    events.extend(frame_rows[0])
    for step in range(SUBSTEPS):
        events.extend(motor_rows[step])
        events.extend(mask_rows[step])
        events.extend(forward_rows[2 * step + 1])
        events.extend(forward_rows[2 * step + 2])
    events.extend(frame_rows[1])
    events.extend(ledger_rows)

    first_by_stream = {name: _first_event(rows) for name, rows in per_stream.items()}
    mismatch_counts = {
        name: sum(
            row["stats"].get(
                "bit_mismatch_scalars", row["stats"].get("mismatch_scalars", 0)
            )
            for row in rows
        )
        for name, rows in per_stream.items()
    }
    result = {
        "protocol": PROTOCOL,
        "source": source,
        "case_order": list(CASE_ORDER),
        "observed_event_order": "constructor call 0; then each motor proposal, step-pre forward, step-post forward; final frame and ledger after tick",
        "call_order": [
            "constructor",
            *[
                item
                for i in range(SUBSTEPS)
                for item in (f"motor-{i}", f"step-pre-{i}", f"step-post-{i}")
            ],
        ],
        "snapshot_channel_order_per_forward": list(STREAMS),
        "readback_timing_perturbs_execution": True,
        "per_stream_mismatch_scalars": mismatch_counts,
        "first_observed_divergence_by_stream": first_by_stream,
        "first_observed_divergence": _first_event(events),
        "comparisons": events,
        "physics_reexecution": False,
        "runtime_cause_proven": False,
        "uncaptured_solver_and_contact_state_excluded": True,
        "tolerance_applied": False,
        "flags": dict(FLAGS),
    }
    encoded_size = len(receiver.old.canonical(result))
    _need(encoded_size <= OUTPUT_CAP, "complete divergence report exceeds 2 MiB")
    result["serialized_bytes"] = encoded_size
    return result


def analyze_closeout(
    run_root,
    closeout_root,
    *,
    source=SOURCE,
    expected_tree=TREE,
    expected_leaves_sha256=LEAVES_SHA256,
    closeout_anchors=CLOSEOUT_ANCHORS,
):
    """Authenticate the full native closeout and run, then independently analyze."""
    _need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and type(expected_tree) is str
        and re.fullmatch(r"[0-9a-f]{40}", expected_tree)
        and type(expected_leaves_sha256) is str
        and _SHA256.fullmatch(expected_leaves_sha256),
        "external source/tree/leaf anchors",
    )
    closeout = _read_closeout(closeout_root, closeout_anchors)
    inventory = json.loads(closeout["inventory.json"])
    _need(
        type(inventory) is dict
        and set(inventory) == set(receiver.CAPS)
        and all(
            type(anchor) is dict
            and set(anchor) == {"bytes", "sha256"}
            and type(anchor["bytes"]) is int
            and 0 < anchor["bytes"] <= receiver.CAPS[name]
            and type(anchor["sha256"]) is str
            and _SHA256.fullmatch(anchor["sha256"])
            for name, anchor in inventory.items()
        ),
        "exact externally authenticated 22-file run inventory",
    )
    # This reads and authenticates every complete raw file, without decoding it.
    raw = receiver.read_inventory(run_root, inventory)
    terminal = closeout["terminal.json"]
    _need(sha256(terminal).hexdigest() == TERMINAL_SHA256, "external terminal SHA-256")
    report = json.loads(closeout["verification.json"])
    _need(
        type(report) is dict
        and report["protocol"]
        == receiver.DATA_PROTOCOL + ":native-independent-closeout"
        and report["decision"] == "fresh-one-step-serial-repeat-negative"
        and report["files"]
        == {
            name: {
                "bytes": len(closeout[name]),
                "sha256": sha256(closeout[name]).hexdigest(),
            }
            for name in ("inventory.json", "receiver.json", "terminal.json")
        }
        and report["source_binding"]
        == {
            "branch": "feat/athletics-obstacle-curriculum",
            "leaf_count": 663,
            "leaves_sha256": expected_leaves_sha256,
            "source": source,
            "tree": expected_tree,
        }
        and report["flags"] == FLAGS,
        "authenticated native closeout consistency and false admission flags",
    )
    native_receiver = closeout["receiver.json"]
    mac_receiver = closeout["mac-receiver.json"]
    _need(native_receiver == mac_receiver, "native and Mac receiver byte identity")
    verified = receiver.verify_directory(
        run_root,
        inventory,
        terminal,
        sha256(terminal).hexdigest(),
        source=source,
        expected_tree=expected_tree,
        expected_leaves_sha256=expected_leaves_sha256,
    )
    serialized = receiver.old.canonical(verified)
    _need(
        serialized == native_receiver,
        "independent current receiver does not reproduce closed receiver bytes",
    )
    _need(
        verified["flags"] == FLAGS
        and verified["paired_decision"] == "one-tick-paired-negative",
        "closed receiver retains literal false flags and paired negative",
    )
    child = json.loads(raw["child.json"])
    result = _analyze_authenticated(raw, child, source)
    _need(
        result["first_observed_divergence"] is not None
        and verified["decision"] == "fresh-one-step-serial-repeat-negative",
        "retained proof must preserve its negative numerical outcome",
    )
    result.update(
        closeout_sha256={
            name: sha256(value).hexdigest() for name, value in closeout.items()
        },
        closeout_receiver_sha256=sha256(native_receiver).hexdigest(),
        independently_reproduced_receiver_sha256=sha256(serialized).hexdigest(),
        raw_file_sha256={
            name: sha256(value).hexdigest() for name, value in raw.items()
        },
        paired_receiver_decision=verified["decision"],
    )
    # Include the count field itself without an off-by-a-few-digits receipt.
    result["serialized_bytes"] = 0
    for _ in range(10):
        size = len(receiver.old.canonical(result))
        if size == result["serialized_bytes"]:
            break
        result["serialized_bytes"] = size
    _need(
        len(receiver.old.canonical(result)) == result["serialized_bytes"],
        "stable complete serialized byte count",
    )
    _need(
        len(receiver.old.canonical(result)) <= OUTPUT_CAP,
        "complete closeout divergence report exceeds 2 MiB",
    )
    return result
