"""Authenticated row-address comparison for a retained BAM load run.

This is a bounded postprocessor. It preserves the original receiver decision and
does not infer runtime cause, waive repeatability, or authorize training.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np

from mjlab_microduck import stance_bam_load_receiver as receiver

PROTOCOL = "microduck-bam-row-address-audit-oct7-v1"
SOURCE = "c3dba2abb556b33db753bbbaed6bb6b00353b856"
TREE = "d580ee7535ecb1f5fdcc7455b447bd56c88ef612"
LEAVES_SHA256 = "a0ea80e4cbd997b8997bb9c2bb1819d71003ca4d7237db918abdc27d6c2a57bd"
RECEIVER_SHA256 = "8d371cb9b2da640aaf7c446fe955e2a421c99b145adabada538f952d64c2f9e9"
OUTPUT_CAP = 2 * 1024**2
FRICTION_DOF = 1
CASE_ORDER = ("serial0", "serial1")
RAW_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/evaluations/bam-load-run-c3dba2abb556"
)
CLOSEOUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/tools/bam-load-run-closeout-c3dba2abb556"
)
CLOSEOUT_CAPS = {
    "inventory.json": {
        "bytes": 5712,
        "sha256": "80305358c7fe4d46fb9a6cb4d2f835bc8368045563b21069394405350c6577c4",
    },
    "terminal.json": {
        "bytes": 390,
        "sha256": "36370523ae5b16d85fbc9a07a02c425f12ec6eb48e386fcf1720f18dd37e3c75",
    },
    "receiver.json": {"bytes": 228221, "sha256": RECEIVER_SHA256},
    "mac-receiver.json": {"bytes": 228221, "sha256": RECEIVER_SHA256},
    "verification.json": {
        "bytes": 1788,
        "sha256": "a2f45ac042d0e4bd60c6f2c644eab88ecdd26d1ce78a6a22c36443b3801160f5",
    },
}
FALSE_FLAGS = {name: False for name in receiver.FLAGS}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _read_anchored_files(root, anchors, *, total_cap):
    root = Path(root)
    _need(root.is_dir() and not root.is_symlink(), "real artifact directory")
    _need(
        type(anchors) is dict
        and {entry.name for entry in root.iterdir()} == set(anchors),
        "exact anchored artifact inventory",
    )
    result = {}
    total = 0
    for name in sorted(anchors):
        anchor = anchors[name]
        _need(
            type(anchor) is dict
            and set(anchor) == {"bytes", "sha256"}
            and type(anchor["bytes"]) is int
            and 0 < anchor["bytes"] <= total_cap
            and type(anchor["sha256"]) is str
            and len(anchor["sha256"]) == 64,
            "literal bounded whole-file anchor " + name,
        )
        descriptor = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            _need(
                stat.S_ISREG(before.st_mode) and before.st_size == anchor["bytes"],
                "regular anchored file " + name,
            )
            chunks = []
            length = 0
            while length <= anchor["bytes"]:
                chunk = os.read(descriptor, min(65536, anchor["bytes"] + 1 - length))
                if not chunk:
                    break
                chunks.append(chunk)
                length += len(chunk)
            after = os.fstat(descriptor)
            _need(
                length == anchor["bytes"]
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
                "stable whole-file read " + name,
            )
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        _need(
            sha256(raw).hexdigest() == anchor["sha256"],
            "external whole-file SHA-256 " + name,
        )
        result[name] = raw
        total += length
        _need(total <= total_cap, "bounded authenticated inventory")
    return result


def _json(raw, label):
    def pairs(items):
        value = {}
        for key, child in items:
            _need(key not in value, "duplicate JSON key " + label)
            value[key] = child
        return value

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("non-finite JSON constant " + label)
        ),
    )


def _validate_fields(fields, label):
    _need(type(fields) is dict and set(fields) == set(receiver.LOAD_FIELDS), label)
    for name, (dtype, shape) in receiver.LOAD_FIELDS.items():
        value = fields[name]
        expected_shape = (receiver.SUBSTEPS, receiver.WORLDS, *shape)
        _need(
            isinstance(value, np.ndarray)
            and value.shape == expected_shape
            and value.dtype == np.dtype(dtype),
            "literal complete load field " + label + ":" + name,
        )
        if np.dtype(dtype).kind == "f":
            _need(
                bool(np.isfinite(value).all()),
                "finite load field " + label + ":" + name,
            )
    counts = fields["nefc"]
    _need(
        bool(((counts >= 0) & (counts <= 512)).all()),
        "bounded complete constraint counts " + label,
    )


def _row_packets(fields, label):
    _validate_fields(fields, label)
    kinds = fields["efc_type"]
    ids = fields["efc_id"]
    force_bits = fields["efc_force"].view(np.dtype("<u4"))
    counts = fields["nefc"]
    packets = []
    for step in range(receiver.SUBSTEPS):
        worlds = []
        for world in range(receiver.WORLDS):
            active_count = int(counts[step, world])
            rows = []
            for row_index in range(active_count):
                if int(kinds[step, world, row_index]) != FRICTION_DOF:
                    continue
                dof = int(ids[step, world, row_index])
                _need(0 <= dof < 20, "active friction row DOF address")
                rows.append([row_index, dof, int(force_bits[step, world, row_index])])
            worlds.append(rows)
        packets.append(worlds)
    return packets


def _cell_decision(left, right):
    left_addresses = [row[1] for row in left]
    right_addresses = [row[1] for row in right]
    left_force_by_address = sorted((row[1], row[2]) for row in left)
    right_force_by_address = sorted((row[1], row[2]) for row in right)
    left_positions = [row[0] for row in left]
    right_positions = [row[0] for row in right]
    left_row_addresses = [(row[0], row[1]) for row in left]
    right_row_addresses = [(row[0], row[1]) for row in right]
    left_address_counts = Counter(left_addresses)
    right_address_counts = Counter(right_addresses)
    same_addresses = sorted(left_addresses) == sorted(right_addresses)
    same_addressed_records = left_force_by_address == right_force_by_address
    return {
        "active_rows": {"serial0": len(left), "serial1": len(right)},
        "active_counts_equal": len(left) == len(right),
        "duplicate_address_rows": {
            "serial0": sum(value - 1 for value in left_address_counts.values()),
            "serial1": sum(value - 1 for value in right_address_counts.values()),
        },
        "row_positions": {"serial0": left_positions, "serial1": right_positions},
        "rows": {"serial0": left, "serial1": right},
        "address_multiset_equal": same_addresses,
        "addressed_force_bits_equal": same_addressed_records,
        # A changed address set is not a matched-DOF force-value comparison.
        "force_bit_change_with_equal_address_multisets": (
            not same_addressed_records if same_addresses else None
        ),
        "row_address_association_equal": left_row_addresses == right_row_addresses,
        "row_force_sequence_equal": left == right,
        "row_ticket_address_permutation_only": (
            left_positions == right_positions
            and left_force_by_address == right_force_by_address
            and left_row_addresses != right_row_addresses
        ),
        "row_placement_changed": left_positions != right_positions,
        "address_multiset_changed": sorted(left_addresses) != sorted(right_addresses),
        "addressed_force_bit_change": left_force_by_address != right_force_by_address,
        "zero_force_rows": {
            case: {
                "positive": sum(
                    (row[2] & 0x7FFFFFFF) == 0 and (row[2] >> 31) == 0
                    for row in selected
                ),
                "negative": sum(
                    (row[2] & 0x7FFFFFFF) == 0 and (row[2] >> 31) == 1
                    for row in selected
                ),
            }
            for case, selected in (("serial0", left), ("serial1", right))
        },
    }


def audit_decoded_fields(fields_by_case):
    """Compare all active friction rows without changing or normalizing raw banks."""
    _need(
        type(fields_by_case) is dict and set(fields_by_case) == set(CASE_ORDER),
        "exact paired load cases",
    )
    rows = {case: _row_packets(fields_by_case[case], case) for case in CASE_ORDER}
    per_world = []
    counts = Counter()
    first = None
    for step in range(receiver.SUBSTEPS):
        for world in range(receiver.WORLDS):
            item = _cell_decision(
                rows["serial0"][step][world], rows["serial1"][step][world]
            )
            item.update(substep=step, world=world)
            per_world.append(item)
            for name in (
                "row_ticket_address_permutation_only",
                "row_placement_changed",
                "address_multiset_changed",
                "addressed_force_bit_change",
                "active_counts_equal",
                "row_address_association_equal",
                "row_force_sequence_equal",
            ):
                counts[name] += int(bool(item[name]))
            if not item["active_counts_equal"]:
                counts["active_count_mismatches"] += 1
            if first is None and not item["row_force_sequence_equal"]:
                first = {"substep": step, "world": world}
    first_step = None if first is None else first["substep"]
    first_worlds = (
        []
        if first_step is None
        else [
            row["world"]
            for row in per_world
            if row["substep"] == first_step and not row["row_force_sequence_equal"]
        ]
    )

    def first_cell(field, expected=True):
        return next(
            (
                {"substep": row["substep"], "world": row["world"]}
                for row in per_world
                if row[field] is expected
            ),
            None,
        )

    def worlds_at_first(field, expected=True):
        cell = first_cell(field, expected)
        return (
            []
            if cell is None
            else [
                row["world"]
                for row in per_world
                if row["substep"] == cell["substep"] and row[field] is expected
            ]
        )

    exact_addressed_multisets = all(
        row["addressed_force_bits_equal"] for row in per_world
    )
    exact_raw_row_sequence = all(row["row_force_sequence_equal"] for row in per_world)
    result = {
        "protocol": PROTOCOL,
        "declared_predecessor_source": SOURCE,
        "declared_predecessor_decision": "fresh-bam-load-repeat-negative",
        "authenticated_closeout": False,
        "raw_provenance_enforced_by_this_pure_view": False,
        "row_address_decision": "all-addressed-force-multisets-exact"
        if exact_addressed_multisets
        else "addressed-force-multiset-negative",
        "all_addressed_force_multisets_exact": exact_addressed_multisets,
        "all_row_sequences_exact": exact_raw_row_sequence,
        "first_affected": first,
        "first_affected_substep": first_step,
        "first_affected_worlds": first_worlds,
        "first_ticket_address_difference": first_cell(
            "row_address_association_equal", False
        ),
        "first_ticket_address_difference_worlds": worlds_at_first(
            "row_address_association_equal", False
        ),
        "first_address_multiset_change": first_cell("address_multiset_changed"),
        "first_address_multiset_change_worlds": worlds_at_first(
            "address_multiset_changed"
        ),
        "first_addressed_force_bit_change": first_cell("addressed_force_bit_change"),
        "first_addressed_force_bit_change_worlds": worlds_at_first(
            "addressed_force_bit_change"
        ),
        "summary": dict(counts),
        "per_world": per_world,
        "runtime_cause_proven": False,
        "training_authorized": False,
        "physical_acceptance": False,
        "full_window_qualified": False,
        "flags": dict(FALSE_FLAGS),
    }
    canonical = receiver.old.canonical(result)
    _need(len(canonical) <= OUTPUT_CAP, "complete row audit within two-MiB report cap")
    return result


def analyze(closeout_root=CLOSEOUT_ROOT, raw_root=RAW_ROOT):
    """Authenticate predecessor closeout and all raw files before row decoding."""
    closeout = _read_anchored_files(
        CLOSEOUT_ROOT if closeout_root is None else closeout_root,
        CLOSEOUT_CAPS,
        total_cap=512 * 1024,
    )
    # No closeout JSON is parsed until every closeout member has matched its
    # literal external length and hash anchor.
    inventory = _json(closeout["inventory.json"], "inventory")
    terminal = _json(closeout["terminal.json"], "terminal")
    native_receiver = _json(closeout["receiver.json"], "native receiver")
    mac_receiver = _json(closeout["mac-receiver.json"], "Mac receiver")
    verification = _json(closeout["verification.json"], "verification")
    _need(
        type(inventory) is dict
        and set(inventory) == set(receiver.CAPS)
        and native_receiver == mac_receiver
        and receiver.old.canonical(native_receiver) == closeout["receiver.json"]
        and closeout["receiver.json"] == closeout["mac-receiver.json"]
        and native_receiver["decision"] == "fresh-bam-load-repeat-negative"
        and verification["source_binding"]
        == {
            "source": SOURCE,
            "branch": "feat/athletics-obstacle-curriculum",
            "tree": TREE,
            "leaf_count": 673,
            "leaves_sha256": LEAVES_SHA256,
        }
        and verification["files"]
        == {
            name: CLOSEOUT_CAPS[name]
            for name in ("inventory.json", "receiver.json", "terminal.json")
        },
        "authenticated native/Mac negative closeout and exact source binding",
    )
    raw = receiver.read_inventory(raw_root, inventory)
    _need(
        type(terminal) is dict
        and sha256(closeout["terminal.json"]).hexdigest()
        == CLOSEOUT_CAPS["terminal.json"]["sha256"],
        "authenticated terminal bytes",
    )
    verified = receiver.verify_directory(
        raw_root,
        inventory,
        closeout["terminal.json"],
        CLOSEOUT_CAPS["terminal.json"]["sha256"],
        source=SOURCE,
        expected_tree=TREE,
        expected_leaves_sha256=LEAVES_SHA256,
    )
    verified_bytes = receiver.old.canonical(verified)
    _need(
        verified_bytes == closeout["receiver.json"] == closeout["mac-receiver.json"]
        and verified["decision"] == "fresh-bam-load-repeat-negative",
        "independent receiver exactly reproduces both negative receipts",
    )
    decoded = {case: receiver.decode_load_packets(raw, case) for case in CASE_ORDER}
    result = audit_decoded_fields(decoded)
    result.update(
        source=SOURCE,
        source_binding=verification["source_binding"],
        original_receiver_decision=verified["decision"],
        authenticated_closeout=True,
        closeout_files=CLOSEOUT_CAPS,
        raw_files=inventory,
        receiver_sha256=RECEIVER_SHA256,
        closeout_inventory_sha256=CLOSEOUT_CAPS["inventory.json"]["sha256"],
        raw_artifact_count=len(raw),
        raw_artifact_total_bytes=sum(map(len, raw.values())),
        source_tree=TREE,
        leaves_sha256=LEAVES_SHA256,
        paired_loads_exact=verified["load_boundary_analysis"][
            "paired_observed_loads_exact"
        ],
        input_commit_consistency_exact=verified["load_boundary_analysis"][
            "actual_input_commit_consistency_exact"
        ],
        reference_decision=verified["load_boundary_analysis"]["reference_decision"],
        preserved_original_decision=verified["decision"],
        original_flags=dict(verified["flags"]),
    )
    _need(
        verified["decision"] == "fresh-bam-load-repeat-negative"
        and verified["paired_decision"] == "one-tick-paired-negative"
        and verified["execution_decision"] == "one-nominal-tick-load-consistent"
        and all(value is False for value in verified["flags"].values())
        and all(value is False for value in result["flags"].values()),
        "preserve negative paired gate and every permanently false flag",
    )
    _need(len(receiver.old.canonical(result)) <= OUTPUT_CAP, "complete bounded report")
    return result


def _write_exclusive(path, payload):
    _need(
        type(payload) is bytes and 0 < len(payload) <= OUTPUT_CAP,
        "bounded complete audit payload before file creation",
    )
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _need(written > 0, "progressing exclusive artifact write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = analyze()
    payload = receiver.old.canonical(result)
    if args.output:
        _write_exclusive(args.output, payload)
    else:
        sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    main()
