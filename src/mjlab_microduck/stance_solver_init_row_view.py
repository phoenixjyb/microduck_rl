"""Pure, non-admitting side view of the retained initialized solver packet.

No physics, owner hooks, canonicalized inputs or GPU imports. Historical
protocols are unchanged. Only the exact retained capture is accepted by the CLI.
"""

from hashlib import sha1, sha256
from math import prod
import os
from pathlib import Path
import re
import stat
import subprocess

from mjlab_microduck import stance_solver_init_control as control
from mjlab_microduck import stance_solver_init_receiver as receiver

PROTOCOL = "microduck-initialized-row-side-view-oct8-v1"
BASE = "4011fdc5e277c06e3bd4975707f5edfcd0073feb"
CAPTURE_SOURCE = "e45a59c412bfe6bcf2de751f98159aa7fe0db463"
BRANCH = "feat/athletics-obstacle-curriculum"
ROOTS = (
    Path("/Users/yanbo/Projects/microduckPlayground/microduck_rl"),
    Path("/home/yanbo/work/microduck_rl-com-entry-20261006"),
)
OWN = {
    "src/mjlab_microduck/stance_solver_init_row_view.py",
    "tests/test_stance_solver_init_row_view.py",
    "docs/experiments/2026-10-08-initialized-row-side-view.md",
}
INVENTORY_SHA = "8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625"
RECEIVER_SHA = "c91d812df1278d1cfe208951d8dad6f23f5240639989d56407a6ca38ed08f999"
TESTS_SHA = "7df590e1a90769cc2326bfe784da60895c7c5169da3347e8ae3ff76b4168d05b"
RAW_FILES, RAW_BYTES = 480, 673600544
ROW_FIELDS = ("context.Jaref", "efc.force", "efc.state")
DIRECT_FIELDS = (
    "data.qpos",
    "data.qvel",
    "data.qM",
    "data.qacc",
    "data.qacc_warmstart",
    "data.qacc_smooth",
    "data.qfrc_smooth",
    "efc.Ma",
    "data.qfrc_constraint",
    "context.cost",
    "context.gauss",
    "context.prev_cost",
    "context.grad",
    "context.grad_dot",
    "context.Mgrad",
    "context.h",
)
FLAGS = dict(receiver.FLAGS)
need = receiver.need


def checked_fields(fields):
    """Literal full packet ABI; no numeric conversion or tolerance."""
    need(
        type(fields) is dict and set(fields) == set(control.SOLVER_INIT_ORDER),
        "complete initialized field set",
    )
    for name, (_, shape, dtype) in control.SOLVER_INIT_SPECS.items():
        field = fields[name]
        width = 1 if dtype == "bool" else 4
        wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        need(
            type(field) is dict and set(field) == {"shape", "dtype", "raw"},
            "literal field schema: " + name,
        )
        need(
            type(field["shape"]) is list
            and all(type(x) is int for x in field["shape"])
            and field["shape"] == list(shape)
            and field["dtype"] == wire
            and type(field["raw"]) is bytes
            and len(field["raw"]) == prod(shape) * width,
            "literal field layout and bytes: " + name,
        )
    need(
        all(x in (0, 1) for x in fields["context.done"]["raw"]),
        "literal boolean done bytes",
    )


def unpack_initialized(snapshot, raw):
    """Decode a previously receiver-validated snapshot with exact packed ABI."""
    need(
        snapshot["phase"] == "initialized-before-search"
        and type(snapshot["forward"]) is int
        and snapshot["forward"] == 4,
        "exact initialized forward-four phase",
    )
    path = receiver.safe_name(snapshot["path"])
    packet = raw[path]
    need(
        type(snapshot["bytes"]) is int
        and snapshot["bytes"] == len(packet) <= control.MAX_PACKET_BYTES
        and snapshot["sha256"] == sha256(packet).hexdigest(),
        "whole initialized packet anchor",
    )
    need(
        set(snapshot["fields"]) == set(control.SOLVER_INIT_ORDER),
        "complete packed field set",
    )
    fields, offset = {}, 0
    for name in control.SOLVER_INIT_ORDER:
        item = snapshot["fields"][name]
        _, shape, dtype = control.SOLVER_INIT_SPECS[name]
        width = 1 if dtype == "bool" else 4
        size = prod(shape) * width
        wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        need(
            type(item) is dict
            and set(item) == {"offset", "bytes", "shape", "dtype"}
            and type(item["offset"]) is int
            and item["offset"] == offset
            and type(item["bytes"]) is int
            and item["bytes"] == size
            and type(item["shape"]) is list
            and all(type(x) is int for x in item["shape"])
            and item["shape"] == list(shape)
            and item["dtype"] == wire,
            "exact contiguous packed slice: " + name,
        )
        fields[name] = {
            "shape": list(shape),
            "dtype": wire,
            "raw": packet[offset : offset + size],
        }
        offset += size
    need(offset == len(packet), "no unclaimed packet bytes")
    checked_fields(fields)
    return fields


def _total():
    return {
        "compared_words": 0,
        "differing_words": 0,
        "exact": None,
        "first_difference": None,
    }


def _compare(total, left, right, left_offset, right_offset, size, identity):
    a = left[left_offset : left_offset + size]
    b = right[right_offset : right_offset + size]
    need(len(a) == len(b) == size and size % 4 == 0, "whole word slices")
    indices = [i for i in range(0, size, 4) if a[i : i + 4] != b[i : i + 4]]
    total["compared_words"] += size // 4
    total["differing_words"] += len(indices)
    if indices and total["first_difference"] is None:
        i = indices[0]
        total["first_difference"] = {
            **identity,
            "component": i // 4,
            "left_byte_offset": left_offset + i,
            "right_byte_offset": right_offset + i,
            "left_word_le_hex": a[i : i + 4].hex(),
            "right_word_le_hex": b[i : i + 4].hex(),
        }
    if total["compared_words"]:
        total["exact"] = total["differing_words"] == 0


def compare_initialized(left, right):
    """Active row storage links plus separately declared literal direct fields.

    Only unique captured payloads and authenticated address/id/type backlinks
    pair contacts. Noncontacts match identical markers at identical offsets.
    Done worlds are excluded, even though this retained run initializes false.
    """
    for fields in (left, right):
        checked_fields(fields)
    contacts = [
        {name: fields[name] for name in receiver.CONTACT_PACKET_ORDER}
        for fields in (left, right)
    ]
    complete = [
        {name: fields[name] for name in receiver.COMPLETE_ORDER}
        for fields in (left, right)
    ]
    linked = receiver.compare_linked_contact_construction(
        contacts[0], complete[0], contacts[1], complete[1]
    )
    counts = [receiver._int_words(x["data.nefc"]) for x in (left, right)]
    need(
        counts[0] == counts[1] and all(0 <= x <= 512 for x in counts[0]),
        "same bounded active row extents",
    )
    active = {
        (world, row) for world, count in enumerate(counts[0]) for row in range(count)
    }
    covered = [set(), set()]
    done = [x["context.done"]["raw"] for x in (left, right)]
    excluded_done = [world for world in range(64) if done[0][world] or done[1][world]]
    totals = {
        kind: {name: _total() for name in ROW_FIELDS}
        for kind in ("contact", "noncontact")
    }

    def pair(kind, world, a, b, identity):
        need(
            (world, a) in active and (world, b) in active,
            "paired active initialized rows",
        )
        if done[0][world] or done[1][world]:
            return
        need(
            (world, a) not in covered[0] and (world, b) not in covered[1],
            "unique disjoint row coverage",
        )
        covered[0].add((world, a))
        covered[1].add((world, b))
        for name in ROW_FIELDS:
            _compare(
                totals[kind][name],
                left[name]["raw"],
                right[name]["raw"],
                (world * 512 + a) * 4,
                (world * 512 + b) * 4,
                4,
                {**identity, "world": world, "left_row": a, "right_row": b},
            )

    for link in linked["compared_payload_links"]:
        need(
            len(link["left_rows"]) == len(link["right_rows"]), "equal linked row count"
        )
        for ordinal, (a, b) in enumerate(zip(link["left_rows"], link["right_rows"])):
            pair(
                "contact",
                link["world"],
                a,
                b,
                {
                    "left_contact_index": link["left_index"],
                    "right_contact_index": link["right_index"],
                    "local_row_ordinal": ordinal,
                },
            )
    types = [receiver._int_words(x["efc.type"]) for x in (left, right)]
    unpaired = []
    for world, count in enumerate(counts[0]):
        for row in range(count):
            i = world * 512 + row
            a, b = types[0][i], types[1][i]
            if a in (5, 6) and b in (5, 6):
                continue
            same = (
                a in range(5)
                and b in range(5)
                and all(
                    left[name]["raw"][4 * i : 4 * i + 4]
                    == right[name]["raw"][4 * i : 4 * i + 4]
                    for name in ("efc.id", "efc.type")
                )
            )
            if same:
                pair("noncontact", world, row, row, {"type": a})
            else:
                unpaired.append({"world": world, "row": row})
    exclusions = {
        name: len(linked[name])
        for name in (
            "unobserved_payload_links",
            "ambiguous_payload_groups",
            "unmatched_payload_groups",
        )
    }
    complete_coverage = all(x == active for x in covered) and not any(
        exclusions.values()
    )
    direct = {}
    for name in DIRECT_FIELDS:
        direct[name] = _total()
        _compare(
            direct[name],
            left[name]["raw"],
            right[name]["raw"],
            0,
            0,
            len(left[name]["raw"]),
            {"shape": left[name]["shape"]},
        )
    return {
        "protocol": PROTOCOL,
        "phase": "initialized-before-search",
        "forward": 4,
        "solver_source_sha256": control.SOLVER_SOURCE_SHA256,
        "row_fields": list(ROW_FIELDS),
        "row_comparisons": totals,
        "active_rows_per_arm": len(active),
        "covered_rows_per_arm": [len(x) for x in covered],
        "uncovered_rows_per_arm": [len(active - x) for x in covered],
        "full_active_row_coverage": complete_coverage if active else None,
        "payload_link_exclusions": exclusions,
        "unpaired_noncontact_offset_count": len(unpaired),
        "unpaired_noncontact_offset_samples": unpaired[:128],
        "excluded_done_worlds": excluded_done,
        "direct_full_carrier_comparisons": direct,
        "direct_scope": "original-offset whole carriers including inactive/padding; not fresh-write validity",
        "uncaptured_contact_parameters": linked["uncaptured_contact_parameters"],
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": "active initialized storage rows, not physical contact identity, complete driver equality, cause or admission; zero comparisons are null",
    }


def _git(root, *args, input=None):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], input=input, timeout=30
    )


def git_binding(root, source, *, worktree):
    """Reconstruct whole historical Git blobs; optionally bind current files."""
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "exact SHA")
    entries = []
    for entry in _git(root, "ls-tree", "-r", "-z", "--full-tree", source).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = path.decode()
        receiver.safe_name(name)
        need(mode in ("100644", "100755") and kind == "blob", "plain Git blob")
        entries.append((name, oid))
    need(0 < len(entries) <= 2000, "bounded source leaf count")
    output = _git(
        root,
        "cat-file",
        "--batch",
        input=b"".join(oid.encode() + b"\n" for _, oid in entries),
    )
    need(len(output) <= 64 * 1024**2, "bounded source blob bytes")
    leaves, offset = {}, 0
    for name, oid in entries:
        end = output.index(b"\n", offset)
        head_oid, kind, size_raw = output[offset:end].decode().split()
        size = int(size_raw)
        need(
            head_oid == oid and kind == "blob" and 0 <= size <= 16 * 1024**2,
            "exact source blob header",
        )
        raw = output[end + 1 : end + 1 + size]
        offset = end + 1 + size
        need(
            output[offset : offset + 1] == b"\n"
            and len(raw) == size
            and sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest() == oid,
            "whole Git blob binding",
        )
        offset += 1
        if worktree:
            path = root / name
            need(
                path.resolve(strict=True) == path
                and not path.is_symlink()
                and stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
                and path.read_bytes() == raw,
                "whole current committed leaf",
            )
        leaves[name] = {
            "bytes": size,
            "git_blob": oid,
            "sha256": sha256(raw).hexdigest(),
        }
    need(offset == len(output), "exact batch source set")
    return {
        "source": source,
        "tree": _git(root, "rev-parse", source + "^{tree}").decode().strip(),
        "branch": BRANCH,
        "leaves": leaves,
    }


def analysis_binding(root, source):
    need(
        root in ROOTS
        and Path.cwd().resolve() == root
        and Path(__file__).resolve().parents[2] == root,
        "exact analysis worktree",
    )
    need(
        _git(root, "rev-parse", "HEAD").decode().strip() == source
        and _git(root, "branch", "--show-current").decode().strip() == BRANCH
        and not _git(root, "status", "--porcelain"),
        "exact clean analysis source",
    )
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE, source],
        check=True,
        timeout=15,
    )
    need(
        set(_git(root, "diff", "--name-only", BASE, source).decode().splitlines())
        == OWN,
        "exact three-path analysis fence",
    )
    return git_binding(root, source, worktree=True)


def anchored(path, cap, expected):
    """Stable regular whole file, authenticated before JSON decoding."""
    path = Path(path).absolute()
    need(path.resolve(strict=True) == path, "plain anchored path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(
            stat.S_ISREG(before.st_mode) and before.st_size <= cap,
            "bounded regular anchor",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(cap + 1)
        after = os.fstat(fd)
        need(
            receiver.component.stat_identity(before)
            == receiver.component.stat_identity(after)
            == receiver.component.stat_identity(path.stat(follow_symlinks=False))
            and len(raw) == before.st_size <= cap
            and sha256(raw).hexdigest() == expected,
            "stable whole external anchor",
        )
        return raw
    finally:
        os.close(fd)


def analyze_retained(root, source):
    """Recompute full historical receiver before interpreting the new side view."""
    binding = analysis_binding(root, source)
    short = CAPTURE_SOURCE[:12]
    closeout = root / "artifacts/tools" / ("solver-init-tick-closeout-" + short)
    raw_root = root / "artifacts/evaluations" / ("solver-init-tick-run-" + short)
    inventory_bytes = anchored(closeout / "inventory.json", 128 * 1024, INVENTORY_SHA)
    inventory = receiver.decode(inventory_bytes)
    need(
        len(inventory) == RAW_FILES
        and sum(x["bytes"] for x in inventory.values()) == RAW_BYTES,
        "exact retained complete leaf set and bytes",
    )
    capture_binding = git_binding(root, CAPTURE_SOURCE, worktree=False)
    native_report = anchored(closeout / "receiver.json", 8 * 1024**2, RECEIVER_SHA)
    recomputed = receiver.verify_run(
        raw_root,
        inventory,
        expected_source=capture_binding,
        expected_tests_sha=TESTS_SHA,
        historical_root=root,
    )
    need(
        receiver.canonical(recomputed) == native_report,
        "whole independently recomputed receiver",
    )
    # Authenticate again rather than trust an earlier mutable filesystem read.
    raw = receiver.authenticate(raw_root, inventory)
    child = receiver.json_packet(raw["child.json"])
    fields = {
        arm: unpack_initialized(child["arms"][arm]["solver_init"]["snapshot"], raw)
        for arm in ("candidate0", "candidate1")
    }
    view = compare_initialized(fields["candidate0"], fields["candidate1"])
    need(binding == analysis_binding(root, source), "analysis source unchanged")
    return {
        "protocol": PROTOCOL,
        "analysis_source_binding": binding,
        "capture_source": CAPTURE_SOURCE,
        "raw_inventory_sha256": INVENTORY_SHA,
        "native_receiver_sha256": RECEIVER_SHA,
        "whole_receiver_recomputed": True,
        "raw_files": RAW_FILES,
        "raw_bytes": RAW_BYTES,
        "initialized_rows": view,
        "flags": dict(FLAGS),
        "decision": "initialized-row-side-view-only",
    }


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not sys.flags.optimize,
        "CUDA-hidden unoptimized pure replay",
    )
    result = analyze_retained(Path.cwd().resolve(), args.source)
    print(receiver.canonical(result).decode())


if __name__ == "__main__":
    main()
