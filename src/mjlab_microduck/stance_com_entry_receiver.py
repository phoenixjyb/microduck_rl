"""Independent whole-byte CoM receiver; never original replay admission."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

PROTOCOL = "microduck-com-actual-entry-receiver-v1"
DATA_PROTOCOL = "microduck-com-actual-entry-probe-v1"
INPUT_BYTES, BANK_BYTES = 12288, 393216
PARENTS = (0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14)
GROUPS = ((6, 15), (5, 10, 14), (4, 9, 13), (3, 8, 12), (2,), (7,), (11,), (1,), (0,))
SIBLING_ORDERS = (
    (2, 7, 11),
    (2, 11, 7),
    (7, 2, 11),
    (7, 11, 2),
    (11, 2, 7),
    (11, 7, 2),
)
CORE_FILES = {
    "declaration.json",
    "report.json",
    "child.json",
    "child.log",
    "entry0.bin",
    "entry1.bin",
}
CAPS = {
    "entry0.bin": INPUT_BYTES,
    "entry1.bin": INPUT_BYTES,
    "concurrent.bin": BANK_BYTES,
    "serial.bin": BANK_BYTES,
    "child.json": 32 * 1024,
    "declaration.json": 128 * 1024,
    "report.json": 256 * 1024,
    "child.log": 1024 * 1024,
}
FLAGS = {
    name: False
    for name in (
        "original_run_entry_captured",
        "original_pair_accepted",
        "actual_kernel_order_observed",
        "runtime_cause_proven",
        "full_window_qualified",
        "training_authorized",
        "physical_acceptance",
    )
}
SERVICE_CAPS = {
    "Restart": "no",
    "RuntimeMaxUSec": "5min",
    "TimeoutStopUSec": "10s",
    "MemoryMax": "6442450944",
    "CPUQuotaPerSecUSec": "2s",
    "Nice": "10",
    "KillMode": "control-group",
    "RemainAfterExit": "yes",
    "LimitFSIZE": "1048576",
    "MemoryAccounting": "yes",
    "NRestarts": "0",
}
VERSIONS = {
    "torch": "2.9.1",
    "warp-lang": "1.12.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.8.1",
    "mjlab": "1.3.0",
    "better-actuator-models": "1.0.1",
}
MACHINE = "7d6778c98cb345788b8c1a410f19ad35"
GPU = "GPU-7d72b360-33bc-2cee-3ff4-a954474011b5"


def false_flags(value):
    return (
        type(value) is dict
        and value == FLAGS
        and all(v is False for v in value.values())
    )


def exact_int(value, expected):
    return type(value) is int and value == expected


def valid_host(value, child_pid, *, idle=False):
    need(
        type(value) is dict
        and value["machine"] == MACHINE
        and value["gpu"] == GPU
        and value["driver"] == "595.95",
        "native monitor identity",
    )
    need(value["driver_model"] == "WDDM", "exact native driver model")
    for key in ("temperature_c", "used_mib", "free_mib"):
        need(type(value[key]) is int and value[key] >= 0, "typed GPU monitor values")
    need(value["temperature_c"] < 75, "bounded GPU temperature")
    rows = value["processes"]
    need(type(rows) is list and len(rows) <= 1, "single GPU child")
    for row in rows:
        need(
            type(row) is dict
            and exact_int(row["pid"], child_pid)
            and (
                (
                    row["memory_status"] == "reported"
                    and type(row["memory_mib"]) is int
                    and 0 < row["memory_mib"] <= 24162
                    and type(row["raw_memory"]) is str
                    and re.fullmatch(r"[0-9]+", row["raw_memory"])
                    and int(row["raw_memory"]) == row["memory_mib"]
                )
                or (
                    row["memory_status"] == "unavailable-wddm"
                    and row["memory_mib"] is None
                    and row["raw_memory"] in ("N/A", "[N/A]")
                )
            ),
            "actual GPU child monitor row",
        )
    if idle:
        need(not rows and value["used_mib"] <= 1024, "declared idle GPU")
    return bool(rows)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("nonfinite JSON constant")
        ),
    )


def _sha(raw, digest):
    need(
        type(raw) is bytes
        and type(digest) is str
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and sha256(raw).hexdigest() == digest,
        "whole external SHA-256 before decode",
    )


def read_inventory(root, anchors):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "regular artifact root")
    need(
        type(anchors) is dict
        and set(anchors) in (CORE_FILES, CORE_FILES | {"concurrent.bin", "serial.bin"}),
        "exact six/eight-file external inventory",
    )
    need(
        {p.name for p in root.iterdir()} == set(anchors),
        "no extra or missing artifact file",
    )
    raw_files = {}
    for name, anchor in anchors.items():
        need(
            type(anchor) is dict and set(anchor) == {"sha256", "bytes"},
            "typed external file anchor",
        )
        need(
            type(anchor["bytes"]) is int
            and 0 <= anchor["bytes"] <= CAPS[name]
            and (anchor["bytes"] > 0 or name == "child.log"),
            "bounded exact anchor length",
        )
        fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(
                stat.S_ISREG(before.st_mode) and before.st_size == anchor["bytes"],
                "regular exact-length file",
            )
            raw = b""
            while len(raw) <= CAPS[name]:
                chunk = os.read(fd, min(65536, CAPS[name] + 1 - len(raw)))
                if not chunk:
                    break
                raw += chunk
            after = os.fstat(fd)
            need(
                len(raw) == anchor["bytes"]
                and (
                    before.st_ino,
                    before.st_dev,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                == (
                    after.st_ino,
                    after.st_dev,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ),
                "stable complete file read",
            )
        finally:
            os.close(fd)
        _sha(raw, anchor["sha256"])
        raw_files[name] = raw
    return raw_files


def analyze_bank(entry, bank):
    need(
        type(entry) is bytes
        and len(entry) == INPUT_BYTES
        and type(bank) is bytes
        and len(bank) == BANK_BYTES,
        "complete fixed entry and bank dimensions",
    )
    expected = np.frombuffer(entry, dtype="<f4").reshape(64, 16, 3).copy()
    outputs = np.frombuffer(bank, dtype="<f4").reshape(32, 64, 16, 3)
    need(
        np.isfinite(expected).all() and np.isfinite(outputs).all(),
        "finite full input/output bank",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        for group in GROUPS:
            for body in group:
                if body != 0:
                    parent = PARENTS[body]
                    expected[:, parent, :] = np.float32(
                        expected[:, parent, :] + expected[:, body, :]
                    )
                    need(
                        np.isfinite(expected[:, parent, :]).all(),
                        "finite intermediate independent recurrence",
                    )
    bits, predicted = outputs.view("<u4"), expected.view("<u4")
    mismatch = bits != predicted[None]
    varying = np.any(bits != bits[0], axis=0)
    numeric = outputs.astype(np.float64)
    delta = np.abs(numeric - expected.astype(np.float64)[None])
    rows = []
    for label, ids in (
        ("world_body_0", [0]),
        ("branch_root_1", [1]),
        ("other_bodies", list(range(2, 16))),
    ):
        rows.append(
            {
                "label": label,
                "mismatched_scalars": int(mismatch[:, :, ids].sum()),
                "varying_cells": int(varying[:, ids].sum()),
                "max_abs_delta": float(delta[:, :, ids].max()),
            }
        )
    return {
        "scalars_compared": int(outputs.size),
        "mismatched_scalars": int(mismatch.sum()),
        "mismatched_repeats": int(np.any(mismatch, axis=(1, 2, 3)).sum()),
        "varying_cells": int(varying.sum()),
        "unique_full_snapshots": len(
            {bank[i * INPUT_BYTES : (i + 1) * INPUT_BYTES] for i in range(32)}
        ),
        "max_abs_delta": float(delta.max()),
        "max_pairwise_repeat_delta": float(
            (numeric.max(axis=0) - numeric.min(axis=0)).max()
        ),
        "body_groups": rows,
        "reference_sha256": sha256(expected.tobytes()).hexdigest(),
    }


def verify(root, source, anchors, terminal_raw, terminal_sha256):
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "independent receiver must hide CUDA",
    )
    need(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source),
        "exact source SHA",
    )
    # Authenticate ALL files and the independent terminal packet before any JSON/array decode.
    raw = read_inventory(root, anchors)
    need(
        type(terminal_raw) is bytes and 0 < len(terminal_raw) <= 64 * 1024,
        "bounded terminal packet",
    )
    _sha(terminal_raw, terminal_sha256)
    log_lines = raw["child.log"].decode("utf-8").splitlines()
    checked_log = "\n".join(
        line
        for line in log_lines
        if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
    )
    need(
        not re.search(
            r"\b(?:nan|nonfinite|overflow|warning|traceback)\b|CUDA error",
            checked_log,
            re.I,
        ),
        "independent finite warning-free owned log",
    )
    declaration, report, child, terminal = (
        _json(raw["declaration.json"]),
        _json(raw["report.json"]),
        _json(raw["child.json"]),
        _json(terminal_raw),
    )
    for packet, suffix in (
        (declaration, ":declaration"),
        (report, ":report"),
        (child, ":child"),
    ):
        need(
            packet["protocol"] == DATA_PROTOCOL + suffix and packet["source"] == source,
            "exact isolated source and protocol",
        )
        need(
            false_flags(packet["flags"]),
            "no admission flags",
        )
    need(
        child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest(),
        "whole declaration binding",
    )
    need(
        report["child_sha256"] == sha256(raw["child.json"]).hexdigest(),
        "whole child binding",
    )
    need(
        declaration["source_binding"]
        == child["source_binding"]
        == report["source_binding"],
        "identical source bindings",
    )
    binding = declaration["source_binding"]
    need(
        type(binding) is dict
        and set(binding) == {"source", "branch", "tree", "leaf_count", "leaves_sha256"}
        and binding["source"] == source
        and binding["branch"] == "feat/athletics-obstacle-curriculum"
        and type(binding["tree"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", binding["tree"])
        and type(binding["leaves_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", binding["leaves_sha256"])
        and type(binding["leaf_count"]) is int
        and 600 < binding["leaf_count"] < 2000,
        "typed exact committed source binding",
    )
    packages = declaration["packages"]
    need(
        packages["versions"] == VERSIONS
        and packages["python"] == "3.12.13"
        and packages["architecture"] == "x86_64",
        "frozen native package binding",
    )
    trees = packages["python_trees"]
    need(
        set(trees) == {"mjlab", "mujoco-warp", "better-actuator-models"}
        and all(
            type(v["files"]) is int
            and 5 < v["files"] < 10000
            and type(v["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", v["sha256"])
            for v in trees.values()
        ),
        "whole installed Python tree binding",
    )
    need(
        child["compiled_descriptor"]["selected_fields_sha256"]
        == "6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f"
        and exact_int(child["runtime_ntendon"], 0),
        "native compiled model binding",
    )
    need(
        terminal["source"] == source
        and terminal["unit"] == "microduck-com-entry-run-" + source[:12] + ".service",
        "exact terminal source and unit",
    )
    live, end = declaration["service_properties"], terminal["service_properties"]
    test_end = declaration["tests_terminal_properties"]
    test_caps = {
        **SERVICE_CAPS,
        "LimitFSIZE": "67108864",
        "MainPID": "0",
        "Result": "success",
        "ExecMainStatus": "0",
        "ActiveState": "active",
        "SubState": "exited",
    }
    need(
        all(test_end.get(key) == value for key, value in test_caps.items())
        and type(test_end["InvocationID"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", test_end["InvocationID"])
        and test_end["InvocationID"] != live["InvocationID"]
        and type(declaration["tests_receipt_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", declaration["tests_receipt_sha256"]),
        "separate completed CPU prerequisite and whole receipt anchor",
    )
    need(
        all(live.get(k) == v and end.get(k) == v for k, v in SERVICE_CAPS.items()),
        "actual live/terminal caps",
    )
    need(
        re.fullmatch(r"[0-9a-f]{32}", live["InvocationID"])
        and live["InvocationID"] == end["InvocationID"],
        "same actual invocation",
    )
    need(
        end["MainPID"] == "0"
        and live["ActiveState"] == "active"
        and live["SubState"] == "running"
        and end["ActiveState"] == "active"
        and end["SubState"] == "exited"
        and end["Result"] == "success"
        and end["ExecMainStatus"] == "0",
        "successful completed native owner",
    )
    need(
        type(child["owner_pid"]) is int
        and child["owner_pid"] > 0
        and str(child["owner_pid"]) == live["MainPID"]
        and exact_int(child["child_ppid"], child["owner_pid"])
        and type(child["child_pid"]) is int
        and child["child_pid"] > 0,
        "actual parent/child PID binding",
    )
    need(
        exact_int(report["returncode"], 0)
        and exact_int(report["observed_child_pid"], child["child_pid"])
        and exact_int(report["observed_child_ppid"], child["owner_pid"])
        and report["gpu_child_observed"] is True,
        "owner-observed child and GPU process",
    )
    valid_host(declaration["host"], child["child_pid"], idle=True)
    monitor = report["monitor"]
    need(type(monitor) is list and 0 < len(monitor) <= 480, "bounded actual monitor")
    seen, previous = False, -1.0
    for row in monitor:
        elapsed = row["elapsed"]
        need(
            type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and previous < elapsed < 240
            and exact_int(row["child_pid"], child["child_pid"]),
            "ordered actual child observation",
        )
        previous = elapsed
        seen |= valid_host(row["host"], child["child_pid"])
    need(seen, "independently checked actual GPU occupancy")
    need(
        exact_int(child["integration_calls"], 0)
        and exact_int(child["physics_steps"], 0)
        and child["graph_created"] is False
        and child["actor_model_created"] is False
        and child["optimizer_created"] is False
        and child["storage_created"] is False
        and child["fixed_state_unchanged"] is True,
        "no integration, policy, optimizer, storage or graph",
    )
    scope = child["capture_receipt"]
    need(
        scope["protocol"] == "microduck-com-actual-entry-capture-v1"
        and scope["status"] == "complete"
        and scope["fault_type"] is None
        and scope["parents"] == list(PARENTS)
        and scope["reversed_levels"]
        == [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2, 7, 11], [1], [0]]
        and false_flags(scope["flags"])
        and len(scope["calls"]) == 2,
        "two complete original CoM call-site records",
    )
    for index, row in enumerate(scope["calls"]):
        need(
            exact_int(row["call_index"], index)
            and exact_int(row["worlds"], 64)
            and row["device"] == "cuda:0"
            and type(row["stream"]) is int
            and row["stream"] >= 0
            and exact_int(row["bytes"], INPUT_BYTES)
            and row["sha256"] == sha256(raw[f"entry{index}.bin"]).hexdigest()
            and exact_int(row["original_launches"], 11)
            and row["boundary"] == "after-init-before-first-accumulation"
            and row["initialization_and_accumulation_output_alias"] is True
            and row["readback_and_device_sync_perturb_timing"] is True,
            "literal fresh entry boundary and alias records",
        )
        layout = row["subtree_com_layout"]
        need(
            layout == scope["calls"][0]["subtree_com_layout"]
            and type(layout["object_id"]) is int
            and layout["object_id"] > 0
            and type(layout["ptr"]) is int
            and layout["ptr"] > 0
            and layout["shape"] == [64, 16]
            and layout["strides"] == [192, 12]
            and layout["dtype"] == "vec3f"
            and layout["contiguous"] is True
            and row["stream"] == scope["calls"][0]["stream"]
            and row["smooth_sha256"]
            == "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
            and row["forward_sha256"]
            == "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
            "same actual array layout, alias, source and stream",
        )
    inputs = [np.frombuffer(raw[f"entry{i}.bin"], dtype="<f4") for i in (0, 1)]
    need(
        all(len(x) == 3072 and np.isfinite(x).all() for x in inputs),
        "whole finite CoM entries",
    )
    different = int(np.count_nonzero(inputs[0].view("<u4") != inputs[1].view("<u4")))
    analyses = {}
    if different:
        need(
            set(raw) == CORE_FILES and child["decision"] == "fresh-entry-inputs-differ",
            "input negative stops repeat banks",
        )
        decision = "fresh-entry-inputs-differ"
    else:
        need(
            set(raw) == CORE_FILES | {"concurrent.bin", "serial.bin"}
            and child["decision"] == "fixed-input-banks-retained"
            and child["repeat_rng_unchanged"] is True,
            "matching input and full repeat banks",
        )
        for mode, launches in (("concurrent", 224), ("serial", 288)):
            record = child["repeat_receipts"][mode]
            need(
                record["mode"] == mode
                and exact_int(record["repeats"], 32)
                and exact_int(record["resets"], 32)
                and exact_int(record["accumulation_launches"], launches)
                and exact_int(record["bytes"], BANK_BYTES)
                and record["input_sha256"] == sha256(raw["entry0.bin"]).hexdigest()
                and record["output_sha256"] == sha256(raw[mode + ".bin"]).hexdigest()
                and record["output_boundary"] == "after-accumulation-before-division"
                and type(record["stream"]) is int
                and record["stream"] == scope["calls"][0]["stream"]
                and type(record["scratch_ptr"]) is int
                and record["scratch_ptr"] > 0
                and record["scratch_shape"] == [64, 16]
                and record["scratch_strides"] == [192, 12]
                and record["input_output_alias"] is True
                and false_flags(record["flags"]),
                "whole aliased reset and original-kernel repeat contract",
            )
            analyses[mode] = analyze_bank(raw["entry0.bin"], raw[mode + ".bin"])
        decision = (
            "isolated-serial-reference-exact"
            if analyses["serial"]["mismatched_scalars"] == 0
            else "isolated-serial-reference-negative"
        )
    return {
        "protocol": PROTOCOL,
        "source": source,
        "input_mismatched_scalars": different,
        "decision": decision,
        "decision_is_original_or_training_admission": False,
        "analyses": analyses,
        "external_anchors": anchors,
        "terminal_sha256": terminal_sha256,
        "instrumentation_changes_timing": True,
        "flags": dict(FLAGS),
    }


def analyze_order_hypotheses(entry, bank):
    """All-cell bit hypotheses, not an observation of atomic arrival order."""
    need(
        type(entry) is bytes
        and len(entry) == INPUT_BYTES
        and type(bank) is bytes
        and len(bank) == BANK_BYTES,
        "complete hypothesis input and bank dimensions",
    )
    initial = np.frombuffer(entry, dtype="<f4").reshape(64, 16, 3)
    outputs = np.frombuffer(bank, dtype="<f4").reshape(32, 64, 16, 3)
    need(
        np.isfinite(initial).all() and np.isfinite(outputs).all(),
        "finite complete hypothesis input and bank",
    )
    candidates = []
    with np.errstate(over="ignore", invalid="ignore"):
        for order in SIBLING_ORDERS:
            expected = initial.copy()
            # Four unchanged groups, then only the three sibling writers vary.
            groups = GROUPS[:4] + tuple((body,) for body in order) + ((1,), (0,))
            for group in groups:
                for body in group:
                    if body:
                        parent = PARENTS[body]
                        expected[:, parent] = np.float32(
                            expected[:, parent] + expected[:, body]
                        )
                        need(
                            np.isfinite(expected[:, parent]).all(),
                            "finite intermediate hypothesis recurrence",
                        )
            candidates.append(expected)
    predicted = np.stack(candidates).view("<u4")
    bits = outputs.view("<u4")
    matches = bits[None] == predicted[:, None]
    reference = predicted[0]
    affected = np.any(bits != reference[None], axis=0)
    rows = []
    for world, body, axis in np.argwhere(affected):
        actual = bits[:, world, body, axis]
        values, counts = np.unique(actual, return_counts=True)
        wanted = int(reference[world, body, axis])

        def ordered(value):
            value = int(value)
            return (~value & 0xFFFFFFFF) if value & 0x80000000 else value ^ 0x80000000

        rows.append(
            {
                "world": int(world),
                "body": int(body),
                "axis": int(axis),
                "initial_bits": f"{int(initial.view('<u4')[world, body, axis]):08x}",
                "reference_bits": f"{wanted:08x}",
                "candidate_bits": [
                    f"{int(value):08x}" for value in predicted[:, world, body, axis]
                ],
                "variants": [
                    {
                        "bits": f"{int(value):08x}",
                        "count": int(count),
                        "compatible_candidate_indices": [
                            int(index)
                            for index in np.flatnonzero(
                                predicted[:, world, body, axis] == value
                            )
                        ],
                    }
                    for value, count in zip(values, counts)
                ],
                "mismatched_repeats": int(np.sum(actual != wanted)),
                "varying": len(values) > 1,
                "max_ordered_bit_distance": max(
                    abs(ordered(value) - ordered(wanted)) for value in values
                ),
            }
        )
    return {
        "scalars_compared": int(bits.size),
        "cells_compared": int(reference.size),
        "candidate_orders": [list(order) for order in SIBLING_ORDERS],
        "candidate_snapshot_sha256": [
            sha256(value.tobytes()).hexdigest() for value in candidates
        ],
        "repeat_compatible_uniform_orders": [
            [
                int(index)
                for index in np.flatnonzero(matches[:, repeat].all(axis=(1, 2, 3)))
            ]
            for repeat in range(32)
        ],
        "unexplained_scalars": int(np.sum(~matches.any(axis=0))),
        "affected_cells": rows,
        "atomic_order_observed": False,
        "compatible_bits_are_causal_or_training_admission": False,
    }


def explain_order_hypotheses(root, source, anchors, terminal_raw, terminal_sha256):
    """Authenticate the complete retained run before separate CPU hypotheses."""
    retained = verify(root, source, anchors, terminal_raw, terminal_sha256)
    need(retained["input_mismatched_scalars"] == 0, "same complete actual-entry inputs")
    raw = read_inventory(root, anchors)
    result = {
        "protocol": PROTOCOL + ":order-hypotheses",
        "source": source,
        "retained_receiver_sha256": sha256(canonical(retained)).hexdigest(),
        "retained_receiver_decision": retained["decision"],
        "terminal_sha256": terminal_sha256,
        "external_anchors": anchors,
        "interpretation": "Compatible float32 hypotheses, not observed atomic order.",
        "analyses": {
            mode: analyze_order_hypotheses(raw["entry0.bin"], raw[mode + ".bin"])
            for mode in ("concurrent", "serial")
        },
        "flags": dict(FLAGS),
    }
    need(
        len(canonical(result)) <= 16 * 1024**2, "bounded complete CPU hypothesis report"
    )
    return result
