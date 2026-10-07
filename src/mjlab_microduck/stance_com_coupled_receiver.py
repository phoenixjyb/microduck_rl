"""Independent whole-byte receiver for fresh instrumented coupled CoM evidence."""

from hashlib import sha256
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

from mjlab_microduck import stance_com_coupled_frames as frames
from mjlab_microduck import stance_com_entry_receiver as old

PROTOCOL = "microduck-com-coupled-receiver-oct7-v1"
DATA_PROTOCOL = "microduck-com-coupled-probe-oct7-v1"
ENTRY_BYTES = 12288
EXPECTED_TESTS = 2019  # Exact collection of the 63 predeclared files.
TEST_FILES_SHA256 = "877c25c3becefada5c808855acae04356087ccb818a4c8e198f3aa2a18175752"
DESCRIPTOR_SHA256 = "91838baeac031299a008709e44cef879bdbc9bc769e84fcb2b318453ac115501"
RNG_KEYS = (
    "caller_cpu_before",
    "caller_cuda_before",
    "private_cpu_start",
    "private_cuda_start",
    "private_cpu_ctor_end",
    "private_cuda_ctor_end",
    "private_cpu_forward_end",
    "private_cuda_forward_end",
    "caller_cpu_after",
    "caller_cuda_after",
)
CPU_SEED_HASHES = {
    "caller_cpu_before": "ba8adae6f1ee70135e097a78de4f08bb885703e3eca406e93e9acf7aafaba8fa",
    "private_cpu_start": "27c3914a3d6d459fd6ba2c1b28bd54f22dd8bf164847973e7bbf3b1771b0316b",
}
CAPS = {
    f"{case}.{kind}.bin": limit
    for case in frames.CASES
    for kind, limit in (
        ("frames", frames.PACKET_BYTES),
        ("entries", 2 * ENTRY_BYTES),
        ("weighted", 2 * ENTRY_BYTES),
        ("rng", 128 * 1024),
    )
}
CAPS.update(
    {
        "declaration.json": 128 * 1024,
        "report.json": 256 * 1024,
        "child.json": 128 * 1024,
        "child.log": 1024**2,
    }
)
CAPS.update({"tests.receipt.json": 128 * 1024, "tests.terminal.json": 4096})
PACKAGE_TREES = {
    "better-actuator-models": {
        "files": 37,
        "sha256": "82150d75c58ec897c4d643728d7c1d264ff25480eef51ed379ef0bb1e9362b8b",
    },
    "mjlab": {
        "files": 189,
        "sha256": "89036723b778f9f9d95fa903f3af92b739cba73a2e7341f2397066c9ad79fcaa",
    },
    "mujoco-warp": {
        "files": 69,
        "sha256": "188d58bfac6ab54e51adfc80a6c504d782aec43856cbb15e54b4109f0ce30a1d",
    },
}


def read_inventory(root, anchors):
    root = Path(root)
    old.need(
        root.is_dir() and not root.is_symlink(), "regular fresh artifact directory"
    )
    old.need(
        type(anchors) is dict
        and set(anchors) == set(CAPS)
        and {p.name for p in root.iterdir()} == set(CAPS),
        "whole exact twenty-two-file inventory",
    )
    result = {}
    for name in sorted(CAPS):
        row = anchors[name]
        old.need(
            type(row) is dict
            and set(row) == {"bytes", "sha256"}
            and type(row["bytes"]) is int
            and 0 < row["bytes"] <= CAPS[name],
            "bounded external anchor",
        )
        fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            old.need(
                stat.S_ISREG(before.st_mode) and before.st_size == row["bytes"],
                "regular whole-length artifact",
            )
            chunks = []
            total = 0
            while total <= CAPS[name]:
                chunk = os.read(fd, min(65536, CAPS[name] + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            after = os.fstat(fd)
            old.need(
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
                "stable whole artifact",
            )
        finally:
            os.close(fd)
        raw = b"".join(chunks)
        old._sha(raw, row["sha256"])
        result[name] = raw
    old.need(sum(map(len, result.values())) < 16 * 1024**2, "whole evidence cap")
    return result


def checked_binding(value, source):
    old.need(
        type(value) is dict
        and set(value) == {"source", "branch", "tree", "leaf_count", "leaves_sha256"}
        and value["source"] == source
        and value["branch"] == "feat/athletics-obstacle-curriculum"
        and type(value["leaf_count"]) is int
        and value["leaf_count"] == 640
        and type(value["tree"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", value["tree"])
        and type(value["leaves_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["leaves_sha256"]),
        "fresh whole committed source binding",
    )


def checked_packages(value):
    old.need(
        type(value) is dict
        and value
        == dict(
            versions=old.VERSIONS,
            python="3.12.13",
            architecture="x86_64",
            python_trees=PACKAGE_TREES,
        ),
        "exact frozen native package trees",
    )


def checked_rng(raw, metadata):
    old.need(
        type(raw) is bytes
        and 0 < len(raw) <= 128 * 1024
        and type(metadata) is dict
        and set(metadata) == set(RNG_KEYS),
        "complete caller/private RNG packet",
    )
    offset = 0
    result = {}
    for name in RNG_KEYS:
        row = metadata[name]
        old.need(
            type(row) is dict
            and set(row) == {"bytes", "sha256"}
            and type(row["bytes"]) is int
            and 0 < row["bytes"] <= 16384,
            "typed bounded actual RNG state",
        )
        length = row["bytes"]
        if "cpu" in name:
            old.need(length == 5056, "exact CPU generator state length")
        state = raw[offset : offset + length]
        old._sha(state, row["sha256"])
        result[name] = state
        offset += length
    old.need(offset == len(raw), "no RNG prefix or trailing bytes")
    for name, expected in CPU_SEED_HASHES.items():
        old.need(
            sha256(result[name]).hexdigest() == expected,
            "literal caller/private CPU seed state",
        )
    for device in ("cpu", "cuda"):
        old.need(
            result[f"caller_{device}_before"] == result[f"caller_{device}_after"]
            and result[f"private_{device}_ctor_end"]
            == result[f"private_{device}_forward_end"],
            "preserved caller streams and no later-forward RNG draw",
        )
    old.need(
        len({len(result[name]) for name in RNG_KEYS if "cuda" in name}) == 1,
        "fixed private/caller CUDA RNG layouts",
    )
    return result


def weighted_comparison(entry_raw, weighted_raw):
    old.need(
        type(entry_raw) is bytes
        and type(weighted_raw) is bytes
        and len(entry_raw) == len(weighted_raw) == ENTRY_BYTES,
        "full actual initialized/weighted CoM arrays",
    )
    initial = np.frombuffer(entry_raw, dtype="<f4").reshape(64, 16, 3)
    actual = np.frombuffer(weighted_raw, dtype="<f4").reshape(64, 16, 3)
    old.need(
        np.isfinite(initial).all() and np.isfinite(actual).all(),
        "finite full CoM arrays",
    )
    expected = initial.copy()
    for level in ((6, 15), (5, 10, 14), (4, 9, 13), (3, 8, 12), (2, 7, 11), (1,), (0,)):
        for body in level:
            if body != 0:
                np.add(
                    expected[:, old.PARENTS[body]],
                    expected[:, body],
                    out=expected[:, old.PARENTS[body]],
                )
    mask = expected.view("<u4") != actual.view("<u4")
    counts = np.moveaxis(mask, 1, 0).reshape(16, -1).sum(1)
    return dict(
        expected_sha256=sha256(expected.tobytes()).hexdigest(),
        actual_sha256=sha256(weighted_raw).hexdigest(),
        scalars=3072,
        bit_mismatch_scalars=int(mask.sum()),
        exact_raw_bits=not bool(mask.any()),
        max_abs_delta=float(
            np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max()
        ),
        body_mismatch_scalars=dict(
            body0=int(counts[0]), body1=int(counts[1]), others=int(counts[2:].sum())
        ),
    )


def checked_scope(scope, mode, entries, weighted):
    old.need(
        type(scope) is dict
        and scope["protocol"]
        == "microduck-com-actual-entry-capture-v1:native-coupled-control"
        and scope["status"] == "complete"
        and scope["fault_type"] is None
        and scope["parents"] == list(old.PARENTS)
        and scope["reversed_levels"]
        == [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2, 7, 11], [1], [0]]
        and old.false_flags(scope["flags"])
        and scope["mode"] == mode
        and scope["dispatched_original_kernel_counts"]
        == [13 if mode == "serial" else 11] * 2
        and all(type(v) is int for v in scope["dispatched_original_kernel_counts"])
        and scope["split_body_ids"] == ([2, 7, 11] if mode == "serial" else [])
        and scope["complete_kernel_count_matches"] is True
        and scope["weighted_boundary"] == "after-accumulation-before-original-division"
        and scope["weighted_boundary_readback_perturbs_timing"] is True
        and scope["weighted_sha256"] == [sha256(raw).hexdigest() for raw in weighted]
        and type(scope["calls"]) is list
        and len(scope["calls"]) == 2,
        "exact native coupled CoM scope",
    )
    for i, row in enumerate(scope["calls"]):
        old.need(
            old.exact_int(row["call_index"], i)
            and old.exact_int(row["worlds"], 64)
            and old.exact_int(row["bytes"], ENTRY_BYTES)
            and old.exact_int(row["original_launches"], 11)
            and row["sha256"] == sha256(entries[i]).hexdigest()
            and row["device"] == "cuda:0"
            and type(row["stream"]) is int
            and row["stream"] >= 0
            and row["stream"] == scope["calls"][0]["stream"]
            and row["boundary"] == "after-init-before-first-accumulation"
            and row["initialization_and_accumulation_output_alias"] is True
            and row["readback_and_device_sync_perturb_timing"] is True
            and row["smooth_sha256"]
            == "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
            and row["forward_sha256"]
            == "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
            "literal actual initialized entry and source/stream",
        )
        layout = row["subtree_com_layout"]
        old.need(
            layout == scope["calls"][0]["subtree_com_layout"]
            and type(layout["object_id"]) is int
            and layout["object_id"] > 0
            and type(layout["ptr"]) is int
            and layout["ptr"] > 0
            and layout["shape"] == [64, 16]
            and layout["strides"] == [192, 12]
            and layout["dtype"] == "vec3f"
            and layout["contiguous"] is True,
            "whole actual alias/layout binding",
        )


def checked_crb(value):
    old.need(
        type(value) is dict
        and value["protocol"] == "football-b1d-crb-runtime-control-v1"
        and value["status"] == "complete"
        and value["mode"] == "serial"
        and value["fault_type"] is None
        and value["constructor_forward_covered"] is True
        and value["runtime_bound"] is True
        and value["initialization_timing_changed"] is True,
        "unchanged serial CRB scope",
    )
    for key, expected in dict(
        max_forward_calls=2,
        topology_id_snapshots=2,
        singleton_child_arrays_allocated=3,
        constructor_forward_calls=1,
        forward_calls=2,
        original_level_launch_requests=14,
        parent_zero_noop_level_requests=4,
        actual_accumulation_launches=18,
        split_child_launches=6,
        dense_qM_launches=2,
    ).items():
        old.need(old.exact_int(value[key], expected), "actual fixed CRB count " + key)
    old.need(
        value["smooth_source_sha256"]
        == "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
        and value["forward_source_sha256"]
        == "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3"
        and type(value["flags"]) is dict
        and set(value["flags"])
        == {
            "native_qualified",
            "original_pair_accepted",
            "cause_proven",
            "full_window_passed",
            "training_authorized",
            "physical_result_accepted",
        }
        and all(v is False for v in value["flags"].values()),
        "pinned CRB sources and false admission",
    )


def verify_directory(root, anchors, terminal_raw, terminal_sha256, *, source):
    old._sha(terminal_raw, terminal_sha256)
    raw = read_inventory(
        root, anchors
    )  # Authenticate every file before ANY JSON/array decode.
    terminal = old._json(terminal_raw)
    declaration, report, child = (
        old._json(raw[name])
        for name in ("declaration.json", "report.json", "child.json")
    )
    old.need(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source),
        "external exact source",
    )
    for value, suffix in (
        (declaration, ":declaration"),
        (report, ":report"),
        (child, ":child"),
    ):
        old.need(
            type(value) is dict
            and value["protocol"] == DATA_PROTOCOL + suffix
            and value["source"] == source
            and old.false_flags(value["flags"]),
            "fresh protocol/source/false flags",
        )
        checked_binding(value["source_binding"], source)
        old.need(
            value["source_binding"] == declaration["source_binding"],
            "one exact committed source",
        )
    checked_packages(declaration["packages"])
    old.need(
        child["packages"] == declaration["packages"],
        "same actual child package binding",
    )
    caps = old.SERVICE_CAPS
    live = declaration["service_properties"]
    old.need(
        all(live.get(k) == v and terminal.get(k) == v for k, v in caps.items())
        and live["ActiveState"] == "active"
        and live["SubState"] == "running"
        and re.fullmatch(r"[1-9][0-9]*", live["MainPID"])
        and re.fullmatch(r"[0-9a-f]{32}", live["InvocationID"])
        and terminal["InvocationID"] == live["InvocationID"]
        and terminal["MainPID"] == "0"
        and terminal["Result"] == "success"
        and terminal["ExecMainStatus"] == "0"
        and terminal["ActiveState"] == "active"
        and terminal["SubState"] == "exited",
        "actual completed native run invocation and all caps",
    )
    old.need(
        old.exact_int(report["returncode"], 0)
        and report["gpu_child_observed"] is True
        and type(report["observed_child_pid"]) is int
        and report["observed_child_pid"] > 0
        and child["child_pid"] == report["observed_child_pid"]
        and type(child["child_pid"]) is int
        and type(child["owner_pid"]) is int
        and child["owner_pid"] == int(live["MainPID"])
        and type(child["child_ppid"]) is int
        and child["child_ppid"] == child["owner_pid"]
        and old.exact_int(report["observed_child_ppid"], child["owner_pid"])
        and report["child_sha256"] == sha256(raw["child.json"]).hexdigest()
        and child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest(),
        "actual owner/child chain and hashes",
    )
    old.valid_host(declaration["host"], child["child_pid"], idle=True)
    old.need(
        type(report["monitor"]) is list and 0 < len(report["monitor"]) <= 480,
        "bounded actual GPU observations",
    )
    seen = False
    previous = -1
    for row in report["monitor"]:
        old.need(
            type(row["elapsed"]) in (int, float)
            and math.isfinite(row["elapsed"])
            and previous <= row["elapsed"] < 240
            and old.exact_int(row["child_pid"], child["child_pid"]),
            "bounded actual child timeline",
        )
        seen |= old.valid_host(row["host"], child["child_pid"])
        old.need(
            row["host"]["used_mib"] <= 12288 and row["host"]["free_mib"] >= 10240,
            "bounded whole-GPU memory exposure",
        )
        previous = row["elapsed"]
    old.need(seen, "GPU child actually observed, not inferred")
    tests = declaration["tests"]
    test_receipt = old._json(raw["tests.receipt.json"])
    test_terminal = old._json(raw["tests.terminal.json"])
    test_caps = {**caps, "LimitFSIZE": "67108864"}
    old.need(
        type(tests) is dict
        and old.exact_int(tests["count"], declaration["expected_tests"])
        and old.exact_int(declaration["expected_tests"], EXPECTED_TESTS)
        and old.exact_int(tests["files"], 63)
        and tests["source_binding"] == declaration["source_binding"]
        and type(tests["receipt_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", tests["receipt_sha256"])
        and tests["terminal"]["InvocationID"] != live["InvocationID"]
        and tests["terminal"]["Result"] == "success"
        and tests["terminal"]["MainPID"] == "0"
        and tests["terminal"]["ExecMainStatus"] == "0"
        and tests["terminal"]["NRestarts"] == "0",
        "fresh current-source independent successful CPU prerequisite",
    )
    old.need(
        test_receipt["protocol"] == DATA_PROTOCOL + ":tests"
        and test_receipt["source_binding"] == declaration["source_binding"]
        and test_receipt["packages"] == declaration["packages"]
        and old.exact_int(test_receipt["tests"], EXPECTED_TESTS)
        and len(test_receipt["test_files"])
        == len(set(test_receipt["test_files"]))
        == 63
        and sha256(old.canonical(test_receipt["test_files"])).hexdigest()
        == TEST_FILES_SHA256
        and test_receipt["retained_descriptor_fixture"]
        == dict(
            path="artifacts/evaluations/com-entry-run-cbcc4d88ca3f/child.json",
            bytes=9948,
            sha256="56370eb3546f19a395a89cf864fc83e25d0690413b6bfbcee459b54b151941b9",
        )
        and old.false_flags(test_receipt["flags"])
        and sha256(raw["tests.receipt.json"]).hexdigest() == tests["receipt_sha256"]
        and tests["terminal"] == test_terminal
        and all(
            test_terminal.get(k) == v and test_receipt["service_properties"].get(k) == v
            for k, v in test_caps.items()
        )
        and test_terminal["ActiveState"] == "active"
        and test_terminal["SubState"] == "exited"
        and test_terminal["InvocationID"]
        == test_receipt["service_properties"]["InvocationID"]
        and re.fullmatch(r"[0-9a-f]{32}", test_terminal["InvocationID"])
        and test_receipt["service_properties"]["ActiveState"] == "active"
        and test_receipt["service_properties"]["SubState"] == "running"
        and re.fullmatch(r"[1-9][0-9]*", test_receipt["service_properties"]["MainPID"]),
        "whole completed CPU receipt and matching actual capped invocation",
    )
    old.need(
        test_receipt["test_environment"]
        == {
            "CUDA_VISIBLE_DEVICES": "",
            "MICRODUCK_STANCE_PROFILE": "wsl-10098-20260930",
            "ATEN_CPU_CAPABILITY": "default",
            "MKL_CBWR": "COMPATIBLE",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
        "whole recorded initial CPU test environment",
    )
    old.need(
        child["case_order"] == list(frames.CASES)
        and type(child["cases"]) is dict
        and set(child["cases"]) == set(frames.CASES)
        and old.exact_int(child["integration_calls"], 0)
        and old.exact_int(child["physics_steps"], 0)
        and all(
            child[k] is False
            for k in (
                "graph_created",
                "actor_model_created",
                "optimizer_created",
                "storage_created",
            )
        ),
        "four predeclared cases and no integration/graph/learner",
    )
    filtered = "\n".join(
        line
        for line in raw["child.log"].decode().splitlines()
        if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
    )
    old.need(
        not re.search(
            r"\b(?:nan|nonfinite|overflow|warning|traceback)\b|CUDA error",
            filtered,
            re.I,
        ),
        "finite warning-free owned log",
    )
    weighted_rows = {}
    entry_rows = []
    rng_reference = None
    descriptor_reference = None
    for case in frames.CASES:
        row = child["cases"][case]
        mode = "original" if case.startswith("original") else "serial"
        old.need(
            row["mode"] == mode
            and row["ntendon"] == 0
            and type(row["ntendon"]) is int
            and row["fixed_state_unchanged"] is True
            and old.exact_int(row["physics_steps"], 0),
            "fixed native case and zero steps",
        )
        descriptor = row["compiled_descriptor"]
        old.need(
            sha256(old.canonical(descriptor)).hexdigest() == DESCRIPTOR_SHA256
            and descriptor["selected_fields_sha256"]
            == "6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f",
            "exact retained native compiled descriptor",
        )
        if descriptor_reference is None:
            descriptor_reference = descriptor
        old.need(
            descriptor == descriptor_reference,
            "identical whole fresh native model descriptions",
        )
        entries = [
            raw[f"{case}.entries.bin"][i * ENTRY_BYTES : (i + 1) * ENTRY_BYTES]
            for i in (0, 1)
        ]
        sums = [
            raw[f"{case}.weighted.bin"][i * ENTRY_BYTES : (i + 1) * ENTRY_BYTES]
            for i in (0, 1)
        ]
        old.need(
            len(raw[f"{case}.entries.bin"])
            == len(raw[f"{case}.weighted.bin"])
            == 2 * ENTRY_BYTES,
            "two whole actual CoM boundaries",
        )
        checked_scope(row["com_scope"], mode, entries, sums)
        checked_crb(row["crb_scope"])
        weighted_rows[case] = [
            weighted_comparison(a, b) for a, b in zip(entries, sums, strict=True)
        ]
        entry_rows.extend(entries)
        rng = checked_rng(raw[f"{case}.rng.bin"], row["rng_metadata"])
        if rng_reference is None:
            rng_reference = rng
        old.need(
            rng == rng_reference,
            "identical complete caller and constructor-private RNG bytes across cases",
        )
        # The whole-authenticated descriptor pins the native compiled home pose.
        decoded = frames._decode_packet(raw[f"{case}.frames.bin"], case)
        qpos = np.asarray(descriptor["initial_qpos"], dtype="<f4")
        old.need(
            qpos.shape == (21,) and np.isfinite(qpos).all(),
            "whole compiled initial qpos",
        )
        for frame in decoded:
            old.need(
                np.array_equal(
                    frame["qpos"].view("<u4"),
                    np.broadcast_to(qpos.view("<u4"), (64, 21)),
                ),
                "actual initial qpos matches frozen native compiled pose",
            )
            old.need(
                all(
                    not frame[name].view("<u4").any()
                    for name in frames.FIXED_INPUT_FIELDS
                    if name != "qpos"
                ),
                "literal positive-zero fixed velocity/time/control/force/warmstart",
            )
    inputs_exact = all(raw == entry_rows[0] for raw in entry_rows)
    arithmetic_exact = all(
        row["exact_raw_bits"]
        for case in ("serial0", "serial1")
        for row in weighted_rows[case]
    )
    comparison = frames.compare(
        {case: raw[f"{case}.frames.bin"] for case in frames.CASES},
        {case: anchors[f"{case}.frames.bin"]["sha256"] for case in frames.CASES},
    )
    decision = (
        "fresh-coupled-control-exact"
        if inputs_exact
        and arithmetic_exact
        and comparison["decision"] == "coupled-serial-repeat-exact"
        else "fresh-coupled-control-negative"
    )
    return dict(
        protocol=PROTOCOL,
        source=source,
        decision=decision,
        actual_initialized_entries_exact=inputs_exact,
        serial_weighted_arithmetic_exact=arithmetic_exact,
        weighted_comparisons=weighted_rows,
        frame_comparison=comparison,
        receiver_files=anchors,
        terminal_sha256=terminal_sha256,
        flags=dict(old.FLAGS),
    )
