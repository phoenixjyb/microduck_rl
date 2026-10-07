"""Independent whole-byte receiver for a fresh four-case coupled RNE control.

This module authenticates an already completed run. It does not qualify CUDA,
infer an historical cause, or admit training or physical behavior.
"""

from hashlib import sha256
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

from mjlab_microduck import stance_com_coupled_frames as frames
from mjlab_microduck import stance_com_coupled_receiver as coupled
from mjlab_microduck import stance_com_entry_receiver as old
from mjlab_microduck import stance_rne_entry_receiver as prior
from mjlab_microduck import stance_com_coupled_idempotence as temporal

PROTOCOL = "microduck-rne-coupled-receiver-oct7-v1"
DATA_PROTOCOL = "microduck-rne-coupled-probe-oct7-v1"
ENTRY_BYTES = 64 * 16 * 6 * 4
FRAME_PACKET_BYTES = 2 * frames.FRAME_BYTES

# Finalize the separate 70-file CPU collection before any native launch.
EXPECTED_TESTS = 2160
TEST_FILES_SHA256 = "ab7999bb7856be69a9aa045de7876ee380fb5dec574163c8a0b022bbb90ff3b9"

DESCRIPTOR_SHA256 = coupled.DESCRIPTOR_SHA256
SELECTED_FIELDS_SHA256 = (
    "6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f"
)
DESCRIPTOR_FIXTURE = {
    "path": "artifacts/evaluations/com-entry-run-cbcc4d88ca3f/child.json",
    "bytes": 9948,
    "sha256": "56370eb3546f19a395a89cf864fc83e25d0690413b6bfbcee459b54b151941b9",
}
RETAINED_TEST_FIXTURES = {
    "files": 12,
    "bytes": 50377522,
    "sha256": "8ac6c0cfa167e0e9248ac3c37beeb1d3b2bd92010347d75f93cd128f3eb09797",
}
PHASE_A = {
    "receipt.json": "1728f97adb06662fcca45baaba470b2744bb0b006882886ea386af38de93414b",
    "junit.xml": "937eef453d3e66f12ed5032722310c745a91f478a42fe657cdfbc3446dc8fd47",
    "pytest.log": "febe0fcdf4aadebe52fa248899de69a3835508270359d8abdcbf2b8f7d922dc2",
}
PHASE_A_SOURCE = "ab88b2621edc14214e96922bc40e7f3685312113"
PHASE_A_TERMINAL = dict(
    {**old.SERVICE_CAPS, "LimitFSIZE": "67108864"},
    InvocationID="aae498825b9f4814ab3575171e2ff7aa",
    MainPID="0",
    Result="success",
    ExecMainStatus="0",
    ActiveState="active",
    SubState="exited",
)
PREDECESSORS = {
    "artifacts/tools/stance-crb-runtime-partial-owner-terminal-2ecee471f7b9.json": "dc2aa2d344fcad58d2054a2238c15d0d3678724852160834dc5a2a29bef8151a",
    "artifacts/tools/stance-crb-runtime-partial-mac-2ecee471f7b9/partial-mac-verification-2ecee471f7b9.json": "5d77483122dc390a53c5d277828f9d8eb7196856b0d950e2445bf9694ab4b167",
    "artifacts/tools/stance-cpu-plant-fields-2ecee471f7b9-20261006-0720/fields.json": "e8160827a52db740a6edc2e89973e5313f0020ed7be79ef609c903a1cf8f6ace",
}
COM_PREDECESSOR = {
    "source": "6cd03291581684680e6ab3875a96f88cf8dd83ca",
    "paired_hashes": {
        "inventory.json": "1048cd38bd8c68c34bba18ea2af65a499d91e07c92ce60fc090593a313036809",
        "terminal.json": "d1cd1c224625fb6171b0da71b24eeec2ec9396ebbb2c51223783e6837efa993f",
        "receiver.json": "51ec8165d49ed398c2704bbe2ce65ce0b9f3e0778dfcd6056c7375d139e113d0",
    },
    "temporal_sha256": "2a871d75faff2d1da3d78ca644fb24332c92cf1ca5830301d5bd77659c4c3e08",
    "paired_decision": "fresh-coupled-control-exact",
    "temporal_decision": "temporal-serial-negative",
}

CASE_ORDER = ("original0", "serial0", "original1", "serial1")
RNE_CLOSEOUT = {
    "inventory.json": {
        "bytes": 1506,
        "sha256": "12eab5e741dae16d23902ec6835684ed83c44412177ff77eb44427581426ebab",
    },
    "terminal.json": {
        "bytes": 390,
        "sha256": "ba67f3013ce3e79f30cd49c0789d065e6bc31e332a06ce9adc98834bb5ed292d",
    },
    "receiver.json": {
        "bytes": 109448,
        "sha256": "f55d4344ad6f635a31c23e58660166b0de889c53b21f7bd1c677944d407e56be",
    },
    "verification.json": {
        "bytes": 1909,
        "sha256": "451afcf742ca12cfc83b9783f6a1b8a7e718edd482d9fa1c63e701dee9c26b56",
    },
    "mac-receiver.json": {
        "bytes": 109448,
        "sha256": "f55d4344ad6f635a31c23e58660166b0de889c53b21f7bd1c677944d407e56be",
    },
}
RNE_PREDECESSOR = {
    "source": "3a50dea5d430ae942a48e81ba5e9391a6e7d3b83",
    "closeout": RNE_CLOSEOUT,
    "decision": "fresh-rne-serial-reference-exact",
    "separate_temporal_exact": False,
}
CASE_CAPS = {
    "entries.bin": 2 * ENTRY_BYTES,
    "outputs.bin": 2 * ENTRY_BYTES,
    "frames.bin": FRAME_PACKET_BYTES,
    "rng.bin": 128 * 1024,
    "com.entries.bin": 2 * coupled.ENTRY_BYTES,
    "com.weighted.bin": 2 * coupled.ENTRY_BYTES,
}
CAPS = {
    case + "." + name: cap for case in CASE_ORDER for name, cap in CASE_CAPS.items()
}
CAPS.update(
    {
        "declaration.json": 128 * 1024,
        "child.json": 128 * 1024,
        "report.json": 256 * 1024,
        "child.log": 1024**2,
        "tests.receipt.json": 128 * 1024,
        "tests.terminal.json": 4096,
    }
)

RNE_FLAGS = {
    "original_run_entry_captured": False,
    "original_pair_accepted": False,
    "actual_kernel_order_observed": False,
    "runtime_cause_proven": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
}

RNE_LEVELS = ((6, 15), (5, 10, 14), (4, 9, 13), (3, 8, 12), (2, 7, 11), (1,), (0,))
RNE_PARENTS = old.PARENTS
COORDINATE_CAP = 16384
RECEIVER_OUTPUT_CAP = 4 * 1024**2


def _need(condition, message):
    old.need(condition, message)


def read_inventory(root, anchors):
    """Read and authenticate every exact artifact before JSON/NumPy decoding."""
    root = Path(root)
    _need(root.is_dir() and not root.is_symlink(), "regular fresh artifact directory")
    _need(
        type(anchors) is dict
        and set(anchors) == set(CAPS)
        and {path.name for path in root.iterdir()} == set(CAPS),
        "exact thirty-file paired artifact inventory",
    )
    result = {}
    for name in sorted(CAPS):
        anchor = anchors[name]
        _need(
            type(anchor) is dict
            and set(anchor) == {"bytes", "sha256"}
            and type(anchor["bytes"]) is int
            and 0 < anchor["bytes"] <= CAPS[name],
            "bounded externally anchored whole artifact",
        )
        descriptor = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            _need(
                stat.S_ISREG(before.st_mode) and before.st_size == anchor["bytes"],
                "regular whole-length artifact",
            )
            chunks = []
            total = 0
            while total <= CAPS[name]:
                chunk = os.read(descriptor, min(65536, CAPS[name] + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
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
                "stable whole artifact read",
            )
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        old._sha(raw, anchor["sha256"])
        result[name] = raw
    _need(sum(map(len, result.values())) < 16 * 1024**2, "whole evidence below 16 MiB")
    return result


def checked_binding(value, source, expected_tree, expected_leaves_sha256):
    _need(
        type(value) is dict
        and set(value) == {"source", "branch", "tree", "leaf_count", "leaves_sha256"}
        and value["source"] == source
        and value["branch"] == "feat/athletics-obstacle-curriculum"
        and type(value["tree"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", value["tree"])
        and value["tree"] == expected_tree
        and type(value["leaf_count"]) is int
        and value["leaf_count"] == 656
        and type(value["leaves_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["leaves_sha256"])
        and value["leaves_sha256"] == expected_leaves_sha256,
        "fresh whole committed source binding",
    )


def _float32(raw, shape, label):
    _need(type(raw) is bytes, label + " immutable bytes")
    count = math.prod(shape)
    _need(len(raw) == count * 4, label + " exact float32 byte length")
    values = np.frombuffer(raw, dtype="<f4").reshape(shape)
    _need(bool(np.isfinite(values).all()), label + " finite float32 values")
    return values


def _ordered_u32(values):
    bits = values.view("<u4").astype(np.uint64)
    sign = (bits & 0x80000000) != 0
    return np.where(sign, 0xFFFFFFFF - bits, bits ^ 0x80000000)


def _field_stats(left, right, *, body_axis=None):
    left_bytes = left.tobytes(order="C")
    right_bytes = right.tobytes(order="C")
    mismatch = left.view("<u4") != right.view("<u4")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    distance = np.abs(
        _ordered_u32(left).astype(np.int64) - _ordered_u32(right).astype(np.int64)
    )
    row = {
        "left_sha256": sha256(left_bytes).hexdigest(),
        "right_sha256": sha256(right_bytes).hexdigest(),
        "scalars": int(left.size),
        "bit_mismatch_scalars": int(mismatch.sum()),
        "max_abs_delta": float(delta.max()),
        "max_ordered_bit_distance": int(distance.max()),
        "exact_raw_bits": not bool(mismatch.any()),
        "world_mismatch_scalars": [
            int(count) for count in mismatch.reshape(64, -1).sum(axis=1)
        ],
    }
    if body_axis is not None:
        per_body = (
            np.moveaxis(mismatch, body_axis, 0)
            .reshape(mismatch.shape[body_axis], -1)
            .sum(1)
        )
        row["body_mismatch_scalars"] = {
            "body0": int(per_body[0]),
            "body1": int(per_body[1]),
            "others": int(per_body[2:].sum()),
        }
    return row


def _rne_array(raw, label):
    return _float32(raw, (64, 16, 6), label)


def _recurrence(entry):
    expected = entry.copy()
    for level in RNE_LEVELS:
        for body in level:
            if body != 0:
                parent = RNE_PARENTS[body]
                np.add(expected[:, parent], expected[:, body], out=expected[:, parent])
                _need(
                    bool(np.isfinite(expected[:, parent]).all()),
                    "finite RNE recurrence",
                )
    return expected


def _check_test_proof(raw, declaration, binding, packages, caps):
    _need(
        EXPECTED_TESTS > 0
        and re.fullmatch(r"[0-9a-f]{64}", TEST_FILES_SHA256) is not None,
        "exact current-source CPU test collection constants are finalized",
    )
    tests = declaration["tests"]
    receipt = old._json(raw["tests.receipt.json"])
    terminal = old._json(raw["tests.terminal.json"])
    test_caps = {**caps, "LimitFSIZE": "67108864"}
    _need(
        declaration["phase_a"]
        == {"source": PHASE_A_SOURCE, "files": PHASE_A, "terminal": PHASE_A_TERMINAL}
        and declaration["predecessors"] == PREDECESSORS
        and declaration["com_predecessor"] == COM_PREDECESSOR
        and declaration["rne_predecessor"] == RNE_PREDECESSOR
        and declaration["case_order"] == list(CASE_ORDER),
        "exact retained Phase A, immutable predecessors and separate CoM negatives",
    )
    _need(
        type(tests) is dict
        and old.exact_int(tests["count"], EXPECTED_TESTS)
        and old.exact_int(declaration["expected_tests"], EXPECTED_TESTS)
        and old.exact_int(tests["files"], len(receipt["test_files"]))
        and tests["source_binding"] == binding
        and type(tests["receipt_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", tests["receipt_sha256"])
        and tests["terminal"] == terminal
        and terminal["InvocationID"]
        != declaration["service_properties"]["InvocationID"]
        and terminal["Result"] == "success"
        and terminal["MainPID"] == "0"
        and terminal["ExecMainStatus"] == "0"
        and terminal["NRestarts"] == "0",
        "completed independent current-source CPU prerequisite",
    )
    files = receipt["test_files"]
    _need(
        receipt["protocol"] == DATA_PROTOCOL + ":tests"
        and receipt["source_binding"] == binding
        and receipt["packages"] == packages
        and old.exact_int(receipt["tests"], EXPECTED_TESTS)
        and type(files) is list
        and len(files) == len(set(files))
        and sha256(old.canonical(files)).hexdigest() == TEST_FILES_SHA256
        and receipt["retained_descriptor_fixture"] == DESCRIPTOR_FIXTURE
        and receipt["retained_test_fixtures"] == RETAINED_TEST_FIXTURES
        and re.fullmatch(r"[0-9a-f]{64}", receipt["junit_sha256"])
        and re.fullmatch(r"[0-9a-f]{64}", receipt["log_sha256"])
        and old.false_flags(receipt["flags"])
        and sha256(raw["tests.receipt.json"]).hexdigest() == tests["receipt_sha256"]
        and all(terminal.get(key) == value for key, value in test_caps.items())
        and all(
            receipt["service_properties"].get(key) == value
            for key, value in test_caps.items()
        )
        and terminal["ActiveState"] == "active"
        and terminal["SubState"] == "exited"
        and terminal["InvocationID"] == receipt["service_properties"]["InvocationID"]
        and re.fullmatch(r"[0-9a-f]{32}", terminal["InvocationID"])
        and receipt["service_properties"]["ActiveState"] == "active"
        and receipt["service_properties"]["SubState"] == "running"
        and re.fullmatch(r"[1-9][0-9]*", receipt["service_properties"]["MainPID"]),
        "whole completed CPU receipt, exact ordered suite and capped invocation",
    )
    _need(
        receipt["test_environment"]
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
        "exact recorded CPU prerequisite environment",
    )


def _check_serial_scope(scope, entries, outputs):
    _need(
        type(scope) is dict
        and scope["protocol"] == "microduck-rne-coupled-serial-oct7-v1"
        and scope["status"] == "complete"
        and scope["fault"] is None
        and old.exact_int(scope["unchanged_postconstraint_passthrough_calls"], 2)
        and old.false_flags(scope["flags"])
        and type(scope["calls"]) is list
        and len(scope["calls"]) == 2,
        "complete actual serial RNE scope and unchanged sensory caller",
    )
    _need(
        scope["dispatch"]
        == {
            "logical_launches_per_bias_call": 7,
            "underlying_launches_per_bias_call": 9,
            "split_level": 4,
            "split_body_order": [2, 7, 11],
            "split_uses_live_initialized_alias": True,
            "sensory_passthrough_calls": 2,
            "closed": True,
        }
        and scope["dispatch"]["closed"] is True
        and scope["dispatch"]["split_uses_live_initialized_alias"] is True,
        "closed literal serial RNE dispatch declaration",
    )
    layout0 = None
    for index, row in enumerate(scope["calls"]):
        _need(
            old.exact_int(row["call_index"], index)
            and old.exact_int(row["worlds"], 64)
            and old.exact_int(row["bytes_per_snapshot"], ENTRY_BYTES)
            and row["device"] == "cuda:0"
            and type(row["stream"]) is int
            and row["stream"] >= 0
            and row["stream"] == scope["calls"][0]["stream"]
            and row["input_sha256"] == sha256(entries[index]).hexdigest()
            and row["output_sha256"] == sha256(outputs[index]).hexdigest()
            and row["input_boundary"] == "after-original-cfrc-before-first-backward"
            and row["output_boundary"]
            == "after-original-backward-before-original-qfrc-bias"
            and row["initializer_caller_code_bound"] is True
            and old.exact_int(row["logical_launches"], 7)
            and old.exact_int(row["underlying_launches"], 9)
            and old.exact_int(row["split_level"], 4)
            and row["split_body_order"] == [2, 7, 11]
            and row["underlying_body_groups"]
            == [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2], [7], [11], [1], [0]]
            and row["split_uses_live_initialized_alias"] is True
            and row["body_rule"] == "body!=0-includes-body1-to-root0"
            and row["input_output_alias"] is True
            and row["readback_and_device_sync_perturb_timing"] is True
            and row["smooth_sha256"]
            == "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
            and row["forward_sha256"]
            == "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
            "literal live initialized serial RNE dispatch, source and snapshots",
        )
        layout = row["layout"]
        _need(
            type(layout) is dict
            and set(layout)
            == {"object_id", "ptr", "shape", "strides", "dtype", "contiguous"}
            and type(layout["object_id"]) is int
            and layout["object_id"] > 0
            and type(layout["ptr"]) is int
            and layout["ptr"] > 0
            and layout["shape"] == [64, 16]
            and layout["strides"] == [384, 24]
            and layout["dtype"] == "spatial_vectorf"
            and layout["contiguous"] is True
            and (layout0 is None or layout == layout0),
            "stable actual serial RNE live spatial storage",
        )
        layout0 = layout


def verify_directory(
    root,
    anchors,
    terminal_raw,
    terminal_sha256,
    *,
    source,
    expected_tree,
    expected_leaves_sha256,
):
    """Authenticate one completed native run and produce a conservative report."""
    _need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and type(expected_tree) is str
        and re.fullmatch(r"[0-9a-f]{40}", expected_tree)
        and type(expected_leaves_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_leaves_sha256),
        "external exact source/tree/leaves anchors",
    )
    _need(
        type(terminal_raw) is bytes and 0 < len(terminal_raw) <= 4096,
        "bounded independent run terminal packet",
    )
    raw = read_inventory(root, anchors)
    old._sha(terminal_raw, terminal_sha256)

    declaration = old._json(raw["declaration.json"])
    report = old._json(raw["report.json"])
    child = old._json(raw["child.json"])
    terminal = old._json(terminal_raw)
    for value, suffix in (
        (declaration, ":declaration"),
        (child, ":child"),
    ):
        _need(
            type(value) is dict
            and value["protocol"] == DATA_PROTOCOL + suffix
            and value["source"] == source
            and old.false_flags(value["flags"]),
            "fresh protocol/source and permanently false admission flags",
        )
        checked_binding(
            value["source_binding"], source, expected_tree, expected_leaves_sha256
        )
    _need(
        type(report) is dict
        and report["protocol"] == DATA_PROTOCOL + ":report"
        and report["source"] == source
        and old.false_flags(report["flags"]),
        "fresh report protocol/source and false flags",
    )
    checked_binding(
        report["source_binding"], source, expected_tree, expected_leaves_sha256
    )
    binding = declaration["source_binding"]
    _need(
        declaration["source_binding"]
        == child["source_binding"]
        == report["source_binding"]
        and child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest()
        and report["child_sha256"] == sha256(raw["child.json"]).hexdigest(),
        "whole declaration/child/report source binding",
    )

    packages = declaration["packages"]
    coupled.checked_packages(packages)
    _need(child["packages"] == packages, "same actual native child package binding")
    live = declaration["service_properties"]
    caps = old.SERVICE_CAPS
    _need(
        type(live) is dict
        and all(
            live.get(key) == value and terminal.get(key) == value
            for key, value in caps.items()
        )
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
        "completed native run with exact retained service caps",
    )
    _need(
        old.exact_int(report["returncode"], 0)
        and report["gpu_child_observed"] is True
        and type(report["observed_child_pid"]) is int
        and report["observed_child_pid"] > 0
        and report["observed_child_ppid"] == int(live["MainPID"])
        and child["child_pid"] == report["observed_child_pid"]
        and type(child["child_pid"]) is int
        and type(child["owner_pid"]) is int
        and child["owner_pid"] == int(live["MainPID"])
        and type(child["child_ppid"]) is int
        and child["child_ppid"] == child["owner_pid"]
        and report["child_sha256"] == sha256(raw["child.json"]).hexdigest()
        and child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest(),
        "observed native owner/child chain and whole child hash",
    )
    old.valid_host(declaration["host"], child["child_pid"], idle=True)
    monitor = report["monitor"]
    _need(type(monitor) is list and 0 < len(monitor) <= 480, "bounded real GPU monitor")
    seen = False
    previous = -1.0
    for row in monitor:
        _need(
            type(row["elapsed"]) in (int, float)
            and math.isfinite(row["elapsed"])
            and previous <= row["elapsed"] < 240
            and old.exact_int(row["child_pid"], child["child_pid"]),
            "bounded monitor timeline and exact child PID",
        )
        seen |= old.valid_host(row["host"], child["child_pid"])
        _need(
            row["host"]["used_mib"] <= 12288 and row["host"]["free_mib"] >= 10240,
            "bounded whole-GPU memory exposure",
        )
        previous = row["elapsed"]
    _need(seen, "actual GPU child observed, not inferred")
    old.valid_host(report["closure_host"], child["child_pid"], idle=True)

    _check_test_proof(raw, declaration, binding, packages, caps)
    log_text = raw["child.log"].decode("utf-8")
    filtered_log = "\n".join(
        line
        for line in log_text.splitlines()
        if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
    )
    _need(
        not re.search(
            r"\b(?:nan|nonfinite|overflow|warning|traceback)\b|CUDA error",
            filtered_log,
            re.I,
        ),
        "finite warning-free owned native child log",
    )

    _need(
        old.exact_int(child["integration_calls"], 0)
        and child["case_order"] == list(CASE_ORDER)
        and type(child["cases"]) is dict
        and set(child["cases"]) == set(CASE_ORDER),
        "one exact four-case native no-integration experiment",
    )
    case_rows, frame_packets, frame_hashes, all_entries, all_com_entries = (
        {},
        {},
        {},
        [],
        [],
    )
    rng_packets = []
    for case in CASE_ORDER:
        record = child["cases"][case]
        mode = "serial" if case.startswith("serial") else "original"
        _need(
            record["mode"] == mode
            and old.exact_int(record["ntendon"], 0)
            and record["fixed_state_unchanged"] is True
            and old.exact_int(record["physics_steps"], 0)
            and all(
                record[key] is False
                for key in (
                    "graph_created",
                    "actor_model_created",
                    "optimizer_created",
                    "storage_created",
                )
            ),
            "literal paired no-step/no-graph/no-learner case",
        )
        descriptor = record["compiled_descriptor"]
        _need(
            sha256(old.canonical(descriptor)).hexdigest() == DESCRIPTOR_SHA256
            and descriptor["selected_fields_sha256"] == SELECTED_FIELDS_SHA256
            and type(descriptor["initial_qpos"]) is list,
            "exact whole native descriptor in every case",
        )
        qpos = np.asarray(descriptor["initial_qpos"], dtype="<f4")
        _need(
            qpos.shape == (21,) and bool(np.isfinite(qpos).all()),
            "finite descriptor home",
        )
        packet = raw[case + ".frames.bin"]
        frame_packets[case], frame_hashes[case] = packet, sha256(packet).hexdigest()
        decoded = frames._decode_packet(packet, case)
        for frame in decoded:
            for name, _shape in frames.FRAME_FIELDS:
                _need(
                    bool(np.isfinite(frame[name]).all()),
                    "finite full paired frame " + name,
                )
            _need(
                np.array_equal(
                    frame["qpos"].view("<u4"),
                    np.broadcast_to(qpos.view("<u4"), (64, 21)),
                ),
                "literal home qpos in all eight complete frames",
            )
            for name in frames.FIXED_INPUT_FIELDS:
                if name != "qpos":
                    _need(
                        not frame[name].view("<u4").any(),
                        "literal positive-zero fixed input " + name,
                    )
        entries = [
            raw[case + ".entries.bin"][i * ENTRY_BYTES : (i + 1) * ENTRY_BYTES]
            for i in range(2)
        ]
        outputs = [
            raw[case + ".outputs.bin"][i * ENTRY_BYTES : (i + 1) * ENTRY_BYTES]
            for i in range(2)
        ]
        entry_arrays = [_rne_array(value, "actual RNE entry") for value in entries]
        output_arrays = [_rne_array(value, "actual RNE output") for value in outputs]
        if mode == "original":
            prior._check_rne_scope(record["rne_scope"], entries, outputs)
        else:
            _check_serial_scope(record["rne_scope"], entries, outputs)
        arithmetic = [
            _field_stats(_recurrence(entry), output, body_axis=1)
            for entry, output in zip(entry_arrays, output_arrays, strict=True)
        ]
        com_entries = [
            raw[case + ".com.entries.bin"][
                i * coupled.ENTRY_BYTES : (i + 1) * coupled.ENTRY_BYTES
            ]
            for i in range(2)
        ]
        weighted = [
            raw[case + ".com.weighted.bin"][
                i * coupled.ENTRY_BYTES : (i + 1) * coupled.ENTRY_BYTES
            ]
            for i in range(2)
        ]
        coupled.checked_scope(record["com_scope"], "serial", com_entries, weighted)
        coupled.checked_crb(record["crb_scope"])
        com_arithmetic = [
            coupled.weighted_comparison(entry, output)
            for entry, output in zip(com_entries, weighted, strict=True)
        ]
        rng = raw[case + ".rng.bin"]
        coupled.checked_rng(rng, record["rng_metadata"])
        rng_packets.append(rng)
        all_entries.extend(entries)
        all_com_entries.extend(com_entries)
        case_rows[case] = {
            "mode": mode,
            "entry_temporal": _field_stats(
                entry_arrays[0], entry_arrays[1], body_axis=1
            ),
            "output_temporal": _field_stats(
                output_arrays[0], output_arrays[1], body_axis=1
            ),
            "rne_arithmetic": arithmetic,
            "rne_arithmetic_exact": all(row["exact_raw_bits"] for row in arithmetic),
            "com_weighted_arithmetic": com_arithmetic,
            "com_serial_arithmetic_exact": all(
                row["exact_raw_bits"] for row in com_arithmetic
            ),
            "rng_sha256": sha256(rng).hexdigest(),
            "rng_states_authenticated": True,
        }
    # All files were whole-authenticated before any metadata/array decode. These
    # independent helpers re-authenticate all four full packets before analysis.
    paired = frames.compare(frame_packets, frame_hashes)
    per_case_temporal = temporal.analyze(frame_packets, frame_hashes)
    rne_inputs_exact = all(value == all_entries[0] for value in all_entries)
    com_inputs_exact = all(value == all_com_entries[0] for value in all_com_entries)
    rng_exact = all(value == rng_packets[0] for value in rng_packets)
    _need(
        rng_exact, "identical caller-restored and private RNG states in all four cases"
    )
    serial_arithmetic_exact = all(
        case_rows[case]["rne_arithmetic_exact"] for case in ("serial0", "serial1")
    )
    com_arithmetic_exact = all(
        row["com_serial_arithmetic_exact"] for row in case_rows.values()
    )
    paired_exact = (
        paired["serial_repeat_exact"] and paired["all_cases_fixed_inputs_exact"]
    )
    temporal_exact = (
        per_case_temporal["serial_temporal_exact"]
        and per_case_temporal["all_cases_fixed_inputs_temporal_exact"]
    )
    accepted = (
        rne_inputs_exact
        and com_inputs_exact
        and serial_arithmetic_exact
        and com_arithmetic_exact
        and paired["serial_repeat_exact"]
        and paired["all_cases_fixed_inputs_exact"]
        and per_case_temporal["serial_temporal_exact"]
        and per_case_temporal["all_cases_fixed_inputs_temporal_exact"]
    )
    result = {
        "protocol": PROTOCOL,
        "source": source,
        "decision": "fresh-coupled-rne-control-exact"
        if accepted
        else "fresh-coupled-rne-control-negative",
        "case_order": list(CASE_ORDER),
        "cases": case_rows,
        "all_actual_initialized_rne_inputs_exact": rne_inputs_exact,
        "all_actual_initialized_com_inputs_exact": com_inputs_exact,
        "all_serial_rne_arithmetic_exact": serial_arithmetic_exact,
        "all_com_serial_arithmetic_exact": com_arithmetic_exact,
        "all_cases_rng_exact": rng_exact,
        "paired_control_decision": (
            "serial-paired-frames-exact"
            if paired_exact
            else "serial-paired-frames-negative"
        ),
        "temporal_control_decision": (
            "serial-temporal-frames-exact"
            if temporal_exact
            else "serial-temporal-frames-negative"
        ),
        "paired_frames": paired,
        "per_case_temporal": per_case_temporal,
        "receiver_files": anchors,
        "terminal_sha256": terminal_sha256,
        "flags": dict(RNE_FLAGS),
    }
    _need(
        len(old.canonical(result)) <= RECEIVER_OUTPUT_CAP,
        "complete receiver below 4 MiB",
    )
    return result
