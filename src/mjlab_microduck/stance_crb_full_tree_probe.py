"""Bounded whole-tree fixed-input CRB schedule controller.

The retained outputs compare concurrent seven-level launches with a nine-group
parent-conflict schedule over one fixed, authenticated completed-forward
full-cinert fixture. This is diagnostic evidence, not a trajectory replay,
kernel-cause proof, or training result.
"""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import numpy as np
import torch
import warp as wp
from mujoco_warp._src import smooth
from mujoco_warp._src.types import vec10 as WP_VEC10

from mjlab_microduck import stance_crb_kernel_probe as frozen
from mjlab_microduck import stance_crb_kernel_repeat as repeat
from mjlab_microduck import stance_crb_launch_trace as crb
from mjlab_microduck import stance_crb_level_plan as planner
from mjlab_microduck import stance_crb_serial_control as serial_control
from mjlab_microduck import stance_crb_serial_control_probe as serial
from mjlab_microduck import stance_crb_full_tree as full_tree
from mjlab_microduck import stance_cuda_probe as cuda_probe
from mjlab_microduck import stance_inertia_order_probe as order
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
MODULE = "mjlab_microduck.stance_crb_full_tree_probe"
PROTOCOL = "football-b1d-crb-full-tree-control-probe-20261006-v1"
BASE_SOURCE = "a3b149d4fef78c1bf1a23a1b44699fabdb599a1a"
SYNC_FROM_SOURCE = "c9708cf68fdb812733f7a9fddd336c36a8241d26"
SERIAL_SOURCE = SYNC_FROM_SOURCE
OWNER_PACKET_SHA256 = "186dea37ef643c510bf3285a5c2a9785dd91c1152a64587fc95fec085db9af02"
OWNER_PACKET_PATH = "artifacts/tools/stance-crb-serial-owner-terminal-c9708cf68fdb.json"
CLOSEOUT_PACKET_SHA256 = (
    "f431fc6f3677fcba16ba530351b2fd970f4d8cf276e2ed76f42978a3cbf7062d"
)
CLOSEOUT_PACKET_PATH = (
    "artifacts/tools/stance-crb-serial-closeout-terminal-c9708cf68fdb.json"
)
SERIAL_TESTS_INVOCATION = "d31a6013091b48f28262383289069818"
SERIAL_RUN_INVOCATION = "2eeda48c67444383b27d0be278b9011a"
SERIAL_SYNC_INVOCATION = "eecef0634bdd46c9bf7a51ba771d48c5"
SERIAL_CLOSEOUT_INVOCATION = "e7f777dd9f924438a5485c407aac9995"
LAUNCH_TOPOLOGY_SHA256 = planner.TOPOLOGY_SHA256
BASELINE_SHA256 = "eec453717d592814c0b15d4b44acd4a8836d5acb941699ac5e5ed27b96805368"
EXPECTED_SHA256 = "5a5cac252692294113c560b0139ec72670cf9dd15143acf349ebe46818e4a1f4"
SMOOTH_SHA256 = frozen.SMOOTH_SHA
CAP = 600
TEST_CAP = 300
FIXTURE_CAP = 300
CLOSEOUT_CAP = 300
CHILD_SECONDS = 240
MEMORY = 6 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
MARGIN = 60
RUN_RESERVE = CAP + CLOSEOUT_CAP + MARGIN
FIXTURE_RESERVE = FIXTURE_CAP + CAP + CLOSEOUT_CAP + MARGIN
TEST_RESERVE = TEST_CAP + FIXTURE_CAP + CAP + CLOSEOUT_CAP + MARGIN
CUTOFF = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc).timestamp()
EXPECTED_TESTS = 1522  # Owner-collected 51-file CUDA-hidden suite, no skips.
DOC = "docs/experiments/2026-10-06-crb-full-tree-control.md"
TEST_FILES = serial.TEST_FILES + (
    "test_stance_crb_level_plan.py",
    "test_stance_crb_full_tree.py",
    "test_stance_crb_full_tree_probe.py",
)
OWN = (
    "src/mjlab_microduck/stance_crb_full_tree_probe.py",
    "tests/test_stance_crb_full_tree_probe.py",
    DOC,
)
RELATED = (
    "src/mjlab_microduck/stance_crb_full_tree.py",
    "tests/test_stance_crb_full_tree.py",
)
PLANNER_FROZEN = (
    "src/mjlab_microduck/stance_crb_level_plan.py",
    "tests/test_stance_crb_level_plan.py",
)
ALLOWED = set(OWN) | set(RELATED)
FLAGS = {
    **{key: False for key, value in serial.FLAGS.items() if value is False},
    **{key: False for key, value in full_tree.FLAGS.items() if value is False},
    "full_tree_schedule_control_qualified": False,
    "full_tree_schedule_proves_original_cause": False,
    "full_native_window_qualified": False,
    "training_authorized": False,
}
KERNEL = smooth._crb_accumulate
KERNEL_NAME = "_crb_accumulate"
JSON_LIMIT = 2 * 1024**2
LOG_LIMIT = 1024**2
MATRIX_BYTES = repeat.WORLDS * repeat.BODIES * repeat.COMPONENTS * 4
OUTPUT_BYTES = repeat.REPEATS * MATRIX_BYTES
RNG_LIMIT = 64 * 1024
TOTAL_BYTES = 12 * 1024**2
FIXTURE_FILES = {
    "declaration.json",
    "baseline.bin",
    "capture-crb.bin",
    "replay-crb.bin",
    "expected.bin",
    "report.json",
}
RUN_FILES = {
    "declaration.json",
    "baseline.bin",
    "capture-crb.bin",
    "replay-crb.bin",
    "expected.bin",
    "concurrent.bin",
    "serial.bin",
    "rng.bin",
    "child-receipt.json",
    "child.log",
    "report.json",
}
TEST_FILES_ON_DISK = {"receipt.json", "pytest.log", "report.json"}
_RUN_DECL_KEYS = {
    "protocol",
    "source",
    "source_binding",
    "host_identity",
    "service_properties",
    "tests_receipt_sha256",
    "fixture_declaration_sha256",
    "fixture_report_sha256",
    "fixture_artifacts",
    "fixture_metadata",
    "topology",
    "topology_sha256",
    "concurrent_levels",
    "serial_groups",
    "schedule",
    "runtime_versions",
    "smooth_sha256",
    "expected_sha256",
    "predecessor_summary",
    *FLAGS,
}
_CHILD_KEYS = {
    "protocol",
    "source",
    "source_binding",
    "declaration_sha256",
    "owner_pid",
    "child_pid",
    "child_ppid",
    "service_properties",
    "host_identity",
    "runtime_versions",
    "smooth_sha256",
    "launch",
    "torch_rng_preserved",
    "torch_rng_state_bytes",
    "torch_cuda_initialized",
    "actor_constructed",
    "model_constructed",
    "optimizer_constructed",
    "random_kernels",
    "concurrent_sha256",
    "serial_sha256",
    "rng_sha256",
    *FLAGS,
}


def unit(source, mode):
    base._hex(source, 40, "exact full-tree source")
    require(mode in ("tests", "fixture", "run"), "declared full-tree mode")
    return f"microduck-crb-fulltree-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    area = "tools" if mode == "tests" else "evaluations"
    return (
        base.execution.ROOT
        / "artifacts"
        / area
        / f"stance-crb-fulltree-{mode}-{source[:12]}"
    )


def source_sync_unit(source):
    base._hex(source, 40, "exact full-tree bootstrap source")
    return f"microduck-crb-fulltree-sync-{source[:12]}.service"


def source_sync_script(source, *, bundle_sha256):
    script = serial.source_sync_script(source, bundle_sha256=bundle_sha256)
    changes = (
        (
            f'test "$(git rev-parse HEAD)" = {serial.SYNC_FROM_SOURCE}\n',
            f'test "$(git rev-parse HEAD)" = {SYNC_FROM_SOURCE}\n',
        ),
        (
            f'if test "$duck_sync_running" != {serial.source_sync_unit(source)}; then\n',
            f'if test "$duck_sync_running" != {source_sync_unit(source)}; then\n',
        ),
    )
    for old, new in changes:
        require(script.count(old) == 1, "one exact frozen full-tree sync guard")
        script = script.replace(old, new, 1)
    return script


def _frozen_paths():
    return (
        set(frozen._frozen_paths())
        | set(frozen.OWN)
        | set(frozen.RELATED)
        | set(order.OWN)
        | set(serial.OWN)
        | set(serial.RELATED)
        | set(PLANNER_FROZEN)
    )


def source_binding(source):
    base._hex(source, 40, "exact committed full-tree source")
    root = Path(__file__).resolve().parents[2]
    require(str(root) == frozen.cpu_probe.SYNC_ROOT, "exact native WSL source root")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=15
        ).stdout

    require(
        git("rev-parse", "HEAD").decode().strip() == source
        and git("branch", "--show-current").decode().strip()
        == "feat/athletics-obstacle-curriculum"
        and git("remote", "get-url", "origin").decode().strip()
        == frozen.cpu_probe.SYNC_ORIGIN
        and not git("status", "--porcelain"),
        "clean exact native full-tree branch, origin and HEAD",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE, source],
        cwd=root,
        check=True,
        timeout=15,
    )
    changed = set(git("diff", "--name-only", BASE_SOURCE, source).decode().splitlines())
    require(changed <= ALLOWED, "only five declared full-tree paths changed")
    frozen_paths = _frozen_paths()
    leaves = {}
    for path in sorted(frozen_paths | set(OWN) | set(RELATED)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(
            raw == git("show", f"{source}:{path}"),
            "whole committed source leaf " + path,
        )
        if path in frozen_paths:
            require(
                raw == git("show", f"{BASE_SOURCE}:{path}"),
                "frozen A3 source leaf " + path,
            )
        leaves[path] = base.digest(raw)
    return {
        "source": source,
        "branch": "feat/athletics-obstacle-curriculum",
        "leaves": leaves,
    }


def _check_window(*, reserve_seconds, now=None):
    require(
        type(reserve_seconds) is int and reserve_seconds > 0,
        "positive full-tree reserved time",
    )
    require(
        (time.time() if now is None else now) + reserve_seconds < CUTOFF,
        "full-tree fixed cutoff",
    )


def _cpu_env():
    env = frozen._cpu_env()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return env


def _gpu_env():
    env = _cpu_env()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def _runtime_versions():
    return frozen.runtime_versions()


def _read_file(path, limit):
    return frozen._read_file(path, limit)


def _inventory(root, names):
    result = {}
    total = 0
    for name in sorted(names):
        cap = (
            LOG_LIMIT
            if name.endswith(".log")
            else JSON_LIMIT
            if name.endswith(".json")
            else RNG_LIMIT
            if name == "rng.bin"
            else 2 * 1024**2
            if name.endswith(".bin")
            else JSON_LIMIT
        )
        raw = _read_file(root / name, cap)
        total += len(raw)
        require(total <= TOTAL_BYTES, "full-tree total artifact byte cap")
        result[name] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return result


def _partial_inventory(root):
    names = RUN_FILES | FIXTURE_FILES | TEST_FILES_ON_DISK
    result = {}
    for name in sorted(names):
        path = root / name
        try:
            metadata = path.lstat()
        except OSError:
            result[name] = {"present": False, "readable": False}
            continue
        if not stat.S_ISREG(metadata.st_mode):
            result[name] = {
                "present": True,
                "regular": False,
                "bytes": metadata.st_size,
            }
            continue
        try:
            cap = (
                LOG_LIMIT
                if name.endswith(".log")
                else JSON_LIMIT
                if name.endswith(".json")
                else RNG_LIMIT
                if name == "rng.bin"
                else 2 * 1024**2
            )
            raw = _read_file(path, cap)
            result[name] = {
                "present": True,
                "regular": True,
                "bytes": len(raw),
                "sha256": base.digest(raw),
            }
        except (OSError, ValueError):
            result[name] = {
                "present": True,
                "regular": True,
                "bytes": metadata.st_size,
                "readable": False,
            }
    return result


def _authenticate_original():
    return frozen._authenticate_original()


def _terminal_record(value, mode, invocation, *, memory_peak=None):
    keys = set(serial._TERMINAL_KEYS)
    if memory_peak is not None:
        keys.add("MemoryPeak")
    require(
        type(value) is dict and set(value) == keys,
        "exact retained terminal service schema",
    )
    expected = {
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "RemainAfterExit": "yes",
        "RuntimeMaxUSec": {
            "sync": "2min",
            "tests": "5min",
            "fixture": "5min",
            "run": "10min",
            "closeout": "2min",
        }[mode],
        "MemoryMax": str(256 * 1024**2 if mode == "sync" else MEMORY),
        "CPUQuotaPerSecUSec": "1s" if mode == "sync" else CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
        "InvocationID": invocation,
    }
    if memory_peak is not None:
        expected["MemoryPeak"] = str(memory_peak)
    require(value == expected, "exact retained invocation and resource caps")


def _properties(source, mode, owner_pid):
    require(mode in ("tests", "fixture", "run"), "full-tree live service mode")
    keys = (
        "MainPID",
        "ActiveState",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "Nice",
        "KillMode",
        "InvocationID",
        "RemainAfterExit",
    )
    value = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in keys
    }
    runtime = "10min" if mode == "run" else "5min"
    expected = {
        "MainPID": str(owner_pid),
        "ActiveState": "active",
        "RuntimeMaxUSec": runtime,
        "MemoryMax": str(MEMORY),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
        "RemainAfterExit": "yes",
    }
    require(
        type(owner_pid) is int
        and owner_pid > 0
        and {key: value[key] for key in expected} == expected,
        "actual full-tree owner PID and retained service caps",
    )
    base._hex(value["InvocationID"], 32, "full-tree service invocation")
    running = base.host.read(
        "systemctl",
        "--user",
        "list-units",
        "--state=running",
        "--no-legend",
        "microduck*",
    )
    require(
        {line.split()[0] for line in running.splitlines()} == {unit(source, mode)},
        "only one owned Duck compute service active",
    )
    return value


def _terminal(source, mode, invocation):
    base._hex(invocation, 32, "completed full-tree invocation")
    keys = (
        "MainPID",
        "ActiveState",
        "SubState",
        "RemainAfterExit",
        "NRestarts",
        "ExecMainStatus",
        "Result",
        "InvocationID",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "Nice",
        "KillMode",
    )
    value = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in keys
    }
    runtime = "10min" if mode == "run" else "5min"
    expected = {
        "MainPID": "0",
        "ActiveState": "active",
        "SubState": "exited",
        "RemainAfterExit": "yes",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": invocation,
        "RuntimeMaxUSec": runtime,
        "MemoryMax": str(MEMORY),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
    }
    require(value == expected, "successful full-tree service with exact terminal caps")
    return value


def _serial_owner_packet_keys():
    return {
        "closeout_memory_at_publish",
        "closeout_process_limits",
        "closeout_service_properties",
        "concurrent",
        "filmbrain",
        "gpu_initialized",
        "host_identity",
        "idle",
        "original",
        "prelaunch_cpu_prediction",
        "protected_services",
        "protocol",
        "result",
        "rollout_cause_proven",
        "run_files",
        "source",
        "source_binding",
        "terminal_run",
        "terminal_sync",
        "terminal_tests",
        "tests_files",
        "tests_receipt_sha256",
        "training_authorized",
    }


def _validate_serial_predecessor(
    owner_raw,
    closeout_raw,
    run_files,
    tests_files,
    report_raw,
    receipt_raw,
    pytest_raw,
    tests_report_raw,
    original,
    concurrent,
):
    require(
        base.digest(owner_raw) == OWNER_PACKET_SHA256,
        "serial owner packet hash before parse",
    )
    require(
        base.digest(closeout_raw) == CLOSEOUT_PACKET_SHA256,
        "serial independent closeout packet hash before parse",
    )
    owner = base.parse_json(owner_raw)
    closeout = base.parse_json(closeout_raw)
    report = base.parse_json(report_raw)
    receipt = base.parse_json(receipt_raw)
    tests_report = base.parse_json(tests_report_raw)
    require(
        type(owner) is dict and set(owner) == _serial_owner_packet_keys(),
        "exact historical serial owner packet schema",
    )
    require(
        type(closeout) is dict
        and set(closeout)
        == {
            "gpu_initialized",
            "owner_packet_sha256",
            "protocol",
            "source",
            "terminal",
            "training_authorized",
        },
        "exact historical serial independent-closeout schema",
    )
    require(
        owner["protocol"] == serial.PROTOCOL + ":independent-owner-closeout"
        and owner["source"] == SERIAL_SOURCE
        and owner["original"] == original
        and owner["concurrent"] == concurrent
        and owner["gpu_initialized"] is False
        and owner["rollout_cause_proven"] is False
        and owner["training_authorized"] is False,
        "serial owner packet binds 38-file predecessor evidence",
    )
    _authenticate_serial_source(owner["source_binding"])
    require(
        run_files == owner["run_files"] and tests_files == owner["tests_files"],
        "all seven serial run and three tests leaves match owner packet",
    )
    require(
        type(report) is dict
        and report.get("protocol") == serial.PROTOCOL + ":run"
        and report.get("source") == SERIAL_SOURCE
        and report.get("status") == "passed"
        and type(report.get("elapsed_seconds")) in (int, float)
        and report["elapsed_seconds"] < serial.CAP
        and report.get("result", {}).get("analysis") == owner["result"]
        and report.get("source_binding") == owner["source_binding"]
        and report.get("host_identity") == owner["host_identity"]
        and report.get("filmbrain") == owner["filmbrain"]
        and report.get("protected_services") == owner["protected_services"]
        and report.get("service_properties", {}).get("InvocationID")
        == SERIAL_RUN_INVOCATION
        and all(report.get(key) is False for key in serial.FLAGS),
        "serial run report independently binds owner packet and false claims",
    )
    require(
        run_files["report.json"]
        == {"bytes": len(report_raw), "sha256": base.digest(report_raw)},
        "serial run whole report bytes bound before interpretation",
    )
    receipt_keys = {
        "protocol",
        "source",
        "source_binding",
        "test_files",
        "passed",
        "skips",
        "pytest_sha256",
        "service_properties",
        *serial.FLAGS,
    }
    require(
        type(receipt) is dict
        and set(receipt) == receipt_keys
        and receipt.get("protocol") == serial.PROTOCOL + ":tests"
        and receipt.get("source") == SERIAL_SOURCE
        and receipt.get("source_binding") == owner["source_binding"]
        and receipt.get("test_files") == list(serial.TEST_FILES)
        and receipt.get("passed") == serial.EXPECTED_TESTS
        and receipt.get("skips") == 0
        and all(receipt.get(key) is False for key in serial.FLAGS),
        "serial test receipt exact frozen successful source-only schema",
    )
    serial._recorded_properties(receipt["service_properties"], "tests")
    require(
        base.digest(receipt_raw) == owner["tests_files"]["receipt.json"]["sha256"]
        and base.digest(pytest_raw) == receipt["pytest_sha256"]
        and tests_files["receipt.json"]
        == {"bytes": len(receipt_raw), "sha256": base.digest(receipt_raw)}
        and tests_files["pytest.log"]
        == {"bytes": len(pytest_raw), "sha256": base.digest(pytest_raw)}
        and tests_files["report.json"]
        == {"bytes": len(tests_report_raw), "sha256": base.digest(tests_report_raw)},
        "serial tests three-file whole-byte inventory bound",
    )
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", pytest_raw)
    require(
        len(totals) == 1
        and int(totals[0][0]) == serial.EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", pytest_raw),
        "serial CPU test count and zero skips",
    )
    tests_result = tests_report.get("result", {})
    require(
        tests_report.get("protocol") == serial.PROTOCOL + ":tests"
        and tests_report.get("source") == SERIAL_SOURCE
        and tests_report.get("mode") == "tests"
        and tests_report.get("status") == "passed"
        and type(tests_report.get("elapsed_seconds")) in (int, float)
        and tests_report["elapsed_seconds"] < serial.TEST_CAP
        and tests_report.get("source_binding") == owner["source_binding"]
        and tests_report.get("service_properties") == receipt["service_properties"]
        and tests_result.get("passed") == serial.EXPECTED_TESTS
        and tests_result.get("pytest_sha256") == receipt["pytest_sha256"]
        and all(tests_report.get(key) is False for key in serial.FLAGS),
        "serial CPU test report matches retained receipt",
    )
    require(
        owner["tests_receipt_sha256"] == base.digest(receipt_raw),
        "serial owner test-receipt hash",
    )
    require(
        owner["terminal_sync"]
        == {
            "ActiveState": "active",
            "CPUQuotaPerSecUSec": "1s",
            "ExecMainStatus": "0",
            "InvocationID": SERIAL_SYNC_INVOCATION,
            "KillMode": "control-group",
            "MainPID": "0",
            "MemoryMax": str(256 * 1024**2),
            "NRestarts": "0",
            "Nice": "10",
            "RemainAfterExit": "yes",
            "Result": "success",
            "RuntimeMaxUSec": "2min",
            "SubState": "exited",
        },
        "terminal serial sync invocation retained",
    )
    _terminal_record(owner["terminal_tests"], "tests", SERIAL_TESTS_INVOCATION)
    _terminal_record(owner["terminal_run"], "run", SERIAL_RUN_INVOCATION)
    _terminal_record(
        closeout["terminal"],
        "closeout",
        SERIAL_CLOSEOUT_INVOCATION,
        memory_peak=1130577920,
    )
    require(
        closeout["protocol"] == serial.PROTOCOL + ":independent-owner-closeout:terminal"
        and closeout["source"] == SERIAL_SOURCE
        and closeout["owner_packet_sha256"] == OWNER_PACKET_SHA256
        and closeout["gpu_initialized"] is False
        and closeout["training_authorized"] is False,
        "independent terminal closeout binds exact owner packet",
    )
    require(
        owner["closeout_service_properties"]
        == {
            "ActiveState": "active",
            "CPUQuotaPerSecUSec": "2s",
            "InvocationID": SERIAL_CLOSEOUT_INVOCATION,
            "KillMode": "control-group",
            "MainPID": owner["closeout_service_properties"].get("MainPID"),
            "MemoryMax": str(MEMORY),
            "Nice": "10",
            "RemainAfterExit": "yes",
            "RuntimeMaxUSec": "2min",
        }
        and str(owner["closeout_service_properties"].get("MainPID", "")).isdecimal()
        and int(owner["closeout_service_properties"]["MainPID"]) > 0,
        "serial closeout owner exact live caps and invocation",
    )
    require(
        owner["closeout_process_limits"]
        == {
            "cpu_quota_percent": 200,
            "kill_mode": "control-group",
            "nice": 10,
            "resident_memory_bytes": MEMORY,
            "service_runtime_seconds": 120,
        }
        and owner["closeout_memory_at_publish"]
        == {"VmPeak": "14786828 kB", "VmRSS": "1522472 kB", "VmSize": "6420100 kB"},
        "serial closeout retained process limits and memory observation",
    )
    owner_non_admitting = {
        key: False for key, value in repeat.FLAGS.items() if value is False
    }
    owner_non_admitting.update(
        serial_schedule_is_cause_proof=False,
        serial_schedule_qualifies_training=False,
        serial_schedule_qualifies_full_window=False,
    )
    require(
        type(owner["result"]) is dict
        and all(owner["result"].get(key) is False for key in owner_non_admitting)
        and owner["result"].get("order_membership_is_not_selection") is True,
        "serial result remains explicitly non-admitting with descriptive membership flag",
    )
    return owner, closeout


def _authenticate_predecessors():
    original = _authenticate_original()
    concurrent = serial._authenticate_concurrent(original)
    run_root, tests_root = _serial_roots()
    owner_path, closeout_path = _serial_packet_roots()
    serial_run_names = set(serial.RUN_FILES)
    serial_test_names = set(serial.TEST_FILES_ON_DISK)
    base._exact_inventory(run_root, serial_run_names)
    base._exact_inventory(tests_root, serial_test_names)
    run_files = serial._inventory(run_root, serial_run_names)
    tests_files = serial._inventory(tests_root, serial_test_names)
    owner_raw = _read_file(owner_path, JSON_LIMIT)
    closeout_raw = _read_file(closeout_path, JSON_LIMIT)
    report_raw = _read_file(run_root / "report.json", JSON_LIMIT)
    receipt_raw = _read_file(tests_root / "receipt.json", JSON_LIMIT)
    pytest_raw = _read_file(tests_root / "pytest.log", LOG_LIMIT)
    tests_report_raw = _read_file(tests_root / "report.json", JSON_LIMIT)
    owner, closeout = _validate_serial_predecessor(
        owner_raw,
        closeout_raw,
        run_files,
        tests_files,
        report_raw,
        receipt_raw,
        pytest_raw,
        tests_report_raw,
        original,
        {
            key: value
            for key, value in concurrent.items()
            if key not in ("fixture", "fixture_source")
        },
    )
    serial_outputs = _read_file(run_root / "outputs.bin", serial.OUTPUT_BYTES)
    serial_analysis = serial_control.analyze(serial_outputs, concurrent["fixture"])
    require(
        serial_analysis == owner["result"],
        "historical serial 32-output analysis independently recomputed from authenticated bytes",
    )
    return {
        "predecessors_authenticated": True,
        "original": original,
        "concurrent": concurrent,
        "serial_owner": owner,
        "serial_closeout": closeout,
        "serial_run_files": run_files,
        "serial_test_files": tests_files,
        "serial_owner_packet_sha256": OWNER_PACKET_SHA256,
        "serial_closeout_packet_sha256": CLOSEOUT_PACKET_SHA256,
    }


def _read_tests(source, binding):
    root = output_path(source, "tests")
    base._exact_inventory(root, TEST_FILES_ON_DISK)
    receipt_raw = _read_file(root / "receipt.json", JSON_LIMIT)
    receipt = base.parse_json(receipt_raw)
    receipt_keys = {
        "protocol",
        "source",
        "source_binding",
        "test_files",
        "passed",
        "skips",
        "pytest_sha256",
        "service_properties",
        *FLAGS,
    }
    require(
        type(receipt) is dict
        and len(receipt) == len(receipt_keys)
        and set(receipt) == receipt_keys
        and receipt.get("protocol") == PROTOCOL + ":tests"
        and receipt.get("source") == source
        and receipt.get("source_binding") == binding
        and receipt.get("test_files") == list(TEST_FILES)
        and type(receipt.get("passed")) is int
        and receipt.get("passed") == EXPECTED_TESTS
        and EXPECTED_TESTS > 0
        and type(receipt.get("skips")) is int
        and receipt.get("skips") == 0
        and all(receipt.get(flag) is False for flag in FLAGS),
        "exact owner-frozen full-tree test receipt",
    )
    log = _read_file(root / "pytest.log", LOG_LIMIT)
    require(
        base.digest(log) == receipt.get("pytest_sha256"),
        "full-tree pytest log whole-byte digest",
    )
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
    require(
        len(totals) == 1
        and int(totals[0][0]) == EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
        "full-tree CPU test exact count without skips",
    )
    _terminal(source, "tests", receipt["service_properties"]["InvocationID"])
    report = base.parse_json(_read_file(root / "report.json", JSON_LIMIT))
    report_keys = {
        "protocol",
        "source",
        "mode",
        "status",
        "source_binding",
        "host_identity",
        "service_properties",
        "passed",
        "skips",
        "pytest_sha256",
        "elapsed_seconds",
        "idle_before",
        "idle_after",
        "filmbrain",
        "protected_services",
        *FLAGS,
    }
    require(
        type(report) is dict
        and len(report) == len(report_keys)
        and set(report) == report_keys
        and report.get("protocol") == PROTOCOL + ":tests"
        and report.get("source") == source
        and report.get("mode") == "tests"
        and report.get("status") == "passed"
        and report.get("source_binding") == binding
        and report.get("service_properties") == receipt["service_properties"]
        and type(report.get("passed")) is int
        and report.get("passed") == EXPECTED_TESTS
        and type(report.get("skips")) is int
        and report.get("skips") == 0
        and report.get("pytest_sha256") == receipt["pytest_sha256"]
        and _valid_elapsed(report.get("elapsed_seconds"), TEST_CAP)
        and all(report.get(flag) is False for flag in FLAGS),
        "full-tree tests report independently matches receipt",
    )
    return base.digest(receipt_raw)


def _valid_elapsed(value, cap):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value < cap


def _fixture_raws(fixture_root):
    baseline = _read_file(fixture_root / "baseline.bin", MATRIX_BYTES)
    return (
        baseline,
        baseline,
        _read_file(fixture_root / "capture-crb.bin", MATRIX_BYTES),
        _read_file(fixture_root / "replay-crb.bin", MATRIX_BYTES),
    )


def _rebuild_fixture_from_files(fixture_root, declaration):
    capture_cinert, replay_cinert, capture_crb, replay_crb = _fixture_raws(fixture_root)
    topology = declaration.get("topology")
    require(
        type(topology) is dict
        and base.digest((canonical(topology) + "\n").encode()) == LAUNCH_TOPOLOGY_SHA256
        and declaration.get("topology_sha256") == LAUNCH_TOPOLOGY_SHA256,
        "fixture declaration exact launch topology digest",
    )
    value = full_tree.fixture(
        capture_cinert,
        replay_cinert,
        capture_crb,
        replay_crb,
        topology,
        LAUNCH_TOPOLOGY_SHA256,
    )
    baseline_raw = _read_file(fixture_root / "baseline.bin", MATRIX_BYTES)
    expected_raw = _read_file(fixture_root / "expected.bin", MATRIX_BYTES)
    require(
        value["baseline"].astype("<f4", copy=False).tobytes(order="C") == baseline_raw
        and value["expected"].astype("<f4", copy=False).tobytes(order="C")
        == expected_raw
        and value["metadata"] == declaration.get("fixture_metadata")
        and value["plan"] == declaration.get("plan"),
        "fixture CPU prediction and all publication bytes independently recomputed",
    )
    require(
        base.digest(baseline_raw) == BASELINE_SHA256
        and base.digest(expected_raw) == EXPECTED_SHA256,
        "full-tree baseline and fixed-nine-group CPU prediction match independently pinned whole-byte hashes",
    )
    return value


def _read_fixture(source, binding):
    root = output_path(source, "fixture")
    base._exact_inventory(root, FIXTURE_FILES)
    inventory = _inventory(root, FIXTURE_FILES)
    report_raw = _read_file(root / "report.json", JSON_LIMIT)
    report = base.parse_json(report_raw)
    declaration_raw = _read_file(root / "declaration.json", JSON_LIMIT)
    declaration = base.parse_json(declaration_raw)
    retained_fixture_artifacts = {
        name: inventory[name]
        for name in (
            "baseline.bin",
            "capture-crb.bin",
            "replay-crb.bin",
            "expected.bin",
        )
    }
    report_keys = {
        "protocol",
        "source",
        "mode",
        "status",
        "source_binding",
        "host_identity",
        "service_properties",
        "test_receipt_sha256",
        "declaration_sha256",
        "fixture_metadata",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "fixture_artifacts",
        "idle_before",
        "idle_after",
        "filmbrain",
        "protected_services",
        "elapsed_seconds",
        *FLAGS,
    }
    declaration_keys = {
        "protocol",
        "source",
        "source_binding",
        "host_identity",
        "service_properties",
        "tests_receipt_sha256",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "topology",
        "topology_sha256",
        "plan",
        "fixture_metadata",
        "fixture_artifacts",
        "schedule",
        "smooth_sha256",
        "runtime_versions",
        "event",
        "fixture_kind",
        "torch_cuda_initialized",
        "cause_proven",
        *FLAGS,
    }
    require(
        type(report) is dict
        and len(report) == len(report_keys)
        and set(report) == report_keys
        and report.get("protocol") == PROTOCOL + ":fixture"
        and report.get("source") == source
        and report.get("mode") == "fixture"
        and report.get("status") == "passed"
        and report.get("source_binding") == binding
        and report.get("fixture_artifacts") == retained_fixture_artifacts
        and report.get("declaration_sha256") == base.digest(declaration_raw)
        and _valid_elapsed(report.get("elapsed_seconds"), FIXTURE_CAP)
        and all(report.get(flag) is False for flag in FLAGS),
        "fixture report is exact retained CPU-only successful artifact record",
    )
    require(
        type(declaration) is dict
        and set(declaration) == declaration_keys
        and declaration.get("protocol") == PROTOCOL + ":fixture"
        and declaration.get("source") == source
        and declaration.get("source_binding") == binding
        and declaration.get("tests_receipt_sha256") == report.get("test_receipt_sha256")
        and declaration.get("predecessor_binding_sha256")
        == report.get("predecessor_binding_sha256")
        and declaration.get("predecessor_summary") == report.get("predecessor_summary")
        and declaration.get("fixture_metadata") == report.get("fixture_metadata")
        and declaration.get("service_properties") == report.get("service_properties")
        and declaration.get("host_identity") == report.get("host_identity")
        and declaration.get("smooth_sha256") == SMOOTH_SHA256
        and declaration.get("runtime_versions") == _runtime_versions()
        and declaration.get("fixture_artifacts") == retained_fixture_artifacts
        and all(declaration.get(flag) is False for flag in FLAGS),
        "fixture declaration source/provenance/runtime binding",
    )
    properties = declaration["service_properties"]
    require(
        properties["RuntimeMaxUSec"] == "5min"
        and properties["MemoryMax"] == str(MEMORY)
        and properties["CPUQuotaPerSecUSec"] == CPU_QUOTA
        and properties["Nice"] == NICE
        and properties["KillMode"] == KILL_MODE,
        "fixture owner capped service record",
    )
    _terminal(source, "fixture", properties["InvocationID"])
    value = _rebuild_fixture_from_files(root, declaration)
    return {
        "root": root,
        "inventory": inventory,
        "report": report,
        "declaration": declaration,
        "fixture": value,
    }


def _monitor(samples):
    require(
        type(samples) is list and 0 < len(samples) <= 121,
        "nonempty sole-child GPU monitor samples",
    )
    row_keys = {"elapsed_seconds", "child_pid", "sample"}
    for row in samples:
        require(
            type(row) is dict and len(row) == 3 and set(row) == row_keys,
            "exact bounded monitor row",
        )
        require(
            type(row["child_pid"]) is int and row["child_pid"] > 0,
            "positive plain monitor PID",
        )
    pids = {row["child_pid"] for row in samples}
    require(
        len(pids) == 1 and type(next(iter(pids))) is int and next(iter(pids)) > 0,
        "monitor samples bind one actual positive child PID",
    )
    child_pid = next(iter(pids))
    expected_services = set(base.gpu_idle_gate.SERVICES) | {
        "user:" + name for name in base.gpu_idle_gate.SERVICES
    }
    previous = -1.0
    for row in samples:
        elapsed = row["elapsed_seconds"]
        require(
            type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and previous < elapsed < CHILD_SECONDS
            and elapsed >= 0,
            "ordered bounded monitor elapsed seconds",
        )
        previous = elapsed
        sample = row["sample"]
        require(
            type(sample) is dict
            and len(sample) == 5
            and set(sample)
            == {
                "temperature_c",
                "memory_used_mib",
                "memory_free_mib",
                "compute_pids",
                "services",
            }
            and type(sample["temperature_c"]) is int
            and 0 <= sample["temperature_c"] < 75
            and type(sample["memory_used_mib"]) is int
            and 0 <= sample["memory_used_mib"] <= 5120
            and type(sample["memory_free_mib"]) is int
            and sample["memory_free_mib"] >= 6144
            and type(sample["compute_pids"]) is list
            and len(sample["compute_pids"]) <= 1
            and all(
                type(pid) is int and pid == child_pid for pid in sample["compute_pids"]
            )
            and type(sample["services"]) is dict
            and len(sample["services"]) == 4
            and set(sample["services"]) == expected_services
            and all(state == "inactive" for state in sample["services"].values()),
            "monitored GPU stays under thermal/VRAM cap with no foreign work or protected-service activity",
        )
    return child_pid


def _live_owner(source, owner_pid):
    service = _properties(source, "run", owner_pid)
    require(os.getppid() == owner_pid, "actual full-tree child parent PID")
    return service


def _child(source, lease_fd, owner_pid, declaration_sha):
    base.training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "full-tree child sees only GPU zero",
    )
    service = _live_owner(source, owner_pid)
    binding = source_binding(source)
    root = output_path(source, "run")
    declaration_raw = _read_file(root / "declaration.json", JSON_LIMIT)
    require(
        base.digest(declaration_raw) == declaration_sha,
        "full-tree declaration whole hash",
    )
    declaration = base.parse_json(declaration_raw)
    require(
        type(declaration) is dict
        and set(declaration) == _RUN_DECL_KEYS
        and declaration.get("protocol") == PROTOCOL + ":run"
        and declaration.get("source") == source
        and declaration.get("source_binding") == binding
        and declaration.get("service_properties") == service
        and declaration.get("schedule")
        == {
            "concurrent": declaration.get("concurrent_levels"),
            "serial": declaration.get("serial_groups"),
        }
        and all(declaration.get(flag) is False for flag in FLAGS),
        "full-tree child declaration exact source/service/schedule",
    )
    require(
        base.host.identity(source) == declaration["host_identity"],
        "pinned host/Python identity before CUDA",
    )
    smooth_raw = _read_file(
        base.execution.ROOT / frozen.SMOOTH_LEAF, repeat.SMOOTH_LIMIT
    )
    require(
        base.digest(smooth_raw) == SMOOTH_SHA256
        and smooth._crb_accumulate is KERNEL
        and _runtime_versions() == declaration["runtime_versions"],
        "pinned smooth source/kernel and package versions before CUDA",
    )
    baseline_raw = _read_file(root / "baseline.bin", MATRIX_BYTES)
    expected_baseline_sha = declaration["fixture_artifacts"]["baseline.bin"]["sha256"]
    require(
        base.digest(baseline_raw)
        == expected_baseline_sha
        == declaration["fixture_metadata"]["baseline_sha256"],
        "baseline immutable input hash before CUDA",
    )
    for name in ("capture-crb.bin", "replay-crb.bin", "expected.bin"):
        raw = _read_file(root / name, MATRIX_BYTES)
        require(
            base.digest(raw) == declaration["fixture_artifacts"][name]["sha256"]
            and bool(np.isfinite(np.frombuffer(raw, dtype="<f4")).all()),
            f"full-tree child finite input {name} hash before CUDA",
        )
    baseline_np = (
        np.frombuffer(baseline_raw, dtype="<f4")
        .reshape(repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
        .copy()
    )
    parents_np = np.asarray(declaration["topology"]["body_parentid"], dtype=np.int32)
    expected_plan = planner.plan(declaration["topology"], LAUNCH_TOPOLOGY_SHA256)
    require(
        parents_np.shape == (repeat.BODIES,)
        and declaration["topology_sha256"] == LAUNCH_TOPOLOGY_SHA256
        and declaration["concurrent_levels"]
        == declaration["topology"]["reversed_body_tree_ids"]
        and declaration["serial_groups"]
        == [group["body_ids"] for group in expected_plan["launch_groups"]]
        and bool(np.isfinite(baseline_np).all()),
        "full-tree finite fixed baseline dimensions and topology schedule",
    )
    require(
        not torch.cuda.is_initialized(),
        "Torch CUDA stays uninitialized before Warp GPU work",
    )
    rng_before = torch.random.get_rng_state().clone()
    device = wp.get_device("cuda:0")
    require(device.is_cuda, "Warp CUDA device zero")
    concurrent_raw, serial_raw = _full_tree_batch(
        device,
        baseline_np,
        parents_np,
        declaration["concurrent_levels"],
        declaration["serial_groups"],
    )
    rng_after = torch.random.get_rng_state().clone()
    require(
        torch.equal(rng_before, rng_after) and not torch.cuda.is_initialized(),
        "Torch CPU RNG preserved and Torch CUDA uninitialized",
    )
    require(
        len(concurrent_raw) == OUTPUT_BYTES and len(serial_raw) == OUTPUT_BYTES,
        "exact 32 full outputs for each schedule",
    )
    rng_raw = rng_before.numpy().tobytes() + rng_after.numpy().tobytes()
    base._write_exclusive(root / "concurrent.bin", concurrent_raw, OUTPUT_BYTES)
    base._write_exclusive(root / "serial.bin", serial_raw, OUTPUT_BYTES)
    base._write_exclusive(root / "rng.bin", rng_raw, RNG_LIMIT)
    receipt = {
        "protocol": PROTOCOL + ":child",
        "source": source,
        "source_binding": binding,
        "declaration_sha256": declaration_sha,
        "owner_pid": owner_pid,
        "child_pid": os.getpid(),
        "child_ppid": os.getppid(),
        "service_properties": service,
        "host_identity": declaration["host_identity"],
        "runtime_versions": _runtime_versions(),
        "smooth_sha256": SMOOTH_SHA256,
        "launch": {
            "kernel": KERNEL_NAME,
            "worlds": repeat.WORLDS,
            "repeats_per_schedule": repeat.REPEATS,
            "concurrent_levels": declaration["concurrent_levels"],
            "serial_groups": declaration["serial_groups"],
            "concurrent_calls_per_repeat": 7,
            "serial_calls_per_repeat": 9,
            "total_launch_calls": repeat.REPEATS * 16,
            "same_working_input_output": True,
            "same_stream": True,
            "reset_once_before_each_full_schedule": True,
            "snapshot_after_complete_schedule": True,
            "sync_once_after_both_banks": True,
            "readback_after_sync_only": True,
            "parent_zero_noops_preserved": [1, 0],
        },
        "torch_rng_preserved": True,
        "torch_rng_state_bytes": int(rng_before.numel()),
        "torch_cuda_initialized": False,
        "actor_constructed": False,
        "model_constructed": False,
        "optimizer_constructed": False,
        "random_kernels": False,
        "concurrent_sha256": base.digest(concurrent_raw),
        "serial_sha256": base.digest(serial_raw),
        "rng_sha256": base.digest(rng_raw),
        **FLAGS,
    }
    base.write_json(root / "child-receipt.json", receipt)
    require(
        _live_owner(source, owner_pid) == service,
        "full-tree child remains under exact parent service",
    )
    return receipt


def _predecessor_summary(authenticated, order_inputs):
    require(
        authenticated.get("predecessors_authenticated") is True,
        "authenticated predecessor set",
    )
    concurrent = authenticated["concurrent"]
    return {
        "legacy_original": authenticated["original"],
        "concurrent": {
            key: value
            for key, value in concurrent.items()
            if key not in ("fixture", "fixture_source")
        },
        "serial_owner_packet_sha256": authenticated["serial_owner_packet_sha256"],
        "serial_closeout_packet_sha256": authenticated["serial_closeout_packet_sha256"],
        "serial_run_files": authenticated["serial_run_files"],
        "serial_test_files": authenticated["serial_test_files"],
        "order_input_inventory": order_inputs[3],
        "order_anchors": order_inputs[4],
    }


def _derive_fixture(authenticated, order_inputs, inputs, traces, topology):
    require(
        authenticated.get("predecessors_authenticated") is True,
        "fifty-file authentication before CPU tensor reads",
    )
    require(
        type(order_inputs) in (tuple, list)
        and len(order_inputs) == 5
        and order_inputs[1]["source"] == order.PARENT
        and order_inputs[1]["schedule"]["worlds"] == repeat.WORLDS,
        "original rollout inputs and 64-world launch authenticated",
    )
    require(
        type(inputs) is list
        and len(inputs) == 2
        and type(traces) is list
        and len(traces) == 2,
        "both full rollout attempts CPU-scored and traced",
    )
    capture_cinert, replay_cinert, capture_crb, replay_crb = _extract_observed_fixture(
        authenticated, traces
    )
    topology_raw = (canonical(topology) + "\n").encode()
    topology_sha = base.digest(topology_raw)
    require(
        topology_sha == LAUNCH_TOPOLOGY_SHA256,
        "fresh compiled launch topology exact digest",
    )
    fixture_value = full_tree.fixture(
        capture_cinert, replay_cinert, capture_crb, replay_crb, topology, topology_sha
    )
    baseline_raw = (
        fixture_value["baseline"].astype("<f4", copy=False).tobytes(order="C")
    )
    expected_raw = (
        fixture_value["expected"].astype("<f4", copy=False).tobytes(order="C")
    )
    require(
        fixture_value["metadata"]["input_sha256"]["capture_cinert"]
        == fixture_value["metadata"]["input_sha256"]["replay_cinert"]
        and baseline_raw == capture_cinert == replay_cinert,
        "full-tree baseline binds byte-identical authenticated complete-forward cinert inputs",
    )
    require(
        base.digest(baseline_raw) == BASELINE_SHA256
        and base.digest(expected_raw) == EXPECTED_SHA256,
        "full-tree completed-forward baseline and nine-group prediction exact pinned hashes",
    )
    return fixture_value, baseline_raw, capture_crb, replay_crb


def _write_fixture(
    source,
    binding,
    host_identity,
    service,
    tests_receipt_sha,
    predecessor,
    order_inputs,
    fixture_value,
    baseline_raw,
    capture_crb_raw,
    replay_crb_raw,
    started,
    idle_before,
    filmbrain,
    protected,
):
    root = output_path(source, "fixture")
    fixture_inventory = {
        "baseline.bin": {
            "bytes": len(baseline_raw),
            "sha256": base.digest(baseline_raw),
        },
        "capture-crb.bin": {
            "bytes": len(capture_crb_raw),
            "sha256": base.digest(capture_crb_raw),
        },
        "replay-crb.bin": {
            "bytes": len(replay_crb_raw),
            "sha256": base.digest(replay_crb_raw),
        },
        "expected.bin": {
            "bytes": MATRIX_BYTES,
            "sha256": base.digest(
                fixture_value["expected"].astype("<f4", copy=False).tobytes(order="C")
            ),
        },
    }
    predecessor_summary = _predecessor_summary(predecessor, order_inputs)
    predecessor_sha = base.digest((canonical(predecessor_summary) + "\n").encode())
    plan = fixture_value["plan"]
    declaration = {
        "protocol": PROTOCOL + ":fixture",
        "source": source,
        "source_binding": binding,
        "host_identity": host_identity,
        "service_properties": service,
        "tests_receipt_sha256": tests_receipt_sha,
        "predecessor_summary": predecessor_summary,
        "predecessor_binding_sha256": predecessor_sha,
        "topology": fixture_value["topology"],
        "topology_sha256": LAUNCH_TOPOLOGY_SHA256,
        "plan": plan,
        "fixture_metadata": fixture_value["metadata"],
        "fixture_artifacts": fixture_inventory,
        "schedule": {
            "concurrent": fixture_value["topology"]["reversed_body_tree_ids"],
            "serial": [group["body_ids"] for group in plan["launch_groups"]],
        },
        "smooth_sha256": SMOOTH_SHA256,
        "runtime_versions": _runtime_versions(),
        "event": {"index": 0, "phase": "scheduled-pre", "step": 0},
        "fixture_kind": "derived-completed-forward-full-cinert",
        "torch_cuda_initialized": False,
        "cause_proven": False,
        **FLAGS,
    }
    declaration_raw = (canonical(declaration) + "\n").encode()
    base._write_exclusive(root / "declaration.json", declaration_raw, JSON_LIMIT)
    base._write_exclusive(root / "baseline.bin", baseline_raw, MATRIX_BYTES)
    base._write_exclusive(root / "capture-crb.bin", capture_crb_raw, MATRIX_BYTES)
    base._write_exclusive(root / "replay-crb.bin", replay_crb_raw, MATRIX_BYTES)
    expected_raw = (
        fixture_value["expected"].astype("<f4", copy=False).tobytes(order="C")
    )
    base._write_exclusive(root / "expected.bin", expected_raw, MATRIX_BYTES)
    return {
        "protocol": PROTOCOL + ":fixture",
        "source": source,
        "mode": "fixture",
        "status": "passed",
        "source_binding": binding,
        "host_identity": host_identity,
        "service_properties": service,
        "test_receipt_sha256": tests_receipt_sha,
        "declaration_sha256": base.digest(declaration_raw),
        "fixture_metadata": fixture_value["metadata"],
        "predecessor_summary": predecessor_summary,
        "predecessor_binding_sha256": predecessor_sha,
        "fixture_artifacts": fixture_inventory,
        "idle_before": idle_before,
        "filmbrain": filmbrain,
        "protected_services": protected,
        "elapsed_seconds": float(time.monotonic() - started),
        **FLAGS,
    }


def execute(source, mode):
    require(mode in ("tests", "fixture", "run"), "declared full-tree mode")
    prior._hidden()
    require(
        base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "exact WSL native profile",
    )
    reserve = {"tests": TEST_RESERVE, "fixture": FIXTURE_RESERVE, "run": RUN_RESERVE}[
        mode
    ]
    _check_window(reserve_seconds=reserve)
    service = _properties(source, mode, os.getpid())
    binding = source_binding(source)
    host_identity = base.host.identity(source)
    output = output_path(source, mode)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "protocol": PROTOCOL + ":" + mode,
        "source": source,
        "mode": mode,
        "status": "failed-retained",
        "source_binding": binding,
        "host_identity": host_identity,
        "service_properties": service,
        **FLAGS,
    }
    try:
        with base.gap.base.files.gpu_lease() as lease_fd:
            idle_before, protected = _read_idle_and_protected()
            filmbrain = base.gap.base.retained.d0.filmbrain_state()
            require(
                not torch.cuda.is_initialized(),
                "supervisor Torch CUDA remains uninitialized",
            )
            if mode == "tests":
                require(
                    EXPECTED_TESTS > 0,
                    "owner freezes positive 51-file CUDA-hidden count before pytest",
                )
                prior._run_process(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *["tests/" + name for name in TEST_FILES],
                    ],
                    output / "pytest.log",
                    TEST_CAP - 30,
                    env=_cpu_env(),
                    fd=lease_fd,
                )
                log = _read_file(output / "pytest.log", LOG_LIMIT)
                totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
                require(
                    len(totals) == 1
                    and int(totals[0][0]) == EXPECTED_TESTS
                    and not re.search(
                        rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log
                    ),
                    "exact full-tree source-only suite, zero skips",
                )
                receipt = {
                    "protocol": PROTOCOL + ":tests",
                    "source": source,
                    "source_binding": binding,
                    "test_files": list(TEST_FILES),
                    "passed": EXPECTED_TESTS,
                    "skips": 0,
                    "pytest_sha256": base.digest(log),
                    "service_properties": service,
                    **FLAGS,
                }
                base.write_json(output / "receipt.json", receipt)
                report.update(
                    status="passed",
                    passed=EXPECTED_TESTS,
                    skips=0,
                    pytest_sha256=receipt["pytest_sha256"],
                )
            elif mode == "fixture":
                tests_receipt_sha = _read_tests(source, binding)
                predecessor = _authenticate_predecessors()
                order_inputs = order.authenticate_inputs()
                require(
                    order_inputs[1]["source"] == order.PARENT
                    and order_inputs[1]["schedule"]["worlds"] == repeat.WORLDS,
                    "authenticated 64-world rollout source",
                )
                rng_before = torch.random.get_rng_state().clone()
                inputs, _scores, traces = prior._score_inputs(
                    order_inputs[0], order.PARENT, order_inputs[1], order_inputs[2]
                )
                try:
                    prior._strict_pair(inputs, traces)
                except ValueError as error:
                    require(
                        str(error) == "paired rollout semantic state exactness",
                        "exact frozen original pair failure before fixture derivation",
                    )
                else:
                    require(
                        False,
                        "original capture/replay semantic state mismatch remains the declared precondition",
                    )
                require(
                    torch.equal(rng_before, torch.random.get_rng_state())
                    and not torch.cuda.is_initialized(),
                    "CPU fixture source loading preserves Torch RNG and CUDA remains uninitialized",
                )
                topology = crb._fresh_topology(repeat.WORLDS)
                fixture_value, baseline_raw, capture_crb_raw, replay_crb_raw = (
                    _derive_fixture(predecessor, order_inputs, inputs, traces, topology)
                )
                require(
                    torch.equal(rng_before, torch.random.get_rng_state())
                    and not torch.cuda.is_initialized(),
                    "CPU topology/fixture extraction preserves Torch RNG and CUDA remains uninitialized",
                )
                require(
                    time.monotonic() - started < FIXTURE_CAP,
                    "full-tree CPU fixture within 300-second cap",
                )
                refreshed = _authenticate_predecessors()
                require(
                    _predecessor_summary(refreshed, order.authenticate_inputs())
                    == _predecessor_summary(predecessor, order_inputs),
                    "all fifty predecessor files and order inputs unchanged after CPU prediction",
                )
                report = _write_fixture(
                    source,
                    binding,
                    host_identity,
                    service,
                    tests_receipt_sha,
                    predecessor,
                    order_inputs,
                    fixture_value,
                    baseline_raw,
                    capture_crb_raw,
                    replay_crb_raw,
                    started,
                    idle_before,
                    filmbrain,
                    protected,
                )
            else:
                tests_receipt_sha = _read_tests(source, binding)
                predecessor = _authenticate_predecessors()
                order_inputs = order.authenticate_inputs()
                predecessor_summary = _predecessor_summary(predecessor, order_inputs)
                fixture_record = _read_fixture(source, binding)
                declaration = fixture_record["declaration"]
                require(
                    declaration["tests_receipt_sha256"] == tests_receipt_sha
                    and declaration["predecessor_binding_sha256"]
                    == base.digest((canonical(predecessor_summary) + "\n").encode())
                    and declaration["predecessor_summary"] == predecessor_summary,
                    "full-tree fixture binds all fifty authenticated predecessor files and own tests",
                )
                require(
                    not torch.cuda.is_initialized(),
                    "Torch CUDA uninitialized before fixed-input child",
                )
                fixture_files = {
                    name: fixture_record["inventory"][name]
                    for name in (
                        "baseline.bin",
                        "capture-crb.bin",
                        "replay-crb.bin",
                        "expected.bin",
                    )
                }
                for name, limit in (
                    ("baseline.bin", MATRIX_BYTES),
                    ("capture-crb.bin", MATRIX_BYTES),
                    ("replay-crb.bin", MATRIX_BYTES),
                    ("expected.bin", MATRIX_BYTES),
                ):
                    base._write_exclusive(
                        output / name,
                        _read_file(fixture_record["root"] / name, limit),
                        limit,
                    )
                expected_raw = _read_file(
                    fixture_record["root"] / "expected.bin", MATRIX_BYTES
                )
                concurrent_levels = declaration["schedule"]["concurrent"]
                serial_groups = declaration["schedule"]["serial"]
                expected_schedule = {
                    "concurrent": declaration["topology"]["reversed_body_tree_ids"],
                    "serial": [
                        group["body_ids"]
                        for group in declaration["plan"]["launch_groups"]
                    ],
                }
                require(
                    declaration["schedule"] == expected_schedule
                    and len(concurrent_levels) == 7
                    and len(serial_groups) == 9
                    and serial_groups
                    == [
                        group["body_ids"]
                        for group in planner.plan(
                            declaration["topology"], LAUNCH_TOPOLOGY_SHA256
                        )["launch_groups"]
                    ],
                    "full-tree two schedules are exact original seven levels and planned nine groups",
                )
                declaration_run = {
                    "protocol": PROTOCOL + ":run",
                    "source": source,
                    "source_binding": binding,
                    "host_identity": host_identity,
                    "service_properties": service,
                    "tests_receipt_sha256": tests_receipt_sha,
                    "fixture_declaration_sha256": base.digest(
                        _read_file(
                            fixture_record["root"] / "declaration.json", JSON_LIMIT
                        )
                    ),
                    "fixture_report_sha256": base.digest(
                        _read_file(fixture_record["root"] / "report.json", JSON_LIMIT)
                    ),
                    "fixture_artifacts": fixture_files,
                    "fixture_metadata": declaration["fixture_metadata"],
                    "topology": declaration["topology"],
                    "topology_sha256": LAUNCH_TOPOLOGY_SHA256,
                    "concurrent_levels": concurrent_levels,
                    "serial_groups": serial_groups,
                    "schedule": {
                        "concurrent": concurrent_levels,
                        "serial": serial_groups,
                    },
                    "runtime_versions": _runtime_versions(),
                    "smooth_sha256": SMOOTH_SHA256,
                    "expected_sha256": base.digest(expected_raw),
                    "predecessor_summary": predecessor_summary,
                    **FLAGS,
                }
                declaration_raw = (canonical(declaration_run) + "\n").encode()
                base._write_exclusive(
                    output / "declaration.json", declaration_raw, JSON_LIMIT
                )
                refreshed = _authenticate_predecessors()
                require(
                    _predecessor_summary(refreshed, order.authenticate_inputs())
                    == predecessor_summary
                    and source_binding(source) == binding
                    and base.host.identity(source) == host_identity,
                    "all fifty predecessor files and exact source/runtime unchanged before GPU child",
                )
                command = [
                    sys.executable,
                    "-m",
                    MODULE,
                    "child",
                    "--source",
                    source,
                    "--lease-fd",
                    str(lease_fd),
                    "--owner-pid",
                    str(os.getpid()),
                    "--declaration-sha",
                    base.digest(declaration_raw),
                ]
                samples = prior._run_process(
                    command,
                    output / "child.log",
                    CHILD_SECONDS,
                    env=_gpu_env(),
                    fd=lease_fd,
                    monitor=True,
                )
                cuda_probe.check_log(output / "child.log")
                base._exact_inventory(output, RUN_FILES - {"report.json"})
                inventory = _inventory(output, RUN_FILES - {"report.json"})
                receipt = base.parse_json(
                    _read_file(output / "child-receipt.json", JSON_LIMIT)
                )
                _validate_child_receipt(
                    receipt,
                    source,
                    binding,
                    declaration_run,
                    base.digest(declaration_raw),
                    service,
                    host_identity,
                )
                child_pid = _monitor(samples)
                require(
                    receipt.get("child_pid") == child_pid
                    and receipt.get("owner_pid") == os.getpid()
                    and receipt.get("child_ppid") == os.getpid(),
                    "live monitor and receipt authenticate actual parent/child PID chain",
                )
                concurrent_raw = _read_file(output / "concurrent.bin", OUTPUT_BYTES)
                serial_raw = _read_file(output / "serial.bin", OUTPUT_BYTES)
                rng_raw = _read_file(output / "rng.bin", RNG_LIMIT)
                state_bytes = receipt.get("torch_rng_state_bytes")
                require(
                    type(state_bytes) is int
                    and state_bytes > 0
                    and len(rng_raw) == state_bytes * 2
                    and rng_raw[:state_bytes] == rng_raw[state_bytes:]
                    and base.digest(concurrent_raw) == receipt["concurrent_sha256"]
                    and base.digest(serial_raw) == receipt["serial_sha256"]
                    and base.digest(rng_raw) == receipt["rng_sha256"],
                    "whole GPU outputs and equal before/after RNG states authenticated before analysis",
                )
                current_fixture = fixture_record["fixture"]
                expected_repeat = expected_raw * repeat.REPEATS
                concurrent_summary = full_tree.analyze(
                    concurrent_raw, current_fixture, schedule="concurrent"
                )
                serial_summary = full_tree.analyze(
                    serial_raw, current_fixture, schedule="serial"
                )
                refreshed = _authenticate_predecessors()
                require(
                    _predecessor_summary(refreshed, order.authenticate_inputs())
                    == predecessor_summary,
                    "all fifty predecessor files unchanged after GPU child and decode",
                )
                result = {
                    "tests_receipt_sha256": tests_receipt_sha,
                    "predecessor_summary": predecessor_summary,
                    "fixture_declaration_sha256": declaration_run[
                        "fixture_declaration_sha256"
                    ],
                    "child_receipt": receipt,
                    "artifacts": inventory,
                    "concurrent_analysis": concurrent_summary,
                    "serial_analysis": serial_summary,
                    "concurrent_matches_fixed_prediction_all_repeats": concurrent_raw
                    == expected_repeat,
                    "serial_matches_fixed_prediction_all_repeats": serial_raw
                    == expected_repeat,
                    "monitor_samples": samples,
                    "child_gpu_ownership_observed_in_monitor": any(
                        child_pid in sample["sample"]["compute_pids"]
                        for sample in samples
                    ),
                }
                require(
                    time.monotonic() - started < CAP,
                    "full-tree parent within 600-second cap",
                )
                idle_after = base.gpu_idle_gate.wait_idle()
                require(
                    source_binding(source) == binding
                    and base.host.identity(source) == host_identity
                    and _properties(source, "run", os.getpid()) == service
                    and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                    and base.gap.base.protected_state() == protected,
                    "unchanged source/runtime, owner service and protected context after fixed-input child",
                )
                report.update(
                    status="passed",
                    result=result,
                    idle_before=idle_before,
                    idle_after=idle_after,
                    filmbrain=filmbrain,
                    protected_services=protected,
                )
            mode_cap = {"tests": TEST_CAP, "fixture": FIXTURE_CAP, "run": CAP}[mode]
            require(
                time.monotonic() - started < mode_cap,
                "full-tree mode remains inside its fixed cap",
            )
            require(
                source_binding(source) == binding
                and base.host.identity(source) == host_identity
                and _properties(source, mode, os.getpid()) == service
                and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                and base.gap.base.protected_state() == protected,
                "source, host, owner service and protected context unchanged at mode closeout",
            )
            idle_after = base.gpu_idle_gate.wait_idle()
            report.update(
                idle_before=idle_before,
                idle_after=idle_after,
                filmbrain=filmbrain,
                protected_services=protected,
            )
    except BaseException as error:
        report.update(
            error_type=type(error).__name__,
            error=str(error),
            partial_artifacts=_partial_inventory(output),
        )
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(output / "report.json", report)
        if report.get("status") == "passed":
            names = (
                TEST_FILES_ON_DISK
                if mode == "tests"
                else FIXTURE_FILES
                if mode == "fixture"
                else RUN_FILES
            )
            _inventory(output, names)
    return report


def _validate_child_receipt(
    receipt, source, binding, declaration, declaration_sha, service, host_identity
):
    keys = _CHILD_KEYS
    expected_launch = {
        "kernel": KERNEL_NAME,
        "worlds": repeat.WORLDS,
        "repeats_per_schedule": repeat.REPEATS,
        "concurrent_levels": declaration["concurrent_levels"],
        "serial_groups": declaration["serial_groups"],
        "concurrent_calls_per_repeat": 7,
        "serial_calls_per_repeat": 9,
        "total_launch_calls": repeat.REPEATS * 16,
        "same_working_input_output": True,
        "same_stream": True,
        "reset_once_before_each_full_schedule": True,
        "snapshot_after_complete_schedule": True,
        "sync_once_after_both_banks": True,
        "readback_after_sync_only": True,
        "parent_zero_noops_preserved": [1, 0],
    }
    require(
        type(receipt) is dict
        and set(receipt) == keys
        and receipt.get("protocol") == PROTOCOL + ":child"
        and receipt.get("source") == source
        and receipt.get("source_binding") == binding
        and receipt.get("declaration_sha256") == declaration_sha
        and receipt.get("service_properties") == service
        and receipt.get("host_identity") == host_identity
        and receipt.get("runtime_versions") == declaration["runtime_versions"]
        and receipt.get("smooth_sha256") == SMOOTH_SHA256
        and receipt.get("launch") == expected_launch
        and receipt.get("torch_rng_preserved") is True
        and receipt.get("torch_cuda_initialized") is False
        and receipt.get("actor_constructed") is False
        and receipt.get("model_constructed") is False
        and receipt.get("optimizer_constructed") is False
        and receipt.get("random_kernels") is False
        and all(receipt.get(flag) is False for flag in FLAGS),
        "exact non-admitting full-tree child receipt",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tests", "fixture", "run", "child"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--declaration-sha")
    args = parser.parse_args()
    if args.mode == "child":
        require(
            args.lease_fd is not None
            and args.owner_pid is not None
            and args.declaration_sha is not None,
            "exact private full-tree child arguments",
        )
        result = _child(
            args.source, args.lease_fd, args.owner_pid, args.declaration_sha
        )
    else:
        require(
            args.lease_fd is None
            and args.owner_pid is None
            and args.declaration_sha is None,
            "no inherited child arguments in owner process",
        )
        result = execute(args.source, args.mode)
    print(
        canonical(
            {
                "protocol": result.get("protocol"),
                "source": result.get("source"),
                "status": result.get("status", "passed"),
            }
        ),
        flush=True,
    )


def _read_idle_and_protected():
    idle = base.gpu_idle_gate.wait_idle()
    protected = base.gap.base.protected_state()
    require(
        bool(protected) and all(state == "inactive" for state in protected.values()),
        "both protected service namespaces inactive",
    )
    return idle, protected


def _serial_roots():
    return serial.output_path(SERIAL_SOURCE, "run"), serial.output_path(
        SERIAL_SOURCE, "tests"
    )


def _serial_packet_roots():
    return (
        base.execution.ROOT / OWNER_PACKET_PATH,
        base.execution.ROOT / CLOSEOUT_PACKET_PATH,
    )


def _authenticate_serial_source(binding):
    root = Path(__file__).resolve().parents[2]
    paths = set(serial._frozen_paths()) | set(serial.OWN) | set(serial.RELATED)
    require(
        type(binding) is dict
        and set(binding) == {"source", "branch", "leaves"}
        and binding.get("source") == SERIAL_SOURCE
        and type(binding.get("leaves")) is dict,
        "pinned historical serial source binding schema",
    )
    for path in sorted(paths):
        raw = subprocess.run(
            ["git", "show", f"{SERIAL_SOURCE}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
        require(
            binding["leaves"].get(path) == base.digest(raw),
            "historical serial whole source leaf " + path,
        )
        if path in serial._frozen_paths():
            frozen_raw = subprocess.run(
                ["git", "show", f"{serial.BASE_SOURCE}:{path}"],
                cwd=root,
                check=True,
                capture_output=True,
                timeout=15,
            ).stdout
            require(raw == frozen_raw, "historical serial frozen source leaf " + path)
    require(
        set(binding["leaves"]) == paths, "historical serial binding exact full leaf set"
    )


def _extract_observed_fixture(authenticated, trace_pair):
    require(
        type(authenticated) is dict
        and authenticated.get("predecessors_authenticated") is True,
        "predecessor fifty-file authentication precedes tensor interpretation",
    )
    require(
        type(trace_pair) in (tuple, list) and len(trace_pair) == 2,
        "exact capture/replay trace pair",
    )
    raws = []
    for attempt, trace in zip(("capture", "replay"), trace_pair, strict=True):
        event = trace["events"][0]
        require(
            event["phase"] == "scheduled-pre" and event["step"] == 0,
            f"{attempt} first trace event is scheduled-pre step zero",
        )
        persistent = event["persistent"]
        arrays = []
        for name in ("cinert", "crb"):
            value = persistent[name]
            require(
                torch.is_tensor(value)
                and value.device.type == "cpu"
                and value.dtype == torch.float32
                and tuple(value.shape)
                == (repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
                and bool(torch.isfinite(value).all()),
                f"{attempt} finite CPU full-tree {name}",
            )
            contiguous = value.detach().contiguous()
            raw = contiguous.numpy().astype("<f4", copy=False).tobytes(order="C")
            require(len(raw) == MATRIX_BYTES, f"{attempt} complete {name} byte matrix")
            arrays.append(raw)
        raws.extend(arrays)
    return raws[0], raws[2], raws[1], raws[3]


def _full_tree_batch(device, baseline_np, parents_np, concurrent_levels, serial_groups):
    """Run both 32-repeat schedules without any intermediate readback."""
    with wp.ScopedDevice(device):
        parents = wp.array(parents_np, dtype=wp.int32, device=device)
        concurrent = [
            (
                wp.array(
                    np.asarray(level, dtype=np.int32), dtype=wp.int32, device=device
                ),
                len(level),
            )
            for level in concurrent_levels
        ]
        serial_groups_device = [
            (
                wp.array(
                    np.asarray(group, dtype=np.int32), dtype=wp.int32, device=device
                ),
                len(group),
            )
            for group in serial_groups
        ]
        baseline = wp.array(baseline_np, dtype=WP_VEC10, device=device)
        working = wp.empty(
            shape=(repeat.WORLDS, repeat.BODIES), dtype=WP_VEC10, device=device
        )
        concurrent_bank = [
            wp.empty(
                shape=(repeat.WORLDS, repeat.BODIES), dtype=WP_VEC10, device=device
            )
            for _ in range(repeat.REPEATS)
        ]
        serial_bank = [
            wp.empty(
                shape=(repeat.WORLDS, repeat.BODIES), dtype=WP_VEC10, device=device
            )
            for _ in range(repeat.REPEATS)
        ]
        stream = wp.get_stream(device)
        for bank, groups in (
            (concurrent_bank, concurrent),
            (serial_bank, serial_groups_device),
        ):
            for index in range(repeat.REPEATS):
                wp.copy(working, baseline, stream=stream)
                for ids, group_size in groups:
                    wp.launch(
                        KERNEL,
                        dim=(repeat.WORLDS, group_size),
                        inputs=[parents, working, ids],
                        outputs=[working],
                        device=device,
                        stream=stream,
                    )
                wp.copy(bank[index], working, stream=stream)
        wp.synchronize_device(device)
        concurrent_raw = (
            np.stack([array.numpy() for array in concurrent_bank])
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        serial_raw = (
            np.stack([array.numpy() for array in serial_bank])
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        return concurrent_raw, serial_raw


if __name__ == "__main__":
    main()
