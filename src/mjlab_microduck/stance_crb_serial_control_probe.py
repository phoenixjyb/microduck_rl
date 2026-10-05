"""Capped supervisor for the separately scheduled serial CRB control.

This diagnostic retains 32 full fixed-input results from three sequential
single-body kernel launches. It is not a runtime rollout or cause proof.
"""

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
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
from mjlab_microduck import stance_crb_serial_control as control
from mjlab_microduck import stance_cuda_probe as cuda_probe
from mjlab_microduck import stance_inertia_order_probe as order
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
MODULE = "mjlab_microduck.stance_crb_serial_control_probe"
PROTOCOL = "football-b1d-crb-serial-schedule-control-probe-20261006-v1"
BASE_SOURCE = "bed9f99feb4dca91d78a99f00b5cdba0361933aa"
SYNC_FROM_SOURCE = "b72cf952a265ac69bbe6affec470f39dbf3ad042"
CONCURRENT_SOURCE = SYNC_FROM_SOURCE
CONCURRENT_REPORT_SHA256 = (
    "2f1e363c3281703cf7a5e583ede2bfcc569d7220a611a66283e14cdaa20c9492"
)
CONCURRENT_TESTS_RECEIPT_SHA256 = (
    "1b71fc3388c39ca117d87d846ac487ea74662a399663700486f907fdf9ad3516"
)
OWNER_PACKET_SHA256 = "82c3c4110113f245bb3302a5354c6194176b30ed81769af6aa9405fddae0d1d1"
OWNER_PACKET_PATH = "artifacts/tools/stance-crb-kernel-owner-terminal-b72cf952a265.json"
CONCURRENT_RUN_INVOCATION = "5c69382e97b14515b616ca4b881671f2"
CONCURRENT_TESTS_INVOCATION = "7c5c21aad5824b189964597d72fa94e4"
BASELINE_SHA256 = frozen.BASELINE_SHA256
LAUNCH_TOPOLOGY_SHA256 = frozen.LAUNCH_TOPOLOGY_SHA256
SMOOTH_SHA256 = frozen.SMOOTH_SHA
CAP = 600
TEST_CAP = 300
CHILD_SECONDS = 240
MEMORY = 6 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
MARGIN = 60
RUN_RESERVE = CAP + TEST_CAP + MARGIN
TEST_RESERVE = TEST_CAP + CAP + TEST_CAP + MARGIN
CUTOFF = datetime(2026, 10, 6, 8, 0, tzinfo=timezone(timedelta(hours=8))).timestamp()
EXPECTED_TESTS = 1436  # Owner-reviewed 48-file CUDA-hidden suite, no skips.
DOC = "docs/experiments/2026-10-06-crb-serial-schedule-control.md"
TEST_FILES = frozen.TEST_FILES + (
    "test_stance_crb_serial_control.py",
    "test_stance_crb_serial_control_probe.py",
)
OWN = (
    "src/mjlab_microduck/stance_crb_serial_control_probe.py",
    "tests/test_stance_crb_serial_control_probe.py",
    DOC,
)
RELATED = (
    "src/mjlab_microduck/stance_crb_serial_control.py",
    "tests/test_stance_crb_serial_control.py",
)
ALLOWED = set(OWN) | set(RELATED)
FLAGS = {
    **frozen.FLAGS,
    **{key: False for key, value in repeat.FLAGS.items() if value is False},
    "serial_schedule_control_completed": False,
    "serial_schedule_proves_original_cause": False,
    "actual_atomic_order_observed": False,
    "full_native_window_qualified": False,
    "training_authorized": False,
}
KERNEL = smooth._crb_accumulate
KERNEL_NAME = "_crb_accumulate"
JSON_LIMIT = 2 * 1024**2
LOG_LIMIT = 1024**2
BASELINE_BYTES = control.WORLDS * control.BODIES * control.COMPONENTS * 4
OUTPUT_BYTES = control.RAW_BYTES
RNG_BYTES = 64 * 1024
TOTAL_BYTES = 8 * 1024**2
RUN_FILES = {
    "declaration.json",
    "baseline.bin",
    "outputs.bin",
    "rng.bin",
    "child-receipt.json",
    "report.json",
    "child.log",
}
TEST_FILES_ON_DISK = {"receipt.json", "pytest.log", "report.json"}
_PACKET_KEYS = {
    "filmbrain",
    "gpu_initialized",
    "host_identity",
    "idle",
    "original",
    "protected_services",
    "protocol",
    "result",
    "rollout_cause_proven",
    "run_files",
    "source",
    "source_binding",
    "terminal_run",
    "terminal_tests",
    "tests_files",
    "tests_receipt_sha256",
    "training_authorized",
}
_TERMINAL_KEYS = {
    "ActiveState",
    "SubState",
    "MainPID",
    "NRestarts",
    "ExecMainStatus",
    "Result",
    "RemainAfterExit",
    "MemoryMax",
    "CPUQuotaPerSecUSec",
    "Nice",
    "KillMode",
    "RuntimeMaxUSec",
    "InvocationID",
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
    "outputs_sha256",
    "rng_sha256",
    *FLAGS,
}


def unit(source, mode):
    base._hex(source, 40, "exact serial-control source")
    require(mode in ("tests", "run"), "declared serial-control mode")
    return f"microduck-crb-serial-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    area = "tools" if mode == "tests" else "evaluations"
    return (
        base.execution.ROOT
        / "artifacts"
        / area
        / f"stance-crb-serial-{mode}-{source[:12]}"
    )


def source_sync_unit(source):
    base._hex(source, 40, "exact serial-control bootstrap source")
    return f"microduck-crb-serial-sync-{source[:12]}.service"


def source_sync_script(source, *, bundle_sha256):
    script = frozen.source_sync_script(source, bundle_sha256=bundle_sha256)
    changes = (
        (
            f'test "$(git rev-parse HEAD)" = {frozen.SYNC_FROM_SOURCE}\n',
            f'test "$(git rev-parse HEAD)" = {SYNC_FROM_SOURCE}\n',
        ),
        (
            f'if test "$duck_sync_running" != {frozen.source_sync_unit(source)}; then\n',
            f'if test "$duck_sync_running" != {source_sync_unit(source)}; then\n',
        ),
    )
    for old, new in changes:
        require(script.count(old) == 1, "one exact frozen sync guard occurrence")
        script = script.replace(old, new, 1)
    return script


def _frozen_paths():
    return (
        set(frozen._frozen_paths())
        | set(frozen.OWN)
        | set(frozen.RELATED)
        | set(order.OWN)
    )


def source_binding(source):
    base._hex(source, 40, "exact committed serial-control source")
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
        "clean exact native serial-control source branch and origin",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE, source],
        cwd=root,
        check=True,
        timeout=15,
    )
    changed = set(git("diff", "--name-only", BASE_SOURCE, source).decode().splitlines())
    require(changed <= ALLOWED, "only the five declared serial-control paths changed")
    frozen_paths = _frozen_paths()
    leaves = {}
    for path in sorted(frozen_paths | set(OWN) | set(RELATED)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(
            raw == git("show", f"{source}:{path}"),
            "exact committed serial-control leaf " + path,
        )
        if path in frozen_paths:
            require(
                raw == git("show", f"{BASE_SOURCE}:{path}"),
                "frozen starting source leaf " + path,
            )
        leaves[path] = base.digest(raw)
    return {
        "source": source,
        "branch": "feat/athletics-obstacle-curriculum",
        "leaves": leaves,
    }


def _check_window(*, reserve_seconds, now=None):
    require(
        type(reserve_seconds) is int and reserve_seconds > 0, "positive reserved window"
    )
    require(
        (time.time() if now is None else now) + reserve_seconds < CUTOFF,
        "serial-control cutoff window",
    )


def _recorded_properties(value, mode):
    return frozen._recorded_properties(value, mode)


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
    return frozen._inventory(root, names)


def _partial_inventory(root):
    return frozen._partial_inventory(root)


def _authenticate_original():
    # Reuses only the legacy immutable 27-file CPU authenticator. It does not
    # inspect the cleared concurrent probe's current service or source binding.
    return frozen._authenticate_original()


def _exact_service(source, mode, owner_pid):
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
    service = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in keys
    }
    _recorded_properties(service, mode)
    require(
        service["MainPID"] == str(owner_pid),
        "actual supervisor owns retained service MainPID",
    )
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
        "only serial-control Duck service running",
    )
    return service


def _terminal_packet(value, mode):
    require(
        type(value) is dict and set(value) == _TERMINAL_KEYS,
        "exact recorded terminal service property keys",
    )
    runtime = "5min" if mode == "tests" else "10min"
    base._hex(value["InvocationID"], 32, "recorded terminal invocation ID")
    expected_invocation = (
        CONCURRENT_TESTS_INVOCATION if mode == "tests" else CONCURRENT_RUN_INVOCATION
    )
    require(
        value
        == {
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "Result": "success",
            "RemainAfterExit": "yes",
            "MemoryMax": str(MEMORY),
            "CPUQuotaPerSecUSec": CPU_QUOTA,
            "Nice": NICE,
            "KillMode": KILL_MODE,
            "RuntimeMaxUSec": runtime,
            "InvocationID": expected_invocation,
        },
        "exact successful terminal historical owner invocation",
    )


def _validate_concurrent_packet(
    packet_raw,
    report_raw,
    run_files,
    tests_files,
    tests_receipt_raw,
    tests_log_raw,
    tests_report_raw,
    original,
):
    require(
        base.digest(packet_raw) == OWNER_PACKET_SHA256,
        "independent concurrent owner packet whole-file hash",
    )
    require(
        base.digest(report_raw) == CONCURRENT_REPORT_SHA256,
        "pinned concurrent run report hash",
    )
    packet = base.parse_json(packet_raw)
    report = base.parse_json(report_raw)
    require(
        type(packet) is dict and set(packet) == _PACKET_KEYS,
        "exact independent owner packet schema",
    )
    require(
        packet["protocol"] == frozen.PROTOCOL + ":independent-owner-closeout"
        and packet["source"] == CONCURRENT_SOURCE
        and packet["original"] == original
        and packet["rollout_cause_proven"] is False
        and packet["training_authorized"] is False
        and packet["gpu_initialized"] is False,
        "owner packet binds cleared concurrent diagnostic and original 27 files",
    )
    require(
        run_files == packet["run_files"] and tests_files == packet["tests_files"],
        "all seven run and three test files match independent terminal packet",
    )
    require(
        run_files["report.json"]
        == {"bytes": len(report_raw), "sha256": base.digest(report_raw)}
        and tests_files["receipt.json"]
        == {"bytes": len(tests_receipt_raw), "sha256": base.digest(tests_receipt_raw)}
        and tests_files["pytest.log"]
        == {"bytes": len(tests_log_raw), "sha256": base.digest(tests_log_raw)}
        and tests_files["report.json"]
        == {"bytes": len(tests_report_raw), "sha256": base.digest(tests_report_raw)},
        "packet inventory binds every independently read report, receipt and log byte",
    )
    require(
        base.digest(tests_receipt_raw)
        == packet["tests_receipt_sha256"]
        == CONCURRENT_TESTS_RECEIPT_SHA256,
        "pinned concurrent CPU tests receipt whole-file hash",
    )
    tests_receipt = base.parse_json(tests_receipt_raw)
    require(
        type(tests_receipt) is dict
        and set(tests_receipt)
        == {
            "protocol",
            "source",
            "source_binding",
            "test_files",
            "passed",
            "skips",
            "pytest_sha256",
            "service_properties",
            *frozen.FLAGS,
        }
        and tests_receipt.get("protocol") == frozen.PROTOCOL + ":tests"
        and tests_receipt.get("source") == CONCURRENT_SOURCE
        and tests_receipt.get("source_binding") == packet["source_binding"]
        and tests_receipt.get("test_files") == list(frozen.TEST_FILES)
        and tests_receipt.get("passed") == frozen.EXPECTED_TESTS
        and tests_receipt.get("skips") == 0
        and base.digest(tests_log_raw) == tests_receipt.get("pytest_sha256")
        and all(tests_receipt.get(key) is False for key in frozen.FLAGS),
        "concurrent test receipt remains a source-only pass",
    )
    _recorded_properties(tests_receipt["service_properties"], "tests")
    require(
        tests_receipt["service_properties"]["InvocationID"]
        == CONCURRENT_TESTS_INVOCATION,
        "concurrent CPU test service invocation binds packet",
    )
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", tests_log_raw)
    require(
        len(totals) == 1
        and int(totals[0][0]) == frozen.EXPECTED_TESTS
        and not re.search(
            rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", tests_log_raw
        ),
        "concurrent CPU test log exact positive count and zero skips",
    )
    tests_report = base.parse_json(tests_report_raw)
    require(
        tests_report.get("protocol") == frozen.PROTOCOL + ":tests"
        and tests_report.get("source") == CONCURRENT_SOURCE
        and tests_report.get("mode") == "tests"
        and tests_report.get("status") == "passed"
        and tests_report.get("source_binding") == packet["source_binding"]
        and tests_report.get("service_properties")
        == tests_receipt["service_properties"]
        and tests_report.get("passed") == frozen.EXPECTED_TESTS
        and tests_report.get("pytest_sha256") == tests_receipt["pytest_sha256"]
        and all(tests_report.get(key) is False for key in frozen.FLAGS),
        "concurrent CPU test report matches whole receipt",
    )
    require(
        report.get("protocol") == frozen.PROTOCOL + ":run"
        and report.get("source") == CONCURRENT_SOURCE
        and report.get("status") == "passed"
        and report.get("source_binding") == packet["source_binding"]
        and report.get("result") == packet["result"]
        and report.get("service_properties", {}).get("InvocationID")
        == CONCURRENT_RUN_INVOCATION
        and report.get("host_identity") == packet["host_identity"]
        and report.get("filmbrain") == packet["filmbrain"]
        and report.get("protected_services") == packet["protected_services"]
        and all(report.get(key) is False for key in frozen.FLAGS),
        "concurrent report and owner packet agree without a cause claim",
    )
    require(
        packet["terminal_run"].get("InvocationID") == CONCURRENT_RUN_INVOCATION
        and packet["terminal_tests"].get("InvocationID") == CONCURRENT_TESTS_INVOCATION,
        "packet binds both recorded terminal invocations",
    )
    _terminal_packet(packet["terminal_run"], "run")
    _terminal_packet(packet["terminal_tests"], "tests")
    require(
        type(packet["result"]) is dict
        and packet["result"].get("bit_variation_observed") is True,
        "concurrent 32-repeat control demonstrates retained bit variation",
    )
    idle = packet["idle"]
    require(
        type(idle) is dict
        and idle.get("protocol") == "two-idle-samples-v1"
        and type(idle.get("samples")) is list
        and len(idle["samples"]) == 2
        and all(
            sample.get("compute_pids") == ""
            and type(sample.get("temperature_c")) is int
            and sample["temperature_c"] < 75
            and type(sample.get("memory_mib")) is int
            and sample["memory_mib"] <= 5120
            and all(
                state == "inactive" for state in sample.get("services", {}).values()
            )
            for sample in idle["samples"]
        ),
        "owner packet retained clean idle GPU and protected-service samples",
    )
    return packet, report


def _concurrent_roots():
    return (
        base.execution.ROOT
        / "artifacts/evaluations/stance-crb-kernel-run-b72cf952a265",
        base.execution.ROOT / "artifacts/tools/stance-crb-kernel-tests-b72cf952a265",
    )


def _authenticate_concurrent(original):
    run_root, tests_root = _concurrent_roots()
    packet_path = base.execution.ROOT / OWNER_PACKET_PATH
    packet_raw = _read_file(packet_path, JSON_LIMIT)
    base._exact_inventory(run_root, RUN_FILES)
    base._exact_inventory(tests_root, frozen.TEST_FILES_ON_DISK)
    run_files = _inventory(run_root, RUN_FILES)
    tests_files = _inventory(tests_root, frozen.TEST_FILES_ON_DISK)
    report_raw = _read_file(run_root / "report.json", JSON_LIMIT)
    tests_log_raw = _read_file(tests_root / "pytest.log", LOG_LIMIT)
    tests_report_raw = _read_file(tests_root / "report.json", JSON_LIMIT)
    tests_receipt = _read_file(tests_root / "receipt.json", JSON_LIMIT)
    packet, report = _validate_concurrent_packet(
        packet_raw,
        report_raw,
        run_files,
        tests_files,
        tests_receipt,
        tests_log_raw,
        tests_report_raw,
        original,
    )
    outputs_raw = _read_file(run_root / "outputs.bin", OUTPUT_BYTES)
    oracle_root = order.output_path(frozen.ORDER_SOURCE, "run")
    oracle_raw = _read_file(oracle_root / "comparison.json", JSON_LIMIT)
    topology = crb._fresh_topology(64)
    require(
        base.digest((canonical(topology) + "\n").encode()) == LAUNCH_TOPOLOGY_SHA256,
        "concurrent exact CPU topology",
    )
    fixture_value = repeat.fixture(oracle_raw, topology)
    require(
        base.digest(fixture_value["baseline"].astype("<f4", copy=False).tobytes())
        == BASELINE_SHA256,
        "concurrent fixture baseline reauthenticated",
    )
    analysis = repeat.analyze(outputs_raw, fixture_value)
    require(
        analysis == packet["result"] and analysis.get("bit_variation_observed") is True,
        "concurrent 32-result variation independently reproduced before serial control",
    )
    return {
        "packet_sha256": OWNER_PACKET_SHA256,
        "run_files": run_files,
        "tests_files": tests_files,
        "report_sha256": CONCURRENT_REPORT_SHA256,
        "result": analysis,
        "source_binding": packet["source_binding"],
        "terminal_run": packet["terminal_run"],
        "terminal_tests": packet["terminal_tests"],
        "fixture": fixture_value,
        "fixture_source": oracle_raw,
    }


def _live_owner(source, owner_pid):
    service = _exact_service(source, "run", owner_pid)
    require(os.getppid() == owner_pid, "actual serial child parent PID")
    return service


def _serial_batch(device, baseline_np, parents_np):
    """Run 32 reset/triplets, each [2],[7],[11], on one stream."""
    with wp.ScopedDevice(device):
        parents = wp.array(parents_np, dtype=wp.int32, device=device)
        levels = [
            wp.array(np.asarray([body], dtype=np.int32), dtype=wp.int32, device=device)
            for body in control.SERIAL_ORDER
        ]
        baseline = wp.array(baseline_np, dtype=WP_VEC10, device=device)
        working = wp.empty(
            shape=(control.WORLDS, control.BODIES), dtype=WP_VEC10, device=device
        )
        output_bank = [
            wp.empty(
                shape=(control.WORLDS, control.BODIES), dtype=WP_VEC10, device=device
            )
            for _ in range(control.REPEATS)
        ]
        stream = wp.get_stream(device)
        for repeat_index in range(control.REPEATS):
            wp.copy(working, baseline, stream=stream)
            for level in levels:
                wp.launch(
                    KERNEL,
                    dim=(control.WORLDS, 1),
                    inputs=[parents, working, level],
                    outputs=[working],
                    device=device,
                    stream=stream,
                )
            wp.copy(output_bank[repeat_index], working, stream=stream)
        wp.synchronize_device(device)
        return (
            np.stack([array.numpy() for array in output_bank])
            .astype("<f4", copy=False)
            .tobytes()
        )


def _child(source, lease_fd, owner_pid, declaration_sha):
    base.training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "serial child sees only declared GPU",
    )
    service = _live_owner(source, owner_pid)
    binding = source_binding(source)
    root = output_path(source, "run")
    declaration_raw = _read_file(root / "declaration.json", JSON_LIMIT)
    require(
        base.digest(declaration_raw) == declaration_sha,
        "parent serial declaration exact hash",
    )
    declaration = base.parse_json(declaration_raw)
    require(
        declaration.get("source") == source
        and declaration.get("source_binding") == binding
        and declaration.get("service_properties") == service
        and declaration.get("schedule")
        == [list(level) for level in control.SERIAL_SCHEDULE],
        "serial declaration source/service/schedule",
    )
    require(
        base.host.identity(source) == declaration["host_identity"],
        "pinned host and Python trees before CUDA",
    )
    smooth_raw = _read_file(
        base.execution.ROOT / frozen.SMOOTH_LEAF, repeat.SMOOTH_LIMIT
    )
    require(
        base.digest(smooth_raw) == SMOOTH_SHA256
        and smooth._crb_accumulate is KERNEL
        and _runtime_versions() == declaration["runtime_versions"],
        "pinned smooth source, kernel identity and package versions before CUDA",
    )
    rng_before = torch.random.get_rng_state().clone()
    baseline_raw = _read_file(root / "baseline.bin", BASELINE_BYTES)
    require(
        base.digest(baseline_raw) == BASELINE_SHA256 == declaration["baseline_sha256"],
        "serial fixed baseline hash before CUDA",
    )
    baseline_np = (
        np.frombuffer(baseline_raw, dtype="<f4")
        .reshape(control.WORLDS, control.BODIES, control.COMPONENTS)
        .copy()
    )
    parents_np = np.asarray(declaration["topology"]["body_parentid"], dtype=np.int32)
    require(
        parents_np.shape == (control.BODIES,)
        and declaration["topology_sha256"] == LAUNCH_TOPOLOGY_SHA256
        and bool(np.isfinite(baseline_np).all()),
        "exact finite launch topology and baseline dimensions",
    )
    require(
        not torch.cuda.is_initialized(),
        "Torch CUDA remains uninitialized before Warp allocation",
    )
    device = wp.get_device("cuda:0")
    require(device.is_cuda, "Warp CUDA device zero")
    outputs_raw = _serial_batch(device, baseline_np, parents_np)
    rng_after = torch.random.get_rng_state().clone()
    require(
        torch.equal(rng_before, rng_after) and not torch.cuda.is_initialized(),
        "Torch CPU RNG preserved and Torch CUDA uninitialized",
    )
    require(len(outputs_raw) == OUTPUT_BYTES, "exact 32 full serial output matrices")
    rng_raw = rng_before.numpy().tobytes() + rng_after.numpy().tobytes()
    base._write_exclusive(root / "outputs.bin", outputs_raw, OUTPUT_BYTES)
    base._write_exclusive(root / "rng.bin", rng_raw, RNG_BYTES)
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
            "dim": [control.WORLDS, 1],
            "calls_per_repeat": 3,
            "repeats": control.REPEATS,
            "schedule": [list(level) for level in control.SERIAL_SCHEDULE],
            "same_input_output": True,
            "reset_once_before_triplet": True,
            "snapshot_after_triplet": True,
            "readback_after_batch": True,
        },
        "torch_rng_preserved": True,
        "torch_rng_state_bytes": int(rng_before.numel()),
        "torch_cuda_initialized": False,
        "actor_constructed": False,
        "model_constructed": False,
        "optimizer_constructed": False,
        "random_kernels": False,
        "outputs_sha256": base.digest(outputs_raw),
        "rng_sha256": base.digest(rng_raw),
        **FLAGS,
    }
    base.write_json(root / "child-receipt.json", receipt)
    require(
        _live_owner(source, owner_pid) == service,
        "serial child remains under exact active owner",
    )
    return receipt


def _terminal(source, mode, invocation):
    base._hex(invocation, 32, "serial service terminal invocation")
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
    actual = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in keys
    }
    require(
        actual
        == {
            "MainPID": "0",
            "ActiveState": "active",
            "SubState": "exited",
            "RemainAfterExit": "yes",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "Result": "success",
            "InvocationID": invocation,
            "RuntimeMaxUSec": "5min" if mode == "tests" else "10min",
            "MemoryMax": str(MEMORY),
            "CPUQuotaPerSecUSec": CPU_QUOTA,
            "Nice": NICE,
            "KillMode": KILL_MODE,
        },
        "successful terminal serial service invocation with exact resource caps",
    )
    return actual


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
        and set(receipt) == receipt_keys
        and receipt.get("protocol") == PROTOCOL + ":tests"
        and receipt.get("source") == source
        and receipt.get("source_binding") == binding
        and receipt.get("test_files") == list(TEST_FILES)
        and type(receipt.get("passed")) is int
        and receipt.get("passed") == EXPECTED_TESTS
        and EXPECTED_TESTS > 0
        and receipt.get("skips") == 0
        and receipt.get("service_properties")
        and all(receipt.get(key) is False for key in FLAGS),
        "exact owner-frozen serial CPU test receipt",
    )
    log = _read_file(root / "pytest.log", LOG_LIMIT)
    require(
        base.digest(log) == receipt["pytest_sha256"], "serial CPU test whole-log hash"
    )
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
    require(
        len(totals) == 1
        and int(totals[0][0]) == EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
        "exact serial CPU suite without skips",
    )
    _recorded_properties(receipt["service_properties"], "tests")
    _terminal(source, "tests", receipt["service_properties"]["InvocationID"])
    report = base.parse_json(_read_file(root / "report.json", JSON_LIMIT))
    require(
        type(report) is dict
        and set(report)
        == {
            "protocol",
            "source",
            "mode",
            "status",
            "source_binding",
            "service_properties",
            "host_identity",
            "result",
            "idle_before",
            "idle_after",
            "filmbrain",
            "protected_services",
            "elapsed_seconds",
            *FLAGS,
        }
        and report.get("protocol") == PROTOCOL + ":tests"
        and report.get("source") == source
        and report.get("mode") == "tests"
        and report.get("status") == "passed"
        and report.get("source_binding") == binding
        and report.get("service_properties") == receipt["service_properties"]
        and type(report.get("elapsed_seconds")) in (int, float)
        and 0 < report["elapsed_seconds"] < TEST_CAP
        and report.get("result")
        == {"passed": EXPECTED_TESTS, "pytest_sha256": receipt["pytest_sha256"]}
        and all(report.get(key) is False for key in FLAGS),
        "serial CPU test report matches receipt and mode cap",
    )
    return base.digest(receipt_raw)


def execute(source, mode):
    require(mode in ("tests", "run"), "declared serial-control mode")
    prior._hidden()
    require(
        base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "exact WSL runtime profile",
    )
    _check_window(reserve_seconds=TEST_RESERVE if mode == "tests" else RUN_RESERVE)
    service = _exact_service(source, mode, os.getpid())
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
        "service_properties": service,
        "host_identity": host_identity,
        **FLAGS,
    }
    try:
        with base.gap.base.files.gpu_lease() as lease_fd:
            idle_before = base.gpu_idle_gate.wait_idle()
            filmbrain = base.gap.base.retained.d0.filmbrain_state()
            protected = base.gap.base.protected_state()
            require(
                bool(protected)
                and all(value == "inactive" for value in protected.values()),
                "both protected service namespaces inactive",
            )
            if mode == "tests":
                require(
                    EXPECTED_TESTS > 0,
                    "owner-frozen positive CUDA-hidden suite count before pytest",
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
                    "exact owner-frozen serial CPU tests",
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
                result = {
                    "passed": EXPECTED_TESTS,
                    "pytest_sha256": receipt["pytest_sha256"],
                }
            else:
                tests_receipt_sha = _read_tests(source, binding)
                original = _authenticate_original()
                concurrent = _authenticate_concurrent(original)
                topology = crb._fresh_topology(64)
                require(
                    base.digest((canonical(topology) + "\n").encode())
                    == LAUNCH_TOPOLOGY_SHA256,
                    "exact fresh serial launch topology",
                )
                baseline = concurrent["fixture"]["baseline"]
                baseline_raw = baseline.astype("<f4", copy=False).tobytes()
                require(
                    base.digest(baseline_raw) == BASELINE_SHA256,
                    "serial fixture baseline matches owner-pinned bytes",
                )
                declaration = {
                    "protocol": PROTOCOL,
                    "source": source,
                    "source_binding": binding,
                    "host_identity": host_identity,
                    "service_properties": service,
                    "tests_receipt_sha256": tests_receipt_sha,
                    "legacy_original": original,
                    "concurrent_evidence": {
                        key: value
                        for key, value in concurrent.items()
                        if key not in ("fixture", "fixture_source")
                    },
                    "topology": topology,
                    "topology_sha256": LAUNCH_TOPOLOGY_SHA256,
                    "baseline_sha256": BASELINE_SHA256,
                    "fixture_metadata": concurrent["fixture"]["metadata"],
                    "fixture_kind": "derived-complete-forward-reduction-fixture",
                    "schedule": [list(level) for level in control.SERIAL_SCHEDULE],
                    "serial_order": list(control.SERIAL_ORDER),
                    "kernel_calls_per_repeat": control.KERNEL_CALLS_PER_REPEAT,
                    "repeats": control.REPEATS,
                    "raw_bytes": control.RAW_BYTES,
                    "runtime_versions": _runtime_versions(),
                    "smooth_sha256": SMOOTH_SHA256,
                    **FLAGS,
                }
                declaration_raw = (canonical(declaration) + "\n").encode()
                base._write_exclusive(
                    output / "declaration.json", declaration_raw, JSON_LIMIT
                )
                base._write_exclusive(
                    output / "baseline.bin", baseline_raw, BASELINE_BYTES
                )
                require(
                    _read_tests(source, binding) == tests_receipt_sha
                    and _authenticate_original() == original
                    and _authenticate_concurrent(original)["packet_sha256"]
                    == concurrent["packet_sha256"],
                    "all frozen tests and 38 predecessor files unchanged before serial child",
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
                require(
                    type(receipt) is dict
                    and set(receipt) == _CHILD_KEYS
                    and receipt.get("protocol") == PROTOCOL + ":child"
                    and receipt.get("source") == source
                    and receipt.get("source_binding") == binding
                    and receipt.get("declaration_sha256")
                    == base.digest(declaration_raw)
                    and receipt.get("owner_pid") == os.getpid()
                    and receipt.get("child_ppid") == os.getpid()
                    and receipt.get("service_properties") == service
                    and receipt.get("host_identity") == host_identity
                    and receipt.get("runtime_versions") == _runtime_versions()
                    and receipt.get("smooth_sha256") == SMOOTH_SHA256
                    and receipt.get("torch_rng_preserved") is True
                    and receipt.get("torch_cuda_initialized") is False
                    and receipt.get("actor_constructed") is False
                    and receipt.get("model_constructed") is False
                    and receipt.get("optimizer_constructed") is False
                    and receipt.get("random_kernels") is False
                    and all(receipt.get(key) is False for key in FLAGS),
                    "exact serial child receipt and non-admitting flags",
                )
                child_pids = {sample.get("child_pid") for sample in samples}
                require(
                    samples
                    and len(child_pids) == 1
                    and type(next(iter(child_pids))) is int
                    and next(iter(child_pids)) > 0
                    and receipt.get("child_pid") in child_pids,
                    "live monitor sample binds actual positive child PID",
                )
                require(
                    receipt.get("launch")
                    == {
                        "kernel": KERNEL_NAME,
                        "dim": [control.WORLDS, 1],
                        "calls_per_repeat": 3,
                        "repeats": control.REPEATS,
                        "schedule": [list(level) for level in control.SERIAL_SCHEDULE],
                        "same_input_output": True,
                        "reset_once_before_triplet": True,
                        "snapshot_after_triplet": True,
                        "readback_after_batch": True,
                    },
                    "receipt binds the exact serial launch sequence",
                )
                outputs_raw = _read_file(output / "outputs.bin", OUTPUT_BYTES)
                rng_raw = _read_file(output / "rng.bin", RNG_BYTES)
                state_bytes = receipt.get("torch_rng_state_bytes")
                require(
                    type(state_bytes) is int
                    and state_bytes > 0
                    and len(rng_raw) == state_bytes * 2
                    and rng_raw[:state_bytes] == rng_raw[state_bytes:]
                    and base.digest(outputs_raw) == receipt["outputs_sha256"]
                    and base.digest(rng_raw) == receipt["rng_sha256"]
                    and base.digest(_read_file(output / "baseline.bin", BASELINE_BYTES))
                    == BASELINE_SHA256,
                    "all raw serial outputs, equal RNG states and baseline hash authenticated before analysis",
                )
                summary = control.analyze(outputs_raw, concurrent["fixture"])
                fixture_after = repeat.fixture(
                    concurrent["fixture_source"], crb._fresh_topology(64)
                )
                require(
                    np.array_equal(
                        fixture_after["baseline"].view("u4"),
                        concurrent["fixture"]["baseline"].view("u4"),
                    )
                    and fixture_after["metadata"] == concurrent["fixture"]["metadata"]
                    and np.array_equal(
                        fixture_after["parents"], concurrent["fixture"]["parents"]
                    )
                    and np.array_equal(
                        fixture_after["level"], concurrent["fixture"]["level"]
                    ),
                    "independent immutable CPU fixture reconstruction after GPU child",
                )
                idle_after = base.gpu_idle_gate.wait_idle()
                require(
                    source_binding(source) == binding
                    and base.host.identity(source) == host_identity
                    and _exact_service(source, mode, os.getpid()) == service
                    and _authenticate_original() == original
                    and _authenticate_concurrent(original)["packet_sha256"]
                    == concurrent["packet_sha256"]
                    and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                    and base.gap.base.protected_state() == protected,
                    "unchanged source/runtime, old 38-file evidence, service and protected context",
                )
                result = {
                    "tests_receipt_sha256": tests_receipt_sha,
                    "legacy_original": original,
                    "concurrent": {
                        key: value
                        for key, value in concurrent.items()
                        if key not in ("fixture", "fixture_source")
                    },
                    "child_receipt": receipt,
                    "artifacts": inventory,
                    "analysis": summary,
                    "monitor_samples": samples,
                }
            require(
                time.monotonic() - started < (TEST_CAP if mode == "tests" else CAP),
                "serial parent remains inside mode-specific fixed cap",
            )
            idle_after = base.gpu_idle_gate.wait_idle()
            require(
                source_binding(source) == binding
                and base.host.identity(source) == host_identity
                and _exact_service(source, mode, os.getpid()) == service
                and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                and base.gap.base.protected_state() == protected,
                "unchanged source, owner service and protected context at parent closeout",
            )
            report.update(
                status="passed",
                result=result,
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
        if mode == "run" and report.get("status") == "passed":
            _inventory(output, RUN_FILES)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tests", "run", "child"))
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
            "exact serial child arguments",
        )
        result = _child(
            args.source, args.lease_fd, args.owner_pid, args.declaration_sha
        )
    else:
        require(
            args.lease_fd is None
            and args.owner_pid is None
            and args.declaration_sha is None,
            "no inherited args in supervisor mode",
        )
        result = execute(args.source, args.mode)
    print(
        canonical(
            {
                "protocol": result["protocol"],
                "source": result["source"],
                "status": result.get("status", "passed"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
