"""Bounded supervisor for the fixed-input CRB accumulation diagnostic.

The parent derives a synthetic CPU fixture from the authenticated retained
oracle; an isolated child launches only the pinned Warp kernel on that fixture.
This is a fixed-input kernel diagnostic, not a replay or rollout qualification.
"""

import argparse
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
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

from mjlab_microduck import stance_crb_archive_cpu_probe as cpu_probe
from mjlab_microduck import stance_crb_kernel_repeat as repeat
from mjlab_microduck import stance_crb_launch_trace as crb
from mjlab_microduck import stance_cuda_probe as cuda_probe
from mjlab_microduck import stance_inertia_order_probe as order
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
MODULE = "mjlab_microduck.stance_crb_kernel_probe"
PROTOCOL = "football-b1d-crb-fixed-input-kernel-probe-20261006-v1"
BASE_SOURCE = "2df4948e5af2c0e10636d61c8d70f0a685fd4d82"
SYNC_FROM_SOURCE = "f190d24dbb4b185f581462c36c0b2104a743cdfd"
ORDER_COMPARISON_SHA = (
    "7731633338071aeb1435c0dd613f1216cc1b528b15edc8e658cbcad8691fa45c"
)
ORDER_REPORT_SHA = "38d860e97cdbe898c560b8522c3191348984c778f8d772d899f6f02ef7a5517a"
ORDER_SOURCE = "766271542f4c168ebe00c56e8e03755c3f44be2b"
BASELINE_SHA256 = "554aa8cccf4dbede45ac8f03b3d6e9e8433b0f567f30da454ab2cacd754a80d6"
LAUNCH_TOPOLOGY_SHA256 = (
    "2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19"
)
SMOOTH_SHA = order.SMOOTH_SHA
SMOOTH_LEAF = order.SMOOTH_LEAF
CAP = 600
TEST_CAP = 300
MEMORY = 6 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
CHILD_SECONDS = 240
MARGIN = 60
RUN_RESERVE = CAP + TEST_CAP + MARGIN
TEST_RESERVE = TEST_CAP + CAP + TEST_CAP + MARGIN
EXPECTED_TESTS = 1415  # Owner-reviewed 46-file CUDA-hidden suite, no skips.
DOC = "docs/experiments/2026-10-06-crb-fixed-input-diagnostic.md"
TEST_FILES = cpu_probe.TEST_FILES + (
    "test_stance_crb_kernel_repeat.py",
    "test_stance_crb_kernel_probe.py",
)
OWN = (
    "src/mjlab_microduck/stance_crb_kernel_probe.py",
    "tests/test_stance_crb_kernel_probe.py",
    DOC,
)
RELATED = (
    "src/mjlab_microduck/stance_crb_kernel_repeat.py",
    "tests/test_stance_crb_kernel_repeat.py",
)
ALLOWED = set(OWN) | set(RELATED)
FLAGS = {
    **{name: False for name, value in order.FLAGS.items() if value is False},
    "fixed_input_kernel_diagnostic_passed": False,
    "native_atomic_order_observed": False,
    "rollout_cause_proven": False,
    "full_native_window_qualified": False,
}
KERNEL = smooth._crb_accumulate
KERNEL_NAME = repeat.KERNEL_NAME
TEST_FILES_ON_DISK = {"receipt.json", "pytest.log", "report.json"}
RUN_FILES = {
    "declaration.json",
    "baseline.bin",
    "outputs.bin",
    "rng.bin",
    "child-receipt.json",
    "report.json",
    "child.log",
}
_CHILD_RECEIPT_KEYS = {
    "protocol",
    "source",
    "source_binding",
    "declaration_sha256",
    "owner_pid",
    "child_pid",
    "child_ppid",
    "service_properties",
    "smooth_sha256",
    "runtime_versions",
    "launch",
    "cpu_rng_preserved",
    "rng_state_bytes",
    "torch_cuda_initialized",
    "actor_constructed",
    "model_constructed",
    "optimizer_constructed",
    "random_kernels",
    "outputs_sha256",
    "rng_sha256",
    *FLAGS,
}
JSON_LIMIT = 2 * 1024**2
LOG_LIMIT = 1024**2
BASELINE_BYTES = repeat.WORLDS * repeat.BODIES * repeat.COMPONENTS * 4
OUTPUT_BYTES = repeat.REPEATS * BASELINE_BYTES
RNG_BYTES = 64 * 1024
TOTAL_BYTES = 8 * 1024**2
CUTOFF = datetime(2026, 10, 6, 8, 0, tzinfo=timezone(timedelta(hours=8))).timestamp()


def unit(source, mode):
    base._hex(source, 40, "exact CRB kernel diagnostic source")
    require(mode in ("tests", "run"), "declared CRB kernel diagnostic mode")
    return f"microduck-crb-kernel-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    area = "tools" if mode == "tests" else "evaluations"
    return (
        base.execution.ROOT
        / "artifacts"
        / area
        / f"stance-crb-kernel-{mode}-{source[:12]}"
    )


def source_sync_unit(source):
    base._hex(source, 40, "exact CRB kernel bootstrap source")
    return f"microduck-crb-kernel-sync-{source[:12]}.service"


def source_sync_script(source, *, bundle_sha256):
    script = cpu_probe.source_sync_script(source, bundle_sha256=bundle_sha256)
    changes = (
        (
            f'test "$(git rev-parse HEAD)" = {cpu_probe.SYNC_FROM_SOURCE}\n',
            f'test "$(git rev-parse HEAD)" = {SYNC_FROM_SOURCE}\n',
        ),
        (
            f'if test "$duck_sync_running" != {cpu_probe.source_sync_unit(source)}; then\n',
            f'if test "$duck_sync_running" != {source_sync_unit(source)}; then\n',
        ),
    )
    for old, new in changes:
        require(script.count(old) == 1, "one exact frozen sync guard")
        script = script.replace(old, new, 1)
    return script


def _frozen_paths():
    return (
        set(cpu_probe.FROZEN_FILES)
        | (set(cpu_probe.OWN) - {cpu_probe.DOC})
        | (set(order.OWN) - {order.OWN[-1]})
    )


def source_binding(source):
    base._hex(source, 40, "exact committed CRB kernel source")
    root = Path(__file__).resolve().parents[2]
    require(str(root) == cpu_probe.SYNC_ROOT, "exact native WSL source root")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=15
        ).stdout

    require(
        git("rev-parse", "HEAD").decode().strip() == source
        and git("branch", "--show-current").decode().strip()
        == "feat/athletics-obstacle-curriculum"
        and git("remote", "get-url", "origin").decode().strip() == cpu_probe.SYNC_ORIGIN
        and not git("status", "--porcelain"),
        "clean exact native source branch and origin",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE, source],
        cwd=root,
        check=True,
        timeout=15,
    )
    changed = set(git("diff", "--name-only", BASE_SOURCE, source).decode().splitlines())
    require(changed <= ALLOWED, "only declared fixed-input kernel diagnostic leaves")
    leaves = {}
    for path in sorted(_frozen_paths() | set(OWN) | set(RELATED)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(raw == git("show", f"{source}:{path}"), "committed source leaf " + path)
        if path in _frozen_paths():
            require(
                raw == git("show", f"{BASE_SOURCE}:{path}"),
                "frozen source leaf " + path,
            )
        leaves[path] = base.digest(raw)
    return dict(
        source=source, branch="feat/athletics-obstacle-curriculum", leaves=leaves
    )


def _recorded_properties(value, mode):
    require(mode in ("tests", "run"), "declared service mode")
    expected = {
        "MainPID": None,
        "ActiveState": "active",
        "RuntimeMaxUSec": "5min" if mode == "tests" else "10min",
        "MemoryMax": str(MEMORY),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
        "InvocationID": None,
        "RemainAfterExit": "yes",
    }
    require(
        type(value) is dict and set(value) == set(expected),
        "exact service property schema",
    )
    require(
        type(value["MainPID"]) is str
        and value["MainPID"].isdecimal()
        and int(value["MainPID"]) > 0,
        "live service owner PID",
    )
    base._hex(value["InvocationID"], 32, "service invocation ID")
    require(
        {k: v for k, v in value.items() if k not in ("MainPID", "InvocationID")}
        == {k: v for k, v in expected.items() if k not in ("MainPID", "InvocationID")},
        "exact retained service caps",
    )
    return True


def _service_values(source, mode):
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
    values = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in keys
    }
    _recorded_properties(values, mode)
    running = base.host.read(
        "systemctl",
        "--user",
        "list-units",
        "--state=running",
        "--no-legend",
        "microduck*",
    )
    require(
        {row.split()[0] for row in running.splitlines()} == {unit(source, mode)},
        "only exact owned Duck service active",
    )
    return values


def _owned_properties(source, mode, owner_pid):
    values = _service_values(source, mode)
    require(values["MainPID"] == str(owner_pid), "service main PID is the active owner")
    return values


def _hidden():
    prior._hidden()
    require(
        base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "exact WSL host profile",
    )


def check_window(*, reserve_seconds, now=None):
    require(
        type(reserve_seconds) is int and reserve_seconds > 0,
        "positive closeout reserve",
    )
    require(
        (time.time() if now is None else now) + reserve_seconds < CUTOFF,
        "fixed-input diagnostic cutoff window",
    )


def runtime_versions():
    versions = {name: version(name) for name in cuda_probe.VERSIONS}
    require(versions == cuda_probe.VERSIONS, "frozen diagnostic runtime versions")
    return {"python": sys.version.split()[0], **versions}


def _cpu_env():
    env = prior.shadow.child_environment()
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
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    return env


def _authenticate_original():
    root, launch, launch_sha, inventory, anchors = order.authenticate_inputs()
    require(
        launch["source"] == order.PARENT and launch["schedule"]["worlds"] == 64,
        "authenticated original 64-world CRB inputs",
    )
    order_root = order.output_path(ORDER_SOURCE, "run")
    base._exact_inventory(order_root, {"comparison.json", "report.json"})
    comparison = base._read_file(order_root / "comparison.json", JSON_LIMIT)
    report = base._read_file(order_root / "report.json", JSON_LIMIT)
    require(
        base.digest(comparison) == ORDER_COMPARISON_SHA
        and base.digest(report) == ORDER_REPORT_SHA,
        "exact frozen order comparison and report hashes",
    )
    map_root = order.frozen_component.output_path(order.FREEZE_SOURCE, "run")
    map_files = _file_entries(map_root, ("comparison.json", "report.json"))
    require(
        map_files["comparison.json"]["sha256"] == anchors["comparison_sha256"]
        and map_files["report.json"]["sha256"] == anchors["report_sha256"],
        "component-map files match original anchors",
    )
    diagnosis_root = order.prior.output_path(order.PARENT)
    diagnosis_files = _file_entries(
        diagnosis_root, ("independent-failure-diagnosis.json",)
    )
    require(
        diagnosis_files["independent-failure-diagnosis.json"]["sha256"]
        == order.DIAGNOSIS_SHA,
        "exact original independent diagnosis hash",
    )
    order_files = _file_entries(order_root, ("comparison.json", "report.json"))
    all_original_files = {
        "rollout": inventory,
        "diagnosis": diagnosis_files,
        "component_map": map_files,
        "order_oracle": order_files,
    }
    require(
        sum(len(group) for group in all_original_files.values()) == 27,
        "all 27 original retained evidence files",
    )
    return dict(
        root=str(root),
        launch_sha256=launch_sha,
        original_files=all_original_files,
        anchors=anchors,
        comparison_sha256=base.digest(comparison),
        report_sha256=base.digest(report),
    )


def _file_entries(root, names):
    result = {}
    for name in names:
        raw = _read_file(root / name, JSON_LIMIT)
        result[name] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return result


def declaration(
    source, binding, original, topology, fixture_metadata, service, host_identity
):
    return {
        "protocol": PROTOCOL,
        "source": source,
        "source_binding": binding,
        "original": original,
        "topology": topology,
        "fixture_metadata": fixture_metadata,
        "service_properties": service,
        "host_identity": host_identity,
        "repeats": repeat.REPEATS,
        "worlds": repeat.WORLDS,
        "bodies": repeat.BODIES,
        "components": repeat.COMPONENTS,
        "raw_bytes": repeat.RAW_BYTES,
        "smooth_sha256": SMOOTH_SHA,
        **FLAGS,
    }


def _read_file(path, limit):
    return base._read_file(path, limit)


def _inventory(root, names):
    result = {}
    total = 0
    for name in sorted(names):
        cap = (
            LOG_LIMIT
            if name.endswith(".log")
            else JSON_LIMIT
            if name.endswith(".json")
            else RNG_BYTES
            if name == "rng.bin"
            else OUTPUT_BYTES
            if name == "outputs.bin"
            else BASELINE_BYTES
        )
        raw = _read_file(root / name, cap)
        total += len(raw)
        require(total <= TOTAL_BYTES, "total retained diagnostic artifact cap")
        result[name] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return result


def _partial_inventory(root):
    result = {}
    for name in sorted(RUN_FILES):
        path = root / name
        try:
            metadata = path.lstat()
        except OSError as error:
            result[name] = {
                "present": False,
                "readable": False,
                "error_type": type(error).__name__,
            }
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            result[name] = {
                "present": True,
                "readable": False,
                "error_type": "UnsafeFile",
            }
            continue
        try:
            raw = _read_file(
                path,
                LOG_LIMIT
                if name.endswith(".log")
                else JSON_LIMIT
                if name.endswith(".json")
                else OUTPUT_BYTES
                if name == "outputs.bin"
                else BASELINE_BYTES
                if name == "baseline.bin"
                else RNG_BYTES,
            )
        except BaseException as error:
            result[name] = {
                "present": True,
                "readable": False,
                "error_type": type(error).__name__,
            }
            continue
        result[name] = {
            "present": True,
            "readable": True,
            "bytes": len(raw),
            "sha256": base.digest(raw),
        }
    return result


def _declaration_sha(path):
    raw = _read_file(path, JSON_LIMIT)
    return base.digest(raw), raw


def _live_owner(source, mode, owner_pid):
    service = _owned_properties(source, mode, owner_pid)
    require(os.getppid() == owner_pid, "actual child parent PID matches service owner")
    return service


def _repeat_batch(device, baseline_np, parents_np, level_np):
    """One stream, aliased production kernel, no intermediate host observation."""
    with wp.ScopedDevice(device):
        parents = wp.array(parents_np, dtype=wp.int32, device=device)
        level = wp.array(level_np, dtype=wp.int32, device=device)
        baseline = wp.array(baseline_np, dtype=WP_VEC10, device=device)
        working = wp.empty(
            shape=(repeat.WORLDS, repeat.BODIES), dtype=WP_VEC10, device=device
        )
        output_bank = [
            wp.empty(
                shape=(repeat.WORLDS, repeat.BODIES), dtype=WP_VEC10, device=device
            )
            for _ in range(repeat.REPEATS)
        ]
        stream = wp.get_stream(device)
        for index in range(repeat.REPEATS):
            wp.copy(working, baseline, stream=stream)
            wp.launch(
                KERNEL,
                dim=(repeat.WORLDS, 3),
                inputs=[parents, working, level],
                outputs=[working],
                device=device,
                stream=stream,
            )
            wp.copy(output_bank[index], working, stream=stream)
        wp.synchronize_device(device)
        return (
            np.stack([array.numpy() for array in output_bank])
            .astype("<f4", copy=False)
            .tobytes()
        )


def _child(source, lease_fd, owner_pid, declaration_sha):
    base.training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "child sees exactly declared GPU"
    )
    service = _live_owner(source, "run", owner_pid)
    binding = source_binding(source)
    root = output_path(source, "run")
    decl_sha, decl_raw = _declaration_sha(root / "declaration.json")
    require(decl_sha == declaration_sha, "parent declaration exact hash")
    decl = base.parse_json(decl_raw)
    require(
        decl["source"] == source
        and decl["source_binding"] == binding
        and decl["service_properties"] == service,
        "child declaration binds live source and owner",
    )
    require(
        base.host.identity(source) == decl["host_identity"],
        "exact pinned host and Python runtime identity before CUDA",
    )
    require(
        base.digest((canonical(decl["topology"]) + "\n").encode())
        == LAUNCH_TOPOLOGY_SHA256
        and decl["fixture_metadata"]["selected_level"] == [2, 7, 11],
        "predeclared exact full launch topology and root-child level",
    )
    smooth_raw = _read_file(base.execution.ROOT / SMOOTH_LEAF, repeat.SMOOTH_LIMIT)
    require(
        base.digest(smooth_raw) == SMOOTH_SHA and smooth._crb_accumulate is KERNEL,
        "exact installed smooth kernel source and identity",
    )
    require(
        runtime_versions() == decl["fixture_metadata"]["runtime_versions"],
        "pinned runtime versions",
    )
    cpu_rng_before = torch.random.get_rng_state().clone()
    immutable_raw = _read_file(root / "baseline.bin", BASELINE_BYTES)
    require(
        base.digest(immutable_raw)
        == decl["fixture_metadata"]["baseline_sha256"]
        == BASELINE_SHA256,
        "parent immutable baseline hash",
    )
    baseline_np = (
        np.frombuffer(immutable_raw, dtype="<f4")
        .reshape(repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
        .copy()
    )
    parents_np = np.asarray(decl["fixture_metadata"]["body_parentid"], dtype=np.int32)
    level_np = np.asarray(decl["fixture_metadata"]["selected_level"], dtype=np.int32)
    require(
        parents_np.shape == (repeat.BODIES,)
        and parents_np.tolist() == decl["topology"]["body_parentid"]
        and level_np.tolist() == [2, 7, 11]
        and baseline_np.shape == (repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
        and bool(np.isfinite(baseline_np).all()),
        "exact finite fixed kernel input dimensions",
    )
    require(
        not torch.cuda.is_initialized(),
        "Torch CUDA remains uninitialized before Warp-only work",
    )
    device = wp.get_device("cuda:0")
    require(device.is_cuda, "Warp CUDA device zero")
    outputs = _repeat_batch(device, baseline_np, parents_np, level_np)
    cpu_rng_after = torch.random.get_rng_state().clone()
    require(
        torch.equal(cpu_rng_before, cpu_rng_after), "caller Torch CPU RNG preserved"
    )
    require(not torch.cuda.is_initialized(), "Torch CUDA stayed uninitialized")
    require(len(outputs) == OUTPUT_BYTES, "all exact repeat outputs retained")
    rng = cpu_rng_before.numpy().tobytes() + cpu_rng_after.numpy().tobytes()
    base._write_exclusive(root / "outputs.bin", outputs, OUTPUT_BYTES)
    base._write_exclusive(root / "rng.bin", rng, RNG_BYTES)
    receipt = {
        "protocol": PROTOCOL + ":child",
        "source": source,
        "source_binding": binding,
        "declaration_sha256": declaration_sha,
        "owner_pid": owner_pid,
        "child_pid": os.getpid(),
        "child_ppid": os.getppid(),
        "service_properties": service,
        "smooth_sha256": SMOOTH_SHA,
        "runtime_versions": runtime_versions(),
        "launch": {
            "kernel": KERNEL_NAME,
            "dim": [repeat.WORLDS, 3],
            "repeats": repeat.REPEATS,
            "same_input_output": True,
            "output_bank": True,
            "readback_after_batch": True,
        },
        "cpu_rng_preserved": True,
        "rng_state_bytes": int(cpu_rng_before.numel()),
        "torch_cuda_initialized": False,
        "actor_constructed": False,
        "model_constructed": False,
        "optimizer_constructed": False,
        "random_kernels": False,
        "outputs_sha256": base.digest(outputs),
        "rng_sha256": base.digest(rng),
        **FLAGS,
    }
    base.write_json(root / "child-receipt.json", receipt)
    require(
        _live_owner(source, "run", owner_pid) == service,
        "child remains under exact owner service",
    )
    return receipt


def _read_tests(source, binding):
    root = output_path(source, "tests")
    base._exact_inventory(root, TEST_FILES_ON_DISK)
    receipt_raw = _read_file(root / "receipt.json", JSON_LIMIT)
    receipt = base.parse_json(receipt_raw)
    require(
        type(receipt) is dict
        and set(receipt)
        == {
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
        and receipt["protocol"] == PROTOCOL + ":tests"
        and receipt["source"] == source
        and receipt["source_binding"] == binding
        and receipt["test_files"] == list(TEST_FILES)
        and type(receipt["passed"]) is int
        and receipt["passed"] == EXPECTED_TESTS
        and EXPECTED_TESTS > 0
        and receipt["skips"] == 0
        and all(receipt[key] is False for key in FLAGS),
        "authenticated owner-frozen new CPU test receipt",
    )
    log = _read_file(root / "pytest.log", LOG_LIMIT)
    require(
        base.digest(log) == receipt["pytest_sha256"], "whole focused CPU pytest log"
    )
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
    require(
        len(totals) == 1
        and int(totals[0][0]) == EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
        "exact CPU tests with zero skips",
    )
    _recorded_properties(receipt["service_properties"], "tests")
    completed(source, "tests", receipt["service_properties"]["InvocationID"])
    report = base.parse_json(_read_file(root / "report.json", JSON_LIMIT))
    require(
        report.get("protocol") == PROTOCOL + ":tests"
        and report.get("source") == source
        and report.get("mode") == "tests"
        and report.get("status") == "passed"
        and report.get("source_binding") == binding
        and report.get("service_properties") == receipt["service_properties"]
        and report.get("passed") == EXPECTED_TESTS
        and report.get("pytest_sha256") == receipt["pytest_sha256"]
        and all(report.get(flag) is False for flag in FLAGS),
        "terminal successful test report matches authenticated receipt",
    )
    return base.digest(receipt_raw)


def completed(source, mode, invocation):
    base._hex(invocation, 32, "finished kernel diagnostic invocation")
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
    require(
        value
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
        "successful terminal retained service invocation",
    )
    return value


def execute(source, mode):
    _hidden()
    require(mode in ("tests", "run"), "declared CPU supervisor mode")
    check_window(reserve_seconds=TEST_RESERVE if mode == "tests" else RUN_RESERVE)
    service = _owned_properties(source, mode, os.getpid())
    binding = source_binding(source)
    host_identity = base.host.identity(source)
    started = time.monotonic()
    root = output_path(source, mode)
    root.mkdir(parents=True, exist_ok=False)
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
        if mode == "tests":
            require(
                EXPECTED_TESTS > 0, "owner-frozen positive test count before pytest"
            )
            env = _cpu_env()
            with base.gap.base.files.gpu_lease() as lease_fd:
                idle_before = base.gpu_idle_gate.wait_idle()
                filmbrain = base.gap.base.retained.d0.filmbrain_state()
                protected = base.gap.base.protected_state()
                require(
                    bool(protected)
                    and all(value == "inactive" for value in protected.values()),
                    "protected services inactive before CPU tests",
                )
                prior._run_process(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *["tests/" + name for name in TEST_FILES],
                    ],
                    root / "pytest.log",
                    TEST_CAP - 30,
                    env=env,
                    fd=lease_fd,
                )
                log = _read_file(root / "pytest.log", LOG_LIMIT)
                totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
                require(
                    len(totals) == 1
                    and int(totals[0][0]) == EXPECTED_TESTS
                    and not re.search(
                        rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log
                    ),
                    "exact frozen test suite",
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
                base.write_json(root / "receipt.json", receipt)
                idle_after = base.gpu_idle_gate.wait_idle()
                require(
                    base.gap.base.retained.d0.filmbrain_state() == filmbrain
                    and base.gap.base.protected_state() == protected
                    and source_binding(source) == binding
                    and _owned_properties(source, mode, os.getpid()) == service,
                    "unchanged source, service, FilmBrain and protected states after CPU tests",
                )
            report.update(
                passed=EXPECTED_TESTS,
                pytest_sha256=receipt["pytest_sha256"],
                idle_before=idle_before,
                idle_after=idle_after,
                filmbrain=filmbrain,
                protected_services=protected,
            )
        else:
            tests_sha = _read_tests(source, binding)
            original = _authenticate_original()
            topology = crb._fresh_topology(64)
            oracle_raw = _read_file(
                order.output_path(ORDER_SOURCE, "run") / "comparison.json", JSON_LIMIT
            )
            fixture_value = repeat.fixture(oracle_raw, topology)
            baseline = fixture_value["baseline"]
            require(
                type(baseline) is np.ndarray
                and baseline.dtype == np.float32
                and baseline.shape == (repeat.WORLDS, repeat.BODIES, repeat.COMPONENTS)
                and baseline.flags.c_contiguous,
                "bounded immutable float32 baseline",
            )
            baseline_raw = baseline.astype("<f4", copy=False).tobytes()
            require(
                len(baseline_raw) == BASELINE_BYTES
                and base.digest(baseline_raw) == BASELINE_SHA256,
                "baseline artifact exact bytes and predeclared hash",
            )
            require(
                base.digest((canonical(topology) + "\n").encode())
                == LAUNCH_TOPOLOGY_SHA256
                and repeat.TOPOLOGY_SHA256
                == "5b445215f52d61d10bc892a14b0d0b15e3041ff3a5dfb6c1964f06211d0cb43a",
                "distinct exact launch and component-map topology hashes",
            )
            require(
                fixture_value["level"].tolist() == [2, 7, 11]
                and np.array_equal(
                    fixture_value["parents"],
                    np.asarray(topology["body_parentid"], dtype="<i4"),
                ),
                "fixture parents and selected body level match exact topology",
            )
            fixture_meta = dict(
                fixture_value["metadata"],
                baseline_sha256=base.digest(baseline_raw),
                body_parentid=topology["body_parentid"],
                selected_level=fixture_value["level"].tolist(),
                runtime_versions=runtime_versions(),
            )
            decl = declaration(
                source,
                binding,
                original,
                topology,
                fixture_meta,
                service,
                host_identity,
            )
            decl_raw = (canonical(decl) + "\n").encode()
            base._write_exclusive(root / "declaration.json", decl_raw, JSON_LIMIT)
            base._write_exclusive(root / "baseline.bin", baseline_raw, BASELINE_BYTES)
            require(
                _read_tests(source, binding) == tests_sha,
                "test receipt unchanged before GPU child",
            )
            with base.gap.base.files.gpu_lease() as lease_fd:
                idle_before = base.gpu_idle_gate.wait_idle()
                filmbrain = base.gap.base.retained.d0.filmbrain_state()
                protected = base.gap.base.protected_state()
                require(
                    bool(protected)
                    and all(value == "inactive" for value in protected.values()),
                    "protected system and user services inactive",
                )
                cmd = [
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
                    base.digest(decl_raw),
                ]
                samples = prior._run_process(
                    cmd,
                    root / "child.log",
                    CHILD_SECONDS,
                    env=_gpu_env(),
                    fd=lease_fd,
                    monitor=True,
                )
                cuda_probe.check_log(root / "child.log")
                base._exact_inventory(root, RUN_FILES - {"report.json"})
                artifacts = _inventory(root, RUN_FILES - {"report.json"})
                receipt = base.parse_json(
                    _read_file(root / "child-receipt.json", JSON_LIMIT)
                )
                require(
                    type(receipt) is dict
                    and set(receipt) == _CHILD_RECEIPT_KEYS
                    and receipt.get("protocol") == PROTOCOL + ":child"
                    and receipt.get("source") == source
                    and receipt.get("source_binding") == binding
                    and receipt.get("declaration_sha256") == base.digest(decl_raw)
                    and receipt.get("owner_pid") == os.getpid()
                    and receipt.get("child_ppid") == os.getpid()
                    and receipt.get("service_properties") == service
                    and receipt.get("runtime_versions") == runtime_versions()
                    and receipt.get("smooth_sha256") == SMOOTH_SHA
                    and receipt.get("cpu_rng_preserved") is True
                    and receipt.get("torch_cuda_initialized") is False
                    and receipt.get("actor_constructed") is False
                    and receipt.get("model_constructed") is False
                    and receipt.get("optimizer_constructed") is False
                    and receipt.get("random_kernels") is False
                    and receipt.get("launch")
                    == {
                        "kernel": KERNEL_NAME,
                        "dim": [repeat.WORLDS, 3],
                        "repeats": repeat.REPEATS,
                        "same_input_output": True,
                        "output_bank": True,
                        "readback_after_batch": True,
                    }
                    and all(receipt.get(key) is False for key in FLAGS),
                    "exact child receipt and non-admitting flags",
                )
                child_pids = {sample.get("child_pid") for sample in samples}
                require(
                    len(child_pids) == 1
                    and all(type(pid) is int and pid > 0 for pid in child_pids)
                    and type(receipt.get("child_pid")) is int
                    and receipt.get("child_pid") in child_pids,
                    "monitored actual child PID binds receipt",
                )
                output_raw = _read_file(root / "outputs.bin", OUTPUT_BYTES)
                rng_raw = _read_file(root / "rng.bin", RNG_BYTES)
                state_bytes = receipt.get("rng_state_bytes")
                require(
                    type(state_bytes) is int
                    and state_bytes > 0
                    and len(rng_raw) == 2 * state_bytes
                    and rng_raw[:state_bytes] == rng_raw[state_bytes:]
                    and base.digest(output_raw) == receipt["outputs_sha256"]
                    and base.digest(rng_raw) == receipt["rng_sha256"]
                    and base.digest(_read_file(root / "baseline.bin", BASELINE_BYTES))
                    == BASELINE_SHA256,
                    "all full raw outputs, equal CPU RNG states and baseline authenticated before analysis",
                )
                result = repeat.analyze(output_raw, fixture_value)
                require(
                    result.get("cause_proven") is False
                    and result.get("full_window_passed") is False
                    and result.get("training_authorized") is False
                    and result.get("physical_result_accepted") is False
                    and result.get("actual_launch_inputs_captured") is False
                    and result.get("bit_variation_is_determinism_proof") is False,
                    "non-admitting fixed-input summary",
                )
                fixture_after = repeat.fixture(oracle_raw, crb._fresh_topology(64))
                require(
                    fixture_after["metadata"] == fixture_value["metadata"]
                    and np.array_equal(
                        fixture_after["baseline"].view("u4"),
                        fixture_value["baseline"].view("u4"),
                    )
                    and np.array_equal(
                        fixture_after["parents"], fixture_value["parents"]
                    )
                    and np.array_equal(fixture_after["level"], fixture_value["level"]),
                    "independent parent fixture is byte-identical after child",
                )
                idle_after = base.gpu_idle_gate.wait_idle()
                require(
                    source_binding(source) == binding
                    and _authenticate_original() == original
                    and base.host.identity(source) == host_identity
                    and _owned_properties(source, mode, os.getpid()) == service
                    and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                    and base.gap.base.protected_state() == protected,
                    "unchanged original input, source, service and protected context",
                )
                report.update(
                    tests_receipt_sha256=tests_sha,
                    original=original,
                    fixture_metadata=fixture_meta,
                    artifacts=artifacts,
                    child_receipt=receipt,
                    result=result,
                    monitor_samples=samples,
                    idle_before=idle_before,
                    idle_after=idle_after,
                    filmbrain=filmbrain,
                    protected_services=protected,
                )
            require(
                time.monotonic() - started < CAP,
                "bounded fixed-input diagnostic parent cap",
            )
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            error_type=type(error).__name__,
            error=str(error),
            partial_artifacts=_partial_inventory(root),
        )
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "report.json", report)
        if mode == "run" and report.get("status") == "passed":
            _inventory(root, RUN_FILES)
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
            "exact child invocation arguments",
        )
        result = _child(
            args.source, args.lease_fd, args.owner_pid, args.declaration_sha
        )
    else:
        require(
            args.lease_fd is None
            and args.owner_pid is None
            and args.declaration_sha is None,
            "supervisor has no inherited child arguments",
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
