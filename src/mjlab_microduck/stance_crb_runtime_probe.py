"""Four-child constructor/eager-forward CRB schedule diagnostic.

This bounded runtime sample changes only the CRB accumulation grouping inside
an owned process-local scope. It does not run physics integration or establish
the cause of the original trajectory mismatch.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import torch
import numpy as np
import mujoco
from mujoco_warp._src import forward as mw_forward
from mujoco_warp._src import smooth as mw_smooth

from mjlab_microduck import stance_crb_full_tree_probe as fulltree
from mjlab_microduck import stance_crb_runtime_control as runtime_control
from mjlab_microduck import stance_crb_serial_control_probe as serial
from mjlab_microduck import stance_cuda_probe as cuda_probe
from mjlab_microduck import stance_recovery_cuda_constructor_rng as constructor
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck import stance_recovery_early_inertia_trace as inertia_trace
from mjlab_microduck import stance_warp_runtime
from mjlab_microduck import stance_plant_evidence as plant_evidence
from mjlab_microduck import stance_recovery_contract as recovery_contract
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
MODULE = "mjlab_microduck.stance_crb_runtime_probe"
PROTOCOL = "football-b1d-crb-runtime-schedule-control-20261006-v1"
BASE_SOURCE = "9d2162d71d4c869d6d9b2481b663b3cb07042dd9"
SYNC_FROM_SOURCE = "dab4f3258e10e79fa3e44629161a76705b8b359d"
HISTORIC_FULLTREE_SOURCE = "d923275040b3538f6ed48b867f1b1e6bc42304d3"
FULLTREE_OWNER_SHA256 = (
    "718b61f3d08e61a2e27410fe8b8b62ae3be52a9a74dad0034e4998bc074ae707"
)
FULLTREE_OWNER_PATH = (
    "artifacts/tools/stance-crb-fulltree-owner-terminal-d923275040b3.json"
)
FULLTREE_TERMINAL_SHA256 = (
    "4806842ddf6b664858875b632246d40df384ef72564f99852b96a1ff1d93dc2f"
)
FULLTREE_TERMINAL_PATH = (
    "artifacts/tools/stance-crb-fulltree-closeout-terminal-d923275040b3.json"
)
FULLTREE_MAC_SHA256 = "20b5bcf1ca84c1f20f41f8cabd0e95090c1f16c49b9aa60050a06de6f7fe0b59"
FULLTREE_MAC_PATH = "artifacts/tools/stance-crb-fulltree-mac-receiver-d923275040b3/mac-verification-v2.json"
FULLTREE_CLEARANCE_SHA256 = (
    "0d495a521888a9a49e90eac31ea7ac2e6fd7696c5e40db691decf3e2945aaac4"
)
FULLTREE_CLEARANCE_PATH = (
    "artifacts/tools/stance-crb-fulltree-clearance-d923275040b3.json"
)
FULLTREE_TEST_INVOCATION = "8cf05493d0ec41c5babde4924d05d8b1"
FULLTREE_FIXTURE_INVOCATION = "75286b04bbc84471ae8387541853fa8a"
FULLTREE_RUN_INVOCATION = "f9779454708f4e9c97ed2714cf5f9d89"
FULLTREE_CLOSEOUT_INVOCATION = "5f50f3f71f8e4aa198dcc96e83ab1371"
FULLTREE_SYNC_INVOCATION = "c76c64903ee84b09b1a290a2e6cd72d1"
SMOOTH_SHA256 = "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
FORWARD_SHA256 = "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3"
SOURCE_ROOT = serial.frozen.cpu_probe.SYNC_ROOT
SOURCE_ORIGIN = serial.frozen.cpu_probe.SYNC_ORIGIN
SOURCE_BRANCH = "feat/athletics-obstacle-curriculum"
MODES = ("tests", "preflight", "run", "closeout")
ATTEMPTS = (
    "concurrent-capture",
    "concurrent-replay",
    "serial-capture",
    "serial-replay",
)
SCHEDULE_MODES = {
    "concurrent-capture": "concurrent",
    "concurrent-replay": "concurrent",
    "serial-capture": "serial",
    "serial-replay": "serial",
}
SECONDS = {"tests": 300, "preflight": 300, "run": 1200, "closeout": 300}
CHILD_SECONDS = 180
SYNC_SECONDS, SYNC_MEMORY, SYNC_CPU_QUOTA = 120, 256 * 1024**2, 100
MEMORY = 6 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
MARGIN, CUTOFF = 60, datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc).timestamp()
RUN_RESERVE = SECONDS["run"] + SECONDS["closeout"] + MARGIN
PREFLIGHT_RESERVE = SECONDS["preflight"] + SECONDS["run"] + SECONDS["closeout"] + MARGIN
TEST_RESERVE = (
    SECONDS["tests"]
    + SECONDS["preflight"]
    + SECONDS["run"]
    + SECONDS["closeout"]
    + MARGIN
)
EXPECTED_TESTS = 1579  # Exact reviewed 53-file CUDA-hidden suite; zero skips.
TEST_FILES = (
    *fulltree.TEST_FILES,
    "test_stance_crb_runtime_control.py",
    "test_stance_crb_runtime_probe.py",
)
DOC = "docs/experiments/2026-10-06-crb-runtime-schedule-control.md"
OWN = (
    "src/mjlab_microduck/stance_crb_runtime_probe.py",
    "tests/test_stance_crb_runtime_probe.py",
    DOC,
)
RELATED = (
    "src/mjlab_microduck/stance_crb_runtime_control.py",
    "tests/test_stance_crb_runtime_control.py",
)
ALLOWED = set(OWN) | set(RELATED)
FROZEN_PATHS = (
    set(fulltree._frozen_paths())
    | set(fulltree.OWN)
    | set(fulltree.RELATED)
    | set(fulltree.PLANNER_FROZEN)
    | set(serial.OWN)
    | set(serial.RELATED)
)
FLAGS = {
    **{key: False for key, value in runtime_control.FLAGS.items() if value is False},
    "runtime_schedule_control_qualified": False,
    "runtime_schedule_proves_original_cause": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
}
JSON_LIMIT, LOG_LIMIT, PAYLOAD_LIMIT = 2 * 1024**2, 1024**2, 2 * 1024**2
TOTAL_LIMIT = 16 * 1024**2
WORLDS = 64
FRAME_FIELDS = {
    **inertia_trace.SHAPES,
    "qpos": (21,),
    "qvel": (20,),
    "time": (),
    "qacc_warmstart": (20,),
    "ctrl": (14,),
    "xfrc_applied": (16, 6),
    "qfrc_applied": (20,),
}
FRAME_BYTES = (
    sum(
        math.prod((WORLDS, *shape)) if shape else WORLDS
        for shape in FRAME_FIELDS.values()
    )
    * 4
)
PAYLOAD_FRAME_LIMIT = 4 * FRAME_BYTES
TEST_FILES_ON_DISK = {"receipt.json", "pytest.log", "report.json"}
RUN_FILES = {
    "declaration.json",
    "report.json",
    *(f"{attempt}.pt" for attempt in ATTEMPTS),
    *(f"{attempt}.json" for attempt in ATTEMPTS),
    *(f"{attempt}.log" for attempt in ATTEMPTS),
}
RUN_CONTENT_FILES = RUN_FILES - {"report.json"}
PREFLIGHT_FILES = {"declaration.json", "report.json"}
CLOSEOUT_FILES = {"receipt.json", "report.json"}
FALSE_FLAGS = {
    **FLAGS,
    **{key: False for key, value in prior.FLAGS.items() if value is False},
}


def unit(source, mode):
    base._hex(source, 40, "exact CRB runtime source")
    require(mode in (*MODES, "sync"), "declared CRB runtime unit mode")
    return f"microduck-crb-runtime-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    area = "tools" if mode in ("tests", "closeout") else "evaluations"
    return (
        base.execution.ROOT
        / "artifacts"
        / area
        / f"stance-crb-runtime-{mode}-{source[:12]}"
    )


def source_sync_unit(source):
    return unit(source, "sync")


def source_sync_script(source, *, bundle_sha256):
    """Retarget only the exact retained head and owned sync-unit guards."""
    script = fulltree.source_sync_script(source, bundle_sha256=bundle_sha256)
    old_head = f'test "$(git rev-parse HEAD)" = {fulltree.SYNC_FROM_SOURCE}\n'
    new_head = f'test "$(git rev-parse HEAD)" = {SYNC_FROM_SOURCE}\n'
    old_unit = (
        f'if test "$duck_sync_running" != {fulltree.source_sync_unit(source)}; then\n'
    )
    new_unit = f'if test "$duck_sync_running" != {source_sync_unit(source)}; then\n'
    require(
        script.count(old_head) == script.count(old_unit) == 1,
        "exact old source and one sync-unit template guards",
    )
    return script.replace(old_head, new_head, 1).replace(old_unit, new_unit, 1)


def source_binding(source):
    base._hex(source, 40, "exact committed runtime source")
    root = Path(__file__).resolve().parents[2]
    require(str(root) == SOURCE_ROOT, "exact native source root")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=15
        ).stdout

    require(
        git("rev-parse", "HEAD").decode().strip() == source
        and git("branch", "--show-current").decode().strip() == SOURCE_BRANCH
        and git("remote", "get-url", "origin").decode().strip() == SOURCE_ORIGIN
        and not git("status", "--porcelain"),
        "clean exact runtime source branch, origin and HEAD",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE, source],
        cwd=root,
        check=True,
        timeout=15,
    )
    changed = set(git("diff", "--name-only", BASE_SOURCE, source).decode().splitlines())
    require(
        changed <= ALLOWED, "only five runtime declaration/adapter/probe paths changed"
    )
    leaves = {}
    for path in sorted(FROZEN_PATHS | set(OWN) | set(RELATED)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(
            raw == git("show", f"{source}:{path}"),
            "whole committed runtime source leaf " + path,
        )
        if path in FROZEN_PATHS:
            require(
                raw == git("show", f"{BASE_SOURCE}:{path}"),
                "frozen 9d2162 source leaf " + path,
            )
        leaves[path] = base.digest(raw)
    return {"source": source, "branch": SOURCE_BRANCH, "leaves": leaves}


def _check_window(*, reserve_seconds, now=None):
    now = time.time() if now is None else now
    require(
        type(reserve_seconds) is int and reserve_seconds > 0,
        "positive remaining runtime reserve",
    )
    require(
        type(now) in (int, float) and math.isfinite(now) and now >= 0,
        "finite runtime clock reading",
    )
    require(
        now + reserve_seconds < CUTOFF, "runtime schedule cutoff with closeout reserve"
    )


def _valid_elapsed(value, cap):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value < cap


def _cpu_env():
    env = serial.frozen._cpu_env()
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
    return fulltree._runtime_versions()


def _read_file(path, limit):
    return fulltree._read_file(path, limit)


def _inventory(root, names):
    result, total = {}, 0
    for name in sorted(names):
        raw = _read_file(root / name, _file_limit(name))
        total += len(raw)
        require(total <= TOTAL_LIMIT, "runtime retained inventory <=16 MiB")
        result[name] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return result


def _file_limit(name):
    return (
        LOG_LIMIT
        if name.endswith(".log")
        else JSON_LIMIT
        if name.endswith(".json")
        else PAYLOAD_LIMIT
    )


def _campaign_inventory(source):
    """Bound all retained files across tests, preflight, run and closeout."""
    declared = {
        "tests": TEST_FILES_ON_DISK,
        "preflight": PREFLIGHT_FILES,
        "run": RUN_FILES,
        "closeout": CLOSEOUT_FILES,
    }
    result, total = {}, 0
    for mode, names in declared.items():
        root = output_path(source, mode)
        try:
            metadata = root.lstat()
        except FileNotFoundError:
            continue
        require(
            stat.S_ISDIR(metadata.st_mode), "runtime campaign artifact mode directory"
        )
        present = set(os.listdir(root))
        require(present <= names, "no undeclared runtime campaign artifact path")
        for name in sorted(present):
            raw = _read_file(root / name, _file_limit(name))
            total += len(raw)
            require(
                total <= TOTAL_LIMIT, "all runtime mode artifacts combined <=16 MiB"
            )
            result[f"{mode}/{name}"] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return {"total_bytes": total, "files": result}


def _valid_campaign_summary(value):
    if type(value) is not dict or set(value) != {"total_bytes", "files"}:
        return False
    total, files = value["total_bytes"], value["files"]
    if (
        type(total) is not int
        or not 0 <= total <= TOTAL_LIMIT
        or type(files) is not dict
    ):
        return False
    observed = 0
    allowed = {
        "tests": TEST_FILES_ON_DISK,
        "preflight": PREFLIGHT_FILES,
        "run": RUN_FILES,
        "closeout": CLOSEOUT_FILES,
    }
    for relative, entry in files.items():
        if type(relative) is not str or relative.count("/") != 1:
            return False
        mode, name = relative.split("/", 1)
        if mode not in allowed or name not in allowed[mode]:
            return False
        if (
            type(entry) is not dict
            or set(entry) != {"bytes", "sha256"}
            or type(entry["bytes"]) is not int
            or not 0 < entry["bytes"] <= _file_limit(name)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            return False
        observed += entry["bytes"]
    return observed == total


def _stage_campaign_snapshot(source, mode):
    stages = ("tests", "preflight", "run", "closeout")
    require(mode in stages, "known runtime campaign publication stage")
    declared = {
        "tests": TEST_FILES_ON_DISK,
        "preflight": PREFLIGHT_FILES,
        "run": RUN_FILES,
        "closeout": CLOSEOUT_FILES,
    }
    result, total = {}, 0
    for stage in stages[: stages.index(mode) + 1]:
        names = declared[stage]
        if stage == mode:
            names = names - {"report.json"}
        inventory = _inventory(output_path(source, stage), names)
        for name, entry in inventory.items():
            result[f"{stage}/{name}"] = entry
            total += entry["bytes"]
            require(total <= TOTAL_LIMIT, "published stage snapshot <=16 MiB")
    return {"total_bytes": total, "files": result}


def _partial_inventory(root):
    names = TEST_FILES_ON_DISK | PREFLIGHT_FILES | RUN_FILES | CLOSEOUT_FILES
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
                else PAYLOAD_LIMIT
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


def _terminal_properties(source, mode, owner_pid):
    unit(source, mode)
    require(type(owner_pid) is int and owner_pid > 0, "actual runtime supervisor PID")
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
    expected = {
        "MainPID": str(owner_pid),
        "ActiveState": "active",
        "RuntimeMaxUSec": {
            "tests": "5min",
            "preflight": "5min",
            "run": "20min",
            "closeout": "5min",
        }[mode],
        "MemoryMax": str(MEMORY),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
        "RemainAfterExit": "yes",
    }
    require(
        {key: value[key] for key in expected} == expected,
        "exact live runtime owner and service caps",
    )
    base._hex(value["InvocationID"], 32, "runtime invocation")
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
        "only one Duck service active during runtime mode",
    )
    return value


def _terminal_record(value, mode, invocation):
    base._hex(invocation, 32, "completed runtime invocation")
    expected = {
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "RemainAfterExit": "yes",
        "RuntimeMaxUSec": {
            "tests": "5min",
            "preflight": "5min",
            "run": "20min",
            "closeout": "5min",
        }[mode],
        "MemoryMax": str(MEMORY),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
        "InvocationID": invocation,
    }
    require(
        type(value) is dict and value == expected,
        "successful exact retained runtime terminal caps and invocation",
    )
    return value


def _terminal(source, mode, invocation):
    base._hex(invocation, 32, "completed runtime unit invocation")
    keys = (
        "ActiveState",
        "SubState",
        "MainPID",
        "NRestarts",
        "ExecMainStatus",
        "Result",
        "RemainAfterExit",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "Nice",
        "KillMode",
        "InvocationID",
    )
    value = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in keys
    }
    return _terminal_record(value, mode, invocation)


def _read_idle_context():
    idle = base.gpu_idle_gate.wait_idle()
    protected = base.gap.base.protected_state()
    filmbrain = base.gap.base.retained.d0.filmbrain_state()
    require(
        type(protected) is dict
        and protected
        and all(state == "inactive" for state in protected.values()),
        "both protected services inactive in system and user namespaces",
    )
    return idle, protected, filmbrain


def _close_context(
    source, mode, binding, host_identity, versions, service, protected, filmbrain
):
    require(source_binding(source) == binding, "unchanged full runtime source closure")
    identity, current_versions = _host(
        source,
        host_identity
        if host_identity.get("source") == source
        else {**host_identity, "source": source},
    )
    require(
        identity == host_identity and current_versions == versions,
        "unchanged actual WSL host/Python trees/packages",
    )
    require(
        _terminal_properties(source, mode, os.getpid()) == service,
        "unchanged runtime owner service identity and caps",
    )
    require(
        base.gap.base.protected_state() == protected
        and base.gap.base.retained.d0.filmbrain_state() == filmbrain,
        "protected namespaces and FilmBrain state unchanged",
    )
    return base.gpu_idle_gate.wait_idle()


def _sha_summary(value):
    return base.digest((canonical(value) + "\n").encode())


def _predecessor_summary(value):
    require(
        value.get("predecessors_authenticated") is True,
        "all 74 predecessor files authenticated",
    )
    legacy = value["legacy"]
    concurrent = {
        key: item
        for key, item in legacy["concurrent"].items()
        if key not in ("fixture", "fixture_source")
    }
    return {
        "fulltree": value["fulltree"],
        "legacy_original": legacy["original"],
        "legacy_concurrent": concurrent,
        "serial_owner_packet_sha256": legacy["serial_owner_packet_sha256"],
        "serial_closeout_packet_sha256": legacy["serial_closeout_packet_sha256"],
        "serial_run_files": legacy["serial_run_files"],
        "serial_test_files": legacy["serial_test_files"],
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
        and set(receipt) == receipt_keys
        and receipt.get("protocol") == PROTOCOL + ":tests"
        and receipt.get("source") == source
        and receipt.get("source_binding") == binding
        and receipt.get("test_files") == list(TEST_FILES)
        and receipt.get("passed") == EXPECTED_TESTS
        and EXPECTED_TESTS > 0
        and receipt.get("skips") == 0
        and all(receipt.get(key) is False for key in FLAGS),
        "exact frozen 53-file CPU-only test receipt",
    )
    log = _read_file(root / "pytest.log", LOG_LIMIT)
    require(
        base.digest(log) == receipt["pytest_sha256"], "whole test log bound to receipt"
    )
    matches = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
    require(
        len(matches) == 1
        and int(matches[0][0]) == EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
        "complete passing runtime source suite with zero skips",
    )
    service = receipt["service_properties"]
    _terminal(source, "tests", service["InvocationID"])
    report = base.parse_json(_read_file(root / "report.json", JSON_LIMIT))
    report_keys = {
        "protocol",
        "source",
        "mode",
        "status",
        "source_binding",
        "host_identity",
        "runtime_versions",
        "service_properties",
        "receipt_sha256",
        "passed",
        "skips",
        "pytest_sha256",
        "idle_before",
        "idle_after",
        "filmbrain",
        "protected_services",
        "elapsed_seconds",
        "campaign_artifacts_before_report",
        *FLAGS,
    }
    require(
        type(report) is dict
        and set(report) == report_keys
        and report.get("protocol") == PROTOCOL + ":tests"
        and report.get("source") == source
        and report.get("mode") == "tests"
        and report.get("status") == "passed"
        and report.get("source_binding") == binding
        and canonical(report.get("service_properties")) == canonical(service)
        and report.get("receipt_sha256") == base.digest(receipt_raw)
        and report.get("passed") == EXPECTED_TESTS
        and report.get("skips") == 0
        and report.get("pytest_sha256") == receipt["pytest_sha256"]
        and _valid_campaign_summary(report.get("campaign_artifacts_before_report"))
        and canonical(report.get("campaign_artifacts_before_report"))
        == canonical(_stage_campaign_snapshot(source, "tests"))
        and _valid_elapsed(report.get("elapsed_seconds"), SECONDS["tests"])
        and all(report.get(key) is False for key in FLAGS),
        "test report independently matches receipt",
    )
    return base.digest(receipt_raw)


def _payload_bytes(payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    require(0 < len(raw) <= PAYLOAD_LIMIT, "each CPU-readable runtime payload <=2 MiB")
    return raw


def _payload_summary(payload):
    frame_hashes = []
    for frame in payload["frames"]:
        frame_hashes.append(
            {
                name: hashlib.sha256(
                    tensor.detach().contiguous().numpy().tobytes()
                ).hexdigest()
                for name, tensor in frame.items()
            }
        )
    caller = payload["caller_rng_states"]
    return {
        "protocol": PROTOCOL + ":child",
        "source": payload["source"],
        "mode": payload["mode"],
        "attempt": payload["attempt"],
        "owner_pid": payload["owner_pid"],
        "child_pid": payload["child_pid"],
        "child_ppid": payload["child_ppid"],
        "host_identity": payload["host_identity"],
        "runtime_versions": payload["runtime_versions"],
        "runtime_metadata": payload["runtime_metadata"],
        "payload_file": payload["attempt"] + ".pt",
        "caller_rng_sha256": {
            key: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            for key, tensor in caller.items()
        },
        "frame_sha256": frame_hashes,
        "scope_receipt": payload["scope_receipt"],
        "constructor_receipt_sha256": _sha_summary(
            {
                key: value
                for key, value in payload["constructor_receipt"].items()
                if key not in ("caller_states", "private_states", "nominal_parameters")
            }
        ),
        "torch_cuda_initialized": True,
        **FLAGS,
    }


def _child(source, attempt, lease_fd, owner_pid, declaration_sha):
    base.training_smoke.inherited_lease(lease_fd)
    mode = SCHEDULE_MODES.get(attempt)
    require(
        mode in ("concurrent", "serial")
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "one isolated CUDA0 runtime attempt",
    )
    require(
        os.getppid() == owner_pid and owner_pid > 0,
        "actual runtime run service parent PID",
    )
    service = _terminal_properties(source, "run", owner_pid)
    binding = source_binding(source)
    root = output_path(source, "run")
    declaration_raw = _read_file(root / "declaration.json", JSON_LIMIT)
    require(
        base.digest(declaration_raw) == declaration_sha,
        "run declaration whole hash before parse",
    )
    declaration = base.parse_json(declaration_raw)
    expected_keys = {
        "protocol",
        "source",
        "source_binding",
        "host_identity",
        "runtime_versions",
        "service_properties",
        "tests_receipt_sha256",
        "preflight_declaration_sha256",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "schedule",
        "runtime_metadata_reference",
        "scope",
        "source_hashes",
        "private_cuda_start_sha256",
        *FLAGS,
    }
    require(
        type(declaration) is dict
        and set(declaration) == expected_keys
        and declaration.get("protocol") == PROTOCOL + ":run"
        and declaration.get("source") == source
        and canonical(declaration.get("source_binding")) == canonical(binding)
        and declaration.get("service_properties") == service
        and declaration.get("schedule") == prior.declaration(source)
        and _metadata_matches_reference(
            declaration.get("runtime_metadata_reference"),
            _compiled_runtime_metadata_reference(source),
        )
        and declaration.get("source_hashes") == _pinned_runtime_sources()
        and all(declaration.get(key) is False for key in FLAGS),
        "exact child source/runtime/schedule declaration",
    )
    identity, versions = _host(source, declaration["host_identity"])
    require(
        identity == declaration["host_identity"]
        and versions == declaration["runtime_versions"],
        "pinned host/dependency identity before CUDA initialization",
    )
    require(
        not torch.cuda.is_initialized(), "child enters without initialized Torch CUDA"
    )
    _pinned_runtime_sources()

    torch.manual_seed(673)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(677)
    caller_before = {
        "cpu_before": torch.random.get_rng_state().clone(),
        "cuda_before": torch.cuda.get_rng_state(0).cpu().clone(),
    }
    require(
        torch.equal(
            caller_before["cpu_before"], _caller_rng_seed_states()["cpu_seed_673"]
        ),
        "actual child caller CPU seed 673",
    )
    control = runtime_control.StanceCrbRuntimeControl(mode=mode, max_forward_calls=4)
    ctor = constructor.CudaConstructorRng(source, lease_fd=lease_fd)
    frames = []
    with control:
        env = ctor.construct(declaration["schedule"])
        private_cuda_start = ctor.receipt["private_states"]["cuda_start"]
        require(
            hashlib.sha256(private_cuda_start.numpy().tobytes()).hexdigest()
            == declaration["private_cuda_start_sha256"],
            "private CUDA constructor start matches authenticated preflight receipt",
        )
        control.bind_runtime(env)
        require(
            type(env) is constructor.ScheduledRecoveryRuntime
            and env.forward_graph is None,
            "exact unmodified eager scheduled runtime",
        )
        frames.append(_snapshot_frame(env))
        for _ in range(3):
            env._forward()
            require(
                not bool(torch.count_nonzero(env.steps)),
                "no physics steps after eager forward",
            )
            frames.append(_snapshot_frame(env))
        require(
            not bool(torch.count_nonzero(env.steps)),
            "no physics steps or dose execution",
        )
    scope_receipt = _validate_scope_receipt(control.receipt, source=source, mode=mode)
    caller_after = {
        "cpu_after": torch.random.get_rng_state().clone(),
        "cuda_after": torch.cuda.get_rng_state(0).cpu().clone(),
    }
    caller_states = {**caller_before, **caller_after}
    require(
        torch.equal(caller_states["cpu_before"], caller_states["cpu_after"])
        and torch.equal(caller_states["cuda_before"], caller_states["cuda_after"]),
        "default caller RNG streams unchanged by constructor and four frames",
    )
    runtime_metadata = _runtime_metadata(env)
    payload = {
        "protocol": PROTOCOL + ":child-payload",
        "source": source,
        "mode": mode,
        "attempt": attempt,
        "owner_pid": owner_pid,
        "child_pid": os.getpid(),
        "child_ppid": os.getppid(),
        "host_identity": identity,
        "runtime_versions": versions,
        "runtime_metadata": runtime_metadata,
        "frames": frames,
        "caller_rng_states": caller_states,
        "constructor_receipt": ctor.receipt,
        "scope_receipt": scope_receipt,
        **FLAGS,
    }
    _validate_payload(
        payload,
        source=source,
        mode=mode,
        attempt=attempt,
        host_identity=identity,
        runtime_versions=versions,
        metadata_reference=declaration["runtime_metadata_reference"],
    )
    payload_raw = _payload_bytes(payload)
    payload_path = root / (attempt + ".pt")
    base._write_exclusive(payload_path, payload_raw, PAYLOAD_LIMIT)
    summary = _payload_summary(payload)
    receipt = {
        **summary,
        "payload_bytes": len(payload_raw),
        "payload_sha256": base.digest(payload_raw),
        "service_properties": service,
        "declaration_sha256": declaration_sha,
        "source_binding": binding,
        "source_hashes": _pinned_runtime_sources(),
        "status": "passed",
        "constructor_calls": 1,
        "forward_calls": 4,
        "physics_steps": [0] * WORLDS,
    }
    base.write_json(root / (attempt + ".json"), receipt)
    require(
        _terminal_properties(source, "run", owner_pid) == service
        and source_binding(source) == binding
        and base.host.identity(source) == identity,
        "unchanged source/runtime/service after isolated runtime child",
    )
    return receipt


def _read_child_receipt(
    path, *, source, mode, attempt, declaration, expected_pid, owner_pid
):
    raw = _read_file(path, JSON_LIMIT)
    receipt = base.parse_json(raw)
    expected = {
        "protocol",
        "source",
        "mode",
        "attempt",
        "owner_pid",
        "child_pid",
        "child_ppid",
        "host_identity",
        "runtime_versions",
        "runtime_metadata",
        "payload_file",
        "caller_rng_sha256",
        "frame_sha256",
        "scope_receipt",
        "constructor_receipt_sha256",
        "torch_cuda_initialized",
        "payload_bytes",
        "payload_sha256",
        "service_properties",
        "declaration_sha256",
        "source_binding",
        "source_hashes",
        "status",
        "constructor_calls",
        "forward_calls",
        "physics_steps",
        *FLAGS,
    }
    require(
        type(receipt) is dict
        and set(receipt) == expected
        and receipt.get("protocol") == PROTOCOL + ":child"
        and receipt.get("source") == source
        and receipt.get("mode") == mode
        and receipt.get("attempt") == attempt
        and type(receipt.get("owner_pid")) is int
        and receipt.get("owner_pid") == owner_pid
        and type(receipt.get("child_pid")) is int
        and receipt.get("child_pid") == expected_pid
        and type(receipt.get("child_ppid")) is int
        and receipt.get("child_ppid") == owner_pid
        and receipt.get("host_identity") == declaration["host_identity"]
        and receipt.get("runtime_versions") == declaration["runtime_versions"]
        and _metadata_matches_reference(
            receipt.get("runtime_metadata"), declaration["runtime_metadata_reference"]
        )
        and receipt.get("service_properties") == declaration["service_properties"]
        and receipt.get("declaration_sha256") == declaration["declaration_sha256"]
        and receipt.get("source_binding") == declaration["source_binding"]
        and receipt.get("source_hashes") == declaration["source_hashes"]
        and receipt.get("status") == "passed"
        and type(receipt.get("constructor_calls")) is int
        and receipt.get("constructor_calls") == 1
        and type(receipt.get("forward_calls")) is int
        and receipt.get("forward_calls") == 4
        and type(receipt.get("physics_steps")) is list
        and len(receipt["physics_steps"]) == WORLDS
        and all(type(item) is int and item == 0 for item in receipt["physics_steps"])
        and receipt.get("torch_cuda_initialized") is True
        and all(receipt.get(key) is False for key in FLAGS),
        "exact child receipt and actual parent PID",
    )
    _validate_scope_receipt(receipt["scope_receipt"], source=source, mode=mode)
    base._hex(receipt.get("payload_sha256"), 64, "child payload whole SHA")
    require(
        type(receipt.get("payload_bytes")) is int
        and 0 < receipt["payload_bytes"] <= PAYLOAD_LIMIT
        and receipt.get("payload_file") == attempt + ".pt",
        "bounded child payload descriptor",
    )
    return raw, receipt


def _load_payload(
    raw,
    receipt,
    *,
    source,
    attempt,
    host_identity,
    versions,
    private_cuda_start,
    metadata_reference,
):
    require(
        len(raw) == receipt["payload_bytes"]
        and base.digest(raw) == receipt["payload_sha256"],
        "child payload size and whole SHA checked before weights-only loading",
    )
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("bounded weights-only child payload decode failed") from error
    require(
        type(payload) is dict
        and "caller_rng_states" in payload
        and "constructor_receipt" in payload,
        "weights-only payload provenance fields before constructor CPU check",
    )
    prepared_caller = payload["caller_rng_states"]
    constructor.check(
        payload["constructor_receipt"], source=source, prepared_caller=prepared_caller
    )
    result = _validate_payload(
        payload,
        source=source,
        mode=receipt["mode"],
        attempt=attempt,
        host_identity=host_identity,
        runtime_versions=versions,
        private_reference=private_cuda_start,
        metadata_reference=metadata_reference,
    )
    require(
        payload["owner_pid"] == receipt["owner_pid"]
        and payload["child_pid"] == receipt["child_pid"]
        and payload["child_ppid"] == receipt["child_ppid"]
        and _metadata_matches_reference(
            payload["runtime_metadata"], receipt["runtime_metadata"]
        )
        and payload["scope_receipt"] == receipt["scope_receipt"],
        "CPU payload values match child JSON receipt",
    )
    require(
        _payload_summary(payload)["caller_rng_sha256"] == receipt["caller_rng_sha256"]
        and _payload_summary(payload)["frame_sha256"] == receipt["frame_sha256"],
        "CPU payload RNG/frame raw hashes match child receipt",
    )
    require(
        _sha_summary(
            {
                key: value
                for key, value in payload["constructor_receipt"].items()
                if key not in ("caller_states", "private_states", "nominal_parameters")
            }
        )
        == receipt["constructor_receipt_sha256"],
        "CPU payload constructor receipt summary hash",
    )
    return payload, result


def _field_comparison(label, left, right):
    require(
        left.keys() == right.keys() == FRAME_FIELDS.keys(),
        "exact paired runtime frame field set",
    )
    fields = {}
    for name in FRAME_FIELDS:
        a, b = left[name], right[name]
        count = a.size
        aa = a.view(np.uint32)
        bb = b.view(np.uint32)
        mismatch = aa != bb
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        fields[name] = {
            "left_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
            "right_sha256": hashlib.sha256(b.tobytes(order="C")).hexdigest(),
            "scalars": int(count),
            "bit_mismatch_scalars": int(mismatch.sum()),
            "max_abs_delta": float(delta.max()) if count else 0.0,
            "exact_raw_bits": not bool(mismatch.any()),
        }
    priority = ("cinert", "cdof", "subtree_com")
    return {
        "label": label,
        "priority_fields_before_crb": list(priority),
        "fields": {
            name: fields[name]
            for name in (
                *priority,
                *(field for field in FRAME_FIELDS if field not in priority),
            )
        },
    }


def _compare_four(payloads):
    require(
        tuple(payloads) == ATTEMPTS, "four runtime children remain in predeclared order"
    )
    for first in ATTEMPTS:
        for other in ATTEMPTS:
            require(
                canonical(payloads[first]["runtime_metadata"])
                == canonical(payloads[other]["runtime_metadata"]),
                "constructor/runtime metadata identical across four fresh children",
            )
    pairs = []
    for mode in ("concurrent", "serial"):
        capture = mode + "-capture"
        replay = mode + "-replay"
        for frame_index in range(4):
            pairs.append(
                _field_comparison(
                    f"{mode}.capture-vs-replay.frame-{frame_index}",
                    payloads[capture]["frames"][frame_index],
                    payloads[replay]["frames"][frame_index],
                )
            )
    for view in ("capture", "replay"):
        left, right = "concurrent-" + view, "serial-" + view
        for frame_index in range(4):
            pairs.append(
                _field_comparison(
                    f"{view}.concurrent-vs-serial.frame-{frame_index}",
                    payloads[left]["frames"][frame_index],
                    payloads[right]["frames"][frame_index],
                )
            )
    for attempt in ATTEMPTS:
        for frame_index in range(1, 4):
            pairs.append(
                _field_comparison(
                    f"{attempt}.constructor-vs-forward-{frame_index}",
                    payloads[attempt]["frames"][0],
                    payloads[attempt]["frames"][frame_index],
                )
            )
    require(len(pairs) == 28, "exact fixed 28 runtime frame comparisons")
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "all_exact_raw_bits": all(
            row["exact_raw_bits"] for pair in pairs for row in pair["fields"].values()
        ),
        "numerical_negative_outcomes_retained": True,
        "comparison_order": [
            "cinert",
            "cdof",
            "subtree_com",
            "crb",
            "remaining_fields",
        ],
        "no_tolerance_or_rerun": True,
    }


def _load_and_compare_run(source, run_root, run_report, previous, preflight):
    owner_pid = _run_owner_pid(run_report)
    inventory = _inventory(run_root, RUN_CONTENT_FILES)
    require(
        inventory == run_report["artifacts"],
        "run complete artifact inventory matches report before payload decode",
    )
    old_receipt = previous["_old_constructor_receipt"]
    private_cuda_start = old_receipt["private_states"]["cuda_start"]
    payloads = {}
    for attempt in ATTEMPTS:
        mode = SCHEDULE_MODES[attempt]
        declaration = {
            **run_report["declaration"],
            "declaration_sha256": run_report["declaration_sha256"],
        }
        raw_json, receipt = _read_child_receipt(
            run_root / (attempt + ".json"),
            source=source,
            mode=mode,
            attempt=attempt,
            declaration=declaration,
            expected_pid=run_report["children"][attempt]["pid"],
            owner_pid=owner_pid,
        )
        payload_raw = _read_file(run_root / receipt["payload_file"], PAYLOAD_LIMIT)
        require(
            inventory[receipt["payload_file"]]
            == {"bytes": len(payload_raw), "sha256": base.digest(payload_raw)},
            "complete child payload inventory authenticates before decode",
        )
        payload, _ = _load_payload(
            payload_raw,
            receipt,
            source=source,
            attempt=attempt,
            host_identity=run_report["host_identity"],
            versions=run_report["runtime_versions"],
            private_cuda_start=private_cuda_start,
            metadata_reference=preflight["declaration"]["runtime_metadata_reference"],
        )
        require(
            payload["mode"] == mode
            and receipt["caller_rng_sha256"]["cuda_before"]
            == receipt["caller_rng_sha256"]["cuda_after"],
            "unchanged caller CUDA RNG state retained",
        )
        payloads[attempt] = payload
    caller_cuda_states = {
        payloads[name]["caller_rng_states"]["cuda_before"].numpy().tobytes()
        for name in ATTEMPTS
    }
    require(
        len(caller_cuda_states) == 1,
        "fixed CUDA caller seed 677 yields retained identical starting state in all fresh children",
    )
    private_endpoints = {}
    for endpoint in ("cpu_start", "cpu_end", "cuda_start", "cuda_end"):
        raw_values = {
            payloads[name]["constructor_receipt"]["private_states"][endpoint]
            .numpy()
            .tobytes()
            for name in ATTEMPTS
        }
        require(
            len(raw_values) == 1,
            "frozen constructor private endpoint matches across all four children: "
            + endpoint,
        )
        private_endpoints[endpoint] = hashlib.sha256(next(iter(raw_values))).hexdigest()
    result = _compare_four(payloads)
    result.update(
        {
            "constructor_private_endpoint_sha256": private_endpoints,
            "constructor_private_cuda_seed_start_sha256": hashlib.sha256(
                private_cuda_start.numpy().tobytes()
            ).hexdigest(),
            "caller_cuda_seed": 677,
            "caller_cuda_seed_resimulated_on_cpu": False,
            "constructor_private_seed_provenance": "authenticated original constructor receipt; not CUDA-resimulated",
            "repeated_forwards_have_unchanged_kinematics_only": True,
            "complete_forward_scratch_state_equivalence_claimed": False,
            "physics_integration_or_dose_executed": False,
            **FLAGS,
        }
    )
    return result


def _run_owner_pid(report):
    owner_pid = report.get("owner_pid")
    service = report.get("service_properties")
    require(
        type(owner_pid) is int
        and owner_pid > 0
        and type(service) is dict
        and service.get("MainPID") == str(owner_pid),
        "published run owner PID binds the live run service",
    )
    return owner_pid


def _runtime_monitor(samples):
    child_pid = fulltree._monitor(samples)
    require(
        all(row["elapsed_seconds"] < CHILD_SECONDS for row in samples)
        and any(child_pid in row["sample"]["compute_pids"] for row in samples),
        "runtime child stays within 180 seconds and actual GPU PID was observed",
    )
    return child_pid


def _write_exclusive_json(path, value):
    raw = (canonical(value) + "\n").encode()
    base._write_exclusive(path, raw, JSON_LIMIT)
    return raw


def _legacy_summary(predecessors):
    legacy = predecessors["legacy"]
    return {
        "fulltree": predecessors["fulltree"],
        "original": legacy["original"],
        "concurrent": {
            key: value
            for key, value in legacy["concurrent"].items()
            if key not in ("fixture", "fixture_source")
        },
        "serial_owner_packet_sha256": legacy["serial_owner_packet_sha256"],
        "serial_closeout_packet_sha256": legacy["serial_closeout_packet_sha256"],
        "serial_run_files": legacy["serial_run_files"],
        "serial_test_files": legacy["serial_test_files"],
    }


def _constructor_anchor(old_ctor):
    private = old_ctor["private_states"]
    return {
        "protocol": constructor.PROTOCOL,
        "source": old_ctor["source"],
        "private_cpu_start_sha256": hashlib.sha256(
            private["cpu_start"].numpy().tobytes()
        ).hexdigest(),
        "private_cuda_start_sha256": hashlib.sha256(
            private["cuda_start"].numpy().tobytes()
        ).hexdigest(),
        "caller_cpu_preserved": old_ctor["caller_cpu_preserved"],
        "caller_cuda_preserved": old_ctor["caller_cuda_preserved"],
    }


def _run_service(source, mode, output, lease_fd):
    if mode == "tests":
        return prior._run_process(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *["tests/" + name for name in TEST_FILES],
            ],
            output / "pytest.log",
            SECONDS["tests"] - 30,
            env=_cpu_env(),
            fd=lease_fd,
        )
    raise ValueError("tests-only process service")


def _execute_tests(
    source,
    binding,
    host_identity,
    versions,
    service,
    output,
    lease_fd,
    idle_before,
    protected,
    filmbrain,
    started,
):
    require(
        EXPECTED_TESTS > 0,
        "owner freezes reviewed positive 53-file focused/native test count before test service",
    )
    _run_service(source, "tests", output, lease_fd)
    log = _read_file(output / "pytest.log", LOG_LIMIT)
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
    require(
        len(totals) == 1
        and int(totals[0][0]) == EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
        "full 53-file source tests pass without skips",
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
    receipt_raw = _write_exclusive_json(output / "receipt.json", receipt)
    idle_after = _close_context(
        source, "tests", binding, host_identity, versions, service, protected, filmbrain
    )
    report = {
        "protocol": PROTOCOL + ":tests",
        "source": source,
        "mode": "tests",
        "status": "passed",
        "source_binding": binding,
        "host_identity": host_identity,
        "runtime_versions": versions,
        "service_properties": service,
        "receipt_sha256": base.digest(receipt_raw),
        "passed": EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": receipt["pytest_sha256"],
        "idle_before": idle_before,
        "idle_after": idle_after,
        "filmbrain": filmbrain,
        "protected_services": protected,
        "elapsed_seconds": float(time.monotonic() - started),
        **FLAGS,
    }
    return report


def _execute_preflight(
    source,
    binding,
    host_identity,
    versions,
    source_hashes,
    service,
    output,
    lease_fd,
    idle_before,
    protected,
    filmbrain,
    started,
):
    base.training_smoke.inherited_lease(lease_fd)
    prior._hidden()
    tests_sha = _read_tests(source, binding)
    predecessors = _authenticate_all_predecessors()
    score = _score_original_pair()
    require(
        not torch.cuda.is_initialized(),
        "preflight remains CUDA-hidden through original trace scoring",
    )
    _host(source, predecessors["fulltree"]["host_identity"])
    schedule = prior.declaration(source)
    runtime_metadata_reference = _compiled_runtime_metadata_reference(source)
    _check_original_trace_topology(score["traces"], runtime_metadata_reference)
    old_ctor = score["old_constructor_receipt"]
    private = old_ctor["private_states"]
    constructor_anchor = {
        "protocol": constructor.PROTOCOL,
        "source": old_ctor["source"],
        "private_cpu_start_sha256": hashlib.sha256(
            private["cpu_start"].numpy().tobytes()
        ).hexdigest(),
        "private_cuda_start_sha256": hashlib.sha256(
            private["cuda_start"].numpy().tobytes()
        ).hexdigest(),
        "caller_cpu_preserved": old_ctor["caller_cpu_preserved"],
        "caller_cuda_preserved": old_ctor["caller_cuda_preserved"],
    }
    predecessor = _legacy_summary(predecessors)
    declaration = {
        "protocol": PROTOCOL + ":preflight",
        "source": source,
        "source_binding": binding,
        "host_identity": host_identity,
        "runtime_versions": versions,
        "service_properties": service,
        "tests_receipt_sha256": tests_sha,
        "source_hashes": source_hashes,
        "schedule": schedule,
        "attempts": list(ATTEMPTS),
        "worlds": WORLDS,
        "runtime_metadata_reference": runtime_metadata_reference,
        "original_trace_topology": {
            key: runtime_metadata_reference[key]
            for key in ("floor_id", "foot_ids", "control_ids")
        },
        "caller_cpu_seed": 673,
        "caller_cuda_seed": 677,
        "private_constructor": constructor_anchor,
        "old_strict_pair_failure": score["failure"],
        "predecessor_summary": predecessor,
        "predecessor_binding_sha256": _sha_summary(predecessor),
        "new_scope_contract": {
            "forward_calls": 4,
            "constructor_calls": 1,
            "original_level_requests": 28,
            "concurrent_accumulation_launches": 28,
            "serial_accumulation_launches": 36,
            "dense_qM_launches": 4,
        },
        "torch_cuda_initialized": False,
        **FLAGS,
    }
    declaration_raw = _write_exclusive_json(output / "declaration.json", declaration)
    _authenticate_all_predecessors()
    require(
        source_binding(source) == binding
        and _runtime_versions() == versions
        and _pinned_runtime_sources() == source_hashes,
        "unchanged source and installed kernels after CPU preflight",
    )
    require(not torch.cuda.is_initialized(), "preflight creates no CUDA context")
    idle_after = _close_context(
        source,
        "preflight",
        binding,
        host_identity,
        versions,
        service,
        protected,
        filmbrain,
    )
    report = {
        "protocol": PROTOCOL + ":preflight",
        "source": source,
        "mode": "preflight",
        "status": "passed",
        "source_binding": binding,
        "host_identity": host_identity,
        "runtime_versions": versions,
        "service_properties": service,
        "declaration_sha256": base.digest(declaration_raw),
        "tests_receipt_sha256": tests_sha,
        "predecessor_summary": predecessor,
        "predecessor_binding_sha256": declaration["predecessor_binding_sha256"],
        "original_strict_pair_failure": score["failure"],
        "private_constructor_anchor": constructor_anchor,
        "runtime_metadata_reference": runtime_metadata_reference,
        "torch_cuda_initialized": False,
        "idle_before": idle_before,
        "idle_after": idle_after,
        "filmbrain": filmbrain,
        "protected_services": protected,
        "elapsed_seconds": float(time.monotonic() - started),
        **FLAGS,
    }
    return report


def _read_preflight(source, binding, tests_sha, host_identity, versions, source_hashes):
    root = output_path(source, "preflight")
    base._exact_inventory(root, PREFLIGHT_FILES)
    declaration_raw = _read_file(root / "declaration.json", JSON_LIMIT)
    report_raw = _read_file(root / "report.json", JSON_LIMIT)
    require(
        base.digest(declaration_raw)
        == base.parse_json(report_raw).get("declaration_sha256"),
        "preflight declaration whole hash before parse",
    )
    declaration = base.parse_json(declaration_raw)
    report = base.parse_json(report_raw)
    metadata_reference = _compiled_runtime_metadata_reference(source)
    declaration_keys = {
        "protocol",
        "source",
        "source_binding",
        "host_identity",
        "runtime_versions",
        "service_properties",
        "tests_receipt_sha256",
        "source_hashes",
        "schedule",
        "attempts",
        "worlds",
        "runtime_metadata_reference",
        "original_trace_topology",
        "caller_cpu_seed",
        "caller_cuda_seed",
        "private_constructor",
        "old_strict_pair_failure",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "new_scope_contract",
        "torch_cuda_initialized",
        *FLAGS,
    }
    report_keys = {
        "protocol",
        "source",
        "mode",
        "status",
        "source_binding",
        "host_identity",
        "runtime_versions",
        "service_properties",
        "declaration_sha256",
        "tests_receipt_sha256",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "original_strict_pair_failure",
        "private_constructor_anchor",
        "runtime_metadata_reference",
        "torch_cuda_initialized",
        "idle_before",
        "idle_after",
        "filmbrain",
        "protected_services",
        "elapsed_seconds",
        "campaign_artifacts_before_report",
        *FLAGS,
    }
    expected_scope = {
        "forward_calls": 4,
        "constructor_calls": 1,
        "original_level_requests": 28,
        "concurrent_accumulation_launches": 28,
        "serial_accumulation_launches": 36,
        "dense_qM_launches": 4,
    }
    expected_trace_topology = {
        key: metadata_reference[key] for key in ("floor_id", "foot_ids", "control_ids")
    }
    require(
        type(declaration) is dict
        and set(declaration) == declaration_keys
        and declaration.get("protocol") == PROTOCOL + ":preflight"
        and declaration.get("source") == source
        and canonical(declaration.get("source_binding")) == canonical(binding)
        and canonical(declaration.get("host_identity")) == canonical(host_identity)
        and canonical(declaration.get("runtime_versions")) == canonical(versions)
        and canonical(declaration.get("service_properties"))
        == canonical(report.get("service_properties"))
        and declaration.get("tests_receipt_sha256") == tests_sha
        and canonical(declaration.get("source_hashes")) == canonical(source_hashes)
        and canonical(declaration.get("schedule"))
        == canonical(prior.declaration(source))
        and declaration.get("caller_cpu_seed") == 673
        and type(declaration.get("caller_cpu_seed")) is int
        and declaration.get("caller_cuda_seed") == 677
        and type(declaration.get("caller_cuda_seed")) is int
        and declaration.get("worlds") == WORLDS
        and type(declaration.get("worlds")) is int
        and declaration.get("attempts") == list(ATTEMPTS)
        and declaration.get("old_strict_pair_failure")
        == "paired rollout semantic state exactness"
        and _metadata_matches_reference(
            declaration.get("runtime_metadata_reference"), metadata_reference
        )
        and canonical(declaration.get("original_trace_topology"))
        == canonical(expected_trace_topology)
        and canonical(declaration.get("private_constructor"))
        == canonical(report.get("private_constructor_anchor"))
        and type(declaration.get("private_constructor")) is dict
        and set(declaration["private_constructor"])
        == {
            "protocol",
            "source",
            "private_cpu_start_sha256",
            "private_cuda_start_sha256",
            "caller_cpu_preserved",
            "caller_cuda_preserved",
        }
        and declaration["private_constructor"].get("protocol") == constructor.PROTOCOL
        and declaration["private_constructor"].get("caller_cpu_preserved") is True
        and declaration["private_constructor"].get("caller_cuda_preserved") is True
        and all(
            re.fullmatch(
                r"[0-9a-f]{64}", declaration["private_constructor"].get(key, "")
            )
            for key in ("private_cpu_start_sha256", "private_cuda_start_sha256")
        )
        and canonical(declaration.get("new_scope_contract"))
        == canonical(expected_scope)
        and _sha_summary(declaration.get("predecessor_summary"))
        == declaration.get("predecessor_binding_sha256")
        and re.fullmatch(
            r"[0-9a-f]{64}", declaration.get("predecessor_binding_sha256", "")
        )
        is not None
        and type(declaration.get("torch_cuda_initialized")) is bool
        and declaration.get("torch_cuda_initialized") is False
        and all(declaration.get(key) is False for key in FLAGS),
        "exact authenticated successful preflight declaration and CPU-compiled metadata",
    )
    require(
        type(report) is dict
        and set(report) == report_keys
        and report.get("protocol") == PROTOCOL + ":preflight"
        and report.get("source") == source
        and report.get("mode") == "preflight"
        and report.get("status") == "passed"
        and canonical(report.get("source_binding")) == canonical(binding)
        and canonical(report.get("host_identity")) == canonical(host_identity)
        and canonical(report.get("runtime_versions")) == canonical(versions)
        and canonical(report.get("host_identity"))
        == canonical(declaration.get("host_identity"))
        and canonical(report.get("runtime_versions"))
        == canonical(declaration.get("runtime_versions"))
        and canonical(report.get("service_properties"))
        == canonical(declaration.get("service_properties"))
        and report.get("tests_receipt_sha256") == tests_sha
        and report.get("predecessor_binding_sha256")
        == declaration.get("predecessor_binding_sha256")
        and canonical(report.get("predecessor_summary"))
        == canonical(declaration.get("predecessor_summary"))
        and report.get("original_strict_pair_failure")
        == declaration.get("old_strict_pair_failure")
        and canonical(report.get("private_constructor_anchor"))
        == canonical(declaration.get("private_constructor"))
        and _metadata_matches_reference(
            report.get("runtime_metadata_reference"),
            declaration.get("runtime_metadata_reference"),
        )
        and report.get("torch_cuda_initialized") is False
        and _valid_elapsed(report.get("elapsed_seconds"), SECONDS["preflight"])
        and _valid_campaign_summary(report.get("campaign_artifacts_before_report"))
        and canonical(report.get("campaign_artifacts_before_report"))
        == canonical(_stage_campaign_snapshot(source, "preflight"))
        and all(report.get(key) is False for key in FLAGS),
        "preflight report source/inputs/failure record",
    )
    _terminal(source, "preflight", report["service_properties"]["InvocationID"])
    return {
        "root": root,
        "declaration": declaration,
        "report": report,
        "declaration_sha256": base.digest(declaration_raw),
        "report_sha256": base.digest(report_raw),
    }


def _execute_run(
    source,
    binding,
    host_identity,
    versions,
    source_hashes,
    service,
    output,
    lease_fd,
    idle_before,
    protected,
    filmbrain,
    started,
):
    tests_sha = _read_tests(source, binding)
    preflight = _read_preflight(
        source, binding, tests_sha, host_identity, versions, source_hashes
    )
    predecessors = _authenticate_all_predecessors()
    score = _score_original_pair()
    predecessor = _legacy_summary(predecessors)
    require(
        _sha_summary(predecessor)
        == preflight["declaration"]["predecessor_binding_sha256"]
        and canonical(predecessor)
        == canonical(preflight["declaration"]["predecessor_summary"])
        and score["failure"] == preflight["declaration"]["old_strict_pair_failure"],
        "run reauthenticates all predecessors and original strict rejection",
    )
    require(
        canonical(_constructor_anchor(score["old_constructor_receipt"]))
        == canonical(preflight["declaration"]["private_constructor"]),
        "preflight constructor endpoints match authenticated original receipt",
    )
    require(
        source_hashes == preflight["declaration"]["source_hashes"]
        and canonical(versions)
        == canonical(preflight["declaration"]["runtime_versions"])
        and canonical(host_identity)
        == canonical(preflight["declaration"]["host_identity"]),
        "run exact preflight runtime/source binding",
    )
    declaration = {
        "protocol": PROTOCOL + ":run",
        "source": source,
        "source_binding": binding,
        "host_identity": host_identity,
        "runtime_versions": versions,
        "service_properties": service,
        "tests_receipt_sha256": tests_sha,
        "preflight_declaration_sha256": preflight["declaration_sha256"],
        "predecessor_summary": predecessor,
        "predecessor_binding_sha256": _sha_summary(predecessor),
        "schedule": prior.declaration(source),
        "runtime_metadata_reference": preflight["declaration"][
            "runtime_metadata_reference"
        ],
        "scope": {
            "max_forward_calls": 4,
            "modes": ["concurrent", "serial"],
            "children": list(ATTEMPTS),
        },
        "source_hashes": source_hashes,
        "private_cuda_start_sha256": preflight["declaration"]["private_constructor"][
            "private_cuda_start_sha256"
        ],
        **FLAGS,
    }
    declaration_raw = _write_exclusive_json(output / "declaration.json", declaration)
    child_declaration = {
        **declaration,
        "declaration_sha256": base.digest(declaration_raw),
    }
    children = {}
    for attempt in ATTEMPTS:
        log_path = output / (attempt + ".log")
        command = [
            sys.executable,
            "-m",
            MODULE,
            "child",
            "--source",
            source,
            "--attempt",
            attempt,
            "--lease-fd",
            str(lease_fd),
            "--owner-pid",
            str(os.getpid()),
            "--declaration-sha",
            child_declaration["declaration_sha256"],
        ]
        samples = prior._run_process(
            command, log_path, CHILD_SECONDS, env=_gpu_env(), fd=lease_fd, monitor=True
        )
        cuda_probe.check_log(log_path)
        child_pid = _runtime_monitor(samples)
        _log_raw = _read_file(log_path, LOG_LIMIT)
        receipt_raw, receipt = _read_child_receipt(
            output / (attempt + ".json"),
            source=source,
            mode=SCHEDULE_MODES[attempt],
            attempt=attempt,
            declaration=child_declaration,
            expected_pid=child_pid,
            owner_pid=os.getpid(),
        )
        payload_raw = _read_file(output / (attempt + ".pt"), PAYLOAD_LIMIT)
        require(
            base.digest(payload_raw) == receipt["payload_sha256"]
            and len(payload_raw) == receipt["payload_bytes"],
            "child payload retained bytes bind receipt before analyzer",
        )
        children[attempt] = {
            "pid": child_pid,
            "receipt_sha256": base.digest(receipt_raw),
            "payload_sha256": receipt["payload_sha256"],
            "monitor_samples": samples,
        }
        accumulated = _campaign_inventory(source)
        require(
            accumulated["total_bytes"] + JSON_LIMIT <= TOTAL_LIMIT,
            "combined prior/current runtime artifacts reserve final report within 16 MiB",
        )
    artifacts = _inventory(output, RUN_CONTENT_FILES)
    require(
        sum(item["bytes"] for item in artifacts.values()) <= TOTAL_LIMIT,
        "four child artifacts remain within 16 MiB total cap",
    )
    idle_after = _close_context(
        source, "run", binding, host_identity, versions, service, protected, filmbrain
    )
    report = {
        "protocol": PROTOCOL + ":run",
        "source": source,
        "mode": "run",
        "owner_pid": os.getpid(),
        "status": "passed",
        "source_binding": binding,
        "host_identity": host_identity,
        "runtime_versions": versions,
        "service_properties": service,
        "declaration_sha256": base.digest(declaration_raw),
        "declaration": declaration,
        "tests_receipt_sha256": tests_sha,
        "preflight_declaration_sha256": preflight["declaration_sha256"],
        "predecessor_summary": predecessor,
        "predecessor_binding_sha256": declaration["predecessor_binding_sha256"],
        "children": children,
        "artifacts": artifacts,
        "idle_before": idle_before,
        "idle_after": idle_after,
        "filmbrain": filmbrain,
        "protected_services": protected,
        "elapsed_seconds": float(time.monotonic() - started),
        **FLAGS,
    }
    return report


def _execute_closeout(
    source,
    binding,
    host_identity,
    versions,
    source_hashes,
    service,
    output,
    lease_fd,
    idle_before,
    protected,
    filmbrain,
    started,
):
    base.training_smoke.inherited_lease(lease_fd)
    prior._hidden()
    tests_sha = _read_tests(source, binding)
    preflight = _read_preflight(
        source, binding, tests_sha, host_identity, versions, source_hashes
    )
    run_root = output_path(source, "run")
    base._exact_inventory(run_root, RUN_FILES)
    run_report_raw = _read_file(run_root / "report.json", JSON_LIMIT)
    run_report = base.parse_json(run_report_raw)
    run_declaration_raw = _read_file(run_root / "declaration.json", JSON_LIMIT)
    require(
        base.digest(run_declaration_raw) == run_report.get("declaration_sha256"),
        "run declaration whole hash before parse",
    )
    run_declaration = base.parse_json(run_declaration_raw)
    run_declaration_keys = {
        "protocol",
        "source",
        "source_binding",
        "host_identity",
        "runtime_versions",
        "service_properties",
        "tests_receipt_sha256",
        "preflight_declaration_sha256",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "schedule",
        "runtime_metadata_reference",
        "scope",
        "source_hashes",
        "private_cuda_start_sha256",
        *FLAGS,
    }
    run_report_keys = {
        "protocol",
        "source",
        "mode",
        "owner_pid",
        "status",
        "source_binding",
        "host_identity",
        "runtime_versions",
        "service_properties",
        "declaration_sha256",
        "declaration",
        "tests_receipt_sha256",
        "preflight_declaration_sha256",
        "predecessor_summary",
        "predecessor_binding_sha256",
        "children",
        "artifacts",
        "idle_before",
        "idle_after",
        "filmbrain",
        "protected_services",
        "elapsed_seconds",
        "campaign_artifacts_before_report",
        *FLAGS,
    }
    require(
        type(run_report) is dict
        and set(run_report) == run_report_keys
        and type(run_declaration) is dict
        and set(run_declaration) == run_declaration_keys
        and run_report.get("protocol") == PROTOCOL + ":run"
        and run_report.get("source") == source
        and run_report.get("mode") == "run"
        and run_report.get("status") == "passed"
        and canonical(run_report.get("source_binding")) == canonical(binding)
        and canonical(run_report.get("host_identity")) == canonical(host_identity)
        and canonical(run_report.get("runtime_versions")) == canonical(versions)
        and canonical(run_report.get("declaration")) == canonical(run_declaration)
        and canonical(run_report.get("service_properties"))
        == canonical(run_declaration.get("service_properties"))
        and run_report.get("tests_receipt_sha256") == tests_sha
        and run_report.get("preflight_declaration_sha256")
        == preflight["declaration_sha256"]
        and run_declaration.get("protocol") == PROTOCOL + ":run"
        and run_declaration.get("source") == source
        and canonical(run_declaration.get("source_binding")) == canonical(binding)
        and canonical(run_declaration.get("host_identity")) == canonical(host_identity)
        and canonical(run_declaration.get("runtime_versions")) == canonical(versions)
        and canonical(run_declaration.get("tests_receipt_sha256"))
        == canonical(tests_sha)
        and run_declaration.get("preflight_declaration_sha256")
        == preflight["declaration_sha256"]
        and canonical(run_declaration.get("schedule"))
        == canonical(prior.declaration(source))
        and _metadata_matches_reference(
            run_declaration.get("runtime_metadata_reference"),
            preflight["declaration"]["runtime_metadata_reference"],
        )
        and canonical(run_declaration.get("scope"))
        == canonical(
            {
                "max_forward_calls": 4,
                "modes": ["concurrent", "serial"],
                "children": list(ATTEMPTS),
            }
        )
        and run_declaration.get("private_cuda_start_sha256")
        == preflight["declaration"]["private_constructor"]["private_cuda_start_sha256"]
        and _valid_elapsed(run_report.get("elapsed_seconds"), SECONDS["run"])
        and _valid_campaign_summary(run_report.get("campaign_artifacts_before_report"))
        and canonical(run_report.get("campaign_artifacts_before_report"))
        == canonical(_stage_campaign_snapshot(source, "run"))
        and all(run_report.get(key) is False for key in FLAGS),
        "exact successful four-child run report and declaration binding",
    )
    _terminal(source, "run", run_report["service_properties"]["InvocationID"])
    _terminal(
        source, "preflight", preflight["report"]["service_properties"]["InvocationID"]
    )
    predecessors = _authenticate_all_predecessors()
    score = _score_original_pair()
    predecessor_summary = _legacy_summary(predecessors)
    require(
        _sha_summary(predecessor_summary)
        == run_declaration["predecessor_binding_sha256"]
        == preflight["declaration"]["predecessor_binding_sha256"]
        and canonical(predecessor_summary)
        == canonical(run_declaration.get("predecessor_summary"))
        and canonical(run_report.get("predecessor_summary"))
        == canonical(run_declaration.get("predecessor_summary"))
        and canonical(predecessor_summary)
        == canonical(preflight["declaration"].get("predecessor_summary"))
        and score["failure"] == "paired rollout semantic state exactness",
        "independent CPU closeout reauthenticates all predecessor evidence and old strict failure",
    )
    require(
        canonical(_constructor_anchor(score["old_constructor_receipt"]))
        == canonical(preflight["declaration"]["private_constructor"]),
        "preflight constructor endpoints match authenticated original receipt",
    )
    require(
        run_declaration.get("source_hashes") == source_hashes
        and run_declaration.get("source_binding") == binding
        and run_declaration.get("host_identity") == host_identity
        and run_declaration.get("runtime_versions") == versions
        and _pinned_runtime_sources() == source_hashes,
        "unchanged exact source/kernel/dependency versions before payload interpretation",
    )
    _runtime_terminal_tests(source)
    numerical = _load_and_compare_run(
        source,
        run_root,
        run_report,
        {"_old_constructor_receipt": score["old_constructor_receipt"]},
        preflight,
    )
    require(
        time.monotonic() - started < SECONDS["closeout"],
        "independent CPU closeout stays within 300 seconds",
    )
    idle_after = _close_context(
        source,
        "closeout",
        binding,
        host_identity,
        versions,
        service,
        protected,
        filmbrain,
    )
    receipt = {
        "protocol": PROTOCOL + ":closeout-receipt",
        "source": source,
        "source_binding": binding,
        "run_report_sha256": base.digest(run_report_raw),
        "run_artifacts": run_report["artifacts"],
        "preflight_report_sha256": preflight["report_sha256"],
        "tests_receipt_sha256": tests_sha,
        "predecessor_summary_sha256": _sha_summary(predecessor_summary),
        "numerical_comparison": numerical,
        "comparison_is_runtime_admission": False,
        "original_cause_proven": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
        **FLAGS,
    }
    receipt_raw = _write_exclusive_json(output / "receipt.json", receipt)
    report = {
        "protocol": PROTOCOL + ":closeout",
        "source": source,
        "mode": "closeout",
        "status": "passed",
        "source_binding": binding,
        "host_identity": host_identity,
        "runtime_versions": versions,
        "service_properties": service,
        "receipt_sha256": base.digest(receipt_raw),
        "run_report_sha256": base.digest(run_report_raw),
        "numerical_comparison": numerical,
        "idle_before": idle_before,
        "idle_after": idle_after,
        "filmbrain": filmbrain,
        "protected_services": protected,
        "elapsed_seconds": float(time.monotonic() - started),
        **FLAGS,
    }
    return report


def _runtime_terminal_tests(source):
    tests = _read_file(output_path(source, "tests") / "receipt.json", JSON_LIMIT)
    value = base.parse_json(tests)
    _terminal(source, "tests", value["service_properties"]["InvocationID"])


def execute(source, mode):
    require(mode in MODES, "declared runtime owner mode")
    require(
        base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "exact native WSL profile",
    )
    reserve = {
        "tests": TEST_RESERVE,
        "preflight": PREFLIGHT_RESERVE,
        "run": RUN_RESERVE,
        "closeout": SECONDS["closeout"] + MARGIN,
    }[mode]
    _check_window(reserve_seconds=reserve)
    prior._hidden()
    binding = source_binding(source)
    service = _terminal_properties(source, mode, os.getpid())
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
        **FLAGS,
    }
    try:
        with base.gap.base.files.gpu_lease() as lease_fd:
            idle_before, protected, filmbrain = _read_idle_context()
            previous = _authenticate_previous_fulltree()
            host_identity, versions = _host(source, previous["host_identity"])
            source_hashes = _pinned_runtime_sources()
            require(
                not torch.cuda.is_initialized(),
                "CPU supervisor Torch CUDA remains uninitialized",
            )
            if mode == "tests":
                report = _execute_tests(
                    source,
                    binding,
                    host_identity,
                    versions,
                    service,
                    output,
                    lease_fd,
                    idle_before,
                    protected,
                    filmbrain,
                    started,
                )
            elif mode == "preflight":
                report = _execute_preflight(
                    source,
                    binding,
                    host_identity,
                    versions,
                    source_hashes,
                    service,
                    output,
                    lease_fd,
                    idle_before,
                    protected,
                    filmbrain,
                    started,
                )
            elif mode == "run":
                report = _execute_run(
                    source,
                    binding,
                    host_identity,
                    versions,
                    source_hashes,
                    service,
                    output,
                    lease_fd,
                    idle_before,
                    protected,
                    filmbrain,
                    started,
                )
            else:
                report = _execute_closeout(
                    source,
                    binding,
                    host_identity,
                    versions,
                    source_hashes,
                    service,
                    output,
                    lease_fd,
                    idle_before,
                    protected,
                    filmbrain,
                    started,
                )
            require(
                time.monotonic() - started < SECONDS[mode],
                "runtime owner remains within mode-specific cap",
            )
            require(
                not torch.cuda.is_initialized(),
                "CPU supervisor does not initialize Torch CUDA",
            )
            _close_context(
                source,
                mode,
                binding,
                host_identity,
                versions,
                service,
                protected,
                filmbrain,
            )
            names = (
                TEST_FILES_ON_DISK
                if mode == "tests"
                else PREFLIGHT_FILES
                if mode == "preflight"
                else RUN_FILES
                if mode == "run"
                else CLOSEOUT_FILES
            ) - {"report.json"}
            base._exact_inventory(output, names)
            _inventory(output, names)
            accumulated = _campaign_inventory(source)
            require(
                accumulated["total_bytes"] + JSON_LIMIT <= TOTAL_LIMIT,
                "combined runtime artifacts reserve final report within 16 MiB",
            )
            report["campaign_artifacts_before_report"] = accumulated
    except BaseException as error:
        report.update(
            status="failed-retained",
            error_type=type(error).__name__,
            error=str(error),
            partial_artifacts=_partial_inventory(output),
        )
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(output / "report.json", report)
    return report


def _host(source, previous_identity):
    identity = base.host.identity(source)
    versions = _runtime_versions()
    require(
        versions == fulltree._runtime_versions() and type(previous_identity) is dict,
        "all pinned native package/runtime versions",
    )
    current_host = {key: value for key, value in identity.items() if key != "source"}
    old_host = {
        key: value for key, value in previous_identity.items() if key != "source"
    }
    require(
        current_host == old_host and identity.get("driver") == "595.95",
        "pinned 595.95 host and full Python dependency-tree identity",
    )
    return identity, versions


def _pinned_runtime_sources():
    for module, expected, label in (
        (mw_smooth, SMOOTH_SHA256, "smooth"),
        (mw_forward, FORWARD_SHA256, "forward dispatch"),
    ):
        raw = _read_file(Path(module.__file__), 1024 * 1024)
        require(base.digest(raw) == expected, f"installed {label} whole source hash")
    return {"smooth_sha256": SMOOTH_SHA256, "forward_sha256": FORWARD_SHA256}


def _hash_packet(path, expected, label):
    raw = _read_file(base.execution.ROOT / path, JSON_LIMIT)
    require(base.digest(raw) == expected, label + " whole SHA before decode")
    return raw


def _historical_source_binding(binding):
    """Check the recorded D923 leaves with git-show, never a live old-source gate."""
    root = Path(__file__).resolve().parents[2]
    expected_paths = (
        fulltree._frozen_paths() | set(fulltree.OWN) | set(fulltree.RELATED)
    )
    require(
        type(binding) is dict
        and set(binding) == {"source", "branch", "leaves"}
        and binding.get("source") == HISTORIC_FULLTREE_SOURCE
        and binding.get("branch") == SOURCE_BRANCH
        and type(binding.get("leaves")) is dict
        and set(binding["leaves"]) == expected_paths,
        "exact historical D923 full-tree source-binding path set",
    )
    for path in sorted(expected_paths):
        raw = subprocess.run(
            ["git", "show", f"{HISTORIC_FULLTREE_SOURCE}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
        require(
            binding["leaves"][path] == base.digest(raw),
            "historical D923 source leaf " + path,
        )
    return {
        "source": HISTORIC_FULLTREE_SOURCE,
        "branch": SOURCE_BRANCH,
        "leaf_count": len(expected_paths),
        "leaves_sha256": base.digest((canonical(binding["leaves"]) + "\n").encode()),
    }


def _terminal_template(mode, invocation, *, peak=None):
    value = {
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
            "closeout": "5min",
        }[mode],
        "MemoryMax": str(256 * 1024**2 if mode == "sync" else MEMORY),
        "CPUQuotaPerSecUSec": "1s" if mode == "sync" else CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
        "InvocationID": invocation,
    }
    if peak is not None:
        value["MemoryPeak"] = str(peak)
    return value


def _authenticate_previous_fulltree():
    """Authenticate all frozen 50+20+4 predecessor leaves before tensor loads."""
    # These whole packet hashes are checked before any of the JSON is decoded.
    owner_raw = _hash_packet(
        FULLTREE_OWNER_PATH, FULLTREE_OWNER_SHA256, "full-tree owner packet"
    )
    terminal_raw = _hash_packet(
        FULLTREE_TERMINAL_PATH, FULLTREE_TERMINAL_SHA256, "full-tree terminal packet"
    )
    mac_raw = _hash_packet(
        FULLTREE_MAC_PATH, FULLTREE_MAC_SHA256, "full-tree Mac receiver"
    )
    clearance_raw = _hash_packet(
        FULLTREE_CLEARANCE_PATH, FULLTREE_CLEARANCE_SHA256, "full-tree clearance"
    )

    # Inventory all 20 source-tagged run/test/fixture artifacts before parsing any.
    roots = {
        "tests": fulltree.output_path(HISTORIC_FULLTREE_SOURCE, "tests"),
        "fixture": fulltree.output_path(HISTORIC_FULLTREE_SOURCE, "fixture"),
        "run": fulltree.output_path(HISTORIC_FULLTREE_SOURCE, "run"),
    }
    manifests = {
        "tests": fulltree._inventory(roots["tests"], fulltree.TEST_FILES_ON_DISK),
        "fixture": fulltree._inventory(roots["fixture"], fulltree.FIXTURE_FILES),
        "run": fulltree._inventory(roots["run"], fulltree.RUN_FILES),
    }
    require(
        sum(map(len, manifests.values())) == 20,
        "exact 20 prior full-tree retained files",
    )

    owner = base.parse_json(owner_raw)
    terminal = base.parse_json(terminal_raw)
    mac = base.parse_json(mac_raw)
    clearance = base.parse_json(clearance_raw)
    require(
        type(owner) is dict
        and owner.get("source") == HISTORIC_FULLTREE_SOURCE
        and owner.get("protocol") == fulltree.PROTOCOL + ":independent-owner-closeout"
        and owner.get("gpu_initialized") is False
        and owner.get("cause_proven") is False
        and owner.get("full_native_window_qualified") is False
        and owner.get("training_authorized") is False,
        "historical full-tree owner packet identity and non-admitting flags",
    )
    require(
        manifests["tests"] == owner["tests_files"]
        and manifests["fixture"] == owner["fixture_files"]
        and manifests["run"] == owner["run_files"],
        "all 20 prior full-tree artifact bytes match owner packet",
    )
    require(
        owner["tests_receipt_sha256"] == manifests["tests"]["receipt.json"]["sha256"],
        "full-tree predecessor tests receipt anchor",
    )
    require(
        owner["terminal"]
        == {
            "sync": _terminal_template("sync", FULLTREE_SYNC_INVOCATION),
            "tests": _terminal_template("tests", FULLTREE_TEST_INVOCATION),
            "fixture": _terminal_template("fixture", FULLTREE_FIXTURE_INVOCATION),
            "run": _terminal_template("run", FULLTREE_RUN_INVOCATION),
        },
        "all five historical full-tree invocations and resource caps",
    )
    require(
        owner["closeout_service_properties"]
        == {
            "ActiveState": "active",
            "CPUQuotaPerSecUSec": "2s",
            "InvocationID": FULLTREE_CLOSEOUT_INVOCATION,
            "KillMode": KILL_MODE,
            "MainPID": owner["closeout_service_properties"].get("MainPID"),
            "MemoryMax": str(MEMORY),
            "Nice": NICE,
            "RemainAfterExit": "yes",
            "RuntimeMaxUSec": "5min",
        }
        and str(owner["closeout_service_properties"]["MainPID"]).isdecimal()
        and int(owner["closeout_service_properties"]["MainPID"]) > 0,
        "historical full-tree closeout owner service record",
    )
    hist_binding = _historical_source_binding(owner["source_binding"])

    require(
        type(terminal) is dict
        and set(terminal)
        == {
            "gpu_initialized",
            "idle",
            "owner_packet_sha256",
            "protocol",
            "source",
            "terminal",
            "training_authorized",
        }
        and terminal["protocol"]
        == fulltree.PROTOCOL + ":independent-owner-closeout:terminal"
        and terminal["source"] == HISTORIC_FULLTREE_SOURCE
        and terminal["owner_packet_sha256"] == FULLTREE_OWNER_SHA256
        and terminal["gpu_initialized"] is False
        and terminal["training_authorized"] is False
        and terminal["terminal"]
        == _terminal_template(
            "closeout", FULLTREE_CLOSEOUT_INVOCATION, peak=1600233472
        ),
        "independent full-tree owner terminal authenticated and successful",
    )
    mac_flags = {
        "caller_rng_preserved": True,
        "cause_proven": False,
        "fresh_native_gpu_attestation": False,
        "full_window_qualified": False,
        "original_full_trajectory_scored_on_mac": False,
        "original_pair_accepted": False,
        "torch_cuda_initialized": False,
        "training_authorized": False,
    }
    require(
        type(mac) is dict
        and mac.get("protocol") == "football-b1d-crb-fulltree-mac-receiver-v2"
        and mac.get("source") == HISTORIC_FULLTREE_SOURCE
        and mac.get("owner_sha256") == FULLTREE_OWNER_SHA256
        and mac.get("terminal_sha256") == FULLTREE_TERMINAL_SHA256
        and mac.get("artifact_files_checked") == 20
        and all(mac.get(key) is value for key, value in mac_flags.items()),
        "full-tree Mac v2 receiver anchor and explicit false claims",
    )
    require(
        type(clearance) is dict
        and clearance.get("source") == HISTORIC_FULLTREE_SOURCE
        and clearance.get("protocol") == fulltree.PROTOCOL + ":completed-unit-clearance"
        and clearance.get("owner_sha256") == FULLTREE_OWNER_SHA256
        and clearance.get("terminal_sha256") == FULLTREE_TERMINAL_SHA256
        and clearance.get("mac_receiver_sha256") == FULLTREE_MAC_SHA256
        and clearance.get("artifacts_preserved") is True
        and clearance.get("training_authorized") is False,
        "full-tree completed-unit clearance anchor and retained artifacts",
    )
    return {
        "owner_sha256": FULLTREE_OWNER_SHA256,
        "terminal_sha256": FULLTREE_TERMINAL_SHA256,
        "mac_sha256": FULLTREE_MAC_SHA256,
        "clearance_sha256": FULLTREE_CLEARANCE_SHA256,
        "artifacts": manifests,
        "historical_source_binding": hist_binding,
        "host_identity": owner["host_identity"],
        "filmbrain": owner["filmbrain"],
        "protected_services": owner["protected_services"],
        "idle_before": owner["idle_before"],
        "idle_after": owner["idle_after"],
        "owner_terminal": terminal["terminal"],
    }


def _authenticate_all_predecessors():
    previous = _authenticate_previous_fulltree()
    legacy = fulltree._authenticate_predecessors()
    require(
        legacy.get("predecessors_authenticated") is True,
        "inherited 50 predecessor files authenticated",
    )
    original = legacy["original"]
    original_inputs = prior.order.authenticate_inputs()
    original_root, _original_launch, original_launch_sha = original_inputs[:3]
    original_anchors = original_inputs[4]
    require(
        type(original) is dict
        and set(original)
        == {
            "root",
            "launch_sha256",
            "original_files",
            "anchors",
            "comparison_sha256",
            "report_sha256",
        }
        and original.get("root") == str(original_root)
        and original.get("launch_sha256") == original_launch_sha
        and original.get("anchors") == original_anchors
        and original.get("comparison_sha256") == fulltree.frozen.ORDER_COMPARISON_SHA
        and original.get("report_sha256") == fulltree.frozen.ORDER_REPORT_SHA
        and {key: len(value) for key, value in original["original_files"].items()}
        == {"rollout": 22, "diagnosis": 1, "component_map": 2, "order_oracle": 2},
        "exact authenticated 27-file original rollout/order evidence and anchors",
    )
    return {"fulltree": previous, "legacy": legacy, "predecessors_authenticated": True}


def _score_original_pair():
    order = prior.order
    order_inputs = order.authenticate_inputs()
    root, launch, sha = order_inputs[0], order_inputs[1], order_inputs[2]
    rng_before = torch.random.get_rng_state().clone()
    inputs, _scores, traces = prior._score_inputs(root, order.PARENT, launch, sha)
    failure = None
    try:
        prior._strict_pair(inputs, traces)
    except ValueError as error:
        failure = str(error)
    require(
        failure == "paired rollout semantic state exactness",
        "original strict pair rejection reproduced exactly",
    )
    require(
        torch.equal(rng_before, torch.random.get_rng_state()),
        "CPU original-source scoring preserves Torch CPU RNG",
    )
    require(
        len(inputs) == len(traces) == 2,
        "both authenticated original forward attempts scored",
    )
    old_receipt = inputs[0]["constructor_receipt"]
    return {
        "order_inputs": order_inputs,
        "failure": failure,
        "inputs": inputs,
        "traces": traces,
        "old_constructor_receipt": old_receipt,
    }


def _tensor_raw(value, label, shape):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and value.layout == torch.strided
        and tuple(value.shape) == shape
        and value.is_contiguous()
        and bool(torch.isfinite(value).all()),
        "owned finite CPU float32 " + label,
    )
    return value.detach().contiguous().numpy().tobytes(order="C")


def _snapshot_frame(env):
    env._sync()
    frame = {}
    for name, suffix in FRAME_FIELDS.items():
        shape = (WORLDS, *suffix) if suffix else (WORLDS,)
        value = env._view(name).detach().cpu().contiguous().clone()
        _tensor_raw(value, name, shape)
        frame[name] = value
    return frame


def _runtime_metadata(env):
    require(
        type(env) is constructor.ScheduledRecoveryRuntime and env.n == WORLDS,
        "exact constructed CUDA64 scheduled runtime",
    )
    require(
        str(env.device) == "cuda:0" and str(env.wp_device) == "cuda:0",
        "runtime and Warp binding remain on physical CUDA0",
    )
    require(
        (env.native.nbody, env.native.nq, env.native.nv, env.native.nu)
        == (16, 21, 20, 14)
        and not env.model.is_sparse,
        "exact dense stance runtime model dimensions",
    )
    require(
        env.steps.shape == (WORLDS,)
        and env.steps.dtype == torch.long
        and not bool(torch.count_nonzero(env.steps)),
        "physics counter remains zero without integration",
    )
    return {
        "runtime_type": f"{type(env).__module__}.{type(env).__qualname__}",
        "worlds": int(env.n),
        "nbody": int(env.native.nbody),
        "nq": int(env.native.nq),
        "nv": int(env.native.nv),
        "nu": int(env.native.nu),
        "floor_id": int(env.floor),
        "foot_ids": [int(value) for value in env.feet],
        "control_ids": [int(value) for value in env.ctrl_ids.detach().cpu().tolist()],
        "physics_steps": [int(value) for value in env.steps.detach().cpu().tolist()],
        "plant_binding": deepcopy(env.binding),
        "schedule": deepcopy(env.schedule_declaration),
    }


def _compiled_runtime_metadata_reference(source):
    """Fresh source-derived CPU compile for exact child ID/plant metadata."""
    schedule = prior.declaration(source)
    entity = stance_warp_runtime.build_entity()
    native = entity.compile()
    require(
        (native.nbody, native.nq, native.nv, native.nu) == (16, 21, 20, 14),
        "fresh CPU-compiled scheduled plant dimensions",
    )
    expected_names = tuple(plant_evidence.JOINTS)
    actuator_names = tuple(
        mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(native.nu)
    )
    actuator_joints = [
        mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_JOINT, int(value))
        for value in native.actuator_trnid[:, 0]
    ]
    require(
        actuator_names == expected_names and tuple(actuator_joints) == expected_names,
        "CPU plant actuator control order and joint IDs",
    )
    body_names = [
        mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        for index in range(native.nbody)
    ]
    body_name = recovery_contract.force.BODY_NAME
    require(body_name in body_names, "CPU plant contains the pinned trunk body")
    plant = plant_evidence.describe(native)
    binding = {
        "selected_plant": plant,
        "nbody": native.nbody,
        "body_names": body_names,
        "body_name": body_name,
        "body_id": int(native.body(body_name).id),
    }
    recovery_contract.force._binding(binding)
    floor_id = int(native.geom("hold_floor").id)
    foot_ids = [int(native.geom(name).id) for name in plant_evidence.FEET]
    control_ids = list(range(native.nu))
    return {
        "runtime_type": "mjlab_microduck.stance_recovery_schedule_runtime.ScheduledRecoveryRuntime",
        "worlds": WORLDS,
        "nbody": native.nbody,
        "nq": native.nq,
        "nv": native.nv,
        "nu": native.nu,
        "floor_id": floor_id,
        "foot_ids": foot_ids,
        "control_ids": control_ids,
        "physics_steps": [0] * WORLDS,
        "plant_binding": binding,
        "schedule": schedule,
    }


def _metadata_matches_reference(metadata, reference):
    expected_keys = {
        "runtime_type",
        "worlds",
        "nbody",
        "nq",
        "nv",
        "nu",
        "floor_id",
        "foot_ids",
        "control_ids",
        "physics_steps",
        "plant_binding",
        "schedule",
    }
    return (
        type(metadata) is dict
        and set(metadata) == expected_keys
        and type(reference) is dict
        and set(reference) == expected_keys
        and canonical(metadata) == canonical(reference)
    )


def _check_original_trace_topology(traces, metadata_reference):
    require(
        type(traces) in (list, tuple) and len(traces) == 2,
        "both authenticated original early traces retained",
    )
    for trace in traces:
        require(
            type(trace) is dict
            and canonical(
                {key: trace.get(key) for key in ("floor_id", "foot_ids", "control_ids")}
            )
            == canonical(
                {
                    key: metadata_reference[key]
                    for key in ("floor_id", "foot_ids", "control_ids")
                }
            ),
            "runtime metadata IDs match authenticated original trace topology",
        )


def _caller_rng_seed_states():
    cpu = torch.Generator(device="cpu").manual_seed(673).get_state().clone()
    return {"cpu_seed_673": cpu}


def _validate_constructor_receipt(
    receipt, *, source, mode, caller_states, private_reference=None
):
    require(
        type(receipt) is dict
        and set(receipt) == constructor.KEYS
        and receipt.get("protocol") == constructor.PROTOCOL
        and receipt.get("source") == source
        and receipt.get("status") == "success"
        and type(receipt.get("constructor_calls")) is int
        and receipt.get("constructor_calls") == 1
        and receipt.get("faulted") is False
        and receipt.get("error_type") is None
        and receipt.get("error") is None,
        "successful exact original private-seed constructor receipt",
    )
    require(
        type(receipt.get("cpu_seed")) is int
        and receipt.get("cpu_seed") == constructor.CPU_SEED
        and type(receipt.get("cuda_seed")) is int
        and receipt.get("cuda_seed") == constructor.CUDA_SEED
        and type(receipt.get("worlds")) is int
        and receipt.get("worlds") == WORLDS,
        "frozen private constructor seeds and world count",
    )
    require(
        receipt.get("caller_cpu_preserved") is True
        and receipt.get("caller_cuda_preserved") is True,
        "constructor preserves actual caller streams",
    )
    states = receipt.get("caller_states")
    require(
        type(states) is dict and set(states) == constructor.CALLER_KEYS,
        "exact caller RNG receipt states",
    )
    require(
        torch.equal(states["cpu_before"], caller_states["cpu_before"])
        and torch.equal(states["cpu_after"], caller_states["cpu_after"])
        and torch.equal(states["cuda_before"], caller_states["cuda_before"])
        and torch.equal(states["cuda_after"], caller_states["cuda_after"]),
        "constructor receipt binds payload caller RNG bytes",
    )
    require(
        torch.equal(states["cpu_before"], caller_states["cpu_after"])
        and torch.equal(states["cuda_before"], caller_states["cuda_after"]),
        "caller RNG state unchanged through all four forwards",
    )
    private = receipt.get("private_states")
    require(
        type(private) is dict and set(private) == constructor.PRIVATE_KEYS,
        "exact constructor-private RNG receipt states",
    )
    all_states = {**states, **private}
    for key, value in all_states.items():
        constructor._state(value, "runtime " + key)
    for device in ("cpu", "cuda"):
        caller_shape = states[device + "_before"].shape
        require(
            states[device + "_after"].shape == caller_shape
            and private[device + "_start"].shape == caller_shape
            and private[device + "_end"].shape == caller_shape,
            "exact fixed " + device + " RNG state layouts",
        )
    expected_cpu_start = (
        torch.Generator(device="cpu").manual_seed(constructor.CPU_SEED).get_state()
    )
    require(
        torch.equal(private["cpu_start"], expected_cpu_start),
        "frozen constructor private CPU seed start",
    )
    constructor._nominal(receipt["nominal_parameters"])
    if private_reference is not None:
        require(
            torch.equal(private["cuda_start"], private_reference),
            "private CUDA constructor seed start matches authenticated prior constructor receipt",
        )
    require(
        type(receipt.get("state_sha256")) is dict
        and set(receipt["state_sha256"]) == set(all_states)
        and all(
            receipt["state_sha256"][k]
            == hashlib.sha256(v.numpy().tobytes()).hexdigest()
            for k, v in all_states.items()
        ),
        "constructor receipt RNG state digests",
    )
    _validate_flags(receipt, constructor.FLAGS, "constructor")
    return private


def _validate_flags(value, expected, label):
    for key, truth in expected.items():
        require(
            type(value.get(key)) is bool and value.get(key) is truth,
            f"{label} non-admission flag {key}",
        )


def _validate_scope_receipt(receipt, *, source, mode):
    base._hex(source, 40, "runtime scope source")
    expected_keys = {
        "protocol",
        "status",
        "mode",
        "max_forward_calls",
        "smooth_source_sha256",
        "forward_source_sha256",
        "topology_id_snapshots",
        "singleton_child_arrays_allocated",
        "initialization_timing_changed",
        "constructor_forward_calls",
        "constructor_forward_covered",
        "runtime_bound",
        "forward_calls",
        "original_level_launch_requests",
        "parent_zero_noop_level_requests",
        "actual_accumulation_launches",
        "split_child_launches",
        "dense_qM_launches",
        "fault_type",
        "flags",
    }
    require(
        type(receipt) is dict
        and set(receipt) == expected_keys
        and receipt.get("protocol") == runtime_control.PROTOCOL
        and receipt.get("mode") == mode
        and receipt.get("status") == "complete"
        and receipt.get("smooth_source_sha256") == SMOOTH_SHA256
        and receipt.get("forward_source_sha256") == FORWARD_SHA256,
        "exact process-local runtime scope completion and installed source pins",
    )
    integer_fields = (
        "max_forward_calls",
        "topology_id_snapshots",
        "singleton_child_arrays_allocated",
        "constructor_forward_calls",
        "forward_calls",
        "original_level_launch_requests",
        "parent_zero_noop_level_requests",
        "actual_accumulation_launches",
        "split_child_launches",
        "dense_qM_launches",
    )
    require(
        all(type(receipt.get(key)) is int for key in integer_fields),
        "plain integer CRB scope counters",
    )
    require(
        receipt.get("max_forward_calls") == 4
        and receipt.get("constructor_forward_calls") == 1
        and receipt.get("constructor_forward_covered") is True
        and receipt.get("runtime_bound") is True
        and receipt.get("forward_calls") == 4
        and receipt.get("original_level_launch_requests") == 28,
        "constructor plus three eager forwards with original seven levels each",
    )
    require(
        receipt.get("topology_id_snapshots") == 4
        and receipt.get("singleton_child_arrays_allocated") == 3
        and receipt.get("initialization_timing_changed") is True
        and receipt.get("parent_zero_noop_level_requests") == 8
        and receipt.get("actual_accumulation_launches")
        == (36 if mode == "serial" else 28)
        and receipt.get("split_child_launches") == (12 if mode == "serial" else 0)
        and receipt.get("dense_qM_launches") == 4
        and receipt.get("fault_type") is None,
        "exact original, split, dense-qM and parent-zero launch counters",
    )
    require(
        type(receipt.get("flags")) is dict
        and set(receipt["flags"]) == set(runtime_control.FLAGS),
        "exact CRB scope flag inventory",
    )
    _validate_flags(receipt.get("flags"), runtime_control.FLAGS, "runtime scope")
    return receipt


def _validate_payload(
    payload,
    *,
    source,
    mode,
    attempt,
    host_identity,
    runtime_versions,
    metadata_reference,
    private_reference=None,
):
    keys = {
        "protocol",
        "source",
        "mode",
        "attempt",
        "owner_pid",
        "child_pid",
        "child_ppid",
        "host_identity",
        "runtime_versions",
        "runtime_metadata",
        "frames",
        "caller_rng_states",
        "constructor_receipt",
        "scope_receipt",
        *FLAGS,
    }
    require(
        type(payload) is dict
        and set(payload) == keys
        and payload.get("protocol") == PROTOCOL + ":child-payload"
        and payload.get("source") == source
        and payload.get("mode") == mode
        and payload.get("attempt") == attempt
        and payload.get("host_identity") == host_identity
        and payload.get("runtime_versions") == runtime_versions,
        "exact bounded child payload provenance schema",
    )
    require(
        type(payload["owner_pid"]) is int
        and payload["owner_pid"] > 0
        and type(payload["child_pid"]) is int
        and payload["child_pid"] > 0
        and type(payload["child_ppid"]) is int
        and payload["child_ppid"] == payload["owner_pid"],
        "actual retained parent/child PID chain",
    )
    caller = payload["caller_rng_states"]
    require(
        type(caller) is dict
        and set(caller) == {"cpu_before", "cpu_after", "cuda_before", "cuda_after"},
        "exact caller RNG state payload",
    )
    require(
        all(
            torch.is_tensor(value)
            and value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.is_contiguous()
            and 0 < value.numel() <= 16384
            for value in caller.values()
        ),
        "bounded CPU caller RNG states",
    )
    require(
        torch.equal(caller["cpu_before"], caller["cpu_after"])
        and torch.equal(caller["cuda_before"], caller["cuda_after"])
        and torch.equal(
            caller["cpu_before"], _caller_rng_seed_states()["cpu_seed_673"]
        ),
        "CPU seed 673 and unchanged caller CPU/CUDA RNG states",
    )
    metadata = payload["runtime_metadata"]
    require(
        type(metadata) is dict
        and set(metadata)
        == {
            "runtime_type",
            "worlds",
            "nbody",
            "nq",
            "nv",
            "nu",
            "floor_id",
            "foot_ids",
            "control_ids",
            "physics_steps",
            "plant_binding",
            "schedule",
        }
        and _metadata_matches_reference(metadata, metadata_reference)
        and metadata.get("runtime_type")
        == "mjlab_microduck.stance_recovery_schedule_runtime.ScheduledRecoveryRuntime"
        and metadata.get("worlds") == WORLDS
        and (
            metadata.get("nbody"),
            metadata.get("nq"),
            metadata.get("nv"),
            metadata.get("nu"),
        )
        == (16, 21, 20, 14)
        and metadata.get("physics_steps") == [0] * WORLDS
        and metadata.get("schedule") == prior.declaration(source),
        "exact CPU-compiled runtime IDs/plant/schedule metadata and zero physics counter",
    )
    frames = payload["frames"]
    require(
        type(frames) is list and len(frames) == 4,
        "exact constructor frame and three eager-forward frames",
    )
    frame_raws = []
    for frame_index, frame in enumerate(frames):
        require(
            type(frame) is dict and set(frame) == set(FRAME_FIELDS),
            f"exact frame field inventory {frame_index}",
        )
        frame_raws.append(
            {
                name: _tensor_raw(
                    frame[name],
                    f"frame {frame_index} {name}",
                    (WORLDS, *suffix) if suffix else (WORLDS,),
                )
                for name, suffix in FRAME_FIELDS.items()
            }
        )
    for field in (
        "qpos",
        "qvel",
        "time",
        "qacc_warmstart",
        "ctrl",
        "xfrc_applied",
        "qfrc_applied",
    ):
        require(
            all(frame_raws[0][field] == frame[field] for frame in frame_raws[1:]),
            "fixed kinematics/controls invariant across observed forwards: " + field,
        )
    for field in ("qvel", "ctrl", "xfrc_applied", "qfrc_applied"):
        require(
            not bool(torch.count_nonzero(frames[0][field])),
            "zero unforced field " + field,
        )
    require(
        not bool(torch.count_nonzero(frames[0]["time"]))
        and not bool(torch.count_nonzero(frames[0]["qacc_warmstart"])),
        "zero runtime time and warmstart before integration",
    )
    private = _validate_constructor_receipt(
        payload["constructor_receipt"],
        source=source,
        mode=mode,
        caller_states=caller,
        private_reference=private_reference,
    )
    _validate_scope_receipt(payload["scope_receipt"], source=source, mode=mode)
    _validate_flags(payload, FLAGS, "child payload")
    return {"frames": frame_raws, "private_states": private}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=(*MODES, "child"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--attempt", choices=ATTEMPTS)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--declaration-sha")
    args = parser.parse_args()
    if args.mode == "child":
        require(
            args.attempt in ATTEMPTS
            and args.lease_fd is not None
            and args.owner_pid is not None
            and args.declaration_sha is not None,
            "exact private four-child invocation arguments",
        )
        result = _child(
            args.source,
            args.attempt,
            args.lease_fd,
            args.owner_pid,
            args.declaration_sha,
        )
    else:
        require(
            args.attempt is None
            and args.lease_fd is None
            and args.owner_pid is None
            and args.declaration_sha is None,
            "no child-only args in CPU owner process",
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


if __name__ == "__main__":
    main()
