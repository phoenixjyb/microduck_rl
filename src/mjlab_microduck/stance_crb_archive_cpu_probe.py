"""Capped WSL CPU witness for the split CRB archive; no GPU or training gate.

The run mode performs one controlled 64-world scheduled CPU tick and archives
the completed launch trace. This single tick covers steps 0-9 before the
declared pulse onset; it makes no force-active execution claim. It is not a
substitute for the original native window, an uninstrumented replay, or any
physical/learned-skill qualification.
"""

import argparse
import io
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import time

import torch
import warp as wp
from mujoco_warp._src import smooth

from mjlab_microduck import stance_crb_launch_trace as crb
from mjlab_microduck import stance_crb_trace_archive as archive
from mjlab_microduck import stance_inertia_component_probe as component
from mjlab_microduck import stance_inertia_order_probe as order
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
MODULE = "mjlab_microduck.stance_crb_archive_cpu_probe"
PROTOCOL = "football-b1d-crb-archive-cpu-probe-20261005-v1"
BASE_SOURCE = "90f8edc7230ed8f7aac3b7e0c2b1ec230d9f1e81"
SYNC_FROM_SOURCE = "766271542f4c168ebe00c56e8e03755c3f44be2b"
SYNC_ROOT = prior.SYNC_ROOT
SYNC_ORIGIN = prior.SYNC_ORIGIN
CAP = 300
MEMORY = 6 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
CHILD_SECONDS = 240
MARGIN = 60
RUN_RESERVE = CAP + MARGIN
CUTOFF = prior.CUTOFF
EXPECTED_TESTS = (
    1374  # Owner-frozen 44-file suite after retained-unit lifecycle review.
)
DOC = "docs/experiments/2026-10-05-cuda64-no-update-rollout.md"
CRB_TESTS = (
    "test_stance_crb_launch_trace.py",
    "test_stance_crb_trace_archive.py",
)
TEST_FILES = order.TEST_FILES + CRB_TESTS + ("test_stance_crb_archive_cpu_probe.py",)
OWN = (
    "src/mjlab_microduck/stance_crb_archive_cpu_probe.py",
    "tests/test_stance_crb_archive_cpu_probe.py",
    DOC,
)
ALLOWED = set(OWN)
FROZEN_FILES = tuple(
    sorted(
        (
            set(prior.OWN_FILES)
            | set(component.OWN)
            | set(order.OWN)
            | {
                "src/mjlab_microduck/stance_crb_launch_trace.py",
                "src/mjlab_microduck/stance_crb_trace_archive.py",
                *("tests/" + name for name in CRB_TESTS),
            }
        )
        - {DOC}
    )
)
ARCHIVE_FILES = {
    "declaration.json",
    "inertia.pt",
    "stages.bin",
    "manifest.json",
    "first-record.pt",
    "rng.pt",
}
RUN_FILES = ARCHIVE_FILES | {"witness.json", "child.log", "report.json"}
TEST_FILES_ON_DISK = {"receipt.json", "pytest.log", "report.json"}
FLAGS = {
    **crb.FLAGS,
    "cpu_archive_roundtrip_qualified": False,
    "full_native_window_qualified": False,
    "optimizer_update_qualified": False,
    "execution_admitted": False,
    "training_update_performed": False,
}
_WITNESS_KEYS = {
    "protocol",
    "source",
    "source_binding",
    "service_properties",
    "original_files",
    "original_anchors",
    "original_pair_accepted",
    "original_pair_error",
    "original_launch_sha256",
    "files",
    "trace_score",
    "trace_tree_sha256_before",
    "trace_tree_sha256_after",
    "caller_rng_preserved",
    "cpu_only",
    "pre_pulse_only",
    "shared_lease_verified",
    "full_native_window_qualified",
    "optimizer_steps",
    *FLAGS,
}


def unit(source, mode):
    base._hex(source, 40, "exact CPU archive probe source")
    require(mode in ("tests", "run"), "declared CPU archive probe mode")
    return f"microduck-crb-archive-cpu-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    area = "tools" if mode == "tests" else "evaluations"
    tag = "tests" if mode == "tests" else "witness"
    return (
        base.execution.ROOT
        / "artifacts"
        / area
        / f"stance-crb-archive-cpu-{tag}-{source[:12]}"
    )


def source_sync_unit(source):
    base._hex(source, 40, "exact CPU archive bootstrap source")
    return f"microduck-crb-archive-cpu-sync-{source[:12]}.service"


def source_sync_script(source, *, bundle_sha256):
    """Retarget only the frozen sync template's expected HEAD and own unit."""
    script = prior.source_sync_script(source, bundle_sha256=bundle_sha256)
    replacements = (
        (
            f'test "$(git rev-parse HEAD)" = {prior.SYNC_FROM_SOURCE}\n',
            f'test "$(git rev-parse HEAD)" = {SYNC_FROM_SOURCE}\n',
        ),
        (
            f'if test "$duck_sync_running" != {prior.source_sync_unit(source)}; then\n',
            f'if test "$duck_sync_running" != {source_sync_unit(source)}; then\n',
        ),
    )
    for old, new in replacements:
        require(script.count(old) == 1, "exact frozen source-sync guard")
        script = script.replace(old, new, 1)
    return script


def source_binding(source):
    base._hex(source, 40, "exact committed CPU archive source")
    root = Path(__file__).resolve().parents[2]
    require(str(root) == SYNC_ROOT, "exact native WSL source root")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=15
        ).stdout

    require(
        git("rev-parse", "HEAD").decode().strip() == source
        and git("branch", "--show-current").decode().strip()
        == "feat/athletics-obstacle-curriculum"
        and git("remote", "get-url", "origin").decode().strip() == SYNC_ORIGIN
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
    require(changed <= ALLOWED, "only declared CRB archive probe changes")
    leaves = {}
    for path in sorted(set(FROZEN_FILES) | set(OWN)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(raw == git("show", f"{source}:{path}"), "committed source leaf " + path)
        if path in FROZEN_FILES:
            require(
                raw == git("show", f"{BASE_SOURCE}:{path}"),
                "frozen prior/source leaf " + path,
            )
        leaves[path] = base.digest(raw)
    return dict(
        source=source, branch="feat/athletics-obstacle-curriculum", leaves=leaves
    )


def declaration(source):
    return prior.declaration(source)


def _recorded_properties(value):
    expected_keys = {
        "MainPID",
        "ActiveState",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "Nice",
        "KillMode",
        "InvocationID",
        "RemainAfterExit",
    }
    require(
        type(value) is dict and set(value) == expected_keys,
        "exact CPU service properties",
    )
    base._hex(value["InvocationID"], 32, "CPU archive service invocation")
    require(
        type(value["MainPID"]) is str
        and value["MainPID"].isdecimal()
        and int(value["MainPID"]) > 0
        and {k: v for k, v in value.items() if k not in {"MainPID", "InvocationID"}}
        == {
            "ActiveState": "active",
            "RuntimeMaxUSec": f"{CAP // 60}min",
            "MemoryMax": str(MEMORY),
            "CPUQuotaPerSecUSec": CPU_QUOTA,
            "Nice": NICE,
            "KillMode": KILL_MODE,
            "RemainAfterExit": "yes",
        },
        "exact CPU archive service caps",
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
    _recorded_properties(values)
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
        "only exact owned CPU archive Duck service active",
    )
    return values


def _owned_properties(source, mode, owner_pid):
    values = _service_values(source, mode)
    require(values["MainPID"] == str(owner_pid), "service main PID is expected owner")
    return values


def _authenticate_original_without_loading(source):
    binding = source_binding(source)
    _root, launch, launch_sha, inventory, anchors = order.authenticate_inputs()
    require(
        launch["source"] == order.PARENT
        and launch["schedule"]["worlds"] == 64
        and anchors["smooth_sha256"] == order.SMOOTH_SHA,
        "exact original failure inputs and installed smooth source",
    )
    return dict(
        source_binding=binding,
        original_files=inventory,
        original_anchors=anchors,
        original_launch_sha256=launch_sha,
        original_pair_accepted=False,
        original_pair_error="paired rollout semantic state exactness",
    )


def _hidden():
    prior._hidden()
    require(
        base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "exact WSL CPU archive environment",
    )


def _file_limit(name):
    if name == "stages.bin":
        return archive.MAX_ARTIFACT_BYTES
    if name.endswith(".pt"):
        return base.RAW_LIMIT
    if name.endswith(".log"):
        return base.LOG_LIMIT
    return base.JSON_LIMIT


def _inventory(root, names=None):
    if names is None:
        names = {path.name for path in root.iterdir()}
    result = {}
    for name in sorted(names):
        raw = base._read_file(root / name, _file_limit(name))
        result[name] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return result


def _partial_inventory(root, mode):
    """Describe only fixed known artifact names; never recurse or follow links."""
    names = (
        {"pytest.log", "receipt.json"}
        if mode == "tests"
        else ARCHIVE_FILES | {"declaration.json", "witness.json", "child.log"}
    )
    result = {}
    for name in sorted(names):
        path = root / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            result[name] = {"present": False, "readable": False}
            continue
        except OSError as error:
            result[name] = {
                "present": True,
                "readable": False,
                "error_type": type(error).__name__,
            }
            continue
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            result[name] = {
                "present": True,
                "readable": False,
                "error_type": "UnsafeFile",
            }
            continue
        try:
            raw = base._read_file(path, _file_limit(name))
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


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device == right.device
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.contiguous().view(torch.uint8).numpy().tobytes()
            == right.contiguous().view(torch.uint8).numpy().tobytes()
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            type(left) is dict
            and type(right) is dict
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if type(left) is float or type(right) is float:
        return (
            type(left) is float
            and type(right) is float
            and struct.pack("<d", left) == struct.pack("<d", right)
        )
    return type(left) is type(right) and left == right


def _tree_digest(value):
    digest = sha256()

    def visit(item):
        if torch.is_tensor(item):
            require(
                item.device.type == "cpu" and item.is_contiguous(),
                "owned CPU trace tensor",
            )
            digest.update(b"tensor\0")
            digest.update(str(item.dtype).encode() + b"\0")
            digest.update(canonical(list(item.shape)).encode() + b"\0")
            digest.update(item.view(torch.uint8).numpy().tobytes())
        elif type(item) is dict:
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(canonical(key).encode() + b"\0")
                visit(item[key])
        elif type(item) in (list, tuple):
            digest.update(b"list\0" if type(item) is list else b"tuple\0")
            for child in item:
                visit(child)
        else:
            digest.update(canonical(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def _parse_canonical_json(raw, label):
    value = base.parse_json(raw)
    require((canonical(value) + "\n").encode() == raw, "canonical " + label)
    return value


def _write_witness_artifacts(
    root,
    declaration_value,
    trace,
    first_record,
    rng_before,
    rng_after,
    source_binding_value,
    service_properties,
    original,
):
    projected = crb._project(trace)
    manifest, inertia_raw, stage_raw = archive.encode(
        trace, declaration_value, first_record
    )
    reconstructed, score = archive.decode(
        manifest,
        inertia_raw,
        stage_raw,
        projected,
        declaration_value,
        first_record,
    )
    before_digest, after_digest = _tree_digest(trace), _tree_digest(reconstructed)
    require(
        _tree_equal(trace, reconstructed) and before_digest == after_digest,
        "exact complete raw-tree archive round trip",
    )
    normalized_record = prior._producer_tree(first_record)
    first_record_raw = evidence.encode(normalized_record)
    rng_raw = evidence.encode({"cpu_before": rng_before, "cpu_after": rng_after})
    payloads = {
        "declaration.json": (canonical(declaration_value) + "\n").encode(),
        "inertia.pt": inertia_raw,
        "stages.bin": stage_raw,
        "manifest.json": (canonical(manifest) + "\n").encode(),
        "first-record.pt": first_record_raw,
        "rng.pt": rng_raw,
    }
    for name, raw in payloads.items():
        limit = _file_limit(name)
        require(
            type(raw) is bytes and 0 < len(raw) <= limit, "bounded retained " + name
        )
        base._write_exclusive(root / name, raw, limit)
    inventory = _inventory(root, ARCHIVE_FILES)
    witness = {
        "protocol": PROTOCOL + ":witness-v1",
        "source": trace["source"],
        "source_binding": source_binding_value,
        "service_properties": service_properties,
        "original_files": original["original_files"],
        "original_anchors": original["original_anchors"],
        "original_pair_accepted": False,
        "original_pair_error": "paired rollout semantic state exactness",
        "original_launch_sha256": original["original_launch_sha256"],
        "files": inventory,
        "trace_score": score,
        "trace_tree_sha256_before": before_digest,
        "trace_tree_sha256_after": after_digest,
        "caller_rng_preserved": torch.equal(rng_before, rng_after),
        "cpu_only": True,
        "pre_pulse_only": True,
        "shared_lease_verified": True,
        "full_native_window_qualified": False,
        "optimizer_steps": 0,
        **FLAGS,
    }
    require(
        witness["caller_rng_preserved"] is True
        and all(witness[name] is False for name in FLAGS),
        "truthful non-admitting CPU witness",
    )
    base.write_json(root / "witness.json", witness)
    return witness


def _cpu_service_env():
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return env


def _child(source, lease_fd, owner_pid):
    base.training_smoke.inherited_lease(lease_fd)
    _hidden()
    require(
        type(owner_pid) is int and owner_pid > 0 and os.getppid() == owner_pid,
        "owned CPU witness child parent PID",
    )
    service = _owned_properties(source, "run", owner_pid)
    start_context = _authenticate_original_without_loading(source)
    start_binding = source_binding(source)
    rng_before = torch.random.get_rng_state().clone()
    declaration_value = declaration(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        runtime = ScheduledRecoveryRuntime(
            declaration_value, device="cpu", solved_field_check="packed"
        )
    require(
        torch.equal(rng_before, torch.random.get_rng_state()),
        "CPU runtime constructor preserves caller RNG",
    )
    observer = crb.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    started = time.monotonic()
    try:
        with observer:
            result = runtime.step_with_schedule(
                torch.zeros(64, 10), capture_control=True
            )
        trace = observer.capture()
    except BaseException:
        # Retain any safely available partial observer state in the parent
        # process log only. Partial/faulted traces are never encoded as success.
        raise
    rng_after = torch.random.get_rng_state().clone()
    require(
        torch.equal(rng_before, rng_after)
        and time.monotonic() - started < CHILD_SECONDS - 30,
        "one-tick CPU witness preserves caller RNG within child cap",
    )
    first_record = {"runtime_result_before_reset": result}
    score = crb.check(trace, declaration_value, first_record)
    require(
        trace["status"] == "complete"
        and trace["worlds"] == 64
        and len(trace["events"]) == 6
        and score["worlds"] == 64
        and all(score[key] is False for key in crb.FLAGS),
        "complete 64-world six-event CPU CRB trace only",
    )
    root = output_path(source, "run")
    witness = _write_witness_artifacts(
        root,
        declaration_value,
        trace,
        first_record,
        rng_before,
        rng_after,
        start_binding,
        service,
        start_context,
    )
    end_context = _authenticate_original_without_loading(source)
    require(
        end_context == start_context
        and source_binding(source) == start_binding
        and _owned_properties(source, "run", owner_pid) == service,
        "unchanged original failure evidence, source and active owner service",
    )
    print(
        canonical(
            {
                "protocol": witness["protocol"],
                "source": source,
                "status": "passed",
                "cpu_only": True,
                "optimizer_steps": 0,
            }
        ),
        flush=True,
    )
    return witness


def _verify_witness(root, source, source_binding_value, expected_service):
    _hidden()
    base._exact_inventory(root, RUN_FILES - {"report.json"})
    raw_files = {
        name: base._read_file(root / name, _file_limit(name))
        for name in sorted(RUN_FILES - {"report.json"})
    }
    witness = _parse_canonical_json(raw_files["witness.json"], "CPU witness JSON")
    require(
        type(witness) is dict
        and set(witness) == _WITNESS_KEYS
        and witness["protocol"] == PROTOCOL + ":witness-v1"
        and witness["source"] == source
        and witness["source_binding"] == source_binding_value
        and witness["service_properties"] == expected_service
        and witness["files"]
        == {
            name: {
                "bytes": len(raw_files[name]),
                "sha256": base.digest(raw_files[name]),
            }
            for name in sorted(ARCHIVE_FILES)
        }
        and type(witness["trace_tree_sha256_before"]) is str
        and type(witness["trace_tree_sha256_after"]) is str
        and witness["trace_tree_sha256_before"] == witness["trace_tree_sha256_after"]
        and witness["caller_rng_preserved"] is True
        and witness["cpu_only"] is True
        and witness["pre_pulse_only"] is True
        and witness["shared_lease_verified"] is True
        and witness["full_native_window_qualified"] is False
        and witness["original_pair_accepted"] is False
        and witness["original_pair_error"] == "paired rollout semantic state exactness"
        and type(witness["optimizer_steps"]) is int
        and witness["optimizer_steps"] == 0
        and all(witness[key] is False for key in FLAGS),
        "exact non-admitting whole-file CPU witness before tensor loads",
    )
    original = _authenticate_original_without_loading(source)
    require(
        original["original_files"] == witness["original_files"]
        and original["original_anchors"] == witness["original_anchors"]
        and original["original_launch_sha256"] == witness["original_launch_sha256"],
        "original 23 artifacts, failure state and installed smooth source reauthenticated",
    )
    declaration_value = _parse_canonical_json(
        raw_files["declaration.json"], "CPU witness declaration"
    )
    require(declaration_value == declaration(source), "exact CPU witness declaration")
    manifest = _parse_canonical_json(raw_files["manifest.json"], "CRB archive manifest")
    # Validate the bounded archive framing, exact source, compiled topology,
    # and raw tensor payload length before loading any tensor-bearing artifact.
    archive._validate_manifest(manifest)
    require(
        manifest["source"] == source
        and manifest["worlds"] == declaration_value["worlds"]
        and manifest["inertia_bytes"] == len(raw_files["inertia.pt"])
        and manifest["stage_bytes"] == len(raw_files["stages.bin"])
        and archive._sha(raw_files["inertia.pt"]) == manifest["inertia_sha256"]
        and archive._sha(raw_files["stages.bin"]) == manifest["stage_sha256"],
        "CRB archive source and whole-payload bindings before tensor loads",
    )
    header, tensor_raw = archive._parse_stage_payload(raw_files["stages.bin"])
    _levels, expected_tensor_bytes = archive._validate_header(header, manifest)
    require(
        len(tensor_raw) == expected_tensor_bytes
        and crb._fresh_topology(manifest["worlds"]) == header["compiled_topology"],
        "CRB archive stage framing and fresh CPU topology before tensor loads",
    )
    # Whole inventory and archive schema are authenticated before these
    # weights-only CPU loads; the split archive decoder itself never loads.
    projected = torch.load(
        io.BytesIO(raw_files["inertia.pt"]), map_location="cpu", weights_only=True
    )
    first_record = torch.load(
        io.BytesIO(raw_files["first-record.pt"]),
        map_location="cpu",
        weights_only=True,
    )
    rng = torch.load(
        io.BytesIO(raw_files["rng.pt"]), map_location="cpu", weights_only=True
    )
    evidence._owned_tree(projected, clone=False)
    evidence._owned_tree(first_record, clone=False)
    evidence._owned_tree(rng, clone=False)
    require(
        type(rng) is dict
        and set(rng) == {"cpu_before", "cpu_after"}
        and all(
            torch.is_tensor(rng[key])
            and rng[key].device.type == "cpu"
            and rng[key].dtype == torch.uint8
            and rng[key].ndim == 1
            and rng[key].numel() > 0
            for key in ("cpu_before", "cpu_after")
        )
        and rng["cpu_before"].shape == rng["cpu_after"].shape
        and torch.equal(rng["cpu_before"], rng["cpu_after"]),
        "whole caller CPU RNG state restored",
    )
    trace, score = archive.decode(
        manifest,
        raw_files["inertia.pt"],
        raw_files["stages.bin"],
        projected,
        declaration_value,
        first_record,
    )
    require(
        _tree_digest(trace) == witness["trace_tree_sha256_after"]
        and score == witness["trace_score"]
        and all(score[key] is False for key in crb.FLAGS),
        "independent hash-first CPU archive closeout",
    )
    return dict(
        witness_sha256=base.digest(raw_files["witness.json"]),
        files={name: witness["files"][name] for name in sorted(ARCHIVE_FILES)},
        trace_score=score,
        trace_tree_sha256=_tree_digest(trace),
    )


def _read_tests(source, binding):
    root = output_path(source, "tests")
    base._exact_inventory(root, TEST_FILES_ON_DISK)
    raw = base._read_file(root / "receipt.json", base.JSON_LIMIT)
    receipt = _parse_canonical_json(raw, "owner-frozen CPU test receipt")
    require(
        type(receipt) is dict
        and set(receipt)
        == {
            "protocol",
            "source",
            "mode",
            "status",
            "source_binding",
            "test_files",
            "passed",
            "skips",
            "pytest_sha256",
            "service_properties",
            *FLAGS,
        }
        and receipt.get("protocol") == PROTOCOL + ":tests"
        and receipt.get("source") == source
        and receipt.get("mode") == "tests"
        and receipt.get("status") == "passed"
        and receipt.get("source_binding") == binding
        and receipt.get("test_files") == list(TEST_FILES)
        and type(receipt.get("passed")) is int
        and receipt["passed"] == EXPECTED_TESTS
        and EXPECTED_TESTS > 0
        and receipt.get("skips") == 0
        and all(receipt.get(name) is False for name in FLAGS),
        "exact owner-frozen zero-skip test receipt",
    )
    log = base._read_file(root / "pytest.log", base.LOG_LIMIT)
    require(base.digest(log) == receipt["pytest_sha256"], "whole CPU test log hash")
    totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
    require(
        len(totals) == 1
        and int(totals[0][0]) == EXPECTED_TESTS
        and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
        "exact owner-frozen CPU test log with zero skips",
    )
    _recorded_properties(receipt["service_properties"])
    completed(source, "tests", receipt["service_properties"]["InvocationID"])
    report = _parse_canonical_json(
        base._read_file(root / "report.json", base.JSON_LIMIT),
        "completed CPU test service report",
    )
    require(
        type(report) is dict
        and report.get("protocol") == PROTOCOL + ":tests"
        and report.get("source") == source
        and report.get("mode") == "tests"
        and report.get("status") == "passed"
        and report.get("source_binding") == binding
        and report.get("service_properties") == receipt["service_properties"]
        and report.get("passed") == EXPECTED_TESTS
        and report.get("pytest_sha256") == receipt["pytest_sha256"]
        and type(report.get("elapsed_seconds")) is float
        and math.isfinite(report["elapsed_seconds"])
        and 0 <= report["elapsed_seconds"] < CAP
        and all(report.get(key) is False for key in FLAGS),
        "exact successful CPU tests report",
    )
    return base.digest(raw)


def completed(source, mode, invocation):
    base._hex(invocation, 32, "completed CPU probe invocation")
    value = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
        )
        for key in (
            "MainPID",
            "ActiveState",
            "SubState",
            "RemainAfterExit",
            "NRestarts",
            "ExecMainStatus",
            "Result",
            "InvocationID",
        )
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
        },
        "successful terminal CPU probe service",
    )
    return value


def execute(source, mode):
    _hidden()
    require(mode in ("tests", "run"), "declared CPU archive mode")
    prior.check_window(
        reserve_seconds=(2 * CAP + MARGIN if mode == "tests" else RUN_RESERVE)
    )
    service = _owned_properties(source, mode, os.getpid())
    binding = source_binding(source)
    host_identity = base.host.identity(source)
    original = _authenticate_original_without_loading(source)
    root = output_path(source, mode)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(
        protocol=PROTOCOL + ":" + mode,
        source=source,
        mode=mode,
        status="failed-retained",
        service_properties=service,
        source_binding=binding,
        host_identity=host_identity,
        original_inputs=original,
        **FLAGS,
    )
    try:
        with base.gap.base.files.gpu_lease() as lease_fd:
            idle_before = base.gpu_idle_gate.wait_idle()
            filmbrain = base.gap.base.retained.d0.filmbrain_state()
            protected = base.gap.base.protected_state()
            require(
                bool(protected)
                and all(value == "inactive" for value in protected.values()),
                "protected AI Mission services inactive",
            )
            if mode == "tests":
                require(
                    EXPECTED_TESTS > 0, "complete test count owner-frozen before launch"
                )
                env = _cpu_service_env()
                prior._run_process(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *["tests/" + name for name in TEST_FILES],
                    ],
                    root / "pytest.log",
                    CAP - 30,
                    env=env,
                    fd=lease_fd,
                )
                log = base._read_file(root / "pytest.log", base.LOG_LIMIT)
                totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
                require(
                    len(totals) == 1
                    and int(totals[0][0]) == EXPECTED_TESTS
                    and not re.search(
                        rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log
                    ),
                    "exact frozen CPU suite, no skips",
                )
                receipt = {
                    "protocol": PROTOCOL + ":tests",
                    "source": source,
                    "mode": "tests",
                    "status": "passed",
                    "source_binding": binding,
                    "test_files": list(TEST_FILES),
                    "passed": EXPECTED_TESTS,
                    "skips": 0,
                    "pytest_sha256": base.digest(log),
                    "service_properties": service,
                    **FLAGS,
                }
                base.write_json(root / "receipt.json", receipt)
                report.update(passed=EXPECTED_TESTS, pytest_sha256=base.digest(log))
            else:
                tests_receipt_sha = _read_tests(source, binding)
                require(
                    _authenticate_original_without_loading(source) == original,
                    "original 23 artifacts, failure state and source closure before child",
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
                ]
                prior._run_process(
                    command,
                    root / "child.log",
                    CHILD_SECONDS,
                    env=_cpu_service_env(),
                    fd=lease_fd,
                )
                witness_result = _verify_witness(root, source, binding, service)
                after_original = _authenticate_original_without_loading(source)
                require(
                    after_original == original
                    and source_binding(source) == binding
                    and base.host.identity(source) == host_identity,
                    "original failed evidence and source closure unchanged after witness",
                )
                report.update(
                    tests_receipt_sha256=tests_receipt_sha,
                    witness=witness_result,
                    cpu_only=True,
                    pre_pulse_only=True,
                    full_native_window_qualified=False,
                    optimizer_steps=0,
                )
            idle_after = base.gpu_idle_gate.wait_idle()
            require(
                source_binding(source) == binding
                and _authenticate_original_without_loading(source) == original
                and base.host.identity(source) == host_identity
                and _owned_properties(source, mode, os.getpid()) == service
                and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                and base.gap.base.protected_state() == protected
                and time.monotonic() - started < CAP,
                "unchanged capped source/host/service/protected context",
            )
            report.update(
                status="passed",
                idle_before=idle_before,
                idle_after=idle_after,
                filmbrain=filmbrain,
                protected_services=protected,
            )
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        try:
            report["partial_artifacts"] = _partial_inventory(root, mode)
        except BaseException as inventory_error:
            report["partial_artifacts_error_type"] = type(inventory_error).__name__
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tests", "run", "child"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    args = parser.parse_args()
    if args.mode == "child":
        require(
            args.lease_fd is not None and args.owner_pid is not None,
            "owned child arguments",
        )
        result = _child(args.source, args.lease_fd, args.owner_pid)
    else:
        require(
            args.lease_fd is None and args.owner_pid is None,
            "supervisor mode arguments",
        )
        result = execute(args.source, args.mode)
    print(
        canonical(
            {
                "protocol": result["protocol"],
                "source": result["source"],
                "status": result.get("status", "passed"),
                "cpu_only": True,
            }
        )
    )


if __name__ == "__main__":
    main()
