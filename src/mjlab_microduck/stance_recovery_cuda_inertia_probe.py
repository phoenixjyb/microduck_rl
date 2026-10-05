"""Fresh October 5 inertia-observed no-update CUDA64 capture/replay, never a trainer.

Closed preparation authenticates ancestry, not this physical rollout. Separate
CPU closeout rehashes every artifact and validates the new evidence contract.
All failed namespaces remain retained; no old service or window is modified.
"""

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
import io
import math
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

import torch

from mjlab_microduck import stance_recovery_cuda_policy_probe as base
from mjlab_microduck import stance_recovery_cuda_preparation_evidence as closed
from mjlab_microduck import stance_recovery_cuda_shadow_probe as shadow
from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_record_replay as physical
from mjlab_microduck import stance_recovery_cuda_storage_evidence as storage
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck import stance_recovery_cuda_constructor_rng as constructor
from mjlab_microduck import stance_recovery_early_inertia_trace as early_trace
from mjlab_microduck import stance_recovery_cuda_rollout_probe as frozen
from mjlab_microduck import stance_constraint_identity_probe as identity
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_transition import PhysicsState
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cuda64-inertia-probe-20261005-v1"
MODULE = "mjlab_microduck.stance_recovery_cuda_inertia_probe"
START, CUTOFF = 1791187928, 1791205200  # October 5 16:12:08 / 21:00 Shanghai.
SEED = 653
ATTEMPTS = ("capture", "replay")
SECONDS = dict(preflight=300, tests=300, run=1200, closeout=600, diagnose=300)
MEMORY = {mode: (6 if mode == "run" else 4) * 1024**3 for mode in SECONDS}
CHILD_SECONDS = 480
MARGIN = 60
RUN_RESERVE = SECONDS["run"] + SECONDS["closeout"] + MARGIN
BASE_SOURCE = "c4fb6062a30ccef0365b21ef9c91aecacb3a8ba9"
TEST_FILES = identity.TEST_FILES + (
    "test_stance_recovery_early_inertia_trace.py",
    "test_stance_recovery_cuda_inertia_probe.py",
)
EXPECTED_TESTS = 1072  # Owner-reviewed complete 37-file suite, no skips.
ALLOWED = (
    "src/mjlab_microduck/stance_recovery_cuda_inertia_probe.py",
    "tests/test_stance_recovery_cuda_inertia_probe.py",
    "docs/experiments/2026-10-05-cuda64-no-update-rollout.md",
)
FROZEN_FILES = tuple(
    sorted(
        set(frozen.OWN_FILES[:-1])
        | set(identity.OWN[:-1])
        | {
            "src/mjlab_microduck/stance_recovery_early_inertia_trace.py",
            "tests/test_stance_recovery_early_inertia_trace.py",
        }
    )
)
OWN_FILES = (*FROZEN_FILES, *ALLOWED)

CHILD_FILES = {
    f"{a}.{suffix}"
    for a in ATTEMPTS
    for suffix in (
        "prepared.pt",
        "prepared.json",
        "constructor.pt",
        "constructor.json",
        "early-trace.pt",
        "early-trace.json",
        "pt",
        "json",
        "log",
    )
}
RUN_FILES = CHILD_FILES | {"launch.json", "checkpoint.pt", "cpu-parent-receipt.json"}
COMPLETE_FILES = RUN_FILES | {"report.json"}
FLAGS = {
    **early_trace.FLAGS,
    "execution_admitted": False,
    "training_update_performed": False,
    "native_transition_qualified": False,
}


def check_window(*, reserve_seconds=0, now=None):
    current = time.time() if now is None else now
    require(
        type(current) in (int, float)
        and math.isfinite(current)
        and type(reserve_seconds) in (int, float)
        and math.isfinite(reserve_seconds)
        and reserve_seconds >= 0
        and START <= current
        and current + reserve_seconds < CUTOFF,
        "fresh October 5 work and complete closeout before 21:00 Shanghai",
    )


def declaration(source):
    cell = next(
        c["id"]
        for c in schedule.lesson.cells("dose", held_out=False)
        if c["onset_step"] == 250
        and c["duration_steps"] == 20
        and c["force_world_newtons"] == [2.0, 0.0, 0.0]
    )
    return schedule.declaration(
        source, "dose", "training", ["zero-wrench"] * 32 + [cell] * 32
    )


def output_path(source):
    base._hex(source, 40, "new rollout source")
    return (
        base.execution.ROOT
        / "artifacts/evaluations"
        / ("stance-cuda64-inertia-" + source[:12])
    )


def tool_path(source, mode):
    output_path(source)
    require(mode in ("preflight", "tests"), "CPU rollout tool mode")
    return (
        base.execution.ROOT
        / "artifacts/tools"
        / (f"cuda64-inertia-{mode}-" + source[:12])
    )


def unit(source, mode):
    output_path(source)
    require(mode in SECONDS, "declared rollout service mode")
    return f"microduck-cuda64-inertia-{mode}-{source[:12]}.service"


def _recorded_properties(value, mode):
    require(
        type(value) is dict
        and set(value)
        == {
            "MainPID",
            "ActiveState",
            "RuntimeMaxUSec",
            "MemoryMax",
            "CPUQuotaPerSecUSec",
            "Nice",
            "KillMode",
            "InvocationID",
        },
        "exact new service properties",
    )
    base._hex(value["InvocationID"], 32, "new service invocation")
    require(
        type(value["MainPID"]) is str
        and value["MainPID"].isdecimal()
        and int(value["MainPID"]) > 0
        and {k: v for k, v in value.items() if k not in ("MainPID", "InvocationID")}
        == {
            "ActiveState": "active",
            "RuntimeMaxUSec": f"{SECONDS[mode] // 60}min",
            "MemoryMax": str(MEMORY[mode]),
            "CPUQuotaPerSecUSec": "2s",
            "Nice": "10",
            "KillMode": "control-group",
        },
        "exact new service caps",
    )


def _properties(source, mode):
    keys = (
        "MainPID",
        "ActiveState",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "Nice",
        "KillMode",
        "InvocationID",
    )
    value = {
        k: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", k, "--value"
        )
        for k in keys
    }
    _recorded_properties(value, mode)
    require(value["MainPID"] == str(os.getpid()), "new capped service owns process")
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
        "only this Duck user service runs",
    )
    return value


def completed(source, mode, invocation):
    base._hex(invocation, 32, "original new service invocation")
    keys = (
        "MainPID",
        "ActiveState",
        "NRestarts",
        "ExecMainStatus",
        "Result",
        "InvocationID",
    )
    value = {
        k: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", k, "--value"
        )
        for k in keys
    }
    require(
        value["InvocationID"] in ("", invocation)
        and {k: v for k, v in value.items() if k != "InvocationID"}
        == {
            "MainPID": "0",
            "ActiveState": "inactive",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "Result": "success",
        },
        "successful original new service",
    )
    return value


def source_binding(source):
    previous = shadow.source_binding(source)  # Pure source check, no old window.
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=15
        ).stdout

    git("merge-base", "--is-ancestor", BASE_SOURCE, source)
    changed = set(git("diff", "--name-only", BASE_SOURCE, source).decode().splitlines())
    require(changed <= set(ALLOWED), "only newly declared inertia supervisor changes")
    leaves = {}
    for relative in OWN_FILES:
        raw = git("show", f"{source}:{relative}")
        require(
            raw == (root / relative).read_bytes(), "exact new source leaf " + relative
        )
        if relative in FROZEN_FILES:
            require(
                raw == git("show", f"{BASE_SOURCE}:{relative}"),
                "unchanged frozen inertia observer and prior readers " + relative,
            )
        leaves[relative] = base.digest(raw)
    return dict(previous_frozen_source=previous, inertia_leaves=leaves)


def trace_descriptor():
    return dict(
        protocol=early_trace.PROTOCOL,
        steps=early_trace.early.STEPS,
        observation_boundary=early_trace.BOUNDARY,
        excluded_intermediates=list(early_trace.EXCLUDED),
        persistent_shapes={k: list(v) for k, v in early_trace.SHAPES.items()},
    )


def _hidden():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized()
        and base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "CUDA-hidden exact WSL new supervisor",
    )


def _limit(name):
    if name in ("capture.pt", "replay.pt"):
        return archive.LIMIT
    return (
        base.RAW_LIMIT
        if name.endswith(".pt")
        else base.LOG_LIMIT
        if name.endswith(".log")
        else base.JSON_LIMIT
    )


def _inventory(root, names=None):
    partial = names is None
    if partial:
        names = {p.name for p in root.iterdir()} - {"report.json"}
    else:
        require(
            set(names) <= {p.name for p in root.iterdir()},
            "every named rollout artifact retained",
        )
    result = {}
    for name in sorted(names):
        path = root / name
        try:
            raw = base._read_file(path, _limit(name))
            result[name] = dict(sha256=base.digest(raw), bytes=len(raw))
        except (OSError, ValueError):
            if not partial:
                raise
            result[name] = dict(readable=False)
    return result


def _run_process(command, log_path, seconds, *, env, fd=None, monitor=False):
    samples = []
    total = 0
    with Path(log_path).open("xb") as log:
        proc = subprocess.Popen(
            command,
            cwd=base.execution.ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=() if fd is None else (fd,),
        )
        start = time.monotonic()
        next_monitor = start
        pipe_open = True
        try:
            while proc.poll() is None or pipe_open:
                left = seconds - (time.monotonic() - start)
                if left <= 0:
                    raise TimeoutError("new owned process exceeded fixed cap")
                readable, _, _ = select.select(
                    [proc.stdout] if pipe_open else [], [], [], min(0.25, left)
                )
                if readable:
                    chunk = os.read(
                        proc.stdout.fileno(), min(65536, base.LOG_LIMIT - total + 1)
                    )
                    if not chunk:
                        pipe_open = False
                    else:
                        allowed = max(0, base.LOG_LIMIT - total)
                        log.write(chunk[:allowed])
                        total += min(len(chunk), allowed)
                        log.flush()
                        require(len(chunk) <= allowed, "new streaming process log cap")
                if monitor and proc.poll() is None and time.monotonic() >= next_monitor:
                    samples.append(
                        dict(
                            elapsed_seconds=float(time.monotonic() - start),
                            child_pid=proc.pid,
                            sample=base.campaign.live_gpu(proc.pid),
                        )
                    )
                    next_monitor = time.monotonic() + 2
            require(
                proc.returncode == 0 and (not monitor or samples),
                "successful new owned process",
            )
        except BaseException:
            base.campaign._kill_owned_group(proc)
            raise
        finally:
            proc.stdout.close()
            log.flush()
            os.fsync(log.fileno())
    return samples


def _tool_receipt(source, mode, operation):
    _hidden()
    check_window(reserve_seconds=sum(SECONDS.values()) + MARGIN)
    service = _properties(source, mode)
    root = tool_path(source, mode)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        protocol=PROTOCOL + ":" + mode, source=source, status="failed-retained", **FLAGS
    )
    try:
        with base.gap.base.files.gpu_lease() as fd:
            idle = base.gpu_idle_gate.wait_idle()
            inputs = source_binding(source)
            prior = closed.checked(source, lease_fd=fd)
            result = operation(root, prior, inputs)
            require(
                source_binding(source) == inputs
                and _properties(source, mode) == service
                and closed.checked(source, lease_fd=fd)["launch_binding"]
                == prior["launch_binding"]
                and time.monotonic() - started < SECONDS[mode],
                "unchanged new CPU tool source/provenance/context",
            )
            receipt.update(
                status="passed",
                service_properties=service,
                source_binding=inputs,
                closed_preparation=prior["launch_binding"],
                idle_before=idle,
                idle_after=base.gpu_idle_gate.wait_idle(),
                **result,
            )
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "receipt.json", receipt)
    return receipt


def preflight(source):
    def prepare(root, prior, inputs):
        before = torch.random.get_rng_state().clone()
        loaded, raw, sha, canonical_sha = base._cpu_parent_prepare(prior["raw_parent"])
        binding = base._validated_cpu_binding(source, loaded["receipt"], canonical_sha)
        require(
            torch.equal(before, torch.random.get_rng_state()),
            "CPU constructor caller RNG preserved",
        )
        base._write_exclusive(root / "cpu-parent-receipt.json", raw, base.JSON_LIMIT)
        return dict(
            cpu_parent_receipt_file_sha256=sha,
            cpu_parent_receipt_canonical_sha256=canonical_sha,
            cpu_parent_binding=binding,
            compiled_plant=physical._compiled_reference(),
        )

    return _tool_receipt(source, "preflight", prepare)


def _read_tool(source, mode, prior):
    root = tool_path(source, mode)
    names = {
        "receipt.json",
        "cpu-parent-receipt.json" if mode == "preflight" else "pytest.log",
    }
    base._exact_inventory(root, names)
    raw = base._read_file(root / "receipt.json", base.JSON_LIMIT)
    rec = base.parse_json(raw)
    require(
        rec.get("protocol") == PROTOCOL + ":" + mode
        and rec.get("source") == source
        and rec.get("status") == "passed"
        and rec.get("source_binding") == source_binding(source)
        and rec.get("closed_preparation") == prior["launch_binding"]
        and all(rec.get(k) is False for k in FLAGS),
        "new closed CPU tool binding",
    )
    _recorded_properties(rec["service_properties"], mode)
    terminal = completed(source, mode, rec["service_properties"]["InvocationID"])
    if mode == "preflight":
        extra = base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT)
        cpu = base.parse_json(extra)
        canonical_sha = base.preparation._cpu_parent_receipt_digest(cpu)
        require(
            base.digest(extra) == rec["cpu_parent_receipt_file_sha256"]
            and canonical_sha == rec["cpu_parent_receipt_canonical_sha256"]
            and base._validated_cpu_binding(source, cpu, canonical_sha)
            == rec["cpu_parent_binding"]
            and rec["compiled_plant"] == physical._compiled_reference(),
            "exact actual CPU preflight inputs",
        )
    else:
        extra = base._read_file(root / "pytest.log", base.LOG_LIMIT)
        require(
            EXPECTED_TESTS > 0
            and rec["test_files"] == list(TEST_FILES)
            and rec["passed"] == EXPECTED_TESTS
            and rec["skips"] == 0
            and rec["log_sha256"] == base.digest(extra),
            "complete declared new native source suite",
        )
    return rec, dict(receipt_sha256=base.digest(raw), terminal=terminal), extra


def tests(source):
    def run_tests(root, prior, inputs):
        _, pre_binding, _ = _read_tool(source, "preflight", prior)
        require(EXPECTED_TESTS > 0, "owner-frozen complete source test count")
        env = {**shadow.child_environment(), "CUDA_VISIBLE_DEVICES": ""}
        _run_process(
            [sys.executable, "-m", "pytest", "-q", *("tests/" + p for p in TEST_FILES)],
            root / "pytest.log",
            SECONDS["tests"] - 60,
            env=env,
        )
        log = base._read_file(root / "pytest.log", base.LOG_LIMIT)
        summary = re.findall(rb"(\d+) passed(?: in [^\n]+)?", log)
        require(
            len(summary) == 1
            and int(summary[0]) == EXPECTED_TESTS
            and not re.search(rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log),
            "exact complete new pytest result, no skips",
        )
        return dict(
            preflight_binding=pre_binding,
            test_files=list(TEST_FILES),
            passed=EXPECTED_TESTS,
            skips=0,
            log_sha256=base.digest(log),
        )

    return _tool_receipt(source, "tests", run_tests)


def _read_launch(source, sha):
    raw = base._read_file(output_path(source) / "launch.json", base.JSON_LIMIT)
    require(base.digest(raw) == sha, "whole new launch bytes before parse")
    launch = base.parse_json(raw)
    require(
        type(launch) is dict
        and set(launch)
        == {
            "protocol",
            "source",
            "source_binding",
            "native_prerequisites",
            "closed_preparation",
            "preflight_binding",
            "tests_binding",
            "service_properties",
            "idle_before",
            "schedule",
            "compiled_plant",
            "cpu_parent_receipt_file_sha256",
            "cpu_parent_receipt_canonical_sha256",
            "cpu_parent_binding",
            "attempts",
            "service_seconds",
            "service_memory_bytes",
            "child_seconds",
            "cutoff_unix",
            "optimizer_steps",
            "return_computation",
            "constructor_rng",
            "early_forward_trace",
            *FLAGS,
        }
        and launch.get("protocol") == PROTOCOL
        and launch.get("source") == source
        and launch.get("schedule") == declaration(source)
        and launch.get("service_seconds") == SECONDS
        and launch.get("service_memory_bytes") == MEMORY
        and launch.get("child_seconds") == CHILD_SECONDS
        and launch.get("cutoff_unix") == CUTOFF
        and launch.get("attempts") == list(ATTEMPTS)
        and type(launch.get("optimizer_steps")) is int
        and launch["optimizer_steps"] == 0
        and launch.get("return_computation") is False
        and launch.get("constructor_rng")
        == dict(
            protocol=constructor.PROTOCOL,
            cpu_seed=constructor.CPU_SEED,
            cuda_seed=constructor.CUDA_SEED,
        )
        and launch.get("early_forward_trace") == trace_descriptor()
        and all(launch.get(k) is False for k in FLAGS),
        "exact new non-admitting launch",
    )
    _recorded_properties(launch["service_properties"], "run")
    return launch


def _prepared_summary(source, sha, raw, metadata, launch):
    # This immutable summary describes preparation BEFORE any runtime/call.
    return dict(
        protocol=base.PROTOCOL + ":child-summary-v1",
        source=source,
        launch_sha256=sha,
        seed=SEED,
        payload_sha256=base.digest(raw),
        payload_bytes=len(raw),
        metadata=metadata,
        private_cuda_rng_state_sha256=metadata["private_cuda_rng_state_sha256"],
        cpu_parent_receipt_file_sha256=launch["cpu_parent_receipt_file_sha256"],
        cpu_parent_receipt_canonical_sha256=launch[
            "cpu_parent_receipt_canonical_sha256"
        ],
        optimizer_steps=0,
        simulator_resets=0,
        training_update_performed=False,
        cuda_child_claims_cpu_provenance_authenticated=False,
        action_sampling=False,
        return_computation=False,
        rollout_collection=False,
        **base.PREPARATION_FALSE_FLAGS,
    )


def _terminal_observed(record):
    for key in ("terminated", "timed_out"):
        tensor = record[key]
        require(
            torch.is_tensor(tensor)
            and tensor.device.type == "cpu"
            and tensor.shape == (64,)
            and tensor.dtype == torch.bool,
            "retained typed first-terminal mask",
        )
    return bool((record["terminated"] | record["timed_out"]).any())


def _producer_tree(value):
    """Normalize only the actual runtime PhysicsState, before bounded ownership.

    No arbitrary dataclass or pickle object is admitted. The existing archive
    copier remains responsible for typed tensors, device transfer and byte caps.
    """
    nodes = 0

    def convert(item, depth=0):
        nonlocal nodes
        nodes += 1
        require(depth <= 32 and nodes <= archive.NODE_BUDGET, "bounded producer tree")
        if type(item) is PhysicsState:
            names = {f.name for f in fields(PhysicsState)}
            require(
                names
                == {
                    "tilt",
                    "root_velocity",
                    "height",
                    "support",
                    "torque",
                    "joint_velocity",
                    "hard_limit",
                    "forbidden_contact",
                    "warning",
                }
                and all(torch.is_tensor(getattr(item, key)) for key in names),
                "exact producer PhysicsState tensor fields",
            )
            return {
                key: convert(getattr(item, key), depth + 1) for key in sorted(names)
            }
        if isinstance(item, Mapping):
            require(all(type(key) is str for key in item), "string producer keys")
            return {key: convert(child, depth + 1) for key, child in item.items()}
        if type(item) in (list, tuple):
            result = [convert(child, depth + 1) for child in item]
            return tuple(result) if type(item) is tuple else result
        return item  # Strict ownership rejects every unsupported leaf next.

    return convert(value)


def child(source, sha, attempt, lease_fd):
    base.training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and attempt in ATTEMPTS,
        "new isolated CUDA0 capture or replay",
    )
    check_window(reserve_seconds=CHILD_SECONDS + SECONDS["closeout"] + MARGIN)
    launch = _read_launch(source, sha)
    require(
        source_binding(source) == launch["source_binding"], "exact new child source"
    )
    context = base._visible_child_context(source, launch)
    root = output_path(source)
    parent = base._read_file(root / "checkpoint.pt", base.checkpoint.LIMIT)
    require(
        base.digest(parent) == base.baseline.CHECKPOINT_SHA256,
        "whole immutable D1 parent",
    )
    cpu_raw = base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT)
    require(
        base.digest(cpu_raw) == launch["cpu_parent_receipt_file_sha256"],
        "whole CPU constructor receipt",
    )
    prepared = base.preparation.prepare_policy(
        parent,
        source=source,
        lease_fd=lease_fd,
        cpu_parent_receipt=base.parse_json(cpu_raw),
        cpu_parent_receipt_sha256=launch["cpu_parent_receipt_canonical_sha256"],
        caller_binding=launch["cpu_parent_binding"],
        seed=SEED,
        worlds=64,
    )
    prep_raw, meta = base._child_payload(prepared, sha)
    base._write_exclusive(root / (attempt + ".prepared.pt"), prep_raw, base.RAW_LIMIT)
    base.write_json(
        root / (attempt + ".prepared.json"),
        _prepared_summary(source, sha, prep_raw, meta, launch),
    )
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(0).cpu().clone()
    require(
        torch.equal(before_cpu, prepared["caller_rng_states"]["cpu_after"])
        and torch.equal(before_cuda, prepared["caller_rng_states"]["cuda_after"]),
        "actual caller streams match preparation closeout",
    )
    ctor = constructor.CudaConstructorRng(source, lease_fd=lease_fd)
    try:
        env = ctor.construct(launch["schedule"])
    finally:
        _retain_constructor(root, source, sha, attempt, ctor.receipt)
    require(
        torch.equal(before_cpu, torch.random.get_rng_state())
        and torch.equal(before_cuda, torch.cuda.get_rng_state(0).cpu()),
        "runtime construction preserves caller CPU and CUDA RNG",
    )
    require(
        torch.equal(
            prepared["private_cuda_generator"].get_state().cpu(),
            prepared["private_cuda_rng_state"],
        ),
        "constructor does not advance the learner private stream",
    )
    require(
        env.binding == launch["compiled_plant"], "actual CUDA runtime selected plant"
    )
    initial = archive._owned_tree(_producer_tree(env.snapshot()), allow_cuda=True)
    initial_controls = archive._owned_tree(env._control_snapshot(), allow_cuda=True)
    collector = transition.CudaTransitionCollector(prepared, env, lease_fd=lease_fd)
    started = time.monotonic()
    records = []
    observer = early_trace.EarlyInertiaTrace(env)
    try:
        with observer:
            # Stop at first terminal; no collection beyond automatic reset.
            for _ in range(28):
                require(
                    time.monotonic() - started < CHILD_SECONDS - 60,
                    "collection serialization reserve",
                )
                record = collector.collect_one(capture_control=True)
                retained = archive.retain_record_cpu(
                    _producer_tree(record), lease_fd=lease_fd
                )
                records.append(retained)
                if _terminal_observed(retained):
                    break
    finally:
        _retain_trace(root, source, sha, attempt, observer.capture())
    archived = dict(
        protocol=archive.PROTOCOL,
        declaration=deepcopy(launch["schedule"]),
        compiled_plant=deepcopy(env.binding),
        learner_seed=SEED,
        records=records,
        **transition.FALSE_FLAGS,
    )
    value = dict(
        protocol=evidence.PROTOCOL,
        source=source,
        launch_sha256=sha,
        seed=SEED,
        attempt=attempt,
        initial_frame=initial,
        constructor_receipt=ctor.receipt,
        archive=archived,
        initial_control_state=initial_controls,
        storage=storage.capture(prepared["storage"], lease_fd=lease_fd),
        model_state_after=base.checkpoint.states_of(
            prepared["actor"], prepared["critic"]
        ),
        optimizer_state_after=deepcopy(prepared["algorithm"].optimizer.state_dict()),
        caller_rng_states=dict(
            cpu_before=before_cpu,
            cpu_after=torch.random.get_rng_state().clone(),
            cuda_before=before_cuda,
            cuda_after=torch.cuda.get_rng_state(0).cpu().clone(),
        ),
        private_cuda_state_final=collector.scope.state,
        elapsed_seconds=float(time.monotonic() - started),
        **transition.FALSE_FLAGS,
    )
    require(
        base._visible_child_context(source, launch) == context
        and source_binding(source) == launch["source_binding"],
        "unchanged child external context",
    )
    raw = evidence.encode(value)
    digest = base._write_exclusive(root / (attempt + ".pt"), raw, archive.LIMIT)
    summary = dict(
        protocol=PROTOCOL + ":child",
        source=source,
        launch_sha256=sha,
        seed=SEED,
        attempt=attempt,
        payload_sha256=digest,
        payload_bytes=len(raw),
        preparation_sha256=base.digest(prep_raw),
        preparation_bytes=len(prep_raw),
        records=len(records),
        optimizer_steps=0,
        **FLAGS,
    )
    base.write_json(root / (attempt + ".json"), summary)
    return summary


def _retain_constructor(root, source, sha, attempt, receipt):
    raw = evidence.encode(receipt)
    digest = base._write_exclusive(
        root / (attempt + ".constructor.pt"), raw, base.RAW_LIMIT
    )
    base.write_json(
        root / (attempt + ".constructor.json"),
        dict(
            protocol=PROTOCOL + ":constructor",
            source=source,
            launch_sha256=sha,
            attempt=attempt,
            payload_sha256=digest,
            payload_bytes=len(raw),
            status=receipt["status"],
            constructor_calls=receipt["constructor_calls"],
            faulted=receipt["faulted"],
            **FLAGS,
        ),
    )


def _retain_trace(root, source, sha, attempt, receipt):
    raw = evidence.encode(receipt)
    digest = base._write_exclusive(
        root / (attempt + ".early-trace.pt"), raw, base.RAW_LIMIT
    )
    base.write_json(
        root / (attempt + ".early-trace.json"),
        dict(
            protocol=PROTOCOL + ":early-trace",
            source=source,
            launch_sha256=sha,
            attempt=attempt,
            payload_sha256=digest,
            payload_bytes=len(raw),
            status=receipt["status"],
            events=len(receipt["events"]),
            **FLAGS,
        ),
    )


def _score_inputs(root, source, launch, sha):
    cpu = base.parse_json(
        base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT)
    )
    inputs, scores, traces = [], [], []
    for attempt in ATTEMPTS:
        summary = base.parse_json(
            base._read_file(root / (attempt + ".json"), base.JSON_LIMIT)
        )
        raw = base._read_file(root / (attempt + ".pt"), archive.LIMIT)
        prep_raw = base._read_file(root / (attempt + ".prepared.pt"), base.RAW_LIMIT)
        prepared_summary = base.parse_json(
            base._read_file(root / (attempt + ".prepared.json"), base.JSON_LIMIT)
        )
        ctor_summary = base.parse_json(
            base._read_file(root / (attempt + ".constructor.json"), base.JSON_LIMIT)
        )
        ctor_raw = base._read_file(root / (attempt + ".constructor.pt"), base.RAW_LIMIT)
        require(
            type(ctor_summary) is dict
            and set(ctor_summary)
            == {
                "protocol",
                "source",
                "launch_sha256",
                "attempt",
                "payload_sha256",
                "payload_bytes",
                "status",
                "constructor_calls",
                "faulted",
                *FLAGS,
            }
            and ctor_summary.get("protocol") == PROTOCOL + ":constructor"
            and ctor_summary.get("source") == source
            and ctor_summary.get("launch_sha256") == sha
            and ctor_summary.get("attempt") == attempt
            and ctor_summary.get("status") == "success"
            and ctor_summary.get("constructor_calls") == 1
            and type(ctor_summary.get("constructor_calls")) is int
            and ctor_summary.get("faulted") is False
            and ctor_summary.get("payload_bytes") == len(ctor_raw)
            and ctor_summary.get("payload_sha256") == base.digest(ctor_raw)
            and all(ctor_summary.get(k) is False for k in FLAGS),
            "whole successful constructor payload before CPU load",
        )
        require(
            summary.get("protocol") == PROTOCOL + ":child"
            and summary.get("source") == source
            and summary.get("launch_sha256") == sha
            and summary.get("seed") == SEED
            and summary.get("attempt") == attempt
            and summary.get("payload_bytes") == len(raw)
            and summary.get("preparation_bytes") == len(prep_raw)
            and summary.get("optimizer_steps") == 0
            and all(summary.get(k) is False for k in FLAGS),
            "new child summary binding",
        )
        value, score = evidence.verify(
            raw,
            summary["payload_sha256"],
            prep_raw,
            summary["preparation_sha256"],
            prepared_summary,
            launch,
            sha,
            cpu,
            attempt=attempt,
        )
        ctor_receipt = torch.load(
            io.BytesIO(ctor_raw), map_location="cpu", weights_only=True
        )
        evidence._owned_tree(ctor_receipt, clone=False)
        require(
            evidence._equal(ctor_receipt, value["constructor_receipt"]),
            "whole constructor file binds to scored body",
        )
        require(
            summary["records"] == len(value["archive"]["records"]),
            "whole actual record count",
        )
        trace_summary = base.parse_json(
            base._read_file(root / (attempt + ".early-trace.json"), base.JSON_LIMIT)
        )
        trace_raw = base._read_file(
            root / (attempt + ".early-trace.pt"), base.RAW_LIMIT
        )
        require(
            type(trace_summary) is dict
            and set(trace_summary)
            == {
                "protocol",
                "source",
                "launch_sha256",
                "attempt",
                "payload_sha256",
                "payload_bytes",
                "status",
                "events",
                *FLAGS,
            }
            and trace_summary["protocol"] == PROTOCOL + ":early-trace"
            and trace_summary["source"] == source
            and trace_summary["launch_sha256"] == sha
            and trace_summary["attempt"] == attempt
            and trace_summary["payload_sha256"] == base.digest(trace_raw)
            and trace_summary["payload_bytes"] == len(trace_raw)
            and trace_summary["status"] == "complete"
            and type(trace_summary["events"]) is int
            and trace_summary["events"] == 2 * early_trace.early.STEPS
            and all(trace_summary[k] is False for k in FLAGS),
            "whole complete early trace before CPU load",
        )
        traced = torch.load(
            io.BytesIO(trace_raw), map_location="cpu", weights_only=True
        )
        trace_score = early_trace.check(
            traced, launch["schedule"], value["archive"]["records"][0]
        )
        inputs.append(value)
        traces.append(traced)
        scores.append(
            {**score, "early_trace": trace_score, "diagnostics": _diagnostics(value)}
        )
    return inputs, scores, traces


def _strict_pair(inputs, traces):
    pair = evidence.paired(*inputs)
    require(
        early_trace.early.forward.throughput.tree_hash(traces[0])
        == early_trace.early.forward.throughput.tree_hash(traces[1]),
        "paired persistent early trace exactness",
    )
    return {**pair, "early_trace_exact": True}


def _score(root, source, launch, sha):
    inputs, scores, traces = _score_inputs(root, source, launch, sha)
    pair = _strict_pair(inputs, traces)
    return scores, pair


def _diagnostics(value):
    records = value["archive"]["records"]
    frames = [value["initial_frame"]] + [
        f for r in records for f in r["runtime_result_before_reset"]["boundaries"][1:]
    ]
    # These are retained-record reductions, not physics or motor re-execution.
    tilt = torch.cat([f["state"]["tilt"] for f in frames])
    velocity = torch.cat([f["state"]["root_velocity"] for f in frames])
    torque = torch.cat([f["state"]["torque"] for f in frames])
    joint_velocity = torch.cat([f["state"]["joint_velocity"] for f in frames])
    soft = torch.cat([f["soft_limit_mask"] for f in frames])
    last = records[-1]
    return dict(
        records=len(records),
        min_pre_reset_step=int(last["pre_reset_steps"].min()),
        max_pre_reset_step=int(last["pre_reset_steps"].max()),
        terminal_rows=int(last["terminated"].sum()),
        timeout_rows=int(last["timed_out"].sum()),
        maximum_tilt_rad=float(tilt.max()),
        maximum_planar_speed_mps=float(velocity[:, :2].norm(dim=1).max()),
        maximum_abs_motor_torque_nm=float(torque.abs().max()),
        maximum_abs_joint_power_w=float((torque * joint_velocity).abs().max()),
        soft_limit_row_exposure=float(soft.any(1).float().mean()),
        summed_environment_reward=float(
            sum(r["environment_reward"].sum() for r in records)
        ),
        motor_thermal_model_evaluated=False,
    )


def _decision(scores, pair):
    require(
        type(scores) is list
        and len(scores) == 2
        and [s.get("attempt") for s in scores] == list(ATTEMPTS)
        and pair.get("paired_semantics_exact") is True,
        "ordered independently checked pair",
    )
    passed = all(
        s.get("complete_28_call_gate") is True
        and s.get("complete_nonzero_pulse_gate") is True
        and s.get("whole_bytes_verified_before_load") is True
        and s.get("individually_cpu_validated") is True
        for s in scores
    )
    return dict(
        short_window_complete_and_consistent=passed,
        decision="complete-no-update-short-window"
        if passed
        else "retained-incomplete-short-window",
        native_rollout_qualified=False,
        training_admitted=False,
    )


def run(source):
    _hidden()
    check_window(reserve_seconds=RUN_RESERVE)
    service = _properties(source, "run")
    started = time.monotonic()
    root = output_path(source)
    root.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, source=source, status="failed-retained", **FLAGS)
    try:
        with base.gap.base.files.gpu_lease() as fd:
            idle = base.gpu_idle_gate.wait_idle()
            prior = closed.checked(source, lease_fd=fd)
            pre, pre_binding, cpu_raw = _read_tool(source, "preflight", prior)
            tests_rec, tests_binding, _ = _read_tool(source, "tests", prior)
            require(
                tests_rec["preflight_binding"] == pre_binding,
                "tests bind exact CPU preflight",
            )
            loaded, fresh_raw, _, _ = base._cpu_parent_prepare(prior["raw_parent"])
            require(
                fresh_raw == cpu_raw, "fresh CPU constructor matches retained preflight"
            )
            del loaded
            launch = dict(
                protocol=PROTOCOL,
                source=source,
                source_binding=source_binding(source),
                native_prerequisites=base._json_safe(
                    prior["current_prerequisites"]["launch_binding"]
                ),
                closed_preparation=prior["launch_binding"],
                preflight_binding=pre_binding,
                tests_binding=tests_binding,
                service_properties=service,
                idle_before=idle,
                schedule=declaration(source),
                compiled_plant=pre["compiled_plant"],
                cpu_parent_receipt_file_sha256=base.digest(cpu_raw),
                cpu_parent_receipt_canonical_sha256=pre[
                    "cpu_parent_receipt_canonical_sha256"
                ],
                cpu_parent_binding=pre["cpu_parent_binding"],
                attempts=list(ATTEMPTS),
                service_seconds=SECONDS,
                service_memory_bytes=MEMORY,
                child_seconds=CHILD_SECONDS,
                cutoff_unix=CUTOFF,
                optimizer_steps=0,
                return_computation=False,
                constructor_rng=dict(
                    protocol=constructor.PROTOCOL,
                    cpu_seed=constructor.CPU_SEED,
                    cuda_seed=constructor.CUDA_SEED,
                ),
                early_forward_trace=trace_descriptor(),
                **FLAGS,
            )
            base._write_exclusive(
                root / "checkpoint.pt", prior["raw_parent"], base.checkpoint.LIMIT
            )
            base._write_exclusive(
                root / "cpu-parent-receipt.json", cpu_raw, base.JSON_LIMIT
            )
            sha = base.write_json(root / "launch.json", launch)
            children = []
            for attempt in ATTEMPTS:
                check_window(reserve_seconds=RUN_RESERVE)
                base.gpu_idle_gate.wait_idle()
                samples = _run_process(
                    [
                        sys.executable,
                        "-m",
                        MODULE,
                        "child",
                        "--source",
                        source,
                        "--launch-sha256",
                        sha,
                        "--attempt",
                        attempt,
                        "--lease-fd",
                        str(fd),
                    ],
                    root / (attempt + ".log"),
                    CHILD_SECONDS,
                    env=shadow.child_environment(),
                    fd=fd,
                    monitor=True,
                )
                children.append(dict(attempt=attempt, samples=samples))
                base.gpu_idle_gate.wait_idle()
            scores, pair = _score(root, source, launch, sha)
            require(
                closed.checked(source, lease_fd=fd)["launch_binding"]
                == prior["launch_binding"]
                and source_binding(source) == launch["source_binding"]
                and _properties(source, "run") == service,
                "unchanged supervised run context",
            )
            base._exact_inventory(root, RUN_FILES)
            report.update(
                status="awaiting-independent-closeout",
                launch_sha256=sha,
                scores=scores,
                pair=pair,
                decision=_decision(scores, pair),
                children=children,
                inventory=_inventory(root, RUN_FILES),
                idle_after=base.gpu_idle_gate.wait_idle(),
            )
    except BaseException as error:
        report.update(
            error_type=type(error).__name__,
            error=str(error),
            partial_inventory=_inventory(root),
        )
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "report.json", report)
    return report


def closeout(source):
    _hidden()
    check_window(reserve_seconds=SECONDS["closeout"] + MARGIN)
    service = _properties(source, "closeout")
    started = time.monotonic()
    root = output_path(source)
    base._exact_inventory(root, COMPLETE_FILES)
    report_raw = base._read_file(root / "report.json", base.JSON_LIMIT)
    report = base.parse_json(report_raw)
    require(
        report.get("protocol") == PROTOCOL
        and report.get("source") == source
        and report.get("status") == "awaiting-independent-closeout"
        and all(report.get(k) is False for k in FLAGS),
        "successful retained run before closeout",
    )
    with base.gap.base.files.gpu_lease() as fd:
        idle = base.gpu_idle_gate.wait_idle()
        launch = _read_launch(source, report["launch_sha256"])
        _recorded_properties(launch["service_properties"], "run")
        terminal = completed(
            source, "run", launch["service_properties"]["InvocationID"]
        )
        prior = closed.checked(source, lease_fd=fd)
        require(
            prior["launch_binding"] == launch["closed_preparation"]
            and source_binding(source) == launch["source_binding"]
            and _inventory(root, RUN_FILES) == report["inventory"],
            "independently rehashed full inventory and fresh context",
        )
        _read_tool(source, "preflight", prior)
        _read_tool(source, "tests", prior)
        scores, pair = _score(root, source, launch, report["launch_sha256"])
        require(
            scores == report["scores"] and pair == report["pair"],
            "independent deterministic CPU score",
        )
        require(
            _decision(scores, pair) == report["decision"],
            "deterministic unchanged numerical short-window gate",
        )
        require(
            source_binding(source) == launch["source_binding"]
            and _properties(source, "closeout") == service
            and closed.checked(source, lease_fd=fd)["launch_binding"]
            == launch["closed_preparation"]
            and time.monotonic() - started < SECONDS["closeout"],
            "unchanged independently closed source and service context",
        )
        result = dict(
            protocol=PROTOCOL + ":closeout",
            source=source,
            report_sha256=base.digest(report_raw),
            files_rehashed=_inventory(root, COMPLETE_FILES),
            scores=scores,
            pair=pair,
            decision=_decision(scores, pair),
            elapsed_seconds=float(time.monotonic() - started),
            service_properties=service,
            run_terminal=terminal,
            idle_before=idle,
            idle_after=base.gpu_idle_gate.wait_idle(),
            **FLAGS,
        )
        base.write_json(root / "independent-closeout.json", result)
    return result


def _failed_terminal(source, invocation):
    base._hex(invocation, 32, "original failed run invocation")
    value = {
        k: base.host.read(
            "systemctl", "--user", "show", unit(source, "run"), "-p", k, "--value"
        )
        for k in (
            "MainPID",
            "ActiveState",
            "NRestarts",
            "ExecMainStatus",
            "Result",
            "InvocationID",
        )
    }
    require(
        value
        == dict(
            MainPID="0",
            ActiveState="failed",
            NRestarts="0",
            ExecMainStatus="1",
            Result="exit-code",
            InvocationID=invocation,
        ),
        "exact terminal original failed run, never restarted",
    )
    return value


def diagnose(source):
    """Independently close a complete failed pair, never a success closeout."""
    _hidden()
    check_window(reserve_seconds=SECONDS["diagnose"] + MARGIN)
    service = _properties(source, "diagnose")
    root = output_path(source)
    base._exact_inventory(root, COMPLETE_FILES)
    started = time.monotonic()
    report_raw = base._read_file(root / "report.json", base.JSON_LIMIT)
    report = base.parse_json(report_raw)
    errors = {
        "paired rollout semantic state exactness",
        "paired constructor private streams and physical fields exact",
        "paired persistent early trace exactness",
    }
    require(
        report.get("protocol") == PROTOCOL
        and report.get("source") == source
        and report.get("status") == "failed-retained"
        and report.get("error_type") == "ValueError"
        and report.get("error") in errors
        and all(report.get(k) is False for k in FLAGS),
        "retained complete numerical pair failure only",
    )
    inventory = _inventory(root, COMPLETE_FILES)
    require(
        {k: v for k, v in inventory.items() if k != "report.json"}
        == report["partial_inventory"],
        "independently rehashed unchanged failed inputs",
    )
    with base.gap.base.files.gpu_lease() as fd:
        idle = base.gpu_idle_gate.wait_idle()
        raw_launch = base._read_file(root / "launch.json", base.JSON_LIMIT)
        sha = base.digest(raw_launch)
        launch = _read_launch(source, sha)
        _recorded_properties(launch["service_properties"], "run")
        invocation = launch["service_properties"]["InvocationID"]
        terminal = _failed_terminal(source, invocation)
        prior = closed.checked(source, lease_fd=fd)
        require(
            prior["launch_binding"] == launch["closed_preparation"]
            and source_binding(source) == launch["source_binding"],
            "fresh exact failed-run source and preparation",
        )
        _read_tool(source, "preflight", prior)
        _read_tool(source, "tests", prior)
        inputs, scores, traces = _score_inputs(root, source, launch, sha)
        pair_error = None
        try:
            _strict_pair(inputs, traces)
        except ValueError as error:
            pair_error = str(error)
        require(
            pair_error == report["error"],
            "independently reproduced exact strict pair failure",
        )
        comparison = early_trace.compare(
            *traces, launch["schedule"], [x["archive"]["records"][0] for x in inputs]
        )
        require(
            _inventory(root, COMPLETE_FILES) == inventory
            and _failed_terminal(source, invocation) == terminal
            and source_binding(source) == launch["source_binding"]
            and closed.checked(source, lease_fd=fd)["launch_binding"]
            == prior["launch_binding"]
            and _properties(source, "diagnose") == service
            and time.monotonic() - started < SECONDS["diagnose"],
            "unchanged independently diagnosed source, inputs and services",
        )
        result = dict(
            protocol=PROTOCOL + ":failure-diagnosis",
            source=source,
            status="failed-pair-independently-diagnosed",
            report_sha256=base.digest(report_raw),
            launch_sha256=sha,
            files_rehashed=inventory,
            scores=scores,
            pair_error=pair_error,
            pair_accepted=False,
            early_trace_comparison=comparison,
            elapsed_seconds=float(time.monotonic() - started),
            service_properties=service,
            failed_run_terminal=terminal,
            idle_before=idle,
            idle_after=base.gpu_idle_gate.wait_idle(),
            **FLAGS,
        )
        base.write_json(root / "independent-failure-diagnosis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=(*SECONDS, "child"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--launch-sha256")
    parser.add_argument("--attempt", choices=ATTEMPTS)
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args()
    if args.mode == "child":
        result = child(args.source, args.launch_sha256, args.attempt, args.lease_fd)
    else:
        result = {
            "preflight": preflight,
            "tests": tests,
            "run": run,
            "closeout": closeout,
            "diagnose": diagnose,
        }[args.mode](args.source)
    print(canonical(result))


if __name__ == "__main__":
    main()
