"""Capped four-child CUDA shadow capture/replay; never a simulation or trainer.

Every capture has a fresh paired CUDA replay. Independent CPU closeout checks
retained records, not CUDA math. Closed namespaces and failed jobs stay immutable.
"""

import argparse
import io
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import torch

from mjlab_microduck import stance_recovery_cuda_policy_probe as base
from mjlab_microduck import stance_recovery_cuda_preparation_evidence as closed
from mjlab_microduck import stance_recovery_cuda_shadow_evidence as evidence
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cuda64-shadow-probe-v1"
MODULE = "mjlab_microduck.stance_recovery_cuda_shadow_probe"
PAIRS = ((653, "capture"), (653, "replay"), (659, "capture"), (659, "replay"))
SECONDS = {"preflight": 180, "tests": 240, "run": 600, "closeout": 180}
MEMORY = {
    "preflight": 2 * 1024**3,
    "tests": 2 * 1024**3,
    "run": 3 * 1024**3,
    "closeout": 2 * 1024**3,
}
CHILD_SECONDS, MARGIN_SECONDS = 120, 60
RUN_RESERVE = SECONDS["run"] + SECONDS["closeout"] + MARGIN_SECONDS
TEST_FILES = (
    "test_stance_recovery_broad_receipt_repair.py",
    "test_stance_recovery_broad_screen.py",
    "test_stance_recovery_ppo_probe.py",
    "test_stance_recovery_ppo_trace.py",
    "test_stance_recovery_ppo_bridge.py",
    "test_stance_recovery_dose_screen.py",
    "test_stance_recovery_campaign_window.py",
    "test_stance_recovery_schedule_trace.py",
    "test_stance_recovery_cuda_schedule_trace.py",
    "test_stance_recovery_ppo_receipt_repair.py",
    "test_stance_recovery_optimizer_trace.py",
    "test_stance_recovery_optimizer_probe.py",
    "test_stance_recovery_terminal_trace.py",
    "test_stance_recovery_terminal_probe.py",
    "test_stance_recovery_terminal_receipt_repair.py",
    "test_stance_recovery_gentle_gap_screen.py",
    "test_stance_recovery_cuda_policy_preparation.py",
    "test_stance_recovery_policy_preparation.py",
    "test_stance_recovery_parent.py",
    "test_stance_recovery_cuda_policy_probe.py",
    "test_stance_recovery_cuda_rng_scope.py",
    "test_stance_recovery_cuda_preparation_evidence.py",
    "test_stance_recovery_cuda_shadow_sampling.py",
    "test_stance_recovery_cuda_shadow_evidence.py",
    "test_stance_recovery_cuda_shadow_probe.py",
)
OWN_FILES = (
    tuple(
        "src/mjlab_microduck/" + n + ".py"
        for n in (
            "stance_recovery_cuda_preparation_evidence",
            "stance_recovery_cuda_rng_scope",
            "stance_recovery_cuda_shadow_sampling",
            "stance_recovery_cuda_shadow_evidence",
            "stance_recovery_cuda_shadow_probe",
        )
    )
    + tuple("tests/" + n for n in TEST_FILES[-5:])
    + ("docs/experiments/2026-10-03-cuda64-shadow-sampler-probe.md",)
)
EXPECTED_TESTS = 679  # Fixed count of the complete declared 25-file source suite.
SYNC_FAILURE_RECEIPT = (
    "44c2b12a3553da6fa177b496672f7c4f0454e031684bcd36c31154a77bea3abb"
)
SYNC_FAILURE_JOURNAL = (
    "a123208f263b4f79782965f2a2e0211deadfd64e49988ed4ff619ce582a55e51"
)
FALSE_FLAGS = {
    **sampling.FALSE_FLAGS,
    "execution_admitted": False,
    "training_job_predeclared": False,
    "finite_optimizer_step_qualified": False,
    "transition_bridge_qualified": False,
    "schedule_installed": False,
    "student_export_available": False,
}
PRE_REPORT_FILES = {
    "launch.json",
    "cpu-parent-receipt.json",
    *(
        f"seed-{seed}-{attempt}.{suffix}"
        for seed, attempt in PAIRS
        for suffix in ("pt", "json", "log")
    ),
}
COMPLETE_FILES = PRE_REPORT_FILES | {"report.json"}


def output_path(source):
    base._hex(source, 40, "shadow source")
    require(source != closed.ARTIFACT_SOURCE, "separate shadow namespace")
    return (
        base.execution.ROOT
        / "artifacts/evaluations"
        / ("stance-wsl-cuda64-shadow-" + source[:12])
    )


def tools_path(source, kind):
    output_path(source)
    require(kind in ("preflight", "tests"), "declared shadow CPU tool")
    return (
        base.execution.ROOT
        / "artifacts/tools"
        / (f"cuda64-shadow-{kind}-" + source[:12])
    )


def unit(source, mode):
    output_path(source)
    require(mode in SECONDS, "declared shadow service mode")
    return f"microduck-cuda64-shadow-{mode}-{source[:12]}.service"


def _properties(source, mode):
    value = {
        k: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", k, "--value"
        )
        for k in (
            "MainPID",
            "ActiveState",
            "RuntimeMaxUSec",
            "MemoryMax",
            "CPUQuotaPerSecUSec",
            "Nice",
            "KillMode",
            "InvocationID",
        )
    }
    _recorded_properties(value, mode)
    require(
        value["MainPID"] == str(os.getpid()),
        "actual capped shadow service owns this process",
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
        {v.split()[0] for v in running.splitlines()} == {unit(source, mode)},
        "only this Duck service runs",
    )
    return value


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
        "exact service property schema",
    )
    base._hex(value["InvocationID"], 32, "retained shadow service invocation")
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
        "exact declared shadow service caps",
    )
    return True


def completed(source, mode, invocation):
    base._hex(invocation, 32, "original shadow invocation")
    state = {
        k: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", k, "--value"
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
        state["InvocationID"] in ("", invocation)
        and {k: v for k, v in state.items() if k != "InvocationID"}
        == {
            "MainPID": "0",
            "ActiveState": "inactive",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "Result": "success",
        },
        "original shadow service is successfully terminal without restart",
    )
    return state


def source_binding(source):
    record = base._source_binding(source)
    root = Path(__file__).resolve().parents[2]
    leaves = {}
    for relative in OWN_FILES:
        raw = subprocess.run(
            ["git", "show", f"{source}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        require(
            raw == (root / relative).read_bytes(),
            "exact committed shadow leaf " + relative,
        )
        leaves[relative] = base.digest(raw)
    return {
        "original_frozen_leaves": record,
        "shadow_leaves": leaves,
        "retained_source_sync_failure": source_sync_failure_binding(),
    }


def source_sync_failure_binding():
    """Preserve the first wrapper's missing-WorkingDirectory failure unchanged."""
    root = (
        base.execution.ROOT
        / "artifacts/tools/cuda64-shadow-source-sync-failure-5eca76e0b85d"
    )
    base._exact_inventory(root, {"receipt.json", "journal.jsonl"})
    raw = base._read_file(root / "receipt.json", base.JSON_LIMIT)
    journal = base._read_file(root / "journal.jsonl", base.LOG_LIMIT)
    require(
        base.digest(raw) == SYNC_FAILURE_RECEIPT
        and base.digest(journal) == SYNC_FAILURE_JOURNAL,
        "whole original source-sync failure and journal",
    )
    unit_name = "microduck-cuda64-shadow-source-sync-5eca76e0b85d.service"
    state = {
        key: base.host.read(
            "systemctl", "--user", "show", unit_name, "-p", key, "--value"
        )
        for key in (
            "MainPID",
            "ActiveState",
            "NRestarts",
            "ExecMainStatus",
            "Result",
            "InvocationID",
            "WorkingDirectory",
        )
    }
    recorded = base.parse_json(raw)
    require(
        state == recorded["original_terminal"],
        "original failed source-sync unit neither reset restarted nor replaced",
    )
    original = base._read_file(
        base.execution.ROOT
        / "artifacts/tools/cuda64-shadow-source-sync-5eca76e0b85d/receipt.json",
        base.JSON_LIMIT,
    )
    require(
        base.digest(original) == recorded["original_receipt_sha256"]
        and len(original) == recorded["original_receipt_bytes"],
        "whole original failed source-sync receipt unchanged",
    )
    return {
        "receipt_sha256": SYNC_FAILURE_RECEIPT,
        "journal_sha256": SYNC_FAILURE_JOURNAL,
        "original_receipt_sha256": recorded["original_receipt_sha256"],
        "original_terminal": state,
    }


def _hidden():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized()
        and base.execution.PROFILE == base.execution.select(base.execution.WSL),
        "initially CUDA-hidden exact WSL shadow supervisor",
    )


def preflight(source):
    _hidden()
    started = time.monotonic()
    base.window.check(
        reserve_seconds=SECONDS["preflight"] + SECONDS["tests"] + RUN_RESERVE
    )
    service = _properties(source, "preflight")
    root = tools_path(source, "preflight")
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "protocol": PROTOCOL + ":preflight-v1",
        "source": source,
        "status": "failed-retained",
        **FALSE_FLAGS,
    }
    try:
        with base.gap.base.files.gpu_lease() as fd:
            idle_before = base.gpu_idle_gate.wait_idle()
            binding = source_binding(source)
            old = closed.checked(source, lease_fd=fd)
            rng = torch.random.get_rng_state().clone()
            loaded, cpu_raw, cpu_sha, canonical_sha = base._cpu_parent_prepare(
                old["raw_parent"]
            )
            cpu_binding = base._validated_cpu_binding(
                source, loaded["receipt"], canonical_sha
            )
            require(
                torch.equal(rng, torch.random.get_rng_state()),
                "preflight caller CPU RNG preserved",
            )
            del loaded
            base._write_exclusive(
                root / "cpu-parent-receipt.json", cpu_raw, base.JSON_LIMIT
            )
            idle_after = base.gpu_idle_gate.wait_idle()
            require(
                closed.checked(source, lease_fd=fd)["launch_binding"]
                == old["launch_binding"]
                and source_binding(source) == binding
                and _properties(source, "preflight") == service,
                "unchanged fresh closed preparation and source across CPU preflight",
            )
            require(
                time.monotonic() - started < SECONDS["preflight"],
                "bounded actual CPU preflight",
            )
            base.window.check(reserve_seconds=SECONDS["tests"] + RUN_RESERVE)
            receipt.update(
                status="actual-hidden-cpu-preflight-checked",
                closed_preparation=old["launch_binding"],
                source_binding=binding,
                cpu_parent_receipt_file_sha256=cpu_sha,
                cpu_parent_receipt_canonical_sha256=canonical_sha,
                cpu_parent_binding=cpu_binding,
                service_properties=service,
                idle_before=idle_before,
                idle_after=idle_after,
            )
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "receipt.json", receipt)
    return receipt


def _closed_preflight(source, current):
    root = tools_path(source, "preflight")
    base._exact_inventory(root, {"receipt.json", "cpu-parent-receipt.json"})
    raw = base._read_file(root / "receipt.json", base.JSON_LIMIT)
    rec = base.parse_json(raw)
    cpu_raw = base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT)
    cpu = base.parse_json(cpu_raw)
    canonical_sha = sampling.preparation._cpu_parent_receipt_digest(cpu)
    require(
        rec.get("protocol") == PROTOCOL + ":preflight-v1"
        and rec.get("source") == source
        and rec.get("status") == "actual-hidden-cpu-preflight-checked"
        and rec.get("closed_preparation") == current["launch_binding"]
        and rec.get("source_binding") == source_binding(source)
        and rec.get("cpu_parent_receipt_file_sha256") == base.digest(cpu_raw)
        and rec.get("cpu_parent_receipt_canonical_sha256") == canonical_sha
        and base._validated_cpu_binding(source, cpu, canonical_sha)
        == rec.get("cpu_parent_binding")
        and all(rec.get(k) is False for k in FALSE_FLAGS),
        "retained exact actual CPU shadow preflight",
    )
    _recorded_properties(rec.get("service_properties"), "preflight")
    state = completed(source, "preflight", rec["service_properties"]["InvocationID"])
    return {"receipt_sha256": base.digest(raw), "terminal": state}, cpu_raw, rec


def _closed_tests(source, preflight_binding):
    root = tools_path(source, "tests")
    base._exact_inventory(root, {"receipt.json", "pytest.log"})
    raw = base._read_file(root / "receipt.json", base.JSON_LIMIT)
    rec = base.parse_json(raw)
    log = base._read_file(root / "pytest.log", base.LOG_LIMIT)
    require(
        rec.get("protocol") == PROTOCOL + ":tests-v1"
        and rec.get("source") == source
        and rec.get("status") == "native-source-tests-passed"
        and rec.get("tests") == ["tests/" + p for p in TEST_FILES]
        and type(rec.get("tests_passed")) is int
        and rec["tests_passed"] == EXPECTED_TESTS
        and rec.get("skips") == 0
        and type(rec.get("skips")) is int
        and rec.get("pytest_log_sha256") == base.digest(log)
        and rec.get("preflight_binding") == preflight_binding
        and rec.get("source_binding") == source_binding(source)
        and all(rec.get(k) is False for k in FALSE_FLAGS),
        "exact complete native shadow source suite",
    )
    _recorded_properties(rec.get("service_properties"), "tests")
    return {
        "receipt_sha256": base.digest(raw),
        "pytest_log_sha256": base.digest(log),
        "terminal": completed(
            source, "tests", rec["service_properties"]["InvocationID"]
        ),
    }


def declaration():
    return {
        "protocol": PROTOCOL,
        "pairs": [[s, a] for s, a in PAIRS],
        "worlds": 64,
        "horizon": 28,
        "service_seconds": SECONDS,
        "service_memory_bytes": MEMORY,
        "child_seconds": CHILD_SECONDS,
        "margin_seconds": MARGIN_SECONDS,
        "launch_reserve_seconds": RUN_RESERVE,
        "raw_bytes_limit": base.RAW_LIMIT,
        "log_bytes_limit": base.LOG_LIMIT,
        "json_bytes_limit": base.JSON_LIMIT,
        "synthetic_zero_observations": True,
        "rollout_collection": False,
        "simulation_steps": 0,
        "optimizer_steps": 0,
        "scope_sampler_qualification_only": True,
        "physical_motion_authorized": False,
        "campaign_window": base.window.declaration(),
        "tests_expected": EXPECTED_TESTS,
        **FALSE_FLAGS,
    }


def _read_launch(source, sha):
    base._hex(sha, 64, "shadow launch SHA256")
    root = output_path(source)
    raw = base._read_file(root / "launch.json", base.JSON_LIMIT)
    require(base.digest(raw) == sha, "whole shadow launch before parse")
    rec = base.parse_json(raw)
    require(
        rec.get("protocol") == PROTOCOL
        and rec.get("source") == source
        and rec.get("declaration") == declaration(),
        "exact fixed shadow declaration",
    )
    _recorded_properties(rec.get("service_properties"), "run")
    cpu_raw = base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT)
    require(
        base.digest(cpu_raw) == rec.get("cpu_parent_receipt_file_sha256"),
        "whole actual CPU parent receipt",
    )
    cpu = base.parse_json(cpu_raw)
    require(
        sampling.preparation._cpu_parent_receipt_digest(cpu)
        == rec.get("cpu_parent_receipt_canonical_sha256")
        and base._validated_cpu_binding(
            source, cpu, rec["cpu_parent_receipt_canonical_sha256"]
        )
        == rec.get("cpu_parent_binding"),
        "exact source-bound canonical CPU receipt",
    )
    return root, rec, cpu


def child(source, launch_sha, seed, attempt, lease_fd):
    base.training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "visible CUDA0 only from child process start",
    )
    require(
        type(seed) is int and (seed, attempt) in PAIRS,
        "one declared capture or fresh replay",
    )
    root, launch, cpu = _read_launch(source, launch_sha)
    require(
        source_binding(source) == launch["source_binding"],
        "exact committed child source",
    )
    context = base._visible_child_context(source, launch)
    parent = base._read_file(
        base.gap.output_path(base.ARTIFACT_SOURCE) / "checkpoint.pt",
        base.checkpoint.LIMIT,
    )
    require(
        base.digest(parent) == base.baseline.CHECKPOINT_SHA256,
        "hash-first frozen D1 parent",
    )
    prepared = sampling.preparation.prepare_policy(
        parent,
        source=source,
        lease_fd=lease_fd,
        cpu_parent_receipt=cpu,
        cpu_parent_receipt_sha256=launch["cpu_parent_receipt_canonical_sha256"],
        caller_binding=launch["cpu_parent_binding"],
        seed=seed,
        worlds=64,
    )
    raw_preparation, meta = base._child_payload(prepared, launch_sha)
    sampled = sampling.sample_shadow(prepared, lease_fd=lease_fd)
    require(
        base._visible_child_context(source, launch) == context
        and source_binding(source) == launch["source_binding"],
        "unchanged child source/runtime/external context",
    )
    payload = {
        "protocol": PROTOCOL + ":child-v1",
        "source": source,
        "launch_sha256": launch_sha,
        "seed": seed,
        "attempt": attempt,
        "preparation_raw": raw_preparation,
        "sampling": sampled,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    tag = f"seed-{seed}-{attempt}"
    sha = base._write_exclusive(root / (tag + ".pt"), raw, base.RAW_LIMIT)
    summary = {
        "protocol": PROTOCOL + ":child-summary-v1",
        "source": source,
        "launch_sha256": launch_sha,
        "seed": seed,
        "attempt": attempt,
        "payload_sha256": sha,
        "payload_bytes": len(raw),
        "preparation_sha256": base.digest(raw_preparation),
        "preparation_metadata": meta,
        "sampling_receipt": sampled["receipt"],
        "optimizer_steps": 0,
        **FALSE_FLAGS,
    }
    base.write_json(root / (tag + ".json"), summary)
    return summary


def child_environment():
    # Explicit frozen startup profile before Python/Torch import, not an
    # after-import check of whichever inherited CPU math settings happened to run.
    return {
        **base.campaign.child_environment(),
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "ATEN_CPU_CAPABILITY": "default",
        "MKL_CBWR": "COMPATIBLE",
        "PYTHONUNBUFFERED": "1",
        "PATH": str(base.execution.ROOT / ".venv/bin")
        + ":/usr/local/bin:/usr/bin:/bin",
    }


def _run_child(source, sha, seed, attempt, fd, root):
    env = child_environment()
    tag = f"seed-{seed}-{attempt}"
    samples = []
    total = 0
    with (root / (tag + ".log")).open("xb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                MODULE,
                "child",
                "--source",
                source,
                "--launch-sha256",
                sha,
                "--seed",
                str(seed),
                "--attempt",
                attempt,
                "--lease-fd",
                str(fd),
            ],
            cwd=base.execution.ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(fd,),
        )
        started = time.monotonic()
        deadline = started + CHILD_SECONDS
        next_monitor = started
        pipe_open = True
        try:
            while proc.poll() is None or pipe_open:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("shadow child exceeded fixed 120-second cap")
                readable, _, _ = select.select(
                    [proc.stdout] if pipe_open else [], [], [], min(0.25, remaining)
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
                        total += min(allowed, len(chunk))
                        log.flush()
                        require(len(chunk) <= allowed, "shadow streaming log cap")
                if time.monotonic() >= next_monitor and proc.poll() is None:
                    samples.append(
                        {
                            "elapsed_seconds": float(time.monotonic() - started),
                            "child_pid": proc.pid,
                            "sample": base.campaign.live_gpu(proc.pid),
                        }
                    )
                    next_monitor = time.monotonic() + 2
            require(
                proc.returncode == 0 and samples,
                "successful child and actual bounded ownership samples",
            )
        except BaseException:
            base.campaign._kill_owned_group(proc)
            raise
        finally:
            proc.stdout.close()
            log.flush()
            os.fsync(log.fileno())
    summary = base.parse_json(base._read_file(root / (tag + ".json"), base.JSON_LIMIT))
    return {**summary, "live_gpu_samples": samples}


def _gpu_samples(children):
    require(
        type(children) is list and len(children) == 4,
        "four ordered fresh CUDA children",
    )
    require(
        [(c.get("seed"), c.get("attempt")) for c in children] == list(PAIRS),
        "exact sequential capture/replay order",
    )
    # The frozen checker expects two ordered seeds. Validate each attempt's pair
    # without altering that checker or treating four children as concurrent.
    for i, j in ((0, 2), (1, 3)):
        base._check_live_gpu_samples([children[i], children[j]])


def _score_child(root, launch, launch_sha, seed, attempt):
    tag = f"seed-{seed}-{attempt}"
    summary = base.parse_json(base._read_file(root / (tag + ".json"), base.JSON_LIMIT))
    raw = base._read_file(root / (tag + ".pt"), base.RAW_LIMIT)
    require(
        type(summary) is dict
        and set(summary)
        == {
            "protocol",
            "source",
            "launch_sha256",
            "seed",
            "attempt",
            "payload_sha256",
            "payload_bytes",
            "preparation_sha256",
            "preparation_metadata",
            "sampling_receipt",
            "optimizer_steps",
            *FALSE_FLAGS,
        }
        and type(summary.get("seed")) is int
        and type(summary.get("payload_bytes")) is int
        and type(summary.get("optimizer_steps")) is int
        and summary["optimizer_steps"] == 0,
        "exact typed non-admitting child summary schema",
    )
    require(
        summary.get("payload_sha256") == base.digest(raw)
        and summary.get("payload_bytes") == len(raw),
        "whole shadow payload before weights-only CPU load",
    )
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    require(
        type(payload) is dict
        and set(payload)
        == {
            "protocol",
            "source",
            "launch_sha256",
            "seed",
            "attempt",
            "preparation_raw",
            "sampling",
        }
        and payload.get("protocol") == PROTOCOL + ":child-v1"
        and payload.get("source") == summary.get("source") == launch["source"]
        and payload.get("launch_sha256") == summary.get("launch_sha256") == launch_sha
        and type(payload.get("seed")) is int
        and payload["seed"] == summary.get("seed") == seed
        and payload.get("attempt") == summary.get("attempt") == attempt
        and summary.get("protocol") == PROTOCOL + ":child-summary-v1"
        and summary.get("sampling_receipt") == payload["sampling"]["receipt"]
        and all(summary.get(k) is False for k in FALSE_FLAGS),
        "exact whole child binding and non-admitting summary",
    )
    prep_raw = payload["preparation_raw"]
    require(
        type(prep_raw) is bytes
        and base.digest(prep_raw) == summary["preparation_sha256"],
        "whole actual preparation subpayload",
    )
    meta = summary["preparation_metadata"]
    old_summary = {
        "protocol": base.PROTOCOL + ":child-summary-v1",
        "source": launch["source"],
        "launch_sha256": launch_sha,
        "seed": seed,
        "payload_sha256": base.digest(prep_raw),
        "payload_bytes": len(prep_raw),
        "metadata": meta,
        "cpu_parent_receipt_file_sha256": launch["cpu_parent_receipt_file_sha256"],
        "cpu_parent_receipt_canonical_sha256": launch[
            "cpu_parent_receipt_canonical_sha256"
        ],
        "optimizer_steps": 0,
        "simulator_resets": 0,
        "training_update_performed": False,
        "cuda_child_claims_cpu_provenance_authenticated": False,
        "action_sampling": False,
        "return_computation": False,
        "rollout_collection": False,
        **base.PREPARATION_FALSE_FLAGS,
    }
    old_summary["private_cuda_rng_state_sha256"] = meta["private_cuda_rng_state_sha256"]
    cpu = base.parse_json(
        base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT)
    )
    base._score_payload(prep_raw, old_summary, launch, launch_sha, cpu, seed)
    prepared = torch.load(io.BytesIO(prep_raw), map_location="cpu", weights_only=True)
    score = evidence.score_sampling(
        payload["sampling"], prepared, source=launch["source"], seed=seed
    )
    return (
        {"sampling": payload["sampling"], "prepared_snapshot": prepared},
        score,
        summary,
    )


def _scores(root, launch, sha):
    inputs = []
    scores = []
    summaries = []
    for seed, attempt in PAIRS:
        value, score, summary = _score_child(root, launch, sha, seed, attempt)
        inputs.append(value)
        scores.append(score)
        summaries.append(summary)
    pairs = [
        evidence.compare_pair(
            inputs[i], inputs[i + 1], source=launch["source"], seed=seed
        )
        for i, seed in ((0, 653), (2, 659))
    ]
    return scores, pairs, summaries


def run(source):
    _hidden()
    started = time.monotonic()
    base.window.check(reserve_seconds=RUN_RESERVE)
    service = _properties(source, "run")
    root = output_path(source)
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": PROTOCOL,
        "source": source,
        "status": "failed-retained",
        **FALSE_FLAGS,
    }
    try:
        with base.gap.base.files.gpu_lease() as fd:
            idle_before = base.gpu_idle_gate.wait_idle()
            current = closed.checked(source, lease_fd=fd)
            preflight_binding, cpu_raw, preflight_rec = _closed_preflight(
                source, current
            )
            tests_binding = _closed_tests(source, preflight_binding)
            loaded, fresh_raw, _, _ = base._cpu_parent_prepare(current["raw_parent"])
            require(
                fresh_raw == cpu_raw,
                "fresh actual parent constructor matches closed preflight",
            )
            del loaded
            binding = source_binding(source)
            launch = {
                "protocol": PROTOCOL,
                "source": source,
                "declaration": declaration(),
                "service_properties": service,
                "source_binding": binding,
                "closed_preparation": current["launch_binding"],
                "native_prerequisites": current["current_prerequisites"][
                    "launch_binding"
                ],
                "preflight_binding": preflight_binding,
                "tests_binding": tests_binding,
                "cpu_parent_receipt_file_sha256": base.digest(cpu_raw),
                "cpu_parent_receipt_canonical_sha256": preflight_rec[
                    "cpu_parent_receipt_canonical_sha256"
                ],
                "cpu_parent_binding": preflight_rec["cpu_parent_binding"],
                "idle_before": idle_before,
                "start_unix": time.time(),
            }
            base._write_exclusive(
                root / "cpu-parent-receipt.json", cpu_raw, base.JSON_LIMIT
            )
            sha = base.write_json(root / "launch.json", launch)
            children = []
            idle_boundaries = []
            for seed, attempt in PAIRS:
                base.window.check(
                    reserve_seconds=SECONDS["closeout"] + MARGIN_SECONDS + CHILD_SECONDS
                )
                idle_boundaries.append(base.gpu_idle_gate.wait_idle())
                children.append(_run_child(source, sha, seed, attempt, fd, root))
            idle_after = base.gpu_idle_gate.wait_idle()
            _gpu_samples(children)
            scores, pairs, _ = _scores(root, launch, sha)
            require(
                closed.checked(source, lease_fd=fd)["launch_binding"]
                == current["launch_binding"]
                and source_binding(source) == binding
                and _properties(source, "run") == service,
                "source/provenance/external context unchanged after all children",
            )
            files = base._complete_inventory(root, PRE_REPORT_FILES)
            require(
                time.monotonic() - started < SECONDS["run"],
                "complete four-child run under fixed cap",
            )
            base.window.check(reserve_seconds=SECONDS["closeout"] + MARGIN_SECONDS)
            report.update(
                status="paired-shadow-captured-awaiting-closeout",
                launch_sha256=sha,
                children=children,
                scores=scores,
                pairs=pairs,
                files=files,
                idle_boundaries=idle_boundaries,
                idle_after=idle_after,
                service_properties=service,
                simulation_steps=0,
                optimizer_steps=0,
            )
    except BaseException as error:
        report.update(
            error_type=type(error).__name__,
            error=str(error),
            partial_files=base._partial_inventory(root),
        )
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "report.json", report)
    return report


def closeout(source, sha):
    _hidden()
    started = time.monotonic()
    base.window.check(reserve_seconds=SECONDS["closeout"] + MARGIN_SECONDS)
    service = _properties(source, "closeout")
    root, launch, _ = _read_launch(source, sha)
    result = {
        "protocol": PROTOCOL + ":closeout-v1",
        "source": source,
        "status": "failed-retained",
        **FALSE_FLAGS,
    }
    try:
        run_terminal = completed(
            source, "run", launch["service_properties"]["InvocationID"]
        )
        with base.gap.base.files.gpu_lease() as fd:
            idle_before = base.gpu_idle_gate.wait_idle()
            current = closed.checked(source, lease_fd=fd)
            preflight_binding, cpu_raw, _ = _closed_preflight(source, current)
            require(
                current["launch_binding"] == launch["closed_preparation"]
                and preflight_binding == launch["preflight_binding"]
                and _closed_tests(source, preflight_binding) == launch["tests_binding"]
                and source_binding(source) == launch["source_binding"],
                "current complete provenance/source/test chain",
            )
            loaded, fresh_raw, _, _ = base._cpu_parent_prepare(current["raw_parent"])
            require(
                fresh_raw
                == cpu_raw
                == base._read_file(root / "cpu-parent-receipt.json", base.JSON_LIMIT),
                "fresh CPU parent constructor independently matches launch",
            )
            del loaded
            report_raw = base._read_file(root / "report.json", base.JSON_LIMIT)
            report = base.parse_json(report_raw)
            require(
                report.get("protocol") == PROTOCOL
                and report.get("source") == source
                and report.get("launch_sha256") == sha
                and report.get("status") == "paired-shadow-captured-awaiting-closeout"
                and all(report.get(k) is False for k in FALSE_FLAGS),
                "successful unchanged non-admitting run report",
            )
            _gpu_samples(report.get("children"))
            inventory = base._complete_inventory(root, COMPLETE_FILES)
            require(
                report.get("files")
                == {k: v for k, v in inventory.items() if k != "report.json"},
                "report binds every complete raw/JSON/log file",
            )
            scores, pairs, summaries = _scores(root, launch, sha)
            require(
                scores == report.get("scores")
                and pairs == report.get("pairs")
                and [
                    {k: v for k, v in child.items() if k != "live_gpu_samples"}
                    for child in report["children"]
                ]
                == summaries,
                "independent complete CPU rescore and exact pair decisions",
            )
            idle_after = base.gpu_idle_gate.wait_idle()
            require(
                closed.checked(source, lease_fd=fd)["launch_binding"]
                == current["launch_binding"]
                and source_binding(source) == launch["source_binding"]
                and _properties(source, "closeout") == service,
                "unchanged final provenance/context and capped closeout",
            )
            require(
                time.monotonic() - started < SECONDS["closeout"],
                "bounded complete independent closeout",
            )
            base.window.check(reserve_seconds=MARGIN_SECONDS)
            result.update(
                status="closed-non-admitting",
                decision="fixed-input-cuda64-shadow-pairs-exact",
                launch_sha256=sha,
                report_sha256=base.digest(report_raw),
                files_rehashed=inventory,
                scores=scores,
                pairs=pairs,
                service_properties=service,
                run_terminal=run_terminal,
                idle_before=idle_before,
                idle_after=idle_after,
                cpu_consistency_rescore_identical=True,
                cuda_math_replayed_by_cpu_closeout=False,
                simulation_steps=0,
                optimizer_steps=0,
            )
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "independent-closeout.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "closeout", "child"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--launch-sha256")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", choices=("capture", "replay"))
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        result = preflight(args.source)
    elif args.mode == "run":
        result = run(args.source)
    elif args.mode == "closeout":
        result = closeout(args.source, args.launch_sha256)
    else:
        result = child(
            args.source, args.launch_sha256, args.seed, args.attempt, args.lease_fd
        )
    print(canonical(result))


if __name__ == "__main__":
    main()
