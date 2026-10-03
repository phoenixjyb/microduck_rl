"""Saved-record-only repair for the first-terminal closeout receipt failure.

This protocol authenticates the immutable original capture and failed closeout,
then independently replays the saved trace with the fixed terminal scorer. It
never creates an environment, collects transitions, resets a simulator, or runs
an optimizer. A successful receipt can attest only the recorded common timeout
and reset; it cannot qualify selective reset or a higher-skill policy.
"""

import argparse
import io
import math
import os
import time
from hashlib import sha256
from pathlib import Path

import torch

from mjlab_microduck import stance_recovery_broad_receipt_repair as broad
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_terminal_probe as probe
from mjlab_microduck import stance_recovery_terminal_trace as trace
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cpu-stochastic-first-terminal-receipt-repair-v1"
ARTIFACT_SOURCE = "9ad48b96abf6528c2f837399dc8fedd83a286464"
ARTIFACT_LAUNCH_SHA256 = (
    "4529c25ce1a728805196a7c8b7be290ab87f019239aae24cf40a6da31fb50bf5"
)
ARTIFACT_PARENT_SHA256 = (
    "2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5"
)
ARTIFACT_CAPTURE_SHA256 = (
    "82ef6175674a520448a3e6cfc19ca075377f1f2f90428cb945883f1e6695db76"
)
ARTIFACT_CAPTURE_METADATA_SHA256 = (
    "76633ec5c77aed675a632190d9c1b2615178930e9f2816a3f0427d19f79ad6e8"
)
ARTIFACT_REPORT_SHA256 = (
    "712a7201bad070b8c431f3310cb27566799f14af7832bda77af0feff9237a98e"
)
ARTIFACT_BYTES = {
    "checkpoint.pt": 256368,
    "launch.json": 25858,
    "capture.pt": 49950070,
    "capture.json": 2664,
    "report.json": 5584,
}
ARTIFACT_HASHES = {
    "checkpoint.pt": ARTIFACT_PARENT_SHA256,
    "launch.json": ARTIFACT_LAUNCH_SHA256,
    "capture.pt": ARTIFACT_CAPTURE_SHA256,
    "capture.json": ARTIFACT_CAPTURE_METADATA_SHA256,
    "report.json": ARTIFACT_REPORT_SHA256,
}
FAILURE_INVOCATION = "ab8b2d101f6f42a1ba94d7f8f9aa5df4"
FAILURE_RECEIPT_SHA256 = (
    "fd83806a4fb92807e35289208f3d33c95ae6af1ef7d12c8ba0fbfedd953c2b1a"
)
FAILURE_JOURNAL_SHA256 = (
    "cce0d9b412a5bd86e3d485b0a188a4e40dc6fa93ea000707ec2bf760e0ea4378"
)
FAILURE_DIRECTORY = "artifacts/tools/terminal-reset-qpos-failure-9ad48b96abf6"
FAILURE_RECEIPT_LIMIT = 2 * 1024**2
FAILURE_JOURNAL_LIMIT = 16 * 1024**2
SERVICE_SECONDS = 180
LAUNCH_RESERVE_SECONDS = 240
MEMORY_BYTES = 2 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
OUTPUT_PREFIX = "stance-wsl-cpu-terminal-receipt-repair-"
ARTIFACT_ROOT = Path("artifacts/evaluations") / (
    "stance-wsl-cpu-stochastic-first-terminal-" + ARTIFACT_SOURCE[:12]
)
TRACE_PATH = "src/mjlab_microduck/stance_recovery_terminal_trace.py"
PROBE_PATH = "src/mjlab_microduck/stance_recovery_terminal_probe.py"
OLD_PROBE_SHA256 = "f863a9edd9c034de9c14aa6230bde6734c0e9924c70a3f795ca55eea2695ca13"
OLD_TRACE_SHA256 = "c18ea88da58ba8eac3f387e02175f1646ccbe0529ce17f716c30d91439524453"
FIXED_TRACE_SHA256 = "a746923941c514e58e9b785b4fa645dc75275e9f7eee457999b3427be110f1c0"
REPAIR_FILES = {"launch.json", "source-inventory.json", "receipt.json"}
FAILURE_SERVICE_STATE = {
    "ActiveState": "failed",
    "MainPID": "0",
    "NRestarts": "0",
    "ExecMainStatus": "1",
    "InvocationID": FAILURE_INVOCATION,
}


def output_path(evaluator_source):
    base = probe.base
    base.files.hex_id(evaluator_source, 40)
    require(
        evaluator_source != ARTIFACT_SOURCE,
        "receipt repair uses a distinct evaluator source namespace",
    )
    return (
        base.host.ROOT
        / "artifacts/evaluations"
        / (OUTPUT_PREFIX + evaluator_source[:12])
    )


def service_name(evaluator_source):
    output_path(evaluator_source)
    return f"microduck-cpu-terminal-receipt-repair-{evaluator_source[:12]}.service"


def _recorded_properties(value):
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
        }
        and type(value["MainPID"]) is str
        and value["MainPID"].isdecimal()
        and int(value["MainPID"]) > 0
        and value["ActiveState"] == "active"
        and value["RuntimeMaxUSec"] == "3min"
        and value["MemoryMax"] == str(MEMORY_BYTES)
        and value["CPUQuotaPerSecUSec"] == CPU_QUOTA
        and value["Nice"] == NICE
        and value["KillMode"] == KILL_MODE,
        "exact recorded 180-second bounded CPU receipt-repair service",
    )
    return True


def service_properties(evaluator_source):
    base = probe.base
    name = service_name(evaluator_source)
    props = {
        key: base.host.read("systemctl", "--user", "show", name, "-p", key, "--value")
        for key in (
            "MainPID",
            "ActiveState",
            "RuntimeMaxUSec",
            "MemoryMax",
            "CPUQuotaPerSecUSec",
            "Nice",
            "KillMode",
        )
    }
    expected = {
        "MainPID": str(os.getpid()),
        "ActiveState": "active",
        "RuntimeMaxUSec": "3min",
        "MemoryMax": str(MEMORY_BYTES),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
    }
    require(props == expected, "exact owned 180-second CPU receipt-repair service")
    running = base.host.read(
        "systemctl",
        "--user",
        "list-units",
        "--state=running",
        "--no-legend",
        "microduck*",
    )
    require(
        {line.split()[0] for line in running.splitlines()} == {name},
        "only this owned Duck service is active during receipt repair",
    )
    return props


def _check_source_context(original_context, evaluator_context, evaluator_source):
    return broad._check_source_context(
        original_context,
        evaluator_context,
        artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source,
    )


def _source_inventory(repo=None):
    """Pin the old evaluator closure and permit only the fixed trace leaf."""
    repo = Path(probe.__file__).resolve().parents[2] if repo is None else Path(repo)
    originals = broad._source_dependency_closure(
        repo, ARTIFACT_SOURCE, [PROBE_PATH, TRACE_PATH]
    )
    require(
        PROBE_PATH in originals
        and TRACE_PATH in originals
        and sha256(originals[PROBE_PATH]).hexdigest() == OLD_PROBE_SHA256
        and sha256(originals[TRACE_PATH]).hexdigest() == OLD_TRACE_SHA256,
        "exact original terminal scorer and probe bytes at artifact revision",
    )
    entries = {}
    for relative, original in originals.items():
        current = (repo / relative).read_bytes()
        old_hash, current_hash = (
            sha256(original).hexdigest(),
            sha256(current).hexdigest(),
        )
        expected = FIXED_TRACE_SHA256 if relative == TRACE_PATH else old_hash
        require(
            current_hash == expected,
            "only the pinned terminal trace leaf may differ: " + relative,
        )
        entries[relative] = {
            "artifact_source_sha256": old_hash,
            "evaluator_source_sha256": current_hash,
        }
    require(
        entries[PROBE_PATH]["artifact_source_sha256"] == OLD_PROBE_SHA256
        and entries[PROBE_PATH]["evaluator_source_sha256"] == OLD_PROBE_SHA256
        and entries[TRACE_PATH]["evaluator_source_sha256"] == FIXED_TRACE_SHA256,
        "terminal probe remains unchanged and scorer leaf is the exact fixed version",
    )
    return {
        "artifact_source": ARTIFACT_SOURCE,
        "old_probe_sha256": OLD_PROBE_SHA256,
        "old_trace_sha256": OLD_TRACE_SHA256,
        "fixed_trace_sha256": FIXED_TRACE_SHA256,
        "dependencies": entries,
    }


def _original_inventory(root):
    base = probe.base
    root = base.files.native._plain_path(root)
    require(
        {path.name for path in root.iterdir()} == probe.RUN_FILES,
        "exact immutable original five-file terminal capture inventory",
    )
    inventory = {}
    for name in sorted(probe.RUN_FILES):
        raw = base.files.file_bytes(root / name, limit=probe.RAW_LIMIT)
        digest = sha256(raw).hexdigest()
        require(
            len(raw) == ARTIFACT_BYTES[name] and digest == ARTIFACT_HASHES[name],
            "whole pinned original terminal artifact bytes: " + name,
        )
        inventory[name] = {"sha256": digest, "bytes": len(raw)}
    return inventory


def _check_failure_evidence(receipt, journal_raw):
    diagnosis = receipt.get("diagnosis") if type(receipt) is dict else None
    require(
        type(receipt) is dict
        and receipt.get("protocol")
        == "cpu-first-terminal-reset-qpos-failure-evidence-v1"
        and receipt.get("artifact_source") == ARTIFACT_SOURCE
        and receipt.get("invocation_id") == FAILURE_INVOCATION
        and receipt.get("service")
        == f"microduck-cpu-terminal-closeout-{ARTIFACT_SOURCE[:12]}.service"
        and receipt.get("failure_stage")
        == "terminal-original-qpos-layout-after-recorded-policy-control-force-validation"
        and receipt.get("error_type") == "ValueError"
        and receipt.get("error") == "trace tensor layout: original reset qpos"
        and receipt.get("journal_sha256") == FAILURE_JOURNAL_SHA256
        and receipt.get("journal_bytes") == len(journal_raw)
        and type(diagnosis) is dict
        and diagnosis.get("terminal_probe_sha256") == OLD_PROBE_SHA256
        and diagnosis.get("terminal_trace_sha256") == OLD_TRACE_SHA256
        and diagnosis.get("capture_bytes") == ARTIFACT_BYTES["capture.pt"]
        and diagnosis.get("initial_qpos_shape") == [2, 21]
        and diagnosis.get("physical_initial_qpos_shape") == [2, 21]
        and diagnosis.get("initial_qpos_dtype") == "torch.float32"
        and diagnosis.get("exact_initial_qpos_equal") is True
        and diagnosis.get("terminal_flags")
        == [
            {
                "world_id": 0,
                "physics_step": 2500,
                "timed_out": True,
                "terminated": False,
            },
            {
                "world_id": 1,
                "physics_step": 2500,
                "timed_out": True,
                "terminated": False,
            },
        ]
        and {
            key: diagnosis.get(key)
            for key in (
                "done",
                "before_live",
                "after_live",
                "before_steps",
                "after_steps",
                "reset_calls",
                "full_timeout_recorded",
                "selective_reset_recorded",
            )
        }
        == {
            "done": [True, True],
            "before_live": [False, False],
            "after_live": [True, True],
            "before_steps": [2500, 2500],
            "after_steps": [0, 0],
            "reset_calls": 1,
            "full_timeout_recorded": True,
            "selective_reset_recorded": False,
        }
        and diagnosis.get("collection")
        == {
            "policy_ticks": 250,
            "elapsed_seconds": 28.991513116052374,
            "stop_reason": "first-natural-terminal",
            "failure": None,
        }
        and receipt.get("original_artifacts_modified") is False
        and receipt.get("original_service_restarted") is False
        and receipt.get("recapture_performed") is False
        and receipt.get("optimizer_steps") == 0
        and receipt.get("terminal_reset_qualified") is False
        and receipt.get("acceptance_changed") is False
        and receipt.get("state") == FAILURE_SERVICE_STATE
        and receipt.get("source_unchanged") is True
        and receipt.get("context", {}).get("source_identity", {}).get("source")
        == ARTIFACT_SOURCE
        and all(
            receipt.get(key) is False
            for key in (
                "checkpoint_admitted",
                "complete_binary_runtime_equivalence_verified",
                "execution_admitted",
                "football_balance_accepted",
                "independent_gpu_attestation",
                "learned_stance_accepted",
                "physical_motion_authorized",
                "recovery_accepted",
                "student_export_available",
                "training_admitted",
                "training_update_performed",
            )
        )
        and all(receipt.get(key) is False for key in baseline.FALSE_FLAGS),
        "pinned failed terminal-qpos closeout with immutable collection evidence",
    )
    return True


def _read_failure(root=None, original=None):
    """Read and authenticate the exact retained two-file failure evidence."""
    base = probe.base
    root = base.host.ROOT / FAILURE_DIRECTORY if root is None else Path(root)
    original = base.host.ROOT / ARTIFACT_ROOT if original is None else Path(original)
    require(
        {path.name for path in root.iterdir()} == {"receipt.json", "journal.log"},
        "exact two-file terminal closeout failure evidence",
    )
    receipt_raw = base.files.file_bytes(
        root / "receipt.json", limit=FAILURE_RECEIPT_LIMIT
    )
    journal = base.files.file_bytes(root / "journal.log", limit=FAILURE_JOURNAL_LIMIT)
    require(
        sha256(receipt_raw).hexdigest() == FAILURE_RECEIPT_SHA256
        and sha256(journal).hexdigest() == FAILURE_JOURNAL_SHA256,
        "whole retained terminal closeout failure receipt and journal hashes",
    )
    receipt = base.files.parse(receipt_raw)
    inventory = _original_inventory(original)
    require(
        receipt.get("original_inventory") == inventory,
        "failed closeout binds all five unchanged original files",
    )
    _check_failure_evidence(receipt, journal)
    return receipt


def _failure_binding(receipt, state):
    require(
        state == FAILURE_SERVICE_STATE
        and receipt.get("state") == FAILURE_SERVICE_STATE,
        "exact original failed service state and invocation ID remain unchanged",
    )
    return {
        "source": ARTIFACT_SOURCE,
        "invocation_id": FAILURE_INVOCATION,
        "receipt_sha256": FAILURE_RECEIPT_SHA256,
        "journal_sha256": FAILURE_JOURNAL_SHA256,
        "service_state": state,
        "original_inventory": receipt["original_inventory"],
    }


def _check_failure_linkage(receipt, fresh, launch):
    """Bind retained failure context to today's fresh prerequisite closures."""
    context = {key: launch[key] for key in fresh[0]}
    require(
        receipt.get("context") == context
        and receipt.get("closed_audit_binding") == fresh[3]
        and receipt.get("closed_update_binding") == fresh[4]
        and receipt.get("launch_failure_binding")
        == fresh[0].get("terminal_launch_failure_binding")
        and receipt.get("old_failed_services_unchanged")
        == probe.previous._old_failed_states(),
        "saved closeout failure binds original context, audit/update, and old failures",
    )
    idle = receipt.get("idle")
    require(
        type(idle) is dict
        and idle.get("protocol") == "two-idle-samples-v1"
        and type(idle.get("samples")) is list
        and len(idle["samples"]) == 2,
        "saved failed closeout retains its two idle samples",
    )
    return True


def _old_service_state():
    base = probe.base
    name = f"microduck-cpu-terminal-closeout-{ARTIFACT_SOURCE[:12]}.service"
    state = {
        key: base.host.read("systemctl", "--user", "show", name, "-p", key, "--value")
        for key in FAILURE_SERVICE_STATE
    }
    require(
        state == FAILURE_SERVICE_STATE,
        "original failed terminal closeout service was not restarted",
    )
    return state


def _check_original(root, fresh, evaluator_source, inventory):
    """Authenticate raw bytes first, then bind the safe-loaded trace to its launch."""
    base = probe.base
    raw = {
        name: base.files.file_bytes(root / name, limit=probe.RAW_LIMIT)
        for name in sorted(probe.RUN_FILES)
    }
    require(
        {
            name: {"sha256": sha256(value).hexdigest(), "bytes": len(value)}
            for name, value in raw.items()
        }
        == inventory,
        "original five-file inventory remains identical before decode",
    )
    launch = base.files.parse(raw["launch.json"])
    metadata = base.files.parse(raw["capture.json"])
    report = base.files.parse(raw["report.json"])
    original_context = {key: launch[key] for key in fresh[0]}
    _check_source_context(original_context, fresh[0], evaluator_source)
    original_fresh = (original_context, *fresh[1:])
    require(
        probe._check_launch(launch, ARTIFACT_SOURCE, original_fresh),
        "actual saved launch matches original prerequisites and context",
    )
    require(
        raw["checkpoint.pt"] == fresh[2]
        and sha256(raw["checkpoint.pt"]).hexdigest() == ARTIFACT_PARENT_SHA256,
        "original parent checkpoint bytes match the immutable fresh parent",
    )
    require(
        probe._check_report(report, metadata, inventory, launch, ARTIFACT_SOURCE),
        "original report and capture metadata remain internally bound",
    )
    capture = raw["capture.pt"]
    value = torch.load(io.BytesIO(capture), map_location="cpu", weights_only=True)
    require(
        type(value) is dict
        and value.get("protocol") == trace.PROTOCOL
        and value.get("binding") == launch["trace_binding"]
        and value.get("declaration") == launch["declaration"]
        and value.get("compiled_plant") == launch["compiled_plant"]
        and value.get("collection") == metadata.get("collection"),
        "safe-loaded immutable payload binds exact original launch and metadata",
    )
    return launch, metadata, value


def _check_replay(score):
    require(
        type(score) is dict and score.get("protocol") == trace.PROTOCOL,
        "exact fixed independent terminal scorer protocol",
    )
    decision = probe._score_decision(score)
    require(
        score.get("full_timeout_qualified") is True
        and score.get("timeout_reset_qualified") is True
        and score.get("selective_reset_qualified") is False,
        "saved trace supports common timeout/reset only, not selective reset",
    )
    return decision


def _lease():
    return {
        "lock_path": str(probe.base.files.LOCK),
        "mechanism": "advisory-flock-exclusive-nonblocking",
        "scope": "saved-record-only-audit",
        "idle_gate": "two-idle-samples-before-and-after",
        "cuda_learner": False,
    }


def receipt_result(
    evaluator_source,
    launch_sha256,
    inventory,
    failure,
    source_inventory,
    context_binding,
    replay,
    service,
    elapsed,
    idle_before,
    idle_after,
):
    base = probe.base
    base.files.hex_id(evaluator_source, 40)
    base.files.hex_id(launch_sha256, 64)
    _recorded_properties(service)
    require(
        evaluator_source != ARTIFACT_SOURCE,
        "receipt separates artifact and evaluator sources",
    )
    expected_inventory = {
        name: {"sha256": ARTIFACT_HASHES[name], "bytes": ARTIFACT_BYTES[name]}
        for name in probe.RUN_FILES
    }
    require(
        type(inventory) is dict
        and set(inventory) == probe.RUN_FILES
        and inventory == expected_inventory
        and inventory["launch.json"]["sha256"] == launch_sha256
        and inventory == failure.get("original_inventory"),
        "all five authenticated original artifacts match failure linkage",
    )
    require(
        type(source_inventory) is dict
        and source_inventory.get("artifact_source") == ARTIFACT_SOURCE
        and source_inventory.get("old_probe_sha256") == OLD_PROBE_SHA256
        and source_inventory.get("old_trace_sha256") == OLD_TRACE_SHA256
        and source_inventory.get("fixed_trace_sha256") == FIXED_TRACE_SHA256
        and source_inventory.get("dependencies", {}).get(TRACE_PATH)
        == {
            "artifact_source_sha256": OLD_TRACE_SHA256,
            "evaluator_source_sha256": FIXED_TRACE_SHA256,
        }
        and source_inventory["dependencies"][PROBE_PATH]
        == {
            "artifact_source_sha256": OLD_PROBE_SHA256,
            "evaluator_source_sha256": OLD_PROBE_SHA256,
        }
        and all(
            item["artifact_source_sha256"] == item["evaluator_source_sha256"]
            for path, item in source_inventory["dependencies"].items()
            if path != TRACE_PATH
        ),
        "exact transitive scorer closure with only fixed terminal trace changed",
    )
    require(
        type(context_binding) is dict
        and set(context_binding) == {"original_context", "evaluator_context"}
        and type(context_binding["original_context"]) is dict
        and context_binding["original_context"].get("source_identity", {}).get("source")
        == ARTIFACT_SOURCE
        and context_binding["evaluator_context"]
        .get("source_identity", {})
        .get("source")
        == evaluator_source
        and _check_source_context(
            context_binding["original_context"],
            context_binding["evaluator_context"],
            evaluator_source,
        ),
        "original/evaluator launch context differs only by source identity",
    )
    require(
        type(elapsed) is float
        and math.isfinite(elapsed)
        and 0 < elapsed < SERVICE_SECONDS,
        "finite receipt-repair runtime is inside the 180-second cap",
    )
    require(
        type(idle_before) is dict and type(idle_after) is dict,
        "two real idle-gate samples bracket saved-record replay",
    )
    decision = _check_replay(replay)
    require(
        type(failure) is dict
        and failure.get("source") == ARTIFACT_SOURCE
        and failure.get("invocation_id") == FAILURE_INVOCATION
        and failure.get("receipt_sha256") == FAILURE_RECEIPT_SHA256
        and failure.get("journal_sha256") == FAILURE_JOURNAL_SHA256
        and failure.get("service_state") == FAILURE_SERVICE_STATE,
        "receipt binds the exact immutable original failed invocation",
    )
    result = dict(
        protocol=PROTOCOL,
        artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source,
        launch_sha256=launch_sha256,
        original_inventory=inventory,
        failure=failure,
        source_inventory=source_inventory,
        context_binding=context_binding,
        original_source_context=context_binding["original_context"],
        independent_replay=replay,
        decision=decision,
        original_full_timeout_reset_qualified=True,
        original_selective_reset_qualified=False,
        audit_simulator_resets=0,
        recollection_performed=False,
        optimizer_steps=0,
        training_update_performed=False,
        execution_admitted=False,
        student_export_available=False,
        service_properties=service,
        lease=_lease(),
        idle_before=idle_before,
        idle_after=idle_after,
        elapsed_seconds=elapsed,
        cuda_initialized=False,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False,
        **baseline.FALSE_FLAGS,
    )
    require(
        result["original_full_timeout_reset_qualified"] is True
        and result["original_selective_reset_qualified"] is False
        and result["audit_simulator_resets"] == 0
        and result["recollection_performed"] is False
        and result["optimizer_steps"] == 0
        and result["cuda_initialized"] is False
        and all(result[key] is False for key in baseline.FALSE_FLAGS),
        "non-admitting saved-record-only terminal receipt",
    )
    return result


def _write_exclusive(path, value):
    base = probe.base
    base.files.write_json(path, value)
    raw = base.files.file_bytes(path, limit=2 * 1024**2)
    return {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def audit(evaluator_source):
    """Re-score only the pinned saved capture inside the existing 240s window."""
    base = probe.base
    out = output_path(evaluator_source)
    started = time.monotonic()
    probe.window.check(reserve_seconds=LAUNCH_RESERVE_SECONDS)
    created = False
    partial = dict(
        protocol=PROTOCOL,
        evaluator_source=evaluator_source,
        status="incomplete",
        failure=None,
        optimizer_steps=0,
        audit_simulator_resets=0,
        recollection_performed=False,
        acceptance_changed=False,
        **baseline.FALSE_FLAGS,
    )
    try:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not torch.cuda.is_initialized(),
            "CUDA-hidden CPU-only receipt repair before any audit work",
        )
        with base.files.gpu_lease():
            idle_before = base.host.wait_idle()
            props = service_properties(evaluator_source)
            fresh = probe._fresh(evaluator_source)
            source_hashes = _source_inventory()
            old_root = base.host.ROOT / ARTIFACT_ROOT
            inventory = _original_inventory(old_root)
            failure_receipt = _read_failure(original=old_root)
            failure_state = _old_service_state()
            failure_link = _failure_binding(failure_receipt, failure_state)
            old_launch_raw = base.files.file_bytes(
                old_root / "launch.json", limit=2 * 1024**2
            )
            require(
                sha256(old_launch_raw).hexdigest() == ARTIFACT_LAUNCH_SHA256,
                "whole original launch hash before parse",
            )
            launch, metadata, payload = _check_original(
                old_root, fresh, evaluator_source, inventory
            )
            _check_failure_linkage(failure_receipt, fresh, launch)
            _check_failure_evidence(
                failure_receipt,
                base.files.file_bytes(
                    (base.host.ROOT / FAILURE_DIRECTORY) / "journal.log",
                    limit=FAILURE_JOURNAL_LIMIT,
                ),
            )
            probe.window.check()
            out.mkdir(parents=False, exist_ok=False)
            created = True
            launch_record = dict(
                protocol=PROTOCOL,
                artifact_source=ARTIFACT_SOURCE,
                evaluator_source=evaluator_source,
                original_launch_sha256=ARTIFACT_LAUNCH_SHA256,
                evaluator_context=fresh[0],
                original_source_context={key: launch[key] for key in fresh[0]},
                source_inventory=source_hashes,
                failure=failure_link,
                original_inventory=inventory,
                service_properties=props,
                launch_reserve_seconds=LAUNCH_RESERVE_SECONDS,
                service_seconds=SERVICE_SECONDS,
                memory_bytes=MEMORY_BYTES,
                cpu_quota=CPU_QUOTA,
                nice=NICE,
                kill_mode=KILL_MODE,
                lease=_lease(),
                idle_before=idle_before,
                cuda_initialized=False,
                optimizer_steps=0,
                audit_simulator_resets=0,
                recollection_performed=False,
                **baseline.FALSE_FLAGS,
            )
            _write_exclusive(out / "launch.json", launch_record)
            _write_exclusive(out / "source-inventory.json", source_hashes)
            raw_capture = base.files.file_bytes(
                old_root / "capture.pt", limit=probe.RAW_LIMIT
            )
            score = trace.verify(
                raw_capture,
                inventory["capture.pt"]["sha256"],
                fresh[2],
                launch["declaration"],
                launch["compiled_plant"],
            )
            decision = _check_replay(score)
            require(
                score.get("collection") == metadata.get("collection"),
                "independent replay collection equals immutable original metadata",
            )
            require(
                decision == "cpu-natural-full-timeout-and-reset-replayed",
                "fixed scorer reproduces only the original common timeout/reset decision",
            )
            require(
                type(payload) is dict
                and type(payload.get("payload")) is dict
                and type(payload["payload"].get("initial")) is dict,
                "safe-loaded immutable payload contains its original initial frame",
            )
            require(
                _source_inventory() == source_hashes
                and probe._fresh(evaluator_source) == fresh
                and _original_inventory(old_root) == inventory
                and _read_failure(original=old_root) == failure_receipt
                and _old_service_state() == failure_state
                and service_properties(evaluator_source) == props
                and not torch.cuda.is_initialized(),
                "prerequisites, failure, source closure, original five files and service unchanged",
            )
            idle_after = base.host.wait_idle()
            require(
                time.monotonic() - started < SERVICE_SECONDS,
                "saved-record replay remains inside 180-second service cap",
            )
            require(
                _source_inventory() == source_hashes
                and probe._fresh(evaluator_source) == fresh
                and _original_inventory(old_root) == inventory
                and _read_failure(original=old_root) == failure_receipt
                and _old_service_state() == failure_state
                and service_properties(evaluator_source) == props
                and not torch.cuda.is_initialized(),
                "final idle gate preserves every pinned input and service state",
            )
            probe.window.check()
            context_binding = {
                "original_context": {key: launch[key] for key in fresh[0]},
                "evaluator_context": fresh[0],
            }
            result = receipt_result(
                evaluator_source,
                ARTIFACT_LAUNCH_SHA256,
                inventory,
                failure_link,
                source_hashes,
                context_binding,
                score,
                props,
                float(time.monotonic() - started),
                idle_before,
                idle_after,
            )
            partial.update(result)
            partial["status"] = "complete"
            partial["output_inventory"] = None
            return partial
    except BaseException as error:
        partial["failure"] = {
            "type": type(error).__name__,
            "message": str(error)[:1000],
        }
        raise
    finally:
        if created:
            require(
                {path.name for path in out.iterdir()}
                == {"launch.json", "source-inventory.json"},
                "receipt repair output contains only its two prepared unique files",
            )
            if partial.get("status") == "complete":
                partial["output_inventory"] = {
                    name: {
                        "sha256": sha256((out / name).read_bytes()).hexdigest(),
                        "bytes": (out / name).stat().st_size,
                    }
                    for name in ("launch.json", "source-inventory.json")
                }
                partial["final_inventory_names"] = sorted(REPAIR_FILES)
            _write_exclusive(out / "receipt.json", partial)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args(argv)
    result = audit(args.source)
    print(
        canonical(
            {
                "output": str(output_path(args.source)),
                "artifact_source": ARTIFACT_SOURCE,
                "evaluator_source": args.source,
                "decision": result["decision"],
                "receipt_sha256": probe.base.host.digest(
                    output_path(args.source) / "receipt.json"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
