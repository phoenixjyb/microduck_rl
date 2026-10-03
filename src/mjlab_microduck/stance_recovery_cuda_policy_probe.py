"""Retain a bounded CUDA64 *preparation-only* qualification.

This is deliberately not a trainer.  The two CUDA children construct the exact
D1 actor/critic, empty CUDA rollout storage and fresh empty Adam requested by
``stance_recovery_cuda_policy_preparation``.  They never create an environment,
sample an action, compute returns, step an optimizer, reset or export a policy.
The CPU closeout checks retained tensors and receipts; it is not CUDA math
replay or independent GPU attestation.
"""

import argparse
import io
import json
import math
import os
import select
import subprocess
import sys
import time
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import torch

from mjlab_microduck import foundation_command_campaign as campaign
from mjlab_microduck import gpu_idle_gate, stance_ppo
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_execution_profile as execution
from mjlab_microduck import stance_recovery_broad_receipt_repair as broad_repair
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_gentle_gap_screen as gap
from mjlab_microduck import stance_training_smoke as training_smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cuda64-policy-preparation-probe-v1"
ARTIFACT_SOURCE = "bb7c059d5dda13b3828887fb13c598bb554548cd"
VERIFIER_RECEIPT_SHA256 = (
    "f6e0df22eabd737e047dc290d8202bbfd5d22452cfb868be2381fa13a66c3e76"
)
ARTIFACT_REPORT_SHA256 = (
    "58fdf6863752f7e3e8238d8cebaf7dd2dd8b7d9fc731c1349c59d8b300570e54"
)
ARTIFACT_LAUNCH_SHA256 = (
    "ca2c7cdc5b7a446f096512b7bbde94e2a53a5dfad6aa4afc4633df73679f4628"
)
ARTIFACT_QUALIFICATION_SHA256 = (
    "3e334001e4304bc166b9f45654eae4c2e29f6c9311c2446732f70ef08f9eceb8"
)
ARTIFACT_CLOSEOUT_SHA256 = (
    "b5f58afa0739b7ce6214b45497f441cf04516e4c1b1ddad9399b3bf55af1304e"
)
ARTIFACT_CLOSEOUT_DIRECTORY = "stance-wsl-cpu-gentle-gap-bb7c059d5dda"
VERIFIER_DIRECTORY = "gentle-gap-verified-bb7c059d5dda"
OUTPUT_PREFIX = "stance-wsl-cuda64-policy-preparation-"
SEEDS = (653, 659)
WORLDS, HORIZON = 64, 28
SUPERVISOR_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN_SECONDS = 360, 120, 180, 60
LAUNCH_RESERVE_SECONDS = SUPERVISOR_SECONDS + CLOSEOUT_SECONDS + MARGIN_SECONDS
SUPERVISOR_MEMORY_BYTES, CLOSEOUT_MEMORY_BYTES = 3 * 1024**3, 2 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
LOG_LIMIT, RAW_LIMIT, JSON_LIMIT = 1024**2, 8 * 1024**2, 2 * 1024**2
HISTORICAL_TRACE_LIMIT = gap.evidence.LIMIT
VERIFIER_LOG_LIMIT = 16 * 1024**2
FALSE_FLAGS = dict(baseline.FALSE_FLAGS)
PREPARATION_FALSE_FLAGS = {
    **FALSE_FLAGS,
    "finite_optimizer_step_qualified": False,
    "transition_bridge_qualified": False,
    "schedule_installed": False,
    "student_export_available": False,
    "training_job_predeclared": False,
    "job_predeclared": False,
    "execution_admitted": False,
    "learner_rng_connected_to_sampler": False,
}
GAP_COMPLETE_FILES = gap.COMPLETE_FILES | {"independent-closeout.json"}
PRE_REPORT_FILES = {
    "launch.json",
    "cpu-parent-receipt.json",
    *(f"seed-{seed}.pt" for seed in SEEDS),
    *(f"seed-{seed}.json" for seed in SEEDS),
    *(f"seed-{seed}.log" for seed in SEEDS),
}
COMPLETE_FILES = PRE_REPORT_FILES | {"report.json"}
CLOSEOUT_FILES = COMPLETE_FILES | {"independent-closeout.json"}
STORAGE_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "dones",
    "values",
    "actions_log_prob",
    "returns",
    "advantages",
)
STORAGE_NONE_FIELDS = (
    "distribution_params",
    "saved_hidden_state_a",
    "saved_hidden_state_c",
)


def digest(raw):
    return sha256(raw).hexdigest()


def _hex(value, length, label):
    require(
        type(value) is str
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value),
        "exact " + label,
    )
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse_json(raw):
    require(type(raw) is bytes and 0 < len(raw) <= JSON_LIMIT, "bounded JSON bytes")
    value = json.loads(raw, object_pairs_hook=_unique_pairs)
    canonical(value)
    return value


def output_path(source):
    _hex(source, 40, "source revision")
    require(
        source != ARTIFACT_SOURCE, "new evaluator source, not closed artifact source"
    )
    return host.ROOT / "artifacts/evaluations" / (OUTPUT_PREFIX + source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ("supervise", "closeout"), "declared CUDA preparation service mode")
    name = "run" if mode == "supervise" else mode
    return f"microduck-cuda64-policy-{name}-{source[:12]}.service"


def service_properties(source, mode):
    seconds = {"supervise": SUPERVISOR_SECONDS, "closeout": CLOSEOUT_SECONDS}[mode]
    memory = {"supervise": SUPERVISOR_MEMORY_BYTES, "closeout": CLOSEOUT_MEMORY_BYTES}[
        mode
    ]
    unit = service_name(source, mode)
    props = {
        key: host.read("systemctl", "--user", "show", unit, "-p", key, "--value")
        for key in (
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
    expected = {
        "MainPID": str(os.getpid()),
        "ActiveState": "active",
        "RuntimeMaxUSec": f"{seconds // 60}min",
        "MemoryMax": str(memory),
        "CPUQuotaPerSecUSec": CPU_QUOTA,
        "Nice": NICE,
        "KillMode": KILL_MODE,
    }
    _hex(props.get("InvocationID"), 32, "active service invocation ID")
    require(
        {key: props[key] for key in expected} == expected,
        "exact independently capped CUDA preparation service",
    )
    running = host.read(
        "systemctl",
        "--user",
        "list-units",
        "--state=running",
        "--no-legend",
        "microduck*",
    )
    require(
        {line.split()[0] for line in running.splitlines()} == {unit},
        "only the exact owned Duck service may run",
    )
    return props


def _recorded_service_properties(value, mode):
    seconds = {"supervise": SUPERVISOR_SECONDS, "closeout": CLOSEOUT_SECONDS}[mode]
    memory = {"supervise": SUPERVISOR_MEMORY_BYTES, "closeout": CLOSEOUT_MEMORY_BYTES}[
        mode
    ]
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
        }
        and type(value.get("MainPID")) is str
        and value["MainPID"].isdecimal()
        and int(value["MainPID"]) > 0
        and type(value.get("InvocationID")) is str
        and len(value["InvocationID"]) == 32
        and all(char in "0123456789abcdef" for char in value["InvocationID"])
        and {
            key: value[key]
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
        == {
            "MainPID": value["MainPID"],
            "ActiveState": "active",
            "RuntimeMaxUSec": f"{seconds // 60}min",
            "MemoryMax": str(memory),
            "CPUQuotaPerSecUSec": CPU_QUOTA,
            "Nice": NICE,
            "KillMode": KILL_MODE,
        },
        "exact recorded service properties",
    )
    return True


def _write_exclusive(path, raw, limit):
    require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded retained bytes")
    path = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return digest(raw)


def write_json(path, value):
    return _write_exclusive(path, (canonical(value) + "\n").encode(), JSON_LIMIT)


def _read_file(path, limit):
    path = Path(path)
    mode = path.lstat().st_mode
    require(
        path.is_file() and not path.is_symlink() and mode & 0o170000 == 0o100000,
        "regular non-symlink evidence file",
    )
    size = path.stat().st_size
    require(0 < size <= limit, "bounded evidence file size")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) == size and 0 < len(raw) <= limit, "stable bounded evidence read")
    return raw


def _exact_inventory(root, expected_names):
    require(root.is_dir() and not root.is_symlink(), "existing evidence directory")
    actual = {item.name for item in root.iterdir()}
    require(actual == set(expected_names), "exact retained evidence inventory")
    return actual


def _check_external_verifier(root=None, *, check_live=True):
    """Authenticate the pinned verifier receipt and its six native journals."""
    root = (
        execution.ROOT / "artifacts/tools" / VERIFIER_DIRECTORY
        if root is None
        else Path(root)
    )
    expected_files = {
        "receipt.json",
        *(
            f"{name}.log"
            for name in (
                "run-manager",
                "run-process",
                "tests-manager",
                "tests-process",
                "closeout-manager",
                "closeout-process",
            )
        ),
    }
    _exact_inventory(root, expected_files)
    receipt_raw = _read_file(root / "receipt.json", JSON_LIMIT)
    require(
        digest(receipt_raw) == VERIFIER_RECEIPT_SHA256,
        "pinned external verifier receipt bytes",
    )
    receipt = parse_json(receipt_raw)
    terminal_template = {
        "ActiveState": "inactive",
        "ExecMainStatus": "0",
        "InvocationID": "",
        "MainPID": "0",
        "NRestarts": "0",
        "Result": "success",
    }
    require(
        receipt.get("protocol") == "cpu-gentle-gap-external-retention-v1"
        and receipt.get("source") == ARTIFACT_SOURCE
        and receipt.get("branch") == "feat/athletics-obstacle-curriculum"
        and receipt.get("artifact_file_count") == 68
        and receipt.get("artifact_total_bytes") == 783930274
        and set(receipt.get("artifact_inventory", {})) == GAP_COMPLETE_FILES
        and receipt.get("run_invocation_id") == "f7bbae5169ef4d07b6ae7b999475b334"
        and receipt.get("closeout_invocation_id") == "c5ade3d140a64b4b9d993d0f4168a73e"
        and receipt.get("source_tests_invocation_id")
        == "b3be89514eb84c73afb0d8b967205387"
        and receipt.get("terminal_states")
        == {name: terminal_template for name in ("run", "closeout", "source_tests")}
        and receipt.get("native_tests_passed") == 367
        and receipt.get("retained_closeout_sha256") == ARTIFACT_CLOSEOUT_SHA256
        and receipt.get("artifact_inventory", {}).get("launch.json", {}).get("sha256")
        == ARTIFACT_LAUNCH_SHA256
        and receipt.get("artifact_inventory", {}).get("report.json", {}).get("sha256")
        == ARTIFACT_REPORT_SHA256
        and receipt.get("artifact_inventory", {})
        .get("cpu-qualification.json", {})
        .get("sha256")
        == ARTIFACT_QUALIFICATION_SHA256
        and receipt.get("decision") == "cpu-gentle-gap-no-deficit"
        and receipt.get("cases_checked") == 21
        and receipt.get("optimizer_steps") == receipt.get("simulator_resets") == 0
        and all(receipt.get(key) is False for key in FALSE_FLAGS),
        "exact successful 68-file no-deficit external retention receipt",
    )
    journal_inventory = receipt.get("journal_inventory")
    require(
        type(journal_inventory) is dict
        and set(journal_inventory) == expected_files - {"receipt.json"},
        "exact six-journal verifier inventory",
    )
    for name, item in journal_inventory.items():
        raw = _read_file(root / name, VERIFIER_LOG_LIMIT)
        require(
            type(item) is dict
            and set(item) == {"sha256", "bytes"}
            and item == {"sha256": digest(raw), "bytes": len(raw)},
            "pinned verifier journal bytes: " + name,
        )
    old_units = {
        f"microduck-cpu-gentle-gap-run-{ARTIFACT_SOURCE[:12]}.service": receipt[
            "run_invocation_id"
        ],
        f"microduck-cpu-gentle-gap-closeout-{ARTIFACT_SOURCE[:12]}.service": receipt[
            "closeout_invocation_id"
        ],
        f"microduck-gentle-gap-tests-{ARTIFACT_SOURCE[:12]}.service": receipt[
            "source_tests_invocation_id"
        ],
    }
    if check_live:
        for unit, old_id in old_units.items():
            state = {
                key: host.read(
                    "systemctl", "--user", "show", unit, "-p", key, "--value"
                )
                for key in terminal_template
            }
            require(
                state["InvocationID"] in ("", old_id)
                and {
                    key: value for key, value in state.items() if key != "InvocationID"
                }
                == {
                    key: value
                    for key, value in terminal_template.items()
                    if key != "InvocationID"
                },
                "old successful invocation remains terminal; only InvocationID GC is allowed",
            )
    return {
        "receipt_sha256": VERIFIER_RECEIPT_SHA256,
        "receipt": receipt,
        "journal_inventory": journal_inventory,
    }


def _authenticate_gap_artifact(source):
    """Rehash the real 68-file native inventory, then run old read-only gates."""
    require(source == ARTIFACT_SOURCE, "exact completed gentle-gap source")
    root = gap.output_path(source)
    verified = _check_external_verifier()
    receipt = verified["receipt"]
    require(
        receipt.get("artifact_inventory")
        == {
            name: {"sha256": item["sha256"], "bytes": item["bytes"]}
            for name, item in receipt["artifact_inventory"].items()
        },
        "typed complete native artifact inventory",
    )
    _exact_inventory(root, GAP_COMPLETE_FILES)
    inventory = {}
    parsed = {}
    for name, expected in receipt["artifact_inventory"].items():
        raw = _read_file(
            root / name, HISTORICAL_TRACE_LIMIT if name.endswith(".pt") else JSON_LIMIT
        )
        require(
            type(expected) is dict
            and set(expected) == {"sha256", "bytes"}
            and expected == {"sha256": digest(raw), "bytes": len(raw)},
            "rehash complete native gentle-gap artifact: " + name,
        )
        inventory[name] = deepcopy(expected)
        if name.endswith(".json"):
            parsed[name] = parse_json(raw)
    require(
        inventory["launch.json"]["sha256"] == ARTIFACT_LAUNCH_SHA256
        and inventory["report.json"]["sha256"] == ARTIFACT_REPORT_SHA256
        and inventory["cpu-qualification.json"]["sha256"]
        == ARTIFACT_QUALIFICATION_SHA256
        and inventory["checkpoint.pt"]["sha256"] == baseline.CHECKPOINT_SHA256
        and inventory["report.json"]["bytes"] == 8697
        and sum(item["bytes"] for item in inventory.values()) == 783930274,
        "pinned whole 68-file native inventory and byte count",
    )
    parent_bytes = _read_file(root / "checkpoint.pt", checkpoint.LIMIT)
    launch = parsed["launch.json"]
    qualification = parsed["cpu-qualification.json"]
    report = parsed["report.json"]
    closeout = parsed["independent-closeout.json"]
    require(
        launch.get("protocol") == gap.PROTOCOL
        and launch.get("source") == source
        and launch.get("parent_checkpoint_sha256") == baseline.CHECKPOINT_SHA256
        and launch.get("optimizer_steps") == launch.get("simulator_resets") == 0
        and all(launch.get(key) is False for key in FALSE_FLAGS)
        and qualification.get("protocol") == gap.PROTOCOL
        and qualification.get("source") == source
        and len(qualification.get("captures", [])) == len(gap.CELL_IDS) == 21
        and len(qualification.get("scores", [])) == len(gap.CELL_IDS)
        and len(qualification.get("prefixes", [])) == len(gap.CELL_IDS)
        and qualification.get("optimizer_steps")
        == qualification.get("simulator_resets")
        == 0
        and all(qualification.get(key) is False for key in FALSE_FLAGS)
        and set(report.get("files", {})) == gap.PREPARE_FILES,
        "pinned saved gap launch, qualification and exact matrix schema",
    )
    close_raw = _read_file(root / "independent-closeout.json", JSON_LIMIT)
    require(
        digest(close_raw) == ARTIFACT_CLOSEOUT_SHA256
        and closeout.get("protocol") == gap.PROTOCOL
        and closeout.get("source") == source
        and closeout.get("whole_cpu_rescore_identical") is True
        and closeout.get("cases_checked") == 21
        and closeout.get("files_rehashed")
        == {name: inventory[name] for name in gap.COMPLETE_FILES}
        and all(closeout.get(key) is False for key in FALSE_FLAGS)
        and report.get("decision") == "cpu-gentle-gap-no-deficit"
        and report.get("screening", {}).get("complete_cases") == 21
        and report.get("screening", {}).get("candidate_deficit_cells") == []
        and report.get("screening", {}).get("prefixes_identical") is True
        and report.get("screening", {}).get("zero_control_valid") is True
        and report.get("screening", {}).get("complete_declared_force_phases") is True
        and report.get("optimizer_steps") == report.get("simulator_resets") == 0
        and all(report.get(key) is False for key in FALSE_FLAGS),
        "native 21-case full-completion no-deficit result with no updates/resets",
    )
    declarations = gap.declarations(source)
    for index, (cell, declaration) in enumerate(zip(gap.CELL_IDS, declarations)):
        capture = parsed[f"case-{index}.json"]
        replay = parsed[f"case-{index}-replay.json"]
        raw_info = inventory[f"case-{index}.pt"]
        expected_capture = qualification["captures"][index]
        score = qualification["scores"][index]
        require(
            capture == expected_capture
            and capture.get("source") == source
            and capture.get("index") == index
            and capture.get("cell") == cell
            and capture.get("declaration") == declaration
            and capture.get("capture_bytes") == raw_info["bytes"]
            and capture.get("capture_sha256") == raw_info["sha256"]
            and capture.get("optimizer_steps") == capture.get("simulator_resets") == 0
            and replay == score
            and replay.get("collection") == capture.get("collection"),
            "saved raw metadata, replay score and qualification align for case "
            + str(index),
        )
    screening = gap._screen_inputs(qualification["scores"], qualification["prefixes"])
    require(
        screening
        == qualification["screening"]
        == report["screening"]
        == closeout["screening"],
        "deterministic saved-score replay reproduces every declared gap gate",
    )
    gap_context_keys = (
        "source_identity",
        "cpu_math_profile",
        "preserved_filmbrain",
        "protected_services",
        "terminal_launch_failure_binding",
    )
    return {
        "inventory": inventory,
        "launch": launch,
        "launch_sha256": ARTIFACT_LAUNCH_SHA256,
        "report_sha256": ARTIFACT_REPORT_SHA256,
        "qualification_sha256": ARTIFACT_QUALIFICATION_SHA256,
        "independent_closeout_sha256": ARTIFACT_CLOSEOUT_SHA256,
        "checkpoint_sha256": digest(parent_bytes),
        "screening": deepcopy(report["screening"]),
        "run_context": {key: deepcopy(launch[key]) for key in gap_context_keys}
        | {"terminal_prerequisites": deepcopy(launch["terminal_prerequisites"])},
        "external_verifier": verified,
    }


def _source_binding(source):
    """Require clean exact source and byte-for-byte unchanged bb7 gap leaves."""
    _hex(source, 40, "source revision")
    repo = Path(gap.__file__).resolve().parents[2]
    relative_paths = (
        "src/mjlab_microduck/stance_recovery_gentle_gap_screen.py",
        "tests/test_stance_recovery_gentle_gap_screen.py",
    )
    current = {}
    for relative in relative_paths:
        old = subprocess.run(
            ["git", "show", f"{ARTIFACT_SOURCE}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        current_raw = (repo / relative).read_bytes()
        require(
            current_raw == old, "immutable closed gentle-gap source leaf: " + relative
        )
        current[relative] = digest(current_raw)
    own_paths = (
        "src/mjlab_microduck/foundation_command_campaign.py",
        "src/mjlab_microduck/gpu_idle_gate.py",
        "src/mjlab_microduck/stance_checkpoint.py",
        "src/mjlab_microduck/stance_cuda_probe.py",
        "src/mjlab_microduck/stance_execution_profile.py",
        "src/mjlab_microduck/stance_ppo.py",
        "src/mjlab_microduck/stance_recovery_broad_receipt_repair.py",
        "src/mjlab_microduck/stance_recovery_campaign_window.py",
        "src/mjlab_microduck/stance_recovery_contract.py",
        "src/mjlab_microduck/stance_recovery_cuda_policy_preparation.py",
        "src/mjlab_microduck/stance_recovery_gentle_gap_screen.py",
        "src/mjlab_microduck/stance_recovery_parent.py",
        "tests/test_stance_recovery_cuda_policy_preparation.py",
        "src/mjlab_microduck/stance_recovery_cuda_policy_probe.py",
        "tests/test_stance_recovery_cuda_policy_probe.py",
        "src/mjlab_microduck/stance_training_smoke.py",
    )
    git_head = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=5,
        )
        .stdout.decode()
        .strip()
    )
    branch = (
        subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=5,
        )
        .stdout.decode()
        .strip()
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        timeout=5,
    ).stdout
    require(
        git_head == source
        and branch == "feat/athletics-obstacle-curriculum"
        and not status,
        "clean exact committed evaluator source",
    )
    pinned = {}
    for relative in own_paths:
        raw = subprocess.run(
            ["git", "show", f"{source}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        require(
            raw == (repo / relative).read_bytes(),
            "exact committed evaluator source leaf",
        )
        pinned[relative] = digest(raw)
    return {
        "source": source,
        "branch": branch,
        "immutable_gap_leaves": current,
        "evaluator_leaves": pinned,
    }


def _native_prerequisites(source):
    source_record = _source_binding(source)
    gap_record = _authenticate_gap_artifact(ARTIFACT_SOURCE)
    current_context = gap._context(source)
    # Authenticate historical parent/update/terminal/failure/source-chain state.
    terminal = gap._terminal_prerequisites(source)
    original_context = gap._closed_parent_context()
    current_record = gap._context_record(
        source, current_context, original_context, terminal
    )
    prior_gap_context = gap_record["run_context"]
    require(
        prior_gap_context["source_identity"].get("source") == ARTIFACT_SOURCE,
        "closed gap run context uses its exact evaluator source",
    )
    broad_repair._check_source_context(
        prior_gap_context,
        current_record,
        artifact_source=ARTIFACT_SOURCE,
        evaluator_source=source,
    )
    return {
        "source": source_record,
        "gap": gap_record,
        "current_context": current_record,
        "terminal_prerequisites": terminal,
        "closed_parent_context": original_context,
        "raw_parent": terminal["fresh"][2],
        "launch_binding": {
            "source": source_record,
            "gap_inventory": gap_record["inventory"],
            "gap_receipts": {
                key: gap_record[key]
                for key in (
                    "launch_sha256",
                    "report_sha256",
                    "qualification_sha256",
                    "independent_closeout_sha256",
                    "checkpoint_sha256",
                    "screening",
                )
            },
            "current_context": current_record,
        },
    }


def _cpu_parent_prepare(raw_parent):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CPU parent must be loaded in an initially CUDA-hidden supervisor",
    )
    from mjlab_microduck import stance_recovery_parent as parent

    loaded = parent.load_parent(raw_parent)
    receipt = loaded["receipt"]
    receipt_raw = (canonical(receipt) + "\n").encode()
    return (
        loaded,
        receipt_raw,
        digest(receipt_raw),
        preparation._cpu_parent_receipt_digest(receipt),
    )


def _make_launch(
    source,
    prerequisites,
    cpu_receipt_sha,
    canonical_cpu_sha,
    service,
    idle_before,
    start_unix,
):
    root = output_path(source)
    receipt = prerequisites["raw_parent"]
    require(
        digest(receipt) == baseline.CHECKPOINT_SHA256, "exact immutable D1 parent bytes"
    )
    binding = preparation.cpu_parent_binding(source, canonical_cpu_sha)
    return {
        "protocol": PROTOCOL,
        "source": source,
        "output_name": root.name,
        "native_prerequisites": _json_safe(prerequisites["launch_binding"]),
        "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
        "parent_state_sha256": preparation.parent.PARENT_STATE_SHA256,
        "parent_identity": preparation.parent.expected_identity(),
        "cpu_parent_receipt_file_sha256": cpu_receipt_sha,
        "cpu_parent_receipt_canonical_sha256": canonical_cpu_sha,
        "cpu_parent_binding": binding,
        "learner_seeds": list(SEEDS),
        "worlds": WORLDS,
        "horizon": HORIZON,
        "preparation_protocol": preparation.PROTOCOL,
        "supervisor_service_seconds": SUPERVISOR_SECONDS,
        "child_timeout_seconds": CHILD_SECONDS,
        "closeout_service_seconds": CLOSEOUT_SECONDS,
        "launch_reserve_seconds": LAUNCH_RESERVE_SECONDS,
        "margin_seconds": MARGIN_SECONDS,
        "raw_bytes_limit": RAW_LIMIT,
        "log_bytes_limit": LOG_LIMIT,
        "supervisor_memory_bytes": SUPERVISOR_MEMORY_BYTES,
        "closeout_memory_bytes": CLOSEOUT_MEMORY_BYTES,
        "cpu_quota": CPU_QUOTA,
        "nice": NICE,
        "kill_mode": KILL_MODE,
        "service_properties": service,
        "idle_before": idle_before,
        "started_unix": start_unix,
        "sequential_children": True,
        "cpu_parent_provenance_authenticated_by_supervisor": True,
        "cuda_child_claims_cpu_provenance_authenticated": False,
        "action_sampling": False,
        "return_computation": False,
        "rollout_collection": False,
        "simulator_resets": 0,
        "optimizer_steps": 0,
        "training_updates": 0,
        "student_exports": 0,
        "campaign_window": window.declaration(),
        **PREPARATION_FALSE_FLAGS,
    }


def _json_safe(value):
    if type(value) is bytes:
        return {"bytes": len(value), "sha256": digest(value)}
    if type(value) is dict:
        return {key: _json_safe(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_safe(item) for item in value]
    return value


def _storage_snapshot(storage):
    result = {
        "observations": {
            key: value.detach().cpu().contiguous().clone()
            for key, value in storage.observations.items()
        }
    }
    for name in STORAGE_FIELDS[1:]:
        value = getattr(storage, name)
        result[name] = value.detach().cpu().contiguous().clone()
    for name in STORAGE_NONE_FIELDS:
        value = getattr(storage, name)
        require(
            value is None, "fresh rollout storage auxiliary field is empty: " + name
        )
        result[name] = value
    return result


def _optimizer_parameter_names(actor, critic, optimizer):
    named = {}
    for group, model in (("actor", actor), ("critic", critic)):
        for name, parameter in model.named_parameters():
            require(
                id(parameter) not in named, "unique actor/critic optimizer parameters"
            )
            named[id(parameter)] = f"{group}.{name}"
    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    require(
        len(owned) == len(set(map(id, owned))) and set(map(id, owned)) == set(named),
        "optimizer owns the exact unique actor/critic parameter union",
    )
    return [named[id(parameter)] for parameter in owned]


def _child_payload(prepared, launch_sha256):
    actor, critic = prepared["actor"], prepared["critic"]
    algorithm, storage = prepared["algorithm"], prepared["storage"]
    receipt = deepcopy(prepared["receipt"])
    states = checkpoint.states_of(actor, critic)
    storage_copy = _storage_snapshot(storage)
    rng = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in prepared["caller_rng_states"].items()
    }
    private_state = (
        prepared["private_cuda_rng_state"].detach().cpu().contiguous().clone()
    )
    require(
        all(value.device.type == "cpu" for value in (*rng.values(), private_state)),
        "retained caller and private RNG states are CPU snapshots",
    )
    optimizer_state = deepcopy(algorithm.optimizer.state_dict())
    parameter_names = _optimizer_parameter_names(actor, critic, algorithm.optimizer)
    metadata = {
        "protocol": PROTOCOL + ":child-v1",
        "source": receipt["source"],
        "launch_sha256": launch_sha256,
        "seed": receipt["learner_seed"],
        "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"],
        "parent_state_sha256": receipt["parent_state_sha256"],
        "state_sha256": checkpoint.state_hash(states),
        "actor_critic_state_schema": {
            group: {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in group_states.items()
            }
            for group, group_states in states.items()
        },
        "actor_parameter_names": [name for name, _ in actor.named_parameters()],
        "critic_parameter_names": [name for name, _ in critic.named_parameters()],
        "optimizer_parameter_names": parameter_names,
        "optimizer_parameter_count": len(parameter_names),
        "storage_step": storage.step,
        "storage_fields": list(STORAGE_FIELDS),
        "storage_is_empty": True,
        "algorithm_is_stock_ppo": type(algorithm).__name__ == "PPO",
        "optimizer_is_actual_adam": type(algorithm.optimizer) is torch.optim.Adam,
        "optimizer_state_is_empty": not algorithm.optimizer.state,
        "private_cuda_rng_state_sha256": digest(private_state.numpy().tobytes()),
        "caller_rng_state_sha256": {
            key: digest(value.numpy().tobytes()) for key, value in rng.items()
        },
        "raw_preparation_receipt": receipt,
    }
    raw_payload = io.BytesIO()
    torch.save(
        {
            "metadata": metadata,
            "actor_critic_states": states,
            "optimizer_state_dict": optimizer_state,
            "storage": storage_copy,
            "caller_rng_states": rng,
            "private_cuda_rng_state": private_state,
        },
        raw_payload,
    )
    raw = raw_payload.getvalue()
    require(0 < len(raw) <= RAW_LIMIT, "bounded CPU snapshot payload")
    return raw, metadata


def _current_launch(source, launch_sha256):
    _hex(launch_sha256, 64, "whole launch SHA256")
    root = output_path(source)
    raw = _read_file(root / "launch.json", JSON_LIMIT)
    require(digest(raw) == launch_sha256, "whole launch bytes before parse")
    launch = parse_json(raw)
    require(
        launch.get("protocol") == PROTOCOL and launch.get("source") == source,
        "source-bound CUDA preparation launch",
    )
    return root, launch, raw


def _visible_child_context(source, launch):
    """Read context without calling the CUDA-hidden legacy context wrapper."""
    expected = launch.get("native_prerequisites", {}).get("current_context")
    require(
        type(expected) is dict, "CPU supervisor recorded full preallocation context"
    )
    current = {
        "source_identity": host.identity(source),
        "cpu_math_profile": gap.base.profile.checked_receipt(),
        "preserved_filmbrain": gap.base.retained.d0.filmbrain_state(),
        "protected_services": gap.base.protected_state(),
    }
    require(
        all(expected.get(key) == value for key, value in current.items()),
        "visible child observes exact launched source/CPU profile/FilmBrain/protected context",
    )
    require(
        all(value == "inactive" for value in current["protected_services"].values()),
        "protected AI Mission services remain inactive",
    )
    return current


def _child(source, launch_sha256, seed, lease_fd):
    # This must precede every CUDA availability/context query or allocation.
    training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "CUDA0 child visibility from process start",
    )
    root, launch, _, cpu_receipt = _check_launch(source, launch_sha256)
    child_context = _visible_child_context(source, launch)
    require(seed in SEEDS and type(seed) is int, "one fixed sequential learner seed")
    receipt_raw = _read_file(root / "cpu-parent-receipt.json", JSON_LIMIT)
    require(
        digest(receipt_raw) == launch["cpu_parent_receipt_file_sha256"],
        "whole supervisor-authenticated CPU receipt file",
    )
    require(parse_json(receipt_raw) == cpu_receipt, "unchanged validated CPU receipt")
    canonical_sha = preparation._cpu_parent_receipt_digest(cpu_receipt)
    require(
        canonical_sha == launch["cpu_parent_receipt_canonical_sha256"],
        "separate canonical CPU parent receipt digest",
    )
    binding = preparation.cpu_parent_binding(source, canonical_sha)
    require(
        canonical(binding) == canonical(launch["cpu_parent_binding"]),
        "source-bound CPU parent binding",
    )
    parent_raw = _read_file(
        gap.output_path(ARTIFACT_SOURCE) / "checkpoint.pt", checkpoint.LIMIT
    )
    require(
        digest(parent_raw) == baseline.CHECKPOINT_SHA256,
        "hash-first unchanged D1 parent bytes",
    )
    prepared = preparation.prepare_policy(
        parent_raw,
        source=source,
        lease_fd=lease_fd,
        cpu_parent_receipt=cpu_receipt,
        cpu_parent_receipt_sha256=canonical_sha,
        caller_binding=binding,
        seed=seed,
        worlds=WORLDS,
    )
    require(
        _visible_child_context(source, launch) == child_context,
        "source/FilmBrain/protected context unchanged across preparation",
    )
    raw, meta = _child_payload(prepared, launch_sha256)
    require(
        meta["source"] == source
        and meta["seed"] == seed
        and meta["state_sha256"] == preparation.parent.PARENT_STATE_SHA256
        and meta["raw_preparation_receipt"][
            "native_cpu_provenance_authenticated_by_this_function"
        ]
        is False,
        "exact non-admitting prepared D1 child payload",
    )
    raw_name = root / f"seed-{seed}.pt"
    payload_sha = _write_exclusive(raw_name, raw, RAW_LIMIT)
    summary = {
        "protocol": PROTOCOL + ":child-summary-v1",
        "source": source,
        "launch_sha256": launch_sha256,
        "seed": seed,
        "payload_sha256": payload_sha,
        "payload_bytes": len(raw),
        "metadata": meta,
        "caller_rng_state_sha256": meta["caller_rng_state_sha256"],
        "private_cuda_rng_state_sha256": meta["private_cuda_rng_state_sha256"],
        "cpu_parent_receipt_file_sha256": launch["cpu_parent_receipt_file_sha256"],
        "cpu_parent_receipt_canonical_sha256": canonical_sha,
        "cuda_child_claims_cpu_provenance_authenticated": False,
        "action_sampling": False,
        "return_computation": False,
        "rollout_collection": False,
        "optimizer_steps": 0,
        "simulator_resets": 0,
        "training_update_performed": False,
        **PREPARATION_FALSE_FLAGS,
    }
    write_json(root / f"seed-{seed}.json", summary)
    return summary


def _run_child(source, launch_sha, seed, fd, env, root):
    log_path = root / f"seed-{seed}.log"
    with log_path.open("xb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mjlab_microduck.stance_recovery_cuda_policy_probe",
                "child",
                "--source",
                source,
                "--launch-sha256",
                launch_sha,
                "--seed",
                str(seed),
                "--lease-fd",
                str(fd),
            ],
            cwd=execution.ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(fd,),
        )
        deadline = time.monotonic() + CHILD_SECONDS
        child_started = time.monotonic()
        next_monitor = time.monotonic()
        total = 0
        gpu_samples = []
        pipe_open = True
        try:
            while proc.poll() is None or pipe_open:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("CUDA child exceeded fixed 120-second timeout")
                readable, _, _ = select.select(
                    [proc.stdout] if pipe_open else [], [], [], min(0.25, remaining)
                )
                if readable:
                    chunk = os.read(
                        proc.stdout.fileno(), min(64 * 1024, LOG_LIMIT - total + 1)
                    )
                    if not chunk:
                        pipe_open = False
                    else:
                        allowed = max(0, LOG_LIMIT - total)
                        if allowed:
                            log.write(chunk[:allowed])
                            total += min(allowed, len(chunk))
                            log.flush()
                        if len(chunk) > allowed:
                            raise ValueError(
                                "CUDA child streaming log byte cap exceeded"
                            )
                if time.monotonic() >= next_monitor and proc.poll() is None:
                    gpu_samples.append(
                        {
                            "elapsed_seconds": float(time.monotonic() - child_started),
                            "child_pid": proc.pid,
                            "sample": campaign.live_gpu(proc.pid),
                        }
                    )
                    next_monitor = time.monotonic() + 2.0
            require(proc.returncode == 0, "CUDA preparation child exited successfully")
            require(
                total <= LOG_LIMIT and log_path.stat().st_size == total,
                "CUDA child streaming log remains under byte cap",
            )
        except BaseException:
            campaign._kill_owned_group(proc)
            raise
        finally:
            proc.stdout.close()
            log.flush()
            os.fsync(log.fileno())
    summary = parse_json(_read_file(root / f"seed-{seed}.json", JSON_LIMIT))
    require(gpu_samples, "at least one bounded live GPU ownership sample per child")
    return {**summary, "live_gpu_samples": gpu_samples}


def _context_after(source, launched):
    current = gap._context(source)
    require(
        current == {key: launched[key] for key in current},
        "unchanged current source/profile/FilmBrain/services context",
    )
    return current


def _partial_inventory(root):
    inventory = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "report.json":
            continue
        limit = (
            RAW_LIMIT
            if path.suffix == ".pt"
            else LOG_LIMIT
            if path.suffix == ".log"
            else JSON_LIMIT
        )
        size = path.stat().st_size
        if size > limit:
            inventory[path.name] = {"bytes": size, "oversized": True}
            continue
        try:
            raw = _read_file(path, limit)
        except (OSError, ValueError):
            inventory[path.name] = {"bytes": size, "readable": False}
            continue
        inventory[path.name] = {"sha256": digest(raw), "bytes": len(raw)}
    return inventory


def _complete_inventory(root, names):
    _exact_inventory(root, names)
    inventory = {}
    for name in sorted(names):
        limit = (
            RAW_LIMIT
            if name.endswith(".pt")
            else LOG_LIMIT
            if name.endswith(".log")
            else JSON_LIMIT
        )
        raw = _read_file(root / name, limit)
        inventory[name] = {"sha256": digest(raw), "bytes": len(raw)}
    return inventory


def _check_live_gpu_samples(children):
    require(
        type(children) is list and len(children) == len(SEEDS),
        "two retained child summaries with live GPU samples",
    )
    expected_services = {
        name for name, _ in execution.service_commands(campaign.SERVICES)
    }
    for seed, child in zip(SEEDS, children):
        require(type(child) is dict, "typed retained child summary")
        samples = child.get("live_gpu_samples")
        require(
            child.get("seed") == seed
            and type(samples) is list
            and 0 < len(samples) <= CHILD_SECONDS // 2 + 2,
            "bounded live GPU sample sequence for fixed child seed",
        )
        previous_elapsed = -1.0
        child_pid = samples[0].get("child_pid") if type(samples[0]) is dict else None
        for row in samples:
            require(
                type(row) is dict
                and set(row) == {"elapsed_seconds", "child_pid", "sample"}
                and type(row["elapsed_seconds"]) is float
                and math.isfinite(row["elapsed_seconds"])
                and previous_elapsed < row["elapsed_seconds"] < CHILD_SECONDS
                and type(row["child_pid"]) is int
                and row["child_pid"] > 0,
                "ordered bounded live GPU sample envelope",
            )
            previous_elapsed = row["elapsed_seconds"]
            require(
                row["child_pid"] == child_pid,
                "one actual child PID throughout monitoring",
            )
            sample = row["sample"]
            require(
                type(sample) is dict
                and set(sample)
                in (
                    {"services", "compute_pids", "temperature_c"},
                    {
                        "services",
                        "compute_pids",
                        "temperature_c",
                        "memory_used_mib",
                        "memory_free_mib",
                    },
                )
                and type(sample["services"]) is dict
                and set(sample["services"]) == expected_services
                and all(value == "inactive" for value in sample["services"].values())
                and type(sample["compute_pids"]) is list
                and all(type(pid) is int for pid in sample["compute_pids"])
                and set(sample["compute_pids"]) <= {row["child_pid"]}
                and type(sample["temperature_c"]) is int
                and 0 <= sample["temperature_c"] < execution.PROFILE["temperature_c"],
                "retained live GPU ownership and temperature sample",
            )
            if execution.PROFILE.get("name") == execution.WSL:
                require(
                    set(sample)
                    == {
                        "services",
                        "compute_pids",
                        "temperature_c",
                        "memory_used_mib",
                        "memory_free_mib",
                    }
                    and type(sample["memory_used_mib"]) is int
                    and 0 <= sample["memory_used_mib"] <= 5120
                    and type(sample["memory_free_mib"]) is int
                    and sample["memory_free_mib"] >= 6144,
                    "retained WSL GPU memory reserve sample",
                )
    return True


def supervise(source):
    """Run the two fixed preparation children inside the predeclared service."""
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden supervisor before child environment creation",
    )
    started = time.monotonic()
    window.check(reserve_seconds=LAUNCH_RESERVE_SECONDS)
    require(execution.PROFILE.get("name") == execution.WSL, "exact frozen WSL profile")
    service = service_properties(source, "supervise")
    root = output_path(source)
    raw_parent = None
    report = {
        "protocol": PROTOCOL,
        "source": source,
        "status": "preparation-failed-retained",
        "optimizer_steps": 0,
        "training_update_performed": False,
        "simulator_resets": 0,
        **PREPARATION_FALSE_FLAGS,
    }
    root.mkdir(parents=True, exist_ok=False)
    try:
        with gap.base.files.gpu_lease() as lease_fd:
            idle_before = gpu_idle_gate.wait_idle()
            prerequisites = _native_prerequisites(source)
            raw_parent = prerequisites["raw_parent"]
            loaded, receipt_raw, receipt_sha, canonical_sha = _cpu_parent_prepare(
                raw_parent
            )
            require(
                digest(raw_parent) == baseline.CHECKPOINT_SHA256,
                "CPU supervisor loaded immutable D1 parent",
            )
            _write_exclusive(root / "cpu-parent-receipt.json", receipt_raw, JSON_LIMIT)
            del loaded
            launch = _make_launch(
                source,
                prerequisites,
                receipt_sha,
                canonical_sha,
                service,
                idle_before,
                time.time(),
            )
            launch_sha = write_json(root / "launch.json", launch)
            report.update(
                launch_sha256=launch_sha,
                decision="preparation-children-pending",
                idle_before=idle_before,
            )
            env = {
                **campaign.child_environment(),
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "ATEN_CPU_CAPABILITY": "default",
                "MKL_CBWR": "COMPATIBLE",
                "PYTHONUNBUFFERED": "1",
            }
            env["PATH"] = (
                str(execution.ROOT / ".venv/bin") + ":/usr/local/bin:/usr/bin:/bin"
            )
            summaries = []
            for index, seed in enumerate(SEEDS):
                remaining_children = len(SEEDS) - index
                require(
                    time.monotonic()
                    - started
                    + remaining_children * CHILD_SECONDS
                    + MARGIN_SECONDS
                    < SUPERVISOR_SECONDS,
                    "full remaining child budget and run-retention margin fit service cap",
                )
                window.check(
                    reserve_seconds=(remaining_children * CHILD_SECONDS)
                    + CLOSEOUT_SECONDS
                    + MARGIN_SECONDS
                )
                if index:
                    gpu_idle_gate.wait_idle()
                summaries.append(
                    _run_child(source, launch_sha, seed, lease_fd, env, root)
                )
                gpu_idle_gate.wait_idle()
            require(
                service_properties(source, "supervise") == service,
                "exact run service invocation and limits unchanged through both children",
            )
            _context_after(source, prerequisites["current_context"])
            final_prerequisites = _native_prerequisites(source)
            require(
                final_prerequisites == prerequisites,
                "authenticated native prerequisites unchanged",
            )
            idle_after = gpu_idle_gate.wait_idle()
            window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
            require(
                service_properties(source, "supervise") == service,
                "supervisor service remains exact after final idle sample",
            )
            _context_after(source, prerequisites["current_context"])
            final_prerequisites = _native_prerequisites(source)
            require(
                final_prerequisites == prerequisites,
                "source and authenticated native inputs remain exact after final idle sample",
            )
            require(
                digest(raw_parent) == baseline.CHECKPOINT_SHA256
                and digest(
                    _read_file(
                        gap.output_path(ARTIFACT_SOURCE) / "checkpoint.pt",
                        checkpoint.LIMIT,
                    )
                )
                == baseline.CHECKPOINT_SHA256,
                "immutable CPU parent bytes remain exact after final idle sample",
            )
            saved_cpu_receipt = _read_file(root / "cpu-parent-receipt.json", JSON_LIMIT)
            require(
                digest(saved_cpu_receipt) == receipt_sha
                and preparation._cpu_parent_receipt_digest(
                    parse_json(saved_cpu_receipt)
                )
                == canonical_sha,
                "whole and canonical CPU parent receipt remain exact after final idle sample",
            )
            pre_report_inventory = _complete_inventory(root, PRE_REPORT_FILES)
            require(
                pre_report_inventory["launch.json"]["sha256"] == launch_sha
                and pre_report_inventory["cpu-parent-receipt.json"]["sha256"]
                == receipt_sha,
                "exclusive launch and CPU receipt bytes remain unchanged",
            )
            require(
                time.monotonic() - started < SUPERVISOR_SECONDS,
                "bounded 360-second preparation supervisor",
            )
            report.update(
                status="prepared-non-admitting",
                decision="cuda64-policy-prepared-awaiting-cpu-closeout",
                children=summaries,
                idle_after=idle_after,
                pre_report_files=pre_report_inventory,
                elapsed_seconds=float(time.monotonic() - started),
                source_unchanged=True,
                prerequisites_unchanged=True,
                parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
                parent_state_sha256=preparation.parent.PARENT_STATE_SHA256,
                service_properties=service,
            )
    except BaseException as error:
        report.update(
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=float(time.monotonic() - started),
        )
        raise
    finally:
        report["files"] = _partial_inventory(root)
        write_json(root / "report.json", report)
    return report


def _validate_launch_record(source, launch):
    require(
        type(launch) is dict
        and launch.get("protocol") == PROTOCOL
        and launch.get("source") == source
        and launch.get("output_name") == output_path(source).name
        and launch.get("preparation_protocol") == preparation.PROTOCOL
        and launch.get("parent_checkpoint_sha256") == baseline.CHECKPOINT_SHA256
        and launch.get("parent_state_sha256") == preparation.parent.PARENT_STATE_SHA256
        and launch.get("parent_identity") == preparation.parent.expected_identity()
        and launch.get("learner_seeds") == list(SEEDS)
        and launch.get("worlds") == WORLDS
        and launch.get("horizon") == HORIZON
        and launch.get("supervisor_service_seconds") == SUPERVISOR_SECONDS
        and launch.get("child_timeout_seconds") == CHILD_SECONDS
        and launch.get("closeout_service_seconds") == CLOSEOUT_SECONDS
        and launch.get("launch_reserve_seconds") == LAUNCH_RESERVE_SECONDS
        and launch.get("margin_seconds") == MARGIN_SECONDS
        and launch.get("supervisor_memory_bytes") == SUPERVISOR_MEMORY_BYTES
        and launch.get("closeout_memory_bytes") == CLOSEOUT_MEMORY_BYTES
        and launch.get("cpu_quota") == CPU_QUOTA
        and launch.get("nice") == NICE
        and launch.get("kill_mode") == KILL_MODE
        and launch.get("raw_bytes_limit") == RAW_LIMIT
        and launch.get("log_bytes_limit") == LOG_LIMIT
        and launch.get("sequential_children") is True
        and launch.get("cpu_parent_provenance_authenticated_by_supervisor") is True
        and launch.get("cuda_child_claims_cpu_provenance_authenticated") is False
        and launch.get("optimizer_steps") == launch.get("training_updates") == 0
        and launch.get("simulator_resets") == launch.get("student_exports") == 0
        and launch.get("action_sampling") is False
        and launch.get("return_computation") is False
        and launch.get("rollout_collection") is False
        and launch.get("campaign_window") == window.declaration()
        and all(launch.get(key) is False for key in PREPARATION_FALSE_FLAGS),
        "exact fixed two-seed preparation-only launch",
    )
    _recorded_service_properties(launch.get("service_properties"), "supervise")
    native = launch.get("native_prerequisites")
    require(
        type(native) is dict
        and set(native)
        == {"source", "gap_inventory", "gap_receipts", "current_context"}
        and native["source"].get("source") == source
        and native["current_context"].get("source_identity", {}).get("source")
        == source,
        "compact JSON-only source/context prerequisite binding",
    )
    binding = launch.get("cpu_parent_binding")
    require(
        type(binding) is dict
        and binding
        == preparation.cpu_parent_binding(
            source, launch.get("cpu_parent_receipt_canonical_sha256")
        ),
        "source-bound separate canonical CPU-parent hash",
    )
    return True


def _check_launch(source, launch_sha):
    root, launch, launch_raw = _current_launch(source, launch_sha)
    _validate_launch_record(source, launch)
    receipt_raw = _read_file(root / "cpu-parent-receipt.json", JSON_LIMIT)
    require(
        digest(receipt_raw) == launch.get("cpu_parent_receipt_file_sha256"),
        "whole actual CPU preparation receipt bytes",
    )
    receipt = parse_json(receipt_raw)
    canonical_sha = preparation._cpu_parent_receipt_digest(receipt)
    require(
        canonical_sha == launch.get("cpu_parent_receipt_canonical_sha256"),
        "canonical CPU preparation receipt digest",
    )
    preparation.validate_cpu_parent_receipt(
        receipt, canonical_sha, launch.get("cpu_parent_binding"), source=source
    )
    _hex(launch_sha, 64, "whole launch SHA256")
    return root, launch, launch_raw, receipt


def _finite_zero(tensor, shape, dtype, label):
    require(
        torch.is_tensor(tensor)
        and tensor.device.type == "cpu"
        and tuple(tensor.shape) == tuple(shape)
        and tensor.dtype == dtype
        and (tensor.dtype == torch.uint8 or torch.isfinite(tensor).all())
        and not torch.count_nonzero(tensor),
        "exact finite zero CPU snapshot " + label,
    )


def _check_storage(storage):
    require(
        type(storage) is dict
        and set(storage) == set(STORAGE_FIELDS) | set(STORAGE_NONE_FIELDS),
        "exact retained storage schema",
    )
    require(
        all(storage[name] is None for name in STORAGE_NONE_FIELDS),
        "rollout storage distribution and hidden-state fields remain empty",
    )
    require(
        type(storage["observations"]) is dict
        and set(storage["observations"]) == {"actor", "critic"},
        "actor/critic observations",
    )
    _finite_zero(
        storage["observations"]["actor"],
        (HORIZON, WORLDS, 44),
        torch.float32,
        "actor observations",
    )
    _finite_zero(
        storage["observations"]["critic"],
        (HORIZON, WORLDS, 50),
        torch.float32,
        "critic observations",
    )
    expected = {
        "actions": ((HORIZON, WORLDS, 10), torch.float32),
        "rewards": ((HORIZON, WORLDS, 1), torch.float32),
        "dones": ((HORIZON, WORLDS, 1), torch.uint8),
        "values": ((HORIZON, WORLDS, 1), torch.float32),
        "actions_log_prob": ((HORIZON, WORLDS, 1), torch.float32),
        "returns": ((HORIZON, WORLDS, 1), torch.float32),
        "advantages": ((HORIZON, WORLDS, 1), torch.float32),
    }
    for name, (shape, dtype) in expected.items():
        _finite_zero(storage[name], shape, dtype, name)
    return True


def _check_caller_rng(rng, metadata, child_receipt):
    require(
        type(rng) is dict
        and set(rng) == {"cpu_before", "cpu_after", "cuda_before", "cuda_after"}
        and all(
            torch.is_tensor(item)
            and item.device.type == "cpu"
            and item.dtype == torch.uint8
            and item.ndim == 1
            and item.numel() > 0
            for item in rng.values()
        ),
        "four raw CPU/CUDA caller RNG byte states",
    )
    hashes = {name: digest(value.numpy().tobytes()) for name, value in rng.items()}
    require(
        hashes
        == metadata.get("caller_rng_state_sha256")
        == child_receipt.get("caller_rng_state_sha256")
        and torch.equal(rng["cpu_before"], rng["cpu_after"])
        and torch.equal(rng["cuda_before"], rng["cuda_after"])
        and child_receipt.get("caller_cpu_rng_unchanged") is True
        and child_receipt.get("caller_cuda_rng_unchanged") is True,
        "whole before/after caller RNG evidence and preservation",
    )
    return hashes


def _check_private_rng(private_state, metadata, child_receipt, summary):
    require(
        torch.is_tensor(private_state)
        and private_state.device.type == "cpu"
        and private_state.dtype == torch.uint8
        and private_state.ndim == 1
        and private_state.numel() > 0,
        "separately retained private CUDA generator bytes",
    )
    private_sha = digest(private_state.numpy().tobytes())
    require(
        private_sha
        == metadata.get("private_cuda_rng_state_sha256")
        == child_receipt.get("private_cuda_rng_state_sha256")
        == summary.get("private_cuda_rng_state_sha256")
        and child_receipt.get("private_cuda_generator_device") == "cuda:0"
        and child_receipt.get("private_cuda_rng_connected_to_sampler") is False,
        "private generator state remains separately retained and unconnected",
    )
    return private_sha


def _check_optimizer_state(state, metadata, actor_states, critic_states):
    require(
        type(state) is dict
        and set(state) == {"state", "param_groups"}
        and state["state"] == {}
        and type(state["param_groups"]) is list
        and len(state["param_groups"]) == 1,
        "fresh empty Adam state dictionary",
    )
    group = state["param_groups"][0]
    names = [
        group_name + "." + name
        for group_name, values in (
            ("actor", metadata.get("actor_parameter_names")),
            ("critic", metadata.get("critic_parameter_names")),
        )
        for name in (values if type(values) is list else [])
    ]
    owned_names = metadata.get("optimizer_parameter_names")
    require(
        type(owned_names) is list
        and len(owned_names) == len(set(owned_names))
        and set(owned_names) == set(names)
        and metadata.get("optimizer_parameter_count") == len(names),
        "actual optimizer parameter ownership is exact unique actor/critic union",
    )
    require(
        all(name in actor_states for name in metadata["actor_parameter_names"])
        and all(name in critic_states for name in metadata["critic_parameter_names"]),
        "optimizer named parameters correspond to retained model state tensors",
    )
    require(
        type(group.get("params")) is list
        and len(group["params"]) == len(names)
        and all(type(value) is int for value in group["params"])
        and len(group["params"]) == len(set(group["params"])),
        "Adam state dictionary owns unique parameter identifiers",
    )
    for key, value in stance_ppo.CONFIG.items():
        if key == "optimizer":
            require(value == "adam", "exact stock Adam selection")
        elif key == "learning_rate":
            require(group.get("lr") == value, "exact declared Adam learning rate")
    require(
        group.get("betas") == (0.9, 0.999)
        and group.get("eps") == 1e-8
        and group.get("weight_decay") == 0,
        "exact stock Adam defaults",
    )
    return True


def _score_payload(raw, summary, launch, launch_sha, receipt, seed):
    require(
        type(raw) is bytes
        and 0 < len(raw) <= RAW_LIMIT
        and digest(raw) == summary.get("payload_sha256")
        and len(raw) == summary.get("payload_bytes"),
        "whole raw child payload before tensor load",
    )
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    require(
        type(payload) is dict
        and set(payload)
        == {
            "metadata",
            "actor_critic_states",
            "optimizer_state_dict",
            "storage",
            "caller_rng_states",
            "private_cuda_rng_state",
        },
        "exact CPU-loadable payload schema",
    )
    metadata = payload["metadata"]
    child_receipt = metadata.get("raw_preparation_receipt")
    child_context = launch.get("native_prerequisites", {}).get("current_context", {})
    expected_child_receipt_keys = {
        "protocol",
        "source",
        "source_identity",
        "parent_checkpoint_sha256",
        "parent_identity",
        "parent_state_sha256",
        "cpu_parent_receipt_sha256",
        "caller_binding",
        "cpu_parent_receipt",
        "cpu_provenance_authentication_required",
        "native_cpu_provenance_authenticated_by_this_function",
        "learner_seed",
        "worlds",
        "horizon",
        "device",
        "actor_device",
        "critic_device",
        "state_sha256_before_transfer",
        "state_sha256_after_transfer",
        "actor_parameter_tensors",
        "critic_parameter_tensors",
        "ppo_config",
        "ppo_source_sha256",
        "storage_step",
        "storage_empty",
        "optimizer_type",
        "optimizer_state_entries",
        "optimizer_empty",
        "private_cuda_generator_device",
        "private_cuda_rng_state_sha256",
        "private_cuda_rng_connected_to_sampler",
        "caller_cpu_rng_unchanged",
        "caller_cuda_rng_unchanged",
        "caller_rng_state_sha256",
        "cuda_initialized",
        "simulator_created",
        "rollout_collected",
        "optimizer_steps",
        "training_update_performed",
        "finite_optimizer_step_qualified",
        "transition_bridge_qualified",
        "schedule_installed",
        "student_export_available",
        "training_job_predeclared",
        "execution_admitted",
    } | set(FALSE_FLAGS)
    require(
        type(metadata) is dict
        and type(child_receipt) is dict
        and set(child_receipt) == expected_child_receipt_keys
        and child_receipt.get("protocol") == preparation.PROTOCOL
        and metadata.get("protocol") == PROTOCOL + ":child-v1"
        and metadata.get("source") == launch["source"] == summary.get("source")
        and metadata.get("launch_sha256") == launch_sha
        and metadata.get("seed") == seed == summary.get("seed")
        and metadata.get("parent_checkpoint_sha256") == baseline.CHECKPOINT_SHA256
        and metadata.get("parent_state_sha256")
        == preparation.parent.PARENT_STATE_SHA256
        and child_receipt.get("source") == launch["source"]
        and child_receipt.get("learner_seed") == seed
        and child_receipt.get("worlds") == WORLDS
        and child_receipt.get("horizon") == HORIZON
        and child_receipt.get("device") == "cuda:0"
        and child_receipt.get("cuda_initialized") is True
        and child_receipt.get("actor_device")
        == child_receipt.get("critic_device")
        == "cuda:0"
        and child_receipt.get("parent_checkpoint_sha256") == baseline.CHECKPOINT_SHA256
        and child_receipt.get("parent_state_sha256")
        == preparation.parent.PARENT_STATE_SHA256
        and child_receipt.get("parent_identity")
        == preparation.parent.expected_identity()
        and child_receipt.get("source_identity") == child_context.get("source_identity")
        and child_receipt.get("cpu_parent_receipt_sha256")
        == launch["cpu_parent_receipt_canonical_sha256"]
        and child_receipt.get("cpu_parent_receipt") == receipt
        and receipt.get("cpu_math_profile") == child_context.get("cpu_math_profile")
        and child_receipt.get("native_cpu_provenance_authenticated_by_this_function")
        is False
        and child_receipt.get("cpu_provenance_authentication_required") is True
        and child_receipt.get("optimizer_state_entries") == 0
        and child_receipt.get("optimizer_empty") is True
        and child_receipt.get("storage_step") == 0
        and child_receipt.get("storage_empty") is True
        and child_receipt.get("simulator_created") is False
        and child_receipt.get("rollout_collected") is False
        and child_receipt.get("finite_optimizer_step_qualified") is False
        and child_receipt.get("transition_bridge_qualified") is False
        and child_receipt.get("schedule_installed") is False
        and child_receipt.get("student_export_available") is False
        and child_receipt.get("training_job_predeclared") is False
        and child_receipt.get("execution_admitted") is False
        and child_receipt.get("private_cuda_rng_connected_to_sampler") is False
        and child_receipt.get("training_update_performed") is False
        and child_receipt.get("optimizer_steps") == 0
        and child_receipt.get("state_sha256_before_transfer")
        == child_receipt.get("state_sha256_after_transfer")
        == preparation.parent.PARENT_STATE_SHA256
        and child_receipt.get("actor_parameter_tensors", 0) > 0
        and child_receipt.get("critic_parameter_tensors", 0) > 0
        and all(child_receipt.get(key) is False for key in FALSE_FLAGS),
        "exact source/launch/seed/CPU receipt-bound non-admitting child receipt",
    )
    require(
        summary.get("launch_sha256") == launch_sha
        and summary.get("protocol") == PROTOCOL + ":child-summary-v1"
        and summary.get("cpu_parent_receipt_file_sha256")
        == launch["cpu_parent_receipt_file_sha256"]
        and summary.get("cpu_parent_receipt_canonical_sha256")
        == launch["cpu_parent_receipt_canonical_sha256"]
        and summary.get("metadata") == metadata
        and summary.get("optimizer_steps") == summary.get("simulator_resets") == 0
        and summary.get("training_update_performed") is False
        and summary.get("cuda_child_claims_cpu_provenance_authenticated") is False
        and summary.get("action_sampling") is False
        and summary.get("return_computation") is False
        and summary.get("rollout_collection") is False
        and all(summary.get(key) is False for key in PREPARATION_FALSE_FLAGS),
        "retained child summary exactly matches raw metadata",
    )
    states = payload["actor_critic_states"]
    require(
        type(states) is dict
        and set(states) == {"actor", "critic"}
        and all(type(states[group]) is dict for group in states),
        "actor/critic snapshot groups",
    )
    require(
        checkpoint.state_hash(states) == preparation.parent.PARENT_STATE_SHA256
        and metadata.get("state_sha256") == preparation.parent.PARENT_STATE_SHA256,
        "actual retained transferred weights remain exact frozen parent",
    )
    for values in states.values():
        require(
            values
            and all(
                torch.is_tensor(value)
                and value.device.type == "cpu"
                and value.dtype == torch.float32
                and torch.isfinite(value).all()
                for value in values.values()
            ),
            "finite CPU snapshot model tensors",
        )
    expected_schema = {
        group: {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in values.items()
        }
        for group, values in states.items()
    }
    require(
        metadata.get("actor_critic_state_schema") == expected_schema,
        "model tensor labels agree with CPU snapshots",
    )
    expected_actor, expected_critic = checkpoint.validate_identity(
        preparation.parent.expected_identity(), evaluation="lean-replication"
    )
    checkpoint.validate_states(states, expected_actor, expected_critic)
    require(
        metadata.get("actor_parameter_names")
        == [name for name, _ in expected_actor.named_parameters()]
        and metadata.get("critic_parameter_names")
        == [name for name, _ in expected_critic.named_parameters()],
        "complete exact parent parameter names, not a metadata-selected subset",
    )
    _check_optimizer_state(
        payload["optimizer_state_dict"], metadata, states["actor"], states["critic"]
    )
    _check_storage(payload["storage"])
    require(
        metadata.get("storage_fields") == list(STORAGE_FIELDS)
        and metadata.get("storage_step") == 0
        and metadata.get("storage_is_empty") is True
        and metadata.get("algorithm_is_stock_ppo") is True
        and metadata.get("optimizer_is_actual_adam") is True
        and metadata.get("optimizer_state_is_empty") is True,
        "actual preparation and zero-storage metadata",
    )
    _check_caller_rng(payload["caller_rng_states"], metadata, child_receipt)
    _check_private_rng(
        payload["private_cuda_rng_state"], metadata, child_receipt, summary
    )
    require(
        child_receipt.get("caller_binding") == launch["cpu_parent_binding"],
        "child uses exact supervisor CPU-parent source binding",
    )
    require(
        child_receipt.get("ppo_config") == stance_ppo.CONFIG
        and child_receipt.get("ppo_source_sha256") == stance_ppo.PINS
        and child_receipt.get("optimizer_type") == "Adam"
        and child_receipt.get("actor_parameter_tensors", 0) > 0
        and child_receipt.get("critic_parameter_tensors", 0) > 0
        and child_receipt["actor_parameter_tensors"]
        + child_receipt["critic_parameter_tensors"]
        == metadata["optimizer_parameter_count"]
        and type(metadata.get("actor_parameter_names")) is list
        and type(metadata.get("critic_parameter_names")) is list
        and len(metadata["actor_parameter_names"])
        == child_receipt["actor_parameter_tensors"]
        and len(metadata["critic_parameter_names"])
        == child_receipt["critic_parameter_tensors"],
        "exact frozen PPO pins/config, parameter counts and pre/post transfer parent hashes",
    )
    return {
        "seed": seed,
        "payload_sha256": digest(raw),
        "state_sha256": checkpoint.state_hash(states),
        "storage_zero": True,
        "optimizer_empty": True,
        "caller_rng_unchanged": True,
        "private_rng_unconnected": True,
        "cuda_math_replayed": False,
        **PREPARATION_FALSE_FLAGS,
    }


def closeout_result(
    source,
    launch_sha,
    report_sha,
    inventory,
    scores,
    service,
    run_invocation_id,
    elapsed,
    idle_before,
    idle_after,
):
    _hex(source, 40, "source revision")
    _hex(launch_sha, 64, "launch digest")
    _hex(report_sha, 64, "report digest")
    require(
        type(inventory) is dict
        and set(inventory) == COMPLETE_FILES
        and type(scores) is list
        and [item["seed"] for item in scores] == list(SEEDS)
        and type(elapsed) is float
        and math.isfinite(elapsed)
        and 0 < elapsed < CLOSEOUT_SECONDS
        and type(idle_before) is dict
        and type(idle_after) is dict,
        "typed independent CPU closeout inputs",
    )
    _recorded_service_properties(service, "closeout")
    _hex(run_invocation_id, 32, "retained run service invocation ID")
    return {
        "protocol": PROTOCOL + ":independent-closeout-v1",
        "source": source,
        "launch_sha256": launch_sha,
        "report_sha256": report_sha,
        "files_rehashed": inventory,
        "scores": scores,
        "whole_cpu_payload_rescore_identical": True,
        "cuda_math_replayed": False,
        "service_properties": service,
        "run_service_invocation_id": run_invocation_id,
        "elapsed_seconds": elapsed,
        "idle_before": idle_before,
        "idle_after": idle_after,
        "optimizer_steps": 0,
        "training_update_performed": False,
        "simulator_resets": 0,
        "action_sampling": False,
        "return_computation": False,
        "rollout_collection": False,
        **PREPARATION_FALSE_FLAGS,
    }


def _completed_run_service(source, invocation_id):
    _hex(invocation_id, 32, "original run invocation ID")
    state = {
        key: host.read(
            "systemctl",
            "--user",
            "show",
            service_name(source, "supervise"),
            "-p",
            key,
            "--value",
        )
        for key in (
            "MainPID",
            "ActiveState",
            "NRestarts",
            "ExecMainStatus",
            "Result",
            "InvocationID",
        )
    }
    require(
        state["InvocationID"] in ("", invocation_id)
        and {key: value for key, value in state.items() if key != "InvocationID"}
        == {
            "MainPID": "0",
            "ActiveState": "inactive",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "Result": "success",
        },
        "original owned run is successfully terminal before independent closeout",
    )
    return state


def closeout(source, launch_sha256):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "independent CPU closeout remains CUDA hidden",
    )
    started = time.monotonic()
    window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
    service = service_properties(source, "closeout")
    root, launch, _, cpu_receipt = _check_launch(source, launch_sha256)
    completed_run = _completed_run_service(
        source, launch["service_properties"]["InvocationID"]
    )
    with gap.base.files.gpu_lease():
        idle_before = gpu_idle_gate.wait_idle()
        prerequisites = _native_prerequisites(source)
        require(
            _json_safe(prerequisites["launch_binding"])
            == launch["native_prerequisites"],
            "current source/history prerequisites unchanged",
        )
        rng_before = torch.random.get_rng_state().clone()
        fresh_parent, fresh_receipt_raw, fresh_receipt_sha, fresh_canonical_sha = (
            _cpu_parent_prepare(prerequisites["raw_parent"])
        )
        require(
            fresh_receipt_sha == launch["cpu_parent_receipt_file_sha256"]
            and fresh_canonical_sha == launch["cpu_parent_receipt_canonical_sha256"]
            and fresh_receipt_raw
            == _read_file(root / "cpu-parent-receipt.json", JSON_LIMIT)
            and fresh_parent["receipt"] == cpu_receipt
            and torch.equal(rng_before, torch.random.get_rng_state()),
            "fresh hidden-CPU parent loader receipt, source bytes and caller RNG reauthenticated",
        )
        del fresh_parent
        report_raw = _read_file(root / "report.json", JSON_LIMIT)
        report = parse_json(report_raw)
        require(
            report.get("protocol") == PROTOCOL
            and report.get("source") == source
            and report.get("launch_sha256") == launch_sha256
            and report.get("status") == "prepared-non-admitting"
            and report.get("decision") == "cuda64-policy-prepared-awaiting-cpu-closeout"
            and report.get("optimizer_steps") == 0
            and report.get("training_update_performed") is False
            and report.get("simulator_resets") == 0
            and all(report.get(key) is False for key in FALSE_FLAGS),
            "successful retained preparation report, no admission",
        )
        _check_live_gpu_samples(report.get("children"))
        require(
            type(report.get("pre_report_files")) is dict
            and report.get("pre_report_files") == report.get("files"),
            "exact complete eight-file pre-report inventory is retained",
        )
        _exact_inventory(root, COMPLETE_FILES)
        inventory = {}
        for name in sorted(COMPLETE_FILES):
            raw = _read_file(
                root / name,
                RAW_LIMIT
                if name.endswith(".pt")
                else LOG_LIMIT
                if name.endswith(".log")
                else JSON_LIMIT,
            )
            inventory[name] = {"sha256": digest(raw), "bytes": len(raw)}
        require(
            report.get("files")
            == {
                name: item for name, item in inventory.items() if name != "report.json"
            },
            "report binds every complete evidence file",
        )
        scores = []
        for seed in SEEDS:
            raw = _read_file(root / f"seed-{seed}.pt", RAW_LIMIT)
            summary_raw = _read_file(root / f"seed-{seed}.json", JSON_LIMIT)
            summary = parse_json(summary_raw)
            reported_child = report["children"][len(scores)]
            require(
                {
                    key: value
                    for key, value in reported_child.items()
                    if key != "live_gpu_samples"
                }
                == summary,
                "report child summary matches exclusive per-seed summary bytes",
            )
            scores.append(
                _score_payload(raw, summary, launch, launch_sha256, cpu_receipt, seed)
            )
        idle_after = gpu_idle_gate.wait_idle()
        require(
            not torch.cuda.is_initialized()
            and time.monotonic() - started < CLOSEOUT_SECONDS,
            "CUDA-hidden bounded CPU-only closeout",
        )
        window.check(reserve_seconds=MARGIN_SECONDS)
        final_prerequisites = _native_prerequisites(source)
        require(
            _json_safe(final_prerequisites["launch_binding"])
            == launch["native_prerequisites"],
            "fresh authenticated source/history context after final idle sample",
        )
        rng_after_idle = torch.random.get_rng_state().clone()
        final_parent, final_receipt_raw, final_receipt_sha, final_canonical_sha = (
            _cpu_parent_prepare(final_prerequisites["raw_parent"])
        )
        require(
            final_receipt_sha == launch["cpu_parent_receipt_file_sha256"]
            and final_canonical_sha == launch["cpu_parent_receipt_canonical_sha256"]
            and final_receipt_raw
            == _read_file(root / "cpu-parent-receipt.json", JSON_LIMIT)
            and final_parent["receipt"] == cpu_receipt
            and torch.equal(rng_after_idle, torch.random.get_rng_state()),
            "fresh hidden CPU parent and receipt reauthenticated after final idle sample",
        )
        del final_parent
        require(
            service_properties(source, "closeout") == service,
            "exact independent-closeout service invocation and limits remain unchanged",
        )
        final_inventory = _complete_inventory(root, COMPLETE_FILES)
        require(
            final_inventory == inventory,
            "whole retained evidence remains unchanged after final idle sample",
        )
        result = closeout_result(
            source,
            launch_sha256,
            digest(report_raw),
            final_inventory,
            scores,
            service,
            launch["service_properties"]["InvocationID"],
            float(time.monotonic() - started),
            idle_before,
            idle_after,
        )
        require(
            _completed_run_service(source, launch["service_properties"]["InvocationID"])
            == completed_run,
            "original run terminal state unchanged during closeout",
        )
        result["run_service_terminal_state"] = completed_run
        window.check(reserve_seconds=MARGIN_SECONDS)
        write_json(root / "independent-closeout.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "child", "closeout"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--launch-sha256")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args(argv)
    if args.mode == "supervise":
        result = supervise(args.source)
    elif args.mode == "child":
        require(
            args.launch_sha256 is not None
            and args.seed is not None
            and args.lease_fd is not None,
            "child requires launch, seed and inherited lease",
        )
        result = _child(args.source, args.launch_sha256, args.seed, args.lease_fd)
    else:
        require(args.launch_sha256 is not None, "closeout requires exact launch digest")
        result = closeout(args.source, args.launch_sha256)
    if args.mode == "child":
        output = {
            key: result[key]
            for key in (
                "protocol",
                "source",
                "seed",
                "launch_sha256",
                "payload_sha256",
                "payload_bytes",
            )
        }
    else:
        output = {
            key: result[key]
            for key in ("protocol", "source", "status", "decision", "elapsed_seconds")
            if key in result
        }
    print(canonical(output), flush=True)


if __name__ == "__main__":
    main()
