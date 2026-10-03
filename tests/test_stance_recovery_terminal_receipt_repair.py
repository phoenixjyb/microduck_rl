"""Pure receipt contracts plus optional checks over retained authentic failure bytes."""

from hashlib import sha256
from pathlib import Path

import pytest

from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_terminal_probe as probe
from mjlab_microduck import stance_recovery_terminal_receipt_repair as repair

EVALUATOR = "a" * 40


def inventory():
    return {
        name: {
            "sha256": repair.ARTIFACT_HASHES[name],
            "bytes": repair.ARTIFACT_BYTES[name],
        }
        for name in probe.RUN_FILES
    }


def context(source, profile="pinned"):
    return {
        "source_identity": {"source": source, "branch": "clean", "revision": "current"},
        "cpu_math_profile": {"profile": profile},
        "preserved_filmbrain": {"generation": 4},
        "protected_services": {"worker": "inactive", "vllm": "inactive"},
        "terminal_launch_failure_binding": {"invocation_id": repair.FAILURE_INVOCATION},
    }


def service():
    return {
        "MainPID": "1234",
        "ActiveState": "active",
        "RuntimeMaxUSec": "3min",
        "MemoryMax": str(repair.MEMORY_BYTES),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
    }


def replay():
    return {
        "protocol": repair.probe.trace.PROTOCOL,
        "collection": {
            "policy_ticks": 250,
            "elapsed_seconds": 28.991513116052374,
            "stop_reason": "first-natural-terminal",
            "failure": None,
        },
        "validated_policy_ticks": 250,
        "full_timeout_qualified": True,
        "timeout_reset_qualified": True,
        "selective_reset_qualified": False,
        "complete_force_phase_checks": True,
        "force": {
            "checked_physics_steps": 2500,
            "window_steps_per_row": [10, 20],
            "delivered_nonzero_steps_per_row": [0, 20],
        },
        **{
            key: True
            for key in (
                "exact_policy_replay",
                "private_rng",
                "caller_rng",
                "recorded_force_prefix_checked",
                "terminal_critic_bootstrap_exact",
                "compiled_plant_checked",
                "control_trace_checked",
                "physical_trace_checked",
                "collection_cap_respected",
            )
        },
        "whole_trajectory_physics_resimulated": False,
        "thermal_model_applied": False,
        **baseline.FALSE_FLAGS,
    }


def failure():
    return {
        "source": repair.ARTIFACT_SOURCE,
        "invocation_id": repair.FAILURE_INVOCATION,
        "receipt_sha256": repair.FAILURE_RECEIPT_SHA256,
        "journal_sha256": repair.FAILURE_JOURNAL_SHA256,
        "service_state": repair.FAILURE_SERVICE_STATE,
        "original_inventory": inventory(),
    }


def source_inventory():
    return repair._source_inventory()


def args(**overrides):
    value = {
        "evaluator_source": EVALUATOR,
        "launch_sha256": repair.ARTIFACT_LAUNCH_SHA256,
        "inventory": inventory(),
        "failure": failure(),
        "source_inventory": source_inventory(),
        "context_binding": {
            "original_context": context(repair.ARTIFACT_SOURCE),
            "evaluator_context": context(EVALUATOR),
        },
        "replay": replay(),
        "service": service(),
        "elapsed": 30.0,
        "idle_before": {"idle": True},
        "idle_after": {"idle": True},
    }
    value.update(overrides)
    return value


def test_distinct_namespace_and_exact_service_limits():
    assert repair.output_path(EVALUATOR).name.endswith(EVALUATOR[:12])
    assert repair.service_name(EVALUATOR) == (
        f"microduck-cpu-terminal-receipt-repair-{EVALUATOR[:12]}.service"
    )
    assert repair.SERVICE_SECONDS == 180 and repair.LAUNCH_RESERVE_SECONDS == 240
    assert repair.MEMORY_BYTES == 2 * 1024**3
    assert repair.CPU_QUOTA == "2s" and repair.NICE == "10"
    assert repair.KILL_MODE == "control-group"
    with pytest.raises(ValueError):
        repair.output_path(repair.ARTIFACT_SOURCE)
    with pytest.raises(ValueError):
        repair.output_path("bad")


@pytest.mark.parametrize(
    "key,value",
    [
        ("MainPID", "0"),
        ("ActiveState", "failed"),
        ("RuntimeMaxUSec", "4min"),
        ("MemoryMax", "infinity"),
        ("CPUQuotaPerSecUSec", "4s"),
        ("Nice", "0"),
        ("KillMode", "process"),
    ],
)
def test_recorded_service_caps_reject_changes(key, value):
    changed = service() | {key: value}
    with pytest.raises(ValueError):
        repair._recorded_properties(changed)


def test_source_context_allows_only_source_identity_change():
    assert repair._check_source_context(
        context(repair.ARTIFACT_SOURCE), context(EVALUATOR), EVALUATOR
    )
    with pytest.raises(ValueError):
        repair._check_source_context(
            context(repair.ARTIFACT_SOURCE), context(EVALUATOR, "changed"), EVALUATOR
        )
    with pytest.raises(ValueError):
        repair._check_source_context(context("b" * 40), context(EVALUATOR), EVALUATOR)


def test_source_closure_allows_only_fixed_trace_leaf():
    result = source_inventory()
    assert result["artifact_source"] == repair.ARTIFACT_SOURCE
    assert result["dependencies"][repair.PROBE_PATH] == {
        "artifact_source_sha256": repair.OLD_PROBE_SHA256,
        "evaluator_source_sha256": repair.OLD_PROBE_SHA256,
    }
    assert result["dependencies"][repair.TRACE_PATH] == {
        "artifact_source_sha256": repair.OLD_TRACE_SHA256,
        "evaluator_source_sha256": repair.FIXED_TRACE_SHA256,
    }
    assert all(
        item["artifact_source_sha256"] == item["evaluator_source_sha256"]
        for path, item in result["dependencies"].items()
        if path != repair.TRACE_PATH
    )


def test_receipt_requires_common_timeout_and_never_selective_reset():
    result = repair.receipt_result(**args())
    assert result["protocol"] == repair.PROTOCOL
    assert result["artifact_source"] == repair.ARTIFACT_SOURCE
    assert result["evaluator_source"] == EVALUATOR
    assert result["decision"] == "cpu-natural-full-timeout-and-reset-replayed"
    assert result["original_full_timeout_reset_qualified"] is True
    assert result["original_selective_reset_qualified"] is False
    assert result["audit_simulator_resets"] == result["optimizer_steps"] == 0
    assert result["recollection_performed"] is False
    assert all(result[key] is False for key in baseline.FALSE_FLAGS)


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "profile",
        "inventory",
        "selective",
        "caps",
        "elapsed",
        "failure",
        "forged_inventory",
        "failed_state",
    ],
)
def test_receipt_rejects_misbound_or_incomplete_inputs(damage):
    values = args()
    if damage == "source":
        values["evaluator_source"] = repair.ARTIFACT_SOURCE
    elif damage == "profile":
        values["context_binding"]["evaluator_context"]["cpu_math_profile"][
            "profile"
        ] = "changed"
    elif damage == "inventory":
        values["inventory"].pop("capture.pt")
    elif damage == "selective":
        values["replay"]["selective_reset_qualified"] = True
    elif damage == "caps":
        values["service"]["RuntimeMaxUSec"] = "4min"
    elif damage == "elapsed":
        values["elapsed"] = 180.0
    elif damage == "forged_inventory":
        values["inventory"]["capture.pt"]["sha256"] = "f" * 64
        values["failure"]["original_inventory"] = values["inventory"]
    elif damage == "failed_state":
        values["failure"]["service_state"] = repair.FAILURE_SERVICE_STATE | {
            "NRestarts": "1"
        }
    else:
        values["failure"]["invocation_id"] = "0" * 32
    with pytest.raises(ValueError):
        repair.receipt_result(**values)


def test_original_inventory_reads_whole_bytes_before_any_decode(tmp_path, monkeypatch):
    for name in probe.RUN_FILES:
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(
        repair, "ARTIFACT_BYTES", {name: len(name.encode()) for name in probe.RUN_FILES}
    )
    monkeypatch.setattr(
        repair,
        "ARTIFACT_HASHES",
        {name: sha256(name.encode()).hexdigest() for name in probe.RUN_FILES},
    )
    got = repair._original_inventory(tmp_path)
    assert set(got) == probe.RUN_FILES
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError):
        repair._original_inventory(tmp_path)


def test_owned_json_write_is_canonical_fsynced_and_exclusive(tmp_path):
    target = tmp_path / "receipt.json"
    result = repair._write_exclusive(target, {"z": 2, "a": 1})
    assert target.read_bytes() == (repair.canonical({"a": 1, "z": 2}) + "\n").encode()
    assert result == {
        "sha256": sha256(target.read_bytes()).hexdigest(),
        "bytes": len(target.read_bytes()),
    }
    with pytest.raises(FileExistsError):
        repair._write_exclusive(target, {"a": 1, "z": 2})


def actual_failure_roots():
    repo = Path(__file__).resolve().parents[1]
    native = repo / repair.FAILURE_DIRECTORY
    original = repo / repair.ARTIFACT_ROOT
    if native.is_dir() and original.is_dir():
        return native, original
    mirror = repo / "artifacts/retained/terminal-reset-failed-9ad48b96abf6.7kplFw"
    if mirror.is_dir():
        return mirror / "diagnosis", mirror / "original"
    pytest.skip("authentic terminal closeout failure mirror is not yet available")


def test_actual_saved_failure_files_authenticate_without_host_or_runtime(monkeypatch):
    diagnosis, original = actual_failure_roots()
    receipt = repair._read_failure(diagnosis, original)
    assert receipt["invocation_id"] == repair.FAILURE_INVOCATION
    assert receipt["original_inventory"] == inventory()
    assert {path.name for path in diagnosis.iterdir()} == {
        "receipt.json",
        "journal.log",
    }


def test_failure_reader_refuses_mutated_receipt_or_extra_file(tmp_path):
    diagnosis, original = actual_failure_roots()
    root = tmp_path / "diagnosis"
    root.mkdir()
    for path in diagnosis.iterdir():
        (root / path.name).write_bytes(path.read_bytes())
    (root / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError):
        repair._read_failure(root, original)


@pytest.mark.parametrize("damage", ["receipt.json", "journal.log", "missing-journal"])
def test_authentic_failure_reader_rejects_whole_file_tampering(tmp_path, damage):
    diagnosis, original = actual_failure_roots()
    root = tmp_path / "diagnosis"
    root.mkdir()
    for path in diagnosis.iterdir():
        if damage == "missing-journal" and path.name == "journal.log":
            continue
        raw = path.read_bytes()
        (root / path.name).write_bytes(raw + b" " if path.name == damage else raw)
    with pytest.raises(ValueError):
        repair._read_failure(root, original)
