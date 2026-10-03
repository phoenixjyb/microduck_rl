"""Pure protocol and explicitly synthetic hook tests; no native qualification."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_optimizer_probe as probe
from mjlab_microduck import stance_recovery_ppo_receipt_repair as audit
from mjlab_microduck import stance_recovery_ppo_trace as trace


def test_protocol_caps_and_static_artifact_inventory():
    assert probe.PROTOCOL == "football-b1d-cpu-scheduled-ppo-one-update-probe-v1"
    assert (
        probe.RUN_SECONDS,
        probe.CLOSEOUT_SECONDS,
        probe.MARGIN_SECONDS,
        probe.LAUNCH_RESERVE,
    ) == (360, 240, 60, 660)
    assert probe.MEMORY_BYTES == 2 * 1024**3
    assert (probe.CPU_QUOTA, probe.NICE, probe.KILL_MODE) == (
        "2s",
        "10",
        "control-group",
    )
    assert len(probe.DIAGNOSTIC_FILES) == 21
    assert {
        "diagnostic-state-step-000.pt",
        "diagnostic-state-step-020.pt",
    } <= probe.DIAGNOSTIC_FILES
    assert "exception-state.pt" not in probe.COMPLETE_FILES
    assert probe.COMPLETE_FILES == probe.PRE_REPORT_FILES | {"report.json"}
    assert probe.CLOSEOUT_FILES == probe.COMPLETE_FILES | {"independent-closeout.json"}
    assert probe.lease_declaration()["scope"] == ["run", "closeout"]
    assert probe.lease_declaration()["idle_gate"] == "base.host.wait_idle-before-and-after"
    assert probe.AUDIT_SOURCE == "75ed100d2b8470c86e0327224073013dab393d23"
    assert probe.ARTIFACT_SOURCE != probe.AUDIT_SOURCE


def test_source_and_service_identity_validation():
    assert probe.output_path("a" * 40).name.endswith("a" * 12)
    for source in (probe.ARTIFACT_SOURCE, probe.AUDIT_SOURCE):
        with pytest.raises(ValueError):
            probe.output_path(source)
    good = {
        "MainPID": "42",
        "ActiveState": "active",
        "RuntimeMaxUSec": "6min",
        "MemoryMax": str(probe.MEMORY_BYTES),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
    }
    assert probe._recorded_properties(good, "run") is True
    bad = dict(good, CPUQuotaPerSecUSec="4s")
    with pytest.raises(ValueError):
        probe._recorded_properties(bad, "run")
    with pytest.raises(ValueError):
        probe._recorded_properties(good, "closeout")
    assert (
        probe._update_flag_after_failure(
            attempted=False, optimizer_steps=0, completed_updates=0
        )
        is False
    )
    assert (
        probe._update_flag_after_failure(
            attempted=True, optimizer_steps=0, completed_updates=0
        )
        is None
    )
    assert (
        probe._update_flag_after_failure(
            attempted=True, optimizer_steps=1, completed_updates=0
        )
        is True
    )


def test_pinned_closed_audit_inventory_is_hash_and_semantics_bound(
    tmp_path, monkeypatch
):
    receipt = dict(
        protocol=audit.PROTOCOL,
        artifact_source=audit.ARTIFACT_SOURCE,
        evaluator_source=probe.AUDIT_SOURCE,
        independent_cpu_replay=True,
        run_admissible=True,
        optimizer_steps=0,
        training_update_performed=False,
        cuda_initialized=False,
        **baseline.FALSE_FLAGS,
    )
    values = {
        "launch.json": {"source": probe.AUDIT_SOURCE},
        "source-inventory.json": {
            "artifact_source": audit.ARTIFACT_SOURCE,
            "evaluator_source": probe.AUDIT_SOURCE,
        },
        "receipt.json": receipt,
    }
    expected_digests = {
        "launch.json": probe.AUDIT_LAUNCH_SHA256,
        "source-inventory.json": probe.AUDIT_SOURCE_INVENTORY_SHA256,
        "receipt.json": probe.AUDIT_RECEIPT_SHA256,
    }
    # Synthetic bytes are not accepted as the pinned live audit: pin mismatch rejects.
    for name, value in values.items():
        (tmp_path / name).write_text(audit.base.files.canonical(value) + "\n")
    monkeypatch.setattr(audit, "output_path", lambda source: tmp_path)
    with pytest.raises(ValueError):
        probe._audit_inventory()
    assert (
        expected_digests["receipt.json"]
        != sha256((tmp_path / "receipt.json").read_bytes()).hexdigest()
    )


def test_retained_closed_audit_mirror_hashes_and_semantics_read_only():
    """Optional evidence mirror check; not host/service/native attestation."""
    repo = Path(__file__).resolve().parents[1]
    direct = (
        repo
        / "artifacts"
        / "evaluations"
        / ("stance-wsl-cpu-ppo-receipt-repair-" + probe.AUDIT_SOURCE[:12])
    )
    roots = (
        [direct]
        if direct.is_dir()
        else list(
            (repo / "artifacts" / "retained").glob(
                "ppo-audit-closed-75ed100d2b84.*/stance-wsl-cpu-ppo-receipt-repair-75ed100d2b84"
            )
        )
    )
    if not roots:
        pytest.skip("closed audit mirror is not present in this checkout")
    result = probe._audit_inventory(roots[0], check_live=False)
    assert result["artifact_source"] == audit.ARTIFACT_SOURCE
    assert result["evaluator_source"] == probe.AUDIT_SOURCE
    assert set(result["inventory"]) == audit.COMPLETE_FILES


def test_exact_old_failed_service_invocation_schema():
    for invocation in probe.OLD_FAILURES.values():
        state = {
            "ActiveState": "failed",
            "MainPID": "0",
            "NRestarts": "0",
            "ExecMainStatus": "1",
            "InvocationID": invocation,
        }
        assert probe._recorded_failed_state(state, invocation)
        with pytest.raises(ValueError):
            probe._recorded_failed_state(dict(state, InvocationID="0" * 32), invocation)


def test_partial_optimizer_flag_preserves_unknown_first_mutation():
    assert (
        probe._update_flag_after_failure(
            attempted=False, optimizer_steps=0, completed_updates=0
        )
        is False
    )
    assert (
        probe._update_flag_after_failure(
            attempted=True, optimizer_steps=0, completed_updates=0
        )
        is None
    )
    assert (
        probe._update_flag_after_failure(
            attempted=True, optimizer_steps=1, completed_updates=0
        )
        is True
    )


def test_raw_budget_is_single_pass_and_refuses_overflow(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe.base.retained,
        "write_capture",
        lambda path, raw: Path(path).write_bytes(raw),
    )
    budget = {"written": 0}
    entry = probe._write_raw(tmp_path / "one.pt", b"data", budget)
    assert entry == {"sha256": sha256(b"data").hexdigest(), "bytes": 4}
    assert budget["written"] == 4
    with pytest.raises(ValueError):
        probe._write_raw(tmp_path / "two.pt", b"xx", {"written": probe.RAW_LIMIT})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["one.pt"]


def test_synthetic_hook_records_each_actual_adam_post_step_once(tmp_path, monkeypatch):
    """Synthetic optimizer only: verifies durable hook progression, not bridge capture."""
    actor = torch.nn.Linear(1, 1)
    actor.distribution = SimpleNamespace(std_param=torch.nn.Parameter(torch.ones(1)))
    critic = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=0.1)
    learner = SimpleNamespace(
        algorithm=SimpleNamespace(optimizer=optimizer),
        actor=actor,
        critic=critic,
        optimizer_steps=0,
        completed_updates=0,
        phase="synthetic",
        faulted=False,
        storage=object(),
        private_rng_state=torch.zeros(2),
    )
    monkeypatch.setattr(
        probe,
        "_diagnostic_value",
        lambda target, step, _context, caller: {
            "synthetic_fixture": True,
            "step": step,
            "observed_counter": target.optimizer_steps,
            "caller": caller,
        },
    )
    monkeypatch.setattr(
        probe, "_serialize", lambda value: str(value["observed_counter"]).encode()
    )
    monkeypatch.setattr(
        probe,
        "_write_raw",
        lambda path, raw, budget: (
            Path(path).write_bytes(raw),
            {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)},
        )[1],
    )
    counter_handle = optimizer.register_step_post_hook(
        lambda *_: setattr(learner, "optimizer_steps", learner.optimizer_steps + 1)
    )
    counts, handles = probe._install_diagnostic_hooks(
        learner, tmp_path, {"written": 0}, torch.zeros(1, dtype=torch.uint8)
    )
    try:
        for _ in range(3):
            optimizer.zero_grad()
            sum(
                parameter.square().sum()
                for parameter in optimizer.param_groups[0]["params"]
            ).backward()
            optimizer.step()
    finally:
        counter_handle.remove()
        for handle in handles:
            handle.remove()
    assert counts == {"pre": 3, "post": 3}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "diagnostic-state-step-001.pt",
        "diagnostic-state-step-002.pt",
        "diagnostic-state-step-003.pt",
    ]
    assert (tmp_path / "diagnostic-state-step-003.pt").read_bytes() == b"3"


def test_synthetic_hook_write_failure_is_not_retried(tmp_path, monkeypatch):
    """A persistence exception aborts the update path and does not replay a step."""
    actor = torch.nn.Linear(1, 1)
    actor.distribution = SimpleNamespace(std_param=torch.nn.Parameter(torch.ones(1)))
    critic = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=0.1)
    learner = SimpleNamespace(
        algorithm=SimpleNamespace(optimizer=optimizer),
        actor=actor,
        critic=critic,
        optimizer_steps=0,
        completed_updates=0,
        phase="synthetic",
        faulted=False,
        storage=object(),
        private_rng_state=torch.zeros(2),
    )
    monkeypatch.setattr(probe, "_diagnostic_value", lambda *_: {"step": 1})
    monkeypatch.setattr(probe, "_serialize", lambda _value: b"synthetic")
    writes = []

    def fail_write(*_args):
        writes.append("attempt")
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(probe, "_write_raw", fail_write)
    _, handles = probe._install_diagnostic_hooks(
        learner, tmp_path, {"written": 0}, torch.zeros(1)
    )
    try:
        optimizer.zero_grad()
        sum(
            parameter.square().sum()
            for parameter in optimizer.param_groups[0]["params"]
        ).backward()
        with pytest.raises(OSError, match="synthetic disk failure"):
            optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
    assert writes == ["attempt"]


def test_synthetic_private_rng_snapshot_labels_fork_state_not_caller_state():
    """CPU-only fork fixture: the in-scope stream is not mislabeled caller RNG."""
    caller_before = torch.random.get_rng_state().clone()
    private = torch.Generator(device="cpu").manual_seed(1234).get_state()
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(private)
        observed = torch.random.get_rng_state().clone()
        snapshot = probe._rng_snapshot(
            "inside-private-update-fork", caller_before, private, observed
        )
        assert snapshot["context"] == "inside-private-update-fork"
        assert torch.equal(snapshot["observed_cpu_rng"], private)
        assert torch.equal(snapshot["learner_private_rng_state"], private)
    assert torch.equal(torch.random.get_rng_state(), caller_before)


def test_transition_binding_rejects_wrong_source_or_compiled_plant(monkeypatch):
    declaration = {"source": "b" * 40}
    plant = {"selected_plant": "synthetic-fixture-only"}
    expected = {"source": "b" * 40, "plant": "synthetic-fixture-only"}
    monkeypatch.setattr(
        trace,
        "binding",
        lambda d, p, _profile: {"source": d["source"], "plant": p["selected_plant"]},
    )
    launch = {
        "declaration": declaration,
        "compiled_plant": plant,
        "cpu_math_profile": {"synthetic_test_fixture": True},
    }
    metadata = {"source": "b" * 40, "binding": expected}
    value = {
        "protocol": trace.PROTOCOL,
        "binding": expected,
        "declaration": declaration,
        "compiled_plant": plant,
    }
    assert probe._check_transition_binding(value, "b" * 40, launch, metadata)
    wrong_source = deepcopy(value)
    wrong_source["binding"]["source"] = "c" * 40
    with pytest.raises(ValueError):
        probe._check_transition_binding(wrong_source, "b" * 40, launch, metadata)
    wrong_plant = deepcopy(value)
    wrong_plant["compiled_plant"] = {"selected_plant": "tampered"}
    with pytest.raises(ValueError):
        probe._check_transition_binding(wrong_plant, "b" * 40, launch, metadata)


def test_synthetic_replay_schema_matches_optimizer_trace_and_rejects_invented_keys(
    monkeypatch,
):
    """Schema fixture only: these fields are not a native replay attestation."""
    monkeypatch.setattr(probe, "_check_trace_score", lambda _score: True)
    replay = {
        "protocol": probe.optimizer_trace.PROTOCOL,
        "optimizer_replay": True,
        "environment_created": False,
        "physics_resimulated": False,
        "exact_gae": True,
        "exact_updated_model": True,
        "exact_adam_state": True,
        "exact_private_rng": True,
        "exact_metrics": True,
        "gradient_hook_steps": 20,
        "moment_hook_steps": 20,
        "whole_trajectory_physics_resimulated": False,
        "thermal_model_applied": False,
        "cuda_initialized": False,
        "optimizer_integration_only": True,
        "source_trace_sha256": "a" * 64,
        "capture_sha256": "b" * 64,
        "source_trace_score": {"synthetic_test_fixture": True},
        **baseline.FALSE_FLAGS,
    }
    assert probe._check_replay_result(replay, "a" * 64, "b" * 64)
    invented = deepcopy(replay)
    invented["optimizer_replay_performed"] = True
    with pytest.raises(ValueError):
        probe._check_replay_result(invented, "a" * 64, "b" * 64)


def test_report_inventory_constructor_matches_closeout_sha_and_byte_shape(tmp_path):
    (tmp_path / "capture.pt").write_bytes(b"synthetic captured raw")
    (tmp_path / "launch.json").write_bytes(b"synthetic launch")
    value = probe._report_inventory(tmp_path)
    assert set(value) == {"capture.pt", "launch.json"}
    for name, entry in value.items():
        raw = (tmp_path / name).read_bytes()
        assert entry == {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def test_synthetic_partial_files_are_diagnostic_only_not_complete_capture():
    present = {
        "checkpoint.pt",
        "launch.json",
        "transition.pt",
        "transition.json",
        "capture-attempt.json",
        "preupdate.pt",
        "diagnostic-state-step-000.pt",
        "diagnostic-state-step-004.pt",
        "exception-state.pt",
        "report.json",
    }
    assert present != probe.COMPLETE_FILES
    assert "exception-state.pt" not in probe.PRE_REPORT_FILES
    assert (
        "optimizer.pt" in probe.COMPLETE_FILES
        and "optimizer.json" in probe.COMPLETE_FILES
    )


def test_synthetic_update_wrapper_calls_capture_once_and_never_retries():
    """Synthetic wrapper fixture only; no learner or training admission."""
    learner = SimpleNamespace(completed_updates=0, optimizer_steps=0)
    calls = []

    def fake_capture(target, raw_parent, raw_trace, trace_sha):
        calls.append((raw_parent, raw_trace, trace_sha))
        target.optimizer_steps = 20
        target.completed_updates = 1
        return {"synthetic_test_fixture": True}

    result = probe._invoke_update_once(
        learner, b"p", b"t", "a" * 64, capture=fake_capture
    )
    assert result == {"synthetic_test_fixture": True}
    assert len(calls) == 1

    failed = SimpleNamespace(completed_updates=0, optimizer_steps=0)
    failure_calls = []

    def fail_once(*_):
        failure_calls.append(1)
        raise RuntimeError("synthetic no-retry sentinel")

    with pytest.raises(RuntimeError, match="synthetic no-retry sentinel"):
        probe._invoke_update_once(failed, b"p", b"t", "a" * 64, capture=fail_once)
    assert failure_calls == [1]


def test_synthetic_marker_failure_removes_observer_hooks_without_capture():
    removed = []

    class Handle:
        def remove(self):
            removed.append(True)

    capture_calls = []

    def fail_marker():
        raise OSError("synthetic marker fsync failure")

    def should_not_capture():
        capture_calls.append(True)
        raise AssertionError("capture must not begin")

    with pytest.raises(OSError, match="synthetic marker fsync failure"):
        probe._marker_then_capture(
            fail_marker, should_not_capture, [Handle(), Handle()]
        )
    assert removed == [True, True]
    assert capture_calls == []
