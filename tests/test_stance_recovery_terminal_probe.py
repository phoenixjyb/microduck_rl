"""Source-only supervisor contracts; synthetic scores never admit a native job."""

from copy import deepcopy
from pathlib import Path

import pytest

from mjlab_microduck import stance_recovery_terminal_probe as p

SOURCE = "a" * 40


def properties(mode):
    seconds = p.RUN_SECONDS if mode == "run" else p.CLOSEOUT_SECONDS
    return {
        "MainPID": "123",
        "ActiveState": "active",
        "RuntimeMaxUSec": f"{seconds // 60}min",
        "MemoryMax": str(p.MEMORY_BYTES),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
    }


def synthetic_score(*, full=False, selective=False, timeout=False):
    """Pure typed receipt fixture, not data from a runtime or verified artifact."""
    return {
        "protocol": p.trace.PROTOCOL,
        "collection": {
            "policy_ticks": 250 if full else 30,
            "elapsed_seconds": 45.0,
            "stop_reason": "first-natural-terminal",
            "failure": None,
        },
        "validated_policy_ticks": 250 if full else 30,
        "full_timeout_qualified": full,
        "timeout_reset_qualified": timeout or full,
        "selective_reset_qualified": selective,
        "complete_force_phase_checks": full,
        "force": {
            "checked_physics_steps": 2500 if full else 300,
            "window_steps_per_row": [10, 20] if full else [0, 20],
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
        **p.baseline.FALSE_FLAGS,
    }


@pytest.mark.parametrize("mode", ["run", "closeout"])
def test_exact_caps_and_distinct_service_names(mode):
    assert p._check_properties(properties(mode), mode)
    assert (
        p.service_name(SOURCE, mode)
        == f"microduck-cpu-terminal-{mode}-{SOURCE[:12]}.service"
    )
    assert p.LAUNCH_RESERVE == 720
    assert p.RUN_SECONDS == 360 and p.CLOSEOUT_SECONDS == 300
    assert p.MEMORY_BYTES == 2 * 1024**3 and p.RAW_LIMIT == 128 * 1024**2


@pytest.mark.parametrize(
    "key,value",
    [
        ("MainPID", "0"),
        ("MainPID", True),
        ("ActiveState", "inactive"),
        ("RuntimeMaxUSec", "1h"),
        ("MemoryMax", "infinity"),
        ("CPUQuotaPerSecUSec", "infinity"),
        ("Nice", "0"),
        ("KillMode", "process"),
    ],
)
def test_changed_recorded_caps_refused(key, value):
    record = properties("run")
    record[key] = value
    with pytest.raises(ValueError):
        p._check_properties(record, "run")


@pytest.mark.parametrize(
    "source",
    [
        p.UPDATE_SOURCE,
        p.previous.AUDIT_SOURCE,
        p.previous.ARTIFACT_SOURCE,
        p.FAILED_SOURCE,
        "bad",
    ],
)
def test_old_artifact_namespace_cannot_be_reused(source):
    with pytest.raises(ValueError):
        p.output_path(source)


def test_file_inventory_and_no_overwrite_are_actual_byte_checks(tmp_path):
    raw = b"source-only evidence bytes"
    item = p._write_raw(tmp_path / "capture.pt", raw)
    assert item == p._inventory(tmp_path, {"capture.pt"})["capture.pt"]
    assert item["bytes"] == len(raw)
    with pytest.raises(FileExistsError):
        p._write_raw(tmp_path / "capture.pt", raw)
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError):
        p._inventory(tmp_path, {"capture.pt"})


def test_partial_inventory_does_not_prove_closed_update(tmp_path):
    (tmp_path / "independent-closeout.json").write_text("{}")
    with pytest.raises(ValueError):
        p._read_closed_update(tmp_path)


def _saved_launch_failure():
    relative = Path("artifacts/tools/terminal-launch-construction-failure-f04e07ce157d")
    original = Path(
        "artifacts/evaluations/stance-wsl-cpu-stochastic-first-terminal-f04e07ce157d"
    )
    native = p.base.host.ROOT
    if (native / relative).is_dir() and (native / original).is_dir():
        return native / relative, native / original
    mirror = (
        Path(__file__).resolve().parents[1]
        / "artifacts/retained/terminal-launch-failed-f04e07ce157d.zujnXr"
    )
    if not mirror.is_dir():
        pytest.skip("optional authentic saved launch failure unavailable")
    return mirror / relative.name, mirror / original.name


def test_actual_launch_failure_is_authenticated_without_fake_native_context():
    root, original = _saved_launch_failure()
    receipt = p._read_launch_failure(root, original)
    assert receipt["source"] == p.FAILED_SOURCE
    assert receipt["state"]["InvocationID"] == p.FAILED_INVOCATION
    assert receipt["policy_calls"] == receipt["optimizer_steps"] == 0
    assert set(receipt["inventory"]) == {"checkpoint.pt", "report.json"}


@pytest.mark.parametrize("changed", ["receipt.json", "journal.log", "report.json"])
def test_saved_launch_failure_whole_bytes_cannot_be_rewritten(tmp_path, changed):
    saved, original = _saved_launch_failure()
    root, capture = tmp_path / "diagnosis", tmp_path / "original"
    root.mkdir()
    capture.mkdir()
    for source, target in ((saved, root), (original, capture)):
        for path in source.iterdir():
            (target / path.name).write_bytes(path.read_bytes())
    target = (capture if changed == "report.json" else root) / changed
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        p._read_launch_failure(root, capture)


def test_actual_retained_one_update_prerequisite_can_be_read_without_fake_host_context():
    native = p.previous.output_path(p.UPDATE_SOURCE)
    mirror = Path(__file__).resolve().parents[1] / (
        "artifacts/retained/ppo-update-closed-0f2451268372.Q9z5r0/"
        "stance-wsl-cpu-scheduled-ppo-one-update-0f2451268372"
    )
    root = native if native.is_dir() else mirror
    if not root.is_dir():
        pytest.skip("optional actual saved one-update evidence unavailable")
    receipt = p._read_closed_update(root)
    assert len(receipt["inventory"]) == 31
    assert receipt["closeout_sha256"] == p.UPDATE_CLOSEOUT_SHA256
    assert receipt["diagnostic_updated_actor_used"] is False
    assert receipt["decision"] == "one-update-cpu-optimizer-integration-replayed"


def launch_fixture(monkeypatch):
    """Pure launch-contract fixture; explicitly bypasses only live CPU-profile inspection.

    It is not a native context, service, parent initialization or launch admission.
    Real production callers obtain their context from ``_fresh`` without this seam.
    """
    native = p.previous.output_path(p.UPDATE_SOURCE)
    mirror = Path(__file__).resolve().parents[1] / (
        "artifacts/retained/ppo-update-closed-0f2451268372.Q9z5r0/"
        "stance-wsl-cpu-scheduled-ppo-one-update-0f2451268372"
    )
    root = native if native.is_dir() else mirror
    if not root.is_dir():
        pytest.skip("optional authentic parent/plant metadata unavailable")
    old = p.base.files.parse((root / "launch.json").read_bytes())
    raw_parent = (root / "checkpoint.pt").read_bytes()
    # Schema validation is real; current-host qualification is intentionally not asserted.
    monkeypatch.setattr(
        p.previous.trace.profile,
        "check_recorded",
        p.previous.trace.profile.validate_receipt,
    )
    declaration = p.previous.collection_probe.declaration(SOURCE)
    compiled = old["compiled_plant"]
    profile = old["cpu_math_profile"]
    fresh = (
        {"cpu_math_profile": profile},
        {"synthetic-prior": True},
        raw_parent,
        {"synthetic-audit": True},
        {"synthetic-update": True},
    )
    launch = {
        "protocol": p.PROTOCOL,
        "source": SOURCE,
        "prerequisite_receipt": fresh[1],
        "closed_audit_binding": fresh[3],
        "closed_update_binding": fresh[4],
        "parent_checkpoint_sha256": p.baseline.CHECKPOINT_SHA256,
        "checkpoint": {
            "sha256": p.baseline.CHECKPOINT_SHA256,
            "bytes": len(raw_parent),
        },
        "policy_call_limit": 250,
        "physics_step_limit": 2500,
        "simulation_seconds": 5.0,
        "collection_seconds": p.trace.WALL_LIMIT,
        "service_seconds": p.RUN_SECONDS,
        "closeout_seconds": p.CLOSEOUT_SECONDS,
        "launch_reserve_seconds": p.LAUNCH_RESERVE,
        "raw_capture_limit_bytes": p.RAW_LIMIT,
        "lease": p._lease(),
        "optimizer_steps": 0,
        "training_update_performed": False,
        "execution_admitted": False,
        "student_export_available": False,
        "cpu_math_profile": profile,
        "declaration": declaration,
        "compiled_plant": compiled,
        "trace_binding": p.trace.binding(declaration, compiled, profile),
        "campaign_window": p.window.declaration(),
        "service_properties": properties("run"),
        **p.baseline.FALSE_FLAGS,
    }
    return launch, fresh


def test_launch_declares_real_units_and_fixed_campaign_schema(monkeypatch):
    launch, fresh = launch_fixture(monkeypatch)
    assert p._check_launch(launch, SOURCE, fresh)
    assert launch["trace_binding"]["horizon"] == 250


def test_actual_launch_builder_emits_native_context_profile_once(monkeypatch):
    launch, fresh = launch_fixture(monkeypatch)
    constructed = p._build_launch(
        SOURCE,
        fresh,
        launch["checkpoint"],
        launch["declaration"],
        launch["compiled_plant"],
        launch["cpu_math_profile"],
        properties("run"),
    )
    assert constructed == launch
    assert p._check_launch(constructed, SOURCE, fresh)


def test_launch_builder_refuses_disagreeing_context_profile(monkeypatch):
    launch, fresh = launch_fixture(monkeypatch)
    fresh[0]["cpu_math_profile"] = {"different-profile": True}
    with pytest.raises(ValueError, match="verified native context"):
        p._build_launch(
            SOURCE,
            fresh,
            launch["checkpoint"],
            launch["declaration"],
            launch["compiled_plant"],
            launch["cpu_math_profile"],
            properties("run"),
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("service_seconds", 361),
        ("closeout_seconds", 301),
        ("launch_reserve_seconds", 660),
        ("collection_seconds", 181),
        ("raw_capture_limit_bytes", 2 * 128 * 1024**2),
        ("policy_call_limit", 2500),
        ("physics_step_limit", 250),
        ("simulation_seconds", 5),
        ("parent_checkpoint_sha256", "f" * 64),
        ("training_update_performed", True),
        ("campaign_window", {"cutoff_unix": 0}),
    ],
)
def test_source_only_launch_contract_refuses_unit_or_cap_drift(monkeypatch, key, value):
    launch, fresh = launch_fixture(monkeypatch)
    launch[key] = value
    with pytest.raises(ValueError):
        p._check_launch(launch, SOURCE, fresh)


def test_separate_gate_decisions():
    assert (
        p._score_decision(synthetic_score(full=True))
        == "cpu-natural-full-timeout-and-reset-replayed"
    )
    assert (
        p._score_decision(synthetic_score(selective=True))
        == "cpu-natural-selective-reset-replayed-full-timeout-open"
    )
    assert (
        p._score_decision(synthetic_score(timeout=True))
        == "cpu-natural-timeout-reset-replayed-full-timeout-open"
    )
    assert (
        p._score_decision(synthetic_score())
        == "cpu-first-terminal-prefix-replayed-no-timeout-reset-qualification"
    )


def test_first_terminal_selective_cannot_also_claim_timeout():
    with pytest.raises(ValueError, match="distinct outcomes"):
        p._score_decision(synthetic_score(selective=True, timeout=True))


def test_selective_reset_requires_checked_prefix_not_unexecuted_full_windows():
    score = synthetic_score(selective=True)
    assert score["collection"]["policy_ticks"] == 30
    assert score["complete_force_phase_checks"] is False
    assert score["force"]["window_steps_per_row"] == [0, 20]
    assert (
        p._score_decision(score)
        == "cpu-natural-selective-reset-replayed-full-timeout-open"
    )
    score["recorded_force_prefix_checked"] = False
    with pytest.raises(ValueError):
        p._score_decision(score)


def run_receipt_fixture():
    """Synthetic document contract only; no host, captured physics or inference."""
    inventory = {
        name: {"sha256": "b" * 64, "bytes": 256 if name == "checkpoint.pt" else 1024}
        for name in p.RUN_FILES
    }
    launch = {
        "service_properties": properties("run"),
        "trace_binding": {"source": SOURCE},
    }
    metadata = {
        "protocol": p.trace.PROTOCOL,
        "source": SOURCE,
        "binding": launch["trace_binding"],
        "collection": {
            "policy_ticks": 250,
            "elapsed_seconds": 45.0,
            "stop_reason": "first-natural-terminal",
            "failure": None,
        },
        "capture_sha256": inventory["capture.pt"]["sha256"],
        "capture_bytes": 1024,
        "optimizer_steps": 0,
        "training_update_performed": False,
        "execution_admitted": False,
        "student_export_available": False,
        **p.baseline.FALSE_FLAGS,
    }
    report = {
        "protocol": p.PROTOCOL,
        "source": SOURCE,
        "decision": "first-terminal-captured-awaiting-independent-closeout",
        "files": {k: v for k, v in inventory.items() if k != "report.json"},
        "capture": metadata,
        "postchecks_passed": True,
        "source_unchanged": True,
        "filmbrain_unchanged": True,
        "protected_services_inactive": True,
        "cuda_initialized": False,
        "optimizer_steps": 0,
        "training_update_performed": False,
        "execution_admitted": False,
        "student_export_available": False,
        "elapsed_seconds": 60.0,
        "raw_bytes_written": 1280,
        "service_properties": deepcopy(launch["service_properties"]),
        **p.baseline.FALSE_FLAGS,
    }
    return report, metadata, inventory, launch


def test_exact_zero_update_run_receipt_contract():
    assert p._check_report(*run_receipt_fixture(), SOURCE)


@pytest.mark.parametrize(
    "key,value",
    [
        ("elapsed_seconds", 360.0),
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
        ("elapsed_seconds", 0.0),
        ("raw_bytes_written", 1281),
        ("training_update_performed", True),
        ("execution_admitted", True),
        ("student_export_available", True),
    ],
)
def test_over_cap_or_admitting_run_report_refused(key, value):
    args = run_receipt_fixture()
    args[0][key] = value
    with pytest.raises(ValueError):
        p._check_report(*args, SOURCE)


def test_different_run_pid_cannot_borrow_launch_caps():
    args = run_receipt_fixture()
    args[0]["service_properties"]["MainPID"] = "456"
    with pytest.raises(ValueError):
        p._check_report(*args, SOURCE)


@pytest.mark.parametrize(
    "key,value",
    [
        ("optimizer_steps", 1),
        ("training_update_performed", True),
        ("execution_admitted", True),
        ("student_export_available", True),
        ("recovery_accepted", True),
        ("capture_sha256", "c" * 64),
        ("capture_bytes", 1000),
        ("unexpected", False),
    ],
)
def test_raw_metadata_admission_or_identity_tamper_refused(key, value):
    args = run_receipt_fixture()
    args[1][key] = value
    with pytest.raises(ValueError):
        p._check_report(*args, SOURCE)


@pytest.mark.parametrize(
    "key",
    [
        "exact_policy_replay",
        "private_rng",
        "caller_rng",
        "recorded_force_prefix_checked",
        "complete_force_phase_checks",
        "terminal_critic_bootstrap_exact",
        "compiled_plant_checked",
        "control_trace_checked",
        "physical_trace_checked",
        "collection_cap_respected",
    ],
)
def test_boolean_terminal_claim_cannot_bypass_independent_evidence(key):
    score = synthetic_score(full=True)
    score[key] = False
    with pytest.raises(ValueError):
        p._score_decision(score)


@pytest.mark.parametrize(
    "mutation",
    [
        {"collection": {"policy_ticks": 249}},
        {"collection": {"elapsed_seconds": 180.0}},
        {"collection": {"elapsed_seconds": float("nan")}},
        {"collection": {"stop_reason": "deadline-exhausted"}},
        {"collection": {"failure": {"stage": "reset"}}},
        {"force": {"checked_physics_steps": 2490}},
        {"force": {"window_steps_per_row": [0, 20]}},
        {"force": {"delivered_nonzero_steps_per_row": [0, 19]}},
        {"selective_reset_qualified": True},
        {"timeout_reset_qualified": False},
        {"whole_trajectory_physics_resimulated": True},
        {"thermal_model_applied": True},
        {"training_admitted": True},
    ],
)
def test_full_timeout_requires_real_counts_and_nonadmitting_scope(mutation):
    score = synthetic_score(full=True)
    for key, value in deepcopy(mutation).items():
        if isinstance(value, dict):
            score[key].update(value)
        else:
            score[key] = value
    with pytest.raises(ValueError):
        p._score_decision(score)
