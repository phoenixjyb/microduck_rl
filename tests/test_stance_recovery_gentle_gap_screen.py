"""Pure protocol and receipt checks for the 21-case CPU gentle-gap screen."""

import json
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256

import pytest

from mjlab_microduck import stance_recovery_gentle_gap_screen as screen
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_trace as evidence

SOURCE = "a" * 40
MAC_AUDIT_MIRROR = "artifacts/retained/terminal-repair-closed-3338b5f381e9.KLlNRi/audit"


def synthetic_terminal_replay():
    """Constructor fixture only, never native terminal evidence."""
    return {
        "protocol": screen.terminal.trace.PROTOCOL,
        "collection": {
            "policy_ticks": 250,
            "elapsed_seconds": 1.0,
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
        **screen.base.baseline.FALSE_FLAGS,
    }


def synthetic_score(cell, *, candidate_pass=True, full=True, force_complete=True):
    ticks, steps = (250, 2500) if full else (30, 300)
    return dict(
        cell=cell,
        protocol=evidence.PROTOCOL,
        strict_actor_restore=True,
        actor_replay_max_abs_error=0.0,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False,
        **screen.base.baseline.FALSE_FLAGS,
        numerical_diagnostic={
            "candidate_pass": candidate_pass,
            "complete_first_attempt": True,
        },
        collection={
            "policy_ticks": ticks,
            "elapsed_seconds": 1.0,
            "stop_reason": "policy-tick-limit"
            if full
            else "all-first-attempts-complete",
        },
        pulse={
            "checked_physics_steps": steps,
            "complete_pulse_delivery": force_complete,
            "complete_phase_checks": force_complete,
            "recorded_phase_checks_valid": True,
            "exact_full_force_arrays_checked": True,
            "unforced_post_arrays_checked": True,
        },
    )


def synthetic_rows():
    return [synthetic_score(cell) for cell in screen.CELL_IDS]


def _closed_audit_files(root, *, selective=False):
    original = {"source_identity": {"source": screen.terminal_repair.ARTIFACT_SOURCE}}
    context = {"source_identity": {"source": screen.ARTIFACT_SOURCE}}
    source_inventory = {"artifact_source": screen.ARTIFACT_SOURCE}
    failure = {"source": screen.terminal_repair.ARTIFACT_SOURCE}
    original_inventory = {"checkpoint.pt": {"sha256": "0" * 64, "bytes": 1}}
    service = {
        "MainPID": "17",
        "ActiveState": "active",
        "RuntimeMaxUSec": "3min",
        "MemoryMax": str(2 * 1024**3),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
    }
    launch = dict(
        protocol=screen.terminal_repair.PROTOCOL,
        artifact_source=screen.terminal_repair.ARTIFACT_SOURCE,
        evaluator_source=screen.ARTIFACT_SOURCE,
        evaluator_context=context,
        original_source_context=original,
        source_inventory=source_inventory,
        failure=failure,
        original_inventory=original_inventory,
        service_properties=service,
        service_seconds=180,
        launch_reserve_seconds=240,
        memory_bytes=2 * 1024**3,
        cpu_quota="2s",
        nice="10",
        kill_mode="control-group",
        optimizer_steps=0,
        recollection_performed=False,
        audit_simulator_resets=0,
        **screen.base.baseline.FALSE_FLAGS,
    )
    values = {"launch.json": launch, "source-inventory.json": source_inventory}
    raw_by_name = {
        name: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        for name, value in values.items()
    }
    inventory = {
        name: {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in raw_by_name.items()
    }
    receipt = dict(
        protocol=screen.terminal_repair.PROTOCOL,
        artifact_source=screen.terminal_repair.ARTIFACT_SOURCE,
        evaluator_source=screen.ARTIFACT_SOURCE,
        decision="cpu-natural-full-timeout-and-reset-replayed",
        status="complete",
        original_full_timeout_reset_qualified=True,
        original_selective_reset_qualified=selective,
        audit_simulator_resets=0,
        recollection_performed=False,
        optimizer_steps=0,
        training_update_performed=False,
        cuda_initialized=False,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False,
        source_inventory=source_inventory,
        failure=failure,
        original_inventory=original_inventory,
        service_properties=service,
        independent_replay=synthetic_terminal_replay(),
        context_binding={"original_context": original, "evaluator_context": context},
        output_inventory=inventory,
        final_inventory_names=sorted(
            {"launch.json", "source-inventory.json", "receipt.json"}
        ),
        **screen.base.baseline.FALSE_FLAGS,
    )
    raw_by_name["receipt.json"] = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()
    for name, raw in raw_by_name.items():
        (root / name).write_bytes(raw)
    return {
        name: (sha256(raw).hexdigest(), len(raw)) for name, raw in raw_by_name.items()
    }


def test_fixed_order_is_zero_plus_twelve_cardinals_plus_eight_diagonals():
    expected = ["zero-wrench"] + [
        f"{direction}-2n-10steps-t{onset}"
        for onset in (250, 500, 750)
        for direction in ("+x", "-x", "+y", "-y")
    ]
    expected += [
        f"diagonal-{direction}-2n-10steps-t{onset}"
        for onset in (375, 625)
        for direction in ("++", "+-", "-+", "--")
    ]
    assert list(screen.CELL_IDS) == expected
    rows = screen.declarations(SOURCE)
    assert len(rows) == 21
    assert [row["cell_ids"][0] for row in rows] == expected
    assert all(
        row["stage"] == "timing" and row["split"] == "held-out" and row["worlds"] == 1
        for row in rows
    )
    assert all(schedule.checked(row) == row for row in rows)


def test_names_caps_and_timing_projection_are_fixed(monkeypatch):
    assert screen.SERVICE_SECONDS == 1440
    assert screen.CLOSEOUT_SECONDS == 480
    assert screen.LAUNCH_RESERVE == 1980
    assert screen.COLLECTION_SECONDS == 60
    assert screen.CASE_RESERVE_SECONDS == 80
    assert screen.PROJECTED_COLLECTION_SECONDS == pytest.approx(1092.711585)
    assert screen.PROJECTED_REPLAY_SECONDS == pytest.approx(306.720015)
    projection = screen._timing_projection()
    assert projection["previous_collection_source"] == screen.PREVIOUS_COLLECTION_SOURCE
    assert projection["previous_replay_source"] == screen.PREVIOUS_REPLAY_SOURCE
    assert screen.ARTIFACT_SOURCE not in (
        screen.PREVIOUS_COLLECTION_SOURCE,
        screen.PREVIOUS_REPLAY_SOURCE,
    )
    assert (
        screen.service_name(SOURCE, "run")
        == "microduck-cpu-gentle-gap-run-aaaaaaaaaaaa.service"
    )
    assert (
        screen.service_name(SOURCE, "closeout")
        == "microduck-cpu-gentle-gap-closeout-aaaaaaaaaaaa.service"
    )
    with pytest.raises(ValueError):
        screen.service_name(SOURCE, "supervise")

    properties = {
        "MainPID": str(screen.os.getpid()),
        "ActiveState": "active",
        "RuntimeMaxUSec": "24min",
        "MemoryMax": str(2 * 1024**3),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
    }

    def read(*args):
        if args[0:4] == ("systemctl", "--user", "list-units", "--state=running"):
            return screen.service_name(SOURCE, "run")
        return properties[args[-2]]

    monkeypatch.setattr(screen.base.host, "read", read)
    assert screen.service_properties(SOURCE, "run") == properties
    properties["RuntimeMaxUSec"] = "25min"
    with pytest.raises(
        ValueError, match="exact independently capped gentle-gap CPU service"
    ):
        screen.service_properties(SOURCE, "run")


def test_matching_full_screen_is_no_deficit_and_non_admitting():
    result = screen._screen_inputs(synthetic_rows(), ["f" * 64] * 21)
    assert result["decision"] == "cpu-gentle-gap-no-deficit"
    assert result["zero_control_valid"] is True
    assert result["complete_declared_force_phases"] is True
    assert result["prefixes_identical"] is True
    assert result["complete_cases"] == result["cases_expected"] == 21
    assert result["candidate_deficit_cells"] == []
    assert result["promotion_authorized"] is False
    assert all(result[key] is False for key in screen.base.baseline.FALSE_FLAGS)


def test_measured_candidate_failure_excludes_zero_control():
    rows = synthetic_rows()
    rows[3] = synthetic_score(screen.CELL_IDS[3], candidate_pass=False)
    result = screen._screen_inputs(rows, ["a" * 64] * 21)
    assert result["decision"] == "cpu-gentle-gap-candidate-deficit"
    assert result["candidate_deficit_cells"] == [screen.CELL_IDS[3]]


@pytest.mark.parametrize("damage", ["zero", "prefix", "force", "partial", "order"])
def test_invalid_zero_prefix_force_partial_or_order_is_inconclusive(damage):
    rows, prefixes = synthetic_rows(), ["b" * 64] * 21
    if damage == "zero":
        rows[0]["numerical_diagnostic"]["candidate_pass"] = False
    elif damage == "prefix":
        prefixes[-1] = "c" * 64
    elif damage == "force":
        rows[5]["pulse"]["complete_phase_checks"] = False
    elif damage == "partial":
        rows[5] = synthetic_score(screen.CELL_IDS[5], full=False)
        rows[5]["collection"]["stop_reason"] = "wall-budget-exhausted"
        rows[5]["numerical_diagnostic"]["complete_first_attempt"] = False
    else:
        rows[1]["cell"] = "unexpected-cell"
    if damage == "order":
        with pytest.raises(
            ValueError, match="ordered independently scored gentle-gap row"
        ):
            screen._screen_inputs(rows, prefixes)
    else:
        result = screen._screen_inputs(rows, prefixes)
        assert result["decision"] == "cpu-gentle-gap-inconclusive"
        assert result["promotion_authorized"] is False


def test_closed_3338_context_reader_uses_exact_schema_and_pins(monkeypatch, tmp_path):
    pins = _closed_audit_files(tmp_path)
    monkeypatch.setattr(screen, "CLOSED_AUDIT_SHA256", pins)
    result = screen._closed_parent_context(tmp_path)
    assert result["source_identity"]["source"] == screen.ARTIFACT_SOURCE


def test_closed_3338_context_reader_rejects_selective_or_extra_file(
    monkeypatch, tmp_path
):
    pins = _closed_audit_files(tmp_path, selective=True)
    monkeypatch.setattr(screen, "CLOSED_AUDIT_SHA256", pins)
    with pytest.raises(ValueError, match="closed 3338 receipt is complete"):
        screen._closed_parent_context(tmp_path)


def test_actual_closed_3338_mirror_or_native_inventory_matches_three_file_pins():
    from pathlib import Path

    mirror = Path(__file__).resolve().parents[1] / MAC_AUDIT_MIRROR
    native = screen.terminal_repair.output_path(screen.ARTIFACT_SOURCE)
    root = mirror if mirror.exists() else native
    if (
        not root.exists()
        and screen.os.environ.get("MICRODUCK_STANCE_PROFILE") != "wsl-10098-20260930"
    ):
        pytest.skip("authentic closed audit mirror is not present in this checkout")
    assert (
        screen._closed_parent_context(root)["source_identity"]["source"]
        == screen.ARTIFACT_SOURCE
    )


def test_closeout_constructor_has_single_false_flag_owner():
    screening = screen._screen_inputs(synthetic_rows(), ["d" * 64] * 21)
    receipt = screen.closeout_result(
        SOURCE,
        "e" * 64,
        b"synthetic report",
        {"screening": screening},
        {"case-0.pt": {"sha256": "0" * 64, "bytes": 1}},
        {
            "MainPID": "17",
            "ActiveState": "active",
            "RuntimeMaxUSec": "8min",
            "MemoryMax": str(2 * 1024**3),
            "CPUQuotaPerSecUSec": "2s",
            "Nice": "10",
            "KillMode": "control-group",
        },
        1.0,
        {"sample": 1},
        {"sample": 2},
    )
    assert receipt["cases_checked"] == 21
    assert receipt["decision"] == "cpu-gentle-gap-no-deficit"
    assert receipt["whole_cpu_rescore_identical"] is True
    assert receipt["idle_before"] == {"sample": 1}
    assert receipt["idle_after"] == {"sample": 2}
    assert all(receipt[key] is False for key in screen.base.baseline.FALSE_FLAGS)


def test_closeout_constructor_detects_duplicate_false_flag_ownership(monkeypatch):
    screening = screen._screen_inputs(synthetic_rows(), ["f" * 64] * 21)
    monkeypatch.setattr(
        screen.base.baseline,
        "FALSE_FLAGS",
        {**screen.base.baseline.FALSE_FLAGS, "optimizer_steps": False},
    )
    with pytest.raises(ValueError, match="one owner for each false-flag field"):
        screen.closeout_result(
            SOURCE,
            "e" * 64,
            b"synthetic report",
            {"screening": screening},
            {},
            {
                "MainPID": "17",
                "ActiveState": "active",
                "RuntimeMaxUSec": "8min",
                "MemoryMax": str(2 * 1024**3),
                "CPUQuotaPerSecUSec": "2s",
                "Nice": "10",
                "KillMode": "control-group",
            },
            1.0,
            {"sample": 1},
            {"sample": 2},
        )


def test_public_entrypoints_acquire_shared_lease_and_capture_idle_gates(monkeypatch):
    events = []

    @contextmanager
    def lease():
        events.append("lease-enter")
        yield
        events.append("lease-exit")

    idle_samples = iter(({"idle": 1}, {"idle": 2}))
    monkeypatch.setattr(screen.base.files, "gpu_lease", lease)
    monkeypatch.setattr(
        screen.base.host,
        "wait_idle",
        lambda: (events.append("idle"), next(idle_samples))[1],
    )
    monkeypatch.setattr(
        screen,
        "_run_leased",
        lambda source, idle: (
            events.append(("run", source, idle)),
            {"decision": "partial"},
        )[1],
    )
    assert screen.run(SOURCE) == {"decision": "partial"}
    assert events == ["lease-enter", "idle", ("run", SOURCE, {"idle": 1}), "lease-exit"]

    events.clear()
    idle_samples = iter(({"idle": 3}, {"idle": 4}))
    monkeypatch.setattr(
        screen,
        "_closeout_leased",
        lambda source, launch, idle: (
            events.append(("closeout", source, launch, idle)),
            {"decision": "partial"},
        )[1],
    )
    assert screen.closeout(SOURCE, "e" * 64) == {"decision": "partial"}
    assert events == [
        "lease-enter",
        "idle",
        ("closeout", SOURCE, "e" * 64, {"idle": 3}),
        "lease-exit",
    ]


def test_prefix_judgement_follows_full_independent_raw_verification(monkeypatch):
    events = []
    score = synthetic_score(screen.CELL_IDS[0])
    monkeypatch.setattr(
        screen.evidence, "verify", lambda *args: (events.append("verify"), score)[1]
    )
    monkeypatch.setattr(screen.torch, "load", lambda *args, **kwargs: {"trace": True})
    monkeypatch.setattr(
        screen.evidence,
        "prefix_hash",
        lambda *args: (events.append("prefix"), "a" * 64)[1],
    )
    replayed, prefix = screen._case_prefix(b"raw", "0" * 64, b"checkpoint", {}, {})
    assert replayed is score and prefix == "a" * 64
    assert events == ["verify", "prefix"]


@pytest.mark.parametrize(
    "damage", ["launch.json", "source-inventory.json", "receipt.json"]
)
def test_closed_audit_reader_rejects_whole_file_tampering(
    monkeypatch, tmp_path, damage
):
    pins = _closed_audit_files(tmp_path)
    monkeypatch.setattr(screen, "CLOSED_AUDIT_SHA256", pins)
    (tmp_path / damage).write_bytes((tmp_path / damage).read_bytes() + b" ")
    with pytest.raises(ValueError, match="pinned closed 3338 terminal audit bytes"):
        screen._closed_parent_context(tmp_path)


def full_constructor_context():
    """Full recorded schema fixture, not an assertion about this host."""
    audit = {
        "source_identity": {"source": screen.ARTIFACT_SOURCE, "machine_id": "fixture"},
        "cpu_math_profile": {"profile": "fixture"},
        "preserved_filmbrain": {"service": "fixture"},
        "protected_services": {"service": "inactive"},
        "terminal_launch_failure_binding": {"source": "fixture"},
    }
    current = deepcopy(audit)
    current["source_identity"]["source"] = SOURCE
    prereq = {
        "fresh": (
            current,
            {"prior": "fixture"},
            b"parent fixture",
            {"audit": "fixture"},
            {"update": "fixture"},
        ),
        "source_inventory": {"closure": "fixture"},
        "failure_binding": {"failure": "fixture"},
    }
    return audit, current, prereq


def test_actual_full_context_constructor_has_single_profile_and_failure_owner():
    audit, current, prereq = full_constructor_context()
    result = screen._context_record(SOURCE, current, audit, prereq)
    assert result["cpu_math_profile"] == current["cpu_math_profile"]
    assert (
        result["terminal_launch_failure_binding"]
        == current["terminal_launch_failure_binding"]
    )
    assert "parent fixture" not in json.dumps(result)
    assert result["terminal_prerequisites"]["closed_update_binding"] == {
        "update": "fixture"
    }


@pytest.mark.parametrize(
    "damage", ["profile", "filmbrain", "protected", "failure", "fresh"]
)
def test_full_context_constructor_rejects_any_extra_identity_exception(damage):
    audit, current, prereq = full_constructor_context()
    if damage == "fresh":
        prereq["fresh"] = ({**current, "extra": "forbidden"}, *prereq["fresh"][1:])
    else:
        key = {
            "profile": "cpu_math_profile",
            "filmbrain": "preserved_filmbrain",
            "protected": "protected_services",
            "failure": "terminal_launch_failure_binding",
        }[damage]
        audit[key] = {"changed": "forbidden"}
    with pytest.raises(ValueError):
        screen._context_record(SOURCE, current, audit, prereq)


def test_exact_static_inventory_and_lease_metadata():
    assert len(screen.CASE_FILES) == 63
    assert len(screen.PREPARE_FILES) == 66
    assert len(screen.COMPLETE_FILES) == 67
    assert len(screen.COMPLETE_FILES | {"independent-closeout.json"}) == 68
    assert screen._lease_declaration() == {
        "lock_path": str(screen.base.files.LOCK),
        "mechanism": "advisory-flock-exclusive-nonblocking",
        "scope": "unchanged-parent-gentle-gap-screen",
        "idle_gate": "two-idle-samples-before-and-after",
        "cuda_learner": False,
    }
