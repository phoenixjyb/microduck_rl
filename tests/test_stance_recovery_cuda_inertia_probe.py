"""CPU-only supervisor contracts; synthetic guards never admit a native rollout."""

import hashlib
import io
from pathlib import Path
from contextlib import contextmanager
import subprocess
import sys

import pytest
import torch

from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_inertia_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "a" * 40


def properties(mode):
    return {
        "MainPID": "123",
        "ActiveState": "active",
        "RuntimeMaxUSec": f"{probe.SECONDS[mode] // 60}min",
        "MemoryMax": str(probe.MEMORY[mode]),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": "a" * 32,
    }


def test_new_window_boundaries_and_finite_numeric_guards():
    probe.check_window(now=probe.START)
    probe.check_window(now=probe.CUTOFF - 1)
    probe.check_window(now=probe.START, reserve_seconds=probe.RUN_RESERVE)
    probe.check_window(
        now=probe.CUTOFF - probe.RUN_RESERVE - 1, reserve_seconds=probe.RUN_RESERVE
    )
    assert probe.START == 1791187928
    assert probe.CUTOFF == 1791205200
    assert (
        probe.RUN_RESERVE
        == probe.SECONDS["run"] + probe.SECONDS["closeout"] + probe.MARGIN
    )
    for now, reserve in (
        (probe.START - 1, 0),
        (probe.CUTOFF, 0),
        (probe.CUTOFF - probe.RUN_RESERVE, probe.RUN_RESERVE),
        (True, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        (probe.START, True),
        (probe.START, float("nan")),
        (probe.START, float("inf")),
        (probe.START, -1),
        (probe.START, "1"),
    ):
        with pytest.raises(ValueError):
            probe.check_window(now=now, reserve_seconds=reserve)


def test_schedule_is_64_explicit_zero_and_early_x2n_rows():
    value = probe.declaration(SOURCE)
    checked = probe.schedule.checked(value)
    assert checked["source"] == SOURCE
    assert checked["stage"] == "dose" and checked["split"] == "training"
    assert checked["worlds"] == 64
    assert checked["cell_ids"] == ["zero-wrench"] * 32 + checked["cell_ids"][32:]
    early = checked["row_cells"][32]
    assert checked["cell_ids"][32:] == [early["id"]] * 32
    assert early["onset_step"] == 250
    assert early["duration_steps"] == 20
    assert early["force_world_newtons"] == [2.0, 0.0, 0.0]
    assert all(
        row["force_world_newtons"] == [0.0, 0.0, 0.0]
        for row in checked["row_cells"][:32]
    )
    assert probe.SEED == 653 and probe.ATTEMPTS == ("capture", "replay")
    assert probe.ATTEMPTS == tuple(dict.fromkeys(probe.ATTEMPTS))


def test_protocol_declares_capped_no_update_services_and_isolated_child():
    assert probe.PROTOCOL == "football-b1d-cuda64-inertia-probe-20261005-v1"
    assert probe.SECONDS == {
        "preflight": 300,
        "tests": 300,
        "run": 1200,
        "closeout": 600,
        "diagnose": 300,
    }
    assert probe.CHILD_SECONDS == 480
    assert probe.MEMORY == {
        "preflight": 4 * 1024**3,
        "tests": 4 * 1024**3,
        "run": 6 * 1024**3,
        "closeout": 4 * 1024**3,
        "diagnose": 4 * 1024**3,
    }
    assert properties("run")["CPUQuotaPerSecUSec"] == "2s"
    assert properties("run")["Nice"] == "10"
    assert properties("run")["KillMode"] == "control-group"
    assert probe.MODULE == "mjlab_microduck.stance_recovery_cuda_inertia_probe"
    assert probe.RUN_RESERVE == 1860
    assert probe.TEST_FILES[-2:] == (
        "test_stance_recovery_early_inertia_trace.py",
        "test_stance_recovery_cuda_inertia_probe.py",
    )
    assert len(probe.TEST_FILES) == 37
    assert all(value is False for value in probe.FLAGS.values())


def test_launch_descriptor_pins_full_persistent_inertia_contract():
    assert probe.trace_descriptor() == {
        "protocol": probe.early_trace.PROTOCOL,
        "steps": probe.early_trace.early.STEPS,
        "observation_boundary": probe.early_trace.BOUNDARY,
        "excluded_intermediates": list(probe.early_trace.EXCLUDED),
        "persistent_shapes": {
            key: list(value) for key, value in probe.early_trace.SHAPES.items()
        },
    }


@pytest.mark.parametrize("mode", list(probe.SECONDS))
def test_exact_recorded_service_properties(mode):
    assert probe._recorded_properties(properties(mode), mode) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("MainPID", "0"),
        ("MainPID", True),
        ("ActiveState", "inactive"),
        ("RuntimeMaxUSec", "19min"),
        ("MemoryMax", str(8 * 1024**3)),
        ("CPUQuotaPerSecUSec", "3s"),
        ("Nice", "0"),
        ("KillMode", "process"),
        ("InvocationID", "a" * 31),
    ],
)
def test_changed_recorded_service_properties_are_rejected(field, value):
    state = properties("run") | {field: value}
    with pytest.raises(ValueError):
        probe._recorded_properties(state, "run")


def test_service_source_and_mode_names_remain_source_bound():
    assert probe.output_path(SOURCE).name == "stance-cuda64-inertia-" + SOURCE[:12]
    assert (
        probe.unit(SOURCE, "run")
        == f"microduck-cuda64-inertia-run-{SOURCE[:12]}.service"
    )
    assert (
        probe.tool_path(SOURCE, "tests").name == f"cuda64-inertia-tests-{SOURCE[:12]}"
    )
    with pytest.raises(ValueError):
        probe.output_path("bad")
    with pytest.raises(ValueError):
        probe.unit(SOURCE, "train")


def test_prepared_summary_is_precollection_and_preserves_actual_preparation_metadata():
    raw = b"actual retained preparation payload bytes"
    metadata = {
        "private_cuda_rng_state_sha256": "b" * 64,
        "protocol": "existing-preparation-summary-v1",
        "source": SOURCE,
        "seed": probe.SEED,
        "worlds": 64,
        "actor_state_sha256": "c" * 64,
    }
    launch = {
        "cpu_parent_receipt_file_sha256": "d" * 64,
        "cpu_parent_receipt_canonical_sha256": "e" * 64,
    }
    summary = probe._prepared_summary(SOURCE, "f" * 64, raw, metadata, launch)
    assert summary["metadata"] == metadata
    assert summary["payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["payload_bytes"] == len(raw)
    assert (
        summary["private_cuda_rng_state_sha256"]
        == metadata["private_cuda_rng_state_sha256"]
    )
    assert summary["seed"] == probe.SEED
    assert summary["optimizer_steps"] == summary["simulator_resets"] == 0
    assert summary["training_update_performed"] is False
    assert summary["action_sampling"] is False
    assert summary["return_computation"] is False
    assert summary["rollout_collection"] is False
    assert summary["cuda_child_claims_cpu_provenance_authenticated"] is False
    assert all(summary[key] is False for key in probe.base.PREPARATION_FALSE_FLAGS)


def test_committed_source_binding_hashes_every_exact_new_leaf_from_source(
    tmp_path, monkeypatch
):
    repo = tmp_path
    for relative in probe.OWN_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("committed:" + relative).encode())
    module_file = repo / "src/mjlab_microduck/stance_recovery_cuda_inertia_probe.py"
    monkeypatch.setattr(probe, "__file__", str(module_file))
    monkeypatch.setattr(
        probe.shadow, "source_binding", lambda _source: {"frozen": True}
    )
    calls = []

    def show(command, **kwargs):
        if command[1] == "merge-base":
            calls.append(("merge-base", kwargs["cwd"]))
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[1] == "diff":
            calls.append(("diff", kwargs["cwd"]))
            raw = ("\n".join(probe.ALLOWED) + "\n").encode()
            return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")
        relative = command[2].split(":", 1)[1]
        calls.append((relative, kwargs["cwd"]))
        return subprocess.CompletedProcess(
            command, 0, stdout=(repo / relative).read_bytes(), stderr=b""
        )

    monkeypatch.setattr(probe.subprocess, "run", show)
    result = probe.source_binding(SOURCE)
    assert result["previous_frozen_source"] == {"frozen": True}
    assert set(result["inertia_leaves"]) == set(probe.OWN_FILES)
    assert all(
        result["inertia_leaves"][name]
        == hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in probe.OWN_FILES
    )
    assert sum(relative not in ("merge-base", "diff") for relative, _ in calls) == (
        len(probe.OWN_FILES) + len(probe.FROZEN_FILES)
    )
    assert all(cwd == repo for _, cwd in calls)


def test_committed_source_binding_rejects_leaf_bytes_different_from_worktree(
    tmp_path, monkeypatch
):
    for relative in probe.OWN_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"current leaf")
    module_file = tmp_path / "src/mjlab_microduck/stance_recovery_cuda_inertia_probe.py"
    monkeypatch.setattr(probe, "__file__", str(module_file))
    monkeypatch.setattr(probe.shadow, "source_binding", lambda _source: {})

    def show(command, **kwargs):
        if command[1] == "merge-base":
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[1] == "diff":
            raw = ("\n".join(probe.ALLOWED) + "\n").encode()
            return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")
        relative = command[2].split(":", 1)[1]
        raw = (
            b"different committed bytes"
            if relative == probe.OWN_FILES[0]
            else (tmp_path / relative).read_bytes()
        )
        return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")

    monkeypatch.setattr(probe.subprocess, "run", show)
    with pytest.raises(ValueError, match="exact new source leaf"):
        probe.source_binding(SOURCE)


def test_source_delta_only_allows_new_inertia_supervisor_module_test_and_doc(
    tmp_path, monkeypatch
):
    for relative in probe.OWN_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    monkeypatch.setattr(
        probe,
        "__file__",
        str(tmp_path / "src/mjlab_microduck/stance_recovery_cuda_inertia_probe.py"),
    )
    monkeypatch.setattr(probe.shadow, "source_binding", lambda _source: {})

    def show(command, **kwargs):
        if command[1] == "merge-base":
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[1] == "diff":
            changed = [
                probe.ALLOWED[0],
                "src/mjlab_microduck/stance_recovery_early_inertia_trace.py",
            ]
            return subprocess.CompletedProcess(
                command, 0, stdout=("\n".join(changed) + "\n").encode(), stderr=b""
            )
        relative = command[2].split(":", 1)[1]
        return subprocess.CompletedProcess(
            command, 0, stdout=(tmp_path / relative).read_bytes(), stderr=b""
        )

    monkeypatch.setattr(probe.subprocess, "run", show)
    with pytest.raises(
        ValueError, match="only newly declared inertia supervisor changes"
    ):
        probe.source_binding(SOURCE)


def test_frozen_v3_supervisor_and_early_inertia_reader_stay_guarded():
    assert probe.BASE_SOURCE.startswith("c4fb6062")
    assert (
        "src/mjlab_microduck/stance_recovery_cuda_rollout_probe.py"
        in probe.FROZEN_FILES
    )
    assert "tests/test_stance_recovery_cuda_rollout_probe.py" in probe.FROZEN_FILES
    assert (
        "src/mjlab_microduck/stance_recovery_early_inertia_trace.py"
        in probe.FROZEN_FILES
    )
    assert "tests/test_stance_recovery_early_inertia_trace.py" in probe.FROZEN_FILES
    assert set(probe.ALLOWED) == {
        "src/mjlab_microduck/stance_recovery_cuda_inertia_probe.py",
        "tests/test_stance_recovery_cuda_inertia_probe.py",
        "docs/experiments/2026-10-05-cuda64-no-update-rollout.md",
    }


def test_source_binding_rejects_changed_frozen_v3_or_inertia_reader(
    tmp_path, monkeypatch
):
    for relative in probe.OWN_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    monkeypatch.setattr(
        probe,
        "__file__",
        str(tmp_path / "src/mjlab_microduck/stance_recovery_cuda_inertia_probe.py"),
    )
    monkeypatch.setattr(probe.shadow, "source_binding", lambda _source: {})
    protected = "src/mjlab_microduck/stance_recovery_cuda_rollout_probe.py"

    def show(command, **kwargs):
        if command[1] == "merge-base":
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[1] == "diff":
            raw = ("\n".join(probe.ALLOWED) + "\n").encode()
            return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")
        revision, relative = command[2].split(":", 1)
        raw = (tmp_path / relative).read_bytes()
        if revision == probe.BASE_SOURCE and relative == protected:
            raw = b"mutated frozen v3 supervisor"
        return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")

    monkeypatch.setattr(probe.subprocess, "run", show)
    with pytest.raises(
        ValueError, match="unchanged frozen inertia observer and prior readers"
    ):
        probe.source_binding(SOURCE)


def test_whole_launch_hash_is_verified_before_json_parse(monkeypatch, tmp_path):
    launch = dict(
        protocol=probe.PROTOCOL,
        source=SOURCE,
        source_binding={},
        native_prerequisites={},
        closed_preparation={},
        preflight_binding={},
        tests_binding={},
        service_properties=properties("run"),
        idle_before={},
        compiled_plant={},
        cpu_parent_receipt_file_sha256="b" * 64,
        cpu_parent_receipt_canonical_sha256="c" * 64,
        cpu_parent_binding={},
        optimizer_steps=0,
        return_computation=False,
        constructor_rng=dict(
            protocol=probe.constructor.PROTOCOL,
            cpu_seed=probe.constructor.CPU_SEED,
            cuda_seed=probe.constructor.CUDA_SEED,
        ),
        early_forward_trace=probe.trace_descriptor(),
        schedule=probe.declaration(SOURCE),
        service_seconds=probe.SECONDS,
        service_memory_bytes=probe.MEMORY,
        child_seconds=probe.CHILD_SECONDS,
        cutoff_unix=probe.CUTOFF,
        attempts=list(probe.ATTEMPTS),
        **probe.FLAGS,
    )
    raw = (canonical(launch) + "\n").encode()
    monkeypatch.setattr(probe, "output_path", lambda _source: tmp_path)
    events = []
    monkeypatch.setattr(probe.base, "_read_file", lambda *args: raw)
    digest = probe.base.digest
    monkeypatch.setattr(
        probe.base, "digest", lambda value: (events.append("hash"), digest(value))[1]
    )
    parse = probe.base.parse_json
    monkeypatch.setattr(
        probe.base,
        "parse_json",
        lambda value: (events.append("parse"), parse(value))[1],
    )
    assert probe._read_launch(SOURCE, digest(raw)) == launch
    assert events == ["hash", "parse"]
    events.clear()
    with pytest.raises(ValueError, match="whole new launch bytes before parse"):
        probe._read_launch(SOURCE, "f" * 64)
    assert events == ["hash"]
    launch["unexpected"] = True
    raw = (canonical(launch) + "\n").encode()
    with pytest.raises(ValueError, match="exact new non-admitting launch"):
        probe._read_launch(SOURCE, digest(raw))


def test_launch_rejects_changed_inertia_descriptor_before_any_tensor_load(
    monkeypatch, tmp_path
):
    launch = dict(
        protocol=probe.PROTOCOL,
        source=SOURCE,
        source_binding={},
        native_prerequisites={},
        closed_preparation={},
        preflight_binding={},
        tests_binding={},
        service_properties=properties("run"),
        idle_before={},
        schedule=probe.declaration(SOURCE),
        compiled_plant={},
        cpu_parent_receipt_file_sha256="b" * 64,
        cpu_parent_receipt_canonical_sha256="c" * 64,
        cpu_parent_binding={},
        attempts=list(probe.ATTEMPTS),
        service_seconds=probe.SECONDS,
        service_memory_bytes=probe.MEMORY,
        child_seconds=probe.CHILD_SECONDS,
        cutoff_unix=probe.CUTOFF,
        optimizer_steps=0,
        return_computation=False,
        constructor_rng=dict(
            protocol=probe.constructor.PROTOCOL,
            cpu_seed=probe.constructor.CPU_SEED,
            cuda_seed=probe.constructor.CUDA_SEED,
        ),
        early_forward_trace=probe.trace_descriptor(),
        **probe.FLAGS,
    )
    launch["early_forward_trace"]["persistent_shapes"]["qM"] = [20, 19]
    raw = (canonical(launch) + "\n").encode()
    monkeypatch.setattr(probe, "output_path", lambda _source: tmp_path)
    monkeypatch.setattr(probe.base, "_read_file", lambda *args: raw)
    monkeypatch.setattr(
        probe.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("descriptor rejected before tensor load"),
    )
    with pytest.raises(ValueError, match="exact new non-admitting launch"):
        probe._read_launch(SOURCE, probe.base.digest(raw))


def test_record_diagnostics_are_explicit_retained_reductions_not_thermal_model():
    frame = {
        "state": {
            "tilt": torch.tensor([0.1, 0.2]),
            "root_velocity": torch.tensor([[3.0, 4.0, 8.0], [0.0, 0.0, 0.0]]),
            "torque": torch.tensor([[2.0], [-3.0]]),
            "joint_velocity": torch.tensor([[4.0], [5.0]]),
        },
        "soft_limit_mask": torch.tensor([[True], [False]]),
    }
    record = {
        "runtime_result_before_reset": {"boundaries": [frame, frame]},
        "pre_reset_steps": torch.tensor([9, 10]),
        "terminated": torch.tensor([True, False]),
        "timed_out": torch.tensor([False, False]),
        "environment_reward": torch.tensor([1.0, -2.0]),
    }
    result = probe._diagnostics(
        {"initial_frame": frame, "archive": {"records": [record]}}
    )
    assert result == {
        "records": 1,
        "min_pre_reset_step": 9,
        "max_pre_reset_step": 10,
        "terminal_rows": 1,
        "timeout_rows": 0,
        "maximum_tilt_rad": pytest.approx(0.2),
        "maximum_planar_speed_mps": 5.0,
        "maximum_abs_motor_torque_nm": 3.0,
        "maximum_abs_joint_power_w": 15.0,
        "soft_limit_row_exposure": 0.5,
        "summed_environment_reward": -1.0,
        "motor_thermal_model_evaluated": False,
    }


def test_child_inherited_lease_failure_precedes_cuda_or_launch(monkeypatch):
    monkeypatch.setattr(
        probe.base.training_smoke,
        "inherited_lease",
        lambda _fd: (_ for _ in ()).throw(
            ValueError("SYNTHETIC inherited lease missing")
        ),
    )
    monkeypatch.setattr(
        probe, "_read_launch", lambda *_: pytest.fail("launch before lease")
    )
    monkeypatch.setattr(
        torch.cuda, "is_initialized", lambda: pytest.fail("CUDA before lease")
    )
    with pytest.raises(ValueError, match="inherited lease missing"):
        probe.child(SOURCE, "b" * 64, "capture", 11)


def test_first_terminal_stops_collection_prefix():
    # Uses the same predicate as the child loop; row liveness does not matter.
    terminal = {
        "terminated": torch.zeros(64, dtype=torch.bool),
        "timed_out": torch.zeros(64, dtype=torch.bool),
    }
    assert probe._terminal_observed(terminal) is False
    terminal["terminated"][7] = True
    assert probe._terminal_observed(terminal) is True
    terminal["terminated"].zero_()
    terminal["timed_out"][12] = True
    assert probe._terminal_observed(terminal) is True


def test_actual_cpu_runtime_snapshot_state_is_normalized_and_owned():
    # Actual CPU MuJoCo-Warp producer shape, not CUDA or native qualification.
    declaration = probe.schedule.declaration(
        SOURCE, "dose", "training", ["zero-wrench"] * 2
    )
    env = probe.constructor.ScheduledRecoveryRuntime(
        declaration, device="cpu", solved_field_check="packed"
    )
    snapshot = env.snapshot()
    assert type(snapshot["state"]) is probe.PhysicsState
    nested = {"initial": snapshot, "boundaries": [snapshot], "params": (torch.ones(2),)}
    owned = archive._owned_tree(probe._producer_tree(nested), allow_cuda=False)
    assert type(owned["initial"]["state"]) is dict
    assert type(owned["boundaries"][0]["state"]) is dict
    assert type(owned["params"]) is tuple
    assert torch.equal(owned["initial"]["state"]["tilt"], snapshot["state"].tilt)
    snapshot["state"].tilt.fill_(123.0)
    assert not (owned["initial"]["state"]["tilt"] == 123.0).any()
    assert owned["initial"]["state"]["torque"].shape == (2, 14)


def test_producer_conversion_does_not_admit_other_objects_or_unbounded_trees():
    from dataclasses import dataclass

    @dataclass
    class Other:
        state: torch.Tensor

    with pytest.raises(ValueError, match="safe archive leaf"):
        archive._owned_tree(
            probe._producer_tree(Other(torch.ones(1))), allow_cuda=False
        )
    deep = None
    for _ in range(34):
        deep = [deep]
    with pytest.raises(ValueError, match="bounded producer tree"):
        probe._producer_tree(deep)


@pytest.mark.parametrize(
    "field,value",
    [
        ("terminated", torch.zeros(64, dtype=torch.int64)),
        ("timed_out", torch.zeros(63, dtype=torch.bool)),
    ],
)
def test_first_terminal_predicate_rejects_malformed_masks(field, value):
    row = {
        "terminated": torch.zeros(64, dtype=torch.bool),
        "timed_out": torch.zeros(64, dtype=torch.bool),
    }
    row[field] = value
    with pytest.raises(ValueError, match="retained typed first-terminal mask"):
        probe._terminal_observed(row)


def test_decision_keeps_early_matched_terminal_as_retained_incomplete():
    pair = {"paired_semantics_exact": True}
    scores = [
        {
            "attempt": "capture",
            "complete_28_call_gate": False,
            "complete_nonzero_pulse_gate": False,
            "whole_bytes_verified_before_load": True,
            "individually_cpu_validated": True,
        },
        {
            "attempt": "replay",
            "complete_28_call_gate": False,
            "complete_nonzero_pulse_gate": False,
            "whole_bytes_verified_before_load": True,
            "individually_cpu_validated": True,
        },
    ]
    result = probe._decision(scores, pair)
    assert result == {
        "short_window_complete_and_consistent": False,
        "decision": "retained-incomplete-short-window",
        "native_rollout_qualified": False,
        "training_admitted": False,
    }
    complete = [
        {**score, "complete_28_call_gate": True, "complete_nonzero_pulse_gate": True}
        for score in scores
    ]
    assert (
        probe._decision(complete, pair)["decision"] == "complete-no-update-short-window"
    )
    assert probe._decision(complete, pair)["native_rollout_qualified"] is False


@pytest.mark.parametrize("damage", ["order", "pair"])
def test_decision_rejects_unordered_or_unmatched_capture_replay(damage):
    scores = [
        {"attempt": "capture"},
        {"attempt": "replay"},
    ]
    pair = {"paired_semantics_exact": True}
    if damage == "order":
        scores.reverse()
    else:
        pair["paired_semantics_exact"] = False
    with pytest.raises(ValueError, match="ordered independently checked pair"):
        probe._decision(scores, pair)


def test_new_body_inventory_uses_256mib_without_widening_preparation_limit(
    tmp_path, monkeypatch
):
    names = {
        "capture.pt",
        "replay.pt",
        "capture.prepared.pt",
        "checkpoint.pt",
        "capture.json",
        "capture.log",
    }
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    limits = {}

    def read(path, limit):
        limits[Path(path).name] = limit
        return Path(path).read_bytes()

    monkeypatch.setattr(probe.base, "_read_file", read)
    result = probe._inventory(tmp_path, names)
    assert set(result) == names
    assert limits["capture.pt"] == limits["replay.pt"] == archive.LIMIT == 256 * 1024**2
    assert (
        limits["capture.prepared.pt"]
        == limits["checkpoint.pt"]
        == probe.base.RAW_LIMIT
        == 8 * 1024**2
    )
    assert limits["capture.log"] == probe.base.LOG_LIMIT
    assert limits["capture.json"] == probe.base.JSON_LIMIT


def test_partial_inventory_marks_oversized_artifacts_unreadable(tmp_path, monkeypatch):
    (tmp_path / "capture.pt").write_bytes(b"capture")
    monkeypatch.setattr(probe, "_limit", lambda _name: 4)
    result = probe._inventory(tmp_path)
    assert result["capture.pt"] == {"readable": False}


def test_constructor_fault_receipt_is_retained_without_success_claim(tmp_path):
    receipt = {
        "status": "fault",
        "constructor_calls": 1,
        "faulted": True,
        "error": "retained construction failure",
        "private_end": torch.arange(8, dtype=torch.uint8),
    }
    probe._retain_constructor(tmp_path, SOURCE, "b" * 64, "capture", receipt)
    raw = (tmp_path / "capture.constructor.pt").read_bytes()
    summary = probe.base.parse_json(
        (tmp_path / "capture.constructor.json").read_bytes()
    )
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    assert len(raw) <= probe.base.RAW_LIMIT
    assert probe.evidence._equal(loaded, receipt)
    assert summary["status"] == "fault" and summary["faulted"] is True
    assert summary["source"] == SOURCE and summary["attempt"] == "capture"
    assert summary["payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["payload_bytes"] == len(raw)
    assert all(summary[key] is False for key in probe.FLAGS)
    with pytest.raises(FileExistsError):
        probe._retain_constructor(tmp_path, SOURCE, "b" * 64, "capture", receipt)


@pytest.mark.parametrize("damage", ["hash", "extra", "boolean_count"])
def test_constructor_authentication_refuses_damage_before_cpu_load(
    tmp_path, monkeypatch, damage
):
    raw = b"not a loadable constructor"
    summary = dict(
        protocol=probe.PROTOCOL + ":constructor",
        source=SOURCE,
        launch_sha256="b" * 64,
        attempt="capture",
        payload_sha256=hashlib.sha256(raw).hexdigest(),
        payload_bytes=len(raw),
        status="success",
        constructor_calls=1,
        faulted=False,
        **probe.FLAGS,
    )
    if damage == "hash":
        summary["payload_sha256"] = "c" * 64
    elif damage == "extra":
        summary["unreviewed"] = True
    else:
        summary["constructor_calls"] = True
    files = {
        "cpu-parent-receipt.json": canonical({}).encode(),
        "capture.json": canonical({}).encode(),
        "capture.pt": b"body",
        "capture.prepared.pt": b"preparation",
        "capture.prepared.json": canonical({}).encode(),
        "capture.constructor.json": canonical(summary).encode(),
        "capture.constructor.pt": raw,
    }
    monkeypatch.setattr(probe.base, "_read_file", lambda path, _limit: files[path.name])

    def forbidden(*_args, **_kwargs):
        pytest.fail("unauthenticated bytes must not reach a loader or scorer")

    monkeypatch.setattr(probe.torch, "load", forbidden)
    monkeypatch.setattr(probe.evidence, "verify", forbidden)
    with pytest.raises(ValueError, match="whole successful constructor payload"):
        probe._score(tmp_path, SOURCE, {}, "b" * 64)


def test_faulted_trace_is_exclusively_retained_with_truthful_status(tmp_path):
    receipt = dict(status="faulted", events=[{"qM": torch.zeros(2, 20, 20)}])
    probe._retain_trace(tmp_path, SOURCE, "b" * 64, "capture", receipt)
    raw = (tmp_path / "capture.early-trace.pt").read_bytes()
    summary = probe.base.parse_json(
        (tmp_path / "capture.early-trace.json").read_bytes()
    )
    assert summary["payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["status"] == "faulted" and summary["events"] == 1
    assert len(raw) <= probe.base.RAW_LIMIT
    assert probe.evidence._equal(
        torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True), receipt
    )
    assert all(summary[k] is False for k in probe.FLAGS)
    with pytest.raises(FileExistsError):
        probe._retain_trace(tmp_path, SOURCE, "b" * 64, "capture", receipt)


@pytest.mark.parametrize("damage", ["hash", "extra", "boolean_count"])
def test_early_trace_authentication_refuses_damage_before_trace_load(
    tmp_path, monkeypatch, damage
):
    """Synthetic envelope seam; no native or body acceptance claim."""
    constructor = {"retained": True}
    ctor_raw = probe.evidence.encode(constructor)
    body_raw, prep_raw, trace_raw = b"body", b"prep", b"not loadable trace"
    sha = "b" * 64
    child = dict(
        protocol=probe.PROTOCOL + ":child",
        source=SOURCE,
        launch_sha256=sha,
        seed=probe.SEED,
        attempt="capture",
        payload_bytes=len(body_raw),
        preparation_bytes=len(prep_raw),
        optimizer_steps=0,
        records=1,
        payload_sha256=hashlib.sha256(body_raw).hexdigest(),
        preparation_sha256=hashlib.sha256(prep_raw).hexdigest(),
        **probe.FLAGS,
    )
    ctor = dict(
        protocol=probe.PROTOCOL + ":constructor",
        source=SOURCE,
        launch_sha256=sha,
        attempt="capture",
        status="success",
        constructor_calls=1,
        faulted=False,
        payload_sha256=hashlib.sha256(ctor_raw).hexdigest(),
        payload_bytes=len(ctor_raw),
        **probe.FLAGS,
    )
    summary = dict(
        protocol=probe.PROTOCOL + ":early-trace",
        source=SOURCE,
        launch_sha256=sha,
        attempt="capture",
        status="complete",
        events=6,
        payload_sha256=hashlib.sha256(trace_raw).hexdigest(),
        payload_bytes=len(trace_raw),
        **probe.FLAGS,
    )
    if damage == "hash":
        summary["payload_sha256"] = "c" * 64
    elif damage == "extra":
        summary["unreviewed"] = True
    else:
        summary["events"] = True
    files = {
        "cpu-parent-receipt.json": b"{}",
        "capture.json": canonical(child).encode(),
        "capture.pt": body_raw,
        "capture.prepared.pt": prep_raw,
        "capture.prepared.json": b"{}",
        "capture.constructor.json": canonical(ctor).encode(),
        "capture.constructor.pt": ctor_raw,
        "capture.early-trace.pt": trace_raw,
        "capture.early-trace.json": canonical(summary).encode(),
    }
    monkeypatch.setattr(probe.base, "_read_file", lambda path, _limit: files[path.name])
    value = {"constructor_receipt": constructor, "archive": {"records": [{}]}}
    monkeypatch.setattr(probe.evidence, "verify", lambda *_args, **_kwargs: (value, {}))
    loads = []

    def load(stream, **kwargs):
        actual = stream.getvalue()
        assert actual == ctor_raw, "unauthenticated trace reached torch.load"
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        loads.append(actual)
        return constructor

    monkeypatch.setattr(probe.torch, "load", load)
    with pytest.raises(ValueError, match="whole complete early trace before CPU load"):
        probe._score_inputs(tmp_path, SOURCE, {}, sha)
    assert loads == [ctor_raw]


@pytest.mark.parametrize("damage", [None, "pid", "restart", "invocation", "success"])
def test_original_failed_terminal_requires_exact_retained_invocation(
    monkeypatch, damage
):
    invocation = "d" * 32
    fields = dict(
        MainPID="0",
        ActiveState="failed",
        NRestarts="0",
        ExecMainStatus="1",
        Result="exit-code",
        InvocationID=invocation,
    )
    if damage == "pid":
        fields["MainPID"] = "123"
    elif damage == "restart":
        fields["NRestarts"] = "1"
    elif damage == "invocation":
        fields["InvocationID"] = ""
    elif damage == "success":
        fields.update(ActiveState="inactive", ExecMainStatus="0", Result="success")
    monkeypatch.setattr(probe.base.host, "read", lambda *args: fields[args[-2]])
    if damage is None:
        assert probe._failed_terminal(SOURCE, invocation) == fields
    else:
        with pytest.raises(ValueError, match="exact terminal original failed run"):
            probe._failed_terminal(SOURCE, invocation)


def test_strict_pair_never_ignores_early_trace_divergence(monkeypatch):
    monkeypatch.setattr(
        probe.evidence, "paired", lambda *_: {"paired_semantics_exact": True}
    )
    with pytest.raises(ValueError, match="paired persistent early trace exactness"):
        probe._strict_pair([{}, {}], [{"qM": torch.zeros(1)}, {"qM": torch.ones(1)}])
    assert probe._strict_pair([{}, {}], [{}, {}])["early_trace_exact"] is True


@pytest.mark.parametrize("signed_zero", [False, True])
def test_strict_pair_keeps_persistent_inertia_raw_bytes_exact(monkeypatch, signed_zero):
    monkeypatch.setattr(
        probe.evidence, "paired", lambda *_: {"paired_semantics_exact": True}
    )
    left = {"events": [{"persistent": {"crb": torch.zeros(2, 16, 10)}}]}
    right = {"events": [{"persistent": {"crb": torch.zeros(2, 16, 10)}}]}
    right["events"][0]["persistent"]["crb"][0, 0, 0] = -0.0 if signed_zero else 0.125
    with pytest.raises(ValueError, match="paired persistent early trace exactness"):
        probe._strict_pair([{}, {}], [left, right])


@pytest.mark.parametrize("recomputed", ["same", "different", "success"])
def test_failure_diagnosis_is_distinct_from_success_and_preserves_original_files(
    tmp_path, monkeypatch, recomputed
):
    """Synthetic orchestration seam, not independent native acceptance."""
    error = "paired rollout semantic state exactness"
    inventory = {
        name: {"bytes": 2, "sha256": "e" * 64} for name in probe.COMPLETE_FILES
    }
    report = dict(
        protocol=probe.PROTOCOL,
        source=SOURCE,
        status="failed-retained",
        error_type="ValueError",
        error=error,
        partial_inventory={k: v for k, v in inventory.items() if k != "report.json"},
        **probe.FLAGS,
    )
    raw = canonical(report).encode()
    (tmp_path / "report.json").write_bytes(raw)
    (tmp_path / "launch.json").write_bytes(b"{}")
    prior = {"launch_binding": {}}
    launch = dict(
        service_properties=properties("run"),
        closed_preparation={},
        source_binding={},
        schedule={},
    )
    monkeypatch.setattr(probe, "_hidden", lambda: None)
    monkeypatch.setattr(probe, "check_window", lambda **_: None)
    monkeypatch.setattr(probe, "output_path", lambda _source: tmp_path)
    monkeypatch.setattr(probe, "_properties", lambda _source, mode: properties(mode))
    monkeypatch.setattr(probe.base, "_exact_inventory", lambda *_: None)
    monkeypatch.setattr(probe, "_inventory", lambda *_: inventory)
    monkeypatch.setattr(probe, "_read_launch", lambda *_: launch)
    monkeypatch.setattr(probe, "source_binding", lambda *_: {})
    monkeypatch.setattr(probe, "_failed_terminal", lambda *_: {"original": True})
    monkeypatch.setattr(probe.closed, "checked", lambda *_, **__: prior)
    modes = []
    monkeypatch.setattr(probe, "_read_tool", lambda _, mode, _prior: modes.append(mode))
    monkeypatch.setattr(probe.base.gpu_idle_gate, "wait_idle", lambda: {"idle": True})

    @contextmanager
    def leased():
        yield 42

    monkeypatch.setattr(probe.base.gap.base.files, "gpu_lease", leased)
    inputs = [{"archive": {"records": [{}]}}, {"archive": {"records": [{}]}}]
    monkeypatch.setattr(probe, "_score_inputs", lambda *_: (inputs, [{}, {}], [{}, {}]))

    def pair(*_):
        if recomputed == "success":
            return {}
        raise ValueError(
            error if recomputed == "same" else "paired persistent early trace exactness"
        )

    monkeypatch.setattr(probe, "_strict_pair", pair)
    monkeypatch.setattr(probe.early_trace, "compare", lambda *_: {"exact": False})
    target = tmp_path / "independent-failure-diagnosis.json"
    if recomputed == "same":
        result = probe.diagnose(SOURCE)
        assert target.exists()
        assert result["pair_accepted"] is False and result["pair_error"] == error
        assert result["status"] == "failed-pair-independently-diagnosed"
        assert all(result[k] is False for k in probe.FLAGS)
    else:
        with pytest.raises(
            ValueError, match="independently reproduced exact strict pair failure"
        ):
            probe.diagnose(SOURCE)
        assert not target.exists()
    assert modes == ["preflight", "tests"]
    assert (tmp_path / "report.json").read_bytes() == raw
    assert (tmp_path / "launch.json").read_bytes() == b"{}"
    assert not (tmp_path / "independent-closeout.json").exists()


def _run(command, log, *, seconds, monkeypatch, log_limit=None):
    captured = []
    original_popen = probe.subprocess.Popen
    monkeypatch.setattr(probe.base.execution, "ROOT", log.parent)

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        captured.append(child)
        return child

    monkeypatch.setattr(probe.subprocess, "Popen", spawn)
    if log_limit is not None:
        monkeypatch.setattr(probe.base, "LOG_LIMIT", log_limit)
    killed = []
    original_kill = probe.base.campaign._kill_owned_group

    def kill(child):
        killed.append(child.pid)
        original_kill(child)

    monkeypatch.setattr(probe.base.campaign, "_kill_owned_group", kill)
    with pytest.raises((ValueError, TimeoutError)):
        probe._run_process(command, log, seconds, env={"PATH": "/usr/bin:/bin"})
    assert len(captured) == 1
    assert killed == [captured[0].pid]
    assert captured[0].poll() is not None
    assert log.is_file()


def test_streaming_process_enforces_log_cap_and_kills_only_owned_child(
    tmp_path, monkeypatch
):
    log = tmp_path / "oversize.log"
    _run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x'*1000000); sys.stdout.flush()",
        ],
        log,
        seconds=3,
        monkeypatch=monkeypatch,
        log_limit=32,
    )
    assert log.stat().st_size <= 32


def test_streaming_process_timeout_reaps_only_owned_child(tmp_path, monkeypatch):
    log = tmp_path / "timeout.log"
    _run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        log,
        seconds=0.1,
        monkeypatch=monkeypatch,
    )


def _source_sync_run(
    tmp_path,
    *,
    units,
    head=None,
    branch="feat/athletics-obstacle-curriculum",
    dirty="",
    remote=None,
    root=None,
    origin=None,
):
    """Run only the returned bootstrap Bash against local Git/systemctl stubs."""
    source = "f" * 40
    stub_bin = tmp_path / "bin"
    state = tmp_path / "state"
    stub_bin.mkdir()
    state.mkdir()
    current_head = probe.SYNC_FROM_SOURCE if head is None else head
    incoming_head = source if remote is None else remote
    (state / "head").write_text(current_head + "\n")
    (state / "incoming").write_text(incoming_head + "\n")
    (state / "branch").write_text(branch + "\n")
    (state / "dirty").write_text(dirty)
    (state / "units").write_text(units)
    (state / "root").write_text((probe.SYNC_ROOT if root is None else root) + "\n")
    (state / "origin").write_text(
        (probe.SYNC_ORIGIN if origin is None else origin) + "\n"
    )
    git_script = """#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\\n' "$*" >> "$SYNC_LOG"
case "$1" in
  status)
    cat "$SYNC_STATE/dirty"
    ;;
  rev-parse)
    case "$2" in
      --show-toplevel) cat "$SYNC_STATE/root" ;;
      HEAD) cat "$SYNC_STATE/head" ;;
      refs/remotes/origin/feat/athletics-obstacle-curriculum)
        test -f "$SYNC_STATE/tracked-ref" || exit 93
        cat "$SYNC_STATE/tracked-ref"
        ;;
      *) exit 91 ;;
    esac
    ;;
  remote)
    test "$2" = get-url && test "$3" = origin
    cat "$SYNC_STATE/origin"
    ;;
  branch) cat "$SYNC_STATE/branch" ;;
  fetch)
    test "$2" = origin
    test "$3" = refs/heads/feat/athletics-obstacle-curriculum:refs/remotes/origin/feat/athletics-obstacle-curriculum
    cat "$SYNC_STATE/incoming" > "$SYNC_STATE/tracked-ref"
    touch "$SYNC_STATE/fetched"
    ;;
  merge)
    touch "$SYNC_STATE/merged"
    test "$2" = --ff-only
    printf '%s\\n' "$3" > "$SYNC_STATE/head"
    ;;
  *) exit 92 ;;
esac
"""
    systemctl_script = """#!/usr/bin/env bash
set -euo pipefail
printf 'systemctl %s\\n' "$*" >> "$SYNC_LOG"
cat "$SYNC_STATE/units"
"""
    git_path = stub_bin / "git"
    service_path = stub_bin / "systemctl"
    git_path.write_text(git_script)
    service_path.write_text(systemctl_script)
    git_path.chmod(0o755)
    service_path.chmod(0o755)
    env = {
        "PATH": f"{stub_bin}:/usr/bin:/bin",
        "SYNC_LOG": str(tmp_path / "calls.log"),
        "SYNC_STATE": str(state),
    }
    process = subprocess.run(
        ["bash", "-c", probe.source_sync_script(source)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    log_path = tmp_path / "calls.log"
    log = log_path.read_text() if log_path.exists() else ""
    return process, log, state, source


def _own_sync_unit(source):
    return probe.source_sync_unit(source) + " loaded active running\n"


def test_source_sync_bootstrap_unit_is_exact_hex_bound_and_cpu_capped():
    source = "f" * 40
    assert probe.SYNC_FROM_SOURCE == "af47d912501a3df7db7f9a9ab7af6c09ba63a215"
    assert probe.SYNC_SECONDS == 120
    assert probe.SYNC_MEMORY == 256 * 1024**2
    assert probe.SYNC_CPU_QUOTA == 100
    assert (
        probe.SYNC_ROOT == "/home/yanbo/work/microduck_rl-stance-replication-20260930"
    )
    assert probe.SYNC_ORIGIN == "https://github.com/phoenixjyb/microduck_rl.git"
    assert probe.source_sync_unit(source) == (
        "microduck-cuda64-inertia-sync-" + source[:12] + ".service"
    )
    assert "set -euo pipefail" in probe.source_sync_script(source)
    for invalid in ("bad", "F" * 40, "f" * 39, "f" * 41):
        with pytest.raises(ValueError):
            probe.source_sync_unit(invalid)
        with pytest.raises(ValueError):
            probe.source_sync_script(invalid)


def test_source_sync_bootstrap_accepts_only_its_own_running_unit(tmp_path):
    source = "f" * 40
    process, log, state, _ = _source_sync_run(
        tmp_path, units=_own_sync_unit(source), remote=source
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == source
    assert (state / "fetched").exists() and (state / "merged").exists()
    assert (
        "git fetch origin refs/heads/feat/athletics-obstacle-curriculum:refs/remotes/origin/feat/athletics-obstacle-curriculum\n"
        in log
    )
    assert ("git merge --ff-only " + source + "\n") in log
    assert (state / "tracked-ref").read_text().strip() == source


@pytest.mark.parametrize(
    "units",
    [
        "",
        "microduck-cuda64-rollout-run-deadbeef.service loaded active running\n",
        _own_sync_unit("f" * 40)
        + "microduck-cuda64-rollout-run-deadbeef.service loaded active running\n",
        _own_sync_unit("f" * 40)
        + "microduck-cuda64-other-run-deadbeef.service loaded active running\n",
    ],
)
def test_source_sync_service_inventory_blocks_before_fetch_or_merge(tmp_path, units):
    process, log, state, _ = _source_sync_run(tmp_path, units=units)
    assert process.returncode != 0
    assert "source-sync must be the sole running Duck service" in process.stderr
    assert "git fetch" not in log and "git merge" not in log
    assert not (state / "fetched").exists() and not (state / "merged").exists()


@pytest.mark.parametrize(
    "head,branch,dirty",
    [
        ("e" * 40, "feat/athletics-obstacle-curriculum", ""),
        (probe.SYNC_FROM_SOURCE, "other-branch", ""),
        (probe.SYNC_FROM_SOURCE, "feat/athletics-obstacle-curriculum", " M file\n"),
    ],
)
def test_source_sync_wrong_head_branch_or_dirty_tree_stops_before_fetch(
    tmp_path, head, branch, dirty
):
    source = "f" * 40
    process, log, state, _ = _source_sync_run(
        tmp_path, units=_own_sync_unit(source), head=head, branch=branch, dirty=dirty
    )
    assert process.returncode != 0
    assert "git fetch" not in log and "git merge" not in log
    assert not (state / "fetched").exists() and not (state / "merged").exists()


@pytest.mark.parametrize(
    "root,origin",
    [
        ("/wrong/checkout", probe.SYNC_ORIGIN),
        (probe.SYNC_ROOT, "https://example.invalid/microduck_rl.git"),
    ],
)
def test_source_sync_wrong_root_or_origin_stops_before_fetch(tmp_path, root, origin):
    source = "f" * 40
    process, log, state, _ = _source_sync_run(
        tmp_path,
        units=_own_sync_unit(source),
        root=root,
        origin=origin,
    )
    assert process.returncode != 0
    assert "git fetch" not in log and "git merge" not in log
    assert not (state / "fetched").exists() and not (state / "merged").exists()


def test_source_sync_remote_head_mismatch_fetches_but_never_merges(tmp_path):
    source = "f" * 40
    process, log, state, _ = _source_sync_run(
        tmp_path, units=_own_sync_unit(source), remote="e" * 40
    )
    assert process.returncode != 0
    assert (
        "git fetch origin refs/heads/feat/athletics-obstacle-curriculum:refs/remotes/origin/feat/athletics-obstacle-curriculum\n"
        in log
    )
    assert "git merge" not in log
    assert (state / "fetched").exists() and not (state / "merged").exists()
    assert (state / "head").read_text().strip() == probe.SYNC_FROM_SOURCE
