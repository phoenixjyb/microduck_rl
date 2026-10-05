"""CUDA-hidden contract tests for the bounded runtime CRB supervisor."""

import hashlib
from contextlib import nullcontext
import math

import numpy as np
import pytest
import torch

from mjlab_microduck import stance_crb_runtime_probe as probe


def _mock_order_input_bytes(monkeypatch, tmp_path):
    root = tmp_path / "authenticated-original-inputs"
    launch = {"schedule": {"worlds": 64}}
    launch_sha = "a" * 64
    inventory = {"original-input": {"bytes": 17, "sha256": "b" * 64}}
    anchors = {
        "comparison_sha256": probe.order.MAP_COMPARISON_SHA,
        "report_sha256": probe.order.MAP_REPORT_SHA,
        "smooth_sha256": probe.order.SMOOTH_SHA,
        "compiled_topology_sha256": probe.order.MAP_TOPOLOGY_SHA,
    }
    calls = {"hidden": 0, "inputs": 0, "anchors": 0}

    def authenticate_inputs():
        calls["inputs"] += 1
        return root, launch, launch_sha, inventory

    def authenticate_map():
        calls["anchors"] += 1
        return anchors

    def hidden():
        calls["hidden"] += 1

    monkeypatch.setattr(probe.order.oracle, "_hidden", hidden)
    monkeypatch.setattr(
        probe.order.frozen_component, "authenticate_inputs", authenticate_inputs
    )
    monkeypatch.setattr(probe.order, "authenticate_map", authenticate_map)
    return root, launch, launch_sha, inventory, anchors, calls


def test_predecessor_wrapper_routes_through_frozen_order_authenticator(
    monkeypatch, tmp_path
):
    root, launch, launch_sha, inventory, anchors, calls = _mock_order_input_bytes(
        monkeypatch, tmp_path
    )
    original_files = {
        "rollout": [None] * 22,
        "diagnosis": [None],
        "component_map": [None] * 2,
        "order_oracle": [None] * 2,
    }
    original = {
        "root": str(root),
        "launch_sha256": launch_sha,
        "original_files": original_files,
        "anchors": anchors,
        "comparison_sha256": probe.fulltree.frozen.ORDER_COMPARISON_SHA,
        "report_sha256": probe.fulltree.frozen.ORDER_REPORT_SHA,
    }
    legacy = {"predecessors_authenticated": True, "original": original}
    previous = {"prior_fulltree": "authenticated"}
    monkeypatch.setattr(probe, "_authenticate_previous_fulltree", lambda: previous)
    monkeypatch.setattr(probe.fulltree, "_authenticate_predecessors", lambda: legacy)

    result = probe._authenticate_all_predecessors()

    assert probe.order is probe.fulltree.order is probe.fulltree.frozen.order
    assert result == {
        "fulltree": previous,
        "legacy": legacy,
        "predecessors_authenticated": True,
    }
    assert calls == {"hidden": 1, "inputs": 1, "anchors": 1}
    assert launch["schedule"]["worlds"] == 64
    assert result["legacy"]["original"]["launch_sha256"] == launch_sha
    assert result["legacy"]["original"]["anchors"] == anchors
    assert {key: len(value) for key, value in original_files.items()} == {
        "rollout": 22,
        "diagnosis": 1,
        "component_map": 2,
        "order_oracle": 2,
    }


def test_original_score_wrapper_uses_order_parent_and_preserves_rng_on_exact_rejection(
    monkeypatch, tmp_path
):
    root, launch, launch_sha, inventory, anchors, calls = _mock_order_input_bytes(
        monkeypatch, tmp_path
    )
    old_constructor = {"old": "authenticated constructor receipt"}
    inputs = [{"constructor_receipt": old_constructor}, {"attempt": 1}]
    traces = [{"trace": 0}, {"trace": 1}]
    score_calls = {}
    strict_calls = {}

    def score_inputs(actual_root, parent, actual_launch, actual_sha):
        score_calls.update(
            root=actual_root,
            parent=parent,
            launch=actual_launch,
            launch_sha=actual_sha,
        )
        return inputs, ["scores"], traces

    def strict_pair(actual_inputs, actual_traces):
        strict_calls.update(inputs=actual_inputs, traces=actual_traces)
        raise ValueError("paired rollout semantic state exactness")

    monkeypatch.setattr(probe.prior, "_score_inputs", score_inputs)
    monkeypatch.setattr(probe.prior, "_strict_pair", strict_pair)
    rng_before = torch.random.get_rng_state().clone()

    result = probe._score_original_pair()

    assert probe.order is probe.fulltree.order is probe.fulltree.frozen.order
    assert calls == {"hidden": 1, "inputs": 1, "anchors": 1}
    assert score_calls == {
        "root": root,
        "parent": probe.order.PARENT,
        "launch": launch,
        "launch_sha": launch_sha,
    }
    assert strict_calls == {"inputs": inputs, "traces": traces}
    assert result["order_inputs"] == (root, launch, launch_sha, inventory, anchors)
    assert result["failure"] == "paired rollout semantic state exactness"
    assert result["old_constructor_receipt"] is old_constructor
    assert torch.equal(rng_before, torch.random.get_rng_state())


def _scope_receipt(mode):
    return {
        "protocol": probe.runtime_control.PROTOCOL,
        "status": "complete",
        "mode": mode,
        "max_forward_calls": 4,
        "smooth_source_sha256": probe.SMOOTH_SHA256,
        "forward_source_sha256": probe.FORWARD_SHA256,
        "topology_id_snapshots": 4,
        "singleton_child_arrays_allocated": 3,
        "initialization_timing_changed": True,
        "constructor_forward_calls": 1,
        "constructor_forward_covered": True,
        "runtime_bound": True,
        "forward_calls": 4,
        "original_level_launch_requests": 28,
        "parent_zero_noop_level_requests": 8,
        "actual_accumulation_launches": 36 if mode == "serial" else 28,
        "split_child_launches": 12 if mode == "serial" else 0,
        "dense_qM_launches": 4,
        "fault_type": None,
        "flags": dict(probe.runtime_control.FLAGS),
    }


@pytest.mark.parametrize("mode", ["concurrent", "serial"])
def test_scope_receipt_requires_exact_forward_and_launch_counts(mode):
    assert probe._validate_scope_receipt(
        _scope_receipt(mode), source=probe.BASE_SOURCE, mode=mode
    )


def test_scope_receipt_rejects_missing_forward_or_changed_accumulation_count():
    value = _scope_receipt("serial")
    value["forward_calls"] = 3
    with pytest.raises(ValueError, match="constructor plus three"):
        probe._validate_scope_receipt(value, source=probe.BASE_SOURCE, mode="serial")
    value = _scope_receipt("serial")
    value["actual_accumulation_launches"] = 35
    with pytest.raises(ValueError, match="exact original, split"):
        probe._validate_scope_receipt(value, source=probe.BASE_SOURCE, mode="serial")


def test_unentered_scope_is_explicitly_not_started():
    scope = probe.runtime_control.StanceCrbRuntimeControl(
        mode="serial", max_forward_calls=4
    )
    assert scope.receipt["status"] == "not-started"
    assert scope.receipt["forward_calls"] == 0
    assert scope.receipt["topology_id_snapshots"] == 0
    with pytest.raises(ValueError, match="plain bounded forward-call cap"):
        probe.runtime_control.StanceCrbRuntimeControl(
            mode="serial", max_forward_calls=5
        )


def test_runtime_window_reserves_all_following_caps_and_cutoff():
    assert probe.TEST_RESERVE == 300 + 300 + 1200 + 300 + 60
    assert probe.PREFLIGHT_RESERVE == 300 + 1200 + 300 + 60
    assert probe.RUN_RESERVE == 1200 + 300 + 60
    probe._check_window(reserve_seconds=60, now=probe.CUTOFF - 61)
    with pytest.raises(ValueError, match="cutoff"):
        probe._check_window(reserve_seconds=60, now=probe.CUTOFF - 60)
    for invalid in (True, float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="finite runtime clock"):
            probe._check_window(reserve_seconds=60, now=invalid)
    for invalid_reserve in (True, 0, -1):
        with pytest.raises(ValueError, match="positive remaining"):
            probe._check_window(reserve_seconds=invalid_reserve, now=probe.CUTOFF - 100)
    assert probe._valid_elapsed(1.0, 2)
    for invalid_elapsed in (True, float("nan"), float("inf"), 0, -1, 2):
        assert not probe._valid_elapsed(invalid_elapsed, 2)


def test_child_payload_size_hash_is_checked_before_decoder(monkeypatch):
    invoked = False

    def forbidden_decode(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("decoder must not run")

    monkeypatch.setattr(probe.torch, "load", forbidden_decode)
    receipt = {"payload_bytes": 4, "payload_sha256": "0" * 64, "mode": "serial"}
    with pytest.raises(ValueError, match="whole SHA checked before"):
        probe._load_payload(
            b"evil",
            receipt,
            source=probe.BASE_SOURCE,
            attempt="serial-capture",
            host_identity={},
            versions={},
            private_cuda_start=torch.zeros(16, dtype=torch.uint8),
            metadata_reference={},
        )
    assert invoked is False


def test_tensor_reader_requires_exact_finite_cpu_float32():
    data = torch.zeros((probe.WORLDS, 21), dtype=torch.float32)
    raw = probe._tensor_raw(data, "qpos", (probe.WORLDS, 21))
    assert len(raw) == probe.WORLDS * 21 * 4
    with pytest.raises(ValueError, match="owned finite CPU float32"):
        probe._tensor_raw(data.double(), "qpos", (probe.WORLDS, 21))
    data[0, 0] = math.nan
    with pytest.raises(ValueError, match="owned finite CPU float32"):
        probe._tensor_raw(data, "qpos", (probe.WORLDS, 21))


def test_metadata_reference_is_rebuilt_from_pinned_cpu_compiled_plant():
    value = probe._compiled_runtime_metadata_reference(probe.BASE_SOURCE)
    assert set(value) == {
        "runtime_type",
        "worlds",
        "nbody",
        "nq",
        "nv",
        "nu",
        "floor_id",
        "foot_ids",
        "control_ids",
        "physics_steps",
        "plant_binding",
        "schedule",
    }
    assert (value["worlds"], value["nbody"], value["nq"], value["nv"], value["nu"]) == (
        64,
        16,
        21,
        20,
        14,
    )
    assert value["floor_id"] == 0
    assert value["foot_ids"] == [29, 79]
    assert value["control_ids"] == list(range(14))
    assert value["physics_steps"] == [0] * 64
    assert value["schedule"] == probe.prior.declaration(probe.BASE_SOURCE)
    assert value["plant_binding"]["nbody"] == 16
    bool_step_standin = {**value, "physics_steps": [False] * probe.WORLDS}
    assert not probe._metadata_matches_reference(bool_step_standin, value)
    bool_body_standin = {
        **value,
        "plant_binding": {**value["plant_binding"], "nbody": True},
    }
    assert not probe._metadata_matches_reference(bool_body_standin, value)


def _publication_campaign(artifacts=None):
    files = {
        path: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for path, raw in (artifacts or {}).items()
    }
    return {"total_bytes": sum(row["bytes"] for row in files.values()), "files": files}


def test_tests_reader_binds_report_to_exact_receipt_bytes(monkeypatch, tmp_path):
    source = probe.BASE_SOURCE
    binding = {"source": source, "leaves": {}}
    service = {"InvocationID": "a" * 32}
    log = f"{probe.EXPECTED_TESTS} passed in 1.25s\n".encode()
    receipt = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": source,
        "source_binding": binding,
        "test_files": list(probe.TEST_FILES),
        "passed": probe.EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": hashlib.sha256(log).hexdigest(),
        "service_properties": service,
        **probe.FLAGS,
    }
    receipt_raw = (probe.canonical(receipt) + "\n").encode()
    report = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": source,
        "mode": "tests",
        "status": "passed",
        "source_binding": binding,
        "host_identity": {"host": "pinned"},
        "runtime_versions": {"python": "pinned"},
        "service_properties": service,
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "passed": probe.EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": receipt["pytest_sha256"],
        "idle_before": {},
        "idle_after": {},
        "filmbrain": {},
        "protected_services": {},
        "elapsed_seconds": 1.0,
        "campaign_artifacts_before_report": _publication_campaign(
            {"tests/receipt.json": receipt_raw, "tests/pytest.log": log}
        ),
        **probe.FLAGS,
    }
    files = {
        "receipt.json": receipt_raw,
        "pytest.log": log,
        "report.json": (probe.canonical(report) + "\n").encode(),
    }
    monkeypatch.setattr(probe, "output_path", lambda _source, _mode: tmp_path)
    monkeypatch.setattr(probe.base, "_exact_inventory", lambda *_args: None)
    monkeypatch.setattr(probe, "_read_file", lambda path, _limit: files[path.name])
    monkeypatch.setattr(probe, "_terminal", lambda *_args: None)
    monkeypatch.setattr(
        probe,
        "_stage_campaign_snapshot",
        lambda _source, _mode: report["campaign_artifacts_before_report"],
    )
    assert probe._read_tests(source, binding) == hashlib.sha256(receipt_raw).hexdigest()

    report["receipt_sha256"] = "f" * 64
    files["report.json"] = (probe.canonical(report) + "\n").encode()
    with pytest.raises(ValueError, match="test report independently matches receipt"):
        probe._read_tests(source, binding)


def _preflight_publication_fixture(monkeypatch, tmp_path):
    source = probe.BASE_SOURCE
    binding = {"source": source, "leaves": {}}
    host = {"source": source, "host": "pinned"}
    versions = {"python": "pinned"}
    source_hashes = {
        "smooth_sha256": probe.SMOOTH_SHA256,
        "forward_sha256": probe.FORWARD_SHA256,
    }
    service = {"InvocationID": "a" * 32}
    schedule = {"groups": [[2], [7], [11]]}
    metadata = {
        "runtime_type": "mjlab_microduck.stance_recovery_schedule_runtime.ScheduledRecoveryRuntime",
        "worlds": probe.WORLDS,
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "floor_id": 0,
        "foot_ids": [29, 79],
        "control_ids": list(range(14)),
        "physics_steps": [0] * probe.WORLDS,
        "plant_binding": {"nbody": 16},
        "schedule": schedule,
    }
    anchor = {
        "protocol": probe.constructor.PROTOCOL,
        "source": probe.BASE_SOURCE,
        "private_cpu_start_sha256": "1" * 64,
        "private_cuda_start_sha256": "2" * 64,
        "caller_cpu_preserved": True,
        "caller_cuda_preserved": True,
    }
    predecessor = {"authenticated": True}
    declaration = {
        "protocol": probe.PROTOCOL + ":preflight",
        "source": source,
        "source_binding": binding,
        "host_identity": host,
        "runtime_versions": versions,
        "service_properties": service,
        "tests_receipt_sha256": "3" * 64,
        "source_hashes": source_hashes,
        "schedule": schedule,
        "attempts": list(probe.ATTEMPTS),
        "worlds": probe.WORLDS,
        "runtime_metadata_reference": metadata,
        "original_trace_topology": {
            key: metadata[key] for key in ("floor_id", "foot_ids", "control_ids")
        },
        "caller_cpu_seed": 673,
        "caller_cuda_seed": 677,
        "private_constructor": anchor,
        "old_strict_pair_failure": "paired rollout semantic state exactness",
        "predecessor_summary": predecessor,
        "predecessor_binding_sha256": probe._sha_summary(predecessor),
        "new_scope_contract": {
            "forward_calls": 4,
            "constructor_calls": 1,
            "original_level_requests": 28,
            "concurrent_accumulation_launches": 28,
            "serial_accumulation_launches": 36,
            "dense_qM_launches": 4,
        },
        "torch_cuda_initialized": False,
        **probe.FLAGS,
    }
    declaration_raw = (probe.canonical(declaration) + "\n").encode()
    report = {
        "protocol": probe.PROTOCOL + ":preflight",
        "source": source,
        "mode": "preflight",
        "status": "passed",
        "source_binding": binding,
        "host_identity": host,
        "runtime_versions": versions,
        "service_properties": service,
        "declaration_sha256": hashlib.sha256(declaration_raw).hexdigest(),
        "tests_receipt_sha256": declaration["tests_receipt_sha256"],
        "predecessor_summary": predecessor,
        "predecessor_binding_sha256": declaration["predecessor_binding_sha256"],
        "original_strict_pair_failure": declaration["old_strict_pair_failure"],
        "private_constructor_anchor": anchor,
        "runtime_metadata_reference": metadata,
        "torch_cuda_initialized": False,
        "idle_before": {},
        "idle_after": {},
        "filmbrain": {},
        "protected_services": {},
        "elapsed_seconds": 1.0,
        "campaign_artifacts_before_report": _publication_campaign(
            {
                "tests/receipt.json": b"pinned prior tests receipt\n",
                "tests/pytest.log": b"1573 passed in 1s\n",
                "tests/report.json": b"pinned prior tests report\n",
                "preflight/declaration.json": declaration_raw,
            }
        ),
        **probe.FLAGS,
    }
    files = {
        "declaration.json": declaration_raw,
        "report.json": (probe.canonical(report) + "\n").encode(),
    }
    monkeypatch.setattr(probe, "output_path", lambda _source, _mode: tmp_path)
    monkeypatch.setattr(probe.base, "_exact_inventory", lambda *_args: None)
    monkeypatch.setattr(probe, "_read_file", lambda path, _limit: files[path.name])
    monkeypatch.setattr(probe, "_terminal", lambda *_args: None)
    monkeypatch.setattr(
        probe,
        "_stage_campaign_snapshot",
        lambda _source, _mode: report["campaign_artifacts_before_report"],
    )
    monkeypatch.setattr(
        probe, "_compiled_runtime_metadata_reference", lambda _source: metadata
    )
    monkeypatch.setattr(probe.prior, "declaration", lambda _source: schedule)
    return source, binding, host, versions, source_hashes, files, declaration, report


def test_preflight_reader_accepts_actual_writer_schema_and_rejects_drift(
    monkeypatch, tmp_path
):
    source, binding, host, versions, hashes, files, declaration, report = (
        _preflight_publication_fixture(monkeypatch, tmp_path)
    )
    reader_args = (
        source,
        binding,
        declaration["tests_receipt_sha256"],
        host,
        versions,
        hashes,
    )
    result = probe._read_preflight(*reader_args)
    assert result["declaration"] == declaration
    assert result["report"] == report

    del declaration["new_scope_contract"]
    declaration_raw = (probe.canonical(declaration) + "\n").encode()
    files["declaration.json"] = declaration_raw
    report["declaration_sha256"] = hashlib.sha256(declaration_raw).hexdigest()
    files["report.json"] = (probe.canonical(report) + "\n").encode()
    with pytest.raises(
        ValueError, match="exact authenticated successful preflight declaration"
    ):
        probe._read_preflight(*reader_args)


@pytest.mark.parametrize(
    "field", ["host_identity", "service_properties", "declaration_sha256"]
)
def test_preflight_reader_rejects_report_host_service_and_hash_drift(
    monkeypatch, tmp_path, field
):
    source, binding, host, versions, hashes, files, declaration, report = (
        _preflight_publication_fixture(monkeypatch, tmp_path)
    )
    if field == "host_identity":
        report[field] = {"source": source, "host": "other"}
    elif field == "service_properties":
        report[field] = {"InvocationID": "b" * 32}
    else:
        report[field] = "f" * 64
    files["report.json"] = (probe.canonical(report) + "\n").encode()
    with pytest.raises(ValueError):
        probe._read_preflight(
            source,
            binding,
            declaration["tests_receipt_sha256"],
            host,
            versions,
            hashes,
        )


def test_stage_campaign_snapshot_includes_prior_stages_and_excludes_future_and_current_report(
    monkeypatch, tmp_path
):
    roots = {mode: tmp_path / mode for mode in probe.MODES}
    declared = {
        "tests": probe.TEST_FILES_ON_DISK,
        "preflight": probe.PREFLIGHT_FILES,
        "run": probe.RUN_FILES,
        "closeout": probe.CLOSEOUT_FILES,
    }
    for mode, names in declared.items():
        roots[mode].mkdir()
        for name in names:
            (roots[mode] / name).write_bytes(f"{mode}/{name}\n".encode())
    monkeypatch.setattr(probe, "output_path", lambda _source, mode: roots[mode])

    snapshot = probe._stage_campaign_snapshot(probe.BASE_SOURCE, "run")
    expected_names = {
        *(f"tests/{name}" for name in probe.TEST_FILES_ON_DISK),
        *(f"preflight/{name}" for name in probe.PREFLIGHT_FILES),
        *(f"run/{name}" for name in probe.RUN_FILES - {"report.json"}),
    }
    assert set(snapshot["files"]) == expected_names
    assert snapshot["total_bytes"] == sum(
        item["bytes"] for item in snapshot["files"].values()
    )

    before = snapshot
    (roots["run"] / "report.json").write_bytes(b"late current-stage report\n")
    (roots["closeout"] / "receipt.json").write_bytes(b"future receipt\n")
    assert probe._stage_campaign_snapshot(probe.BASE_SOURCE, "run") == before

    (roots["tests"] / "receipt.json").write_bytes(b"tampered earlier-stage receipt\n")
    assert probe._stage_campaign_snapshot(probe.BASE_SOURCE, "run") != before


def test_field_comparison_is_signed_zero_sensitive_and_retains_delta():
    left = {
        name: np.zeros(
            (probe.WORLDS, *shape) if shape else (probe.WORLDS,), dtype="<f4"
        )
        for name, shape in probe.FRAME_FIELDS.items()
    }
    right = {name: value.copy() for name, value in left.items()}
    right["crb"].flat[0] = np.float32(-0.0)
    right["qpos"].flat[1] = np.float32(0.25)
    result = probe._field_comparison("serial-vs-concurrent", left, right)
    assert result["fields"]["crb"]["bit_mismatch_scalars"] == 1
    assert result["fields"]["qpos"]["bit_mismatch_scalars"] == 1
    assert result["fields"]["qpos"]["max_abs_delta"] == 0.25
    assert result["fields"]["cinert"]["exact_raw_bits"] is True


def test_four_child_comparator_retains_finite_negative_results():
    frames = []
    for _ in range(4):
        frames.append(
            {
                name: np.zeros(
                    (probe.WORLDS, *shape) if shape else (probe.WORLDS,), dtype="<f4"
                )
                for name, shape in probe.FRAME_FIELDS.items()
            }
        )
    payloads = {}
    for attempt in probe.ATTEMPTS:
        child_frames = [
            {name: value.copy() for name, value in frame.items()} for frame in frames
        ]
        payloads[attempt] = {
            "runtime_metadata": {"worlds": probe.WORLDS},
            "frames": child_frames,
        }
    payloads["concurrent-replay"]["frames"][0]["crb"].flat[0] = np.float32(1e-6)
    payloads["serial-replay"]["frames"][3]["cinert"].flat[3] = np.float32(-1e-6)
    summary = probe._compare_four(payloads)
    assert summary["pair_count"] == 28
    assert [pair["label"] for pair in summary["pairs"][-12:]][:4] == [
        "concurrent-capture.constructor-vs-forward-1",
        "concurrent-capture.constructor-vs-forward-2",
        "concurrent-capture.constructor-vs-forward-3",
        "concurrent-replay.constructor-vs-forward-1",
    ]
    assert summary["all_exact_raw_bits"] is False
    assert summary["numerical_negative_outcomes_retained"] is True
    assert summary["no_tolerance_or_rerun"] is True


def test_exact_source_sync_template_retargets_both_guards_once():
    script = probe.source_sync_script(probe.BASE_SOURCE, bundle_sha256="a" * 64)
    assert script.count(f"= {probe.SYNC_FROM_SOURCE}\n") == 1
    assert script.count(probe.source_sync_unit(probe.BASE_SOURCE)) == 1
    assert probe.fulltree.SYNC_FROM_SOURCE not in script
    assert hashlib.sha256(script.encode()).hexdigest() != "0" * 64


def test_child_environment_is_sanitized_and_cuda_is_explicit(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-value")
    cpu = probe._cpu_env()
    gpu = probe._gpu_env()
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0"
    assert "AWS_SECRET_ACCESS_KEY" not in cpu
    assert "AWS_SECRET_ACCESS_KEY" not in gpu
    assert cpu["OMP_NUM_THREADS"] == gpu["OMP_NUM_THREADS"] == "1"


def test_terminal_record_rejects_service_cap_or_invocation_drift():
    invocation = "b" * 32
    valid = {
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "RemainAfterExit": "yes",
        "RuntimeMaxUSec": "20min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": probe.CPU_QUOTA,
        "Nice": probe.NICE,
        "KillMode": probe.KILL_MODE,
        "InvocationID": invocation,
    }
    assert probe._terminal_record(valid, "run", invocation) == valid
    for key, value in (
        ("RuntimeMaxUSec", "10min"),
        ("MemoryMax", "1"),
        ("CPUQuotaPerSecUSec", "1s"),
        ("Nice", "0"),
        ("KillMode", "process"),
    ):
        changed = dict(valid, **{key: value})
        with pytest.raises(ValueError, match="terminal caps"):
            probe._terminal_record(changed, "run", invocation)


def test_artifact_caps_and_child_log_inventory_are_four_separate_files():
    assert {f"{attempt}.log" for attempt in probe.ATTEMPTS} <= probe.RUN_FILES
    assert "child.log" not in probe.RUN_FILES
    assert probe.PAYLOAD_LIMIT == 2 * 1024**2
    assert probe.TOTAL_LIMIT == 16 * 1024**2


def test_campaign_cap_combines_files_across_modes(tmp_path, monkeypatch):
    roots = {
        mode: tmp_path / mode for mode in ("tests", "preflight", "run", "closeout")
    }
    names = {
        "tests": ("report.json", "receipt.json", "pytest.log"),
        "preflight": ("declaration.json", "report.json"),
        "run": tuple(f"{attempt}.pt" for attempt in probe.ATTEMPTS),
        "closeout": (),
    }
    sizes = {
        "report.json": 2 * 1024**2,
        "receipt.json": 2 * 1024**2,
        "pytest.log": 1024**2,
        "declaration.json": 2 * 1024**2,
    }
    for mode, filenames in names.items():
        roots[mode].mkdir()
        for name in filenames:
            (roots[mode] / name).write_bytes(b"x" * sizes.get(name, 2 * 1024**2))
    monkeypatch.setattr(probe, "output_path", lambda _source, mode: roots[mode])
    assert all(
        sum(path.stat().st_size for path in roots[mode].iterdir()) <= probe.TOTAL_LIMIT
        for mode in roots
    )
    with pytest.raises(ValueError, match="all runtime mode artifacts combined"):
        probe._campaign_inventory(probe.BASE_SOURCE)


def test_cpu_payload_reader_runs_frozen_constructor_check_with_prepared_states(
    monkeypatch,
):
    metadata = probe._compiled_runtime_metadata_reference(probe.BASE_SOURCE)
    caller = {
        name: torch.arange(8, dtype=torch.uint8)
        for name in ("cpu_before", "cpu_after", "cuda_before", "cuda_after")
    }
    payload = {
        "constructor_receipt": {"receipt": "safe weights-only value"},
        "caller_rng_states": caller,
        "owner_pid": 17,
        "child_pid": 18,
        "child_ppid": 17,
        "runtime_metadata": metadata,
        "scope_receipt": {"scope": "complete"},
    }
    buffer = __import__("io").BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    caller_hashes = {
        name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        for name, tensor in caller.items()
    }
    constructor_hash = probe._sha_summary(payload["constructor_receipt"])
    monkeypatch.setattr(
        probe, "_validate_payload", lambda *args, **kwargs: {"cpu_checked": True}
    )
    monkeypatch.setattr(
        probe,
        "_payload_summary",
        lambda _payload: {
            "caller_rng_sha256": caller_hashes,
            "frame_sha256": [],
            "constructor_receipt_sha256": constructor_hash,
        },
    )
    observed = {}

    def frozen_check(receipt, *, source, prepared_caller):
        observed.update(source=source, receipt=receipt, prepared_caller=prepared_caller)
        return {"checked": True}

    monkeypatch.setattr(probe.constructor, "check", frozen_check)
    receipt = {
        "payload_bytes": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "concurrent",
        "owner_pid": 17,
        "child_pid": 18,
        "child_ppid": 17,
        "runtime_metadata": payload["runtime_metadata"],
        "scope_receipt": payload["scope_receipt"],
        "caller_rng_sha256": caller_hashes,
        "frame_sha256": [],
        "constructor_receipt_sha256": constructor_hash,
    }
    result, validation = probe._load_payload(
        raw,
        receipt,
        source=probe.BASE_SOURCE,
        attempt="concurrent-capture",
        host_identity={},
        versions={},
        private_cuda_start=torch.zeros(8, dtype=torch.uint8),
        metadata_reference=metadata,
    )
    assert validation == {"cpu_checked": True}
    assert observed == {
        "source": probe.BASE_SOURCE,
        "receipt": payload["constructor_receipt"],
        "prepared_caller": result["caller_rng_states"],
    }


def test_child_receipt_requires_exact_runtime_metadata_and_live_parent_pid(monkeypatch):
    metadata = probe._compiled_runtime_metadata_reference(probe.BASE_SOURCE)
    declaration = {
        "host_identity": {"host": "pinned"},
        "runtime_versions": {"python": "pinned"},
        "service_properties": {"InvocationID": "a" * 32},
        "declaration_sha256": "b" * 64,
        "source_binding": {"source": probe.BASE_SOURCE},
        "source_hashes": {
            "smooth_sha256": probe.SMOOTH_SHA256,
            "forward_sha256": probe.FORWARD_SHA256,
        },
        "runtime_metadata_reference": metadata,
    }
    receipt = {
        "protocol": probe.PROTOCOL + ":child",
        "source": probe.BASE_SOURCE,
        "mode": "concurrent",
        "attempt": "concurrent-capture",
        "owner_pid": 17,
        "child_pid": 18,
        "child_ppid": 17,
        "host_identity": declaration["host_identity"],
        "runtime_versions": declaration["runtime_versions"],
        "runtime_metadata": declaration["runtime_metadata_reference"],
        "payload_file": "concurrent-capture.pt",
        "caller_rng_sha256": {"cuda_before": "c" * 64, "cuda_after": "c" * 64},
        "frame_sha256": [],
        "scope_receipt": _scope_receipt("concurrent"),
        "constructor_receipt_sha256": "d" * 64,
        "torch_cuda_initialized": True,
        "payload_bytes": 7,
        "payload_sha256": "e" * 64,
        "service_properties": declaration["service_properties"],
        "declaration_sha256": declaration["declaration_sha256"],
        "source_binding": declaration["source_binding"],
        "source_hashes": declaration["source_hashes"],
        "status": "passed",
        "constructor_calls": 1,
        "forward_calls": 4,
        "physics_steps": [0] * probe.WORLDS,
        **probe.FLAGS,
    }
    monkeypatch.setattr(
        probe, "_read_file", lambda _path, _limit: probe.canonical(receipt).encode()
    )
    raw, decoded = probe._read_child_receipt(
        "receipt.json",
        source=probe.BASE_SOURCE,
        mode="concurrent",
        attempt="concurrent-capture",
        declaration=declaration,
        expected_pid=18,
        owner_pid=17,
    )
    assert raw == probe.canonical(receipt).encode()
    assert decoded == receipt
    receipt["constructor_calls"] = True
    monkeypatch.setattr(
        probe, "_read_file", lambda _path, _limit: probe.canonical(receipt).encode()
    )
    with pytest.raises(ValueError, match="exact child receipt"):
        probe._read_child_receipt(
            "receipt.json",
            source=probe.BASE_SOURCE,
            mode="concurrent",
            attempt="concurrent-capture",
            declaration=declaration,
            expected_pid=18,
            owner_pid=17,
        )


def test_run_reader_binds_declaration_hash_and_compares_all_four_private_endpoints(
    monkeypatch, tmp_path
):
    endpoint = {
        name: torch.arange(8, dtype=torch.uint8)
        for name in ("cpu_start", "cpu_end", "cuda_start", "cuda_end")
    }
    old_receipt = {"private_states": {**endpoint, "cuda_start": endpoint["cuda_start"]}}
    declaration = {"runtime_metadata_reference": {"worlds": probe.WORLDS}}
    report = {
        "owner_pid": 17,
        "service_properties": {"MainPID": "17"},
        "declaration": {"source": probe.BASE_SOURCE},
        "declaration_sha256": "f" * 64,
        "children": {
            attempt: {"pid": index + 18} for index, attempt in enumerate(probe.ATTEMPTS)
        },
        "host_identity": {"host": "pinned"},
        "runtime_versions": {"python": "pinned"},
        "artifacts": {
            attempt + ".pt": {
                "bytes": len(b"payload"),
                "sha256": hashlib.sha256(b"payload").hexdigest(),
            }
            for attempt in probe.ATTEMPTS
        },
    }
    monkeypatch.setattr(probe, "_inventory", lambda _root, _names: report["artifacts"])
    observed_declarations = []

    def child_receipt(
        _path, *, source, mode, attempt, declaration, expected_pid, owner_pid
    ):
        observed_declarations.append(declaration["declaration_sha256"])
        return b"receipt", {
            "payload_file": attempt + ".pt",
            "payload_sha256": hashlib.sha256(b"payload").hexdigest(),
            "payload_bytes": len(b"payload"),
            "caller_rng_sha256": {"cuda_before": "c" * 64, "cuda_after": "c" * 64},
        }

    monkeypatch.setattr(probe, "_read_child_receipt", child_receipt)
    monkeypatch.setattr(probe, "_read_file", lambda _path, _limit: b"payload")
    caller_cuda = torch.zeros(8, dtype=torch.uint8)

    def load(_raw, _receipt, *, attempt, **_kwargs):
        return {
            "mode": probe.SCHEDULE_MODES[attempt],
            "caller_rng_states": {"cuda_before": caller_cuda},
            "constructor_receipt": {"private_states": endpoint},
        }, {}

    monkeypatch.setattr(probe, "_load_payload", load)
    monkeypatch.setattr(
        probe, "_compare_four", lambda _payloads: {"pair_count": 28, "pairs": []}
    )
    result = probe._load_and_compare_run(
        probe.BASE_SOURCE,
        tmp_path,
        report,
        {"_old_constructor_receipt": old_receipt},
        {"declaration": declaration},
    )
    assert observed_declarations == ["f" * 64] * 4
    assert result["pair_count"] == 28
    assert set(result["constructor_private_endpoint_sha256"]) == {
        "cpu_start",
        "cpu_end",
        "cuda_start",
        "cuda_end",
    }


def test_completed_run_owner_binds_published_live_service_pid():
    assert (
        probe._run_owner_pid({"owner_pid": 17, "service_properties": {"MainPID": "17"}})
        == 17
    )
    for value in (
        {},
        {"owner_pid": True, "service_properties": {"MainPID": "1"}},
        {"owner_pid": 17, "service_properties": {"MainPID": "18"}},
    ):
        with pytest.raises(ValueError, match="published run owner PID"):
            probe._run_owner_pid(value)


def test_runtime_monitor_requires_actual_gpu_pid_and_own_child_time_cap(monkeypatch):
    monkeypatch.setattr(probe.fulltree, "_monitor", lambda _rows: 18)
    rows = [{"elapsed_seconds": 179.0, "sample": {"compute_pids": [18]}}]
    assert probe._runtime_monitor(rows) == 18
    rows[0]["sample"]["compute_pids"] = []
    with pytest.raises(ValueError, match="actual GPU PID"):
        probe._runtime_monitor(rows)
    rows[0]["sample"]["compute_pids"] = [18]
    rows[0]["elapsed_seconds"] = 180.0
    with pytest.raises(ValueError, match="180 seconds"):
        probe._runtime_monitor(rows)


def _mock_execute_context(monkeypatch, tmp_path, *, fail=False):
    output = tmp_path / "tests-output"
    host = {"source": probe.BASE_SOURCE, "host": "pinned"}
    versions = {"python": "pinned"}
    binding = {"source": probe.BASE_SOURCE, "branch": probe.SOURCE_BRANCH, "leaves": {}}
    service = {"MainPID": "17", "ActiveState": "active", "InvocationID": "a" * 32}
    monkeypatch.setattr(probe.base.execution, "PROFILE", "wsl")
    monkeypatch.setattr(probe.base.execution, "WSL", "wsl")
    monkeypatch.setattr(probe.base.execution, "select", lambda _profile: "wsl")
    monkeypatch.setattr(probe.prior, "_hidden", lambda: None)
    monkeypatch.setattr(probe, "_check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe, "source_binding", lambda _source: binding)
    monkeypatch.setattr(
        probe, "_terminal_properties", lambda _source, _mode, _pid: service
    )
    monkeypatch.setattr(
        probe, "output_path", lambda _source, mode: tmp_path / f"{mode}-output"
    )
    monkeypatch.setattr(probe.base.gap.base.files, "gpu_lease", lambda: nullcontext(9))
    monkeypatch.setattr(
        probe,
        "_read_idle_context",
        lambda: (
            {"idle": True},
            {"user": "inactive", "system": "inactive"},
            {"filmbrain": "unchanged"},
        ),
    )
    monkeypatch.setattr(
        probe, "_authenticate_previous_fulltree", lambda: {"host_identity": host}
    )
    monkeypatch.setattr(probe, "_host", lambda _source, _previous: (host, versions))
    monkeypatch.setattr(probe, "_runtime_versions", lambda: versions)
    monkeypatch.setattr(
        probe,
        "_pinned_runtime_sources",
        lambda: {
            "smooth_sha256": probe.SMOOTH_SHA256,
            "forward_sha256": probe.FORWARD_SHA256,
        },
    )
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(probe, "_close_context", lambda *_args: {"idle": True})
    monkeypatch.setattr(probe, "_inventory", lambda *_args: {})
    monkeypatch.setattr(probe, "_partial_inventory", lambda _output: {"partial": True})

    def run_tests(*_args):
        if fail:
            raise ValueError("synthetic mode failure")
        (output / "receipt.json").write_bytes(b"{}\n")
        (output / "pytest.log").write_bytes(b"synthetic log\n")
        return {
            "protocol": probe.PROTOCOL + ":tests",
            "source": probe.BASE_SOURCE,
            "mode": "tests",
            "status": "passed",
            "source_binding": binding,
            "service_properties": service,
        }

    monkeypatch.setattr(probe, "_execute_tests", run_tests)
    writes = []
    writer = probe.base.write_json

    def write_once(path, value):
        writes.append(path)
        return writer(path, value)

    monkeypatch.setattr(probe.base, "write_json", write_once)
    return output, writes


def test_execute_writes_one_final_report_after_success(monkeypatch, tmp_path):
    output, writes = _mock_execute_context(monkeypatch, tmp_path)
    report = probe.execute(probe.BASE_SOURCE, "tests")
    assert report["status"] == "passed"
    assert writes == [output / "report.json"]
    assert (output / "report.json").is_file()


def test_execute_failure_retains_one_partial_report(monkeypatch, tmp_path):
    output, writes = _mock_execute_context(monkeypatch, tmp_path, fail=True)
    with pytest.raises(ValueError, match="synthetic mode failure"):
        probe.execute(probe.BASE_SOURCE, "tests")
    assert writes == [output / "report.json"]
    saved = probe.base.parse_json((output / "report.json").read_bytes())
    assert saved["status"] == "failed-retained"
    assert saved["partial_artifacts"] == {"partial": True}


def test_execute_late_closeout_failure_downgrades_inner_success(monkeypatch, tmp_path):
    output, writes = _mock_execute_context(monkeypatch, tmp_path)

    def fail_closeout(*_args):
        raise RuntimeError("synthetic final-context failure")

    monkeypatch.setattr(probe, "_close_context", fail_closeout)
    with pytest.raises(RuntimeError, match="synthetic final-context failure"):
        probe.execute(probe.BASE_SOURCE, "tests")
    assert writes == [output / "report.json"]
    saved = probe.base.parse_json((output / "report.json").read_bytes())
    assert saved["status"] == "failed-retained"
    assert saved["error"] == "synthetic final-context failure"
    assert saved["partial_artifacts"] == {"partial": True}


def test_execute_content_validation_failure_downgrades_before_publication(
    monkeypatch, tmp_path
):
    output, writes = _mock_execute_context(monkeypatch, tmp_path)

    def fail_inventory(*_args):
        raise ValueError("synthetic final content failure")

    monkeypatch.setattr(probe.base, "_exact_inventory", fail_inventory)
    with pytest.raises(ValueError, match="synthetic final content failure"):
        probe.execute(probe.BASE_SOURCE, "tests")
    assert writes == [output / "report.json"]
    saved = probe.base.parse_json((output / "report.json").read_bytes())
    assert saved["status"] == "failed-retained"
    assert saved["error"] == "synthetic final content failure"
