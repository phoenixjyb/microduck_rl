"""CUDA-hidden and mocked supervisor contract tests for full-tree control."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import os
import time

import numpy as np
import pytest
import torch

from mjlab_microduck import stance_crb_full_tree_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "a" * 40
INVOCATION = "c" * 32
PARENTS = [0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14]
LEVELS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2, 7, 11], [1], [0]]
GROUPS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2], [7], [11], [1], [0]]


def _service(mode="run", pid=None, invocation=INVOCATION):
    return {
        "MainPID": str(os.getpid() if pid is None else pid),
        "ActiveState": "active",
        "RuntimeMaxUSec": "10min" if mode == "run" else "5min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": invocation,
        "RemainAfterExit": "yes",
    }


def _topology():
    return {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": list(PARENTS),
        "reversed_body_tree_ids": [list(level) for level in LEVELS],
    }


def _fixture():
    zeros = np.zeros((64, 16, 10), dtype="<f4").tobytes()
    return probe.full_tree.fixture(
        zeros, zeros, zeros, zeros, _topology(), probe.LAUNCH_TOPOLOGY_SHA256
    )


def test_protocol_names_mode_caps_reserves_cutoff_and_51_file_count():
    assert probe.BASE_SOURCE == "a3b149d4fef78c1bf1a23a1b44699fabdb599a1a"
    assert probe.SYNC_FROM_SOURCE == "c9708cf68fdb812733f7a9fddd336c36a8241d26"
    assert [probe.unit(SOURCE, mode) for mode in ("tests", "fixture", "run")] == [
        f"microduck-crb-fulltree-{mode}-{SOURCE[:12]}.service"
        for mode in ("tests", "fixture", "run")
    ]
    assert len(probe.TEST_FILES) == 51 and probe.EXPECTED_TESTS == 1522
    assert (probe.TEST_CAP, probe.FIXTURE_CAP, probe.CAP, probe.CHILD_SECONDS) == (
        300,
        300,
        600,
        240,
    )
    assert (probe.RUN_RESERVE, probe.FIXTURE_RESERVE, probe.TEST_RESERVE) == (
        960,
        1260,
        1560,
    )
    assert probe.MEMORY == 6 * 1024**3 and probe.TOTAL_BYTES == 12 * 1024**2
    assert probe.CUTOFF == 1791244800
    with pytest.raises(ValueError):
        probe.unit(SOURCE, "sync")
    with pytest.raises(ValueError, match="cutoff"):
        probe._check_window(reserve_seconds=1, now=probe.CUTOFF)
    probe._check_window(reserve_seconds=1, now=probe.CUTOFF - 2)


def test_sync_template_retargets_only_exact_head_and_unit_guards(monkeypatch):
    template = (
        f'test "$(git rev-parse HEAD)" = {probe.serial.SYNC_FROM_SOURCE}\n'
        f'if test "$duck_sync_running" != {probe.serial.source_sync_unit(SOURCE)}; then\n'
    )
    monkeypatch.setattr(
        probe.serial, "source_sync_script", lambda *_args, **_kwargs: template
    )
    result = probe.source_sync_script(SOURCE, bundle_sha256="b" * 64)
    assert result.count(probe.SYNC_FROM_SOURCE) == 1
    assert result.count(probe.source_sync_unit(SOURCE)) == 1
    assert probe.serial.SYNC_FROM_SOURCE not in result
    assert probe.serial.source_sync_unit(SOURCE) not in result
    monkeypatch.setattr(
        probe.serial,
        "source_sync_script",
        lambda *_args, **_kwargs: template + template,
    )
    with pytest.raises(ValueError, match="one exact"):
        probe.source_sync_script(SOURCE, bundle_sha256="b" * 64)


def test_source_binding_preserves_base_leaves_and_rejects_unowned_changes(monkeypatch):
    root = Path(probe.__file__).resolve().parents[2]
    monkeypatch.setattr(probe.frozen.cpu_probe, "SYNC_ROOT", str(root))
    frozen_paths = probe._frozen_paths()
    paths = sorted(frozen_paths | set(probe.OWN) | set(probe.RELATED))
    leaves = {path: ("whole-leaf:" + path).encode() for path in paths}
    changed = sorted(probe.ALLOWED)

    def run(args, **_kwargs):
        command = args[1:]
        if command[:2] == ["rev-parse", "HEAD"]:
            stdout = SOURCE
        elif command[:2] == ["branch", "--show-current"]:
            stdout = "feat/athletics-obstacle-curriculum"
        elif command[:3] == ["remote", "get-url", "origin"]:
            stdout = probe.frozen.cpu_probe.SYNC_ORIGIN
        elif command[:2] == ["status", "--porcelain"]:
            stdout = ""
        elif command[:2] == ["diff", "--name-only"]:
            stdout = "\n".join(changed)
        elif command[0] == "show":
            target, path = command[1].split(":", 1)
            stdout = (
                leaves[path]
                if target == SOURCE
                or (target == probe.BASE_SOURCE and path in frozen_paths)
                else ("base-leaf:" + path).encode()
            )
        else:
            stdout = b""
        return SimpleNamespace(
            stdout=stdout.encode() if isinstance(stdout, str) else stdout
        )

    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(
        probe.base,
        "_read_file",
        lambda path, _limit: leaves[str(path).replace(str(root) + "/", "")],
    )
    result = probe.source_binding(SOURCE)
    assert set(result["leaves"]) == set(paths)
    changed.append("src/other.py")
    with pytest.raises(ValueError, match="five declared"):
        probe.source_binding(SOURCE)


def test_predecessor_packet_hashes_are_checked_before_json_decode(monkeypatch):
    monkeypatch.setattr(
        probe.base,
        "parse_json",
        lambda _raw: pytest.fail("packet decoded before pinned whole hash"),
    )
    with pytest.raises(ValueError, match="owner packet hash"):
        probe._validate_serial_predecessor(
            b"not the owner packet", b"not closeout", {}, {}, b"", b"", b"", b"", {}, {}
        )


def test_fixture_input_gate_precedes_tensor_interpretation_and_orders_raws():
    with pytest.raises(ValueError, match="predecessor fifty-file"):
        probe._extract_observed_fixture({}, object())
    traces = []
    for _ in range(2):
        traces.append(
            {
                "events": [
                    {
                        "phase": "scheduled-pre",
                        "step": 0,
                        "persistent": {
                            "cinert": torch.zeros((64, 16, 10), dtype=torch.float32),
                            "crb": torch.zeros((64, 16, 10), dtype=torch.float32),
                        },
                    }
                ]
            }
        )
    capture_cinert, replay_cinert, capture_crb, replay_crb = (
        probe._extract_observed_fixture({"predecessors_authenticated": True}, traces)
    )
    assert len(capture_cinert) == probe.MATRIX_BYTES
    assert capture_cinert == replay_cinert == capture_crb == replay_crb


def test_fixture_and_analyzer_retain_concurrent_and_serial_decisions_without_acceptance():
    fixture = _fixture()
    outputs = np.repeat(fixture["expected"][None, ...], 32, axis=0)
    raw = outputs.astype("<f4", copy=False).tobytes()
    concurrent = probe.full_tree.analyze(raw, fixture, schedule="concurrent")
    serial = probe.full_tree.analyze(raw, fixture, schedule="serial")
    assert (
        concurrent["schedule"] == "concurrent"
        and concurrent["decision"] == "diagnostic-only-concurrent"
    )
    assert (
        serial["schedule"] == "serial"
        and serial["decision"] == "stable-exact-candidate"
    )
    assert concurrent["flags"] == probe.full_tree.FLAGS
    assert all(value is False for key, value in probe.FLAGS.items())
    altered = outputs.copy()
    altered[0, 0, 1, 0] = np.float32(1.0)
    diagnostic = probe.full_tree.analyze(
        altered.astype("<f4", copy=False).tobytes(), fixture, schedule="serial"
    )
    assert diagnostic["decision"] == "not-stable-or-not-candidate"
    assert diagnostic["root_mismatched_scalars_from_prediction"] > 0


def test_batch_has_512_alias_launches_two_ordered_banks_one_sync_then_64_reads(
    monkeypatch,
):
    class FakeArray:
        def __init__(self, data=None, label="array"):
            self.data = data
            self.label = label

        def numpy(self):
            records["events"].append(("read", self))
            return np.zeros((64, 16, 10), dtype=np.float32)

    records = {"events": [], "launches": []}
    stream = object()
    device = SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(probe.wp, "ScopedDevice", lambda _device: nullcontext())

    def array(data=None, **_kwargs):
        return FakeArray(data)

    def empty(*, shape, **_kwargs):
        return FakeArray(label=f"empty-{len(records['events'])}")

    def copy(dst, src, *, stream=None):
        records["events"].append(("copy", dst, src, stream))

    def launch(kernel, *, dim, inputs, outputs, device=None, stream=None):
        item = (kernel, dim, inputs, outputs, device, stream)
        records["launches"].append(item)
        records["events"].append(("launch", item))

    def sync(_device):
        records["events"].append(("sync",))

    monkeypatch.setattr(probe.wp, "array", array)
    monkeypatch.setattr(probe.wp, "empty", empty)
    monkeypatch.setattr(probe.wp, "get_stream", lambda _device: stream)
    monkeypatch.setattr(probe.wp, "copy", copy)
    monkeypatch.setattr(probe.wp, "launch", launch)
    monkeypatch.setattr(probe.wp, "synchronize_device", sync)
    concurrent, serial = probe._full_tree_batch(
        device,
        np.zeros((64, 16, 10), dtype=np.float32),
        np.asarray(PARENTS, dtype=np.int32),
        LEVELS,
        GROUPS,
    )
    assert (
        len(records["launches"]) == 512
        and len(concurrent) == len(serial) == probe.OUTPUT_BYTES
    )
    assert sum(event[0] == "sync" for event in records["events"]) == 1
    sync_index = next(
        index for index, event in enumerate(records["events"]) if event[0] == "sync"
    )
    assert all(event[0] == "read" for event in records["events"][sync_index + 1 :])
    assert sum(event[0] == "read" for event in records["events"]) == 64
    launch_events = [event[1] for event in records["events"] if event[0] == "launch"]
    copies = [event for event in records["events"] if event[0] == "copy"]
    assert len(copies) == 128
    for repeat_idx in range(32):
        for bank, schedule, base_launch in ((0, LEVELS, 0), (1, GROUPS, 32 * 7)):
            event_index = (
                repeat_idx * (1 + 7 + 1)
                if bank == 0
                else 32 * 9 + repeat_idx * (1 + 9 + 1)
            )
            reset = copies[repeat_idx * 2 + bank * 64]
            snapshot = copies[repeat_idx * 2 + bank * 64 + 1]
            assert reset[3] is stream and snapshot[3] is stream
            working = reset[1]
            start = base_launch + repeat_idx * len(schedule)
            launches = launch_events[start : start + len(schedule)]
            assert [call[1] for call in launches] == [
                (64, len(group)) for group in schedule
            ]
            assert [call[2][2].data.tolist() for call in launches] == schedule
            assert all(
                call[0] is probe.KERNEL
                and call[2][1] is working
                and call[3][0] is working
                and call[5] is stream
                for call in launches
            )
            assert reset[1] is working and reset[2] is not working
            assert snapshot[2] is working and snapshot[1] is not working
            assert records["events"][event_index][0] == "copy"


def test_terminal_record_enforces_all_five_resource_caps():
    record = {
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "RemainAfterExit": "yes",
        "RuntimeMaxUSec": "10min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": INVOCATION,
    }
    assert probe._terminal_record(record, "run", INVOCATION) is None
    for field, bad in (
        ("RuntimeMaxUSec", "5min"),
        ("MemoryMax", "1"),
        ("CPUQuotaPerSecUSec", "1s"),
        ("Nice", "0"),
        ("KillMode", "process"),
    ):
        changed = dict(record, **{field: bad})
        with pytest.raises(ValueError, match="resource caps"):
            probe._terminal_record(changed, "run", INVOCATION)


def test_monitor_rejects_foreign_gpu_work_or_cap_violations():
    valid = [
        {
            "child_pid": 321,
            "elapsed_seconds": 0.0,
            "sample": {
                "temperature_c": 40,
                "memory_used_mib": 1024,
                "memory_free_mib": 12000,
                "compute_pids": [321],
                "services": {
                    **{name: "inactive" for name in probe.base.gpu_idle_gate.SERVICES},
                    **{
                        "user:" + name: "inactive"
                        for name in probe.base.gpu_idle_gate.SERVICES
                    },
                },
            },
        }
    ]
    assert probe._monitor(valid) == 321
    with pytest.raises(ValueError):
        probe._monitor(
            [{**valid[0], "sample": {**valid[0]["sample"], "compute_pids": [777]}}]
        )
    with pytest.raises(ValueError):
        probe._monitor(
            [{**valid[0], "sample": {**valid[0]["sample"], "memory_used_mib": 5121}}]
        )


def test_child_source_guard_follows_lease_and_precedes_warp_cuda(monkeypatch):
    events = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        probe.base.training_smoke,
        "inherited_lease",
        lambda fd: events.append(("lease", fd)),
    )
    monkeypatch.setattr(probe, "_live_owner", lambda *_args: _service())
    monkeypatch.setattr(
        probe,
        "source_binding",
        lambda _source: (_ for _ in ()).throw(ValueError("source closure")),
    )
    monkeypatch.setattr(
        probe.wp,
        "get_device",
        lambda *_args: pytest.fail("CUDA queried before source closure"),
    )
    with pytest.raises(ValueError, match="source closure"):
        probe._child(SOURCE, 19, os.getpid(), "d" * 64)
    assert events == [("lease", 19)]


def test_fixture_declaration_report_and_reader_roundtrip(tmp_path, monkeypatch):
    fixture = _fixture()
    baseline = fixture["baseline"].astype("<f4", copy=False).tobytes()
    expected = fixture["expected"].astype("<f4", copy=False).tobytes()
    monkeypatch.setattr(probe, "BASELINE_SHA256", probe.base.digest(baseline))
    monkeypatch.setattr(probe, "EXPECTED_SHA256", probe.base.digest(expected))
    root = tmp_path / "fixture"
    root.mkdir()
    monkeypatch.setattr(probe, "output_path", lambda *_args: root)
    binding = {
        "source": SOURCE,
        "branch": "feat/athletics-obstacle-curriculum",
        "leaves": {},
    }
    monkeypatch.setattr(probe, "source_binding", lambda _source: binding)
    monkeypatch.setattr(probe, "_runtime_versions", lambda: {"python": "test"})
    monkeypatch.setattr(probe, "_terminal", lambda *_args: _service("fixture"))
    service = _service("fixture")
    predecessor = {
        "predecessors_authenticated": True,
        "original": {"source": "legacy"},
        "concurrent": {"source": "concurrent"},
        "serial_owner_packet_sha256": "1" * 64,
        "serial_closeout_packet_sha256": "2" * 64,
        "serial_run_files": {},
        "serial_test_files": {},
    }
    order_inputs = (None, {"source": "parent"}, None, {}, {})
    report = probe._write_fixture(
        SOURCE,
        binding,
        {"host": "test"},
        service,
        "3" * 64,
        predecessor,
        order_inputs,
        fixture,
        baseline,
        bytes(probe.MATRIX_BYTES),
        bytes(probe.MATRIX_BYTES),
        time.monotonic(),
        {"idle": True},
        {"filmbrain": "same"},
        {"protected": "inactive"},
    )
    report["idle_after"] = {"idle": True}
    probe.base.write_json(root / "report.json", report)
    record = probe._read_fixture(SOURCE, binding)
    assert record["declaration"]["tests_receipt_sha256"] == "3" * 64
    assert record["fixture"]["metadata"] == fixture["metadata"]
    assert record["inventory"]["baseline.bin"]["sha256"] == probe.base.digest(baseline)


def test_child_validates_full_declaration_and_all_inputs_before_warp_cuda(
    tmp_path, monkeypatch
):
    fixture = _fixture()
    raws = {
        "baseline.bin": fixture["baseline"].astype("<f4", copy=False).tobytes(),
        "capture-crb.bin": fixture["capture_crb"].astype("<f4", copy=False).tobytes(),
        "replay-crb.bin": fixture["replay_crb"].astype("<f4", copy=False).tobytes(),
        "expected.bin": fixture["expected"].astype("<f4", copy=False).tobytes(),
    }
    monkeypatch.setattr(
        probe, "BASELINE_SHA256", probe.base.digest(raws["baseline.bin"])
    )
    monkeypatch.setattr(
        probe, "EXPECTED_SHA256", probe.base.digest(raws["expected.bin"])
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", lambda _fd: None)
    service = _service()
    monkeypatch.setattr(probe, "_live_owner", lambda *_args: service)
    binding = {
        "source": SOURCE,
        "branch": "feat/athletics-obstacle-curriculum",
        "leaves": {},
    }
    monkeypatch.setattr(probe, "source_binding", lambda _source: binding)
    host_identity = {"host": "test"}
    monkeypatch.setattr(probe.base.host, "identity", lambda _source: host_identity)
    monkeypatch.setattr(probe, "_runtime_versions", lambda: {"python": "test"})
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(probe, "output_path", lambda *_args: root)
    inventory = {
        name: {"bytes": len(raw), "sha256": probe.base.digest(raw)}
        for name, raw in raws.items()
    }
    declaration = {
        "protocol": probe.PROTOCOL + ":run",
        "source": SOURCE,
        "source_binding": binding,
        "host_identity": host_identity,
        "service_properties": service,
        "tests_receipt_sha256": "4" * 64,
        "fixture_declaration_sha256": "5" * 64,
        "fixture_report_sha256": "6" * 64,
        "fixture_artifacts": inventory,
        "fixture_metadata": fixture["metadata"],
        "topology": fixture["topology"],
        "topology_sha256": probe.LAUNCH_TOPOLOGY_SHA256,
        "concurrent_levels": LEVELS,
        "serial_groups": GROUPS,
        "schedule": {"concurrent": LEVELS, "serial": GROUPS},
        "runtime_versions": {"python": "test"},
        "smooth_sha256": probe.SMOOTH_SHA256,
        "expected_sha256": inventory["expected.bin"]["sha256"],
        "predecessor_summary": {},
        **probe.FLAGS,
    }
    declaration_raw = (canonical(declaration) + "\n").encode()
    (root / "declaration.json").write_bytes(declaration_raw)
    for name, raw in raws.items():
        (root / name).write_bytes(raw)
    original_read = probe._read_file

    def read_file(path, limit):
        path = Path(path)
        if path.parent == root and path.name in (set(raws) | {"declaration.json"}):
            raw = (
                declaration_raw if path.name == "declaration.json" else raws[path.name]
            )
            assert len(raw) <= limit
            return raw
        if path.name == "smooth.py":
            raw = Path(probe.smooth.__file__).read_bytes()
            assert len(raw) <= limit
            return raw
        return original_read(path, limit)

    monkeypatch.setattr(probe, "_read_file", read_file)
    monkeypatch.setattr(
        probe.wp,
        "get_device",
        lambda _name: (_ for _ in ()).throw(ValueError("reached Warp device query")),
    )
    with pytest.raises(ValueError, match="reached Warp device query"):
        probe._child(SOURCE, 19, os.getpid(), probe.base.digest(declaration_raw))


def test_child_receipt_exactly_binds_launch_and_non_admitting_flags():
    binding = {"source": SOURCE, "leaves": {}}
    service = _service()
    host_identity = {"source": SOURCE}
    declaration = {
        "runtime_versions": probe._runtime_versions(),
        "concurrent_levels": LEVELS,
        "serial_groups": GROUPS,
    }
    launch = {
        "kernel": probe.KERNEL_NAME,
        "worlds": 64,
        "repeats_per_schedule": 32,
        "concurrent_levels": LEVELS,
        "serial_groups": GROUPS,
        "concurrent_calls_per_repeat": 7,
        "serial_calls_per_repeat": 9,
        "total_launch_calls": 32 * 16,
        "same_working_input_output": True,
        "same_stream": True,
        "reset_once_before_each_full_schedule": True,
        "snapshot_after_complete_schedule": True,
        "sync_once_after_both_banks": True,
        "readback_after_sync_only": True,
        "parent_zero_noops_preserved": [1, 0],
    }
    receipt = {
        "protocol": probe.PROTOCOL + ":child",
        "source": SOURCE,
        "source_binding": binding,
        "declaration_sha256": "e" * 64,
        "owner_pid": os.getpid(),
        "child_pid": os.getpid() + 1,
        "child_ppid": os.getpid(),
        "service_properties": service,
        "host_identity": host_identity,
        "runtime_versions": declaration["runtime_versions"],
        "smooth_sha256": probe.SMOOTH_SHA256,
        "launch": launch,
        "torch_rng_preserved": True,
        "torch_rng_state_bytes": 5056,
        "torch_cuda_initialized": False,
        "actor_constructed": False,
        "model_constructed": False,
        "optimizer_constructed": False,
        "random_kernels": False,
        "concurrent_sha256": "1" * 64,
        "serial_sha256": "2" * 64,
        "rng_sha256": "3" * 64,
        **probe.FLAGS,
    }
    probe._validate_child_receipt(
        receipt, SOURCE, binding, declaration, "e" * 64, service, host_identity
    )
    receipt["model_constructed"] = True
    with pytest.raises(ValueError, match="child receipt"):
        probe._validate_child_receipt(
            receipt, SOURCE, binding, declaration, "e" * 64, service, host_identity
        )


def test_cpu_child_environments_are_sanitized_and_old_live_gates_are_not_called(
    monkeypatch,
):
    monkeypatch.setenv("FULL_TREE_SECRET", "must-not-leak")
    assert probe._cpu_env()["CUDA_VISIBLE_DEVICES"] == ""
    assert probe._gpu_env()["CUDA_VISIBLE_DEVICES"] == "0"
    assert "FULL_TREE_SECRET" not in probe._cpu_env()
    source = Path(probe.__file__).read_text()
    assert "frozen.completed(" not in source
    assert "serial.completed(" not in source
    assert "serial._read_tests(" not in source
    assert (
        "concurrent_raw == expected_repeat and serial_raw == expected_repeat"
        not in source
    )


def test_fixed_artifact_sizes_and_total_inventory_cap(tmp_path):
    for name, size in (
        ("baseline.bin", probe.MATRIX_BYTES),
        ("concurrent.bin", probe.OUTPUT_BYTES),
        ("serial.bin", probe.OUTPUT_BYTES),
    ):
        (tmp_path / name).write_bytes(b"x" * size)
    items = probe._inventory(tmp_path, {"baseline.bin", "concurrent.bin", "serial.bin"})
    assert items["serial.bin"]["bytes"] == probe.OUTPUT_BYTES
    assert sum(record["bytes"] for record in items.values()) < probe.TOTAL_BYTES


@pytest.mark.parametrize(
    "elapsed", [True, False, -1, 0, float("nan"), float("inf"), 300]
)
def test_elapsed_report_gate_rejects_nonpositive_nonfinite_bool_or_cap(elapsed):
    assert probe._valid_elapsed(elapsed, 300) is False
    assert probe._valid_elapsed(1.5, 300) is True


@pytest.mark.parametrize(
    "damage",
    [
        "missing-pids",
        "empty-services",
        "float-memory",
        "bool-pid",
        "nan-elapsed",
        "duplicate-elapsed",
    ],
)
def test_monitor_requires_complete_typed_ordered_wsl_samples(damage):
    row = {
        "child_pid": 321,
        "elapsed_seconds": 0.0,
        "sample": {
            "temperature_c": 40,
            "memory_used_mib": 1024,
            "memory_free_mib": 12000,
            "compute_pids": [],
            "services": {
                name: "inactive"
                for name in (
                    *probe.base.gpu_idle_gate.SERVICES,
                    *("user:" + name for name in probe.base.gpu_idle_gate.SERVICES),
                )
            },
        },
    }
    assert probe._monitor([row]) == 321  # Idle polls do not imply observed ownership.
    rows = [row]
    if damage == "missing-pids":
        del row["sample"]["compute_pids"]
    elif damage == "empty-services":
        row["sample"]["services"] = {}
    elif damage == "float-memory":
        row["sample"]["memory_used_mib"] = 1024.0
    elif damage == "bool-pid":
        row["child_pid"] = True
    elif damage == "nan-elapsed":
        row["elapsed_seconds"] = float("nan")
    else:
        rows.append(row.copy())
    with pytest.raises(ValueError):
        probe._monitor(rows)


@pytest.mark.parametrize(
    "damage",
    ["none", "extra-report-key", "negative-elapsed", "bool-elapsed", "bool-skip-count"],
)
def test_tests_reader_closes_report_schema_and_plain_counts(
    tmp_path, monkeypatch, damage
):
    binding = {"source": SOURCE}
    service = _service("tests")
    log = f"{probe.EXPECTED_TESTS} passed in 1.00s\n".encode()
    receipt = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "source_binding": binding,
        "test_files": list(probe.TEST_FILES),
        "passed": probe.EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": probe.base.digest(log),
        "service_properties": service,
        **probe.FLAGS,
    }
    report = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "source_binding": binding,
        "mode": "tests",
        "status": "passed",
        "host_identity": {"source": SOURCE},
        "service_properties": service,
        "passed": probe.EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": receipt["pytest_sha256"],
        "elapsed_seconds": 2.0,
        "idle_before": {},
        "idle_after": {},
        "filmbrain": {},
        "protected_services": {},
        **probe.FLAGS,
    }
    if damage == "extra-report-key":
        report["extra"] = None
    elif damage == "negative-elapsed":
        report["elapsed_seconds"] = -1.0
    elif damage == "bool-elapsed":
        report["elapsed_seconds"] = True
    elif damage == "bool-skip-count":
        receipt["skips"] = False
    (tmp_path / "pytest.log").write_bytes(log)
    probe.base.write_json(tmp_path / "receipt.json", receipt)
    probe.base.write_json(tmp_path / "report.json", report)
    monkeypatch.setattr(probe, "output_path", lambda *_args: tmp_path)
    monkeypatch.setattr(probe, "_terminal", lambda *_args: None)
    if damage == "none":
        assert probe._read_tests(SOURCE, binding)
    else:
        with pytest.raises(ValueError):
            probe._read_tests(SOURCE, binding)
