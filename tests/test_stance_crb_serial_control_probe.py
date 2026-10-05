"""CPU and mocked lifecycle tests for the serial CRB schedule supervisor."""

from hashlib import sha256
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest

from mjlab_microduck import stance_crb_serial_control_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "a" * 40
INVOCATION = "c" * 32


def _properties(mode="run", pid=None, invocation=INVOCATION):
    return {
        "MainPID": str(os.getpid() if pid is None else pid),
        "ActiveState": "active",
        "RuntimeMaxUSec": "5min" if mode == "tests" else "10min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": invocation,
        "RemainAfterExit": "yes",
    }


def _terminal(mode, invocation):
    return {
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "RemainAfterExit": "yes",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "5min" if mode == "tests" else "10min",
        "InvocationID": invocation,
    }


def test_names_caps_reserves_cutoff_and_owner_frozen_count():
    assert probe.BASE_SOURCE == "bed9f99feb4dca91d78a99f00b5cdba0361933aa"
    assert probe.SYNC_FROM_SOURCE == "b72cf952a265ac69bbe6affec470f39dbf3ad042"
    assert (
        probe.unit(SOURCE, "tests")
        == f"microduck-crb-serial-tests-{SOURCE[:12]}.service"
    )
    assert (
        probe.unit(SOURCE, "run") == f"microduck-crb-serial-run-{SOURCE[:12]}.service"
    )
    assert (
        probe.source_sync_unit(SOURCE)
        == f"microduck-crb-serial-sync-{SOURCE[:12]}.service"
    )
    assert (
        probe.output_path(SOURCE, "tests").name
        == f"stance-crb-serial-tests-{SOURCE[:12]}"
    )
    assert probe.CAP == 600 and probe.TEST_CAP == 300 and probe.CHILD_SECONDS == 240
    assert probe.RUN_RESERVE == 960 and probe.TEST_RESERVE == 1260
    assert probe.MEMORY == 6 * 1024**3 and probe.EXPECTED_TESTS == 1436
    assert set(probe.frozen.FLAGS) <= set(probe.FLAGS)
    assert all(probe.FLAGS[name] is False for name in probe.frozen.FLAGS)
    assert len(probe.TEST_FILES) == 48
    with pytest.raises(ValueError):
        probe.unit("A" * 40, "run")
    with pytest.raises(ValueError):
        probe.unit(SOURCE, "child")
    with pytest.raises(ValueError, match="cutoff"):
        probe._check_window(reserve_seconds=1, now=probe.CUTOFF)
    probe._check_window(reserve_seconds=1, now=probe.CUTOFF - 2)


def test_retained_properties_are_exact_per_mode():
    assert probe._recorded_properties(_properties("tests"), "tests")
    assert probe._recorded_properties(_properties(), "run")
    for field, bad in (
        ("MemoryMax", "1"),
        ("RuntimeMaxUSec", "5min"),
        ("CPUQuotaPerSecUSec", "4s"),
        ("Nice", "0"),
        ("KillMode", "process"),
    ):
        values = _properties()
        values[field] = bad
        with pytest.raises(ValueError):
            probe._recorded_properties(values, "run")


def test_sync_template_retargets_exact_native_head_and_owned_unit(monkeypatch):
    template = (
        f'test "$(git rev-parse HEAD)" = {probe.frozen.SYNC_FROM_SOURCE}\n'
        f'if test "$duck_sync_running" != {probe.frozen.source_sync_unit(SOURCE)}; then\n'
    )
    monkeypatch.setattr(
        probe.frozen, "source_sync_script", lambda *_args, **_kwargs: template
    )
    actual = probe.source_sync_script(SOURCE, bundle_sha256="d" * 64)
    assert actual.count(probe.SYNC_FROM_SOURCE) == 1
    assert actual.count(probe.source_sync_unit(SOURCE)) == 1
    assert probe.frozen.SYNC_FROM_SOURCE not in actual
    assert probe.frozen.source_sync_unit(SOURCE) not in actual


def test_source_binding_limits_changes_and_authenticates_all_frozen_leaves(monkeypatch):
    root = Path(probe.__file__).resolve().parents[2]
    monkeypatch.setattr(probe.frozen.cpu_probe, "SYNC_ROOT", str(root))
    paths = sorted(probe._frozen_paths() | set(probe.OWN) | set(probe.RELATED))
    leaves = {name: ("content:" + name).encode() for name in paths}
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
            stdout = leaves[command[1].split(":", 1)[1]]
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
    changed.append("src/extra.py")
    with pytest.raises(ValueError, match="five declared"):
        probe.source_binding(SOURCE)


def test_serial_batch_uses_exact_96_launch_sequence_one_stream_and_no_intermediate_sync(
    monkeypatch,
):
    class FakeArray:
        def __init__(self, data=None, label=None):
            self.data = data
            self.label = label

        def numpy(self):
            assert records["syncs"] == 1
            assert len(records["launches"]) == 96
            assert len(records["copies"]) == 64
            records["events"].append("readback")
            return np.zeros(
                (probe.control.WORLDS, probe.control.BODIES, probe.control.COMPONENTS),
                dtype=np.float32,
            )

    stream = object()
    device = SimpleNamespace(is_cuda=True)
    records = {"launches": [], "copies": [], "syncs": 0, "arrays": [], "events": []}
    monkeypatch.setattr(probe.wp, "ScopedDevice", lambda _device: nullcontext())

    def make_array(data=None, **_kwargs):
        value = FakeArray(data, "array")
        records["arrays"].append(value)
        return value

    def make_empty(*, shape, **_kwargs):
        value = FakeArray(label=f"empty-{len(records['arrays'])}")
        records["arrays"].append(value)
        return value

    def copy(destination, source, *, stream=None):
        records["events"].append("copy")
        records["copies"].append((destination, source, stream))

    def launch(kernel, *, dim, inputs, outputs, device=None, stream=None):
        records["events"].append("launch")
        records["launches"].append((kernel, dim, inputs, outputs, device, stream))

    def synchronize(_device):
        records["events"].append("synchronize")
        records["syncs"] += 1

    monkeypatch.setattr(probe.wp, "array", make_array)
    monkeypatch.setattr(probe.wp, "empty", make_empty)
    monkeypatch.setattr(probe.wp, "get_stream", lambda _device: stream)
    monkeypatch.setattr(probe.wp, "copy", copy)
    monkeypatch.setattr(probe.wp, "launch", launch)
    monkeypatch.setattr(
        probe.wp,
        "synchronize_device",
        synchronize,
    )
    raw = probe._serial_batch(
        device,
        np.zeros((64, 16, 10), dtype=np.float32),
        np.asarray(
            [0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14], dtype=np.int32
        ),
    )

    assert len(records["launches"]) == 32 * 3
    assert records["syncs"] == 1
    assert len(records["copies"]) == 64
    assert len(raw) == probe.OUTPUT_BYTES
    assert records["events"] == (
        ["copy", "launch", "launch", "launch", "copy"] * 32
        + ["synchronize"]
        + ["readback"] * 32
    )
    for repeat_index in range(32):
        reset = records["copies"][repeat_index * 2]
        snapshot = records["copies"][repeat_index * 2 + 1]
        triplet = records["launches"][repeat_index * 3 : repeat_index * 3 + 3]
        working = reset[0]
        assert reset[1] is records["arrays"][4]
        assert all(copy_record[2] is stream for copy_record in (reset, snapshot))
        assert [call[1] for call in triplet] == [(64, 1)] * 3
        assert [call[2][2].data.tolist() for call in triplet] == [[2], [7], [11]]
        assert all(
            call[0] is probe.KERNEL
            and call[2][1] is working
            and call[3][0] is working
            and call[5] is stream
            for call in triplet
        )
        assert snapshot[0] is not working and snapshot[1] is working


def test_child_inherited_lease_and_source_guards_precede_warp_cuda(monkeypatch):
    events = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        probe.base.training_smoke,
        "inherited_lease",
        lambda fd: events.append(("lease", fd)),
    )
    monkeypatch.setattr(probe, "_live_owner", lambda *_args: _properties())
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
        probe._child(SOURCE, 17, os.getpid() - 1, "e" * 64)
    assert events == [("lease", 17)]


def test_concurrent_owner_packet_binds_38_files_and_terminal_invocations(monkeypatch):
    original = {"legacy": "authenticated-27"}
    binding = {"source": probe.CONCURRENT_SOURCE}
    run_files = {name: {"bytes": 1, "sha256": "1" * 64} for name in probe.RUN_FILES}
    tests_files = {
        name: {"bytes": 1, "sha256": "2" * 64} for name in probe.TEST_FILES_ON_DISK
    }
    pytest_log = f"{probe.frozen.EXPECTED_TESTS} passed in 1.00s\n".encode()
    test_service = _properties("tests", invocation=probe.CONCURRENT_TESTS_INVOCATION)
    run_service = _properties("run", invocation=probe.CONCURRENT_RUN_INVOCATION)
    test_receipt = {
        "protocol": probe.frozen.PROTOCOL + ":tests",
        "source": probe.CONCURRENT_SOURCE,
        "source_binding": binding,
        "test_files": list(probe.frozen.TEST_FILES),
        "passed": probe.frozen.EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": probe.base.digest(pytest_log),
        "service_properties": test_service,
        **probe.frozen.FLAGS,
    }
    test_report = {
        "protocol": probe.frozen.PROTOCOL + ":tests",
        "source": probe.CONCURRENT_SOURCE,
        "mode": "tests",
        "status": "passed",
        "source_binding": binding,
        "service_properties": test_service,
        "passed": probe.frozen.EXPECTED_TESTS,
        "pytest_sha256": test_receipt["pytest_sha256"],
        **probe.frozen.FLAGS,
    }
    result = {"bit_variation_observed": True}
    host_identity = {"source": probe.CONCURRENT_SOURCE}
    filmbrain = {"filmbrain": "steady"}
    protected = {"worker": "inactive", "user-worker": "inactive"}
    report = {
        "protocol": probe.frozen.PROTOCOL + ":run",
        "source": probe.CONCURRENT_SOURCE,
        "status": "passed",
        "source_binding": binding,
        "result": result,
        "service_properties": run_service,
        "host_identity": host_identity,
        "filmbrain": filmbrain,
        "protected_services": protected,
        **probe.frozen.FLAGS,
    }
    idle = {
        "protocol": "two-idle-samples-v1",
        "samples": [
            {
                "compute_pids": "",
                "temperature_c": 30,
                "memory_mib": 643,
                "services": {"worker": "inactive"},
            }
        ]
        * 2,
        "timeout_s": 10,
    }
    packet = {
        "filmbrain": filmbrain,
        "gpu_initialized": False,
        "host_identity": host_identity,
        "idle": idle,
        "original": original,
        "protected_services": protected,
        "protocol": probe.frozen.PROTOCOL + ":independent-owner-closeout",
        "result": result,
        "rollout_cause_proven": False,
        "run_files": run_files,
        "source": probe.CONCURRENT_SOURCE,
        "source_binding": binding,
        "terminal_run": _terminal("run", probe.CONCURRENT_RUN_INVOCATION),
        "terminal_tests": _terminal("tests", probe.CONCURRENT_TESTS_INVOCATION),
        "tests_files": tests_files,
        "tests_receipt_sha256": "",
        "training_authorized": False,
    }
    report_raw = (canonical(report) + "\n").encode()
    tests_receipt_raw = (canonical(test_receipt) + "\n").encode()
    tests_report_raw = (canonical(test_report) + "\n").encode()
    packet["tests_receipt_sha256"] = sha256(tests_receipt_raw).hexdigest()
    run_files["report.json"] = {
        "bytes": len(report_raw),
        "sha256": sha256(report_raw).hexdigest(),
    }
    tests_files["receipt.json"] = {
        "bytes": len(tests_receipt_raw),
        "sha256": sha256(tests_receipt_raw).hexdigest(),
    }
    tests_files["pytest.log"] = {
        "bytes": len(pytest_log),
        "sha256": sha256(pytest_log).hexdigest(),
    }
    tests_files["report.json"] = {
        "bytes": len(tests_report_raw),
        "sha256": sha256(tests_report_raw).hexdigest(),
    }
    packet_raw = (canonical(packet) + "\n").encode()
    monkeypatch.setattr(probe, "OWNER_PACKET_SHA256", sha256(packet_raw).hexdigest())
    monkeypatch.setattr(
        probe, "CONCURRENT_REPORT_SHA256", sha256(report_raw).hexdigest()
    )
    monkeypatch.setattr(
        probe, "CONCURRENT_TESTS_RECEIPT_SHA256", packet["tests_receipt_sha256"]
    )
    monkeypatch.setattr(probe, "_recorded_properties", lambda *_args: True)
    validated, validated_report = probe._validate_concurrent_packet(
        packet_raw,
        report_raw,
        run_files,
        tests_files,
        tests_receipt_raw,
        pytest_log,
        tests_report_raw,
        original,
    )
    assert validated == packet and validated_report == report

    bad_files = dict(run_files)
    bad_files["outputs.bin"] = {"bytes": 2, "sha256": "3" * 64}
    with pytest.raises(ValueError, match="seven run"):
        probe._validate_concurrent_packet(
            packet_raw,
            report_raw,
            bad_files,
            tests_files,
            tests_receipt_raw,
            pytest_log,
            tests_report_raw,
            original,
        )


def test_terminal_packet_rejects_restart_or_wrong_invocation():
    terminal = _terminal("run", probe.CONCURRENT_RUN_INVOCATION)
    assert probe._terminal_packet(terminal, "run") is None
    terminal["NRestarts"] = "1"
    with pytest.raises(ValueError):
        probe._terminal_packet(terminal, "run")


def test_live_terminal_authenticates_exit_identity_and_all_resource_caps(monkeypatch):
    values = {
        "MainPID": "0",
        "ActiveState": "active",
        "SubState": "exited",
        "RemainAfterExit": "yes",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": INVOCATION,
        "RuntimeMaxUSec": "10min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": probe.CPU_QUOTA,
        "Nice": probe.NICE,
        "KillMode": probe.KILL_MODE,
    }
    monkeypatch.setattr(
        probe.base.host, "read", lambda *_args: values[_args[_args.index("-p") + 1]]
    )
    observed = probe._terminal(SOURCE, "run", INVOCATION)
    assert observed == values
    for field, bad_value in (
        ("RuntimeMaxUSec", "5min"),
        ("MemoryMax", "1"),
        ("CPUQuotaPerSecUSec", "1s"),
        ("Nice", "0"),
        ("KillMode", "process"),
    ):
        values[field] = bad_value
        with pytest.raises(ValueError, match="resource caps"):
            probe._terminal(SOURCE, "run", INVOCATION)
        values[field] = {
            "RuntimeMaxUSec": "10min",
            "MemoryMax": str(probe.MEMORY),
            "CPUQuotaPerSecUSec": probe.CPU_QUOTA,
            "Nice": probe.NICE,
            "KillMode": probe.KILL_MODE,
        }[field]


def test_subprocess_environment_sanitization_and_no_live_old_reader_calls(monkeypatch):
    monkeypatch.setenv("SERIAL_CONTROL_SECRET", "must-not-leak")
    assert probe._cpu_env()["CUDA_VISIBLE_DEVICES"] == ""
    assert probe._gpu_env()["CUDA_VISIBLE_DEVICES"] == "0"
    assert "SERIAL_CONTROL_SECRET" not in probe._cpu_env()
    source = Path(probe.__file__).read_text()
    assert "frozen._read_tests(" not in source
    assert "frozen.completed(" not in source
    assert "frozen.source_binding(" not in source


def test_inventory_carries_fixed_binary_limits_and_total_cap(tmp_path):
    (tmp_path / "baseline.bin").write_bytes(b"b" * probe.BASELINE_BYTES)
    (tmp_path / "outputs.bin").write_bytes(b"o" * probe.OUTPUT_BYTES)
    result = probe._inventory(tmp_path, {"baseline.bin", "outputs.bin"})
    assert result["outputs.bin"]["bytes"] == probe.OUTPUT_BYTES
    assert sum(record["bytes"] for record in result.values()) <= probe.TOTAL_BYTES


def test_current_cpu_receipt_report_and_terminal_are_bound(monkeypatch, tmp_path):
    binding = {"source": SOURCE, "branch": "feat/athletics-obstacle-curriculum"}
    service = _properties("tests")
    log = f"{probe.EXPECTED_TESTS} passed in 1.00s\n".encode()
    log_hash = sha256(log).hexdigest()
    receipt = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "source_binding": binding,
        "test_files": list(probe.TEST_FILES),
        "passed": probe.EXPECTED_TESTS,
        "skips": 0,
        "pytest_sha256": log_hash,
        "service_properties": service,
        **probe.FLAGS,
    }
    report = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "mode": "tests",
        "status": "passed",
        "source_binding": binding,
        "service_properties": service,
        "host_identity": {},
        "result": {"passed": probe.EXPECTED_TESTS, "pytest_sha256": log_hash},
        "idle_before": {},
        "idle_after": {},
        "filmbrain": {},
        "protected_services": {},
        "elapsed_seconds": 2.0,
        **probe.FLAGS,
    }

    def publish():
        receipt_raw = (canonical(receipt) + "\n").encode()
        (tmp_path / "receipt.json").write_bytes(receipt_raw)
        (tmp_path / "report.json").write_bytes((canonical(report) + "\n").encode())
        (tmp_path / "pytest.log").write_bytes(log)
        return sha256(receipt_raw).hexdigest()

    terminal_calls = []
    monkeypatch.setattr(probe, "output_path", lambda *_args: tmp_path)
    monkeypatch.setattr(probe, "_terminal", lambda *args: terminal_calls.append(args))
    expected = publish()
    assert probe._read_tests(SOURCE, binding) == expected
    assert terminal_calls == [(SOURCE, "tests", INVOCATION)]
    for elapsed in (-1.0, 0.0, probe.TEST_CAP, True):
        report["elapsed_seconds"] = elapsed
        publish()
        with pytest.raises(ValueError, match="mode cap"):
            probe._read_tests(SOURCE, binding)
    report["elapsed_seconds"] = 2.0
    report["result"]["passed"] -= 1
    publish()
    with pytest.raises(ValueError, match="report matches"):
        probe._read_tests(SOURCE, binding)
    report["result"]["passed"] += 1
    receipt["status"] = "passed"
    publish()
    with pytest.raises(ValueError, match="test receipt"):
        probe._read_tests(SOURCE, binding)
