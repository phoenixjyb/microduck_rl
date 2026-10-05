"""Contract tests for the fixed-input CRB kernel supervisor."""

from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import os

import numpy as np
import pytest

from mjlab_microduck import stance_crb_kernel_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "a" * 40
INVOCATION = "b" * 32


def _properties(mode="run", pid=None):
    return {
        "MainPID": str(os.getpid() if pid is None else pid),
        "ActiveState": "active",
        "RuntimeMaxUSec": "5min" if mode == "tests" else "10min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": INVOCATION,
        "RemainAfterExit": "yes",
    }


def test_names_caps_reserves_and_expired_window_are_explicit():
    assert probe.BASE_SOURCE == "2df4948e5af2c0e10636d61c8d70f0a685fd4d82"
    assert probe.SYNC_FROM_SOURCE == "f190d24dbb4b185f581462c36c0b2104a743cdfd"
    assert (
        probe.unit(SOURCE, "tests")
        == f"microduck-crb-kernel-tests-{SOURCE[:12]}.service"
    )
    assert (
        probe.unit(SOURCE, "run") == f"microduck-crb-kernel-run-{SOURCE[:12]}.service"
    )
    assert (
        probe.source_sync_unit(SOURCE)
        == f"microduck-crb-kernel-sync-{SOURCE[:12]}.service"
    )
    assert (
        probe.output_path(SOURCE, "run").name == f"stance-crb-kernel-run-{SOURCE[:12]}"
    )
    assert probe.RUN_RESERVE == 960
    assert probe.TEST_RESERVE == 1260
    assert probe.CAP == 600 and probe.TEST_CAP == 300 and probe.CHILD_SECONDS == 240
    assert probe.MEMORY == 6 * 1024**3
    assert probe.EXPECTED_TESTS == 1415
    for bad in ("BAD", "a" * 39, "A" * 40, "a" * 41):
        with pytest.raises(ValueError):
            probe.unit(bad, "run")
    with pytest.raises(ValueError):
        probe.unit(SOURCE, "child")
    with pytest.raises(ValueError, match="cutoff"):
        probe.check_window(reserve_seconds=1, now=probe.CUTOFF)
    probe.check_window(reserve_seconds=60, now=probe.CUTOFF - 61)


def test_recorded_service_caps_are_mode_specific_and_reject_pid_or_limit_drift():
    assert probe._recorded_properties(_properties("tests"), "tests")
    assert probe._recorded_properties(_properties(), "run")
    for mutation in (
        {"RuntimeMaxUSec": "5min"},
        {"MemoryMax": str(5 * 1024**3)},
        {"CPUQuotaPerSecUSec": "3s"},
        {"Nice": "0"},
        {"KillMode": "process"},
    ):
        value = _properties()
        value.update(mutation)
        with pytest.raises(ValueError):
            probe._recorded_properties(value, "run")
    value = _properties()
    value["MainPID"] = "0"
    with pytest.raises(ValueError):
        probe._recorded_properties(value, "run")


def test_source_sync_retargets_only_expected_base_and_unit(monkeypatch):
    template = (
        f'test "$(git rev-parse HEAD)" = {probe.cpu_probe.SYNC_FROM_SOURCE}\n'
        f'if test "$duck_sync_running" != {probe.cpu_probe.source_sync_unit(SOURCE)}; then\n'
    )
    monkeypatch.setattr(
        probe.cpu_probe, "source_sync_script", lambda *_a, **_k: template
    )
    rewritten = probe.source_sync_script(SOURCE, bundle_sha256="c" * 64)
    assert rewritten.count(probe.SYNC_FROM_SOURCE) == 1
    assert rewritten.count(probe.source_sync_unit(SOURCE)) == 1
    assert probe.cpu_probe.SYNC_FROM_SOURCE not in rewritten
    assert probe.cpu_probe.source_sync_unit(SOURCE) not in rewritten


@pytest.mark.parametrize("damage", [None, "invocation", "memory", "status"])
def test_terminal_service_keeps_exact_caps_and_successful_invocation(
    monkeypatch, damage
):
    value = dict(
        _properties(),
        MainPID="0",
        SubState="exited",
        NRestarts="0",
        ExecMainStatus="0",
        Result="success",
    )
    if damage == "invocation":
        value["InvocationID"] = "c" * 32
    elif damage == "memory":
        value["MemoryMax"] = str(7 * 1024**3)
    elif damage == "status":
        value["ExecMainStatus"] = "1"
    monkeypatch.setattr(probe.base.host, "read", lambda *args: value[args[-2]])
    if damage is None:
        assert probe.completed(SOURCE, "run", INVOCATION) == value
    else:
        with pytest.raises(ValueError, match="terminal"):
            probe.completed(SOURCE, "run", INVOCATION)


def test_source_binding_requires_clean_exact_native_committed_and_frozen_leaves(
    monkeypatch,
):
    root = Path(probe.__file__).resolve().parents[2]
    monkeypatch.setattr(probe.cpu_probe, "SYNC_ROOT", str(root))
    paths = sorted(probe._frozen_paths() | set(probe.OWN) | set(probe.RELATED))
    values = {path: ("leaf:" + path).encode() for path in paths}
    changed = sorted(probe.ALLOWED)

    def run(args, **kwargs):
        command = args[1:]
        if command[:2] == ["rev-parse", "HEAD"]:
            out = SOURCE
        elif command[:2] == ["branch", "--show-current"]:
            out = "feat/athletics-obstacle-curriculum"
        elif command[:3] == ["remote", "get-url", "origin"]:
            out = probe.cpu_probe.SYNC_ORIGIN
        elif command[:2] == ["status", "--porcelain"]:
            out = ""
        elif command[:2] == ["diff", "--name-only"]:
            out = "\n".join(changed)
        elif command[0] == "show":
            out = values[command[1].split(":", 1)[1]]
        else:
            out = b""
        return SimpleNamespace(stdout=out.encode() if isinstance(out, str) else out)

    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(
        probe.base,
        "_read_file",
        lambda path, _limit: values[str(path).replace(str(root) + "/", "")],
    )
    binding = probe.source_binding(SOURCE)
    assert binding["source"] == SOURCE and set(binding["leaves"]) == set(paths)

    changed.append("src/not-allowed.py")
    with pytest.raises(ValueError, match="declared"):
        probe.source_binding(SOURCE)


def test_inventory_enforces_per_file_and_total_caps_and_partial_symlink_safety(
    tmp_path,
):
    for name, data in (
        ("baseline.bin", b"b" * probe.BASELINE_BYTES),
        ("outputs.bin", b"o" * probe.OUTPUT_BYTES),
    ):
        (tmp_path / name).write_bytes(data)
    inventory = probe._inventory(tmp_path, {"baseline.bin", "outputs.bin"})
    assert inventory["outputs.bin"]["bytes"] == probe.OUTPUT_BYTES
    assert sum(item["bytes"] for item in inventory.values()) <= probe.TOTAL_BYTES
    with pytest.raises(OSError):
        probe._inventory(tmp_path, {"outputs.bin", "missing.bin"})

    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(b"secret")
    try:
        (tmp_path / "child.log").symlink_to(outside)
        partial = probe._partial_inventory(tmp_path)
        assert partial["child.log"] == {
            "present": True,
            "readable": False,
            "error_type": "UnsafeFile",
        }
    finally:
        outside.unlink(missing_ok=True)


def test_authenticated_test_receipt_requires_whole_log_report_and_terminal_invocation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 7)
    monkeypatch.setattr(probe, "output_path", lambda *_args: tmp_path)
    monkeypatch.setattr(probe.base, "_exact_inventory", lambda *_args: None)
    log = b".......                                                                  [100%]\n7 passed in 1.25s\n"
    service = _properties("tests")
    receipt = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "source_binding": {"source": SOURCE},
        "test_files": list(probe.TEST_FILES),
        "passed": 7,
        "skips": 0,
        "pytest_sha256": probe.base.digest(log),
        "service_properties": service,
        **probe.FLAGS,
    }
    report = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "mode": "tests",
        "status": "passed",
        "source_binding": receipt["source_binding"],
        "service_properties": service,
        "passed": 7,
        "pytest_sha256": receipt["pytest_sha256"],
        **probe.FLAGS,
    }
    data = {
        "receipt.json": (canonical(receipt) + "\n").encode(),
        "pytest.log": log,
        "report.json": (canonical(report) + "\n").encode(),
    }
    monkeypatch.setattr(probe, "_read_file", lambda path, _limit: data[Path(path).name])
    monkeypatch.setattr(probe, "completed", lambda *_args: {"InvocationID": INVOCATION})
    assert probe._read_tests(SOURCE, receipt["source_binding"]) == probe.base.digest(
        data["receipt.json"]
    )

    bad = dict(receipt)
    bad["skips"] = 1
    data["receipt.json"] = (canonical(bad) + "\n").encode()
    with pytest.raises(ValueError):
        probe._read_tests(SOURCE, receipt["source_binding"])


def test_child_inherited_lease_and_source_guard_precede_any_cuda_work(
    monkeypatch, tmp_path
):
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
        lambda _source: (_ for _ in ()).throw(ValueError("source guard")),
    )
    monkeypatch.setattr(
        probe.wp,
        "get_device",
        lambda *_args: pytest.fail("CUDA work before source guard"),
    )
    with pytest.raises(ValueError, match="source guard"):
        probe._child(SOURCE, 42, os.getpid() - 1, "d" * 64)
    assert events == [("lease", 42)]


def test_declaration_hash_mismatch_stops_child_before_runtime_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", lambda _fd: None)
    monkeypatch.setattr(probe, "_live_owner", lambda *_args: _properties())
    monkeypatch.setattr(probe, "source_binding", lambda _source: {"source": SOURCE})
    monkeypatch.setattr(probe, "output_path", lambda *_args: tmp_path)
    (tmp_path / "declaration.json").write_bytes(b"{}\n")
    monkeypatch.setattr(
        probe.base.host,
        "identity",
        lambda *_args: pytest.fail("host query before declaration hash"),
    )
    monkeypatch.setattr(
        probe.wp,
        "get_device",
        lambda *_args: pytest.fail("CUDA before declaration hash"),
    )
    with pytest.raises(ValueError, match="declaration exact hash"):
        probe._child(SOURCE, 13, os.getpid() - 1, "0" * 64)


def test_fixed_fixture_and_runtime_claims_are_bounded_and_non_admitting():
    assert probe.BASELINE_BYTES == 40960
    assert probe.OUTPUT_BYTES == 1310720
    assert probe.TOTAL_BYTES == 8 * 1024**2
    assert probe.FLAGS and all(value is False for value in probe.FLAGS.values())
    assert probe.KERNEL is probe.smooth._crb_accumulate
    assert probe.KERNEL_NAME == "_crb_accumulate"
    assert probe.runtime_versions()["python"] == probe.sys.version.split()[0]


def test_launch_topology_and_component_map_topology_hashes_are_distinct():
    assert (
        probe.LAUNCH_TOPOLOGY_SHA256
        == "2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19"
    )
    assert (
        probe.repeat.TOPOLOGY_SHA256
        == "5b445215f52d61d10bc892a14b0d0b15e3041ff3a5dfb6c1964f06211d0cb43a"
    )
    assert probe.LAUNCH_TOPOLOGY_SHA256 != probe.repeat.TOPOLOGY_SHA256


def test_subprocess_environments_are_sanitized_and_mode_specific(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_TEST_SECRET", "must-not-leak")
    cpu = probe._cpu_env()
    gpu = probe._gpu_env()
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0"
    for env in (cpu, gpu):
        assert env["ATEN_CPU_CAPABILITY"] == "default"
        assert env["MKL_CBWR"] == "COMPATIBLE"
        assert all(
            env[key] == "1"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
    assert "SUPERVISOR_TEST_SECRET" not in cpu and "SUPERVISOR_TEST_SECRET" not in gpu


def test_no_expired_historical_window_is_called():
    source = Path(probe.__file__).read_text()
    assert "prior.check_window(" not in source
    assert "cpu_probe.check_window(" not in source


def test_repeat_batch_resets_full_aliased_input_and_reads_only_after_all_launches(
    monkeypatch,
):
    """A mocked command schedule tests ownership/order, not native arithmetic."""
    events = []
    stream, device = object(), object()
    baseline = np.zeros((64, 16, 10), dtype="<f4")
    baseline[:, 1, :] = 2
    baseline[:, 2, :] = 3
    baseline[:, 7, :] = 5
    baseline[:, 11, :] = 7
    baseline[:, 15, 0] = np.float32(-0.0)

    class Array:
        def __init__(self, values):
            self.values = values.copy()

        def numpy(self):
            assert events.count("launch") == 32 and events.count("sync") == 1
            events.append("read")
            return self.values.copy()

    def allocate(values, *, dtype, device):
        return Array(np.asarray(values))

    def empty(*, shape, dtype, device):
        return Array(np.full((*shape, 10), np.nan, dtype="<f4"))

    def copy(target, source, *, stream):
        assert stream is expected_stream
        target.values[...] = source.values
        events.append("copy")

    def launch(kernel, *, dim, inputs, outputs, device, stream):
        assert kernel is probe.KERNEL and dim == (64, 3)
        assert stream is expected_stream and device is expected_device
        parents, working, level = inputs
        assert outputs[0] is working
        assert np.array_equal(working.values.view("<u4"), baseline.view("<u4"))
        assert parents.values.tolist() == probe.repeat.EXPECTED_PARENTS
        assert level.values.tolist() == [2, 7, 11]
        for child in level.values:
            working.values[:, 1] += working.values[:, child]
        events.append("launch")

    def sync(device):
        assert device is expected_device and events == ["copy", "launch", "copy"] * 32
        events.append("sync")

    expected_stream, expected_device = stream, device
    monkeypatch.setattr(probe.wp, "ScopedDevice", lambda _device: nullcontext())
    monkeypatch.setattr(probe.wp, "array", allocate)
    monkeypatch.setattr(probe.wp, "empty", empty)
    monkeypatch.setattr(probe.wp, "get_stream", lambda _device: stream)
    monkeypatch.setattr(probe.wp, "copy", copy)
    monkeypatch.setattr(probe.wp, "launch", launch)
    monkeypatch.setattr(probe.wp, "synchronize_device", sync)
    raw = probe._repeat_batch(
        device,
        baseline,
        np.asarray(probe.repeat.EXPECTED_PARENTS),
        np.asarray([2, 7, 11]),
    )
    result = np.frombuffer(raw, dtype="<f4").reshape(32, 64, 16, 10)
    assert len(raw) == probe.OUTPUT_BYTES and np.all(result[:, :, 1] == 17)
    assert np.all(result[:, :, 15, 0].view("<u4") == 0x80000000)
    assert events == ["copy", "launch", "copy"] * 32 + ["sync"] + ["read"] * 32
