"""CPU-only contract tests for the bounded Ada runtime smoke."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mjlab_microduck import ada_runtime_smoke as smoke


def sample(used=5000, free=11376, utilization=5, temperature=40):
    raw = (f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, {used}, {free}, "
           f"{utilization}, {temperature}")
    return {"gpu": smoke.parse_gpu(raw)}


def receipt():
    values = [float(index * 2 + 1) for index in range(smoke.BOUNDS["elements"])]
    return dict(protocol=smoke.PROTOCOL, gpu_uuid=smoke.GPU, name=smoke.NAME,
                capability=[8, 9], torch_cuda="12.8", torch="2.9.1", warp="1.12.0",
                elements=32, torch_values=values.copy(), warp_values=values.copy(),
                torch_peak_bytes=4096, warp_precompiled_headers=False, flags=deepcopy(smoke.FLAGS))


def test_import_is_stdlib_only_and_inert_in_fresh_process():
    spec = importlib.util.find_spec("mjlab_microduck.ada_runtime_smoke")
    assert spec is not None and spec.origin == smoke.__file__
    code = (
        "import sys; import mjlab_microduck.ada_runtime_smoke; "
        "assert not {'torch', 'warp', 'mujoco', 'mujoco_warp'} & sys.modules.keys()"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_binary_source_reader_preserves_exact_whitespace_and_bytes():
    assert smoke.read(sys.executable, "-c", "import sys; sys.stdout.buffer.write(b' \\r\\nx \\n')", binary=True) == b" \r\nx \n"


@pytest.mark.parametrize("raw", [
    "", "one,two", f"GPU-wrong, {smoke.DRIVER}, 8.9, 16376, 7000, 9376, 5, 40",
    f"{smoke.GPU}, wrong, 8.9, 16376, 7000, 9376, 5, 40",
    f"{smoke.GPU}, {smoke.DRIVER}, 8.6, 16376, 7000, 9376, 5, 40",
    f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, N/A, 9376, 5, 40",
    f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, -1, 9376, 5, 40",
    f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, 7000, 9376, 101, 40",
    f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, 7000, 9376, 5, 121",
    f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 24576, 7000, 9376, 5, 40",
    "\n".join([f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, 7000, 9376, 5, 40"] * 2),
])
def test_malformed_or_changed_ada_telemetry_is_refused(raw):
    with pytest.raises(ValueError):
        smoke.parse_gpu(raw)


@pytest.mark.parametrize("field,value", [
    ("uuid", "GPU-other"), ("driver", "other"), ("capability", [8, 6]),
    ("free_mib", 10239), ("used_mib", 12289), ("temperature_c", 65),
    ("utilization_percent", 86),
])
def test_capacity_rejects_identity_and_each_resource_boundary(field, value):
    current = sample()
    current["gpu"][field] = value
    with pytest.raises(ValueError):
        smoke.capacity(current)


def test_exact_ada_capacity_boundaries_are_admitted():
    assert smoke.capacity(sample(used=6136, free=10240, utilization=85, temperature=64))


def test_impossible_used_plus_free_counters_are_rejected():
    with pytest.raises(ValueError, match="valid GPU counters"):
        sample(used=12288, free=10240)


@pytest.mark.parametrize("growth", [2048, 2049])
def test_growth_guard_is_per_gpu_against_its_own_baseline(growth):
    baseline = sample(used=4000, free=12376)
    current = sample(used=baseline["gpu"]["used_mib"] + growth,
                     free=baseline["gpu"]["free_mib"] - growth)
    if growth == 2048:
        assert smoke.capacity(current, baseline) is current
    else:
        with pytest.raises(ValueError, match="aggregate VRAM growth"):
            smoke.capacity(current, baseline)


def test_telemetry_queries_exact_device_fields(monkeypatch):
    calls = []

    def read(*args):
        calls.append(args)
        return f"{smoke.GPU}, {smoke.DRIVER}, 8.9, 16376, 7000, 9376, 5, 40"

    monkeypatch.setattr(smoke, "read", read)
    result = smoke.telemetry()
    assert calls == [("nvidia-smi", "--query-gpu=" +
                      "uuid,driver_version,compute_cap,memory.total,memory.used,memory.free,"
                      "utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits")]
    assert result["gpu"]["capability"] == [8, 9]


def test_child_receipt_is_closed_and_checks_all_32_values_and_false_flags():
    result = receipt()
    assert smoke.validate_child(result) is None


@pytest.mark.parametrize("which,index,value", [
    ("torch_values", 0, 0.0), ("warp_values", 31, 64.0),
    ("torch_values", 4, True), ("warp_values", 9, float("nan")),
])
def test_child_receipt_rejects_any_bad_affine_element(which, index, value):
    result = receipt()
    result[which][index] = value
    with pytest.raises(ValueError):
        smoke.validate_child(result)


@pytest.mark.parametrize("flag", list(smoke.FLAGS))
def test_child_receipt_never_promotes_any_qualification_flag(flag):
    result = receipt()
    result["flags"][flag] = True
    with pytest.raises(ValueError, match="qualification flags"):
        smoke.validate_child(result)


@pytest.mark.parametrize("field,value", [
    ("protocol", "training"), ("gpu_uuid", "GPU-other"), ("name", "other"),
    ("capability", [8, 6]), ("torch_cuda", "13.0"), ("torch", "new"),
    ("warp", "new"), ("elements", 31), ("torch_peak_bytes", 0),
    ("torch_peak_bytes", smoke.BOUNDS["torch_allocator_bytes"] + 1),
    ("warp_precompiled_headers", True), ("warp_precompiled_headers", 0),
])
def test_child_receipt_rejects_changed_identity_or_bound(field, value):
    result = receipt()
    result[field] = value
    with pytest.raises(ValueError):
        smoke.validate_child(result)


def test_child_receipt_rejects_extra_keys_and_non_false_flag_values():
    result = receipt()
    result["training"] = False
    with pytest.raises(ValueError, match="closed child schema"):
        smoke.validate_child(result)
    result = receipt()
    result["flags"][next(iter(smoke.FLAGS))] = 0
    with pytest.raises(ValueError, match="qualification flags"):
        smoke.validate_child(result)


def test_inherited_lease_requires_same_regular_empty_owned_file(tmp_path, monkeypatch):
    path = tmp_path / "lease"
    path.touch()
    monkeypatch.setattr(smoke, "LOCK", path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        identity = smoke.lease_identity(fd)
        assert identity == {"device": os.fstat(fd).st_dev, "inode": os.fstat(fd).st_ino, "bytes": 0}
    finally:
        os.close(fd)

    path.write_text("not empty")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        with pytest.raises(ValueError, match="existing owned advisory lease"):
            smoke.lease_identity(fd)
    finally:
        os.close(fd)


def test_lease_rejects_symlink_and_non_regular_descriptor(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.touch()
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.setattr(smoke, "LOCK", link)
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        with pytest.raises(ValueError, match="existing owned advisory lease"):
            smoke.lease_identity(fd)
    finally:
        os.close(fd)

    monkeypatch.setattr(smoke, "LOCK", tmp_path)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(ValueError, match="existing owned advisory lease"):
            smoke.lease_identity(fd)
    finally:
        os.close(fd)


def test_write_json_is_exclusive_finite_and_bounded(tmp_path):
    target = tmp_path / "receipt.json"
    smoke.write_json(target, {"ok": True})
    assert json.loads(target.read_bytes()) == {"ok": True}
    with pytest.raises(FileExistsError):
        smoke.write_json(target, {"ok": False})
    with pytest.raises(ValueError):
        smoke.write_json(tmp_path / "nan.json", {"value": float("nan")})
    with pytest.raises(ValueError, match="bounded evidence"):
        smoke.write_json(tmp_path / "large.json", {"value": "x" * 300000})


def owner_fixture(monkeypatch, tmp_path, samples, *, child_returncode=0,
                  service_states=None, foreign_states=None, watchdog=None):
    root = tmp_path / "repo"
    root.mkdir()
    lock = tmp_path / "existing.lock"
    lock.touch()
    output = tmp_path / "fresh-evidence"
    monkeypatch.setattr(smoke, "ROOT", root)
    monkeypatch.setattr(smoke, "LOCK", lock)
    monkeypatch.setattr(smoke, "directory", lambda source: output)
    monkeypatch.setattr(smoke, "source_check", lambda source: {"commit": source})
    monkeypatch.setattr(smoke, "owned_service", lambda source: {"unit": "owned-test"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Model the fresh CPU owner without mutating pytest's process-wide imports.
    monkeypatch.setattr(smoke, "sys", SimpleNamespace(modules={}, executable=sys.executable))
    monkeypatch.setattr(smoke, "READ_DEADLINE", None)

    telemetry_items = iter(samples)

    def telemetry():
        item = next(telemetry_items)
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)

    monkeypatch.setattr(smoke, "telemetry", telemetry)
    service_items = iter(service_states or [{"service": "inactive"}] * 10)

    def service_snapshot():
        return next(service_items)

    monkeypatch.setattr(smoke, "service_snapshot", service_snapshot)
    foreign_items = iter(foreign_states or [[], [], []])
    monkeypatch.setattr(smoke, "foreign_processes", lambda owned_pid=None: next(foreign_items))
    actions = []

    class OwnedChild:
        pid = 54321

        def __init__(self, command, **kwargs):
            actions.append("spawn-owned-child")
            self.returncode = child_returncode
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            assert kwargs["env"]["CUDA_CACHE_DISABLE"] == "1"
            assert len(kwargs["pass_fds"]) == 1
            assert all(Path(kwargs["env"][key]).is_dir() for key in smoke.CACHES)
            (output / "child.log").touch(exist_ok=True)
            if child_returncode == 0:
                smoke.write_json(output / "child.json", receipt())

        def poll(self):
            return self.returncode

        def terminate(self):
            actions.append("terminate-owned-child")
            if self.returncode is None:
                self.returncode = -15

        def wait(self, timeout):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("owned-child", timeout)
            return self.returncode

        def kill(self):
            actions.append("kill-owned-child")
            self.returncode = -9

    monkeypatch.setattr(smoke.subprocess, "Popen", OwnedChild)
    if watchdog is not None:
        class ImmediateWatchdog:
            def __init__(self, delay, callback):
                self.callback = callback

            def start(self):
                self.callback()

            def cancel(self):
                pass

            def join(self, timeout):
                pass

        monkeypatch.setattr(smoke.threading, "Timer", ImmediateWatchdog)
    return output, actions


def test_owner_success_retains_receipt_and_diagnostic_only_flags(monkeypatch, tmp_path):
    foreign = [{"pid": 1592, "name": "GroundingDINO"}]
    output, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample(used=5001, free=11375), sample()],
                                    foreign_states=[foreign, foreign, foreign])
    smoke.supervise("a" * 40)
    report = json.loads((output / "report.json").read_bytes())
    assert report["decision"] == "ada-torch-warp-smoke-complete-not-training"
    assert report["foreign_before"] == foreign
    assert report["foreign_unchanged"] and report["services_unchanged"] and report["lease_unchanged"]
    assert len(report["telemetry"]) == 3 and actions == ["spawn-owned-child"]
    assert set(report["files"]) == {"launch.json", "child.log", "child.json"}
    assert not any(report["flags"].values())


def test_admission_refusal_never_spawns_or_touches_foreign_workload(monkeypatch, tmp_path):
    output, actions = owner_fixture(monkeypatch, tmp_path, [sample(free=10239)])
    with pytest.raises(ValueError):
        smoke.supervise("a" * 40)
    report = json.loads((output / "report.json").read_bytes())
    assert actions == [] and report["decision"] == "failed"
    assert report["files"] == {} and report["telemetry"] == [sample(free=10239)]


@pytest.mark.parametrize("changed", ["telemetry", "foreign", "service"])
def test_owner_monitor_change_stops_only_child_and_retains_failure(monkeypatch, tmp_path, changed):
    telemetry_items = [sample(), sample(temperature=65), sample()]
    services = [{"service": "inactive"}, {"service": "changed"}]
    foreign = [[{"pid": 1592, "name": "GroundingDINO"}],
               [{"pid": 1592, "name": "GroundingDINO"}, {"pid": 9000, "name": "other"}],
               [{"pid": 1592, "name": "GroundingDINO"}, {"pid": 9000, "name": "other"}]]
    if changed == "telemetry":
        telemetry_items = [sample(), RuntimeError("telemetry lost")]
    elif changed == "foreign":
        telemetry_items = [sample(), sample()]
    else:
        telemetry_items = [sample(), sample()]
    output, actions = owner_fixture(
        monkeypatch, tmp_path, telemetry_items,
        child_returncode=None,
        service_states=services if changed == "service" else None,
        foreign_states=foreign if changed == "foreign" else None)
    with pytest.raises((ValueError, RuntimeError)):
        smoke.supervise("a" * 40)
    report = json.loads((output / "report.json").read_bytes())
    assert actions == ["spawn-owned-child", "terminate-owned-child"]
    assert report["decision"] == "failed" and report["child_exit"] == -15
    assert not any(report["flags"].values())


def test_child_failure_is_retained_without_success_or_retry(monkeypatch, tmp_path):
    output, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample()], child_returncode=2)
    with pytest.raises(ValueError, match="owned child failed"):
        smoke.supervise("a" * 40)
    report = json.loads((output / "report.json").read_bytes())
    assert report["decision"] == "failed" and report["child_exit"] == 2
    assert actions == ["spawn-owned-child"]


def test_independent_watchdog_kills_only_owned_child_and_reports_failure(monkeypatch, tmp_path):
    output, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample()],
                                    child_returncode=None, watchdog=True)
    with pytest.raises(ValueError, match="owned child deadline"):
        smoke.supervise("a" * 40)
    report = json.loads((output / "report.json").read_bytes())
    assert actions == ["spawn-owned-child", "kill-owned-child"]
    assert report["child_exit"] == -9 and report["decision"] == "failed"

