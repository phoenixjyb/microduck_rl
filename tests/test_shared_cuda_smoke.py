"""CPU refusal tests; these do not initialize CUDA or qualify a simulator."""

from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from mjlab_microduck import shared_cuda_smoke as smoke


def sample(used=7422, free=16740, util=14, temp=32):
    raw = f"{smoke.GPU}, {smoke.DRIVER}, 24467, {used}, {free}, {util}, {temp}"
    return dict(wsl=smoke.parse_gpu(raw), windows=smoke.parse_gpu(raw))


def receipt():
    return dict(protocol=smoke.PROTOCOL, device="cuda:0", gpu_uuid=smoke.GPU,
                gpu_name="NVIDIA RTX PRO 4000 Blackwell", torch_cuda="12.8", capability=[12, 0], torch="2.9.1",
                rounds=8, elements=4 * 1024**2, all_values_exact=True, autograd_enabled=False,
                allocator_cap_bytes=512 * 1024**2, peak_allocated_bytes=48 * 1024**2,
                peak_reserved_bytes=64 * 1024**2, flags=deepcopy(smoke.FLAGS))


def test_capacity_is_not_idle_and_never_claims_training():
    assert smoke.capacity(sample(util=14)) == sample(util=14)
    assert not any(smoke.FLAGS.values())
    top_imports = [node for node in ast.parse(Path(smoke.__file__).read_text()).body
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {node.module if isinstance(node, ast.ImportFrom) else alias.name
             for node in top_imports for alias in node.names}
    assert not names.intersection({"torch", "warp", "mujoco", "mujoco_warp"})


@pytest.mark.parametrize("raw", ["", "one,two", "\n".join([f"{smoke.GPU}, {smoke.DRIVER}, 24467, 0, 20000, 0, 32"] * 2),
    f"GPU-wrong, {smoke.DRIVER}, 24467, 7422, 16740, 0, 32",
    f"{smoke.GPU}, wrong, 24467, 7422, 16740, 0, 32",
    f"{smoke.GPU}, {smoke.DRIVER}, 24467, N/A, 16740, 0, 32",
    f"{smoke.GPU}, {smoke.DRIVER}, 24467, -1, 16740, 0, 32",
    f"{smoke.GPU}, {smoke.DRIVER}, 24467, 7422, 16740, 101, 32",
    f"{smoke.GPU}, {smoke.DRIVER}, 24467, 7422, 16740, 0, 121",
    f"{smoke.GPU}, {smoke.DRIVER}, 24576, 7422, 16740, 0, 32"])
def test_malformed_or_changed_gpu_refused(raw):
    with pytest.raises(ValueError):
        smoke.parse_gpu(raw)


@pytest.mark.parametrize("view", ["windows", "wsl"])
@pytest.mark.parametrize("field,value", [("used_mib", 12289), ("free_mib", 10239),
    ("temperature_c", 65), ("utilization_percent", 86), ("uuid", "other"), ("driver", "other")])
def test_each_host_view_independently_guards_resources(view, field, value):
    current = sample()
    current[view][field] = value
    with pytest.raises(ValueError):
        smoke.capacity(current)


@pytest.mark.parametrize("view", ["windows", "wsl"])
def test_growth_is_aggregate_not_false_process_attribution(view):
    baseline, current = sample(), sample()
    current[view]["used_mib"] += 2048
    smoke.capacity(current, baseline)
    current[view]["used_mib"] += 1
    with pytest.raises(ValueError, match="aggregate memory growth"):
        smoke.capacity(current, baseline)


def test_exact_resource_boundaries_and_sequential_counters():
    smoke.capacity(sample(used=12288, free=10240, util=85, temp=64))
    current = sample()
    current["windows"]["used_mib"] += 50
    smoke.capacity(current, sample())  # Sequential readings need not be equal.


def test_exact_child_receipt_is_diagnostic_only():
    smoke.validate_child(receipt())


@pytest.mark.parametrize("field,value", [("protocol", "training"), ("device", "cpu"),
    ("gpu_uuid", "other"), ("gpu_name", "other"), ("torch_cuda", "13.2"),
    ("capability", [8, 9]), ("torch", "new"), ("rounds", 7), ("elements", 1),
    ("all_values_exact", False), ("autograd_enabled", True), ("allocator_cap_bytes", 1024**3),
    ("peak_allocated_bytes", 0), ("peak_reserved_bytes", 129 * 1024**2)])
def test_changed_or_over_budget_child_refused(field, value):
    result = receipt()
    result[field] = value
    with pytest.raises(ValueError):
        smoke.validate_child(result)


@pytest.mark.parametrize("flag", list(smoke.FLAGS))
def test_no_child_can_promote_diagnostic(flag):
    result = receipt()
    result["flags"][flag] = True
    with pytest.raises(ValueError):
        smoke.validate_child(result)


def test_json_is_exclusive_and_finite(tmp_path):
    target = tmp_path / "report.json"
    smoke.write_json(target, receipt())
    with pytest.raises(FileExistsError):
        smoke.write_json(target, receipt())
    with pytest.raises(ValueError):
        smoke.write_json(tmp_path / "nan.json", {"value": float("nan")})
    with pytest.raises(ValueError, match="bounded evidence"):
        smoke.write_json(tmp_path / "large.json", {"value": "x" * 300000})


def owner_fixture(monkeypatch, tmp_path, samples, *, returncode=None, stubborn=False):
    root = tmp_path / "repo"
    root.mkdir()
    target = tmp_path / "frozen-env"
    target.mkdir()
    (root / ".venv").symlink_to(target, target_is_directory=True)
    lock = tmp_path / "existing.lock"
    lock.touch()
    out = tmp_path / "fresh-evidence"
    monkeypatch.setattr(smoke, "ROOT", root)
    monkeypatch.setattr(smoke, "LOCK", lock)
    monkeypatch.setattr(smoke, "directory", lambda source: out)
    monkeypatch.setattr(smoke, "source_check", lambda source: {"commit": source})
    monkeypatch.setattr(smoke, "owned_service", lambda source: {"unit": "owned-test"})
    monkeypatch.setattr(smoke, "service_snapshot", lambda: {"preserved": True})
    monkeypatch.setattr(smoke, "lease_identity", lambda fd: {"existing": True})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Other collected suites may import Torch. Model the separately fresh owner
    # without mutating the process-wide import table or weakening the CLI guard.
    monkeypatch.setattr(smoke, "sys", SimpleNamespace(modules={}, executable=sys.executable))
    readings = iter(samples)
    monkeypatch.setattr(smoke, "READ_DEADLINE", None)

    def telemetry():
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(smoke, "telemetry", telemetry)
    actions = []

    class OwnedChild:
        def __init__(self, command, **kwargs):
            actions.append("spawn-owned-child")
            self.returncode = returncode
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            assert len(kwargs["pass_fds"]) == 1
            if returncode == 0:
                smoke.write_json(out / "child.json", receipt())

        def poll(self):
            return self.returncode

        def terminate(self):
            actions.append("terminate-owned-child")
            if not stubborn:
                self.returncode = -15

        def wait(self, timeout):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("owned-child", timeout)
            return self.returncode

        def kill(self):
            actions.append("kill-owned-child")
            self.returncode = -9

    monkeypatch.setattr(smoke.subprocess, "Popen", OwnedChild)
    return out, actions


@pytest.mark.parametrize("failure", [sample(free=10000), RuntimeError("telemetry lost")])
def test_monitor_failure_terminates_only_owned_child_and_retains_failure(monkeypatch, tmp_path, failure):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(), failure])
    with pytest.raises((ValueError, RuntimeError)):
        smoke.supervise("a" * 40)
    report = json.loads((out / "report.json").read_bytes())
    assert actions == ["spawn-owned-child", "terminate-owned-child"]
    assert report["decision"] == "failed" and report["child_exit"] == -15
    assert not any(report["flags"].values())
    assert set(report["files"]) == {"launch.json", "child.log"}


def test_stubborn_owned_child_killed_without_foreign_pid_actions(monkeypatch, tmp_path):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample(temp=65)], stubborn=True)
    with pytest.raises(ValueError):
        smoke.supervise("a" * 40)
    assert actions == ["spawn-owned-child", "terminate-owned-child", "kill-owned-child"]
    assert json.loads((out / "report.json").read_bytes())["child_exit"] == -9


def test_failed_admission_never_spawns_or_waits(monkeypatch, tmp_path):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(free=10000)])
    with pytest.raises(ValueError):
        smoke.supervise("a" * 40)
    assert actions == []
    report = json.loads((out / "report.json").read_bytes())
    assert report["decision"] == "failed" and report["files"] == {}


def test_completed_owner_retains_both_views_but_no_training_admission(monkeypatch, tmp_path):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample(used=7800), sample(used=7422)], returncode=0)
    smoke.supervise("a" * 40)
    report = json.loads((out / "report.json").read_bytes())
    assert actions == ["spawn-owned-child"]
    assert report["decision"] == "shared-cuda-tensor-smoke-complete-not-training"
    assert len(report["telemetry"]) == 3
    assert report["post_exit_telemetry"] == sample(used=7422)
    assert report["services_unchanged"] and not any(report["flags"].values())
    assert set(report["files"]) == {"launch.json", "child.log", "child.json"}


def test_deadline_terminates_owned_child(monkeypatch, tmp_path):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample()])
    ticks = iter([0, 0, 0, 61, 61])
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="deadline"):
        smoke.supervise("a" * 40)
    assert actions == ["spawn-owned-child", "terminate-owned-child"]
    assert json.loads((out / "report.json").read_bytes())["decision"] == "failed"


def test_post_exit_telemetry_failure_cannot_publish_success(monkeypatch, tmp_path):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample(), sample(free=10000)], returncode=0)
    with pytest.raises(ValueError):
        smoke.supervise("a" * 40)
    assert actions == ["spawn-owned-child"]
    assert json.loads((out / "report.json").read_bytes())["decision"] == "failed"


def test_read_timeout_is_clipped_to_deadline_and_expiry_never_queries(monkeypatch):
    monkeypatch.setattr(smoke, "READ_DEADLINE", 2.5)
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 2.)
    calls = []

    def run(args, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.read("read-only-tool") == "ok"
    assert calls == [.5]
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 2.5)
    with pytest.raises(ValueError, match="read deadline"):
        smoke.read("read-only-tool")
    assert calls == [.5]


def test_independent_watchdog_kills_owned_child_before_blocking_monitor_returns(monkeypatch, tmp_path):
    out, actions = owner_fixture(monkeypatch, tmp_path, [sample(), sample()])

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
    with pytest.raises(ValueError, match="deadline"):
        smoke.supervise("a" * 40)
    assert actions == ["spawn-owned-child", "kill-owned-child"]
    assert json.loads((out / "report.json").read_bytes())["child_exit"] == -9


def test_child_receipt_extra_keys_refused():
    result = receipt()
    result["optimizer"] = "none"
    with pytest.raises(ValueError, match="schema"):
        smoke.validate_child(result)


def test_literal_device_uuid_not_capability_or_string_format_guess():
    raw = list(uuid.UUID(smoke.GPU.removeprefix("GPU-")).bytes)
    assert smoke.physical_uuid(raw) == smoke.GPU
    for bad in (raw[:-1], raw + [0], [256] * 16, [True] * 16, bytes(raw)):
        with pytest.raises(ValueError, match="16-byte"):
            smoke.physical_uuid(bad)
