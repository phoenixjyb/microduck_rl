"""Synthetic identity checks and bounded CPU-only child supervision."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_solver_supervisor as supervisor


def identity(pid=10, birth=100, session=10, state="R", parent=None):
    return supervisor.Identity(pid, os.getpid() if parent is None else parent,
                               session, session, birth, state)


class Proc:
    def __init__(self, values):
        self.values = values
        self.sent = []

    def read(self, pid):
        return next((value for value in self.values if value.pid == pid), None)

    def snapshot(self):
        return list(self.values)

    def send(self, value, signum):
        self.sent.append((value, signum))
        self.values.remove(value)
        if value.pid == 10:
            self.values.append(identity(state="Z"))
        return True


def test_parse_comm_with_spaces_parentheses_and_birth():
    raw = b"10 (tool (compiler) ) S 3 10 10 " + b"0 " * 15 + b"100 0 0\n"
    value = supervisor.parse_stat(raw)
    assert value == supervisor.Identity(10, 3, 10, 10, 100, "S")


@pytest.mark.parametrize("raw", [b"x", b"10 (tool) S 1", b"x" * 4097,
                                 b"0 (tool) S 1 1 1 " + b"0 " * 15 + b"100"])
def test_reject_bad_or_unbounded_stat(raw):
    with pytest.raises(ValueError):
        supervisor.parse_stat(raw)


def test_owned_session_never_adopts_foreign_or_reused_root():
    proc = Proc([identity(), identity(11, 101), identity(20, 102, session=20)])
    child = SimpleNamespace(pid=10)
    owned = supervisor.OwnedSession(child, proc)
    assert {value.pid for value in owned.members()} == {10, 11}
    # The exited root remains unreaped through the final inventory.
    proc.values = [identity(11, 101), identity(12, 103), identity(state="Z")]
    assert {value.pid for value in owned.members()} == {11, 12}
    # Once every original anchor is gone, a reused SID is not ours.
    proc.values = [identity(10, 300), identity(13, 301)]
    with pytest.raises(RuntimeError, match="root anchor unavailable"):
        owned.members()
    assert proc.sent == []


def test_task_cap_and_fresh_session_requirement():
    proc = Proc([identity()])
    owned = supervisor.OwnedSession(SimpleNamespace(pid=10), proc)
    proc.values += [identity(pid, 101) for pid in range(11, 75)]
    with pytest.raises(RuntimeError, match="task cap"):
        owned.members()
    with pytest.raises(RuntimeError, match="fresh process session"):
        supervisor.OwnedSession(SimpleNamespace(pid=11), proc)


def test_cleanup_signals_only_owned_birth_bound_members():
    foreign = identity(20, 99, 20)
    proc = Proc([identity(), identity(11, 101), foreign])
    owned = supervisor.OwnedSession(SimpleNamespace(pid=10, poll=lambda: -15), proc)
    receipt = owned.stop(0.02, 0.001)
    assert receipt["remaining"] == []
    assert proc.values == [foreign, identity(state="Z")]
    assert {value.pid for value, _ in proc.sent} == {10, 11}
    assert all(signum == signal.SIGTERM for _, signum in proc.sent)


def test_pidfd_rechecks_birth_after_open_without_signalling_reuse(monkeypatch):
    opened, sent, closed = [], [], []
    monkeypatch.setattr(os, "pidfd_open", lambda pid, flags: opened.append(pid) or 9, raising=False)
    monkeypatch.setattr(os, "P_PIDFD", 3, raising=False)
    monkeypatch.setattr(os, "WNOWAIT", 0x1000000, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *args: sent.append(args), raising=False)
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd))
    proc = supervisor.LinuxProc()
    monkeypatch.setattr(proc, "read", lambda pid: identity(birth=200))
    assert proc.send(identity(), signal.SIGTERM) is False
    assert opened == [10] and closed == [9] and sent == []


def test_probe_refusal_prevents_spawn(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    with (tmp_path / "log").open("wb") as log:
        with pytest.raises(ValueError, match="busy"):
            supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                                 env=dict(os.environ), log=log, proc=Proc([]), timeout=1,
                                 probe=lambda: (_ for _ in ()).throw(ValueError("busy")))
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 601])
def test_budget_invalid_before_spawn(timeout, tmp_path):
    with pytest.raises(ValueError, match="bounded"):
        supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                             env={}, log=None, probe=lambda: None, timeout=timeout)


def test_native_cpu_supervision_or_non_linux_refusal(tmp_path):
    # On WSL this exercises real pidfds, process birth checks, deadlines and an
    # inherited-session grandchild. On macOS require refusal before any spawn;
    # that platform result is explicitly not native pidfd evidence.
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        with pytest.raises(RuntimeError, match="pidfd support"):
            supervisor.LinuxProc()
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    with (tmp_path / "log").open("wb") as log:
        receipt = supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                                      env=env, log=log, probe=lambda: None,
                                      timeout=2, probe_timeout=0.2, grace=0.1)
        assert receipt["returncode"] == 0 and receipt["observed_session_members"] == []
        assert receipt["process_tree_retirement_proven"] is False
        assert receipt["native_qualified"] is False
        code = ("import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                "print(p.pid,flush=True); time.sleep(30)")
        with pytest.raises(RuntimeError, match="child-deadline"):
            supervisor.supervise((sys.executable, "-c", code), cwd=str(tmp_path),
                                 env=env, log=log, probe=lambda: None,
                                 timeout=0.3, probe_timeout=0.1, grace=0.1, interval=0.01)
        with pytest.raises(RuntimeError, match="child-exit-4"):
            supervisor.supervise((sys.executable, "-c", "raise SystemExit(4)"),
                                 cwd=str(tmp_path), env=env, log=log,
                                 probe=lambda: None, timeout=2, grace=0.1)
    pid = int((tmp_path / "log").read_text().strip())
    current = supervisor.LinuxProc().read(pid)
    assert current is None or current.state == "Z"


def test_native_cpu_child_exit_with_descendant_is_not_success(tmp_path):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        with pytest.raises(RuntimeError, match="pidfd support"):
            supervisor.LinuxProc()
        return
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "print(p.pid,flush=True);time.sleep(0.15)")
    with (tmp_path / "log").open("wb") as log:
        with pytest.raises(RuntimeError, match="child-left-descendants"):
            supervisor.supervise((sys.executable, "-c", code), cwd=str(tmp_path),
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), log=log,
                                 probe=lambda: None, timeout=2, grace=0.1, interval=0.01)
    pid = int((tmp_path / "log").read_text().strip())
    current = supervisor.LinuxProc().read(pid)
    assert current is None or current.state == "Z"


def test_stuck_post_spawn_probe_cannot_block_deadline(monkeypatch, tmp_path):
    root = identity()
    proc = Proc([root])
    child = SimpleNamespace(pid=10, poll=lambda: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    checks = []

    def probe():
        checks.append(1)
        if len(checks) > 1:
            time.sleep(0.3)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="admission-probe-timeout"):
        supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                             env={}, log=None, probe=probe, timeout=0.2,
                             probe_timeout=0.02, grace=0.02, interval=0.01, proc=proc)
    assert time.monotonic() - start < 0.2
    assert proc.sent == [(root, signal.SIGTERM)]


def test_final_unreaped_scan_catches_pre_exit_inventory_race(monkeypatch, tmp_path):
    class LateFork(Proc):
        def __init__(self):
            super().__init__([identity(state="Z")])
            self.scans = 0

        def snapshot(self):
            self.scans += 1
            if self.scans == 3:
                self.values.append(identity(11, 101))
            return super().snapshot()

        def status(self, child):
            return 0  # WNOWAIT: zombie root stays present throughout scans.

    proc = LateFork()
    child = SimpleNamespace(pid=10, poll=lambda: 0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="child-left-descendants"):
        supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                             env={}, log=None, probe=lambda: None, timeout=1,
                             probe_timeout=0.1, grace=0.02, interval=0.01, proc=proc)
    assert any(value.pid == 11 for value, _ in proc.sent)


def test_inventory_elapsed_time_cannot_mint_late_success(monkeypatch, tmp_path):
    class SlowScan(Proc):
        def snapshot(self):
            time.sleep(0.03)
            return super().snapshot()

    proc = SlowScan([identity(state="Z")])
    child = SimpleNamespace(pid=10, poll=lambda: 0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="child-deadline"):
        supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                             env={}, log=None, probe=lambda: None, timeout=0.02,
                             probe_timeout=0.1, grace=0.02, interval=0.01, proc=proc)


def test_root_attach_failure_uses_held_root_cleanup(monkeypatch, tmp_path):
    class AttachFailure(Proc):
        def bind(self, child):
            raise PermissionError("stat unavailable")

        def abort_root(self, child, grace):
            self.aborted = child.pid
            return {"remaining": None, "errors": ["cgroup backstop required"]}

    proc = AttachFailure([])
    child = SimpleNamespace(pid=10, poll=lambda: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="stat unavailable"):
        supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                             env={}, log=None, probe=lambda: None, timeout=1,
                             grace=0.02, proc=proc)
    assert proc.aborted == 10 and proc.sent == []


def test_completed_but_over_budget_probe_is_still_failure(monkeypatch, tmp_path):
    proc = Proc([identity()])
    child = SimpleNamespace(pid=10, poll=lambda: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    calls = []
    def probe():
        calls.append(1)
        if len(calls) > 1:
            time.sleep(0.025)
    with pytest.raises(RuntimeError, match="admission-probe-timeout"):
        supervisor.supervise((sys.executable, "-c", "pass"), cwd=str(tmp_path),
                             env={}, log=None, probe=probe, timeout=0.3,
                             probe_timeout=0.02, grace=0.02, interval=0.05, proc=proc)
