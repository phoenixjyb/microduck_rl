"""Bounded Linux supervision of one newly spawned, reviewed child session.

No GPU admission, CUDA import, process discovery by name, or service mutation.
The caller must hold its lease and admission checks, supply a bounded log file,
and run inside a capped user unit with KillMode=control-group as the backstop
for session escapes. This is not a sandbox for hostile child code.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


@dataclass(frozen=True)
class Identity:
    pid: int
    parent: int
    group: int
    session: int
    birth: int
    state: str

    @property
    def key(self):
        return self.pid, self.birth


def parse_stat(raw):
    if type(raw) is not bytes or len(raw) > 4096:
        raise ValueError("bounded proc stat bytes")
    head, delimiter, tail = raw.rpartition(b") ")
    if not delimiter or b" (" not in head:
        raise ValueError("complete proc stat comm delimiter")
    pid = int(head.split(b" (", 1)[0])
    fields = tail.split()
    if len(fields) < 20:
        raise ValueError("complete proc stat identity")
    value = Identity(pid, int(fields[1]), int(fields[2]), int(fields[3]),
                     int(fields[19]), fields[0].decode("ascii"))
    if pid <= 0 or value.birth <= 0 or len(value.state) != 1:
        raise ValueError("positive proc birth identity")
    return value


class LinuxProc:
    """Signal only the process referenced by a freshly verified pidfd."""

    def __init__(self):
        if (not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")
                or not hasattr(os, "P_PIDFD") or not hasattr(os, "WNOWAIT")):
            raise RuntimeError("Linux pidfd support required before spawn")
        self.root_fd = None

    def bind(self, child):
        # Popen's child has not been waited/reaped, so its PID cannot yet be
        # reused. Keep this exact root handle throughout final inventory.
        self.root_fd = os.pidfd_open(child.pid, 0)
        return self.read(child.pid)

    def status(self, child):
        info = os.waitid(os.P_PIDFD, self.root_fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if info is None or not info.si_pid:
            return None
        if info.si_pid != child.pid:
            raise RuntimeError("held root pidfd exit identity")
        return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status

    def abort_root(self, child, grace):
        if self.root_fd is None:
            return {"remaining": None, "errors": ["root attach unavailable; cgroup backstop required"]}
        errors = []
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                signal.pidfd_send_signal(self.root_fd, signum)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=grace)
                return {"remaining": None, "errors": ["session not inspected; cgroup backstop required"]}
            except subprocess.TimeoutExpired:
                errors.append("root wait timed out")
        return {"remaining": None, "errors": errors}

    def signal_root(self, signum):
        if self.root_fd is None:
            raise RuntimeError("held original root handle unavailable")
        try:
            signal.pidfd_send_signal(self.root_fd, signum)
            return True
        except ProcessLookupError:
            return False

    def close(self):
        if self.root_fd is not None:
            os.close(self.root_fd)
            self.root_fd = None

    def read(self, pid):
        try:
            with (Path("/proc") / str(pid) / "stat").open("rb") as handle:
                return parse_stat(handle.read(4097))
        except (FileNotFoundError, ProcessLookupError):
            return None

    def snapshot(self):
        names = []
        with os.scandir("/proc") as entries:
            for entry in entries:
                if entry.name.isdecimal():
                    names.append(int(entry.name))
                    if len(names) > 8192:
                        raise RuntimeError("bounded global proc inventory exceeded")
        return [value for pid in names if (value := self.read(pid)) is not None]

    def send(self, identity, signum):
        try:
            descriptor = os.pidfd_open(identity.pid, 0)
        except ProcessLookupError:
            return False
        try:
            current = self.read(identity.pid)
            if (current is None or current.key != identity.key
                    or (current.parent, current.group, current.session) !=
                    (identity.parent, identity.group, identity.session)
                    or current.state == "Z"):
                return False
            try:
                signal.pidfd_send_signal(descriptor, signum)
                return True
            except ProcessLookupError:
                return False
        finally:
            os.close(descriptor)


def _seconds(value, label, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError("bounded " + label)
    return float(value)


class OwnedSession:
    """Birth-bound root and same-session descendants, never foreign processes."""

    def __init__(self, child, proc):
        self.child, self.proc = child, proc
        self.root = proc.bind(child) if hasattr(proc, "bind") else proc.read(child.pid)
        if (self.root is None or self.root.parent != os.getpid()
                or self.root.group != child.pid or self.root.session != child.pid):
            raise RuntimeError("new child must own its fresh process session")
        self.known = {self.root.key}

    def members(self):
        snapshot = self.proc.snapshot()
        anchors = [value for value in snapshot if value.key == self.root.key
                   and (value.parent, value.group, value.session) ==
                   (self.root.parent, self.root.group, self.root.session)]
        if len(anchors) != 1:
            # The root must remain unreaped (WNOWAIT) until cleanup/success.
            # Tick-resolution birth stamps alone are not generation handles.
            raise RuntimeError("unreaped original root anchor unavailable")
        births = dict(self.known)
        members = [value for value in snapshot
                   if value.session == self.root.session and value.birth >= self.root.birth
                   and (value.pid not in births or births[value.pid] == value.birth)]
        if len(members) > 64:
            raise RuntimeError("owned session task cap exceeded")
        self.known.update(value.key for value in members)
        return [value for value in members if value.state != "Z"]

    def stop(self, grace, interval):
        sent = []
        sent_keys = set()
        failures = []
        for signum, budget in ((signal.SIGTERM, grace), (signal.SIGKILL, grace)):
            end = time.monotonic() + budget
            while True:
                try:
                    members = self.members()
                    if not members:
                        self.child.poll()
                        return {"remaining": [], "signals": sent, "errors": failures}
                    for member in members:
                        key = member.key + (int(signum),)
                        if key not in sent_keys and self.proc.send(member, signum):
                            sent.append([member.pid, member.birth, int(signum)])
                            sent_keys.add(key)
                except Exception as error:
                    failures.append(type(error).__name__ + ": " + str(error))
                    # Restrict fallback to the birth-bound direct child. The
                    # enclosing capped user unit must retire any unseen tasks.
                    if hasattr(self.proc, "signal_root"):
                        self.proc.signal_root(signum)
                if time.monotonic() >= end:
                    break
                time.sleep(interval)
        try:
            remaining = [[value.pid, value.birth] for value in self.members()]
        except Exception as error:
            failures.append(type(error).__name__ + ": " + str(error))
            remaining = None
        self.child.poll()
        return {"remaining": remaining, "signals": sent, "errors": failures}


def supervise(command, *, cwd, env, log, probe, timeout, probe_timeout=5,
              grace=2, interval=0.05, pass_fds=(), proc=None):
    """Observe one reviewed child with joined-descendant behavior and a fresh probe.

    Probe work runs in a daemon thread so a stuck telemetry call cannot prevent
    child timeout/cleanup. A stuck probe or cleanup error forbids a receipt.
    No command shell or PIPE is used. Lease FDs are held by the caller and may
    be inherited explicitly. A reviewed child must not detach, daemonize, or
    leave unjoined tool descendants. Proc scans are non-atomic and cannot prove
    an arbitrary forking tree empty; retirement of the enclosing service's
    cgroup must be independently verified before native acceptance/lease release.
    """
    timeout = _seconds(timeout, "child timeout", 600)
    probe_timeout = _seconds(probe_timeout, "probe timeout", 30)
    grace = _seconds(grace, "cleanup grace", 10)
    interval = _seconds(interval, "poll interval", 1)
    if interval < 0.01:
        raise ValueError("poll interval must be at least 0.01 seconds")
    if (type(command) is not tuple or not command or
            any(type(arg) is not str or not arg or "\0" in arg for arg in command)
            or not Path(command[0]).is_absolute() or not callable(probe)):
        raise ValueError("reviewed literal argv and admission probe")
    if type(env) is not dict or not Path(cwd).is_absolute():
        raise ValueError("explicit child environment and absolute working directory")
    proc = LinuxProc() if proc is None else proc
    # Initial admission occurs before spawn; the caller must bound this check.
    probe()
    start = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True,
                             stdin=subprocess.DEVNULL, pass_fds=pass_fds)
    owned = None
    results = queue.Queue(maxsize=1)

    def check():
        try:
            probe()
            results.put((None, time.monotonic()))
        except BaseException as error:
            results.put((error, time.monotonic()))

    thread, probe_start = None, None
    reason = None
    last_probe = None
    exit_seen = None
    try:
        owned = OwnedSession(child, proc)
        while reason is None:
            members = owned.members()
            status = proc.status(child) if hasattr(proc, "status") else child.poll()
            now = time.monotonic()  # Inventory time counts against the budget.
            if status is not None and exit_seen is None:
                exit_seen = now
            if now - start >= timeout:
                reason = "child-deadline"
                break
            if thread is not None:
                try:
                    error, completed_at = results.get_nowait()
                except queue.Empty:
                    if now - probe_start >= probe_timeout:
                        reason = "admission-probe-timeout"
                        break
                else:
                    if completed_at - probe_start >= probe_timeout:
                        reason = "admission-probe-timeout"
                        break
                    if error is not None:
                        reason = "admission-probe-failed: " + type(error).__name__ + ": " + str(error)
                        break
                    last_probe = now
                    thread = None
                    if status is not None and probe_start >= exit_seen:
                        if status != 0:
                            reason = "child-exit-" + str(status)
                        elif members:
                            reason = "child-left-descendants"
                        else:
                            # Root is now exited but unreaped: cannot fork again
                            # and cannot be replaced by an unrelated SID owner.
                            if owned.members():
                                reason = "child-left-descendants"
                                break
                            if time.monotonic() - start >= timeout:
                                reason = "child-deadline"
                                break
                            if child.poll() != 0:
                                raise RuntimeError("final root reap disagrees with held exit")
                            return {"decision": "reviewed-owned-root-exited-no-native-qualification",
                                    "returncode": 0, "root": list(owned.root.key),
                                    "elapsed": time.monotonic() - start,
                                    "observed_session_members": [],
                                    "process_tree_retirement_proven": False,
                                    "native_qualified": False, "training_authorized": False}
            if thread is None and (last_probe is None or status is not None or now - last_probe >= 1):
                probe_start = now
                thread = threading.Thread(target=check, daemon=True)
                thread.start()
            time.sleep(interval)
    except BaseException as error:
        reason = "supervision-failed: " + type(error).__name__ + ": " + str(error)
    finally:
        # Success has already reaped the root after its final inventory. All
        # failure paths keep it unreaped through stop's descendant scans.
        if reason is None and owned is not None and hasattr(proc, "close"):
            proc.close()
    try:
        cleanup = (owned.stop(grace, interval) if owned is not None else
                   proc.abort_root(child, grace))
    finally:
        if hasattr(proc, "close"):
            proc.close()
    raise RuntimeError(str({"reason": reason, "cleanup": cleanup,
                            "returncode": child.poll(),
                            "root": list(owned.root.key) if owned is not None else None}))
