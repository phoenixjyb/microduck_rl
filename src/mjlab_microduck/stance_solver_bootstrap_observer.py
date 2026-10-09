"""One-shot entered-Python bootstrap integrity; never native qualification.

The profile authenticates entered Warp code against its literal whole source
tree and direct standard-library callees against installed source bytes.
Unentered paths, native C calls and other threads remain outside this evidence.
"""
from hashlib import sha1, sha256
from pathlib import Path
import sys
import sysconfig
import threading
import time
from types import BuiltinFunctionType, CodeType, ModuleType

from mjlab_microduck import stance_solver_gradient_probe as probe
from mjlab_microduck import stance_solver_replay_receiver as pins

PROTOCOL = "microduck-entered-bootstrap-python-oct9-v1"
BASE = "20c00062c7d81dd4185cdb61e25a3a2efc68faca"
OWN = frozenset({
    "src/mjlab_microduck/stance_solver_bootstrap_observer.py",
    "tests/test_stance_solver_bootstrap_observer.py",
    "docs/experiments/2026-10-09-bootstrap-observer.md",
})
TESTS = probe.TESTS + ("tests/test_stance_solver_bootstrap_observer.py",)
FLAGS = probe.FLAGS
COORDINATOR_PIN = (30079, "4f85d93674b21fcf5f9c41a053da8ba3a43168bccb5b5dda4fae04076917b584")
MAX_CALLS, MAX_ROWS, INIT_SECONDS = 20000, 256, 10
EXTERNAL = frozenset({"ctypes", "platform", "os", "posixpath", "genericpath",
                      "collections.abc", "warnings", "importlib._bootstrap",
                      "importlib._bootstrap_external"})
need, read_plain, canonical = probe.need, probe.read_plain, probe.canonical


def source_binding(root):
    """New three-path fence; never alters a predecessor's fence."""
    cmd = probe.retained.command
    need(root == root.resolve(strict=True) and Path.cwd().resolve() == root
         and Path(__file__).resolve().parents[2] == root
         and cmd("git", "branch", "--show-current").decode().strip() == probe.BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact bootstrap-observer branch")
    source = cmd("git", "rev-parse", "HEAD").decode().strip()
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN,
         "separate exact three-path bootstrap-observer fence")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        head, name = row.split(b"\t", 1)
        mode, kind, oid = head.decode().split(); name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed observer leaf")
        raw = read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
             "whole committed observer blob")
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(source=source, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(), leaves=leaves)


def warp_inventory(root):
    need(root == root.resolve(strict=True) and root.is_dir(), "canonical installed Warp root")
    paths = sorted(set(root.rglob("*.py")) | {p for p in (root / "native").rglob("*") if p.is_file()})
    need(len(paths) == 460, "complete frozen Warp source leaf count")
    raw = {str(p.relative_to(root)): read_plain(p) for p in paths}
    leaves = {name: dict(bytes=len(v), sha256=sha256(v).hexdigest()) for name, v in raw.items()}
    need(sum(v["bytes"] for v in leaves.values()) == 9292904
         and sha256(canonical(leaves)).hexdigest() == pins.WARP_SOURCE_TREE_SHA256,
         "whole literal frozen Warp source tree before compilation")
    return leaves, raw


def code_graph(raw, filename):
    """Compile, never execute; include nested functions and comprehensions."""
    need(type(raw) is bytes and 0 < len(raw) <= 4 * 1024**2, "bounded Python source graph")
    root = compile(raw, filename, "exec", dont_inherit=True, optimize=0)
    out, pending = [], [root]
    while pending:
        code = pending.pop()
        out.append(code)
        need(len(out) <= 10000, "bounded recursive Python code graph")
        pending.extend(c for c in code.co_consts if type(c) is CodeType)
    return tuple(out)


class BootstrapObserver(probe.dispatch._Sealed):
    """Trusted main-thread integrity observer, not a hostile-process sandbox."""
    __slots__ = ("binding", "method", "method_code", "root", "leaves", "sources", "graphs",
                 "modules", "external_sources", "rows", "calls", "used", "active", "completed",
                 "failed", "thread", "deadline", "hook", "setter", "getter", "thread_hooks", "_sealed")

    def __init__(self, binding):
        need(type(binding) is probe.BootstrapBinding and not binding.used and binding.runtime is None,
             "fresh held bootstrap binding")
        binding.check()
        need(sys.getprofile() is None and sys.gettrace() is None and threading.getprofile() is None
             and threading.gettrace() is None and threading.current_thread() is threading.main_thread(),
             "no foreign hooks and main-thread-only bootstrap observation")
        self.binding, self.method = binding, binding.initialize
        self.method_code = self.method.__func__.__code__
        method_path = Path(probe.__file__).resolve(strict=True)
        raw = read_plain(method_path)
        need((len(raw), sha256(raw).hexdigest()) == COORDINATOR_PIN, "whole literal predecessor coordinator source")
        need(any(c.co_qualname == "BootstrapBinding.initialize" and self.method_code == c
                 for c in code_graph(raw, str(method_path)))
             and self.method.__func__.__globals__ is probe.__dict__, "original held bootstrap coordinator body")
        self.root = Path(binding.wp.__file__).resolve(strict=True).parent
        self.leaves, self.sources = warp_inventory(self.root)
        self.graphs, self.modules, self.external_sources, self.rows = {}, {}, {}, {}
        self.calls, self.used, self.active, self.completed, self.failed = 0, False, False, False, False
        self.thread, self.deadline = threading.get_ident(), 0.0
        self.setter, self.getter = sys.setprofile, sys.getprofile
        self.thread_hooks = (threading.setprofile, threading.settrace, threading.getprofile, threading.gettrace)
        need(all(type(f) is BuiltinFunctionType and f.__module__ == "sys"
                 and f.__self__ is sys and f.__name__ == name
                 for name, f in (("setprofile", self.setter), ("getprofile", self.getter))),
             "original CPython profile API metadata")
        self.hook = self._profile
        self._sealed = True

    def _frame(self, frame, warp):
        name = frame.f_globals.get("__name__")
        module = sys.modules.get(name)
        if name == "collections.abc" and frame.f_code.co_filename == "<frozen _collections_abc>":
            module = sys.modules.get("_collections_abc")
        need(type(module) is ModuleType and module.__dict__ is frame.f_globals,
             "canonical entered Python module globals")
        path = Path(module.__file__).resolve(strict=True)
        filename = frame.f_code.co_filename
        if warp:
            need(path.is_relative_to(self.root) and path.suffix == ".py"
                 and filename == str(path), "entered Warp canonical source path")
            rel = str(path.relative_to(self.root))
            need(rel in self.sources, "entered Warp source belongs to literal whole tree")
            raw, anchor = self.sources[rel], self.leaves[rel]
        else:
            stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
            need(name in EXTERNAL and path.is_relative_to(stdlib) and path.suffix == ".py"
                 and "site-packages" not in path.parts
                 and (filename.startswith("<frozen ") or filename == str(path)),
                 "direct external callee is installed standard-library Python")
            raw = read_plain(path)
            anchor = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
            old = self.external_sources.setdefault(str(path), anchor)
            need(old == anchor, "unchanged observed installed stdlib source")
        key = (str(path), filename)
        if key not in self.graphs: self.graphs[key] = code_graph(raw, filename)
        need(any(frame.f_code == c for c in self.graphs[key]), "source-equivalent entered Python body")
        held = self.modules.setdefault(name, module)
        need(held is module, "held entered Python module identity")
        row = ("warp" if warp else "direct-stdlib", name, frame.f_code.co_qualname, frame.f_code.co_firstlineno, str(path))
        if row not in self.rows:
            need(len(self.rows) < MAX_ROWS, "bounded unique observed bootstrap frames")
            self.rows[row] = 0
        self.rows[row] += 1

    def _profile(self, frame, event, arg):
        try:
            need(self.active and threading.get_ident() == self.thread and time.monotonic() < self.deadline,
                 "active bounded same-thread bootstrap profile")
            if event == "c_call" and arg in (self.setter, sys.settrace):
                # Only this observer's finally block may remove its own hook.
                need(frame.f_code is type(self).initialize.__code__ and frame.f_globals is globals(),
                     "foreign profile/trace mutation during bootstrap")
            if event != "call": return
            need(all(frame.f_code is not f.__code__ for f in self.thread_hooks[:2]),
                 "foreign thread profile/trace mutation during bootstrap")
            object.__setattr__(self, "calls", self.calls + 1)
            need(self.calls <= MAX_CALLS, "bounded bootstrap Python calls")
            name = frame.f_globals.get("__name__", "")
            caller = frame.f_back.f_globals.get("__name__", "") if frame.f_back else ""
            if type(name) is str and (name == "warp" or name.startswith("warp.")):
                self._frame(frame, True)
            elif type(caller) is str and (caller == "warp" or caller.startswith("warp.")):
                self._frame(frame, False)
        except BaseException:
            object.__setattr__(self, "failed", True)
            raise

    def initialize(self):
        need(not self.used and not self.active and self.binding.runtime is None and not self.binding.used,
             "one-shot fresh observed initialization")
        need(self.getter() is None and sys.gettrace() is None and threading.getprofile() is None
             and threading.gettrace() is None and threading.get_ident() == self.thread
             and sys.setprofile is self.setter and sys.getprofile is self.getter
             and self.thread_hooks == (threading.setprofile, threading.settrace, threading.getprofile, threading.gettrace)
             and self.method.__func__.__code__ is self.method_code,
             "unchanged bootstrap profile admission")
        self.binding.check()
        need(warp_inventory(self.root)[0] == self.leaves, "unchanged whole Warp tree before init")
        object.__setattr__(self, "used", True)
        object.__setattr__(self, "active", True)
        object.__setattr__(self, "deadline", time.monotonic() + INIT_SECONDS)
        try:
            self.setter(self.hook)
            self.method()
            need(self.getter() is self.hook and not self.failed, "owned bootstrap profile retained")
        finally:
            current = self.getter()
            if current is self.hook: self.setter(None)
            object.__setattr__(self, "active", False)
            need(current is self.hook or (current is None and self.failed),
                 "preserve foreign replacement hook and refuse success")
        self.binding.check()
        need(self.binding.used and self.binding.runtime is not None
             and self.binding.record()["held_constructor_completed"] is True,
             "completed original held runtime constructor")
        need(warp_inventory(self.root)[0] == self.leaves, "unchanged whole Warp tree after init")
        for path, anchor in self.external_sources.items():
            raw = read_plain(Path(path))
            need(anchor == dict(bytes=len(raw), sha256=sha256(raw).hexdigest()), "unchanged observed stdlib source after init")
        need(self.getter() is None and sys.gettrace() is None
             and all(f() is None for f in self.thread_hooks[2:])
             and self.thread_hooks == (threading.setprofile, threading.settrace, threading.getprofile, threading.gettrace)
             and self.rows and not self.failed,
             "completed observer with no leaked hooks")
        object.__setattr__(self, "completed", True)

    def record(self):
        need(self.completed and not self.active and not self.failed, "successful observed initialization only")
        self.binding.check()
        return dict(protocol=PROTOCOL, scope="main-thread-entered-Warp-and-direct-stdlib-Python-during-init",
            warp_source_tree_sha256=pins.WARP_SOURCE_TREE_SHA256,
            frames=[dict(kind=k[0], module=k[1], qualified=k[2], line=k[3], path=k[4], calls=v)
                    for k, v in sorted(self.rows.items())],
            stdlib_sources=self.external_sources.copy(), python_call_events=self.calls,
            profile_removed=True, timing_changed_by_profile=True, entered_python_bodies_checked=True,
            whole_transitive_dependencies_authenticated=False, native_c_bodies_authenticated=False,
            stdlib_origin_authenticated=False, callable_defaults_authenticated=False,
            other_threads_observed=False, gpu_only_paths_observed=False, qualification=dict(FLAGS))
