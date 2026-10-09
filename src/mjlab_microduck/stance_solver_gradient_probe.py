"""Capped shared-capacity stopped gradient diagnostic; never training.

The CPU owner authenticates before spawning an inherited-lease CUDA child.
Only external closeout may authenticate complete output and unit retirement.
"""
import argparse
import ast
from collections import Counter
import fcntl
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import FunctionType
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_solver_replay_probe as retained
from mjlab_microduck import stance_solver_supervisor as supervisor
from mjlab_microduck import stance_solver_gradient_prefix as prefix
from mjlab_microduck import stance_solver_gradient_dispatch as dispatch
from mjlab_microduck import stance_solver_gradient_runtime as api
from mjlab_microduck import stance_solver_gradient_executable as executable
from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-stopped-gradient-probe-oct9-v1"
RESTORE_PROTOCOL = "microduck-gradient-parent-restoration-oct9-v1"
BOOTSTRAP_PROTOCOL = "microduck-gradient-runtime-bootstrap-oct9-v1"
BASE = "df973002b20fa99771aa353b27d8111e5a97166a"
OWN = frozenset({
    "src/mjlab_microduck/stance_solver_gradient_probe.py",
    "src/mjlab_microduck/stance_solver_gradient_receiver.py",
    "tests/test_stance_solver_gradient_probe.py",
    "tests/test_stance_solver_gradient_receiver.py",
    "docs/experiments/2026-10-09-gradient-native-preparation.md",
})
TESTS = api.TESTS + ("tests/test_stance_solver_gradient_probe.py", "tests/test_stance_solver_gradient_receiver.py")
FLAGS = prefix.FLAGS
CACHE_NAMES = ("warp-cache", "cuda-cache", "xdg-cache", "torch-cache")
DCL_KEYS = frozenset({"protocol", "source", "source_binding", "native_root", "owner_pid", "runtime", "service",
                      "cpu_tests", "lease", "services", "baseline", "bounds", "deadline_unix", "replay_input", "parent",
                      "exclusive_gpu_claimed", "flags"})
ROOT, BRANCH, BOUNDS, UNIT_CAPS = retained.ROOT, retained.BRANCH, retained.BOUNDS, retained.UNIT_CAPS
shared, frozen, need = retained.shared, retained.frozen, retained.need
read_plain, read_json, canonical = retained.read_plain, retained.read_json, retained.canonical


def unit(source):
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "literal source SHA")
    return "microduck-stopped-gradient-" + source[:12] + ".service"


def output(source):
    unit(source)
    return ROOT / "artifacts/evaluations" / ("stopped-gradient-" + source[:12])


def solver_source_path():
    return (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)


def source_binding(source):
    need(Path.cwd().resolve() == ROOT and Path(__file__).resolve().parents[2] == ROOT,
         "literal native source root")
    return committed_source(ROOT, source)


def committed_source(root, source):
    """Portable CPU source inventory; never weakens the native root fence."""
    unit(source)
    cmd = retained.command
    need(root == root.resolve(strict=True) and Path.cwd().resolve() == root
         and Path(__file__).resolve().parents[2] == root
         and cmd("git", "rev-parse", "HEAD").decode().strip() == source
         and cmd("git", "branch", "--show-current").decode().strip() == BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact native source branch")
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN,
         "new exact five-path source fence; historical fences unchanged")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed file")
        raw = read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid, "whole committed blob " + name)
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(source=source, branch=BRANCH, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(), leaves=leaves)


def unit_record(source, pid):
    keys = (*UNIT_CAPS, "Id", "MainPID", "ActiveState", "ControlGroup", "InvocationID")
    raw = retained.command("systemctl", "--user", "show", unit(source), *[x for k in keys for x in ("-p", k)]).decode()
    row = dict(line.split("=", 1) for line in raw.splitlines())
    need({k: row.get(k) for k in UNIT_CAPS} == UNIT_CAPS and row["Id"] == unit(source)
         and row["MainPID"] == str(pid) and row["ActiveState"] == "active"
         and row["InvocationID"] == os.environ.get("INVOCATION_ID")
         and re.fullmatch(r"[0-9a-f]{32}", row["InvocationID"])
         and row["ControlGroup"] == retained.CGROUP_PARENT + unit(source), "same active capped exec owner unit")
    return dict(name=unit(source), **row)


def tests_record(path, source):
    raw = read_plain(path)
    value = read_json(path)
    need(canonical(value) == raw, "canonical paired CPU evidence")
    need(set(value) == {"source", "test_files", "mac", "native", "flags"}
         and value["source"] == source and value["test_files"] == list(TESTS)
         and value["flags"] == FLAGS, "source-bound paired focused CPU evidence")
    cases = []
    for platform in ("mac", "native"):
        a = value[platform]
        need(type(a) is dict and set(a) == {"path", "bytes", "sha256", "tests"}
             and type(a["tests"]) is int and a["tests"] > 1199, "new focused collection anchor")
        p = Path(a["path"])
        need(p.is_absolute() and p.is_relative_to(ROOT / "artifacts/tools") and p.suffix == ".xml", "retained CPU XML path")
        xml = read_plain(p)
        need(len(xml) == a["bytes"] and sha256(xml).hexdigest() == a["sha256"], "whole CPU XML hash")
        tree = ET.fromstring(xml)
        rows = [tree] if tree.tag == "testsuite" else list(tree)
        need(rows and all(r.tag == "testsuite" and all(int(r.get(k, "-1")) == 0 for k in ("errors", "failures", "skipped")) for r in rows)
             and sum(int(r.get("tests", "-1")) for r in rows) == a["tests"], "all focused CPU tests passed without skips")
        actual = tree.findall(".//testcase")
        need(len(actual) == a["tests"] and not any(tree.findall(".//" + k) for k in ("failure", "error", "skipped")),
             "whole CPU testcase results")
        cases.append(Counter((c.get("classname"), c.get("name")) for c in actual))
    need(value["mac"]["tests"] == value["native"]["tests"], "matched focused collection")
    need(cases[0] == cases[1], "matched exact CPU case multiset")
    return dict(path=str(path), bytes=len(raw), sha256=sha256(raw).hexdigest(), evidence=value)


def restoration_record(raw):
    need(type(raw) is bytes and len(raw) == prefix.PACKET_BYTES, "complete restoration bank")
    return dict(protocol=RESTORE_PROTOCOL, initial_bank=dict(bytes=len(raw), sha256=sha256(raw).hexdigest()),
                fields=list(prefix.ORDER), preflight_complete=True, same_stream_copy=True,
                gpu_numpy_readback=False, timing_changed_by_staging=True, qualification=dict(FLAGS))


def restore_bank(wp, arrays, stream, guard, raw, anchor):
    """Authenticate and preflight all 26 fields before the first H2D copy.

    No GPU numpy view; same-stream copies/sync deliberately affect timing.
    Caller boundary capture remains the independent whole-bank check.
    """
    import numpy as np
    need(type(raw) is bytes and len(raw) == prefix.PACKET_BYTES
         and type(anchor) is dict and set(anchor) == {"bytes", "sha256"}
         and type(anchor["bytes"]) is int and anchor == dict(bytes=len(raw), sha256=sha256(raw).hexdigest()),
         "whole authenticated parent-restoration bank before decode")
    values = prefix.unpack_bank(raw)
    specs = {n: (logical, host, dtype) for n, logical, host, dtype in dispatch.SPECS}
    need(type(arrays) is dict and set(arrays) == set(prefix.ORDER)
         and len(values) == 26, "complete 26-field staging inputs")
    held_array, held_copy, held_sync = wp.array, wp.copy, wp.synchronize_stream
    held_numpy = held_array.numpy
    entries = (held_numpy, held_copy, held_sync)
    codes = tuple(f.__code__ for f in entries)
    device, stream_handle = stream.device, stream.cuda_stream
    held_guard, guard_code = guard, getattr(guard, "__code__", None)
    need(guard_code is not None and type(stream_handle) is int and stream_handle > 0,
         "held staging guard and explicit stream")
    source_arrays = dict(arrays)
    layouts = {}
    guard()
    for name, a in arrays.items():
        logical, host, dtype = specs[name]
        need(type(a) is held_array and a.dtype is getattr(wp, dtype)
             and tuple(a.shape) == logical and a.device is device and device.is_cuda is True
             and a.requires_grad is False and a.is_contiguous is True
             and type(a.ptr) is int and a.ptr > 0, "literal full staging destination layout")
        layouts[name] = (a.ptr, len(values[name]))
    spans = sorted((p, p + size) for p, size in layouts.values())
    need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "disjoint complete destination bank")
    staged = {}
    for name, field_bytes in values.items():
        logical, host, dtype = specs[name]
        wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        host_array = np.frombuffer(field_bytes, dtype=wire).reshape(host).copy()
        src = held_array(host_array, dtype=getattr(wp, dtype), device="cpu", pinned=True, requires_grad=False)
        staged[name] = (src, src.ptr)
    def check():
        guard()
        need(guard is held_guard and guard.__code__ is guard_code
             and wp.array is held_array and held_array.numpy is held_numpy
             and wp.copy is held_copy and wp.synchronize_stream is held_sync
             and all(f.__code__ is c for f, c in zip(entries, codes))
             and stream.device is device and stream.cuda_stream == stream_handle
             and set(arrays) == set(source_arrays), "held typed staging entries and stream")
        for name, a in source_arrays.items():
            logical, host, dtype = specs[name]
            need(arrays[name] is a and type(a) is held_array and a.dtype is getattr(wp, dtype)
                 and tuple(a.shape) == logical and a.device is device
                 and device.is_cuda is True and a.requires_grad is False
                 and a.is_contiguous is True and (a.ptr, len(values[name])) == layouts[name],
                 "unchanged full destination bank")
        host_spans = []
        for name, (src, ptr) in staged.items():
            logical, host, dtype = specs[name]
            wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
            need(type(src) is held_array and src.dtype is getattr(wp, dtype)
                 and tuple(src.shape) == logical and src.device.is_cpu is True
                 and src.device.is_cuda is False and src.pinned is True
                 and src.requires_grad is False and src.is_contiguous is True
                 and src.ptr == ptr and type(ptr) is int and ptr > 0,
                 "literal typed pinned CPU staging")
            view = held_numpy(src)
            need(tuple(view.shape) == host and view.dtype.str == wire
                 and view.flags.c_contiguous and view.nbytes == len(values[name])
                 and view.tobytes(order="C") == values[name], "whole typed parent staging bytes")
            host_spans.append((ptr, ptr + len(values[name])))
        host_spans.sort()
        need(all(a[1] <= b[0] for a, b in zip(host_spans, host_spans[1:]))
             and not any(a[0] < b[1] and b[0] < a[1] for a in spans for b in host_spans),
             "all typed staging buffers disjoint from each other and destination bank")
    check()  # every field, including late gradient/norm, before first copy
    for name, (src, _) in staged.items():
        check()
        held_copy(arrays[name], src, stream=stream)
        check()
        held_sync(stream)
        check()
    return restoration_record(raw)


def parent_inputs(root, solver_path):
    """Reproduce the literal predecessor receiver before re-reading its banks."""
    prefix.audit_retained(root, solver_path)
    directory = root / "artifacts/evaluations/efc-transition-7717b4cbef3a"
    inv_path = root / "artifacts/tools/efc-transition-closeout-7717b4cbef3a/inventory.json"
    inv_raw = read_plain(inv_path)
    need(sha256(inv_raw).hexdigest() == prefix.PRIOR_INVENTORY_SHA256, "unchanged literal parent inventory")
    inventory = prefix.receiver.old._json(inv_raw, "parent inventory")
    banks = {arm: read_plain(directory / (arm + "-gauss.after.bin")) for arm in ("reference", "control")}
    need(all(inventory[a + "-gauss.after.bin"] == dict(bytes=len(v), sha256=sha256(v).hexdigest())
             for a, v in banks.items()), "whole re-read authenticated parent Gauss banks")
    packet = prefix.efc.historical(root)
    initial = prefix.initial_banks(banks, packet)
    record = dict(prior_source=prefix.PRIOR_SOURCE, inventory_sha256=prefix.PRIOR_INVENTORY_SHA256,
                  retirement_sha256=prefix.PRIOR_RETIREMENT_SHA256, receiver_sha256=prefix.PRIOR_RESULT_SHA256,
                  initial_banks={a: dict(bytes=len(v), sha256=sha256(v).hexdigest()) for a, v in initial.items()},
                  gauss_banks={a: dict(bytes=len(v), sha256=sha256(v).hexdigest()) for a, v in banks.items()})
    return record, banks, initial, packet


def cache_record(root):
    leaves = {}
    for name in CACHE_NAMES:
        directory = root / name
        need(not directory.is_symlink(), "plain owned cache directory")
        if not directory.exists():
            continue
        need(directory.is_dir(), "owned cache directory")
        for path in sorted(directory.rglob("*")):
            need(not path.is_symlink() and (path.is_dir() or path.is_file()), "plain cache leaf")
            if path.is_file():
                raw = read_plain(path)
                leaves[str(path.relative_to(root))] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    need(len(leaves) <= 128 and sum(r["bytes"] for r in leaves.values()) <= 64 * 1024**2, "bounded owned cache inventory")
    return dict(roots=list(CACHE_NAMES), absent_before_init=True, leaves=leaves)


class BootstrapBinding(dispatch._Sealed):
    """Source-equivalent init and all Runtime methods, then one held constructor.

    This is selected transitive Python integrity, not whole-interpreter/native
    origin proof. Native library bytes and loaded paths are checked separately.
    """
    __slots__ = ("wp", "context", "cls", "rows", "runtime", "used", "_sealed")

    def __init__(self, wp, context, api_guard):
        need(type(api_guard) is api.WarpApiGuard and context.runtime is None, "fresh uninitialized runtime bootstrap")
        api_guard.check()
        raw = read_plain(api_guard.paths["warp._src.context"])
        need((len(raw), sha256(raw).hexdigest()) == api.PINS["warp._src.context"], "whole pinned bootstrap source")
        tree = ast.parse(raw)
        code = compile(raw, str(api_guard.paths["warp._src.context"]), "exec", dont_inherit=True, optimize=0)
        cls = context.Runtime
        need(type(cls) is type and cls.__module__ == context.__name__ and cls.__qualname__ == "Runtime"
             and cls.__bases__ == (object,) and wp.init is context.init, "plain frozen Runtime class and init alias")
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
        names = tuple(n.name for n in node.body if isinstance(n, ast.FunctionDef))
        need(len(names) == 17 and len(set(names)) == 17, "all frozen Runtime methods")
        self.wp, self.context, self.cls = wp, context, cls
        self.rows = []
        for owner, name, qualified in [(context, "init", "init"), *((cls, n, "Runtime." + n) for n in names)]:
            expected, defaults = api._definition(code, tree, qualified)
            fn = owner.__dict__.get(name)
            need(type(fn) is FunctionType and fn.__code__ == expected and fn.__globals__ is context.__dict__
                 and fn.__closure__ is None and fn.__kwdefaults__ is None and not fn.__dict__
                 and api._value(fn.__defaults__, defaults), "original frozen runtime bootstrap body/defaults")
            self.rows.append((owner, name, fn, fn.__code__, fn.__defaults__, defaults))
        self.rows = tuple(self.rows)
        self.runtime, self.used = None, False
        self._sealed = True
        self.check()

    def check(self):
        need(self.context.Runtime is self.cls and self.wp.init is self.context.init
             and self.context.runtime is self.runtime and self.cls.__module__ == "warp._src.context"
             and self.cls.__qualname__ == "Runtime" and self.cls.__bases__ == (object,), "held runtime bootstrap objects")
        for owner, name, fn, code, defaults, expected in self.rows:
            need(owner.__dict__.get(name) is fn and fn.__code__ is code and fn.__globals__ is self.context.__dict__
                 and fn.__closure__ is None and fn.__kwdefaults__ is None and not fn.__dict__
                 and fn.__defaults__ is defaults and api._value(defaults, expected), "held runtime bootstrap entries")

    def initialize(self):
        need(not self.used and self.runtime is None, "one-shot fresh runtime initialization")
        object.__setattr__(self, "used", True)
        self.check()
        self.rows[0][2]()
        need(type(self.context.runtime) is self.cls, "actual frozen Runtime constructor result")
        object.__setattr__(self, "runtime", self.context.runtime)
        self.check()

    def record(self):
        need(self.used and self.runtime is not None, "completed runtime bootstrap")
        self.check()
        return dict(protocol=BOOTSTRAP_PROTOCOL, methods=["init", *["Runtime." + r[1] for r in self.rows[1:]]],
                    fresh_runtime_before_init=True, selected_bootstrap_bodies_checked=True,
                    held_constructor_completed=True, transitive_dependencies_authenticated=False, flags=dict(FLAGS))


def child(args):
    root = output(args.source)
    need(args.child and os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.getppid() == args.owner_pid,
         "fresh inherited-lease CUDA child")
    raw = read_plain(root / "declaration.json")
    need(sha256(raw).hexdigest() == args.declaration_sha256, "whole declaration hash before decode")
    dcl = read_json(root / "declaration.json")
    need(set(dcl) == DCL_KEYS, "closed gradient child declaration")
    parent, _, initial, _ = parent_inputs(ROOT, solver_source_path())
    need(dcl["protocol"] == PROTOCOL and dcl["source"] == args.source and dcl["owner_pid"] == args.owner_pid
         and dcl["native_root"] == str(root) and dcl["source_binding"] == source_binding(args.source)
         and dcl["runtime"] == retained.runtime_binding() and dcl["service"] == unit_record(args.source, args.owner_pid)
         and dcl["parent"] == parent and dcl["bounds"] == BOUNDS
         and dcl["flags"] == FLAGS and dcl["deadline_unix"] == args.deadline
         and shared.lease_identity(args.lease_fd) == dcl["lease"], "complete child admission before CUDA import")
    retained.check_deadline(args.deadline, BOUNDS["child_seconds"] + BOUNDS["closeout_seconds"] + BOUNDS["margin_seconds"])
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    need(shared.service_snapshot() == dcl["services"], "unchanged protected services before child initialization")
    shared.capacity(shared.telemetry(), dcl["baseline"])
    for key, name in (("WARP_CACHE_PATH", "warp-cache"), ("CUDA_CACHE_PATH", "cuda-cache"),
                      ("XDG_CACHE_HOME", "xdg-cache"), ("TORCH_EXTENSIONS_DIR", "torch-cache")):
        p = root / name
        need(os.environ.get(key) == str(p) and not p.exists() and not p.is_symlink(), "absent private cache before CUDA init")
    import numpy as np
    import warp as wp
    from warp._src import context as wc, types as wt, build as wb
    before_api = api.WarpApiGuard(wp=wp, context=wc, types=wt, build=wb)
    bootstrap = BootstrapBinding(wp, wc, before_api)
    frozen.configure_compiler(wp)
    bootstrap.initialize()
    runtime_api = api.WarpApiGuard(wp=wp, context=wc, types=wt, build=wb)
    bootstrap.check()
    device = wp.get_device("cuda:0")
    need(device.uuid == shared.GPU and device.arch == 120 and device.is_cuda is True
         and device.context > 0 and not device.is_capturing, "actual mapped sm120 GPU")
    for attr, name in (("core", "warp.so"), ("llvm", "warp-clang.so")):
        need(str(getattr(wc.runtime, attr)._name) == dcl["runtime"]["libraries"][name]["path"], "actual pinned loaded runtime library")
    import mujoco
    import mujoco_warp as mw
    from mujoco_warp._src import solver, types
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_plant_evidence import describe
    from mjlab_microduck.stance_com_coupled_receiver import DESCRIPTOR_SHA256
    need(int(types.ConstraintType.CONTACT_ELLIPTIC) == 7, "frozen domain enum")
    stream = wp.Stream(device)
    with wp.ScopedDevice(device), wp.ScopedStream(stream):
        plant = build_entity().compile()
        need(sha256(canonical(describe(plant))).hexdigest() == DESCRIPTOR_SHA256, "same compiled plant descriptor")
        model = mw.put_model(plant)
        data = mw.put_data(plant, mujoco.MjData(plant), nworld=64, nconmax=128, njmax=512)
        ctx = solver.create_solver_context(model, data)
        retained.recipe_signature(model, data)
        runtime_api.check(); bootstrap.check()
        prep = executable.GradientModuleExecutable(solver=solver, wp=wp, context=wc, device=device,
                                                   directory=root / "compiled-gradient")
        prep.compile()
        runtime_api.check(); bootstrap.check()
        sass, tool = {}, dcl["runtime"]["tool"]
        for n, symbol in prep.symbols.items():
            need(retained.tool_binding() == tool, "same held disassembler before use")
            with (root / (n + ".sass")).open("xb") as out, (root / (n + ".stderr")).open("xb") as err:
                subprocess.run((tool["path"], "--dump-sass", "--function", symbol, str(prep.binary_path)),
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=20, check=True)
                out.flush(); os.fsync(out.fileno()); err.flush(); os.fsync(err.fileno())
            need((root / (n + ".stderr")).stat().st_size == 0 and retained.tool_binding() == tool,
                 "joined unchanged successful offline disassembler")
            sass[n] = read_plain(root / (n + ".sass"))
        bindings = prep.load(sass)
        runtime_api.check(); bootstrap.check()
        arrays = dispatch._arrays(model, data, ctx)
        held_array, held_copy, held_sync = wp.array, wp.copy, wp.synchronize_stream
        codes = tuple(f.__code__ for f in (held_copy, held_sync))
        def guard():
            runtime_api.check(); bootstrap.check(); prep.check()
            need(wp.array is held_array and wp.copy is held_copy and wp.synchronize_stream is held_sync
                 and all(f.__code__ is c for f, c in zip((held_copy, held_sync), codes))
                 and wp.get_stream(device) is stream and wc.runtime.tape is None and device.uuid == shared.GPU,
                 "same scratch copy/sync/runtime/stream")
            for b in bindings.values(): b.assert_unchanged()
        arms = {}
        for arm in ("reference", "control"):
            restoration = restore_bank(wp, arrays, stream, guard, initial[arm], parent["initial_banks"][arm])
            guard()
            observer = dispatch.GradientDispatchObserver(solver=solver, wp=wp, runtime=wc.runtime,
                bindings=bindings, model=model, data=data, context=ctx, stream=stream)
            try:
                observer.run()
            finally:
                # Preserve complete failure boundaries too; no successful receipt
                # is written if observer.run/record fails.
                for phase in prefix.PHASES:
                    if phase in observer.capture.packets:
                        frozen.write(root / (arm + "-" + phase + ".bin"), observer.capture.raw(phase))
            guard()  # Own observer has restored the launch alias; never mask its hook.
            arms[arm] = dict(observer=observer.record(), restoration=restoration)
        receipt = dict(protocol=PROTOCOL, source=args.source, declaration_sha256=args.declaration_sha256,
            owner_pid=args.owner_pid, child_pid=os.getpid(), native_root=str(root),
            device=dict(alias=str(device), arch=device.arch, context=device.context, object_id=id(device), gpu_uuid=device.uuid),
            module=prep.record(), arms=arms, api=runtime_api.record(), bootstrap=bootstrap.record(),
            caches=cache_record(root), flags=dict(FLAGS))
    need(retained.runtime_binding() == dcl["runtime"] and source_binding(args.source) == dcl["source_binding"]
         and shared.lease_identity(args.lease_fd) == dcl["lease"], "whole post-run runtime/source/lease pins")
    frozen.write(root / "receipt.json", receipt)


def owner(args):
    need(not args.child and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only capped owner")
    need(type(args.cpu_evidence) is str and bool(args.cpu_evidence), "explicit paired CPU evidence path")
    retained.check_deadline(args.deadline, BOUNDS["service_seconds"] + BOUNDS["closeout_seconds"] + BOUNDS["margin_seconds"])
    source, runtime = source_binding(args.source), retained.runtime_binding()
    service, tests = unit_record(args.source, os.getpid()), tests_record(Path(args.cpu_evidence), args.source)
    parent, _, _, _ = parent_inputs(ROOT, solver_source_path())
    root = output(args.source)
    need(root.parent.resolve(strict=True) == root.parent and not root.exists(), "unique absent diagnostic output")
    fd = os.open(shared.LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        lease = shared.lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        services = shared.service_snapshot()
        baseline = shared.capacity(shared.telemetry())
        root.mkdir(mode=0o700)
        for platform in ("mac", "native"):
            anchor = tests["evidence"][platform]
            xml = read_plain(Path(anchor["path"]))
            need(len(xml) == anchor["bytes"] and sha256(xml).hexdigest() == anchor["sha256"],
                 "unchanged CPU XML before copying")
            frozen.write(root / (platform + "-cpu.xml"), xml)
        dcl = dict(protocol=PROTOCOL, source=args.source, source_binding=source, native_root=str(root), owner_pid=os.getpid(),
            runtime=runtime, service=service, cpu_tests=tests, lease=lease, services=services, baseline=baseline,
            bounds=BOUNDS, deadline_unix=args.deadline, replay_input=retained.INPUT, parent=parent,
            exclusive_gpu_claimed=False, flags=dict(FLAGS))
        frozen.write(root / "declaration.json", dcl)
        env = retained.child_env(root)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        argv = (sys.executable, "-m", "mjlab_microduck.stance_solver_gradient_probe", "--child", "--source", args.source,
                "--deadline", str(args.deadline), "--owner-pid", str(os.getpid()), "--lease-fd", str(fd),
                "--declaration-sha256", sha256(canonical(dcl)).hexdigest())
        samples = []
        def probe():
            shared.READ_DEADLINE = time.monotonic() + BOUNDS["probe_seconds"] - 1
            retained.check_deadline(args.deadline, BOUNDS["margin_seconds"])
            need(shared.lease_identity(fd) == lease and shared.service_snapshot() == services, "same held lease/protected services")
            need(len(samples) < 1024, "bounded telemetry collection")
            samples.append(shared.capacity(shared.telemetry(), baseline))
        try:
            with (root / "child.log").open("xb") as log:
                result = supervisor.supervise(argv, cwd=ROOT, env=env, log=log, probe=probe,
                    timeout=BOUNDS["child_seconds"], probe_timeout=BOUNDS["probe_seconds"],
                    grace=BOUNDS["cleanup_seconds"], interval=0.25, pass_fds=(fd,))
                log.flush(); os.fsync(log.fileno())
            retained.assert_cgroup_owner_only(service, os.getpid())
            need(source_binding(args.source) == source and retained.runtime_binding() == runtime, "whole source/runtime pins after child")
            probe()
            frozen.write(root / "telemetry.json", dict(samples=samples, flags=dict(FLAGS)))
            frozen.write(root / "supervision.json", dict(result=result, kernel_cgroup_only_owner=True,
                unit_retirement_independently_required=True, flags=dict(static.FLAGS)))
        except BaseException as error:
            frozen.write(root / "failure.json", dict(error_type=type(error).__name__, error=str(error)[:4096],
                unit_retirement_independently_required=True, flags=dict(FLAGS)))
            raise
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--cpu-evidence")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--declaration-sha256")
    args = parser.parse_args()
    child(args) if args.child else owner(args)


if __name__ == "__main__": main()
