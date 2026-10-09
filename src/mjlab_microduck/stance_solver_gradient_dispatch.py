"""Stopped initialization-gradient caller and complete readback preparation.

No device imports, collector admission or module compilation. The native owner
must separately authenticate parent bytes, executable loading, external inventory
and retirement. Trusted in-process integrity, not a hostile Python sandbox.
"""
import ast
from hashlib import sha1, sha256
import inspect
from math import prod
from pathlib import Path
import sys
from threading import get_ident
from types import FunctionType, MappingProxyType

from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_cost_dispatch as cost
from mjlab_microduck import stance_solver_dispatch_guard as shared
from mjlab_microduck import stance_solver_gradient_prefix as prefix
from mjlab_microduck import stance_solver_init_control as init

PROTOCOL = "microduck-stopped-gradient-dispatch-oct9-v1"
BASE = "870d933dc69e7bc3070e2e3b3c645f31c7f2b99c"
OWN = frozenset({"src/mjlab_microduck/stance_solver_gradient_dispatch.py",
    "tests/test_stance_solver_gradient_dispatch.py",
    "docs/experiments/2026-10-09-gradient-dispatch-preparation.md"})
TESTS = prefix.TESTS + ("tests/test_stance_solver_gradient_dispatch.py",)
NAMES = ("update_gradient_zero_grad_dot", "update_gradient_grad")
SPECS = tuple((n, *init.SOLVER_INIT_SPECS[n]) for n in prefix.ORDER)
need = prefix.need


def source_binding(root):
    """CPU preparation's own fence; never widen an older native source fence."""
    cmd = prefix.prior.retained.command
    need(type(root) is type(Path.cwd()) and root == root.resolve(strict=True)
         and Path.cwd().resolve() == root
         and cmd("git", "branch", "--show-current").decode().strip() == prefix.prior.BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact gradient adapter feature branch")
    source = cmd("git", "rev-parse", "HEAD").decode().strip()
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN,
         "separate exact three-path gradient adapter fence")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed gradient adapter leaf")
        raw = prefix.prior.read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
             "whole committed gradient adapter blob")
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(commit=source, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(), base=BASE, leaves=leaves)


def _arrays(m, d, ctx):
    roots = dict(model=m, data=d, contact=d.contact, efc=d.efc, context=ctx)
    result = {}
    for name in prefix.ORDER:
        value = roots[name.split(".")[0]]
        for part in name.split(".")[1:]:
            value = getattr(value, part)
        result[name] = value
    return result


class _Sealed:
    __slots__ = ()

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("gradient dispatch bindings are sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("gradient dispatch bindings are sealed")
        object.__delattr__(self, name)


class GradientStageCapture(_Sealed):
    """One-shot 26-field same-stream snapshots; dispatch itself is opaque here."""
    __slots__ = ("wp", "arrays", "stream", "device", "device_identity", "stream_handle",
        "owner_thread", "guard", "guard_code", "array_type", "entries", "numpy_entry",
        "sources", "layouts", "host", "host_layouts", "packets", "hashes", "next_stage",
        "failed", "active", "_sealed")

    def __init__(self, *, wp, arrays, stream, guard):
        need(sys.byteorder == "little" and type(arrays) is dict and set(arrays) == set(prefix.ORDER)
             and callable(guard) and getattr(guard, "__code__", None) is not None,
             "complete gradient capture with Python owner guard")
        self.wp, self.arrays, self.stream, self.guard = wp, MappingProxyType(dict(arrays)), stream, guard
        self.guard_code, self.device = guard.__code__, stream.device
        self.device_identity = (self.device.arch, self.device.context)
        self.stream_handle, self.owner_thread, self.array_type = stream.cuda_stream, get_ident(), wp.array
        need(self.device.is_cuda is True and self.device.arch == 120
             and type(self.device.context) is int and self.device.context > 0
             and type(self.stream_handle) is int and self.stream_handle > 0,
             "explicit sm120 gradient stream")
        self.entries = MappingProxyType({n: (getattr(wp, n), getattr(wp, n).__code__)
            for n in ("empty", "copy", "synchronize_stream")})
        self.numpy_entry = (wp.array.numpy, wp.array.numpy.__code__)
        self.sources = tuple(arrays[n] for n in prefix.ORDER)
        self.layouts = self._layouts(self.sources, host=False)
        self.host = tuple(self.entries["empty"][0](shape=shape, dtype=getattr(wp, dtype),
            device="cpu", pinned=True, requires_grad=False) for _, shape, _, dtype in SPECS)
        self.host_layouts = self._layouts(self.host, host=True)
        need(not any(a[1] < b[1] + b[-1] and b[1] < a[1] + a[-1]
            for a in self.layouts for b in self.host_layouts), "disjoint gradient source/staging spans")
        self.packets, self.hashes = MappingProxyType({}), MappingProxyType({})
        self.next_stage, self.failed, self.active = 0, False, False
        self._sealed = True
        self._check()

    def _layouts(self, arrays, *, host):
        rows = []
        for array, (_, shape, _, dtype) in zip(arrays, SPECS):
            need(type(array) is self.array_type and tuple(array.shape) == shape
                 and all(type(x) is int for x in array.shape) and array.dtype is getattr(self.wp, dtype)
                 and array.is_contiguous is True and array.requires_grad is False
                 and type(array.ptr) is int and array.ptr > 0, "literal gradient array layout")
            need((array.device.is_cpu is True and array.device.is_cuda is False and array.pinned is True)
                 if host else array.device is self.device, "literal gradient array device")
            rows.append((id(array), array.ptr, shape, id(array.dtype), id(array.device),
                         prod(shape) * (1 if dtype == "bool" else 4)))
        spans = sorted((r[1], r[1] + r[-1]) for r in rows)
        need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "nonoverlapping gradient allocations")
        return tuple(rows)

    def _check(self):
        need(get_ident() == self.owner_thread and self.wp.array is self.array_type
             and self.wp.array.numpy is self.numpy_entry[0] and self.wp.array.numpy.__code__ is self.numpy_entry[1]
             and all(getattr(self.wp, n) is fn and fn.__code__ is code for n, (fn, code) in self.entries.items()),
             "held gradient readback runtime entries")
        need(self.guard.__code__ is self.guard_code and self.stream.device is self.device
             and type(self.stream.cuda_stream) is int and self.stream.cuda_stream == self.stream_handle
             and self.device.is_cuda is True and (self.device.arch, self.device.context) == self.device_identity,
             "held gradient guard and stream/device")
        need(self._layouts(self.sources, host=False) == self.layouts
             and self._layouts(self.host, host=True) == self.host_layouts
             and all(self.arrays[n] is a for n, a in zip(prefix.ORDER, self.sources)),
             "unchanged gradient source/staging allocations")
        need(set(self.packets) == set(self.hashes) and all(type(raw) is bytes
             and len(raw) == prefix.PACKET_BYTES and sha256(raw).hexdigest() == self.hashes[p]
             for p, raw in self.packets.items()), "unchanged gradient retained bytes")
        self.guard()

    def _snapshot(self, phase):
        need(self.active and not self.failed and len(self.packets) < len(prefix.PHASES)
             and phase == prefix.PHASES[len(self.packets)]
             and phase.startswith(prefix.STAGES[self.next_stage] + "."), "ordered active gradient snapshot")
        self._check()
        self.entries["synchronize_stream"][0](self.stream)
        self._check()
        for dst, src in zip(self.host, self.sources):
            self.entries["copy"][0](dst, src, stream=self.stream)
            self._check()
        self.entries["synchronize_stream"][0](self.stream)
        self._check()
        fields = {}
        for array, (name, _, shape, dtype) in zip(self.host, SPECS):
            view = self.numpy_entry[0](array)
            need(tuple(view.shape) == shape and view.dtype.str == {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
                 and view.flags.c_contiguous is True and type(view.nbytes) is int
                 and view.nbytes == prod(shape) * (1 if dtype == "bool" else 4), "whole canonical gradient CPU view")
            fields[name] = view.tobytes(order="C")
            self._check()
        raw = prefix.pack_bank(fields)
        object.__setattr__(self, "packets", MappingProxyType(dict(self.packets, **{phase: raw})))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes, **{phase: sha256(raw).hexdigest()})))

    def bracket(self, stage, dispatch):
        if not (not self.failed and not self.active and self.next_stage < 2
                and stage == prefix.STAGES[self.next_stage] and callable(dispatch)):
            object.__setattr__(self, "failed", True)
            raise ValueError("ordered one-shot gradient bracket")
        object.__setattr__(self, "active", True)
        try:
            self._snapshot(stage + ".before")
            result = dispatch()
            self._check()
            self._snapshot(stage + ".after")
            object.__setattr__(self, "next_stage", self.next_stage + 1)
            return result
        except BaseException:
            object.__setattr__(self, "failed", True)
            raise
        finally:
            object.__setattr__(self, "active", False)

    def raw(self, phase):
        need(phase in prefix.PHASES and phase in self.packets, "retained complete gradient snapshot")
        raw = self.packets[phase]
        need(type(raw) is bytes and len(raw) == prefix.PACKET_BYTES
             and sha256(raw).hexdigest() == self.hashes[phase], "stable gradient snapshot")
        return raw

    def record(self):
        need(not self.failed and not self.active and self.next_stage == 2
             and set(self.packets) == set(prefix.PHASES), "complete gradient capture only")
        self._check()
        return dict(protocol=PROTOCOL, stages=list(prefix.STAGES), fields=list(prefix.ORDER),
            packets={p: dict(bytes=prefix.PACKET_BYTES, sha256=self.hashes[p]) for p in prefix.PHASES},
            timing_changed_by_readback=True, copy_stream_handle=self.stream_handle,
            capture_origin_authenticated=False, gpu_dispatch_authenticated=False,
            qualification=dict(prefix.FLAGS))


class _PrefixStop(BaseException):
    """Private intentional stop; only the exact per-observer instance is accepted."""


class GradientDispatchObserver(_Sealed):
    """Observe original _update_gradient's first two launches, then never resume it."""
    __slots__ = ("solver", "wp", "runtime", "bindings", "model", "data", "context", "stream",
        "device", "stream_handle", "owner_thread", "path", "caller", "caller_code", "kernels",
        "functions", "codes", "types", "original", "original_code", "signature", "entries",
        "trace_entries", "wrapper", "arrays", "capture", "source_lines", "branch_line", "stop",
        "used", "active", "busy", "completed", "calls", "stopped", "_sealed")

    def __init__(self, *, solver, wp, runtime, bindings, model, data, context, stream):
        need(type(bindings) is dict and set(bindings) == set(prefix.STAGES)
             and all(type(b) is artifacts.LoadedModuleBinding and b.artifact.format == "cubin"
                     for b in bindings.values()), "two already-bound gradient CUBIN objects")
        self.solver, self.wp, self.runtime = solver, wp, runtime
        self.bindings = MappingProxyType(dict(bindings))
        self.model, self.data, self.context, self.stream = model, data, context, stream
        self.device, self.stream_handle, self.owner_thread = bindings[prefix.STAGES[0]].device, stream.cuda_stream, get_ident()
        self.path = (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)
        need(Path(solver.__file__).resolve(strict=True) == self.path, "installed gradient solver path")
        raw = self.path.read_bytes()
        prefix.verify_source(raw)
        self.caller = solver._update_gradient
        need(type(self.caller) is FunctionType, "literal original gradient caller")
        self.caller_code = self.caller.__code__
        codes = cost._codes(raw, self.caller_code.co_filename)
        need(Path(self.caller_code.co_filename).resolve(strict=True) == self.path
             and self.caller_code == codes["_update_gradient"], "frozen original gradient caller code")
        self.kernels = tuple(bindings[s].kernel for s in prefix.STAGES)
        self.functions = tuple(k.func for k in self.kernels)
        need(all(type(f) is FunctionType and f.__code__ == codes[n]
             and Path(f.__code__.co_filename).resolve(strict=True) == self.path
             for f, n in zip(self.functions, NAMES)), "frozen original gradient kernel codes")
        self.codes, self.types = tuple(f.__code__ for f in self.functions), solver.types
        fn = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "_update_gradient")
        self.source_lines, self.branch_line = tuple(n.lineno for n in fn.body[:2]), fn.body[2].lineno
        need(self.source_lines == (2925, 2927) and self.branch_line == 2934,
             "exact stopped gradient call sites and excluded next branch")
        self.original, self.original_code = wp.launch, wp.launch.__code__
        self.signature = inspect.signature(self.original)
        need(set(self.signature.parameters) == set(("kernel", "dim", "inputs", "outputs", "adj_inputs",
            "adj_outputs", "device", "stream", "adjoint", "record_tape", "record_cmd", "max_blocks", "block_dim")),
            "literal gradient launch parameter inventory")
        self.entries = MappingProxyType({n: (getattr(wp, n), getattr(wp, n).__code__)
            for n in ("get_device", "get_stream", "synchronize_stream")})
        self.trace_entries = (sys.gettrace, sys.getprofile)
        self.wrapper, self.arrays = self._launch, MappingProxyType(_arrays(model, data, context))
        self.capture = None
        self.used = self.active = self.busy = self.completed = self.stopped = False
        self.calls, self.stop = 0, _PrefixStop()
        self._check()
        self.capture = GradientStageCapture(wp=wp, arrays=dict(self.arrays), stream=stream, guard=self._check)
        self._sealed = True

    def _check(self):
        s, wp = self.solver, self.wp
        need(get_ident() == self.owner_thread and (sys.gettrace, sys.getprofile) == self.trace_entries
             and self.trace_entries[0]() is None and self.trace_entries[1]() is None,
             "owned gradient thread without trace/profile instrumentation")
        need(sha256(self.path.read_bytes()).hexdigest() == prefix.static.SOLVER_SHA256
             and sys.modules.get(prefix.static.MODULE) is s and s.__name__ == prefix.static.MODULE
             and Path(s.__file__).resolve(strict=True) == self.path and s.wp is wp and s.types is self.types
             and s._update_gradient is self.caller and self.caller.__code__ is self.caller_code
             and self.caller.__globals__ is s.__dict__ and self.caller.__defaults__ is None
             and self.caller.__kwdefaults__ is None and self.caller.__closure__ is None,
             "unchanged gradient source/caller namespace")
        for name, kernel, fn, code in zip(NAMES, self.kernels, self.functions, self.codes):
            need(getattr(s, name) is kernel and kernel.func is fn and fn.__code__ is code
                 and fn.__globals__ is s.__dict__ and fn.__defaults__ is None
                 and fn.__kwdefaults__ is None and fn.__closure__ is None, "unchanged gradient kernel identity/body")
        first, second = (self.bindings[s] for s in prefix.STAGES)
        for binding in self.bindings.values():
            binding.assert_unchanged()
            need(binding.device is self.device and binding.block_dim == 256, "same gradient executable device/block")
        need(first.module is second.module and first.executable is second.executable and first.artifact == second.artifact,
             "same explicitly supplied two-entry gradient module")
        need(wp.launch is (self.wrapper if self.active else self.original) and self.original.__code__ is self.original_code
             and inspect.signature(self.original) == self.signature
             and all(getattr(wp, n) is fn and fn.__code__ is code for n, (fn, code) in self.entries.items()),
             "owned gradient launch hook and runtime entries")
        need(self.runtime.tape is None and self.entries["get_device"][0]() is self.device
             and self.entries["get_stream"][0](self.device) is self.stream and self.stream.device is self.device
             and type(self.stream.cuda_stream) is int and self.stream.cuda_stream == self.stream_handle > 0
             and self.device.is_cuda is True and self.device.arch == 120, "eager explicit same-device gradient stream")
        need(self.model.is_sparse is False and type(self.model.nv) is int and self.model.nv == 20
             and type(self.data.nworld) is int and self.data.nworld == 64
             and type(self.data.njmax) is int and self.data.njmax == 512,
             "literal bounded gradient recipe")
        actual = _arrays(self.model, self.data, self.context)
        need(all(actual[n] is a for n, a in self.arrays.items()), "unchanged gradient caller arrays")
        if self.capture is not None:
            need(self.capture._layouts(tuple(actual[n] for n in prefix.ORDER), host=False) == self.capture.layouts,
                 "unchanged gradient caller buffer layouts")

    def _launch(self, *args, **kwargs):
        self._check()
        need(self.active and not self.busy and self.calls < 2, "two ordered direct gradient launches only")
        index = self.calls
        frame = inspect.currentframe().f_back
        try:
            need(frame.f_code is self.caller_code and frame.f_globals is self.solver.__dict__
                 and frame.f_lineno == self.source_lines[index] and frame.f_locals.get("m") is self.model
                 and frame.f_locals.get("d") is self.data and frame.f_locals.get("ctx") is self.context,
                 "exact direct original gradient caller site")
        finally:
            del frame
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call, d, ctx = bound.arguments, self.data, self.context
        dim, inputs, outputs = ((64, (ctx.done,), (ctx.grad_dot,)),
            ((64, 20), (d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done), (ctx.grad, ctx.grad_dot)))[index]
        need(call["kernel"] is self.kernels[index] and type(call["dim"]) is type(dim) and call["dim"] == dim
             and (type(dim) is int or all(type(x) is int for x in call["dim"])), "exact gradient kernel/dimensions")
        for name, expected in (("inputs", inputs), ("outputs", outputs)):
            need(type(call[name]) is list and len(call[name]) == len(expected)
                 and all(a is b for a, b in zip(call[name], expected)), "exact gradient argument identities/order")
        need(call["device"] is None and call["stream"] is None and call["adjoint"] is False
             and call["record_cmd"] is False and call["record_tape"] is True
             and type(call["adj_inputs"]) is list and not call["adj_inputs"]
             and type(call["adj_outputs"]) is list and not call["adj_outputs"]
             and type(call["max_blocks"]) is int and call["max_blocks"] == 0
             and type(call["block_dim"]) is int and call["block_dim"] == 256, "unchanged gradient eager launch defaults")
        object.__setattr__(self, "busy", True)
        try:
            def dispatch():
                self._check()
                result = self.original(*args, **kwargs)
                self._check()
                return result
            result = self.capture.bracket(prefix.STAGES[index], dispatch)
            object.__setattr__(self, "calls", index + 1)
            if index == 1:
                self._check()
                raise self.stop
            return result
        finally:
            object.__setattr__(self, "busy", False)

    def run(self):
        need(not self.used and not self.active, "one-shot stopped gradient observer")
        object.__setattr__(self, "used", True)
        need(shared._HOOK_LOCK.acquire(blocking=False), "exclusive shared launch hook")
        try:
            self._check()
            self.capture._check()
            need(not self.capture.failed and not self.capture.active and self.capture.next_stage == 0
                 and not self.capture.packets, "fresh gradient capture")
            self.wp.launch = self.wrapper
            object.__setattr__(self, "active", True)
            try:
                self.caller(self.model, self.data, self.context)
            except _PrefixStop as stop:
                need(stop is self.stop and self.calls == 2 and not self.busy,
                     "exact intentional stop after both completed gradient captures")
                object.__setattr__(self, "stopped", True)
            else:
                raise ValueError("normal gradient caller return is not a prefix stop")
            self._check()
            self.entries["synchronize_stream"][0](self.stream)
            self._check()
            self.capture.record()
            object.__setattr__(self, "completed", True)
        except BaseException:
            object.__setattr__(self.capture, "failed", True)
            raise
        finally:
            foreign = self.active and self.wp.launch is not self.wrapper
            if self.active and not foreign:
                self.wp.launch = self.original
            object.__setattr__(self, "active", False)
            shared._HOOK_LOCK.release()
            if foreign:
                object.__setattr__(self, "completed", False)
                object.__setattr__(self.capture, "failed", True)
                raise ValueError("foreign gradient launch hook preserved; no successful receipt")

    def record(self):
        need(self.completed and self.stopped and not self.active and not self.busy and self.calls == 2,
             "completed intentionally stopped gradient prefix only")
        self._check()
        return dict(protocol=PROTOCOL, base=BASE, caller="_update_gradient", path="initialization-prefix",
            call_site_lines=list(self.source_lines), excluded_next_branch_line=self.branch_line,
            stop="private-sentinel-after-second-launch-and-readback-before-caller-resumption",
            call_sites_observed=True, intentional_stop_observed=True, normal_caller_completion=False,
            next_branch_observed=False, trace_hooks_installed=False, incremental_iteration_observed=False,
            bindings={n: b.record() for n, b in self.bindings.items()}, capture=self.capture.record(),
            parent_capture_authenticated=False, explicit_load_provenance_authenticated=False,
            native_gpu_execution_authenticated=False, external_unit_retirement_authenticated=False,
            qualification=dict(prefix.FLAGS))
