"""One-shot dense dispatch guard, not a supervisor or numerical acceptance gate.

Importing this module initializes no device. The caller owns the fresh child,
lease, caps, compiler/runtime pins, explicit CUBIN load and raw data retention.
This is observed-object integrity, not a sandbox against hostile Python code.
"""
from hashlib import sha256
import inspect
from pathlib import Path
import sys
from threading import Lock
from types import CodeType

from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-dense-solver-dispatch-guard-oct8-v1"
_HOOK_LOCK = Lock()
need = static.need


class DenseSolverDispatchGuard:
    """Delegate one unchanged _update_constraint call, counting its dense launch."""

    def __init__(self, *, solver, wp, runtime, binding, model, data, context, stream):
        need(type(binding) is artifacts.LoadedModuleBinding and binding.artifact.format == "cubin",
             "already explicitly loaded CUBIN binding")
        self.solver, self.wp, self.runtime, self.binding = solver, wp, runtime, binding
        self.model, self.data, self.context, self.stream = model, data, context, stream
        self.path = (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)
        need(Path(solver.__file__).resolve(strict=True) == self.path, "installed frozen solver path")
        raw = self.path.read_bytes()
        self.source = static.verify_source(raw)
        self.caller, self.kernel = solver._update_constraint, getattr(solver, static.TARGET)
        self.caller_code, self.kernel_code = self.caller.__code__, self.kernel.func.__code__
        need(Path(self.caller_code.co_filename).resolve(strict=True) == self.path
             and Path(self.kernel_code.co_filename).resolve(strict=True) == self.path,
             "loaded code filenames resolve to pinned solver")
        # Compile only, never execute/import the source. CPython code equality binds
        # the loaded function bodies, not merely source text adjacent to them.
        expected = compile(raw, self.caller_code.co_filename, "exec", dont_inherit=True, optimize=0)
        codes = {c.co_name: c for c in expected.co_consts if isinstance(c, CodeType)}
        expected_kernel = compile(raw, self.kernel_code.co_filename, "exec", dont_inherit=True, optimize=0)
        kernel_codes = {c.co_name: c for c in expected_kernel.co_consts if isinstance(c, CodeType)}
        need(self.caller_code == codes["_update_constraint"]
             and self.kernel_code == kernel_codes[static.TARGET], "pinned loaded caller and kernel code")
        self.original = wp.launch
        self.original_code = self.original.__code__
        self.signature = inspect.signature(self.original)
        self.entries = {name: (getattr(wp, name), getattr(wp, name).__code__)
                        for name in ("get_device", "get_stream", "synchronize_stream")}
        self.dependencies = {name: solver.__dict__[name] for name in (
            "update_constraint_init_cost", "update_constraint_efc", "update_constraint_gauss_cost", "ceil")}
        self.wrapper = self._launch
        self.stream_handle = stream.cuda_stream
        self.active = self.used = self.completed = False
        self.calls = 0
        self.layouts = self._layouts()
        self._guard()

    def _layouts(self):
        m, d, ctx, wp = self.model, self.data, self.context, self.wp
        need(m.is_sparse is False and type(m.nv) is int and m.nv == 20
             and type(d.nworld) is int and d.nworld == 64
             and type(d.njmax) is int and d.njmax == 512, "literal bounded dense recipe")
        values = (d.nefc, d.efc.J, d.efc.force, ctx.done, d.qfrc_constraint)
        shapes = ((64,), (64, 512, 20), (64, 512), (64,), (64, 20))
        dtypes = (wp.int32, wp.float32, wp.float32, wp.bool, wp.float32)
        result = []
        for value, shape, dtype in zip(values, shapes, dtypes):
            need(type(value.ptr) is int and value.ptr > 0 and tuple(value.shape) == shape
                 and value.dtype is dtype and value.device is self.binding.device
                 and value.is_contiguous is True and value.requires_grad is False,
                 "bounded literal target array layout and device")
            result.append((id(value), value.ptr, shape, id(dtype)))
        need(len({x[1] for x in result}) == 5, "nonaliasing target buffers")
        sizes = (64 * 4, 64 * 512 * 20 * 4, 64 * 512 * 4, 64, 64 * 20 * 4)
        spans = sorted((row[1], row[1] + size) for row, size in zip(result, sizes))
        need(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])),
             "nonoverlapping contiguous target buffer ranges")
        return tuple(result)

    def _guard(self):
        b, s, wp = self.binding, self.solver, self.wp
        b.assert_unchanged()
        need(sha256(self.path.read_bytes()).hexdigest() == static.SOLVER_SHA256,
             "installed solver bytes unchanged")
        need(s.__name__ == static.MODULE and sys.modules.get(static.MODULE) is s
             and s.wp is wp and s._update_constraint is self.caller
             and self.caller.__code__ is self.caller_code
             and self.caller.__globals__ is s.__dict__
             and self.caller.__defaults__ == (False,) and self.caller.__defaults__[0] is False
             and self.caller.__kwdefaults__ is None
             and getattr(s, static.TARGET) is self.kernel and b.kernel is self.kernel
             and self.kernel.func.__code__ is self.kernel_code
             and self.kernel.func.__globals__ is s.__dict__
             and self.kernel.func.__defaults__ is None and self.kernel.func.__kwdefaults__ is None
             and all(s.__dict__[name] is value for name, value in self.dependencies.items()),
             "unchanged literal solver namespace, caller, target and dependencies")
        need(wp.launch is (self.wrapper if self.active else self.original)
             and self.original.__code__ is self.original_code
             and inspect.signature(self.original) == self.signature
             and all(getattr(wp, name) is fn and fn.__code__ is code
                     for name, (fn, code) in self.entries.items()), "owned launch and runtime entries")
        need(self.runtime.tape is None and b.device.arch == 120 and b.block_dim == 256
             and self.entries["get_device"][0]() is b.device
             and self.entries["get_stream"][0](b.device) is self.stream
             and self.stream.device is b.device and type(self.stream.cuda_stream) is int
             and self.stream.cuda_stream == self.stream_handle > 0, "eager sm120 stream without tape")
        need(self._layouts() == self.layouts, "same target buffer identities and pointers")

    def _launch(self, *args, **kwargs):
        self._guard()
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        if call["kernel"] is self.kernel:
            frame = inspect.currentframe().f_back
            try:
                need(self.active and self.calls == 0 and frame.f_code is self.caller_code
                     and frame.f_globals is self.solver.__dict__
                     and frame.f_lineno == self.source["dispatch"]["dispatch_line"]
                     and frame.f_locals.get("m") is self.model
                     and frame.f_locals.get("d") is self.data
                     and frame.f_locals.get("ctx") is self.context,
                     "single direct target call from pinned dense branch")
            finally:
                del frame
            d, ctx = self.data, self.context
            expected = (d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done)
            need(type(call["dim"]) is tuple and call["dim"] == (64, 20)
                 and all(type(x) is int for x in call["dim"])
                 and type(call["inputs"]) is list and len(call["inputs"]) == 5
                 and all(call["inputs"][i] is expected[i] for i in (0, 1, 2, 4))
                 and type(call["inputs"][3]) is int and call["inputs"][3] == 512
                 and type(call["outputs"]) is list and len(call["outputs"]) == 1
                 and call["outputs"][0] is d.qfrc_constraint, "exact target launch arguments")
            need(call["device"] is None and call["stream"] is None
                 and call["adjoint"] is False and call["record_cmd"] is False
                 and call["record_tape"] is True
                 and call["adj_inputs"] == [] and call["adj_outputs"] == []
                 and type(call["max_blocks"]) is int and call["max_blocks"] == 0
                 and type(call["block_dim"]) is int and call["block_dim"] == 256,
                 "unchanged eager forward launch defaults")
            self.calls += 1
        result = self.original(*args, **kwargs)
        self._guard()
        return result

    def run(self):
        need(not self.used and not self.active, "one-shot observer")
        need(_HOOK_LOCK.acquire(blocking=False), "exclusive owned launch hook")
        self.used = True
        try:
            self._guard()
            self.wp.launch = self.wrapper
            self.active = True
            self.caller(self.model, self.data, self.context)
            need(self.calls == 1, "exactly one dense target dispatch")
            self.entries["synchronize_stream"][0](self.stream)
            self._guard()
            self.completed = True
        finally:
            foreign = self.active and self.wp.launch is not self.wrapper
            if self.active and not foreign:
                self.wp.launch = self.original
            self.active = False
            _HOOK_LOCK.release()
            if foreign:
                self.completed = False
                raise ValueError("foreign launch hook preserved, no successful receipt")

    def record(self):
        need(self.completed and not self.active and self.calls == 1, "completed one-shot dispatch")
        self._guard()
        return dict(protocol=PROTOCOL, decision="dense-solver-dispatch-guard-complete-not-qualification",
                    observed_target_calls=self.calls, caller="_update_constraint",
                    dispatch_line=self.source["dispatch"]["dispatch_line"],
                    dimensions=[64, 20], input_order=self.source["dispatch"]["inputs"],
                    output_order=self.source["dispatch"]["outputs"],
                    layouts=[dict(object_id=i, pointer=p, shape=list(s), dtype_object_id=t)
                             for i, p, s, t in self.layouts],
                    stream_object_id=id(self.stream), stream_handle=self.stream.cuda_stream,
                    binding=self.binding.record(), flags=dict(static.FLAGS),
                    numerical_acceptance=False, driver_loaded_code_observed=False)
