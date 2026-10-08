"""Four-stage frozen caller observer; no device initialization or admission.

The owner supplies four already-bound executables and a restored scratch bank.
This observes Python call sites and loaded objects, not loader provenance,
driver-resident bytes, external retirement, numerical or training acceptance.
It is trusted in-process integrity, not a hostile-Python sandbox.
"""
import ast
from hashlib import sha256
import inspect
from math import ceil, prod
from pathlib import Path
import sys
from threading import get_ident
from types import CodeType, FunctionType, MappingProxyType

from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_dispatch_guard as dense_guard
from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-caller-cost-dispatch-oct9-v1"
BASE = "7aec9cebc2178be909ba119a0fa2cb504b264695"
UTIL_MODULE = "mujoco_warp._src.warp_util"
UTIL_SHA256 = "b0d156ddbced4848cd6cbcd70977f1e96c230fce0d746a6acfa911549afb447c"
NAMES = ("update_constraint_init_cost", "update_constraint_efc",
         static.TARGET, "update_constraint_gauss_cost")
EXTRA_SPECS = (("contact.friction", (8192,), "vec5", 20),
               ("context.changed_efc_ids", (64, 512), "int32", 4),
               ("context.changed_efc_count", (64,), "int32", 4))
need = static.need


def _codes(raw, filename):
    return {c.co_name: c for c in compile(raw, filename, "exec", dont_inherit=True,
                                        optimize=0).co_consts if isinstance(c, CodeType)}


def _nested(code, name):
    values = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == name]
    need(len(values) == 1, "unique frozen nested code")
    return values[0]


def _closure(fn):
    return tuple(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ())))


def _arrays(m, d, ctx):
    roots = dict(model=m, data=d, contact=d.contact, efc=d.efc, context=ctx)
    result = {}
    for name in (*stages.ORDER, *(s[0] for s in EXTRA_SPECS)):
        if name == "contact.nacon":
            result[name] = d.nacon
            continue
        value = roots[name.split(".")[0]]
        for part in name.split(".")[1:]:
            value = getattr(value, part)
        result[name] = value
    result["contact.nacon"] = d.nacon
    return result


class CostDispatchObserver:
    """Consume exactly one unchanged dense _update_constraint(False) call."""

    __slots__ = ("solver", "util", "wp", "runtime", "bindings", "model", "data", "context",
                 "stream", "device", "stream_handle", "owner_thread", "path", "util_path",
                 "caller", "caller_code", "kernels", "kernel_functions", "kernel_codes", "factories",
                 "factory_functions", "factory_codes", "wrapper_codes", "cache", "cache_entries",
                 "source_lines", "types", "vec5", "math", "original", "original_code",
                 "signature", "entries", "wrapper", "arrays", "layouts", "capture",
                 "used", "active", "busy", "completed", "calls", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("caller cost-dispatch bindings are sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("caller cost-dispatch bindings are sealed")
        object.__delattr__(self, name)

    def __init__(self, *, solver, util, wp, runtime, bindings, model, data, context, stream):
        need(type(bindings) is dict and set(bindings) == set(stages.STAGES)
             and all(type(b) is artifacts.LoadedModuleBinding and b.artifact.format == "cubin"
                     for b in bindings.values()), "four already-bound CUBIN stage objects")
        self.solver, self.util, self.wp, self.runtime = solver, util, wp, runtime
        self.bindings = MappingProxyType(dict(bindings))
        self.model, self.data, self.context, self.stream = model, data, context, stream
        self.device = self.bindings["dense"].device
        self.stream_handle, self.owner_thread = stream.cuda_stream, get_ident()
        root = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src"
        self.path, self.util_path = (root / "solver.py").resolve(strict=True), (root / "warp_util.py").resolve(strict=True)
        need(Path(solver.__file__).resolve(strict=True) == self.path
             and Path(util.__file__).resolve(strict=True) == self.util_path, "installed frozen source paths")
        raw, uraw = self.path.read_bytes(), self.util_path.read_bytes()
        static.verify_source(raw)
        need(sha256(uraw).hexdigest() == UTIL_SHA256, "whole frozen cache wrapper source")
        self.caller = solver._update_constraint
        need(type(self.caller) is FunctionType, "literal Python caller")
        self.caller_code = self.caller.__code__
        self.factories = (solver.update_constraint_efc, solver.update_constraint_gauss_cost)
        need(all(type(f) is FunctionType and Path(f.__code__.co_filename).resolve(strict=True) == self.util_path
                 for f in self.factories), "literal frozen Python cache wrappers")
        codes, ucodes = _codes(raw, self.caller_code.co_filename), _codes(uraw, self.factories[0].__code__.co_filename)
        need(Path(self.caller_code.co_filename).resolve(strict=True) == self.path
             and self.caller_code == codes["_update_constraint"], "frozen loaded caller code")
        self.wrapper_codes = tuple(f.__code__ for f in self.factories)
        need(all(c == _nested(ucodes["cache_kernel"], "wrapper") for c in self.wrapper_codes),
             "frozen loaded cache wrapper bodies")
        self.factory_functions = tuple(f.__wrapped__ for f in self.factories)
        need(all(type(f) is FunctionType and f.__code__ == codes[n]
                 for f, n in zip(self.factory_functions, (NAMES[1], NAMES[3]))), "frozen factory bodies")
        self.factory_codes = tuple(f.__code__ for f in self.factory_functions)
        self.kernels = tuple(self.bindings[s].kernel for s in stages.STAGES)
        self.kernel_functions = tuple(k.func for k in self.kernels)
        expected = (codes[NAMES[0]], _nested(self.factory_codes[0], "kernel"),
                    codes[NAMES[2]], _nested(self.factory_codes[1], "kernel"))
        need(all(type(f) is FunctionType and Path(f.__code__.co_filename).resolve(strict=True) == self.path
                 and f.__code__ == code for f, code in zip(self.kernel_functions, expected)),
             "four frozen loaded kernel bodies")
        self.kernel_codes = tuple(f.__code__ for f in self.kernel_functions)
        self.cache = util._KERNEL_CACHE
        need(type(self.cache) is dict, "literal kernel cache")
        self.cache_entries = MappingProxyType({(hash(False), hash(NAMES[1])): self.kernels[1],
                                              (hash(20), hash(50), hash(NAMES[3])): self.kernels[3]})
        tree = ast.parse(raw)
        caller = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_update_constraint")
        launches = sorted((n for n in ast.walk(caller) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                           and n.func.value.id == "wp" and n.func.attr == "launch"), key=lambda n: n.lineno)
        need(len(launches) == 5, "frozen caller launch inventory including excluded sparse branch")
        self.source_lines = tuple(launches[i].lineno for i in (0, 1, 3, 4))
        self.types, self.vec5, self.math = solver.types, solver.types.vec5, solver.math
        self.original, self.original_code = wp.launch, wp.launch.__code__
        self.signature = inspect.signature(self.original)
        need(set(self.signature.parameters) == set(("kernel", "dim", "inputs", "outputs", "adj_inputs",
             "adj_outputs", "device", "stream", "adjoint", "record_tape", "record_cmd", "max_blocks", "block_dim")),
             "literal eager launch parameter inventory")
        self.entries = MappingProxyType({n: (getattr(wp, n), getattr(wp, n).__code__)
                                         for n in ("get_device", "get_stream", "synchronize_stream")})
        self.wrapper = self._launch
        self.arrays = MappingProxyType(_arrays(model, data, context))
        self.layouts = self._layouts()
        self.used = self.active = self.busy = self.completed = False
        self.calls, self.capture = 0, None
        self._check()
        self.capture = stages.CostStageCapture(wp=wp, arrays={n: self.arrays[n] for n in stages.ORDER},
                                               stream=stream, guard=self._check)
        self._sealed = True

    def _layouts(self):
        m, d, wp = self.model, self.data, self.wp
        need(m.is_sparse is False and type(m.nv) is int and m.nv == 20
             and type(d.nworld) is int and d.nworld == 64 and type(d.njmax) is int and d.njmax == 512,
             "literal bounded dense recipe")
        actual = _arrays(m, d, self.context)
        need(all(actual[n] is a for n, a in self.arrays.items()), "same caller argument allocations")
        specs = tuple((n, shape, dtype, 1 if dtype == "bool" else 4)
                      for n, shape, _, dtype in stages.SPECS) + EXTRA_SPECS
        rows = []
        for name, shape, dtype, size in specs:
            a, kind = actual[name], self.vec5 if dtype == "vec5" else getattr(wp, dtype)
            need(type(a) is wp.array and type(a.ptr) is int and a.ptr > 0
                 and tuple(a.shape) == shape and all(type(x) is int for x in a.shape)
                 and a.dtype is kind and a.device is self.device
                 and a.is_contiguous is True and a.requires_grad is False, "literal caller array layout/device")
            rows.append((id(a), a.ptr, shape, id(kind), prod(shape) * size))
        spans = sorted((r[1], r[1] + r[-1]) for r in rows)
        need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "nonoverlapping caller buffers")
        return tuple(rows)

    def _check(self):
        s, u, wp = self.solver, self.util, self.wp
        need(get_ident() == self.owner_thread, "same owned caller thread")
        need(sha256(self.path.read_bytes()).hexdigest() == static.SOLVER_SHA256
             and sha256(self.util_path.read_bytes()).hexdigest() == UTIL_SHA256, "unchanged frozen source bytes")
        need(sys.modules.get(static.MODULE) is s and s.__name__ == static.MODULE
             and sys.modules.get(UTIL_MODULE) is u and u.__name__ == UTIL_MODULE
             and Path(s.__file__).resolve(strict=True) == self.path
             and Path(u.__file__).resolve(strict=True) == self.util_path
             and s.wp is wp and s.types is self.types and s.types.vec5 is self.vec5 and s.math is self.math
             and s.ceil is ceil and s._update_constraint is self.caller
             and self.caller.__code__ is self.caller_code and self.caller.__globals__ is s.__dict__
             and self.caller.__defaults__ == (False,) and self.caller.__defaults__[0] is False
             and self.caller.__kwdefaults__ is None and self.caller.__closure__ is None,
             "unchanged frozen caller namespace and defaults")
        need(s.update_constraint_init_cost is self.kernels[0] and getattr(s, NAMES[2]) is self.kernels[2],
             "unchanged top-level kernel identities")
        need(u._KERNEL_CACHE is self.cache and type(self.cache) is dict
             and all(self.cache.get(k) is v for k, v in self.cache_entries.items()), "held factory cache entries")
        for name, wrapper, wc, fn, code in zip((NAMES[1], NAMES[3]), self.factories, self.wrapper_codes,
                                             self.factory_functions, self.factory_codes):
            need(getattr(s, name) is wrapper and type(wrapper) is FunctionType
                 and wrapper.__code__ is wc and wrapper.__globals__ is u.__dict__
                 and wrapper.__defaults__ is None and wrapper.__kwdefaults__ is None
                 and _closure(wrapper) == (("func", fn),) and wrapper.__wrapped__ is fn
                 and fn.__code__ is code and fn.__globals__ is s.__dict__ and fn.__name__ == name
                 and fn.__defaults__ is None and fn.__kwdefaults__ is None and fn.__closure__ is None,
                 "frozen cache wrappers and original factory code")
        expected_closures = ((), (("TRACK_CHANGES", False),), (), (("dofs_per_thread", 50), ("nv", 20)))
        for kernel, fn, code, expected in zip(self.kernels, self.kernel_functions, self.kernel_codes, expected_closures):
            values = _closure(fn)
            need(kernel.func is fn and fn.__code__ is code and fn.__globals__ is s.__dict__ and fn.__defaults__ is None
                 and fn.__kwdefaults__ is None and len(values) == len(expected)
                 and all(n == en and type(v) is type(ev) and v == ev for (n, v), (en, ev) in zip(values, expected)),
                 "unchanged kernel functions and literal closure values")
        for b in self.bindings.values():
            b.assert_unchanged()
            need(b.device is self.device and b.block_dim == 256 and self.device.arch == 120,
                 "same sm120 executable device/block")
        b0, be, bd, bg = (self.bindings[n] for n in stages.STAGES)
        need(b0.module is bd.module and b0.executable is bd.executable
             and b0.artifact == bd.artifact and len({id(b0.module), id(be.module), id(bg.module)}) == 3,
             "shared top-level and separate unique-factory modules")
        need(wp.launch is (self.wrapper if self.active else self.original)
             and self.original.__code__ is self.original_code and inspect.signature(self.original) == self.signature
             and all(getattr(wp, n) is fn and fn.__code__ is code for n, (fn, code) in self.entries.items()),
             "owned launch hook and held runtime entries")
        need(self.runtime.tape is None and self.entries["get_device"][0]() is self.device
             and self.entries["get_stream"][0](self.device) is self.stream and self.stream.device is self.device
             and type(self.stream.cuda_stream) is int and self.stream.cuda_stream == self.stream_handle > 0,
             "eager same-device explicit stream without tape")
        need(self._layouts() == self.layouts, "unchanged caller buffer identities and pointers")
        if self.capture is not None:
            need(not any(a[1] < b[1] + b[-1] and b[1] < a[1] + a[-1]
                         for a in self.layouts for b in self.capture.host_layouts),
                 "all caller buffers disjoint from stage host staging")

    def _arguments(self, index):
        m, d, c = self.model, self.data, self.context
        return ((64, (c.cost, c.done), (c.gauss, c.cost, c.prev_cost)),
                ((64, 512), (m.opt.impratio_invsqrt, d.ne, d.nf, d.nefc, d.contact.friction,
                            d.contact.dim, d.contact.efc_address, d.efc.type, d.efc.id, d.efc.D,
                            d.efc.frictionloss, d.nacon, c.Jaref, c.done),
                 (d.efc.force, d.efc.state, c.cost, c.changed_efc_ids, c.changed_efc_count)),
                ((64, 20), (d.nefc, d.efc.J, d.efc.force, 512, c.done), (d.qfrc_constraint,)),
                ((64, 1), (d.qacc, d.qfrc_smooth, d.qacc_smooth, d.efc.Ma, c.done), (c.gauss, c.cost)))[index]

    def _launch(self, *args, **kwargs):
        self._check()
        need(self.active and not self.busy and self.calls < 4, "single ordered direct caller dispatch")
        index = self.calls
        frame = inspect.currentframe().f_back
        try:
            need(frame.f_code is self.caller_code and frame.f_globals is self.solver.__dict__
                 and frame.f_lineno == self.source_lines[index]
                 and frame.f_locals.get("m") is self.model and frame.f_locals.get("d") is self.data
                 and frame.f_locals.get("ctx") is self.context and frame.f_locals.get("track_changes") is False,
                 "exact direct frozen caller stage site")
            if index == 3:
                need(type(frame.f_locals.get("dofs_per_thread")) is int and frame.f_locals["dofs_per_thread"] == 50
                     and type(frame.f_locals.get("threads_per_efc")) is int and frame.f_locals["threads_per_efc"] == 1,
                     "single-thread Gauss recipe")
        finally:
            del frame
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        dim, inputs, outputs = self._arguments(index)
        need(call["kernel"] is self.kernels[index] and type(call["dim"]) is type(dim) and call["dim"] == dim
             and (type(dim) is int or all(type(x) is int for x in call["dim"])), "exact stage kernel and dimensions")
        for name, expected in (("inputs", inputs), ("outputs", outputs)):
            need(type(call[name]) is list and len(call[name]) == len(expected)
                 and all((type(a) is int and a == b) if type(b) is int else a is b
                         for a, b in zip(call[name], expected)), "exact stage argument identities/order")
        need(call["device"] is None and call["stream"] is None and call["adjoint"] is False
             and call["record_cmd"] is False and call["record_tape"] is True
             and type(call["adj_inputs"]) is list and not call["adj_inputs"]
             and type(call["adj_outputs"]) is list and not call["adj_outputs"]
             and type(call["max_blocks"]) is int and call["max_blocks"] == 0
             and type(call["block_dim"]) is int and call["block_dim"] == 256, "unchanged eager forward launch defaults")
        object.__setattr__(self, "busy", True)
        try:
            def dispatch():
                self._check()
                result = self.original(*args, **kwargs)
                self._check()
                return result
            result = self.capture.bracket(stages.STAGES[index], dispatch)
            object.__setattr__(self, "calls", index + 1)
            return result
        finally:
            object.__setattr__(self, "busy", False)

    def run(self):
        need(not self.used and not self.active, "one-shot caller observer")
        object.__setattr__(self, "used", True)
        need(dense_guard._HOOK_LOCK.acquire(blocking=False), "exclusive shared launch hook")
        try:
            self._check()
            self.capture._check()
            need(not self.capture.failed and not self.capture.active and self.capture.next_stage == 0
                 and not self.capture.packets, "fresh unused stage capture")
            self.wp.launch = self.wrapper
            object.__setattr__(self, "active", True)
            self.caller(self.model, self.data, self.context)
            need(self.calls == 4, "four completed ordered caller stages")
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
            dense_guard._HOOK_LOCK.release()
            if foreign:
                object.__setattr__(self, "completed", False)
                object.__setattr__(self.capture, "failed", True)
                raise ValueError("foreign launch hook preserved; no successful caller receipt")

    def record(self):
        need(self.completed and not self.active and not self.busy and self.calls == 4, "completed one-shot caller only")
        self._check()
        return dict(protocol=PROTOCOL, base=BASE, caller="_update_constraint", track_changes=False,
                    observed_stages=list(stages.STAGES), call_site_lines=list(self.source_lines),
                    call_sites_observed=True, timing_changed_by_readback=True,
                    bindings={n: b.record() for n, b in self.bindings.items()}, capture=self.capture.record(),
                    omitted_packet_fields=[s[0] for s in EXTRA_SPECS],
                    explicit_load_provenance_authenticated=False, native_gpu_execution_authenticated=False,
                    capture_origin_authenticated=False, external_unit_retirement_authenticated=False,
                    qualification=dict(stages.FLAGS))
