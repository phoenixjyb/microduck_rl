"""Fresh three-module cost-caller executable preparation, never admission.

The owner pins the runtime, owns a fresh process, lease, caps and disassembler.
No CUDA package is imported here. Retained compile inputs are not driver bytes.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import FunctionType, MappingProxyType

from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_cost_dispatch as dispatch
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_executable as retained
from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-caller-cost-executable-oct9-v1"
GROUPS = {"shared": ("init_cost", "dense"), "efc": ("efc",), "gauss": ("gauss",)}
need = static.need


def _same_map(actual, expected):
    return type(actual) is dict and set(actual) == set(expected) and all(actual[k] is v for k, v in expected.items())


def verify_kernel_graph(solver, util, kernels):
    """Authenticate loaded frontend code without compile/load/launch."""
    root = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src"
    path, upath = (root / "solver.py").resolve(strict=True), (root / "warp_util.py").resolve(strict=True)
    need(Path(solver.__file__).resolve(strict=True) == path and Path(util.__file__).resolve(strict=True) == upath,
         "frozen installed frontend paths")
    raw, uraw = path.read_bytes(), upath.read_bytes()
    static.verify_source(raw)
    need(sha256(uraw).hexdigest() == dispatch.UTIL_SHA256
         and sys.modules.get(static.MODULE) is solver and sys.modules.get(dispatch.UTIL_MODULE) is util,
         "frozen cache utility and module identities")
    caller = solver._update_constraint
    need(type(caller) is FunctionType and Path(caller.__code__.co_filename).resolve(strict=True) == path,
         "frozen Python caller filename")
    codes = dispatch._codes(raw, caller.__code__.co_filename)
    need(caller.__code__ == codes["_update_constraint"] and caller.__globals__ is solver.__dict__
         and caller.__defaults__ == (False,) and caller.__defaults__[0] is False
         and caller.__kwdefaults__ is None and caller.__closure__ is None, "frozen caller body/defaults")
    wrappers = (solver.update_constraint_efc, solver.update_constraint_gauss_cost)
    for w, name in zip(wrappers, (dispatch.NAMES[1], dispatch.NAMES[3])):
        need(type(w) is FunctionType and Path(w.__code__.co_filename).resolve(strict=True) == upath,
             "frozen factory wrapper filename")
        ucodes = dispatch._codes(uraw, w.__code__.co_filename)
        f = w.__wrapped__
        need(w.__code__ == dispatch._nested(ucodes["cache_kernel"], "wrapper")
             and w.__globals__ is util.__dict__ and w.__defaults__ is None and w.__kwdefaults__ is None
             and type(f) is FunctionType and dispatch._closure(w) == (("func", f),)
             and f.__code__ == codes[name] and f.__globals__ is solver.__dict__ and f.__name__ == name
             and f.__defaults__ is None and f.__kwdefaults__ is None and f.__closure__ is None,
             "frozen cache wrappers and original factory bodies")
    if kernels is None:
        return
    need(type(kernels) is dict and set(kernels) == set(stages.STAGES), "complete four kernel graph")
    expected = (codes[dispatch.NAMES[0]], dispatch._nested(codes[dispatch.NAMES[1]], "kernel"),
                codes[dispatch.NAMES[2]], dispatch._nested(codes[dispatch.NAMES[3]], "kernel"))
    closures = ((), (("TRACK_CHANGES", False),), (), (("dofs_per_thread", 50), ("nv", 20)))
    for stage, code, values in zip(stages.STAGES, expected, closures):
        f = kernels[stage].func
        actual = dispatch._closure(f)
        need(type(f) is FunctionType and f.__code__ == code and f.__globals__ is solver.__dict__
             and f.__defaults__ is None and f.__kwdefaults__ is None and len(actual) == len(values)
             and all(n == en and type(v) is type(ev) and v == ev for (n, v), (en, ev) in zip(actual, values)),
             "frozen four kernel bodies and literal specialization cells")
    need(solver.update_constraint_init_cost is kernels["init_cost"]
         and getattr(solver, static.TARGET) is kernels["dense"] and type(util._KERNEL_CACHE) is dict
         and util._KERNEL_CACHE.get((hash(False), hash(dispatch.NAMES[1]))) is kernels["efc"]
         and util._KERNEL_CACHE.get((hash(20), hash(50), hash(dispatch.NAMES[3]))) is kernels["gauss"],
         "four frontend objects and exact specialized cache entries")


def materialize_kernels(solver, util):
    """Create only the two declared factory specializations in a fresh child."""
    verify_kernel_graph(solver, util, None)
    keys = ((hash(False), hash(dispatch.NAMES[1])), (hash(20), hash(50), hash(dispatch.NAMES[3])))
    need(type(util._KERNEL_CACHE) is dict and all(k not in util._KERNEL_CACHE for k in keys),
         "absent declared factory entries before materialization")
    kernels = dict(init_cost=solver.update_constraint_init_cost, dense=getattr(solver, static.TARGET),
                   efc=solver.update_constraint_efc(False), gauss=solver.update_constraint_gauss_cost(20, 50))
    verify_kernel_graph(solver, util, kernels)
    need(kernels["init_cost"].module is kernels["dense"].module
         and len({id(k.module) for k in kernels.values()}) == 3, "three literal module groups")
    return kernels


class CostModuleExecutable:
    """One compile and explicit load; shared module binds both entry points."""
    __slots__ = ("wp", "context", "device", "directory", "role", "kernels", "module", "graph_guard",
                 "entries", "symbols", "functions", "codes", "adjoints", "module_hash", "options",
                 "config", "kernel_options", "cache", "baseline", "failures", "failed_baseline",
                 "device_identity", "binary_path", "meta_path", "source_path", "generated", "artifact",
                 "disassembly", "bindings", "executable", "state", "compile_used", "load_used", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False): raise AttributeError("cost executable bindings are sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False): raise AttributeError("cost executable bindings are sealed")
        object.__delattr__(self, name)

    def __init__(self, *, wp, context, device, directory, role, kernels, graph_guard):
        need(role in GROUPS and type(kernels) is dict and tuple(kernels) == GROUPS[role]
             and callable(graph_guard) and getattr(graph_guard, "__code__", None) is not None, "literal module group")
        need(type(directory) is type(Path()) and directory.is_absolute() and directory.name == "compiled-" + role
             and directory.parent.resolve(strict=True) == directory.parent and directory.parent.is_dir()
             and directory.resolve(strict=False) == directory and not directory.exists(), "fresh absent canonical module directory")
        self.wp, self.context, self.device, self.directory, self.role = wp, context, device, directory, role
        self.kernels, self.graph_guard = MappingProxyType(dict(kernels)), graph_guard
        self.module = next(iter(kernels.values())).module
        need(all(k.module is self.module for k in kernels.values()), "same actual owning module")
        self.cache, self.failures = self.module.execs, self.module.failed_builds
        self.baseline, self.failed_baseline = MappingProxyType(dict(self.cache)), frozenset(self.failures)
        artifacts.assert_fresh_module(self.module, device, 256)
        self.module_hash = context.Module.get_module_hash(self.module, 256)
        need(type(self.module_hash) is bytes and len(self.module_hash) == 32, "whole module hash")
        self.symbols = MappingProxyType({n: k.get_mangled_name() + "_cuda_kernel_forward" for n, k in kernels.items()})
        self.functions = tuple(k.func for k in kernels.values())
        self.codes, self.adjoints = tuple(f.__code__ for f in self.functions), tuple(k.adj for k in kernels.values())
        self.options = retained._json_snapshot(self.module.options, "module options")[0]
        self.kernel_options = tuple(retained._json_snapshot(k.options, "kernel options")[0] for k in kernels.values())
        self.config = retained._json_snapshot({n: getattr(wp.config, n) for n in retained._CONFIG_KEYS}, "compiler config")[0]
        self.device_identity = (str(device), device.arch, device.context)
        rows = [(context.Module, n) for n in ("_compile", "load", "get_module_hash")]
        rows += [(context.ModuleExec, "get_kernel_hooks"), (context.ModuleBuilder, "codegen"),
                 (wp._src.build, "build_cuda"), (wp._src.build, "load_cuda"), (wp, "get_device")]
        rows += [(self.module, n) for n in ("get_module_hash", "_get_compile_arch", "_get_compile_output_name",
                                           "_get_meta_name", "get_module_identifier")]
        rows += [(k, "get_mangled_name") for k in kernels.values()]
        rows += [(self, "graph_guard")]
        self.entries = tuple((o, n, getattr(o, n), getattr(o, n).__code__) for o, n in rows)
        binary_name = self.module._get_compile_output_name(device, output_arch=120, use_ptx=False)
        names = (binary_name, self.module._get_meta_name(), self.module.get_module_identifier() + ".cu")
        need(len(set(names)) == 3 and all(type(n) is str and Path(n).name == n and n not in (".", "..")
             and "/" not in n and "\\" not in n for n in names) and binary_name.endswith(".cubin"), "literal generated names")
        self.binary_path, self.meta_path, self.source_path = (directory / n for n in names)
        self.generated = self.artifact = self.disassembly = self.executable = None
        self.bindings = MappingProxyType({})
        self.state, self.compile_used, self.load_used = "new", False, False
        self._sealed = True
        self.check(fresh=True)

    def _files(self):
        need(self.directory.resolve(strict=True) == self.directory and self.directory.is_dir()
             and {p.name for p in self.directory.iterdir()} == {self.binary_path.name, self.meta_path.name, self.source_path.name},
             "exact compiled module file set")
        rows = {}
        for name, path, cap in (("binary", self.binary_path, retained.MAX_BINARY_BYTES),
                                ("metadata", self.meta_path, retained.MAX_META_BYTES),
                                ("source", self.source_path, retained.MAX_SOURCE_BYTES)):
            raw, identity = retained._read_plain(path, cap, name)
            rows[name] = retained._file_record(path, raw, identity)
        if self.generated is not None:
            need(json.dumps(rows, sort_keys=True) == self.generated, "unchanged generated bytes and identities")
        return rows

    def check(self, *, fresh=False):
        self.graph_guard()
        m, d, wp = self.module, self.device, self.wp
        need(all(retained._same_entry(o, n, f, c) for o, n, f, c in self.entries), "held compiler/loader/runtime entries")
        need(wp.get_device() is d and d.is_cuda is True and d.is_capturing is False and d.arch == 120
             and type(d.arch) is int and type(d.context) is int and d.context > 0
             and (str(d), d.arch, d.context) == self.device_identity and self.context.runtime.tape is None,
             "same eager sm120 context without tape")
        need(retained._json_snapshot(m.options, "options")[0] == self.options
             and retained._json_snapshot({n: getattr(wp.config, n) for n in retained._CONFIG_KEYS}, "config")[0] == self.config
             and m.get_module_hash(256) == self.module_hash, "same module/config/frontend hash")
        for (n, k), f, code, adj, opts in zip(self.kernels.items(), self.functions, self.codes, self.adjoints, self.kernel_options):
            need(k.module is m and k.func is f and f.__code__ is code and k.adj is adj
                 and retained._json_snapshot(k.options, "kernel options")[0] == opts
                 and (m.options | k.options).get("enable_backward") is False
                 and k.get_mangled_name() + "_cuda_kernel_forward" == self.symbols[n], "same module entry points")
        need(m.execs is self.cache and m.failed_builds is self.failures
             and frozenset(self.failures) == self.failed_baseline, "same executable cache objects/failures")
        if fresh or self.state in ("new", "compiling", "compiled", "loading"):
            need(_same_map(self.cache, self.baseline), "unchanged fresh executable cache")
            artifacts.assert_fresh_module(m, d, 256)
        elif self.state == "loaded":
            need(_same_map(self.cache, dict(self.baseline) | {(d.context, 256): self.executable}), "exact explicit-load cache delta")
            for b in self.bindings.values(): b.assert_unchanged()
        need(self.state != "failed", "failed executable cannot be reused")

    def compile(self):
        need(not self.compile_used and self.state == "new", "one-shot module compile")
        object.__setattr__(self, "compile_used", True)
        try:
            self.check(fresh=True)
            need(not self.directory.exists() and self.module._get_compile_arch(self.device) == 120, "fresh sm120 compile destination")
            object.__setattr__(self, "state", "compiling")
            fn = next(f for o, n, f, c in self.entries if o is self.context.Module and n == "_compile")
            need(fn(self.module, self.device, self.directory, self.binary_path.name, 120, False) is True, "fresh actual CUBIN compilation")
            self.check(fresh=True)
            rows = self._files()
            object.__setattr__(self, "generated", json.dumps(rows, sort_keys=True))
            b, meta = rows["binary"], rows["metadata"]
            object.__setattr__(self, "artifact", artifacts.retain_artifact(self.binary_path, self.meta_path,
                binary_size=b["bytes"], binary_sha256=b["sha256"], metadata_size=meta["bytes"], metadata_sha256=meta["sha256"]))
            from mjlab_microduck import stance_solver_cost_binary as binary
            raw = retained._read_plain(self.binary_path, retained.MAX_BINARY_BYTES, "cubin")[0]
            for n, symbol in self.symbols.items(): binary.select_cubin(raw, n, symbol)
            object.__setattr__(self, "state", "compiled")
        except BaseException:
            object.__setattr__(self, "state", "failed")
            raise

    def load(self, sass):
        need(not self.load_used, "one-shot module load")
        object.__setattr__(self, "load_used", True)
        try:
            need(self.state == "compiled" and type(sass) is dict and set(sass) == set(self.kernels), "complete per-entry SASS before load")
            self.check(fresh=True)
            self._files()
            from mjlab_microduck import stance_solver_cost_binary as binary
            raw = retained._read_plain(self.binary_path, retained.MAX_BINARY_BYTES, "cubin")[0]
            checks = {n: binary.verify_disassembly(raw, sass[n], n, symbol) for n, symbol in self.symbols.items()}
            object.__setattr__(self, "disassembly", json.dumps(checks, sort_keys=True))
            self.artifact.assert_unchanged()
            self.check(fresh=True)
            self._files()
            object.__setattr__(self, "state", "loading")
            fn = next(f for o, n, f, c in self.entries if o is self.context.Module and n == "load")
            result = fn(self.module, self.device, block_dim=256, binary_path=str(self.binary_path), output_arch=120, meta_path=str(self.meta_path))
            need(type(result) is self.context.ModuleExec
                 and _same_map(self.cache, dict(self.baseline) | {(self.device.context, 256): result}), "new exact loaded ModuleExec cache entry")
            hook_fn = next(f for o, n, f, c in self.entries if o is self.context.ModuleExec and n == "get_kernel_hooks")
            bindings = {n: artifacts.bind_loaded_module(self.artifact, k, self.device, result, hook_fn(result, k),
                        block_dim=256, expected_module_hash=self.module_hash) for n, k in self.kernels.items()}
            object.__setattr__(self, "bindings", MappingProxyType(bindings))
            object.__setattr__(self, "executable", result)
            object.__setattr__(self, "state", "loaded")
            self.check()
            self._files()
            return dict(self.bindings)
        except BaseException:
            object.__setattr__(self, "state", "failed")
            raise

    def record(self):
        need(self.state == "loaded", "complete explicit module load only")
        self.check()
        self._files()
        return dict(protocol=PROTOCOL, role=self.role, generated=json.loads(self.generated),
                    module_hash=self.module_hash.hex(), offline_disassembly=json.loads(self.disassembly),
                    explicit_load=dict(binary_path=str(self.binary_path), metadata_path=str(self.meta_path),
                        output_arch=120, block_dim=256, returned_executable_is_cache_entry=True,
                        module_object_id=id(self.module), executable_object_id=id(self.executable), device_object_id=id(self.device)),
                    bindings={n: b.record() for n, b in self.bindings.items()}, flags=dict(stages.FLAGS),
                    loaded_binary_bytes_observed=False, driver_jit_machine_code_observed=False, actual_dispatch_observed=False)
