"""Fresh two-entry gradient executable preparation, never native admission.

No CUDA package is imported. The future owner must authenticate pristine runtime
APIs before construction and independently prove input origin and retirement.
"""
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys
from types import FunctionType, MappingProxyType

from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_gradient_dispatch as adapter
from mjlab_microduck import stance_solver_gradient_binary as binary
from mjlab_microduck import stance_solver_executable as retained
from mjlab_microduck import stance_solver_cost_executable as helpers

PROTOCOL = "microduck-gradient-executable-oct9-v1"
BASE = "0fd6a33fba5feca9849e6968569c1c76b9f6588d"
OWN = frozenset({
    "src/mjlab_microduck/stance_solver_gradient_binary.py",
    "src/mjlab_microduck/stance_solver_gradient_executable.py",
    "tests/test_stance_solver_gradient_binary.py",
    "tests/test_stance_solver_gradient_executable.py",
    "docs/experiments/2026-10-09-gradient-executable-preparation.md",
})
TESTS = adapter.TESTS + ("tests/test_stance_solver_gradient_binary.py",
                         "tests/test_stance_solver_gradient_executable.py")
need = adapter.need
_same_map = helpers._same_map


def source_binding(root):
    """Separate exact five-path fence; previous contracts stay immutable."""
    cmd = adapter.prefix.prior.retained.command
    need(type(root) is type(Path.cwd()) and root == root.resolve(strict=True)
         and Path.cwd().resolve() == root
         and cmd("git", "branch", "--show-current").decode().strip() == adapter.prefix.prior.BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact gradient executable branch")
    source = cmd("git", "rev-parse", "HEAD").decode().strip()
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN,
         "separate exact five-path gradient executable fence")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed executable leaf")
        raw = adapter.prefix.prior.read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
             "whole committed executable blob")
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(commit=source, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(),
                base=BASE, leaves=leaves)


def verify_kernel_graph(solver, wp, context, kernels):
    """Authenticate original top-level frontend code, without compile/load/launch."""
    path = (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)
    need(Path(solver.__file__).resolve(strict=True) == path
         and sys.modules.get(adapter.prefix.static.MODULE) is solver
         and solver.__name__ == adapter.prefix.static.MODULE and solver.wp is wp,
         "frozen installed gradient module path and namespace")
    raw = path.read_bytes()
    adapter.prefix.verify_source(raw)
    codes = adapter.cost._codes(raw, str(path))
    caller = solver._update_gradient
    need(type(caller) is FunctionType and caller.__code__ == codes["_update_gradient"]
         and Path(caller.__code__.co_filename).resolve(strict=True) == path
         and caller.__globals__ is solver.__dict__ and caller.__defaults__ is None
         and caller.__kwdefaults__ is None and caller.__closure__ is None,
         "frozen original gradient caller body/globals/defaults")
    need(type(kernels) is dict and tuple(kernels) == adapter.prefix.STAGES,
         "literal ordered two-entry gradient graph")
    for stage, name in zip(adapter.prefix.STAGES, adapter.NAMES):
        k, f = kernels[stage], kernels[stage].func
        need(type(k) is context.Kernel and getattr(solver, name) is k
             and type(f) is FunctionType and f.__code__ == codes[name]
             and Path(f.__code__.co_filename).resolve(strict=True) == path
             and f.__globals__ is solver.__dict__ and f.__defaults__ is None
             and f.__kwdefaults__ is None and f.__closure__ is None,
             "original top-level gradient kernel object/code/globals/defaults")
    need(len({id(k.module) for k in kernels.values()}) == 1, "one original owning gradient module")


def identify_kernels(solver, wp, context):
    """Identify existing top-level kernels; no factory or cache specialization."""
    kernels = {s: getattr(solver, n) for s, n in zip(adapter.prefix.STAGES, adapter.NAMES)}
    verify_kernel_graph(solver, wp, context, kernels)
    return kernels


class GradientModuleExecutable:
    """One fresh compile and explicit load of the two original gradient entries."""
    __slots__ = ("solver", "caller", "types", "context_types", "wp", "context", "device", "directory", "role", "kernels", "module", "graph_guard",
                 "entries", "symbols", "functions", "codes", "adjoints", "module_hash", "options",
                 "config", "kernel_options", "cache", "baseline", "failures", "failed_baseline",
                 "device_identity", "binary_path", "meta_path", "source_path", "generated", "artifact",
                 "disassembly", "bindings", "executable", "state", "compile_used", "load_used", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False): raise AttributeError("gradient executable bindings are sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False): raise AttributeError("gradient executable bindings are sealed")
        object.__delattr__(self, name)

    def __init__(self, *, solver, wp, context, device, directory):
        kernels = identify_kernels(solver, wp, context)
        role = "gradient"
        self.solver, self.caller, self.types = solver, solver._update_gradient, solver.types
        self.context_types = (context.Kernel, context.Module, context.ModuleExec, context.ModuleBuilder)
        graph_guard = self._frontend
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
        need(all(type(symbol) is str and binary.SYMBOLS[n].fullmatch(symbol)
                 for n, symbol in self.symbols.items()), "literal original gradient forward symbols")
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

    def _frontend(self):
        need(self.solver._update_gradient is self.caller and self.solver.types is self.types
             and all(actual is held for actual, held in zip(
                 (self.context.Kernel, self.context.Module, self.context.ModuleExec, self.context.ModuleBuilder),
                 self.context_types)), "held gradient caller/types and runtime classes")
        verify_kernel_graph(self.solver, self.wp, self.context, dict(self.kernels))

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
                    bindings={n: b.record() for n, b in self.bindings.items()}, flags=dict(adapter.prefix.FLAGS),
                    loaded_binary_bytes_observed=False, driver_jit_machine_code_observed=False, actual_dispatch_observed=False)
