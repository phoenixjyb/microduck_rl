"""Retain and explicitly load the frozen dense solver target executable.

The caller owns CUDA initialization, environment and runtime authentication,
the GPU lease, resource limits, disassembler invocation, and dispatch capture.
This class binds generated compile inputs and observed Warp module objects; it
does not observe driver-loaded machine code, dispatch, numerical cause, or
qualification. Importing it initializes no device and imports no CUDA package.
"""

from hashlib import sha256
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import sys
from types import MethodType

from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_disassembly_scope as disassembly
from mjlab_microduck import stance_solver_target_binding as target

PROTOCOL = "microduck-dense-solver-executable-oct8-v1"
BLOCK_DIM = 256
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_BINARY_BYTES = artifacts.MAX_BINARY_BYTES
MAX_META_BYTES = artifacts.MAX_META_BYTES
_CONFIG_KEYS = (
    "mode",
    "optimization_level",
    "verify_fp",
    "llvm_cuda",
    "cache_kernels",
    "verify_autograd_array_access",
    "use_precompiled_headers",
)
FLAGS = {
    "native_qualified": False,
    "full_window_qualified": False,
    "runtime_cause_proven": False,
    "training_authorized": False,
    "physical_acceptance": False,
}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _code(function, label):
    value = getattr(function, "__code__", None)
    _need(value is not None, label + " Python code object")
    return value


def _same_entry(owner, name, function, code):
    current = getattr(owner, name)
    if current is function:
        return _code(current, name) is code
    return (isinstance(current, MethodType) and isinstance(function, MethodType)
            and current.__func__ is function.__func__
            and current.__self__ is function.__self__
            and _code(current, name) is code)


def _json_snapshot(value, label):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("plain JSON snapshot " + label) from error
    return raw, result


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_plain(path, cap, label):
    _need(type(path) is type(Path()) and path.is_absolute(), "absolute plain path " + label)
    _need(path.resolve(strict=True) == path, "canonical non-symlink path " + label)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode), "regular file " + label)
        _need(0 < before.st_size <= cap, "bounded file size " + label)
        chunks = []
        remaining = cap + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        _need(
            _identity(before) == _identity(after) == _identity(current)
            and stat.S_ISREG(current.st_mode)
            and path.resolve(strict=True) == path,
            "stable regular file identity " + label,
        )
    finally:
        os.close(fd)
    _need(len(raw) == before.st_size and len(raw) <= cap, "whole bounded file " + label)
    return raw, _identity(before)


def _file_record(path, raw, identity):
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "identity": list(identity),
    }


class DenseSolverExecutable:
    """One-shot compile, disassembly bind, and explicit CUBIN load for the target."""

    def __init__(self, wp, context, solver, device, directory):
        self.wp, self.context, self.solver, self.device = wp, context, solver, device
        self.directory = directory
        self.state = "new"
        self.consumed = False
        self.load_consumed = False
        self.binding = None
        self.compile_record = None
        self.loaded_cache_snapshot = None
        self.disassembly = None
        self.artifact = None
        self.generated = None
        self.explicit_load = None

        _need(type(directory) is type(Path()) and directory.is_absolute(),
              "absolute plain compiled directory")
        _need(directory.name == "compiled-dense-solver", "literal compiled directory name")
        parent = directory.parent
        _need(parent.is_dir() and parent.resolve(strict=True) == parent,
              "existing canonical directory parent")
        _need(directory.resolve(strict=False) == directory and not directory.exists(),
              "fresh absent compiled directory")

        self.path = (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py")
        self.path = self.path.resolve(strict=True)
        _need(Path(solver.__file__).resolve(strict=True) == self.path,
              "installed frozen solver path")
        source_raw, self.source_identity = _read_plain(self.path, target.MAX_SOURCE_BYTES,
                                                        "installed solver source")
        self.source = target.verify_source(source_raw)
        self.source_sha256 = sha256(source_raw).hexdigest()
        self.caller = solver._update_constraint
        self.kernel = getattr(solver, target.TARGET)
        self.caller_code = _code(self.caller, "solver caller")
        self.kernel_code = _code(self.kernel.func, "solver target kernel")
        self.kernel_adjoint = self.kernel.adj
        self.symbol_fn = self.kernel.get_mangled_name
        self.symbol_code = _code(self.symbol_fn, "target symbol")
        self.symbol = self.symbol_fn() + "_cuda_kernel_forward"
        expected = compile(source_raw, self.caller_code.co_filename, "exec",
                           dont_inherit=True, optimize=0)
        expected_codes = {code.co_name: code for code in expected.co_consts
                          if hasattr(code, "co_name")}
        _need(self.caller_code == expected_codes.get("_update_constraint"),
              "loaded frozen solver caller code")
        expected_kernel = compile(source_raw, self.kernel_code.co_filename, "exec",
                                  dont_inherit=True, optimize=0)
        kernel_codes = {code.co_name: code for code in expected_kernel.co_consts
                        if hasattr(code, "co_name")}
        _need(self.kernel_code == kernel_codes.get(target.TARGET),
              "loaded frozen dense target code")
        _need(Path(self.caller_code.co_filename).resolve(strict=True) == self.path
              and Path(self.kernel_code.co_filename).resolve(strict=True) == self.path,
              "loaded solver code filenames resolve to pinned path")

        self.module = self.kernel.module
        self.compile_fn = context.Module._compile
        self.load_fn = context.Module.load
        self.hash_fn = context.Module.get_module_hash
        self.module_hash_access = self.module.get_module_hash
        self.hooks_fn = context.ModuleExec.get_kernel_hooks
        self.codegen_fn = context.ModuleBuilder.codegen
        self.compile_arch_fn = self.module._get_compile_arch
        self.output_name_fn = self.module._get_compile_output_name
        self.meta_name_fn = self.module._get_meta_name
        self.identifier_fn = self.module.get_module_identifier
        self.build_cuda_fn = wp._src.build.build_cuda
        self.load_cuda_fn = wp._src.build.load_cuda
        self.get_device_fn = wp.get_device
        self.entrypoints = {
            name: (fn, _code(fn, name))
            for name, fn in (
                ("compile", self.compile_fn),
                ("load", self.load_fn),
                ("module_hash", self.hash_fn),
                ("module_hash_access", self.module_hash_access),
                ("kernel_hooks", self.hooks_fn),
                ("source_codegen", self.codegen_fn),
                ("compile_arch", self.compile_arch_fn),
                ("output_name", self.output_name_fn),
                ("metadata_name", self.meta_name_fn),
                ("module_identifier", self.identifier_fn),
                ("runtime_build_cuda", self.build_cuda_fn),
                ("runtime_load_cuda", self.load_cuda_fn),
                ("get_device", self.get_device_fn),
                ("target_symbol", self.symbol_fn),
            )
        }
        self.module_hash = self.hash_fn(self.module, BLOCK_DIM)
        _need(type(self.module_hash) is bytes and len(self.module_hash) == 32,
              "plain SHA256 module hash")
        self.module_options_raw, self.module_options = _json_snapshot(self.module.options,
                                                                        "module options")
        self.kernel_options_raw, self.kernel_options = _json_snapshot(self.kernel.options,
                                                                        "kernel options")
        self.config_raw, self.config = _json_snapshot(
            {key: getattr(wp.config, key) for key in _CONFIG_KEYS}, "Warp compiler config"
        )
        self.device_context = device.context
        self.device_arch = device.arch
        self.device_alias = str(device)
        self.cache_object = self.module.execs
        self.failed_builds_object = self.module.failed_builds
        self.cache_baseline = dict(self.module.execs)
        self.failed_baseline = set(self.module.failed_builds)
        self.binary_name = self.module._get_compile_output_name(
            device, output_arch=self.device_arch, use_ptx=False
        )
        self.meta_name = self.module._get_meta_name()
        self.module_identifier = self.module.get_module_identifier()
        generated_names = (self.binary_name, self.meta_name, self.module_identifier + ".cu")
        _need(all(type(name) is str and name and Path(name).name == name
                  and "/" not in name and "\\" not in name and name not in (".", "..")
                  for name in generated_names)
              and self.binary_name.endswith(".cubin")
              and self.meta_name.endswith(".meta")
              and len(set(generated_names)) == 3,
              "plain distinct generated CUBIN, metadata, and source filenames")
        self.binary_path = directory / self.binary_name
        self.meta_path = directory / self.meta_name
        self.source_path = directory / (self.module_identifier + ".cu")
        self.runtime = context.runtime
        self.tape = self.runtime.tape
        self._guard(precompile=True)

    def _guard(self, *, precompile=False):
        s, wp, d, m = self.solver, self.wp, self.device, self.module
        _need(s.__name__ == target.MODULE and sys.modules.get(target.MODULE) is s
              and s.wp is wp
              and Path(s.__file__).resolve(strict=True) == self.path,
              "same installed solver module and path")
        source_raw, source_identity = _read_plain(
            self.path, target.MAX_SOURCE_BYTES, "installed solver source"
        )
        _need(sha256(source_raw).hexdigest() == self.source_sha256
              and source_identity == self.source_identity,
              "same frozen solver source bytes and identity")
        _need(s._update_constraint is self.caller and self.caller.__code__ is self.caller_code
              and self.caller.__globals__ is s.__dict__
              and self.caller.__defaults__ == (False,)
              and self.caller.__defaults__[0] is False and self.caller.__kwdefaults__ is None
              and getattr(s, target.TARGET) is self.kernel
              and self.kernel.module is m and self.kernel.func.__code__ is self.kernel_code
              and self.kernel.func.__globals__ is s.__dict__
              and self.kernel.adj is self.kernel_adjoint
              and self.symbol_fn() + "_cuda_kernel_forward" == self.symbol
              and self.kernel.func.__defaults__ is None and self.kernel.func.__kwdefaults__ is None,
              "same frozen caller, target, and owning module")
        _need(self.wp.get_device is self.get_device_fn
              and self.get_device_fn() is d and d.is_cuda is True
              and d.is_capturing is False and type(d.arch) is int and d.arch == 120
              and type(d.context) is int and d.context > 0
              and d.context == self.device_context and str(d) == self.device_alias,
              "same current eager sm120 CUDA device/context")
        _need(self.runtime is self.context.runtime and self.runtime.tape is None
              and self.tape is None, "no active Warp tape")
        _need(m is self.kernel.module and type(m.options) is dict
              and m.options.get("block_dim") == BLOCK_DIM
              and type(m.options.get("block_dim")) is int
              and m.options.get("strip_hash") is False
              and (m.options | self.kernel.options).get("enable_backward") is False,
              "literal target module options with backward disabled")
        _need(self.hash_fn(m, BLOCK_DIM) == self.module_hash
              and _same_entry(m, "get_module_hash", self.module_hash_access,
                              self.entrypoints["module_hash_access"][1])
              and self.module_hash_access(BLOCK_DIM) == self.module_hash,
              "same source/options module hash")
        _need(_json_snapshot(m.options, "module options")[0] == self.module_options_raw
              and _json_snapshot(self.kernel.options, "kernel options")[0] == self.kernel_options_raw,
              "unchanged module and target options")
        _need(_json_snapshot({key: getattr(wp.config, key) for key in _CONFIG_KEYS},
                             "Warp compiler config")[0] == self.config_raw,
              "unchanged Warp compiler config")
        _need(all(_same_entry(owner, name, fn, code)
                  for (owner, name), (fn, code) in self._entry_owner_rows()),
              "unchanged compiler, loader, hashing, hooks and runtime source entries")
        _need(m.execs is self.cache_object and m.failed_builds is self.failed_builds_object,
              "same module cache objects")
        if precompile:
            _need(m.execs == self.cache_baseline and d.context not in m.failed_builds
                  and not any(type(key) is tuple and key and key[0] == d.context
                              for key in m.execs),
                  "fresh target module cache in current context")
            _need(self.module_options.get("block_dim") == BLOCK_DIM
                  and self.module_options.get("strip_hash") is False,
                  "source hashed block 256 module")
        elif self.state in ("compiling", "compiled"):
            _need(m.execs == self.cache_baseline and m.failed_builds == self.failed_baseline,
                  "compile did not load or poison executable cache")
        elif self.state in ("loading", "loaded"):
            _need(m.failed_builds == self.failed_baseline,
                  "load did not poison executable cache")
            if self.state == "loaded":
                _need(self.loaded_cache_snapshot is not None
                      and m.execs == self.loaded_cache_snapshot,
                      "same cache after bound explicit load")

    def _entry_owner_rows(self):
        return (
            ((self.context.Module, "_compile"), self.entrypoints["compile"]),
            ((self.context.Module, "load"), self.entrypoints["load"]),
            ((self.context.Module, "get_module_hash"), self.entrypoints["module_hash"]),
            ((self.module, "get_module_hash"), self.entrypoints["module_hash_access"]),
            ((self.context.ModuleExec, "get_kernel_hooks"), self.entrypoints["kernel_hooks"]),
            ((self.context.ModuleBuilder, "codegen"), self.entrypoints["source_codegen"]),
            ((self.module, "_get_compile_arch"), self.entrypoints["compile_arch"]),
            ((self.module, "_get_compile_output_name"), self.entrypoints["output_name"]),
            ((self.module, "_get_meta_name"), self.entrypoints["metadata_name"]),
            ((self.module, "get_module_identifier"), self.entrypoints["module_identifier"]),
            ((self.wp._src.build, "build_cuda"), self.entrypoints["runtime_build_cuda"]),
            ((self.wp._src.build, "load_cuda"), self.entrypoints["runtime_load_cuda"]),
            ((self.wp, "get_device"), self.entrypoints["get_device"]),
            ((self.kernel, "get_mangled_name"), self.entrypoints["target_symbol"]),
        )

    def _directory_guard(self):
        _need(self.directory.resolve(strict=True) == self.directory
              and self.directory.is_dir(), "same compiled output directory")
        names = {path.name for path in self.directory.iterdir()}
        _need(names == {self.binary_name, self.meta_name, self.source_path.name},
              "exact generated source, metadata, and CUBIN set")
        self._assert_generated_unchanged()

    def _assert_generated_unchanged(self):
        if self.generated is None:
            return
        for name, path, cap in (
            ("source", self.source_path, MAX_SOURCE_BYTES),
            ("metadata", self.meta_path, MAX_META_BYTES),
            ("binary", self.binary_path, MAX_BINARY_BYTES),
        ):
            raw, identity = _read_plain(path, cap, "generated " + name)
            row = self.generated[name]
            _need(len(raw) == row["bytes"] and sha256(raw).hexdigest() == row["sha256"]
                  and identity == tuple(row["identity"]), "unchanged generated " + name)

    def compile(self):
        _need(not self.consumed and self.state == "new", "one-shot compile")
        self.consumed = True
        try:
            self._guard(precompile=True)
            arch = self.module._get_compile_arch(self.device)
            _need(type(arch) is int and arch == self.device_arch == 120,
                  "literal sm120 compile architecture")
            self._need_fresh_directory()
            self.state = "compiling"
            compiled = self.compile_fn(
                self.module, self.device, self.directory, self.binary_name, arch, False
            )
            _need(compiled is True, "fresh CUBIN compilation performed")
            self._guard(precompile=False)
            self._directory_guard()
            rows, raw_files = {}, {}
            for key, path, cap in (
                ("source", self.source_path, MAX_SOURCE_BYTES),
                ("metadata", self.meta_path, MAX_META_BYTES),
                ("binary", self.binary_path, MAX_BINARY_BYTES),
            ):
                raw, identity = _read_plain(path, cap, "generated " + key)
                raw_files[key] = raw
                rows[key] = _file_record(path, raw, identity)
            _need(self.binary_path.suffix == ".cubin", "literal target CUBIN output")
            self.generated = rows
            self.artifact = artifacts.retain_artifact(
                self.binary_path, self.meta_path,
                binary_size=rows["binary"]["bytes"],
                binary_sha256=rows["binary"]["sha256"],
                metadata_size=rows["metadata"]["bytes"],
                metadata_sha256=rows["metadata"]["sha256"],
            )
            selected = disassembly.select_target_cubin(raw_files["binary"])
            _need(selected["cubin"]["sha256"] == rows["binary"]["sha256"],
                  "target function in retained compile-input CUBIN")
            _need(selected["target"]["symbol"] == self.symbol,
                  "selected CUBIN symbol matches the actual frozen target")
            self.compile_record = {
                "performed": True,
                "output_arch": arch,
                "use_ptx": False,
                "module_hash": self.module_hash.hex(),
                "module_options": self.module_options,
                "kernel_options": self.kernel_options,
                "warp_config": self.config,
                "generated": rows,
                "target_compile_input": selected,
            }
            self.state = "compiled"
            self._guard(precompile=False)
            self._directory_guard()
            return deepcopy(self.compile_record)
        except Exception:
            self.state = "failed"
            raise

    def _need_fresh_directory(self):
        _need(self.directory.parent.is_dir()
              and self.directory.parent.resolve(strict=True) == self.directory.parent
              and not self.directory.exists(), "fresh canonical compiled directory")

    def load(self, sass_stdout):
        _need(not self.load_consumed, "one-shot load")
        self.load_consumed = True
        if not self.consumed or self.state != "compiled":
            self.consumed = True
            self.state = "failed"
            raise ValueError("compile must complete before one-shot load")
        self.state = "loading"
        try:
            self._guard(precompile=False)
            self._directory_guard()
            binary, _ = _read_plain(self.binary_path, MAX_BINARY_BYTES, "compiled CUBIN")
            self.disassembly = disassembly.verify_target_disassembly(binary, sass_stdout)
            _need(self.disassembly["cubin"]["sha256"] == self.generated["binary"]["sha256"],
                  "SASS matches retained generated CUBIN")
            self._guard(precompile=True)
            _need(self.module.execs == self.cache_baseline
                  and not any(type(key) is tuple and key and key[0] == self.device.context
                              for key in self.module.execs),
                  "still fresh target module cache immediately before explicit load")
            _need(self.artifact is not None, "retained CUBIN and metadata anchors")
            self.artifact.assert_unchanged()
            self._assert_generated_unchanged()
            artifacts.assert_fresh_module(self.module, self.device, BLOCK_DIM)
            executable = self.load_fn(
                self.module, self.device, block_dim=BLOCK_DIM,
                binary_path=str(self.binary_path), output_arch=self.device_arch,
                meta_path=str(self.meta_path),
            )
            _need(type(executable) is self.context.ModuleExec
                  and executable is not None
                  and self.module.execs.get((self.device_context, BLOCK_DIM)) is executable,
                  "explicit load returned a new cached ModuleExec")
            _need(self.cache_baseline == {
                key: value for key, value in self.module.execs.items()
                if key != (self.device_context, BLOCK_DIM)
            }, "explicit load preserved other executable cache entries")
            hooks = self.hooks_fn(executable, self.kernel)
            binding = artifacts.bind_loaded_module(
                self.artifact, self.kernel, self.device, executable, hooks,
                block_dim=BLOCK_DIM, expected_module_hash=self.module_hash,
            )
            self.binding = binding
            self.explicit_load = {
                "binary_path": str(self.binary_path),
                "metadata_path": str(self.meta_path),
                "output_arch": self.device_arch,
                "block_dim": BLOCK_DIM,
                "module_object_id": id(self.module),
                "executable_object_id": id(executable),
                "device_object_id": id(self.device),
                "returned_executable_is_cache_entry": True,
            }
            self.loaded_cache_snapshot = dict(self.module.execs)
            self.state = "loaded"
            self._guard(precompile=False)
            self._directory_guard()
            binding.assert_unchanged()
            return binding
        except Exception:
            self.state = "failed"
            raise

    def record(self):
        _need(self.state == "loaded" and self.binding is not None
              and self.compile_record is not None and self.disassembly is not None,
              "complete executable preparation before receipt")
        self._guard(precompile=False)
        self._directory_guard()
        self.binding.assert_unchanged()
        result = {
            "protocol": PROTOCOL,
            "decision": "dense-solver-executable-prepared-not-dispatch-or-qualification",
            "source": {"path": str(self.path), "sha256": self.source_sha256,
                       "identity": list(self.source_identity)},
            "target": target.TARGET,
            "device": {"alias": self.device_alias, "arch": self.device_arch,
                       "context": self.device_context, "object_id": id(self.device)},
            "compile": self.compile_record,
            "offline_disassembly": self.disassembly,
            "loaded_binding": self.binding.record(),
            "explicit_load": self.explicit_load,
            "runtime_entries": {
                name: {"object_id": id(fn), "code_object_id": id(code),
                       "qualname": code.co_qualname}
                for name, (fn, code) in self.entrypoints.items()
            },
            "actual_dispatch_observed": False,
            "loaded_binary_bytes_observed": False,
            "driver_jit_machine_code_observed": False,
            "numerical_cause_proven": False,
            "flags": dict(FLAGS),
        }
        return deepcopy(result)
