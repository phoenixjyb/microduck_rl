"""Synthetic preparation-flow checks only; no Warp load, CUDA, or disassembly."""

from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import textwrap
from types import FunctionType, ModuleType, SimpleNamespace as NS

import pytest

from mjlab_microduck import stance_solver_executable as executable
from mjlab_microduck import stance_solver_target_binding as target


_ENCODING_A = "0123456789abcdef"
_ENCODING_B = "fedcba9876543210"


class FakeDevice:
    is_cuda = True
    is_capturing = False
    context = 9876
    arch = 120
    max_shared_memory_per_block = 65536
    alias = "cuda:0"

    def __str__(self):
        return self.alias


_LIVE_SOLVER_PATH = (
    Path(sys.prefix)
    / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
).resolve(strict=True)
_LIVE_SOLVER_BYTES = _LIVE_SOLVER_PATH.read_bytes()
assert sha256(_LIVE_SOLVER_BYTES).hexdigest() == target.SOLVER_SHA256


@pytest.fixture(autouse=True)
def live_solver_source_unchanged():
    assert _LIVE_SOLVER_PATH.read_bytes() == _LIVE_SOLVER_BYTES
    yield
    assert _LIVE_SOLVER_PATH.read_bytes() == _LIVE_SOLVER_BYTES


def _target_symbol():
    return target.TARGET + "_1234abcd_cuda_kernel_forward"


def _cubin(code):
    section_names = b"\0.shstrtab\0.strtab\0.symtab\0.text\0"
    symbol = _target_symbol()
    strings = b"\0" + symbol.encode("ascii") + b"\0"
    symbol_row = struct.pack(
        "<IBBHQQ", strings.index(symbol.encode("ascii") + b"\0"), 18, 16, 4, 0, len(code)
    )
    data = bytearray(b"\0" * 64)
    sections = [(0,) * 10]
    for name, kind, flags, payload, link, entry in (
        (b".shstrtab", 3, 0, section_names, 0, 0),
        (b".strtab", 3, 0, strings, 0, 0),
        (b".symtab", 2, 0, b"\0" * 24 + symbol_row, 2, 24),
        (b".text", 1, 6, code, 0, 0),
    ):
        sections.append((section_names.index(name + b"\0"), kind, flags, 0, len(data),
                         len(payload), link, 0, 1, entry))
        data.extend(payload)
    section_offset = len(data)
    data.extend(b"".join(struct.pack("<IIQQQQIIQQ", *row) for row in sections))
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, ident, 2, 190, 1, 0, 0,
                     section_offset, 0, 64, 0, 0, 64, 5, 1)
    return bytes(data)


def _sass(code):
    assert len(code) == 16
    first = int.from_bytes(code[:8], "little")
    second = int.from_bytes(code[8:], "little")
    return (
        f"\n\tcode for sm_120\n\t\tFunction : {_target_symbol()}\n"
        '\t.headerflags\t@"EF_CUDA_64BIT_ADDRESS EF_CUDA_SM120 '
        'EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"\n'
        f"/*0000*/ MOV R1, R2 ; /* 0x{first:016x} */\n"
        f"                    /* 0x{second:016x} */\n"
        "\t\t..........\n"
    ).encode("ascii")


@pytest.fixture
def case(tmp_path, monkeypatch):
    device = FakeDevice()
    module_hash = bytes(range(32))
    module = NS(
        execs={}, failed_builds=set(),
        options={"block_dim": 256, "strip_hash": False, "enable_backward": False,
                 "fast_math": False, "fuse_fp": True, "lineinfo": False,
                 "compile_time_trace": False},
        get_module_hash=lambda block: module_hash,
        _get_compile_output_name=lambda dev, output_arch=None, use_ptx=None:
            "solver.sm120.cubin",
        _get_meta_name=lambda: "solver.meta",
        get_module_identifier=lambda: "solver-hash",
        _get_compile_arch=lambda dev: 120,
    )
    kernel = NS(module=module, adj=object(), options={},
                func=NS(__code__=None),
                get_mangled_name=lambda: target.TARGET + "_1234abcd")
    source_raw = _LIVE_SOLVER_BYTES
    prefix = tmp_path / "frozen-prefix"
    solver_path = (
        prefix
        / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
    )
    solver_path.parent.mkdir(parents=True)
    solver_path.write_bytes(source_raw)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    source = compile(source_raw, str(solver_path), "exec", dont_inherit=True, optimize=0)
    solver_codes = {c.co_name: c for c in source.co_consts if hasattr(c, "co_name")}
    from types import FunctionType
    solver = ModuleType(target.MODULE)
    solver.__file__ = str(solver_path)
    solver.wp = NS()
    solver._update_constraint = FunctionType(solver_codes["_update_constraint"], solver.__dict__,
                                              argdefs=(False,))
    kernel.func = FunctionType(solver_codes[target.TARGET], solver.__dict__)
    setattr(solver, target.TARGET, kernel)
    monkeypatch.setitem(sys.modules, target.MODULE, solver)

    events = []
    compile_extras = []
    load_behavior = [None]
    hooks_behavior = [None]
    code = bytes.fromhex(_ENCODING_A + _ENCODING_B)
    raw_cubin = _cubin(code)
    sass = _sass(code)

    def compile_fn(mod, dev, directory, output_name, arch, use_ptx):
        events.append(("compile", mod, dev, directory, output_name, arch, use_ptx))
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "solver-hash.cu").write_bytes(b"// synthetic generated source\n")
        (directory / "solver.meta").write_text(
            json.dumps({_target_symbol() + "_smem_bytes": 0}), encoding="ascii"
        )
        (directory / "solver.sm120.cubin").write_bytes(raw_cubin)
        for extra in compile_extras:
            path = directory / extra
            if extra.endswith("/"):
                path.mkdir()
            else:
                path.write_bytes(b"unexpected")
        return True

    def load_fn(mod, dev, block_dim=None, binary_path=None, output_arch=None, meta_path=None):
        events.append(("load", mod, dev, block_dim, binary_path, output_arch, meta_path))
        if load_behavior[0] == "none":
            return None
        executable_object = ModuleExec()
        executable_object.handle = 98765
        executable_object.device = dev
        executable_object.module_hash = module_hash
        executable_object.kernel_hooks = {}
        executable_object.meta = {_target_symbol() + "_smem_bytes": 0}
        if load_behavior[0] == "wrong-object":
            return NS()
        mod.execs[(dev.context, block_dim)] = executable_object
        if load_behavior[0] == "other-cache-entry":
            mod.execs[(dev.context, block_dim - 128)] = object()
        return executable_object

    def hash_fn(mod, block_dim=None):
        assert mod is module and block_dim == 256
        return module_hash

    def hooks_fn(exec_obj, target_kernel):
        if hooks_behavior[0] == "raise":
            raise RuntimeError("synthetic hook failure")
        if hooks_behavior[0] == "none":
            return None
        if hooks_behavior[0] == "malformed":
            return NS(forward=76543)
        hooks = NS(forward=76543, backward=None, forward_smem_bytes=0, backward_smem_bytes=0)
        exec_obj.kernel_hooks[target_kernel.adj] = hooks
        return hooks

    def codegen_fn(builder, language):
        return "// synthetic codegen"

    class Module:
        _compile = staticmethod(compile_fn)
        load = staticmethod(load_fn)
        get_module_hash = staticmethod(hash_fn)

    class ModuleExec:
        get_kernel_hooks = staticmethod(hooks_fn)

    class ModuleBuilder:
        codegen = codegen_fn

    def build_cuda(*args, **kwargs):
        events.append(("runtime_build_cuda", args, kwargs))

    def load_cuda(*args, **kwargs):
        events.append(("runtime_load_cuda", args, kwargs))

    runtime = NS(tape=None)
    context = NS(Module=Module, ModuleExec=ModuleExec, ModuleBuilder=ModuleBuilder,
                 runtime=runtime)
    wp = NS(config=NS(mode="release", optimization_level=None, verify_fp=False,
                      llvm_cuda=False, cache_kernels=True,
                      verify_autograd_array_access=False, use_precompiled_headers=False),
            _src=NS(build=NS(build_cuda=build_cuda, load_cuda=load_cuda)),
            get_device=lambda: device)
    solver.wp = wp
    parent = tmp_path / "canonical"
    parent.mkdir()
    destination = parent / "compiled-dense-solver"
    collector = executable.DenseSolverExecutable(wp, context, solver, device, destination)
    return NS(collector=collector, device=device, module=module, kernel=kernel,
              context=context, wp=wp, events=events, sass=sass, cubin=raw_cubin,
              destination=destination, source_raw=source_raw, solver_path=solver_path,
              compile_extras=compile_extras, load_behavior=load_behavior,
              hooks_behavior=hooks_behavior)


def test_compile_then_matching_sass_explicit_load_and_record(case):
    compile_record = case.collector.compile()
    assert compile_record["performed"] is True
    assert compile_record["target_compile_input"]["target"]["symbol"] == _target_symbol()
    assert [event[0] for event in case.events] == ["compile"]
    binding = case.collector.load(case.sass)
    assert binding.kernel is case.kernel
    assert [event[0] for event in case.events] == ["compile", "load"]
    assert case.events[1][3:] == (
        256, str(case.destination / "solver.sm120.cubin"), 120,
        str(case.destination / "solver.meta"),
    )
    record = case.collector.record()
    assert record["decision"] == "dense-solver-executable-prepared-not-dispatch-or-qualification"
    assert record["actual_dispatch_observed"] is False
    assert record["loaded_binary_bytes_observed"] is False
    assert record["driver_jit_machine_code_observed"] is False
    assert not any(record["flags"].values())
    assert all(value is False for value in record["loaded_binding"].values()
               if type(value) is bool)


def test_compile_is_one_shot_and_preserves_generated_files_after_failure(case):
    case.collector.compile()
    with pytest.raises(ValueError, match="one-shot"):
        case.collector.compile()
    with pytest.raises(ValueError, match="exactly matches"):
        case.collector.load(_sass(b"X" * 16))
    assert case.destination.exists()
    assert len(list(case.destination.iterdir())) == 3
    with pytest.raises(ValueError):
        case.collector.record()


def test_load_before_compile_consumes_collector(case):
    with pytest.raises(ValueError, match="compile must complete"):
        case.collector.load(case.sass)
    with pytest.raises(ValueError, match="one-shot compile"):
        case.collector.compile()
    assert not case.events


@pytest.mark.parametrize("extra", ["extra.txt", "nested"])
def test_compile_refuses_extra_generated_artifact(case, extra):
    case.compile_extras.append("nested/" if extra == "nested" else extra)
    with pytest.raises(ValueError, match="exact generated"):
        case.collector.compile()
    assert case.collector.state == "failed"


@pytest.mark.parametrize("mutation", ["device", "tape", "arch", "module-option",
                                      "kernel-option", "config", "hash", "source",
                                      "compile-entry", "build-entry", "codegen-entry",
                                      "same-code-build-entry"])
def test_precompile_integrity_mutations_refused_before_compilation(case, mutation):
    c = case.collector
    if mutation == "device":
        case.wp.get_device = lambda: NS()
    elif mutation == "tape":
        case.context.runtime.tape = object()
    elif mutation == "arch":
        case.device.arch = 90
    elif mutation == "module-option":
        case.module.options["block_dim"] = 128
    elif mutation == "kernel-option":
        case.kernel.options["enable_backward"] = True
    elif mutation == "config":
        case.wp.config.verify_fp = True
    elif mutation == "hash":
        case.module.get_module_hash = lambda *_: b"X" * 32
    elif mutation == "source":
        case.solver_path.write_bytes(case.source_raw + b"\n")
    elif mutation == "compile-entry":
        case.context.Module._compile = staticmethod(lambda *args: True)
    elif mutation == "build-entry":
        case.wp._src.build.build_cuda = lambda *a, **k: None
    elif mutation == "same-code-build-entry":
        original = case.wp._src.build.build_cuda
        case.wp._src.build.build_cuda = FunctionType(
            original.__code__, original.__globals__, original.__name__,
            original.__defaults__, original.__closure__
        )
    else:
        case.context.ModuleBuilder.codegen = lambda *a: "changed"
    with pytest.raises((ValueError, OSError)):
        c.compile()
    assert not case.events


def test_preloaded_target_module_refused_without_clearing(case):
    existing = object()
    case.module.execs[(case.device.context, 128)] = existing
    with pytest.raises(ValueError, match="fresh target module cache"):
        case.collector.compile()
    assert case.module.execs[(case.device.context, 128)] is existing
    assert not case.events


def test_filename_path_injection_is_refused_at_construction(case, tmp_path):
    case.module._get_compile_output_name = lambda *a, **k: "../escape.cubin"
    directory = tmp_path / "path-injection" / "compiled-dense-solver"
    directory.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="plain distinct generated"):
        executable.DenseSolverExecutable(
            case.wp, case.context, case.collector.solver, case.device, directory
        )
    assert not directory.exists()


@pytest.mark.parametrize("mutation", ["binary", "metadata", "source", "extra"])
def test_artifact_mutation_after_compile_refused_before_load(case, mutation):
    case.collector.compile()
    path = {
        "binary": case.destination / "solver.sm120.cubin",
        "metadata": case.destination / "solver.meta",
        "source": case.destination / "solver-hash.cu",
        "extra": case.destination / "unexpected.txt",
    }[mutation]
    if mutation == "extra":
        path.write_bytes(b"extra")
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        case.collector.load(case.sass)
    assert [event[0] for event in case.events] == ["compile"]


def test_symlink_generated_leaf_is_refused_before_load(case, tmp_path):
    case.collector.compile()
    source = case.destination / "solver-hash.cu"
    target_file = tmp_path / "outside.cu"
    target_file.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target_file)
    with pytest.raises(ValueError):
        case.collector.load(case.sass)
    assert [event[0] for event in case.events] == ["compile"]


def test_oversized_generated_leaf_is_refused_before_load(case, tmp_path, monkeypatch):
    # Exercise an actual over-cap regular file without exceeding the supervisor's
    # unchanged 8 MiB LimitFSIZE. The full production bound is checked separately
    # through synthetic fstat metadata for the held generated leaf.
    small = tmp_path / "over-small-cap.bin"
    small.write_bytes(b"x" * 1025)
    with pytest.raises(ValueError, match="bounded file size"):
        executable._read_plain(small, 1024, "small cap fixture")
    case.collector.compile()
    binary = case.destination / "solver.sm120.cubin"
    assert executable.MAX_BINARY_BYTES == 8 * 1024**2
    held = binary.stat()
    original_fstat = os.fstat

    def oversized_fstat(fd):
        actual = original_fstat(fd)
        if (actual.st_dev, actual.st_ino) == (held.st_dev, held.st_ino):
            return NS(st_mode=actual.st_mode, st_size=executable.MAX_BINARY_BYTES + 1)
        return actual

    monkeypatch.setattr(os, "fstat", oversized_fstat)
    with pytest.raises(ValueError, match="bounded file size"):
        case.collector.load(case.sass)
    assert [event[0] for event in case.events] == ["compile"]


def test_failed_or_cached_load_does_not_make_successful_binding(case):
    case.collector.compile()
    case.load_behavior[0] = "other-cache-entry"
    with pytest.raises(ValueError, match="preserved other executable"):
        case.collector.load(case.sass)
    assert [event[0] for event in case.events] == ["compile", "load"]
    with pytest.raises(ValueError):
        case.collector.record()


@pytest.mark.parametrize("behavior", ["none", "wrong-object", "other-cache-entry"])
def test_unexpected_explicit_load_result_refuses_binding(case, behavior):
    case.collector.compile()
    case.load_behavior[0] = behavior
    with pytest.raises(ValueError):
        case.collector.load(case.sass)
    assert [event[0] for event in case.events] == ["compile", "load"]
    with pytest.raises(ValueError):
        case.collector.record()


@pytest.mark.parametrize("behavior", ["raise", "none", "malformed"])
def test_unexpected_hook_lookup_refuses_binding(case, behavior):
    case.collector.compile()
    case.hooks_behavior[0] = behavior
    with pytest.raises((ValueError, RuntimeError, AttributeError)):
        case.collector.load(case.sass)
    assert [event[0] for event in case.events] == ["compile", "load"]
    with pytest.raises(ValueError):
        case.collector.record()


def test_loaded_object_mutations_refuse_receipt(case):
    case.collector.compile()
    binding = case.collector.load(case.sass)
    binding.device.arch = 90
    with pytest.raises(ValueError):
        case.collector.record()


def test_returned_receipts_are_detached_from_internal_anchors(case):
    compile_record = case.collector.compile()
    compile_record["module_options"]["block_dim"] = 7
    assert case.collector.compile_record["module_options"]["block_dim"] == 256
    case.collector.load(case.sass)
    first = case.collector.record()
    first["compile"]["module_options"]["block_dim"] = 9
    first["flags"]["native_qualified"] = True
    second = case.collector.record()
    assert second["compile"]["module_options"]["block_dim"] == 256
    assert second["flags"]["native_qualified"] is False


@pytest.mark.parametrize("stdout", [b"", b"not sass", _sass(b"Z" * 16)])
def test_invalid_disassembly_refused_before_explicit_load(case, stdout):
    case.collector.compile()
    with pytest.raises(ValueError):
        case.collector.load(stdout)
    assert [event[0] for event in case.events] == ["compile"]


def test_import_does_not_initialize_cuda_packages():
    source_root = Path(executable.__file__).resolve().parents[2]
    script = textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {str(source_root)!r})
        from mjlab_microduck import stance_solver_executable
        assert not {{"warp", "torch", "mujoco", "mujoco_warp"}} & set(sys.modules)
        """
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mutation", ["symbol", "adjoint", "false-default-as-zero"])
def test_target_identity_mutations_refused_before_compile(case, mutation):
    if mutation == "symbol":
        case.collector.kernel.get_mangled_name = lambda: target.TARGET + "_87654321"
    elif mutation == "adjoint":
        case.collector.kernel.adj = object()
    else:
        case.collector.caller.__defaults__ = (0,)
    with pytest.raises(ValueError):
        case.collector.compile()
    assert not case.events


def test_selected_cubin_symbol_must_match_actual_kernel_before_load(case, monkeypatch):
    original = executable.disassembly.select_target_cubin

    def changed(raw):
        record = original(raw)
        record["target"]["symbol"] = target.TARGET + "_87654321_cuda_kernel_forward"
        return record

    monkeypatch.setattr(executable.disassembly, "select_target_cubin", changed)
    with pytest.raises(ValueError, match="selected CUBIN symbol"):
        case.collector.compile()
    assert [event[0] for event in case.events] == ["compile"]
    with pytest.raises(ValueError):
        case.collector.record()


def test_same_code_different_function_does_not_replace_held_entry(case):
    from types import FunctionType

    original = case.context.Module._compile
    clone = FunctionType(original.__code__, original.__globals__, original.__name__,
                         original.__defaults__, original.__closure__)
    assert clone is not original and clone.__code__ is original.__code__
    case.context.Module._compile = staticmethod(clone)
    with pytest.raises(ValueError, match="unchanged compiler"):
        case.collector.compile()
    assert not case.events


def test_same_bound_method_reaccess_is_allowed_but_other_receiver_is_not():
    class Receiver:
        def method(self):
            return None

    first, second = Receiver(), Receiver()
    held = first.method
    assert executable._same_entry(first, "method", held, held.__code__)
    assert not executable._same_entry(second, "method", held, held.__code__)


@pytest.mark.parametrize("mutation", ["failed-build", "foreign-cache-entry"])
def test_postload_foreign_cache_state_refuses_receipt_without_clearing(case, mutation):
    case.collector.compile()
    case.collector.load(case.sass)
    if mutation == "failed-build":
        case.module.failed_builds.add(case.device.context)
    else:
        case.module.execs[(case.device.context, 128)] = object()
    with pytest.raises(ValueError):
        case.collector.record()
    if mutation == "failed-build":
        assert case.device.context in case.module.failed_builds
    else:
        assert (case.device.context, 128) in case.module.execs
