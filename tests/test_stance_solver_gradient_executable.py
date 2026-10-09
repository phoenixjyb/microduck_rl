"""Original frontend code with synthetic loader objects; no CUDA execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, ModuleType, SimpleNamespace as NS

import pytest
from test_stance_solver_cost_binary import cubin, sass
from mjlab_microduck import stance_solver_gradient_executable as ex
from mjlab_microduck import stance_solver_gradient_binary as binary


@pytest.fixture
def setup(tmp_path, monkeypatch):
    device = NS(is_cuda=True, arch=120, context=1234, is_capturing=False, max_shared_memory_per_block=65536)
    wp = NS(config=NS(**{n: False for n in ex.retained._CONFIG_KEYS}))
    wp.get_device = lambda: device
    def noop(*args): pass
    wp._src = NS(build=NS(build_cuda=noop, load_cuda=noop))
    path = (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)
    solver = ModuleType(ex.adapter.prefix.static.MODULE)
    solver.__file__, solver.wp, solver.types = str(path), wp, NS()
    codes = ex.adapter.cost._codes(path.read_bytes(), str(path))
    solver._update_gradient = FunctionType(codes["_update_gradient"], solver.__dict__)
    module_hash = bytes([6]) * 32
    m = NS(options=dict(block_dim=256, strip_hash=False, enable_backward=False), execs={}, failed_builds=set())
    m.get_module_hash = lambda block: module_hash
    m._get_compile_arch = lambda d: 120
    m._get_compile_output_name = lambda d, **kw: "gradient.cubin"
    m._get_meta_name = lambda: "gradient.meta"
    m.get_module_identifier = lambda: "gradient"
    kernels = {}
    for stage, name in zip(ex.adapter.prefix.STAGES, ex.adapter.NAMES):
        k = NS(func=FunctionType(codes[name], solver.__dict__), module=m, adj=object(), options={})
        k.get_mangled_name = lambda name=name: name + "_1234abcd"
        setattr(solver, name, k)
        kernels[stage] = k
    actions, events = {}, []
    class Exec:
        def get_kernel_hooks(self, kernel): return self.kernel_hooks[kernel.adj]
    class Module:
        def get_module_hash(module, block): return module.get_module_hash(block)
        def _compile(module, device, directory, name, arch, ptx):
            events.append("compile")
            if "compile" in actions: return actions["compile"](module, directory)
            assert arch == 120 and ptx is False and not directory.exists()
            directory.mkdir()
            (directory / name).write_bytes(cubin([(k.get_mangled_name() + "_cuda_kernel_forward", bytes(range(i*16,(i+1)*16))) for i,k in enumerate(kernels.values())]))
            (directory / "gradient.meta").write_text(json.dumps({k.get_mangled_name() + "_cuda_kernel_forward_smem_bytes": 0 for k in kernels.values()}))
            (directory / "gradient.cu").write_text("// synthetic frontend, never compiled\n")
            return True
        def load(module, device, *, block_dim, binary_path, output_arch, meta_path):
            events.append("load")
            if "load" in actions: return actions["load"](module)
            assert block_dim == 256 and output_arch == 120
            e = Exec()
            e.device, e.handle, e.module_hash = device, 9876, module_hash
            e.meta = json.loads(Path(meta_path).read_text())
            e.kernel_hooks = {k.adj: NS(forward=1234+i, backward=None, forward_smem_bytes=0, backward_smem_bytes=0) for i,k in enumerate(kernels.values())}
            module.execs[(device.context,256)] = e
            if "post-load" in actions: actions["post-load"](module, e)
            return e
    class Builder:
        def codegen(self): pass
    context = NS(Kernel=NS, Module=Module, ModuleExec=Exec, ModuleBuilder=Builder, runtime=NS(tape=None))
    monkeypatch.setitem(sys.modules, solver.__name__, solver)
    def make(): return ex.GradientModuleExecutable(solver=solver, wp=wp, context=context, device=device, directory=tmp_path / "compiled-gradient")
    def disassembly(p): return {s:sass(p.symbols[s], bytes(range(i*16,(i+1)*16))) for i,s in enumerate(kernels)}
    return NS(make=make, disassembly=disassembly, solver=solver, wp=wp, context=context,
              device=device, module=m, kernels=kernels, events=events, actions=actions, directory=tmp_path / "compiled-gradient")


def test_one_fresh_explicit_load_binds_both_entries(setup):
    p = setup.make()
    p.compile()
    bindings = p.load(setup.disassembly(p))
    assert tuple(bindings) == ex.adapter.prefix.STAGES and setup.events == ["compile", "load"]
    a, b = bindings.values()
    assert a.module is b.module is setup.module
    assert a.executable is b.executable is p.executable and a.artifact is b.artifact
    record = p.record()
    assert record["explicit_load"]["returned_executable_is_cache_entry"] is True
    assert len(record["flags"]) == 6 and not any(record["flags"].values())
    assert all(record[n] is False for n in ("loaded_binary_bytes_observed", "driver_jit_machine_code_observed", "actual_dispatch_observed"))
    with pytest.raises(ValueError): p.compile()
    with pytest.raises(ValueError): p.load(setup.disassembly(p))


@pytest.mark.parametrize("mutation", ("module-path", "namespace", "caller-code", "caller-default", "caller-globals",
    "kernel-code", "kernel-default", "kernel-globals", "split-module", "kernel-type", "cache-other-block",
    "cached-target", "failed-context", "backward-enabled", "directory-exists", "wrong-symbol"))
def test_constructor_refuses_nonoriginal_or_nonfresh_frontend(setup, mutation):
    s, m, k = setup.solver, setup.module, setup.kernels["gradient"]
    if mutation == "module-path": s.__file__ = __file__
    elif mutation == "namespace": s.wp = object()
    elif mutation == "caller-code": s._update_gradient.__code__ = (lambda *a:None).__code__
    elif mutation == "caller-default": s._update_gradient.__defaults__ = (False,)
    elif mutation == "caller-globals": s._update_gradient = FunctionType(s._update_gradient.__code__, {})
    elif mutation == "kernel-code": k.func.__code__ = (lambda *a:None).__code__
    elif mutation == "kernel-default": k.func.__defaults__ = (False,)
    elif mutation == "kernel-globals": k.func = FunctionType(k.func.__code__, {})
    elif mutation == "split-module": k.module = NS()
    elif mutation == "kernel-type": setup.context.Kernel = object
    elif mutation == "cache-other-block": m.execs[(setup.device.context,128)] = object()
    elif mutation == "cached-target": m.execs[(setup.device.context,256)] = object()
    elif mutation == "failed-context": m.failed_builds.add(setup.device.context)
    elif mutation == "backward-enabled": m.options["enable_backward"] = True
    elif mutation == "wrong-symbol": k.get_mangled_name = lambda:"update_constraint_dense_1234abcd"
    else: setup.directory.mkdir()
    with pytest.raises(ValueError): setup.make()
    assert not setup.events


@pytest.mark.parametrize("mutation", ("caller-object", "equal-caller-code-clone", "types", "runtime-class", "kernel-object", "kernel-code",
    "cache", "cache-object", "failed-object", "failed", "options", "config", "compile-entry", "load-entry",
    "symbol", "tape", "device", "artifact", "extra-file", "metadata", "generated-source"))
def test_postcompile_mutation_never_loads(setup, mutation):
    p = setup.make()
    p.compile()
    if mutation == "caller-object": setup.solver._update_gradient = FunctionType(p.caller.__code__, setup.solver.__dict__)
    elif mutation == "equal-caller-code-clone":
        clone = p.caller.__code__.replace()
        assert clone == p.caller.__code__ and clone is not p.caller.__code__
        p.caller.__code__ = clone
    elif mutation == "types": setup.solver.types = NS()
    elif mutation == "runtime-class": setup.context.Module = NS()
    elif mutation == "kernel-object": setattr(setup.solver, ex.adapter.NAMES[1], NS())
    elif mutation == "kernel-code": p.kernels["gradient"].func.__code__ = (lambda *a:None).__code__
    elif mutation == "cache": p.module.execs[(p.device.context,128)] = object()
    elif mutation == "cache-object": p.module.execs = dict(p.module.execs)
    elif mutation == "failed-object": p.module.failed_builds = set()
    elif mutation == "failed": p.module.failed_builds.add(p.device.context)
    elif mutation == "options": p.module.options["strip_hash"] = True
    elif mutation == "config": setup.wp.config.use_precompiled_headers = True
    elif mutation == "compile-entry": setup.context.Module._compile = lambda *a:True
    elif mutation == "load-entry": setup.context.Module.load = lambda *a,**kw:None
    elif mutation == "symbol": p.kernels["gradient"].get_mangled_name = lambda:"forged"
    elif mutation == "tape": setup.context.runtime.tape = object()
    elif mutation == "device": p.device.context += 1
    elif mutation == "artifact": p.binary_path.write_bytes(b"forged")
    elif mutation == "extra-file": (p.directory/"extra").write_bytes(b"unbound")
    elif mutation == "metadata": p.meta_path.write_bytes(b"{}")
    else: p.source_path.write_bytes(b"changed")
    with pytest.raises((ValueError, AttributeError)): p.load(setup.disassembly(p))
    assert p.state == "failed" and p.executable is None and setup.events == ["compile"]


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong-span", "swap", "bytearray"))
def test_complete_disassembly_is_required_before_load(setup, mutation):
    p = setup.make()
    p.compile()
    evidence = setup.disassembly(p)
    if mutation == "missing": del evidence["gradient"]
    elif mutation == "extra": evidence["dense"] = b""
    elif mutation == "wrong-span": evidence["gradient"] = sass(p.symbols["gradient"], bytes(range(16)))
    elif mutation == "swap": evidence = dict(zip(evidence, reversed(list(evidence.values()))))
    else: evidence["gradient"] = bytearray(evidence["gradient"])
    with pytest.raises(ValueError): p.load(evidence)
    assert setup.events == ["compile"] and p.state == "failed" and not p.module.execs


@pytest.mark.parametrize("stage", ("compile", "load"))
def test_backend_failure_consumes_attempt_without_retry(setup, stage):
    def failed(*args): raise RuntimeError("owned backend failed")
    setup.actions[stage] = failed
    p = setup.make()
    if stage == "load": p.compile()
    with pytest.raises(RuntimeError): p.compile() if stage == "compile" else p.load(setup.disassembly(p))
    assert p.state == "failed"
    with pytest.raises(ValueError): p.compile()
    with pytest.raises(ValueError): p.load({})


def test_partial_load_cache_is_preserved_and_no_receipt(setup):
    p = setup.make()
    p.compile()
    foreign = object()
    def fail(m):
        m.execs[(p.device.context,256)] = foreign
        return foreign
    setup.actions["load"] = fail
    with pytest.raises(ValueError): p.load(setup.disassembly(p))
    assert p.module.execs[(p.device.context,256)] is foreign
    with pytest.raises(ValueError): p.record()


@pytest.mark.parametrize("mutation", ("cache-delta", "hook", "metadata", "binary", "source", "caller"))
def test_loaded_mutation_refuses_later_receipt(setup, mutation):
    p = setup.make()
    p.compile()
    p.load(setup.disassembly(p))
    if mutation == "cache-delta": p.module.execs[(p.device.context,128)] = object()
    elif mutation == "hook": p.executable.kernel_hooks[p.kernels["gradient"].adj].forward += 1
    elif mutation == "metadata": p.executable.meta[p.symbols["gradient"] + "_smem_bytes"] = 16
    elif mutation == "binary": p.binary_path.write_bytes(b"changed")
    elif mutation == "source": p.source_path.write_bytes(b"changed")
    else: setup.solver._update_gradient.__defaults__ = (False,)
    with pytest.raises(ValueError): p.record()


def test_compile_cannot_populate_executable_cache(setup):
    p = setup.make()
    def contaminates(module, directory):
        module.execs[(p.device.context,256)] = object()
        return True
    setup.actions["compile"] = contaminates
    with pytest.raises(ValueError): p.compile()
    assert p.state == "failed" and setup.events == ["compile"]
    with pytest.raises(ValueError): p.load({})


def test_plausible_loaded_executable_with_extra_cache_key_refuses(setup):
    p = setup.make()
    p.compile()
    foreign = object()
    def extra(m, e): m.execs[(p.device.context,128)] = foreign
    setup.actions["post-load"] = extra
    with pytest.raises(ValueError): p.load(setup.disassembly(p))
    assert p.state == "failed" and p.module.execs[(p.device.context,128)] is foreign
    assert (p.device.context,256) in p.module.execs
    with pytest.raises(ValueError): p.record()


def test_load_before_compile_consumes_load_and_refuses_compile(setup):
    p = setup.make()
    with pytest.raises(ValueError): p.load({})
    with pytest.raises(ValueError): p.compile()
    assert not setup.events and p.state == "failed"


@pytest.mark.parametrize("name", ("solver", "caller", "context_types", "kernels", "module", "generated", "state", "_sealed"))
def test_bindings_are_sealed(setup, name):
    p = setup.make()
    with pytest.raises(AttributeError): setattr(p, name, None)
    with pytest.raises(AttributeError): delattr(p, name)


def test_import_is_inert():
    subprocess.run([sys.executable, "-c", "from mjlab_microduck import stance_solver_gradient_executable; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"], check=True, timeout=10)


def test_frozen_cpu_frontend_identifies_original_pair_without_compile_load_launch(tmp_path):
    code = '''
import warp as wp
wp.init()
from warp._src import context as wc
from mujoco_warp._src import solver
from mjlab_microduck import stance_solver_gradient_executable as ex
def forbidden(*args, **kwargs): raise AssertionError('no compile/load/launch in CPU frontend check')
wc.Module._compile = wc.Module.load = wp.launch = forbidden
wp._src.build.build_cuda = wp._src.build.load_cuda = forbidden
kernels = ex.identify_kernels(solver, wp, wc)
assert tuple(kernels) == ex.adapter.prefix.STAGES
assert kernels['zero_grad_dot'].module is kernels['gradient'].module
for k in kernels.values():
    assert not k.module.execs and not k.module.failed_builds
    assert len(wc.Module.get_module_hash(k.module,256)) == 32 and k.get_mangled_name()
    assert not k.module.execs
print('CPU_ORIGINAL_GRADIENT_PAIR_FRONTEND_ONLY')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WARP_CACHE_PATH=str(tmp_path/"cache"),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = subprocess.run([sys.executable,"-c",code], capture_output=True,text=True,timeout=60,env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CPU_ORIGINAL_GRADIENT_PAIR_FRONTEND_ONLY' in result.stdout
