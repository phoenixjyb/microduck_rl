"""Synthetic compile/load objects only; no CUDA or compiler execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from test_stance_solver_cost_dispatch import case, capture_case, example
from test_stance_solver_cost_binary import cubin, sass, symbol
from mjlab_microduck import stance_solver_cost_executable as ex
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_executable as retained


@pytest.fixture
def setup(case, tmp_path):
    wp = case.wp
    kernels = dict(zip(stages.STAGES, case.kernels))
    for n, k in kernels.items():
        k.get_mangled_name = lambda n=n: symbol(n).removesuffix("_cuda_kernel_forward")
    wp.config = NS(**{n: False for n in retained._CONFIG_KEYS})
    def noop(*args): pass
    wp._src = NS(build=NS(build_cuda=noop, load_cuda=noop))
    class Exec:
        def get_kernel_hooks(self, kernel): return self.kernel_hooks[kernel.adj]
    class Module:
        def get_module_hash(m, block): return m.get_module_hash(block)
        def _compile(m, device, directory, name, arch, ptx):
            assert arch == 120 and ptx is False and not directory.exists()
            directory.mkdir()
            owned = [(n, k) for n, k in kernels.items() if k.module is m]
            (directory / name).write_bytes(cubin([(symbol(n), bytes(range(16))) for n, k in owned]))
            (directory / "module.meta").write_text(json.dumps({symbol(n) + "_smem_bytes": 0 for n, k in owned}))
            (directory / "module.cu").write_text("// synthetic generated source, not compiled\n")
            return True
        def load(m, device, *, block_dim, binary_path, output_arch, meta_path):
            assert block_dim == 256 and output_arch == 120
            e = Exec()
            e.device, e.handle, e.module_hash = device, 7890, m.get_module_hash(256)
            e.meta, e.kernel_hooks = json.loads(Path(meta_path).read_text()), {}
            for n, k in kernels.items():
                if k.module is m:
                    e.kernel_hooks[k.adj] = NS(forward=1234 + list(kernels).index(n), backward=None,
                                              forward_smem_bytes=0, backward_smem_bytes=0)
            m.execs[(device.context, 256)] = e
            return e
    class Builder:
        def codegen(self): pass
    context = NS(Module=Module, ModuleExec=Exec, ModuleBuilder=Builder, runtime=case.runtime)
    for m in {id(k.module): k.module for k in kernels.values()}.values():
        m.execs.clear()
        m._get_compile_arch = lambda d: 120
        m._get_compile_output_name = lambda d, **kw: "module.cubin"
        m._get_meta_name = lambda: "module.meta"
        m.get_module_identifier = lambda: "module"
    def graph_guard(): ex.verify_kernel_graph(case.solver, case.util, kernels)
    def make(role):
        return ex.CostModuleExecutable(wp=wp, context=context, device=case.stream.device,
            directory=tmp_path / ("compiled-" + role), role=role,
            kernels={n: kernels[n] for n in ex.GROUPS[role]}, graph_guard=graph_guard)
    return NS(make=make, context=context, kernels=kernels, wp=wp, case=case)


@pytest.mark.parametrize("role", ex.GROUPS)
def test_three_fresh_module_paths_bind_exact_entry_points(setup, role):
    p = setup.make(role)
    p.compile()
    bound = p.load({n: sass(p.symbols[n], bytes(range(16))) for n in p.kernels})
    assert tuple(bound) == ex.GROUPS[role]
    r = p.record()
    assert not any(r["flags"].values()) and r["explicit_load"]["returned_executable_is_cache_entry"]
    assert r["loaded_binary_bytes_observed"] is r["actual_dispatch_observed"] is False
    if role == "shared":
        assert bound["init_cost"].artifact is bound["dense"].artifact
        assert bound["init_cost"].executable is bound["dense"].executable
    with pytest.raises(ValueError): p.compile()
    with pytest.raises(ValueError): p.load({})


@pytest.mark.parametrize("mutation", ("cached-other-block", "failed-build", "wrapper", "closure", "kernel-code"))
def test_frontend_preparation_refusal(setup, mutation):
    c = setup.case
    if mutation == "cached-other-block": c.kernels[2].module.execs[(c.stream.device.context, 128)] = object()
    elif mutation == "failed-build": c.kernels[2].module.failed_builds.add(c.stream.device.context)
    elif mutation == "wrapper": c.solver.update_constraint_efc.__code__ = c.solver.update_constraint_efc.__code__.replace(co_name="forged")
    elif mutation == "closure": c.kernels[1].func.__closure__[0].cell_contents = True
    else: c.kernels[0].func.__code__ = (lambda *a: None).__code__
    with pytest.raises(ValueError): setup.make("shared")


@pytest.mark.parametrize("mutation", ("cache", "compile-entry", "config", "artifact", "symbol", "failed", "load-entry"))
def test_preload_mutations_never_load(setup, mutation):
    p = setup.make("shared")
    p.compile()
    if mutation == "cache": p.module.execs[(p.device.context, 128)] = object()
    elif mutation == "compile-entry": setup.context.Module._compile = lambda *a: True
    elif mutation == "config": setup.wp.config.use_precompiled_headers = True
    elif mutation == "artifact": p.binary_path.write_bytes(b"forged")
    elif mutation == "symbol": p.kernels["dense"].get_mangled_name = lambda: "forged"
    elif mutation == "failed": p.module.failed_builds.add(p.device.context)
    else: setup.context.Module.load = lambda *a, **kw: None
    with pytest.raises(ValueError): p.load({n: sass(p.symbols[n], bytes(range(16))) for n in p.kernels})
    assert p.state == "failed" and p.executable is None


def test_disassembly_mismatch_consumes_load(setup):
    p = setup.make("efc")
    p.compile()
    with pytest.raises(ValueError): p.load({"efc": sass(p.symbols["efc"], bytes(reversed(range(16))))})
    with pytest.raises(ValueError): p.load({"efc": sass(p.symbols["efc"], bytes(range(16)))})
    assert not p.module.execs and p.state == "failed"


def test_no_load_before_compile_or_retry(setup):
    p = setup.make("gauss")
    with pytest.raises(ValueError): p.load({})
    with pytest.raises(ValueError): p.compile()
    assert p.state == "failed"


@pytest.mark.parametrize("name", ("module", "kernels", "graph_guard", "generated", "state", "_sealed"))
def test_immutable_executable_bindings(setup, name):
    p = setup.make("shared")
    with pytest.raises(AttributeError): setattr(p, name, None)
    with pytest.raises(AttributeError): delattr(p, name)


def test_existing_factory_entries_cannot_be_materialized_again(setup):
    with pytest.raises(ValueError, match="absent declared"): ex.materialize_kernels(setup.case.solver, setup.case.util)


def test_import_is_inert():
    code = "from mjlab_microduck import stance_solver_cost_executable; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_frozen_cpu_abi_materializes_three_modules_without_compile_load_launch(tmp_path):
    code = '''
import warp as wp
wp.init()
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected compile/load/launch in CPU ABI check')
wp.launch = forbidden
import mujoco, mujoco_warp as mw
from mujoco_warp._src import solver, warp_util
from warp._src import context as wc
from mjlab_microduck.stance_warp_runtime import build_entity
from mjlab_microduck import stance_solver_cost_executable as ex
from mjlab_microduck import stance_solver_replay_probe as retained
wc.Module._compile = wc.Module.load = forbidden
wp._src.build.build_cuda = wp._src.build.load_cuda = forbidden
with wp.ScopedDevice('cpu'):
    native = build_entity().compile()
    model = mw.put_model(native)
    data = mw.put_data(native, mujoco.MjData(native), nworld=64, nconmax=128, njmax=512)
    ctx = solver.create_solver_context(model, data)
    retained.recipe_signature(model, data)
    assert ctx.changed_efc_ids.shape == (64,512)
    assert ctx.changed_efc_count.shape == (64,)
    kernels = ex.materialize_kernels(solver, warp_util)
    assert len({id(k.module) for k in kernels.values()}) == 3
    for k in kernels.values():
        assert k.module.execs == {} and not k.module.failed_builds
        digest = wc.Module.get_module_hash(k.module, 256)
        assert len(digest) == 32 and k.get_mangled_name()
        assert k.module.execs == {}
print('CPU_THREE_MODULE_FRONTEND_ONLY')
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", WARP_CACHE_PATH=str(tmp_path / "cache")))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU_THREE_MODULE_FRONTEND_ONLY" in result.stdout
