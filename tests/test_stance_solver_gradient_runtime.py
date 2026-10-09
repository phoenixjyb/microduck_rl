"""Selected Python API bodies: CPU-only checks, no runtime calls or GPU proof."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import pytest

from mjlab_microduck import stance_solver_gradient_runtime as boundary


@pytest.fixture
def runtime():
    import warp as wp
    from warp._src import build, context, types
    return SimpleNamespace(wp=wp, context=context, types=types, build=build,
                           make=lambda: boundary.WarpApiGuard(wp=wp, context=context, types=types, build=build))


def test_authenticate_real_cpu_frontend_without_runtime_calls(runtime):
    old_runtime = runtime.context.runtime
    guard = runtime.make()
    record = guard.record()
    assert runtime.context.runtime is old_runtime
    assert record["protocol"] == boundary.PROTOCOL and len(record["entries"]) == 24
    assert len(record["sources"]) == 4
    assert record["selected_python_bodies_authenticated"] is True
    assert record["held_direct_globals_checked"] is True
    assert record["transitive_dependencies_authenticated"] is False
    assert record["initialized_runtime_origin_authenticated"] is False
    assert record["native_execution_observed"] is False
    assert record["flags"] == boundary.executable.adapter.prefix.FLAGS


@pytest.mark.parametrize("name", boundary.CONTEXT_FUNCTIONS)
def test_preinstalled_api_shim_refused(runtime, monkeypatch, name):
    def shim(*args, **kwargs):
        pytest.fail("shim must not be called")
    monkeypatch.setattr(runtime.context, name, shim)
    monkeypatch.setattr(runtime.wp, name, shim)
    with pytest.raises(ValueError):
        runtime.make()


@pytest.mark.parametrize("owner_name,method", [(o, m) for o, methods in boundary.CONTEXT_METHODS.items() for m in methods])
def test_preinstalled_loader_descriptor_refused(runtime, monkeypatch, owner_name, method):
    monkeypatch.setattr(getattr(runtime.context, owner_name), method, lambda *a, **kw: None)
    with pytest.raises(ValueError):
        runtime.make()


@pytest.mark.parametrize("owner_name,method", (("array", "__init__"), ("array", "numpy"), ("build", "build_cuda"), ("build", "load_cuda")))
def test_preinstalled_readback_or_compiler_refused(runtime, monkeypatch, owner_name, method):
    owner = runtime.types.array if owner_name == "array" else runtime.build
    monkeypatch.setattr(owner, method, lambda *a, **kw: None)
    with pytest.raises(ValueError):
        runtime.make()


@pytest.mark.parametrize("mutation", ("code", "defaults", "list-default", "kwdefaults", "wrapped", "globals", "alias", "kernel-alias", "builtin", "same-builtin"))
def test_preinstalled_callable_metadata_or_alias_refused(runtime, monkeypatch, mutation):
    f = runtime.context.launch
    if mutation == "code":
        monkeypatch.setattr(f, "__code__", (lambda *a: None).__code__)
    elif mutation == "defaults":
        monkeypatch.setattr(f, "__defaults__", (*f.__defaults__[:-1], True))
    elif mutation == "list-default":
        monkeypatch.setattr(f, "__defaults__", ([1], *f.__defaults__[1:]))
    elif mutation == "kwdefaults":
        monkeypatch.setattr(f, "__kwdefaults__", {"stream": None})
    elif mutation == "wrapped":
        monkeypatch.setattr(f, "__wrapped__", f, raising=False)
    elif mutation == "globals":
        clone = FunctionType(f.__code__, dict(f.__globals__), argdefs=f.__defaults__)
        monkeypatch.setattr(runtime.context, "launch", clone)
        monkeypatch.setattr(runtime.wp, "launch", clone)
    elif mutation == "alias":
        monkeypatch.setattr(runtime.wp, "launch", lambda *a, **kw: None)
    elif mutation == "kernel-alias":
        monkeypatch.setattr(runtime.wp, "Kernel", object)
    elif mutation == "same-builtin":
        import builtins
        monkeypatch.setitem(runtime.context.__dict__, "len", builtins.len)
    else:
        monkeypatch.setitem(runtime.context.__dict__, "len", lambda *a: 0)
    with pytest.raises(ValueError):
        runtime.make()


@pytest.mark.parametrize("mutation", ("function", "equal-code", "default-tuple", "mutable-default", "keyword-defaults",
    "descriptor", "static-descriptor", "class", "class-name", "array-base", "alias", "module", "module-file", "module-name",
    "src-module", "runtime", "helper", "helper-code", "builtin-shadow", "attribute"))
def test_held_boundary_mutation_refuses_receipt(runtime, monkeypatch, mutation):
    guard = runtime.make()
    f = runtime.context.launch
    original_class_name = runtime.context.Module.__qualname__
    if mutation == "function":
        clone = FunctionType(f.__code__, f.__globals__, argdefs=f.__defaults__)
        monkeypatch.setattr(runtime.context, "launch", clone)
        monkeypatch.setattr(runtime.wp, "launch", clone)
    elif mutation == "equal-code":
        clone = f.__code__.replace()
        assert clone == f.__code__ and clone is not f.__code__
        monkeypatch.setattr(f, "__code__", clone)
    elif mutation == "default-tuple":
        monkeypatch.setattr(f, "__defaults__", tuple(list(f.__defaults__)))
    elif mutation == "mutable-default":
        f.__defaults__[0].append(1)
    elif mutation == "keyword-defaults":
        monkeypatch.setattr(f, "__kwdefaults__", {})
    elif mutation == "descriptor":
        monkeypatch.setattr(runtime.context.Module, "load", lambda *a, **kw: None)
    elif mutation == "static-descriptor":
        monkeypatch.setattr(runtime.context.Module, "load", staticmethod(runtime.context.Module.load))
    elif mutation == "class":
        monkeypatch.setattr(runtime.context, "Module", object)
    elif mutation == "class-name":
        runtime.context.Module.__qualname__ = "Forged"
    elif mutation == "array-base":
        monkeypatch.setattr(runtime.types, "Array", object)
    elif mutation == "alias":
        monkeypatch.setattr(runtime.wp, "array", object)
    elif mutation == "module":
        monkeypatch.setitem(sys.modules, runtime.context.__name__, SimpleNamespace())
    elif mutation == "module-file":
        monkeypatch.setattr(runtime.context, "__file__", __file__)
    elif mutation == "module-name":
        monkeypatch.setattr(runtime.context, "__name__", "forged")
    elif mutation == "src-module":
        monkeypatch.setattr(runtime.wp._src, "build", object)
    elif mutation == "runtime":
        monkeypatch.setattr(runtime.context, "runtime", object())
    elif mutation == "helper":
        monkeypatch.setattr(runtime.context, "init", lambda: None)
    elif mutation == "helper-code":
        monkeypatch.setattr(runtime.context.init, "__code__", (lambda: None).__code__)
    elif mutation == "builtin-shadow":
        import builtins
        monkeypatch.setitem(runtime.context.__dict__, "len", builtins.len)
    else:
        monkeypatch.setattr(f, "injected", True, raising=False)
    try:
        with pytest.raises(ValueError):
            guard.record()
    finally:
        if mutation == "mutable-default":
            f.__defaults__[0].clear()
        elif mutation == "class-name":
            runtime.context.Module.__qualname__ = original_class_name


@pytest.mark.parametrize("mutation", ("clone", "equal-default-values"))
def test_initial_source_equivalence_does_not_prove_object_origin(runtime, monkeypatch, mutation):
    f = runtime.context.launch
    if mutation == "clone":
        clone = FunctionType(f.__code__, f.__globals__, argdefs=f.__defaults__)
        monkeypatch.setattr(runtime.context, "launch", clone)
        monkeypatch.setattr(runtime.wp, "launch", clone)
    else:
        monkeypatch.setattr(f, "__defaults__", ([], [], [], [], *f.__defaults__[4:]))
    assert runtime.make().record()["initialized_runtime_origin_authenticated"] is False


def test_initial_nonselected_dependency_origin_is_not_claimed(runtime, monkeypatch):
    monkeypatch.setattr(runtime.context, "init", lambda: None)
    record = runtime.make().record()
    assert record["transitive_dependencies_authenticated"] is False


def test_preinstalled_python_builtin_shim_refused_in_fresh_process():
    code = """
import builtins
import warp as wp
from warp._src import context, types, build
from mjlab_microduck.stance_solver_gradient_runtime import WarpApiGuard
old = builtins.print
builtins.print = lambda *args, **kw: None
refused = False
try:
    try:
        WarpApiGuard(wp=wp, context=context, types=types, build=build)
    except ValueError as error:
        refused = 'builtin API dependency' in str(error)
finally:
    builtins.print = old
assert refused
print('refused')
"""
    p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and p.stdout.strip() == "refused", p.stderr


def test_source_mutation_is_refused_before_receipt(runtime, monkeypatch):
    guard = runtime.make()
    read = boundary.executable.retained._read_plain
    def changed(path, cap, name):
        raw, identity = read(path, cap, name)
        return raw + b"\n", identity
    monkeypatch.setattr(boundary.executable.retained, "_read_plain", changed)
    with pytest.raises(ValueError):
        guard.record()


def test_guard_is_sealed_and_receipt_does_not_mutate_it(runtime):
    guard = runtime.make()
    with pytest.raises(AttributeError):
        guard.runtime = object()
    with pytest.raises(AttributeError):
        del guard.rows
    with pytest.raises(TypeError):
        guard.modules["warp"] = object()
    record = guard.record()
    record["flags"]["training_authorized"] = True
    assert guard.record()["flags"]["training_authorized"] is False


@pytest.mark.parametrize("raw", (b"def f(a=unknown): pass", b"def f(a=object()): pass", b"@staticmethod\ndef f(): pass", b"def f(*, a=1): pass"))
def test_closed_default_and_decorator_language(raw):
    with pytest.raises(ValueError):
        boundary._definition(compile(raw, "frozen.py", "exec"), ast.parse(raw), "f")


@pytest.mark.parametrize("actual,expected", ((True, 1), (1, True), ((1,), [1]), ([1], []), ("x", None)))
def test_default_comparison_is_typed(actual, expected):
    assert not boundary._value(actual, expected)


def test_import_is_inert_in_fresh_cpu_process():
    code = """
import sys
from mjlab_microduck import stance_solver_gradient_runtime
assert not any(n == 'warp' or n.startswith('warp.') for n in sys.modules)
assert 'mujoco' not in sys.modules
print('inert')
"""
    p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and p.stdout.strip() == "inert", p.stderr


def test_authentication_never_calls_runtime_in_fresh_cpu_process():
    code = """
import json, sys
import warp as wp
from warp._src import context, types, build
from mjlab_microduck.stance_solver_gradient_runtime import WarpApiGuard
names = {'launch', 'copy', 'empty', 'get_device', 'get_stream', 'synchronize_stream', 'init',
         '_compile', 'load', 'build_cuda', 'load_cuda', 'numpy'}
def forbidden(frame, event, arg):
    if event == 'call' and frame.f_globals.get('__name__', '').startswith('warp.') and frame.f_code.co_name in names:
        raise RuntimeError('runtime function called during authentication')
sys.setprofile(forbidden)
try:
    g = WarpApiGuard(wp=wp, context=context, types=types, build=build)
    result = g.record()
finally:
    sys.setprofile(None)
assert context.runtime is None
print(json.dumps(result))
"""
    p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    record = json.loads(p.stdout)
    assert len(record["entries"]) == 24 and not any(record["flags"].values())
