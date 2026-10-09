"""Fresh CUDA-hidden entry guards, not native or GPU route evidence."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from mjlab_microduck import stance_solver_bootstrap_observer as observer


def run(code, tmp_path):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", WARP_CACHE_PATH=str(tmp_path / "cache"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=30)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()


FRONTEND = '''
import sys
from types import FunctionType
import warp as wp
from warp._src import context as c, types as t, build as b
from mjlab_microduck import stance_solver_bootstrap_observer as o
guard = o.probe.api.WarpApiGuard(wp=wp,context=c,types=t,build=b)
binding = o.probe.BootstrapBinding(wp,c,guard)
'''


def test_actual_cpu_bootstrap_records_unselected_entered_helpers(tmp_path):
    run(FRONTEND + '''
watch = o.BootstrapObserver(binding)
assert c.runtime is None
watch.initialize()
receipt = watch.record()
names = {r['qualified'] for r in receipt['frames'] if r['kind']=='warp'}
assert {'init','Runtime.__init__','Device.__init__','init_kernel_cache'} <= names
assert receipt['entered_python_bodies_checked'] is True
assert receipt['profile_removed'] is True and sys.getprofile() is None
assert receipt['stdlib_sources']
assert all(receipt[k] is False for k in ('whole_transitive_dependencies_authenticated',
    'native_c_bodies_authenticated','other_threads_observed','gpu_only_paths_observed'))
assert not any(receipt['qualification'].values())
assert not any(n.split('.')[0] in {'torch','mujoco','mujoco_warp'} for n in sys.modules)
try:watch.initialize()
except ValueError:pass
else:raise AssertionError('reused observer')
''', tmp_path)


@pytest.mark.parametrize("target", ("device", "cache", "allocator", "stdlib", "external-globals"))
@pytest.mark.parametrize("when", ("before", "after"))
def test_unselected_shim_refused_at_entry_before_its_body(tmp_path, target, when):
    run(FRONTEND + f"target={target!r}; when={when!r}\n" + r'''
def install():
    if target=='stdlib':
        import platform
        namespace=platform.__dict__
        destination=platform
        name='system'
    else:
        namespace=c.__dict__ if target!='cache' else b.__dict__
        destination=c.Device if target in ('device','external-globals') else c.CpuDefaultAllocator if target=='allocator' else b
        name='__init__' if target!='cache' else 'init_kernel_cache'
    namespace['_observer_canary']=[]
    exec('def shim(*args, **kwargs):\n    _observer_canary.append(1)\n',namespace)
    fn=namespace['shim']
    if target=='external-globals':
        outside={'_observer_canary':namespace['_observer_canary'],'__name__':'outside_bootstrap'}
        fn=FunctionType(fn.__code__,outside)
    setattr(destination,name,fn)
    return namespace['_observer_canary']
if when=='before':canary=install()
watch=o.BootstrapObserver(binding)
if when=='after':canary=install()
try:watch.initialize()
except ValueError:pass
else:raise AssertionError('shim accepted')
assert canary==[], 'shim body was entered'
assert sys.getprofile() is None
assert watch.failed and not watch.completed and watch.used
try:watch.record()
except ValueError:pass
else:raise AssertionError('failed receipt accepted')
''', tmp_path)


@pytest.mark.parametrize("hook", ("profile", "trace", "thread-profile", "thread-trace"))
def test_foreign_hooks_refused_and_preserved(tmp_path, hook):
    run(FRONTEND + f"hook={hook!r}\n" + '''
import threading
def foreign(*args):return foreign
setter,getter={'profile':(sys.setprofile,sys.getprofile),'trace':(sys.settrace,sys.gettrace),
 'thread-profile':(threading.setprofile,threading.getprofile),
 'thread-trace':(threading.settrace,threading.gettrace)}[hook]
setter(foreign)
try:
    try:o.BootstrapObserver(binding)
    except ValueError:pass
    else:raise AssertionError('foreign hook accepted')
    assert getter() is foreign and c.runtime is None
finally:setter(None)
''', tmp_path)


@pytest.mark.parametrize("fault", ("deadline", "calls", "rows", "tree", "coordinator", "sealed", "method"))
def test_bounds_and_authentication_refused(tmp_path, fault):
    run(FRONTEND + f"fault={fault!r}\n" + '''
if fault=='coordinator':o.COORDINATOR_PIN=(1,'0'*64)
if fault=='method':
    o.probe.BootstrapBinding.initialize=lambda self: None
if fault in ('coordinator','method'):
    try:o.BootstrapObserver(binding)
    except ValueError:pass
    else:raise AssertionError('bad coordinator accepted')
else:
    watch=o.BootstrapObserver(binding)
    if fault=='sealed':
        try:watch.used=True
        except AttributeError:pass
        else:raise AssertionError('mutable observer')
    else:
        if fault=='deadline':o.INIT_SECONDS=0
        elif fault=='calls':o.MAX_CALLS=0
        elif fault=='rows':o.MAX_ROWS=0
        else:
            original=o.warp_inventory
            o.warp_inventory=lambda root: ({}, {})
        try:watch.initialize()
        except ValueError:pass
        else:raise AssertionError('bad bound accepted')
        assert not watch.completed and sys.getprofile() is None
''', tmp_path)


def test_constructor_failure_no_hook_leak_or_receipt(tmp_path):
    run(FRONTEND + '''
watch=o.BootstrapObserver(binding)
# C function argument failure is not a replacement Python body. It illustrates
# why body equivalence does not establish valid values or native semantics.
c.warp_home=None
try:watch.initialize()
except TypeError:pass
else:raise AssertionError('bad native argument accepted')
assert sys.getprofile() is None and not watch.completed
try:watch.record()
except ValueError:pass
else:raise AssertionError('failure produced receipt')
''', tmp_path)


def test_indirect_stdlib_helper_is_explicitly_outside_body_scope(tmp_path):
    run(FRONTEND + '''
import platform
original=platform.uname
calls=[]
def indirect():
    calls.append(1)
    return original()
platform.uname=indirect
watch=o.BootstrapObserver(binding)
watch.initialize()
receipt=watch.record()
assert calls and receipt['whole_transitive_dependencies_authenticated'] is False
assert receipt['stdlib_origin_authenticated'] is False
assert sys.getprofile() is None
''', tmp_path)


def test_body_equivalent_clone_does_not_authenticate_defaults(tmp_path):
    run(FRONTEND + '''
original=b.init_kernel_cache
b.init_kernel_cache=FunctionType(original.__code__,original.__globals__,original.__name__,('changed-default',))
watch=o.BootstrapObserver(binding)
watch.initialize()  # Runtime supplies the argument explicitly.
assert watch.record()['callable_defaults_authenticated'] is False
assert sys.getprofile() is None
''', tmp_path)


@pytest.mark.parametrize('hook', ('profile','trace'))
def test_indirect_thread_hook_mutation_refused_before_body(tmp_path, hook):
    run(FRONTEND + f"hook={hook!r}\n" + '''
import platform,threading
original=platform.uname
def indirect():
    getattr(threading,'set'+hook)(lambda *args:None)
    return original()
platform.uname=indirect
watch=o.BootstrapObserver(binding)
try:watch.initialize()
except ValueError:pass
else:raise AssertionError('thread hook mutation accepted')
assert sys.getprofile() is None
assert threading.getprofile() is None and threading.gettrace() is None
assert watch.failed and not watch.completed
''', tmp_path)


def test_nested_graph_compilation_does_not_execute_source():
    raw=b"def f():\n    return [x*x for x in range(3)]\nraise RuntimeError('must not execute')\n"
    graph=observer.code_graph(raw,"fixture.py")
    assert {c.co_qualname for c in graph}=={'<module>','f'}  # 3.12 inlines list comprehensions
    raw=b"def f():\n    return (x*x for x in range(3))\n"
    assert 'f.<locals>.<genexpr>' in {c.co_qualname for c in observer.code_graph(raw,"fixture.py")}


@pytest.mark.parametrize("raw", (None, b"", b"x"*(4*1024**2+1)))
def test_invalid_source_graph_bounds(raw):
    with pytest.raises(ValueError):observer.code_graph(raw,"fixture.py")


def test_whole_warp_inventory_and_symlink_rejection(tmp_path):
    with pytest.raises(ValueError):observer.warp_inventory(tmp_path)
    link=tmp_path/'alias';link.symlink_to(tmp_path)
    with pytest.raises(ValueError):observer.warp_inventory(link)


def test_inert_import_new_source_fence_and_cpu_test_declaration(tmp_path):
    assert len(observer.OWN)==3 and observer.BASE=='20c00062c7d81dd4185cdb61e25a3a2efc68faca'
    assert len(observer.TESTS)==27 and observer.TESTS[:-1]==observer.probe.TESTS
    assert observer.probe.BASE=='df973002b20fa99771aa353b27d8111e5a97166a'
    run("from mjlab_microduck import stance_solver_bootstrap_observer;import sys;assert not any(n.split('.')[0] in {'warp','torch','numpy','mujoco','mujoco_warp'} for n in sys.modules)",tmp_path)
