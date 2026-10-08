"""Synthetic CPU tests for the passive post-init solver observer."""

from collections import OrderedDict
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mjlab_microduck import stance_contact_boundary_control as contact
from mjlab_microduck import stance_solver_init_control as solver_init
from mjlab_microduck import stance_friction_runtime_control as friction


class FakeArray:
    def __init__(self, host, device, pointer, *, shape=None):
        self.host = host
        self.device = device
        self.ptr = pointer if host.nbytes else None
        self.shape = tuple(host.shape if shape is None else shape)
        item_width = host.dtype.itemsize
        for suffix in host.shape[len(self.shape) :]:
            item_width *= suffix
        canonical = []
        for dimension in reversed(self.shape):
            canonical.append(item_width)
            item_width *= dimension
        self.strides = tuple(reversed(canonical))
        self.dtype = str(host.dtype)
        self.is_contiguous = bool(host.flags.c_contiguous)

    def numpy(self):
        return self.host


def _packet_arrays(device):
    pointer = 0x100000
    arrays = {}
    for name in solver_init.SOLVER_INIT_ORDER:
        shape, host_shape, dtype = solver_init._SOLVER_INIT_SPECS[name]
        host = np.zeros(
            host_shape,
            dtype={"float32": np.float32, "int32": np.int32, "bool": np.bool_}[dtype],
        )
        arrays[name] = FakeArray(host, device, pointer, shape=shape)
        pointer += max(host.nbytes, 4) + 64
    for name in (
        "model.opt.impratio_invsqrt",
        "model.opt.tolerance",
        "model.opt.ls_tolerance",
        "model.opt.timestep",
        "model.stat.meaninertia",
    ):
        arrays[name].strides = (0,)
        arrays[name].is_contiguous = False
    return arrays


def _observer(monkeypatch, *, callback=None, arrays=None):
    monkeypatch.setattr(solver_init, "_TOTAL_CAPTURE_BYTES", 0)
    device = SimpleNamespace(context=44, is_cuda=True, is_capturing=False)
    stream = SimpleNamespace(device=device, cuda_stream=91)
    recipe = solver_init.SOLVER_RECIPE
    model = SimpleNamespace(
        nv=20,
        nv_pad=20,
        block_dim=SimpleNamespace(**recipe["block_dim"]),
        opt=SimpleNamespace(
            solver=recipe["solver"],
            cone=recipe["cone"],
            integrator=recipe["integrator"],
            disableflags=recipe["disableflags"],
            enableflags=recipe["enableflags"],
            iterations=recipe["iterations"],
            ls_iterations=recipe["ls_iterations"],
            ls_parallel=recipe["ls_parallel"],
            ls_parallel_min_step=recipe["ls_parallel_min_step"],
            graph_conditional=recipe["graph_conditional"],
            run_collision_detection=recipe["run_collision_detection"],
            timestep=FakeArray(np.zeros((1,), dtype=np.float32), device, 0x300000),
        ),
        stat=SimpleNamespace(),
        is_sparse=False,
    )
    data = SimpleNamespace(
        nworld=recipe["nworld"],
        njmax=recipe["njmax"],
        njmax_nnz=recipe["njmax_nnz"],
        naconmax=8192,
    )
    context_type = type("SolverContext", (), {})
    context_type.__module__ = "fake_solver_module"
    sink = OrderedDict()
    calls = []

    def original(actual_model, actual_data, actual_context, grad=True):
        calls.append((actual_model, actual_data, actual_context, grad))
        if callback is not None:
            callback(actual_model, actual_data, actual_context, grad)
        return "original-result"

    fake_module = ModuleType("fake_solver_module")
    fake_warp_util = ModuleType("fake_warp_util")
    fake_forward_module = ModuleType("fake_forward_module")
    fake_forward_warp = ModuleType("fake_forward_warp")
    fake_module.SolverContext = context_type
    fake_module.init_context = original
    fake_module.dispatch_grad = True
    exec(
        "def _solve(m, d, ctx): return init_context(m, d, ctx, grad=dispatch_grad)\n"
        "def _solve_body(m, d, ctx): return _solve(m, d, ctx)",
        fake_module.__dict__,
    )
    fake_warp_util.solve_body = fake_module._solve_body
    exec(
        "def solve_public(*args, **kwargs): return solve_body(*args, **kwargs)",
        fake_warp_util.__dict__,
    )
    fake_module.solve = fake_warp_util.solve_public
    fake_module.solve.__module__ = fake_module.__name__
    fake_module.solve.__wrapped__ = fake_module._solve_body
    fake_forward_module.solve_public = fake_module.solve
    exec(
        "def forward_body(m, d, ctx): return solve_public(m, d, ctx)",
        fake_forward_module.__dict__,
    )
    fake_forward_warp.forward_body = fake_forward_module.forward_body
    exec(
        "def forward_public(*args, **kwargs): return forward_body(*args, **kwargs)",
        fake_forward_warp.__dict__,
    )
    fake_forward_module.forward = fake_forward_warp.forward_public
    fake_forward_module.forward.__module__ = fake_forward_module.__name__
    fake_forward_module.forward.__wrapped__ = fake_forward_module.forward_body
    fake_module.init_context.__module__ = fake_module.__name__
    # Keep module identity valid for the observer's source binding checks.
    import sys

    for module in (fake_module, fake_warp_util, fake_forward_module, fake_forward_warp):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    observer = object.__new__(solver_init.RuntimeSolverInitObserver)
    observer.solver_module = fake_module
    observer.forward_module = fake_forward_module
    observer.source_sha256 = None
    observer.wrapper_source_sha256 = None
    observer.forward_source_sha256 = None
    observer._solver_source_path = None
    observer._wrapper_source_path = None
    observer._forward_source_path = None
    observer.solver_hook_active = False
    observer.init_function = original
    observer.init_code = original.__code__
    observer.solve_function = fake_module._solve
    observer.solve_code = fake_module._solve.__code__
    observer.solve_public_function = fake_module.solve
    observer.solve_public_code = fake_module.solve.__code__
    observer.solve_body_function = fake_module._solve_body
    observer.solve_body_code = fake_module._solve_body.__code__
    observer.solve_wrapper_globals = fake_warp_util.__dict__
    observer.solve_wrapper_module = fake_warp_util
    observer.forward_public_function = fake_forward_module.forward
    observer.forward_public_code = fake_forward_module.forward.__code__
    observer.forward_body_function = fake_forward_module.forward_body
    observer.forward_body_code = fake_forward_module.forward_body.__code__
    observer.forward_wrapper_globals = fake_forward_warp.__dict__
    observer.forward_wrapper_module = fake_forward_warp
    observer.init_wrapper = observer._init_context
    observer.init_calls = []
    observer.init_active = False
    observer.solver_packet = None
    observer.solver_bytes = 0
    observer.solver_layouts = None
    observer.arm = "original"
    observer.model = model
    observer.data = data
    observer.device = device
    observer.stream = stream
    observer.wp = SimpleNamespace(synchronize_stream=lambda actual: actual is stream)
    observer.sink = lambda name, raw: sink.__setitem__(name, raw)
    observer.entries = []
    observer.current_contact_nnz = object()
    observer._guard = lambda: None
    if arrays is None:
        arrays = _packet_arrays(device)
    model.opt.impratio_invsqrt = arrays["model.opt.impratio_invsqrt"]
    model.opt.tolerance = arrays["model.opt.tolerance"]
    model.opt.ls_tolerance = arrays["model.opt.ls_tolerance"]
    model.stat.meaninertia = arrays["model.stat.meaninertia"]
    observer._recipe_binding = solver_init._recipe_signature(model, data)
    monkeypatch.setattr(solver_init, "_arrays", lambda *_: arrays)
    return observer, context_type, sink, calls


def _forward(observer, context_type, index):
    observer.entries.append(
        {
            "index": index,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    observer.solver_module.init_context = observer.init_wrapper
    return observer.forward_module.forward(
        observer.model, observer.data, context_type()
    )


def _dispatch(observer, model, data, context, *, grad=True):
    observer.solver_module.dispatch_grad = grad
    observer.solver_module.init_context = observer.init_wrapper
    return observer.forward_module.forward(model, data, context)


def test_full_schedule_calls_original_once_and_publishes_only_post_return_forward4(
    monkeypatch,
):
    returned = {"ok": False}
    events = []

    def after_original(*_):
        events.append("original")
        returned["ok"] = True

    observer, context_type, sink, calls = _observer(
        monkeypatch, callback=after_original
    )
    for index in range(friction.FORWARDS):
        context = context_type()
        # The packet is assembled only after the fourth original returns.
        if index == solver_init.SAMPLED_FORWARD:
            original_arrays = solver_init._arrays

            def arrays_after_return(*args):
                assert returned["ok"]
                events.append("snapshot")
                return original_arrays(*args)

            monkeypatch.setattr(solver_init, "_arrays", arrays_after_return)
        observer.entries.append(
            {
                "index": index,
                "model_object_id": id(observer.model),
                "data_object_id": id(observer.data),
            }
        )
        observer.solver_module.init_context = observer.init_wrapper
        result = observer.forward_module.forward(observer.model, observer.data, context)
        assert result == "original-result"
    assert len(calls) == friction.FORWARDS
    assert len(observer.init_calls) == friction.FORWARDS
    assert events[solver_init.SAMPLED_FORWARD : solver_init.SAMPLED_FORWARD + 2] == [
        "original",
        "snapshot",
    ]
    assert events.count("snapshot") == 1
    assert observer.solver_packet["phase"] == "initialized-before-search"
    assert observer.solver_packet["forward"] == 4
    assert list(sink) == ["solver-init/original/forward-04.initialized.bin"]
    assert observer.solver_packet["bytes"] == len(next(iter(sink.values())))
    assert observer.solver_packet["fields"]["context.done"]["dtype"] == "|b1"
    assert observer.solver_packet["layouts"]["data.qM"]["shape"] == [64, 20, 20]


def test_receipt_binds_source_recipe_calls_and_false_flags(monkeypatch):
    observer, context_type, _, _ = _observer(monkeypatch)
    for index in range(friction.FORWARDS):
        _forward(observer, context_type, index)
    observer.active = False
    receipt = observer.solver_receipt()
    assert receipt["protocol"] == solver_init.PROTOCOL
    assert receipt["source_identity"]["solve_code_id"] == id(observer.solve_code)
    assert receipt["source_identity"]["forward_code_id"] == id(
        observer.forward_public_code
    )
    assert receipt["recipe"] == solver_init.SOLVER_RECIPE
    assert all(value is False for value in receipt["flags"].values())
    assert receipt["calls"][4]["phase"] == "pre-search"
    assert receipt["calls"][20]["phase"] == "unsampled-init-context"


def test_original_exception_never_emits_or_records_success(monkeypatch):
    def fail(*_):
        raise RuntimeError("init failed")

    observer, context_type, sink, calls = _observer(monkeypatch, callback=fail)
    observer.entries.append(
        {
            "index": 0,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    observer.solver_module.init_context = observer.init_wrapper
    with pytest.raises(RuntimeError, match="init failed"):
        _dispatch(observer, observer.model, observer.data, context_type())
    assert len(calls) == 1
    assert observer.init_calls == []
    assert observer.solver_packet is None
    assert not sink


@pytest.mark.parametrize("bad", [False, 1])
def test_grad_false_or_nonliteral_grad_refuses_before_original(monkeypatch, bad):
    observer, context_type, sink, calls = _observer(monkeypatch)
    observer.entries.append(
        {
            "index": 0,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    with pytest.raises(ValueError, match="grad=True"):
        _dispatch(observer, observer.model, observer.data, context_type(), grad=bad)
    assert not calls and not sink


def test_foreign_context_model_or_data_refuses_before_original(monkeypatch):
    observer, context_type, sink, calls = _observer(monkeypatch)
    observer.entries.append(
        {
            "index": 0,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    observer.solver_module.init_context = observer.init_wrapper
    for model, data, context in (
        (SimpleNamespace(), observer.data, context_type()),
        (observer.model, SimpleNamespace(), context_type()),
        (observer.model, observer.data, SimpleNamespace()),
    ):
        with pytest.raises(ValueError, match="identities"):
            observer.forward_module.forward(model, data, context)
    assert not calls and not sink


def test_schedule_guard_refuses_uncompleted_or_out_of_order_friction_entry(monkeypatch):
    observer, context_type, sink, calls = _observer(monkeypatch)
    observer.entries = []
    with pytest.raises(ValueError, match="schedule"):
        _dispatch(observer, observer.model, observer.data, context_type())
    assert not calls and not sink


def test_direct_external_init_dispatch_refuses_even_with_matching_schedule(monkeypatch):
    observer, context_type, sink, calls = _observer(monkeypatch)
    observer.entries.append(
        {
            "index": 0,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    with pytest.raises(ValueError, match="public solver and forward"):
        observer._init_context(observer.model, observer.data, context_type(), grad=True)
    assert not calls and not sink


def test_direct_public_solve_without_public_forward_refuses(monkeypatch):
    observer, context_type, sink, calls = _observer(monkeypatch)
    observer.entries.append(
        {
            "index": 0,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    observer.solver_module.init_context = observer.init_wrapper
    with pytest.raises(ValueError, match="public solver and forward"):
        observer.solve_public_function(observer.model, observer.data, context_type())
    assert not calls and not sink


def test_reentrant_dispatch_refuses_without_second_original_call(monkeypatch):
    observer, context_type, sink, calls = _observer(monkeypatch)

    def recurse(*_):
        _dispatch(observer, observer.model, observer.data, context_type())

    observer.init_function = lambda *args, **kwargs: recurse()
    observer.init_code = observer.init_function.__code__
    observer.solver_module.init_context = observer.init_wrapper
    observer.active = False
    observer._guard = lambda: None
    observer.entries.append(
        {
            "index": 0,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    with pytest.raises(ValueError, match="nonreentrant"):
        _dispatch(observer, observer.model, observer.data, context_type())
    assert not sink
    assert observer.init_calls == []


def test_layout_and_raw_packet_preserve_signed_zero_nan_and_padded_bytes(monkeypatch):
    device = SimpleNamespace(context=44, is_cuda=True, is_capturing=False)
    arrays = _packet_arrays(device)
    values = arrays["data.qacc"].host
    values[0, 0] = np.float32(-0.0)
    values[0, 1] = np.float32(np.nan)
    layouts = solver_init._layouts(arrays, device)
    assert layouts["data.qacc"]["host_dtype"] == "float32"
    raw, fields = solver_init._pack(arrays)
    field = fields["data.qacc"]
    copied = np.frombuffer(
        raw[field["offset"] : field["offset"] + field["bytes"]], dtype="<f4"
    )
    assert copied[0].view(np.uint32) == np.float32(-0.0).view(np.uint32)
    assert np.isnan(copied[1])


@pytest.mark.parametrize("mutation", ["stride", "alias"])
def test_layout_refuses_changed_stride_or_overlapping_storage(monkeypatch, mutation):
    device = SimpleNamespace(context=44, is_cuda=True, is_capturing=False)
    arrays = _packet_arrays(device)
    if mutation == "stride":
        arrays["data.qacc"].strides = (0, 4)
    else:
        arrays["data.qacc"].ptr = arrays["data.qvel"].ptr
    with pytest.raises(ValueError):
        solver_init._layouts(arrays, device)


def test_layout_accepts_and_retains_zero_stride_on_singleton_option_arrays():
    device = SimpleNamespace(context=44, is_cuda=True, is_capturing=False)
    arrays = _packet_arrays(device)
    scalar = arrays["model.opt.timestep"]
    assert scalar.shape == (1,)
    scalar.strides = (0,)
    scalar.is_contiguous = False
    layout = solver_init._solver_identity(scalar, device)
    assert layout["shape"] == [1]
    assert layout["strides"] == [0]
    assert layout["span"] == layout["bytes"] == 4


def test_native_cpu_allocation_matches_frozen_solver_init_abi(tmp_path):
    """Allocate the installed native carriers, but never call forward/solve."""
    script = textwrap.dedent(
        """\
        import json
        from types import SimpleNamespace
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        from mujoco_warp._src import forward, solver
        from mjlab_microduck.stance_warp_runtime import build_entity
        from mjlab_microduck.stance_solver_init_control import (
            SOLVER_INIT_ORDER, SOLVER_INIT_SPECS, _arrays, _layouts,
            _recipe_signature, SOLVER_RECIPE,
        )
        wp.init()
        def forbidden(*_args, **_kwargs):
            raise AssertionError("native allocation ABI test must not execute physics")
        solver.solve = forbidden
        solver._solve = forbidden
        forward.forward = forbidden
        device = wp.get_device("cpu")
        native = build_entity().compile()
        native_data = mujoco.MjData(native)
        with wp.ScopedDevice(device):
            model = mw.put_model(native)
            data = mw.put_data(native, native_data, nworld=64, nconmax=128, njmax=512)
            context = solver.create_solver_context(model, data)
            local_nnz = wp.zeros((64,), dtype=wp.int32, device=device)
            arrays = _arrays(SimpleNamespace(current_contact_nnz=local_nnz), model, data, context)
            layouts = _layouts(arrays, device)
            recipe, _ = _recipe_signature(model, data)
            assert recipe == SOLVER_RECIPE, repr(recipe)
            assert set(arrays) == set(SOLVER_INIT_ORDER) == set(SOLVER_INIT_SPECS)
            print(json.dumps({
                "recipe": recipe,
                "layouts": layouts,
                "count": len(arrays),
                "context_type": type(context).__name__,
            }, sort_keys=True))
        """
    )
    cache = tmp_path / "warp-cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "WARP_CACHE_PATH": str(cache),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence["recipe"] == solver_init.SOLVER_RECIPE
    assert evidence["count"] == len(solver_init.SOLVER_INIT_ORDER) == 66
    for name in (
        "model.opt.impratio_invsqrt",
        "model.opt.tolerance",
        "model.opt.ls_tolerance",
        "model.opt.timestep",
        "model.stat.meaninertia",
    ):
        assert evidence["layouts"][name]["shape"] == [1]
        assert evidence["layouts"][name]["strides"] == [0]
        assert evidence["layouts"][name]["bytes"] == 4
    assert evidence["layouts"]["context.gauss"]["shape"] == [64]
    assert evidence["layouts"]["context.prev_cost"]["shape"] == [64]
    assert evidence["layouts"]["context.done"]["host_dtype"] == "bool"
    assert evidence["context_type"] == "SolverContext"


def test_duplicate_leaf_refusal_and_packet_cap_do_not_publish(monkeypatch):
    observer, context_type, sink, _ = _observer(monkeypatch)
    monkeypatch.setattr(solver_init, "MAX_PACKET_BYTES", 1)
    for index in range(solver_init.SAMPLED_FORWARD + 1):
        observer.entries.append(
            {
                "index": index,
                "model_object_id": id(observer.model),
                "data_object_id": id(observer.data),
            }
        )
        with (
            pytest.raises(ValueError, match="packet cap")
            if index == solver_init.SAMPLED_FORWARD
            else _nullcontext()
        ):
            _dispatch(observer, observer.model, observer.data, context_type())
    assert not sink


@pytest.mark.parametrize(
    ("cap_name", "cap", "message"),
    [
        ("MAX_INSTANCE_BYTES", 1, "arm cap"),
        ("MAX_ALL_ARMS_BYTES", 1, "all-arms cap"),
    ],
)
def test_instance_and_all_arm_caps_refuse_without_publication(
    monkeypatch, cap_name, cap, message
):
    observer, context_type, sink, _ = _observer(monkeypatch)
    monkeypatch.setattr(solver_init, cap_name, cap)
    for index in range(solver_init.SAMPLED_FORWARD + 1):
        observer.entries.append(
            {
                "index": index,
                "model_object_id": id(observer.model),
                "data_object_id": id(observer.data),
            }
        )
        if index == solver_init.SAMPLED_FORWARD:
            with pytest.raises(ValueError, match=message):
                _dispatch(observer, observer.model, observer.data, context_type())
        else:
            _dispatch(observer, observer.model, observer.data, context_type())
    assert not sink


def test_out_of_range_active_count_refuses_without_packet(monkeypatch):
    observer, context_type, sink, _ = _observer(monkeypatch)
    arrays = solver_init._arrays(
        observer, observer.model, observer.data, context_type()
    )
    arrays["data.ne"].host[0] = observer.data.njmax + 1
    monkeypatch.setattr(solver_init, "_arrays", lambda *_: arrays)
    for index in range(solver_init.SAMPLED_FORWARD + 1):
        observer.entries.append(
            {
                "index": index,
                "model_object_id": id(observer.model),
                "data_object_id": id(observer.data),
            }
        )
        if index == solver_init.SAMPLED_FORWARD:
            with pytest.raises(ValueError, match="bounded initialized count"):
                _dispatch(observer, observer.model, observer.data, context_type())
        else:
            _dispatch(observer, observer.model, observer.data, context_type())
    assert not sink


def test_source_guard_refuses_foreign_init_hook_without_replacing_it(monkeypatch):
    monkeypatch.setattr(
        contact.RuntimeContactBoundaryObserver, "_guard", lambda self: None
    )
    observer, _, _, _ = _observer(monkeypatch)
    observer.active = True

    def foreign(*_args):
        return None

    observer.solver_module.init_context = foreign
    with pytest.raises(ValueError, match="solver init and solve"):
        solver_init.RuntimeSolverInitObserver._guard(observer)
    assert observer.solver_module.init_context is foreign


def test_source_guard_refuses_replaced_public_forward(monkeypatch):
    monkeypatch.setattr(
        contact.RuntimeContactBoundaryObserver, "_guard", lambda self: None
    )
    observer, _, _, _ = _observer(monkeypatch)
    observer.active = True
    observer.solver_module.init_context = observer.init_function
    observer.forward_module.forward = lambda *args: None
    with pytest.raises(ValueError, match="solver init and solve"):
        solver_init.RuntimeSolverInitObserver._guard(observer)


def test_default_constructor_pins_installed_solver_and_forward_without_physics(
    monkeypatch,
):
    from mujoco_warp._src import forward, solver

    monkeypatch.setattr(
        contact.RuntimeContactBoundaryObserver,
        "__init__",
        lambda *_args, **_kwargs: None,
    )
    observer = solver_init.RuntimeSolverInitObserver(
        "original", contact_kernel=object()
    )
    assert observer.solver_module is solver
    assert observer.forward_module is forward
    assert observer.solve_function is solver._solve
    assert observer.solve_body_function is inspect.unwrap(solver.solve)
    assert observer.forward_body_function is inspect.unwrap(forward.forward)
    assert observer.source_sha256 == solver_init.SOLVER_SOURCE_SHA256
    assert observer.wrapper_source_sha256 == solver_init.SOLVER_WRAPPER_SOURCE_SHA256
    assert observer.forward_source_sha256 == solver_init.FORWARD_SOURCE_SHA256


def test_context_installs_and_releases_only_its_solver_hook(monkeypatch):
    def base_enter(self):
        self.active = True
        return self

    def base_exit(self, *_):
        self.active = False
        return False

    monkeypatch.setattr(contact.RuntimeContactBoundaryObserver, "__enter__", base_enter)
    monkeypatch.setattr(contact.RuntimeContactBoundaryObserver, "__exit__", base_exit)
    observer, context_type, _, _ = _observer(monkeypatch)
    observer.active = False
    observer._guard = lambda: None
    observer.__enter__()
    assert observer.solver_module.init_context is observer.init_wrapper
    for index in range(friction.FORWARDS):
        _forward(observer, context_type, index)
    observer.__exit__(None, None, None)
    assert observer.solver_module.init_context is observer.init_function


def test_real_inherited_contact_observer_context_lifecycle(monkeypatch):
    observer, context_type, _, _ = _observer(monkeypatch)
    backend_module = ModuleType("fake_frozen_solver_backend")
    exec(
        "def original_kernel(): pass\ndef candidate_kernel(): pass\ndef contact_kernel_func(): pass",
        backend_module.__dict__,
    )
    sys.modules[backend_module.__name__] = backend_module
    original = SimpleNamespace(key="original", func=backend_module.original_kernel)
    candidate = SimpleNamespace(key="candidate", func=backend_module.candidate_kernel)
    contact_kernel = SimpleNamespace(
        key="contact",
        func=backend_module.contact_kernel_func,
        module=SimpleNamespace(kernel_key="contact-module"),
    )

    def contact_factory(*_):
        return contact_kernel

    def make_constraint(*_):
        return None

    def launch(*_args, **_kwargs):
        return None

    constraint = SimpleNamespace(
        _friction_dof=original,
        _efc_contact_init=contact_factory,
        make_constraint=make_constraint,
    )
    observer.constraint = constraint
    observer.original = original
    observer.candidate = candidate
    observer.make = constraint.make_constraint
    observer.make_code = inspect.unwrap(observer.make).__code__
    observer.launch = launch
    observer.launch_code = launch.__code__
    observer.kernel_codes = tuple(
        (kernel, kernel.func, kernel.func.__code__) for kernel in (original, candidate)
    )
    observer.make_wrapper = observer._make
    observer.launch_wrapper = observer._launch
    observer.active = False
    observer.inside = False
    observer.wp = SimpleNamespace(
        launch=launch, synchronize_stream=lambda actual: actual is observer.stream
    )
    observer.guard = lambda: None
    observer.contact_factory = contact_factory
    observer.contact_factory_func = inspect.unwrap(contact_factory)
    observer.contact_factory_code = contact_factory.__code__
    observer.contact_kernel = contact_kernel
    observer.contact_warp_module = contact_kernel.module
    observer.contact_function = contact_kernel.func
    observer.contact_code = contact_kernel.func.__code__
    observer.contact_module_name = backend_module.__name__
    observer.contact_module = backend_module

    observer.__enter__()
    assert observer.solver_module.init_context is observer.init_wrapper
    for index in range(friction.FORWARDS):
        _forward(observer, context_type, index)
    observer.__exit__(None, None, None)
    assert observer.solver_module.init_context is observer.init_function
    assert constraint.make_constraint is observer.make
    assert observer.wp.launch is launch


def test_close_preserves_foreign_solver_hook(monkeypatch):
    def base_enter(self):
        self.active = True
        return self

    def base_exit(self, *_):
        self.active = False
        return False

    monkeypatch.setattr(contact.RuntimeContactBoundaryObserver, "__enter__", base_enter)
    monkeypatch.setattr(contact.RuntimeContactBoundaryObserver, "__exit__", base_exit)
    observer, _, _, _ = _observer(monkeypatch)
    observer.active = False
    observer._guard = lambda: None
    observer.__enter__()

    def foreign(*_args):
        return None

    observer.solver_module.init_context = foreign
    with pytest.raises(ValueError, match="foreign solver init hook"):
        observer.__exit__(RuntimeError, RuntimeError("stop"), None)
    assert observer.solver_module.init_context is foreign


def test_solve_function_code_change_refuses(monkeypatch):
    monkeypatch.setattr(
        contact.RuntimeContactBoundaryObserver, "_guard", lambda self: None
    )
    observer, _, _, _ = _observer(monkeypatch)
    observer.active = True
    observer.solver_module.init_context = observer.init_wrapper
    observer.solver_module._solve = lambda *args: None
    with pytest.raises(ValueError, match="solver init and solve"):
        solver_init.RuntimeSolverInitObserver._guard(observer)


def test_extra_init_dispatch_after_21_is_refused(monkeypatch):
    observer, context_type, sink, calls = _observer(monkeypatch)
    for index in range(friction.FORWARDS):
        _forward(observer, context_type, index)
    observer.entries.append(
        {
            "index": friction.FORWARDS,
            "model_object_id": id(observer.model),
            "data_object_id": id(observer.data),
        }
    )
    with pytest.raises(ValueError, match="nonreentrant bounded"):
        _dispatch(observer, observer.model, observer.data, context_type())
    assert len(calls) == friction.FORWARDS
    assert len(sink) == 1


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False
