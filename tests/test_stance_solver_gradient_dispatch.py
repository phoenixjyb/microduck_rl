"""Original CPython caller, fake arrays/executables; never CUDA evidence."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType, ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from test_stance_solver_gradient_prefix import parents, packet, prospective  # register sealed CPU fixtures
from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_gradient_dispatch as adapter
from mjlab_microduck import stance_solver_gradient_prefix as g


class Array:
    def numpy(self):
        assert self.device.is_cpu, "no GPU convenience readback"
        return self.values


@pytest.fixture
def case(prospective, tmp_path, monkeypatch):
    device = NS(is_cuda=True, is_cpu=False, arch=120, context=1234,
                is_capturing=False, max_shared_memory_per_block=65536)
    cpu = NS(is_cuda=False, is_cpu=True)
    wp = NS(array=Array, float32=object(), int32=object(), bool=object())
    stream = NS(device=device, cuda_stream=5678)
    arrays, hosts, events, actions, trace = {}, [], [], [], []
    scenario = NS(arm="control")
    first = g.unpack_bank(prospective.arms["control"][g.PHASES[0]])
    for index, (name, shape, host_shape, dtype) in enumerate(adapter.SPECS):
        a = Array()
        a.ptr, a.shape, a.dtype, a.device = 0x1000000 + index * 0x400000, shape, getattr(wp, dtype), device
        a.is_contiguous, a.requires_grad = True, False
        wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        a.values = np.frombuffer(first[name], dtype=wire).reshape(host_shape).copy()
        arrays[name] = a

    def empty(*, shape, dtype, device, requires_grad, pinned):
        assert device == "cpu" and requires_grad is False and pinned is True
        a = Array()
        a.ptr, a.shape, a.dtype, a.device = 0x20000000 + len(hosts) * 0x400000, shape, dtype, cpu
        a.is_contiguous, a.requires_grad, a.pinned = True, False, True
        a.values = np.empty(shape, dtype={wp.float32: "<f4", wp.int32: "<i4", wp.bool: "|b1"}[dtype])
        hosts.append(a)
        return a

    def copy(dst, src, *, stream):
        assert stream.cuda_stream == 5678
        trace.append("copy")
        np.copyto(dst.values, src.values)

    def sync(stream):
        assert stream.cuda_stream == 5678
        trace.append("sync")

    def launch(kernel, dim, inputs=[], outputs=[], adj_inputs=[], adj_outputs=[],
               device=None, stream=None, adjoint=False, record_tape=True,
               record_cmd=False, max_blocks=0, block_dim=256):
        events.append(kernel)
        for action in actions:
            action(kernel)
        fields = g.unpack_bank(prospective.arms[scenario.arm][g.STAGES[len(events) - 1] + ".after"])
        for name, _, shape, dtype in adapter.SPECS:
            wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
            np.copyto(arrays[name].values, np.frombuffer(fields[name], dtype=wire).reshape(shape))

    def get_device(): return device
    def get_stream(value):
        assert value is device
        return stream

    wp.empty, wp.copy, wp.synchronize_stream = empty, copy, sync
    wp.launch, wp.get_device, wp.get_stream = launch, get_device, get_stream
    path = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
    solver = ModuleType(g.static.MODULE)
    solver.__file__, solver.wp, solver.types = str(path), wp, NS()
    codes = adapter.cost._codes(path.read_bytes(), str(path))
    solver._update_gradient = FunctionType(codes["_update_gradient"], solver.__dict__)
    kernels = tuple(NS(func=FunctionType(codes[n], solver.__dict__)) for n in adapter.NAMES)
    module_hash = bytes([5]) * 32
    module = NS(options=dict(block_dim=256, strip_hash=False, enable_backward=False),
                execs={}, failed_builds=set(), get_module_hash=lambda block: module_hash)
    metadata, hooks = {}, {}
    for index, (name, k) in enumerate(zip(adapter.NAMES, kernels)):
        setattr(solver, name, k)
        k.module, k.adj, k.options = module, object(), {}
        mangled = name + "_123abcff"
        k.get_mangled_name = lambda value=mangled: value
        metadata[mangled + "_cuda_kernel_forward_smem_bytes"] = 0
        hooks[k.adj] = NS(forward=6789 + index, backward=None, forward_smem_bytes=0, backward_smem_bytes=0)
    binary, meta = tmp_path / "synthetic.cubin", tmp_path / "synthetic.meta"
    binary.write_bytes(b"\x7fELF synthetic; never loaded")
    meta.write_bytes(json.dumps(metadata).encode())
    artifact = artifacts.retain_artifact(binary, meta, binary_size=binary.stat().st_size,
        binary_sha256=sha256(binary.read_bytes()).hexdigest(), metadata_size=meta.stat().st_size,
        metadata_sha256=sha256(meta.read_bytes()).hexdigest())
    executable = NS(handle=9876, device=device, module_hash=module_hash, kernel_hooks=hooks, meta=metadata)
    module.execs[(device.context, 256)] = executable
    bindings = {stage: artifacts.bind_loaded_module(artifact, k, device, executable, hooks[k.adj],
        block_dim=256, expected_module_hash=module_hash) for stage, k in zip(g.STAGES, kernels)}
    # .opt carries the retained input field, but no .solver: line 2934 must fail.
    model = NS(is_sparse=False, nv=20, opt=NS())
    data, ctx = NS(nworld=64, njmax=512, contact=NS(), efc=NS()), NS()
    roots = dict(model=model, data=data, contact=data.contact, efc=data.efc, context=ctx)
    for name, a in arrays.items():
        parts = name.split(".")
        obj = roots[parts[0]]
        for part in parts[1:-1]: obj = getattr(obj, part)
        setattr(obj, parts[-1], a)
    monkeypatch.setitem(sys.modules, g.static.MODULE, solver)
    runtime = NS(tape=None)
    inputs = dict(solver=solver, wp=wp, runtime=runtime, bindings=bindings,
                  model=model, data=data, context=ctx, stream=stream)
    observer = adapter.GradientDispatchObserver(**inputs)
    return NS(observer=observer, inputs=inputs, solver=solver, wp=wp, runtime=runtime,
              model=model, data=data, context=ctx, stream=stream, kernels=kernels,
              bindings=bindings, arrays=arrays, hosts=hosts, events=events,
              actions=actions, trace=trace, original=launch, scenario=scenario)


def assert_released(case):
    assert case.wp.launch is case.original and sys.gettrace() is None and sys.getprofile() is None
    assert adapter.shared._HOOK_LOCK.acquire(blocking=False)
    adapter.shared._HOOK_LOCK.release()


@pytest.mark.parametrize("arm", ["reference", "control"])
def test_original_two_sites_stop_before_any_solver_option_read(case, prospective, arm):
    case.scenario.arm = arm
    first = g.unpack_bank(prospective.arms[arm][g.PHASES[0]])
    for name, _, shape, dtype in adapter.SPECS:
        wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        np.copyto(case.arrays[name].values, np.frombuffer(first[name], dtype=wire).reshape(shape))
    case.observer.run()
    assert case.events == list(case.kernels)
    r = case.observer.record()
    assert r["call_site_lines"] == [2925, 2927] and r["excluded_next_branch_line"] == 2934
    assert r["intentional_stop_observed"] and r["call_sites_observed"]
    assert not any(r[n] for n in ("normal_caller_completion", "next_branch_observed",
        "trace_hooks_installed", "incremental_iteration_observed", "parent_capture_authenticated",
        "explicit_load_provenance_authenticated", "native_gpu_execution_authenticated",
        "external_unit_retirement_authenticated"))
    assert not any(r["qualification"].values())
    assert len(r["capture"]["fields"]) == 26
    assert case.trace.count("copy") == 104 and case.trace.count("sync") == 9
    assert all(case.observer.capture.raw(p) == prospective.arms[arm][p] for p in g.PHASES)
    assert_released(case)


def test_single_use_and_no_early_receipt(case):
    with pytest.raises(ValueError): case.observer.record()
    case.observer.run()
    with pytest.raises(ValueError, match="one-shot"): case.observer.run()


@pytest.mark.parametrize("fault", ["caller", "caller-code", "defaults", "kwdefaults", "kernel", "kernel-code",
    "kernel-defaults", "namespace", "source", "tape", "stream", "device", "device-context", "arch",
    "array", "pointer", "shape", "dtype", "grad-alias", "norm-alias", "host-pointer", "host-pinned",
    "host-alias", "copy", "numpy", "get-stream", "launch", "launch-code", "nv-bool", "nworld", "sparse"])
def test_changed_bindings_fail_before_dispatch(case, monkeypatch, fault):
    o, s, wp = case.observer, case.solver, case.wp
    if fault == "caller": s._update_gradient = lambda *a: None
    elif fault == "caller-code": s._update_gradient.__code__ = (lambda *a: None).__code__
    elif fault == "defaults": s._update_gradient.__defaults__ = (None,)
    elif fault == "kwdefaults": s._update_gradient.__kwdefaults__ = {"unused": 1}
    elif fault == "kernel": s.update_gradient_grad = case.kernels[0]
    elif fault == "kernel-code": case.kernels[0].func.__code__ = (lambda *a: None).__code__
    elif fault == "kernel-defaults": case.kernels[1].func.__defaults__ = (None,)
    elif fault == "namespace": s.wp = NS()
    elif fault == "source": monkeypatch.setattr(Path, "read_bytes", lambda _: b"wrong source")
    elif fault == "tape": case.runtime.tape = object()
    elif fault == "stream": case.stream.cuda_stream += 1
    elif fault == "device": case.stream.device = NS()
    elif fault == "device-context": case.stream.device.context += 1
    elif fault == "arch": case.stream.device.arch = 121
    elif fault == "array": case.context.grad = case.context.grad_dot
    elif fault == "pointer": case.context.grad.ptr += 1
    elif fault == "shape": case.context.grad.shape = (64, 19)
    elif fault == "dtype": case.context.grad.dtype = wp.int32
    elif fault == "grad-alias": case.context.grad.ptr = case.data.qfrc_smooth.ptr
    elif fault == "norm-alias": case.context.grad_dot.ptr = case.context.grad.ptr
    elif fault == "host-pointer": case.hosts[-1].ptr += 1
    elif fault == "host-pinned": case.hosts[-1].pinned = False
    elif fault == "host-alias": case.hosts[-1].ptr = case.context.grad.ptr
    elif fault == "copy": wp.copy = lambda *a, **k: None
    elif fault == "numpy": monkeypatch.setattr(Array, "numpy", lambda _: None)
    elif fault == "get-stream": wp.get_stream = lambda _: case.stream
    elif fault == "launch": wp.launch = lambda *a, **k: None
    elif fault == "launch-code": wp.launch.__code__ = wp.launch.__code__.replace(co_name="substituted")
    elif fault == "nv-bool": case.model.nv = True
    elif fault == "nworld": case.data.nworld = 63
    else: case.model.is_sparse = True
    with pytest.raises((ValueError, AttributeError)): o.run()
    assert not case.events and o.capture.failed
    with pytest.raises(ValueError): o.record()
    assert adapter.shared._HOOK_LOCK.acquire(blocking=False)
    adapter.shared._HOOK_LOCK.release()


@pytest.mark.parametrize("at", [0, 1])
def test_launch_failure_propagates_with_only_complete_before(case, at):
    def fail(_):
        if len(case.events) == at + 1: raise RuntimeError("synthetic launch failure")
    case.actions.append(fail)
    with pytest.raises(RuntimeError, match="synthetic launch failure"): case.observer.run()
    assert len(case.events) == at + 1 and case.observer.capture.failed
    assert set(case.observer.capture.packets) == set(g.PHASES[:at*2+1])
    assert_released(case)


def test_premature_same_sentinel_is_not_success(case):
    def fail(_): raise case.observer.stop
    case.actions.append(fail)
    with pytest.raises(ValueError, match="exact intentional stop"): case.observer.run()
    assert not case.observer.stopped and not case.observer.completed
    assert_released(case)


def test_unrelated_stop_instance_is_not_success(case):
    def fail(_): raise adapter._PrefixStop()
    case.actions.append(fail)
    with pytest.raises(ValueError, match="exact intentional stop"): case.observer.run()
    assert not case.observer.completed
    assert_released(case)


def test_recursive_or_indirect_launch_rejected(case):
    case.actions.append(lambda _: case.wp.launch(case.kernels[1], dim=(64, 20)))
    with pytest.raises(ValueError, match="two ordered"): case.observer.run()
    assert case.events == [case.kernels[0]]
    assert_released(case)


def test_third_launch_refused_before_delegation(case):
    # Reflection only manufactures the otherwise unreachable negative state.
    object.__setattr__(case.observer, "active", True)
    object.__setattr__(case.observer, "calls", 2)
    case.wp.launch = case.observer.wrapper
    try:
        with pytest.raises(ValueError, match="two ordered"): case.wp.launch(case.kernels[1], dim=(64, 20))
        assert not case.events
    finally:
        case.wp.launch = case.original
        object.__setattr__(case.observer, "active", False)


def test_call_site_drift_refused(case):
    object.__setattr__(case.observer, "source_lines", (2926, 2927))
    with pytest.raises(ValueError, match="original gradient caller site"): case.observer.run()
    assert not case.events
    assert_released(case)


@pytest.mark.parametrize("index,value", [(4, "cuda:0"), (5, object()), (6, True), (7, False), (8, True), (9, 1), (10, 128)])
def test_launch_default_changes_refused(case, index, value):
    values = list(case.wp.launch.__defaults__)
    values[index] = value
    case.wp.launch.__defaults__ = tuple(values)
    with pytest.raises(ValueError, match="runtime entries"): case.observer.run()
    assert not case.events
    assert_released(case)


@pytest.mark.parametrize("index", [2, 3])
def test_in_place_adjoint_default_list_change_refused(case, index):
    case.wp.launch.__defaults__[index].append(object())
    with pytest.raises(ValueError, match="eager launch defaults"): case.observer.run()
    assert not case.events and not case.observer.capture.packets
    assert_released(case)


@pytest.mark.parametrize("name", ["guard", "stream", "sources", "layouts", "host", "host_layouts", "packets", "next_stage", "_sealed"])
def test_sealed_capture(case, name):
    with pytest.raises(AttributeError, match="sealed"): setattr(case.observer.capture, name, object())
    with pytest.raises(AttributeError, match="sealed"): delattr(case.observer.capture, name)


def test_snapshot_outside_bracket_refused(case):
    with pytest.raises(ValueError, match="ordered active"): case.observer.capture._snapshot(g.PHASES[0])
    assert not case.observer.capture.packets


def test_snapshot_host_view_wrong_dtype_refused(case, monkeypatch):
    def wrong_view(array):
        assert array.device.is_cpu
        return array.values.view("<i4") if array is case.hosts[-1] else array.values
    monkeypatch.setattr(Array, "numpy", wrong_view)
    o = adapter.GradientDispatchObserver(**case.inputs)
    with pytest.raises(ValueError, match="canonical gradient CPU view"): o.run()
    assert not case.events and not o.capture.packets
    assert_released(case)


def test_normal_return_is_not_success(case):
    # Deliberately bypass code checks to exercise the outer normal-return guard.
    object.__setattr__(case.observer, "caller", lambda *a: None)
    object.__setattr__(case.observer.capture, "guard", lambda: None)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(adapter.GradientDispatchObserver, "_check", lambda _: None)
    object.__setattr__(case.observer.capture, "guard_code", case.observer.capture.guard.__code__)
    try:
        with pytest.raises(ValueError, match="normal gradient caller return"): case.observer.run()
        assert not case.events and not case.observer.completed
    finally:
        monkey.undo()
    assert_released(case)


def test_foreign_launch_hook_preserved(case):
    foreign = lambda *a, **k: None
    case.actions.append(lambda _: setattr(case.wp, "launch", foreign))
    with pytest.raises(ValueError, match="foreign gradient launch hook preserved"): case.observer.run()
    assert case.wp.launch is foreign and not case.observer.completed and case.observer.capture.failed
    assert adapter.shared._HOOK_LOCK.acquire(blocking=False)
    adapter.shared._HOOK_LOCK.release()


@pytest.mark.parametrize("kind", ["trace", "profile"])
def test_existing_trace_profile_refused_without_overwrite(case, kind):
    setter, getter = (sys.settrace, sys.gettrace) if kind == "trace" else (sys.setprofile, sys.getprofile)
    def foreign(*a): return foreign
    setter(foreign)
    try:
        with pytest.raises(ValueError, match="trace/profile"): case.observer.run()
        assert getter() is foreign and not case.events and case.wp.launch is case.original
    finally:
        setter(None)
    assert_released(case)


@pytest.mark.parametrize("kind", ["trace", "profile"])
def test_foreign_trace_profile_during_launch_preserved(case, kind):
    setter, getter = (sys.settrace, sys.gettrace) if kind == "trace" else (sys.setprofile, sys.getprofile)
    def foreign(*a): return foreign
    case.actions.append(lambda _: setter(foreign))
    try:
        with pytest.raises(ValueError, match="trace/profile"): case.observer.run()
        assert getter() is foreign and case.wp.launch is case.original and not case.observer.completed
    finally:
        setter(None)
    assert_released(case)


def test_capture_failure_does_not_issue_stop(case):
    original = case.wp.copy
    def fail(dst, src, *, stream):
        if len(case.events) == 2: raise RuntimeError("synthetic after-readback failure")
        original(dst, src, stream=stream)
    # Install before construction so the held runtime entry is the failing one.
    case.wp.copy = fail
    o = adapter.GradientDispatchObserver(**case.inputs)
    with pytest.raises(RuntimeError, match="after-readback failure"): o.run()
    assert len(case.events) == 2 and not o.stopped and not o.completed and o.calls == 1
    assert set(o.capture.packets) == set(g.PHASES[:3])
    assert_released(case)


def test_shared_lock_excludes_adapter(case):
    assert adapter.shared._HOOK_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="exclusive shared"): case.observer.run()
    finally:
        adapter.shared._HOOK_LOCK.release()
    assert not case.events and case.wp.launch is case.original
    with pytest.raises(ValueError, match="one-shot"): case.observer.run()


@pytest.mark.parametrize("name", ["caller", "caller_code", "wrapper", "source_lines", "branch_line", "stop", "capture", "calls", "_sealed"])
def test_sealed_observer(case, name):
    with pytest.raises(AttributeError, match="sealed"): setattr(case.observer, name, object())
    with pytest.raises(AttributeError, match="sealed"): delattr(case.observer, name)


def test_opaque_capture_does_not_authenticate_dispatch(case):
    c = case.observer.capture
    for stage in g.STAGES: c.bracket(stage, lambda: None)
    assert not c.record()["gpu_dispatch_authenticated"]
    assert not any(c.record()["qualification"].values())
    with pytest.raises(ValueError): case.observer.record()
    with pytest.raises(ValueError, match="fresh gradient capture"): case.observer.run()


def test_preserves_unrelated_nan_payload_bits(case):
    # Complete readback is raw, not a numerical filter of irrelevant fields.
    raw = np.array([0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000], dtype="<u4")
    case.arrays["efc.J"].values.view("<u4")[0, 511, :4] = raw
    c = case.observer.capture
    c.bracket(g.STAGES[0], lambda: None)
    for p in g.PHASES[:2]:
        fields = g.unpack_bank(c.raw(p))
        assert fields["efc.J"][511*20*4:511*20*4+16] == raw.tobytes()


def test_import_inert():
    code = "import sys; from mjlab_microduck import stance_solver_gradient_dispatch; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp','numpy'} for n in sys.modules)"
    subprocess.run((sys.executable, "-c", code), check=True, timeout=15)


def test_separate_fence_does_not_widen_prior_contracts():
    assert adapter.BASE == "870d933dc69e7bc3070e2e3b3c645f31c7f2b99c"
    assert len(adapter.OWN) == 3 and len(adapter.TESTS) == 21
    assert g.BASE == "d79ecf5a384aed92a9292b3f34bed9cced2af1af" and len(g.OWN) == 3
    assert adapter.OWN.isdisjoint(g.OWN)
    assert adapter.cost.BASE == "7aec9cebc2178be909ba119a0fa2cb504b264695"


@pytest.mark.parametrize("fault", ["branch", "dirty", "paths", "mode", "blob"])
def test_source_binding_fail_closed(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path)
    def command(*args):
        if args[1] == "branch": return b"wrong" if fault == "branch" else g.prior.BRANCH.encode()
        if args[1] == "status": return b" M owned" if fault == "dirty" else b""
        if args[1] == "rev-parse": return b"a" * 40
        if args[1] == "merge-base": return b""
        if args[1] == "diff": return b"wrong" if fault == "paths" else "\n".join(sorted(adapter.OWN)).encode()
        if args[1] == "ls-tree":
            return (b"120000" if fault == "mode" else b"100644") + b" blob " + b"a" * 40 + b"\tfile\0"
        pytest.fail("unexpected git command")
    monkeypatch.setattr(g.prior.retained, "command", command)
    monkeypatch.setattr(g.prior, "read_plain", lambda _: b"different blob")
    with pytest.raises(ValueError): adapter.source_binding(tmp_path.resolve())
