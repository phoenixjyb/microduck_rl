"""Frozen CPython caller with synthetic kernels/binaries/arrays; no CUDA proof."""
from hashlib import sha256
import inspect
import json
from math import ceil
from pathlib import Path
import subprocess
import sys
from types import CodeType, FunctionType, ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from test_stance_solver_cost_stages import capture_case, example  # register CPU fixtures
from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_cost_dispatch as adapter
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_dispatch_guard as dense_guard
from mjlab_microduck import stance_solver_target_binding as static


def cell(value):
    return (lambda: value).__closure__[0]


@pytest.fixture
def case(capture_case, example, tmp_path, monkeypatch):
    c, wp = capture_case, capture_case.wp
    device = c.stream.device
    device.is_capturing, device.max_shared_memory_per_block = False, 65536
    events, actions = [], []

    def launch(kernel, dim, inputs=[], outputs=[], adj_inputs=[], adj_outputs=[],
               device=None, stream=None, adjoint=False, record_tape=True,
               record_cmd=False, max_blocks=0, block_dim=256):
        events.append(kernel)
        for action in actions:
            action(kernel)
        stage = stages.STAGES[len(events) - 1]
        fields = stages.unpack_bank(example.arms["control"][stage + ".after"])
        for name, _, host, dtype in stages.SPECS:
            wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
            np.copyto(c.arrays[name].values, np.frombuffer(fields[name], dtype=wire).reshape(host))

    def get_device():
        return device

    def get_stream(value):
        assert value is device
        return c.stream

    wp.launch, wp.get_device, wp.get_stream = launch, get_device, get_stream
    root = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src"
    solver, util = ModuleType(static.MODULE), ModuleType(adapter.UTIL_MODULE)
    solver.__file__, util.__file__ = str(root / "solver.py"), str(root / "warp_util.py")
    solver.wp, solver.ceil, solver.types, solver.math = wp, ceil, NS(vec5=object()), NS()
    codes = adapter._codes((root / "solver.py").read_bytes(), solver.__file__)
    ucodes = adapter._codes((root / "warp_util.py").read_bytes(), util.__file__)
    solver._update_constraint = FunctionType(codes["_update_constraint"], solver.__dict__, argdefs=(False,))
    util._KERNEL_CACHE = {}
    factories, kernels = [], []
    for name, values in ((adapter.NAMES[1], {"TRACK_CHANGES": False}),
                         (adapter.NAMES[3], {"dofs_per_thread": 50, "nv": 20})):
        fn = FunctionType(codes[name], solver.__dict__)
        wrapper = FunctionType(adapter._nested(ucodes["cache_kernel"], "wrapper"), util.__dict__, closure=(cell(fn),))
        wrapper.__wrapped__ = fn
        setattr(solver, name, wrapper)
        factories.append(wrapper)
        kc = adapter._nested(codes[name], "kernel")
        kf = FunctionType(kc, solver.__dict__, closure=tuple(cell(values[n]) for n in kc.co_freevars))
        kernels.append(NS(func=kf))
    init = NS(func=FunctionType(codes[adapter.NAMES[0]], solver.__dict__))
    dense = NS(func=FunctionType(codes[static.TARGET], solver.__dict__))
    kernels = (init, kernels[0], dense, kernels[1])
    solver.update_constraint_init_cost, solver.update_constraint_init_qfrc_constraint_dense = init, dense
    util._KERNEL_CACHE[(hash(False), hash(adapter.NAMES[1]))] = kernels[1]
    util._KERNEL_CACHE[(hash(20), hash(50), hash(adapter.NAMES[3]))] = kernels[3]
    monkeypatch.setitem(sys.modules, static.MODULE, solver)
    monkeypatch.setitem(sys.modules, adapter.UTIL_MODULE, util)
    bindings = {}
    for group, indices in enumerate(((0, 2), (1,), (3,))):
        module_hash = bytes([group + 1]) * 32
        module = NS(options=dict(block_dim=256, strip_hash=False, enable_backward=False),
                    execs={}, failed_builds=set(), get_module_hash=lambda block, h=module_hash: h)
        metadata, hooks = {}, {}
        for index in indices:
            k = kernels[index]
            k.module, k.adj, k.options = module, object(), {}
            name = stages.STAGES[index] + "_123abcff"
            k.get_mangled_name = lambda name=name: name
            metadata[name + "_cuda_kernel_forward_smem_bytes"] = 0
            hooks[k.adj] = NS(forward=6789 + index, backward=None, forward_smem_bytes=0, backward_smem_bytes=0)
        binary, meta = tmp_path / f"stage{group}.cubin", tmp_path / f"stage{group}.meta"
        binary.write_bytes(b"\x7fELF mock, never loaded " + bytes([group]))
        meta.write_bytes(json.dumps(metadata).encode())
        artifact = artifacts.retain_artifact(binary, meta, binary_size=binary.stat().st_size,
            binary_sha256=sha256(binary.read_bytes()).hexdigest(), metadata_size=meta.stat().st_size,
            metadata_sha256=sha256(meta.read_bytes()).hexdigest())
        executable = NS(handle=9876 + group, device=device, module_hash=module_hash, kernel_hooks=hooks, meta=metadata)
        module.execs[(device.context, 256)] = executable
        for index in indices:
            k = kernels[index]
            bindings[stages.STAGES[index]] = artifacts.bind_loaded_module(artifact, k, device, executable,
                hooks[k.adj], block_dim=256, expected_module_hash=module_hash)
    extra = {}
    for index, (name, shape, dtype, _) in enumerate(adapter.EXTRA_SPECS):
        a = wp.array()
        a.ptr, a.shape = 0x18000000 + index * 0x400000, shape
        a.dtype = solver.types.vec5 if dtype == "vec5" else getattr(wp, dtype)
        a.device, a.is_contiguous, a.requires_grad = device, True, False
        extra[name] = a
    model = NS(is_sparse=False, nv=20, opt=NS())
    data, ctx = NS(nworld=64, njmax=512, contact=NS(), efc=NS()), NS()
    roots = dict(model=model, data=data, contact=data.contact, efc=data.efc, context=ctx)
    for name, a in (c.arrays | extra).items():
        if name == "contact.nacon":
            data.nacon = a
            continue
        parts = name.split(".")
        obj = roots[parts[0]]
        for part in parts[1:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], a)
    runtime = NS(tape=None)
    inputs = dict(solver=solver, util=util, wp=wp, runtime=runtime, bindings=bindings,
                  model=model, data=data, context=ctx, stream=c.stream)
    observer = adapter.CostDispatchObserver(**inputs)
    return NS(observer=observer, inputs=inputs, solver=solver, util=util, wp=wp, runtime=runtime,
              model=model, data=data, context=ctx, kernels=kernels, bindings=bindings,
              stream=c.stream, events=events, actions=actions, original=launch, trace=c.trace)


def test_four_frozen_call_sites_and_raw_capture(case, example):
    case.observer.run()
    assert case.events == list(case.kernels) and case.wp.launch is case.original
    r = case.observer.record()
    assert r["call_site_lines"] == [2156, 2180, 2197, 2214]
    assert r["call_sites_observed"] and r["timing_changed_by_readback"]
    assert not any(r["qualification"].values())
    assert all(r[n] is False for n in ("explicit_load_provenance_authenticated", "native_gpu_execution_authenticated",
                                      "capture_origin_authenticated", "external_unit_retirement_authenticated"))
    assert all(case.observer.capture.raw(p) == example.arms["control"][p] for p in stages.PHASES)
    assert case.trace.count("copy") == 192 and case.trace.count("sync") == 17


def test_single_use_and_no_early_record(case):
    with pytest.raises(ValueError): case.observer.record()
    case.observer.run()
    with pytest.raises(ValueError, match="one-shot"): case.observer.run()


@pytest.mark.parametrize("name", ("solver", "bindings", "capture", "calls", "layouts", "original", "_sealed"))
def test_adapter_bindings_sealed(case, name):
    with pytest.raises(AttributeError, match="sealed"): setattr(case.observer, name, None)
    with pytest.raises(AttributeError, match="sealed"): delattr(case.observer, name)
    with pytest.raises(TypeError): case.observer.bindings["gauss"] = case.bindings["dense"]


@pytest.mark.parametrize("mutation", ("sparse", "nv", "worlds", "njmax", "tape", "stream", "dtype", "alias",
    "friction-overlap", "changed-shape", "replace-bank", "factory", "factory-body", "wrapper-body", "closure",
    "bool-closure", "cache", "cache-entry", "kernel-code", "kernel-func", "caller-default", "ceil", "hook", "metadata"))
def test_mutation_refused_before_delegate(case, mutation):
    if mutation == "sparse": case.model.is_sparse = True
    elif mutation == "nv": case.model.nv = 21
    elif mutation == "worlds": case.data.nworld = True
    elif mutation == "njmax": case.data.njmax = 511
    elif mutation == "tape": case.runtime.tape = object()
    elif mutation == "stream": case.stream.cuda_stream += 1
    elif mutation == "dtype": case.data.contact.friction.dtype = case.wp.float32
    elif mutation == "alias": case.context.changed_efc_count.ptr = case.data.nefc.ptr
    elif mutation == "friction-overlap": case.data.contact.friction.ptr = case.data.efc.J.ptr + 4
    elif mutation == "changed-shape": case.context.changed_efc_ids.shape = (64, 511)
    elif mutation == "replace-bank": case.data.qacc = case.data.qacc_smooth
    elif mutation == "factory": case.solver.update_constraint_efc = lambda x: case.kernels[1]
    elif mutation == "factory-body": case.solver.update_constraint_efc.__wrapped__.__code__ = (lambda x: None).__code__
    elif mutation == "wrapper-body":
        fn = case.solver.update_constraint_efc
        fn.__code__ = fn.__code__.replace(co_name="forged_wrapper")
    elif mutation == "closure": case.kernels[1].func.__closure__[0].cell_contents = True
    elif mutation == "bool-closure": case.kernels[3].func.__closure__[1].cell_contents = True
    elif mutation == "cache": case.util._KERNEL_CACHE = dict(case.util._KERNEL_CACHE)
    elif mutation == "cache-entry": case.util._KERNEL_CACHE[(hash(False), hash(adapter.NAMES[1]))] = case.kernels[3]
    elif mutation == "kernel-code": case.kernels[0].func.__code__ = (lambda *a: None).__code__
    elif mutation == "kernel-func": case.kernels[2].func = lambda *a: None
    elif mutation == "caller-default": case.solver._update_constraint.__defaults__ = (0,)
    elif mutation == "ceil": case.solver.ceil = lambda _: 1
    elif mutation == "hook": case.bindings["gauss"].hooks.forward += 1
    else: case.bindings["efc"].executable.meta["extra"] = 0
    with pytest.raises((ValueError, AttributeError)): case.observer.run()
    assert not case.events and case.wp.launch is case.original
    with pytest.raises(ValueError): case.observer.record()


@pytest.mark.parametrize("stage", stages.STAGES)
@pytest.mark.parametrize("mutation", ("kernel", "dimension", "boolean-dim", "input-order", "output", "device",
                                       "stream", "adjoint", "tape", "block", "adj-inputs", "max-blocks"))
def test_every_stage_launch_boundary(case, monkeypatch, stage, mutation):
    original = inspect.BoundArguments.apply_defaults
    index = stages.STAGES.index(stage)
    def corrupt(bound):
        original(bound)
        a = bound.arguments
        if a["kernel"] is not case.kernels[index]: return
        if mutation == "kernel": a["kernel"] = case.kernels[(index + 1) % 4]
        elif mutation == "dimension": a["dim"] = (64, 2)
        elif mutation == "boolean-dim": a["dim"] = True if index == 0 else (64, True)
        elif mutation == "input-order": a["inputs"] = list(reversed(a["inputs"]))
        elif mutation == "output": a["outputs"] = [object()]
        elif mutation == "device": a["device"] = case.stream.device
        elif mutation == "stream": a["stream"] = case.stream
        elif mutation == "adjoint": a["adjoint"] = True
        elif mutation == "tape": a["record_tape"] = False
        elif mutation == "block": a["block_dim"] = 128
        elif mutation == "adj-inputs": a["adj_inputs"] = [object()]
        else: a["max_blocks"] = True
    monkeypatch.setattr(inspect.BoundArguments, "apply_defaults", corrupt)
    with pytest.raises(ValueError): case.observer.run()
    assert len(case.events) == index and case.wp.launch is case.original
    with pytest.raises(ValueError): case.observer.record()


def test_recursive_real_caller_refused(case):
    case.actions.append(lambda _: case.solver._update_constraint(case.model, case.data, case.context))
    with pytest.raises(ValueError, match="single ordered"): case.observer.run()
    assert case.events == [case.kernels[0]] and case.wp.launch is case.original


def test_manual_launch_refused(case):
    case.actions.append(lambda _: case.wp.launch(case.kernels[1], dim=(64, 512)))
    with pytest.raises(ValueError): case.observer.run()
    assert len(case.events) == 1 and case.wp.launch is case.original


def test_failure_retains_before_and_releases_shared_hook(case):
    def fail(_): raise RuntimeError("synthetic launch failure")
    case.actions.append(fail)
    with pytest.raises(RuntimeError): case.observer.run()
    assert len(case.observer.capture.raw("init_cost.before")) == stages.PACKET_BYTES
    with pytest.raises(ValueError): case.observer.capture.raw("init_cost.after")
    assert case.wp.launch is case.original
    assert dense_guard._HOOK_LOCK.acquire(blocking=False)
    dense_guard._HOOK_LOCK.release()


def test_foreign_hook_not_overwritten(case):
    foreign = lambda *a, **k: None
    case.actions.append(lambda _: setattr(case.wp, "launch", foreign))
    with pytest.raises(ValueError, match="foreign launch hook preserved"): case.observer.run()
    assert case.wp.launch is foreign
    with pytest.raises(ValueError): case.observer.record()


def test_existing_dense_guard_lock_excludes_new_adapter(case):
    assert dense_guard._HOOK_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="exclusive shared"): case.observer.run()
    finally:
        dense_guard._HOOK_LOCK.release()
    assert not case.events and case.wp.launch is case.original
    with pytest.raises(ValueError, match="one-shot"): case.observer.run()


@pytest.mark.parametrize("mutation", ("missing-stage", "factory-code", "kernel-code", "factory-closure"))
def test_constructor_authenticates_not_just_holds_code(case, mutation):
    inputs = dict(case.inputs)
    if mutation == "missing-stage":
        inputs["bindings"] = {n: b for n, b in case.bindings.items() if n != "gauss"}
    elif mutation == "factory-code":
        case.solver.update_constraint_efc.__wrapped__.__code__ = (lambda x: None).__code__
    elif mutation == "kernel-code":
        case.kernels[0].func.__code__ = (lambda *a: None).__code__
    else:
        case.kernels[1].func.__closure__[0].cell_contents = True
    with pytest.raises(ValueError): adapter.CostDispatchObserver(**inputs)
    assert not case.events


def test_preconsumed_capture_cannot_be_relabelled_as_caller(case):
    case.observer.capture.bracket("init_cost", lambda: None)
    with pytest.raises(ValueError, match="fresh unused"): case.observer.run()
    assert not case.events and case.wp.launch is case.original


def test_post_delegate_mutation_fails_receipt(case):
    case.actions.append(lambda _: setattr(case.data.qacc, "ptr", 42))
    with pytest.raises(ValueError): case.observer.run()
    assert len(case.events) == 1 and case.wp.launch is case.original
    with pytest.raises(ValueError): case.observer.record()


def test_import_is_inert():
    code = "from mjlab_microduck import stance_solver_cost_dispatch; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
