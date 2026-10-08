"""Real frozen Python code with synthetic objects; no GPU execution evidence."""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import CodeType, FunctionType, ModuleType, SimpleNamespace as NS

import pytest
from mjlab_microduck import stance_cuda_artifact_binding as artifacts
from mjlab_microduck import stance_solver_dispatch_guard as guard
from mjlab_microduck import stance_solver_target_binding as static


class Device:
    context = 1234
    arch = 120
    is_cuda = True
    is_capturing = False
    max_shared_memory_per_block = 65536

    def __str__(self):
        return "cuda:0"


@pytest.fixture
def case(tmp_path, monkeypatch):
    device = Device()
    wp = NS(int32=object(), float32=object(), bool=object())
    stream = NS(device=device, cuda_stream=4567)
    events, actions, syncs = [], [], []

    def launch(kernel, dim, inputs=[], outputs=[], adj_inputs=[], adj_outputs=[],
               device=None, stream=None, adjoint=False, record_tape=True,
               record_cmd=False, max_blocks=0, block_dim=256):
        events.append(kernel)
        for action in actions:
            action(kernel)

    def get_device():
        return device

    def get_stream(value):
        assert value is device
        return stream

    def synchronize_stream(value):
        assert value is stream
        syncs.append(value)

    wp.launch, wp.get_device, wp.get_stream, wp.synchronize_stream = launch, get_device, get_stream, synchronize_stream
    solver = ModuleType(static.MODULE)
    path = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
    solver.__file__ = str(path)
    solver.wp = wp
    codes = {c.co_name: c for c in compile(path.read_bytes(), str(path), "exec", dont_inherit=True, optimize=0).co_consts
             if isinstance(c, CodeType)}
    solver._update_constraint = FunctionType(codes["_update_constraint"], solver.__dict__, argdefs=(False,))
    kernel_func = FunctionType(codes[static.TARGET], solver.__dict__)
    symbol = static.TARGET + "_123abcff_cuda_kernel_forward"
    module_hash = b"H" * 32
    module = NS(options=dict(block_dim=256, strip_hash=False, enable_backward=False),
                execs={}, failed_builds=set(), get_module_hash=lambda b: module_hash)
    kernel = NS(func=kernel_func, module=module, adj=object(), options={},
                get_mangled_name=lambda: symbol.removesuffix("_cuda_kernel_forward"))
    setattr(solver, static.TARGET, kernel)
    solver.update_constraint_init_cost = object()
    solver.update_constraint_efc = lambda changes: object()
    solver.update_constraint_gauss_cost = lambda nv, threads: object()
    solver.ceil = lambda x: int(x + 0.9999)
    monkeypatch.setitem(sys.modules, static.MODULE, solver)
    binary, meta = tmp_path / "target.cubin", tmp_path / "target.meta"
    binary.write_bytes(b"\x7fELF synthetic mock only, never loaded")
    meta.write_bytes(json.dumps({symbol + "_smem_bytes": 0}).encode())
    artifact = artifacts.retain_artifact(binary, meta,
        binary_size=binary.stat().st_size, binary_sha256=sha256(binary.read_bytes()).hexdigest(),
        metadata_size=meta.stat().st_size, metadata_sha256=sha256(meta.read_bytes()).hexdigest())
    hooks = NS(forward=6789, backward=None, forward_smem_bytes=0, backward_smem_bytes=0)
    executable = NS(handle=9876, device=device, module_hash=module_hash,
                    kernel_hooks={kernel.adj: hooks}, meta={symbol + "_smem_bytes": 0})
    module.execs[(device.context, 256)] = executable
    binding = artifacts.bind_loaded_module(artifact, kernel, device, executable, hooks,
        block_dim=256, expected_module_hash=module_hash)
    def array(shape, dtype, pointer):
        return NS(ptr=pointer, shape=shape, dtype=dtype, device=device,
                  is_contiguous=True, requires_grad=False)
    nefc = array((64,), wp.int32, 0x100000)
    jacobian = array((64, 512, 20), wp.float32, 0x200000)
    force = array((64, 512), wp.float32, 0x500000)
    done = array((64,), wp.bool, 0x600000)
    output = array((64, 20), wp.float32, 0x700000)
    model = NS(is_sparse=False, nv=20, opt=NS(impratio_invsqrt=object()))
    efc = NS(J=jacobian, force=force, type=object(), id=object(), D=object(),
             frictionloss=object(), state=object(), Ma=object())
    data = NS(nworld=64, njmax=512, nefc=nefc, efc=efc, qfrc_constraint=output,
              ne=object(), nf=object(), nacon=object(), contact=NS(friction=object(), dim=object(), efc_address=object()),
              qacc=object(), qfrc_smooth=object(), qacc_smooth=object())
    context = NS(done=done, cost=object(), gauss=object(), prev_cost=object(), Jaref=object(),
                 changed_efc_ids=object(), changed_efc_count=object())
    runtime = NS(tape=None)
    observer = guard.DenseSolverDispatchGuard(solver=solver, wp=wp, runtime=runtime,
        binding=binding, model=model, data=data, context=context, stream=stream)
    return NS(observer=observer, wp=wp, solver=solver, kernel=kernel, binding=binding,
              model=model, data=data, context=context, runtime=runtime, stream=stream,
              events=events, actions=actions, syncs=syncs, original=launch)


def test_frozen_caller_delegates_one_target_and_restores(case):
    case.observer.run()
    assert len(case.events) == 4 and sum(k is case.kernel for k in case.events) == 1
    assert case.syncs == [case.stream]
    assert case.wp.launch is case.original
    record = case.observer.record()
    assert record["observed_target_calls"] == 1
    assert not any(record["flags"].values())
    assert record["numerical_acceptance"] is record["driver_loaded_code_observed"] is False
    assert record["binding"]["loaded_binary_bytes_observed"] is False


def test_single_use_and_no_early_receipt(case):
    with pytest.raises(ValueError):
        case.observer.record()
    case.observer.run()
    with pytest.raises(ValueError, match="one-shot"):
        case.observer.run()


@pytest.mark.parametrize("mutation", ["sparse", "worlds", "nv", "njmax", "pointer", "alias", "overlap", "dtype", "grad", "shape",
    "device", "tape", "stream", "new-stream", "kernel", "caller", "caller-default", "dependency", "metadata", "forward"])
def test_preflight_mutations_do_not_dispatch(case, mutation):
    if mutation == "sparse": case.model.is_sparse = True
    elif mutation == "worlds": case.data.nworld = True
    elif mutation == "nv": case.model.nv = 21
    elif mutation == "njmax": case.data.njmax = 511
    elif mutation == "pointer": case.data.efc.J.ptr += 1
    elif mutation == "alias": case.data.qfrc_constraint.ptr = case.data.nefc.ptr
    elif mutation == "overlap": case.data.qfrc_constraint.ptr = case.data.efc.J.ptr + 4
    elif mutation == "dtype": case.data.efc.J.dtype = object()
    elif mutation == "grad": case.data.efc.force.requires_grad = True
    elif mutation == "shape": case.context.done.shape = (1,)
    elif mutation == "device": case.data.nefc.device = Device()
    elif mutation == "tape": case.runtime.tape = object()
    elif mutation == "stream": case.stream.cuda_stream = 0
    elif mutation == "new-stream": case.stream.cuda_stream += 1
    elif mutation == "kernel": setattr(case.solver, static.TARGET, object())
    elif mutation == "caller": case.solver._update_constraint = lambda *args: None
    elif mutation == "caller-default": case.solver._update_constraint.__defaults__ = (True,)
    elif mutation == "dependency": case.solver.update_constraint_efc = lambda c: object()
    elif mutation == "metadata": case.binding.executable.meta["extra"] = 0
    else: case.binding.hooks.forward += 1
    with pytest.raises(ValueError):
        case.observer.run()
    assert case.events == [] and case.wp.launch is case.original


def test_manual_target_launch_inside_owned_scope_refused(case):
    def manual(kernel):
        if kernel is not case.kernel:
            case.wp.launch(case.kernel, dim=(64, 20))
    case.actions.append(manual)
    with pytest.raises(ValueError, match="direct target call"):
        case.observer.run()
    assert not any(k is case.kernel for k in case.events)
    assert case.wp.launch is case.original


def test_duplicate_actual_caller_target_refused(case):
    def duplicate(kernel):
        if kernel is case.kernel:
            case.observer.caller(case.model, case.data, case.context)
    case.actions.append(duplicate)
    with pytest.raises(ValueError, match="single direct"):
        case.observer.run()
    assert sum(k is case.kernel for k in case.events) == 1
    assert case.wp.launch is case.original


def test_mutation_after_original_target_detected(case):
    case.actions.append(lambda k: setattr(case.data.efc.force, "ptr", 42) if k is case.kernel else None)
    with pytest.raises(ValueError):
        case.observer.run()
    with pytest.raises(ValueError):
        case.observer.record()
    assert case.wp.launch is case.original


def test_foreign_hook_preserved_on_failure(case):
    foreign = lambda *a, **k: None
    case.actions.append(lambda k: setattr(case.wp, "launch", foreign))
    with pytest.raises(ValueError, match="foreign launch hook preserved"):
        case.observer.run()
    assert case.wp.launch is foreign


@pytest.mark.parametrize("mutation", ["dimensions", "boolean-dimension", "input-order", "boolean-njmax", "outputs",
    "device", "stream", "adjoint", "record-cmd", "record-tape", "block", "adj-inputs", "max-blocks"])
def test_wrapper_boundary_arguments_refused_before_target_delegate(case, monkeypatch, mutation):
    # Mutate a synthetic BoundArguments carrier, not the frozen caller source.
    # This protects the validation branch independently of the source/code fence.
    original = inspect.BoundArguments.apply_defaults
    def corrupt(bound):
        original(bound)
        args = bound.arguments
        if args["kernel"] is not case.kernel:
            return
        if mutation == "dimensions": args["dim"] = (64, 21)
        elif mutation == "boolean-dimension": args["dim"] = (True, 20)
        elif mutation == "input-order": args["inputs"] = [case.data.nefc, case.data.efc.force, case.data.efc.J, 512, case.context.done]
        elif mutation == "boolean-njmax": args["inputs"] = [case.data.nefc, case.data.efc.J, case.data.efc.force, True, case.context.done]
        elif mutation == "outputs": args["outputs"] = [object()]
        elif mutation == "device": args["device"] = case.binding.device
        elif mutation == "stream": args["stream"] = case.stream
        elif mutation == "adjoint": args["adjoint"] = True
        elif mutation == "record-cmd": args["record_cmd"] = True
        elif mutation == "record-tape": args["record_tape"] = False
        elif mutation == "block": args["block_dim"] = 128
        elif mutation == "adj-inputs": args["adj_inputs"] = [object()]
        else: args["max_blocks"] = 1
    monkeypatch.setattr(inspect.BoundArguments, "apply_defaults", corrupt)
    with pytest.raises(ValueError):
        case.observer.run()
    assert not any(k is case.kernel for k in case.events)
    assert case.wp.launch is case.original


def test_original_launch_exception_restores_owned_hook_and_refuses_receipt(case):
    def failure(kernel):
        raise RuntimeError("synthetic delegated launch failure")
    case.actions.append(failure)
    with pytest.raises(RuntimeError, match="delegated launch failure"):
        case.observer.run()
    assert case.wp.launch is case.original
    with pytest.raises(ValueError):
        case.observer.record()
    assert guard._HOOK_LOCK.acquire(blocking=False)
    guard._HOOK_LOCK.release()


def test_import_initializes_no_device_packages():
    code = "from mjlab_microduck import stance_solver_dispatch_guard; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
