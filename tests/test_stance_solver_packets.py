"""Synthetic stream/copy/dispatch packets only, never native GPU evidence."""
from hashlib import sha256
import struct
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_stance_solver_dispatch_guard import case as dispatch_case
from mjlab_microduck import stance_solver_dispatch_guard as guard
from mjlab_microduck import stance_solver_packets as packets


class Array:
    def numpy(self):
        return self.values


@pytest.fixture
def case(dispatch_case):
    c = dispatch_case
    c.trace, c.copy_actions = [], []
    wp = c.wp
    wp.array = Array
    cpu = NS(is_cpu=True, is_cuda=False)
    destinations = []
    counter = [0]

    def empty(*, shape, dtype, device, requires_grad, pinned):
        assert device == "cpu" and requires_grad is False and pinned is True
        counter[0] += 1
        a = Array()
        a.shape, a.dtype, a.device = shape, dtype, cpu
        a.ptr = 0x100000000 + counter[0] * 0x400000
        a.is_contiguous, a.requires_grad, a.pinned = True, False, True
        a.values = np.empty(shape, dtype={wp.int32: "<i4", wp.float32: "<f4", wp.bool: "?"}[dtype])
        destinations.append(a)
        return a

    def copy(dst, src, *, stream):
        assert stream is c.stream
        c.trace.append(("copy", src.name))
        for action in c.copy_actions:
            action(dst, src)
        np.copyto(dst.values, src.values)

    def synchronize_stream(stream):
        assert stream is c.stream
        c.trace.append(("sync",))

    wp.empty, wp.copy, wp.synchronize_stream = empty, copy, synchronize_stream
    old = (c.data.nefc, c.data.efc.J, c.data.efc.force, c.context.done, c.data.qfrc_constraint)
    arrays = []
    for previous, (name, shape, dtype, wire, _) in zip(old, packets.SPECS):
        a = Array()
        a.__dict__.update(vars(previous))
        a.values = np.zeros(shape, dtype=wire)
        a.name, a.pinned = name, False
        arrays.append(a)
    c.data.nefc, c.data.efc.J, c.data.efc.force, c.context.done, c.data.qfrc_constraint = arrays
    c.data.nefc.values[:] = 1
    c.data.efc.J.values[:, 0, :] = 1
    c.data.efc.force.values[:, 0] = 2
    c.context.done.values[1] = True
    c.data.qfrc_constraint.values[:] = 7

    def advance(kernel):
        if len(c.events) == 2:
            c.trace.append(("efc",))
            c.data.efc.force.values[:, 0] = 3
        elif kernel is c.kernel:
            c.trace.append(("target",))
            c.data.qfrc_constraint.values[~c.context.done.values] = 3
        elif len(c.events) == 4:
            c.trace.append(("gauss",))
            c.data.qfrc_constraint.values[:] += 100  # Prove post packet precedes later code.
    c.actions.append(advance)
    c.observer = guard.DenseSolverDispatchGuard(solver=c.solver, wp=wp, runtime=c.runtime,
        binding=c.binding, model=c.model, data=c.data, context=c.context, stream=c.stream)
    c.capture = packets.DenseSolverPacketCapture(wp=wp, data=c.data, context=c.context, stream=c.stream)
    c.arrays, c.destinations = arrays, destinations
    return c


def field(capture, phase, name):
    row = next(r for r in capture.record()["packets"][phase]["fields"] if r["name"] == name)
    return capture.raw(phase)[row["offset"]:row["offset"] + row["bytes"]]


def test_actual_literal_launch_bracket_not_whole_caller(case):
    c = case
    c.observer.run(capture=c.capture)
    copies = [("copy", name) for name, *_ in packets.SPECS]
    assert c.trace == [("efc",), ("sync",), *copies, ("sync",), ("target",),
                       ("sync",), *copies, ("sync",), ("gauss",), ("sync",)]
    assert np.frombuffer(field(c.capture, "before", "force"), dtype="<f4")[0] == 3
    assert np.frombuffer(field(c.capture, "after", "qfrc_constraint"), dtype="<f4")[0] == 3
    assert c.data.qfrc_constraint.values[0, 0] == 103
    output = np.frombuffer(field(c.capture, "after", "qfrc_constraint"), dtype="<f4").reshape(64, 20)
    assert np.all(output[1] == 7)  # done-world preexisting sentinel preserved.
    result = c.observer.record()
    pair = result["packets"]
    assert pair["complete_pair"] is pair["input_bytes_unchanged"] is True
    assert pair["packet_bytes"] == 2757952 and pair["pair_bytes"] == 5515904
    assert not any(pair["flags"].values()) and not result["numerical_acceptance"]
    assert not pair["driver_loaded_code_observed"]
    for phase in ("before", "after"):
        raw = c.capture.raw(phase)
        assert type(raw) is bytes and len(raw) == packets.PACKET_BYTES
        assert pair["packets"][phase]["sha256"] == sha256(raw).hexdigest()
        assert sum(r["bytes"] for r in pair["packets"][phase]["fields"]) == len(raw)
    assert c.wp.launch is c.original


def test_raw_inactive_rows_preserve_special_bits(case):
    c = case
    special = struct.pack("<IIII", 0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000)
    c.data.efc.J.values.view("<u4")[0, 511, :4] = np.frombuffer(special, dtype="<u4")
    c.observer.run(capture=c.capture)
    offset = (511 * 20) * 4
    assert field(c.capture, "before", "J")[offset:offset + 16] == special
    assert field(c.capture, "after", "J")[offset:offset + 16] == special


def test_raw_boolean_carrier_is_not_normalized(case):
    case.context.done.values.view("u1")[2] = 0x80
    case.observer.run(capture=case.capture)
    assert field(case.capture, "before", "done")[2] == 0x80
    assert field(case.capture, "after", "done")[2] == 0x80


def test_changed_input_bits_are_reported_not_accepted(case):
    case.actions.append(lambda k: case.data.efc.force.values.__setitem__((0, 0), 9)
                        if k is case.kernel else None)
    case.observer.run(capture=case.capture)
    assert case.capture.record()["input_bytes_unchanged"] is False
    assert not any(case.capture.record()["flags"].values())


def test_delegated_target_failure_preserves_pre_packet_only(case):
    def fail(kernel):
        if kernel is case.kernel: raise RuntimeError("synthetic target fault")
    case.actions.append(fail)
    with pytest.raises(RuntimeError, match="target fault"):
        case.observer.run(capture=case.capture)
    assert case.wp.launch is case.original and case.capture.failed
    assert len(case.capture.raw("before")) == packets.PACKET_BYTES
    with pytest.raises(ValueError): case.capture.raw("after")
    with pytest.raises(ValueError): case.observer.record()


def test_foreign_hook_during_post_copy_is_preserved(case):
    foreign = lambda *a, **k: None
    def change(dst, src):
        if any(k is case.kernel for k in case.events): case.wp.launch = foreign
    case.copy_actions.append(change)
    with pytest.raises(ValueError, match="foreign launch hook preserved"):
        case.observer.run(capture=case.capture)
    assert case.wp.launch is foreign and case.capture.failed
    assert len(case.capture.raw("before")) == packets.PACKET_BYTES
    with pytest.raises(ValueError): case.observer.record()


def test_receipt_is_detached(case):
    case.observer.run(capture=case.capture)
    receipt = case.capture.record()
    receipt["packets"]["before"]["fields"][0]["shape"].append(9)
    receipt["flags"]["training_authorized"] = True
    assert case.capture.record()["packets"]["before"]["fields"][0]["shape"] == [64]
    assert not any(case.capture.record()["flags"].values())


def test_early_and_repeated_capture_refused(case):
    with pytest.raises(ValueError):
        case.capture.record()
    with pytest.raises(ValueError):
        case.capture.raw("before")
    case.observer.run(capture=case.capture)
    with pytest.raises(ValueError):
        case.observer.run(capture=case.capture)
    with pytest.raises(ValueError):
        case.capture._snapshot("before")


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("mutation", ["copy-failure", "shape", "dtype", "contiguous", "size", "source-replace", "source-ptr"])
def test_capture_failures_refuse_receipt_restore_hook(case, phase, mutation):
    c = case
    def break_copy(dst, src):
        after = any(row[0] == "target" for row in c.trace)
        if after != (phase == "after") or src is not c.data.efc.J:
            return
        if mutation == "copy-failure":
            raise RuntimeError("synthetic copy failure")
        elif mutation == "shape": dst.values = np.zeros((1,), dtype="<f4")
        elif mutation == "dtype": dst.values = dst.values.astype("<f8")
        elif mutation == "contiguous": dst.values = dst.values[:, :, ::-1]
        elif mutation == "size": dst.shape = (1,)
        elif mutation == "source-replace":
            c.data.efc.J = Array()
            c.data.efc.J.__dict__.update(vars(src))
        else: src.ptr += 4
    c.copy_actions.append(break_copy)
    with pytest.raises((ValueError, RuntimeError)):
        c.observer.run(capture=c.capture)
    assert c.wp.launch is c.original and c.capture.failed
    with pytest.raises(ValueError): c.observer.record()
    with pytest.raises(ValueError): c.capture.record()
    if phase == "before":
        assert not any(k is c.kernel for k in c.events)
    else:
        assert sum(k is c.kernel for k in c.events) == 1
        assert len(c.capture.raw("before")) == packets.PACKET_BYTES
        assert not any(row[0] == "gauss" for row in c.trace)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_same_stream_sync_failure_is_consumed(case, phase):
    c = case
    def sync(stream):
        if (any(k is c.kernel for k in c.events)) == (phase == "after"):
            raise RuntimeError("synthetic sync fault")
    c.wp.synchronize_stream = sync
    # Bind this synthetic entry before guard and capture construction.
    c.observer = guard.DenseSolverDispatchGuard(solver=c.solver, wp=c.wp, runtime=c.runtime,
        binding=c.binding, model=c.model, data=c.data, context=c.context, stream=c.stream)
    c.capture = packets.DenseSolverPacketCapture(wp=c.wp, data=c.data, context=c.context, stream=c.stream)
    with pytest.raises(RuntimeError, match="sync fault"):
        c.observer.run(capture=c.capture)
    assert c.wp.launch is c.original and c.capture.failed
    with pytest.raises(ValueError): c.capture.record()


@pytest.mark.parametrize("mutation", ["copy-entry", "empty-entry", "numpy-entry", "numpy-code", "staging-ptr", "staging-alias", "staging-device", "staging-pinned"])
def test_held_readback_entries_and_staging_mutations_refused(case, monkeypatch, mutation):
    c = case
    def mutate(kernel):
        if len(c.events) != 1: return
        if mutation == "copy-entry": c.wp.copy = lambda *a, **k: None
        elif mutation == "empty-entry": c.wp.empty = lambda *a, **k: None
        elif mutation == "numpy-entry": monkeypatch.setattr(Array, "numpy", lambda self: self.values)
        elif mutation == "numpy-code": monkeypatch.setattr(Array.numpy, "__code__", (lambda self: self.values).__code__)
        elif mutation == "staging-ptr": c.destinations[0].ptr += 1
        elif mutation == "staging-alias": c.destinations[0].ptr = c.destinations[1].ptr
        elif mutation == "staging-device": c.destinations[0].device = c.stream.device
        else: c.destinations[0].pinned = False
    c.actions.append(mutate)
    with pytest.raises(ValueError): c.observer.run(capture=c.capture)
    assert not any(k is c.kernel for k in c.events) and c.wp.launch is c.original
    with pytest.raises(ValueError): c.capture.record()


def test_duck_typed_or_subclass_capture_refused_before_dispatch(case):
    class Other(packets.DenseSolverPacketCapture): pass
    for capture in (NS(), object(), Other(wp=case.wp, data=case.data, context=case.context, stream=case.stream)):
        with pytest.raises(ValueError, match="exact optional"):
            case.observer.run(capture=capture)
    assert not case.events and not case.observer.used


def test_wrong_data_capture_refused(case):
    other = packets.DenseSolverPacketCapture(wp=case.wp, data=NS(**vars(case.data)), context=case.context, stream=case.stream)
    with pytest.raises(ValueError, match="bound to exact"):
        case.observer.run(capture=other)
    assert not case.events and case.wp.launch is case.original


@pytest.mark.parametrize("phase", ["before", "after"])
def test_retained_raw_mutation_refuses_receipt(case, phase):
    case.observer.run(capture=case.capture)
    raw = case.capture.packets[phase]
    case.capture.packets[phase] = b"X" + raw[1:]
    with pytest.raises(ValueError): case.capture.raw(phase)
    with pytest.raises(ValueError): case.capture.record()
    with pytest.raises(ValueError): case.observer.record()


def test_other_thread_run_refused_without_dispatch(case):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        attempt = pool.submit(case.observer.run, capture=case.capture)
        with pytest.raises(ValueError, match="owned dispatch thread"):
            attempt.result(timeout=5)
    assert not case.events and case.wp.launch is case.original


def test_import_initializes_no_device_packages():
    code = "from mjlab_microduck import stance_solver_packets; import sys; assert not any(n.split('.')[0] in {'numpy','warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
