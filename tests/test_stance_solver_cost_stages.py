"""Synthetic packet/readback tests, never native dispatch evidence."""
from fractions import Fraction
from hashlib import sha256
import os
import struct
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_stance_solver_scratch import _decoded
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_gauss_audit as gauss


def anchors(arms):
    return {arm: {phase: dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
                  for phase, raw in banks.items()} for arm, banks in arms.items()}


@pytest.fixture(scope="module")
def example():
    packet, arms = _decoded(), {}
    for arm, initial in stages.initial_banks(packet).items():
        fields, banks = stages.unpack_bank(initial), {}
        for stage in stages.STAGES:
            banks[stage + ".before"] = stages.pack_bank(fields)
            if stage == "init_cost":
                fields["context.prev_cost"] = fields["context.cost"]
                fields["context.gauss"] = fields["context.cost"] = bytes(256)
            elif stage == "efc":
                force, state = bytearray(131072), bytearray(131072)
                for world in range(64):
                    struct.pack_into("<f", force, world * 512 * 4, 2.0)
                    struct.pack_into("<i", state, world * 512 * 4, 1)
                fields["efc.force"], fields["efc.state"] = bytes(force), bytes(state)
                # Opaque atomic cost intentionally differs; not a row oracle.
                fields["context.cost"] = struct.pack("<64f", *([1.25 if arm == "reference" else 1.5] * 64))
            elif stage == "dense":
                fields["data.qfrc_constraint"] = struct.pack("<f", 2.0) + bytes(5116)
            else:
                contribution = 0.0 if arm == "reference" else 10.0
                costs = struct.unpack("<64f", fields["context.cost"])
                fields["context.gauss"] = struct.pack("<64f", *([contribution] * 64))
                fields["context.cost"] = struct.pack("<64f", *(gauss.round32(Fraction(x) + Fraction(contribution)) for x in costs))
            banks[stage + ".after"] = stages.pack_bank(fields)
        arms[arm] = banks
    return NS(packet=packet, arms=arms, anchors=anchors(arms))


def change_from(banks, phase, name, raw):
    result = dict(banks)
    for key in stages.PHASES[stages.PHASES.index(phase):]:
        fields = stages.unpack_bank(result[key])
        fields[name] = raw
        result[key] = stages.pack_bank(fields)
    return result


def test_complete_contract(example):
    result = stages.receive_banks(example.arms, example.anchors, example.packet)
    assert result["decision"] == "caller-cost-stage-packet-contract-only"
    assert result["packet_bytes"] == 3729992
    assert result["total_bytes"] == 59679872 < stages.MAX_TOTAL_BYTES
    assert not any(result["qualification"].values())
    assert not result["capture_origin_authenticated"] and not result["gpu_dispatch_authenticated"]
    for arm, row in result["arms"].items():
        assert row["setup_reset"] and row["prior_cost_transferred"] and row["gauss_cost_addition"]
        assert not row["efc_cost_numerically_qualified"]
        assert row["dense"]["mismatches"] == row["gauss"]["outside_bound"] == 0
        assert row["gauss"]["rows"][0]["captured"] == (10.0 if arm == "control" else 0.0)


def test_literal_control_only_seven_fields(example):
    initial = stages.initial_banks(example.packet)
    a, b = map(stages.unpack_bank, (initial["reference"], initial["control"]))
    overrides = stages.control_fields()
    assert set(overrides) == set(gauss.NAMES) | {"context.gauss", "context.cost", "context.prev_cost"}
    assert all(b[name] == overrides.get(name, a[name]) for name in stages.ORDER)
    assert stages.control_manifest()["synthetic_only"]
    assert not any(stages.control_manifest()["qualification"].values())


@pytest.mark.parametrize("stage,name", [("init_cost", "efc.force"), ("efc", "context.gauss"),
    ("dense", "context.prev_cost"), ("gauss", "data.qfrc_constraint"),
    ("init_cost", "data.nefc"), ("efc", "data.qacc"), ("dense", "efc.state"), ("gauss", "efc.force")])
def test_disallowed_writes(example, stage, name):
    raw = stages.unpack_bank(example.arms["control"][stage + ".after"])[name]
    banks = change_from(example.arms["control"], stage + ".after", name, b"\xff" + raw[1:])
    with pytest.raises(ValueError, match="non-output"):
        stages.analyze_arm(banks, arm="control")


def test_adjacent_continuity(example):
    banks = dict(example.arms["control"])
    fields = stages.unpack_bank(banks["gauss.before"])
    fields["context.cost"] = struct.pack("<64f", *([3.0] * 64))
    banks["gauss.before"] = stages.pack_bank(fields)
    with pytest.raises(ValueError, match="continuity"):
        stages.analyze_arm(banks, arm="control")


@pytest.mark.parametrize("name", ("context.gauss", "context.cost"))
def test_negative_zero_reset(example, name):
    banks = change_from(example.arms["control"], "init_cost.after", name, struct.pack("<64I", *([0x80000000] * 64)))
    with pytest.raises(ValueError, match="positive-zero"):
        stages.analyze_arm(banks, arm="control")


def test_prior_cost_exact_transfer(example):
    banks = change_from(example.arms["control"], "init_cost.after", "context.prev_cost", bytes(256))
    with pytest.raises(ValueError, match="prior-cost"):
        stages.analyze_arm(banks, arm="control")


@pytest.mark.parametrize("value", (-1.0, float("nan"), float("inf")))
def test_bad_efc_cost(example, value):
    banks = change_from(example.arms["control"], "efc.after", "context.cost", struct.pack("<64f", *([value] * 64)))
    with pytest.raises(ValueError, match="finite|nonnegative"):
        stages.analyze_arm(banks, arm="control")


@pytest.mark.parametrize("name", ("efc.force", "efc.state"))
def test_inactive_tail(example, name):
    raw = stages.unpack_bank(example.arms["control"]["efc.after"])[name]
    banks = change_from(example.arms["control"], "efc.after", name, raw[:-1] + b"\xff")
    with pytest.raises(ValueError, match="inactive EFC tail"):
        stages.analyze_arm(banks, arm="control")


@pytest.mark.parametrize("phase,name,raw,message", [
    ("dense.after", "data.qfrc_constraint", bytes(5120), "dense arithmetic"),
    ("gauss.after", "context.gauss", struct.pack("<64f", *([11.0] * 64)), "Gauss arithmetic"),
    ("gauss.after", "context.cost", bytes(256), "immediate pre-stage"),
    ("efc.after", "context.cost", struct.pack("<64f", *([2.0**30] * 64)), "nonvacuous"),
    ("init_cost.before", "context.done", b"\1" + bytes(63), "all-active"),
    ("init_cost.before", "data.nefc", bytes(256), "bounded active"),
    ("init_cost.before", "data.ne", struct.pack("<64i", *([47] * 64)), "bounded active"),
    ("init_cost.before", "efc.type", struct.pack("<i", 7) + bytes(131068), "elliptic"),
])
def test_wrong_outputs_and_domain(example, phase, name, raw, message):
    banks = change_from(example.arms["control"], phase, name, raw)
    with pytest.raises(ValueError, match=message):
        stages.analyze_arm(banks, arm="control")


@pytest.mark.parametrize("phase", stages.PHASES)
def test_missing_phase(example, phase):
    banks = dict(example.arms["control"])
    del banks[phase]
    with pytest.raises(ValueError, match="complete declared"):
        stages.analyze_arm(banks, arm="control")


@pytest.mark.parametrize("which", ("missing", "extra", "mutable", "short"))
def test_closed_packet_layout(example, which):
    fields = stages.unpack_bank(example.arms["control"]["init_cost.before"])
    if which == "missing": del fields["data.qacc"]
    elif which == "extra": fields["foreign"] = b""
    elif which == "mutable": fields["data.qacc"] = bytearray(fields["data.qacc"])
    else: fields["data.qacc"] = fields["data.qacc"][:-1]
    with pytest.raises(ValueError): stages.pack_bank(fields)


@pytest.mark.parametrize("arm", ("reference", "control"))
def test_all_hashes_before_any_decode(example, monkeypatch, arm):
    arms = {key: dict(value) for key, value in example.arms.items()}
    arms[arm]["gauss.after"] += b"x"
    monkeypatch.setattr(stages, "unpack_bank", lambda _: pytest.fail("decoded before all hashes verified"))
    with pytest.raises(ValueError, match="whole external"):
        stages.receive_banks(arms, example.anchors, example.packet)


def test_reanchored_wrong_initial_bank(example):
    arms = {key: dict(value) for key, value in example.arms.items()}
    raw = arms["control"]["init_cost.before"]
    arms["control"]["init_cost.before"] = b"\xff" + raw[1:]
    with pytest.raises(ValueError, match="historical restoration"):
        stages.receive_banks(arms, anchors(arms), example.packet)


def test_unsealed_initial_packet():
    with pytest.raises(ValueError, match="sealed authenticated"):
        stages.initial_banks(NS(raw=b"", sha256=sha256(b"").hexdigest(), _sealed=True))


def test_import_is_inert():
    code = "import sys; from mjlab_microduck import stance_solver_cost_stages; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp','numpy'} for n in sys.modules)"
    subprocess.run((sys.executable, "-c", code), check=True, timeout=15,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


class Array:
    def numpy(self):
        assert self.device.is_cpu, "GPU convenience readback forbidden"
        return self.values


@pytest.fixture
def capture_case(example):
    device = NS(is_cuda=True, is_cpu=False, arch=120, context=1234)
    cpu = NS(is_cuda=False, is_cpu=True)
    wp = NS(array=Array, float32=object(), int32=object(), bool=object())
    stream = NS(device=device, cuda_stream=5678)
    trace, allocations, arrays = [], [], {}
    first = stages.unpack_bank(example.arms["control"]["init_cost.before"])
    for index, (name, shape, host_shape, dtype) in enumerate(stages.SPECS):
        a = Array()
        a.ptr, a.shape, a.dtype, a.device = 0x1000000 + index * 0x400000, shape, getattr(wp, dtype), device
        a.is_contiguous, a.requires_grad = True, False
        wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        a.values = np.frombuffer(first[name], dtype=wire).reshape(host_shape).copy()
        arrays[name] = a

    def empty(*, shape, dtype, device, requires_grad, pinned):
        assert device == "cpu" and requires_grad is False and pinned is True
        a = Array()
        a.ptr = 0x20000000 + len(allocations) * 0x400000
        a.shape, a.dtype, a.device = shape, dtype, cpu
        a.is_contiguous, a.requires_grad, a.pinned = True, False, True
        a.values = np.empty(shape, dtype={wp.float32: "<f4", wp.int32: "<i4", wp.bool: "|b1"}[dtype])
        allocations.append(a)
        return a

    def copy(dst, src, *, stream):
        assert stream.cuda_stream == 5678
        trace.append("copy")
        np.copyto(dst.values, src.values)

    def sync(stream):
        assert stream.cuda_stream == 5678
        trace.append("sync")

    def guard(): trace.append("guard")

    wp.empty, wp.copy, wp.synchronize_stream = empty, copy, sync
    capture = stages.CostStageCapture(wp=wp, arrays=arrays, stream=stream, guard=guard)

    def dispatch(stage):
        trace.append(stage)
        fields = stages.unpack_bank(example.arms["control"][stage + ".after"])
        for name, _, host, dtype in stages.SPECS:
            wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
            np.copyto(arrays[name].values, np.frombuffer(fields[name], dtype=wire).reshape(host))
        return stage + "-result"

    return NS(capture=capture, wp=wp, arrays=arrays, stream=stream, trace=trace, dispatch=dispatch)


def test_ordered_same_stream_cpu_capture(capture_case, example):
    c = capture_case
    for stage in stages.STAGES:
        assert c.capture.bracket(stage, lambda stage=stage: c.dispatch(stage)) == stage + "-result"
    record = c.capture.record()
    assert record["stages"] == list(stages.STAGES) and not record["kernel_dispatch_authenticated"]
    assert not any(record["qualification"].values())
    assert c.trace.count("sync") == 16 and c.trace.count("copy") == 192
    assert all(c.capture.raw(phase) == example.arms["control"][phase] for phase in stages.PHASES)
    with pytest.raises(ValueError, match="one-shot"): c.capture.bracket("gauss", lambda: None)


def test_failure_preserves_only_before(capture_case):
    def fail(): raise RuntimeError("synthetic dispatch failure")
    c = capture_case.capture
    with pytest.raises(RuntimeError, match="dispatch failure"): c.bracket("init_cost", fail)
    assert len(c.raw("init_cost.before")) == stages.PACKET_BYTES
    with pytest.raises(ValueError): c.raw("init_cost.after")
    with pytest.raises(ValueError): c.record()
    with pytest.raises(ValueError): c.bracket("init_cost", lambda: None)


def test_bad_order_consumes_capture(capture_case):
    c = capture_case.capture
    with pytest.raises(ValueError, match="ordered"): c.bracket("gauss", lambda: None)
    with pytest.raises(ValueError): c.bracket("init_cost", lambda: None)


@pytest.mark.parametrize("mutation", ("pointer", "stream", "copy", "replace-array", "host-pointer", "device-context", "device-arch"))
def test_binding_mutation_before_dispatch(capture_case, mutation):
    c = capture_case
    if mutation == "pointer": c.arrays["data.qacc"].ptr += 1
    elif mutation == "stream": c.stream.cuda_stream += 1
    elif mutation == "copy": c.wp.copy = lambda *args, **kwargs: None
    elif mutation == "replace-array": c.arrays["data.qacc"] = object()
    elif mutation == "device-context": c.stream.device.context += 1
    elif mutation == "device-arch": c.stream.device.arch = 121
    else: c.capture.host[0].ptr += 1
    called = []
    with pytest.raises(ValueError): c.capture.bracket("init_cost", lambda: called.append(True))
    assert not called and c.capture.failed and not c.capture.packets


def test_retained_bytes_mutation(capture_case):
    c = capture_case
    c.capture.bracket("init_cost", lambda: c.dispatch("init_cost"))
    original = c.capture.raw("init_cost.before")
    replacement = bytes(stages.PACKET_BYTES)
    with pytest.raises(TypeError):
        c.capture.packets["init_cost.before"] = replacement
    with pytest.raises(TypeError):
        c.capture.hashes["init_cost.before"] = sha256(replacement).hexdigest()
    assert c.capture.raw("init_cost.before") == original


@pytest.mark.parametrize("name", ("guard", "held_guard", "wp", "stream", "held_stream", "sources", "layouts", "next_stage", "_sealed"))
def test_sealed_capture_bindings(capture_case, name):
    with pytest.raises(AttributeError, match="sealed"):
        setattr(capture_case.capture, name, object())
    with pytest.raises(AttributeError, match="sealed"):
        delattr(capture_case.capture, name)


def test_dispatch_cannot_coordinately_rebind_guard(capture_case):
    c = capture_case.capture
    original = c.guard

    def dispatch():
        replacement = lambda: None
        c.guard = replacement
        c.held_guard = replacement

    with pytest.raises(AttributeError, match="sealed"):
        c.bracket("init_cost", dispatch)
    assert c.failed and c.guard is c.held_guard is original
    assert len(c.raw("init_cost.before")) == stages.PACKET_BYTES
    with pytest.raises(ValueError): c.record()


def test_runtime_entry_directory_is_read_only(capture_case):
    with pytest.raises(TypeError):
        capture_case.capture.entries["copy"] = (lambda: None, None)


def test_no_manual_snapshot_outside_bracket(capture_case):
    with pytest.raises(ValueError, match="active ordered"):
        capture_case.capture._snapshot("init_cost.before")
    assert not capture_case.capture.packets


def test_capture_preserves_inactive_nan_payload(capture_case):
    c = capture_case
    special = (0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000)
    c.arrays["efc.J"].values.view("<u4")[0, 511, :4] = special
    c.capture.bracket("init_cost", lambda: None)
    before, after = (stages.unpack_bank(c.capture.raw("init_cost." + phase)) for phase in ("before", "after"))
    offset = 511 * 20 * 4
    expected = struct.pack("<4I", *special)
    assert before["efc.J"][offset:offset + 16] == after["efc.J"][offset:offset + 16] == expected
