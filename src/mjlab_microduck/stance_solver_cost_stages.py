"""Inert caller-stage capture/CPU contract preparation, not GPU admission.

The future native owner must authenticate dispatches, executable bindings,
source/runtime, lease, caps and retirement independently. This module does not
import a device package or launch a kernel. Synthetic controls are not physics.
"""

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import math
from math import prod
import struct
import sys
from threading import get_ident
from types import MappingProxyType

from mjlab_microduck import stance_solver_gauss_audit as gauss
from mjlab_microduck import stance_solver_init_control as init
from mjlab_microduck import stance_solver_scratch as scratch

PROTOCOL = "microduck-caller-cost-stage-preparation-oct8-v1"
CONTROL_PROTOCOL = "microduck-caller-cost-stage-synthetic-control-oct8-v1"
FLAGS = dict(gauss.FLAGS)
STAGES = ("init_cost", "efc", "dense", "gauss")
PHASES = tuple(stage + "." + phase for stage in STAGES for phase in ("before", "after"))
ORDER = scratch.RESTORE_ORDER
SPECS = tuple((name, *init.SOLVER_INIT_SPECS[name]) for name in ORDER)
PACKET_BYTES = sum(prod(host) * (1 if dtype == "bool" else 4)
                   for _, _, host, dtype in SPECS)
MAX_TOTAL_BYTES = 64 * 1024**2
WRITES = MappingProxyType({
    "init_cost": frozenset(("context.gauss", "context.cost", "context.prev_cost")),
    "efc": frozenset(("efc.force", "efc.state", "context.cost")),
    "dense": frozenset(("data.qfrc_constraint",)),
    "gauss": frozenset(("context.gauss", "context.cost")),
})
need = gauss.need


def pack_bank(fields):
    need(type(fields) is dict and set(fields) == set(ORDER), "closed complete stage bank")
    for name, _, shape, dtype in SPECS:
        need(type(fields[name]) is bytes
             and len(fields[name]) == prod(shape) * (1 if dtype == "bool" else 4),
             "literal stage field bytes " + name)
    return b"".join(fields[name] for name in ORDER)


def unpack_bank(raw):
    need(type(raw) is bytes and len(raw) == PACKET_BYTES, "literal whole stage packet")
    fields, offset = {}, 0
    for name, _, shape, dtype in SPECS:
        size = prod(shape) * (1 if dtype == "bool" else 4)
        fields[name] = raw[offset:offset + size]
        offset += size
    return fields


def control_fields():
    """Literal control independent of historical output values."""
    values = {name: struct.pack("<1280f", *([1.0 if name in ("efc.Ma", "data.qacc") else 0.0] * 1280))
              for name in gauss.NAMES}
    values["context.gauss"] = struct.pack("<64f", *(float((1 if w % 2 == 0 else -1) * (4096 + w))
                                                     for w in range(64)))
    values["context.cost"] = struct.pack("<64f", *(float(1024 + w) for w in range(64)))
    values["context.prev_cost"] = struct.pack("<64f", *(float(-2048 - w) for w in range(64)))
    return values


def control_manifest():
    return dict(protocol=CONTROL_PROTOCOL, synthetic_only=True, expected_gauss=10.0,
                overrides={name: dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
                           for name, raw in sorted(control_fields().items())},
                qualification=dict(FLAGS))


def initial_banks(packet):
    need(type(packet) is scratch.DecodedPacket and packet._sealed is True
         and sha256(packet.raw).hexdigest() == packet.sha256,
         "sealed authenticated historical packet only")
    reference = {name: packet.fields[name] for name in ORDER}
    control = dict(reference, **control_fields())
    return dict(reference=pack_bank(reference), control=pack_bank(control))


def _floats(raw):
    values = struct.unpack("<" + str(len(raw) // 4) + "f", raw)
    need(all(math.isfinite(x) for x in values), "finite stage values")
    return values


def _domain(fields):
    need(fields["context.done"] == bytes(64), "all-active stage recipe")
    counts = struct.unpack("<64i", fields["data.nefc"])
    ne = struct.unpack("<64i", fields["data.ne"])
    nf = struct.unpack("<64i", fields["data.nf"])
    need(all(0 < count <= 512 and 0 <= a <= count and 0 <= b <= count - a
             for count, a, b in zip(counts, ne, nf)), "bounded active stage rows")
    types = struct.unpack("<32768i", fields["efc.type"])
    need(all(types[w * 512 + row] != 7 for w, count in enumerate(counts) for row in range(count)),
         "no active elliptic rows with omitted friction")
    return counts


def _dense(before, after, counts):
    # Same separate product/sum hypothesis and tolerances as the retained dense
    # replay receiver. This is arithmetic consistency, not CUDA instruction proof.
    max_error, mismatches = 0.0, 0
    for world, count in enumerate(counts):
        for dof in range(20):
            expected = 0.0
            for row in range(count):
                j = struct.unpack_from("<f", before["efc.J"], ((world * 512 + row) * 20 + dof) * 4)[0]
                force = struct.unpack_from("<f", before["efc.force"], (world * 512 + row) * 4)[0]
                need(math.isfinite(j) and math.isfinite(force), "finite active dense terms")
                term = gauss.round32(Fraction(j) * Fraction(force))
                expected = gauss.round32(Fraction(expected) + Fraction(term))
            actual = struct.unpack_from("<f", after["data.qfrc_constraint"], (world * 20 + dof) * 4)[0]
            need(math.isfinite(actual), "finite dense output")
            error = abs(actual - expected)
            max_error = max(max_error, error)
            mismatches += error > 2e-5 + 2e-5 * max(abs(actual), abs(expected))
    need(mismatches == 0, "dense arithmetic consistency")
    return dict(compared_dofs=1280, mismatches=mismatches, max_absolute_error=max_error,
                absolute_tolerance=2e-5, relative_tolerance=2e-5)


def analyze_arm(packets, *, arm):
    """Analyze already hash-authenticated banks; does not prove capture origin."""
    need(type(packets) is dict and set(packets) == set(PHASES)
         and arm in ("reference", "control"), "complete declared stage snapshots")
    fields = {phase: unpack_bank(packets[phase]) for phase in PHASES}
    counts = _domain(fields[PHASES[0]])
    changed = {}
    for stage in STAGES:
        before, after = fields[stage + ".before"], fields[stage + ".after"]
        changes = {name for name in ORDER if before[name] != after[name]}
        need(changes <= WRITES[stage], "unchanged non-output stage fields " + stage)
        changed[stage] = sorted(changes)
    for index in range(len(STAGES) - 1):
        need(packets[STAGES[index] + ".after"] == packets[STAGES[index + 1] + ".before"],
             "exact adjacent stage-bank continuity")
    before, after = fields["init_cost.before"], fields["init_cost.after"]
    need(after["context.gauss"] == after["context.cost"] == bytes(256),
         "positive-zero setup reset")
    need(after["context.prev_cost"] == before["context.cost"], "bit-exact prior-cost transfer")
    _floats(before["context.cost"])
    before, after = fields["efc.before"], fields["efc.after"]
    need(all(x >= 0.0 for x in _floats(after["context.cost"])), "nonnegative finite EFC cost")
    for world, count in enumerate(counts):
        for name in ("efc.force", "efc.state"):
            start, end = (world * 512 + count) * 4, (world + 1) * 512 * 4
            need(before[name][start:end] == after[name][start:end], "unchanged inactive EFC tail")
    dense = _dense(fields["dense.before"], fields["dense.after"], counts)
    before, after = fields["gauss.before"], fields["gauss.after"]
    audit = gauss.analyze({name: before[name] for name in gauss.NAMES}
                         | {"context.done": before["context.done"], "context.gauss": after["context.gauss"]})
    need(audit["outside_bound"] == 0, "Gauss arithmetic consistency")
    costs_before, costs_after = _floats(before["context.cost"]), _floats(after["context.cost"])
    contributions = _floats(after["context.gauss"])
    expected_cost = struct.pack("<64f", *(gauss.round32(Fraction(a) + Fraction(b))
                                          for a, b in zip(costs_before, contributions)))
    need(after["context.cost"] == expected_cost, "Gauss add to immediate pre-stage cost")
    if arm == "control":
        need(all(fields["init_cost.before"][name] == raw for name, raw in control_fields().items()),
             "literal synthetic initial control")
        need(after["context.gauss"] == struct.pack("<64f", *([10.0] * 64)), "nonzero Gauss control equals ten")
        need(all(a != b for a, b in zip(costs_before, costs_after)), "nonvacuous control cost addition")
    return dict(arm=arm, changed_fields=changed, setup_reset=True, prior_cost_transferred=True,
                inactive_efc_tails_unchanged=True, efc_cost_numerically_qualified=False,
                dense=dense, gauss=audit, gauss_cost_addition=True, qualification=dict(FLAGS))


def receive_banks(arms, anchors, packet):
    """Authenticate all sixteen complete banks before interpreting any one.

    `anchors` must come from the future collector's independently authenticated
    complete raw inventory, not from metadata adjacent to these banks.
    """
    expected = {"reference", "control"}
    need(type(arms) is dict and type(anchors) is dict
         and set(arms) == set(anchors) == expected, "exact two-arm stage envelope")
    need(16 * PACKET_BYTES < MAX_TOTAL_BYTES, "bounded sixteen-bank protocol")
    for arm in sorted(expected):
        need(type(arms[arm]) is dict and type(anchors[arm]) is dict
             and set(arms[arm]) == set(anchors[arm]) == set(PHASES), "exact stage bank directory")
        for phase in PHASES:
            raw, anchor = arms[arm][phase], anchors[arm][phase]
            need(type(raw) is bytes and len(raw) == PACKET_BYTES
                 and type(anchor) is dict and set(anchor) == {"bytes", "sha256"}
                 and type(anchor["bytes"]) is int and anchor["bytes"] == len(raw)
                 and type(anchor["sha256"]) is str and anchor["sha256"] == sha256(raw).hexdigest(),
                 "whole external stage packet anchor")
    initial = initial_banks(packet)
    need(all(arms[arm]["init_cost.before"] == initial[arm] for arm in expected),
         "both arms bound to complete historical restoration and literal overrides")
    results = {arm: analyze_arm(arms[arm], arm=arm) for arm in sorted(expected)}
    for name in ("efc.force", "efc.state", "data.qfrc_constraint"):
        phase = "dense.after" if name == "data.qfrc_constraint" else "efc.after"
        need(unpack_bank(arms["reference"][phase])[name] == unpack_bank(arms["control"][phase])[name],
             "paired non-Gauss output banks agree")
    return dict(protocol=PROTOCOL, decision="caller-cost-stage-packet-contract-only",
                banks=deepcopy(anchors), packet_bytes=PACKET_BYTES, total_bytes=16 * PACKET_BYTES,
                control=control_manifest(), arms=results, capture_origin_authenticated=False,
                executable_binding_authenticated=False, gpu_dispatch_authenticated=False,
                external_unit_retirement_authenticated=False, qualification=dict(FLAGS))


class CostStageCapture:
    """One-shot ordered same-stream readback, with an external owner's guard.

    `bracket` never authenticates its dispatch callable as a kernel. The native
    caller adapter must do that separately. Failed attempts cannot be retried.
    Sealed bindings provide observed-object integrity for trusted in-process
    callers, not a sandbox against arbitrary Python reflection/instrumentation.
    """

    __slots__ = ("wp", "arrays", "stream", "guard", "held_guard", "guard_code",
                 "held_stream", "device", "device_identity", "stream_handle",
                 "array_type", "owner_thread", "entries", "numpy_entry", "sources",
                 "layouts", "host", "host_layouts", "packets", "hashes", "next_stage",
                 "failed", "active", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("caller-stage capture bindings are sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("caller-stage capture bindings are sealed")
        object.__delattr__(self, name)

    def __init__(self, *, wp, arrays, stream, guard):
        need(sys.byteorder == "little" and type(arrays) is dict and set(arrays) == set(ORDER)
             and callable(guard), "complete caller-stage capture inputs")
        self.wp, self.arrays, self.stream, self.guard = wp, arrays, stream, guard
        self.held_guard, self.guard_code = guard, getattr(guard, "__code__", None)
        need(self.guard_code is not None, "held Python owner guard")
        self.held_stream, self.device = stream, stream.device
        self.stream_handle = stream.cuda_stream
        need(self.device.is_cuda is True and self.device.arch == 120
             and type(self.device.context) is int and self.device.context > 0
             and type(self.stream_handle) is int and self.stream_handle > 0,
             "bounded explicit sm120 stage stream")
        self.device_identity = (self.device.arch, self.device.context)
        self.array_type, self.owner_thread = wp.array, get_ident()
        self.entries = MappingProxyType({name: (getattr(wp, name), getattr(wp, name).__code__)
                                         for name in ("empty", "copy", "synchronize_stream")})
        self.numpy_entry = (wp.array.numpy, wp.array.numpy.__code__)
        self.sources = tuple(arrays[name] for name in ORDER)
        self.layouts = self._layouts(self.sources, host=False)
        self.host = tuple(self.entries["empty"][0](shape=shape, dtype=getattr(wp, dtype),
                          device="cpu", pinned=True, requires_grad=False)
                          for _, shape, _, dtype in SPECS)
        self.host_layouts = self._layouts(self.host, host=True)
        need(not any(a[1] < b[1] + b[-1] and b[1] < a[1] + a[-1]
                     for a in self.layouts for b in self.host_layouts), "disjoint source/staging spans")
        self.packets, self.hashes = MappingProxyType({}), MappingProxyType({})
        self.next_stage, self.failed, self.active = 0, False, False
        self._sealed = True
        self._check()

    def _layouts(self, arrays, *, host):
        rows = []
        for array, (_, shape, _, dtype) in zip(arrays, SPECS):
            need(type(array) is self.array_type and tuple(array.shape) == shape
                 and array.dtype is getattr(self.wp, dtype) and array.is_contiguous is True
                 and array.requires_grad is False and type(array.ptr) is int and array.ptr > 0,
                 "literal stage array allocation")
            need((array.device.is_cpu is True and array.device.is_cuda is False and array.pinned is True)
                 if host else array.device is self.stream.device, "literal stage array device")
            rows.append((id(array), array.ptr, shape, id(array.dtype), id(array.device),
                         prod(shape) * (1 if dtype == "bool" else 4)))
        spans = sorted((row[1], row[1] + row[-1]) for row in rows)
        need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "nonoverlapping stage array spans")
        return tuple(rows)

    def _check(self):
        need(get_ident() == self.owner_thread and self.wp.array is self.array_type
             and self.wp.array.numpy is self.numpy_entry[0]
             and self.wp.array.numpy.__code__ is self.numpy_entry[1]
             and all(getattr(self.wp, name) is fn and fn.__code__ is code
                     for name, (fn, code) in self.entries.items()), "held stage runtime entries")
        need(self.guard is self.held_guard and self.guard.__code__ is self.guard_code
             and self.stream is self.held_stream and self.stream.device is self.device
             and self.stream.cuda_stream == self.stream_handle and self.device.is_cuda is True
             and (self.device.arch, self.device.context) == self.device_identity,
             "held owner guard and explicit stage stream")
        need(set(self.arrays) == set(ORDER)
             and all(self.arrays[name] is array for name, array in zip(ORDER, self.sources))
             and self._layouts(self.sources, host=False) == self.layouts
             and self._layouts(self.host, host=True) == self.host_layouts,
             "same stage source and staging allocations")
        need(set(self.packets) == set(self.hashes)
             and all(type(raw) is bytes and len(raw) == PACKET_BYTES
                     and sha256(raw).hexdigest() == self.hashes[phase]
                     for phase, raw in self.packets.items()), "unchanged retained stage bytes")
        self.guard()

    def _snapshot(self, phase):
        need(self.active and not self.failed and len(self.packets) < len(PHASES)
             and phase == PHASES[len(self.packets)]
             and phase.startswith(STAGES[self.next_stage] + "."),
             "single active ordered stage snapshot")
        self._check()
        self.entries["synchronize_stream"][0](self.stream)
        self._check()
        for dst, src in zip(self.host, self.sources):
            self.entries["copy"][0](dst, src, stream=self.stream)
            self._check()
        self.entries["synchronize_stream"][0](self.stream)
        self._check()
        fields = {}
        for array, (name, _, host_shape, dtype) in zip(self.host, SPECS):
            host = self.numpy_entry[0](array)
            wire = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
            need(tuple(host.shape) == host_shape and host.dtype.str == wire
                 and host.flags.c_contiguous is True
                 and type(host.nbytes) is int
                 and host.nbytes == prod(host_shape) * (1 if dtype == "bool" else 4),
                 "whole canonical stage CPU view")
            fields[name] = host.tobytes(order="C")
            self._check()
        raw = pack_bank(fields)
        object.__setattr__(self, "packets", MappingProxyType(dict(self.packets, **{phase: raw})))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes, **{phase: sha256(raw).hexdigest()})))

    def bracket(self, stage, dispatch):
        if not (not self.failed and not self.active and self.next_stage < len(STAGES)
                and stage == STAGES[self.next_stage] and callable(dispatch)):
            object.__setattr__(self, "failed", True)
            raise ValueError("ordered one-shot stage bracket")
        object.__setattr__(self, "active", True)
        try:
            self._snapshot(stage + ".before")
            result = dispatch()
            self._check()
            self._snapshot(stage + ".after")
            object.__setattr__(self, "next_stage", self.next_stage + 1)
            return result
        except BaseException:
            object.__setattr__(self, "failed", True)
            raise
        finally:
            object.__setattr__(self, "active", False)

    def raw(self, phase):
        need(phase in PHASES and phase in self.packets, "retained complete stage snapshot")
        raw = self.packets[phase]
        need(type(raw) is bytes and len(raw) == PACKET_BYTES
             and sha256(raw).hexdigest() == self.hashes[phase], "stable retained stage snapshot")
        return raw

    def record(self):
        need(not self.failed and not self.active and self.next_stage == 4
             and set(self.packets) == set(PHASES), "complete ordered stage capture only")
        self._check()
        return dict(protocol=PROTOCOL, stages=list(STAGES), timing_changed_by_readback=True,
                    packets={phase: dict(bytes=PACKET_BYTES, sha256=self.hashes[phase]) for phase in PHASES},
                    copy_stream_handle=self.stream_handle,
                    capture_origin_authenticated=False, executable_binding_authenticated=False,
                    gpu_dispatch_authenticated=False, external_unit_retirement_authenticated=False,
                    kernel_dispatch_authenticated=False, qualification=dict(FLAGS))
