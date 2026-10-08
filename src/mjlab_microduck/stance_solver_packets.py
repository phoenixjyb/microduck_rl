"""One-shot raw readback at a guarded dense dispatch, not GPU admission.

No device packages are imported. The owner supplies the authenticated runtime,
scratch state, executable binding, stream, supervisor and workload lease. Copies
change timing. This is not a sandbox against hostile Python instrumentation.
"""
from copy import deepcopy
from hashlib import sha256
from math import prod
import sys

from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-dense-solver-packets-oct8-v1"
# Complete banks, including inactive/padding rows and preexisting output bits.
SPECS = (
    ("nefc", (64,), "int32", "<i4", 4),
    ("J", (64, 512, 20), "float32", "<f4", 4),
    ("force", (64, 512), "float32", "<f4", 4),
    ("done", (64,), "bool", "|b1", 1),
    ("qfrc_constraint", (64, 20), "float32", "<f4", 4),
)
PACKET_BYTES = sum(prod(shape) * width for _, shape, _, _, width in SPECS)
MAX_PAIR_BYTES = 8 * 1024**2
need = static.need


class DenseSolverPacketCapture:
    """Retain immutable whole pre/post bytes using explicit same-stream copies."""

    def __init__(self, *, wp, data, context, stream):
        need(sys.byteorder == "little" and 2 * PACKET_BYTES < MAX_PAIR_BYTES,
             "literal little-endian bounded packet recipe")
        self.wp, self.data, self.context, self.stream = wp, data, context, stream
        self.array_type = wp.array
        self.entries = {name: (getattr(wp, name), getattr(wp, name).__code__)
                        for name in ("empty", "copy", "synchronize_stream")}
        self.numpy_entry = (wp.array.numpy, wp.array.numpy.__code__)
        self.arrays = (data.nefc, data.efc.J, data.efc.force, context.done, data.qfrc_constraint)
        self.guard = None
        self.host = ()
        self.layouts = self._layouts(self.arrays, host=False)
        self.host_layouts = ()
        self.phase = "new"
        self.completed = self.failed = False
        self.packets = {}
        self.packet_hashes = {}
        self._check()

    def _layouts(self, arrays, *, host):
        result = []
        need(len(arrays) == len(SPECS), "complete five-buffer packet layout")
        for value, (_, shape, dtype, _, width) in zip(arrays, SPECS):
            need(type(value) is self.array_type and tuple(value.shape) == shape
                 and value.dtype is getattr(self.wp, dtype)
                 and type(value.ptr) is int and value.ptr > 0
                 and value.is_contiguous is True and value.requires_grad is False,
                 "literal contiguous packet buffer")
            if host:
                need(getattr(value.device, "is_cpu", None) is True
                     and value.device.is_cuda is False
                     and value.pinned is True, "owned pinned CPU staging buffer")
            else:
                need(value.device is self.stream.device, "same source stream device")
            result.append((id(value), value.ptr, shape, id(value.dtype), id(value.device),
                           prod(shape) * width))
        spans = sorted((v[1], v[1] + v[-1]) for v in result)
        need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])),
             "nonoverlapping packet buffers")
        return tuple(result)

    def _check(self):
        wp = self.wp
        need(wp.array is self.array_type
             and wp.array.numpy is self.numpy_entry[0]
             and wp.array.numpy.__code__ is self.numpy_entry[1]
             and all(getattr(wp, name) is fn and fn.__code__ is code
                     for name, (fn, code) in self.entries.items()), "held packet runtime entries")
        current = (self.data.nefc, self.data.efc.J, self.data.efc.force,
                   self.context.done, self.data.qfrc_constraint)
        need(all(a is b for a, b in zip(current, self.arrays))
             and self._layouts(current, host=False) == self.layouts,
             "same packet source objects and layouts")
        if self.host:
            need(self._layouts(self.host, host=True) == self.host_layouts,
                 "same owned CPU staging buffers")
        need(set(self.packets) == set(self.packet_hashes)
             and all(type(raw) is bytes and len(raw) == PACKET_BYTES
                     and sha256(raw).hexdigest() == self.packet_hashes[phase]
                     for phase, raw in self.packets.items()), "unchanged retained packet bytes")

    def _attach(self, observer):
        from mjlab_microduck.stance_solver_dispatch_guard import DenseSolverDispatchGuard
        need(type(observer) is DenseSolverDispatchGuard and self.phase == "new"
             and self.guard is None and not self.failed
             and observer.wp is self.wp and observer.data is self.data
             and observer.context is self.context and observer.stream is self.stream,
             "one-shot capture bound to exact dispatch guard")
        self.guard = observer
        self.phase = "attached"
        self._check()
        self.host = tuple(self.entries["empty"][0](shape=shape, dtype=getattr(self.wp, dtype),
                          device="cpu", requires_grad=False, pinned=True)
                          for _, shape, dtype, _, _ in SPECS)
        self.host_layouts = self._layouts(self.host, host=True)
        self._assert_bound(observer)

    def _assert_bound(self, observer):
        need(self.guard is observer and not self.failed, "unchanged live capture ownership")
        self._check()

    def _snapshot(self, phase):
        need(phase in ("before", "after")
             and self.phase == ("attached" if phase == "before" else "before-complete")
             and not self.failed, "single ordered packet snapshot")
        self.phase = phase + "-started"  # Failure consumes this attempt.
        self.guard._guard()
        # The first sync retires the preceding EFC-force writer or the target.
        self.entries["synchronize_stream"][0](self.stream)
        self.guard._guard()
        for dst, src in zip(self.host, self.arrays):
            self.entries["copy"][0](dst, src, stream=self.stream)
            self.guard._guard()
        self.entries["synchronize_stream"][0](self.stream)
        self.guard._guard()
        chunks = []
        for value, (_, shape, _, wire, width) in zip(self.host, SPECS):
            # Only CPU array.numpy is used, never GPU convenience/default-stream
            # readback. The separately pinned runtime/source remains owner duty.
            host = self.numpy_entry[0](value)
            need(tuple(host.shape) == shape and host.dtype.str == wire
                 and host.flags.c_contiguous is True
                 and type(host.nbytes) is int and host.nbytes == prod(shape) * width,
                 "whole canonical CPU readback layout")
            raw = host.tobytes(order="C")
            need(type(raw) is bytes and len(raw) == prod(shape) * width,
                 "complete raw packet field bytes")
            chunks.append(raw)
            self.guard._guard()
        raw = b"".join(chunks)
        need(len(raw) == PACKET_BYTES, "whole fixed packet size")
        self.packets[phase] = raw
        self.packet_hashes[phase] = sha256(raw).hexdigest()
        self.phase = phase + "-complete"

    def _finish(self, success):
        self.completed = success is True and self.phase == "after-complete"
        self.failed = not self.completed

    def raw(self, phase):
        """Complete retained phase bytes remain available even after a later fault."""
        need(phase in ("before", "after") and phase in self.packets,
             "fully retained packet phase")
        need(type(self.packets[phase]) is bytes and len(self.packets[phase]) == PACKET_BYTES
             and sha256(self.packets[phase]).hexdigest() == self.packet_hashes[phase],
             "unchanged retained packet phase bytes")
        return self.packets[phase]

    def record(self):
        need(self.completed and not self.failed and self.guard.completed
             and not self.guard.active and self.guard.capture is self,
             "successful guarded packet pair only")
        self.guard._guard()
        rows = {}
        for phase in ("before", "after"):
            raw, offset, fields = self.raw(phase), 0, []
            for name, shape, _, wire, width in SPECS:
                size = prod(shape) * width
                fields.append(dict(name=name, shape=list(shape), dtype=wire, offset=offset,
                                   bytes=size, sha256=sha256(raw[offset:offset + size]).hexdigest()))
                offset += size
            rows[phase] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest(), fields=fields)
        inputs_size = PACKET_BYTES - 64 * 20 * 4
        return deepcopy(dict(protocol=PROTOCOL, decision="guarded-packets-only-not-qualification",
            packets=rows, complete_pair=True, input_bytes_unchanged=(
                self.raw("before")[:inputs_size] == self.raw("after")[:inputs_size]),
            packet_bytes=PACKET_BYTES, pair_bytes=2 * PACKET_BYTES,
            copy_stream_handle=self.stream.cuda_stream,
            timing_changed_by_readback=True, numerical_acceptance=False,
            driver_loaded_code_observed=False, flags=dict(static.FLAGS)))
