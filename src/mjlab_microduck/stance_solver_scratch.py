"""Authenticated one-shot restoration for a dense solver scratch bootstrap.

This module does not import Warp, Torch, MuJoCo, or the installed solver. The
owner supplies staging and copy callables after it has authenticated and
prepared the runtime. Restoring these arrays is a scratch setup action only;
it does not execute a solver or qualify a device.
"""

from hashlib import sha256
from math import isfinite
import struct
from types import MappingProxyType


PROTOCOL = "microduck-dense-solver-scratch-oct8-v1"
MAX_PACKET_BYTES = 8 * 1024**2
FLAGS = MappingProxyType({
    "native_qualified": False,
    "full_window_qualified": False,
    "runtime_cause_proven": False,
    "training_authorized": False,
    "physical_acceptance": False,
})

# The unchanged _update_constraint(False) call has five array arguments at its
# dense target. These and the arrays read/written by its three setup launches
# must be restored from one same-boundary packet.
RESTORE_ORDER = (
    "model.opt.impratio_invsqrt",
    "data.ne",
    "data.nf",
    "data.nefc",
    "contact.nacon",
    "data.qacc",
    "data.qacc_smooth",
    "data.qfrc_smooth",
    "data.qfrc_constraint",
    "contact.dim",
    "contact.efc_address",
    "efc.type",
    "efc.id",
    "efc.J",
    "efc.D",
    "efc.frictionloss",
    "efc.force",
    "efc.state",
    "efc.Ma",
    "context.Jaref",
    "context.gauss",
    "context.cost",
    "context.prev_cost",
    "context.done",
)
_DECODED_PACKET_SEAL = object()


def _need(value, message):
    if not value:
        raise ValueError(message)


def _contract():
    # This project module is stdlib-only and does not import the solver. Resolve
    # the frozen capture schema lazily so importing this file stays inert.
    from mjlab_microduck import stance_solver_init_control as control

    _need(len(control.SOLVER_INIT_ORDER) == 66, "literal 66-field solver-init packet")
    _need(set(control.SOLVER_INIT_ORDER) == set(control.SOLVER_INIT_SPECS),
          "complete frozen solver-init layout schema")
    _need(control.SOLVER_RECIPE["nv"] == 20
          and control.SOLVER_RECIPE["nworld"] == 64
          and control.SOLVER_RECIPE["njmax"] == 512
          and control.SOLVER_RECIPE["is_sparse"] is False,
          "literal dense 64-world solver recipe")
    return control


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _check_recipe(recipe, expected):
    _need(type(recipe) is dict and set(recipe) == set(expected)
          and recipe == expected and recipe["cone"] == 0,
          "exact dense pyramidal solver recipe")
    integer_keys = (
        "solver", "cone", "disableflags", "enableflags", "iterations",
        "ls_iterations", "integrator", "nv", "nv_pad", "nworld", "njmax",
        "njmax_nnz",
    )
    bool_keys = ("ls_parallel", "graph_conditional", "run_collision_detection", "is_sparse")
    _need(all(type(recipe[name]) is int for name in integer_keys)
          and all(type(recipe[name]) is bool for name in bool_keys),
          "plain integer and boolean recipe scalars")
    _need(type(recipe["ls_parallel_min_step"]) in (int, float)
          and isfinite(recipe["ls_parallel_min_step"]),
          "finite plain linesearch step")
    block_dim = recipe["block_dim"]
    _need(type(block_dim) is dict and set(block_dim) == set(expected["block_dim"])
          and all(type(value) is int and value > 0 for value in block_dim.values()),
          "exact plain integer block dimensions")


def _warp_dtype(host_dtype, trailing):
    vector = {
        ("float32", (2,)): "<class 'warp._src.types.vec2f'>",
        ("float32", (3,)): "<class 'warp._src.types.vec3f'>",
        ("float32", (5,)): "<class 'mujoco_warp._src.types.vec5f'>",
        ("float32", (3, 3)): "<class 'warp._src.types.mat33f'>",
        ("int32", (2,)): "<class 'warp._src.types.vec2i'>",
    }
    return vector.get((host_dtype, tuple(trailing)), {
        "float32": "<class 'warp._src.types.float32'>",
        "int32": "<class 'warp._src.types.int32'>",
        "bool": "<class 'warp._src.types.bool'>",
    }[host_dtype])


def decode_packet(raw, snapshot, expected_sha256, recipe, elliptic_type):
    """Authenticate and decode the complete retained 66-field packet.

    `expected_sha256` must be taken from an independently authenticated raw
    inventory. Hash verification precedes every slice or scalar decode.
    `elliptic_type` is the caller's value from the pinned MuJoCo-Warp enum.
    """
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_PACKET_BYTES
          and type(expected_sha256) is str,
          "immutable packet bytes and external SHA256 anchor")
    _need(len(expected_sha256) == 64
          and all(character in "0123456789abcdef" for character in expected_sha256),
          "plain lowercase SHA256 anchor")
    actual = sha256(raw).hexdigest()
    _need(actual == expected_sha256, "whole supplied solver-init packet SHA256")
    _need(type(snapshot) is dict and set(snapshot) == {
              "path", "bytes", "sha256", "fields", "layouts", "forward", "phase",
              "context_object_id",
          }
          and snapshot.get("sha256") == actual
          and type(snapshot.get("bytes")) is int
          and type(snapshot.get("context_object_id")) is int
          and snapshot["context_object_id"] > 0
          and snapshot.get("bytes") == len(raw), "snapshot whole-byte binding")
    _need(type(elliptic_type) is int and type(elliptic_type) is not bool,
          "literal elliptic constraint enum value")

    control = _contract()
    _check_recipe(recipe, control.SOLVER_RECIPE)
    _need(type(snapshot.get("forward")) is int and snapshot["forward"] == 4
          and type(snapshot.get("phase")) is str
          and snapshot["phase"] == "initialized-before-search"
          and type(snapshot.get("path")) is str
          and snapshot["path"] == "solver-init/original/forward-04.initialized.bin",
          "retained post-init forward-four packet")
    fields, layouts = snapshot.get("fields"), snapshot.get("layouts")
    order, specs = control.SOLVER_INIT_ORDER, control.SOLVER_INIT_SPECS
    _need(type(fields) is dict and set(fields) == set(order),
          "exact solver-init field directory")
    _need(type(layouts) is dict and set(layouts) == set(order),
          "exact solver-init allocation layout directory")

    decoded, offset = {}, 0
    wire = {"float32": ("<f4", 4), "int32": ("<i4", 4), "bool": ("|b1", 1)}
    for name in order:
        logical, host_shape, host_dtype = specs[name]
        _need(all(type(value) is int and value >= 0
                  for value in tuple(logical) + tuple(host_shape)),
              "plain integer frozen schema dimensions " + name)
        dtype, width = wire[host_dtype]
        entry = fields[name]
        size = 1
        for dimension in host_shape:
            size *= dimension
        size *= width
        _need(type(entry) is dict
              and set(entry) == {"offset", "bytes", "shape", "dtype"}
              and type(entry["offset"]) is int and entry["offset"] == offset
              and type(entry["bytes"]) is int and entry["bytes"] == size
              and type(entry["shape"]) is list
              and all(type(value) is int and value >= 0 for value in entry["shape"])
              and entry["shape"] == list(host_shape)
              and type(entry["dtype"]) is str
              and entry["dtype"] == dtype,
              "literal packet offset/shape/dtype " + name)
        layout = layouts[name]
        _need(type(layout) is dict and set(layout) == {
            "object_id", "pointer", "span", "device", "context", "warp_dtype",
            "shape", "strides", "host_shape", "host_dtype", "bytes",
        }, "complete retained allocation metadata " + name)
        _need(type(layout["device"]) is str
              and type(layout["warp_dtype"]) is str
              and type(layout["host_dtype"]) is str,
              "plain string retained layout metadata " + name)
        expected_warp_dtype = _warp_dtype(host_dtype, host_shape[len(logical):])
        pointer = layout["pointer"]
        if size:
            _need(type(pointer) is int and pointer > 0,
                  "retained source pointer " + name)
        else:
            _need(pointer is None or (type(pointer) is int and pointer == 0),
                  "null empty source allocation " + name)
        stride_width = width
        for dimension in host_shape[len(logical):]:
            stride_width *= dimension
        expected_strides, stride = [], stride_width
        for dimension in reversed(logical):
            expected_strides.insert(0, stride)
            stride *= dimension
        source_strides = layout["strides"]
        _need(type(source_strides) is list and len(source_strides) == len(logical)
              and all(type(actual) is int and actual >= 0
                      and (actual == expected or (dimension == 1 and actual == 0))
                      for actual, expected, dimension
                      in zip(source_strides, expected_strides, logical)),
              "retained source strides match packed layout " + name)
        _need(type(layout["object_id"]) is int and layout["object_id"] > 0
              and layout["device"] == "cuda:0"
              and type(layout["context"]) is int and layout["context"] > 0,
              "retained source object/device/context " + name)
        _need(type(layout["shape"]) is list
              and all(type(value) is int and value >= 0 for value in layout["shape"])
              and type(layout["host_shape"]) is list
              and all(type(value) is int and value >= 0 for value in layout["host_shape"])
              and type(layout["span"]) is int and type(layout["bytes"]) is int
              and layout["shape"] == list(logical)
              and layout["host_shape"] == list(host_shape)
              and layout["host_dtype"] == host_dtype
              and layout["warp_dtype"] == expected_warp_dtype
              and layout["bytes"] == size and layout["span"] == size,
              "retained allocation layout matches frozen schema " + name)
        end = offset + size
        decoded[name] = raw[offset:end]
        _need(len(decoded[name]) == size, "whole field bytes " + name)
        offset = end
    _need(offset == len(raw), "no unclaimed solver-init packet bytes")

    # Prove contact.friction is unreachable in this exact frozen call. The
    # installed kernel reads it only under the CONTACT_ELLIPTIC row branch;
    # cone=0 alone is insufficient, so inspect every row below each nefc.
    counts = struct.unpack("<64i", decoded["data.nefc"])
    _need(all(0 < count <= recipe["njmax"] for count in counts),
          "bounded active dense constraint rows")
    equality = struct.unpack("<64i", decoded["data.ne"])
    friction = struct.unpack("<64i", decoded["data.nf"])
    _need(all(0 <= ne <= count and 0 <= nf <= count - ne
              for ne, nf, count in zip(equality, friction, counts)),
          "bounded equality/friction rows within active EFC bank")
    contact_count = struct.unpack("<i", decoded["contact.nacon"])[0]
    _need(0 <= contact_count <= 8192, "bounded active contact count")
    efc_types = struct.unpack("<32768i", decoded["efc.type"])
    _need(all(value == 0 for value in decoded["context.done"]),
          "all worlds active for scratch update")
    active_j_nonzero = False
    for world, count in enumerate(counts):
        start = world * recipe["njmax"]
        j_offset = world * recipe["njmax"] * recipe["nv"] * 4
        active_j = struct.unpack_from(
            "<" + str(count * recipe["nv"]) + "f", decoded["efc.J"], j_offset
        )
        _need(all(isfinite(value) for value in active_j),
              "finite active dense Jacobian rows")
        active_j_nonzero |= any(value != 0.0 for value in active_j)
        for row in range(count):
            index = start + row
            if efc_types[index] == elliptic_type:
                _need(False, "active elliptic rows require omitted contact friction")
    _need(active_j_nonzero, "nonzero active dense Jacobian for meaningful replay")

    missing = set(RESTORE_ORDER) - set(decoded)
    _need(not missing, "all dense solver scratch sources retained")
    return DecodedPacket(
        raw, actual, decoded, layouts, specs, recipe, _seal=_DECODED_PACKET_SEAL
    )


class DecodedPacket:
    """Sealed immutable authenticated packet view retaining all source bytes."""

    __slots__ = ("raw", "sha256", "fields", "layouts", "specs", "recipe", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("decoded packet is sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("decoded packet is sealed")
        object.__delattr__(self, name)

    def __init__(self, raw, digest, fields, layouts, specs, recipe, *, _seal=None):
        _need(_seal is _DECODED_PACKET_SEAL, "decoded packets come only from authentication")
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "fields", MappingProxyType(dict(fields)))
        object.__setattr__(self, "layouts", MappingProxyType({
            name: MappingProxyType({
                "shape": tuple(value["shape"]),
                "host_shape": tuple(value["host_shape"]),
                "host_dtype": value["host_dtype"],
                "warp_dtype": value["warp_dtype"],
                "strides": tuple(value["strides"]),
                "span": value["span"],
            })
            for name, value in layouts.items()
        }))
        object.__setattr__(self, "specs", MappingProxyType({
            name: (tuple(spec[0]), tuple(spec[1]), spec[2])
            for name, spec in specs.items()
        }))
        object.__setattr__(self, "recipe", _freeze(recipe))
        object.__setattr__(self, "_sealed", True)


class DenseSolverScratchRestorer:
    """One-shot copies; `guard` must recheck the caller's held runtime bindings.

    `stage` must return a fresh CPU Warp array whose NumPy view is the exact
    supplied host-byte payload. This class verifies that view before each
    copy, but the caller-supplied runtime guard remains responsible for the
    solver/module/device identities and for excluding concurrent mutation.
    """

    __slots__ = (
        "packet", "arrays", "device", "stream", "stage", "copy", "synchronize",
        "guard", "destination_signature", "_arrays_input", "_array_value_ids",
        "_callback_ids", "_state", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("scratch restorer bindings are frozen")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("scratch restorer bindings are frozen")
        object.__delattr__(self, name)

    def __init__(self, packet, arrays, device, stream, *, stage, copy, synchronize, guard):
        _need(type(packet) is DecodedPacket and getattr(packet, "_sealed", False),
              "sealed authenticated decoded packet")
        _need(type(arrays) is dict and set(arrays) == set(RESTORE_ORDER),
              "exact caller-owned scratch destination set")
        _need(all(callable(value) for value in (stage, copy, synchronize, guard)),
              "caller-held staging/copy/sync/guard callables")
        object.__setattr__(self, "packet", packet)
        object.__setattr__(self, "arrays", MappingProxyType(dict(arrays)))
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "copy", copy)
        object.__setattr__(self, "synchronize", synchronize)
        object.__setattr__(self, "guard", guard)
        object.__setattr__(self, "_arrays_input", arrays)
        object.__setattr__(self, "_array_value_ids", tuple(
            (name, id(arrays[name])) for name in RESTORE_ORDER
        ))
        object.__setattr__(self, "_callback_ids", tuple(
            id(value) for value in (packet, arrays, device, stream, stage, copy, synchronize, guard)
        ))
        object.__setattr__(self, "_state", {"consumed": False, "completed": False, "sources": []})
        object.__setattr__(self, "destination_signature", self._destination_signature())
        object.__setattr__(self, "_sealed", True)

    @property
    def consumed(self):
        return self._state["consumed"]

    @property
    def completed(self):
        return self._state["completed"]

    @property
    def sources(self):
        return tuple(self._state["sources"])

    def _destination_signature(self):
        return tuple(
            (
                name,
                id(value),
                getattr(value, "ptr", None),
                tuple(getattr(value, "shape", ())),
                tuple(getattr(value, "strides", ())),
                str(getattr(value, "dtype", "")),
                getattr(value, "device", None) is self.device,
            )
            for name, value in sorted(self.arrays.items())
        )

    def _guard(self):
        _need(self.packet is not None and self._callback_ids == tuple(
            id(value) for value in (
                self.packet, self._arrays_input, self.device, self.stream, self.stage,
                self.copy, self.synchronize, self.guard,
            )
        ), "packet, arrays, device, stream and callbacks remain bound")
        _need(type(self._arrays_input) is dict
              and set(self._arrays_input) == set(RESTORE_ORDER)
              and tuple((name, id(self._arrays_input[name])) for name in RESTORE_ORDER)
              == self._array_value_ids,
              "caller array mapping identity and values remain bound")
        self.guard()
        _need(self._destination_signature() == self.destination_signature,
              "caller-owned destination identities/layouts unchanged")

    @staticmethod
    def _method_identity(value, name):
        method = getattr(value, name)
        return (
            getattr(method, "__self__", None) is value,
            getattr(method, "__func__", getattr(type(value), name, None)),
        )

    def _stage_signature(self, source):
        return (
            id(source),
            getattr(source, "ptr", None),
            tuple(getattr(source, "shape", ())),
            tuple(getattr(source, "strides", ())),
            str(getattr(source, "dtype", "")),
            str(getattr(source, "device", "")),
            self._method_identity(source, "numpy"),
        )

    def _verify_stage(self, name, source, signature):
        logical, host_shape, host_dtype = self.packet.specs[name]
        layout = self.packet.layouts[name]
        _need(self._stage_signature(source) == signature,
              "CPU staging identity/pointer/strides unchanged " + name)
        _need(tuple(source.shape) == tuple(logical)
              and str(source.dtype) == layout["warp_dtype"]
              and str(source.device) == "cpu"
              and type(source.ptr) is int and source.ptr > 0,
              "CPU staging allocation layout/device " + name)
        width = 1 if host_dtype == "bool" else 4
        for dimension in host_shape[len(logical):]:
            width *= dimension
        expected_strides, stride = [], width
        for dimension in reversed(logical):
            expected_strides.insert(0, stride)
            stride *= dimension
        _need(len(source.strides) == len(expected_strides)
              and all(actual == expected or (dimension == 1 and actual == 0)
                      for actual, expected, dimension
                      in zip(source.strides, expected_strides, logical)),
              "CPU staging contiguous strides " + name)
        host = source.numpy()
        _need(tuple(host.shape) == tuple(host_shape)
              and str(host.dtype) == host_dtype
              and type(host.nbytes) is int
              and host.nbytes == len(self.packet.fields[name]),
              "CPU staging host view layout " + name)
        raw = host.tobytes(order="C")
        _need(type(raw) is bytes and raw == self.packet.fields[name],
              "whole CPU staging bytes match authenticated source " + name)

    def restore(self):
        _need(not self.consumed, "single-use scratch restore")
        self._state["consumed"] = True
        self._guard()
        # Validate every destination and stage every source before the first
        # asynchronous write. The caller's guard rechecks source/runtime state
        # around each copy; failures never produce a success receipt.
        staged = []
        ranges, source_ranges = [], []
        for name in RESTORE_ORDER:
            destination = self.arrays[name]
            logical, host_shape, host_dtype = self.packet.specs[name]
            layout = self.packet.layouts[name]
            strides = tuple(getattr(destination, "strides", ()))
            _need(tuple(getattr(destination, "shape", ())) == tuple(logical)
                  and len(strides) == len(logical)
                  and str(getattr(destination, "dtype", "")) == layout["warp_dtype"],
                  "destination shape/dtype " + name)
            _need(getattr(destination, "device", None) is self.device,
                  "destination device identity " + name)
            pointer = getattr(destination, "ptr", None)
            _need(type(pointer) is int and pointer > 0,
                  "live destination allocation " + name)
            host_shape = self.packet.specs[name][1]
            host_dtype = self.packet.specs[name][2]
            tail_width = 1 if host_dtype == "bool" else 4
            for dimension in host_shape[len(logical):]:
                tail_width *= dimension
            expected_strides, width = [], tail_width
            for dimension in reversed(logical):
                expected_strides.insert(0, width)
                width *= dimension
            _need(all(actual == expected or (dimension == 1 and actual == 0)
                      for actual, expected, dimension
                      in zip(strides, expected_strides, logical)),
                  "destination contiguous strides " + name)
            span = (0 if any(dimension == 0 for dimension in logical) else
                    tail_width + sum((dimension - 1) * stride
                                     for dimension, stride in zip(logical, strides)))
            ranges.append((pointer, pointer + span, name))
            source = self.stage(
                name, self.packet.fields[name], tuple(logical), tuple(host_shape),
                host_dtype, layout["warp_dtype"],
            )
            _need(tuple(getattr(source, "shape", ())) == tuple(logical)
                  and str(getattr(source, "dtype", "")) == layout["warp_dtype"]
                  and str(getattr(source, "device", "")) == "cpu",
                  "CPU source staging layout/device " + name)
            signature = self._stage_signature(source)
            self._verify_stage(name, source, signature)
            staged.append((name, destination, source, signature))
            source_ranges.append((source.ptr, source.ptr + len(self.packet.fields[name]), name))
        ranges.sort()
        _need(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])),
              "nonoverlapping caller-owned destination allocations")
        source_ranges.sort()
        _need(all(left[1] <= right[0]
                  for left, right in zip(source_ranges, source_ranges[1:])),
              "nonoverlapping fresh CPU staging allocations")
        self._guard()
        for name, destination, source, signature in staged:
            self._guard()
            for staged_name, _staged_destination, staged_source, staged_signature in staged:
                self._verify_stage(staged_name, staged_source, staged_signature)
            self.copy(destination, source, stream=self.stream)
            self._state["sources"].append(source)
            self._verify_stage(name, source, signature)
            self._guard()
        self.synchronize(self.stream)
        self._guard()
        for name, _destination, source, signature in staged:
            self._verify_stage(name, source, signature)
        self._state["completed"] = True
        return self.receipt()

    def receipt(self):
        _need(self.completed, "all scratch copies synchronized and guarded")
        return {
            "protocol": PROTOCOL,
            "decision": "dense-solver-scratch-restored-no-dispatch-or-qualification",
            "source_sha256": self.packet.sha256,
            "source_bytes": len(self.packet.raw),
            "restored_fields": list(RESTORE_ORDER),
            "retained_source_bytes": True,
            "solver_called": False,
            "forward_called": False,
            "flags": dict(FLAGS),
        }
