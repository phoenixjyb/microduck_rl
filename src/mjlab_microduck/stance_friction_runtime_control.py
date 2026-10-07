"""Scoped dense-friction dispatch and full-bank capture, not admission.

Backend objects are supplied by the native child; importing this module has no
GPU or package side effects. The injectable boundary permits CPU hook tests.
"""

from hashlib import sha256
import inspect
from math import prod
from threading import Lock, active_count, current_thread, main_thread

PROTOCOL = "microduck-friction-runtime-control-oct8-v1"
ARMS = ("original", "candidate0", "candidate1")
FORWARDS = 21
INPUT_ORDER = ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep")
OUTPUT_ORDER = (
    "nf",
    "nefc",
    "type",
    "id",
    "row_nnz",
    "row_adr",
    "col_ind",
    "J",
    "pos",
    "margin",
    "D",
    "vel",
    "aref",
    "frictionloss",
    "efc_nnz",
)
BANK_ORDER = ("nf", "nefc", "efc_nnz", *OUTPUT_ORDER[2:-1], "force")
_LOCK = Lock()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(array, device):
    """Actual strided storage span, not broadcast-expanded numpy byte length."""
    shape, strides = tuple(array.shape), tuple(array.strides)
    need(len(shape) == len(strides), "array rank and strides")
    need(
        all(type(x) is int and x >= 0 for x in shape + strides),
        "plain array dimensions",
    )
    host = array.numpy()
    need(str(host.dtype) in ("float32", "int32"), "raw four-byte host dtype")
    need(
        host.shape[: len(shape)] == shape and array.device is device,
        "actual CUDA array shape/device",
    )
    item = host.dtype.itemsize * prod(host.shape[len(shape) :])
    span = (
        0
        if not prod(shape)
        else item + sum((s - 1) * t for s, t in zip(shape, strides))
    )
    pointer = array.ptr
    need(
        (type(pointer) is int and pointer > 0)
        or (span == 0 and (pointer is None or (type(pointer) is int and pointer == 0))),
        "live nonempty allocation or actual null empty array pointer",
    )
    if span:
        need(
            array.is_contiguous
            or (len(shape) > 0 and strides[0] == 0 and span == item * prod(shape[1:])),
            "contiguous or exact leading broadcast allocation",
        )
    return {
        "object_id": id(array),
        "pointer": pointer,
        "span": span,
        "device": str(device),
        "context": device.context,
        "warp_dtype": str(array.dtype),
        "shape": list(shape),
        "strides": list(strides),
        "host_shape": list(host.shape),
        "host_dtype": str(host.dtype),
        "bytes": host.nbytes,
    }


def pack(arrays, order):
    raw, offsets = [], {}
    offset = 0
    for name in order:
        host = arrays[name].numpy()
        need(str(host.dtype) in ("float32", "int32"), "capture float32/int32 only")
        data = host.astype(
            "<f4" if host.dtype.kind == "f" else "<i4", copy=False
        ).tobytes(order="C")
        offsets[name] = {
            "offset": offset,
            "bytes": len(data),
            "shape": list(host.shape),
            "dtype": "<f4" if host.dtype.kind == "f" else "<i4",
        }
        raw.append(data)
        offset += len(data)
    result = b"".join(raw)
    return result, offsets


class RuntimeFrictionObserver:
    """Own only the dynamic constraint entry and its target launch window."""

    def __init__(self, arm, *, wp, constraint, candidate, device, stream, sink, guard):
        need(arm in ARMS, "literal runtime arm")
        self.arm, self.wp, self.constraint = arm, wp, constraint
        self.original = constraint._friction_dof
        self.candidate, self.device, self.stream = candidate, device, stream
        self.sink, self.guard = sink, guard
        self.make = constraint.make_constraint
        self.make_code = inspect.unwrap(self.make).__code__
        self.launch = wp.launch
        self.launch_code = self.launch.__code__
        self.kernel_codes = tuple(
            (k, k.func, k.func.__code__) for k in (self.original, candidate)
        )
        self.make_wrapper, self.launch_wrapper = self._make, self._launch
        self.entries, self.other_launches = [], []
        self.active = self.inside = False
        self.model = self.data = None
        self.bound = None
        self.targets = 0

    def _guard(self):
        need(
            current_thread() is main_thread() and active_count() == 1,
            "single main Python thread",
        )
        need(
            self.constraint._friction_dof is self.original
            and inspect.unwrap(self.make).__code__ is self.make_code
            and self.launch.__code__ is self.launch_code
            and all(k.func is f and f.__code__ is c for k, f, c in self.kernel_codes),
            "held frozen dispatch and kernel code",
        )
        need(
            self.wp.launch is (self.launch_wrapper if self.inside else self.launch)
            and self.constraint.make_constraint
            is (self.make_wrapper if self.active else self.make),
            "owned constraint/launch windows only",
        )
        need(
            not self.device.is_capturing
            and self.device.is_cuda
            and self.stream.device is self.device
            and self.stream.cuda_stream > 0,
            "one eager CUDA device and nonzero stream",
        )
        self.guard()

    def __enter__(self):
        need(
            not self.active and _LOCK.acquire(blocking=False),
            "exclusive new runtime observer",
        )
        try:
            self._guard()
            self.constraint.make_constraint = self.make_wrapper
            self.active = True
            self._guard()
            return self
        except BaseException:
            if self.constraint.make_constraint is self.make_wrapper:
                self.constraint.make_constraint = self.make
            self.active = False
            _LOCK.release()
            raise

    def __exit__(self, kind, value, traceback):
        foreign = (
            self.constraint.make_constraint is not self.make_wrapper
            or self.wp.launch is not self.launch
        )
        if self.constraint.make_constraint is self.make_wrapper:
            self.constraint.make_constraint = self.make
        self.active = False
        _LOCK.release()
        need(not foreign, "foreign hook not overwritten during close")
        if kind is None:
            self._guard()
            need(len(self.entries) == FORWARDS, "exact21 completed constraint targets")

    def _make(self, model, data):
        self._guard()
        need(
            not self.inside and len(self.entries) < FORWARDS,
            "bounded nonreentrant constraint entry",
        )
        need(
            model.nv == 20
            and model.nq == 21
            and model.nbody == 16
            and model.nu == 14
            and model.ntendon == 0
            and model.is_sparse is False
            and model.opt.disableflags == 0
            and data.nworld == 64
            and data.njmax == 512
            and data.njmax_nnz == 10240,
            "literal nominal dense stance plant and capacity",
        )
        if self.model is None:
            self.model, self.data = model, data
        need(model is self.model and data is self.data, "same per-arm model/data")
        count = len(self.entries)
        self.current_forward = count
        self.targets = 0
        self.inside = True
        self.wp.launch = self.launch_wrapper
        try:
            result = self.make(model, data)
            self._guard()
            need(
                self.targets == 1 and len(self.entries) == count + 1,
                "one target per constraint entry",
            )
            return result
        finally:
            owned = self.wp.launch is self.launch_wrapper
            if owned:
                self.wp.launch = self.launch
            self.inside = False
            need(owned, "foreign launch hook not overwritten")

    def _arrays(self, inputs, outputs):
        model, data = self.model, self.data
        expected_inputs = [
            20,
            model.opt.timestep,
            0,
            model.dof_solref,
            model.dof_solimp,
            model.dof_frictionloss,
            model.dof_invweight0,
            False,
            data.qvel,
            512,
            10240,
        ]
        arrays_in = {
            name: inputs[index] for name, index in zip(INPUT_ORDER, (5, 8, 6, 3, 4, 1))
        }
        expected_outputs = [
            data.nf,
            data.nefc,
            data.efc.type,
            data.efc.id,
            data.efc.J_rownnz,
            data.efc.J_rowadr,
            data.efc.J_colind,
            data.efc.J,
            data.efc.pos,
            data.efc.margin,
            data.efc.D,
            data.efc.vel,
            data.efc.aref,
            data.efc.frictionloss,
        ]
        need(
            all(
                actual is expected
                for actual, expected in zip(inputs, expected_inputs)
                if not isinstance(expected, (int, bool))
            ),
            "actual model/data input objects",
        )
        need(
            all(
                actual is expected
                for actual, expected in zip(outputs[:14], expected_outputs)
            ),
            "actual data output objects",
        )
        arrays_out = dict(zip(OUTPUT_ORDER, outputs)) | {"force": data.efc.force}
        return arrays_in, arrays_out

    def _launch(self, kernel, *args, **kwargs):
        self._guard()
        if kernel is not self.original:
            self.other_launches.append(
                {"forward": self.current_forward, "kernel": kernel.key}
            )
            return self.launch(kernel, *args, **kwargs)
        need(
            not args and set(kwargs) == {"dim", "inputs", "outputs"},
            "exact frozen friction invocation",
        )
        inputs, outputs = kwargs["inputs"], kwargs["outputs"]
        need(
            type(inputs) is list
            and len(inputs) == 11
            and type(outputs) is list
            and len(outputs) == 15,
            "26 argument lists",
        )
        need(
            type(kwargs["dim"]) is tuple
            and len(kwargs["dim"]) == 2
            and all(type(value) is int for value in kwargs["dim"])
            and kwargs["dim"] == (64, 20)
            and all(type(inputs[i]) is int for i in (0, 2, 9, 10))
            and [inputs[i] for i in (0, 2, 9, 10)] == [20, 0, 512, 10240]
            and inputs[7] is False
            and self.targets == 0,
            "literal dense friction launch and single target",
        )
        arrays_in, arrays_out = self._arrays(inputs, outputs)
        all_arrays = {"input." + k: v for k, v in arrays_in.items()} | {
            "bank." + k: v for k, v in arrays_out.items()
        }
        layouts = {k: identity(v, self.device) for k, v in all_arrays.items()}
        ranges = sorted(
            (v["pointer"], v["pointer"] + v["span"], k)
            for k, v in layouts.items()
            if v["span"]
        )
        need(
            all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])),
            "nonoverlapping actual kernel banks and inputs",
        )
        need(
            all(v.is_contiguous for v in arrays_out.values()),
            "contiguous written banks and dense-unused scratch",
        )
        stable = {k: v for k, v in layouts.items() if k != "bank.efc_nnz"}
        if self.bound is None:
            self.bound = stable
        need(stable == self.bound, "stable per-arm allocation layout")
        index = len(self.entries)
        packets = {}
        for label, arrays, order in (
            ("inputs", arrays_in, INPUT_ORDER),
            ("bank", arrays_out, BANK_ORDER),
        ):
            raw, offsets = pack(arrays, order)
            name = f"{self.arm}/entry-{index:02d}.{label}.before.bin"
            self.sink(name, raw)
            packets[label + ".before"] = {
                "path": name,
                "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(),
                "fields": offsets,
            }
        selected = self.original if self.arm == "original" else self.candidate
        result = self.launch(selected, **kwargs, record_tape=False)
        self.wp.synchronize_stream(self.stream)
        self._guard()
        need(
            layouts == {k: identity(v, self.device) for k, v in all_arrays.items()},
            "entry layouts closed unchanged",
        )
        for label, arrays, order in (
            ("inputs", arrays_in, INPUT_ORDER),
            ("bank", arrays_out, BANK_ORDER),
        ):
            raw, offsets = pack(arrays, order)
            name = f"{self.arm}/entry-{index:02d}.{label}.after.bin"
            self.sink(name, raw)
            packets[label + ".after"] = {
                "path": name,
                "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(),
                "fields": offsets,
            }
        self.entries.append(
            {
                "index": index,
                "arm": self.arm,
                "packets": packets,
                "layouts": layouts,
                "dim": [64, 20],
                "scalars": dict(
                    nv=20, disableflags=0, is_sparse=False, njmax=512, njmax_nnz=10240
                ),
                "kernel": selected.key,
                "kernel_object_id": id(selected),
                "kernel_function_id": id(selected.func),
                "kernel_code_id": id(selected.func.__code__),
                "launch_function_id": id(self.launch),
                "launch_code_id": id(self.launch_code),
                "make_function_id": id(self.make),
                "make_code_id": id(self.make_code),
                "stream": self.stream.cuda_stream,
                "device_context": self.device.context,
                "model_object_id": id(self.model),
                "data_object_id": id(self.data),
                "record_tape": False,
            }
        )
        self.targets += 1
        return result

    def receipt(self):
        need(
            not self.active and len(self.entries) == FORWARDS,
            "closed complete observer",
        )
        return {
            "protocol": PROTOCOL,
            "arm": self.arm,
            "entries": self.entries,
            "other_launches": self.other_launches,
            "owned_hooks_restored": True,
        }
