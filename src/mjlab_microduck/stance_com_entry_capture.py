"""Narrow original-call-site CoM observation; no replay or training admission."""

from copy import deepcopy
from hashlib import sha256
from threading import Lock, current_thread, main_thread

import numpy as np
import warp as wp
import mujoco_warp as public
from mujoco_warp._src import forward, smooth

from mjlab_microduck import stance_crb_runtime_control as source_checks
from mjlab_microduck import stance_forward_reduction_plan as planner

PROTOCOL = "microduck-com-actual-entry-capture-v1"
MAX_CALLS = 2
_LOCK = Lock()
_ENTRY = smooth.com_pos
_PUBLIC_ENTRY = public.com_pos
_LAUNCH = wp.launch
_KERNELS = tuple(
    getattr(smooth, name)
    for name in (
        "_subtree_com_init",
        "_subtree_com_acc",
        "_subtree_div",
        "_cinert",
        "_cdof",
    )
)
_FORWARD = forward.forward
_POSITION = forward.fwd_position
_FLAGS = {
    "original_run_entry_captured": False,
    "original_pair_accepted": False,
    "actual_kernel_order_observed": False,
    "runtime_cause_proven": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _fingerprint(array):
    return (
        id(array),
        int(array.ptr),
        tuple(array.shape),
        tuple(array.strides),
        array.dtype,
        array.device,
        array.is_contiguous,
    )


def _stream(device):
    return None if str(device) == "cpu" else int(wp.get_stream(device).cuda_stream or 0)


def _sources():
    _need(
        source_checks._source_digest(smooth.__file__)
        == source_checks.PINNED_SMOOTH_SHA256
        and source_checks._source_digest(forward.__file__)
        == source_checks.PINNED_FORWARD_SHA256,
        "exact pinned CoM source and forward dispatch",
    )
    _need(
        _ENTRY is _PUBLIC_ENTRY
        and forward.smooth is smooth
        and forward.forward is _FORWARD
        and forward.fwd_position is _POSITION
        and smooth.wp is wp
        and public.com_pos is _PUBLIC_ENTRY
        and tuple(
            getattr(smooth, name)
            for name in (
                "_subtree_com_init",
                "_subtree_com_acc",
                "_subtree_div",
                "_cinert",
                "_cdof",
            )
        )
        == _KERNELS,
        "exact CoM callable and kernel references",
    )


def _layout(model, data):
    _need(
        (model.nbody, model.nq, model.nv, model.nu) == (16, 21, 20, 14)
        and model.is_sparse is False,
        "fixed dense MicroDuck dimensions",
    )
    device = model.body_parentid.device
    _need(
        type(data.nworld) is int
        and (
            (data.nworld == 2 and str(device) == "cpu")
            or (data.nworld == 64 and str(device) == "cuda:0")
        ),
        "two CPU or 64 CUDA worlds",
    )
    _need(
        not device.is_capturing and wp.get_device() == device,
        "bound eager current device",
    )
    parents = source_checks._warp_int_ids(
        model.body_parentid, 16, device, "CoM parents"
    )
    _need(parents == source_checks.EXPECTED_PARENTS, "fixed CoM parents")
    _need(
        type(model.body_tree) is tuple and len(model.body_tree) == 7,
        "seven CoM tree levels",
    )
    levels = tuple(
        source_checks._warp_int_ids(row, len(expected), device, "CoM level")
        for row, expected in zip(
            model.body_tree, source_checks.EXPECTED_DEPTH_LEVELS, strict=True
        )
    )
    _need(levels == source_checks.EXPECTED_DEPTH_LEVELS, "fixed CoM tree IDs")
    for name in ("subtree_com", "xipos"):
        array = getattr(data, name)
        _need(
            array.dtype is wp.vec3
            and array.shape == (data.nworld, 16)
            and array.device == device
            and array.is_contiguous is True,
            "bound contiguous CoM vec3 arrays",
        )
    _need(
        data.xipos is not data.subtree_com and data.xipos.ptr != data.subtree_com.ptr,
        "distinct position and CoM storage",
    )
    for name in ("body_mass", "body_subtreemass"):
        array = getattr(model, name)
        _need(
            array.dtype is wp.float32
            and array.shape in ((1, 16), (data.nworld, 16))
            and array.device == device
            and array.is_contiguous is True,
            "bound per-world CoM mass rows",
        )
    return device


class ComEntryCapture:
    """Observe exactly two eager CoM calls without replacing their kernels.

    Reading back the actual initialized aliased array changes timing. The scope
    never claims that a fresh instrumented entry is the old rejected run entry.
    """

    def __init__(self):
        self._status = "not-started"
        self._fault = None
        self._entries = []
        self._records = []
        self._binding = None
        self._entry_wrapper = None
        self._owns_lock = False
        self._inside = False

    def __enter__(self):
        _need(self._status == "not-started", "one-shot CoM scope")
        _need(current_thread() is main_thread(), "main-thread CoM scope")
        _need(_LOCK.acquire(blocking=False), "exclusive process-local CoM scope")
        self._owns_lock = True
        try:
            _OBSERVER_CHECK(self)
            _sources()
            _need(
                smooth.com_pos is _ENTRY and wp.launch is _LAUNCH,
                "unmodified CoM entry and launch",
            )
            self._entry_wrapper = self._call
            smooth.com_pos = self._entry_wrapper
            self._status = "active"
            return self
        except BaseException as error:
            self._fault = type(error).__name__
            self._status = "faulted"
            _LOCK.release()
            self._owns_lock = False
            raise

    def __exit__(self, error_type, error, traceback):
        foreign = smooth.com_pos is not self._entry_wrapper or wp.launch is not _LAUNCH
        if smooth.com_pos is self._entry_wrapper:
            smooth.com_pos = _ENTRY
        if self._owns_lock:
            _LOCK.release()
            self._owns_lock = False
        if (
            error is not None
            or foreign
            or self._fault is not None
            or len(self._entries) != MAX_CALLS
        ):
            self._status = "faulted"
            self._fault = self._fault or (
                error_type.__name__ if error_type else "IncompleteOrForeignScope"
            )
            if error is None:
                raise ValueError("complete exact two-call CoM capture required")
        else:
            self._status = "complete"
        return False

    def _call(self, model, data):
        try:
            _OBSERVER_CHECK(self)
            _need(
                current_thread() is main_thread()
                and not self._inside
                and self._status == "active",
                "owned nonrecursive main-thread CoM entry",
            )
            _need(
                len(self._entries) < MAX_CALLS
                and smooth.com_pos is self._entry_wrapper
                and wp.launch is _LAUNCH,
                "bounded unmodified CoM dispatch",
            )
            _sources()
            device = _layout(model, data)
            self._before_call(model, data, device)
            arrays = [
                model.body_parentid,
                *model.body_tree,
                model.body_mass,
                model.body_subtreemass,
                data.xipos,
                data.subtree_com,
            ]
            binding = (
                id(model),
                id(data),
                data.nworld,
                tuple(_fingerprint(a) for a in arrays),
                _stream(device),
            )
            if self._binding is None:
                self._binding = binding
            _need(
                binding == self._binding,
                "unchanged CoM objects, aliases, stream and layout",
            )
            stages = [
                (
                    _KERNELS[0],
                    (data.nworld, 16),
                    [model.body_mass, data.xipos],
                    [data.subtree_com],
                )
            ]
            stages.extend(
                (
                    _KERNELS[1],
                    (data.nworld, row.size),
                    [model.body_parentid, data.subtree_com, row],
                    [data.subtree_com],
                )
                for row in reversed(model.body_tree)
            )
            stages.extend(
                [
                    (
                        _KERNELS[2],
                        (data.nworld, 16),
                        [model.body_subtreemass, data.subtree_com],
                        [data.subtree_com],
                    ),
                    (
                        _KERNELS[3],
                        (data.nworld, 16),
                        [
                            model.body_rootid,
                            model.body_mass,
                            model.body_inertia,
                            data.xipos,
                            data.ximat,
                            data.subtree_com,
                        ],
                        [data.cinert],
                    ),
                    (
                        _KERNELS[4],
                        (data.nworld, model.njnt),
                        [
                            model.body_rootid,
                            model.jnt_type,
                            model.jnt_dofadr,
                            model.jnt_bodyid,
                            data.xmat,
                            data.xanchor,
                            data.xaxis,
                            data.subtree_com,
                        ],
                        [data.cdof],
                    ),
                ]
            )
            index = 0
            raw = None

            def launch(*args, **kwargs):
                nonlocal index, raw
                _need(
                    current_thread() is main_thread()
                    and wp.launch is launch
                    and self._inside,
                    "owned CoM launch wrapper",
                )
                _need(
                    not device.is_capturing
                    and _stream(device) == binding[-1]
                    and wp.get_device() == device,
                    "unchanged eager CoM stream/device",
                )
                _need(
                    index < len(stages)
                    and len(args) == 1
                    and set(kwargs) == {"dim", "inputs", "outputs"},
                    "exact ordered CoM launch signature",
                )
                kernel, dim, inputs, outputs = stages[index]
                _need(
                    args[0] is kernel
                    and type(kwargs["dim"]) is tuple
                    and kwargs["dim"] == dim
                    and type(kwargs["inputs"]) is list
                    and len(kwargs["inputs"]) == len(inputs)
                    and all(
                        a is b for a, b in zip(kwargs["inputs"], inputs, strict=True)
                    )
                    and type(kwargs["outputs"]) is list
                    and len(kwargs["outputs"]) == len(outputs)
                    and all(
                        a is b for a, b in zip(kwargs["outputs"], outputs, strict=True)
                    ),
                    "exact bound original CoM kernels, dimensions and array aliases",
                )
                _need(
                    tuple(_fingerprint(a) for a in arrays) == binding[3],
                    "CoM arrays unchanged during call",
                )
                result = self._dispatch(index, args, kwargs)
                if index == 0:
                    wp.synchronize_device(device)
                    values = data.subtree_com.numpy()
                    _need(
                        values.shape == (data.nworld, 16, 3)
                        and values.dtype == np.dtype(np.float32)
                        and np.isfinite(values).all(),
                        "finite actual mass-weighted CoM entry",
                    )
                    raw = values.astype("<f4", copy=False).tobytes(order="C")
                index += 1
                return result

            self._inside = True
            wp.launch = launch
            try:
                result = _ENTRY(model, data)
                _need(
                    result is None and index == 11 and type(raw) is bytes,
                    "complete unchanged CoM call with observed initialization boundary",
                )
                _need(
                    wp.launch is launch and smooth.com_pos is self._entry_wrapper,
                    "CoM scope references unchanged at return",
                )
                self._after_call(model, data, device)
                _sources()
            finally:
                if wp.launch is launch:
                    wp.launch = _LAUNCH
                else:
                    self._fault = self._fault or "ForeignLaunchReference"
                self._inside = False
            self._entries.append(raw)
            self._records.append(
                {
                    "call_index": len(self._entries) - 1,
                    "bytes": len(raw),
                    "sha256": sha256(raw).hexdigest(),
                    "worlds": data.nworld,
                    "device": str(device),
                    "stream": binding[-1],
                    "subtree_com_layout": {
                        "object_id": id(data.subtree_com),
                        "ptr": int(data.subtree_com.ptr),
                        "shape": list(data.subtree_com.shape),
                        "strides": list(data.subtree_com.strides),
                        "dtype": "vec3f",
                        "contiguous": data.subtree_com.is_contiguous,
                    },
                    "smooth_sha256": source_checks.PINNED_SMOOTH_SHA256,
                    "forward_sha256": source_checks.PINNED_FORWARD_SHA256,
                    "original_launches": index,
                    "boundary": "after-init-before-first-accumulation",
                    "initialization_and_accumulation_output_alias": True,
                    "readback_and_device_sync_perturb_timing": True,
                }
            )
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise

    def _check_observer_methods(self):
        _need(
            type(self)
            in (ComEntryCapture, CpuSerialComControl, NativeCoupledComControl)
            and getattr(self._call, "__func__", None) is _OBSERVER_CALL,
            "exact observation or two-world CPU-only coupled scope",
        )
        if type(self) is ComEntryCapture:
            _need(
                all(
                    getattr(getattr(self, name), "__func__", None) is method
                    for name, method in _OBSERVER_METHODS
                ),
                "bound original observer methods",
            )
        elif type(self) is NativeCoupledComControl:
            _need(
                all(
                    getattr(getattr(self, name), "__func__", None) is method
                    for name, method in _NATIVE_CONTROL_METHODS
                ),
                "bound native coupled methods",
            )

    def _before_call(self, model, data, device):
        """Default observation changes no dispatch or array."""

    def _dispatch(self, index, args, kwargs):
        return _LAUNCH(*args, **kwargs)

    def _after_call(self, model, data, device):
        """Default observation requires no additional dispatch."""

    @property
    def entries(self):
        return tuple(self._entries)

    @property
    def receipt(self):
        return deepcopy(
            {
                "protocol": PROTOCOL,
                "status": self._status,
                "fault_type": self._fault,
                "parents": list(source_checks.EXPECTED_PARENTS),
                "reversed_levels": [
                    list(row) for row in source_checks.EXPECTED_REVERSED_LEVELS
                ],
                "calls": self._records,
                "flags": dict(_FLAGS),
            }
        )


class CpuSerialComControl(ComEntryCapture):
    """Two-world CPU prototype only; not accepted by the native receiver."""

    def __init__(self):
        super().__init__()
        self._split_rows = None
        self._split_binding = None
        self._kernel_counts = []

    def __enter__(self):
        _need(str(wp.get_device()) == "cpu", "CPU-only coupled CoM prototype")
        return super().__enter__()

    def _before_call(self, model, data, device):
        _need(
            type(self) is CpuSerialComControl
            and str(device) == "cpu"
            and type(data.nworld) is int
            and data.nworld == 2,
            "exact two-world CPU-only coupled CoM prototype before kernels",
        )
        if self._split_rows is None:
            self._split_rows = tuple(
                wp.array([body], dtype=wp.int32, device=device) for body in (2, 7, 11)
            )
            self._split_binding = tuple(_fingerprint(row) for row in self._split_rows)
        self._check_rows(device)
        self._kernel_counts.append(0)

    def _check_rows(self, device):
        _need(
            type(self._split_rows) is tuple
            and len(self._split_rows) == 3
            and tuple(_fingerprint(row) for row in self._split_rows)
            == self._split_binding,
            "unchanged CPU sibling-control storage",
        )
        _need(
            tuple(
                source_checks._warp_int_ids(row, 1, device, "CPU control singleton")
                for row in self._split_rows
            )
            == ((2,), (7,), (11,)),
            "exact CPU sibling-control IDs",
        )

    def _dispatch(self, index, args, kwargs):
        _need(
            type(self) is CpuSerialComControl
            and str(kwargs["outputs"][0].device) == "cpu"
            and self._kernel_counts,
            "owned CPU-only control dispatch",
        )
        if index == 5:
            self._check_rows(kwargs["outputs"][0].device)
            _need(
                args[0] is _KERNELS[1]
                and kwargs["dim"] == (2, 3)
                and kwargs["inputs"][1] is kwargs["outputs"][0],
                "one bound original aliased CPU sibling group",
            )
            for row in self._split_rows:
                _LAUNCH(
                    args[0],
                    dim=(2, 1),
                    inputs=[*kwargs["inputs"][:2], row],
                    outputs=kwargs["outputs"],
                )
                self._kernel_counts[-1] += 1
            return None
        result = _LAUNCH(*args, **kwargs)
        self._kernel_counts[-1] += 1
        return result

    def _after_call(self, model, data, device):
        self._check_rows(device)
        _need(self._kernel_counts[-1] == 13, "complete thirteen-call CPU CoM dispatch")

    @property
    def receipt(self):
        result = super().receipt
        result.update(
            protocol=PROTOCOL + ":cpu-serial-control",
            cpu_only_prototype=True,
            dispatch_transformation="split-siblings-2-7-11-only",
            dispatched_original_kernel_counts=list(self._kernel_counts),
            complete_kernel_count_matches=(
                result["status"] == "complete" and self._kernel_counts == [13, 13]
            ),
        )
        return result


class NativeCoupledComControl(ComEntryCapture):
    """Fresh CUDA64 paired-control protocol; never old observation admission.

    Both modes read the actual weighted array before division. Only serial mode
    splits the original sibling launch. Resource/lease/source admission belongs
    to the separate fresh supervisor, not this process-local hook.
    """

    def __init__(self, *, mode):
        _need(
            type(mode) is str and mode in ("original", "serial"),
            "literal coupled CoM mode",
        )
        super().__init__()
        self.mode = self._declared_mode = mode
        self._split_rows = self._split_binding = None
        self._kernel_counts = []
        self._weighted_entries = []

    def __enter__(self):
        _need(str(wp.get_device()) == "cuda:0", "native coupled CoM requires CUDA0")
        return super().__enter__()

    def _before_call(self, model, data, device):
        _need(
            type(self) is NativeCoupledComControl
            and type(self.mode) is str
            and self.mode == self._declared_mode
            and str(device) == "cuda:0"
            and type(data.nworld) is int
            and data.nworld == 64,
            "exact fixed-mode CUDA64 coupled CoM before kernels",
        )
        if self._split_rows is None:
            # Identical allocation/readback setup in both modes; only dispatch varies.
            self._split_rows = tuple(
                wp.array([body], dtype=wp.int32, device=device) for body in (2, 7, 11)
            )
            self._split_binding = tuple(_fingerprint(row) for row in self._split_rows)
        self._check_rows(device)
        self._kernel_counts.append(0)

    def _check_rows(self, device):
        _need(
            type(self._split_rows) is tuple
            and len(self._split_rows) == 3
            and tuple(_fingerprint(row) for row in self._split_rows)
            == self._split_binding,
            "unchanged native sibling-control storage",
        )
        _need(
            tuple(
                source_checks._warp_int_ids(row, 1, device, "native singleton")
                for row in self._split_rows
            )
            == ((2,), (7,), (11,)),
            "exact native sibling-control IDs",
        )

    def _dispatch(self, index, args, kwargs):
        device = kwargs["outputs"][0].device
        _need(
            type(self) is NativeCoupledComControl
            and str(device) == "cuda:0"
            and self.mode == self._declared_mode
            and self._kernel_counts,
            "owned fixed-mode native coupled dispatch",
        )
        if index == 8:
            _need(
                args[0] is _KERNELS[2]
                and len(self._weighted_entries) == len(self._kernel_counts) - 1,
                "once per-call actual weighted sum before original division",
            )
            wp.synchronize_device(device)
            values = kwargs["outputs"][0].numpy()
            _need(
                values.shape == (64, 16, 3)
                and values.dtype == np.dtype(np.float32)
                and np.isfinite(values).all(),
                "finite complete native weighted array",
            )
            self._weighted_entries.append(
                values.astype("<f4", copy=False).tobytes(order="C")
            )
        if index == 5:
            # Mirror the validation readbacks in both modes. Only the kernel
            # dispatch order is the paired intervention at this boundary.
            self._check_rows(device)
        if index == 5 and self.mode == "serial":
            _need(
                args[0] is _KERNELS[1]
                and kwargs["dim"] == (64, 3)
                and kwargs["inputs"][1] is kwargs["outputs"][0],
                "one bound original aliased native sibling group",
            )
            for row in self._split_rows:
                _LAUNCH(
                    args[0],
                    dim=(64, 1),
                    inputs=[*kwargs["inputs"][:2], row],
                    outputs=kwargs["outputs"],
                )
                self._kernel_counts[-1] += 1
            return None
        result = _LAUNCH(*args, **kwargs)
        self._kernel_counts[-1] += 1
        return result

    def _after_call(self, model, data, device):
        self._check_rows(device)
        expected = 13 if self.mode == "serial" else 11
        _need(
            self._kernel_counts[-1] == expected
            and len(self._weighted_entries) == len(self._kernel_counts),
            "complete native coupled dispatch and weighted boundary",
        )

    @property
    def weighted_entries(self):
        return tuple(self._weighted_entries)

    @property
    def receipt(self):
        result = super().receipt
        expected = 13 if self.mode == "serial" else 11
        result.update(
            protocol=PROTOCOL + ":native-coupled-control",
            mode=self.mode,
            dispatched_original_kernel_counts=list(self._kernel_counts),
            split_body_ids=[2, 7, 11] if self.mode == "serial" else [],
            weighted_boundary="after-accumulation-before-original-division",
            weighted_boundary_readback_perturbs_timing=True,
            weighted_sha256=[sha256(raw).hexdigest() for raw in self._weighted_entries],
            complete_kernel_count_matches=result["status"] == "complete"
            and self._kernel_counts == [expected, expected]
            and len(self._weighted_entries) == 2,
        )
        return result


_OBSERVER_CALL = ComEntryCapture._call
_OBSERVER_CHECK = ComEntryCapture._check_observer_methods
_OBSERVER_METHODS = tuple(
    (name, getattr(ComEntryCapture, name))
    for name in ("_before_call", "_dispatch", "_after_call")
)
_NATIVE_CONTROL_METHODS = tuple(
    (name, getattr(NativeCoupledComControl, name))
    for name in ("_before_call", "_dispatch", "_after_call", "_check_rows")
)


def repeat_entry(model, data, raw, digest, mode):
    """Run original CoM accumulation on a detached aliased scratch array.

    Source/CPU tests or this return alone cannot establish native admission.
    No division/downstream kernels or physics integration are launched here.
    """
    _sources()
    _need(
        wp.launch is _LAUNCH and smooth.com_pos is _ENTRY,
        "no capture hook during repeats",
    )
    device = _layout(model, data)
    _need(
        type(raw) is bytes
        and len(raw) == data.nworld * 16 * 3 * 4
        and type(digest) is str
        and sha256(raw).hexdigest() == digest,
        "whole exact entry input bytes",
    )
    _need(
        type(mode) is str and mode in ("concurrent", "serial"), "fixed CoM repeat mode"
    )
    values = np.frombuffer(raw, dtype="<f4").reshape(data.nworld, 16, 3)
    _need(np.isfinite(values).all(), "finite repeat entry")
    scratch = wp.array(values.copy(), dtype=wp.vec3, device=device)
    fixed = wp.array(values.copy(), dtype=wp.vec3, device=device)
    topology = {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": list(source_checks.EXPECTED_PARENTS),
        "reversed_body_tree_ids": [
            list(row) for row in source_checks.EXPECTED_REVERSED_LEVELS
        ],
    }
    plan = planner.plan(topology, planner.TOPOLOGY_SHA256, "subtree_com")
    groups = (
        list(reversed(model.body_tree))
        if mode == "concurrent"
        else [
            wp.array(row["body_ids"], dtype=wp.int32, device=device)
            for row in plan["launch_groups"]
        ]
    )
    binding, stream = _fingerprint(scratch), _stream(device)
    snapshots = []
    for _ in range(32):
        _need(
            not device.is_capturing
            and _stream(device) == stream
            and wp.launch is _LAUNCH,
            "unchanged eager repeat stream and dispatch",
        )
        wp.copy(scratch, fixed)
        for group in groups:
            _LAUNCH(
                _KERNELS[1],
                dim=(data.nworld, group.size),
                inputs=[model.body_parentid, scratch, group],
                outputs=[scratch],
            )
        wp.synchronize_device(device)
        _need(_fingerprint(scratch) == binding, "in-place repeat alias preserved")
        result = scratch.numpy()
        _need(np.isfinite(result).all(), "finite repeat weighted sum")
        snapshots.append(result.astype("<f4", copy=False).tobytes(order="C"))
    _sources()
    output = b"".join(snapshots)
    return output, {
        "mode": mode,
        "repeats": 32,
        "resets": 32,
        "accumulation_launches": 32 * len(groups),
        "bytes": len(output),
        "input_sha256": digest,
        "output_sha256": sha256(output).hexdigest(),
        "output_boundary": "after-accumulation-before-division",
        "stream": stream,
        "scratch_ptr": int(scratch.ptr),
        "scratch_shape": list(scratch.shape),
        "scratch_strides": list(scratch.strides),
        "input_output_alias": True,
        "flags": dict(_FLAGS),
    }
