"""Observe fresh initialized RNE inputs; no historical or training admission.

The observer keeps every original backward launch. Detached repeat banks never
write the live Data array or replace a forward result. Synchronization/readback
changes timing and is explicitly reported; actual atomic arrival order is not
observed.
"""

from copy import deepcopy
from hashlib import sha256
import inspect
from threading import Lock, current_thread, main_thread

import numpy as np
import warp as wp
import mujoco_warp as public
from mujoco_warp._src import forward, smooth

from mjlab_microduck import stance_crb_runtime_control as source_checks

PROTOCOL = "microduck-rne-actual-entry-oct7-v1"
MAX_CALLS, REPEATS = 2, 32
_LOCK = Lock()
_ENTRY = smooth._rne_cfrc_backward
_RNE = smooth.rne
_RNE_CODE = inspect.unwrap(_RNE).__code__
_POST = smooth.rne_postconstraint
_POST_CODE = inspect.unwrap(_POST).__code__
_LAUNCH = wp.launch
_KERNEL = smooth._cfrc_backward
_FORWARD = forward.forward
_VELOCITY = forward.fwd_velocity
_PREFIX = tuple(
    (name, getattr(smooth, name))
    for name in (
        "_rne_cacc_world",
        "_rne_cacc_forward",
        "_rne_cfrc",
        "_cacc_world",
        "_cacc_branch",
        "_cfrc",
        "_qfrc_bias",
    )
)
FLAGS = {
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
        "exact pinned RNE and forward sources",
    )
    _need(
        smooth.rne is _RNE
        and public.rne is _RNE
        and smooth.rne_postconstraint is _POST
        and public.rne_postconstraint is _POST
        and inspect.unwrap(_POST).__code__ is _POST_CODE
        and inspect.unwrap(_RNE).__code__ is _RNE_CODE
        and smooth._cfrc_backward is _KERNEL
        and forward.smooth is smooth
        and smooth.wp is wp
        and forward.forward is _FORWARD
        and forward.fwd_velocity is _VELOCITY
        and all(getattr(smooth, name) is value for name, value in _PREFIX),
        "unmodified RNE initialization, dot product and forward references",
    )


def _layout(model, data):
    _need(
        (model.nbody, model.nq, model.nv, model.nu) == (16, 21, 20, 14)
        and model.is_sparse is False,
        "fixed dense MicroDuck RNE dimensions",
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
        "bound eager current RNE device",
    )
    _need(
        source_checks._warp_int_ids(model.body_parentid, 16, device, "RNE parents")
        == source_checks.EXPECTED_PARENTS,
        "fixed RNE parents",
    )
    _need(
        type(model.body_tree) is tuple and len(model.body_tree) == 7,
        "seven RNE tree levels",
    )
    levels = tuple(
        source_checks._warp_int_ids(row, len(expected), device, "RNE level")
        for row, expected in zip(
            model.body_tree, source_checks.EXPECTED_DEPTH_LEVELS, strict=True
        )
    )
    _need(levels == source_checks.EXPECTED_DEPTH_LEVELS, "fixed RNE tree IDs")
    for name in ("cfrc_int", "cacc", "cvel"):
        array = getattr(data, name)
        _need(
            array.dtype is wp.spatial_vector
            and array.shape == (data.nworld, 16)
            and array.device == device
            and array.is_contiguous is True,
            "bound contiguous RNE spatial arrays",
        )
    _need(
        len({int(getattr(data, name).ptr) for name in ("cfrc_int", "cacc", "cvel")})
        == 3,
        "distinct force, acceleration and velocity storage",
    )
    return device


def _snapshot(array, worlds):
    wp.synchronize_device(array.device)
    values = array.numpy()
    _need(
        values.shape == (worlds, 16, 6)
        and values.dtype == np.dtype(np.float32)
        and bool(np.isfinite(values).all()),
        "finite full RNE spatial snapshot",
    )
    return values.astype("<f4", copy=False).tobytes(order="C")


class RneEntryCapture:
    """Two original RNE backward calls, immediately after its original initializer."""

    def __init__(self):
        self._status, self._fault = "not-started", None
        self._entries, self._outputs, self._records = [], [], []
        self._binding, self._wrapper = None, None
        self._owns_lock, self._inside = False, False
        self._post_calls = 0

    def __enter__(self):
        _need(
            self._status == "not-started" and current_thread() is main_thread(),
            "one-shot main-thread RNE scope",
        )
        _need(_LOCK.acquire(blocking=False), "exclusive process-local RNE scope")
        self._owns_lock = True
        try:
            _methods(self)
            _sources()
            _need(
                smooth._rne_cfrc_backward is _ENTRY and wp.launch is _LAUNCH,
                "unmodified RNE backward and launch",
            )
            self._wrapper = self._call
            smooth._rne_cfrc_backward = self._wrapper
            self._status = "active"
            return self
        except BaseException as error:
            self._status, self._fault = "faulted", type(error).__name__
            self._owns_lock = False
            _LOCK.release()
            raise

    def __exit__(self, error_type, error, traceback):
        foreign = (
            smooth._rne_cfrc_backward is not self._wrapper or wp.launch is not _LAUNCH
        )
        if smooth._rne_cfrc_backward is self._wrapper:
            smooth._rne_cfrc_backward = _ENTRY
        if self._owns_lock:
            self._owns_lock = False
            _LOCK.release()
        if (
            error is not None
            or foreign
            or self._fault
            or len(self._entries) != MAX_CALLS
            or self._post_calls != MAX_CALLS
        ):
            self._status = "faulted"
            self._fault = self._fault or (
                error_type.__name__ if error_type else "IncompleteOrForeignScope"
            )
            if error is None:
                raise ValueError("complete exact two-call RNE capture required")
        else:
            self._status = "complete"
        return False

    def _call(self, model, data):
        try:
            _methods(self)
            frame = inspect.currentframe()
            try:
                caller = frame.f_back
                is_post = caller.f_code is _POST_CODE
                _need(
                    (is_post or caller.f_code is _RNE_CODE)
                    and caller.f_locals.get("m") is model
                    and caller.f_locals.get("d") is data
                    and (is_post or caller.f_locals.get("flg_acc") is False),
                    "actual unchanged RNE initializer caller, no direct scratch entry",
                )
            finally:
                del frame, caller
            _need(
                current_thread() is main_thread()
                and not self._inside
                and self._status == "active"
                and (is_post or len(self._entries) < MAX_CALLS)
                and smooth._rne_cfrc_backward is self._wrapper
                and wp.launch is _LAUNCH,
                "bounded owned nonrecursive RNE call",
            )
            _sources()
            device = _layout(model, data)
            arrays = [
                model.body_parentid,
                *model.body_tree,
                data.cfrc_int,
                data.cacc,
                data.cvel,
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
            _need(binding == self._binding, "unchanged RNE arrays, aliases and stream")
            if is_post:
                # Sensor acceleration uses the same helper after the bias dot.
                # Preserve that other original caller; it is not bias input.
                _need(
                    self._post_calls < MAX_CALLS
                    and len(self._entries) == self._post_calls + 1,
                    "one bounded original sensory passthrough after each bias capture",
                )
                result = _ENTRY(model, data)
                _need(
                    result is None
                    and smooth._rne_cfrc_backward is self._wrapper
                    and wp.launch is _LAUNCH,
                    "unchanged original sensory passthrough",
                )
                _sources()
                self._post_calls += 1
                return result
            initialized = _snapshot(data.cfrc_int, data.nworld)
            levels = tuple(reversed(model.body_tree))
            index = 0

            def launch(*args, **kwargs):
                nonlocal index
                _need(
                    current_thread() is main_thread()
                    and self._inside
                    and wp.launch is launch
                    and index < 7
                    and len(args) == 1
                    and set(kwargs) == {"dim", "inputs", "outputs"},
                    "owned exact ordered RNE launch signature",
                )
                row = levels[index]
                inputs = [model.body_parentid, data.cfrc_int, row]
                _need(
                    args[0] is _KERNEL
                    and type(kwargs["dim"]) is list
                    and kwargs["dim"] == [data.nworld, row.size]
                    and type(kwargs["inputs"]) is list
                    and len(kwargs["inputs"]) == 3
                    and all(
                        a is b for a, b in zip(kwargs["inputs"], inputs, strict=True)
                    )
                    and type(kwargs["outputs"]) is list
                    and len(kwargs["outputs"]) == 1
                    and kwargs["outputs"][0] is data.cfrc_int,
                    "original aliased RNE kernel, dimensions and arrays",
                )
                _need(
                    not device.is_capturing
                    and wp.get_device() == device
                    and _stream(device) == binding[-1]
                    and tuple(_fingerprint(a) for a in arrays) == binding[3],
                    "unchanged eager RNE storage/device/stream during dispatch",
                )
                result = _LAUNCH(*args, **kwargs)
                index += 1
                return result

            self._inside = True
            wp.launch = launch
            try:
                result = _ENTRY(model, data)
                _need(
                    result is None
                    and index == 7
                    and wp.launch is launch
                    and smooth._rne_cfrc_backward is self._wrapper,
                    "complete unchanged seven-launch RNE backward",
                )
                accumulated = _snapshot(data.cfrc_int, data.nworld)
                _sources()
            finally:
                if wp.launch is launch:
                    wp.launch = _LAUNCH
                else:
                    self._fault = self._fault or "ForeignLaunchReference"
                self._inside = False
            self._entries.append(initialized)
            self._outputs.append(accumulated)
            self._records.append(
                {
                    "call_index": len(self._entries) - 1,
                    "worlds": data.nworld,
                    "device": str(device),
                    "stream": binding[-1],
                    "bytes_per_snapshot": len(initialized),
                    "input_sha256": sha256(initialized).hexdigest(),
                    "output_sha256": sha256(accumulated).hexdigest(),
                    "input_boundary": "after-original-cfrc-before-first-backward",
                    "output_boundary": "after-original-backward-before-original-qfrc-bias",
                    "initializer_caller_code_bound": True,
                    "original_launches": index,
                    "body_rule": "body!=0-includes-body1-to-root0",
                    "input_output_alias": True,
                    "layout": {
                        "object_id": id(data.cfrc_int),
                        "ptr": int(data.cfrc_int.ptr),
                        "shape": list(data.cfrc_int.shape),
                        "strides": list(data.cfrc_int.strides),
                        "dtype": "spatial_vectorf",
                        "contiguous": True,
                    },
                    "smooth_sha256": source_checks.PINNED_SMOOTH_SHA256,
                    "forward_sha256": source_checks.PINNED_FORWARD_SHA256,
                    "readback_and_device_sync_perturb_timing": True,
                }
            )
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise

    @property
    def entries(self):
        return tuple(self._entries)

    @property
    def outputs(self):
        return tuple(self._outputs)

    @property
    def receipt(self):
        return {
            "protocol": PROTOCOL,
            "status": self._status,
            "fault": self._fault,
            "calls": deepcopy(self._records),
            "unchanged_postconstraint_passthrough_calls": self._post_calls,
            "flags": dict(FLAGS),
        }


_PINNED_METHODS = tuple(
    (name, getattr(RneEntryCapture, name))
    for name in ("__enter__", "__exit__", "_call")
)


def _methods(scope):
    _need(
        type(scope) is RneEntryCapture
        and all(
            getattr(getattr(scope, name), "__func__", None) is method
            for name, method in _PINNED_METHODS
        ),
        "exact unmodified RNE observer methods",
    )


def repeat_entry(model, data, raw, digest, mode):
    """Reset and reduce detached scratch 32 times; preserve every live force bit."""
    _need(
        type(mode) is str and mode in ("concurrent", "serial"),
        "literal RNE repeat mode",
    )
    _need(
        type(raw) is bytes and len(raw) in (768, 24576),
        "immutable full RNE input bytes",
    )
    _need(
        type(digest) is str and len(digest) == 64 and sha256(raw).hexdigest() == digest,
        "whole RNE input SHA-256 before decode",
    )
    values = np.frombuffer(raw, dtype="<f4").reshape(-1, 16, 6)
    _need(bool(np.isfinite(values).all()), "finite authenticated RNE repeat input")
    _need(
        current_thread() is main_thread()
        and smooth._rne_cfrc_backward is _ENTRY
        and wp.launch is _LAUNCH,
        "unmodified detached RNE repeat dispatch",
    )
    _sources()
    device = _layout(model, data)
    _need(
        values.shape == (data.nworld, 16, 6), "repeat input matches bound world count"
    )
    before = _snapshot(data.cfrc_int, data.nworld)
    live_binding = _fingerprint(data.cfrc_int)
    scratch = wp.empty((data.nworld, 16), dtype=wp.spatial_vector, device=device)
    _need(scratch.ptr != data.cfrc_int.ptr, "detached RNE scratch never live output")
    scratch_binding = _fingerprint(scratch)
    rows = tuple(wp.array([body], dtype=wp.int32, device=device) for body in (2, 7, 11))
    _need(
        tuple(
            source_checks._warp_int_ids(row, 1, device, "RNE singleton") for row in rows
        )
        == ((2,), (7,), (11,)),
        "exact RNE serial singleton IDs",
    )
    bank, launches = [], 0
    stream = _stream(device)
    levels = tuple(reversed(model.body_tree))
    plan_arrays = [model.body_parentid, *levels, *rows]
    plan_binding = tuple(_fingerprint(array) for array in plan_arrays)
    for _ in range(REPEATS):
        # Identical ID readbacks in both modes, before any reduction dispatch.
        _need(
            tuple(
                source_checks._warp_int_ids(row, row.size, device, "RNE repeat level")
                for row in levels
            )
            == source_checks.EXPECTED_REVERSED_LEVELS
            and tuple(
                source_checks._warp_int_ids(row, 1, device, "RNE singleton")
                for row in rows
            )
            == ((2,), (7,), (11,)),
            "unchanged RNE repeat IDs in both modes",
        )
        scratch.assign(values)
        for level_index, row in enumerate(levels):
            split = mode == "serial" and level_index == 4
            for current in rows if split else (row,):
                _need(
                    not device.is_capturing
                    and wp.get_device() == device
                    and _stream(device) == stream
                    and wp.launch is _LAUNCH
                    and _fingerprint(scratch) == scratch_binding
                    and _fingerprint(data.cfrc_int) == live_binding
                    and tuple(_fingerprint(array) for array in plan_arrays)
                    == plan_binding,
                    "unchanged detached RNE repeat storage/device/stream",
                )
                _LAUNCH(
                    _KERNEL,
                    dim=[data.nworld, current.size],
                    inputs=[model.body_parentid, scratch, current],
                    outputs=[scratch],
                )
                launches += 1
        bank.append(_snapshot(scratch, data.nworld))
    after = _snapshot(data.cfrc_int, data.nworld)
    _need(
        before == after and _fingerprint(data.cfrc_int) == live_binding,
        "all live RNE force bits unchanged by detached repeat",
    )
    _sources()
    _need(
        launches == (224 if mode == "concurrent" else 288),
        "complete bounded RNE repeat count",
    )
    output = b"".join(bank)
    return output, {
        "protocol": PROTOCOL + ":detached-repeat",
        "mode": mode,
        "repeats": REPEATS,
        "resets": REPEATS,
        "worlds": data.nworld,
        "bytes_per_snapshot": len(raw),
        "input_sha256": digest,
        "bank_sha256": sha256(output).hexdigest(),
        "accumulation_launches": launches,
        "body_rule": "body!=0-includes-body1-to-root0",
        "live_array_unchanged": True,
        "live_before_sha256": sha256(before).hexdigest(),
        "live_after_sha256": sha256(after).hexdigest(),
        "detached_input_role": "supplied-bytes-not-independent-stage-provenance",
        "readback_and_device_sync_perturb_timing": True,
        "flags": dict(FLAGS),
    }
