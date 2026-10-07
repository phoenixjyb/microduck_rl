"""Bounded serial sibling dispatch on the actual coupled RNE path.

This control preserves the initialized live ``Data.cfrc_int`` array and only
splits the original sibling level during bias RNE. The sensory postconstraint
caller remains a direct, unobserved pass-through. It is an instrumentation
control, not a determinism or training-admission claim.
"""

from copy import deepcopy
from hashlib import sha256
import inspect
from threading import Lock, current_thread, main_thread

import warp as wp
from mujoco_warp._src import forward, smooth

from mjlab_microduck import stance_rne_entry_capture as observer

PROTOCOL = "microduck-rne-coupled-serial-oct7-v1"
MAX_CALLS = 2
SPLIT_LEVEL_INDEX = 4
SPLIT_BODY_IDS = (2, 7, 11)
_LOCK = Lock()
_ENTRY = smooth._rne_cfrc_backward
_RNE = smooth.rne
_RNE_CODE = inspect.unwrap(_RNE).__code__
_POST = smooth.rne_postconstraint
_POST_CODE = inspect.unwrap(_POST).__code__
_KERNEL = smooth._cfrc_backward
_LAUNCH = wp.launch
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


def _sources():
    observer._sources()
    _need(
        smooth.rne is _RNE
        and smooth.rne_postconstraint is _POST
        and inspect.unwrap(_RNE).__code__ is _RNE_CODE
        and inspect.unwrap(_POST).__code__ is _POST_CODE
        and smooth._cfrc_backward is _KERNEL
        and forward.forward is _FORWARD
        and forward.fwd_velocity is _VELOCITY
        and all(getattr(smooth, name) is value for name, value in _PREFIX),
        "pinned RNE caller, kernel and forward aliases",
    )


def _methods(scope):
    _need(
        type(scope) is SerialRneCoupledControl
        and all(
            getattr(getattr(scope, name), "__func__", None) is method
            for name, method in _PINNED_METHODS
        ),
        "exact unmodified serial RNE control methods",
    )


class SerialRneCoupledControl:
    """Split only actual bias-RNE level 4 for constructor and eager forward."""

    def __init__(self):
        self._status, self._fault = "not-started", None
        self._entries, self._outputs, self._records = [], [], []
        self._binding, self._wrapper = None, None
        self._owns_lock, self._inside = False, False
        self._post_calls = 0
        self._singletons = None
        self._singleton_binding = None

    def __enter__(self):
        _need(
            self._status == "not-started" and current_thread() is main_thread(),
            "one-shot main-thread serial RNE scope",
        )
        _need(_LOCK.acquire(blocking=False), "exclusive serial RNE scope")
        self._owns_lock = True
        try:
            _methods(self)
            _sources()
            _need(
                smooth._rne_cfrc_backward is _ENTRY and wp.launch is _LAUNCH,
                "unmodified RNE backward and Warp launch",
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
                raise ValueError("complete exact serial RNE scope required")
        else:
            self._status = "complete"
        return False

    def _prepare_singletons(self, device):
        if self._singletons is None:
            rows = tuple(
                wp.array([body], dtype=wp.int32, device=device)
                for body in SPLIT_BODY_IDS
            )
            actual_ids = tuple(
                observer.source_checks._warp_int_ids(row, 1, device, "RNE singleton")
                for row in rows
            )
            _need(
                actual_ids == tuple((body,) for body in SPLIT_BODY_IDS),
                "exact RNE split body IDs",
            )
            self._singletons = rows
            self._singleton_binding = tuple(observer._fingerprint(row) for row in rows)
        _need(
            tuple(observer._fingerprint(row) for row in self._singletons)
            == self._singleton_binding
            and tuple(
                observer.source_checks._warp_int_ids(row, 1, device, "RNE singleton")
                for row in self._singletons
            )
            == tuple((body,) for body in SPLIT_BODY_IDS),
            "unchanged exact serial singleton arrays",
        )

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
                    "actual pinned RNE initializer caller and bound model/data",
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
                "bounded owned nonrecursive serial RNE call",
            )
            if is_post:
                device = model.body_parentid.device
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
                    tuple(observer._fingerprint(array) for array in arrays),
                    observer._stream(device),
                )
                _need(
                    self._post_calls < MAX_CALLS
                    and len(self._entries) == self._post_calls + 1,
                    "one untouched sensory passthrough after each bias call",
                )
                _need(
                    self._binding is not None
                    and binding == self._binding
                    and not device.is_capturing
                    and wp.get_device() == device,
                    "sensory passthrough remains on bound live RNE arrays/device/stream",
                )
                result = _ENTRY(model, data)
                _need(
                    result is None
                    and smooth._rne_cfrc_backward is self._wrapper
                    and wp.launch is _LAUNCH,
                    "unchanged sensory postconstraint helper",
                )
                _sources()
                self._post_calls += 1
                return result

            _sources()
            device = observer._layout(model, data)
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
                tuple(observer._fingerprint(array) for array in arrays),
                observer._stream(device),
            )
            if self._binding is None:
                self._binding = binding
            _need(binding == self._binding, "unchanged serial RNE arrays/device/stream")
            initialized = observer._snapshot(data.cfrc_int, data.nworld)
            self._prepare_singletons(device)
            levels = tuple(reversed(model.body_tree))
            logical_index = 0
            underlying_launches = 0
            underlying_groups = []
            level_ids = observer.source_checks.EXPECTED_REVERSED_LEVELS

            def launch(*args, **kwargs):
                nonlocal logical_index, underlying_launches
                _need(
                    current_thread() is main_thread()
                    and self._inside
                    and wp.launch is launch
                    and logical_index < 7
                    and len(args) == 1
                    and set(kwargs) == {"dim", "inputs", "outputs"},
                    "owned exact ordered logical RNE launch",
                )
                row = levels[logical_index]
                _need(
                    args[0] is _KERNEL
                    and type(kwargs["dim"]) is list
                    and kwargs["dim"] == [data.nworld, row.size]
                    and type(kwargs["inputs"]) is list
                    and len(kwargs["inputs"]) == 3
                    and all(
                        actual is expected
                        for actual, expected in zip(
                            kwargs["inputs"],
                            [model.body_parentid, data.cfrc_int, row],
                            strict=True,
                        )
                    )
                    and type(kwargs["outputs"]) is list
                    and len(kwargs["outputs"]) == 1
                    and kwargs["outputs"][0] is data.cfrc_int,
                    "original RNE kernel and actual aliased live force array",
                )
                _need(
                    not device.is_capturing
                    and wp.get_device() == device
                    and observer._stream(device) == binding[-1]
                    and tuple(observer._fingerprint(array) for array in arrays)
                    == binding[3],
                    "unchanged eager RNE storage/device/stream",
                )
                index = logical_index
                logical_index += 1
                if index == SPLIT_LEVEL_INDEX:
                    _need(
                        observer.source_checks._warp_int_ids(
                            row, row.size, device, "RNE split level"
                        )
                        == (SPLIT_BODY_IDS),
                        "exact ordered RNE sibling level",
                    )
                    for body, singleton in zip(
                        SPLIT_BODY_IDS, self._singletons, strict=True
                    ):
                        _need(
                            observer.source_checks._warp_int_ids(
                                singleton, 1, device, "RNE singleton"
                            )
                            == (body,),
                            "exact ordered RNE singleton body ID",
                        )
                        _LAUNCH(
                            _KERNEL,
                            dim=[data.nworld, 1],
                            inputs=[model.body_parentid, data.cfrc_int, singleton],
                            outputs=[data.cfrc_int],
                        )
                        underlying_launches += 1
                        underlying_groups.append([body])
                    return None
                _LAUNCH(*args, **kwargs)
                underlying_launches += 1
                underlying_groups.append(list(level_ids[index]))
                return None

            self._inside = True
            wp.launch = launch
            try:
                result = _ENTRY(model, data)
                _need(
                    result is None
                    and logical_index == 7
                    and underlying_launches == 9
                    and wp.launch is launch
                    and smooth._rne_cfrc_backward is self._wrapper,
                    "seven logical RNE requests and nine actual serial launches",
                )
                accumulated = observer._snapshot(data.cfrc_int, data.nworld)
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
                    "logical_launches": logical_index,
                    "underlying_launches": underlying_launches,
                    "split_level": SPLIT_LEVEL_INDEX,
                    "split_body_order": list(SPLIT_BODY_IDS),
                    "underlying_body_groups": underlying_groups,
                    "body_rule": "body!=0-includes-body1-to-root0",
                    "split_uses_live_initialized_alias": True,
                    "input_output_alias": True,
                    "layout": {
                        "object_id": id(data.cfrc_int),
                        "ptr": int(data.cfrc_int.ptr),
                        "shape": list(data.cfrc_int.shape),
                        "strides": list(data.cfrc_int.strides),
                        "dtype": "spatial_vectorf",
                        "contiguous": True,
                    },
                    "smooth_sha256": observer.source_checks.PINNED_SMOOTH_SHA256,
                    "forward_sha256": observer.source_checks.PINNED_FORWARD_SHA256,
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
            "dispatch": {
                "logical_launches_per_bias_call": 7,
                "underlying_launches_per_bias_call": 9,
                "split_level": SPLIT_LEVEL_INDEX,
                "split_body_order": list(SPLIT_BODY_IDS),
                "split_uses_live_initialized_alias": True,
                "sensory_passthrough_calls": self._post_calls,
                "closed": self._status == "complete",
            },
            "flags": dict(FLAGS),
        }


_PINNED_METHODS = tuple(
    (name, getattr(SerialRneCoupledControl, name))
    for name in ("__enter__", "__exit__", "_prepare_singletons", "_call")
)
