"""Bounded one-step serial reduction control on the live stance runtime.

This diagnostic owns a fresh three-entry dispatch scope for exactly one
constructor forward and one ten-substep eager step. It never changes the
historical controls, replaces a kernel, or qualifies a full window.
"""

from copy import deepcopy
from hashlib import sha256
import inspect
from threading import Lock, active_count, current_thread, main_thread

import numpy as np
import torch
import warp as wp
import mujoco_warp as public
from mujoco_warp._src import forward, smooth

from mjlab_microduck import stance_com_entry_capture as com_checks
from mjlab_microduck import stance_crb_runtime_control as crb_checks
from mjlab_microduck import stance_rne_entry_capture as rne_checks
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

PROTOCOL = "microduck-serial-one-step-control-oct7-v1"
WORLDS = (2, 64)
FORWARDS = 1 + 2 * 10
COM_SPLIT_INDEX = 5
CRB_SPLIT_INDEX = 4
RNE_SPLIT_INDEX = 4
SPLIT_BODY_IDS = (2, 7, 11)
_LOCK = Lock()

_CRB = smooth.crb
_COM = smooth.com_pos
_RNE_ENTRY = smooth._rne_cfrc_backward
_RNE = smooth.rne
_RNE_CODE = inspect.unwrap(_RNE).__code__
_POST = smooth.rne_postconstraint
_POST_CODE = inspect.unwrap(_POST).__code__
_CRB_KERNEL = smooth._crb_accumulate
_QM_KERNEL = smooth._qM_dense
_COM_KERNELS = com_checks._KERNELS
_COM_KERNEL_NAMES = (
    "_subtree_com_init",
    "_subtree_com_acc",
    "_subtree_div",
    "_cinert",
    "_cdof",
)
_RNE_KERNEL = smooth._cfrc_backward
_PINNED_LAUNCH = wp.launch
_LAUNCH = _PINNED_LAUNCH
_FWD_POSITION_CODE = inspect.unwrap(forward.fwd_position).__code__
_RUNTIME_STEP = WarpStanceRuntime.step
_RUNTIME_FORWARD = WarpStanceRuntime._forward
_RUNTIME_RESET = WarpStanceRuntime.reset
_RUNTIME_CODES = tuple(
    fn.__code__ for fn in (_RUNTIME_STEP, _RUNTIME_FORWARD, _RUNTIME_RESET)
)
_PARENTS = crb_checks.EXPECTED_PARENTS
_LEVELS = crb_checks.EXPECTED_DEPTH_LEVELS
_FALSE_FLAGS = {
    "original_run_entry_captured": False,
    "native_qualified": False,
    "full_window_qualified": False,
    "training_authorized": False,
    "physical_acceptance": False,
    "original_pair_accepted": False,
    "actual_kernel_order_observed": False,
    "runtime_cause_proven": False,
}
_FLAG_NAMES = tuple(_FALSE_FLAGS)
_RNE_PREFIX = tuple(rne_checks._PREFIX)
_ACCUMULATION_GROUPS = tuple(tuple(row) for row in crb_checks.EXPECTED_REVERSED_LEVELS)
_COM_GROUPS = _ACCUMULATION_GROUPS
_CRB_GROUPS = _COM_GROUPS
_RNE_GROUPS = _COM_GROUPS
_PINNED_REFS = (
    ("_CRB", _CRB),
    ("_COM", _COM),
    ("_RNE_ENTRY", _RNE_ENTRY),
    ("_RNE", _RNE),
    ("_POST", _POST),
    ("_CRB_KERNEL", _CRB_KERNEL),
    ("_QM_KERNEL", _QM_KERNEL),
    ("_COM_KERNELS", _COM_KERNELS),
    ("_RNE_KERNEL", _RNE_KERNEL),
    ("_LAUNCH", _PINNED_LAUNCH),
    ("_PINNED_LAUNCH", _PINNED_LAUNCH),
    ("_RUNTIME_STEP", _RUNTIME_STEP),
    ("_RUNTIME_FORWARD", _RUNTIME_FORWARD),
    ("_RUNTIME_RESET", _RUNTIME_RESET),
    ("_RUNTIME_CODES", _RUNTIME_CODES),
    ("_RNE_CODE", _RNE_CODE),
    ("_POST_CODE", _POST_CODE),
    ("_FWD_POSITION_CODE", _FWD_POSITION_CODE),
    ("_PARENTS", _PARENTS),
    ("_LEVELS", _LEVELS),
    ("_FLAG_NAMES", _FLAG_NAMES),
    ("PROTOCOL", PROTOCOL),
    ("FORWARDS", FORWARDS),
    ("COM_SPLIT_INDEX", COM_SPLIT_INDEX),
    ("CRB_SPLIT_INDEX", CRB_SPLIT_INDEX),
    ("RNE_SPLIT_INDEX", RNE_SPLIT_INDEX),
    ("SPLIT_BODY_IDS", SPLIT_BODY_IDS),
    ("_COM_KERNEL_NAMES", _COM_KERNEL_NAMES),
    ("_RNE_PREFIX", _RNE_PREFIX),
    ("_ACCUMULATION_GROUPS", _ACCUMULATION_GROUPS),
)
_PINNED_REF_TABLE = _PINNED_REFS
_ARRAY_NAMES = (
    "body_parentid",
    "body_tree_0",
    "body_tree_1",
    "body_tree_2",
    "body_tree_3",
    "body_tree_4",
    "body_tree_5",
    "body_tree_6",
    "body_mass",
    "body_subtreemass",
    "body_rootid",
    "body_inertia",
    "dof_bodyid",
    "dof_parentid",
    "dof_armature",
    "jnt_type",
    "jnt_dofadr",
    "jnt_bodyid",
    "crb",
    "cinert",
    "qM",
    "cdof",
    "subtree_com",
    "xipos",
    "ximat",
    "xmat",
    "xanchor",
    "xaxis",
    "cfrc_int",
    "cacc",
    "cvel",
)


def _serial_groups(groups, split_index):
    result = []
    for index, group in enumerate(groups):
        result.extend(
            [[body] for body in SPLIT_BODY_IDS]
            if index == split_index
            else [list(group)]
        )
    return result


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _source_checks(scope=None):
    crb_checks._check_caller_refs()
    _need(
        _PINNED_REFS is _PINNED_REF_TABLE
        and all(globals().get(name) is value for name, value in _PINNED_REFS)
        and tuple(_FALSE_FLAGS) == _FLAG_NAMES
        and all(value is False for value in _FALSE_FLAGS.values())
        and (
            wp.launch is _PINNED_LAUNCH
            or (
                scope is not None
                and scope._inside
                and wp.launch is scope._launch_wrapper
            )
        ),
        "immutable pinned callables and owned per-entry launch wrapper",
    )
    _need(
        crb_checks._source_digest(smooth.__file__) == crb_checks.PINNED_SMOOTH_SHA256
        and crb_checks._source_digest(forward.__file__)
        == crb_checks.PINNED_FORWARD_SHA256,
        "exact pinned smooth and forward source bytes",
    )
    entries = (
        (smooth.crb, _CRB, None if scope is None else scope._wrappers.get("crb")),
        (smooth.com_pos, _COM, None if scope is None else scope._wrappers.get("com")),
        (
            smooth._rne_cfrc_backward,
            _RNE_ENTRY,
            None if scope is None else scope._wrappers.get("rne"),
        ),
    )
    _need(
        all(
            actual is original or (wrapper is not None and actual is wrapper)
            for actual, original, wrapper in entries
        ),
        "pinned original or owned active reduction entries",
    )
    _need(
        forward.smooth is smooth
        and smooth.wp is wp
        and com_checks._ENTRY is _COM
        and com_checks._PUBLIC_ENTRY is public.com_pos
        and public.com_pos is _COM
        and public.crb is _CRB
        and smooth.rne is _RNE
        and public.rne is _RNE
        and smooth.rne_postconstraint is _POST
        and public.rne_postconstraint is _POST
        and inspect.unwrap(_RNE).__code__ is _RNE_CODE
        and inspect.unwrap(_POST).__code__ is _POST_CODE
        and inspect.unwrap(forward.fwd_position).__code__ is _FWD_POSITION_CODE
        and smooth._crb_accumulate is _CRB_KERNEL
        and smooth._qM_dense is _QM_KERNEL
        and smooth._cfrc_backward is _RNE_KERNEL
        and tuple(getattr(smooth, name) for name in _COM_KERNEL_NAMES) == _COM_KERNELS
        and tuple((name, getattr(smooth, name)) for name, _ in _RNE_PREFIX)
        == _RNE_PREFIX
        and WarpStanceRuntime.step is _RUNTIME_STEP
        and WarpStanceRuntime._forward is _RUNTIME_FORWARD
        and WarpStanceRuntime.reset is _RUNTIME_RESET
        and tuple(
            fn.__code__ for fn in (_RUNTIME_STEP, _RUNTIME_FORWARD, _RUNTIME_RESET)
        )
        == _RUNTIME_CODES,
        "pinned forward, reduction kernels, and runtime step",
    )


def _methods(scope):
    _need(
        type(scope) is SerialStepControl
        and all(
            getattr(getattr(scope, name), "__func__", None) is method
            for name, method in _PINNED_METHODS
        ),
        "exact owned serial-step control methods",
    )


def _fingerprint(array):
    return (
        id(array),
        int(array.ptr),
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.strides),
        array.dtype,
        array.device,
        array.is_contiguous,
    )


def _host_f32(array, shape, label):
    values = array.numpy()
    _need(
        isinstance(values, np.ndarray)
        and values.shape == shape
        and values.dtype == np.dtype(np.float32)
        and np.isfinite(values).all(),
        f"finite complete {label} readback",
    )
    return values.astype("<f4", copy=False).tobytes(order="C")


def _static_rows(array, worlds, tail_shape, dtype, label):
    _need(
        array.dtype is dtype
        and array.shape in ((1, *tail_shape), (worlds, *tail_shape)),
        f"{label} exact static model shape and dtype",
    )
    if array.shape[0] == 1 or array.is_contiguous:
        _need(array.is_contiguous, f"{label} ordinary contiguous rows")
    else:
        _need(
            array.shape[0] == worlds
            and array.strides[0] == 0
            and array.strides[-1] > 0
            and all(
                array.strides[index]
                == array.strides[index + 1] * array.shape[index + 1]
                for index in range(1, array.ndim - 1)
            ),
            f"{label} exact leading-axis static broadcast layout",
        )


class SerialStepControl:
    """Split only the declared live sibling group during one actual step."""

    def __init__(self):
        self._status = "not-started"
        self._fault = None
        self._owns_lock = False
        self._used = False
        self._inside = False
        self._active_kind = None
        self._active_stage = 0
        self._binding = None
        self._model = self._data = self._device = None
        self._arrays = None
        self._wrappers = {}
        self._launch_wrapper = self._launch
        self._runtime = None
        self._step_used = False
        self._completed_forwards = 0
        self._event_state = "com"
        self._calls = []
        self._actual_groups = []
        self._dispatch_trace = []
        self._com_initialized = []
        self._com_weighted = []
        self._rne_inputs = []
        self._rne_outputs = []
        self._split_rows = None
        self._split_binding = None
        self._entry_counts = {"com": 0, "crb": 0, "rne_bias": 0, "rne_post": 0}

    def _guard(self):
        _methods(self)
        _need(
            self._status == "active"
            and self._owns_lock
            and self._fault is None
            and current_thread() is main_thread()
            and active_count() == 1,
            "active isolated main-thread one-shot scope",
        )
        _source_checks(self)

    def __enter__(self):
        if type(self) is not SerialStepControl or self._used:
            raise ValueError("one-shot isolated main-thread serial-step scope")
        if current_thread() is not main_thread() or active_count() != 1:
            self._status, self._fault = "faulted", "ThreadBoundaryViolation"
            raise ValueError("one-shot isolated main-thread serial-step scope")
        self._used = True
        if not _LOCK.acquire(blocking=False):
            self._status, self._fault = "faulted", "ProcessScopeBusy"
            raise ValueError("exclusive serial-step process lock")
        self._owns_lock = True
        try:
            _methods(self)
            _source_checks()
            _need(
                smooth.crb is _CRB
                and smooth.com_pos is _COM
                and smooth._rne_cfrc_backward is _RNE_ENTRY
                and wp.launch is _LAUNCH
                and public.crb is _CRB
                and public.com_pos is _COM
                and public.rne is _RNE
                and public.rne_postconstraint is _POST,
                "unmodified original smooth/public/launch aliases",
            )
            self._wrappers = {
                "crb": self._call_crb,
                "com": self._call_com,
                "rne": self._call_rne,
            }
            smooth.crb = self._wrappers["crb"]
            smooth.com_pos = self._wrappers["com"]
            smooth._rne_cfrc_backward = self._wrappers["rne"]
            self._status = "active"
            return self
        except BaseException as error:
            self._fault = type(error).__name__
            self._status = "faulted"
            self._restore_owned()
            self._release()
            raise

    def _restore_owned(self):
        if smooth.crb is self._wrappers.get("crb"):
            smooth.crb = _CRB
        if smooth.com_pos is self._wrappers.get("com"):
            smooth.com_pos = _COM
        if smooth._rne_cfrc_backward is self._wrappers.get("rne"):
            smooth._rne_cfrc_backward = _RNE_ENTRY

    def _release(self):
        if self._owns_lock:
            self._owns_lock = False
            _LOCK.release()

    def __exit__(self, error_type, error, _traceback):
        foreign = (
            smooth.crb is not self._wrappers.get("crb")
            or smooth.com_pos is not self._wrappers.get("com")
            or smooth._rne_cfrc_backward is not self._wrappers.get("rne")
            or wp.launch is not _LAUNCH
        )
        if error is not None:
            self._fault = self._fault or error_type.__name__
        if foreign:
            self._fault = self._fault or "ForeignHookOrLaunch"
        self._restore_owned()
        self._release()
        try:
            _source_checks()
        except BaseException as source_error:
            self._fault = self._fault or type(source_error).__name__
        complete = (
            error is None
            and not foreign
            and self._fault is None
            and self._runtime is not None
            and self._step_used
            and self._completed_forwards == FORWARDS
            and self._event_state == "com"
            and self._entry_counts
            == {
                "com": FORWARDS,
                "crb": FORWARDS,
                "rne_bias": FORWARDS,
                "rne_post": FORWARDS,
            }
            and len(self._com_initialized) == FORWARDS
            and len(self._com_weighted) == FORWARDS
            and len(self._rne_inputs) == FORWARDS
            and len(self._rne_outputs) == FORWARDS
            and len(self._calls) == FORWARDS
        )
        if complete:
            self._status = "complete"
        else:
            self._status = "faulted"
            self._fault = self._fault or "IncompleteOrForeignScope"
            if error is None:
                raise ValueError(
                    "complete exact one-constructor/one-step scope required"
                )
        return False

    def _validate_model_data(self, model, data):
        _need(
            type(model.nbody) is int
            and (model.nbody, model.nq, model.nv, model.nu) == (16, 21, 20, 14)
            and model.is_sparse is False,
            "fixed dense MicroDuck model dimensions",
        )
        device = crb_checks._validate_bound_model(model, data)
        _need(
            type(data.nworld) is int
            and (
                (data.nworld == 2 and str(device) == "cpu")
                or (data.nworld == 64 and str(device) == "cuda:0")
            ),
            "two CPU worlds or 64 CUDA0 worlds only",
        )
        for name in ("subtree_com", "xipos"):
            array = getattr(data, name)
            _need(
                array.dtype is wp.vec3
                and array.shape == (data.nworld, 16)
                and array.device == device
                and array.is_contiguous is True,
                "exact contiguous live CoM vec3 arrays",
            )
        _need(
            data.xipos is not data.subtree_com
            and data.xipos.ptr != data.subtree_com.ptr,
            "distinct live CoM position and accumulation arrays",
        )
        _static_rows(model.body_mass, data.nworld, (16,), wp.float32, "body_mass")
        _static_rows(
            model.body_subtreemass, data.nworld, (16,), wp.float32, "body_subtreemass"
        )
        for name in ("cfrc_int", "cacc", "cvel"):
            array = getattr(data, name)
            _need(
                array.dtype is wp.spatial_vector
                and array.shape == (data.nworld, 16)
                and array.device == device
                and array.is_contiguous is True,
                "exact contiguous live RNE spatial arrays",
            )
        _need(
            len({int(getattr(data, name).ptr) for name in ("cfrc_int", "cacc", "cvel")})
            == 3,
            "distinct live RNE force, acceleration and velocity arrays",
        )
        _static_rows(
            model.dof_armature,
            data.nworld,
            (model.nv,),
            wp.float32,
            "dof_armature",
        )
        _need(
            model.body_inertia.device == device
            and model.body_inertia.dtype is wp.vec3
            and model.body_inertia.shape in ((1, 16), (data.nworld, 16)),
            "exact static body-inertia row shape/device",
        )
        if model.body_inertia.shape[0] == 1:
            _need(model.body_inertia.is_contiguous, "body inertia contiguous singleton")
        elif model.body_inertia.is_contiguous is not True:
            _need(
                model.body_inertia.strides[0] == 0
                and model.body_inertia.strides[1] > 0,
                "body inertia exact leading-axis broadcast layout",
            )
        arrays = (
            model.body_parentid,
            *model.body_tree,
            model.body_mass,
            model.body_subtreemass,
            model.body_rootid,
            model.body_inertia,
            model.dof_bodyid,
            model.dof_parentid,
            model.dof_armature,
            model.jnt_type,
            model.jnt_dofadr,
            model.jnt_bodyid,
            data.crb,
            data.cinert,
            data.qM,
            data.cdof,
            data.subtree_com,
            data.xipos,
            data.ximat,
            data.xmat,
            data.xanchor,
            data.xaxis,
            data.cfrc_int,
            data.cacc,
            data.cvel,
        )
        static = {
            id(model.body_mass),
            id(model.body_subtreemass),
            id(model.dof_armature),
            id(model.body_inertia),
        }
        _need(
            all(
                array.device == device and (id(array) in static or array.is_contiguous)
                for array in arrays
            ),
            "all live arrays contiguous except exact static broadcast models",
        )
        fingerprint = (
            id(model),
            id(data),
            data.nworld,
            tuple(_fingerprint(array) for array in arrays),
            crb_checks._array_fingerprint(model.body_parentid),
            com_checks._stream(device),
        )
        if self._binding is None:
            self._model, self._data, self._device = model, data, device
            self._arrays = arrays
            self._binding = fingerprint
        _need(
            model is self._model
            and data is self._data
            and fingerprint == self._binding
            and not device.is_capturing
            and wp.get_device() == device,
            "unchanged live model/data arrays, layout, device and stream",
        )
        return device

    def _layout_receipt(self):
        return [
            {
                "name": name,
                "object_id": int(fingerprint[0]),
                "ptr": int(fingerprint[1]),
                "shape": list(fingerprint[2]),
                "strides": list(fingerprint[3]),
                "dtype": str(fingerprint[4]),
                "device": str(fingerprint[5]),
                "contiguous": bool(fingerprint[6]),
            }
            for name, fingerprint in zip(_ARRAY_NAMES, self._binding[3], strict=True)
        ]

    def _make_cycle(self, kind, caller):
        self._guard()
        _need(not self._inside, "non-nested reduction entries")
        _need(wp.launch is _LAUNCH, "original Warp launcher before each entry")
        _need(
            self._completed_forwards < FORWARDS,
            "hard 21-forward bound before any readback or append",
        )
        device = self._validate_model_data(*caller)
        if kind == "com":
            _need(self._event_state == "com", "one CoM call per forward cycle")
        elif kind == "crb":
            _need(self._event_state == "crb", "CRB follows original CoM")
        elif kind == "rne_bias":
            _need(self._event_state == "rne_bias", "bias RNE follows original CRB")
        else:
            _need(self._event_state == "rne_post", "sensory RNE follows bias RNE")
        self._inside = True
        self._active_kind = kind
        self._active_stage = 0
        return device

    def _call_com(self, model, data):
        try:
            frame = inspect.currentframe()
            try:
                _need(
                    frame.f_back.f_code is _FWD_POSITION_CODE,
                    "actual pinned position-phase CoM caller",
                )
            finally:
                del frame
            self._make_cycle("com", (model, data))
            index = self._completed_forwards
            _need(len(self._calls) < FORWARDS, "bounded per-forward receipts")
            self._calls.append(
                {
                    "forward_index": index,
                    "phase": "constructor"
                    if index == 0
                    else "step-pre"
                    if index % 2 == 1
                    else "step-post",
                    "model_id": id(model),
                    "data_id": id(data),
                    "device": str(self._device),
                    "stream": self._binding[-1],
                    "smooth_sha256": crb_checks.PINNED_SMOOTH_SHA256,
                    "forward_sha256": crb_checks.PINNED_FORWARD_SHA256,
                    "array_layouts": self._layout_receipt(),
                }
            )
            self._actual_groups.append({"com": [], "crb": [], "rne_bias": []})
            self._active_kind = "com"
            self._active_stage = 0
            wp.launch = self._launch_wrapper
            result = _COM(model, data)
            _need(
                result is None
                and self._active_stage == 11
                and wp.launch is self._launch_wrapper,
                "exact original 11-call CoM body with three singleton siblings",
            )
            _need(
                smooth.com_pos is self._wrappers["com"]
                and tuple(_fingerprint(a) for a in self._arrays) == self._binding[3],
                "owned CoM hook and live arrays preserved",
            )
            _source_checks(self)
            self._event_state = "crb"
            self._entry_counts["com"] += 1
            self._calls[index]["com"] = {
                "logical_launches": 11,
                "underlying_launches": 13,
                "split_index": COM_SPLIT_INDEX,
                "split_body_order": list(SPLIT_BODY_IDS),
                "underlying_body_groups": _serial_groups(
                    _COM_GROUPS, COM_SPLIT_INDEX - 1
                ),
                "observed_body_groups": deepcopy(self._actual_groups[index]["com"]),
                "initialized_sha256": sha256(self._com_initialized[index]).hexdigest(),
                "weighted_sha256": sha256(self._com_weighted[index]).hexdigest(),
            }
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise
        finally:
            if wp.launch is self._launch_wrapper:
                wp.launch = _LAUNCH
            elif wp.launch is not _LAUNCH:
                self._fault = self._fault or "ForeignWarpLaunch"
            self._inside = False
            self._active_kind = None

    def _call_crb(self, model, data):
        try:
            frame = inspect.currentframe()
            try:
                _need(
                    frame.f_back.f_code is _FWD_POSITION_CODE,
                    "actual pinned position-phase CRB caller",
                )
            finally:
                del frame
            self._guard()
            _need(
                self._event_state == "crb"
                and not self._inside
                and current_thread() is main_thread()
                and smooth.crb is self._wrappers["crb"]
                and wp.launch is _LAUNCH,
                "owned nonrecursive actual CRB call after CoM",
            )
            self._validate_model_data(model, data)
            index = self._completed_forwards
            self._inside, self._active_kind, self._active_stage = True, "crb", 0
            wp.launch = self._launch_wrapper
            result = _CRB(model, data)
            _need(
                result is None
                and self._active_stage == 8
                and wp.launch is self._launch_wrapper
                and smooth.crb is self._wrappers["crb"],
                "exact seven logical CRB levels plus dense qM",
            )
            _source_checks(self)
            _need(
                tuple(_fingerprint(a) for a in self._arrays) == self._binding[3],
                "CRB call preserves all bound live array identities/layouts",
            )
            self._event_state = "rne_bias"
            self._entry_counts["crb"] += 1
            self._calls[index]["crb"] = {
                "logical_accumulation_launches": 7,
                "underlying_accumulation_launches": 9,
                "qM_launches": 1,
                "split_index": CRB_SPLIT_INDEX,
                "split_body_order": list(SPLIT_BODY_IDS),
                "underlying_body_groups": _serial_groups(_CRB_GROUPS, CRB_SPLIT_INDEX),
                "observed_body_groups": deepcopy(self._actual_groups[index]["crb"]),
            }
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise
        finally:
            if wp.launch is self._launch_wrapper:
                wp.launch = _LAUNCH
            elif wp.launch is not _LAUNCH:
                self._fault = self._fault or "ForeignWarpLaunch"
            self._inside = False
            self._active_kind = None

    def _call_rne(self, model, data):
        try:
            _methods(self)
            frame = inspect.currentframe()
            try:
                caller = frame.f_back
                caller_code = caller.f_code
                if caller_code is _POST_CODE:
                    kind = "rne_post"
                    _need(
                        caller.f_locals.get("m") is model
                        and caller.f_locals.get("d") is data,
                        "sensory RNE exact bound caller arrays",
                    )
                else:
                    kind = "rne_bias"
                    _need(
                        caller_code is _RNE_CODE
                        and caller.f_locals.get("m") is model
                        and caller.f_locals.get("d") is data
                        and caller.f_locals.get("flg_acc") is False,
                        "actual pinned bias-RNE caller with flg_acc false",
                    )
            finally:
                del frame, caller
            self._make_cycle(kind, (model, data))
            index = self._completed_forwards
            if kind == "rne_post":
                result = _RNE_ENTRY(model, data)
                _need(
                    result is None
                    and smooth._rne_cfrc_backward is self._wrappers["rne"]
                    and wp.launch is _LAUNCH,
                    "untouched sensory RNE passthrough",
                )
                _source_checks(self)
                self._event_state = "com"
                self._completed_forwards += 1
                self._entry_counts["rne_post"] += 1
                self._calls[index]["rne_sensory"] = "untouched-original-passthrough"
                return result

            _need(
                kind == "rne_bias"
                and smooth._rne_cfrc_backward is self._wrappers["rne"]
                and wp.launch is _LAUNCH
                and len(self._rne_inputs) < FORWARDS,
                "bounded owned bias-RNE dispatch",
            )
            entry = rne_checks._snapshot(data.cfrc_int, data.nworld)
            self._active_kind, self._active_stage, self._inside = "rne", 0, True
            wp.launch = self._launch_wrapper
            try:
                result = _RNE_ENTRY(model, data)
                _need(
                    result is None
                    and self._active_stage == 7
                    and wp.launch is self._launch_wrapper,
                    "seven logical RNE launches and nine underlying launches",
                )
                output = rne_checks._snapshot(data.cfrc_int, data.nworld)
                _need(
                    smooth._rne_cfrc_backward is self._wrappers["rne"]
                    and tuple(_fingerprint(a) for a in self._arrays)
                    == self._binding[3],
                    "actual live aliased RNE input/output remains bound",
                )
                _source_checks(self)
                self._rne_inputs.append(entry)
                self._rne_outputs.append(output)
            finally:
                if wp.launch is self._launch_wrapper:
                    wp.launch = _LAUNCH
                elif wp.launch is not _LAUNCH:
                    self._fault = self._fault or "ForeignWarpLaunch"
                self._inside = False
                self._active_kind = None
            self._event_state = "rne_post"
            self._entry_counts["rne_bias"] += 1
            self._calls[index]["rne_bias"] = {
                "logical_launches": 7,
                "underlying_launches": 9,
                "split_index": RNE_SPLIT_INDEX,
                "split_body_order": list(SPLIT_BODY_IDS),
                "underlying_body_groups": _serial_groups(_RNE_GROUPS, RNE_SPLIT_INDEX),
                "observed_body_groups": deepcopy(
                    self._actual_groups[index]["rne_bias"]
                ),
                "input_sha256": sha256(entry).hexdigest(),
                "output_sha256": sha256(output).hexdigest(),
                "input_boundary": "after-original-cfrc-before-first-backward",
                "output_boundary": "after-original-backward-before-original-qfrc-bias",
                "input_output_alias": True,
                "bytes_per_snapshot": len(entry),
                "snapshot_layout": {
                    "object_id": id(data.cfrc_int),
                    "ptr": int(data.cfrc_int.ptr),
                    "shape": list(data.cfrc_int.shape),
                    "strides": list(data.cfrc_int.strides),
                    "dtype": "spatial_vectorf",
                    "device": str(self._device),
                    "stream": self._binding[-1],
                    "contiguous": True,
                },
                "readback_and_device_sync_perturbs_timing": True,
            }
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise
        finally:
            if self._active_kind == "rne_post":
                self._inside = False
                self._active_kind = None

    def _launch(self, *args, **kwargs):
        self._guard()
        _need(
            self._inside
            and self._active_kind in ("com", "crb", "rne")
            and wp.launch is self._launch_wrapper
            and len(args) == 1
            and type(kwargs) is dict
            and set(kwargs) == {"dim", "inputs", "outputs"},
            "owned exact launch within one original reduction entry",
        )
        device = self._device
        _need(
            not device.is_capturing
            and wp.get_device() == device
            and com_checks._stream(device) == self._binding[-1]
            and tuple(_fingerprint(a) for a in self._arrays) == self._binding[3],
            "live arrays/device/stream unchanged at launch boundary",
        )
        index = self._active_stage
        if self._active_kind == "com":
            self._launch_com(index, args[0], kwargs)
        elif self._active_kind == "crb":
            self._launch_crb(index, args[0], kwargs)
        else:
            self._launch_rne(index, args[0], kwargs)

    def _launch_com(self, index, kernel, kwargs):
        _need(index < 11, "exact original CoM logical stages")
        rows = tuple(reversed(self._model.body_tree))
        stages = [
            (
                com_checks._KERNELS[0],
                (self._data.nworld, 16),
                [self._model.body_mass, self._data.xipos],
                [self._data.subtree_com],
            )
        ]
        stages.extend(
            (
                com_checks._KERNELS[1],
                (self._data.nworld, row.size),
                [self._model.body_parentid, self._data.subtree_com, row],
                [self._data.subtree_com],
            )
            for row in rows
        )
        stages.extend(
            [
                (
                    com_checks._KERNELS[2],
                    (self._data.nworld, 16),
                    [self._model.body_subtreemass, self._data.subtree_com],
                    [self._data.subtree_com],
                ),
                (
                    com_checks._KERNELS[3],
                    (self._data.nworld, 16),
                    [
                        self._model.body_rootid,
                        self._model.body_mass,
                        self._model.body_inertia,
                        self._data.xipos,
                        self._data.ximat,
                        self._data.subtree_com,
                    ],
                    [self._data.cinert],
                ),
                (
                    com_checks._KERNELS[4],
                    (self._data.nworld, self._model.njnt),
                    [
                        self._model.body_rootid,
                        self._model.jnt_type,
                        self._model.jnt_dofadr,
                        self._model.jnt_bodyid,
                        self._data.xmat,
                        self._data.xanchor,
                        self._data.xaxis,
                        self._data.subtree_com,
                    ],
                    [self._data.cdof],
                ),
            ]
        )
        expected_kernel, expected_dim, expected_inputs, expected_outputs = stages[index]
        _need(
            kernel is expected_kernel
            and type(kwargs["dim"]) is tuple
            and kwargs["dim"] == expected_dim
            and type(kwargs["inputs"]) is list
            and len(kwargs["inputs"]) == len(expected_inputs)
            and all(
                a is b for a, b in zip(kwargs["inputs"], expected_inputs, strict=True)
            )
            and type(kwargs["outputs"]) is list
            and len(kwargs["outputs"]) == len(expected_outputs)
            and all(
                a is b for a, b in zip(kwargs["outputs"], expected_outputs, strict=True)
            ),
            "exact original CoM kernel/dimensions/arrays",
        )
        if index == 8:
            _need(len(self._com_weighted) < FORWARDS, "bounded weighted CoM snapshots")
            wp.synchronize_device(self._device)
            self._com_weighted.append(
                _host_f32(
                    self._data.subtree_com,
                    (self._data.nworld, 16, 3),
                    "pre-division accumulated weighted CoM",
                )
            )
        if index == 5:
            _need(
                crb_checks._warp_int_ids(
                    rows[index - 1], 3, self._device, "CoM sibling level"
                )
                == SPLIT_BODY_IDS,
                "exact ordered live CoM sibling group",
            )
            self._split(index, kernel, kwargs, "com")
        else:
            if 1 <= index <= 7:
                self._trace_group("com", kwargs["inputs"][2])
            _LAUNCH(kernel, **kwargs)
        self._active_stage += 1
        if index == 0:
            _need(len(self._com_initialized) < FORWARDS, "bounded CoM init snapshots")
            wp.synchronize_device(self._device)
            self._com_initialized.append(
                _host_f32(
                    self._data.subtree_com,
                    (self._data.nworld, 16, 3),
                    "initialized weighted CoM",
                )
            )

    def _launch_crb(self, index, kernel, kwargs):
        _need(index < 8, "exact original seven CRB levels and qM")
        if index < 7:
            row = tuple(reversed(self._model.body_tree))[index]
            _need(
                kernel is _CRB_KERNEL
                and type(kwargs["dim"]) is tuple
                and kwargs["dim"] == (self._data.nworld, row.shape[0])
                and type(kwargs["inputs"]) is list
                and len(kwargs["inputs"]) == 3
                and kwargs["inputs"][0] is self._model.body_parentid
                and kwargs["inputs"][1] is self._data.crb
                and kwargs["inputs"][2] is row
                and type(kwargs["outputs"]) is list
                and len(kwargs["outputs"]) == 1
                and kwargs["outputs"][0] is self._data.crb,
                "exact original CRB logical level and aliased output",
            )
            if index == CRB_SPLIT_INDEX:
                _need(
                    crb_checks._warp_int_ids(row, 3, self._device, "CRB sibling level")
                    == SPLIT_BODY_IDS,
                    "exact ordered live CRB sibling group",
                )
                self._split(index, kernel, kwargs, "crb")
            else:
                self._trace_group("crb", kwargs["inputs"][2])
                _LAUNCH(kernel, **kwargs)
        else:
            _need(
                kernel is _QM_KERNEL
                and type(kwargs["dim"]) is tuple
                and kwargs["dim"] == (self._data.nworld, self._model.nv)
                and type(kwargs["inputs"]) is list
                and len(kwargs["inputs"]) == 5
                and all(
                    a is b
                    for a, b in zip(
                        kwargs["inputs"],
                        [
                            self._model.dof_bodyid,
                            self._model.dof_parentid,
                            self._model.dof_armature,
                            self._data.cdof,
                            self._data.crb,
                        ],
                        strict=True,
                    )
                )
                and type(kwargs["outputs"]) is list
                and len(kwargs["outputs"]) == 1
                and kwargs["outputs"][0] is self._data.qM,
                "exact original dense qM launch",
            )
            _LAUNCH(kernel, **kwargs)
        self._active_stage += 1

    def _launch_rne(self, index, kernel, kwargs):
        _need(index < 7, "exact original seven logical RNE stages")
        row = tuple(reversed(self._model.body_tree))[index]
        _need(
            kernel is _RNE_KERNEL
            and type(kwargs["dim"]) is list
            and kwargs["dim"] == [self._data.nworld, row.size]
            and type(kwargs["inputs"]) is list
            and len(kwargs["inputs"]) == 3
            and all(
                a is b
                for a, b in zip(
                    kwargs["inputs"],
                    [self._model.body_parentid, self._data.cfrc_int, row],
                    strict=True,
                )
            )
            and type(kwargs["outputs"]) is list
            and len(kwargs["outputs"]) == 1
            and kwargs["outputs"][0] is self._data.cfrc_int,
            "exact original RNE kernel and live aliased force array",
        )
        if index == RNE_SPLIT_INDEX:
            _need(
                crb_checks._warp_int_ids(row, 3, self._device, "RNE sibling level")
                == SPLIT_BODY_IDS,
                "exact ordered live RNE sibling group",
            )
            self._split(index, kernel, kwargs, "rne")
        else:
            self._trace_group("rne_bias", kwargs["inputs"][2])
            _LAUNCH(kernel, **kwargs)
        self._active_stage += 1

    def _split(self, index, kernel, kwargs, kind):
        if self._split_rows is None:
            self._split_rows = tuple(
                wp.array([body], dtype=wp.int32, device=self._device)
                for body in SPLIT_BODY_IDS
            )
            self._split_binding = tuple(_fingerprint(row) for row in self._split_rows)
        _need(
            tuple(_fingerprint(row) for row in self._split_rows) == self._split_binding
            and tuple(
                crb_checks._warp_int_ids(row, 1, self._device, "serial singleton")
                for row in self._split_rows
            )
            == tuple((body,) for body in SPLIT_BODY_IDS),
            "unchanged exact singleton arrays in body order",
        )
        _need(
            kwargs["inputs"][1] is kwargs["outputs"][0]
            and kwargs["outputs"][0]
            is {
                "com": self._data.subtree_com,
                "crb": self._data.crb,
                "rne": self._data.cfrc_int,
            }[kind],
            "serial split accumulates in the original live alias",
        )
        for row in self._split_rows:
            if kind == "com":
                inputs = [*kwargs["inputs"][:2], row]
            else:
                inputs = [self._model.body_parentid, kwargs["inputs"][1], row]
            dim = (self._data.nworld, 1) if kind != "rne" else [self._data.nworld, 1]
            self._trace_group("rne_bias" if kind == "rne" else kind, row)
            _LAUNCH(kernel, dim=dim, inputs=inputs, outputs=kwargs["outputs"])

    def _trace_group(self, kind, row):
        _need(
            kind in ("com", "crb", "rne_bias")
            and len(self._dispatch_trace) < FORWARDS * 27,
            "bounded observed accumulation body-group trace",
        )
        ids = crb_checks._warp_int_ids(
            row, int(row.shape[0]), self._device, "observed accumulation group"
        )
        self._actual_groups[self._completed_forwards][kind].append(list(ids))
        self._dispatch_trace.append(
            {
                "forward_index": self._completed_forwards,
                "entry": kind,
                "body_ids": list(ids),
            }
        )

    def bind_runtime(self, env):
        self._guard()
        _need(
            self._runtime is None
            and not self._step_used
            and type(env) is WarpStanceRuntime
            and env.forward_graph is None
            and env.faulted is False
            and env.model is self._model
            and env.data is self._data
            and env.n == self._data.nworld
            and env.device == torch.device(str(self._device))
            and env.wp_device == self._device
            and getattr(env._forward, "__func__", None) is _RUNTIME_FORWARD
            and getattr(env.step, "__func__", None) is _RUNTIME_STEP
            and not {"step", "_forward", "reset"} & set(env.__dict__)
            and self._completed_forwards == 1
            and self._event_state == "com",
            "exact eager runtime after one constructor forward",
        )
        self._runtime = env
        return self

    def step(self, env=None, actions=None, *, capture_control=True):
        self._guard()
        if env is None:
            env = self._runtime
        _need(
            not self._step_used
            and env is self._runtime
            and type(capture_control) is bool
            and capture_control is True
            and actions is not None
            and torch.is_tensor(actions)
            and tuple(actions.shape) == (env.n, 10)
            and actions.dtype is torch.float32
            and actions.device == env.device
            and not actions.any()
            and env.forward_graph is None
            and not env.faulted
            and getattr(env._forward, "__func__", None) is _RUNTIME_FORWARD
            and getattr(env.step, "__func__", None) is _RUNTIME_STEP
            and not {"step", "_forward", "reset"} & set(env.__dict__)
            and bool(env.live.all())
            and not bool(env.steps.any())
            and self._completed_forwards == 1
            and self._event_state == "com",
            "exact single nominal zero-action step after constructor",
        )
        self._step_used = True
        try:
            result = _RUNTIME_STEP(env, actions, capture_control=True)
            _need(
                type(result) is dict
                and result.get("optimizer_launched") is False
                and len(result.get("boundaries", ())) == 11
                and bool((env.steps == 10).all())
                and self._completed_forwards == FORWARDS
                and self._event_state == "com",
                "one complete nominal ten-substep eager step, no reset/extra forward",
            )
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise

    @property
    def com_initialized(self):
        return tuple(self._com_initialized)

    @property
    def com_weighted(self):
        return tuple(self._com_weighted)

    @property
    def rne_inputs(self):
        return tuple(self._rne_inputs)

    @property
    def rne_outputs(self):
        return tuple(self._rne_outputs)

    @property
    def receipt(self):
        return deepcopy(
            {
                "protocol": PROTOCOL,
                "status": self._status,
                "fault": self._fault,
                "forward_calls": self._completed_forwards,
                "constructor_forward_calls": min(self._completed_forwards, 1),
                "step_calls": int(self._step_used),
                "worlds": None if self._binding is None else self._binding[2],
                "device": None if self._device is None else str(self._device),
                "calls": self._calls,
                "dispatch_trace": self._dispatch_trace,
                "com_initialized_sha256": [
                    sha256(raw).hexdigest() for raw in self._com_initialized
                ],
                "com_weighted_sha256": [
                    sha256(raw).hexdigest() for raw in self._com_weighted
                ],
                "rne_input_sha256": [
                    sha256(raw).hexdigest() for raw in self._rne_inputs
                ],
                "rne_output_sha256": [
                    sha256(raw).hexdigest() for raw in self._rne_outputs
                ],
                "dispatch": {
                    "forward_cap": FORWARDS,
                    "com_original_logical_launches": 11,
                    "com_underlying_launches": 13,
                    "com_split_index": COM_SPLIT_INDEX,
                    "crb_logical_accumulation_launches": 7,
                    "crb_underlying_accumulation_launches": 9,
                    "crb_qM_launches": 1,
                    "crb_split_index": CRB_SPLIT_INDEX,
                    "rne_bias_logical_launches": 7,
                    "rne_bias_underlying_launches": 9,
                    "rne_split_index": RNE_SPLIT_INDEX,
                    "split_body_order": list(SPLIT_BODY_IDS),
                    "sensory_rne_untouched_passthrough_calls": self._entry_counts[
                        "rne_post"
                    ],
                    "constructor_plus_one_step_only": True,
                    "all_observation_readbacks_perturb_timing": True,
                    "closed": self._status == "complete",
                },
                "entry_counts": dict(self._entry_counts),
                "flags": dict.fromkeys(_FLAG_NAMES, False),
            }
        )


_PINNED_METHODS = tuple(
    (name, getattr(SerialStepControl, name))
    for name in (
        "_guard",
        "__enter__",
        "__exit__",
        "_restore_owned",
        "_release",
        "_validate_model_data",
        "_layout_receipt",
        "_make_cycle",
        "_call_com",
        "_call_crb",
        "_call_rne",
        "_launch",
        "_launch_com",
        "_launch_crb",
        "_launch_rne",
        "_split",
        "_trace_group",
        "bind_runtime",
        "step",
    )
)
