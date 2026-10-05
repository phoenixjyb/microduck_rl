"""One-shot process-local CRB launch control for the pinned stance topology.

The scope wraps the installed CRB entry and Warp launch function only while
that exact entry is running. It never substitutes a kernel, reads a solved
CRB result, changes an input array, or qualifies native behavior.
"""

from copy import deepcopy
from hashlib import sha256
import os
import stat
from threading import Lock, active_count, current_thread, main_thread

import numpy as np
import warp as wp

import mujoco_warp as public_mjwarp
from mujoco_warp._src import forward as pinned_forward
from mujoco_warp._src import smooth as pinned_smooth
from mujoco_warp._src.types import vec10

PROTOCOL = "football-b1d-crb-runtime-control-v1"
PINNED_SMOOTH_SHA256 = (
    "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
)
PINNED_FORWARD_SHA256 = (
    "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3"
)
MAX_FORWARD_CALLS = 4
MAX_WORLDS = (2, 64)
EXPECTED_PARENTS = (0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14)
EXPECTED_REVERSED_LEVELS = (
    (6, 15),
    (5, 10, 14),
    (4, 9, 13),
    (3, 8, 12),
    (2, 7, 11),
    (1,),
    (0,),
)
EXPECTED_DEPTH_LEVELS = tuple(reversed(EXPECTED_REVERSED_LEVELS))
SPLIT_DEPTH_INDEX = 2
SPLIT_BODY_IDS = (2, 7, 11)
FLAGS = {
    "native_qualified": False,
    "original_pair_accepted": False,
    "cause_proven": False,
    "full_window_passed": False,
    "training_authorized": False,
    "physical_result_accepted": False,
}

_PROCESS_SCOPE_LOCK = Lock()
_PINNED_CRB = pinned_smooth.crb
_PINNED_ACCUMULATE = pinned_smooth._crb_accumulate
_PINNED_QM_DENSE = pinned_smooth._qM_dense
_PINNED_WP_LAUNCH = wp.launch
_PINNED_FORWARD = pinned_forward.forward
_PINNED_FWD_POSITION = pinned_forward.fwd_position
_PINNED_PUBLIC_FORWARD = public_mjwarp.forward
_PINNED_PUBLIC_FWD_POSITION = public_mjwarp.fwd_position
_PINNED_PUBLIC_CRB = public_mjwarp.crb
_ACTIVE_SCOPE = None


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _source_digest(path):
    _need(type(path) is str and path.endswith(".py"), "pinned Python source path")
    before_path = os.lstat(path)
    _need(
        not stat.S_ISLNK(before_path.st_mode)
        and stat.S_ISREG(before_path.st_mode)
        and 0 < before_path.st_size <= 512 * 1024,
        "bounded regular non-symlink pinned source",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _need(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= 512 * 1024
            and (before.st_dev, before.st_ino)
            == (before_path.st_dev, before_path.st_ino),
            "stable bounded pinned source descriptor",
        )
        chunks = []
        total = 0
        while total <= 512 * 1024:
            chunk = os.read(descriptor, min(64 * 1024, 512 * 1024 + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        _need(
            total == before.st_size
            and total <= 512 * 1024
            and (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ),
            "pinned source descriptor remained stable while hashing",
        )
        return sha256(b"".join(chunks)).hexdigest()
    finally:
        os.close(descriptor)


def _check_caller_refs():
    _need(
        pinned_smooth.wp is wp
        and pinned_smooth._crb_accumulate is _PINNED_ACCUMULATE
        and pinned_smooth._qM_dense is _PINNED_QM_DENSE
        and pinned_forward.smooth is pinned_smooth
        and pinned_forward.forward is _PINNED_FORWARD
        and pinned_forward.fwd_position is _PINNED_FWD_POSITION
        and public_mjwarp.forward is _PINNED_PUBLIC_FORWARD
        and public_mjwarp.fwd_position is _PINNED_PUBLIC_FWD_POSITION
        and public_mjwarp.crb is _PINNED_PUBLIC_CRB,
        "pinned forward module and public call aliases",
    )


def _check_sources():
    _need(
        _source_digest(pinned_smooth.__file__) == PINNED_SMOOTH_SHA256
        and _source_digest(pinned_forward.__file__) == PINNED_FORWARD_SHA256,
        "exact authenticated smooth and forward sources",
    )


def _array_fingerprint(array):
    return (
        id(array),
        int(array.ptr),
        tuple(int(size) for size in array.shape),
        tuple(int(stride) for stride in array.strides),
        array.dtype,
        array.device,
        array.is_contiguous,
    )


def _warp_int_ids(array, size, device, label):
    _need(
        array.dtype is wp.int32
        and array.ndim == 1
        and array.shape == (size,)
        and array.device == device
        and array.is_contiguous is True,
        f"{label} Warp int32 vector layout/device",
    )
    values = array.numpy()
    _need(
        isinstance(values, np.ndarray)
        and values.shape == (size,)
        and values.dtype == np.dtype(np.int32),
        f"{label} host ID snapshot layout",
    )
    return tuple(int(item) for item in values.tolist())


def _validate_mode(mode):
    _need(
        type(mode) is str and mode in {"serial", "concurrent"},
        "serial or concurrent mode",
    )


def _validate_bound_model(model, data):
    _need(
        type(model.nbody) is int
        and model.nbody == 16
        and type(model.nq) is int
        and model.nq == 21
        and type(model.nv) is int
        and model.nv == 20
        and type(model.nu) is int
        and model.nu == 14
        and model.is_sparse is False,
        "exact dense 16-body MicroDuck CRB model dimensions",
    )
    _need(
        type(data.nworld) is int and data.nworld in MAX_WORLDS, "two or 64 CRB worlds"
    )
    device = model.body_parentid.device
    _need(
        (data.nworld == 2 and str(device) == "cpu")
        or (data.nworld == 64 and str(device) == "cuda:0"),
        "two-world CPU fixture or 64-world CUDA runtime only",
    )
    _need(not device.is_capturing, "CRB schedule control refuses Warp capture")
    _need(
        _warp_int_ids(model.body_parentid, 16, device, "body parents")
        == EXPECTED_PARENTS,
        "exact pinned body parent vector",
    )
    _need(
        type(model.body_tree) is tuple and len(model.body_tree) == 7,
        "exact seven-level body tree tuple",
    )
    tree_ids = tuple(
        _warp_int_ids(level, len(expected), device, "body tree level")
        for level, expected in zip(model.body_tree, EXPECTED_DEPTH_LEVELS, strict=True)
    )
    _need(tree_ids == EXPECTED_DEPTH_LEVELS, "exact pinned depth-level body IDs")
    _need(
        data.crb.dtype is vec10
        and data.crb.shape == (data.nworld, model.nbody)
        and data.crb.device == device
        and data.cinert.dtype is vec10
        and data.cinert.shape == (data.nworld, model.nbody)
        and data.cinert.device == device
        and data.crb is not data.cinert
        and data.qM.dtype is wp.float32
        and data.qM.shape == (data.nworld, model.nv, model.nv)
        and data.qM.device == device,
        "exact same-device dense CRB, cinert, and qM arrays",
    )
    _need(
        model.dof_bodyid.dtype is wp.int32
        and model.dof_bodyid.shape == (model.nv,)
        and model.dof_parentid.dtype is wp.int32
        and model.dof_parentid.shape == (model.nv,)
        and model.dof_armature.dtype is wp.float32
        and model.dof_armature.ndim == 2
        and model.dof_armature.shape[1] == model.nv
        and model.dof_armature.shape[0] in (1, data.nworld)
        and data.cdof.dtype is wp.spatial_vector
        and data.cdof.shape == (data.nworld, model.nv)
        and all(
            array.device == device
            for array in (
                model.dof_bodyid,
                model.dof_parentid,
                model.dof_armature,
                data.cdof,
            )
        ),
        "exact dense qM input array types, shapes, and device",
    )
    return device


def _bound_model_layout_unchanged(model, data, device, worlds):
    return (
        type(model.nbody) is int
        and model.nbody == 16
        and type(model.nq) is int
        and model.nq == 21
        and type(model.nv) is int
        and model.nv == 20
        and type(model.nu) is int
        and model.nu == 14
        and model.is_sparse is False
        and type(data.nworld) is int
        and data.nworld == worlds
        and model.body_parentid.device == device
        and type(model.body_tree) is tuple
        and len(model.body_tree) == 7
        and data.crb.dtype is vec10
        and data.crb.shape == (data.nworld, model.nbody)
        and data.crb.device == device
        and data.cinert.dtype is vec10
        and data.cinert.shape == (data.nworld, model.nbody)
        and data.cinert.device == device
        and data.crb is not data.cinert
        and data.qM.dtype is wp.float32
        and data.qM.shape == (data.nworld, model.nv, model.nv)
        and data.qM.device == device
        and model.dof_bodyid.dtype is wp.int32
        and model.dof_bodyid.shape == (model.nv,)
        and model.dof_bodyid.device == device
        and model.dof_parentid.dtype is wp.int32
        and model.dof_parentid.shape == (model.nv,)
        and model.dof_parentid.device == device
        and model.dof_armature.dtype is wp.float32
        and model.dof_armature.ndim == 2
        and model.dof_armature.shape[1] == model.nv
        and model.dof_armature.shape[0] in (1, data.nworld)
        and model.dof_armature.device == device
        and data.cdof.dtype is wp.spatial_vector
        and data.cdof.shape == (data.nworld, model.nv)
        and data.cdof.device == device
    )


class StanceCrbRuntimeControl:
    """One-shot launch schedule scope; bind its exact runtime before stepping.

    Enter before constructing ScheduledRecoveryRuntime to cover its first
    reset forward. After construction call bind_runtime(env) before caller-
    directed forwards or stepping. Concurrent mode retains all seven original
    accumulation launches; serial mode splits only bodies 2, 7, and 11.
    """

    def __init__(self, *, mode, max_forward_calls=4):
        _validate_mode(mode)
        _need(
            type(max_forward_calls) is int
            and 1 <= max_forward_calls <= MAX_FORWARD_CALLS,
            "plain bounded forward-call cap",
        )
        self.mode = mode
        self.max_forward_calls = max_forward_calls
        self._used = False
        self._active = False
        self._completed = False
        self._faulted = False
        self._fault_type = None
        self._original_crb = None
        self._original_launch = None
        self._entry_callable = None
        self._model = None
        self._data = None
        self._device = None
        self._array_refs = None
        self._array_fingerprints = None
        self._worlds = None
        self._declared_mode = mode
        self._declared_cap = max_forward_calls
        self._singleton_body_arrays = None
        self._runtime_bound = False
        self._crb_active = False
        self._constructor_forward_calls = 0
        self._forward_calls = 0
        self._topology_id_snapshots = 0
        self._level_launch_requests = 0
        self._parent_zero_noop_level_requests = 0
        self._actual_accumulate_launches = 0
        self._split_child_launches = 0
        self._qM_launches = 0

    def __enter__(self):
        global _ACTIVE_SCOPE
        if self._used:
            self._faulted = True
            self._fault_type = self._fault_type or "ScopeReuse"
            raise ValueError("one-shot CRB scope")
        self._used = True
        if current_thread() is not main_thread() or active_count() != 1:
            self._faulted = True
            self._fault_type = "ThreadBoundaryViolation"
            raise ValueError("CRB scope requires isolated main Python thread")
        if not _PROCESS_SCOPE_LOCK.acquire(blocking=False):
            self._faulted = True
            self._fault_type = "ProcessScopeBusy"
            raise ValueError("another process CRB scope is active")
        try:
            _need(_ACTIVE_SCOPE is None, "no nested process CRB scope")
            _need(
                pinned_smooth.crb is _PINNED_CRB
                and pinned_smooth._crb_accumulate is _PINNED_ACCUMULATE
                and pinned_smooth._qM_dense is _PINNED_QM_DENSE
                and wp.launch is _PINNED_WP_LAUNCH,
                "original pinned smooth and Warp launch references",
            )
            _check_caller_refs()
            _check_sources()
            source_sha256 = PINNED_SMOOTH_SHA256
            self._forward_source_sha256 = PINNED_FORWARD_SHA256
            self._original_crb = _PINNED_CRB
            self._original_launch = _PINNED_WP_LAUNCH
            self._entry_callable = self._controlled_crb
            self._source_sha256 = source_sha256
            _ACTIVE_SCOPE = self
            self._active = True
            pinned_smooth.crb = self._entry_callable
            return self
        except BaseException:
            _ACTIVE_SCOPE = None
            self._active = False
            _PROCESS_SCOPE_LOCK.release()
            self._faulted = True
            self._fault_type = self._fault_type or "ScopeEntryFailure"
            raise

    def __exit__(self, kind, error, _tb):
        global _ACTIVE_SCOPE
        if kind is not None:
            self._faulted = True
            self._fault_type = kind.__name__
        if self._crb_active:
            self._faulted = True
            self._fault_type = self._fault_type or "ReentrantExit"
        if (
            type(self.mode) is not str
            or self.mode != self._declared_mode
            or type(self.max_forward_calls) is not int
            or self.max_forward_calls != self._declared_cap
        ):
            self._faulted = True
            self._fault_type = self._fault_type or "ModeOrCapChanged"
        if pinned_smooth.crb is self._entry_callable:
            pinned_smooth.crb = self._original_crb
        else:
            self._faulted = True
            self._fault_type = self._fault_type or "SmoothReferenceChanged"
        if wp.launch is not self._original_launch:
            self._faulted = True
            self._fault_type = self._fault_type or "WarpLaunchReferenceChanged"
        try:
            _check_caller_refs()
            _check_sources()
        except BaseException as source_error:
            self._faulted = True
            self._fault_type = self._fault_type or type(source_error).__name__
        self._active = False
        if _ACTIVE_SCOPE is self:
            _ACTIVE_SCOPE = None
        _PROCESS_SCOPE_LOCK.release()
        if kind is None and not (
            self._runtime_bound
            and self._constructor_forward_calls == 1
            and self._forward_calls == self.max_forward_calls
            and not self._faulted
        ):
            self._faulted = True
            self._fault_type = self._fault_type or "IncompleteScope"
            raise ValueError("complete bound CRB control scope")
        if kind is None and not self._faulted:
            self._completed = True
        return False

    def _check_thread(self):
        if current_thread() is not main_thread() or active_count() != 1:
            self._faulted = True
            self._fault_type = self._fault_type or "ThreadBoundaryViolation"
            raise ValueError("exclusive isolated main Python thread")
        if _ACTIVE_SCOPE is not self or not self._active:
            self._faulted = True
            self._fault_type = self._fault_type or "InactiveScope"
            raise ValueError("active owned CRB scope")
        _need(
            type(self.mode) is str
            and self.mode == self._declared_mode
            and type(self.max_forward_calls) is int
            and self.max_forward_calls == self._declared_cap,
            "constructor-bound immutable CRB mode and call cap",
        )

    def _bind_first_arrays(self, model, data):
        device = _validate_bound_model(model, data)
        self._model = model
        self._data = data
        self._device = device
        self._worlds = data.nworld
        self._array_refs = (
            model.body_parentid,
            model.body_tree,
            tuple(model.body_tree),
            data.cinert,
            data.crb,
            data.qM,
            model.dof_bodyid,
            model.dof_parentid,
            model.dof_armature,
            data.cdof,
        )
        arrays = (
            model.body_parentid,
            *model.body_tree,
            data.cinert,
            data.crb,
            data.qM,
            model.dof_bodyid,
            model.dof_parentid,
            model.dof_armature,
            data.cdof,
        )
        _need(
            all(
                array.is_contiguous is True and array.device == device
                for array in arrays
            ),
            "contiguous same-device bound Warp arrays",
        )
        self._array_fingerprints = tuple(_array_fingerprint(array) for array in arrays)
        self._singleton_body_arrays = tuple(
            wp.array([body], dtype=wp.int32, device=device) for body in SPLIT_BODY_IDS
        )
        _need(
            all(
                array.device == device
                and array.dtype is wp.int32
                and array.shape == (1,)
                and array.is_contiguous is True
                for array in self._singleton_body_arrays
            ),
            "same-device singleton child arrays",
        )

    def _check_arrays_unchanged(self, model, data):
        _need(
            model is self._model and data is self._data,
            "same bound model and data identity",
        )
        (
            parent_ref,
            tree_ref,
            tree_levels,
            cinert_ref,
            crb_ref,
            qM_ref,
            dof_bodyid_ref,
            dof_parentid_ref,
            dof_armature_ref,
            cdof_ref,
        ) = self._array_refs
        _need(
            model.body_parentid is parent_ref
            and model.body_tree is tree_ref
            and type(model.body_tree) is tuple
            and len(model.body_tree) == 7
            and all(a is b for a, b in zip(model.body_tree, tree_levels, strict=True))
            and data.cinert is cinert_ref
            and data.crb is crb_ref
            and data.qM is qM_ref
            and model.dof_bodyid is dof_bodyid_ref
            and model.dof_parentid is dof_parentid_ref
            and model.dof_armature is dof_armature_ref
            and data.cdof is cdof_ref,
            "bound model/data array identities unchanged",
        )
        _need(
            _bound_model_layout_unchanged(model, data, self._device, self._worlds)
            and tuple(
                _array_fingerprint(array)
                for array in (
                    model.body_parentid,
                    *model.body_tree,
                    data.cinert,
                    data.crb,
                    data.qM,
                    model.dof_bodyid,
                    model.dof_parentid,
                    model.dof_armature,
                    data.cdof,
                )
            )
            == self._array_fingerprints,
            "bound CRB array storage/layout/device unchanged",
        )

    def _check_topology_contents(self, model):
        _need(
            _warp_int_ids(model.body_parentid, 16, self._device, "body parents")
            == EXPECTED_PARENTS,
            "body parent IDs remain exact before CRB launch",
        )
        tree_ids = tuple(
            _warp_int_ids(level, len(expected), self._device, "body tree level")
            for level, expected in zip(
                model.body_tree, EXPECTED_DEPTH_LEVELS, strict=True
            )
        )
        _need(
            tree_ids == EXPECTED_DEPTH_LEVELS,
            "body tree level IDs remain exact before CRB launch",
        )
        self._topology_id_snapshots += 1

    @staticmethod
    def _device_capturing(model, data):
        return bool(
            model.body_parentid.device.is_capturing or data.crb.device.is_capturing
        )

    def _controlled_crb(self, model, data):
        try:
            self._check_thread()
            _need(not self._faulted, "faulted CRB scope cannot continue")
            _need(not self._crb_active, "nonreentrant pinned CRB call")
            _need(
                self._forward_calls < self.max_forward_calls,
                "bounded maximum CRB forward-call count",
            )
            _need(
                wp.launch is self._original_launch, "unmodified Warp launch before CRB"
            )
            _need(
                pinned_smooth.crb is self._entry_callable,
                "owned pinned CRB entry remains installed",
            )
            _check_caller_refs()
            _check_sources()
            _need(
                not self._device_capturing(model, data),
                "CRB control refuses active Warp capture",
            )
            if self._model is None:
                self._bind_first_arrays(model, data)
                self._topology_id_snapshots += 1
            else:
                self._check_arrays_unchanged(model, data)
                self._check_topology_contents(model)
            self._crb_active = True
            self._forward_calls += 1
            if not self._runtime_bound:
                self._constructor_forward_calls += 1
            stage = {"level": 0, "qm": 0, "launches": 0}
            launch_wrapper = self._make_launch_wrapper(model, data, stage)
            wp.launch = launch_wrapper
            try:
                result = self._original_crb(model, data)
                _need(
                    pinned_smooth.crb is self._entry_callable,
                    "owned pinned CRB entry remained installed",
                )
                _check_caller_refs()
                _check_sources()
                _need(result is None, "pinned smooth.crb returns None")
                _need(
                    stage["level"] == 7 and stage["qm"] == 1,
                    "exact seven CRB levels and one dense qM launch",
                )
                expected_actual = 9 if self.mode == "serial" else 7
                _need(
                    stage["launches"] == expected_actual + 1,
                    "exact mode-specific accumulation and dense qM launch count",
                )
            finally:
                if wp.launch is launch_wrapper:
                    wp.launch = self._original_launch
                else:
                    self._faulted = True
                    self._fault_type = self._fault_type or "WarpLaunchReferenceChanged"
            return result
        except BaseException as error:
            self._faulted = True
            self._fault_type = self._fault_type or type(error).__name__
            raise
        finally:
            self._crb_active = False

    def _make_launch_wrapper(self, model, data, stage):
        original_launch = self._original_launch
        tree_levels = model.body_tree

        def controlled_launch(*args, **kwargs):
            try:
                self._check_thread()
                _need(self._crb_active, "Warp launch is limited to active original CRB")
                _need(
                    not self._device_capturing(model, data),
                    "CRB control refuses active Warp capture",
                )
                _need(
                    len(args) == 1
                    and type(kwargs) is dict
                    and set(kwargs) == {"dim", "inputs", "outputs"},
                    "exact pinned Warp launch call signature",
                )
                kernel = args[0]
                if kernel is _PINNED_ACCUMULATE:
                    level_index = stage["level"]
                    _need(
                        level_index < 7, "exact seven ordered CRB accumulation requests"
                    )
                    tree_index = 6 - level_index
                    tree = tree_levels[tree_index]
                    inputs = kwargs["inputs"]
                    outputs = kwargs["outputs"]
                    _need(
                        type(inputs) is list
                        and len(inputs) == 3
                        and inputs[0] is model.body_parentid
                        and inputs[1] is data.crb
                        and inputs[2] is tree
                        and type(outputs) is list
                        and len(outputs) == 1
                        and outputs[0] is data.crb
                        and type(kwargs["dim"]) is tuple
                        and kwargs["dim"] == (data.nworld, tree.shape[0]),
                        "exact bound CRB level launch arrays and dimensions",
                    )
                    self._level_launch_requests += 1
                    stage["level"] += 1
                    if level_index in (5, 6):
                        self._parent_zero_noop_level_requests += 1
                    if level_index == 4 and self.mode == "serial":
                        for singleton in self._singleton_body_arrays:
                            original_launch(
                                _PINNED_ACCUMULATE,
                                dim=(data.nworld, 1),
                                inputs=[model.body_parentid, data.crb, singleton],
                                outputs=[data.crb],
                            )
                            self._actual_accumulate_launches += 1
                            self._split_child_launches += 1
                            stage["launches"] += 1
                        return None
                    result = original_launch(*args, **kwargs)
                    self._actual_accumulate_launches += 1
                    stage["launches"] += 1
                    return result
                _need(
                    kernel is _PINNED_QM_DENSE,
                    "only original dense qM kernel follows CRB",
                )
                _need(
                    stage["level"] == 7 and stage["qm"] == 0,
                    "qM launch follows all seven CRB levels once",
                )
                inputs = kwargs["inputs"]
                outputs = kwargs["outputs"]
                _need(
                    type(kwargs["dim"]) is tuple
                    and kwargs["dim"] == (data.nworld, model.nv)
                    and type(inputs) is list
                    and len(inputs) == 5
                    and inputs[0] is model.dof_bodyid
                    and inputs[1] is model.dof_parentid
                    and inputs[2] is model.dof_armature
                    and inputs[3] is data.cdof
                    and inputs[4] is data.crb
                    and type(outputs) is list
                    and len(outputs) == 1
                    and outputs[0] is data.qM,
                    "exact original dense qM passthrough arrays and dimensions",
                )
                stage["qm"] += 1
                self._qM_launches += 1
                stage["launches"] += 1
                return original_launch(*args, **kwargs)
            except BaseException as error:
                self._faulted = True
                self._fault_type = self._fault_type or type(error).__name__
                raise

        return controlled_launch

    def bind_runtime(self, env):
        """Bind the constructed runtime after its constructor forward is covered."""
        self._check_thread()
        try:
            _need(not self._faulted, "faulted CRB scope cannot bind a runtime")
            _need(
                pinned_smooth.crb is self._entry_callable,
                "owned pinned CRB entry remains installed",
            )
            _check_caller_refs()
            _check_sources()
            from mjlab_microduck.stance_recovery_schedule_runtime import (
                ScheduledRecoveryRuntime,
            )
            from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

            _need(
                not self._runtime_bound
                and type(env) is ScheduledRecoveryRuntime
                and env.forward_graph is None
                and type(env.n) is int
                and env.n == self._worlds
                and str(env.device) == str(self._device)
                and env.wp_device == self._device
                and getattr(env._forward, "__func__", None)
                is WarpStanceRuntime._forward
                and getattr(env._scheduled_forward, "__func__", None)
                is ScheduledRecoveryRuntime._scheduled_forward
                and not {"_forward", "_scheduled_forward"} & set(env.__dict__),
                "exact unmodified eager scheduled runtime",
            )
            _need(
                self._model is env.model
                and self._data is env.data
                and self._constructor_forward_calls == 1
                and self._forward_calls == 1
                and not self._faulted,
                "one constructor reset forward bound to runtime model/data",
            )
            self._runtime_bound = True
        except BaseException as error:
            self._faulted = True
            self._fault_type = self._fault_type or type(error).__name__
            raise

    @property
    def receipt(self):
        """Detached bounded counters; never native qualification."""
        return deepcopy(
            {
                "protocol": PROTOCOL,
                "status": "faulted"
                if self._faulted
                else "active"
                if self._active
                else "complete"
                if self._completed
                else "not-started",
                "mode": self.mode,
                "max_forward_calls": self.max_forward_calls,
                "smooth_source_sha256": getattr(self, "_source_sha256", None),
                "forward_source_sha256": getattr(self, "_forward_source_sha256", None),
                "topology_id_snapshots": self._topology_id_snapshots,
                "singleton_child_arrays_allocated": (
                    0
                    if self._singleton_body_arrays is None
                    else len(self._singleton_body_arrays)
                ),
                "initialization_timing_changed": self._model is not None,
                "constructor_forward_calls": self._constructor_forward_calls,
                "constructor_forward_covered": self._runtime_bound
                and self._constructor_forward_calls == 1,
                "runtime_bound": self._runtime_bound,
                "forward_calls": self._forward_calls,
                "original_level_launch_requests": self._level_launch_requests,
                "parent_zero_noop_level_requests": self._parent_zero_noop_level_requests,
                "actual_accumulation_launches": self._actual_accumulate_launches,
                "split_child_launches": self._split_child_launches,
                "dense_qM_launches": self._qM_launches,
                "fault_type": self._fault_type,
                "flags": dict(FLAGS),
            }
        )
