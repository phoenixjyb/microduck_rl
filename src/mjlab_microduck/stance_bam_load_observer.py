"""Read-only BAM load/friction snapshots around one existing serial step.

This observer composes with SerialStepControl; it does not alter that
control's dispatch, reproduce BAM arithmetic, or turn observations into a
runtime-cause or training-admission claim.
"""

from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
from threading import Lock, active_count, current_thread, main_thread
from types import MethodType

import numpy as np
import torch
import bam.mjlab as bam_bridge

from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
from mjlab_microduck.stance_control_state import BamStateCommit
from mjlab_microduck.stance_serial_step_control import SerialStepControl
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

PROTOCOL = "microduck-bam-load-observer-oct7-v1"
CALLS = 10
MAX_FIELD_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
WORLD_COUNTS = (2, 64)
NV = 20
JOINTS = 14
EFC_ROWS = 512

_LOCK = Lock()
_BAM_COMPUTE = bam_bridge.BamActuator.compute
_BAM_COMPUTE_CODE = _BAM_COMPUTE.__code__
_DOF_FRICTION = bam_bridge.BamActuator._dof_friction_force
_DOF_FRICTION_CODE = _DOF_FRICTION.__code__
_BUDGET = FrictionDRBamActuator._compute_friction_budget
_BUDGET_CODE = _BUDGET.__code__
_BAM_BUDGET = bam_bridge.BamActuator._compute_friction_budget
_BAM_BUDGET_CODE = _BAM_BUDGET.__code__
_COMMIT_COMPUTE = BamStateCommit.compute
_COMMIT_COMPUTE_CODE = _COMMIT_COMPUTE.__code__
_COMMIT_UNWRAPPED = inspect.unwrap(_COMMIT_COMPUTE)
_COMMIT_UNWRAPPED_CODE = _COMMIT_UNWRAPPED.__code__
_RUNTIME_STEP = WarpStanceRuntime.step
_RUNTIME_STEP_CODE = inspect.unwrap(_RUNTIME_STEP).__code__
_F32_FIELDS = (
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_actuator",
    "efc_force",
    "qfrc_friction",
    "budget_motor",
    "budget_external",
    "budget_stribeck",
    "budget_output",
    "friction_scale",
)
_I32_FIELDS = ("efc_type", "efc_id", "nefc")
_ALL_FIELDS = (*_F32_FIELDS, *_I32_FIELDS)
_M6_SHA256 = "61c699362fb3fabdde93eeba5e1ad3bf4ef9ca2f71d03e316b1924ff005b20d3"
_BAM_SHA256 = "af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440"
_ADAPTER_SHA256 = "e9b349a4be9910ea2c452cacfa78ccfc7fadb9f60e2f5ab8b54eed835e09b2b4"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _tensor_bytes(value, shape, dtype, device, label):
    _need(
        torch.is_tensor(value)
        and tuple(value.shape) == tuple(shape)
        and value.dtype is dtype
        and value.device == device
        and value.is_contiguous(),
        f"{label}: exact shape/dtype/device/contiguous layout",
    )
    if dtype.is_floating_point:
        _need(bool(torch.isfinite(value).all()), f"{label}: finite complete tensor")
        host = value.detach().to(device="cpu").contiguous().numpy()
        _need(host.dtype == np.dtype("float32"), f"{label}: float32 wire dtype")
        return host.astype("<f4", copy=False).tobytes(order="C")
    host = value.detach().to(device="cpu").contiguous().numpy()
    _need(host.dtype == np.dtype("int32"), f"{label}: int32 wire dtype")
    return host.astype("<i4", copy=False).tobytes(order="C")


def _module_pin(module, expected, label):
    _need(_digest(module.__file__) == expected, f"pinned {label} source bytes")


class BamLoadObserver:
    """Capture ten real actuator proposals inside an unchanged one-step scope.

    Enter while SerialStepControl is active, construct and bind the runtime,
    then call bind_runtime before its one step. Temporary hooks are restored
    only when still owned by this object.
    """

    def __init__(self):
        self._status = "not-started"
        self._fault = None
        self._used = False
        self._owns_lock = False
        self._inside_compute = False
        self._runtime = None
        self._control = None
        self._motor = None
        self._data = None
        self._bridge_data = None
        self._device = None
        self._worlds = None
        self._original_compute = None
        self._compute_wrapper = None
        self._dof_wrapper = lambda staged, nv: self._dof_friction_wrapper(staged, nv)
        self._budget_wrapper = lambda staged, motor, external, stribeck: (
            self._budget_call_wrapper(staged, motor, external, stribeck)
        )
        self._entries = []
        self._call_records = []
        self._runtime_parameters = []
        self._current = None
        self._m6_json = None
        self._raw_layouts = None
        self._m6_path = Path(bam_bridge.__file__).parent / "params/xl330/m6.json"

    def __enter__(self):
        if self._used or type(self) is not BamLoadObserver:
            raise ValueError("one-shot exact BAM load observation scope")
        if current_thread() is not main_thread() or active_count() != 1:
            self._status, self._fault = "faulted", "ThreadBoundaryViolation"
            raise ValueError("isolated main-thread observer required")
        self._used = True
        if not _LOCK.acquire(blocking=False):
            self._status, self._fault = "faulted", "ObserverScopeBusy"
            raise ValueError("exclusive BAM observer scope required")
        self._owns_lock = True
        try:
            self._source_checks()
            _need(
                bam_bridge.BamActuator.compute is _BAM_COMPUTE
                and bam_bridge.BamActuator._dof_friction_force is _DOF_FRICTION
                and FrictionDRBamActuator._compute_friction_budget is _BUDGET,
                "unmodified pinned BAM compute and friction methods",
            )
            raw = self._m6_path.read_bytes()
            _need(sha256(raw).hexdigest() == _M6_SHA256, "whole pinned XL330 m6 JSON")
            self._m6_json = json.loads(raw)
            self._status = "active"
            return self
        except BaseException as error:
            self._fault = type(error).__name__
            self._status = "faulted"
            self._release()
            raise

    def _source_checks(self):
        _module_pin(bam_bridge, _BAM_SHA256, "installed BAM")
        from mjlab_microduck import stance_control_state as state
        from mjlab_microduck.actuator import friction_dr_bam as adapter

        _module_pin(
            state,
            "7974953ded3c96e4a05ba05d15bfc82e12ea3b8255b150c8636ac07f580d26b9",
            "BAM state commit",
        )
        _module_pin(adapter, _ADAPTER_SHA256, "stance friction adapter")
        _need(
            sha256(self._m6_path.read_bytes()).hexdigest() == _M6_SHA256,
            "whole pinned XL330 m6 parameter JSON remains unchanged",
        )
        _need(
            bam_bridge.BamActuator.compute is _BAM_COMPUTE
            and bam_bridge.BamActuator.compute.__code__ is _BAM_COMPUTE_CODE
            and (
                bam_bridge.BamActuator._dof_friction_force is self._dof_wrapper
                or (
                    bam_bridge.BamActuator._dof_friction_force is _DOF_FRICTION
                    and _DOF_FRICTION.__code__ is _DOF_FRICTION_CODE
                )
            )
            and (
                FrictionDRBamActuator._compute_friction_budget is self._budget_wrapper
                or (
                    FrictionDRBamActuator._compute_friction_budget is _BUDGET
                    and _BUDGET.__code__ is _BUDGET_CODE
                )
            )
            and BamStateCommit.compute is _COMMIT_COMPUTE
            and BamStateCommit.compute.__code__ is _COMMIT_COMPUTE_CODE
            and inspect.unwrap(_COMMIT_COMPUTE) is _COMMIT_UNWRAPPED
            and _COMMIT_UNWRAPPED.__code__ is _COMMIT_UNWRAPPED_CODE
            and _DOF_FRICTION.__code__ is _DOF_FRICTION_CODE
            and _BUDGET.__code__ is _BUDGET_CODE
            and bam_bridge.BamActuator._compute_friction_budget is _BAM_BUDGET
            and _BAM_BUDGET.__code__ is _BAM_BUDGET_CODE
            and WarpStanceRuntime.step is _RUNTIME_STEP
            and inspect.unwrap(_RUNTIME_STEP).__code__ is _RUNTIME_STEP_CODE,
            "pinned compute sources and original runtime step",
        )

    def bind_runtime(self, control, env):
        self._guard()
        _need(
            self._runtime is None
            and type(control) is SerialStepControl
            and control._status == "active"
            and control._owns_lock
            and control._runtime is env
            and control._completed_forwards == 1
            and not control._step_used
            and type(env) is WarpStanceRuntime
            and env.forward_graph is None
            and not env.faulted
            and env.model is control._model
            and env.data is control._data
            and env.n in WORLD_COUNTS
            and (
                (env.n == 2 and str(env.wp_device) == "cpu")
                or (env.n == 64 and str(env.wp_device) == "cuda:0")
            )
            and type(env.motor) is BamStateCommit
            and type(env.motor).compute is _COMMIT_COMPUTE
            and type(env.motor.actuator) is FrictionDRBamActuator
            and "compute" not in env.motor.__dict__,
            "exact bound nominal runtime after constructor forward and before step",
        )
        _need(
            bam_bridge.BamActuator._dof_friction_force is _DOF_FRICTION
            and FrictionDRBamActuator._compute_friction_budget is _BUDGET
            and "compute" not in env.motor.__dict__,
            "unmodified BAM hooks and no instance-level compute override",
        )
        self._control, self._runtime, self._motor = control, env, env.motor
        self._data, self._worlds, self._device = env.data, env.n, env.device
        self._bridge_data = env.motor.actuator._data
        _need(
            self._bridge_data is env.data_bridge
            and self._bridge_data.struct is env.data,
            "actual BAM bridge aliases the bound raw solver data",
        )
        self._raw_layouts = self._capture_raw_layouts(env.data)
        self._original_compute = env.motor.compute
        self._compute_wrapper = MethodType(self._commit_compute_wrapper, env.motor)
        env.motor.compute = self._compute_wrapper
        bam_bridge.BamActuator._dof_friction_force = self._dof_wrapper
        FrictionDRBamActuator._compute_friction_budget = self._budget_wrapper
        return self

    @staticmethod
    def _capture_raw_layouts(data):
        names = (
            "qfrc_bias",
            "qfrc_constraint",
            "qfrc_actuator",
            "nefc",
        )
        arrays = {name: getattr(data, name) for name in names}
        arrays.update(
            {"efc." + name: getattr(data.efc, name) for name in ("type", "id", "force")}
        )
        layouts = []
        for name, array in arrays.items():
            layouts.append(
                {
                    "name": name,
                    "object_id": id(array),
                    "ptr": int(array.ptr),
                    "shape": [int(value) for value in array.shape],
                    "strides": [int(value) for value in array.strides],
                    "dtype": str(array.dtype),
                    "device": str(array.device),
                    "contiguous": bool(array.is_contiguous),
                }
            )
        return tuple(layouts)

    def _guard(self):
        _need(
            self._status == "active"
            and self._owns_lock
            and self._fault is None
            and current_thread() is main_thread()
            and active_count() == 1,
            "active isolated one-shot BAM observer",
        )
        self._source_checks()
        if self._runtime is not None:
            _need(
                self._motor.__dict__.get("compute") is self._compute_wrapper
                and bam_bridge.BamActuator._dof_friction_force is self._dof_wrapper
                and FrictionDRBamActuator._compute_friction_budget
                is self._budget_wrapper
                and self._runtime.data is self._data
                and self._motor.actuator._data is self._bridge_data
                and self._bridge_data.struct is self._data
                and self._capture_raw_layouts(self._data) == self._raw_layouts
                and self._runtime.model is self._control._model
                and self._runtime.forward_graph is None,
                "owned hooks and unchanged bound runtime",
            )

    def _snapshot_data(self, staged):
        _need(
            type(staged) is FrictionDRBamActuator
            and staged._data is self._bridge_data
            and staged._num_envs == self._worlds
            and staged._dof_ids.shape == (JOINTS,)
            and staged._dof_ids.dtype is torch.int64
            and staged._dof_ids.device == self._device,
            "exact staged actuator and bound live data",
        )
        _need(
            torch.equal(staged._dof_ids, self._runtime.dofs)
            and torch.equal(staged._dof_ids, torch.arange(6, 20, device=self._device)),
            "literal ordered controlled DOF addresses",
        )
        packet = {}
        for name in ("qfrc_bias", "qfrc_constraint", "qfrc_actuator"):
            wrapped = getattr(self._bridge_data, name)
            raw = getattr(self._data, name)
            tensor = staged._as_tensor(wrapped)
            _need(
                wrapped.wp_array is raw and tensor.data_ptr() == int(raw.ptr),
                "actual shared BAM input storage " + name,
            )
            packet[name] = _tensor_bytes(
                tensor,
                (self._worlds, NV),
                torch.float32,
                self._device,
                name,
            )
        efc = self._bridge_data.efc
        for name in ("type", "id"):
            wrapped = getattr(efc, name)
            raw = getattr(self._data.efc, name)
            tensor = staged._as_tensor(wrapped)
            _need(
                wrapped.wp_array is raw and tensor.data_ptr() == int(raw.ptr),
                "actual shared BAM constraint storage " + name,
            )
            packet["efc_" + name] = _tensor_bytes(
                tensor,
                (self._worlds, EFC_ROWS),
                torch.int32,
                self._device,
                "efc_" + name,
            )
        force = staged._as_tensor(efc.force)
        counts = staged._as_tensor(self._bridge_data.nefc)
        _need(
            efc.force.wp_array is self._data.efc.force
            and force.data_ptr() == int(self._data.efc.force.ptr)
            and self._bridge_data.nefc.wp_array is self._data.nefc
            and counts.data_ptr() == int(self._data.nefc.ptr),
            "actual shared constraint force/count storage",
        )
        packet["efc_force"] = _tensor_bytes(
            force,
            (self._worlds, EFC_ROWS),
            torch.float32,
            self._device,
            "efc_force",
        )
        packet["nefc"] = _tensor_bytes(
            counts,
            (self._worlds,),
            torch.int32,
            self._device,
            "nefc",
        )
        return packet

    def _commit_compute_wrapper(self, motor, command, live):
        self._guard()
        _need(
            motor is self._motor
            and not self._inside_compute
            and len(self._entries) < CALLS
            and self._control._step_used
            and inspect.currentframe().f_back.f_code is _RUNTIME_STEP_CODE
            and self._original_compute.__self__ is self._motor
            and self._original_compute.__func__ is _COMMIT_COMPUTE
            and torch.is_tensor(live)
            and tuple(live.shape) == (self._worlds,)
            and live.dtype is torch.bool
            and live.device == self._device
            and bool(live.all()),
            "exact in-order live motor proposal within the pinned runtime step",
        )
        self._inside_compute = True
        try:
            step_before = self._runtime.steps.detach().cpu().tolist()
            _need(
                step_before == [len(self._entries)] * self._worlds,
                "exact ascending runtime substep counter before each proposal",
            )
            self._current = {"packet": {}, "staged_id": None}
            result = self._original_compute(command, live)
            _need(
                set(self._current)
                == {
                    "packet",
                    "qfric",
                    "budget",
                    "staged_id",
                    "staged_object",
                    "runtime_parameters",
                }
                and len(self._entries) < CALLS,
                "one friction scan and one friction budget per proposal",
            )
            packet = self._current["packet"]
            packet.update(self._current.pop("qfric"))
            packet.update(self._current.pop("budget"))
            record = packet
            record["staged_id"] = self._current["staged_id"]
            self._runtime_parameters.append(self._current.pop("runtime_parameters"))
            _need(
                set(record) == set(_ALL_FIELDS) | {"staged_id"},
                "complete named BAM load packet",
            )
            sizes = [len(record[name]) for name in _ALL_FIELDS]
            prior_bytes = sum(
                len(row[name]) for row in self._entries for name in _ALL_FIELDS
            )
            _need(
                all(size <= MAX_FIELD_BYTES for size in sizes)
                and prior_bytes + sum(sizes) <= MAX_TOTAL_BYTES,
                "full-call bytes fit declared caps",
            )
            self._entries.append(record)
            self._call_records.append(
                {
                    "proposal_index": len(self._entries) - 1,
                    "steps_before": step_before,
                    "runtime_data_id": id(self._data),
                    "bridge_data_id": id(self._bridge_data),
                    "staged_actuator_id": record["staged_id"],
                    "compute_calls": len(self._entries),
                    "friction_scan_calls": len(self._entries),
                    "budget_calls": len(self._entries),
                }
            )
            return result
        except BaseException as error:
            self._fault = self._fault or type(error).__name__
            raise
        finally:
            self._inside_compute = False
            self._current = None

    def _dof_friction_wrapper(self, staged, nv):
        self._guard()
        _need(
            self._inside_compute
            and self._current is not None
            and "qfric" not in self._current
            and nv == NV
            and staged._data is self._bridge_data,
            "one ordered actual friction scan inside each staged proposal",
        )
        staged_id = id(staged)
        snapshot = self._snapshot_data(staged)
        friction = _DOF_FRICTION(staged, nv)
        snapshot["qfrc_friction"] = _tensor_bytes(
            friction,
            (self._worlds, NV),
            torch.float32,
            self._device,
            "qfrc_friction",
        )
        self._current["qfric"] = snapshot
        self._current["staged_id"] = staged_id
        self._current["staged_object"] = staged
        return friction

    def _budget_call_wrapper(
        self, staged, motor_torque, external_torque, stribeck_coeff
    ):
        self._guard()
        _need(
            self._inside_compute
            and self._current is not None
            and "qfric" in self._current
            and "budget" not in self._current
            and id(staged) == self._current["staged_id"]
            and staged._data is self._bridge_data,
            "one ordered actual budget call on the same staged actuator",
        )
        runtime_parameters = self._validate_runtime_parameters(staged)
        args = {
            "budget_motor": _tensor_bytes(
                motor_torque,
                (self._worlds, JOINTS),
                torch.float32,
                self._device,
                "budget_motor",
            ),
            "budget_external": _tensor_bytes(
                external_torque,
                (self._worlds, JOINTS),
                torch.float32,
                self._device,
                "budget_external",
            ),
            "budget_stribeck": _tensor_bytes(
                stribeck_coeff,
                (self._worlds, JOINTS),
                torch.float32,
                self._device,
                "budget_stribeck",
            ),
            "friction_scale": _tensor_bytes(
                staged.friction_scale,
                (self._worlds, 1),
                torch.float32,
                self._device,
                "friction_scale",
            ),
        }
        output = _BUDGET(staged, motor_torque, external_torque, stribeck_coeff)
        args["budget_output"] = _tensor_bytes(
            output,
            (self._worlds, JOINTS),
            torch.float32,
            self._device,
            "budget_output",
        )
        _need(
            bool((staged.friction_scale == 1.0).all()),
            "nominal unit friction scale at every actual budget call",
        )
        self._current["budget"] = args
        self._current["runtime_parameters"] = runtime_parameters
        return output

    def _validate_runtime_parameters(self, staged):
        model = staged._bam_model
        _need(
            staged.cfg.model == "m6"
            and staged.cfg.motor_name == "xl330"
            and model.name == "m6"
            and model.actuator_name == "xl330"
            and all(
                getattr(model, name) is True
                for name in ("stribeck", "load_dependent", "directional", "quadratic")
            ),
            "exact loaded M6 XL330 parameter flags and labels",
        )
        names = (
            "kt",
            "R",
            "armature",
            "q_offset",
            "friction_base",
            "friction_stribeck",
            "load_friction_motor",
            "load_friction_external",
            "load_friction_motor_stribeck",
            "load_friction_external_stribeck",
            "load_friction_motor_quad",
            "load_friction_external_quad",
            "dtheta_stribeck",
            "alpha",
            "friction_viscous",
        )
        values = {}
        for name in names:
            raw_value = getattr(model, name)
            value = getattr(raw_value, "value", raw_value)
            _need(
                type(value) is float and value == self._m6_json[name],
                f"pinned M6 parameter {name}",
            )
            values[name] = value
        return values

    def _restore_owned(self):
        if (
            self._motor is not None
            and self._motor.__dict__.get("compute") is self._compute_wrapper
        ):
            del self._motor.__dict__["compute"]
        if bam_bridge.BamActuator._dof_friction_force is self._dof_wrapper:
            bam_bridge.BamActuator._dof_friction_force = _DOF_FRICTION
        if FrictionDRBamActuator._compute_friction_budget is self._budget_wrapper:
            FrictionDRBamActuator._compute_friction_budget = _BUDGET

    def _release(self):
        if self._owns_lock:
            self._owns_lock = False
            _LOCK.release()

    def __exit__(self, error_type, error, _traceback):
        foreign = self._runtime is not None and (
            self._motor.__dict__.get("compute") is not self._compute_wrapper
            or bam_bridge.BamActuator._dof_friction_force is not self._dof_wrapper
            or FrictionDRBamActuator._compute_friction_budget
            is not self._budget_wrapper
        )
        if error is not None:
            self._fault = self._fault or error_type.__name__
        if foreign:
            self._fault = self._fault or "ForeignHookReplacement"
        self._restore_owned()
        self._release()
        try:
            self._source_checks()
        except BaseException as source_error:
            self._fault = self._fault or type(source_error).__name__
        complete = (
            error is None
            and not foreign
            and self._fault is None
            and self._runtime is not None
            and self._control._step_used
            and self._control._completed_forwards == 21
            and self._control._event_state == "com"
            and self._control._fault is None
            and len(self._entries) == CALLS
            and len(self._call_records) == CALLS
            and len(self._runtime_parameters) == CALLS
            and all(
                set(row) == set(_ALL_FIELDS) | {"staged_id"} for row in self._entries
            )
        )
        self._status = "complete" if complete else "faulted"
        if not complete and error is None:
            raise ValueError("complete ten-call closed BAM load observation required")
        return False

    @property
    def packets(self):
        _need(self._status == "complete", "closed complete observer packets only")
        _need(
            all(set(row) == set(_ALL_FIELDS) | {"staged_id"} for row in self._entries),
            "exact complete packet field set for all ten calls",
        )
        result = {
            name: b"".join(row[name] for row in self._entries) for name in _ALL_FIELDS
        }
        _need(
            all(len(raw) <= MAX_FIELD_BYTES for raw in result.values())
            and sum(map(len, result.values())) <= MAX_TOTAL_BYTES,
            "all complete named packets within caps",
        )
        return result

    @property
    def receipt(self):
        return deepcopy(
            {
                "protocol": PROTOCOL,
                "status": self._status,
                "fault": self._fault,
                "calls": len(self._entries),
                "expected_calls": CALLS,
                "call_records": deepcopy(self._call_records),
                "worlds": self._worlds,
                "device": None if self._device is None else str(self._device),
                "runtime_id": None if self._runtime is None else id(self._runtime),
                "data_id": None if self._data is None else id(self._data),
                "bridge_data_id": None
                if self._bridge_data is None
                else id(self._bridge_data),
                "motor_id": None if self._motor is None else id(self._motor),
                "call_order": list(range(len(self._entries))),
                "raw_data_call_ids": [id(self._data)] * len(self._entries),
                "bridge_data_call_ids": [id(self._bridge_data)] * len(self._entries),
                "raw_array_layouts": self._raw_layouts,
                "staged_actuator_ids": [row.get("staged_id") for row in self._entries],
                "runtime_parameters": deepcopy(self._runtime_parameters),
                "runtime_parameters_checked_each_budget": len(self._runtime_parameters)
                == CALLS,
                "model_flags": {
                    "name": "m6",
                    "actuator": "xl330",
                    "stribeck": True,
                    "load_dependent": True,
                    "directional": True,
                    "quadratic": True,
                },
                "m6_json_sha256": _M6_SHA256,
                "m6_parameters": self._m6_json,
                "installed_bam_sha256": _BAM_SHA256,
                "adapter_sha256": _ADAPTER_SHA256,
                "state_commit_sha256": "7974953ded3c96e4a05ba05d15bfc82e12ea3b8255b150c8636ac07f580d26b9",
                "actual_bam_inputs_share_bound_raw_storage": True
                if self._status == "complete"
                else False,
                "source_pins": {
                    "bam.mjlab": _BAM_SHA256,
                    "stance_control_state.py": "7974953ded3c96e4a05ba05d15bfc82e12ea3b8255b150c8636ac07f580d26b9",
                    "friction_dr_bam.py": _ADAPTER_SHA256,
                },
                "method_pins": {
                    "BamStateCommit.compute": {
                        "object_id": id(_COMMIT_COMPUTE),
                        "code_name": _COMMIT_COMPUTE_CODE.co_qualname,
                    },
                    "BamActuator.compute": {
                        "object_id": id(_BAM_COMPUTE),
                        "code_name": _BAM_COMPUTE_CODE.co_qualname,
                    },
                    "BamActuator._dof_friction_force": {
                        "object_id": id(_DOF_FRICTION),
                        "code_name": _DOF_FRICTION_CODE.co_qualname,
                    },
                    "FrictionDRBamActuator._compute_friction_budget": {
                        "object_id": id(_BUDGET),
                        "code_name": _BUDGET_CODE.co_qualname,
                    },
                    "BamActuator._compute_friction_budget": {
                        "object_id": id(_BAM_BUDGET),
                        "code_name": _BAM_BUDGET_CODE.co_qualname,
                    },
                },
                "calls_sha256": {
                    name: [sha256(row[name]).hexdigest() for row in self._entries]
                    for name in _ALL_FIELDS
                    if self._entries and name in self._entries[0]
                },
                "field_lengths": {
                    name: len(self._entries[0][name]) if self._entries else None
                    for name in _ALL_FIELDS
                },
                "original_compute_called_once_per_proposal": self._status == "complete",
                "original_friction_scan_called_once_per_proposal": self._status
                == "complete",
                "original_budget_called_once_per_proposal": self._status == "complete",
                "runtime_cause_proven": False,
                "original_run_entry_captured": False,
                "native_qualified": False,
                "full_window_qualified": False,
                "training_authorized": False,
                "physical_acceptance": False,
            }
        )
