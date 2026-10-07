"""Passive construction-time contact-boundary capture over runtime friction control.

The observer records exact CPU/GPU-array byte packets around contact allocation
and at constraint-construction completion. It never runs the solver or admits
runtime, training, or physical behavior.
"""

from hashlib import sha256
import inspect
import sys
from threading import Lock

from mjlab_microduck import stance_friction_runtime_control as friction_control


PROTOCOL = "microduck-contact-boundary-control-oct8-v1"
SAMPLED_FORWARDS = 7
CONTACT_CAPACITY = 8192
NJMAX = 512
NJMAX_NNZ = 10240
EFC_BYTES = 4_068_352
MAX_PACKET_BYTES = 16 * 1024**2
MAX_INSTANCE_BYTES = 80 * 1024**2
MAX_ALL_ARMS_BYTES = 240 * 1024**2
CONTEXT_FIELDS = ("context.geomcollisionid", "context.pos", "context.frame")
FLAGS = {
    name: False
    for name in (
        "runtime_cause_proven",
        "native_qualified",
        "full_window_qualified",
        "training_authorized",
        "physical_acceptance",
    )
}
CONTACT_INPUT_ORDER = (
    "model.body_weldid",
    "model.body_dofnum",
    "model.body_dofadr",
    "model.dof_parentid",
    "model.geom_bodyid",
    "model.flex_vertadr",
    "model.flex_vertbodyid",
    "contact.nacon",
    "contact.dist",
    "contact.dim",
    "contact.includemargin",
    "contact.worldid",
    "contact.geom",
    "contact.flex",
    "contact.vert",
    "contact.type",
)
CONTACT_OUTPUT_ORDER = (
    "data.nefc",
    "contact.efc_address",
    "efc.id",
    "efc.J_rownnz",
    "efc.J_rowadr",
    "local.efc_nnz",
)
CONTACT_ARRAY_ORDER = CONTACT_INPUT_ORDER + CONTACT_OUTPUT_ORDER
CONTACT_PACKET_ORDER = CONTACT_ARRAY_ORDER + CONTEXT_FIELDS
COMPLETE_ORDER = (
    "data.ne",
    "data.nf",
    "data.nl",
    "data.nefc",
    "data.qpos",
    "data.qvel",
    "data.qacc",
    "data.qacc_warmstart",
    "data.ctrl",
    "efc.type",
    "efc.id",
    "efc.J_rownnz",
    "efc.J_rowadr",
    "efc.J_colind",
    "efc.J",
    "efc.pos",
    "efc.margin",
    "efc.D",
    "efc.vel",
    "efc.aref",
    "efc.frictionloss",
    "efc.force",
    "efc.state",
    "efc.Ma",
    "efc.Jqvel",
    "local.efc_nnz",
)
EFC_FIELDS = (
    "efc.type",
    "efc.id",
    "efc.J_rownnz",
    "efc.J_rowadr",
    "efc.J_colind",
    "efc.J",
    "efc.pos",
    "efc.margin",
    "efc.D",
    "efc.vel",
    "efc.aref",
    "efc.frictionloss",
    "efc.force",
    "efc.state",
    "efc.Ma",
    "efc.Jqvel",
)
STALE_PRIOR_FIELDS = ("efc.force", "efc.state", "efc.Ma", "efc.Jqvel")
_TOTAL_CAPTURE_BYTES = 0
_CAPTURE_LOCK = Lock()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _array_specs():
    """Literal CPU-proven host/logical layouts for this retained plant."""
    result = {}

    def add(name, logical, host, dtype):
        result[name] = (tuple(logical), tuple(host), dtype)

    for name in ("body_weldid", "body_dofnum", "body_dofadr"):
        add("model." + name, (16,), (16,), "int32")
    add("model.dof_parentid", (20,), (20,), "int32")
    add("model.geom_bodyid", (82,), (82,), "int32")
    add("model.flex_vertadr", (0,), (0,), "int32")
    add("model.flex_vertbodyid", (0,), (0,), "int32")

    add("contact.nacon", (1,), (1,), "int32")
    for name, dtype in (
        ("dist", "float32"),
        ("dim", "int32"),
        ("includemargin", "float32"),
        ("worldid", "int32"),
        ("type", "int32"),
    ):
        add("contact." + name, (CONTACT_CAPACITY,), (CONTACT_CAPACITY,), dtype)
    for name in ("geom", "flex", "vert"):
        add(
            "contact." + name,
            (CONTACT_CAPACITY,),
            (CONTACT_CAPACITY, 2),
            "int32",
        )
    add(
        "contact.efc_address",
        (CONTACT_CAPACITY, 4),
        (CONTACT_CAPACITY, 4),
        "int32",
    )
    add("contact.pos", (CONTACT_CAPACITY,), (CONTACT_CAPACITY, 3), "float32")
    add(
        "contact.frame",
        (CONTACT_CAPACITY, 3),
        (CONTACT_CAPACITY, 3, 3),
        "float32",
    )
    add("context.geomcollisionid", (CONTACT_CAPACITY,), (CONTACT_CAPACITY,), "int32")
    add("context.pos", (CONTACT_CAPACITY,), (CONTACT_CAPACITY, 3), "float32")
    add("context.frame", (CONTACT_CAPACITY, 3), (CONTACT_CAPACITY, 3, 3), "float32")

    for name in ("ne", "nf", "nl", "nefc"):
        add("data." + name, (64,), (64,), "int32")
    add("data.qpos", (64, 21), (64, 21), "float32")
    for name in ("qvel", "qacc", "qacc_warmstart"):
        add("data." + name, (64, 20), (64, 20), "float32")
    add("data.ctrl", (64, 14), (64, 14), "float32")

    for name in ("type", "id", "state"):
        add("efc." + name, (64, 512), (64, 512), "int32")
    for name in ("J_rownnz", "J_rowadr"):
        add("efc." + name, (64, 0), (64, 0), "int32")
    add("efc.J_colind", (64, 0, 0), (64, 0, 0), "int32")
    add("efc.J", (64, 512, 20), (64, 512, 20), "float32")
    for name in ("pos", "margin", "D", "vel", "aref", "frictionloss", "force"):
        add("efc." + name, (64, 512), (64, 512), "float32")
    add("efc.Ma", (64, 20), (64, 20), "float32")
    add("efc.Jqvel", (64, 512), (64, 512), "float32")
    add("local.efc_nnz", (64,), (64,), "int32")
    return result


_SPECS = _array_specs()


def _layouts(arrays, device, names, label):
    _need(set(arrays) == set(names), "exact array set " + label)
    result = {}
    for name in names:
        layout = friction_control.identity(arrays[name], device)
        logical, host, dtype = _SPECS[name]
        _need(
            tuple(layout["shape"]) == logical
            and tuple(layout["host_shape"]) == host
            and layout["host_dtype"] == dtype,
            "literal array shape and dtype " + label + "." + name,
        )
        result[name] = layout
    return result


def _nonalias(layouts, label):
    ranges = sorted(
        (value["pointer"], value["pointer"] + value["span"], name)
        for name, value in layouts.items()
        if value["span"]
    )
    _need(
        all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])),
        "nonoverlapping actual allocations " + label,
    )


def _contact_arrays(model, data, local_nnz):
    contact, efc = data.contact, data.efc
    return {
        "model.body_weldid": model.body_weldid,
        "model.body_dofnum": model.body_dofnum,
        "model.body_dofadr": model.body_dofadr,
        "model.dof_parentid": model.dof_parentid,
        "model.geom_bodyid": model.geom_bodyid,
        "model.flex_vertadr": model.flex_vertadr,
        "model.flex_vertbodyid": model.flex_vertbodyid,
        "contact.nacon": data.nacon,
        "contact.dist": contact.dist,
        "contact.dim": contact.dim,
        "contact.includemargin": contact.includemargin,
        "contact.worldid": contact.worldid,
        "contact.geom": contact.geom,
        "contact.flex": contact.flex,
        "contact.vert": contact.vert,
        "contact.type": contact.type,
        "data.nefc": data.nefc,
        "contact.efc_address": contact.efc_address,
        "efc.id": efc.id,
        "efc.J_rownnz": efc.J_rownnz,
        "efc.J_rowadr": efc.J_rowadr,
        "local.efc_nnz": local_nnz,
        "context.geomcollisionid": contact.geomcollisionid,
        "context.pos": contact.pos,
        "context.frame": contact.frame,
    }


def _complete_arrays(data, local_nnz):
    efc = data.efc
    return {
        "data.ne": data.ne,
        "data.nf": data.nf,
        "data.nl": data.nl,
        "data.nefc": data.nefc,
        "data.qpos": data.qpos,
        "data.qvel": data.qvel,
        "data.qacc": data.qacc,
        "data.qacc_warmstart": data.qacc_warmstart,
        "data.ctrl": data.ctrl,
        "efc.type": efc.type,
        "efc.id": efc.id,
        "efc.J_rownnz": efc.J_rownnz,
        "efc.J_rowadr": efc.J_rowadr,
        "efc.J_colind": efc.J_colind,
        "efc.J": efc.J,
        "efc.pos": efc.pos,
        "efc.margin": efc.margin,
        "efc.D": efc.D,
        "efc.vel": efc.vel,
        "efc.aref": efc.aref,
        "efc.frictionloss": efc.frictionloss,
        "efc.force": efc.force,
        "efc.state": efc.state,
        "efc.Ma": efc.Ma,
        "efc.Jqvel": efc.Jqvel,
        "local.efc_nnz": local_nnz,
    }


class RuntimeContactBoundaryObserver(friction_control.RuntimeFrictionObserver):
    """Add first-seven-forward contact and pre-solver boundary packets."""

    def __init__(self, arm, *, contact_kernel, **kwargs):
        super().__init__(arm, **kwargs)
        self.contact_factory = getattr(self.constraint, "_efc_contact_init", None)
        _need(callable(self.contact_factory), "callable frozen contact factory")
        self.contact_factory_func = inspect.unwrap(self.contact_factory)
        self.contact_factory_code = getattr(self.contact_factory_func, "__code__", None)
        _need(self.contact_factory_code is not None, "unwrapped contact factory code")
        self.contact_kernel = contact_kernel
        self.contact_warp_module = getattr(contact_kernel, "module", None)
        _need(
            self.contact_warp_module is not None,
            "cached contact Warp kernel module object",
        )
        self.contact_function = getattr(contact_kernel, "func", None)
        _need(callable(self.contact_function), "passed cached contact kernel function")
        self.contact_code = getattr(self.contact_function, "__code__", None)
        _need(self.contact_code is not None, "passed cached contact kernel code")
        self.contact_module_name = getattr(self.contact_function, "__module__", None)
        self.contact_module = sys.modules.get(self.contact_module_name)
        _need(
            self.contact_module is not None
            and self.contact_function.__globals__ is self.contact_module.__dict__,
            "loaded contact kernel module object",
        )
        self.boundary_entries = []
        self.boundary_bytes = 0
        self.bound_contact_layouts = None
        self.bound_complete_layouts = None
        self.current_contact_targets = 0
        self.current_contact_nnz = None
        self.current_contact_layouts = None
        self.current_friction_nnz = None
        self.current_contact_call = None

    def _guard(self):
        super()._guard()
        _need(
            getattr(self.constraint, "_efc_contact_init", None) is self.contact_factory
            and inspect.unwrap(self.contact_factory).__code__
            is self.contact_factory_code
            and self.contact_kernel.func is self.contact_function
            and self.contact_function.__code__ is self.contact_code,
            "held cached contact factory and kernel code",
        )
        _need(
            self.contact_kernel.module is self.contact_warp_module,
            "held cached contact Warp kernel module",
        )
        _need(
            self.contact_function.__module__ == self.contact_module_name
            and sys.modules.get(self.contact_module_name) is self.contact_module
            and self.contact_function.__globals__ is self.contact_module.__dict__,
            "held contact kernel module identity",
        )

    def _make(self, model, data):
        cone = model.opt.cone
        _need(
            type(cone) is int and cone == 0 and model.is_sparse is False,
            "plain dense pyramidal cone-zero contact model",
        )
        _need(
            type(data.naconmax) is int
            and data.naconmax == CONTACT_CAPACITY
            and type(data.naccdmax) is int
            and data.naccdmax == CONTACT_CAPACITY,
            "literal full contact and CCD capacities",
        )
        _need(
            self.contact_factory(cone, False) is self.contact_kernel,
            "exact cached cone-zero dense contact kernel",
        )
        forward = len(self.entries)
        self.current_contact_targets = 0
        self.current_contact_nnz = None
        self.current_contact_layouts = None
        self.current_friction_nnz = None
        self.current_contact_call = None
        result = super()._make(model, data)
        self._guard()
        if forward < SAMPLED_FORWARDS:
            _need(
                self.current_contact_targets == 1
                and self.current_contact_nnz is not None
                and self.current_contact_call is not None,
                "exactly one sampled contact allocation target",
            )
            self.wp.synchronize_stream(self.stream)
            self._guard()
            arrays = _complete_arrays(data, self.current_contact_nnz)
            layouts = _layouts(arrays, self.device, COMPLETE_ORDER, "complete")
            _nonalias(layouts, "construction-complete bank")
            stable_complete = {
                name: layout
                for name, layout in layouts.items()
                if name != "local.efc_nnz"
            }
            _need(
                self.bound_complete_layouts is None
                or stable_complete == self.bound_complete_layouts,
                "stable sampled construction-complete layouts",
            )
            self.bound_complete_layouts = stable_complete
            efc_bytes = sum(layouts[name]["bytes"] for name in EFC_FIELDS)
            _need(efc_bytes == EFC_BYTES, "complete EFC bank byte length")
            friction_layout = self.entries[-1]["layouts"]["bank.efc_nnz"]
            _need(
                layouts["local.efc_nnz"] == friction_layout
                and layouts["local.efc_nnz"]
                == self.current_contact_layouts["local.efc_nnz"],
                "same per-forward ephemeral NNZ allocation across contact and friction",
            )
            raw, fields = friction_control.pack(arrays, COMPLETE_ORDER)
            name = f"boundary/{self.arm}/forward-{forward:02d}.complete.bin"
            packet = self._emit(name, raw)
            self.boundary_entries.append(
                {
                    "forward": forward,
                    "phase": "construction-complete-BEFORE-solver",
                    "contact_call": self.current_contact_call,
                    "contact": self.current_contact_layouts,
                    "complete_layouts": layouts,
                    "contact_before": self._current_contact_before,
                    "contact_after": self._current_contact_after,
                    "complete": packet,
                    "complete_fields": fields,
                    "context_fields": list(CONTEXT_FIELDS),
                    "stale_prior_fields": list(STALE_PRIOR_FIELDS),
                }
            )
        return result

    def _launch(self, kernel, *args, **kwargs):
        if kernel is self.original:
            if self.current_forward < SAMPLED_FORWARDS:
                _need(
                    type(kwargs.get("outputs")) is list
                    and len(kwargs["outputs"]) == 15
                    and self.current_friction_nnz is None,
                    "one sampled friction launch with literal output list",
                )
                self.current_friction_nnz = kwargs["outputs"][14]
            result = super()._launch(kernel, *args, **kwargs)
            if self.current_forward < SAMPLED_FORWARDS:
                _need(
                    self.entries[-1]["layouts"]["bank.efc_nnz"]
                    == friction_control.identity(
                        self.current_friction_nnz, self.device
                    ),
                    "held sampled friction ephemeral NNZ metadata",
                )
            return result
        if (
            kernel is not self.contact_kernel
            or self.current_forward >= SAMPLED_FORWARDS
        ):
            return super()._launch(kernel, *args, **kwargs)

        self._guard()
        _need(
            not args and set(kwargs) == {"dim", "inputs", "outputs"},
            "exact contact launch keyword ABI",
        )
        inputs, outputs = kwargs["inputs"], kwargs["outputs"]
        _need(
            type(inputs) is list
            and len(inputs) == 18
            and type(outputs) is list
            and len(outputs) == 6,
            "exact18 contact inputs and six outputs",
        )
        _need(
            type(kwargs["dim"]) is int
            and kwargs["dim"] == CONTACT_CAPACITY
            and type(inputs[7]) is int
            and inputs[7] == NJMAX
            and type(inputs[8]) is int
            and inputs[8] == NJMAX_NNZ
            and inputs[9] is self.data.nacon
            and self.current_friction_nnz is not None
            and self.current_contact_targets == 0,
            "friction-first contact dimensions, capacities and count object",
        )
        _need(
            all(
                actual is expected
                for actual, expected in zip(
                    inputs[:7],
                    (
                        self.model.body_weldid,
                        self.model.body_dofnum,
                        self.model.body_dofadr,
                        self.model.dof_parentid,
                        self.model.geom_bodyid,
                        self.model.flex_vertadr,
                        self.model.flex_vertbodyid,
                    ),
                )
            )
            and all(
                actual is expected
                for actual, expected in zip(
                    inputs[10:],
                    (
                        self.data.contact.dist,
                        self.data.contact.dim,
                        self.data.contact.includemargin,
                        self.data.contact.worldid,
                        self.data.contact.geom,
                        self.data.contact.flex,
                        self.data.contact.vert,
                        self.data.contact.type,
                    ),
                )
            ),
            "actual model and contact input objects",
        )
        efc = self.data.efc
        _need(
            all(
                actual is expected
                for actual, expected in zip(
                    outputs,
                    (
                        self.data.nefc,
                        self.data.contact.efc_address,
                        efc.id,
                        efc.J_rownnz,
                        efc.J_rowadr,
                        outputs[5],
                    ),
                )
            ),
            "actual contact output banks",
        )
        _need(
            outputs[5] is self.current_friction_nnz
            and self.entries[-1]["layouts"]["bank.efc_nnz"]
            == friction_control.identity(outputs[5], self.device),
            "contact reuses exact friction ephemeral NNZ object and metadata",
        )
        self.current_contact_call = {
            "kernel_key": self.contact_kernel.key,
            "kernel_object_id": id(self.contact_kernel),
            "kernel_function_id": id(self.contact_function),
            "kernel_code_id": id(self.contact_code),
            "module_object_id": id(self.contact_warp_module),
            "factory_object_id": id(self.contact_factory),
            "factory_function_id": id(self.contact_factory_func),
            "factory_code_id": id(self.contact_factory_code),
            "cone": self.model.opt.cone,
            "is_sparse": self.model.is_sparse,
            "dim": kwargs["dim"],
            "njmax": inputs[7],
            "njmax_nnz": inputs[8],
            "input_count": len(inputs),
            "output_count": len(outputs),
        }
        self.current_contact_nnz = outputs[5]
        arrays = _contact_arrays(self.model, self.data, self.current_contact_nnz)
        layouts_before = _layouts(
            arrays, self.device, CONTACT_PACKET_ORDER, "contact-before"
        )
        _nonalias(layouts_before, "contact allocation call")
        _need(
            all(arrays[name].is_contiguous for name in CONTACT_OUTPUT_ORDER),
            "contiguous written contact allocation outputs",
        )
        _need(
            layouts_before["contact.efc_address"]["host_shape"]
            == [CONTACT_CAPACITY, 4],
            "full contact address capacity",
        )
        stable = {
            name: layout
            for name, layout in layouts_before.items()
            if name != "local.efc_nnz"
        }
        if self.bound_contact_layouts is not None:
            _need(
                stable == self.bound_contact_layouts,
                "stable sampled contact allocation layouts",
            )
        self.bound_contact_layouts = stable
        self.current_contact_layouts = layouts_before
        before_inputs, _ = friction_control.pack(arrays, CONTACT_INPUT_ORDER)
        before_context, _ = friction_control.pack(arrays, CONTEXT_FIELDS)
        raw_before, fields_before = friction_control.pack(arrays, CONTACT_PACKET_ORDER)
        index = self.current_forward
        before_name = f"boundary/{self.arm}/forward-{index:02d}.contact.before.bin"
        before_packet = self._emit(before_name, raw_before)
        result = super()._launch(kernel, *args, **kwargs)
        self.wp.synchronize_stream(self.stream)
        self._guard()
        layouts_after = _layouts(
            arrays, self.device, CONTACT_PACKET_ORDER, "contact-after"
        )
        _need(
            layouts_after == layouts_before,
            "contact call allocation identity unchanged",
        )
        after_inputs, _ = friction_control.pack(arrays, CONTACT_INPUT_ORDER)
        after_context, _ = friction_control.pack(arrays, CONTEXT_FIELDS)
        _need(before_inputs == after_inputs, "contact inputs unchanged by allocation")
        _need(
            before_context == after_context,
            "contact context fields unchanged by allocation",
        )
        raw_after, fields_after = friction_control.pack(arrays, CONTACT_PACKET_ORDER)
        after_name = f"boundary/{self.arm}/forward-{index:02d}.contact.after.bin"
        after_packet = self._emit(after_name, raw_after)
        context_fields = list(CONTEXT_FIELDS)
        before_packet.update(fields=fields_before, context_fields=context_fields)
        after_packet.update(fields=fields_after, context_fields=context_fields)
        self._current_contact_before = before_packet
        self._current_contact_after = after_packet
        self.current_contact_targets += 1
        return result

    def _emit(self, name, raw):
        global _TOTAL_CAPTURE_BYTES
        _need(type(raw) is bytes, "immutable boundary packet bytes")
        _need(len(raw) <= MAX_PACKET_BYTES, "bounded boundary packet leaf")
        _need(
            self.boundary_bytes + len(raw) <= MAX_INSTANCE_BYTES,
            "bounded boundary bytes per arm",
        )
        with _CAPTURE_LOCK:
            _need(
                _TOTAL_CAPTURE_BYTES + len(raw) <= MAX_ALL_ARMS_BYTES,
                "bounded boundary bytes across all arms",
            )
            self.sink(name, raw)
            _TOTAL_CAPTURE_BYTES += len(raw)
        self.boundary_bytes += len(raw)
        return {
            "path": name,
            "bytes": len(raw),
            "sha256": sha256(raw).hexdigest(),
        }

    def boundary_receipt(self):
        _need(
            not self.active and len(self.entries) == friction_control.FORWARDS,
            "closed complete friction observer before boundary receipt",
        )
        _need(
            len(self.boundary_entries) == SAMPLED_FORWARDS,
            "exact first-seven contact boundary entries",
        )
        return {
            "protocol": PROTOCOL,
            "arm": self.arm,
            "entries": self.boundary_entries,
            "packet_count": SAMPLED_FORWARDS * 3,
            "captured_bytes": self.boundary_bytes,
            "flags": dict(FLAGS),
        }
