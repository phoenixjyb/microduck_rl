"""Passive solver initialization checkpoint capture over contact observation.

This observer records the state immediately after the frozen
``solver.init_context`` call returns.  It does not invoke a solver, search, or
iteration itself and does not change any historical contact-boundary packet.
"""

from hashlib import sha256
import inspect
from pathlib import Path
import sys
from threading import Lock

from mjlab_microduck import stance_contact_boundary_control as contact_control
from mjlab_microduck import stance_friction_runtime_control as friction_control


PROTOCOL = "microduck-solver-init-control-oct8-v1"
SOLVER_SOURCE_SHA256 = (
    "bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a"
)
SOLVER_WRAPPER_SOURCE_SHA256 = (
    "b0d156ddbced4848cd6cbcd70977f1e96c230fce0d746a6acfa911549afb447c"
)
FORWARD_SOURCE_SHA256 = (
    "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3"
)
SOLVER_RECIPE = {
    "solver": 2,
    "cone": 0,
    "disableflags": 0,
    "enableflags": 0,
    "iterations": 100,
    "ls_iterations": 50,
    "ls_parallel": False,
    "ls_parallel_min_step": 1.0e-6,
    "graph_conditional": True,
    "integrator": 0,
    "run_collision_detection": True,
    "nv": 20,
    "nv_pad": 20,
    "is_sparse": False,
    "nworld": 64,
    "njmax": 512,
    "njmax_nnz": 10240,
    "block_dim": {
        "segmented_sort": 128,
        "euler_dense": 32,
        "actuator_velocity": 32,
        "ray": 64,
        "contact_sort": 64,
        "energy_vel_kinetic": 32,
        "cholesky_factorize": 32,
        "cholesky_solve": 32,
        "cholesky_factorize_solve": 32,
        "solve_LD_sparse_fused": 64,
        "update_gradient_cholesky": 64,
        "update_gradient_cholesky_blocked": 32,
        "update_gradient_JTDAJ_sparse": 64,
        "update_gradient_JTDAJ_dense": 96,
        "linesearch_iterative": 32,
        "contact_jac_tiled": 32,
        "qderiv_actuator_dense": 32,
    },
}
SAMPLED_FORWARD = 4
MAX_PACKET_BYTES = 8 * 1024**2
MAX_INSTANCE_BYTES = 8 * 1024**2
MAX_ALL_ARMS_BYTES = 24 * 1024**2
FLAGS = dict(contact_control.FLAGS)
_TOTAL_CAPTURE_BYTES = 0
_CAPTURE_LOCK = Lock()
_HOOK_LOCK = Lock()

# Names are fixed in packet order.  Shapes are both logical Warp dimensions and
# host ndarray dimensions; matrix/contact layouts retain their native shapes.
SOLVER_INIT_ORDER = tuple(
    dict.fromkeys(
        (
            "model.opt.impratio_invsqrt",
            "model.opt.tolerance",
            "model.opt.ls_tolerance",
            "model.opt.timestep",
            "model.stat.meaninertia",
            "data.ne",
            "data.nf",
            "data.nl",
            "data.nefc",
            "data.solver_niter",
            "data.qpos",
            "data.qvel",
            "data.qacc",
            "data.qacc_warmstart",
            "data.qacc_smooth",
            "data.qM",
            "data.qfrc_smooth",
            "data.qfrc_constraint",
            *contact_control.CONTACT_PACKET_ORDER,
            *contact_control.COMPLETE_ORDER,
            "context.Jaref",
            "context.gauss",
            "context.search_dot",
            "context.cost",
            "context.prev_cost",
            "context.done",
            "context.grad",
            "context.grad_dot",
            "context.Mgrad",
            "context.h",
        )
    )
)


def _solver_init_specs():
    """Frozen CPU host layouts for the installed dense nv=20, 64-world ABI."""
    result = {}

    def add(name, shape, dtype):
        result[name] = (tuple(shape), tuple(shape), dtype)

    for name in ("impratio_invsqrt", "tolerance", "ls_tolerance"):
        add("model.opt." + name, (1,), "float32")
    add("model.opt.timestep", (1,), "float32")
    add("model.stat.meaninertia", (1,), "float32")
    for name in ("ne", "nf", "nl", "nefc", "solver_niter"):
        add("data." + name, (64,), "int32")
    add("data.qpos", (64, 21), "float32")
    for name in (
        "qvel",
        "qacc",
        "qacc_warmstart",
        "qacc_smooth",
        "qfrc_smooth",
        "qfrc_constraint",
    ):
        add("data." + name, (64, 20), "float32")
    add("data.qM", (64, 20, 20), "float32")
    # Existing contact protocol literals include model/contact/context payloads
    # and the complete EFC bank.  Reuse those specifications without editing it.
    result.update(
        {
            name: spec
            for name, spec in contact_control._SPECS.items()
            if name in SOLVER_INIT_ORDER
        }
    )
    add("context.Jaref", (64, 512), "float32")
    add("context.gauss", (64,), "float32")
    add("context.search_dot", (64,), "float32")
    add("context.cost", (64,), "float32")
    add("context.prev_cost", (64,), "float32")
    add("context.done", (64,), "bool")
    add("context.grad", (64, 20), "float32")
    add("context.grad_dot", (64,), "float32")
    add("context.Mgrad", (64, 20), "float32")
    add("context.h", (64, 20, 20), "float32")
    return result


SOLVER_INIT_SPECS = _solver_init_specs()
_SOLVER_INIT_SPECS = SOLVER_INIT_SPECS


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _solver_identity(array, device):
    """Exact layout identity, with zero stride only on singleton axes."""
    shape, strides = tuple(array.shape), tuple(array.strides)
    host = array.numpy()
    _need(
        len(shape) == len(strides)
        and all(type(value) is int and value >= 0 for value in shape + strides)
        and str(host.dtype) in ("float32", "int32", "bool")
        and host.shape[: len(shape)] == shape
        and array.device is device,
        "literal solver-init carrier layout and device",
    )
    # NumPy reports arbitrary zero strides for some zero-length dimensions.
    # Reconstruct Warp's canonical C strides over logical axes, with any
    # trailing vector/matrix dtype width included in each logical item.
    item_width = host.dtype.itemsize
    for suffix in host.shape[len(shape) :]:
        item_width *= suffix
    canonical = []
    for dimension in reversed(shape):
        canonical.append(item_width)
        item_width *= dimension
    expected_strides = tuple(reversed(canonical))
    _need(
        all(
            actual == expected or (actual == 0 and dim == 1)
            for actual, expected, dim in zip(strides, expected_strides, shape)
        ),
        "solver-init stride matches canonical host layout except singleton zero stride",
    )
    item = host.dtype.itemsize
    for suffix in host.shape[len(shape) :]:
        item *= suffix
    span = (
        0
        if not all(shape)
        else item + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
    )
    _need(
        span == host.nbytes
        and (
            span == 0
            or array.is_contiguous
            or any(stride == 0 and dim == 1 for stride, dim in zip(strides, shape))
        ),
        "contiguous solver-init storage with legal singleton broadcast",
    )
    pointer = array.ptr
    _need(
        (type(pointer) is int and pointer > 0) or (span == 0 and pointer in (None, 0)),
        "live solver-init carrier allocation",
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


def _layouts(arrays, device):
    _need(set(arrays) == set(SOLVER_INIT_ORDER), "exact solver-init packet array set")
    result = {}
    for name in SOLVER_INIT_ORDER:
        layout = _solver_identity(arrays[name], device)
        logical, host, dtype = _SOLVER_INIT_SPECS[name]
        _need(
            tuple(layout["shape"]) == logical
            and tuple(layout["host_shape"]) == host
            and layout["host_dtype"] == dtype,
            "literal solver-init shape and dtype " + name,
        )
        result[name] = layout
    ranges = sorted(
        (value["pointer"], value["pointer"] + value["span"], name)
        for name, value in result.items()
        if value["span"]
    )
    _need(
        all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])),
        "nonoverlapping solver-init allocations",
    )
    return result


def _pack(arrays):
    chunks, fields, offset = [], {}, 0
    for name in SOLVER_INIT_ORDER:
        host = arrays[name].numpy()
        dtype = host.dtype
        _need(str(dtype) in ("float32", "int32", "bool"), "raw solver-init dtype")
        wire_dtype = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[str(dtype)]
        raw = host.astype(wire_dtype, copy=False).tobytes(order="C")
        fields[name] = {
            "offset": offset,
            "bytes": len(raw),
            "shape": list(host.shape),
            "dtype": wire_dtype,
        }
        chunks.append(raw)
        offset += len(raw)
    return b"".join(chunks), fields


def _check_counts(observer, arrays):
    limits = {
        "data.ne": observer.data.njmax,
        "data.nf": observer.data.njmax,
        "data.nl": observer.data.njmax,
        "data.nefc": observer.data.njmax,
        "data.solver_niter": observer.model.opt.iterations,
        "contact.nacon": observer.data.naconmax,
        "local.efc_nnz": observer.data.njmax_nnz,
    }
    for name, limit in limits.items():
        values = arrays[name].numpy().reshape(-1)
        _need(
            all(type(limit) is int and 0 <= int(value) <= limit for value in values),
            "bounded initialized count " + name,
        )


def _arrays(observer, model, data, context):
    base = contact_control._contact_arrays(model, data, observer.current_contact_nnz)
    complete = contact_control._complete_arrays(data, observer.current_contact_nnz)
    return {
        "model.opt.impratio_invsqrt": model.opt.impratio_invsqrt,
        "model.opt.tolerance": model.opt.tolerance,
        "model.opt.ls_tolerance": model.opt.ls_tolerance,
        "model.opt.timestep": model.opt.timestep,
        "model.stat.meaninertia": model.stat.meaninertia,
        "data.ne": data.ne,
        "data.nf": data.nf,
        "data.nl": data.nl,
        "data.nefc": data.nefc,
        "data.solver_niter": data.solver_niter,
        "data.qpos": data.qpos,
        "data.qvel": data.qvel,
        "data.qacc": data.qacc,
        "data.qacc_warmstart": data.qacc_warmstart,
        "data.qacc_smooth": data.qacc_smooth,
        "data.qM": data.qM,
        "data.qfrc_smooth": data.qfrc_smooth,
        "data.qfrc_constraint": data.qfrc_constraint,
        **base,
        **complete,
        "context.Jaref": context.Jaref,
        "context.gauss": context.gauss,
        "context.search_dot": context.search_dot,
        "context.cost": context.cost,
        "context.prev_cost": context.prev_cost,
        "context.done": context.done,
        "context.grad": context.grad,
        "context.grad_dot": context.grad_dot,
        "context.Mgrad": context.Mgrad,
        "context.h": context.h,
    }


def _recipe_signature(model, data):
    opt = model.opt
    for name in ("ls_parallel", "graph_conditional", "run_collision_detection"):
        _need(type(getattr(opt, name)) is bool, "literal boolean solver option " + name)
    _need(type(model.is_sparse) is bool, "literal boolean sparse recipe")
    integer_values = {
        "solver": opt.solver,
        "cone": opt.cone,
        "disableflags": opt.disableflags,
        "enableflags": opt.enableflags,
        "iterations": opt.iterations,
        "ls_iterations": opt.ls_iterations,
        "integrator": opt.integrator,
        "nv": model.nv,
        "nv_pad": model.nv_pad,
        "nworld": data.nworld,
        "njmax": data.njmax,
        "njmax_nnz": data.njmax_nnz,
    }
    _need(
        all(
            isinstance(value, int) and type(value) is not bool
            for value in integer_values.values()
        ),
        "literal integer solver recipe settings",
    )
    _need(
        all(type(value) is int for value in vars(model.block_dim).values()),
        "literal integer solver block dimensions",
    )
    recipe = {
        "solver": int(opt.solver),
        "cone": int(opt.cone),
        "disableflags": int(opt.disableflags),
        "enableflags": int(opt.enableflags),
        "iterations": int(opt.iterations),
        "ls_iterations": int(opt.ls_iterations),
        "ls_parallel": opt.ls_parallel,
        "ls_parallel_min_step": opt.ls_parallel_min_step,
        "graph_conditional": opt.graph_conditional,
        "integrator": int(opt.integrator),
        "run_collision_detection": opt.run_collision_detection,
        "nv": int(model.nv),
        "nv_pad": int(model.nv_pad),
        "is_sparse": model.is_sparse,
        "nworld": int(data.nworld),
        "njmax": int(data.njmax),
        "njmax_nnz": int(data.njmax_nnz),
        "block_dim": dict(vars(model.block_dim)),
    }
    option_bytes = tuple(
        (
            name,
            str(array.numpy().dtype),
            tuple(array.shape),
            array.numpy().tobytes(order="C"),
        )
        for name, array in (
            ("impratio_invsqrt", opt.impratio_invsqrt),
            ("tolerance", opt.tolerance),
            ("ls_tolerance", opt.ls_tolerance),
            ("timestep", opt.timestep),
            ("meaninertia", model.stat.meaninertia),
        )
    )
    return recipe, option_bytes


class RuntimeSolverInitObserver(contact_control.RuntimeContactBoundaryObserver):
    """Observe the post-init, pre-search checkpoint for friction forward four."""

    def __init__(
        self, arm, *, contact_kernel, solver_module=None, forward_module=None, **kwargs
    ):
        super().__init__(arm, contact_kernel=contact_kernel, **kwargs)
        if solver_module is None:
            from mujoco_warp._src import solver as solver_module
        if forward_module is None:
            from mujoco_warp._src import forward as forward_module

        self.solver_module = solver_module
        self.forward_module = forward_module
        self.source_sha256 = None
        self.wrapper_source_sha256 = None
        self.forward_source_sha256 = None
        if solver_module.__name__ == "mujoco_warp._src.solver":
            source_path = Path(self.solver_module.__file__).resolve(strict=True)
            self.source_sha256 = sha256(source_path.read_bytes()).hexdigest()
            _need(
                self.source_sha256 == SOLVER_SOURCE_SHA256,
                "pinned installed solver.py source",
            )
        if forward_module.__name__ == "mujoco_warp._src.forward":
            forward_path = Path(forward_module.__file__).resolve(strict=True)
            self.forward_source_sha256 = sha256(forward_path.read_bytes()).hexdigest()
            _need(
                self.forward_source_sha256 == FORWARD_SOURCE_SHA256,
                "pinned installed forward.py source",
            )
        self.init_function = self.solver_module.init_context
        self.init_code = getattr(self.init_function, "__code__", None)
        self.solve_function = self.solver_module._solve
        self.solve_code = getattr(self.solve_function, "__code__", None)
        self.solve_public_function = self.solver_module.solve
        self.solve_public_code = getattr(self.solve_public_function, "__code__", None)
        self.solve_body_function = inspect.unwrap(self.solve_public_function)
        self.solve_body_code = getattr(self.solve_body_function, "__code__", None)
        self.solve_wrapper_globals = self.solve_public_function.__globals__
        self.solve_wrapper_module = sys.modules.get(
            self.solve_wrapper_globals.get("__name__")
        )
        self.forward_public_function = self.forward_module.forward
        self.forward_public_code = getattr(
            self.forward_public_function, "__code__", None
        )
        self.forward_body_function = inspect.unwrap(self.forward_public_function)
        self.forward_body_code = getattr(self.forward_body_function, "__code__", None)
        self.forward_wrapper_globals = self.forward_public_function.__globals__
        self.forward_wrapper_module = sys.modules.get(
            self.forward_wrapper_globals.get("__name__")
        )
        _need(
            self.init_code is not None
            and self.solve_code is not None
            and self.solve_public_code is not None
            and self.solve_body_code is not None
            and self.forward_public_code is not None
            and self.forward_body_code is not None,
            "solver source code objects",
        )
        _need(
            self.init_function.__globals__ is self.solver_module.__dict__
            and self.solve_function.__globals__ is self.solver_module.__dict__
            and self.solve_body_function.__globals__ is self.solver_module.__dict__
            and self.solve_wrapper_module is not None
            and self.solve_wrapper_globals is self.solve_wrapper_module.__dict__
            and self.forward_body_function.__globals__ is self.forward_module.__dict__
            and self.forward_wrapper_module is not None
            and self.forward_wrapper_globals is self.forward_wrapper_module.__dict__,
            "solver source module globals",
        )
        _need(
            sys.modules.get(self.solver_module.__name__) is self.solver_module
            and self.init_function.__module__ == self.solver_module.__name__
            and self.solve_function.__module__ == self.solver_module.__name__
            and self.solve_public_function.__module__ == self.solver_module.__name__
            and self.forward_body_function.__module__ == self.forward_module.__name__,
            "loaded solver source module identity",
        )
        self.init_wrapper = self._init_context
        self.solver_hook_active = False
        self.init_calls = []
        self.init_active = False
        self.solver_packet = None
        self.solver_bytes = 0
        self.solver_layouts = None
        self._recipe_binding = None
        self._solver_source_path = (
            Path(self.solver_module.__file__).resolve(strict=True)
            if self.source_sha256 is not None
            else None
        )
        self._wrapper_source_path = (
            Path(self.solve_wrapper_module.__file__).resolve(strict=True)
            if self.solve_wrapper_module.__name__ == "mujoco_warp._src.warp_util"
            else None
        )
        if self._wrapper_source_path is not None:
            self.wrapper_source_sha256 = sha256(
                self._wrapper_source_path.read_bytes()
            ).hexdigest()
            _need(
                self.wrapper_source_sha256 == SOLVER_WRAPPER_SOURCE_SHA256,
                "pinned installed event_scope wrapper source",
            )
        self._forward_source_path = (
            Path(self.forward_module.__file__).resolve(strict=True)
            if self.forward_source_sha256 is not None
            else None
        )

    def _check_solver_source(self):
        if self._solver_source_path is not None:
            current = sha256(self._solver_source_path.read_bytes()).hexdigest()
            _need(
                current == self.source_sha256 == SOLVER_SOURCE_SHA256,
                "solver.py unchanged around initialized capture",
            )
        if self._wrapper_source_path is not None:
            current = sha256(self._wrapper_source_path.read_bytes()).hexdigest()
            _need(
                current == self.wrapper_source_sha256 == SOLVER_WRAPPER_SOURCE_SHA256,
                "event_scope source unchanged around initialized capture",
            )
        if self._forward_source_path is not None:
            current = sha256(self._forward_source_path.read_bytes()).hexdigest()
            _need(
                current == self.forward_source_sha256 == FORWARD_SOURCE_SHA256,
                "forward.py unchanged around initialized capture",
            )

    def _make(self, model, data):
        result = super()._make(model, data)
        recipe, option_bytes = _recipe_signature(model, data)
        _need(recipe == SOLVER_RECIPE, "literal measured dense Newton2 solver recipe")
        binding = (recipe, option_bytes)
        if self._recipe_binding is None:
            self._recipe_binding = binding
        _need(
            binding == self._recipe_binding,
            "solver recipe and raw option bytes unchanged from first forward",
        )
        return result

    def _guard(self):
        super()._guard()
        _need(
            self.solver_module.init_context
            is (self.init_wrapper if self.solver_hook_active else self.init_function)
            and self.solver_module._solve is self.solve_function
            and self.solver_module._solve.__code__ is self.solve_code
            and self.solver_module.solve is self.solve_public_function
            and self.solver_module.solve.__code__ is self.solve_public_code
            and inspect.unwrap(self.solve_public_function) is self.solve_body_function
            and self.solve_body_function.__code__ is self.solve_body_code
            and self.solve_public_function.__globals__ is self.solve_wrapper_globals
            and sys.modules.get(self.solve_wrapper_module.__name__)
            is self.solve_wrapper_module
            and self.forward_module.forward is self.forward_public_function
            and self.forward_public_function.__code__ is self.forward_public_code
            and inspect.unwrap(self.forward_public_function)
            is self.forward_body_function
            and self.forward_body_function.__code__ is self.forward_body_code
            and self.forward_public_function.__globals__ is self.forward_wrapper_globals
            and self.forward_body_function.__globals__ is self.forward_module.__dict__
            and sys.modules.get(self.forward_module.__name__) is self.forward_module
            and sys.modules.get(self.forward_wrapper_module.__name__)
            is self.forward_wrapper_module
            and self.init_function.__code__ is self.init_code
            and self.init_function.__globals__ is self.solver_module.__dict__
            and self.solve_function.__globals__ is self.solver_module.__dict__
            and self.solve_body_function.__globals__ is self.solver_module.__dict__
            and sys.modules.get(self.solver_module.__name__) is self.solver_module,
            "held solver init and solve function identities",
        )

    def __enter__(self):
        _need(_HOOK_LOCK.acquire(blocking=False), "exclusive solver init hook")
        try:
            super().__enter__()
            _need(
                self.solver_module.init_context is self.init_function,
                "solver init hook remains unowned before installation",
            )
            self.solver_module.init_context = self.init_wrapper
            self.solver_hook_active = True
            self._guard()
            return self
        except BaseException:
            if self.solver_module.init_context is self.init_wrapper:
                self.solver_module.init_context = self.init_function
            self.solver_hook_active = False
            if self.active:
                super().__exit__(*sys.exc_info())
            _HOOK_LOCK.release()
            raise

    def __exit__(self, kind, value, traceback):
        foreign = self.solver_module.init_context is not self.init_wrapper
        if not foreign:
            self.solver_module.init_context = self.init_function
            self.solver_hook_active = False
        try:
            super().__exit__(kind, value, traceback)
        finally:
            _HOOK_LOCK.release()
        _need(not foreign, "foreign solver init hook not overwritten during close")
        if kind is None:
            _need(
                len(self.init_calls) == friction_control.FORWARDS,
                "exact21 solver init calls",
            )
            _need(self.solver_packet is not None, "one initialized solver packet")

    def _init_context(self, model, data, context, grad=True):
        self._guard()
        self._check_solver_source()
        index = len(self.init_calls)
        _need(
            not self.init_active and index < friction_control.FORWARDS,
            "nonreentrant bounded init call",
        )
        _need(
            type(grad) is bool and grad is True,
            "literal forward init_context grad=True",
        )
        _need(
            model is self.model
            and data is self.data
            and type(context) is self.solver_module.SolverContext,
            "exact forward model/data and SolverContext identities",
        )
        _need(
            len(self.entries) == index + 1
            and self.entries[index]["index"] == index
            and self.entries[index]["model_object_id"] == id(model)
            and self.entries[index]["data_object_id"] == id(data),
            "init call schedule follows completed friction forward",
        )
        # The immediate caller must be the frozen private solve routine.  Its
        # caller chain then proves this came through the public solve and
        # forward event_scope wrappers, rather than a direct init_context call.
        solve_frame = sys._getframe(1)
        public_frame = solve_frame.f_back
        public_wrapper_frame = public_frame.f_back if public_frame is not None else None
        forward_frame = (
            public_wrapper_frame.f_back if public_wrapper_frame is not None else None
        )
        forward_wrapper_frame = (
            forward_frame.f_back if forward_frame is not None else None
        )
        try:
            _need(
                solve_frame is not None
                and solve_frame.f_code is self.solve_code
                and solve_frame.f_globals is self.solver_module.__dict__
                and solve_frame.f_locals.get("m") is model
                and solve_frame.f_locals.get("d") is data
                and solve_frame.f_locals.get("ctx") is context
                and public_frame is not None
                and public_frame.f_code is self.solve_body_code
                and public_frame.f_globals is self.solver_module.__dict__
                and public_wrapper_frame is not None
                and public_wrapper_frame.f_code is self.solve_public_code
                and public_wrapper_frame.f_globals is self.solve_wrapper_globals
                and forward_frame is not None
                and forward_frame.f_code is self.forward_body_code
                and forward_frame.f_globals is self.forward_module.__dict__
                and forward_frame.f_locals.get("m") is model
                and forward_frame.f_locals.get("d") is data
                and forward_wrapper_frame is not None
                and forward_wrapper_frame.f_code is self.forward_public_code
                and forward_wrapper_frame.f_globals is self.forward_wrapper_globals,
                "init_context caller is held solver._solve within public solver and forward",
            )
        finally:
            # Break the parent/child frame cycle before any initialized arrays
            # are retained or the original solver function is called.
            solve_frame = public_frame = public_wrapper_frame = None
            forward_frame = forward_wrapper_frame = None
        _need(
            _recipe_signature(model, data) == self._recipe_binding,
            "solver recipe and raw option bytes unchanged before init_context",
        )
        call = {
            "forward": index,
            "arm": self.arm,
            "grad": True,
            "model_object_id": id(model),
            "data_object_id": id(data),
            "context_object_id": id(context),
            "init_function_id": id(self.init_function),
            "init_code_id": id(self.init_code),
            "solve_function_id": id(self.solve_function),
            "solve_code_id": id(self.solve_code),
            "solve_public_function_id": id(self.solve_public_function),
            "solve_public_code_id": id(self.solve_public_code),
            "solve_body_code_id": id(self.solve_body_code),
            "forward_function_id": id(self.forward_public_function),
            "forward_code_id": id(self.forward_public_code),
            "forward_body_code_id": id(self.forward_body_code),
            "stream": self.stream.cuda_stream,
            "device_context": self.device.context,
            "phase": "pre-search"
            if index == SAMPLED_FORWARD
            else "unsampled-init-context",
        }
        self.init_active = True
        try:
            result = self.init_function(model, data, context, grad=True)
            self._check_solver_source()
            self._guard()
            _need(
                _recipe_signature(model, data) == self._recipe_binding,
                "solver recipe and raw option bytes unchanged at initialized boundary",
            )
            if index == SAMPLED_FORWARD:
                self.wp.synchronize_stream(self.stream)
                self._guard()
                arrays = _arrays(self, model, data, context)
                layouts = _layouts(arrays, self.device)
                if self.solver_layouts is None:
                    self.solver_layouts = layouts
                _need(
                    layouts == self.solver_layouts,
                    "stable initialized solver carrier layouts",
                )
                _check_counts(self, arrays)
                raw, fields = _pack(arrays)
                self._check_solver_source()
                name = f"solver-init/{self.arm}/forward-{index:02d}.initialized.bin"
                self.solver_packet = self._emit_solver(name, raw)
                self.solver_packet.update(
                    fields=fields,
                    layouts=layouts,
                    forward=index,
                    phase="initialized-before-search",
                    context_object_id=id(context),
                )
                self._check_solver_source()
            self.init_calls.append(call)
            return result
        finally:
            self.init_active = False

    def _emit_solver(self, name, raw):
        global _TOTAL_CAPTURE_BYTES
        _need(type(raw) is bytes, "immutable solver-init packet bytes")
        _need(len(raw) <= MAX_PACKET_BYTES, "solver-init packet cap")
        _need(self.solver_bytes + len(raw) <= MAX_INSTANCE_BYTES, "solver-init arm cap")
        with _CAPTURE_LOCK:
            _need(
                _TOTAL_CAPTURE_BYTES + len(raw) <= MAX_ALL_ARMS_BYTES,
                "solver-init all-arms cap",
            )
            self.sink(name, raw)
            _TOTAL_CAPTURE_BYTES += len(raw)
        self.solver_bytes += len(raw)
        return {"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    def solver_receipt(self):
        _need(
            not self.active and len(self.init_calls) == friction_control.FORWARDS,
            "closed complete solver-init observer",
        )
        _need(
            self.solver_packet is not None, "one successful forward-four solver packet"
        )
        recipe, _ = self._recipe_binding
        return {
            "protocol": PROTOCOL,
            "arm": self.arm,
            "calls": self.init_calls,
            "snapshot": self.solver_packet,
            "packet_count": 1,
            "captured_bytes": self.solver_bytes,
            "source_identity": {
                "init_function_id": id(self.init_function),
                "init_code_id": id(self.init_code),
                "solve_function_id": id(self.solve_function),
                "solve_code_id": id(self.solve_code),
                "solve_public_function_id": id(self.solve_public_function),
                "solve_public_code_id": id(self.solve_public_code),
                "solve_body_code_id": id(self.solve_body_code),
                "forward_function_id": id(self.forward_public_function),
                "forward_code_id": id(self.forward_public_code),
                "forward_body_code_id": id(self.forward_body_code),
                "solver_source_sha256": self.source_sha256,
                "solver_wrapper_source_sha256": self.wrapper_source_sha256,
                "forward_source_sha256": self.forward_source_sha256,
            },
            "recipe": recipe,
            "flags": dict(FLAGS),
        }
