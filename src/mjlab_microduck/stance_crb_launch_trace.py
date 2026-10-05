"""Scoped CRB accumulation launch snapshots, never a native-order proof.

The observer wraps only the process-local ``warp.launch`` reference while an
owned first-six forward invocation is running.  It recognizes the pinned
MuJoCo Warp kernel by identity and retains before/after CPU snapshots around
each actual compiled body-tree level.  Atomic order within a launch remains
unobserved; the extra synchronization makes this instrumentation distinct from
an uninstrumented CUDA replay.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import stat
import threading

import mujoco_warp as mjwarp
import torch
import warp as wp

from mujoco_warp._src import smooth as pinned_smooth

from mjlab_microduck import stance_recovery_early_inertia_trace as inertia
from mjlab_microduck import stance_warp_runtime
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-crb-launch-trace-v1"
BOUNDARY = "before-after-each-complete-crb-accumulation-launch"
SMOOTH_SOURCE_SHA256 = (
    "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
)
SMOOTH_MODULE = "mujoco_warp._src.smooth"
MAX_STAGE_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_WORLDS = 64
MAX_BODIES = 16
VEC10 = 10
SNAPSHOTS_PER_STAGE = 4  # before/after CRB and before/after cinert
_PATCH_LOCK = threading.Lock()
FLAGS = {
    **inertia.FLAGS,
    "crb_launch_trace_qualified": False,
    "crb_launch_cause_proven": False,
    "actual_atomic_order_observed": False,
}
_BASE_KEYS = {
    "protocol",
    "source",
    "worlds",
    "step_limit",
    "floor_id",
    "foot_ids",
    "control_ids",
    "status",
    "events",
    "observation_boundary",
    "excluded_intermediates",
    *inertia.FLAGS,
}
_TRACE_KEYS = _BASE_KEYS | {
    "launch_trace_boundary",
    "smooth_module",
    "smooth_source_sha256",
    "accumulation_kernel",
    "compiled_topology",
    "compiled_topology_sha256",
    "stage_payload_tensor_bytes",
    "stage_payload_limit_bytes",
    "partial_crb_accumulation",
    *FLAGS,
}
_STAGE_KEYS = {
    "stage_index",
    "body_tree_ids",
    "dim",
    "before_crb",
    "before_cinert",
    "after_crb",
    "after_cinert",
}
_INHERITED_EVENT_KEYS = {"phase", "step", "inputs", "solved", "qM", "persistent"}
_TOPOLOGY_KEYS = {
    "nbody",
    "nq",
    "nv",
    "nu",
    "worlds",
    "body_parentid",
    "reversed_body_tree_ids",
}


def _source_hash(path):
    try:
        source_path = Path(path)
        metadata = source_path.lstat()
        if (
            source_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 1024 * 1024
        ):
            return None
        descriptor = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > 1024 * 1024
                or (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino)
            ):
                return None
            raw = stream.read(1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
            attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                len(raw) > 1024 * 1024
                or len(raw) != before.st_size
                or any(
                    getattr(before, key) != getattr(after, key) for key in attributes
                )
            ):
                return None
        return sha256(raw).hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _array_ids(value):
    return [int(item) for item in wp.to_torch(value).detach().cpu().tolist()]


def _topology_payload(native, model, worlds):
    require(
        (native.nbody, native.nq, native.nv, native.nu) == (16, 21, 20, 14)
        and worlds in (2, 64)
        and not model.is_sparse,
        "exact dense stance CRB launch topology",
    )
    parent_ids = [int(item) for item in native.body_parentid.tolist()]
    require(
        _array_ids(model.body_parentid) == parent_ids,
        "compiled Warp body parents match native parent topology",
    )
    levels = [_array_ids(tree) for tree in reversed(model.body_tree)]
    require(
        len(levels) > 0
        and len(levels) <= native.nbody
        and all(0 < len(level) <= native.nbody for level in levels)
        and all(0 <= body < native.nbody for level in levels for body in level)
        and sorted(body for level in levels for body in level)
        == list(range(native.nbody)),
        "compiled reversed body-tree levels partition every body exactly once",
    )
    return {
        "nbody": int(native.nbody),
        "nq": int(native.nq),
        "nv": int(native.nv),
        "nu": int(native.nu),
        "worlds": int(worlds),
        "body_parentid": parent_ids,
        "reversed_body_tree_ids": levels,
    }


def _fresh_topology(worlds):
    """Bind retained level IDs to a fresh CPU compilation, without a physics step."""
    with torch.random.fork_rng(devices=[]):
        native = stance_warp_runtime.build_entity().compile()
    with wp.ScopedDevice(wp.get_device("cpu")):
        model = mjwarp.put_model(native)
        return _topology_payload(native, model, worlds)


def _raw_equal(left, right):
    return (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.dtype == right.dtype
        and left.device.type == right.device.type
        and left.shape == right.shape
        and left.detach().contiguous().cpu().numpy().tobytes()
        == right.detach().contiguous().cpu().numpy().tobytes()
    )


def _check_tensor(value, worlds, label):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and tuple(value.shape) == (worlds, MAX_BODIES, VEC10)
        and value.is_contiguous()
        and bool(torch.isfinite(value).all()),
        "owned finite CPU float32 " + label,
    )


def _plain_int(value):
    return type(value) is int


def _exact_keys(value, expected):
    return (
        type(value) is dict and len(value) == len(expected) and set(value) == expected
    )


def _plain_int_list(value):
    return (
        type(value) is list
        and len(value) <= MAX_BODIES
        and all(_plain_int(item) for item in value)
    )


def _preflight_owned_metadata(value):
    """Reject ambiguous Python scalar types before compiling or traversing."""
    require(
        _plain_int(value["worlds"])
        and value["worlds"] in (2, 64)
        and _plain_int(value["step_limit"])
        and _plain_int(value["floor_id"])
        and _plain_int_list(value["foot_ids"])
        and _plain_int_list(value["control_ids"]),
        "exact integer CRB trace metadata",
    )
    require(
        _plain_int(value["stage_payload_tensor_bytes"])
        and _plain_int(value["stage_payload_limit_bytes"]),
        "exact integer CRB payload limits",
    )
    topology = value["compiled_topology"]
    require(
        _exact_keys(topology, _TOPOLOGY_KEYS)
        and all(
            _plain_int(topology[name]) for name in ("nbody", "nq", "nv", "nu", "worlds")
        )
        and _plain_int_list(topology["body_parentid"])
        and type(topology["reversed_body_tree_ids"]) is list
        and 0 < len(topology["reversed_body_tree_ids"]) <= MAX_BODIES
        and all(_plain_int_list(level) for level in topology["reversed_body_tree_ids"]),
        "exact integer compiled topology metadata",
    )
    require(
        type(value["events"]) is list
        and len(value["events"]) == 2 * inertia.early.STEPS,
        "exact complete CRB event count",
    )
    for event_index, event in enumerate(value["events"]):
        require(
            _exact_keys(event, _INHERITED_EVENT_KEYS | {"crb_accumulation_stages"})
            and type(event["phase"]) is str
            and _plain_int(event["step"])
            and type(event["crb_accumulation_stages"]) is list,
            "exact event metadata before topology compilation",
        )
        expected_phase = "scheduled-pre" if event_index % 2 == 0 else "unforced-post"
        require(
            event["phase"] == expected_phase and event["step"] == event_index // 2,
            "ordered event metadata before topology compilation",
        )
        require(
            0 < len(event["crb_accumulation_stages"]) <= MAX_BODIES,
            "bounded stage count before traversal",
        )
        for stage_index, stage in enumerate(event["crb_accumulation_stages"]):
            require(
                _exact_keys(stage, _STAGE_KEYS)
                and _plain_int(stage["stage_index"])
                and stage["stage_index"] == stage_index
                and _plain_int_list(stage["body_tree_ids"])
                and _plain_int_list(stage["dim"]),
                "exact stage scalar types before topology compilation",
            )


def _launch_parameter(args, kwargs, name, position):
    if name in kwargs:
        return kwargs[name]
    require(len(args) > position, "complete Warp launch arguments")
    return args[position]


class CrbLaunchTrace(inertia.EarlyInertiaTrace):
    """Early inertia observer with a narrowly scoped process-local launch hook."""

    def __init__(self, env, *, smooth_module, launch_module=wp):
        require(
            type(smooth_module).__name__ == "module"
            and smooth_module is pinned_smooth
            and getattr(smooth_module, "__name__", None) == SMOOTH_MODULE
            and _source_hash(getattr(smooth_module, "__file__", None))
            == SMOOTH_SOURCE_SHA256
            and getattr(smooth_module, "wp", None) is launch_module
            and callable(getattr(smooth_module, "_crb_accumulate", None))
            and smooth_module._crb_accumulate is pinned_smooth._crb_accumulate
            and callable(getattr(launch_module, "launch", None)),
            "exact pinned smooth module, kernel source and Warp launch module",
        )
        self._topology_snapshot = None
        super().__init__(env)
        self.smooth_module = smooth_module
        self.launch_module = launch_module
        self.kernel = smooth_module._crb_accumulate
        self._original_launch = launch_module.launch
        self._launch_hook = self._dispatch_launch
        self._owner_thread = None
        self._current_levels = None
        self._current_phase = None
        self._topology_snapshot = self._topology_from_env()
        self._topology_sha256 = sha256(
            canonical(self._topology_snapshot).encode("utf-8")
        ).hexdigest()
        stage_bytes = (
            2
            * inertia.early.STEPS
            * len(self._topology_snapshot["reversed_body_tree_ids"])
            * env.n
            * MAX_BODIES
            * VEC10
            * torch.tensor([], dtype=torch.float32).element_size()
            * SNAPSHOTS_PER_STAGE
        )
        require(
            stage_bytes <= MAX_STAGE_PAYLOAD_BYTES,
            "bounded separate CRB stage payload",
        )
        self._stage_payload_tensor_bytes = stage_bytes
        self._partial_stage_event = None
        self._inflight_stage = None

    def _topology(self):
        super()._topology()
        if self._topology_snapshot is not None:
            current = self._topology_from_env()
            require(
                current == self._topology_snapshot,
                "compiled Warp body-tree topology remains unchanged",
            )

    def _topology_from_env(self):
        self.env._sync()
        return _topology_payload(self.env.native, self.env.model, self.env.n)

    def _owned_snapshot(self, name):
        value = wp.to_torch(getattr(self.env.data, name)).detach().cpu().clone()
        require(
            value.shape == (self.env.n, MAX_BODIES, VEC10)
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all()),
            "finite exact CRB launch tensor " + name,
        )
        return value.contiguous()

    def _device_matches(self, args, kwargs):
        device = kwargs.get("device")
        if device is None and len(args) > 5:
            device = args[5]
        current = wp.get_device() if device is None else wp.get_device(device)
        return str(current) == str(self.env.wp_device)

    def _dispatch_launch(self, kernel, *args, **kwargs):
        original = self._original_launch
        if kernel is not self.kernel or self._current_levels is None:
            return original(kernel, *args, **kwargs)
        inputs = kwargs.get("inputs")
        if inputs is None and len(args) > 1:
            inputs = args[1]
        outputs = kwargs.get("outputs")
        if outputs is None and len(args) > 2:
            outputs = args[2]
        # Same kernel on another model/data is outside this observer.  It is
        # forwarded byte-for-byte and does not consume an owned level slot.
        if (
            type(inputs) not in (list, tuple)
            or len(inputs) < 2
            or inputs[0] is not self.env.model.body_parentid
            or inputs[1] is not self.env.data.crb
        ):
            return original(kernel, *args, **kwargs)
        if threading.get_ident() != self._owner_thread:
            self.faulted = True
            self.env.faulted = True
            raise ValueError("owned CRB launch attempted from a foreign thread")
        require(
            not args and _exact_keys(kwargs, {"dim", "inputs", "outputs"}),
            "exact pinned smooth.crb Warp launch signature",
        )
        level_index = len(self._current_levels)
        expected_levels = self._topology_snapshot["reversed_body_tree_ids"]
        require(
            level_index < len(expected_levels), "bounded CRB body-tree launch count"
        )
        require(
            len(inputs) == 3
            and inputs[2] is self._expected_level_arrays[level_index]
            and type(outputs) in (list, tuple)
            and len(outputs) == 1
            and outputs[0] is self.env.data.crb,
            "exact owned CRB launch array identities and level order",
        )
        dim = _launch_parameter(args, kwargs, "dim", 0)
        require(
            tuple(dim) == (self.env.n, len(expected_levels[level_index]))
            and self._device_matches(args, kwargs),
            "exact owned CRB launch dimension and device",
        )
        tree_ids = _array_ids(inputs[2])
        require(
            tree_ids == expected_levels[level_index],
            "exact compiled reversed body-tree IDs",
        )
        self.env._sync()
        before_crb = self._owned_snapshot("crb")
        before_cinert = self._owned_snapshot("cinert")
        self._inflight_stage = {
            "stage_index": level_index,
            "body_tree_ids": tree_ids,
            "dim": [self.env.n, len(tree_ids)],
            "before_crb": before_crb,
            "before_cinert": before_cinert,
        }
        try:
            result = original(kernel, *args, **kwargs)
            self.env._sync()
            after_crb = self._owned_snapshot("crb")
            after_cinert = self._owned_snapshot("cinert")
            require(
                _raw_equal(before_cinert, after_cinert),
                "CRB accumulation leaves cinert unchanged",
            )
        except BaseException:
            self.faulted = True
            self.env.faulted = True
            raise
        self._current_levels.append(
            {
                "stage_index": level_index,
                "body_tree_ids": tree_ids,
                "dim": [self.env.n, len(tree_ids)],
                "before_crb": before_crb,
                "before_cinert": before_cinert,
                "after_crb": after_crb,
                "after_cinert": after_cinert,
            }
        )
        self._inflight_stage = None
        return result

    def _invoke_with_launch_scope(self, invoke, levels):
        require(
            _PATCH_LOCK.acquire(blocking=False),
            "exclusive process-local Warp launch observer",
        )
        self._owner_thread = threading.get_ident()
        self._current_levels = levels
        try:
            require(
                self.launch_module.launch is self._original_launch,
                "Warp launch callable remains exclusively available",
            )
            self.launch_module.launch = self._launch_hook
            result = invoke()
        except BaseException:
            self.faulted = True
            self.env.faulted = True
            raise
        finally:
            hook_changed = self.launch_module.launch is not self._launch_hook
            if not hook_changed:
                self.launch_module.launch = self._original_launch
            if hook_changed:
                self.faulted = True
                self.env.faulted = True
            self._current_levels = None
            self._owner_thread = None
            _PATCH_LOCK.release()
        if hook_changed:
            raise ValueError("Warp launch hook changed during owned invocation")
        return result

    def _event(self, phase, step, invoke):
        self._topology()
        levels = []
        self._current_phase = (phase, step)
        try:
            result = super()._event(
                phase,
                step,
                lambda: self._invoke_with_launch_scope(invoke, levels),
            )
            require(
                len(levels) == len(self._topology_snapshot["reversed_body_tree_ids"]),
                "complete CRB accumulation level trace",
            )
            self.events[-1]["crb_accumulation_stages"] = levels
            self._partial_stage_event = None
            return result
        except BaseException:
            self.faulted = True
            self.env.faulted = True
            partial = list(levels)
            if self._inflight_stage is not None:
                partial.append({**self._inflight_stage, "pending_after_snapshot": True})
            self._partial_stage_event = {
                "phase": phase,
                "step": step,
                "stages": partial,
            }
            raise
        finally:
            self._current_phase = None

    def __enter__(self):
        require(
            self.launch_module.launch is self._original_launch,
            "Warp launch callable remains unchanged before observer entry",
        )
        self._expected_level_arrays = list(reversed(self.env.model.body_tree))
        return super().__enter__()

    def __exit__(self, kind, error, traceback):
        failure = None
        try:
            return super().__exit__(kind, error, traceback)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            current = self.launch_module.launch
            foreign_replacement = current not in (
                self._original_launch,
                self._launch_hook,
            )
            if current is self._launch_hook:
                self.launch_module.launch = self._original_launch
            if foreign_replacement:
                self.faulted = True
                self.env.faulted = True
                if failure is None and kind is None:
                    raise ValueError("foreign Warp launch replacement preserved")

    def capture(self):
        value = super().capture()
        payload_stages = [
            stage
            for event in value["events"]
            for stage in event.get("crb_accumulation_stages", [])
        ]
        if self._partial_stage_event is not None:
            payload_stages.extend(self._partial_stage_event["stages"])
        payload_bytes = sum(
            tensor.numel() * tensor.element_size()
            for stage in payload_stages
            for tensor in stage.values()
            if torch.is_tensor(tensor)
        )
        value.update(
            protocol=PROTOCOL,
            launch_trace_boundary=BOUNDARY,
            smooth_module=SMOOTH_MODULE,
            smooth_source_sha256=SMOOTH_SOURCE_SHA256,
            accumulation_kernel="mujoco_warp._src.smooth._crb_accumulate",
            compiled_topology=deepcopy(self._topology_snapshot),
            compiled_topology_sha256=self._topology_sha256,
            stage_payload_tensor_bytes=payload_bytes,
            stage_payload_limit_bytes=MAX_STAGE_PAYLOAD_BYTES,
            partial_crb_accumulation=deepcopy(self._partial_stage_event),
            **FLAGS,
        )
        require(
            payload_bytes <= MAX_STAGE_PAYLOAD_BYTES,
            "bounded retained CRB stage payload",
        )
        return value


def _project(value):
    require(
        _exact_keys(value, _TRACE_KEYS)
        and value["protocol"] == PROTOCOL
        and value["launch_trace_boundary"] == BOUNDARY
        and value["smooth_module"] == SMOOTH_MODULE
        and value["smooth_source_sha256"] == SMOOTH_SOURCE_SHA256
        and value["accumulation_kernel"] == "mujoco_warp._src.smooth._crb_accumulate"
        and all(value[name] is False for name in FLAGS)
        and type(value["events"]) is list
        and len(value["events"]) == 2 * inertia.early.STEPS,
        "exact non-admitting CRB launch trace metadata",
    )
    events = []
    for event in value["events"]:
        require(
            _exact_keys(event, _INHERITED_EVENT_KEYS | {"crb_accumulation_stages"}),
            "exact inherited and CRB-stage event schema",
        )
        events.append(
            {
                key: item
                for key, item in event.items()
                if key != "crb_accumulation_stages"
            }
        )
    projected = {
        key: item for key, item in value.items() if key not in _TRACE_KEYS - _BASE_KEYS
    }
    projected.update(protocol=inertia.PROTOCOL, events=events, **inertia.FLAGS)
    return projected


def check(value, declaration, first_record):
    """Validate stage bytes/topology before unchanged persistent-inertia checks."""
    inertia._hidden()
    require(
        _exact_keys(value, _TRACE_KEYS),
        "exact CRB launch trace fields before traversal",
    )
    _project(value)
    _preflight_owned_metadata(value)
    require(
        value["partial_crb_accumulation"] is None,
        "complete trace has no pending CRB stage",
    )
    inertia.early.evidence._owned_tree(value, clone=False)
    require(value["worlds"] in (2, 64), "two or 64 retained CRB worlds")
    expected_topology = _fresh_topology(value["worlds"])
    require(
        type(value["compiled_topology"]) is dict
        and value["compiled_topology"] == expected_topology
        and value["compiled_topology_sha256"]
        == sha256(canonical(expected_topology).encode("utf-8")).hexdigest(),
        "stage IDs bind to fresh compiled Warp body-tree topology",
    )
    expected_levels = expected_topology["reversed_body_tree_ids"]
    expected_tensor_bytes = (
        2
        * inertia.early.STEPS
        * len(expected_levels)
        * value["worlds"]
        * MAX_BODIES
        * VEC10
        * torch.tensor([], dtype=torch.float32).element_size()
        * SNAPSHOTS_PER_STAGE
    )
    require(
        type(value["stage_payload_tensor_bytes"]) is int
        and value["stage_payload_tensor_bytes"] == expected_tensor_bytes
        and type(value["stage_payload_limit_bytes"]) is int
        and value["stage_payload_limit_bytes"] == MAX_STAGE_PAYLOAD_BYTES
        and expected_tensor_bytes <= MAX_STAGE_PAYLOAD_BYTES
        and type(value["events"]) is list
        and len(value["events"]) == 2 * inertia.early.STEPS,
        "exact bounded complete CRB stage payload",
    )
    for event_index, event in enumerate(value["events"]):
        require(
            _exact_keys(event, _INHERITED_EVENT_KEYS | {"crb_accumulation_stages"})
            and type(event["crb_accumulation_stages"]) is list
            and len(event["crb_accumulation_stages"]) == len(expected_levels),
            "complete per-forward CRB stage list",
        )
        require(
            _exact_keys(event["persistent"], set(inertia.SHAPES)),
            "exact inherited persistent fields before stage validation",
        )
        inertia._persistent(event["persistent"], value["worlds"])
        previous_after = None
        event_cinert = None
        for stage_index, stage in enumerate(event["crb_accumulation_stages"]):
            require(
                _exact_keys(stage, _STAGE_KEYS)
                and stage["stage_index"] == stage_index
                and stage["body_tree_ids"] == expected_levels[stage_index]
                and stage["dim"]
                == [value["worlds"], len(expected_levels[stage_index])],
                "exact actual body-tree stage order and dimension",
            )
            for name in ("before_crb", "before_cinert", "after_crb", "after_cinert"):
                _check_tensor(stage[name], value["worlds"], name)
            require(
                _raw_equal(stage["before_cinert"], stage["after_cinert"]),
                "cinert unchanged across each accumulation launch",
            )
            if event_cinert is not None:
                require(
                    _raw_equal(event_cinert, stage["before_cinert"]),
                    "cinert bytes unchanged across adjacent accumulation stages",
                )
            else:
                event_cinert = stage["before_cinert"]
            if stage_index == 0:
                require(
                    _raw_equal(stage["before_crb"], stage["before_cinert"]),
                    "first accumulation begins after cinert-to-crb initialization",
                )
            if previous_after is not None:
                require(
                    _raw_equal(previous_after, stage["before_crb"]),
                    "adjacent CRB stages share exact intermediate bytes",
                )
            previous_after = stage["after_crb"]
        require(
            _raw_equal(previous_after, event["persistent"]["crb"]),
            "last CRB stage binds to inherited completed-forward persistent CRB",
        )
        require(
            _raw_equal(event_cinert, event["persistent"]["cinert"]),
            "all CRB stages bind to inherited persistent cinert",
        )
    projected = _project(value)
    score = inertia.check(projected, declaration, first_record)
    return {
        "protocol": PROTOCOL + ":score",
        "source": score["source"],
        "worlds": score["worlds"],
        "events": score["events"],
        "stages_per_event": len(expected_levels),
        "compiled_topology_sha256": value["compiled_topology_sha256"],
        "stage_payload_tensor_bytes": expected_tensor_bytes,
        "actual_atomic_order_observed": False,
        **FLAGS,
    }
