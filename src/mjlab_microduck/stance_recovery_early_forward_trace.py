"""Bounded observational forward hooks, not a solver or replay-gate waiver.

The exact inherited methods run once, with the original arguments and results.
Additional synchronization/CPU reads are diagnostic instrumentation: a traced
run is not evidence that an uninstrumented native schedule is deterministic.
"""

from copy import deepcopy
import os

import torch

from mjlab_microduck import stance_forward_probe as forward
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-early-forward-trace-v1"
STEPS = 3
INPUTS = (*forward.KINEMATICS, "ctrl", "xfrc_applied", "qfrc_applied", "qM")
FLAGS = {
    **evidence.FALSE_FLAGS,
    "native_trace_qualified": False,
    "native_kernel_cause_proven": False,
    "uninstrumented_replay_qualified": False,
}


class EarlyForwardTrace:
    """One exact runtime instance, one hook lifetime, six full-batch phases."""

    def __init__(self, env):
        require(
            type(env) is ScheduledRecoveryRuntime
            and env.n in (2, 64)
            and env.forward_graph is None
            and getattr(env._scheduled_forward, "__func__", None)
            is ScheduledRecoveryRuntime._scheduled_forward
            and getattr(env._forward, "__func__", None) is WarpStanceRuntime._forward
            and not {"_forward", "_scheduled_forward"} & set(env.__dict__),
            "exact unmodified eager scheduled runtime for observation",
        )
        checked = schedule.checked(env.schedule_declaration)
        require(checked["worlds"] == env.n, "trace schedule world binding")
        self.env = env
        self.source = checked["source"]
        self.events = []
        self.faulted = False
        self._used = False
        self._active = False
        self._pending = None
        self._originals = {
            "_scheduled_forward": env._scheduled_forward,
            "_forward": env._forward,
        }
        self._hooks = {
            "_scheduled_forward": self._scheduled,
            "_forward": self._unforced,
        }

    def __enter__(self):
        if self._used:
            self.faulted = True
            raise ValueError("one-shot forward observation lifetime")
        require(
            not {"_forward", "_scheduled_forward"} & set(self.env.__dict__)
            and all(
                getattr(getattr(self.env, name), "__func__", None) is method.__func__
                for name, method in self._originals.items()
            )
            and self.env.forward_graph is None,
            "forward hooks are still exclusively available",
        )
        self._used = self._active = True
        for name, hook in self._hooks.items():
            setattr(self.env, name, hook)
        return self

    def __exit__(self, kind, _error, _tb):
        changed = []
        for name, hook in self._hooks.items():
            if self.env.__dict__.get(name) is hook:
                delattr(self.env, name)
            else:
                changed.append(name)
        self._active = False
        self.faulted |= kind is not None or bool(changed)
        if changed:
            self.env.faulted = True
            raise ValueError("owned forward observer hook was replaced")
        return False

    def _inputs(self):
        self.env._sync()
        values = {name: self.env._view(name).detach().cpu().clone() for name in INPUTS}
        evidence._owned_tree(values, clone=False)
        return values

    def _event(self, phase, step, invoke):
        require(self._active, "active owned forward observation")
        before = self._inputs()
        result = invoke()  # Exactly one inherited call; no extra solve/integration.
        solved = forward.output(self.env)
        after = self._inputs()
        # Forward must not change integration state or the caller's controls.
        require(
            all(evidence._equal(before[k], after[k]) for k in INPUTS if k != "qM"),
            "observed forward preserves integration and control inputs",
        )
        self.events.append(
            dict(phase=phase, step=step, inputs=before, solved=solved, qM=after["qM"])
        )
        require(len(self.events) <= 2 * STEPS, "bounded early phase count")
        return result

    def _scheduled(self, steps, accepted):
        original = self._originals["_scheduled_forward"]
        if len(self.events) == 2 * STEPS:
            return original(steps, accepted)
        try:
            step = len(self.events) // 2
            require(
                self._pending is None
                and len(self.events) % 2 == 0
                and steps.shape == accepted.shape == (self.env.n,)
                and steps.dtype == torch.long
                and accepted.dtype == torch.bool
                and (steps == step).all()
                and accepted.all(),
                "ordered complete early scheduled phase",
            )
            result = self._event(
                "scheduled-pre", step, lambda: original(steps, accepted)
            )
            self._pending = step
            return result
        except BaseException:
            self.faulted = True
            raise

    def _unforced(self):
        original = self._originals["_forward"]
        if self._pending is None:
            return original()
        try:
            result = self._event("unforced-post", self._pending, original)
            self._pending = None
            return result
        except BaseException:
            self.faulted = True
            raise

    def capture(self):
        require(not self._active, "retain trace after hook restoration")
        return deepcopy(
            dict(
                protocol=PROTOCOL,
                source=self.source,
                worlds=self.env.n,
                step_limit=STEPS,
                floor_id=self.env.floor,
                foot_ids=list(self.env.feet),
                control_ids=self.env.ctrl_ids.detach().cpu().tolist(),
                status=(
                    "faulted"
                    if self.faulted
                    else "complete"
                    if len(self.events) == 2 * STEPS
                    else "incomplete"
                ),
                events=self.events,
                **FLAGS,
            )
        )


def check(value, declaration, first_record):
    """CPU-only structural/trajectory binding, not native solver re-execution."""
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden trace reader",
    )
    evidence._owned_tree(value, clone=False)
    checked = schedule.checked(declaration)
    require(
        type(value) is dict
        and set(value)
        == {
            "protocol",
            "source",
            "worlds",
            "step_limit",
            "floor_id",
            "foot_ids",
            "control_ids",
            "status",
            "events",
            *FLAGS,
        }
        and value["protocol"] == PROTOCOL
        and value["source"] == checked["source"]
        and type(value["worlds"]) is int
        and value["worlds"] == checked["worlds"]
        and value["worlds"] in (2, 64)
        and type(value["step_limit"]) is int
        and value["step_limit"] == STEPS
        and type(value["floor_id"]) is int
        and value["floor_id"] >= 0
        and type(value["foot_ids"]) is list
        and len(value["foot_ids"]) == 2
        and all(type(x) is int and x >= 0 for x in value["foot_ids"])
        and len({value["floor_id"], *value["foot_ids"]}) == 3
        and type(value["control_ids"]) is list
        and len(value["control_ids"]) == 14
        and all(type(x) is int for x in value["control_ids"])
        and sorted(value["control_ids"]) == list(range(14))
        and value["status"] == "complete"
        and all(value[k] is False for k in FLAGS),
        "exact complete non-admitting early trace",
    )
    n = value["worlds"]
    events = value["events"]
    boundaries = first_record["runtime_result_before_reset"]["boundaries"]
    require(
        type(events) is list
        and len(events) == 2 * STEPS
        and len(boundaries) >= STEPS + 1,
        "six early solves and corresponding retained boundaries",
    )
    for index, event in enumerate(events):
        step, post = index // 2, bool(index % 2)
        require(
            type(event) is dict
            and set(event) == {"phase", "step", "inputs", "solved", "qM"}
            and type(event["step"]) is int
            and event["step"] == step
            and event["phase"] == ("unforced-post" if post else "scheduled-pre")
            and type(event["inputs"]) is dict
            and set(event["inputs"]) == set(INPUTS),
            "exact ordered early forward event",
        )
        forward.validate_output(event["solved"], n)
        inputs = event["inputs"]
        for key in forward.KINEMATICS:
            shape = (n,) if key == "time" else (n, 21 if key == "qpos" else 20)
            forward.trace.tensor(inputs[key], shape, torch.float32, key)
            require(
                evidence._equal(inputs[key], event["solved"]["kinematics"][key]),
                "forward preserves recorded integration field " + key,
            )
        for key, cols in (("ctrl", 14), ("qfrc_applied", 20)):
            forward.trace.tensor(inputs[key], (n, cols), torch.float32, key)
        xfrc = inputs["xfrc_applied"]
        require(
            torch.is_tensor(xfrc)
            and xfrc.device.type == "cpu"
            and xfrc.dtype == torch.float32
            and xfrc.ndim == 3
            and xfrc.shape[0] == n
            and xfrc.shape[2] == 6
            and 0 < xfrc.shape[1] <= 64
            and torch.isfinite(xfrc).all()
            and not xfrc.any()
            and not inputs["qfrc_applied"].any(),
            "early prefix has exact zero applied-force arrays",
        )
        for matrix in (inputs["qM"], event["qM"]):
            require(
                torch.is_tensor(matrix)
                and matrix.device.type == "cpu"
                and matrix.dtype == torch.float32
                and matrix.ndim == 3
                and matrix.shape == (n, 20, 20)
                and torch.isfinite(matrix).all(),
                "finite complete dense per-world mass matrix",
            )
        boundary = boundaries[step + int(post)]
        clock = torch.zeros(n, dtype=torch.float32)
        for _ in range(step + int(post)):
            clock.add_(0.002)
        require(
            torch.equal(inputs["time"], clock),
            "early trace phase clock binds to declared physics step",
        )
        require(
            all(evidence._equal(inputs[k], boundary[k]) for k in ("qpos", "qvel")),
            "trace inputs bind to actual retained trajectory",
        )
        proposal = first_record["runtime_result_before_reset"]["control_evidence"][
            "proposals"
        ][step]
        require(
            evidence._equal(
                inputs["ctrl"][:, value["control_ids"]], proposal["committed"]["ctrl"]
            ),
            "trace controls bind to actual committed actuator order",
        )
    return dict(
        protocol=PROTOCOL + ":score",
        events=len(events),
        steps=STEPS,
        source=value["source"],
        worlds=n,
        **FLAGS,
    )


def _differences(left, right):
    count, first = 0, []

    def visit(a, b, path):
        nonlocal count
        if evidence._equal(a, b):
            return
        if torch.is_tensor(a) and torch.is_tensor(b):
            count += 1
            if len(first) < 24:
                item = dict(
                    path=path,
                    left_shape=list(a.shape),
                    right_shape=list(b.shape),
                    dtype=str(a.dtype),
                )
                if a.shape == b.shape and a.dtype == b.dtype:
                    item["different_elements"] = int((a != b).sum())
                    if a.numel():
                        item["max_abs_delta"] = float(
                            (a.double() - b.double()).abs().max()
                        )
                first.append(item)
        elif type(a) is dict and type(b) is dict and set(a) == set(b):
            for key in sorted(a):
                visit(a[key], b[key], path + "." + key)
        elif type(a) is list and type(b) is list and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "[" + str(index) + "]")
        else:
            count += 1
            if len(first) < 24:
                first.append(dict(path=path, left=str(a)[:160], right=str(b)[:160]))

    visit(left, right, "event")
    return dict(
        differing_leaves=count,
        first_differences=first,
        details_truncated=count > len(first),
    )


def _contact_multiset(table):
    # Diagnostic only: address allocation is excluded explicitly. The strict
    # native pair still compares all ordered contact and constraint fields.
    fields = set(table) - {"efc_address"}
    return sorted(
        forward.throughput.tree_hash({k: table[k][index] for k in fields})
        for index in range(table["worldid"].numel())
    )


def compare(left, right, declaration, records):
    """Compare separately checked traces; never turn proximity into acceptance."""
    require(
        type(records) in (tuple, list) and len(records) == 2,
        "paired independent trajectory records",
    )
    check(left, declaration, records[0])
    check(right, declaration, records[1])
    metadata = {
        "protocol",
        "source",
        "worlds",
        "step_limit",
        "floor_id",
        "foot_ids",
        "control_ids",
        "status",
        *FLAGS,
    }
    require(
        all(evidence._equal(left[k], right[k]) for k in metadata),
        "paired early trace metadata",
    )
    events = []
    for index, (a, b) in enumerate(zip(left["events"], right["events"])):
        groups = {
            k: evidence._equal(a["solved"][k], b["solved"][k]) for k in a["solved"]
        }
        inputs = {k: evidence._equal(a["inputs"][k], b["inputs"][k]) for k in INPUTS}
        events.append(
            dict(
                index=index,
                phase=a["phase"],
                step=a["step"],
                exact=evidence._equal(a, b),
                input_fields_exact=inputs,
                solved_groups_exact=groups,
                mass_matrix_after_exact=evidence._equal(a["qM"], b["qM"]),
                contact_multiset_ignoring_addresses_exact=(
                    _contact_multiset(a["solved"]["contacts"])
                    == _contact_multiset(b["solved"]["contacts"])
                ),
                **_differences(a, b),
            )
        )
    return dict(
        protocol=PROTOCOL + ":comparison",
        source=left["source"],
        exact=evidence._equal(left, right),
        earliest_differing_event=next(
            (x["index"] for x in events if not x["exact"]), None
        ),
        contact_multiset_ignored_fields=["efc_address"],
        events=events,
        **FLAGS,
    )
