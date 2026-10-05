"""Persistent post-forward inertia diagnostics, not a physics change or waiver.

The inherited observer invokes each original runtime method exactly once. These
additional synchronized CPU copies observe the mass/bias path after the complete
forward (including acceleration sensors). They are not stage-boundary samples:
the accelerometer's postconstraint RNE overwrites cacc/cfrc_int/cfrc_ext, which
are deliberately excluded. Instrumented runs do not qualify uninstrumented
CUDA replay, prove a kernel cause, or admit an optimizer update.
"""

import os

import torch

from mjlab_microduck import stance_recovery_early_forward_trace as early
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-early-inertia-trace-v1"
BOUNDARY = "after-complete-forward-including-acceleration-sensors"
EXCLUDED = ("cacc", "cfrc_int", "cfrc_ext")
# This is a reporting order, not a temporal ordering of the upstream kernels.
SHAPES = {
    "subtree_com": (16, 3),
    "cinert": (16, 10),
    "cdof": (20, 6),
    "crb": (16, 10),
    "qM": (20, 20),
    "qLD": (20, 20),
    "cvel": (16, 6),
    "cdof_dot": (20, 6),
    "qfrc_bias": (20,),
    "qfrc_smooth": (20,),
}
FLAGS = {
    **early.FLAGS,
    "persistent_inertia_qualified": False,
    "inertia_kernel_cause_proven": False,
}


def _hidden():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden persistent inertia reader",
    )


def _persistent(values, worlds):
    require(
        type(values) is dict and set(values) == set(SHAPES),
        "exact persistent post-forward fields",
    )
    for name, suffix in SHAPES.items():
        early.forward.trace.tensor(
            values[name], (worlds, *suffix), torch.float32, "persistent " + name
        )


def _bytes_equal(left, right):
    # The frozen early reader keeps its original value-equality semantics. New
    # duplicate fields additionally bind raw bytes, including signed zero.
    return early.evidence._equal(left, right) and (
        left.detach().contiguous().numpy().tobytes()
        == right.detach().contiguous().numpy().tobytes()
    )


class EarlyInertiaTrace(early.EarlyForwardTrace):
    """Separate observer protocol; unchanged runtime and old trace reader."""

    def __init__(self, env):
        super().__init__(env)
        self._topology()

    def _topology(self):
        require(
            (
                self.env.native.nbody,
                self.env.native.nq,
                self.env.native.nv,
                self.env.native.nu,
            )
            == (16, 21, 20, 14)
            and not self.env.model.is_sparse
            and self.env.model.sensor_rne_postconstraint
            and self.env.forward_graph is None
            and getattr(self.env._view, "__func__", None)
            is early.WarpStanceRuntime._view
            and getattr(self.env._sync, "__func__", None)
            is early.WarpStanceRuntime._sync,
            "exact dense accelerometer stance observation topology",
        )

    def _event(self, phase, step, invoke):
        self._topology()
        result = super()._event(phase, step, invoke)
        self.env._sync()
        values = {name: self.env._view(name).detach().cpu().clone() for name in SHAPES}
        early.evidence._owned_tree(values, clone=False)
        _persistent(values, self.env.n)
        event = self.events[-1]
        require(
            _bytes_equal(values["qM"], event["qM"])
            and _bytes_equal(
                values["qfrc_bias"], event["solved"]["dynamics"]["qfrc_bias"]
            ),
            "persistent fields bind to the same completed forward",
        )
        event["persistent"] = values
        return result

    def capture(self):
        value = super().capture()  # Includes owned copies, only after restoration.
        value.update(
            protocol=PROTOCOL,
            observation_boundary=BOUNDARY,
            excluded_intermediates=list(EXCLUDED),
            **FLAGS,
        )
        return value


def _projection(value):
    """Project only after schema checks; do not bless damaged old trace fields."""
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
            "observation_boundary",
            "excluded_intermediates",
            *FLAGS,
        }
        and value["protocol"] == PROTOCOL
        and value["observation_boundary"] == BOUNDARY
        and type(value["excluded_intermediates"]) is list
        and value["excluded_intermediates"] == list(EXCLUDED)
        and all(value[name] is False for name in FLAGS)
        and type(value["events"]) is list,
        "exact non-admitting persistent trace metadata",
    )
    events = []
    for event in value["events"]:
        require(
            type(event) is dict
            and set(event) == {"phase", "step", "inputs", "solved", "qM", "persistent"},
            "exact persistent forward event",
        )
        require(
            type(event["persistent"]) is dict
            and set(event["persistent"]) == set(SHAPES),
            "exact persistent post-forward fields before projection",
        )
        events.append({key: item for key, item in event.items() if key != "persistent"})
    projected = {
        key: item
        for key, item in value.items()
        if key not in {"observation_boundary", "excluded_intermediates", *FLAGS}
    }
    projected.update(protocol=early.PROTOCOL, events=events, **early.FLAGS)
    return projected


def check(value, declaration, first_record):
    """Bound to all six original trajectory/control phases; CPU evidence only."""
    _hidden()  # Before accessing or traversing caller data.
    early.evidence._owned_tree(value, clone=False)
    projected = _projection(value)
    score = early.check(projected, declaration, first_record)
    for event in value["events"]:
        require(
            event["inputs"]["xfrc_applied"].shape == (value["worlds"], 16, 6),
            "persistent trace exact compiled body dimension",
        )
        _persistent(event["persistent"], value["worlds"])
        require(
            _bytes_equal(event["persistent"]["qM"], event["qM"])
            and _bytes_equal(
                event["persistent"]["qfrc_bias"],
                event["solved"]["dynamics"]["qfrc_bias"],
            ),
            "persistent fields bind to the same completed forward",
        )
    return dict(
        protocol=PROTOCOL + ":score",
        source=score["source"],
        worlds=score["worlds"],
        events=score["events"],
        steps=score["steps"],
        observation_boundary=BOUNDARY,
        excluded_intermediates=list(EXCLUDED),
        **FLAGS,
    )


def compare(left, right, declaration, records):
    """Report raw persistent differences; neither tolerance nor row canonicalization."""
    _hidden()
    require(
        type(records) in (list, tuple) and len(records) == 2,
        "paired persistent trajectory records",
    )
    check(left, declaration, records[0])
    check(right, declaration, records[1])
    prior = early.compare(_projection(left), _projection(right), declaration, records)
    events = []
    for index, (a, b) in enumerate(zip(left["events"], right["events"], strict=True)):
        fields = {}
        for name in SHAPES:
            x, y = a["persistent"][name], b["persistent"][name]
            bits = x.contiguous().view(torch.int32) != y.contiguous().view(torch.int32)
            changed = bits.flatten(start_dim=1).any(dim=1)
            fields[name] = dict(
                exact=_bytes_equal(x, y),
                different_elements=int(bits.sum()),
                max_abs_delta=float((x.double() - y.double()).abs().max()),
                differing_worlds=int(changed.sum()),
                first_differing_worlds=changed.nonzero().flatten().tolist()[:8],
                world_details_truncated=int(changed.sum()) > 8,
            )
        events.append(
            dict(
                index=index,
                phase=a["phase"],
                step=a["step"],
                exact=early.forward.throughput.tree_hash(a)
                == early.forward.throughput.tree_hash(b),
                raw_event_exact=prior["events"][index]["exact"],
                fields=fields,
                first_listed_differing_field=next(
                    (k for k in SHAPES if not fields[k]["exact"]), None
                ),
            )
        )
    return dict(
        protocol=PROTOCOL + ":comparison",
        source=left["source"],
        worlds=left["worlds"],
        exact=early.forward.throughput.tree_hash(left)
        == early.forward.throughput.tree_hash(right),
        earliest_differing_event=next(
            (x["index"] for x in events if not x["exact"]), None
        ),
        observation_boundary=BOUNDARY,
        excluded_intermediates=list(EXCLUDED),
        reporting_order_is_not_kernel_temporal_order=True,
        persistent_exactness_includes_signed_zero=True,
        original_trace_comparison=prior,
        events=events,
        **FLAGS,
    )
