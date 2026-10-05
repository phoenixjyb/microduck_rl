"""Post-hoc identity diagnostics; never reorder physics or weaken exact replay.

Some contact constraints have repeated (type, id) keys. Those rows are explicitly
ambiguous, not matched by an invented occurrence number or sorted numerics.
"""

from collections import Counter
import os

import torch

from mjlab_microduck import stance_recovery_early_forward_trace as early
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_forward_probe import CONSTRAINTS

PROTOCOL = "football-b1d-constraint-identity-v1"
FIELDS = ("J", "D", "aref", "force", "state")
FRICTION_TYPE = 1  # Pinned MuJoCo ConstraintType.FRICTION_DOF.
FLAGS = {
    **early.FLAGS,
    "constraint_identity_qualified": False,
    "constraint_order_waived": False,
}


def _hidden():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden constraint identity diagnostic",
    )


def _keys(rows):
    require(type(rows) is dict and set(rows) == set(CONSTRAINTS), "complete row table")
    require(
        torch.is_tensor(rows["id"]) and rows["id"].ndim == 1,
        "one-dimensional row identities",
    )
    count = rows["id"].numel()
    require(count <= 512, "bounded active row table")
    for name, value in rows.items():
        shape = (count, 20) if name == "J" else (count,)
        dtype = torch.int32 if name in ("type", "id", "state") else torch.float32
        require(
            torch.is_tensor(value)
            and value.device.type == "cpu"
            and value.shape == shape
            and value.dtype == dtype
            and torch.isfinite(value).all(),
            "finite typed row field " + name,
        )
    return list(zip(rows["type"].tolist(), rows["id"].tolist()))


def _rows(left, right):
    """Compare one world's tables without changing or pairing duplicate rows."""
    _hidden()
    evidence._owned_tree(left, clone=False)
    evidence._owned_tree(right, clone=False)
    a, b = _keys(left), _keys(right)
    ca, cb = Counter(a), Counter(b)
    duplicates = {}
    for label, counter in (("left", ca), ("right", cb)):
        repeated = sorted((k, count) for k, count in counter.items() if count > 1)
        duplicates[label] = [
            {"key": list(k), "count": count} for k, count in repeated[:16]
        ]
        duplicates[label + "_count"] = len(repeated)
        duplicates[label + "_truncated"] = len(repeated) > 16
    ambiguous = bool(duplicates["left_count"] or duplicates["right_count"])
    same_keys = ca == cb
    unique = sorted(k for k in ca.keys() & cb.keys() if ca[k] == cb[k] == 1)
    ia, ib = {k: i for i, k in enumerate(a)}, {k: i for i, k in enumerate(b)}
    fields = {}
    for name in FIELDS:
        changed, maximum = [], 0.0
        for key in unique:
            x, y = left[name][ia[key]], right[name][ib[key]]
            if not torch.equal(x, y):
                changed.append(list(key))
                maximum = max(maximum, float((x.double() - y.double()).abs().max()))
        fields[name] = dict(
            exact=None if ambiguous else bool(same_keys and not changed),
            compared_unique_rows=len(unique),
            differing_unique_rows=len(changed),
            max_abs_delta=maximum,
            first_differing_keys=changed[:8],
            details_truncated=len(changed) > 8,
        )
    return dict(
        raw_order_exact=a == b,
        raw_identity_order_sha256={
            label: early.forward.throughput.tree_hash([list(k) for k in keys])
            for label, keys in (("left", a), ("right", b))
        },
        raw_identity_prefix={
            "left": [list(k) for k in a[:16]],
            "right": [list(k) for k in b[:16]],
        },
        identity_rows={"left": len(a), "right": len(b)},
        key_sets_exact=same_keys,
        duplicate_identities=duplicates,
        comparison_complete=bool(not ambiguous and same_keys),
        fields=fields,
        **FLAGS,
    )


def compare(left, right, declaration, records):
    _hidden()
    prior = early.compare(left, right, declaration, records)
    events = []
    for index, (a, b) in enumerate(zip(left["events"], right["events"])):
        worlds = []
        for world, (x, y) in enumerate(
            zip(a["solved"]["constraints"], b["solved"]["constraints"])
        ):
            friction = []
            for rows, event in ((x, a), (y, b)):
                mask = rows["type"] == FRICTION_TYPE
                require(
                    int(mask.sum())
                    == int(event["solved"]["solver"]["nf"][world])
                    == 14,
                    "actual friction rows bind to retained counter",
                )
                friction.append(
                    {key: value[mask].clone() for key, value in rows.items()}
                )
            worlds.append(
                dict(world=world, all_rows=_rows(x, y), friction_rows=_rows(*friction))
            )
        events.append(
            dict(
                index=index,
                phase=a["phase"],
                step=a["step"],
                raw_event_exact=prior["events"][index]["exact"],
                nf_exact=evidence._equal(
                    a["solved"]["solver"]["nf"], b["solved"]["solver"]["nf"]
                ),
                worlds=worlds,
                dynamics={
                    key: dict(
                        exact=torch.equal(value, b["solved"]["dynamics"][key]),
                        max_abs_delta=float(
                            (value.double() - b["solved"]["dynamics"][key].double())
                            .abs()
                            .max()
                        ),
                    )
                    for key, value in a["solved"]["dynamics"].items()
                },
            )
        )
    return dict(
        protocol=PROTOCOL,
        source=left["source"],
        worlds=left["worlds"],
        earliest_raw_differing_event=prior["earliest_differing_event"],
        events=events,
        **FLAGS,
    )
