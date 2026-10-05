"""Bounded declarative partition plan for the exact stance CRB topology.

This module only plans body-ID groups. It does not adapt or launch the kernel,
and it does not qualify an observed result or infer an original kernel cause.
"""

from hashlib import sha256
import json
import re

from mjlab_microduck import stance_crb_kernel_repeat as repeat

PROTOCOL = "football-b1d-crb-level-plan-v1"
TOPOLOGY_SHA256 = "2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19"
WORLDS = 64
BODIES = 16
EXPECTED_PARENTS = (0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14)
EXPECTED_LEVELS = (
    (6, 15),
    (5, 10, 14),
    (4, 9, 13),
    (3, 8, 12),
    (2, 7, 11),
    (1,),
    (0,),
)
EXPECTED_GROUPS = (
    ((6, 15),),
    ((5, 10, 14),),
    ((4, 9, 13),),
    ((3, 8, 12),),
    ((2,), (7,), (11,)),
    ((1,),),
    ((0,),),
)
_FIELDS = frozenset(
    {
        "nbody",
        "nq",
        "nv",
        "nu",
        "worlds",
        "body_parentid",
        "reversed_body_tree_ids",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TOPOLOGY_BYTES = 4096


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _bounded_topology(topology):
    """Validate primitive types and maximum sizes before hashing or traversal."""
    _need(
        type(topology) is dict and len(topology) == 7, "exact bounded topology object"
    )
    _need(all(type(key) is str for key in topology), "plain string topology keys")
    _need(frozenset(topology) == _FIELDS, "exact seven topology fields")
    for key, expected in (
        ("nbody", BODIES),
        ("nq", 21),
        ("nv", 20),
        ("nu", 14),
        ("worlds", WORLDS),
    ):
        _need(
            type(topology[key]) is int and topology[key] == expected, f"topology {key}"
        )

    parents = topology["body_parentid"]
    _need(
        type(parents) is list and len(parents) == BODIES, "bounded body parent vector"
    )
    _need(
        all(type(parent) is int and 0 <= parent < BODIES for parent in parents),
        "plain in-range body parent IDs",
    )

    levels = topology["reversed_body_tree_ids"]
    _need(
        type(levels) is list and len(levels) == len(EXPECTED_LEVELS),
        "bounded reversed level count",
    )
    total_ids = 0
    for level in levels:
        _need(
            type(level) is list and 0 < len(level) <= BODIES,
            "bounded nonempty body level",
        )
        total_ids += len(level)
        _need(total_ids <= BODIES, "bounded total level body IDs")
        _need(
            all(type(body) is int and 0 <= body < BODIES for body in level),
            "plain in-range level body IDs",
        )
    _need(total_ids == BODIES, "all topology body IDs are present")

    return tuple(parents), tuple(tuple(level) for level in levels)


def _validate_authenticated_topology(topology, parents, levels):
    """Apply the frozen semantic topology contract after digest authentication."""
    # The primitive schema is already bounded at no more than 16 IDs in either
    # collection, so the retained validator cannot traverse excess data here.
    repeat._validate_topology(topology)
    _need(tuple(parents) == EXPECTED_PARENTS, "exact authenticated body parent vector")
    _need(
        levels == EXPECTED_LEVELS,
        "exact authenticated reversed body levels",
    )
    flattened = [body for level in levels for body in level]
    _need(
        sorted(flattened) == list(range(BODIES)), "topology levels partition body IDs"
    )


def _canonical_topology_bytes(topology):
    try:
        encoded = (
            json.dumps(topology, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical topology encoding") from exc
    _need(len(encoded) <= _MAX_TOPOLOGY_BYTES, "canonical topology byte cap")
    return encoded


def _partition_level(body_ids, parents):
    """Greedily place stable body IDs into minimum parent-conflict rounds."""
    groups = []
    target_occurrences = {}
    for body in body_ids:
        parent = parents[body]
        if parent == 0:
            round_index = 0
        else:
            round_index = target_occurrences.get(parent, 0)
            target_occurrences[parent] = round_index + 1
        while len(groups) <= round_index:
            groups.append([])
        groups[round_index].append(body)
    return groups


def plan(topology: dict, topology_sha256: str) -> dict:
    """Return detached groups and witnesses for the pinned topology only."""
    parents, levels = _bounded_topology(topology)
    _need(
        type(topology_sha256) is str and _SHA256.fullmatch(topology_sha256) is not None,
        "lowercase topology SHA-256",
    )
    canonical_raw = _canonical_topology_bytes(topology)
    actual_sha256 = sha256(canonical_raw).hexdigest()
    _need(
        topology_sha256 == TOPOLOGY_SHA256 and actual_sha256 == TOPOLOGY_SHA256,
        "exact canonical authenticated topology SHA-256",
    )
    _validate_authenticated_topology(topology, parents, levels)

    launch_groups = []
    level_rows = []
    for stage_index, level in enumerate(levels):
        groups = _partition_level(level, parents)
        _need(
            tuple(tuple(group) for group in groups) == EXPECTED_GROUPS[stage_index],
            "minimum stable unique-parent level partition",
        )
        group_indices = []
        for group_index, group in enumerate(groups):
            body_ids = list(group)
            parent_ids = [parents[body] for body in group]
            atomic_targets = [parent for parent in parent_ids if parent != 0]
            noops = [
                body
                for body, parent in zip(body_ids, parent_ids, strict=True)
                if parent == 0
            ]
            group_indices.append(len(launch_groups))
            launch_groups.append(
                {
                    "launch_index": len(launch_groups),
                    "stage_index": stage_index,
                    "group_index": group_index,
                    "body_ids": body_ids,
                    "parent_ids": parent_ids,
                    "nonzero_parent_targets": atomic_targets,
                    "parent_zero_noop_body_ids": noops,
                    "dim": [WORLDS, len(body_ids)],
                }
            )
        level_rows.append(
            {
                "stage_index": stage_index,
                "body_tree_ids": list(level),
                "launch_group_indices": group_indices,
                "nonzero_parent_target_multiplicities": _target_multiplicities(
                    level, parents
                ),
                "minimum_group_count": _minimum_groups(level, parents),
            }
        )

    body_launch = {}
    body_order = []
    for group in launch_groups:
        for body in group["body_ids"]:
            _need(body not in body_launch, "planned body appears only once")
            body_launch[body] = group["launch_index"]
            body_order.append(body)
    _need(set(body_launch) == set(range(BODIES)), "plan covers every body once")
    child_before_parent = all(
        parent == 0 or body_launch[body] < body_launch[parent]
        for body, parent in enumerate(parents)
    )
    _need(child_before_parent, "every non-world child launches before its parent")

    witnesses = {
        "topology_digest_matches_pinned": actual_sha256 == TOPOLOGY_SHA256,
        "body_coverage_exactly_once": True,
        "stable_body_id_order_within_groups": all(
            group["body_ids"] == sorted(group["body_ids"]) for group in launch_groups
        ),
        "unique_nonzero_parent_targets_per_group": all(
            len(group["nonzero_parent_targets"])
            == len(set(group["nonzero_parent_targets"]))
            for group in launch_groups
        ),
        "child_before_parent": child_before_parent,
        "per_level_group_count_is_minimal": all(
            len(level["launch_group_indices"]) == level["minimum_group_count"]
            for level in level_rows
        ),
        "parent_zero_noops_retained": [
            body
            for group in launch_groups
            for body in group["parent_zero_noop_body_ids"]
        ]
        == [1, 0],
        "expected_launch_group_count": len(launch_groups) == 9,
        "covered_body_ids_in_stable_level_order": body_order
        == [body for level in levels for body in level],
        "unchanged_original_level_boundaries": all(
            [
                body
                for group in launch_groups
                if group["stage_index"] == stage_index
                for body in group["body_ids"]
            ]
            == list(level)
            for stage_index, level in enumerate(levels)
        ),
        "launch_stages_monotonic": all(
            left["stage_index"] <= right["stage_index"]
            for left, right in zip(launch_groups, launch_groups[1:])
        ),
    }
    _need(all(witnesses.values()), "all plan witness checks")
    return {
        "protocol": PROTOCOL,
        "topology_sha256": actual_sha256,
        "worlds": WORLDS,
        "levels": level_rows,
        "launch_groups": launch_groups,
        "witnesses": witnesses,
        "flags": {
            "fresh_cpu_compile_performed": False,
            "actual_kernel_launch_performed": False,
            "actual_kernel_order_observed": False,
            "original_pair_accepted": False,
            "cause_proven": False,
            "full_window_passed": False,
            "training_authorized": False,
            "physical_result_accepted": False,
        },
    }


def _minimum_groups(body_ids, parents):
    counts = _target_multiplicities(body_ids, parents)
    return max([1, *(row["body_count"] for row in counts)])


def _target_multiplicities(body_ids, parents):
    counts = {}
    for body in body_ids:
        parent = parents[body]
        if parent != 0:
            counts[parent] = counts.get(parent, 0) + 1
    return [
        {"parent_id": parent, "body_count": counts[parent]} for parent in sorted(counts)
    ]
