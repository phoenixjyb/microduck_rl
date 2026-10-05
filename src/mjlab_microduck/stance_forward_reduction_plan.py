"""Source-only single-writer plans; no runtime or numerical qualification."""

from copy import deepcopy

from mjlab_microduck import stance_crb_level_plan as topology_contract

PROTOCOL = "football-b1d-forward-reduction-plan-v1"
SMOOTH_SHA256 = "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
TOPOLOGY_SHA256 = topology_contract.TOPOLOGY_SHA256
_RULES = {
    "crb": {"symbol": "_crb_accumulate", "active_rule": "parent_id != 0"},
    "subtree_com": {"symbol": "_subtree_com_acc", "active_rule": "body_id != 0"},
    "rne_backward": {"symbol": "_cfrc_backward", "active_rule": "body_id != 0"},
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _active(body, parents, reduction):
    return parents[body] != 0 if reduction == "crb" else body != 0


def _partition(level, parents, reduction):
    groups, counts = [], {}
    for body in level:
        target = parents[body]
        if _active(body, parents, reduction):
            round_index = counts.get(target, 0)
            counts[target] = round_index + 1
        else:
            round_index = 0
        while len(groups) <= round_index:
            groups.append([])
        groups[round_index].append(body)
    return groups, counts


def plan(topology, topology_sha256, reduction):
    """Authenticate the fixed topology and derive reduction-specific witnesses."""
    _need(
        type(reduction) is str and 0 < len(reduction) <= 32 and reduction in _RULES,
        "fixed reduction name",
    )
    # Reuse only its exact topology authentication. Do not copy CRB no-op or
    # target-zero semantics from its returned groups/witnesses.
    topology_contract.plan(topology, topology_sha256)
    parents = tuple(topology["body_parentid"])
    levels = tuple(tuple(row) for row in topology["reversed_body_tree_ids"])
    groups, stage_rows = [], []
    for stage_index, level in enumerate(levels):
        partitions, counts = _partition(level, parents, reduction)
        indices = []
        for group_index, bodies in enumerate(partitions):
            active = [body for body in bodies if _active(body, parents, reduction)]
            noops = [body for body in bodies if not _active(body, parents, reduction)]
            targets = [parents[body] for body in active]
            indices.append(len(groups))
            groups.append(
                {
                    "launch_index": len(groups),
                    "stage_index": stage_index,
                    "group_index": group_index,
                    "body_ids": list(bodies),
                    "parent_ids": [parents[body] for body in bodies],
                    "active_body_ids": active,
                    "noop_body_ids": noops,
                    "write_targets": targets,
                    "dim": [64, len(bodies)],
                }
            )
        stage_rows.append(
            {
                "stage_index": stage_index,
                "body_tree_ids": list(level),
                "launch_group_indices": indices,
                "target_multiplicities": [
                    {"parent_id": target, "body_count": count}
                    for target, count in sorted(counts.items())
                ],
                "minimum_group_count": max([1, *counts.values()]),
            }
        )
    body_order = [body for group in groups for body in group["body_ids"]]
    positions = {
        body: group["launch_index"] for group in groups for body in group["body_ids"]
    }
    active_all = [body for group in groups for body in group["active_body_ids"]]
    noop_all = [body for group in groups for body in group["noop_body_ids"]]
    witnesses = {
        "exact_topology_authenticated": True,
        "body_coverage_exactly_once": sorted(body_order) == list(range(16)),
        "unchanged_level_and_body_order": body_order
        == [body for level in levels for body in level],
        "expected_nine_groups": len(groups) == 9,
        "active_children_precede_parents_including_root": all(
            positions[body] < positions[parents[body]] for body in active_all
        ),
        "unique_active_targets_including_zero": all(
            len(group["write_targets"]) == len(set(group["write_targets"]))
            for group in groups
        ),
        "per_level_grouping_is_minimal": all(
            len(row["launch_group_indices"]) == row["minimum_group_count"]
            for row in stage_rows
        ),
        "activity_rule_noops_exact": noop_all
        == ([1, 0] if reduction == "crb" else [0]),
        "root_target_activity_exact": (
            [body for body in active_all if parents[body] == 0]
            == ([] if reduction == "crb" else [1])
        ),
    }
    _need(all(witnesses.values()), "all fixed reduction-plan witnesses")
    return {
        "protocol": PROTOCOL,
        "reduction": reduction,
        "topology_sha256": TOPOLOGY_SHA256,
        "worlds": 64,
        "kernel_contract": {
            **deepcopy(_RULES[reduction]),
            "expected_smooth_sha256": SMOOTH_SHA256,
        },
        "levels": stage_rows,
        "launch_groups": groups,
        "witnesses": witnesses,
        "flags": {
            "fresh_installed_source_authenticated": False,
            "fresh_model_compiled": False,
            "actual_kernel_launch_performed": False,
            "actual_order_observed": False,
            "original_pair_accepted": False,
            "cause_proven": False,
            "full_window_qualified": False,
            "training_authorized": False,
            "physical_result_accepted": False,
        },
    }
