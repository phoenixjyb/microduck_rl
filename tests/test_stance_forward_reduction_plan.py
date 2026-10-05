"""Independent structural checks; no compile, kernel or policy execution."""

from collections import Counter
from copy import deepcopy

import pytest

from mjlab_microduck import stance_forward_reduction_plan as planner

REDUCTIONS = ("crb", "subtree_com", "rne_backward")
PARENTS = [0, 0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 1, 11, 12, 13, 14]
LEVELS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2, 7, 11], [1], [0]]
GROUPS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2], [7], [11], [1], [0]]


def topology():
    return {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": list(PARENTS),
        "reversed_body_tree_ids": deepcopy(LEVELS),
    }


def run(reduction):
    return planner.plan(topology(), planner.TOPOLOGY_SHA256, reduction)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_literal_groups_and_root_activity_are_distinct(reduction):
    result = run(reduction)
    assert [row["body_ids"] for row in result["launch_groups"]] == GROUPS
    root_child = result["launch_groups"][7]
    world = result["launch_groups"][8]
    assert world["active_body_ids"] == [] and world["noop_body_ids"] == [0]
    if reduction == "crb":
        assert root_child["active_body_ids"] == []
        assert root_child["noop_body_ids"] == [1]
        assert root_child["write_targets"] == []
        assert result["levels"][5]["target_multiplicities"] == []
    else:
        assert root_child["active_body_ids"] == [1]
        assert root_child["noop_body_ids"] == []
        assert root_child["write_targets"] == [0]
        assert result["levels"][5]["target_multiplicities"] == [
            {"parent_id": 0, "body_count": 1}
        ]
    assert all(result["witnesses"].values())
    assert all(value is False for value in result["flags"].values())


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_independently_recompute_every_structural_witness(reduction):
    result = run(reduction)
    groups = result["launch_groups"]
    seen, positions = [], {}
    for index, group in enumerate(groups):
        assert group["launch_index"] == index
        bodies = group["body_ids"]
        expected_active = [
            body
            for body in bodies
            if (PARENTS[body] != 0 if reduction == "crb" else body != 0)
        ]
        assert group["active_body_ids"] == expected_active
        assert group["noop_body_ids"] == [
            body for body in bodies if body not in expected_active
        ]
        assert group["parent_ids"] == [PARENTS[body] for body in bodies]
        assert group["write_targets"] == [PARENTS[body] for body in expected_active]
        assert len(set(group["write_targets"])) == len(group["write_targets"])
        assert group["dim"] == [64, len(bodies)]
        seen.extend(bodies)
        positions.update({body: index for body in bodies})
    assert seen == [body for level in LEVELS for body in level]
    assert sorted(seen) == list(range(16)) and len(set(seen)) == 16
    for body in seen:
        if PARENTS[body] != 0 if reduction == "crb" else body != 0:
            assert positions[body] < positions[PARENTS[body]]
    for index, level in enumerate(LEVELS):
        active = [
            body
            for body in level
            if (PARENTS[body] != 0 if reduction == "crb" else body != 0)
        ]
        counts = Counter(PARENTS[body] for body in active)
        expected_min = max([1, *counts.values()])
        row = result["levels"][index]
        selected = [group for group in groups if group["stage_index"] == index]
        assert [body for group in selected for body in group["body_ids"]] == level
        assert row["body_tree_ids"] == level
        assert row["minimum_group_count"] == expected_min == len(selected)
        assert row["target_multiplicities"] == [
            {"parent_id": parent, "body_count": count}
            for parent, count in sorted(counts.items())
        ]
        assert row["launch_group_indices"] == [
            group["launch_index"] for group in selected
        ]


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_detached_inputs_outputs_and_other_plans(reduction):
    original = topology()
    untouched = deepcopy(original)
    result = planner.plan(original, planner.TOPOLOGY_SHA256, reduction)
    assert original == untouched
    original["body_parentid"][1] = 9
    assert result["launch_groups"][7]["parent_ids"] == [0]
    result["launch_groups"][0]["body_ids"][0] = 99
    result["kernel_contract"]["symbol"] = "changed"
    second = run(reduction)
    assert second["launch_groups"][0]["body_ids"] == [6, 15]
    assert second["kernel_contract"]["symbol"] != "changed"


@pytest.mark.parametrize(
    "invalid", [None, True, 0, {}, [], "CRB", "rne", "", "crb" * 1000]
)
def test_invalid_reduction_rejected_before_topology_traversal(monkeypatch, invalid):
    def forbidden(*_args):
        raise AssertionError("must reject reduction before authenticating topology")

    monkeypatch.setattr(planner.topology_contract, "plan", forbidden)
    with pytest.raises(ValueError, match="fixed reduction name"):
        planner.plan(None, None, invalid)


@pytest.mark.parametrize(
    "mutation", ["bool", "extra", "huge", "changed-parent", "changed-level"]
)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_malformed_topology_rejected_for_each_activity_rule(mutation, reduction):
    value = topology()
    if mutation == "bool":
        value["worlds"] = True
    elif mutation == "extra":
        value["unexpected"] = 1
    elif mutation == "huge":
        value["reversed_body_tree_ids"][0] *= 1000
    elif mutation == "changed-parent":
        value["body_parentid"][1] = 1
    else:
        value["reversed_body_tree_ids"][0].reverse()
    with pytest.raises(ValueError):
        planner.plan(value, planner.TOPOLOGY_SHA256, reduction)


@pytest.mark.parametrize("digest", [None, True, 0, "a" * 64, "0" * 64])
def test_bad_or_unbound_topology_digest_rejected(digest):
    with pytest.raises(ValueError):
        planner.plan(topology(), digest, "subtree_com")


def test_kernel_symbols_and_expected_source_pin_are_declarations_only():
    assert {run(name)["kernel_contract"]["symbol"] for name in REDUCTIONS} == {
        "_crb_accumulate",
        "_subtree_com_acc",
        "_cfrc_backward",
    }
    assert all(
        run(name)["kernel_contract"]["expected_smooth_sha256"] == planner.SMOOTH_SHA256
        for name in REDUCTIONS
    )
    assert all(
        not run(name)["flags"]["fresh_installed_source_authenticated"]
        for name in REDUCTIONS
    )


def test_oversized_reduction_rejected_before_dictionary_lookup(monkeypatch):
    class ForbiddenLookup(dict):
        def __contains__(self, _key):
            raise AssertionError("oversized name must not reach dictionary hashing")

    monkeypatch.setattr(planner, "_RULES", ForbiddenLookup())
    with pytest.raises(ValueError, match="fixed reduction name"):
        planner.plan(None, None, "crb" * 1000)
