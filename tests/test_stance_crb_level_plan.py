"""CUDA-hidden structural tests for the exact-topology CRB level planner."""

import hashlib
import json

import pytest

from mjlab_microduck import stance_crb_level_plan as planner
from mjlab_microduck import stance_crb_kernel_repeat as repeat


def _topology():
    return {
        "nbody": 16,
        "nq": 21,
        "nv": 20,
        "nu": 14,
        "worlds": 64,
        "body_parentid": list(planner.EXPECTED_PARENTS),
        "reversed_body_tree_ids": [list(level) for level in planner.EXPECTED_LEVELS],
    }


def _digest(topology):
    raw = (
        json.dumps(topology, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode()
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_exact_plan_preserves_levels_splits_parent_collisions_and_keeps_noops():
    topology = _topology()
    result = planner.plan(topology, _digest(topology))
    expected = [
        [6, 15],
        [5, 10, 14],
        [4, 9, 13],
        [3, 8, 12],
        [2],
        [7],
        [11],
        [1],
        [0],
    ]
    assert [group["body_ids"] for group in result["launch_groups"]] == expected
    assert result["topology_sha256"] == planner.TOPOLOGY_SHA256
    assert len(result["launch_groups"]) == 9
    assert result["launch_groups"][4]["nonzero_parent_targets"] == [1]
    assert result["launch_groups"][5]["nonzero_parent_targets"] == [1]
    assert result["launch_groups"][6]["nonzero_parent_targets"] == [1]
    assert result["launch_groups"][7]["parent_zero_noop_body_ids"] == [1]
    assert result["launch_groups"][8]["parent_zero_noop_body_ids"] == [0]
    assert result["levels"][4]["minimum_group_count"] == 3
    assert result["levels"][4]["nonzero_parent_target_multiplicities"] == [
        {"parent_id": 1, "body_count": 3}
    ]
    assert all(result["witnesses"].values())
    assert all(value is False for value in result["flags"].values())


def test_returned_witnesses_recompute_independently():
    topology = _topology()
    result = planner.plan(topology, _digest(topology))
    parents = topology["body_parentid"]
    groups = result["launch_groups"]

    flattened = [body for group in groups for body in group["body_ids"]]
    assert sorted(flattened) == list(range(16))
    assert len(flattened) == len(set(flattened))
    body_launch = {
        body: group["launch_index"] for group in groups for body in group["body_ids"]
    }
    for group in groups:
        body_ids = group["body_ids"]
        parent_ids = [parents[body] for body in body_ids]
        assert body_ids == sorted(body_ids)
        assert group["parent_ids"] == parent_ids
        nonzero = [parent for parent in parent_ids if parent != 0]
        assert len(nonzero) == len(set(nonzero))
        assert group["nonzero_parent_targets"] == nonzero
        assert group["parent_zero_noop_body_ids"] == [
            body
            for body, parent in zip(body_ids, parent_ids, strict=True)
            if parent == 0
        ]
    for body, parent in enumerate(parents):
        if parent != 0:
            assert body_launch[body] < body_launch[parent]
    for stage_index, level in enumerate(result["levels"]):
        ids = [
            body
            for group in groups
            if group["stage_index"] == stage_index
            for body in group["body_ids"]
        ]
        assert ids == level["body_tree_ids"]
    assert all(
        left["stage_index"] <= right["stage_index"]
        for left, right in zip(groups, groups[1:])
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-field",
        "extra-field",
        "bool-worlds",
        "wrong-dimension",
        "short-parents",
        "bool-parent",
        "changed-parent",
        "short-levels",
        "long-level",
        "bool-body-id",
        "duplicate-body-id",
        "missing-body-id",
        "reordered-level",
    ],
)
def test_invalid_or_unbounded_topology_is_rejected_before_planning(mutation):
    topology = _topology()
    if mutation == "missing-field":
        del topology["nq"]
    elif mutation == "extra-field":
        topology["unused"] = 0
    elif mutation == "bool-worlds":
        topology["worlds"] = True
    elif mutation == "wrong-dimension":
        topology["nbody"] = 17
    elif mutation == "short-parents":
        topology["body_parentid"].pop()
    elif mutation == "bool-parent":
        topology["body_parentid"][2] = True
    elif mutation == "changed-parent":
        topology["body_parentid"][2] = 3
    elif mutation == "short-levels":
        topology["reversed_body_tree_ids"].pop()
    elif mutation == "long-level":
        topology["reversed_body_tree_ids"][0] = list(range(17))
    elif mutation == "bool-body-id":
        topology["reversed_body_tree_ids"][0][0] = True
    elif mutation == "duplicate-body-id":
        topology["reversed_body_tree_ids"][0][1] = 6
    elif mutation == "missing-body-id":
        topology["reversed_body_tree_ids"][0].pop()
    elif mutation == "reordered-level":
        topology["reversed_body_tree_ids"][0].reverse()
    with pytest.raises(ValueError):
        planner.plan(topology, "0" * 64)


@pytest.mark.parametrize("bad_digest", ["", "a" * 63, "G" * 64, "0" * 64, 7, True])
def test_invalid_or_mismatched_topology_hash_is_rejected(bad_digest):
    with pytest.raises(ValueError):
        planner.plan(_topology(), bad_digest)


def test_plan_is_detached_from_input_and_each_other():
    topology = _topology()
    result = planner.plan(topology, _digest(topology))
    topology["reversed_body_tree_ids"][0][0] = 99
    assert result["launch_groups"][0]["body_ids"] == [6, 15]

    topology = _topology()
    result = planner.plan(topology, _digest(topology))
    result["launch_groups"][0]["body_ids"][0] = 99
    assert topology["reversed_body_tree_ids"][0] == [6, 15]

    independent = planner.plan(topology, _digest(topology))
    assert independent["launch_groups"][0]["body_ids"] == [6, 15]


def test_primitive_bounds_reject_before_hash_or_frozen_topology_traversal(monkeypatch):
    topology = _topology()
    topology["body_parentid"][2] = True
    called = []

    def forbidden(*_args):
        called.append(True)
        raise AssertionError(
            "must not hash or validate frozen topology before primitive bounds"
        )

    monkeypatch.setattr(planner, "sha256", forbidden)
    monkeypatch.setattr(repeat, "_validate_topology", forbidden)
    with pytest.raises(ValueError):
        planner.plan(topology, planner.TOPOLOGY_SHA256)
    assert called == []


def test_wrong_digest_rejects_before_frozen_semantic_validation(monkeypatch):
    called = []

    def forbidden(_topology):
        called.append(True)
        raise AssertionError(
            "frozen topology validation must follow hash authentication"
        )

    monkeypatch.setattr(repeat, "_validate_topology", forbidden)
    with pytest.raises(ValueError, match="authenticated topology SHA-256"):
        planner.plan(_topology(), "0" * 64)
    assert called == []
