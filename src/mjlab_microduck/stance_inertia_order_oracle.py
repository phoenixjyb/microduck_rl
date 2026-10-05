"""Float32 arithmetic-order plausibility diagnostics for one mapped CRB node.

This enumerates possible arithmetic parenthesizations from completed-forward
snapshots.  It observes no native execution order and is never a qualification
or kernel-cause oracle.
"""

from itertools import permutations
import os

import torch

from mjlab_microduck import stance_inertia_component_map as component
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-stance-inertia-order-oracle-v1"
ROOT_BODY = 1
ROOT_NAME = "trunk_base"
CHILDREN = ((2, "yaw2roll"), (7, "neck"), (11, "bearing_roll"))
ORDERS = tuple(permutations(tuple(index for index, _ in CHILDREN)))
FLAGS = {
    **component.FLAGS,
    "addition_order_qualified": False,
    "addition_order_cause_proven": False,
}


def _hidden():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden inertia order oracle",
    )


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def _bits_nested(tensor):
    return [[f"0x{int(value):08x}" for value in row] for row in _bits(tensor).tolist()]


def _candidate(root_cinert, child_crb, order):
    """Replay exactly three sequential CPU float32 additions in the given order."""
    require(
        torch.is_tensor(root_cinert)
        and root_cinert.device.type == "cpu"
        and root_cinert.dtype == torch.float32
        and root_cinert.ndim == 2
        and root_cinert.shape[1] == 10
        and bool(torch.isfinite(root_cinert).all()),
        "finite CPU float32 root cinert matrix",
    )
    require(
        type(child_crb) is dict
        and set(child_crb) == {index for index, _ in CHILDREN}
        and type(order) is tuple
        and len(order) == 3
        and set(order) == set(child_crb),
        "complete direct-child order",
    )
    for child, tensor in child_crb.items():
        require(
            torch.is_tensor(tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tuple(tensor.shape) == tuple(root_cinert.shape)
            and bool(torch.isfinite(tensor).all()),
            f"finite CPU float32 child CRB {child}",
        )
    result = root_cinert.detach().clone()
    for child in order:
        result.add_(child_crb[child])
        require(
            bool(torch.isfinite(result).all()),
            "float32 candidate remains finite after each addition",
        )
    return result


def _summary(candidate, observed):
    candidate_bits, observed_bits = _bits(candidate), _bits(observed)
    differing = candidate_bits != observed_bits
    per_world = differing.any(dim=1)
    return {
        "exact_elements": int((~differing).sum()),
        "different_elements": int(differing.sum()),
        "differing_worlds": int(per_world.sum()),
        "max_abs_delta": float((candidate.double() - observed.double()).abs().max()),
    }


def _membership(observed, candidates):
    observed_bits = _bits(observed)
    candidate_bits = [_bits(value) for value in candidates]
    rows = []
    for world in range(observed.shape[0]):
        for component_index in range(observed.shape[1]):
            capture_mask = sum(
                1 << order_index
                for order_index, values in enumerate(candidate_bits)
                if values[world, component_index]
                == observed_bits[world, component_index]
            )
            rows.append(
                {
                    "world": world,
                    "component_index": component_index,
                    "capture_membership_mask": capture_mask,
                    "capture_matching_orders": [
                        index
                        for index in range(len(ORDERS))
                        if capture_mask & (1 << index)
                    ],
                }
            )
    return rows


def _node_binding(compiled_map):
    bodies = compiled_map["bodies"]
    binding = compiled_map["compiled_plant"]
    require(
        type(binding) is dict
        and binding["nbody"] == 16
        and binding["body_names"] == [body["name"] for body in bodies]
        and binding["selected_plant"]["topology"][:3] == [21, 20, 14]
        and len(bodies) == 16
        and bodies[ROOT_BODY]["name"] == ROOT_NAME
        and bodies[ROOT_BODY]["index"] == ROOT_BODY,
        "compiled trunk-base root index and name",
    )
    root = bodies[ROOT_BODY]
    require(
        root["parent_index"] == 0
        and root["parent_name"] == "world"
        and root["ancestry"][-2:]
        == [
            {"index": 0, "name": "world"},
            {"index": ROOT_BODY, "name": ROOT_NAME},
        ],
        "compiled trunk-base root parent and ancestry",
    )
    direct = [body for body in bodies if body["parent_index"] == ROOT_BODY]
    expected = {index: name for index, name in CHILDREN}
    require(
        {body["index"]: body["name"] for body in direct} == expected,
        "compiled trunk-base has exact direct-child topology",
    )
    require(
        all(
            bodies[index]["parent_index"] == ROOT_BODY
            and bodies[index]["parent_name"] == ROOT_NAME
            and bodies[index]["ancestry"][-2:]
            == [
                {"index": ROOT_BODY, "name": ROOT_NAME},
                {"index": index, "name": name},
            ]
            for index, name in CHILDREN
        ),
        "compiled direct-child parent and ancestry bindings",
    )
    return {
        "root": bodies[ROOT_BODY],
        "children": [bodies[index] for index, _ in CHILDREN],
    }


def compare(left, right, declaration, records, compiled_plant):
    """Enumerate all child-add orders for the earliest retained step-zero event."""
    _hidden()
    comparison = component.compare(left, right, declaration, records, compiled_plant)
    require(
        comparison["worlds"] in (2, 64),
        "two-world CPU test or full 64-world retained trace",
    )
    node = _node_binding(comparison["compiled_map"])
    full = comparison["original_trace_comparison"]
    require(
        full["earliest_differing_event"] == 0
        and comparison["earliest_persistent_differing_event"] == 0,
        "earliest retained difference is first event",
    )
    left_event, right_event = left["events"][0], right["events"][0]
    require(
        left_event["phase"] == right_event["phase"] == "scheduled-pre"
        and left_event["step"] == right_event["step"] == 0,
        "first event is scheduled-pre step zero",
    )
    left_fields = left_event["persistent"]
    right_fields = right_event["persistent"]
    left_root = left_fields["cinert"][:, ROOT_BODY, :].contiguous()
    right_root = right_fields["cinert"][:, ROOT_BODY, :].contiguous()
    require(
        torch.equal(_bits(left_root), _bits(right_root)),
        "root cinert premise is raw-bit identical across pair",
    )
    left_children = {
        index: left_fields["crb"][:, index, :].contiguous() for index, _ in CHILDREN
    }
    right_children = {
        index: right_fields["crb"][:, index, :].contiguous() for index, _ in CHILDREN
    }
    require(
        all(
            torch.equal(_bits(left_children[index]), _bits(right_children[index]))
            for index, _ in CHILDREN
        ),
        "all direct-child CRB premises are raw-bit identical across pair",
    )
    capture_root = left_fields["crb"][:, ROOT_BODY, :].contiguous()
    replay_root = right_fields["crb"][:, ROOT_BODY, :].contiguous()
    candidates = [_candidate(left_root, left_children, order) for order in ORDERS]
    order_rows = []
    for order, candidate in zip(ORDERS, candidates, strict=True):
        order_rows.append(
            {
                "order": list(order),
                "candidate_values": candidate.tolist(),
                "candidate_uint32_bits": _bits_nested(candidate),
                "capture": _summary(candidate, capture_root),
                "replay": _summary(candidate, replay_root),
            }
        )
    capture_membership = _membership(capture_root, candidates)
    replay_membership = _membership(replay_root, candidates)
    cell_membership = [
        {
            "world": capture["world"],
            "component_index": capture["component_index"],
            "capture_membership_mask": capture["capture_membership_mask"],
            "capture_matching_orders": capture["capture_matching_orders"],
            "replay_membership_mask": replay["capture_membership_mask"],
            "replay_matching_orders": replay["capture_matching_orders"],
        }
        for capture, replay in zip(capture_membership, replay_membership, strict=True)
    ]
    capture_bits = _bits(capture_root)
    replay_bits = _bits(replay_root)
    observed_differences = capture_bits != replay_bits
    observed_world_differences = observed_differences.any(dim=1)
    capture_unmatched = sum(
        row["capture_membership_mask"] == 0 for row in capture_membership
    )
    replay_unmatched = sum(
        row["capture_membership_mask"] == 0 for row in replay_membership
    )
    capture_world_matches = sum(
        bool(torch.equal(capture_bits[world], candidate_bits[world]))
        for world in range(capture_bits.shape[0])
        for candidate_bits in (_bits(value) for value in candidates)
    )
    replay_world_matches = sum(
        bool(torch.equal(replay_bits[world], candidate_bits[world]))
        for world in range(replay_bits.shape[0])
        for candidate_bits in (_bits(value) for value in candidates)
    )
    return {
        "protocol": PROTOCOL + ":comparison",
        "source": comparison["source"],
        "worlds": comparison["worlds"],
        "event": {"index": 0, "phase": "scheduled-pre", "step": 0},
        "node_binding": node,
        "compiled_topology_sha256": comparison["compiled_map"]["topology_sha256"],
        "root_cinert_values": left_root.tolist(),
        "root_cinert_uint32_bits": _bits_nested(left_root),
        "child_crb": [
            {
                "body_index": index,
                "body_name": name,
                "values": left_children[index].tolist(),
                "uint32_bits": _bits_nested(left_children[index]),
            }
            for index, name in CHILDREN
        ],
        "capture_observed_root_crb": {
            "values": capture_root.tolist(),
            "uint32_bits": _bits_nested(capture_root),
        },
        "replay_observed_root_crb": {
            "values": replay_root.tolist(),
            "uint32_bits": _bits_nested(replay_root),
        },
        "orders": order_rows,
        "cell_membership": cell_membership,
        "membership_summary": {
            "capture_unmatched_elements": int(capture_unmatched),
            "replay_unmatched_elements": int(replay_unmatched),
            "different_observed_elements": int(observed_differences.sum()),
            "differing_observed_worlds": int(observed_world_differences.sum()),
            "capture_complete_root_world_matches": capture_world_matches,
            "replay_complete_root_world_matches": replay_world_matches,
            "all_observed_cells_have_a_candidate": (
                capture_unmatched == 0 and replay_unmatched == 0
            ),
            "interpretation": "arithmetic-membership summary only; no winner or admission",
        },
        "full_comparison": comparison,
        "interpretation": (
            "complete-forward snapshots support arithmetic plausibility only; "
            "reported order is not actual kernel order"
        ),
        "actual_kernel_order_observed": False,
        "order_membership_is_not_selection": True,
        "order_membership_claims_actual_execution": False,
        **FLAGS,
    }
