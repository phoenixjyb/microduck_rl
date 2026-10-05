"""Map persistent inertia report coordinates onto the compiled stance model.

This is a CPU-only diagnostic view.  It preserves the observer's raw-bit
comparison and does not claim solver ordering, a kernel cause, or admission.
"""

from hashlib import sha256
import os

import mujoco
import torch

from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_recovery_early_inertia_trace as inertia
from mjlab_microduck import stance_warp_runtime
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-stance-inertia-component-map-v1"
SAMPLE_LIMIT = 32
FLAGS = {
    **inertia.FLAGS,
    "component_map_qualified": False,
    "reduction_order_cause_proven": False,
}
_BODY_FIELDS = {"subtree_com", "cinert", "crb", "cvel"}
_DOF_FIELDS = {"cdof", "cdof_dot", "qfrc_bias", "qfrc_smooth"}
_MATRIX_FIELDS = {"qM", "qLD"}


def _hidden():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden component map reader",
    )


def _joint_dof_count(joint_type):
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def _body_names(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        for index in range(model.nbody)
    ]


def _joint_names(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or ""
        for index in range(model.njnt)
    ]


def _topology(model):
    require(
        (model.nbody, model.nq, model.nv, model.nu) == (16, 21, 20, 14),
        "exact compiled inertia component topology",
    )
    body_names = _body_names(model)
    joint_names = _joint_names(model)
    require(
        len(body_names) == model.nbody
        and all(type(name) is str and name for name in body_names)
        and len(set(body_names)) == model.nbody
        and len(joint_names) == model.njnt
        and all(type(name) is str and name for name in joint_names)
        and len(set(joint_names)) == model.njnt,
        "compiled body and joint names are unique",
    )
    parents = [int(value) for value in model.body_parentid]
    require(
        len(parents) == model.nbody
        and parents[0] == 0
        and all(0 <= parent < model.nbody for parent in parents),
        "compiled body parent ranges",
    )
    bodies = []
    for index, parent in enumerate(parents):
        path = []
        seen = set()
        cursor = index
        while True:
            require(cursor not in seen, "compiled body ancestry is acyclic")
            seen.add(cursor)
            path.append(cursor)
            next_parent = parents[cursor]
            if cursor == 0:
                require(next_parent == 0, "compiled world body is root")
                break
            cursor = next_parent
        path.reverse()
        bodies.append(
            {
                "index": index,
                "name": body_names[index],
                "parent_index": parent,
                "parent_name": body_names[parent],
                "ancestry": [
                    {"index": ancestor, "name": body_names[ancestor]}
                    for ancestor in path
                ],
            }
        )

    jnt_dofadr = [int(value) for value in model.jnt_dofadr]
    jnt_types = [int(value) for value in model.jnt_type]
    jnt_bodies = [int(value) for value in model.jnt_bodyid]
    dof_joint = [int(value) for value in model.dof_jntid]
    dof_body = [int(value) for value in model.dof_bodyid]
    require(
        len(dof_joint) == model.nv
        and len(dof_body) == model.nv
        and len(jnt_dofadr) == model.njnt
        and len(jnt_types) == model.njnt
        and len(jnt_bodies) == model.njnt,
        "compiled joint and dof table lengths",
    )
    dofs = []
    for index, (joint_index, body_index) in enumerate(
        zip(dof_joint, dof_body, strict=True)
    ):
        require(
            0 <= joint_index < model.njnt
            and 0 <= body_index < model.nbody
            and body_index == jnt_bodies[joint_index],
            "compiled dof joint and body ranges",
        )
        require(jnt_types[joint_index] in (0, 1, 2, 3), "compiled dof joint type")
        local_offset = index - jnt_dofadr[joint_index]
        require(
            0 <= local_offset < _joint_dof_count(jnt_types[joint_index]),
            "compiled dof local joint offset",
        )
        dofs.append(
            {
                "index": index,
                "body_index": body_index,
                "joint_index": joint_index,
                "joint_name": joint_names[joint_index],
                "joint_type": jnt_types[joint_index],
                "local_offset": local_offset,
                "body_name": body_names[body_index],
                "body_ancestry": bodies[body_index]["ancestry"],
            }
        )
    require(
        sorted(item["index"] for item in dofs) == list(range(model.nv)),
        "compiled dof indices cover full generalized velocity range",
    )
    return bodies, dofs


def compiled_map():
    """Compile the actual CPU plant and return validated report-coordinate maps."""
    _hidden()
    rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = stance_warp_runtime.build_entity().compile()
    require(
        torch.equal(torch.random.get_rng_state(), rng),
        "component mapping preserves caller RNG",
    )
    bodies, dofs = _topology(model)
    binding = fixture.compiled_binding(model)
    require(
        type(binding) is dict
        and binding["nbody"] == len(bodies)
        and binding["body_names"] == [item["name"] for item in bodies],
        "compiled fixture binding matches body topology",
    )
    payload = {"compiled_plant": binding, "bodies": bodies, "dofs": dofs}
    payload["topology_sha256"] = sha256(canonical(payload).encode("utf-8")).hexdigest()
    return payload


def _bits(value):
    return int(value.contiguous().view(torch.int32).item()) & 0xFFFFFFFF


def _coordinate_map(name, coordinate, bodies, dofs):
    result = {"coordinate": list(coordinate)}
    if name in _BODY_FIELDS:
        body_index = coordinate[1]
        result["body"] = bodies[body_index]
        if len(coordinate) > 2:
            result["component_index"] = coordinate[2]
    elif name in _MATRIX_FIELDS:
        result["row_dof"] = dofs[coordinate[1]]
        result["column_dof"] = dofs[coordinate[2]]
    elif name in _DOF_FIELDS:
        result["dof"] = dofs[coordinate[1]]
        if len(coordinate) > 2:
            result["component_index"] = coordinate[2]
    return result


def _field_diff(name, left, right, bodies, dofs):
    left_bits = left.contiguous().view(torch.int32)
    right_bits = right.contiguous().view(torch.int32)
    changed = left_bits != right_bits
    coords = changed.nonzero(as_tuple=False)
    samples = []
    for row in coords[:SAMPLE_LIMIT].tolist():
        coord = tuple(row)
        a, b = left[coord], right[coord]
        samples.append(
            {
                **_coordinate_map(name, coord, bodies, dofs),
                "world_index": coord[0],
                "left": float(a),
                "right": float(b),
                "left_bits": f"0x{_bits(a):08x}",
                "right_bits": f"0x{_bits(b):08x}",
            }
        )
    world_changed = changed.flatten(start_dim=1).any(dim=1)
    return {
        "exact": not bool(changed.any()),
        "different_elements": int(changed.sum()),
        "differing_worlds": int(world_changed.sum()),
        "max_abs_delta": float((left.double() - right.double()).abs().max()),
        "samples": samples,
        "samples_truncated": int(coords.shape[0]) > SAMPLE_LIMIT,
    }


def compare(left, right, declaration, records, compiled_plant):
    """Return order-sensitive per-event component diagnostics, never qualification."""
    _hidden()
    require(
        type(records) in (tuple, list) and len(records) == 2, "paired component records"
    )
    left_score = inertia.check(left, declaration, records[0])
    right_score = inertia.check(right, declaration, records[1])
    require(
        left_score["source"] == right_score["source"]
        and left_score["worlds"] == right_score["worlds"],
        "paired traces share source and world dimensions",
    )
    prior = inertia.compare(left, right, declaration, records)
    mapping = compiled_map()
    require(
        type(compiled_plant) is dict
        and canonical(compiled_plant) == canonical(mapping["compiled_plant"]),
        "supplied compiled plant matches fresh CPU compilation",
    )
    bodies, dofs = mapping["bodies"], mapping["dofs"]
    event_rows = []
    earliest_prior = next(
        (
            index
            for index, event in enumerate(prior["events"])
            if any(not event["fields"][name]["exact"] for name in inertia.SHAPES)
        ),
        None,
    )
    for index, (a, b) in enumerate(zip(left["events"], right["events"], strict=True)):
        if index != earliest_prior:
            continue
        field_rows = {
            name: _field_diff(
                name, a["persistent"][name], b["persistent"][name], bodies, dofs
            )
            for name in inertia.SHAPES
        }
        event_rows.append(
            {
                "index": index,
                "phase": a["phase"],
                "step": a["step"],
                "fields": field_rows,
                "first_differing_field": next(
                    (name for name in inertia.SHAPES if not field_rows[name]["exact"]),
                    None,
                ),
            }
        )
    earliest = next(
        (
            row["index"]
            for row in event_rows
            if row["first_differing_field"] is not None
        ),
        None,
    )
    return {
        "protocol": PROTOCOL + ":comparison",
        "source": left_score["source"],
        "worlds": left_score["worlds"],
        "events": event_rows,
        "earliest_persistent_differing_event": earliest,
        "original_trace_comparison": prior,
        "compiled_map": mapping,
        "reporting_order_is_not_kernel_temporal_order": True,
        "persistent_exactness_includes_signed_zero": True,
        **FLAGS,
    }
