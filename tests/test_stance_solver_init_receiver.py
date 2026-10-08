"""Pure CPU validation of passive solver-init packets."""

from copy import deepcopy
from hashlib import sha256

import pytest
from math import prod

from mjlab_microduck import stance_solver_init_control as control
from mjlab_microduck import stance_solver_init_receiver as receiver


def _device():
    return {"context": 44, "stream": 55}


def _layout(name, shape, host_shape, dtype, logical, pointer, object_id):
    width = 1 if dtype == "bool" else 4
    trailing = host_shape[len(logical) :]
    width *= prod(trailing or (1,))
    strides, running = [], width
    for dimension in reversed(logical):
        strides.insert(0, running)
        running *= dimension
    size = prod(host_shape) * (1 if dtype == "bool" else 4)
    base_dtype = {
        "float32": "<class 'warp._src.types.float32'>",
        "int32": "<class 'warp._src.types.int32'>",
        "bool": "<class 'warp._src.types.bool'>",
    }[dtype]
    vector_dtype = {
        ("float32", (2,)): "<class 'warp._src.types.vec2f'>",
        ("float32", (3,)): "<class 'warp._src.types.vec3f'>",
        ("float32", (5,)): "<class 'mujoco_warp._src.types.vec5f'>",
        ("float32", (3, 3)): "<class 'warp._src.types.mat33f'>",
        ("int32", (2,)): "<class 'warp._src.types.vec2i'>",
    }
    return {
        "object_id": object_id,
        "pointer": pointer if size else None,
        "span": size,
        "device": "cuda:0",
        "context": 44,
        "warp_dtype": vector_dtype.get(
            (dtype, tuple(host_shape[len(logical) :])), base_dtype
        ),
        "shape": list(logical),
        "strides": strides,
        "host_shape": list(host_shape),
        "host_dtype": dtype,
        "bytes": size,
    }


def _fixture():
    order, specs = control.SOLVER_INIT_ORDER, control.SOLVER_INIT_SPECS
    chunks, fields, layouts, carriers = [], {}, {}, {}
    offset, pointer = 0, 4096
    for object_id, name in enumerate(order, 1000):
        logical, host_shape, dtype = specs[name]
        size = prod(host_shape) * (1 if dtype == "bool" else 4)
        wire_dtype = {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype]
        raw = bytes(size)
        if name == "data.qacc":
            raw = bytes.fromhex("000000800000c07f") + raw[8:]
        fields[name] = {
            "offset": offset,
            "bytes": size,
            "shape": list(host_shape),
            "dtype": wire_dtype,
        }
        chunks.append(raw)
        carriers[name] = {"shape": list(host_shape), "dtype": wire_dtype, "raw": raw}
        layout = _layout(name, logical, host_shape, dtype, logical, pointer, object_id)
        layouts[name] = layout
        pointer += max(size, 64) + 64
        offset += size
    packet_raw = b"".join(chunks)
    call_ids = {
        "init_function_id": 10,
        "init_code_id": 11,
        "solve_function_id": 12,
        "solve_code_id": 13,
        "solve_public_function_id": 14,
        "solve_public_code_id": 15,
        "solve_body_code_id": 16,
        "forward_function_id": 17,
        "forward_code_id": 18,
        "forward_body_code_id": 19,
    }
    calls, friction_entries, scope_calls = [], [], []
    for index in range(21):
        model_id, data_id = 100 + index, 200 + index
        calls.append(
            {
                "forward": index,
                "arm": "candidate0",
                "grad": True,
                "model_object_id": model_id,
                "data_object_id": data_id,
                "context_object_id": 300 + index,
                **call_ids,
                "stream": 55,
                "device_context": 44,
                "phase": "pre-search" if index == 4 else "unsampled-init-context",
            }
        )
        friction_entries.append(
            {
                "identities": {"model_object_id": model_id, "data_object_id": data_id},
                "stream": 55,
            }
        )
        scope_calls.append({"model_id": model_id, "data_id": data_id})
    receipt = {
        "protocol": control.PROTOCOL,
        "arm": "candidate0",
        "calls": calls,
        "snapshot": {
            "path": "solver-init/candidate0/forward-04.initialized.bin",
            "bytes": len(packet_raw),
            "sha256": sha256(packet_raw).hexdigest(),
            "fields": fields,
            "layouts": layouts,
            "forward": 4,
            "phase": "initialized-before-search",
            "context_object_id": 304,
        },
        "packet_count": 1,
        "captured_bytes": len(packet_raw),
        "source_identity": {
            **call_ids,
            "solver_source_sha256": control.SOLVER_SOURCE_SHA256,
            "solver_wrapper_source_sha256": control.SOLVER_WRAPPER_SOURCE_SHA256,
            "forward_source_sha256": control.FORWARD_SOURCE_SHA256,
        },
        "recipe": deepcopy(control.SOLVER_RECIPE),
        "flags": dict(control.FLAGS),
    }
    boundary_contact = {name: layouts[name] for name in receiver.CONTACT_PACKET_ORDER}
    boundary_complete = {name: layouts[name] for name in receiver.COMPLETE_ORDER}
    contact_packet = {name: carriers[name] for name in receiver.CONTACT_PACKET_ORDER}
    complete_packet = {name: carriers[name] for name in receiver.COMPLETE_ORDER}
    boundary = {
        "packet_fields": {
            4: {"contact_after": contact_packet, "complete": complete_packet}
        },
    }
    boundary_record = {
        "entries": [
            {},
            {},
            {},
            {},
            {
                "contact": boundary_contact,
                "complete_layouts": boundary_complete,
            },
        ]
    }
    recipe_record = {"control_scope": {"calls": scope_calls}}
    raw = {receipt["snapshot"]["path"]: packet_raw}
    return receipt, raw, friction_entries, boundary, boundary_record, recipe_record


def test_capture_plan_keeps_contact_plan_and_adds_solver_init_exactly():
    plan = receiver.expected_capture_plan()
    assert plan["protocol"] == receiver.BOUNDARY_CONTROL_PROTOCOL
    assert plan["solver_init"] == {
        "protocol": control.PROTOCOL,
        "forward": 4,
        "phase": "initialized-before-search",
        "packet_count_per_arm": 1,
        "max_bytes_per_arm": 8 * 1024**2,
        "max_bytes_all_arms": 24 * 1024**2,
        "order": list(control.SOLVER_INIT_ORDER),
        "flags": dict(control.FLAGS),
    }


def test_solver_init_receipt_accepts_complete_authenticated_cpu_fixture():
    receipt, raw, friction, boundary, boundary_record, recipe = _fixture()
    result = receiver.solver_init_snapshot(
        receipt,
        raw,
        "candidate0",
        _device(),
        friction,
        boundary,
        boundary_record,
        recipe,
    )
    assert len(result["fields"]) == len(control.SOLVER_INIT_ORDER)
    assert result["fields"]["context.done"]["dtype"] == "|b1"
    assert result["fields"]["context.done"]["raw"] == bytes(64)


@pytest.mark.parametrize(
    "damage",
    (
        "schema",
        "hash",
        "grad",
        "schedule",
        "layout",
        "dtype",
        "cap",
        "recipe",
        "source",
    ),
)
def test_solver_init_receipt_rejects_adversarial_schema_and_bindings(
    monkeypatch, damage
):
    receipt, raw, friction, boundary, boundary_record, recipe = _fixture()
    if damage == "schema":
        receipt["unexpected"] = None
    elif damage == "hash":
        raw[receipt["snapshot"]["path"]] += b"x"
    elif damage == "grad":
        receipt["calls"][4]["grad"] = False
    elif damage == "schedule":
        receipt["calls"][4]["forward"] = 3
    elif damage == "layout":
        receipt["snapshot"]["layouts"]["data.qpos"]["host_shape"] = [64, 20]
    elif damage == "dtype":
        receipt["snapshot"]["fields"]["context.done"]["dtype"] = "<i4"
    elif damage == "cap":
        monkeypatch.setattr(receiver, "MAX_SOLVER_INIT_ARM_BYTES", 1)
    elif damage == "recipe":
        receipt["recipe"]["cone"] = 1
    else:
        receipt["source_identity"]["solve_public_code_id"] += 1
    with pytest.raises(ValueError):
        receiver.solver_init_snapshot(
            receipt,
            raw,
            "candidate0",
            _device(),
            friction,
            boundary,
            boundary_record,
            recipe,
        )


def test_solver_init_packet_preserves_signed_zero_nan_and_boolean_wire_layout():
    receipt, raw, friction, boundary, boundary_record, recipe = _fixture()
    result = receiver.solver_init_snapshot(
        receipt,
        raw,
        "candidate0",
        _device(),
        friction,
        boundary,
        boundary_record,
        recipe,
    )
    assert result["fields"]["context.done"]["dtype"] == "|b1"
    assert result["fields"]["data.qacc"]["raw"][:4].hex() == "00000080"
    assert result["fields"]["data.qacc"]["raw"][4:8].hex() == "0000c07f"


def test_layout_accepts_only_singleton_zero_stride_broadcast():
    scalar = {"shape": [1], "dtype": "<f4", "raw": bytes(4)}
    layout = _layout("scalar", (1,), (1,), "float32", (1,), 4096, 10)
    layout["strides"] = [0]
    assert receiver.layout_solver_init(layout, scalar, _device(), (1,), "float32") == (
        4096,
        4100,
    )

    vector = {"shape": [2], "dtype": "<f4", "raw": bytes(8)}
    layout = _layout("vector", (2,), (2,), "float32", (2,), 8192, 11)
    layout["strides"] = [0]
    with pytest.raises(ValueError, match="singleton-broadcast"):
        receiver.layout_solver_init(layout, vector, _device(), (2,), "float32")


def test_candidate_repeat_uses_literal_fields_and_backlink_row_helper():
    arms = {}
    for arm in ("candidate0", "candidate1"):
        receipt, raw, friction, boundary, boundary_record, recipe = _fixture()
        receipt["arm"] = arm
        for call in receipt["calls"]:
            call["arm"] = arm
        old_path = receipt["snapshot"]["path"]
        new_path = old_path.replace("candidate0", arm)
        receipt["snapshot"]["path"] = new_path
        raw[new_path] = raw.pop(old_path)
        arms[arm] = receiver.solver_init_snapshot(
            receipt, raw, arm, _device(), friction, boundary, boundary_record, recipe
        )
    report = receiver.compare_solver_init(arms)
    assert report["candidate0_candidate1_non_efc_literal_repeat_exact"] is True
    assert report["candidate0_candidate1_efc_full_carrier_raw_storage_only"] is True
    assert (
        report["candidate0_candidate1_efc_backlink_row_view"][
            "active_row_count_per_arm"
        ]
        == 0
    )
    assert all(value is False for value in report["flags"].values())
