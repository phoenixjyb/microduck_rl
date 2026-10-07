"""CPU-only tests for passive boundary evidence validation."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from mjlab_microduck import stance_contact_boundary_receiver as receiver
from mjlab_microduck import stance_contact_boundary_probe as producer
from test_stance_contact_boundary_control import _environment, _run
from test_stance_friction_prefix_cuda_receiver import compiled_fixture, device
from test_stance_friction_runtime_numerical import _entry, WORLDS, CAPACITY


def _enriched_numerical_entries():
    return {
        arm: [
            {**_entry(), "layouts": {}, "identities": {}, "stream": 123}
            for _ in range(21)
        ]
        for arm in receiver.ARMS
    }


def test_numerical_projection_preserves_metadata_and_raw_carrier_identity():
    entries = _enriched_numerical_entries()
    before = deepcopy(entries)
    projected = receiver.numerical_entry_projection(entries)
    assert entries == before
    for arm in receiver.ARMS:
        for original, numeric in zip(entries[arm], projected[arm], strict=True):
            assert set(numeric) == receiver.numerical.ENTRY_FIELDS
            assert set(original) - set(numeric) == {"layouts", "identities", "stream"}
            for key in receiver.numerical.ENTRY_FIELDS:
                assert numeric[key] is original[key]
            assert numeric["inputs"]["qvel"]["raw"] is original["inputs"]["qvel"]["raw"]
    report = receiver.numerical.audit_entries(
        projected, worlds=WORLDS, capacity=CAPACITY, forwards=21
    )
    assert entries == before
    assert report["component_exact_without_overflow"] is True
    assert all(value is False for value in report["flags"].values())
    with pytest.raises(ValueError, match="exact entry fields"):
        receiver.numerical.audit_entries(
            entries, worlds=WORLDS, capacity=CAPACITY, forwards=21
        )


@pytest.mark.parametrize(
    "damage",
    ("extra-arm", "missing-arm", "row-count", "row-type", "extra-field")
    + tuple("missing-" + key for key in sorted(receiver.numerical.ENTRY_FIELDS))
    + ("missing-layouts", "missing-identities", "missing-stream"),
)
def test_numerical_projection_refuses_unknown_missing_or_malformed_schema(damage):
    entries = _enriched_numerical_entries()
    row = entries["original"][0]
    if damage == "extra-arm":
        entries["extra"] = []
    elif damage == "missing-arm":
        del entries["candidate1"]
    elif damage == "row-count":
        entries["original"].pop()
    elif damage == "row-type":
        entries["original"][0] = None
    elif damage == "extra-field":
        row["extra"] = None
    else:
        del row[damage.removeprefix("missing-")]
    with pytest.raises(ValueError, match="exact (observer|enriched observer)"):
        receiver.numerical_entry_projection(entries)


def test_projection_does_not_relax_numerical_carrier_validation():
    entries = _enriched_numerical_entries()
    entries["original"][0]["inputs"]["qvel"]["raw"] = b"bad"
    projected = receiver.numerical_entry_projection(entries)
    with pytest.raises(ValueError):
        receiver.numerical.audit_entries(
            projected, worlds=WORLDS, capacity=CAPACITY, forwards=21
        )


def test_cpu_receipt_binds_exact_child_thread_settings():
    receiver.cpu_thread_settings({"cpu_test_threads": dict(producer.CPU_TEST_THREADS)})


@pytest.mark.parametrize(
    "settings",
    (
        None,
        {},
        {"OMP_NUM_THREADS": "1"},
        {name: 1 for name in producer.CPU_TEST_THREADS},
        {**producer.CPU_TEST_THREADS, "EXTRA_THREADS": "1"},
    ),
)
def test_cpu_receipt_refuses_missing_partial_or_changed_thread_settings(settings):
    with pytest.raises(ValueError, match="exact CPU-only child thread settings"):
        receiver.cpu_thread_settings({"cpu_test_threads": settings})


def _thread_fixture():
    env = dict(producer.OWNER_THREAD_ENV)
    budget = {"owner": dict(env), "cuda_child": dict(producer.CUDA_CHILD_THREAD_ENV)}
    owner_pid, child_pid = 101, 202

    def snapshot(role, pid, settings, threads):
        return {
            "role": role,
            "pid": pid,
            "settings": dict(settings),
            "threads": threads,
        }

    phases = []
    for index, (phase, role) in enumerate(receiver._expected_thread_phase_order()):
        phases.append(
            {
                "protocol": producer.PROTOCOL + ":phase",
                "phase": phase,
                "role": role,
                "pid": child_pid,
                "threads": 1 + (index % 7),
                "observed_cpu_thread_env": dict(budget["cuda_child"]),
            }
        )
    owner_start = snapshot("owner", owner_pid, budget["owner"], 1)
    child_observations = {
        "pre_import": snapshot("cuda_child", child_pid, budget["cuda_child"], 1),
        "after_warp_init": snapshot("cuda_child", child_pid, budget["cuda_child"], 5),
        "after_recipe": snapshot("cuda_child", child_pid, budget["cuda_child"], 7),
    }
    by_key = {(row["phase"], row["role"]): row for row in phases}
    for field, key in (
        ("pre_import", ("torch-import-start", None)),
        ("after_warp_init", ("warp-init-done", None)),
        ("after_recipe", ("recipe-case-done", "candidate1")),
    ):
        row = by_key[key]
        child_observations[field] = {
            "role": "cuda_child",
            "pid": row["pid"],
            "settings": dict(row["observed_cpu_thread_env"]),
            "threads": row["threads"],
        }
    declaration = {
        "owner_pid": owner_pid,
        "thread_budget": deepcopy(budget),
        "owner_thread_start": deepcopy(owner_start),
    }
    child = {
        "owner_pid": owner_pid,
        "child_pid": child_pid,
        "thread_budget": deepcopy(budget),
        "thread_observations": child_observations,
    }
    owner = {
        "owner_pid": owner_pid,
        "child_pid": child_pid,
        "thread_budget": deepcopy(budget),
        "thread_observations": {
            "pre_import": deepcopy(owner_start),
            "after_child": snapshot("owner", owner_pid, budget["owner"], 3),
        },
    }
    return declaration, child, owner, phases


def _thread_log(phases, *, newline=True):
    prefix = b"Warp initialization output\n"
    suffix = b'{"complete":true}\n'
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        + (b"\n" if newline else b"")
        for row in phases
    ]
    return prefix + b"".join(lines) + suffix


def test_thread_budget_proof_accepts_literal_maps_snapshots_and_all_phases():
    declaration, child, owner, phases = _thread_fixture()
    proof = receiver.thread_budget_proof(declaration, child, owner, _thread_log(phases))

    assert len(phases) == proof["phase_count"] == 40
    assert proof["phase_order"][0] == {"phase": "torch-import-start", "role": None}
    assert proof["phase_order"][-1] == {
        "phase": "recipe-case-done",
        "role": "candidate1",
    }
    assert proof["owner_threads"] == {"pre_import": 1, "after_child": 3}
    assert proof["child_threads"] == {
        "pre_import": 1,
        "after_warp_init": phases[4]["threads"],
        "after_recipe": phases[-1]["threads"],
    }


@pytest.mark.parametrize(
    "damage",
    (
        "budget-map",
        "budget-role",
        "snapshot-role",
        "snapshot-pid",
        "snapshot-count-bool",
        "snapshot-count-high",
        "snapshot-map-type",
        "log-missing",
        "log-extra",
        "log-reordered",
        "log-pid",
        "log-map",
        "log-fields",
        "log-protocol",
        "log-role",
        "log-overlong",
        "log-no-newline",
    ),
)
def test_thread_budget_proof_rejects_budget_snapshot_and_log_corruption(damage):
    declaration, child, owner, phases = _thread_fixture()
    raw_log = _thread_log(phases)
    if damage == "budget-map":
        declaration["thread_budget"]["owner"]["OMP_NUM_THREADS"] = "2"
    elif damage == "budget-role":
        child["thread_budget"]["unexpected"] = {}
    elif damage == "snapshot-role":
        owner["thread_observations"]["after_child"]["role"] = "cuda_child"
    elif damage == "snapshot-pid":
        child["thread_observations"]["after_recipe"]["pid"] += 1
    elif damage == "snapshot-count-bool":
        child["thread_observations"]["after_warp_init"]["threads"] = True
    elif damage == "snapshot-count-high":
        owner["thread_observations"]["after_child"]["threads"] = 65
    elif damage == "snapshot-map-type":
        child["thread_observations"]["pre_import"]["settings"]["MKL_NUM_THREADS"] = 1
    elif damage == "log-missing":
        phases.pop(0)
        raw_log = _thread_log(phases)
    elif damage == "log-extra":
        phases.append(deepcopy(phases[-1]))
        raw_log = _thread_log(phases)
    elif damage == "log-reordered":
        phases[0], phases[1] = phases[1], phases[0]
        raw_log = _thread_log(phases)
    elif damage == "log-pid":
        phases[0]["pid"] += 1
        raw_log = _thread_log(phases)
    elif damage == "log-map":
        phases[2]["observed_cpu_thread_env"]["OPENBLAS_NUM_THREADS"] = "2"
        raw_log = _thread_log(phases)
    elif damage == "log-fields":
        phases[0]["extra"] = True
        raw_log = _thread_log(phases)
    elif damage == "log-protocol":
        phases[0]["protocol"] = "other:phase"
        raw_log = _thread_log(phases)
    elif damage == "log-role":
        phases[1]["role"] = "unexpected"
        raw_log = _thread_log(phases)
    elif damage == "log-overlong":
        phases[0]["padding"] = "x" * 1100
        raw_log = _thread_log(phases)
    else:
        raw_log = _thread_log(phases, newline=False)

    with pytest.raises(ValueError):
        receiver.thread_budget_proof(declaration, child, owner, raw_log)


def _field(raw, shape, dtype="<f4"):
    return {"raw": raw, "shape": list(shape), "dtype": dtype}


def _matrix_frame_fixture():
    # Synthetic CUDA identity labels only; no native/CUDA claim.
    gpu = device()
    carrier = _field(bytes(8192 * 36), (8192, 3, 3))
    value = {
        "object_id": 1,
        "pointer": 4096,
        "span": len(carrier["raw"]),
        "device": "cuda:0",
        "context": gpu["context"],
        "warp_dtype": "<class 'warp._src.types.mat33f'>",
        "shape": [8192],
        "strides": [36],
        "host_shape": [8192, 3, 3],
        "host_dtype": "float32",
        "bytes": len(carrier["raw"]),
    }
    return value, carrier, gpu


def test_original_contact_frame_matrix_layout_has_one_logical_dimension():
    value, carrier, gpu = _matrix_frame_fixture()
    assert receiver.layout(value, carrier, gpu) == (4096, 4096 + 8192 * 36)


@pytest.mark.parametrize(
    "damage",
    ("vec3-view", "matrix-wrong-rank", "scalar-type", "stride", "host-shape", "bytes"),
)
def test_matrix_frame_layout_refuses_reinterpretation_or_malformed_metadata(damage):
    value, carrier, gpu = _matrix_frame_fixture()
    if damage in ("vec3-view", "matrix-wrong-rank"):
        value.update(shape=[8192, 3], strides=[36, 12])
        if damage == "vec3-view":
            value["warp_dtype"] = "<class 'warp._src.types.vec3f'>"
    elif damage == "scalar-type":
        value["warp_dtype"] = "<class 'warp._src.types.float32'>"
    elif damage == "stride":
        value["strides"] = [12]
    elif damage == "host-shape":
        value["host_shape"] = [8192, 9]
    else:
        value["bytes"] -= 4
    with pytest.raises(ValueError):
        receiver.layout(value, carrier, gpu)


def test_unpack_packet_preserves_order_and_float_bits_without_decoding():
    # -0.0 and a quiet NaN payload are deliberately compared as bytes, not floats.
    first = bytes.fromhex("00000080 4523c17f")
    second = bytes.fromhex("0000803f 00000000")
    raw_bytes = first + second
    metadata = {
        "z_first": {"offset": 0, "bytes": 8, "shape": [2], "dtype": "<f4"},
        "a_second": {"offset": 8, "bytes": 8, "shape": [2], "dtype": "<f4"},
    }
    record = {
        "path": "arm/raw.bin",
        "bytes": len(raw_bytes),
        "sha256": sha256(raw_bytes).hexdigest(),
        "fields": metadata,
    }

    decoded = receiver.unpack_packet(
        record, {"arm/raw.bin": raw_bytes}, order=("z_first", "a_second")
    )

    assert list(decoded) == ["z_first", "a_second"]
    assert decoded["z_first"]["raw"] == first
    assert decoded["a_second"]["raw"] == second
    assert struct.unpack("<I", decoded["z_first"]["raw"][:4])[0] == 0x80000000
    assert struct.unpack("<I", decoded["z_first"]["raw"][4:])[0] == 0x7FC12345


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(bytes=7),
        lambda record: record.update(sha256="0" * 64),
        lambda record: record["fields"]["z_first"].update(offset=4),
        lambda record: record["fields"]["z_first"].update(dtype="=f4"),
        lambda record: record["fields"].pop("a_second"),
    ],
)
def test_unpack_packet_rejects_malformed_or_unbound_sections(mutate):
    raw_bytes = bytes(16)
    record = {
        "path": "arm/raw.bin",
        "bytes": len(raw_bytes),
        "sha256": sha256(raw_bytes).hexdigest(),
        "fields": {
            "z_first": {"offset": 0, "bytes": 8, "shape": [2], "dtype": "<f4"},
            "a_second": {"offset": 8, "bytes": 8, "shape": [2], "dtype": "<f4"},
        },
    }
    mutate(record)

    with pytest.raises(ValueError):
        receiver.unpack_packet(
            record, {"arm/raw.bin": raw_bytes}, order=("z_first", "a_second")
        )


def test_inactive_contact_capacity_bits_are_retained_but_not_active_extent():
    left_raw = bytes.fromhex("0000803f 0000c07f 00000080")
    right_raw = bytes.fromhex("0000803f 0100c07f 00000080")
    left = _field(left_raw, (3,))
    right = _field(right_raw, (3,))

    # The full raw carrier differs in inactive slot 1, while the authenticated
    # nacon prefix is one contact and therefore remains byte-identical.
    assert left["raw"] != right["raw"]
    assert receiver._active_segments(
        "contact_before", "contact.dist", left, 1, [0] * 64
    ) == [(0, 4)]
    assert receiver._first_raw_word(left_raw[:4], right_raw[:4]) is None


def test_active_signed_zero_difference_is_reported_as_raw_word():
    left = _field(bytes.fromhex("00000000 0000803f"), (2,))
    right = _field(bytes.fromhex("00000080 0000803f"), (2,))

    segment_left = receiver._active_segments(
        "contact_before", "contact.dist", left, 1, [0] * 64
    )[0]
    segment_right = receiver._active_segments(
        "contact_before", "contact.dist", right, 1, [0] * 64
    )[0]
    delta = receiver._first_raw_word(
        left["raw"][segment_left[0] : sum(segment_left)],
        right["raw"][segment_right[0] : sum(segment_right)],
    )

    assert delta == {
        "word_index": 0,
        "left_word_le_hex": "00000000",
        "right_word_le_hex": "00000080",
    }


def test_active_extent_counts_are_bounded_plain_ints():
    invalid_nacon = _field(struct.pack("<i", 8193), (1,), "<i4")
    invalid_stage = {
        "contact.nacon": invalid_nacon,
        "data.nefc": _field(struct.pack("<64i", *([0] * 64)), (64,), "<i4"),
    }
    with pytest.raises(ValueError):
        receiver._active_first_difference(
            invalid_stage, invalid_stage, "contact_before"
        )
    with pytest.raises(ValueError):
        receiver._int_words(_field(struct.pack("<i", 1), (1,), "<f4"))


def _payload_bank(count=2):
    bank = {
        name: _field(bytes(4 * receiver.prod(shape)), shape, dtype)
        for name, (shape, dtype) in receiver.CONTACT_CARRIERS.items()
    }
    bank["contact.nacon"]["raw"] = struct.pack("<i", count)
    bank["contact.dist"]["raw"] = struct.pack("<2I", 0, 0x80000000) + bytes(
        4 * (receiver.CONTACT_CAPACITY - 2)
    )
    return bank


def _complete_view_bank(world=2, count=4):
    bank = {
        name: _field(bytes(4 * receiver.prod(shape)), shape, dtype)
        for name, (shape, dtype) in receiver.COMPLETE_CARRIERS.items()
    }
    counts = [0] * 64
    counts[world] = count
    bank["data.nefc"]["raw"] = struct.pack("<64i", *counts)
    return bank


def _change_raw_word(bank, name, index, word=b"\x00\x00\x00\x80"):
    raw = bank[name]["raw"]
    bank[name]["raw"] = raw[: index * 4] + word + raw[(index + 1) * 4 :]


def test_coordinate_view_identifies_matrix_slot_and_contact_world_without_mutation():
    left, right = _payload_bank(8), _payload_bank(8)
    index = 7 * 9 + 2 * 3 + 1
    _change_raw_word(right, "context.frame", index)
    before = deepcopy((left, right))
    result = receiver.describe_first_active_difference(left, right, "contact_before")
    assert (left, right) == before
    assert result["shape"] == [8192, 3, 3]
    assert result["row_major_coordinate"] == [7, 2, 1]
    assert result["byte_offset"] == index * 4
    assert result["delta"]["right_word_le_hex"] == "00000080"
    assert result["active_extent"] == {
        "kind": "active-contact-prefix",
        "contact_index": 7,
        "left_count": 8,
        "right_count": 8,
        "left_world": 0,
        "right_world": 0,
    }
    assert result["stale_prior_solver_storage"] is False
    assert not any(result["flags"].values())


@pytest.mark.parametrize("name,stale", (("efc.force", True), ("efc.J", False)))
def test_coordinate_view_identifies_active_world_row_and_stale_solver_storage(
    name, stale
):
    left, right = _complete_view_bank(), _complete_view_bank()
    width = receiver.prod(left[name]["shape"][2:])
    index = (2 * 512 + 3) * width + (7 if width > 1 else 0)
    _change_raw_word(right, name, index)
    result = receiver.describe_first_active_difference(left, right, "complete")
    assert result["row_major_coordinate"] == [2, 3] + ([7] if width > 1 else [])
    assert result["active_extent"] == {
        "kind": "per-world-active-EFC-rows",
        "world": 2,
        "left_count": 4,
        "right_count": 4,
    }
    assert result["stale_prior_solver_storage"] is stale
    assert not any(result["flags"].values())


def test_coordinate_view_corrects_dense_jqvel_phase_without_rewriting_history():
    left, right = _complete_view_bank(), _complete_view_bank()
    _change_raw_word(right, "efc.Jqvel", 2 * 512 + 3)
    before = deepcopy((left, right))
    result = receiver.describe_first_active_difference(left, right, "complete")
    assert result["protocol"] == "microduck-boundary-raw-coordinate-view-oct8-v2"
    assert result["historical_capture_stale_label"] is True
    assert result["stale_prior_solver_storage"] is False
    assert result["recomputed_during_dense_construction"] is True
    assert result["phase_source_sha256"] == receiver.DENSE_CONSTRUCTION_SOURCE_SHA256
    assert receiver.STALE_PRIOR_FIELDS == (
        "efc.force",
        "efc.state",
        "efc.Ma",
        "efc.Jqvel",
    )
    assert (left, right) == before
    assert not any(result["flags"].values())


def _row_view_banks():
    contact = _payload_bank(3)
    complete = _complete_view_bank(world=0, count=4)
    counts = [4, 1] + [0] * 62
    for bank in (contact, complete):
        bank["data.nefc"]["raw"] = struct.pack("<64i", *counts)
    contact["contact.efc_address"]["raw"] = struct.pack("<i", -1) * (8192 * 4)
    for index, (world, dim, kind, block) in enumerate(
        ((0, 3, 1, (0, 1, 2, 3)), (1, 1, 3, (0,)), (0, 3, 2, ()))
    ):
        for name, value in (("worldid", world), ("dim", dim), ("type", kind)):
            _change_raw_word(
                contact, "contact." + name, index, struct.pack("<i", value)
            )
        for offset, row in enumerate(block):
            _change_raw_word(
                contact,
                "contact.efc_address",
                index * 4 + offset,
                struct.pack("<i", row),
            )
            _change_raw_word(
                complete, "efc.id", world * 512 + row, struct.pack("<i", index)
            )
            _change_raw_word(
                complete,
                "efc.type",
                world * 512 + row,
                struct.pack("<i", 5 if dim == 1 else 6),
            )
    return contact, complete


def test_row_view_joins_only_observed_addresses_and_preserves_all_bytes():
    contact, complete = _row_view_banks()
    # Inactive capacity and unused frictionless/non-constraint slots are opaque.
    for index in (5, 6, 7, 8, 9, 10, 11, 12):
        _change_raw_word(contact, "contact.efc_address", index, struct.pack("<i", -999))
    before = deepcopy((contact, complete))
    report = receiver.describe_contact_constraint_rows(contact, complete)
    assert report["joined_row_count"] == 5
    assert report["contacts"][0]["row_indices"] == [0, 1, 2, 3]
    assert report["contacts"][1]["row_indices"] == [0]
    assert report["contacts"][2]["status"] == "no-complete-contact-row-not-read"
    assert report["gpu_eligibility_recomputed"] is False
    assert report["uncaptured_contact_parameters"] == [
        "friction",
        "solref",
        "solreffriction",
        "solimp",
    ]
    assert (contact, complete) == before
    assert not any(report["flags"].values())


@pytest.mark.parametrize("unused", (-1, 0, 499, -999))
def test_row_view_never_reads_addresses_without_observed_complete_rows(unused):
    contact, complete = _row_view_banks()
    for index in range(4):
        _change_raw_word(
            contact, "contact.efc_address", index, struct.pack("<i", unused)
        )
        _change_raw_word(complete, "efc.type", index, struct.pack("<i", 0))
    report = receiver.describe_contact_constraint_rows(contact, complete)
    assert report["contacts"][0]["status"] == "no-complete-contact-row-not-read"
    assert "row_indices" not in report["contacts"][0]
    assert report["joined_row_count"] == 1


def test_row_view_handles_noncontact_prefix_without_reindexing_contact_rows():
    contact, complete = _row_view_banks()
    for bank in (contact, complete):
        _change_raw_word(bank, "data.nefc", 0, struct.pack("<i", 6))
    for row in range(6):
        _change_raw_word(
            complete, "efc.type", row, struct.pack("<i", 0 if row < 2 else 6)
        )
    for offset in range(4):
        _change_raw_word(
            contact, "contact.efc_address", offset, struct.pack("<i", offset + 2)
        )
    report = receiver.describe_contact_constraint_rows(contact, complete)
    assert report["contacts"][0]["row_indices"] == [2, 3, 4, 5]
    assert report["joined_row_count"] == 5


@pytest.mark.parametrize(
    "damage", ("orphan", "extra", "missing", "nonconstraint", "overflow", "capacity")
)
def test_row_view_refuses_contradictory_completed_contact_rows(damage):
    contact, complete = _row_view_banks()
    if damage == "orphan":
        _change_raw_word(complete, "efc.id", 0, struct.pack("<i", 3))
    elif damage == "extra":
        for bank in (contact, complete):
            _change_raw_word(bank, "data.nefc", 0, struct.pack("<i", 5))
        _change_raw_word(complete, "efc.type", 4, struct.pack("<i", 6))
    elif damage == "missing":
        _change_raw_word(complete, "efc.type", 0, struct.pack("<i", 0))
    elif damage == "nonconstraint":
        _change_raw_word(contact, "contact.type", 0, struct.pack("<i", 2))
    elif damage == "overflow":
        for offset in range(4):
            _change_raw_word(
                contact, "contact.efc_address", offset, struct.pack("<i", -1)
            )
    else:
        for bank in (contact, complete):
            _change_raw_word(bank, "data.nefc", 0, struct.pack("<i", 513))
    with pytest.raises(ValueError):
        receiver.describe_contact_constraint_rows(contact, complete)


def _linked_construction_pair():
    left_after, left_complete = _row_view_banks()
    right_after, right_complete = deepcopy((left_after, left_complete))
    # Reorder captured contact slots, but retain separate constraint row offsets.
    for name, carrier in right_after.items():
        if (name.startswith("contact.") and name != "contact.nacon") or name.startswith(
            "context."
        ):
            width = 4 * receiver.prod(carrier["shape"][1:])
            raw = carrier["raw"]
            carrier["raw"] = raw[width : 2 * width] + raw[:width] + raw[2 * width :]
    for bank in (right_after, right_complete):
        counts = [6, 4] + [0] * 62
        bank["data.nefc"]["raw"] = struct.pack("<64i", *counts)
    for index in (*range(6), *range(512, 516)):
        _change_raw_word(right_complete, "efc.type", index, struct.pack("<i", 0))
    for world, left_index, right_index, left_rows, right_rows in (
        (0, 0, 1, range(4), range(2, 6)),
        (1, 1, 0, (0,), (3,)),
    ):
        for ordinal, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
            _change_raw_word(
                right_after,
                "contact.efc_address",
                right_index * 4 + ordinal,
                struct.pack("<i", right_row),
            )
            _change_raw_word(
                right_complete,
                "efc.id",
                world * 512 + right_row,
                struct.pack("<i", right_index),
            )
            _change_raw_word(
                right_complete,
                "efc.type",
                world * 512 + right_row,
                struct.pack("<i", 6 if world == 0 else 5),
            )
            for name in receiver.CONSTRUCTION_ROW_FIELDS:
                width = receiver.prod(left_complete[name]["shape"][2:])
                for component in range(width):
                    word = struct.pack(
                        "<I", 0x3F800000 + world * 10000 + ordinal * 100 + component
                    )
                    _change_raw_word(
                        left_complete,
                        name,
                        (world * 512 + left_row) * width + component,
                        word,
                    )
                    _change_raw_word(
                        right_complete,
                        name,
                        (world * 512 + right_row) * width + component,
                        word,
                    )
    return left_after, left_complete, right_after, right_complete


def test_linked_construction_compares_local_ordinals_at_original_offsets():
    banks = _linked_construction_pair()
    before = deepcopy(banks)
    report = receiver.compare_linked_contact_construction(*banks)
    assert banks == before
    assert len(report["compared_payload_links"]) == 2
    assert len(report["unobserved_payload_links"]) == 1
    assert report["compared_payload_links"][0]["left_index"] == 0
    assert report["compared_payload_links"][0]["right_index"] == 1
    assert report["compared_payload_links"][0]["left_rows"] == [0, 1, 2, 3]
    assert report["compared_payload_links"][0]["right_rows"] == [2, 3, 4, 5]
    assert report["field_comparisons"]["efc.J"]["compared_words"] == 100
    assert report["field_comparisons"]["efc.Jqvel"]["compared_words"] == 5
    assert all(value["exact"] is True for value in report["field_comparisons"].values())
    assert report["fields"] == [
        "efc.J",
        "efc.pos",
        "efc.margin",
        "efc.D",
        "efc.vel",
        "efc.aref",
        "efc.frictionloss",
        "efc.Jqvel",
    ]
    assert report["prior_solver_fields_excluded"] == [
        "efc.force",
        "efc.state",
        "efc.Ma",
    ]
    assert report["phase_source_sha256"] == receiver.DENSE_CONSTRUCTION_SOURCE_SHA256
    assert not any(report["flags"].values())


@pytest.mark.parametrize("name", receiver.CONSTRUCTION_ROW_FIELDS)
def test_linked_construction_reports_raw_delta_with_both_original_offsets(name):
    banks = _linked_construction_pair()
    width = receiver.prod(banks[3][name]["shape"][2:])
    component = 7 if width > 1 else 0
    _change_raw_word(banks[3], name, 4 * width + component)
    before = deepcopy(banks)
    result = receiver.compare_linked_contact_construction(*banks)["field_comparisons"][
        name
    ]
    assert banks == before
    assert result["exact"] is False
    assert result["differing_words"] == 1
    delta = result["first_difference"]
    assert (delta["local_row_ordinal"], delta["component"]) == (2, component)
    assert (delta["left_row"], delta["right_row"]) == (2, 4)
    assert delta["left_byte_offset"] == (2 * width + component) * 4
    assert delta["right_byte_offset"] == (4 * width + component) * 4


@pytest.mark.parametrize(
    "words,exact",
    (
        ((0, 0x80000000), False),
        ((0x7FC12345, 0x7FC12345), True),
        ((0x7FC12345, 0x7FC12346), False),
    ),
)
def test_linked_construction_preserves_signed_zero_and_nan_payload_bits(words, exact):
    banks = _linked_construction_pair()
    for bank, index, word in ((banks[1], 0, words[0]), (banks[3], 2, words[1])):
        _change_raw_word(bank, "efc.Jqvel", index, struct.pack("<I", word))
    result = receiver.compare_linked_contact_construction(*banks)["field_comparisons"][
        "efc.Jqvel"
    ]
    assert result["exact"] is exact


@pytest.mark.parametrize("case", ("empty", "unobserved", "duplicates"))
def test_linked_construction_no_comparisons_is_null_not_vacuous_exact(case):
    after = _payload_bank(0 if case == "empty" else 2)
    complete = _complete_view_bank(count=0)
    if case == "duplicates":
        _change_raw_word(after, "contact.dist", 1, struct.pack("<I", 0))
    report = receiver.compare_linked_contact_construction(
        after, complete, deepcopy(after), deepcopy(complete)
    )
    assert report["compared_payload_links"] == []
    assert all(
        value
        == {
            "compared_words": 0,
            "differing_words": 0,
            "exact": None,
            "first_difference": None,
        }
        for value in report["field_comparisons"].values()
    )
    if case == "duplicates":
        assert len(report["ambiguous_payload_groups"]) == 1
    elif case == "unobserved":
        assert len(report["unobserved_payload_links"]) == 2
        assert all(
            link["left_status"]
            == link["right_status"]
            == "no-complete-contact-row-not-read"
            for link in report["unobserved_payload_links"]
        )


def test_linked_construction_does_not_infer_all_drivers_equal_from_equal_row_fields():
    banks = _linked_construction_pair()
    _change_raw_word(banks[3], "data.qvel", 0)
    report = receiver.compare_linked_contact_construction(*banks)
    assert all(value["exact"] is True for value in report["field_comparisons"].values())
    assert report["captured_state_inputs_exact"] == {
        "data.qpos": True,
        "data.qvel": False,
        "data.ctrl": True,
    }
    assert report["all_construction_drivers_asserted_equal"] is False


@pytest.mark.parametrize("damage", ("model", "backlink", "unknown"))
def test_linked_construction_refuses_differing_model_or_malformed_banks(damage):
    banks = _linked_construction_pair()
    if damage == "model":
        _change_raw_word(banks[2], "model.body_weldid", 0, struct.pack("<i", 1))
    elif damage == "backlink":
        _change_raw_word(banks[3], "efc.id", 2, struct.pack("<i", 99))
    else:
        banks[3]["extra"] = banks[3]["efc.id"]
    before = deepcopy(banks)
    with pytest.raises(ValueError):
        receiver.compare_linked_contact_construction(*banks)
    assert banks == before


def _post_forward_load_pair():
    a, ac, b, bc = _linked_construction_pair()
    loads = []
    for complete in (ac, bc):
        load = {
            name: deepcopy(complete[name])
            if name in complete
            else _field(bytes(4 * receiver.prod(shape)), shape, dtype)
            for name, (shape, dtype) in receiver.SOLVED_LOAD_CARRIERS.items()
        }
        loads.append(load)
    return a, ac, loads[0], b, bc, loads[1]


@pytest.mark.parametrize("forward,call", receiver.POST_FORWARD_LOAD_PAIRS)
def test_load_view_keeps_phase_separate_from_stale_construction_force(forward, call):
    banks = _post_forward_load_pair()
    _change_raw_word(banks[1], "efc.force", 0, struct.pack("<I", 0x7FC12345))
    _change_raw_word(banks[4], "efc.force", 2, struct.pack("<I", 0x80000000))
    before = deepcopy(banks)
    report = receiver.compare_linked_post_forward_load(
        *banks, forward=forward, load_call=call
    )
    assert banks == before
    assert report["forward"] == forward and report["load_call"] == call
    assert report["observed_linked_force_words"] == 5
    assert report["force_exact"] is True
    assert report["aggregate_qfrc_constraint"]["contact_isolated"] is False
    assert report["active_id_type_count_continuity_checked"] is True
    assert len(report["compared_payload_links"]) == 2
    assert "not construction efc.force" in report["force_phase"]
    assert not any(report["flags"].values())


def test_load_view_force_delta_retains_both_original_row_offsets():
    banks = _post_forward_load_pair()
    _change_raw_word(banks[5], "efc.force", 4, struct.pack("<I", 0x80000000))
    report = receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)
    assert report["force_exact"] is False
    assert report["differing_force_words"] == 1
    delta = report["first_force_difference"]
    assert (delta["left_row"], delta["right_row"], delta["local_row_ordinal"]) == (
        2,
        4,
        2,
    )
    assert (delta["left_byte_offset"], delta["right_byte_offset"]) == (8, 16)
    assert delta["left_word_le_hex"] == "00000000"
    assert delta["right_word_le_hex"] == "00000080"


def test_load_view_aggregate_load_delta_is_not_contact_isolated():
    banks = _post_forward_load_pair()
    _change_raw_word(banks[5], "data.qfrc_constraint", 3 * 20 + 6)
    result = receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)
    assert result["force_exact"] is True
    aggregate = result["aggregate_qfrc_constraint"]
    assert aggregate["compared_words"] == 1280
    assert aggregate["differing_words"] == 1 and aggregate["exact"] is False
    assert aggregate["first_difference"]["world"] == 3
    assert aggregate["first_difference"]["dof"] == 6
    assert aggregate["contact_isolated"] is False


@pytest.mark.parametrize(
    "words,exact",
    (
        ((0, 0x80000000), False),
        ((0x7FC12345, 0x7FC12345), True),
        ((0x7FC12345, 0x7FC12346), False),
    ),
)
def test_load_view_compares_opaque_force_words_without_float_normalization(
    words, exact
):
    banks = _post_forward_load_pair()
    for bank, index, word in ((banks[2], 0, words[0]), (banks[5], 2, words[1])):
        _change_raw_word(bank, "efc.force", index, struct.pack("<I", word))
    report = receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)
    assert report["force_exact"] is exact


@pytest.mark.parametrize(
    "forward,call",
    (
        (-2, -1),
        (1, 0),
        (3, 1),
        (5, 2),
        (7, 3),
        (8, 4),
        (4, 3),
        (True, 0),
        (0, False),
        (0.0, 0),
    ),
)
def test_load_view_refuses_unsampled_or_nonliteral_stage_pairs(forward, call):
    with pytest.raises(ValueError, match="literal predeclared"):
        receiver.compare_linked_post_forward_load(
            *_post_forward_load_pair(), forward=forward, load_call=call
        )


@pytest.mark.parametrize("arm", (2, 5))
@pytest.mark.parametrize("field", ("data.nefc", "efc.id", "efc.type"))
def test_load_view_refuses_any_active_count_id_type_discontinuity(arm, field):
    banks = _post_forward_load_pair()
    _change_raw_word(
        banks[arm],
        field,
        0 if arm == 2 or field == "data.nefc" else 2,
        struct.pack("<i", 99),
    )
    before = deepcopy(banks)
    with pytest.raises(ValueError):
        receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)
    assert banks == before


def test_load_view_ignores_inactive_id_type_force_capacity_and_preserves_it():
    banks = _post_forward_load_pair()
    for field in ("efc.id", "efc.type", "efc.force"):
        _change_raw_word(banks[5], field, 500, struct.pack("<I", 0x7FC12345))
    before = deepcopy(banks)
    result = receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)
    assert result["force_exact"] is True and banks == before


@pytest.mark.parametrize(
    "damage", ("missing", "unknown", "shape", "boolshape", "dtype", "raw")
)
def test_load_view_refuses_malformed_load_carriers(damage):
    banks = _post_forward_load_pair()
    load = banks[5]
    if damage == "missing":
        del load["efc.force"]
    elif damage == "unknown":
        load["extra"] = load["efc.force"]
    elif damage in ("shape", "boolshape"):
        load["efc.force"]["shape"] = [True if damage == "boolshape" else 63, 512]
    elif damage == "dtype":
        load["efc.force"]["dtype"] = "=f4"
    else:
        load["efc.force"]["raw"] = b"short"
    with pytest.raises(ValueError):
        receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)


@pytest.mark.parametrize("case", ("empty", "unobserved", "duplicates"))
def test_load_view_zero_compared_force_words_is_null(case):
    after = _payload_bank(0 if case == "empty" else 2)
    complete = _complete_view_bank(count=0)
    if case == "duplicates":
        _change_raw_word(after, "contact.dist", 1, struct.pack("<I", 0))
    load = {
        name: _field(bytes(4 * receiver.prod(shape)), shape, dtype)
        for name, (shape, dtype) in receiver.SOLVED_LOAD_CARRIERS.items()
    }
    result = receiver.compare_linked_post_forward_load(
        after,
        complete,
        load,
        deepcopy(after),
        deepcopy(complete),
        deepcopy(load),
        forward=0,
        load_call=0,
    )
    assert result["observed_linked_force_words"] == 0
    assert result["force_exact"] is None and result["first_force_difference"] is None
    assert result["compared_payload_links"] == []


@pytest.mark.parametrize("all_contacts", (False, True))
def test_load_view_one_sided_observed_rows_are_not_compared(all_contacts):
    banks = _post_forward_load_pair()
    indices = [2, 3, 4, 5] + ([512 + 3] if all_contacts else [])
    for index in indices:
        for bank in (banks[4], banks[5]):
            _change_raw_word(bank, "efc.type", index, struct.pack("<i", 0))
    result = receiver.compare_linked_post_forward_load(*banks, forward=4, load_call=2)
    assert result["observed_linked_force_words"] == (0 if all_contacts else 1)
    assert result["force_exact"] is (None if all_contacts else True)
    assert (
        result["unobserved_payload_links"][0]["left_status"]
        == "observed-address-backlink"
    )
    assert (
        result["unobserved_payload_links"][0]["right_status"]
        == "no-complete-contact-row-not-read"
    )
    assert result["snapshot_provenance_established_by_helper"] is False


@pytest.mark.parametrize(
    "damage",
    (
        "count",
        "world",
        "condim",
        "partial",
        "negative",
        "extent",
        "order",
        "id",
        "type",
        "duplicate",
        "nefc",
        "schema",
    ),
)
def test_row_view_refuses_malformed_or_ambiguous_storage_association(damage):
    contact, complete = _row_view_banks()
    if damage == "count":
        _change_raw_word(contact, "contact.nacon", 0, struct.pack("<i", 8193))
    elif damage in ("world", "condim"):
        _change_raw_word(
            contact,
            "contact." + ("worldid" if damage == "world" else "dim"),
            0,
            struct.pack("<i", 64 if damage == "world" else 6),
        )
    elif damage in ("partial", "negative", "extent", "order"):
        _change_raw_word(
            contact,
            "contact.efc_address",
            1,
            struct.pack(
                "<i", {"partial": -1, "negative": -2, "extent": 4, "order": 2}[damage]
            ),
        )
    elif damage in ("id", "type"):
        _change_raw_word(complete, "efc." + damage, 0, struct.pack("<i", 99))
    elif damage == "duplicate":
        _change_raw_word(contact, "contact.worldid", 1, struct.pack("<i", 0))
    elif damage == "nefc":
        _change_raw_word(complete, "data.nefc", 0, struct.pack("<i", 5))
    else:
        del contact["context.pos"]
    before = deepcopy((contact, complete))
    with pytest.raises(ValueError):
        receiver.describe_contact_constraint_rows(contact, complete)
    assert (contact, complete) == before


@pytest.mark.parametrize("stage", ("contact_before", "contact_after", "complete"))
def test_coordinate_view_does_not_promote_inactive_capacity_differences(stage):
    if stage == "complete":
        left, right = _complete_view_bank(), _complete_view_bank()
        _change_raw_word(right, "efc.force", 2 * 512 + 4)
    else:
        left, right = _payload_bank(), _payload_bank()
        _change_raw_word(right, "contact.dist", 2)
    assert receiver.describe_first_active_difference(left, right, stage) is None


@pytest.mark.parametrize("stage", ("contact_before", "complete"))
def test_coordinate_view_count_mismatch_identifies_count_not_arbitrary_row(stage):
    if stage == "complete":
        left, right = _complete_view_bank(count=4), _complete_view_bank(count=5)
        expected = [2]
    else:
        left, right = _payload_bank(1), _payload_bank(2)
        expected = [0]
    result = receiver.describe_first_active_difference(left, right, stage)
    assert result["row_major_coordinate"] == expected
    assert result["delta"]["extent_counts_equal"] is False
    assert (
        result["active_extent"]["right_count"] - result["active_extent"]["left_count"]
        == 1
    )


@pytest.mark.parametrize("stage", (None, True, "solver", "contact"))
def test_coordinate_view_refuses_unknown_stage(stage):
    with pytest.raises(ValueError, match="literal boundary stage"):
        receiver.describe_first_active_difference(
            _payload_bank(), _payload_bank(), stage
        )


@pytest.mark.parametrize(
    "damage", ("unknown", "missing", "shape", "raw", "count", "world")
)
def test_coordinate_view_refuses_malformed_or_out_of_range_banks(damage):
    left, right = _payload_bank(), _payload_bank()
    if damage == "unknown":
        right["extra"] = right["contact.dist"]
    elif damage == "missing":
        del right["context.pos"]
    elif damage == "shape":
        right["context.frame"]["shape"] = [8192, 3]
    elif damage == "raw":
        right["context.frame"]["raw"] = b"bad"
    elif damage == "count":
        right["contact.nacon"]["raw"] = struct.pack("<i", 8193)
    else:
        _change_raw_word(right, "contact.dist", 0, b"\x01\x00\x00\x00")
        _change_raw_word(right, "contact.worldid", 0, struct.pack("<i", 64))
    with pytest.raises(ValueError):
        receiver.describe_first_active_difference(left, right, "contact_before")


def test_contact_payload_view_links_unique_byte_payloads_without_mutation():
    left = _payload_bank()
    right = deepcopy(left)
    raw = right["contact.dist"]["raw"]
    right["contact.dist"]["raw"] = raw[4:8] + raw[:4] + raw[8:]
    before = deepcopy((left, right))
    report = receiver.compare_active_contact_payloads(left, right)
    assert (left, right) == before
    assert report["record_order_exact"] is False
    assert report["payload_multiset_equal"] is True
    assert report["unique_payload_links"] == [
        {"left_index": 0, "right_index": 1},
        {"left_index": 1, "right_index": 0},
    ]
    assert report["ambiguous_payload_groups"] == []
    assert report["output_fields_compared"] is False
    assert not any(report["flags"].values())


def test_duplicate_contact_payloads_are_ambiguous_and_never_arbitrarily_paired():
    left = _payload_bank()
    left["contact.dist"]["raw"] = bytes(4 * receiver.CONTACT_CAPACITY)
    report = receiver.compare_active_contact_payloads(left, deepcopy(left))
    assert report["payload_multiset_equal"] is True
    assert report["unique_payload_links"] == []
    assert len(report["ambiguous_payload_groups"]) == 1
    assert report["ambiguous_payload_groups"][0]["left_indices"] == [0, 1]
    assert report["ambiguous_payload_groups"][0]["right_indices"] == [0, 1]


@pytest.mark.parametrize(
    "name",
    tuple(
        key
        for key in receiver.CONTACT_INPUT_ORDER
        if key.startswith("contact.") and key != "contact.nacon"
    )
    + receiver.CONTACT_CONTEXT_ORDER,
)
def test_contact_payload_links_require_every_input_and_context_field(name):
    left, right = _payload_bank(), _payload_bank()
    raw = right[name]["raw"]
    right[name]["raw"] = bytes([raw[0] ^ 1]) + raw[1:]
    report = receiver.compare_active_contact_payloads(left, right)
    assert report["payload_multiset_equal"] is False
    assert len(report["unmatched_payload_groups"]) == 2


def test_contact_payload_view_does_not_pair_missing_or_changed_multiplicity():
    left, right = _payload_bank(), _payload_bank(1)
    report = receiver.compare_active_contact_payloads(left, right)
    assert report["contact_counts"] == {"left": 2, "right": 1}
    assert report["payload_multiset_equal"] is False
    assert report["unique_payload_links"] == [{"left_index": 0, "right_index": 0}]
    assert len(report["unmatched_payload_groups"]) == 1


def test_contact_payload_view_preserves_nan_bits_and_refuses_numeric_equivalence():
    left, right = _payload_bank(), _payload_bank()
    left["contact.dist"]["raw"] = (
        bytes.fromhex("0100c07f") + left["contact.dist"]["raw"][4:]
    )
    right["contact.dist"]["raw"] = (
        bytes.fromhex("0200c07f") + right["contact.dist"]["raw"][4:]
    )
    report = receiver.compare_active_contact_payloads(left, right)
    assert report["payload_multiset_equal"] is False
    assert len(report["unmatched_payload_groups"]) == 2


def test_contact_payload_view_separates_inactive_capacity_outputs_and_model_inputs():
    left, right = _payload_bank(), _payload_bank()
    raw = right["contact.dist"]["raw"]
    right["contact.dist"]["raw"] = raw[:-4] + bytes.fromhex("0100c07f")
    right["contact.efc_address"]["raw"] = (
        bytes.fromhex("01000000") + right["contact.efc_address"]["raw"][4:]
    )
    right["model.body_weldid"]["raw"] = (
        bytes.fromhex("01000000") + right["model.body_weldid"]["raw"][4:]
    )
    report = receiver.compare_active_contact_payloads(left, right)
    assert report["record_order_exact"] is True
    assert report["payload_multiset_equal"] is True
    assert report["model_inputs_exact"] is False
    assert report["output_fields_compared"] is False


@pytest.mark.parametrize("count", (0, 1, 8192))
def test_contact_payload_view_bounds_empty_single_and_full_active_prefix(count):
    bank = _payload_bank(count)
    report = receiver.compare_active_contact_payloads(bank, deepcopy(bank))
    assert report["contact_counts"] == {"left": count, "right": count}
    assert report["record_order_exact"] and report["payload_multiset_equal"]


@pytest.mark.parametrize(
    "damage",
    (
        "extra",
        "missing",
        "shape-bool",
        "shape",
        "dtype",
        "raw-type",
        "raw-size",
        "count-negative",
        "count-overflow",
    ),
)
def test_contact_payload_view_rejects_malformed_or_unbounded_carriers(damage):
    left, right = _payload_bank(), _payload_bank()
    carrier = right["contact.dist"]
    if damage == "extra":
        right["unknown"] = carrier
    elif damage == "missing":
        del right["context.frame"]
    elif damage == "shape-bool":
        carrier["shape"] = [True]
    elif damage == "shape":
        carrier["shape"] = [1]
    elif damage == "dtype":
        carrier["dtype"] = "<i4"
    elif damage == "raw-type":
        carrier["raw"] = bytearray(carrier["raw"])
    elif damage == "raw-size":
        carrier["raw"] = carrier["raw"][:-4]
    else:
        right["contact.nacon"]["raw"] = struct.pack(
            "<i", -1 if damage == "count-negative" else 8193
        )
    with pytest.raises(ValueError):
        receiver.compare_active_contact_payloads(left, right)


def test_capture_plan_is_literal_bounded_and_non_admitting():
    plan = receiver.expected_capture_plan()

    assert plan["forwards"] == list(range(7))
    assert plan["packet_count_per_arm"] == 21
    assert plan["max_bytes_per_arm"] == 80 * 1024**2
    assert plan["max_bytes_all_arms"] == 240 * 1024**2
    assert plan["phase"] == "construction-complete-BEFORE-solver"
    assert plan["contact_input_order"] == list(receiver.CONTACT_INPUT_ORDER)
    assert plan["contact_output_order"] == list(receiver.CONTACT_OUTPUT_ORDER)
    assert set(plan["flags"].values()) == {False}


def test_compiled_contact_refuses_missing_explicit_role():
    with pytest.raises(ValueError):
        receiver.compiled_contact({}, {}, {}, "/tmp/run", {"arch": 120})


def _contact_compiled_fixture():
    compiled, raw, inventory, directory = compiled_fixture()
    contact = deepcopy(compiled["original"])
    role_root = Path(directory) / "compiled-contact"
    symbol = "_efc_contact_init__locals__kernel_45be4f8f_cuda_kernel_forward"
    files = {
        "generated_source": ("contact.cu", b"// _efc_contact_init generated source"),
        "binary": ("contact.cubin", b"\x7fELFcontact"),
        "metadata": ("contact.meta", json.dumps({symbol + "_smem_bytes": 0}).encode()),
    }
    for kind, (name, body) in files.items():
        relative = "compiled-contact/" + name
        raw[relative] = body
        inventory[relative] = {"bytes": len(body), "sha256": sha256(body).hexdigest()}
        contact[kind] = {
            "path": str(role_root / name),
            "bytes": len(body),
            "sha256": sha256(body).hexdigest(),
        }

    ids = {
        "kernel": 501,
        "module": 502,
        "device": 10,
        "executable": 503,
        "hooks": 504,
    }
    binding = contact["binding"]
    binding["symbol"] = symbol
    binding["module_hash"] = "f" * 64
    binding["observed_object_ids"] = ids
    for kind in ("binary", "metadata"):
        for suffix in ("path", "bytes", "sha256"):
            binding[kind + "_" + suffix] = contact[kind][suffix]
    contact["module_options"] = {
        "block_dim": 256,
        "enable_backward": False,
        "strip_hash": False,
    }
    contact["output_arch"] = 120
    contact["explicit_load"] = {
        "module_object_id": ids["module"],
        "returned_executable_id": ids["executable"],
        "device_object_id": ids["device"],
        "block_dim": 256,
        "binary_path": contact["binary"]["path"],
        "meta_path": contact["metadata"]["path"],
        "output_arch": 120,
        "fresh_cache_before": True,
    }
    return contact, raw, inventory, directory


def test_compiled_contact_validates_complete_explicit_role():
    value, raw, inventory, directory = _contact_compiled_fixture()

    result = receiver.compiled_contact(value, raw, inventory, directory, device())

    assert result["kernel_object_id"] == 501
    assert result["module_object_id"] == 502
    assert result["runtime_entries"] == value["runtime_entries"]


@pytest.mark.parametrize(
    "mutation",
    ("load", "metadata", "identity", "cubin", "options", "path", "flags", "symbol"),
)
def test_compiled_contact_refuses_role_mismatches(mutation):
    value, raw, inventory, directory = _contact_compiled_fixture()
    if mutation == "load":
        value["explicit_load"]["fresh_cache_before"] = False
    elif mutation == "metadata":
        name = "compiled-contact/contact.meta"
        raw[name] = json.dumps({"wrong_cuda_kernel_forward_smem_bytes": 0}).encode()
    elif mutation == "identity":
        value["binding"]["observed_object_ids"]["module"] += 1
    elif mutation == "cubin":
        raw["compiled-contact/contact.cubin"] = b"notELF"
    elif mutation == "options":
        value["module_options"]["enable_backward"] = True
    elif mutation == "path":
        value["generated_source"]["path"] = "/tmp/contact.cu"
    elif mutation == "flags":
        value["binding"]["training_authorized"] = True
    else:
        value["binding"]["symbol"] = "friction_cuda_kernel_forward"

    with pytest.raises(ValueError):
        receiver.compiled_contact(value, raw, inventory, directory, device())


def _fake_boundary_capture(monkeypatch, arm):
    env = _environment(monkeypatch, arm=arm)
    env.contact_kernel.key = "_efc_contact_init__locals__kernel"
    env.contact.dist.host.view(np.uint32)[1] = 0x7FC12345
    _run(env)
    boundary_record = env.observer.boundary_receipt()
    friction_record = env.observer.receipt()

    def freeze_fake_layout(layout):
        shape = layout["shape"]
        host_shape = layout["host_shape"]
        dtype = layout["host_dtype"]
        layout["device"] = "cuda:0"
        if len(host_shape) == len(shape):
            layout["warp_dtype"] = (
                "<class 'warp._src.types.float32'>"
                if dtype == "float32"
                else "<class 'warp._src.types.int32'>"
            )
        elif len(host_shape) == len(shape) + 2 and host_shape[-2:] == [3, 3]:
            layout["warp_dtype"] = "<class 'warp._src.types.mat33f'>"
        else:
            vector_types = {
                ("float32", 2): "<class 'warp._src.types.vec2f'>",
                ("float32", 3): "<class 'warp._src.types.vec3f'>",
                ("float32", 5): "<class 'mujoco_warp._src.types.vec5f'>",
                ("int32", 2): "<class 'warp._src.types.vec2i'>",
            }
            layout["warp_dtype"] = vector_types[(dtype, host_shape[-1])]

    # Adapt only fake identity labels; keep captured raw bytes and shapes intact.
    for entry in boundary_record["entries"]:
        for layouts in (entry["contact"], entry["complete_layouts"]):
            for layout in layouts.values():
                freeze_fake_layout(layout)
    for entry in friction_record["entries"][: receiver.BOUNDARY_FORWARDS]:
        for layout in entry["layouts"].values():
            freeze_fake_layout(layout)

    friction_entries = [
        {
            "identities": {
                "model_object_id": entry["model_object_id"],
                "data_object_id": entry["data_object_id"],
            },
            "stream": entry["stream"],
            "dim": entry["dim"],
            "layouts": entry["layouts"],
        }
        for entry in friction_record["entries"][: receiver.BOUNDARY_FORWARDS]
    ]
    recipe = {
        "control_scope": {
            "calls": [
                {
                    "model_id": id(env.model),
                    "data_id": id(env.data),
                    "stream": env.observer.stream.cuda_stream,
                }
                for _ in range(receiver.BOUNDARY_FORWARDS)
            ]
        }
    }
    first_call = boundary_record["entries"][0]["contact_call"]
    capture = {
        "record": boundary_record,
        "raw": env.sink,
        "device": {
            "context": env.observer.device.context,
            "stream": env.observer.stream.cuda_stream,
        },
        "compiled_role": {
            "kernel_object_id": first_call["kernel_object_id"],
            "module_object_id": first_call["module_object_id"],
        },
        "friction_entries": friction_entries,
        "recipe": recipe,
    }
    capture["result"] = _verify_capture(capture)
    return capture


def _verify_capture(capture, record=None):
    return receiver.contact_boundary(
        capture["record"] if record is None else record,
        capture["raw"],
        capture["record"]["arm"],
        capture["device"],
        capture["compiled_role"],
        capture["friction_entries"],
        capture["recipe"],
    )


def test_positive_boundary_receipts_from_fake_observer_cover_three_arms(monkeypatch):
    captures = {arm: _fake_boundary_capture(monkeypatch, arm) for arm in receiver.ARMS}
    results = {arm: captures[arm]["result"] for arm in receiver.ARMS}
    comparison = receiver.compare_contact_boundaries(results)

    assert all(
        len(results[arm]["entries"]) == receiver.BOUNDARY_FORWARDS
        for arm in receiver.ARMS
    )
    assert all(
        results[arm]["captured_bytes"] == captures[arm]["record"]["captured_bytes"]
        for arm in receiver.ARMS
    )
    assert comparison["candidate0_candidate1_exact_through_sampled_boundaries"] is True
    assert comparison["active_extent_repeat_exact"] is True
    assert comparison["flags"] == receiver.FLAGS
    assert results["original"]["packet_fields"][0]["contact_before"]["contact.nacon"][
        "raw"
    ] == bytes(4)
    assert results["original"]["packet_fields"][0]["contact_before"]["contact.dist"][
        "raw"
    ][4:8] == bytes.fromhex("4523c17f")


@pytest.mark.parametrize(
    "damage",
    (
        "contact_context",
        "complete_packet",
        "complete_fields",
        "contact_identity",
        "contact_span",
    ),
)
def test_positive_receipt_rejects_corruption_in_each_packet_schema_branch(
    monkeypatch, damage
):
    capture = _fake_boundary_capture(monkeypatch, "original")
    corrupted = deepcopy(capture["record"])
    entry = corrupted["entries"][0]
    if damage == "contact_context":
        entry["contact_before"].pop("context_fields")
    elif damage == "complete_packet":
        entry["complete"]["fields"] = entry["complete_fields"]
    elif damage == "complete_fields":
        entry["complete_fields"].pop("efc.Ma")
    elif damage == "contact_identity":
        entry["contact"]["data.nefc"]["object_id"] += 1
    else:
        entry["contact"]["contact.dist"]["span"] -= 4

    with pytest.raises(ValueError):
        _verify_capture(capture, corrupted)
