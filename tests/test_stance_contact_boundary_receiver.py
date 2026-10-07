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
