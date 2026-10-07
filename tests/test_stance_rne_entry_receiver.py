"""Synthetic whole-evidence negatives; never native CUDA execution."""

from hashlib import sha256
import json

import numpy as np
import pytest
import torch
import warp as wp

from mjlab_microduck import stance_com_coupled_receiver as coupled
from mjlab_microduck import stance_com_entry_receiver as old
from mjlab_microduck import stance_rne_entry_receiver as receiver

SOURCE = "a" * 40
TREE = "b" * 40
LEAVES = "c" * 64


def _sha(raw):
    return sha256(raw).hexdigest()


def _canonical(value):
    return old.canonical(value)


def _package_packet():
    return {
        "versions": old.VERSIONS,
        "python": "3.12.13",
        "architecture": "x86_64",
        "python_trees": coupled.PACKAGE_TREES,
    }


def _cpu_state(seed):
    return torch.Generator(device="cpu").manual_seed(seed).get_state().numpy().tobytes()


def _rng_packet():
    states = {
        key: _cpu_state(673 if key.startswith("caller") else 977)
        if "cpu" in key
        else (
            (677 if key.startswith("caller") else 983).to_bytes(8, "little") + b"\0" * 8
        )
        for key in coupled.RNG_KEYS
    }
    raw = b"".join(states[key] for key in coupled.RNG_KEYS)
    metadata = {
        key: {"bytes": len(value), "sha256": _sha(value)}
        for key, value in states.items()
    }
    return raw, metadata


def _world_array(seed=0):
    return (np.full((64, 16, 6), seed, dtype="<f4") / np.float32(10)).astype("<f4")


def _rne_sum(values):
    result = values.copy()
    for level in receiver.RNE_LEVELS:
        for body in level:
            if body:
                np.add(
                    result[:, receiver.RNE_PARENTS[body]],
                    result[:, body],
                    out=result[:, receiver.RNE_PARENTS[body]],
                )
    return result


def _frame_packet(descriptor):
    chunks = []
    for _ in range(2):
        for name, shape in coupled.frames.FRAME_FIELDS:
            values = np.zeros((64, *shape) if shape else (64,), dtype="<f4")
            if name == "qpos":
                values[:] = np.asarray(descriptor["initial_qpos"], dtype="<f4")
            chunks.append(values.tobytes(order="C"))
    return b"".join(chunks)


def _com_entry():
    return np.zeros((64, 16, 3), dtype="<f4").tobytes()


def _com_scope(entries, weighted):
    calls = [
        {
            "call_index": i,
            "worlds": 64,
            "bytes": coupled.ENTRY_BYTES,
            "original_launches": 11,
            "sha256": _sha(entries[i]),
            "device": "cuda:0",
            "stream": 0,
            "boundary": "after-init-before-first-accumulation",
            "initialization_and_accumulation_output_alias": True,
            "readback_and_device_sync_perturb_timing": True,
            "smooth_sha256": "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f",
            "forward_sha256": "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
            "subtree_com_layout": {
                "object_id": 111,
                "ptr": 222,
                "shape": [64, 16],
                "strides": [192, 12],
                "dtype": "vec3f",
                "contiguous": True,
            },
        }
        for i in range(2)
    ]
    return {
        "protocol": "microduck-com-actual-entry-capture-v1:native-coupled-control",
        "status": "complete",
        "fault_type": None,
        "parents": list(old.PARENTS),
        "reversed_levels": [list(row) for row in coupled.old.GROUPS[:4]]
        + [[2, 7, 11], [1], [0]],
        "flags": old.FLAGS,
        "mode": "serial",
        "dispatched_original_kernel_counts": [13, 13],
        "split_body_ids": [2, 7, 11],
        "complete_kernel_count_matches": True,
        "weighted_boundary": "after-accumulation-before-original-division",
        "weighted_boundary_readback_perturbs_timing": True,
        "weighted_sha256": [_sha(value) for value in weighted],
        "calls": calls,
    }


def _crb_scope():
    return {
        "protocol": "football-b1d-crb-runtime-control-v1",
        "status": "complete",
        "mode": "serial",
        "fault_type": None,
        "constructor_forward_covered": True,
        "runtime_bound": True,
        "initialization_timing_changed": True,
        "max_forward_calls": 2,
        "topology_id_snapshots": 2,
        "singleton_child_arrays_allocated": 3,
        "constructor_forward_calls": 1,
        "forward_calls": 2,
        "original_level_launch_requests": 14,
        "parent_zero_noop_level_requests": 4,
        "actual_accumulation_launches": 18,
        "split_child_launches": 6,
        "dense_qM_launches": 2,
        "smooth_source_sha256": "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f",
        "forward_source_sha256": "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
        "flags": {
            "native_qualified": False,
            "original_pair_accepted": False,
            "cause_proven": False,
            "full_window_passed": False,
            "training_authorized": False,
            "physical_result_accepted": False,
        },
    }


def _rne_scope(entries, outputs):
    layout = {
        "object_id": 333,
        "ptr": 444,
        "shape": [64, 16],
        "strides": [384, 24],
        "dtype": "spatial_vectorf",
        "contiguous": True,
    }
    return {
        "protocol": "microduck-rne-actual-entry-oct7-v1",
        "status": "complete",
        "fault": None,
        "unchanged_postconstraint_passthrough_calls": 2,
        "flags": receiver.RNE_FLAGS,
        "calls": [
            {
                "call_index": i,
                "worlds": 64,
                "device": "cuda:0",
                "stream": 0,
                "bytes_per_snapshot": receiver.ENTRY_BYTES,
                "input_sha256": _sha(entries[i]),
                "output_sha256": _sha(outputs[i]),
                "input_boundary": "after-original-cfrc-before-first-backward",
                "output_boundary": "after-original-backward-before-original-qfrc-bias",
                "initializer_caller_code_bound": True,
                "original_launches": 7,
                "body_rule": "body!=0-includes-body1-to-root0",
                "input_output_alias": True,
                "layout": dict(layout),
                "smooth_sha256": "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f",
                "forward_sha256": "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
                "readback_and_device_sync_perturb_timing": True,
            }
            for i in range(2)
        ],
    }


def _repeat_scope(mode, bank, entry):
    return {
        "protocol": "microduck-rne-actual-entry-oct7-v1:detached-repeat",
        "mode": mode,
        "repeats": 32,
        "resets": 32,
        "worlds": 64,
        "bytes_per_snapshot": receiver.ENTRY_BYTES,
        "input_sha256": _sha(entry),
        "bank_sha256": _sha(bank),
        "accumulation_launches": 224 if mode == "concurrent" else 288,
        "body_rule": "body!=0-includes-body1-to-root0",
        "live_array_unchanged": True,
        "live_before_sha256": "d" * 64,
        "live_after_sha256": "d" * 64,
        "readback_and_device_sync_perturb_timing": True,
        "detached_input_role": "supplied-bytes-not-independent-stage-provenance",
        "flags": receiver.RNE_FLAGS,
    }


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(receiver, "EXPECTED_TESTS", 7)
    test_files = ["tests/test_one.py", "tests/test_two.py"]
    monkeypatch.setattr(receiver, "TEST_FILES_SHA256", _sha(_canonical(test_files)))

    descriptor = {
        "selected_fields_sha256": receiver.SELECTED_FIELDS_SHA256,
        "initial_qpos": [0.0] * 21,
        "synthetic": True,
    }
    monkeypatch.setattr(receiver, "DESCRIPTOR_SHA256", _sha(_canonical(descriptor)))
    frame_raw = _frame_packet(descriptor)
    entry_array = _world_array(0)
    entry = entry_array.tobytes()
    output = _rne_sum(entry_array).tobytes()
    entries, outputs = entry + entry, output + output
    bank = output * 32
    com_entry = _com_entry()
    com_weighted = com_entry
    rng_raw, rng_metadata = _rng_packet()
    binding = {
        "source": SOURCE,
        "branch": "feat/athletics-obstacle-curriculum",
        "tree": TREE,
        "leaf_count": 649,
        "leaves_sha256": LEAVES,
    }
    packages = _package_packet()
    live = dict(
        old.SERVICE_CAPS,
        MainPID="100",
        InvocationID="1" * 32,
        ActiveState="active",
        SubState="running",
    )
    run_terminal = dict(
        old.SERVICE_CAPS,
        MainPID="0",
        InvocationID="1" * 32,
        ActiveState="active",
        SubState="exited",
        Result="success",
        ExecMainStatus="0",
    )
    test_caps = {**old.SERVICE_CAPS, "LimitFSIZE": "67108864"}
    test_live = dict(
        test_caps,
        MainPID="99",
        InvocationID="2" * 32,
        ActiveState="active",
        SubState="running",
    )
    test_terminal = dict(
        test_caps,
        MainPID="0",
        InvocationID="2" * 32,
        ActiveState="active",
        SubState="exited",
        Result="success",
        ExecMainStatus="0",
    )
    test_receipt = {
        "protocol": receiver.DATA_PROTOCOL + ":tests",
        "source_binding": binding,
        "packages": packages,
        "tests": 7,
        "test_files": test_files,
        "retained_descriptor_fixture": receiver.DESCRIPTOR_FIXTURE,
        "retained_test_fixtures": receiver.RETAINED_TEST_FIXTURES,
        "junit_sha256": "3" * 64,
        "log_sha256": "4" * 64,
        "flags": old.FLAGS,
        "service_properties": test_live,
        "test_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "MICRODUCK_STANCE_PROFILE": "wsl-10098-20260930",
            "ATEN_CPU_CAPABILITY": "default",
            "MKL_CBWR": "COMPATIBLE",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
    }
    test_raw = _canonical(test_receipt)
    declaration = {
        "protocol": receiver.DATA_PROTOCOL + ":declaration",
        "source": SOURCE,
        "source_binding": binding,
        "packages": packages,
        "service_properties": live,
        "host": {
            "machine": old.MACHINE,
            "gpu": old.GPU,
            "driver": "595.95",
            "driver_model": "WDDM",
            "temperature_c": 30,
            "used_mib": 640,
            "free_mib": 23500,
            "processes": [],
        },
        "phase_a": {
            "source": receiver.PHASE_A_SOURCE,
            "files": receiver.PHASE_A,
            "terminal": receiver.PHASE_A_TERMINAL,
        },
        "predecessors": receiver.PREDECESSORS,
        "com_predecessor": receiver.COM_PREDECESSOR,
        "expected_tests": 7,
        "tests": {
            "count": 7,
            "files": len(test_files),
            "source_binding": binding,
            "receipt_sha256": _sha(test_raw),
            "terminal": test_terminal,
        },
        "flags": old.FLAGS,
    }
    declaration_raw = _canonical(declaration)
    child = {
        "protocol": receiver.DATA_PROTOCOL + ":child",
        "source": SOURCE,
        "source_binding": binding,
        "packages": packages,
        "declaration_sha256": _sha(declaration_raw),
        "owner_pid": 100,
        "child_pid": 101,
        "child_ppid": 100,
        "compiled_descriptor": descriptor,
        "ntendon": 0,
        "fixed_state_unchanged": True,
        "physics_steps": 0,
        "graph_created": False,
        "actor_model_created": False,
        "optimizer_created": False,
        "storage_created": False,
        "integration_calls": 0,
        "com_scope": _com_scope([com_entry] * 2, [com_weighted] * 2),
        "crb_scope": _crb_scope(),
        "rne_scope": _rne_scope([entry] * 2, [output] * 2),
        "repeat_scopes": {
            mode: _repeat_scope(mode, bank, entry) for mode in ("concurrent", "serial")
        },
        "rng_metadata": rng_metadata,
        "flags": receiver.RNE_FLAGS,
    }
    child_raw = _canonical(child)
    gpu_host = {
        "machine": old.MACHINE,
        "gpu": old.GPU,
        "driver": "595.95",
        "driver_model": "WDDM",
        "temperature_c": 31,
        "used_mib": 1000,
        "free_mib": 23000,
        "processes": [
            {
                "pid": 101,
                "memory_status": "reported",
                "memory_mib": 300,
                "raw_memory": "300",
            }
        ],
    }
    report = {
        "protocol": receiver.DATA_PROTOCOL + ":report",
        "source": SOURCE,
        "source_binding": binding,
        "child_sha256": _sha(child_raw),
        "returncode": 0,
        "observed_child_pid": 101,
        "observed_child_ppid": 100,
        "gpu_child_observed": True,
        "monitor": [{"elapsed": 0.5, "child_pid": 101, "host": gpu_host}],
        "closure_host": {
            "machine": old.MACHINE,
            "gpu": old.GPU,
            "driver": "595.95",
            "driver_model": "WDDM",
            "temperature_c": 30,
            "used_mib": 640,
            "free_mib": 23500,
            "processes": [],
        },
        "flags": old.FLAGS,
    }
    files = {
        "entries.bin": entries,
        "outputs.bin": outputs,
        "concurrent.bin": bank,
        "serial.bin": bank,
        "frames.bin": frame_raw,
        "rng.bin": rng_raw,
        "com.entries.bin": com_entry * 2,
        "com.weighted.bin": com_weighted * 2,
        "declaration.json": declaration_raw,
        "child.json": child_raw,
        "report.json": _canonical(report),
        "child.log": b"synthetic fixture, not CUDA execution\n",
        "tests.receipt.json": test_raw,
        "tests.terminal.json": _canonical(test_terminal),
    }
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
    anchors = {
        name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in files.items()
    }
    return {
        "root": tmp_path,
        "files": files,
        "anchors": anchors,
        "terminal": _canonical(run_terminal),
        "binding": binding,
        "descriptor": descriptor,
        "entry": entry,
        "output": output,
        "bank": bank,
    }


def verify(value):
    return receiver.verify_directory(
        value["root"],
        value["anchors"],
        value["terminal"],
        _sha(value["terminal"]),
        source=SOURCE,
        expected_tree=TREE,
        expected_leaves_sha256=LEAVES,
    )


def _replace_artifact(value, name, raw):
    (value["root"] / name).write_bytes(raw)
    value["files"][name] = raw
    value["anchors"][name] = {"bytes": len(raw), "sha256": _sha(raw)}


def _rewrite_child(value, mutate):
    child = json.loads(value["files"]["child.json"])
    mutate(child)
    child_raw = _canonical(child)
    _replace_artifact(value, "child.json", child_raw)
    report = json.loads(value["files"]["report.json"])
    report["child_sha256"] = _sha(child_raw)
    _replace_artifact(value, "report.json", _canonical(report))


def test_complete_synthetic_packet_has_full_rows_and_false_admission(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    report = verify(value)
    assert report["decision"] == "fresh-rne-serial-reference-exact"
    assert report["initialized_entries_exact"]
    assert report["constructor_forward_arithmetic_exact"]
    assert report["repeat_comparisons"]["serial"]["exact_all_repeats"]
    assert len(report["repeat_comparisons"]["serial"]["repeat_rows"]) == 32
    assert len(report["repeat_comparisons"]["serial"]["world_rows"]) == 64
    assert set(report["frame_temporal_fields"]) == {
        name for name, _ in coupled.frames.FRAME_FIELDS
    }
    assert len(report["constructor_forward_arithmetic"]) == 2
    assert report["flags"] == receiver.RNE_FLAGS
    assert all(value is False for value in report["flags"].values())


def test_all_fourteen_whole_hashes_precede_json_or_numpy_decode(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    bad_name = "serial.bin"
    value["anchors"][bad_name]["sha256"] = "0" * 64
    json_calls = []
    numpy_calls = []
    original_json = old._json
    original_frombuffer = np.frombuffer

    def spy_json(raw):
        json_calls.append(raw)
        return original_json(raw)

    def spy_frombuffer(*args, **kwargs):
        numpy_calls.append(args)
        return original_frombuffer(*args, **kwargs)

    monkeypatch.setattr(old, "_json", spy_json)
    monkeypatch.setattr(np, "frombuffer", spy_frombuffer)
    with pytest.raises(ValueError, match="whole external SHA"):
        verify(value)
    assert not json_calls and not numpy_calls


@pytest.mark.parametrize(
    "mutation", ["short", "oversized", "mutable", "wrong-hash", "extra"]
)
def test_exact_inventory_and_immutable_external_anchors(
    tmp_path, monkeypatch, mutation
):
    value = fixture(tmp_path, monkeypatch)
    if mutation == "short":
        _replace_artifact(value, "entries.bin", value["files"]["entries.bin"][:-4])
    elif mutation == "oversized":
        _replace_artifact(value, "serial.bin", value["files"]["serial.bin"] + b"x")
    elif mutation == "mutable":
        value["anchors"]["entries.bin"] = bytearray(b"x")
    elif mutation == "wrong-hash":
        value["anchors"]["entries.bin"]["sha256"] = "0" * 64
    else:
        (value["root"] / "extra.bin").write_bytes(b"extra")
    with pytest.raises((ValueError, TypeError)):
        verify(value)


def test_rejects_nonfinite_authenticated_frames_and_entries(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    raw = bytearray(value["files"]["entries.bin"])
    raw[:4] = np.asarray([np.inf], dtype="<f4").tobytes()
    _replace_artifact(value, "entries.bin", bytes(raw))
    with pytest.raises(ValueError, match="finite"):
        verify(value)


def test_negative_serial_bank_retains_bounded_coordinates_and_root_counts(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    bank = (
        np.frombuffer(value["files"]["serial.bin"], dtype="<f4")
        .copy()
        .reshape(32, 64, 16, 6)
    )
    bank[4, 0, 1, 0] = np.float32(1.0)
    raw = bank.astype("<f4").tobytes()
    _replace_artifact(value, "serial.bin", raw)
    _rewrite_child(
        value,
        lambda child: child["repeat_scopes"]["serial"].update(bank_sha256=_sha(raw)),
    )
    report = verify(value)
    serial = report["repeat_comparisons"]["serial"]
    assert report["decision"] == "fresh-rne-negative"
    assert serial["bit_mismatch_scalars"] == 1
    assert serial["body_mismatch_scalars"] == {"body0": 0, "body1": 1, "others": 0}
    assert serial["mismatch_coordinates"] == [[4, 0, 1, 0, 0, 1065353216, 1065353216]]


def test_refuses_to_truncate_repeat_mismatch_coordinates(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    bank = (
        np.frombuffer(value["files"]["serial.bin"], dtype="<f4")
        .copy()
        .reshape(32, 64, 16, 6)
    )
    bank[:3, :, :, :] = 1.0
    raw = bank.astype("<f4").tobytes()
    _replace_artifact(value, "serial.bin", raw)
    _rewrite_child(
        value,
        lambda child: child["repeat_scopes"]["serial"].update(bank_sha256=_sha(raw)),
    )
    with pytest.raises(ValueError, match="coordinate cap exceeded"):
        verify(value)


def test_constructor_forward_signed_zero_is_a_real_temporal_mismatch(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    entries = (
        np.frombuffer(value["files"]["entries.bin"], dtype="<f4")
        .copy()
        .reshape(2, 64, 16, 6)
    )
    entries[1, 0, 0, 0] = np.float32(-0.0)
    entries_raw = entries.astype("<f4").tobytes()
    outputs = np.stack([_rne_sum(item) for item in entries]).astype("<f4").tobytes()
    _replace_artifact(value, "entries.bin", entries_raw)
    _replace_artifact(value, "outputs.bin", outputs)
    _rewrite_child(
        value,
        lambda child: child.update(
            rne_scope=_rne_scope(
                [
                    entries_raw[: receiver.ENTRY_BYTES],
                    entries_raw[receiver.ENTRY_BYTES :],
                ],
                [outputs[: receiver.ENTRY_BYTES], outputs[receiver.ENTRY_BYTES :]],
            )
        ),
    )
    report = verify(value)
    assert report["initialized_entries_exact"] is False
    assert report["decision"] == "fresh-rne-negative"
    assert report["entry_temporal"]["bit_mismatch_scalars"] == 1
    assert report["entry_temporal"]["max_ordered_bit_distance"] == 1


def test_captured_forward_output_negative_is_reported_without_admission_gate(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    outputs = np.frombuffer(value["files"]["outputs.bin"], dtype="<f4").copy()
    outputs[receiver.ENTRY_BYTES // 4] = np.float32(1.0)
    raw = outputs.astype("<f4").tobytes()
    _replace_artifact(value, "outputs.bin", raw)
    _rewrite_child(
        value,
        lambda child: child.update(
            rne_scope=_rne_scope(
                [value["entry"], value["entry"]],
                [raw[: receiver.ENTRY_BYTES], raw[receiver.ENTRY_BYTES :]],
            )
        ),
    )
    report = verify(value)
    assert report["constructor_forward_arithmetic_exact"] is False
    assert report["decision"] == "fresh-rne-serial-reference-exact"


def test_frame_temporal_rows_report_bias_delta_and_signed_zero_exactness(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    packet = bytearray(value["files"]["frames.bin"])
    # qfrc_bias is the ninth field in each field-major frame; alter eager frame world 0, dof 0.
    frame_bytes = coupled.frames.FRAME_BYTES
    qfrc_bias_offset = sum(
        64 * int(np.prod(shape) if shape else 1) * 4
        for name, shape in coupled.frames.FRAME_FIELDS[:8]
    )
    offset = frame_bytes + qfrc_bias_offset
    packet[offset : offset + 4] = np.asarray([np.float32(-0.0)], dtype="<f4").tobytes()
    _replace_artifact(value, "frames.bin", bytes(packet))
    report = verify(value)
    row = report["frame_temporal_fields"]["qfrc_bias"]
    assert row["bit_mismatch_scalars"] == 1
    assert row["max_ordered_bit_distance"] == 1
    assert row["exact_raw_bits"] is False


def test_fixed_input_mutation_fails_even_when_both_frames_match(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    packet = bytearray(value["files"]["frames.bin"])
    # qvel follows qpos; identical mutation in both frames violates positive-zero input.
    offset = sum(
        64 * int(np.prod(shape) if shape else 1) * 4
        for _, shape in coupled.frames.FRAME_FIELDS[:11]
    )
    packet[offset : offset + 4] = np.asarray([np.float32(1.0)], dtype="<f4").tobytes()
    frame_bytes = coupled.frames.FRAME_BYTES
    packet[offset + frame_bytes : offset + frame_bytes + 4] = np.asarray(
        [np.float32(1.0)], dtype="<f4"
    ).tobytes()
    _replace_artifact(value, "frames.bin", bytes(packet))
    with pytest.raises(ValueError, match="positive-zero fixed input"):
        verify(value)


@pytest.mark.parametrize(
    "field",
    [
        "original_launches",
        "input_output_alias",
        "body_rule",
        "initializer_caller_code_bound",
    ],
)
def test_rne_scope_rejects_unproven_dispatch_claim(tmp_path, monkeypatch, field):
    value = fixture(tmp_path, monkeypatch)

    def mutate(child):
        call = child["rne_scope"]["calls"][0]
        call[field] = 6 if field == "original_launches" else False

    _rewrite_child(value, mutate)
    with pytest.raises(ValueError, match="literal initialized RNE boundary"):
        verify(value)


def test_wrong_source_tree_or_leaf_anchor_rejected(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="committed source binding"):
        receiver.verify_directory(
            value["root"],
            value["anchors"],
            value["terminal"],
            _sha(value["terminal"]),
            source=SOURCE,
            expected_tree="d" * 40,
            expected_leaves_sha256=LEAVES,
        )


def test_no_subset_or_nonliteral_case_list_is_accepted(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    del value["anchors"]["outputs.bin"]
    with pytest.raises(ValueError, match="fourteen-file"):
        verify(value)


def test_spatial_layout_stride_matches_actual_cpu_allocation():
    # A real allocation verifies the source assumption, not GPU qualification.
    wp.init()
    array = wp.empty((64, 16), dtype=wp.spatial_vector, device="cpu")
    assert array.shape == (64, 16)
    assert array.strides == (384, 24)
    assert array.dtype is wp.spatial_vector


def test_assumed_padded_spatial_stride_is_rejected(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)

    def mutate(child):
        for call in child["rne_scope"]["calls"]:
            call["layout"]["strides"] = [1536, 24]

    _rewrite_child(value, mutate)
    with pytest.raises(ValueError, match="stable spatial-vector"):
        verify(value)


def test_wrong_whole_retained_cpu_fixture_binding_is_rejected(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    receipt = json.loads(value["files"]["tests.receipt.json"])
    receipt["retained_test_fixtures"]["sha256"] = "0" * 64
    raw = _canonical(receipt)
    _replace_artifact(value, "tests.receipt.json", raw)
    declaration = json.loads(value["files"]["declaration.json"])
    declaration["tests"]["receipt_sha256"] = _sha(raw)
    raw = _canonical(declaration)
    _replace_artifact(value, "declaration.json", raw)
    _rewrite_child(value, lambda child: child.update(declaration_sha256=_sha(raw)))
    with pytest.raises(ValueError, match="whole completed CPU receipt"):
        verify(value)
