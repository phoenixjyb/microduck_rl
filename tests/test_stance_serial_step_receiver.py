"""Explicitly synthetic complete one-tick receiver tests, never CUDA proof."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

import test_stance_rne_entry_receiver as entry_fixture
from mjlab_microduck import stance_serial_step_receiver as receiver

SOURCE, TREE, LEAVES = entry_fixture.SOURCE, entry_fixture.TREE, entry_fixture.LEAVES
canon = receiver.old.canonical


def sha(raw):
    return sha256(raw).hexdigest()


def scope_fixture(banks):
    layouts = [
        dict(
            name=name,
            object_id=1000 + i,
            ptr=4000 + i * 4096,
            shape=shape,
            strides=strides,
            dtype=f"<class '{dtype}'>",
            device="cuda:0",
            contiguous=True,
        )
        for i, (name, (shape, strides, dtype)) in enumerate(receiver.LAYOUTS.items())
    ]
    # These are independently declared synthetic layouts, not CUDA readbacks.
    calls = []
    for i in range(21):
        calls.append(
            dict(
                forward_index=i,
                phase="constructor" if i == 0 else "step-pre" if i % 2 else "step-post",
                model_id=17,
                data_id=18,
                device="cuda:0",
                stream=0,
                smooth_sha256=receiver.SMOOTH_SHA,
                forward_sha256=receiver.FORWARD_SHA,
                array_layouts=deepcopy(layouts),
                rne_sensory="untouched-original-passthrough",
                com=dict(
                    logical_launches=11,
                    underlying_launches=13,
                    split_index=5,
                    split_body_order=[2, 7, 11],
                    underlying_body_groups=deepcopy(receiver.GROUPS),
                    observed_body_groups=deepcopy(receiver.GROUPS),
                    initialized_sha256=sha(banks["com.entries.bin"][i]),
                    weighted_sha256=sha(banks["com.weighted.bin"][i]),
                ),
                crb=dict(
                    logical_accumulation_launches=7,
                    underlying_accumulation_launches=9,
                    qM_launches=1,
                    split_index=4,
                    split_body_order=[2, 7, 11],
                    underlying_body_groups=deepcopy(receiver.GROUPS),
                    observed_body_groups=deepcopy(receiver.GROUPS),
                ),
                rne_bias=dict(
                    logical_launches=7,
                    underlying_launches=9,
                    split_index=4,
                    split_body_order=[2, 7, 11],
                    underlying_body_groups=deepcopy(receiver.GROUPS),
                    observed_body_groups=deepcopy(receiver.GROUPS),
                    input_sha256=sha(banks["rne.entries.bin"][i]),
                    output_sha256=sha(banks["rne.outputs.bin"][i]),
                    input_boundary="after-original-cfrc-before-first-backward",
                    output_boundary="after-original-backward-before-original-qfrc-bias",
                    input_output_alias=True,
                    bytes_per_snapshot=receiver.RNE_BYTES,
                    snapshot_layout={
                        **{
                            k: layouts[-3][k]
                            for k in (
                                "object_id",
                                "ptr",
                                "shape",
                                "strides",
                                "device",
                                "contiguous",
                            )
                        },
                        "dtype": "spatial_vectorf",
                        "stream": 0,
                    },
                    readback_and_device_sync_perturbs_timing=True,
                ),
            )
        )
    return dict(
        protocol="microduck-serial-one-step-control-oct7-v1",
        status="complete",
        fault=None,
        forward_calls=21,
        constructor_forward_calls=1,
        step_calls=1,
        worlds=64,
        device="cuda:0",
        calls=calls,
        entry_counts={k: 21 for k in ("com", "crb", "rne_bias", "rne_post")},
        com_initialized_sha256=[sha(v) for v in banks["com.entries.bin"]],
        com_weighted_sha256=[sha(v) for v in banks["com.weighted.bin"]],
        rne_input_sha256=[sha(v) for v in banks["rne.entries.bin"]],
        rne_output_sha256=[sha(v) for v in banks["rne.outputs.bin"]],
        dispatch_trace=[
            dict(forward_index=i, entry=e, body_ids=g)
            for i in range(21)
            for e in ("com", "crb", "rne_bias")
            for g in deepcopy(receiver.GROUPS)
        ],
        dispatch=dict(
            forward_cap=21,
            com_original_logical_launches=11,
            com_underlying_launches=13,
            com_split_index=5,
            crb_logical_accumulation_launches=7,
            crb_underlying_accumulation_launches=9,
            crb_qM_launches=1,
            crb_split_index=4,
            rne_bias_logical_launches=7,
            rne_bias_underlying_launches=9,
            rne_split_index=4,
            split_body_order=[2, 7, 11],
            sensory_rne_untouched_passthrough_calls=21,
            constructor_plus_one_step_only=True,
            all_observation_readbacks_perturb_timing=True,
            closed=True,
        ),
        flags=dict(receiver.FLAGS, native_qualified=False),
    )


def fixture(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    value = entry_fixture.fixture(old, monkeypatch)
    binding = dict(value["binding"], leaf_count=663)
    prior = json.loads(value["files"]["child.json"])
    descriptor = dict(prior["compiled_descriptor"], qids=list(range(7, 21)))
    monkeypatch.setattr(receiver, "DESCRIPTOR_SHA256", sha(canon(descriptor)))
    monkeypatch.setattr(receiver, "EXPECTED_TESTS", 7)
    receipt = json.loads(value["files"]["tests.receipt.json"])
    monkeypatch.setattr(
        receiver, "TEST_FILES_SHA256", sha(canon(receipt["test_files"]))
    )
    receipt.update(protocol=receiver.DATA_PROTOCOL + ":tests", source_binding=binding)
    test_raw = canon(receipt)
    declaration = json.loads(value["files"]["declaration.json"])
    declaration.update(
        protocol=receiver.DATA_PROTOCOL + ":declaration",
        source_binding=binding,
        rne_predecessor=receiver.PREDECESSOR,
        case_order=list(receiver.CASE_ORDER),
    )
    declaration["tests"].update(source_binding=binding, receipt_sha256=sha(test_raw))
    declaration_raw = canon(declaration)
    packets = {}
    clock = np.float32(0)
    for _ in range(10):
        clock = np.float32(clock + np.float32(0.002))
    frames = []
    for boundary in range(2):
        for name, shape in receiver.frames.FRAME_FIELDS:
            a = np.zeros((64, *shape) if shape else (64,), dtype="<f4")
            if name == "time" and boundary == 1:
                a[:] = clock
            frames.append(a.tobytes())
    packets["frames.bin"] = b"".join(frames)
    banks = {
        name: [b"\0" * size] * 21
        for name, size in (
            ("rne.entries.bin", receiver.RNE_BYTES),
            ("rne.outputs.bin", receiver.RNE_BYTES),
            ("com.entries.bin", receiver.COM_BYTES),
            ("com.weighted.bin", receiver.COM_BYTES),
        )
    }
    packets.update({name: b"".join(bank) for name, bank in banks.items()})
    packets["rng.bin"], rng_metadata = entry_fixture._rng_packet()
    motors, masks = [], []
    for i in range(10):
        for name, shape in receiver.MOTOR_FIELDS.items():
            a = np.zeros((64, *shape), dtype="<f4")
            if name == "committed.voltage":
                a[:] = 7.5
            if name == "committed.kp":
                a[:] = 200.0
            motors.append(a.tobytes())
        masks.extend(
            (
                np.full((64,), i, dtype="<i8").tobytes(),
                b"\1" * 64,
                b"\1" * 64,
                b"\0" * 64,
            )
        )
    packets["motor.bin"], packets["masks.bin"] = b"".join(motors), b"".join(masks)
    ledger = dict(
        reward=[0.0] * 64,
        terminated=[False] * 64,
        timed_out=[False] * 64,
        live=[True] * 64,
        episode_steps=[10] * 64,
        executed_steps=[10] * 64,
        term_sums={
            k: [0.0] * 64
            for k in (
                "upright",
                "stillness",
                "height",
                "support",
                "motor",
                "joint_speed",
                "correction",
                "correction_change",
            )
        },
    )
    cases = {
        case: dict(
            compiled_descriptor=descriptor,
            ntendon=0,
            nominal_parameters={
                "voltage": 7.5,
                "drop_gain": 0.1,
                "kp_scale": 1.0,
                "kd_scale": 1.0,
                "friction_scale": 1.0,
            },
            physics_steps=10,
            graph_created=False,
            actor_model_created=False,
            optimizer_created=False,
            storage_created=False,
            explicit_reset_calls=0,
            control_scope=scope_fixture(banks),
            rng_metadata=rng_metadata,
            ledger=ledger,
            rng_end_boundary="after-one-nominal-step",
            control_capture=True,
            control_ids=list(range(14)),
        )
        for case in receiver.CASE_ORDER
    }
    child = {
        k: deepcopy(prior[k])
        for k in ("source", "packages", "owner_pid", "child_pid", "child_ppid", "flags")
    }
    child.update(
        protocol=receiver.DATA_PROTOCOL + ":child",
        source_binding=binding,
        declaration_sha256=sha(declaration_raw),
        case_order=list(receiver.CASE_ORDER),
        cases=cases,
    )
    child_raw = canon(child)
    report = json.loads(value["files"]["report.json"])
    report.update(
        protocol=receiver.DATA_PROTOCOL + ":report",
        source_binding=binding,
        child_sha256=sha(child_raw),
    )
    files = {
        case + "." + name: raw
        for case in receiver.CASE_ORDER
        for name, raw in packets.items()
    }
    files.update(
        {
            "declaration.json": declaration_raw,
            "child.json": child_raw,
            "report.json": canon(report),
            "child.log": value["files"]["child.log"],
            "tests.receipt.json": test_raw,
            "tests.terminal.json": value["files"]["tests.terminal.json"],
        }
    )
    root = tmp_path / "fresh"
    root.mkdir()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return dict(
        root=root,
        files=files,
        anchors={n: dict(bytes=len(v), sha256=sha(v)) for n, v in files.items()},
        terminal=value["terminal"],
        binding=binding,
    )


def verify(value, **kwargs):
    return receiver.verify_directory(
        value["root"],
        value["anchors"],
        value["terminal"],
        sha(value["terminal"]),
        source=SOURCE,
        expected_tree=kwargs.get("tree", TREE),
        expected_leaves_sha256=kwargs.get("leaves", LEAVES),
    )


def replace(value, name, raw):
    (value["root"] / name).write_bytes(raw)
    value["files"][name] = raw
    value["anchors"][name] = dict(bytes=len(raw), sha256=sha(raw))


def rewrite_child(value, mutate):
    child = json.loads(value["files"]["child.json"])
    mutate(child)
    raw = canon(child)
    replace(value, "child.json", raw)
    report = json.loads(value["files"]["report.json"])
    report["child_sha256"] = sha(raw)
    replace(value, "report.json", canon(report))


def test_complete_synthetic_positive_keeps_all_admissions_closed(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    result = verify(value)
    assert result["decision"] == "fresh-one-step-serial-repeat-exact"
    assert result["execution_decision"] == "one-nominal-tick-consistent"
    assert result["arithmetic_decision"] == "moving-reductions-reference-exact"
    assert (
        len(result["field_comparisons"]) == 34
        and len(result["packet_comparisons"]) == 8
    )
    assert len(result["receiver_files"]) == 22
    assert all(v is False for v in result["flags"].values())
    assert (
        result["simulated_seconds"] == 0.02
        and result["full_policy_capture_replay"] is False
    )
    # Constructor and post-step clocks differ: no unchanged-time gate is applied.
    assert (
        value["files"]["serial0.frames.bin"][: receiver.frames.FRAME_BYTES]
        != value["files"]["serial0.frames.bin"][receiver.frames.FRAME_BYTES :]
    )


@pytest.mark.parametrize("name", list(receiver.CAPS))
def test_all_whole_files_authenticate_before_any_decode(tmp_path, monkeypatch, name):
    value = fixture(tmp_path, monkeypatch)
    raw = value["files"][name]
    (value["root"] / name).write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    monkeypatch.setattr(
        receiver.old, "_json", lambda *_: pytest.fail("unauthenticated JSON")
    )
    monkeypatch.setattr(
        receiver.np,
        "frombuffer",
        lambda *_args, **_kwargs: pytest.fail("unauthenticated array"),
    )
    with pytest.raises(ValueError):
        verify(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "group",
        "observed",
        "trace",
        "cap",
        "layout",
        "dtype",
        "pointer",
        "snapshot-pointer",
        "snapshot-bytes",
        "phase",
        "source",
        "sensory",
        "flags",
        "constructor",
        "step",
        "subset",
        "actor",
        "reset",
    ],
)
def test_scope_and_execution_claims_are_literal(tmp_path, monkeypatch, mutation):
    value = fixture(tmp_path, monkeypatch)

    def change(child):
        record = child["cases"]["serial0"]
        scope = record["control_scope"]
        row = scope["calls"][1]
        if mutation == "group":
            row["rne_bias"]["underlying_body_groups"][4] = [7]
        elif mutation == "observed":
            row["crb"]["observed_body_groups"][4] = [7]
        elif mutation == "trace":
            scope["dispatch_trace"][0]["body_ids"] = [15, 6]
        elif mutation == "cap":
            scope["dispatch"]["forward_cap"] = 562
        elif mutation == "layout":
            row["array_layouts"][8]["strides"] = [64, 4]
        elif mutation == "dtype":
            row["array_layouts"][8]["dtype"] = "float32"
        elif mutation == "pointer":
            row["array_layouts"][0]["ptr"] += 4
        elif mutation == "snapshot-pointer":
            row["rne_bias"]["snapshot_layout"]["ptr"] += 4
        elif mutation == "snapshot-bytes":
            row["rne_bias"]["bytes_per_snapshot"] -= 4
        elif mutation == "phase":
            row["phase"] = "step-post"
        elif mutation == "source":
            row["smooth_sha256"] = "0" * 64
        elif mutation == "sensory":
            row["rne_sensory"] = "serial"
        elif mutation == "flags":
            scope["flags"]["native_qualified"] = True
        elif mutation == "constructor":
            scope["constructor_forward_calls"] = 2
        elif mutation == "step":
            record["physics_steps"] = 9
        elif mutation == "subset":
            child["cases"].pop("serial1")
        elif mutation == "actor":
            record["actor_model_created"] = True
        elif mutation == "reset":
            record["explicit_reset_calls"] = 1

    rewrite_child(value, change)
    with pytest.raises(ValueError):
        verify(value)


def field_offset(name):
    offset = 0
    for field, shape in receiver.frames.FRAME_FIELDS:
        if field == name:
            return offset
        offset += 64 * (int(np.prod(shape)) if shape else 1) * 4
    raise AssertionError(name)


@pytest.mark.parametrize(
    "mutation",
    [
        "clock",
        "initial-zero",
        "force",
        "nonfinite",
        "torque",
        "voltage",
        "mask",
        "counter",
        "rng",
    ],
)
def test_physical_and_motor_consistency_failures_are_not_repeat_passes(
    tmp_path, monkeypatch, mutation
):
    value = fixture(tmp_path, monkeypatch)
    name = "serial0.frames.bin"
    raw = bytearray(value["files"][name])
    offset = receiver.frames.FRAME_BYTES + field_offset("time")
    number = np.float32(0.018)
    if mutation == "initial-zero":
        offset, number = field_offset("qvel"), np.float32(-0.0)
    elif mutation == "force":
        offset, number = (
            receiver.frames.FRAME_BYTES + field_offset("xfrc_applied"),
            np.float32(0.01),
        )
    elif mutation == "nonfinite":
        offset, number = field_offset("crb"), np.float32(np.nan)
    elif mutation in ("torque", "voltage"):
        name = "serial0.motor.bin"
        raw = bytearray(value["files"][name])
        offset = 0
        for field, shape in receiver.MOTOR_FIELDS.items():
            if field == ("torque" if mutation == "torque" else "committed.voltage"):
                break
            offset += 64 * int(np.prod(shape)) * 4
        number = np.float32(0.37 if mutation == "torque" else 8.0)
    elif mutation in ("mask", "counter"):
        name = "serial0.masks.bin"
        raw = bytearray(value["files"][name])
        if mutation == "mask":
            raw[512] = 0
        else:
            raw[:8] = (1).to_bytes(8, "little")
    elif mutation == "rng":
        name = "serial0.rng.bin"
        raw = bytearray(value["files"][name])
        raw[0] ^= 1
    if mutation not in ("mask", "counter", "rng"):
        raw[offset : offset + 4] = number.tobytes()
    replace(value, name, bytes(raw))
    with pytest.raises(ValueError):
        verify(value)


def test_full_pair_negative_is_retained_not_tolerance_normalized(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    raw = bytearray(value["files"]["serial1.frames.bin"])
    offset = receiver.frames.FRAME_BYTES + field_offset("qfrc_bias")
    raw[offset : offset + 4] = np.float32(1e-9).tobytes()
    replace(value, "serial1.frames.bin", bytes(raw))
    result = verify(value)
    assert result["paired_decision"] == "one-tick-paired-negative"
    assert result["decision"] == "fresh-one-step-serial-repeat-negative"
    assert result["arithmetic_decision"] == "moving-reductions-reference-exact"
    assert all(v is False for v in result["flags"].values())


def test_external_source_tree_inventory_and_terminal_not_self_asserted(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        verify(value, tree="d" * 40)
    with pytest.raises(ValueError):
        verify(value, leaves="e" * 64)
    value["terminal"] = value["terminal"][:-1] + b"X"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify(value)


def test_inventory_cannot_include_unanchored_files(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    (value["root"] / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError):
        verify(value)


def test_finite_reward_mismatch_is_retained_as_pair_negative(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    rewrite_child(
        value,
        lambda child: child["cases"]["serial1"]["ledger"]["reward"].__setitem__(
            53, 0.125
        ),
    )
    result = verify(value)
    assert result["execution_decision"] == "one-nominal-tick-consistent"
    assert result["paired_decision"] == "one-tick-paired-negative"
    assert result["decision"] == "fresh-one-step-serial-repeat-negative"
    assert result["ledger_comparison"]["exact_retained_values"] is False
    assert result["ledger_field_comparisons"][0]["bit_mismatch_scalars"] == 1
    assert all(value is False for value in result["flags"].values())


@pytest.mark.parametrize("number", [True, 0, 0.1, float("nan")])
def test_reward_requires_actual_finite_float32_values(tmp_path, monkeypatch, number):
    value = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        rewrite_child(
            value,
            lambda child: child["cases"]["serial0"]["ledger"]["reward"].__setitem__(
                0, number
            ),
        )
        verify(value)
