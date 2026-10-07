"""Synthetic four-case receiver contracts, never CUDA or training evidence."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

import test_stance_rne_entry_receiver as entry_fixture
from mjlab_microduck import stance_rne_coupled_receiver as receiver
from mjlab_microduck import stance_rne_entry_receiver as prior

SOURCE, TREE, LEAVES = entry_fixture.SOURCE, entry_fixture.TREE, entry_fixture.LEAVES


def sha(raw):
    return sha256(raw).hexdigest()


def canon(value):
    return receiver.old.canonical(value)


def fixture(tmp_path, monkeypatch):
    old_root = tmp_path / "old"
    old_root.mkdir()
    value = entry_fixture.fixture(old_root, monkeypatch)
    monkeypatch.setattr(receiver, "EXPECTED_TESTS", 7)
    monkeypatch.setattr(receiver, "TEST_FILES_SHA256", prior.TEST_FILES_SHA256)
    monkeypatch.setattr(receiver, "DESCRIPTOR_SHA256", prior.DESCRIPTOR_SHA256)
    binding = dict(value["binding"], leaf_count=656)
    declaration = json.loads(value["files"]["declaration.json"])
    declaration.update(
        protocol=receiver.DATA_PROTOCOL + ":declaration",
        source_binding=binding,
        rne_predecessor=receiver.RNE_PREDECESSOR,
        case_order=list(receiver.CASE_ORDER),
    )
    declaration["tests"]["source_binding"] = binding
    receipt = json.loads(value["files"]["tests.receipt.json"])
    receipt.update(protocol=receiver.DATA_PROTOCOL + ":tests", source_binding=binding)
    test_raw = canon(receipt)
    declaration["tests"]["receipt_sha256"] = sha(test_raw)
    declaration_raw = canon(declaration)
    old_child = json.loads(value["files"]["child.json"])
    cases = {}
    files = {}
    for case in receiver.CASE_ORDER:
        mode = "serial" if case.startswith("serial") else "original"
        record = {
            k: deepcopy(old_child[k])
            for k in (
                "compiled_descriptor",
                "ntendon",
                "fixed_state_unchanged",
                "physics_steps",
                "graph_created",
                "actor_model_created",
                "optimizer_created",
                "storage_created",
                "com_scope",
                "crb_scope",
                "rne_scope",
                "rng_metadata",
            )
        }
        record["mode"] = mode
        if mode == "serial":
            scope = record["rne_scope"]
            scope["protocol"] = "microduck-rne-coupled-serial-oct7-v1"
            scope["dispatch"] = {
                "logical_launches_per_bias_call": 7,
                "underlying_launches_per_bias_call": 9,
                "split_level": 4,
                "split_body_order": [2, 7, 11],
                "split_uses_live_initialized_alias": True,
                "sensory_passthrough_calls": 2,
                "closed": True,
            }
            for row in scope["calls"]:
                row.pop("original_launches")
                row.update(
                    logical_launches=7,
                    underlying_launches=9,
                    split_level=4,
                    split_body_order=[2, 7, 11],
                    split_uses_live_initialized_alias=True,
                    underlying_body_groups=[
                        [6, 15],
                        [5, 10, 14],
                        [4, 9, 13],
                        [3, 8, 12],
                        [2],
                        [7],
                        [11],
                        [1],
                        [0],
                    ],
                )
        cases[case] = record
        for name in receiver.CASE_CAPS:
            files[case + "." + name] = value["files"][name]
    child = {
        k: deepcopy(old_child[k])
        for k in (
            "source",
            "packages",
            "owner_pid",
            "child_pid",
            "child_ppid",
            "integration_calls",
            "flags",
        )
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
    root = tmp_path / "paired"
    root.mkdir()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return dict(
        root=root,
        files=files,
        anchors={k: dict(bytes=len(v), sha256=sha(v)) for k, v in files.items()},
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


def test_complete_four_case_synthetic_positive_keeps_all_admission_false(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    result = verify(value)
    assert result["decision"] == "fresh-coupled-rne-control-exact"
    assert result["all_actual_initialized_rne_inputs_exact"]
    assert result["all_actual_initialized_com_inputs_exact"]
    assert result["all_serial_rne_arithmetic_exact"]
    assert result["paired_frames"]["serial_repeat_exact"]
    assert result["per_case_temporal"]["serial_temporal_exact"]
    assert len(result["paired_frames"]["comparisons"]) == 12
    assert len(result["per_case_temporal"]["temporal_comparisons"]) == 4
    assert all(flag is False for flag in result["flags"].values())
    assert len(result["receiver_files"]) == 30


@pytest.mark.parametrize(
    "file",
    ["original0.frames.bin", "serial1.outputs.bin", "child.json", "tests.receipt.json"],
)
def test_every_whole_anchor_precedes_any_json_or_array_decode(
    tmp_path, monkeypatch, file
):
    value = fixture(tmp_path, monkeypatch)
    raw = value["files"][file]
    (value["root"] / file).write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
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
        "split-order",
        "group-order",
        "launches",
        "alias",
        "post-count",
        "dispatch-open",
        "layout",
    ],
)
def test_serial_dispatch_claims_are_literal_and_bound(tmp_path, monkeypatch, mutation):
    value = fixture(tmp_path, monkeypatch)

    def mutate(child):
        scope = child["cases"]["serial0"]["rne_scope"]
        row = scope["calls"][0]
        if mutation == "split-order":
            row["split_body_order"] = [7, 2, 11]
        elif mutation == "group-order":
            row["underlying_body_groups"][4] = [7]
        elif mutation == "launches":
            row["underlying_launches"] = 7
        elif mutation == "alias":
            row["split_uses_live_initialized_alias"] = False
        elif mutation == "post-count":
            scope["unchanged_postconstraint_passthrough_calls"] = 1
        elif mutation == "dispatch-open":
            scope["dispatch"]["closed"] = False
        elif mutation == "layout":
            row["layout"]["strides"] = [1536, 24]

    rewrite_child(value, mutate)
    with pytest.raises(ValueError):
        verify(value)


def field_offset(name):
    return sum(
        64 * (int(np.prod(shape)) if shape else 1) * 4
        for field, shape in receiver.frames.FRAME_FIELDS[
            : next(
                i
                for i, (field, _) in enumerate(receiver.frames.FRAME_FIELDS)
                if field == name
            )
        ]
    )


def test_serial_temporal_delta_is_negative_despite_paired_equality(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    for case in ("serial0", "serial1"):
        packet = bytearray(value["files"][case + ".frames.bin"])
        offset = receiver.frames.FRAME_BYTES + field_offset("qfrc_bias")
        packet[offset : offset + 4] = np.float32(0.25).tobytes()
        replace(value, case + ".frames.bin", bytes(packet))
    result = verify(value)
    assert result["paired_frames"]["serial_repeat_exact"]
    assert not result["per_case_temporal"]["serial_temporal_exact"]
    assert result["per_case_temporal"]["negative_coordinate_count"] == 2
    assert result["decision"] == "fresh-coupled-rne-control-negative"
    assert all(flag is False for flag in result["flags"].values())


def test_serial_repeat_delta_is_negative_despite_per_case_temporal_equality(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    packet = bytearray(value["files"]["serial1.frames.bin"])
    for boundary in (0, 1):
        offset = boundary * receiver.frames.FRAME_BYTES + field_offset("qfrc_bias")
        packet[offset : offset + 4] = np.float32(0.25).tobytes()
    replace(value, "serial1.frames.bin", bytes(packet))
    result = verify(value)
    assert result["per_case_temporal"]["serial_temporal_exact"]
    assert not result["paired_frames"]["serial_repeat_exact"]
    assert result["decision"] == "fresh-coupled-rne-control-negative"


def test_original_negative_is_retained_without_turning_into_historical_cause(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    packet = bytearray(value["files"]["original0.frames.bin"])
    offset = receiver.frames.FRAME_BYTES + field_offset("qfrc_smooth")
    packet[offset : offset + 4] = np.float32(-0.25).tobytes()
    replace(value, "original0.frames.bin", bytes(packet))
    result = verify(value)
    assert not result["per_case_temporal"]["original_temporal_exact"]
    assert result["decision"] == "fresh-coupled-rne-control-exact"
    assert result["flags"]["runtime_cause_proven"] is False


def test_wrong_actual_initialized_serial_entry_blocks_gate_even_with_unchanged_frames(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    raw = bytearray(value["files"]["serial0.entries.bin"])
    raw[0:4] = np.float32(0.25).tobytes()
    replace(value, "serial0.entries.bin", bytes(raw))
    rewrite_child(
        value,
        lambda c: c["cases"]["serial0"]["rne_scope"]["calls"][0].update(
            input_sha256=sha(bytes(raw[: receiver.ENTRY_BYTES]))
        ),
    )
    result = verify(value)
    assert not result["all_actual_initialized_rne_inputs_exact"]
    assert not result["all_serial_rne_arithmetic_exact"]
    assert result["decision"] == "fresh-coupled-rne-control-negative"


@pytest.mark.parametrize(
    "mutation",
    ["case-order", "subset", "graph", "step", "mode", "nonfinite", "fixed-zero", "rng"],
)
def test_missing_case_or_changed_physical_contract_refused(
    tmp_path, monkeypatch, mutation
):
    value = fixture(tmp_path, monkeypatch)
    if mutation == "nonfinite" or mutation == "fixed-zero":
        raw = bytearray(value["files"]["serial0.frames.bin"])
        offset = (
            field_offset("qvel") if mutation == "fixed-zero" else field_offset("crb")
        )
        raw[offset : offset + 4] = (
            np.float32(-0.0) if mutation == "fixed-zero" else np.float32(np.nan)
        ).tobytes()
        replace(value, "serial0.frames.bin", bytes(raw))
    elif mutation == "rng":
        raw = bytearray(value["files"]["serial0.rng.bin"])
        raw[0] ^= 1
        replace(value, "serial0.rng.bin", bytes(raw))
    else:

        def mutate(c):
            if mutation == "case-order":
                c["case_order"].reverse()
            elif mutation == "subset":
                c["cases"].pop("original1")
            elif mutation == "graph":
                c["cases"]["serial0"]["graph_created"] = True
            elif mutation == "step":
                c["cases"]["serial0"]["physics_steps"] = 1
            elif mutation == "mode":
                c["cases"]["serial0"]["mode"] = "original"

        rewrite_child(value, mutate)
    with pytest.raises(ValueError):
        verify(value)


def test_external_source_tree_and_leaf_inventory_cannot_be_self_asserted(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        verify(value, tree="d" * 40)
    with pytest.raises(ValueError):
        verify(value, leaves="e" * 64)


def test_exact_inventory_has_no_detached_banks_subset_or_unanchored_addition(
    tmp_path, monkeypatch
):
    value = fixture(tmp_path, monkeypatch)
    (value["root"] / "serial.bin").write_bytes(b"not allowed")
    with pytest.raises(ValueError):
        verify(value)
