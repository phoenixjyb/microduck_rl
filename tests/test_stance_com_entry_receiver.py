"""Synthetic whole-byte negative controls; no CUDA/ownership acceptance."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from mjlab_microduck import stance_com_entry_receiver as r

SOURCE = "c" * 40


def prediction(entry):
    values = np.frombuffer(entry, dtype="<f4").reshape(64, 16, 3).copy()
    for group in r.GROUPS:
        for body in group:
            if body:
                np.add(
                    values[:, r.PARENTS[body]],
                    values[:, body],
                    out=values[:, r.PARENTS[body]],
                )
    return values.tobytes() * 32


def fixture(tmp_path, *, input_negative=False):
    entry = (np.arange(3072, dtype=np.float32).reshape(64, 16, 3) / 1024).tobytes()
    second = entry
    if input_negative:
        changed = np.frombuffer(entry, dtype="<f4").copy()
        changed[7] += np.float32(0.25)
        second = changed.tobytes()
    flags = dict(r.FLAGS)
    binding = {
        "source": SOURCE,
        "branch": "feat/athletics-obstacle-curriculum",
        "tree": "d" * 40,
        "leaf_count": 622,
        "leaves_sha256": "e" * 64,
    }
    live = {
        **r.SERVICE_CAPS,
        "MainPID": "123",
        "InvocationID": "a" * 32,
        "ActiveState": "active",
        "SubState": "running",
    }
    declaration = {
        "protocol": r.DATA_PROTOCOL + ":declaration",
        "source": SOURCE,
        "source_binding": binding,
        "service_properties": live,
        "tests_receipt_sha256": "e" * 64,
        "tests_terminal_properties": {
            **r.SERVICE_CAPS,
            "LimitFSIZE": "67108864",
            "MainPID": "0",
            "Result": "success",
            "ExecMainStatus": "0",
            "ActiveState": "active",
            "SubState": "exited",
            "InvocationID": "b" * 32,
        },
        "packages": {
            "versions": r.VERSIONS,
            "python": "3.12.13",
            "architecture": "x86_64",
            "python_trees": {
                name: {"files": 100, "sha256": "e" * 64}
                for name in ("mjlab", "mujoco-warp", "better-actuator-models")
            },
        },
        "host": {
            "machine": r.MACHINE,
            "gpu": r.GPU,
            "driver": "595.95",
            "driver_model": "WDDM",
            "temperature_c": 30,
            "used_mib": 666,
            "free_mib": 23496,
            "processes": [],
        },
        "flags": flags,
    }
    declaration_raw = r.canonical(declaration)
    calls = [
        {
            "call_index": i,
            "worlds": 64,
            "device": "cuda:0",
            "stream": 0,
            "subtree_com_layout": {
                "object_id": 10,
                "ptr": 1000,
                "shape": [64, 16],
                "strides": [192, 12],
                "dtype": "vec3f",
                "contiguous": True,
            },
            "smooth_sha256": "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f",
            "forward_sha256": "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
            "bytes": 12288,
            "sha256": sha256(data).hexdigest(),
            "original_launches": 11,
            "boundary": "after-init-before-first-accumulation",
            "initialization_and_accumulation_output_alias": True,
            "readback_and_device_sync_perturb_timing": True,
        }
        for i, data in enumerate((entry, second))
    ]
    child = {
        "protocol": r.DATA_PROTOCOL + ":child",
        "source": SOURCE,
        "source_binding": binding,
        "declaration_sha256": sha256(declaration_raw).hexdigest(),
        "owner_pid": 123,
        "child_pid": 456,
        "child_ppid": 123,
        "compiled_descriptor": {
            "selected_fields_sha256": "6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f"
        },
        "runtime_ntendon": 0,
        "integration_calls": 0,
        "physics_steps": 0,
        "graph_created": False,
        "actor_model_created": False,
        "optimizer_created": False,
        "storage_created": False,
        "fixed_state_unchanged": True,
        "flags": flags,
        "capture_receipt": {
            "protocol": "microduck-com-actual-entry-capture-v1",
            "status": "complete",
            "fault_type": None,
            "parents": list(r.PARENTS),
            "reversed_levels": [
                [6, 15],
                [5, 10, 14],
                [4, 9, 13],
                [3, 8, 12],
                [2, 7, 11],
                [1],
                [0],
            ],
            "flags": flags,
            "calls": calls,
        },
        "decision": "fresh-entry-inputs-differ"
        if input_negative
        else "fixed-input-banks-retained",
        "repeat_rng_unchanged": not input_negative,
        "repeat_receipts": {},
    }
    raw_files = {
        "declaration.json": declaration_raw,
        "entry0.bin": entry,
        "entry1.bin": second,
        "child.log": b"synthetic fixture, not native evidence\n",
    }
    if not input_negative:
        for mode, count in (("concurrent", 224), ("serial", 288)):
            bank = prediction(entry)
            raw_files[mode + ".bin"] = bank
            child["repeat_receipts"][mode] = {
                "mode": mode,
                "repeats": 32,
                "resets": 32,
                "accumulation_launches": count,
                "bytes": 393216,
                "input_sha256": sha256(entry).hexdigest(),
                "output_sha256": sha256(bank).hexdigest(),
                "output_boundary": "after-accumulation-before-division",
                "stream": 0,
                "scratch_ptr": 2000,
                "scratch_shape": [64, 16],
                "scratch_strides": [192, 12],
                "input_output_alias": True,
                "flags": flags,
            }
    raw_files["child.json"] = r.canonical(child)
    report = {
        "protocol": r.DATA_PROTOCOL + ":report",
        "source": SOURCE,
        "source_binding": binding,
        "child_sha256": sha256(raw_files["child.json"]).hexdigest(),
        "returncode": 0,
        "observed_child_pid": 456,
        "observed_child_ppid": 123,
        "gpu_child_observed": True,
        "monitor": [
            {
                "elapsed": 1.0,
                "child_pid": 456,
                "host": {
                    **declaration["host"],
                    "processes": [
                        {
                            "pid": 456,
                            "memory_mib": 1024,
                            "memory_status": "reported",
                            "raw_memory": "1024",
                        }
                    ],
                },
            }
        ],
        "flags": flags,
    }
    raw_files["report.json"] = r.canonical(report)
    terminal = {
        "source": SOURCE,
        "unit": "microduck-com-entry-run-" + SOURCE[:12] + ".service",
        "service_properties": {
            **live,
            "MainPID": "0",
            "SubState": "exited",
            "Result": "success",
            "ExecMainStatus": "0",
        },
    }
    for name, raw in raw_files.items():
        (tmp_path / name).write_bytes(raw)
    return raw_files, terminal


def verify(tmp_path, files, terminal):
    anchors = {
        name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        for name, raw in files.items()
    }
    raw = r.canonical(terminal)
    return r.verify(tmp_path, SOURCE, anchors, raw, sha256(raw).hexdigest())


def resign(tmp_path, files, *, child=None, declaration=None, report=None):
    if declaration is not None:
        files["declaration.json"] = r.canonical(declaration)
        if child is None:
            child = json.loads(files["child.json"])
        child["declaration_sha256"] = sha256(files["declaration.json"]).hexdigest()
    if child is not None:
        files["child.json"] = r.canonical(child)
        if report is None:
            report = json.loads(files["report.json"])
        report["child_sha256"] = sha256(files["child.json"]).hexdigest()
    if report is not None:
        files["report.json"] = r.canonical(report)
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)


@pytest.fixture(autouse=True)
def cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_complete_banks_independent_reference_and_no_admission(tmp_path):
    files, terminal = fixture(tmp_path)
    result = verify(tmp_path, files, terminal)
    assert result["decision"] == "isolated-serial-reference-exact"
    assert result["decision_is_original_or_training_admission"] is False
    assert all(value is False for value in result["flags"].values())
    assert all(row["scalars_compared"] == 98304 for row in result["analyses"].values())
    assert all(row["mismatched_scalars"] == 0 for row in result["analyses"].values())
    report = json.loads(files["report.json"])
    process = report["monitor"][0]["host"]["processes"][0]
    process.update(
        memory_mib=None, memory_status="unavailable-wddm", raw_memory="[N/A]"
    )
    resign(tmp_path, files, report=report)
    assert verify(tmp_path, files, terminal)["decision"] == result["decision"]
    process["memory_mib"] = 0
    resign(tmp_path, files, report=report)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


def test_input_negative_never_accepts_output_banks(tmp_path):
    files, terminal = fixture(tmp_path, input_negative=True)
    result = verify(tmp_path, files, terminal)
    assert result["decision"] == "fresh-entry-inputs-differ"
    assert result["input_mismatched_scalars"] == 1 and result["analyses"] == {}
    files["serial.bin"] = prediction(files["entry0.bin"])
    (tmp_path / "serial.bin").write_bytes(files["serial.bin"])
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize("name", sorted(r.CAPS))
def test_each_whole_file_corruption_stops_before_decode(tmp_path, monkeypatch, name):
    files, terminal = fixture(tmp_path)
    original = files[name]
    (tmp_path / name).write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    def forbidden(raw):
        raise AssertionError("no JSON decode before ALL complete hashes")

    monkeypatch.setattr(r, "_json", forbidden)
    with pytest.raises(ValueError, match="whole external"):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize("key", list(r.SERVICE_CAPS))
def test_terminal_cap_changes_rejected_even_with_new_hash(tmp_path, key):
    files, terminal = fixture(tmp_path)
    terminal["service_properties"][key] = "foreign"
    with pytest.raises(ValueError, match="actual live/terminal caps"):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "key,value",
    [
        ("MainPID", "999"),
        ("InvocationID", "b" * 32),
        ("ExecMainStatus", "1"),
        ("Result", "exit-code"),
    ],
)
def test_completed_invocation_required(tmp_path, key, value):
    files, terminal = fixture(tmp_path)
    terminal["service_properties"][key] = value
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "key,value",
    [
        ("integration_calls", 1),
        ("physics_steps", 1),
        ("graph_created", True),
        ("actor_model_created", True),
        ("optimizer_created", True),
        ("storage_created", True),
        ("fixed_state_unchanged", False),
        ("child_ppid", 999),
    ],
)
def test_changed_execution_contract_rejected_with_resigned_files(tmp_path, key, value):
    files, terminal = fixture(tmp_path)
    child = json.loads(files["child.json"])
    child[key] = value
    resign(tmp_path, files, child=child)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "key,value",
    [
        ("worlds", 63),
        ("boundary", "post-forward"),
        ("device", "cpu"),
        ("original_launches", 10),
        ("readback_and_device_sync_perturb_timing", False),
        ("initialization_and_accumulation_output_alias", False),
        ("stream", True),
    ],
)
def test_exact_actual_entry_stage_required(tmp_path, key, value):
    files, terminal = fixture(tmp_path)
    child = json.loads(files["child.json"])
    child["capture_receipt"]["calls"][0][key] = value
    resign(tmp_path, files, child=child)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize("body", [0, 1, 15])
def test_root_trunk_and_other_cell_variations_retained(tmp_path, body):
    files, terminal = fixture(tmp_path)
    values = (
        np.frombuffer(files["serial.bin"], dtype="<f4").copy().reshape(32, 64, 16, 3)
    )
    values[7, 13, body, 2] += np.float32(0.5)
    files["serial.bin"] = values.tobytes()
    child = json.loads(files["child.json"])
    child["repeat_receipts"]["serial"]["output_sha256"] = sha256(
        files["serial.bin"]
    ).hexdigest()
    resign(tmp_path, files, child=child)
    result = verify(tmp_path, files, terminal)
    assert result["decision"] == "isolated-serial-reference-negative"
    row = result["analyses"]["serial"]
    assert row["mismatched_scalars"] == row["varying_cells"] == 1
    assert row["unique_full_snapshots"] == 2


def test_constant_wrong_bank_is_not_variability_acceptance(tmp_path):
    files, terminal = fixture(tmp_path)
    values = (
        np.frombuffer(files["serial.bin"], dtype="<f4").copy().reshape(32, 64, 16, 3)
    )
    values[:, 13, 0, 2] += np.float32(0.5)
    result = r.analyze_bank(files["entry0.bin"], values.tobytes())
    assert result["mismatched_scalars"] == 32 and result["varying_cells"] == 0
    assert result["unique_full_snapshots"] == 1


def test_signed_zero_variation_is_raw_bit_negative():
    entry = np.zeros((64, 16, 3), dtype="<f4").tobytes()
    values = np.zeros((32, 64, 16, 3), dtype="<f4")
    values[7, 13, 0, 2] = np.float32(-0.0)
    row = r.analyze_bank(entry, values.tobytes())
    assert row["mismatched_scalars"] == row["varying_cells"] == 1
    assert row["max_abs_delta"] == 0


def test_symlink_extra_file_and_duplicate_json_rejected(tmp_path):
    files, terminal = fixture(tmp_path)
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(ValueError, match="extra or missing"):
        verify(tmp_path, files, terminal)
    with pytest.raises(ValueError, match="duplicate"):
        r._json(b'{"x":1,"x":2}')
    with pytest.raises(ValueError, match="nonfinite"):
        r._json(b'{"x":NaN}')


def test_nonfinite_bank_rejected():
    entry = np.zeros((64, 16, 3), dtype="<f4").tobytes()
    values = np.zeros((32, 64, 16, 3), dtype="<f4")
    values[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite full"):
        r.analyze_bank(entry, values.tobytes())


def explain(tmp_path, files, terminal):
    anchors = {
        name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        for name, raw in files.items()
    }
    raw = r.canonical(terminal)
    return r.explain_order_hypotheses(
        tmp_path, SOURCE, anchors, raw, sha256(raw).hexdigest()
    )


def test_authenticated_hypotheses_preserve_original_receiver(tmp_path):
    files, terminal = fixture(tmp_path)
    original = verify(tmp_path, files, terminal)
    result = explain(tmp_path, files, terminal)
    assert (
        result["retained_receiver_sha256"] == sha256(r.canonical(original)).hexdigest()
    )
    assert result["retained_receiver_decision"] == original["decision"]
    assert all(value is False for value in result["flags"].values())
    for row in result["analyses"].values():
        assert row["scalars_compared"] == 98304 and row["cells_compared"] == 3072
        assert (
            len(row["candidate_orders"]) == len(row["candidate_snapshot_sha256"]) == 6
        )
        assert row["unexplained_scalars"] == 0 and row["affected_cells"] == []
        assert row["atomic_order_observed"] is False
        assert row["compatible_bits_are_causal_or_training_admission"] is False


def test_hypothesis_authentication_precedes_arithmetic(tmp_path, monkeypatch):
    files, terminal = fixture(tmp_path)
    (tmp_path / "concurrent.bin").write_bytes(b"x" * r.BANK_BYTES)
    monkeypatch.setattr(
        r,
        "analyze_order_hypotheses",
        lambda *_: pytest.fail("arithmetic before complete authentication"),
    )
    with pytest.raises(ValueError, match="whole external"):
        explain(tmp_path, files, terminal)


def test_hypotheses_reject_differing_actual_entry_inputs(tmp_path):
    files, terminal = fixture(tmp_path, input_negative=True)
    with pytest.raises(ValueError, match="same complete"):
        explain(tmp_path, files, terminal)


@pytest.mark.parametrize("body", [0, 1, 15])
def test_hypotheses_do_not_hide_constant_wrong_cells(body):
    entry = np.zeros((64, 16, 3), dtype="<f4").tobytes()
    values = np.zeros((32, 64, 16, 3), dtype="<f4")
    values[:, 13, body, 2] = np.float32(0.5)
    row = r.analyze_order_hypotheses(entry, values.tobytes())
    assert row["unexplained_scalars"] == 32
    assert row["repeat_compatible_uniform_orders"] == [[]] * 32
    (cell,) = row["affected_cells"]
    assert (cell["world"], cell["body"], cell["axis"]) == (13, body, 2)
    assert cell["mismatched_repeats"] == 32 and cell["varying"] is False
    assert cell["variants"] == [
        {"bits": "3f000000", "count": 32, "compatible_candidate_indices": []}
    ]


def test_cancellation_orders_are_hypotheses_not_observed_order():
    initial = np.zeros((64, 16, 3), dtype="<f4")
    initial[3, 2, 0], initial[3, 7, 0], initial[3, 11, 0] = 1e20, -1e20, 1
    values = np.frombuffer(prediction(initial.tobytes()), dtype="<f4").copy()
    values = values.reshape(32, 64, 16, 3)
    values[0, 3, [0, 1], 0] = 0
    row = r.analyze_order_hypotheses(initial.tobytes(), values.tobytes())
    assert row["candidate_orders"] == [
        [2, 7, 11],
        [2, 11, 7],
        [7, 2, 11],
        [7, 11, 2],
        [11, 2, 7],
        [11, 7, 2],
    ]
    assert row["unexplained_scalars"] == 0
    assert row["repeat_compatible_uniform_orders"][0] == [1, 3, 4, 5]
    assert row["repeat_compatible_uniform_orders"][1:] == [[0, 2]] * 31
    assert len(row["affected_cells"]) == 2
    assert all(cell["mismatched_repeats"] == 1 for cell in row["affected_cells"])
    assert row["atomic_order_observed"] is False


@pytest.mark.parametrize("kind", ["cross-world", "cross-axis"])
def test_cellwise_compatibility_does_not_invent_one_uniform_order(kind):
    initial = np.zeros((64, 16, 3), dtype="<f4")
    for world, axis in ((3, 0), (3, 2), (4, 0)):
        initial[world, 2, axis] = 1e20
        initial[world, 7, axis] = -1e20
        initial[world, 11, axis] = 1
    values = np.frombuffer(prediction(initial.tobytes()), dtype="<f4").copy()
    values = values.reshape(32, 64, 16, 3)
    world, axis = (4, 0) if kind == "cross-world" else (3, 2)
    values[:, world, [0, 1], axis] = 0
    row = r.analyze_order_hypotheses(initial.tobytes(), values.tobytes())
    assert row["unexplained_scalars"] == 0
    assert row["repeat_compatible_uniform_orders"] == [[]] * 32
    assert row["atomic_order_observed"] is False
    assert row["compatible_bits_are_causal_or_training_admission"] is False


def test_hypotheses_keep_signed_zero_as_raw_negative():
    entry = np.zeros((64, 16, 3), dtype="<f4").tobytes()
    values = np.zeros((32, 64, 16, 3), dtype="<f4")
    values[7, 13, 0, 2] = np.float32(-0.0)
    row = r.analyze_order_hypotheses(entry, values.tobytes())
    assert row["unexplained_scalars"] == 1
    (cell,) = row["affected_cells"]
    assert cell["reference_bits"] == "00000000"
    assert cell["max_ordered_bit_distance"] == 1
    assert cell["mismatched_repeats"] == 1 and cell["varying"] is True


@pytest.mark.parametrize("kind", ["input", "bank", "intermediate"])
def test_nonfinite_hypotheses_rejected(kind):
    initial = np.zeros((64, 16, 3), dtype="<f4")
    values = np.zeros((32, 64, 16, 3), dtype="<f4")
    if kind == "input":
        initial[0, 1, 0] = np.inf
    elif kind == "bank":
        values[0, 0, 0, 0] = np.nan
    else:
        initial[0, [2, 7], 0] = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="finite"):
        r.analyze_order_hypotheses(initial.tobytes(), values.tobytes())


def test_hypotheses_require_whole_fixed_dimensions():
    entry = np.zeros((64, 16, 3), dtype="<f4").tobytes()
    with pytest.raises(ValueError, match="complete hypothesis"):
        r.analyze_order_hypotheses(entry, b"x")


def test_terminal_whole_hash_precedes_json_decode(tmp_path, monkeypatch):
    files, terminal = fixture(tmp_path)
    anchors = {
        name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        for name, raw in files.items()
    }
    monkeypatch.setattr(
        r, "_json", lambda _: (_ for _ in ()).throw(AssertionError("premature decode"))
    )
    with pytest.raises(ValueError, match="whole external"):
        r.verify(tmp_path, SOURCE, anchors, r.canonical(deepcopy(terminal)), "0" * 64)


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("child", "physics_steps", False),
        ("child", "integration_calls", False),
        ("child", "runtime_ntendon", False),
        ("child", "child_ppid", True),
        ("report", "returncode", False),
        ("report", "observed_child_pid", True),
    ],
)
def test_boolean_integer_impersonation_rejected(tmp_path, target, key, value):
    files, terminal = fixture(tmp_path)
    packet = json.loads(files[target + ".json"])
    packet[key] = value
    resign(tmp_path, files, **{target: packet})
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "no-gpu",
        "foreign-pid",
        "hot",
        "wrong-driver",
        "nan-time",
        "late",
        "duplicate-time",
    ],
)
def test_native_monitor_is_checked_independently(tmp_path, kind):
    files, terminal = fixture(tmp_path)
    report = json.loads(files["report.json"])
    rows = report["monitor"]
    if kind == "empty":
        report["monitor"] = []
    elif kind == "no-gpu":
        rows[0]["host"]["processes"] = []
    elif kind == "foreign-pid":
        rows[0]["host"]["processes"][0]["pid"] = 999
    elif kind == "hot":
        rows[0]["host"]["temperature_c"] = 75
    elif kind == "wrong-driver":
        rows[0]["host"]["driver"] = "changed"
    elif kind == "nan-time":
        rows[0]["elapsed"] = "NaN"
    elif kind == "late":
        rows[0]["elapsed"] = 240
    else:
        rows.append(deepcopy(rows[0]))
    resign(tmp_path, files, report=report)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "kind",
    [
        "branch",
        "source",
        "tree",
        "leaf-count",
        "version",
        "architecture",
        "descriptor",
        "numeric-flags",
    ],
)
def test_resigned_provenance_still_requires_exact_contract(tmp_path, kind):
    files, terminal = fixture(tmp_path)
    declaration, child, report = (
        json.loads(files[name + ".json"]) for name in ("declaration", "child", "report")
    )
    if kind in ("branch", "source", "tree", "leaf-count"):
        key = "leaf_count" if kind == "leaf-count" else kind
        for packet in (declaration, child, report):
            packet["source_binding"][key] = False if kind == "leaf-count" else "foreign"
    elif kind == "version":
        declaration["packages"]["versions"]["torch"] = "changed"
    elif kind == "architecture":
        declaration["packages"]["architecture"] = "arm64"
    elif kind == "descriptor":
        child["compiled_descriptor"]["selected_fields_sha256"] = "0" * 64
    else:
        child["capture_receipt"]["flags"] = {name: 0 for name in r.FLAGS}
    resign(tmp_path, files, declaration=declaration, child=child, report=report)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "message",
    [
        "NaN",
        "nonfinite",
        "overflow",
        "Warning",
        "Traceback",
        "CUDA error",
        "[mdp] Patches 1-2 active: NaN-safe reward/advantage Warning",
    ],
)
def test_independent_receiver_rejects_resigned_log_signals(tmp_path, message):
    files, terminal = fixture(tmp_path)
    files["child.log"] = (message + "\n").encode()
    resign(tmp_path, files)
    with pytest.raises(ValueError, match="owned log"):
        verify(tmp_path, files, terminal)


def test_only_exact_known_registration_line_is_excluded(tmp_path):
    files, terminal = fixture(tmp_path)
    files["child.log"] = b"[mdp] Patches 1-2 active: NaN-safe reward/advantage\n"
    resign(tmp_path, files)
    assert verify(tmp_path, files, terminal)["flags"] == r.FLAGS
