"""CPU-only contracts; actual frozen plant replay is a separate retained run."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mjlab_microduck import ada_collision_replay as p


def states():
    values = {k: np.zeros((2, p.p.base.WIDTHS[k]), np.float32) for k in p.p.base.UNCHANGED}
    values["qpos"][0, 2] = .11728236
    values["qpos"][1, 2] = .11708236
    arrays = {"/data/" + k: np.frombuffer(v.tobytes(), np.uint8) for k, v in values.items()}
    report = {"child": {"input_manifest": {k: {"sha256": sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}}
    return report, {"prepared-inputs.npz": arrays, "gpu-fields.npz": values}


def test_exact_prepared_states_no_offset_recalculation():
    report, banks = states()
    actual = p.state_inputs(report, banks)
    for k, v in actual.items():
        assert v.tobytes() == banks["gpu-fields.npz"][k].tobytes()
        assert not np.shares_memory(v, banks["gpu-fields.npz"][k])
    assert actual["qpos"][1, 2] == np.float32(.11708236)


@pytest.mark.parametrize("key", p.COLLISION_OPTIONS)
def test_collision_scalar_options_are_not_reporting_only(key):
    opt = SimpleNamespace(**p.COLLISION_OPTIONS)
    assert p.collision_options(opt) == p.COLLISION_OPTIONS
    setattr(opt, key, 888)
    with pytest.raises(ValueError): p.collision_options(opt)


@pytest.mark.parametrize("damage", ["none", "dtype", "shape", "ulp", "flag"])
def test_warp_ccd_tolerance_is_bound_as_exact_float32_array_not_scalar(damage):
    value = np.asarray([1e-6], dtype=np.float32)
    if damage == "dtype": value = value.astype(np.float64)
    if damage == "shape": value = np.repeat(value, 2)
    if damage == "ulp": value[0] = np.nextafter(value[0], np.float32(1))
    opt = SimpleNamespace(**p.COLLISION_OPTIONS)
    opt.ccd_tolerance = SimpleNamespace(numpy=lambda: value)
    if damage == "flag": opt.disableflags = 1
    if damage == "none": assert p.collision_options(opt, warp=True) == p.WARP_COLLISION_OPTIONS
    else:
        with pytest.raises(ValueError): p.collision_options(opt, warp=True)


@pytest.mark.parametrize("key", p.p.base.UNCHANGED)
@pytest.mark.parametrize("damage", ["hash", "length", "state"])
def test_state_binding_refuses(key, damage):
    report, banks = states()
    if damage == "hash": report["child"]["input_manifest"]["/data/" + key]["sha256"] = "0" * 64
    elif damage == "length": banks["prepared-inputs.npz"]["/data/" + key] = np.zeros(3, np.uint8)
    else: banks["gpu-fields.npz"][key][0, 0] = 1.
    with pytest.raises(ValueError): p.state_inputs(report, banks)


def candidates():
    native = {}
    warp = {k: np.zeros((2,)) for k in p.CONTACT}
    warp.update(worldid=np.array([1, 1]), type=np.array([1, 1]), geom=np.array([[0, 29], [29, 0]]), dim=np.array([3, 3]),
                efc_address=np.full((2, 4), -1), dist=np.array([-.01, .02]), includemargin=np.zeros(2))
    for w in range(2):
        count = w
        for k in p.CONTACT: native[f"{w}/{k}"] = np.zeros(count)
        native[f"{w}/geom"] = np.array([[0, 29]] * count).reshape(count, 2)
        native[f"{w}/dim"] = np.full(count, 3)
        native[f"{w}/efc_address"] = np.full(count, -1)
        native[f"{w}/dist"] = np.full(count, -.01)
    return native, warp


def test_candidate_scratch_addresses_not_inclusion_keys_and_pair_order_preserved():
    native, warp = candidates()
    result = p.candidate_summary(native, warp)
    assert result["native"][0]["count"] == 0
    assert result["native"][1]["within_includemargin_count"] == 1
    assert result["warp_cpu"][1]["constraint_type_bit_count"] == 2
    assert result["warp_cpu"][1]["groups"] == [
        {"ordered_key": [0, 29, 3, True], "raw_slots": [0]},
        {"ordered_key": [29, 0, 3, False], "raw_slots": [1]}]


@pytest.mark.parametrize("side", ["native", "warp"])
@pytest.mark.parametrize("damage", ["efc", "nan"])
def test_candidate_solver_or_nonfinite_refuses(side, damage):
    native, warp = candidates()
    target = native["1/efc_address" if damage == "efc" else "1/dist"] if side == "native" else warp["efc_address" if damage == "efc" else "dist"]
    target.flat[0] = 0 if damage == "efc" else np.nan
    with pytest.raises(ValueError): p.candidate_summary(native, warp)


def test_kinematic_residual_is_descriptive_not_tolerance():
    a = {"geom_xpos": np.zeros((2, 82, 3), np.float64)}
    b = {"geom_xpos": np.ones((2, 82, 3), np.float32)}
    result = p.differences(a, b)["geom_xpos"]
    assert result == dict(max_abs=1., float32_bit_mismatches=492)


@pytest.mark.parametrize("damage", ["keys", "shape", "nan"])
def test_invalid_kinematic_bank_refuses(damage):
    a, b = {"xpos": np.zeros((2, 16, 3))}, {"xpos": np.zeros((2, 16, 3))}
    if damage == "keys": b = {}
    if damage == "shape": b["xpos"] = np.zeros((1, 16, 3))
    if damage == "nan": b["xpos"][0, 0, 0] = np.nan
    with pytest.raises(ValueError): p.differences(a, b)


def test_all_model_array_bytes_and_layout_bound(monkeypatch):
    class Array:
        dtype = "synthetic"
        shape = (1, 3)
        strides = (0, 4)
        def numpy(self): return np.arange(3, dtype=np.float32).reshape(1, 3)
    array = Array(); key = "/model/mesh_vert"
    monkeypatch.setattr(p.p, "all_arrays", lambda *args: {key: array})
    row = dict(dtype="synthetic", shape=[1, 3], strides=[0, 4])
    report = {"child": {"input_manifest": {key: row}}}
    banks = {"prepared-inputs.npz": {key: np.frombuffer(array.numpy().tobytes(), dtype=np.uint8)}}
    assert p.model_binding(None, report, banks) == {key: sha256(array.numpy().tobytes()).hexdigest()}
    row["shape"] = [3]
    with pytest.raises(ValueError): p.model_binding(None, report, banks)
    row["shape"] = [1, 3]; banks["prepared-inputs.npz"][key] = np.zeros(12, np.uint8)
    with pytest.raises(ValueError): p.model_binding(None, report, banks)
    report["child"]["input_manifest"]["/model/missing"] = row.copy()
    with pytest.raises(ValueError): p.model_binding(None, report, banks)


def test_complete_warp_candidate_prefix_including_sidecars():
    from dataclasses import dataclass
    class Array:
        def __init__(self, value): self.value = np.asarray(value)
        def numpy(self): return self.value
    @dataclass
    class Contact:
        worldid: object
        geomcollisionid: object
        type: object
    contact = Contact(Array([1, 1, 88]), Array([3, 2, 88]), Array([1, 1, 88]))
    data = SimpleNamespace(nacon=Array([2]), naconmax=3, contact=contact)
    result = p.warp_candidates(data)
    assert set(result) == {"worldid", "geomcollisionid", "type"}
    assert result["geomcollisionid"].tolist() == [3, 2]
    data.nacon = Array([4])
    with pytest.raises(ValueError): p.warp_candidates(data)


def test_exclusive_finite_retention_and_array_hashes(tmp_path):
    arrays = {"geometry": np.ones((2, 3), np.float32)}
    path = tmp_path / "replay.npz"
    manifest = p.retain(path, arrays)
    assert manifest["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert manifest["arrays"]["geometry"]["sha256"] == sha256(arrays["geometry"].tobytes()).hexdigest()
    with pytest.raises(FileExistsError): p.retain(path, arrays)
    arrays["geometry"][0, 0] = np.nan
    with pytest.raises(ValueError): p.retain(tmp_path / "bad.npz", arrays)
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize("cuda,source", [("0", "a" * 40), ("", "bad"), ("", "a" * 40)])
def test_cli_refuses_before_output(tmp_path, cuda, source):
    output = tmp_path / "replay.json"
    command = [sys.executable, "-m", p.__name__, "--input", str(tmp_path), "--output", str(output),
               "--cache", str(tmp_path / "cache"), "--source", source]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=cuda))
    assert result.returncode != 0
    assert "explicit CPU-hidden" in result.stderr or "clean exact collision" in result.stderr
    assert not output.exists() and not (tmp_path / "cache").exists()


def test_call_boundary_source_fence():
    import ast
    tree = ast.parse(Path(p.__file__).read_text())
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert {"mujoco.mj_kinematics", "mujoco.mj_collision", "kinematics", "mjwarp.collision"} <= calls
    forbidden = {"mujoco.mj_forward", "mujoco.mj_step", "mjwarp.forward", "mjwarp.fwd_position",
                 "mjwarp.make_constraint", "mujoco.mj_contactForce", "plant.describe", "plant.reference", "WarpStanceRuntime"}
    assert not calls & forbidden


@pytest.mark.parametrize("damage", ["flag", "source", "input", "model", "calls", "options", "counter", "payload"])
def test_receiver_refuses_changed_contract_before_array_decoding(tmp_path, monkeypatch, damage):
    source = "a" * 40
    report = dict(files={}, child=dict(input_manifest={}, plant=dict(selected_fields_sha256="b" * 64)))
    monkeypatch.setattr(p.prior, "authenticated_banks", lambda root: (report, {}))
    monkeypatch.setattr(p.p.base.host, "read", lambda *args, **kwargs: b"module")
    result = dict(protocol=p.PROTOCOL, source=source, predecessor_report_sha256=p.prior.REPORT_SHA256,
                  predecessor_source=p.prior.SOURCE, decision="collision-boundary-diagnostic-not-admission",
                  input_file_sha256={}, model_array_sha256={}, flags=dict(p.p.FLAGS),
                  training_authorized=False, physical_motion_authorized=False,
                  new_constraint_calls=0, new_solver_calls=0, new_integration_steps=0,
                  module_sha256=sha256(b"module").hexdigest(), versions=p.VERSIONS,
                  measured_native_kinematics_calls=2, native_collision_calls=2,
                  allocation_native_kinematics_calls=1, measured_warp_kinematics_calls=1, warp_cpu_collision_calls=1,
                  warp_collision_options=dict(p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11),
                  plant_bindings=[dict(collision_options=p.COLLISION_OPTIONS, selected_fields_sha256="b" * 64)] * 2,
                  counters=dict(native=[dict(nefc=0)] * 2, warp_cpu=dict(nefc=[0, 0])),
                  payload=dict(file="result.npz", bytes=1, sha256="0" * 64))
    if damage == "flag": result["flags"][next(iter(result["flags"]))] = True
    if damage == "source": result["source"] = "c" * 40
    if damage == "input": result["input_file_sha256"] = {"unbound": "c" * 64}
    if damage == "model": result["model_array_sha256"] = {"unbound": "c" * 64}
    if damage == "calls": result["native_collision_calls"] = 3
    if damage == "options": result["warp_collision_options"]["disableflags"] = 1
    if damage == "counter": result["counters"]["warp_cpu"]["nefc"][0] = 1
    if damage == "payload": (tmp_path / "result.npz").write_bytes(b"X")
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError): p.receive(path, source, tmp_path)
