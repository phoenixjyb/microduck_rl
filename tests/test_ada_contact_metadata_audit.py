"""Portable generator-label checks, not physical contact or CUDA acceptance."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
from mjlab_microduck import ada_contact_metadata_audit as p


def table():
    return dict(worldid=np.array([1, 1], np.int32), slot=np.array([0, 1], np.int32),
                geom=np.array([[0, 29], [0, 29]], np.int32), dim=np.array([3, 3], np.int32),
                type=np.array([1, 1], np.int32), geomcollisionid=np.array([0, 1], np.int32),
                dist=np.array([-.0001, -.0002], np.float32), pos=np.ones((2, 3), np.float32),
                frame=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)), friction=np.ones((2, 5), np.float32))


def test_import_does_not_load_runtime_libraries():
    code = "import sys; import mjlab_microduck.ada_contact_metadata_audit; assert not {'numpy','torch','warp','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_permutation_joins_labels_not_slot_or_position_and_does_not_mutate():
    cpu = table(); ada = {k: v[::-1].copy() for k, v in cpu.items()}
    before = {k: v.tobytes() for k, v in ada.items()}
    ada["pos"][0, 0] = 123.
    result = p.compare(cpu, ada)
    assert [r["ada_raw_slot"] for r in result["rows"]] == [0, 1]
    assert result["rows"][1]["common_fields"]["pos"]["ada_minus_cpu"] == [122., 0., 0.]
    assert result["rows"][0]["common_fields"]["pos"]["bytes_equal"]
    assert not result["physical_point_identity_established"]
    assert all(not x["physical_point_identity_established"] for x in result["rows"])
    assert all(v.tobytes() == before[k] for k, v in ada.items() if k != "pos")


def test_independent_raw_slot_assignment_cannot_drive_alignment():
    cpu = table(); ada = {k: v[::-1].copy() for k, v in cpu.items()}
    ada["slot"] = np.arange(2, dtype=np.int32)
    result = p.compare(cpu, ada)
    assert [r["ada_raw_slot"] for r in result["rows"]] == [1, 0]
    assert [r["cpu_raw_slot"] for r in result["rows"]] == [0, 1]
    assert all(r["common_fields"]["dist"]["bytes_equal"] for r in result["rows"])


@pytest.mark.parametrize("key", list(p.INT_FIELDS | p.FLOAT_FIELDS))
@pytest.mark.parametrize("damage", ["missing", "dtype", "shape"])
def test_incomplete_or_changed_layout_refuses(key, damage):
    value = table()
    if damage == "missing": value.pop(key)
    if damage == "dtype": value[key] = value[key].astype(np.int64 if key in p.INT_FIELDS else np.float64)
    if damage == "shape": value[key] = value[key][:1]
    with pytest.raises(ValueError): p.metadata_index(value)


@pytest.mark.parametrize("key", p.FLOAT_FIELDS)
def test_nonfinite_common_fields_refuse(key):
    value = table(); value[key].flat[0] = np.nan
    with pytest.raises(ValueError): p.metadata_index(value)


@pytest.mark.parametrize("key", ["worldid", "geom", "dim", "type", "geomcollisionid"])
def test_changed_key_cannot_be_matched_using_nearest_positions(key):
    cpu, ada = table(), table()
    ada[key].flat[0] += 1
    with pytest.raises(ValueError): p.compare(cpu, ada)


def test_reversed_pair_duplicate_full_key_and_extra_field_refuse():
    for damage in ("reversed", "duplicate", "extra", "slot"):
        cpu, ada = table(), table()
        if damage == "reversed": ada["geom"][0] = [29, 0]
        if damage == "duplicate": ada["geomcollisionid"][1] = 0
        if damage == "extra": ada["force"] = np.zeros((2, 6), np.float32)
        if damage == "slot": ada["slot"][1] = 0
        with pytest.raises(ValueError): p.compare(cpu, ada)


def test_duplicate_group_keys_are_not_physical_identity():
    value = table()
    assert np.array_equal(value["geom"][0], value["geom"][1])
    assert len(p.metadata_index(value)) == 2
    value["geomcollisionid"][1] = 0
    with pytest.raises(ValueError, match="unique complete"): p.metadata_index(value)


def test_one_ulp_and_signed_zero_are_not_normalized():
    a = np.array([0., 1.], np.float32)
    b = np.array([-0., np.nextafter(np.float32(1), np.float32(2))], np.float32)
    result = p.field_difference(a, b)
    assert result["float32_bit_mismatches"] == 2 and not result["bytes_equal"]
    assert result["max_abs"] == 2**-23


def test_fixed_json_authentication_precedes_reception(tmp_path, monkeypatch):
    path = tmp_path / "replay.json"; path.write_bytes(b"{}")
    monkeypatch.setattr(p.replay, "receive", lambda *args: pytest.fail("must refuse before decoding"))
    with pytest.raises(ValueError, match="fixed collision replay JSON"): p.audit(tmp_path, path)


def test_fixed_bank_binding_precedes_predecessor_decode(tmp_path, monkeypatch):
    from hashlib import sha256
    path = tmp_path / "replay.json"; path.write_bytes(b"{}")
    monkeypatch.setattr(p, "REPLAY_SHA256", sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(p.replay, "receive", lambda *args: ({"payload": {"sha256": "0" * 64}}, {}))
    monkeypatch.setattr(p.prior, "authenticated_banks", lambda *args: pytest.fail("must refuse mismatched bank"))
    with pytest.raises(ValueError, match="fixed complete collision"): p.audit(tmp_path, path)


def test_complete_audit_has_stage_pose_and_false_flag_bindings(tmp_path, monkeypatch):
    from hashlib import sha256
    cpu, ada = table(), table()
    ada["pos"][0, 0] += np.float32(2**-23)
    arrays = {"warp_cpu/contact/" + k: v for k, v in cpu.items() if k != "slot"}
    arrays["warp_cpu/contact/includemargin"] = np.zeros(2, np.float32)
    arrays["warp_cpu/contact/efc_address"] = np.full((2, 4), -1, np.int32)
    active = {("sidecar/" if k in ("type", "geomcollisionid") else "contacts/") + k: v for k, v in ada.items()}
    active["contacts/efc_address"] = np.array([14, 18], np.int32)
    prepared = {}
    for name in p.replay.KINEMATIC:
        current = np.zeros((2, 3, 3), np.float32)
        saved = current.copy(); saved.flat[0] = np.float32(2**-23)
        arrays["warp_cpu/pose/" + name] = current
        prepared["/data/" + name] = np.frombuffer(saved.tobytes(), np.uint8)
    path = tmp_path / "replay.json"; path.write_bytes(b"{}")
    monkeypatch.setattr(p, "REPLAY_SHA256", sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(p.replay, "receive", lambda *args: ({"payload": {"sha256": p.REPLAY_BANK_SHA256}}, arrays))
    monkeypatch.setattr(p.prior, "authenticated_banks", lambda *args: ({"files": {"bound": "hash"}},
                        {"gpu-active.npz": active, "prepared-inputs.npz": prepared}))
    result = p.audit(tmp_path, path)
    assert set(result["complete_actual_pose_fields"]) == set(p.replay.KINEMATIC)
    assert all(v["max_abs"] == 2**-23 and v["float32_bit_mismatches"] == 1
               for v in result["complete_actual_pose_fields"].values())
    assert not result["actual_pose_bytes_equal"]
    assert result["stage_binding"]["cpu_efc_addresses"] == [[-1] * 4] * 2
    assert result["stage_binding"]["ada_efc_addresses"] == [14, 18]
    assert not result["stage_binding"]["ada_includemargin_retained"]
    assert not result["stage_binding"]["force_or_efc_address_compared_as_common_fields"]
    for key in ("training_authorized", "physical_motion_authorized", "solver_qualified",
                "same_pose_collision_response_isolated", "compiled_generator_execution_identity_established"):
        assert result[key] is False
    assert all(v is False for v in result["flags"].values())
    assert result["new_collision_calls"] == result["new_solver_calls"] == result["new_integration_steps"] == 0


@pytest.mark.parametrize("cuda,source", [("0", "a" * 40), ("", "bad"), ("", "a" * 40)])
def test_cli_refuses_before_output(tmp_path, cuda, source):
    path = tmp_path / "result.json"
    command = [sys.executable, "-m", p.__name__, "--input", str(tmp_path), "--replay", str(tmp_path / "replay.json"),
               "--output", str(path), "--source", source]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=cuda))
    assert result.returncode != 0
    assert "explicit CPU-hidden" in result.stderr or "clean exact metadata" in result.stderr
    assert not path.exists()


def test_no_new_physics_force_or_tolerance_calls():
    import ast
    tree = ast.parse(Path(p.__file__).read_text())
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ""
             for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not set(calls) & {"mj_forward", "mj_collision", "mj_step", "collision", "kinematics", "solve", "allclose", "isclose"}
