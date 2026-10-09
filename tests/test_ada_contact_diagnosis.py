"""Portable synthetic arithmetic and authentication checks, not CUDA proof."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mjlab_microduck import ada_contact_diagnosis as p


def contacts(count=1):
    values = dict(worldid=np.ones(count, dtype=np.int32), slot=np.arange(count, dtype=np.int32),
                  geom=np.tile([0, 29], (count, 1)).astype(np.int32), dim=np.full(count, 3, dtype=np.int32),
                  dist=np.full(count, -.0001, dtype=np.float32), pos=np.tile([1., 2., 3.], (count, 1)).astype(np.float32),
                  frame=np.tile(np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32), (count, 1, 1)),
                  friction=np.zeros((count, 5), dtype=np.float32),
                  force=np.tile([2., 3., 4., 5., 6., 7.], (count, 1)).astype(np.float32),
                  efc_address=np.arange(count, dtype=np.int32) * 4)
    return {"contacts/" + k: v for k, v in values.items()}


def test_import_is_inert():
    code = "import sys; import mjlab_microduck.ada_contact_diagnosis; assert not {'numpy','torch','warp','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_non_symmetric_frame_and_origin_moment():
    row = p.resultants(contacts())[0]
    assert row["world_force_N"] == [3., 4., 2.]
    assert row["world_torque_about_origin_Nm"] == [-2., 14., 3.]
    assert row["frame_orthonormality_max_abs"] == 0.
    assert row["minimum_pairwise_position_distance_m"] is None


def test_multiplicity_is_summed_not_averaged_or_normalized():
    bank = contacts(2)
    row = p.resultants(bank)[0]
    assert row["count"] == 2 and row["exact_distinct_positions"] == 1
    assert row["minimum_pairwise_position_distance_m"] == 0.
    assert row["world_force_N"] == [6., 8., 4.]
    assert row["world_torque_about_origin_Nm"] == [-4., 28., 6.]
    bank["contacts/pos"][1, 0] += 1.
    row = p.resultants(bank)[0]
    assert row["exact_distinct_positions"] == 2 and row["minimum_pairwise_position_distance_m"] == 1.


def test_ordered_pair_reversal_is_not_silently_merged():
    bank = contacts(2)
    bank["contacts/geom"][1] = [29, 0]
    rows = p.resultants(bank)
    assert [r["key"] for r in rows] == [[1, 0, 29, 3, True], [1, 29, 0, 3, True]]
    assert all(r["count"] == 1 and r["world_force_N"] == [3., 4., 2.] for r in rows)


def test_all_rows_reconstruct_contact_and_friction_without_input_mutation():
    active, fields = {}, {"qfrc_constraint": np.zeros((2, 20), dtype=np.float32)}
    for w in range(2):
        J = np.zeros((3, 20), dtype=np.float32); J[0, 6] = 1.; J[1, 2] = 2.; J[2, 2] = -1.
        active.update({f"rows/{w}/J": J, f"rows/{w}/force": np.array([3., 4., 5.], dtype=np.float32),
                       f"rows/{w}/type": np.array([1, 6, 6], dtype=np.int32)})
        fields["qfrc_constraint"][w, 2] = 3.; fields["qfrc_constraint"][w, 6] = 3.
    before = {k: v.tobytes() for k, v in active.items()}
    result = p.reconstruction(active, fields)
    assert all(r["max_abs"] == 0. for r in result)
    assert all(r["friction_generalized_force"][6] == r["contact_generalized_force"][2] == 3. for r in result)
    assert before == {k: v.tobytes() for k, v in active.items()}


def test_nonfinite_generalized_arithmetic_refuses():
    active = {f"rows/{w}/{k}": v.copy() for w in range(2) for k, v in
              dict(J=np.zeros((1, 20), dtype=np.float32), force=np.array([np.nan], dtype=np.float32),
                   type=np.array([1], dtype=np.int32)).items()}
    with pytest.raises(ValueError): p.reconstruction(active, {"qfrc_constraint": np.zeros((2, 20), dtype=np.float32)})


@pytest.mark.parametrize("damage", ["unbound", "symlink", "missing", "extra"])
def test_frozen_file_authentication_refuses_before_decoding(tmp_path, damage):
    for name in set(p.p.FILES) | {"report.json"}: (tmp_path / name).write_bytes(b"{}")
    if damage == "symlink":
        (tmp_path / "cpu-fields.npz").unlink()
        (tmp_path / "cpu-fields.npz").symlink_to(tmp_path / "gpu-fields.npz")
    if damage == "missing": (tmp_path / "motor.npz").unlink()
    if damage == "extra": (tmp_path / "unregistered.bin").write_bytes(b"x")
    with pytest.raises(ValueError): p.authenticated_banks(tmp_path)


def test_empty_contact_resultants():
    assert p.resultants(contacts(0)) == []


@pytest.mark.parametrize("name", p.p.FILES)
def test_manifest_byte_corruption_refuses_before_semantic_decoder(tmp_path, monkeypatch, name):
    child = {"synthetic": True}
    data = {n: b"opaque" for n in p.p.FILES}
    data["child.json"] = json.dumps(child).encode()
    report = dict(source={"commit": p.SOURCE}, decision=p.p.COLLECTION_DECISION,
                  specification=p.p.specification(), bounds=p.p.BOUNDS, child_exit=0,
                  child=child, files={n: sha256(raw).hexdigest() for n, raw in data.items()})
    raw = json.dumps(report).encode()
    monkeypatch.setattr(p, "REPORT_SHA256", sha256(raw).hexdigest())
    for n, value in data.items(): (tmp_path / n).write_bytes(value + (b"tampered" if n == name else b""))
    (tmp_path / "report.json").write_bytes(raw)
    monkeypatch.setattr(p.p, "receive_payloads", lambda *args: pytest.fail("must authenticate before decoding"))
    with pytest.raises(ValueError, match="every manifest file authenticated"): p.authenticated_banks(tmp_path)


def test_no_new_physics_or_tolerance_source():
    source = Path(p.__file__).read_text()
    for forbidden in ("mj_collision(", "mj_forward(", "mj_step(", ".integrate(", "allclose(", "cuda("):
        assert forbidden not in source


def test_cli_refuses_without_hidden_cuda(tmp_path):
    code = [sys.executable, "-m", p.__name__, "--input", str(tmp_path), "--output", str(tmp_path / "result.json"), "--source", "a" * 40]
    result = subprocess.run(code, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "explicit CPU-hidden diagnosis" in result.stderr
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("source", ["not-a-sha", "a" * 40])
def test_cli_refuses_wrong_source_before_output(tmp_path, source):
    code = [sys.executable, "-m", p.__name__, "--input", str(tmp_path), "--output", str(tmp_path / "result.json"), "--source", source]
    result = subprocess.run(code, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "clean exact diagnosis source" in result.stderr
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("damage", ["none", "branch", "dirty", "module"])
def test_main_source_closure_and_exclusive_output(monkeypatch, tmp_path, damage):
    source, output = "a" * 40, tmp_path / "result.json"
    monkeypatch.chdir(Path(p.__file__).resolve().parents[2])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(sys, "argv", ["probe", "--source", source, "--input", str(tmp_path), "--output", str(output)])
    def read(*command, binary=False):
        if command[1:3] == ("rev-parse", "HEAD"): return source
        if command[1:3] == ("branch", "--show-current"): return "foreign" if damage == "branch" else p.p.base.host.BRANCH
        if command[1:3] == ("status", "--porcelain"): return " M foreign.txt" if damage == "dirty" else ""
        if command[1] == "show":
            assert binary and command[2] == source + ":src/mjlab_microduck/ada_contact_diagnosis.py"
            return b"changed" if damage == "module" else Path(p.__file__).read_bytes()
        raise AssertionError(command)
    monkeypatch.setattr(p.p.base.host, "read", read)
    monkeypatch.setattr(p, "diagnose", lambda root: {"decision": "synthetic-only", "aggregate_differences": []})
    if damage == "none":
        p.main()
        result = json.loads(output.read_bytes())
        assert result["diagnosis_source"] == source
        assert result["diagnostic_module_sha256"] == sha256(Path(p.__file__).read_bytes()).hexdigest()
        before = output.read_bytes()
        with pytest.raises(FileExistsError): p.main()
        assert output.read_bytes() == before
    else:
        with pytest.raises(ValueError): p.main()
        assert not output.exists()
