"""Historical file identity and CPU reconstruction are separate from admission."""

import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck import robot_source_reconstruction as audit


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def test_working_blob_identity_is_not_just_filename(tmp_path):
    data = b"historical model bytes"
    p = tmp_path / "robot.xml"
    p.write_bytes(data)
    entry = ("100644", "blob", blob(data))
    result = audit.verify_blob(tmp_path, p.name, entry)
    assert result == dict(git_blob=blob(data), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    p.write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="historical"):
        audit.verify_blob(tmp_path, p.name, entry)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "", ".", "a/../b", "a\\b", "a//b"])
def test_unsafe_source_path_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="safe source"):
        audit.verify_blob(tmp_path, name, ("100644", "blob", "a" * 40))


@pytest.mark.parametrize("entry", [("120000", "blob", "a" * 40), ("160000", "commit", "a" * 40)])
def test_nonregular_git_entry_rejected(tmp_path, entry):
    with pytest.raises(ValueError, match="regular"):
        audit.verify_blob(tmp_path, "model", entry)


def test_source_symlink_is_not_a_provenance_shortcut(tmp_path):
    (tmp_path / "real").write_bytes(b"same")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValueError, match="symlink"):
        audit.verify_blob(tmp_path, "link", ("100644", "blob", blob(b"same")))


def model_xml(meshes='<mesh file="z.stl"/><mesh file="a.stl"/>', extra="", meshdir="assets"):
    return f'<mujoco><compiler meshdir="{meshdir}"/><asset>{meshes}</asset>{extra}</mujoco>'.encode()


def test_all_meshes_bound_in_stable_order():
    assert audit.mesh_paths(model_xml()) == [
        "src/mjlab_microduck/robot/microduck/assets/a.stl",
        "src/mjlab_microduck/robot/microduck/assets/z.stl",
    ]


@pytest.mark.parametrize("xml", [
    model_xml(extra='<include file="other.xml"/>'),
    model_xml(extra='<texture file="texture.png"/>'),
    model_xml(meshdir="../outside"),
    model_xml(meshes=""),
    model_xml(meshes='<mesh file="a.stl"/><mesh file="a.stl"/>'),
    model_xml(meshes='<mesh file="../a.stl"/>'),
    model_xml(meshes='<mesh file="/a.stl"/>'),
    model_xml(meshes='<mesh file="sub/a.stl"/>'),
    model_xml(meshes='<mesh/>'),
])
def test_unknown_or_unsafe_model_asset_cannot_be_silently_skipped(xml):
    with pytest.raises(ValueError):
        audit.mesh_paths(xml)


def test_cuda_guard_precedes_all_artifact_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="hide CUDA"):
        audit.reconstruct(tmp_path, tmp_path, tmp_path / "missing")


def test_cli_cannot_overwrite_closed_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["audit", str(tmp_path), str(tmp_path / "binding.json"),
                                    "--output", str(tmp_path / "new.json")])
    monkeypatch.setattr(audit, "reconstruct", lambda *_: pytest.fail("must fail before reconstruction"))
    with pytest.raises(ValueError, match="outside closed"):
        audit.main()


def test_real_cpu_reconstruction_has_explicit_scope_when_artifacts_available(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    evidence = next((repo / "artifacts" / p / "f1y-yaw-support-paired-s499-v1"
                     for p in ("diagnostics", "experiments")
                     if (repo / "artifacts" / p / "f1y-yaw-support-paired-s499-v1/manifest.json").is_file()), None)
    binding = repo / "artifacts/audits/f1y-retained-skill-binding-v1.json"
    if evidence is None or not binding.exists():
        pytest.skip("optional retained evidence unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = audit.reconstruct(repo, evidence, binding)
    json.dumps(result, allow_nan=False)
    assert len(result["source_inventory"]["files"]) == 136
    assert result["source_inventory"]["mesh_files"] == 38
    assert result["reconstructed_robot"]["motor_columns"] == result["reconstructed_robot"]["joint_names"]
    for flag in ("historical_training_model_captured", "historical_effective_model_equivalence",
                 "resolved_policy_action_pipeline_verified", "behavioral_retention_verified",
                 "policy_acceptance", "transition_authorized", "physical_motion_authorized", "cuda_initialized"):
        assert result[flag] is False
    assert result["simulation_steps_executed"] == result["optimizer_updates"] == 0
