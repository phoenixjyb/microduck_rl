"""CPU robot-only reconstruction, explicitly not a captured training model."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import xml.etree.ElementTree as ET

import yaml

from mjlab_microduck.retained_skill_binding import SOURCE, declared_config, require, verified_f1y_inventory


PROTOCOL = "f1y-robot-only-cpu-reconstruction-v1"
BINDING_SHA = "62a3bf5a34abda909246205011a54c1c7261f571ba7ddc9e5afce2a8ea502bf2"
PACKAGE = "src/mjlab_microduck/"
XML = PACKAGE + "robot/microduck/robot_walk.xml"
ARRAYS = ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_type", "jnt_axis",
          "jnt_pos", "jnt_range", "qpos0", "key_qpos", "actuator_trnid", "actuator_gear",
          "actuator_ctrlrange", "actuator_forcerange", "dof_armature", "dof_damping",
          "dof_frictionloss", "geom_friction", "geom_contype", "geom_conaffinity", "geom_condim")


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify_blob(repo: Path, name: str, entry: tuple[str, str, str]) -> dict:
    """Working bytes must equal the immutable Git blob, not just HEAD's path."""
    path = PurePosixPath(name)
    require(bool(name) and name != "." and "\\" not in name and "\0" not in name
            and not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
            "safe source path")
    mode, kind, oid = entry
    require(mode in ("100644", "100755") and kind == "blob", "regular source blob")
    file = repo
    for part in path.parts:
        file = file / part
        require(not file.is_symlink(), "source symlink not allowed")
    data = file.read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    require(digest == oid, "working bytes differ from historical source: " + name)
    return dict(git_blob=oid, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def mesh_paths(xml: bytes) -> list[str]:
    """Strict resolver for this flat mesh-only model; never skip unknown files."""
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    require(root.tag == "mujoco" and compiler is not None
            and compiler.attrib.get("meshdir") == "assets", "known mesh directory")
    require(not list(root.iter("include")), "includes require a separate resolver")
    meshes = root.findall("./asset/mesh")
    require(bool(meshes), "mesh inventory required")
    require({id(e) for e in root.iter() if "file" in e.attrib} == {id(e) for e in meshes},
            "unhandled external model asset")
    names = []
    for mesh in meshes:
        name = mesh.attrib.get("file", "")
        path = PurePosixPath(name)
        require(bool(name) and path.name == name and name not in (".", "..")
                and "\\" not in name and "\0" not in name, "flat safe mesh name")
        names.append(PurePosixPath(XML).parent.joinpath("assets", name).as_posix())
    require(len(names) == len(set(names)), "unique mesh inventory")
    return sorted(names)


def source_inventory(repo: Path) -> dict:
    raw = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", SOURCE, "--", PACKAGE], cwd=repo)
    entries = {}
    for row in raw.split(b"\0"):
        if row:
            metadata, name = row.decode().split("\t", 1)
            entries[name] = tuple(metadata.split())
    require(XML in entries, "historical robot XML required")
    verified = {XML: verify_blob(repo, XML, entries[XML])}
    meshes = mesh_paths((repo / XML).read_bytes())
    names = sorted({k for k in entries if k.endswith(".py")} | set(meshes))
    require(len(meshes) == 38 and sum(k.endswith(".py") for k in names) == 97,
            "fixed source/mesh coverage")
    for name in names:
        require(name in entries, "asset missing from training Git tree: " + name)
        verified[name] = verify_blob(repo, name, entries[name])
    return dict(training_source=SOURCE, python_files=97, mesh_files=38, files=verified)


def reconstruct(repo: Path, evidence: Path, binding_path: Path) -> dict:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide CUDA for CPU reconstruction")
    repo, evidence = repo.resolve(), evidence.resolve()
    require(repo == Path(__file__).resolve().parents[2], "use the executing source worktree")
    require(identity(binding_path)["sha256"] == BINDING_SHA, "immutable partial binding")
    binding = json.loads(binding_path.read_text())
    sources = source_inventory(repo)  # Bind historical project bytes before config imports.
    verified_f1y_inventory(evidence)
    with contextlib.redirect_stdout(io.StringIO()):
        import torch
        import numpy as np
        import mjlab
        import bam
        from mjlab.entity import Entity
        from mjlab_microduck.foundation_yaw_experiment import prepare_config
        from mjlab_microduck.motor_measurement_audit import motor_layout

        require(not torch.cuda.is_initialized(), "CUDA must remain uninitialized")
        cfg, agent = prepare_config("pilot", arm="yaw")
        declared = declared_config(yaml.dump(asdict(cfg), sort_keys=False),
                                   yaml.dump(asdict(agent), sort_keys=False))
        require(declared == binding["declared_config"], "exact retained config declarations")
        robot = Entity(cfg.scene.entities["robot"])
        model = robot.spec.compile()
        cfg.sim.mujoco.apply(model)
        names, joint_ids = motor_layout(robot)
        require(list(names) == binding["recorded_evaluation"]["motor_columns"], "recorded motor order")
        require(model.nu == 14 and model.nq == 21 and model.nv == 20, "rigid robot-only topology")
        arrays = {}
        for name in ARRAYS:
            value = getattr(model, name)
            require(bool(np.isfinite(value).all()), "finite compiled field: " + name)
            arrays[name] = value.tolist()
        require(not torch.cuda.is_initialized(), "CPU-only compilation")
        bam_root, mjlab_root = Path(bam.__file__).resolve().parent, Path(mjlab.__file__).resolve().parent
        params = Path(cfg.scene.entities["robot"].articulation.actuators[0]._resolved_json_path).resolve()
        require(params == bam_root / "params/xl330/m6.json", "actual bundled BAM parameter source")
        dependencies = {f"{label}/{p.relative_to(root).as_posix()}": identity(p)
                        for label, root in (("bam", bam_root), ("mjlab", mjlab_root))
                        for p in sorted(root.rglob("*.py"))}
        dependencies["bam/params/xl330/m6.json"] = identity(params)
        snapshot = dict(
            nq=model.nq, nv=model.nv, nu=model.nu, bodies=model.nbody, geoms=model.ngeom,
            meshes=model.nmesh, physics_timestep_s=model.opt.timestep, decimation=cfg.decimation,
            motor_columns=list(names), local_joint_ids=list(joint_ids),
            actuator_names=list(robot.actuator_names), joint_names=list(robot.joint_names),
            arrays=arrays,
        )
    return dict(
        protocol=PROTOCOL, decision="reconstructed-robot-only-not-historical-equivalence",
        partial_binding_sha256=BINDING_SHA, source_inventory=sources,
        current_python_dependencies=dependencies,
        current_package_versions={k: importlib.metadata.version(k) for k in
            ("better-actuator-models", "mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")},
        audit_implementation_sha256=identity(Path(__file__))["sha256"],
        declared_config_matches_retained=True, reconstructed_robot=snapshot,
        matched_recorded_motor_columns=True, cuda_initialized=False,
        scope="current CPU robot-only compilation; no scene, randomization or runtime actuator state",
        historical_training_model_captured=False, historical_effective_model_equivalence=False,
        resolved_policy_action_pipeline_verified=False, behavioral_retention_verified=False,
        policy_acceptance=False, transition_authorized=False, physical_motion_authorized=False,
        simulation_steps_executed=0, optimizer_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("binding", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(args.evidence.resolve()),
            "new output outside closed evidence required")
    report = reconstruct(Path(__file__).resolve().parents[2], args.evidence, args.binding)
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(data).hexdigest(),
                         decision=report["decision"])))


if __name__ == "__main__":
    main()
