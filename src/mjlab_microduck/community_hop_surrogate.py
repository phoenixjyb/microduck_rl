"""Native CPU stock/backlash plant for the separately labelled C1-S probe.

No policy inference or physics stepping occurs during construction/preflight.
This is our declared surrogate, not the author's recovered training runtime.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

from . import community_hop_rehearsal as c1
from .community_policy_inspection import _read_payload, inspect_policy_payload


DECLARATION_PATH = "docs/experiments/2026-10-01-community-hop-c1s-predeclaration.json"
DECLARATION_SHA256 = "236d7ca25e1a0508e6a913944f822a44244fb032362bd4ee3a38e5315c9eb700"
BRANCH = "feat/athletics-obstacle-curriculum"
RUNNER_PATHS = (
    "src/mjlab_microduck/community_hop_surrogate.py",
    "src/mjlab_microduck/community_hop_diagnostic.py",
    "src/mjlab_microduck/community_hop_baseline.py",
    "src/mjlab_microduck/community_policy_inspection.py",
    "src/mjlab_microduck/community_hop_rehearsal.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_native() -> None:
    # Native imports/compilation live inside the owned, hard-bounded child.
    # Reading the declaration and constructing the watchdog require stdlib only.
    global mujoco, np
    import mujoco
    import numpy as np


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   text=True, timeout=10).strip()


def load_declaration(root: Path) -> tuple[dict, bytes]:
    try:
        payload = _read_payload(root / DECLARATION_PATH)
    except ValueError as exc:
        raise ValueError("predeclaration could not be read safely") from exc
    require(sha256(payload) == DECLARATION_SHA256, "immutable predeclaration SHA256 mismatch")
    return json.loads(payload), payload


def source_binding(root: Path, *, clean_required: bool) -> dict:
    require(git(root, "branch", "--show-current") == BRANCH, "exact feature branch required")
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    require(not clean_required or not status, "policy execution requires a clean repository")
    sources = {name: sha256(_read_payload(root / name)) for name in RUNNER_PATHS}
    return {"runner_source_commit": git(root, "rev-parse", "HEAD"),
            "runner_source_sha256": sources, "repository_clean": not bool(status)}


def verified_assets(root: Path, plan: dict) -> tuple[bytes, dict[str, bytes], dict]:
    """Bind the XML plus every tracked asset to immutable compiler buffers."""
    plant = plan["plant"]
    scene = _read_payload(root / plant["scene_path"])
    robot = _read_payload(root / plant["robot_path"])
    require(sha256(scene) == plant["scene_sha256"], "scene SHA256 mismatch")
    require(sha256(robot) == plant["robot_sha256"], "robot SHA256 mismatch")
    folder = str(Path(plant["scene_path"]).parent / "assets")
    tree = git(root, "rev-parse", "HEAD:" + folder)
    require(tree == plant["asset_tree_sha1"], "stock asset tree mismatch")
    # Compare actual bytes with Git blobs, not merely HEAD's directory identity.
    records = git(root, "ls-tree", "-r", "HEAD", "--", folder).splitlines()
    assets = {Path(plant["robot_path"]).name: robot}
    hashes = {}
    for row in records:
        mode_type_hash, name = row.split("\t", 1)
        mode, kind, blob = mode_type_hash.split()
        require(kind == "blob" and mode == "100644", "regular tracked assets required")
        payload = _read_payload(root / name)
        actual_blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        require(actual_blob == blob, "working asset bytes differ from declared tree: " + name)
        assets["assets/" + str(Path(name).relative_to(folder))] = payload
        hashes[name] = sha256(payload)
    require(bool(hashes), "empty asset closure")
    return scene, assets, {"asset_tree_sha1": tree, "asset_sha256": hashes}


def verified_policy(root: Path, record: dict, interface: dict) -> tuple[bytes, dict]:
    import onnx

    payload = _read_payload(root / record["local_path"])
    report = inspect_policy_payload(payload, record["sha256"])
    require(len(payload) == record["bytes"], "policy byte count mismatch")
    require(report["runtime_inputs"] == [interface["input"]]
            and report["outputs"] == [interface["output"]], "exact API1 signature required")
    model = onnx.load_model_from_string(payload)
    metadata = {}
    for pair in model.metadata_props:
        require(pair.key not in metadata, "duplicate ONNX metadata key")
        metadata[pair.key] = pair.value
    require(metadata.get("joint_names", "").split(",") == interface["joint_names"],
            "policy servo order mismatch")
    require(metadata.get("observation_names") ==
            "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
            "policy observation order mismatch")
    require(float(metadata.get("action_scale", "nan")) == interface["action_scale"],
            "policy action scale mismatch")
    # Published metadata is rounded to three decimals; the declaration's exact
    # HOME is used for runtime, never silently replaced by those rounded values.
    home = [float(value) for value in metadata.get("default_joint_pos", "").split(",")]
    require(len(home) == 14 and all(math.isfinite(x) and abs(x - h) <= 0.0005
                                  for x, h in zip(home, interface["home_rad"])),
            "policy rounded HOME metadata mismatch")
    return payload, {**report, "metadata": metadata,
                     "repo": record["repo"], "revision": record["revision"]}


def verified_bam(plan: dict):
    from bam.model import _resolve_json_path, load_model_from_dict

    payload = _read_payload(Path(_resolve_json_path(None, "xl330", "m6")))
    require(sha256(payload) == plan["plant"]["bam_parameters_sha256"], "BAM parameter SHA256 mismatch")
    parameters = json.loads(payload)
    model = load_model_from_dict(parameters)
    model.actuator.vin = plan["plant"]["vin_v"]
    model.actuator.kp = plan["plant"]["kp_fw"]
    model.actuator.max_current = plan["plant"]["max_current_a"]
    return model, parameters


def verified_versions(plan: dict) -> dict:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be explicitly empty")
    versions = {name: importlib.metadata.version(name) for name in plan["runtime"]["versions"]}
    require(versions == plan["runtime"]["versions"], "frozen runtime version mismatch")
    direct = json.loads(importlib.metadata.distribution("better-actuator-models").read_text("direct_url.json") or "{}")
    require(direct.get("vcs_info", {}).get("commit_id") == plan["plant"]["bam_git_revision"],
            "installed BAM Git revision mismatch")
    return {"versions": versions, "bam_git_revision": direct["vcs_info"]["commit_id"],
            "cuda_visible_devices": "", "provider_requested": "CPUExecutionProvider",
            "sessions_created": False, "policy_inferences": 0, "physics_steps": 0}


def friction_without_self(data, dofs: np.ndarray) -> np.ndarray:
    """Subtract each DOF's own dry-friction constraint from the previous load.

    MuJoCo 3.10 efc_id for mjCNSTR_FRICTION_DOF is a DOF index, not a joint
    index. These differ on a free-base/backlash robot. No installed BAM patch.
    """
    load_native()
    dry = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    own = np.array([np.sum(data.efc_force[dry & (data.efc_id == dof)]) for dof in dofs])
    return -data.qfrc_bias[dofs] + data.qfrc_constraint[dofs] - own


class OutputSideBamController:
    """BAM motor math with output-side firmware feedback; no torque hard clip.

    Current torque and rotor speed feed BAM's friction budget; the external load
    is the preceding solve's constraint load minus its own DOF dry friction.
    Only motor DOFs are edited. Passive hinge properties remain stock.
    """

    def __init__(self, model, data, bam, joint_names):
        load_native()
        self.model, self.data, self.bam = model, data, bam
        self.act = np.array([model.actuator(name).id for name in joint_names])
        joints = np.array([model.joint(name).id for name in joint_names])
        passive = np.array([model.joint("passive_" + name + "_backlash").id for name in joint_names])
        require(np.array_equal(model.actuator_trnid[self.act, 0], joints), "actuator-joint binding mismatch")
        self.q = model.jnt_qposadr[joints]
        self.dof = model.jnt_dofadr[joints]
        self.bq = model.jnt_qposadr[passive]
        self.bdof = model.jnt_dofadr[passive]
        self.target = np.zeros(14)
        self.voltage = self.torque = self.current = np.zeros(14)
        model.dof_armature[self.dof] = bam.actuator.get_extra_inertia()
        mujoco.mj_setConst(model, data)

    def update(self):
        rotor_q = self.data.qpos[self.q]
        encoder_q = rotor_q + self.data.qpos[self.bq]
        rotor_dq = self.data.qvel[self.dof]
        actuator = self.bam.actuator
        self.voltage = np.asarray(actuator.compute_control(self.target, encoder_q, rotor_dq,
                                                         c1.PHYSICS_DT_S))
        self.torque = np.asarray(actuator.compute_torque(self.voltage, True, rotor_q, rotor_dq))
        self.current = (self.voltage - self.bam.kt.value * rotor_dq) / self.bam.R.value
        friction, damping = self.bam.compute_frictions(self.torque,
                                                       friction_without_self(self.data, self.dof), rotor_dq)
        self.data.ctrl[self.act] = self.torque
        self.model.dof_frictionloss[self.dof] = friction
        self.model.dof_damping[self.dof] = damping


class NativePlant:
    def __init__(self, scene: bytes, assets: dict[str, bytes], plan: dict, bam):
        load_native()
        spec = mujoco.MjSpec.from_string(
            scene.decode(), include={name: payload for name, payload in assets.items() if name.endswith(".xml")},
            assets={name: payload for name, payload in assets.items() if not name.endswith(".xml")})
        # Source actuators are position servos. Strip all inherited actuator
        # clipping/bias, but do not alter contacts, equalities or passive DOFs.
        for actuator in spec.actuators:
            actuator.set_to_motor()
            actuator.gear = [1.0, 0, 0, 0, 0, 0]
            actuator.biasprm = [0.] * 10
            actuator.ctrllimited = False
            actuator.forcelimited = False
        spec.option.timestep = c1.PHYSICS_DT_S
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.plan = plan
        self.controller = OutputSideBamController(self.model, self.data, bam, plan["interface"]["joint_names"])
        m, d, ctrl = self.model, self.data, self.controller
        root_joint = m.joint("trunk_base_freejoint")
        self.root_q = int(root_joint.qposadr[0])
        self.root_dof = int(root_joint.dofadr[0])
        self.root_body = m.body("trunk_base").id
        self.floor = m.geom(plan["plant"]["floor_geom"]).id
        self.feet = [m.geom(name).id for name in plan["plant"]["foot_geoms"]]
        self.home = np.array(plan["interface"]["home_rad"])
        ranges = m.jnt_range[[m.joint(name).id for name in plan["interface"]["joint_names"]]]
        center = ranges.mean(axis=1)
        half = (ranges[:, 1] - ranges[:, 0]) * plan["plant"]["soft_joint_limit_factor"] / 2
        self.soft_range = np.column_stack((center - half, center + half))
        d.qpos[:] = 0
        d.qpos[self.root_q:self.root_q + 3] = plan["plant"]["initial_base_xyz_m"]
        d.qpos[self.root_q + 3:self.root_q + 7] = plan["plant"]["initial_base_quaternion_wxyz"]
        d.qpos[ctrl.q] = self.home
        d.qvel[:] = 0
        d.ctrl[:] = 0
        ctrl.target = self.home.copy()
        mujoco.mj_forward(m, d)
        require((m.nq, m.nv, m.nu) == (35, 34, 14), "unexpected compiled dimensions")
        require(np.all(m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
                and np.all(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(m.actuator_gainprm[:, 0] == 1)
                and np.all(m.actuator_biasprm == 0)
                and not np.any(m.actuator_ctrllimited) and not np.any(m.actuator_forcelimited),
                "unclipped unit-gain torque motors required")
        require(m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
                and m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
                and m.opt.iterations == 100 and m.opt.ls_iterations == 50
                and m.opt.tolerance == 1e-8
                and np.array_equal(m.opt.gravity, [0, 0, -9.81]), "source solver settings mismatch")
        self.ground_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] != 0 and
                            ((int(m.geom_contype[i]) & int(m.geom_conaffinity[self.floor])) or
                             (int(m.geom_conaffinity[i]) & int(m.geom_contype[self.floor])))]
        require(len(self.ground_geoms) == 10 and set(self.feet) <= set(self.ground_geoms),
                "stock floor-capable geom binding mismatch")
        require(m.geom_type[self.floor] == mujoco.mjtGeom.mjGEOM_PLANE
                and np.array_equal(d.geom_xpos[self.floor], [0, 0, 0])
                and np.allclose(d.geom_xmat[self.floor].reshape(3, 3), np.eye(3)),
                "horizontal zero-height floor required")
        require(all(m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in self.feet),
                "mesh soles required")
        self.sole_vertices = []
        for geom in self.feet:
            mesh = m.geom_dataid[geom]
            first, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.sole_vertices.append(m.mesh_vert[first:first + count].copy())

    def prepare(self):
        mujoco.mj_step1(self.model, self.data)

    def unsafe_for_inference(self) -> bool:
        values = self.kinematics()
        return (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or max(abs(x) for x in values["base_roll_pitch_yaw_rad"][:2]) >= math.radians(60)
                or self.contacts()["body_contact"])

    def observation_state(self) -> dict:
        m, d, ctrl = self.model, self.data, self.controller
        rot = d.xmat[self.root_body].reshape(3, 3)
        # Free-joint angular velocity is local; linear translation is world.
        return {"angular_velocity": d.qvel[self.root_dof + 3:self.root_dof + 6].tolist(),
                "projected_gravity": (rot.T @ np.array([0., 0., -1.])).tolist(),
                "servo_position": d.qpos[ctrl.q].tolist(),
                "backlash_position": d.qpos[ctrl.bq].tolist(),
                "servo_velocity": d.qvel[ctrl.dof].tolist(),
                "backlash_velocity": d.qvel[ctrl.bdof].tolist(),
                "home_position": self.home.tolist()}

    def kinematics(self) -> dict:
        m, d, ctrl = self.model, self.data, self.controller
        rot = d.xmat[self.root_body].reshape(3, 3)
        pitch = math.asin(float(np.clip(-rot[2, 0], -1, 1)))
        rpy = [math.atan2(rot[2, 1], rot[2, 2]), pitch, math.atan2(rot[1, 0], rot[0, 0])]
        clearance = []
        for geom, vertices in zip(self.feet, self.sole_vertices):
            # Compiled vertices already use the canonical mesh frame. The
            # compiler folded mesh_pos/quat into geom_xpos/xmat: do NOT apply
            # those offsets again. Convex-hull and full-mesh extrema coincide.
            geom_rot = d.geom_xmat[geom].reshape(3, 3)
            clearance.append(float(np.min(vertices @ geom_rot[2]) + d.geom_xpos[geom, 2]))
        encoder = d.qpos[ctrl.q] + d.qpos[ctrl.bq]
        return {"base_xy_m": d.qpos[self.root_q:self.root_q + 2].tolist(),
                "base_velocity_world_m_s": d.qvel[self.root_dof:self.root_dof + 3].tolist(),
                "base_roll_pitch_yaw_rad": rpy,
                "base_angular_velocity_rad_s": (rot @ d.qvel[self.root_dof + 3:self.root_dof + 6]).tolist(),
                "foot_clearance_m": clearance,
                "motor_velocity_rad_s": d.qvel[ctrl.dof].tolist(),
                "soft_limit_exposed": ((encoder < self.soft_range[:, 0]) |
                                       (encoder > self.soft_range[:, 1])).tolist()}

    def contacts(self) -> dict:
        touched, force, body = [False, False], [0., 0.], False
        for index, contact in enumerate(self.data.contact):
            pair = [int(contact.geom1), int(contact.geom2)]
            if self.floor not in pair or contact.efc_address < 0:
                continue
            other = pair[1] if pair[0] == self.floor else pair[0]
            if other in self.feet:
                foot = self.feet.index(other)
                touched[foot] = True
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                force[foot] += abs(float(wrench[0]))
            elif self.model.geom_bodyid[other] != 0:
                body = True
        return {"foot_contact": touched, "foot_normal_force_n": force, "body_contact": body}

    def advance_and_measure(self, applied: list[float], *, terminal: bool) -> dict:
        before = self.kinematics()
        self.controller.target = self.home + np.asarray(applied)
        self.controller.update()
        torque = {"motor_current_a": self.controller.current.tolist(),
                  "motor_torque_nm": self.controller.torque.tolist()}
        old_time = self.data.time
        if terminal:
            mujoco.mj_forward(self.model, self.data)
        else:
            # prepare() built constraints using the preceding friction budget.
            # Refresh position/velocity stages after BAM edits, before this
            # substep's solve; otherwise efc_frictionloss would lag by 5 ms.
            mujoco.mj_step1(self.model, self.data)
            # mj_step1 prepared CURRENT kinematics above. mj_step2 solves its
            # forces and then integrates. Contact forces still refer to that
            # pre-integration state, not to the newly advanced qpos/qvel.
            mujoco.mj_step2(self.model, self.data)
        reset = abs(self.data.time - old_time - (0 if terminal else c1.PHYSICS_DT_S)) > 1e-8
        return {**before, **torque, **self.contacts(), "reset_event": bool(reset)}

    def receipt(self) -> dict:
        m, d, ctrl = self.model, self.data, self.controller
        def geom_record(i):
            return {"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                    "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])),
                    "contype": int(m.geom_contype[i]), "conaffinity": int(m.geom_conaffinity[i]),
                    "condim": int(m.geom_condim[i]), "friction": m.geom_friction[i].tolist(),
                    "solref": m.geom_solref[i].tolist(), "solimp": m.geom_solimp[i].tolist()}
        return {"compiled_mjb_sha256": sha256(self.binary_model()),
                "dimensions": {"nq": m.nq, "nv": m.nv, "nu": m.nu},
                "initial_qpos": d.qpos.tolist(), "initial_qvel": d.qvel.tolist(),
                "motor_joint_order": self.plan["interface"]["joint_names"],
                "motor_qpos_indices": ctrl.q.tolist(), "motor_dof_indices": ctrl.dof.tolist(),
                "backlash_qpos_indices": ctrl.bq.tolist(), "backlash_dof_indices": ctrl.bdof.tolist(),
                "motor_armature": m.dof_armature[ctrl.dof].tolist(),
                "passive_armature": m.dof_armature[ctrl.bdof].tolist(),
                "passive_damping": m.dof_damping[ctrl.bdof].tolist(),
                "passive_frictionloss": m.dof_frictionloss[ctrl.bdof].tolist(),
                "actuator_gain": m.actuator_gainprm.tolist(), "actuator_bias": m.actuator_biasprm.tolist(),
                "actuator_gain_type": m.actuator_gaintype.tolist(), "actuator_bias_type": m.actuator_biastype.tolist(),
                "actuator_gear": m.actuator_gear.tolist(),
                "actuator_ctrl_limited": m.actuator_ctrllimited.tolist(),
                "actuator_force_limited": m.actuator_forcelimited.tolist(),
                "encoder_soft_ranges_rad": self.soft_range.tolist(),
                "ground_enabled_geoms": [geom_record(i) for i in self.ground_geoms],
                "floor": geom_record(self.floor), "physics_dt_s": m.opt.timestep,
                "solver": int(m.opt.solver), "integrator": int(m.opt.integrator),
                "iterations": m.opt.iterations, "ls_iterations": m.opt.ls_iterations,
                "tolerance": m.opt.tolerance, "gravity": m.opt.gravity.tolist(),
                "passive_joint_ranges_rad": m.jnt_range[[m.joint("passive_" + name + "_backlash").id
                                                         for name in self.plan["interface"]["joint_names"]]].tolist(),
                "passive_joint_limit_solref": m.jnt_solref[[m.joint("passive_" + name + "_backlash").id
                                                           for name in self.plan["interface"]["joint_names"]]].tolist(),
                "base_velocity_reference": "free-joint-origin-world-linear-world-angular",
                "clearance_method": "compiled-mesh-vertex-world-z-minimum",
                "friction_load_method": "current-motor-torque-previous-solve-external-minus-own-dof-friction"}

    def binary_model(self) -> bytes:
        binary = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, None, binary)
        return binary.tobytes()


def preflight(root: Path, *, clean_required: bool = False):
    plan, declaration_bytes = load_declaration(root)
    require(sha256(_read_payload(root / "uv.lock")) == plan["runtime"]["uv_lock_sha256"], "uv.lock SHA256 mismatch")
    require(sha256(_read_payload(root / plan["measurement"]["scorer_path"])) ==
            plan["measurement"]["scorer_sha256"], "frozen measurement scorer mismatch")
    sources = source_binding(root, clean_required=clean_required)
    runtime = verified_versions(plan)
    policies, reports = {}, {}
    for kind, record in plan["policies"].items():
        policies[kind], reports[kind] = verified_policy(root, record, plan["interface"])
    scene, assets, closure = verified_assets(root, plan)
    bam, parameters = verified_bam(plan)
    plant = NativePlant(scene, assets, plan, bam)
    receipt = {"experiment_id": plan["experiment_id"], "case_id": "preflight",
               "measurement_protocol": c1.PROTOCOL,
               "measurement_scorer_sha256": plan["measurement"]["scorer_sha256"],
               "predeclaration_sha256": sha256(declaration_bytes), **sources,
               "runtime": runtime, "policies": reports, "asset_closure": closure,
               "plant": plant.receipt(), "bam_parameters": parameters,
               "bam_settings": {name: plan["plant"][name] for name in
                                ("vin_v", "kp_fw", "max_current_a", "voltage_drop_gain",
                                 "firmware_position_view", "back_emf_and_friction_velocity_view")},
               "decision": "static-preflight-passed-no-policy-execution",
               **plan["not_claimed"]}
    return plan, declaration_bytes, policies, plant, receipt
