"""Bounded B0-A pose search and conservative point-force static analysis."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.football_contact_fixture import FEET, ROBOT_XML, inspect_contacts, vertical_placement_probe
from mjlab_microduck.football_wrench_probe import wrench_matrix

PROTOCOL = 'football-b0a-pose-and-ideal-load-v1'
LIMITS = np.array([.005, .005, .005, .05, .05])
POSE_TOL_M = 1e-8


def asset_hashes():
    """Pin this fixture's XML and all referenced mesh files; reject new assets."""
    root = ET.parse(ROBOT_XML).getroot()
    require(not root.findall('.//include'), 'no unpinned includes')
    require(not root.findall('.//texture') and not root.findall('.//hfield'), 'no unpinned asset types')
    compiler = root.find('compiler')
    directory = ROBOT_XML.parent / compiler.get('meshdir', '')
    paths = [ROBOT_XML]
    for mesh in root.findall('.//asset/mesh'):
        require(mesh.get('file') is not None, 'file-backed mesh')
        paths.append(directory / mesh.get('file'))
    return {str(p.relative_to(ROBOT_XML.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def _geometry(model, data):
    mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_collision(model, data)
    ball = model.geom('b0_ball_surface').id
    bodies = [i for i in range(1, model.nbody) if i != model.geom_bodyid[ball]]
    mass = float(model.body_mass[bodies].sum())
    com = (data.xipos[bodies] * model.body_mass[bodies, None]).sum(0) / mass
    contacts = []
    for name in FEET:
        foot = model.geom(name).id
        matches = [c for c in data.contact if set(map(int, c.geom)) == {foot, ball}]
        require(len(matches) == 1, 'exactly one collision candidate per foot throughout search')
        contacts.append(matches[0])
    return com, contacts


def adjust_pose():
    model, data, baseline = vertical_placement_probe(support='free')
    root = int(model.joint('trunk_base_freejoint').qposadr[0])
    initial = data.qpos.copy(); calls = 0
    def objective(offset):
        nonlocal calls
        calls += 1
        data.qpos[:] = initial
        data.qpos[root:root+3] += offset[:3]
        xyzw = Rotation.from_euler('xy', offset[3:]).as_quat()
        data.qpos[root+3:root+7] = xyzw[[3, 0, 1, 2]]
        com, contacts = _geometry(model, data)
        center = data.geom_xpos[model.geom('b0_ball_surface').id]
        p, q = [c.pos - center for c in contacts]
        length = np.linalg.norm((q-p)[:2])
        require(length > .01, 'separated support contacts')
        line_distance = (p[0]*q[1] - p[1]*q[0]) / length
        return np.array([contacts[0].dist-.0001, contacts[1].dist-.0001,
                         com[0]-center[0], com[1]-center[1], line_distance])
    fit = least_squares(objective, np.zeros(5), bounds=(-LIMITS, LIMITS),
                        max_nfev=80, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    residual = objective(fit.x)
    geometry = inspect_contacts(model, data)
    good = (np.max(np.abs(residual)) <= POSE_TOL_M
            and geometry['geometry_within_joint_ranges']
            and not geometry['forbidden_contact_candidates']
            and geometry['minimum_forbidden_external_clearance_m'] >= 0)
    report = dict(protocol=PROTOCOL, search_status=int(fit.status), search_message=fit.message,
        solver_nfev=int(fit.nfev), actual_geometry_calls=calls, max_nfev=80,
        offset_xyz_m_roll_pitch_rad=fit.x.tolist(), absolute_offset_bounds=LIMITS.tolist(),
        residual_m=residual.tolist(), residual_tolerance_m=POSE_TOL_M,
        adjusted_geometry_consistent=bool(good), baseline_qpos=initial.tolist(), geometry=geometry,
        asset_sha256=asset_hashes(), policy_acceptance=False, physical_motion_authorized=False)
    canonical(report); return model, data, report


def constrained_forces(points, frames, friction, robot_com, robot_mass, ball_center, ball_mass):
    """Two forces on robot plus floor force on ball; four positive rays/contact."""
    points = np.asarray(points, dtype=float); frames = np.asarray(frames, dtype=float)
    friction = np.asarray(friction, dtype=float)
    require(points.shape == (3, 3) and frames.shape == (3, 3, 3)
            and friction.shape == (3, 2), 'three contact frames and friction pairs')
    require(np.isfinite(points).all() and np.isfinite(frames).all()
            and np.isfinite(friction).all() and (friction >= 0).all(), 'finite contacts')
    require(np.allclose(frames @ frames.transpose(0, 2, 1), np.eye(3), atol=1e-8), 'orthonormal contact frames')
    require(np.shape(robot_com) == np.shape(ball_center) == (3,)
            and np.isfinite(robot_com).all() and np.isfinite(ball_center).all()
            and np.isfinite(robot_mass) and robot_mass > 0
            and np.isfinite(ball_mass) and ball_mass > 0, 'finite positive plant')
    rays = []
    for frame, mu in zip(frames, friction):
        n, t1, t2 = frame
        rays.append(np.stack([n+mu[0]*t1, n-mu[0]*t1, n+mu[1]*t2, n-mu[1]*t2], axis=1))
    matrix = np.zeros((12, 12))
    for i in range(2):
        matrix[:6, 4*i:4*i+4] = wrench_matrix(points[i], robot_com) @ rays[i]
        matrix[6:, 4*i:4*i+4] = -wrench_matrix(points[i], ball_center) @ rays[i]
    matrix[6:, 8:] = wrench_matrix(points[2], ball_center) @ rays[2]
    target = np.zeros(12); target[2] = robot_mass*9.81; target[8] = ball_mass*9.81
    scale = np.tile([1, 1, 1, 1/.11, 1/.11, 1/.11], 2)
    fit = linprog(np.ones(12), A_eq=matrix*scale[:, None], b_eq=target*scale,
                  bounds=(0, None), method='highs', options={
                      'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    result = dict(lp_status=int(fit.status), lp_message=fit.message, constrained_equilibrium_feasible=False,
        points_world_m=points.tolist(), frames_world=frames.tolist(), sliding_friction=friction.tolist(),
        forces_world_n=None, ray_coefficients_n=None, scaled_residual_max_n=None,
        friction_pyramid_exposure=None,
        contact_model='four-ray unilateral Coulomb pyramid; no contact couples or compliance',
        motor_load_verified=False, stance_feasibility_verified=False, policy_acceptance=False)
    if fit.success:
        residual = (matrix @ fit.x - target)*scale
        good = np.max(np.abs(residual)) <= 1e-6 and min(fit.x) >= -1e-9
        forces = np.array([rays[i] @ fit.x[i*4:i*4+4] for i in range(3)])
        exposure = []
        for frame, mu, force in zip(frames, friction, forces):
            normal, t1, t2 = frame @ force
            terms = [abs(t)/u if u > 0 else (0. if abs(t) < 1e-9 else float('inf'))
                     for t, u in zip((t1, t2), mu)]
            require(np.isfinite(terms).all(), 'zero-friction direction must have no tangential load')
            used = sum(terms)
            exposure.append(dict(normal_force_n=float(normal),
                scaled_tangential_l1_n=float(used), pyramid_slack_n=float(normal-used),
                pyramid_utilization=float(used/normal) if normal > 1e-9 else None))
        result.update(constrained_equilibrium_feasible=bool(good),
            forces_world_n=forces.tolist(), friction_pyramid_exposure=exposure,
            ray_coefficients_n=fit.x.tolist(), scaled_residual_max_n=float(np.max(np.abs(residual))))
    canonical(result); return result


def fixture_contacts(model, data):
    ball = model.geom('b0_ball_surface').id
    ordered = []
    for name in (*FEET, 'b0_floor'):
        other = model.geom(name).id
        matches = [c for c in data.contact if set(map(int, c.geom)) == {ball, other}]
        require(len(matches) == 1, 'exact declared contact topology')
        c = matches[0]; require(c.dim == 3 and c.dist < c.includemargin, 'active-margin point contact')
        # First two forces act on foot; last acts on ball.
        receiver = ball if name == 'b0_floor' else other
        sign = 1 if int(c.geom[1]) == receiver else -1
        ordered.append((c.pos.copy(), sign*c.frame.reshape(3, 3).copy(), c.friction[:2].copy()))
    return tuple(np.array([row[i] for row in ordered]) for i in range(3))


def gravity_generalized(model, data):
    gravity = np.zeros(model.nv)
    for body in range(1, model.nbody):
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jac, None, body)
        gravity += jac.T @ (model.body_mass[body] * model.opt.gravity)
    return gravity


def ideal_joint_loads(model, data, points, forces):
    """Rigid ideal static demand only; deliberately excludes passive/BAM forces."""
    total = gravity_generalized(model, data)
    ball_body = model.body('b0_football').id
    for i, name in enumerate(FEET):
        foot_body = int(model.geom(name).bodyid[0])
        for body, force in ((foot_body, forces[i]), (ball_body, -np.asarray(forces[i]))):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, np.asarray(points[i]), body)
            total += jac.T @ force
    jac = np.zeros((3, model.nv)); mujoco.mj_jac(model, data, jac, None, np.asarray(points[2]), ball_body)
    total += jac.T @ forces[2]
    joints = {model.joint(j).name: float(-total[model.jnt_dofadr[j]]) for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    free = {model.joint(j).name: total[model.jnt_dofadr[j]:model.jnt_dofadr[j]+6].tolist()
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    maximum = max(float(np.max(np.abs(value))) for value in free.values())
    return dict(ideal_hinge_torque_nm=joints, free_joint_residual_force_then_moment=free,
                ideal_free_root_equilibrium_consistent=maximum <= 1e-6,
                free_root_residual_max_mixed_n_nm=maximum,
                free_joint_residual_frames='generalized coordinates: world translation, local-body rotation',
                passive_forces_included=False, bam_executed=False, motor_acceptance=False)


def run_probe():
    model, data, result = adjust_pose()
    result.update(constrained_loads=None, ideal_joint_loads=None, physics_steps=0,
                  mujoco_force_solver_executed=False, learned_balance=False)
    if result['adjusted_geometry_consistent']:
        points, frames, friction = fixture_contacts(model, data)
        g = result['geometry']
        loads = constrained_forces(points, frames, friction, g['robot_com_world_m'], g['robot_mass_kg'],
                                   g['ball_center_world_m'], float(model.body('b0_football').mass[0]))
        result['constrained_loads'] = loads
        if loads['constrained_equilibrium_feasible']:
            result['ideal_joint_loads'] = ideal_joint_loads(model, data, points, loads['forces_world_n'])
    canonical(result); return result
