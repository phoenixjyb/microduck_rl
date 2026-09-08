"""Single predeclared interior allocation on an immutable retained B0-A pose."""

import hashlib
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck import football_stance_probe as stance
from mjlab_microduck.football_contact_fixture import build_fixture, inspect_contacts

PROTOCOL = 'football-b0m-fixed-half-utilization-v1'
INPUT_SHA256 = 'd15226a8eacb69a347f4ece725528ecb102e9fde05e69e9035cb861c6e5bebf6'
UTILIZATION_CAP = .5


def read_pose(path):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INPUT_SHA256, 'exact immutable B0-A input hash')
    envelope = json.loads(raw)
    require(envelope['protocol'] == stance.PROTOCOL and envelope['policy_acceptance'] is False,
            'diagnostic-only B0-A envelope')
    require(hashlib.sha256(Path(stance.__file__).read_bytes()).hexdigest()
            == envelope['probe_source_sha256'], 'unchanged B0-A implementation')
    saved = envelope['result']
    require(saved['adjusted_geometry_consistent'] and saved['constrained_loads']['constrained_equilibrium_feasible'],
            'retained feasible static candidate')
    require(saved['asset_sha256'] == stance.asset_hashes(), 'unchanged complete fixture asset hashes')
    model, data = build_fixture(support='free')
    qpos = np.asarray(saved['geometry']['qpos'], dtype=float)
    require(qpos.shape == data.qpos.shape and np.isfinite(qpos).all(), 'finite compatible qpos')
    data.qpos[:] = qpos
    geometry = inspect_contacts(model, data)
    require(geometry['geometry_within_joint_ranges'] and not geometry['forbidden_contact_candidates']
            and geometry['minimum_forbidden_external_clearance_m'] >= 0, 'nonpenetrating declared geometry')
    points, frames, friction = stance.fixture_contacts(model, data)
    loads = saved['constrained_loads']
    for current, previous in ((points, loads['points_world_m']), (frames, loads['frames_world']),
            (friction, loads['sliding_friction']),
            (geometry['robot_com_world_m'], saved['geometry']['robot_com_world_m']),
            (geometry['robot_mass_kg'], saved['geometry']['robot_mass_kg']),
            (geometry['ball_center_world_m'], saved['geometry']['ball_center_world_m'])):
        require(np.allclose(current, previous, atol=1e-8, rtol=0), 'reconciled fixture geometry and contacts')
    # Fixed ball plant from B0-A: geometry report excludes mass, but native value
    # must match the declared .43 kg and the thin-shell inertia in build_fixture.
    require(float(model.body('b0_football').mass[0]) == .43, 'declared ball mass')
    require(np.array_equal(model.opt.gravity, [0., 0., -9.81])
            and np.allclose(model.body('b0_football').inertia, [.43*.11**2*2/3]*3,
                            atol=1e-12, rtol=0), 'declared gravity and shell inertia')
    return model, data, saved, geometry, points, frames, friction


def full_friction_exposure(frames, friction, forces):
    frames = np.asarray(frames, dtype=float); friction = np.asarray(friction, dtype=float)
    forces = np.asarray(forces, dtype=float)
    require(frames.shape == (3, 3, 3) and friction.shape == (3, 2) and forces.shape == (3, 3),
            'exact three-contact shape')
    require(np.isfinite(forces).all() and np.isfinite(frames).all()
            and np.isfinite(friction).all() and (friction > 0).all(), 'finite positive friction')
    out = []
    for frame, mu, force in zip(frames, friction, forces):
        normal, t1, t2 = frame @ force
        scaled = abs(t1)/mu[0] + abs(t2)/mu[1]
        out.append(dict(normal_force_n=float(normal), scaled_tangential_l1_n=float(scaled),
            full_pyramid_utilization=float(scaled/normal) if normal > 1e-9 else None,
            full_pyramid_slack_n=float(normal-scaled)))
    return out


def run_probe(path):
    model, data, saved, geometry, points, frames, friction = read_pose(path)
    loads = stance.constrained_forces(points, frames, UTILIZATION_CAP*friction,
        geometry['robot_com_world_m'], geometry['robot_mass_kg'], geometry['ball_center_world_m'], .43)
    report = dict(protocol=PROTOCOL, input_sha256=INPUT_SHA256, utilization_cap=UTILIZATION_CAP,
        qpos=data.qpos.tolist(), asset_sha256=saved['asset_sha256'],
        actual_contact_sliding_friction=friction.tolist(), allocation=loads,
        full_friction_exposure=None, interior_allocation_consistent=False,
        original_allocation=saved['constrained_loads'], original_ideal_joint_loads=saved['ideal_joint_loads'],
        ideal_joint_loads=None, named_ideal_torque_delta_nm=None,
        pose_search_repeated=False, contact_friction_modified=False, physics_steps=0,
        mujoco_force_solver_executed=False, motor_acceptance=False, learned_balance=False,
        policy_acceptance=False, physical_motion_authorized=False)
    if loads['constrained_equilibrium_feasible']:
        exposure = full_friction_exposure(frames, friction, loads['forces_world_n'])
        consistent = all(e['normal_force_n'] > 1e-9 and e['full_pyramid_utilization'] is not None
                         and e['full_pyramid_utilization'] <= UTILIZATION_CAP+1e-8 for e in exposure)
        report.update(full_friction_exposure=exposure, interior_allocation_consistent=consistent)
        if consistent:
            ideal = stance.ideal_joint_loads(model, data, points, loads['forces_world_n'])
            require(ideal['ideal_free_root_equilibrium_consistent'], 'balanced ideal root loads')
            old = saved['ideal_joint_loads']['ideal_hinge_torque_nm']
            report.update(ideal_joint_loads=ideal, named_ideal_torque_delta_nm={
                name: value-old[name] for name, value in ideal['ideal_hinge_torque_nm'].items()})
    require(np.array_equal(data.qpos, saved['geometry']['qpos']) and not np.any(data.qvel)
            and not np.any(data.ctrl) and data.time == 0, 'pose, zero motion and control preserved')
    require(np.array_equal(stance.fixture_contacts(model, data)[2], friction), 'contact friction unchanged')
    canonical(report); return report
