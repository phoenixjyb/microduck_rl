import hashlib

import mujoco
import numpy as np
import pytest

from mjlab_microduck import football_stance_probe as probe
from mjlab_microduck.football_contact_fixture import ROBOT_XML


@pytest.fixture(scope='module')
def adjusted():
    return probe.adjust_pose()


def test_bounded_pose_preserves_hinges_and_passes_geometry(adjusted):
    model, data, report = adjusted
    assert report['adjusted_geometry_consistent']
    assert report['solver_nfev'] <= 80
    assert report['actual_geometry_calls'] <= 80*6+1
    assert max(abs(np.array(report['residual_m']))) <= 1e-8
    assert np.all(abs(np.array(report['offset_xyz_m_roll_pitch_rad'])) <= probe.LIMITS)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            q = model.jnt_qposadr[j]
            assert data.qpos[q] == report['baseline_qpos'][q]
    assert data.time == 0 and np.all(data.qvel == 0) and np.all(data.ctrl == 0)
    assert not report['geometry']['forbidden_contact_candidates']


def test_xml_and_all_mesh_hashes_are_real(adjusted):
    hashes = adjusted[2]['asset_sha256']
    assert len(hashes) > 10 and 'robot_allcollisions.xml' in hashes
    for name, digest in hashes.items():
        assert hashlib.sha256((ROBOT_XML.parent/name).read_bytes()).hexdigest() == digest


def test_feasible_symmetric_contacts_and_unilateral_friction_bounds():
    points = [[0, -.03, .21], [0, .03, .21], [0, 0, 0]]
    frames = np.tile([[0, 0, 1], [1, 0, 0], [0, 1, 0]], (3, 1, 1))
    r = probe.constrained_forces(points, frames, np.full((3, 2), .6), [0, 0, .35], .7, [0, 0, .11], .43)
    assert r['constrained_equilibrium_feasible']
    assert min(r['ray_coefficients_n']) >= -1e-9
    assert r['forces_world_n'][2][2] == pytest.approx(1.13*9.81)
    assert all(e['pyramid_slack_n'] >= -1e-9 for e in r['friction_pyramid_exposure'])
    assert not r['motor_load_verified'] and not r['policy_acceptance']


def test_negative_normal_force_is_not_allowed():
    frames = np.tile([[0, 0, -1], [1, 0, 0], [0, -1, 0]], (3, 1, 1))
    r = probe.constrained_forces([[0, -.03, .21], [0, .03, .21], [0, 0, 0]],
        frames, np.full((3, 2), .6), [0, 0, .35], .7, [0, 0, .11], .43)
    assert not r['constrained_equilibrium_feasible'] and r['forces_world_n'] is None


def test_bad_frames_rejected():
    with pytest.raises(ValueError, match='orthonormal'):
        probe.constrained_forces(np.zeros((3, 3)), np.zeros((3, 3, 3)), np.ones((3, 2)),
                                 [0, 0, .35], .7, [0, 0, .11], .43)


def test_gravity_jacobian_matches_potential_energy_gradient(adjusted):
    model, original, _ = adjusted
    data = mujoco.MjData(model); data.qpos[:] = original.qpos
    mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
    gravity = probe.gravity_generalized(model, data)
    qpos = data.qpos.copy(); numerical = []
    for dof in range(model.nv):
        tangent = np.zeros(model.nv); tangent[dof] = 1
        potential = []
        for sign in (1, -1):
            data.qpos[:] = qpos
            mujoco.mj_integratePos(model, data.qpos, tangent, sign*1e-6)
            mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
            potential.append(-np.sum(model.body_mass[:, None]*data.xipos*model.opt.gravity))
        numerical.append(-(potential[0]-potential[1])/2e-6)
    np.testing.assert_allclose(gravity, numerical, atol=1e-8, rtol=1e-7)


def test_native_contact_forces_and_ideal_joint_loads_balance_free_roots(adjusted):
    model, data, report = adjusted
    points, frames, friction = probe.fixture_contacts(model, data)
    g = report['geometry']
    r = probe.constrained_forces(points, frames, friction, g['robot_com_world_m'],
        g['robot_mass_kg'], g['ball_center_world_m'], .43)
    assert r['constrained_equilibrium_feasible'] and r['scaled_residual_max_n'] <= 1e-6
    loads = probe.ideal_joint_loads(model, data, points, r['forces_world_n'])
    assert len(loads['ideal_hinge_torque_nm']) == 14
    assert loads['ideal_free_root_equilibrium_consistent']
    for value in loads['free_joint_residual_force_then_moment'].values():
        assert np.max(abs(np.asarray(value))) <= 1e-6
    assert not loads['bam_executed'] and not loads['motor_acceptance']
    assert data.time == 0 and np.all(data.ctrl == 0)


def test_probe_does_not_run_mujoco_dynamics(monkeypatch):
    def forbidden(*a, **k): pytest.fail('dynamics routine invoked')
    for name in ('mj_step', 'mj_forward', 'mj_inverse', 'mj_fwdConstraint', 'mj_fwdActuation'):
        monkeypatch.setattr(mujoco, name, forbidden)
    r = probe.run_probe()
    assert r['physics_steps'] == 0 and not r['mujoco_force_solver_executed']
    assert not r['learned_balance'] and not r['physical_motion_authorized']
