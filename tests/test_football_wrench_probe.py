import numpy as np
import pytest

from mjlab_microduck import football_wrench_probe as probe


def test_wrench_moment_sign_and_translation_invariance():
    point = np.array([1., 2., 3.]); center = np.array([.2, .3, .4]); force = np.array([4., 5., 6.])
    np.testing.assert_allclose(probe.wrench_matrix(point, center) @ force,
                               np.r_[force, np.cross(point-center, force)])
    np.testing.assert_allclose(probe.wrench_matrix(point+7, center+7), probe.wrench_matrix(point, center))


@pytest.mark.parametrize('free', [False, True])
def test_symmetric_known_static_solution_without_stance_promotion(free):
    kwargs = dict(ball_center=[0, 0, .11], ball_mass=.43, floor_point=[0, 0, 0]) if free else {}
    r = probe.relaxed_equilibrium([[0, -.03, .21], [0, .03, .21]], [0, 0, .35], .7, **kwargs)
    assert r['relaxed_equilibrium_consistent']
    assert sum(f[2] for f in r['least_squares_foot_forces_world_n']) == pytest.approx(.7*9.81)
    if free: assert r['least_squares_floor_force_on_ball_world_n'][2] == pytest.approx(1.13*9.81)
    assert not r['stance_feasibility_verified'] and not r['motor_load_verified']


def test_com_offset_cannot_be_balanced_about_two_point_support_line():
    r = probe.relaxed_equilibrium([[0, -.03, .21], [0, .03, .21]], [.01, 0, .35], .7)
    assert not r['relaxed_equilibrium_consistent']
    assert r['scaled_residual_max_n'] > .01


def test_rank_deficient_coincident_contacts_remain_finite():
    r = probe.relaxed_equilibrium([[0, 0, 0], [0, 0, 0]], [0, 0, .1], .7)
    assert r['matrix_rank'] == 3 and r['relaxed_equilibrium_consistent']


@pytest.mark.parametrize('kwargs', [dict(ball_mass=.43), dict(ball_center=[0, 0, .11]),
                                  dict(ball_center=[0, 0, .11], ball_mass=-1, floor_point=[0, 0, 0])])
def test_partial_or_invalid_free_ball_fields_rejected(kwargs):
    with pytest.raises(ValueError): probe.relaxed_equilibrium([[0, 0, 0]], [0, 0, .1], .7, **kwargs)


@pytest.mark.parametrize('support', ['fixed', 'free'])
def test_native_fixture_probe_never_steps_or_claims_motor_acceptance(support, monkeypatch):
    import mujoco
    def forbidden(*args, **kwargs): pytest.fail('force solver or integration invoked')
    for name in ('mj_step', 'mj_forward', 'mj_fwdConstraint', 'mj_fwdActuation'):
        monkeypatch.setattr(mujoco, name, forbidden)
    r = probe.probe_fixture(support=support)
    assert not r['relaxed_equilibrium_consistent']
    assert r['physics_steps'] == 0 and not r['mujoco_force_solver_executed']
    assert not r['policy_acceptance'] and not r['friction_constraints_enforced']
    feet = [c for c in r['contact_candidates'] if 'b0_floor' not in c['names']]
    assert len(feet) == 2 and all(c['dim'] == 3 for c in feet)
    assert all(c['friction'][0] == pytest.approx(1.) for c in feet)
