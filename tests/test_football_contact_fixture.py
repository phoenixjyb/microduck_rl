"""Real native CPU geometry only; not learned stance or coupled ball dynamics."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from mjlab_microduck import football_contact_fixture as fixture
from mjlab_microduck.football_balance_plan import FootballPlant


@pytest.mark.parametrize('support,free_count',[('fixed',1),('free',2)])
def test_explicit_fixture_topology_and_shell_inertia(support,free_count):
    model,data=fixture.build_fixture(support=support)
    assert sum(model.jnt_type==0)==free_count and model.neq==0 and model.nmocap==0
    assert model.nu==14 and data.time==0 and np.all(data.qvel==0)
    np.testing.assert_allclose(model.body('b0_football').inertia,[FootballPlant().inertia_kg_m2]*3)
    assert model.body('b0_football').mass[0]==pytest.approx(.43)


@pytest.mark.parametrize('support',['fixed','free'])
def test_vertical_geometry_candidate_has_two_feet_near_sphere_without_stance_claim(support):
    source=fixture.ROBOT_XML.read_bytes()
    model,data,r=fixture.vertical_placement_probe(support=support)
    assert r['both_feet_near_ball'] and r['geometry_within_joint_ranges']
    assert min(x['signed_distance_m'] for x in r['feet'].values())==pytest.approx(.0001,abs=1e-6)
    assert r['minimum_forbidden_external_clearance_m']>0
    assert r['robot_mass_kg']==pytest.approx(.73724318)
    assert r['robot_com_world_m'][2]>.22
    assert not r['forbidden_contact_candidates']
    assert r['ball_fixed_assistance']==(support=='fixed')
    assert r['physics_steps']==0 and data.time==0 and np.all(data.ctrl==0)
    assert not any(r[k] for k in ('stance_feasibility_verified','motor_load_verified','policy_acceptance','force_solver_executed'))
    assert fixture.ROBOT_XML.read_bytes()==source
    assert r['robot_xml_sha256']==hashlib.sha256(source).hexdigest()
    for foot in r['feet'].values():
        normal=np.array(foot['ball_outward_normal_world'])
        assert np.linalg.norm(normal)==pytest.approx(1) and normal[2]>0
        assert np.linalg.norm(np.array(foot['ball_point_world_m'])-r['ball_center_world_m'])==pytest.approx(.11,abs=1e-6)


def test_contact_inspection_does_not_integrate_or_change_borrowed_pose():
    model,data,_=fixture.vertical_placement_probe()
    qpos,qvel,ctrl=data.qpos.copy(),data.qvel.copy(),data.ctrl.copy()
    fixture.inspect_contacts(model,data)
    np.testing.assert_array_equal(data.qpos,qpos);np.testing.assert_array_equal(data.qvel,qvel)
    np.testing.assert_array_equal(data.ctrl,ctrl);assert data.time==0


def test_forbidden_robot_ball_contact_is_reported_not_hidden():
    model,data=fixture.build_fixture()
    # Move the ball into the nominal torso, an intentionally invalid reset.
    ball_q=int(model.joint('b0_ball_freejoint').qposadr[0]);data.qpos[ball_q+2]=.16
    report=fixture.inspect_contacts(model,data)
    assert report['minimum_forbidden_external_clearance_m']<0
    assert report['forbidden_contact_candidates']
    assert not report['stance_feasibility_verified']


def test_invalid_support_and_too_small_nominal_ball_do_not_fake_a_fit():
    with pytest.raises(ValueError):fixture.build_fixture(support='hidden-tether')
    with pytest.raises(ValueError,match='bracket'):
        fixture.vertical_placement_probe(replace(FootballPlant(),radius_m=.02))


def test_joint_limit_violation_is_reported_and_force_routines_are_never_called(monkeypatch):
    def forbidden(*a,**k):pytest.fail('geometry probe invoked integration or force evaluation')
    for name in ('mj_step','mj_forward','mj_fwdActuation','mj_fwdConstraint'):
        monkeypatch.setattr(fixture.mujoco,name,forbidden)
    model,data,_=fixture.vertical_placement_probe()
    data.qpos[model.joint('left_knee').qposadr[0]]=1.8
    r=fixture.inspect_contacts(model,data)
    assert not r['geometry_within_joint_ranges']
    assert not next(j for j in r['joint_ranges'] if j['joint']=='left_knee')['within_range']
    assert r['physics_steps']==0 and not r['motor_load_verified']
