"""Native CPU ball-only physics checks, not Duck balancing or trained behavior."""

from dataclasses import replace
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from mjlab_microduck.football_balance_plan import FootballPlant,STAGES,ball_world_xml,upper_surface


def test_separate_progression_and_no_capability_claim():
    assert [x[0] for x in STAGES] == ['B0','B1','B2','B3','B4','B5','B6']
    p = FootballPlant(); evidence = p.provenance()
    assert .68 <= 2*math.pi*p.radius_m <= .70 and .410 <= p.mass_kg <= .450
    assert not any(evidence[k] for k in ('pressure_or_deformation_calibrated','robot_included',
                                       'policy_acceptance','physical_motion_authorized'))
    historical = Path(__file__).resolve().parents[1]/'src/mjlab_microduck/robot/microduck/ball.xml'
    old = ET.parse(historical).getroot()
    assert old.find('.//geom').get('size') == '0.035'
    assert old.find('.//inertial').get('mass') == '0.015'


@pytest.mark.parametrize('field,value',[('radius_m',0),('radius_m',float('nan')),('mass_kg',-.1),
    ('mass_kg',True),('sliding_friction',float('inf')),('step_s',.1)])
def test_invalid_plant_refused(field,value):
    with pytest.raises(ValueError): replace(FootballPlant(),**{field:value})


@pytest.mark.parametrize('support,nq',[('fixed',0),('free',7)])
def test_native_compilation_has_explicit_shell_inertia_and_no_actuator(support,nq):
    p = FootballPlant(); model = mujoco.MjModel.from_xml_string(ball_world_xml(p,support=support))
    assert model.nq == nq and model.nu == 0
    assert model.body('football').mass[0] == pytest.approx(p.mass_kg)
    np.testing.assert_allclose(model.body('football').inertia,[p.inertia_kg_m2]*3)


def test_native_free_ball_passive_roll_and_rest_remain_finite():
    p = FootballPlant(); model = mujoco.MjModel.from_xml_string(ball_world_xml(p))
    rest = mujoco.MjData(model)
    for _ in range(500): mujoco.mj_step(model,rest)
    assert np.isfinite(rest.qpos).all() and np.isfinite(rest.qvel).all()
    assert rest.qpos[2] == pytest.approx(p.radius_m,abs=.001)
    assert np.linalg.norm(rest.qvel) < .001
    rolling = mujoco.MjData(model)
    # Initial pure rolling: v_x=R*omega_y. No robot, actuators or external force.
    rolling.qvel[0] = .05; rolling.qvel[4] = .05/p.radius_m
    for _ in range(500): mujoco.mj_step(model,rolling)
    assert np.isfinite(rolling.qpos).all() and np.isfinite(rolling.qvel).all()
    assert .03 < rolling.qpos[0] < .07 and abs(rolling.qpos[1]) < .001
    assert rolling.qpos[2] == pytest.approx(p.radius_m,abs=.001)


def test_symmetric_surface_estimate_is_not_a_stance_solution():
    left,right = upper_surface(.11,0,.042),upper_surface(.11,0,-.042)
    assert left == right and .21 < left['height_above_floor_m'] < .22
    assert math.degrees(left['normal_tilt_rad']) == pytest.approx(22.446,abs=.01)
    assert left['stance_feasibility_verified'] is False
    with pytest.raises(ValueError): upper_surface(.035,0,.042)
    with pytest.raises(ValueError): ball_world_xml(support='hidden-assistance')
