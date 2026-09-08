"""B0 rigid full-collision geometry fixture, not a stance or dynamics controller."""

import hashlib
from pathlib import Path
import re

import mujoco
import numpy as np

from mjlab_microduck.football_balance_plan import FootballPlant
from mjlab_microduck.first_attempt_smoke import canonical,require

VARIANT = 'football-b0-rigid-full-collision-geometry-v1'
ROBOT_XML = Path(__file__).parent/'robot/microduck/robot_allcollisions.xml'
FEET = ('left_foot_collision','right_foot_collision')


def build_fixture(plant=FootballPlant(),*,support='free'):
    """Fresh model/data; preserve the repository robot and historical tasks."""
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    require(type(plant) is FootballPlant,'typed ball plant');plant.__post_init__()
    require(support in ('fixed','free'),'explicit ball support')
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.modelname = VARIANT+'-'+support
    spec.option.timestep = plant.step_s
    spec.option.gravity = [0,0,-9.81]
    spec.worldbody.add_geom(name='b0_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2,2,.1],friction=[plant.sliding_friction,0,0],condim=3)
    ball = spec.worldbody.add_body(name='b0_football',pos=[0,0,plant.radius_m],
        mass=plant.mass_kg,inertia=[plant.inertia_kg_m2]*3,explicitinertial=True)
    if support=='free':ball.add_freejoint(name='b0_ball_freejoint')
    ball.add_geom(name='b0_ball_surface',type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[plant.radius_m,0,0],friction=[plant.sliding_friction,0,0],condim=3,margin=.001)
    model=spec.compile();data=mujoco.MjData(model)
    root=model.joint('trunk_base_freejoint').qposadr[0]
    data.qpos[root:root+7]=[0,0,.12,1,0,0,0]
    for i in range(model.njnt):
        if model.jnt_type[i]!=mujoco.mjtJoint.mjJNT_HINGE:continue
        name=model.joint(i).name
        matches=[v for expr,v in HOME_FRAME.joint_pos.items() if re.fullmatch(expr,name)]
        require(len(matches)==1,'exact nominal hinge pose')
        data.qpos[model.jnt_qposadr[i]]=matches[0]
    mujoco.mj_kinematics(model,data)
    return model,data


def _distance(model,data,a,b):
    segment=np.zeros(6)
    distance=float(mujoco.mj_geomDistance(model,data,a,b,2.,segment))
    require(np.isfinite(segment).all() and np.isfinite(distance) and distance<2.,'resolved bounded distance')
    return distance,segment


def inspect_contacts(model,data):
    """Refresh derived kinematics/contact buffers only; no integration/forces."""
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(),'finite fixture state')
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_collision(model,data)
    ball=model.geom('b0_ball_surface').id;floor=model.geom('b0_floor').id
    feet=[model.geom(name).id for name in FEET]
    ball_center=data.geom_xpos[ball]
    robot=[g for g in range(model.ngeom) if model.geom_bodyid[g] not in (0,model.geom_bodyid[ball])
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    robot_bodies=[b for b in range(1,model.nbody) if b!=model.geom_bodyid[ball]]
    robot_mass=float(model.body_mass[robot_bodies].sum())
    robot_com=(data.xipos[robot_bodies]*model.body_mass[robot_bodies,None]).sum(0)/robot_mass
    foot_reports={}
    for name,g in zip(FEET,feet):
        distance,segment=_distance(model,data,g,ball)
        normal=segment[3:]-ball_center
        require(np.linalg.norm(normal)>1e-8,'resolved sphere surface normal')
        normal/=np.linalg.norm(normal)
        foot_reports[name]=dict(signed_distance_m=distance,foot_point_world_m=segment[:3].tolist(),
            ball_point_world_m=segment[3:].tolist(),ball_outward_normal_world=normal.tolist(),
            normal_tilt_rad=float(np.arccos(np.clip(normal[2],-1,1))))
    forbidden_distances=[]
    for g in robot:
        for target in (floor,ball):
            if g in feet and target==ball:continue
            distance,_=_distance(model,data,g,target)
            forbidden_distances.append(dict(geometry=model.geom(g).name or f'geom-{g}',
                target=model.geom(target).name,signed_distance_m=distance))
    allowed={frozenset((ball,floor)),*(frozenset((ball,g)) for g in feet)}
    contacts=[]
    for contact in data.contact:
        a,b=map(int,contact.geom)
        contacts.append(dict(geometry_ids=[a,b],names=[model.geom(g).name or f'geom-{g}' for g in (a,b)],
            signed_distance_m=float(contact.dist),position_world_m=contact.pos.tolist(),
            normal_from_first_to_second=contact.frame[:3].tolist(),allowed_pair=frozenset((a,b)) in allowed))
    joint_ranges=[]
    for j in range(model.njnt):
        if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:continue
        q=float(data.qpos[model.jnt_qposadr[j]]);lo,hi=map(float,model.jnt_range[j])
        joint_ranges.append(dict(joint=model.joint(j).name,qpos_rad=q,range_rad=[lo,hi],
            within_range=bool(model.jnt_limited[j] and lo<=q<=hi)))
    result=dict(variant=VARIANT,feet=foot_reports,forbidden_distances=forbidden_distances,
        contact_candidates=contacts,joint_ranges=joint_ranges,
        qpos=data.qpos.tolist(),ball_center_world_m=ball_center.tolist(),
        robot_mass_kg=robot_mass,robot_com_world_m=robot_com.tolist(),
        geometry_within_joint_ranges=all(j['within_range'] for j in joint_ranges),
        minimum_forbidden_external_clearance_m=min(x['signed_distance_m'] for x in forbidden_distances),
        forbidden_contact_candidates=[c for c in contacts if not c['allowed_pair']],
        force_solver_executed=False,physics_steps=0,simulation_time_s=float(data.time),
        stance_feasibility_verified=False,motor_load_verified=False,policy_acceptance=False,
        physical_motion_authorized=False)
    canonical(result);return result


def vertical_placement_probe(plant=FootballPlant(),*,support='free'):
    """Only translate the nominal rigid robot vertically; no IK or stability solve."""
    model,data=build_fixture(plant,support=support)
    root=int(model.joint('trunk_base_freejoint').qposadr[0])
    ball=model.geom('b0_ball_surface').id;feet=[model.geom(n).id for n in FEET]
    def clearance(z):
        data.qpos[root+2]=z;mujoco.mj_kinematics(model,data)
        return min(_distance(model,data,g,ball)[0] for g in feet)
    target=.0001  # Near contact with a small nonpenetrating gap, not support force.
    lo,hi=.12+plant.radius_m,.12+2*plant.radius_m+.15
    require(clearance(lo)<target<clearance(hi),'upper-hemisphere nominal vertical placement bracket')
    for _ in range(40):
        mid=(lo+hi)/2
        if clearance(mid)<target:lo=mid
        else:hi=mid
    clearance(hi)
    result=inspect_contacts(model,data)
    result.update(support=support,ball_fixed_assistance=support=='fixed',
        ball_radius_m=plant.radius_m,ball_mass_kg=plant.mass_kg,
        ball_inertia_kg_m2=model.body('b0_football').inertia.tolist(),
        contact_margin_m=.001,target_nearest_foot_gap_m=target,
        method='40-step vertical bisection at existing HOME_FRAME; no joint optimization',
        robot_xml_sha256=hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
        all_asset_bytes_pinned=False,actuator_model='unstepped XML actuators, not validated BAM dynamics',
        both_feet_near_ball=all(0<=f['signed_distance_m']<=.0003 for f in result['feet'].values()))
    canonical(result);return model,data,result
