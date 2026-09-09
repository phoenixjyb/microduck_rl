"""CPU compiled-plant and recorded-state consistency, not live provenance.

Does not integrate physics, attest a solver run or authorize training. Selected
native model fields are compared explicitly; this is not a full MJB fingerprint.
"""
from hashlib import sha256

import mujoco
import torch
import mjlab  # Initialize task discovery before flat_hold imports bam.mjlab.

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.football_contact_fixture import FEET
from mjlab_microduck.football_flat_hold import place_on_floor
from mjlab_microduck.football_stance_probe import asset_hashes
from mjlab_microduck.stance_lesson_contract import JOINTS
from mjlab_microduck.stance_warp_runtime import build_entity
from mjlab_microduck.stance_contact_evidence import contact_summary

PROTOCOL = 'football-b1n-compiled-plant-v1'
ARRAYS = ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat',
    'jnt_type', 'jnt_bodyid', 'jnt_qposadr', 'jnt_dofadr', 'jnt_range', 'jnt_axis', 'jnt_pos',
    'dof_armature', 'dof_frictionloss', 'dof_damping', 'geom_type', 'geom_bodyid',
    'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_condim', 'geom_contype',
    'geom_conaffinity', 'geom_solref', 'geom_solimp', 'actuator_trnid', 'actuator_trntype',
    'actuator_gear', 'actuator_forcerange', 'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm')
ATOL, RTOL = 1e-6, 1e-5


def describe(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('init_state').id)
    data.qpos[:7] = [0, 0, .2, 1, 0, 0, 0]
    data.qvel[:] = 0; data.ctrl[:] = 0
    place_on_floor(model, data)
    require(not data.warning.number.any(), 'compiled reset warning')
    qids = [int(model.joint(n).qposadr[0]) for n in JOINTS]
    dofs = [int(model.joint(n).dofadr[0]) for n in JOINTS]
    values = {name: getattr(model, name).tolist() for name in ARRAYS}
    return dict(protocol=PROTOCOL, assets=asset_hashes(),
        topology=[model.nq, model.nv, model.nu, model.ngeom, model.neq, model.nmocap],
        joints=list(JOINTS), qids=qids, dofs=dofs,
        floor=int(model.geom('hold_floor').id), feet=[int(model.geom(n).id) for n in FEET],
        ranges=model.jnt_range[[model.joint(n).id for n in JOINTS]].tolist(),
        initial_qpos=torch.tensor(data.qpos.copy(), dtype=torch.float32).tolist(),
        selected_fields=list(ARRAYS), selected_fields_sha256=sha256(canonical(values).encode()).hexdigest(),
        options=dict(timestep=model.opt.timestep, gravity=model.opt.gravity.tolist(),
            integrator=int(model.opt.integrator), solver=int(model.opt.solver),
            iterations=model.opt.iterations, tolerance=model.opt.tolerance))


def reference():
    """Fresh CPU compile on every verification; never trust the supplied mapping."""
    return describe(build_entity().compile())


def runtime_bytes(source, model):
    return (canonical(dict(source=source, plant=describe(model)))+'\n').encode()


def checked_runtime(runtime, source):
    require(type(runtime) is dict and set(runtime) == {'source', 'plant'}
            and runtime['source'] == source, 'exact runtime plant envelope')
    plant = reference()
    require(canonical(runtime['plant']) == canonical(plant), 'compiled plant mismatch')
    return plant


def close(actual, expected, label):
    require(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), 'plant state mismatch: '+label)


def check_frame(frame, plant):
    qpos, vel, state = frame['qpos'], frame['qvel'], frame['state']
    q = qpos[:, plant['qids']]
    ranges = torch.tensor(plant['ranges'], dtype=torch.float32)
    nominal = torch.tensor(plant['initial_qpos'], dtype=torch.float32)[plant['qids']]
    hard = ((q < ranges[:, 0]) | (q > ranges[:, 1])).any(1)
    soft = (q-ranges.mean(1)).abs() > .45*(ranges[:, 1]-ranges[:, 0])
    require(torch.equal(hard, state['hard_limit']) and torch.equal(soft, frame['soft_limit_mask']),
            'compiled joint limit masks')
    require(torch.equal(vel[:, plant['dofs']], state['joint_velocity']), 'compiled joint velocities')
    quat = qpos[:, 3:7]
    close(quat.square().sum(1), torch.ones(len(qpos)), 'unit root quaternion')
    w, x, y, z = quat.unbind(1)
    gravity = torch.stack((2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1), 1)
    actor = frame['observation']['actor']; critic = frame['observation']['critic']
    close(actor[:, :3], gravity, 'projected gravity')
    close(actor[:, 3:6], vel[:, 3:6]/5, 'body angular velocity')
    close(actor[:, 6:20], (q-nominal)/.5, 'joint offsets')
    close(actor[:, 20:34], vel[:, plant['dofs']]/10, 'joint speed observation')
    # Compare cos(tilt) to avoid acos amplification close to upright. The raw
    # recorded tilt still faces the original, unchanged numerical gates.
    close(state['tilt'].cos(), -gravity[:, 2], 'tilt')
    require(((state['tilt'] >= 0) & (state['tilt'] <= torch.pi)).all(), 'bounded tilt')
    require((actor[:, 34:].abs() <= 1).all(), 'bounded correction observation')
    close(critic[:, 44:47], vel[:, :3], 'critic root velocity')
    close(critic[:, 47], (qpos[:, 2]-.12)/.1, 'critic height')
    close(critic[:, 48:], state['support']/10, 'critic support')


def check_contacts(record, plant, worlds):
    raw = record['contacts']; count = len(raw['worldid'])
    shapes = dict(worldid=(count,), geom=(count, 2), dim=(count,), dist=(count,),
        pos=(count, 3), frame=(count, 3, 3), friction=(count, 5), force=(count, 6))
    require(set(raw) == set(shapes) | {'efc_address'}, 'exact terminal contact fields')
    table = {}
    for name, shape in shapes.items():
        integer = name in ('worldid', 'geom', 'dim')
        # Do not truncate malformed floating-point geometry/world IDs.
        value = torch.tensor(raw[name])
        require(not integer or count == 0 or value.dtype == torch.int64, 'integer terminal contact IDs')
        require((count == 0 and value.numel() == 0) or tuple(value.shape) == shape, 'terminal contact shape')
        table[name] = value.to(torch.int64 if integer else torch.float32).reshape(shape)
    address = torch.tensor(raw['efc_address'])
    require(count == 0 or address.dtype == torch.int64, 'integer contact addresses')
    require((count == 0 and address.numel() == 0) or
            (address.ndim == 2 and address.shape[0] == count and address.shape[1] >= 1),
            'terminal address shape')
    table['efc_address'] = address.to(torch.int64).reshape(count, -1) if count else torch.empty((0, 1), dtype=torch.int64)
    require((table['worldid'] == record['world_id']).all(), 'terminal contact world binding')
    require((table['geom'] < plant['topology'][3]).all(), 'compiled contact geometry range')
    require(torch.isin(table['dim'], torch.tensor([1, 3, 4, 6])).all(), 'contact dimensionality')
    support, forbidden = contact_summary(table, worlds, floor_id=plant['floor'], foot_ids=plant['feet'])
    row = record['world_id']
    close(support[row], torch.tensor(record['state']['support'], dtype=torch.float32), 'terminal contact support')
    require(bool(forbidden[row]) == record['state']['forbidden_contact'], 'terminal forbidden contact')


def check_trace(payload, plant):
    """Call after structural/continuity replay, before retaining the bundle."""
    initial = payload['initial']; n = payload['binding']['worlds']
    nominal = torch.tensor(plant['initial_qpos'], dtype=torch.float32).expand(n, -1)
    require(torch.equal(initial['qpos'], nominal) and not initial['qvel'].any()
            and not initial['physics_steps'].any() and not initial['observation']['actor'][:, 34:].any(),
            'exact nominal first reset')
    check_frame(initial, plant)
    for tick in payload['ticks']:
        for frame in tick['boundaries']: check_frame(frame, plant)
    terminals = payload['ticks'][-1]['terminal_records'] if payload['ticks'] else []
    for record in terminals:
        if record is not None: check_contacts(record, plant, n)
    return dict(compiled_plant_checked=True, nominal_reset_checked=True,
        kinematic_observations_checked=True, terminal_contact_summary_checked=True,
        terminal_contact_records_checked=sum(r is not None for r in terminals),
        plant_comparison_atol=ATOL, plant_comparison_rtol=RTOL)
