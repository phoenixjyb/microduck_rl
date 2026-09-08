"""One bounded native BAM floor hold; not a learned policy or GPU parity test."""

from collections import deque
import inspect
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck import football_bam_probe as bam
from mjlab_microduck.first_attempt_smoke import canonical, require, sha256
from mjlab_microduck.football_contact_fixture import FEET, _distance

PROTOCOL = 'football-flat-hold-v1'
STEPS = 500
DT = .002


class FixedDelay:
    def __init__(self, initial):
        self.initial = np.asarray(initial, dtype=float).copy()
        require(self.initial.shape == (14,) and np.isfinite(self.initial).all(), 'delay target')
        self.reset()

    def reset(self):
        self.queue = deque(self.initial.copy() for _ in range(3))

    def push(self, target):
        target = np.asarray(target, dtype=float)
        require(target.shape == (14,) and np.isfinite(target).all(), 'delay target')
        result = self.queue.popleft()
        self.queue.append(target.copy())
        return result


def place_on_floor(model, data):
    root = int(model.joint('trunk_base_freejoint').qposadr[0])
    floor = model.geom('hold_floor').id
    feet = [model.geom(n).id for n in FEET]
    def gaps(z):
        data.qpos[root+2] = z
        mujoco.mj_kinematics(model, data)
        return [_distance(model, data, f, floor)[0] for f in feet]
    lo, hi = .04, .30
    require(min(gaps(lo)) < .0001 < min(gaps(hi)), 'floor placement bracket')
    for _ in range(40):
        mid = (lo+hi)/2
        if min(gaps(mid)) < .0001: lo = mid
        else: hi = mid
    result = gaps(hi)
    require(all(0 <= g <= .0003 for g in result), 'both feet near floor')
    mujoco.mj_forward(model, data)
    return result


def observe(model, data, actuator):
    """Read a boundary after mj_forward; force and geometry share that state."""
    require(all(np.isfinite(v).all() for v in
        (data.qpos, data.qvel, data.qacc, data.qfrc_actuator)), 'nonfinite boundary')
    floor = model.geom('hold_floor').id
    feet = [model.geom(n).id for n in FEET]
    allowed = {frozenset((floor, f)) for f in feet}
    support = {n: 0. for n in FEET}; contacts = []; forbidden = []
    for i, c in enumerate(data.contact):
        pair = frozenset(map(int, c.geom))
        force = np.zeros(6); mujoco.mj_contactForce(model, data, i, force)
        require(np.isfinite(force).all(), 'nonfinite contact force')
        record = dict(geoms=[model.geom(int(g)).name for g in c.geom],
            distance_m=float(c.dist), force_contact_frame=force.tolist(),
            position_world_m=c.pos.tolist(), frame=c.frame.tolist(),
            friction=c.friction.tolist(), dim=int(c.dim))
        contacts.append(record)
        if pair in allowed:
            foot = next(f for f in feet if f in pair)
            support[model.geom(foot).name] += max(0., float(force[0]))
        elif c.dist <= 0: forbidden.append(record['geoms'])
    root_joint = model.joint('trunk_base_freejoint')
    root = int(root_joint.qposadr[0]); root_dof = int(root_joint.dofadr[0])
    quat = data.qpos[root+3:root+7]
    mat = np.zeros(9); mujoco.mju_quat2Mat(mat, quat)
    dofs = actuator._dof_ids.numpy()
    ids = [model.joint(n).id for n in actuator.target_names]
    q = data.qpos[model.jnt_qposadr[ids]]
    ranges = model.jnt_range[ids]; center = ranges.mean(1); half = (ranges[:,1]-ranges[:,0])/2
    torque = data.qfrc_actuator[dofs]; speed = data.qvel[dofs]
    out = dict(time_s=float(data.time), qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
        root_height_m=float(data.qpos[root+2]), tilt_rad=float(np.arccos(np.clip(mat[8], -1, 1))),
        root_speed_mps=float(np.linalg.norm(data.qvel[root_dof:root_dof+3])),
        hinge_speed_rad_s=speed.tolist(), applied_torque_nm=torque.tolist(),
        absolute_mechanical_power_w=float(np.abs(torque*speed).sum()),
        torque_squared_sum_nm2=float((torque**2).sum()),
        soft_limit_exposure_fraction=float((np.abs(q-center) > .9*half).mean()),
        outside_hard_limit=bool(((q < ranges[:,0]) | (q > ranges[:,1])).any()),
        warning_count=int(data.warning.number.sum()), support_normal_n=support,
        contacts=contacts, forbidden_contacts=forbidden)
    canonical(out); return out


def stop_reasons(frame, proposed=None):
    canonical(frame)
    reasons = []
    for condition, label in (
        (frame['warning_count'] > 0, 'mujoco-warning'),
        (bool(frame['forbidden_contacts']), 'forbidden-contact'),
        (frame['root_height_m'] < .08, 'root-height'),
        (frame['tilt_rad'] > .35, 'tilt'),
        (frame['root_speed_mps'] > 1., 'root-speed'),
        (max(map(abs, frame['hinge_speed_rad_s'])) > 10., 'hinge-speed'),
        (max(map(abs, frame['applied_torque_nm'])) > .36, 'applied-torque'),
        (frame['outside_hard_limit'], 'hard-joint-limit'),
        (frame['time_s'] >= .1 and min(frame['support_normal_n'].values()) <= .01, 'lost-foot-support'),
    ):
        if condition: reasons.append(label)
    if proposed is not None:
        require(np.isfinite(proposed).all(), 'nonfinite proposed torque')
        if max(map(abs, proposed)) > .36: reasons.append('proposed-torque')
    return reasons


def run_hold():
    model, data, actuator = bam.build_component_fixture(flat_floor=True)
    gaps = place_on_floor(model, data)
    require(model.opt.timestep == DT and model.neq == 0 and model.nmocap == 0,
            'unassisted declared fixture')
    names = actuator.target_names
    target = np.array([data.qpos[model.joint(n).qposadr[0]] for n in names])
    control_ids = []
    for name in names:
        ids = np.flatnonzero((model.actuator_trnid[:,0] == model.joint(name).id)
                            & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
        require(len(ids) == 1, 'unique joint motor transmission')
        control_ids.append(int(ids[0]))
    delay = FixedDelay(target); actuator.reset()
    frames = []; commands = []; steps = 0; reasons = []; error = None
    try:
        for boundary in range(STEPS+1):
            # Refresh post-integration state before associating contacts/loads.
            mujoco.mj_forward(model, data)
            frame = observe(model, data, actuator); frames.append(frame)
            reasons = stop_reasons(frame)
            if reasons or boundary == STEPS: break
            command = bam.compute_snapshot(model, data, actuator, delay.push(target))
            commands.append(dict(boundary=boundary, applied=False, **command))
            reasons = stop_reasons(frame, command['torque_nm'])
            if reasons: break
            data.ctrl[control_ids] = command['torque_nm']
            dofs = actuator._dof_ids.numpy()
            model.dof_frictionloss[dofs] = command['frictionloss_nm']
            model.dof_damping[dofs] = command['damping_nm_s_per_rad']
            mujoco.mj_step(model, data)
            steps += 1; commands[-1]['applied'] = True
    except ValueError as exc:
        reasons = ['invalid-numerical-evidence']; error = str(exc)
    report = dict(protocol=PROTOCOL, physics_steps=steps, duration_s=float(data.time),
        decision='bounded-fixture-complete' if steps == STEPS and not reasons else 'diagnostic-abort',
        stop_reasons=reasons, error=error, initial_foot_gap_m=gaps,
        initial_target_rad=target.tolist(), joints=names, fixed_delay_steps=3,
        frames=frames, commands=commands, final_boundary_observed=bool(frames and frames[-1]['time_s'] == data.time),
        torque_squared_integral_nm2_s=sum(f['torque_squared_sum_nm2']*DT for f in frames[:steps]),
        torque_integral_method='left-rectangle refreshed-boundary load proxy, not continuous torque',
        thermal_calibration=False, learned_stance=False, ball_balance=False,
        gpu_parity=False, physical_motion_authorized=False,
        asset_sha256=bam.asset_hashes(), source_sha256=sha256(Path(__file__)),
        bam_bridge_sha256=sha256(Path(bam.__file__)),
        runtime_versions={p: bam.version(p) for p in ('mujoco', 'torch', 'better-actuator-models', 'mjlab')},
        motor_parameters_sha256=sha256(Path(actuator.cfg._resolved_json_path)),
        training_adapter_sha256=sha256(Path(inspect.getfile(bam.BamActuator))),
        compiled_solver=dict(integrator=int(model.opt.integrator), solver=int(model.opt.solver),
            iterations=int(model.opt.iterations), tolerance=float(model.opt.tolerance),
            timestep_s=float(model.opt.timestep), gravity=model.opt.gravity.tolist()))
    canonical(report); return report
