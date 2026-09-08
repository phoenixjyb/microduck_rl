"""Owned CPU snapshots for BAM's training compute path; never steps a robot."""

from copy import deepcopy
from importlib.metadata import version
import inspect
from pathlib import Path
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from bam.actuator import TorchBackend
from bam.mjlab import BamActuator
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microduck.first_attempt_smoke import canonical, require, sha256
from mjlab_microduck.football_contact_fixture import ROBOT_XML
from mjlab_microduck.football_stance_probe import asset_hashes

PROTOCOL = 'football-bam-cpu-component-v1'


def native_snapshot(model, data):
    """Copy native fields, including actual active constraint IDs, to CPU only."""
    def tensor(value, dtype=torch.float32):
        a = np.asarray(value)
        require(np.isfinite(a).all(), 'finite native snapshot')
        return torch.tensor(a.copy(), dtype=dtype, device='cpu').unsqueeze(0)

    n = data.nefc
    row_type = np.asarray(data.efc_type[:n])
    row_id = np.asarray(data.efc_id[:n])
    friction = row_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    require(((row_id[friction] >= 0) & (row_id[friction] < model.nv)).all(),
            'friction DOF IDs in range')
    return SimpleNamespace(
        qfrc_bias=tensor(data.qfrc_bias),
        qfrc_constraint=tensor(data.qfrc_constraint),
        qfrc_actuator=tensor(data.qfrc_actuator),
        nefc=torch.tensor([n], dtype=torch.long, device='cpu'),
        efc=SimpleNamespace(type=tensor(row_type, torch.long),
                            id=tensor(row_id, torch.long),
                            force=tensor(data.efc_force[:n])),
    )


def build_component_fixture():
    """Fresh suspended full robot, motor spec conversion, explicit CPU binding.

    Private BAM fields are deliberately localized here. This is not a substitute
    for Entity/Simulation.initialize, delayed get_command, or GPU validation.
    """
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME, actuators

    cfg = deepcopy(actuators)
    require(cfg.motor_name == 'xl330' and cfg.model == 'm6' and cfg.kp_fw == 200.,
            'declared XL330 m6 firmware')
    require(cfg.vin_range == (6.5, 8.2) and cfg.vin_drop_gain_range == (0., .2)
            and cfg.vin_min == 6., 'declared startup ranges')
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    names = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    require(len(names) == 14 and len(set(names)) == 14, 'exact rigid 14-hinge robot')
    actuator = cfg.build(None, list(range(14)), names)
    actuator.edit_spec(spec, names)
    spec.option.timestep = .002
    spec.option.gravity = [0, 0, -9.81]
    model = spec.compile()
    data = mujoco.MjData(model)
    root = int(model.joint('trunk_base_freejoint').qposadr[0])
    data.qpos[root:root+7] = [0, 0, .5, 1, 0, 0, 0]
    for name in names:
        matches = [q for expr, q in HOME_FRAME.joint_pos.items() if re.fullmatch(expr, name)]
        require(len(matches) == 1, 'unique nominal hinge target')
        data.qpos[model.joint(name).qposadr[0]] = matches[0]
    mujoco.mj_forward(model, data)  # Derived forces only; no time integration.

    # One explicitly selected startup instance, not sampled domain randomization.
    actuator._device = 'cpu'
    actuator._num_envs = 1
    actuator._dof_ids = torch.tensor(
        [int(model.joint(n).dofadr[0]) for n in names], dtype=torch.long, device='cpu')
    actuator._dt = float(model.opt.timestep)
    actuator._base_kp = float(actuator._bam_model.actuator.kp)
    actuator._bam_model.actuator.backend = TorchBackend()
    for key in ('kp_scale', 'kd_scale', 'friction_scale'):
        setattr(actuator, key, torch.ones(1, 1, device='cpu'))
        setattr(actuator, 'default_'+key, torch.ones(1, 1, device='cpu'))
    actuator.vin_tensor = torch.tensor([[7.5]], device='cpu')
    actuator.vin_drop_gain = torch.tensor([[.1]], device='cpu')
    actuator._prev_motor_torque = torch.zeros(1, 14, device='cpu')
    actuator._data = native_snapshot(model, data)
    actuator._mjwarp_model = SimpleNamespace(
        dof_frictionloss=torch.zeros(1, model.nv, device='cpu'),
        dof_damping=torch.zeros(1, model.nv, device='cpu'))
    return model, data, actuator


def compute_snapshot(model, data, actuator, target):
    """Evaluate one component call; writes only owned adapter tensors, not MuJoCo."""
    target = np.asarray(target, dtype=float)
    require(target.shape == (14,) and np.isfinite(target).all(), 'finite 14-target shape')
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), 'finite joint state')
    names = actuator.target_names
    pos = torch.tensor([[data.qpos[model.joint(n).qposadr[0]] for n in names]],
                       dtype=torch.float32, device='cpu')
    vel = torch.tensor([[data.qvel[model.joint(n).dofadr[0]] for n in names]],
                       dtype=torch.float32, device='cpu')
    actuator._data = native_snapshot(model, data)
    cmd = ActuatorCmd(
        position_target=torch.tensor(target[None], dtype=torch.float32, device='cpu'),
        velocity_target=torch.zeros_like(pos), effort_target=torch.zeros_like(pos),
        pos=pos, vel=vel)
    with torch.no_grad():
        torque = actuator.compute(cmd)
    fields = actuator._mjwarp_model
    report = dict(joints=names, torque_nm=torque[0].tolist(),
        frictionloss_nm=fields.dof_frictionloss[0, actuator._dof_ids].tolist(),
        damping_nm_s_per_rad=fields.dof_damping[0, actuator._dof_ids].tolist(),
        effective_voltage_v=float(actuator._bam_model.actuator.vin.item()))
    canonical(report)
    require(torque.shape == (1, 14) and min(report['frictionloss_nm']) >= 0,
            'finite shaped motor output and nonnegative friction')
    return report


def run_probe():
    model, data, actuator = build_component_fixture()
    names = actuator.target_names
    nominal = np.array([data.qpos[model.joint(n).qposadr[0]] for n in names])
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    samples = []
    for offset in (0., .02, -.02):
        actuator.reset()
        sample = compute_snapshot(model, data, actuator, nominal+offset)
        samples.append(dict(target_offset_rad=offset, **sample))
    require(data.time == 0 and np.array_equal(qpos, data.qpos)
            and np.array_equal(qvel, data.qvel) and np.array_equal(ctrl, data.ctrl),
            'native state and controls unchanged')
    source_paths = {
        'probe': Path(__file__), 'bam_training_adapter': Path(inspect.getfile(BamActuator)),
        'bam_voltage_actuator': Path(inspect.getfile(TorchBackend)),
        'repository_actuator': Path(inspect.getfile(type(actuator))),
        'bam_motor_class': Path(inspect.getfile(type(actuator._bam_model.actuator))),
        'bam_model': Path(inspect.getfile(type(actuator._bam_model))),
        'motor_parameters': Path(actuator.cfg._resolved_json_path),
    }
    return dict(protocol=PROTOCOL, samples=samples, asset_sha256=asset_hashes(),
        source_sha256={k: sha256(p) for k, p in source_paths.items()},
        runtime_versions={p: version(p) for p in ('mujoco', 'torch', 'better-actuator-models', 'mjlab')},
        selected_startup=dict(voltage_v=7.5, drop_gain_v_per_nm=.1, kp_fw=200.,
                              kp_scale=1., kd_scale=1., friction_scale=1.),
        compiled_motor_spec=dict(
            dof_ids=actuator._dof_ids.tolist(),
            armature=model.dof_armature[actuator._dof_ids.numpy()].tolist(),
            actuator_forcerange_nm=model.actuator_forcerange.tolist(),
            bias_type=model.actuator_biastype.tolist(),
            gear=model.actuator_gear.tolist()),
        timestep_s=float(model.opt.timestep), qpos=qpos.tolist(),
        physics_steps=0, force_solver_executed=True,
        native_control_written=False, command_delay_executed=False,
        full_runtime_initialization=False, gpu_equivalence_verified=False,
        suspended_component_fixture=True, contact_dynamics_verified=False,
        motor_capacity_verified=False, learned_balance=False, physical_motion_authorized=False)
