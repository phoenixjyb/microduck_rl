from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from bam.mjlab import BamActuator

from mjlab_microduck import football_bam_probe as probe


def nominal(model, actuator):
    return np.array([model.joint(n).qposadr[0] for n in actuator.target_names], dtype=int)


def test_native_floating_base_friction_is_indexed_by_dof_not_joint():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos="0 0 2"><freejoint/><geom size=".1"/>
      <body pos="0 0 .2"><joint name="hinge" frictionloss=".1"/>
      <geom size=".1"/></body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); data.qvel[6] = 1.
    mujoco.mj_forward(model, data)
    assert model.joint('hinge').id == 1
    assert model.joint('hinge').dofadr[0] == 6
    snapshot = probe.native_snapshot(model, data)
    assert snapshot.efc.id.tolist() == [[6]]
    fake = SimpleNamespace(_data=snapshot, _device='cpu')
    fake._as_tensor = lambda x: x
    force = BamActuator._dof_friction_force(fake, model.nv)
    assert force[0, 6].item() == pytest.approx(data.efc_force[0])
    assert force[0, 1].item() == 0
    # A contact-row ID is not a DOF index and must not be scattered.
    snapshot.efc.type[0, 0] = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
    snapshot.efc.id[0, 0] = 9999
    assert torch.count_nonzero(BamActuator._dof_friction_force(fake, model.nv)) == 0
    assert data.time == 0


def test_real_spec_is_motor_mode_with_bam_armature_and_no_stale_pd():
    model, data, actuator = probe.build_component_fixture()
    dofs = actuator._dof_ids.numpy()
    assert model.nu == 14
    assert np.all(model.actuator_biastype == int(mujoco.mjtBias.mjBIAS_NONE))
    assert np.all(model.actuator_gaintype == int(mujoco.mjtGain.mjGAIN_FIXED))
    assert np.all(model.actuator_gainprm[:, 0] == 1)
    assert np.all(model.actuator_gear[:, 0] == 1)
    assert not np.any(model.actuator_gear[:, 1:])
    assert np.allclose(model.dof_armature[dofs], actuator._bam_model.armature.value)
    assert not np.any(model.dof_damping[dofs])
    assert not np.any(model.dof_frictionloss[dofs])
    assert np.allclose(model.dof_solref[dofs], [-5e4, -2e2])
    ceiling = 8.2*actuator._bam_model.kt.value/actuator._bam_model.R.value
    assert np.allclose(model.actuator_forcerange, [-ceiling, ceiling])
    assert data.time == 0 and not np.any(data.ctrl)


def test_compute_uses_real_training_method_and_leaves_native_buffers_unchanged(monkeypatch):
    model, data, actuator = probe.build_component_fixture()
    target = data.qpos[nominal(model, actuator)] + .02
    before = {k: getattr(data, k).copy() for k in ('qpos', 'qvel', 'ctrl', 'qfrc_actuator')}
    native_friction = model.dof_frictionloss.copy()
    original = BamActuator.compute
    calls = []
    def audited(self, cmd):
        calls.append(cmd.pos.device.type)
        return original(self, cmd)
    monkeypatch.setattr(BamActuator, 'compute', audited)
    result = probe.compute_snapshot(model, data, actuator, target)
    assert calls == ['cpu']
    assert len(result['torque_nm']) == 14 and min(result['torque_nm']) > 0
    assert min(result['frictionloss_nm']) > 0
    for key, saved in before.items(): assert np.array_equal(getattr(data, key), saved)
    assert np.array_equal(model.dof_frictionloss, native_friction)
    assert data.time == 0
    actuator._data.qfrc_bias.fill_(123)
    assert not np.all(data.qfrc_bias == 123)  # No shared numpy/Torch memory.


@pytest.mark.parametrize('target', [np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)])
def test_invalid_commands_rejected_before_compute(target, monkeypatch):
    model, data, actuator = probe.build_component_fixture()
    monkeypatch.setattr(actuator, 'compute', lambda c: pytest.fail('invalid input used'))
    with pytest.raises(ValueError, match='finite 14-target'):
        probe.compute_snapshot(model, data, actuator, target)


def test_lagged_voltage_sag_and_reset_are_actual_bam_state():
    model, data, actuator = probe.build_component_fixture()
    target = data.qpos[nominal(model, actuator)] + .02
    first = probe.compute_snapshot(model, data, actuator, target)
    second = probe.compute_snapshot(model, data, actuator, target)
    assert first['effective_voltage_v'] == 7.5
    assert second['effective_voltage_v'] == pytest.approx(
        max(6., 7.5-.1*sum(abs(t) for t in first['torque_nm'])))
    actuator.reset()
    assert probe.compute_snapshot(model, data, actuator, target) == first


def test_nonfinite_native_state_rejected_before_compute(monkeypatch):
    model, data, actuator = probe.build_component_fixture()
    target = data.qpos[nominal(model, actuator)].copy()
    data.qvel[6] = np.nan
    monkeypatch.setattr(actuator, 'compute', lambda c: pytest.fail('invalid state used'))
    with pytest.raises(ValueError, match='finite joint state'):
        probe.compute_snapshot(model, data, actuator, target)


def test_native_snapshot_rejects_nonfinite_forces():
    model, data, _ = probe.build_component_fixture()
    data.qfrc_bias[6] = np.inf
    with pytest.raises(ValueError, match='finite native snapshot'):
        probe.native_snapshot(model, data)


def test_m6_native_formula_is_not_silently_substituted_for_training_formula():
    _, _, actuator = probe.build_component_fixture()
    model = actuator._bam_model
    # Same-sign loads: native gates the quadratic term off; training keeps it.
    motor, external = .2, .1
    native, _ = model.compute_frictions(motor, external, 0.)
    training = actuator._compute_friction_budget(
        torch.tensor([[motor]], dtype=torch.float64),
        torch.tensor([[external]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)).item()
    assert training-native == pytest.approx(model.load_friction_external_quad.value*external**2)
    # Equal magnitudes: native strict inequalities select neither direction.
    native, _ = model.compute_frictions(.1, -.1, 0.)
    training = actuator._compute_friction_budget(torch.tensor([[.1]], dtype=torch.float64),
        torch.tensor([[-.1]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)).item()
    assert training-native == pytest.approx(model.load_friction_motor_quad.value*.1**2)


def test_independent_component_report_never_steps_or_claims_balance(monkeypatch):
    def forbidden(*a, **k): pytest.fail('integration or inverse dynamics called')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr(mujoco, 'mj_inverse', forbidden)
    report = probe.run_probe()
    assert [s['target_offset_rad'] for s in report['samples']] == [0, .02, -.02]
    assert report['samples'][0]['torque_nm'] == [0.]*14
    assert np.allclose(report['samples'][1]['torque_nm'],
                       -np.asarray(report['samples'][2]['torque_nm']), atol=1e-6)
    assert all(s['effective_voltage_v'] == 7.5 for s in report['samples'])
    assert report['physics_steps'] == 0
    assert not report['learned_balance'] and not report['gpu_equivalence_verified']
    assert not report['command_delay_executed'] and not report['native_control_written']
    assert len(report['asset_sha256']) > 10
