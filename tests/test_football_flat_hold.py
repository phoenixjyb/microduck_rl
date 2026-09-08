from copy import deepcopy

import mujoco
import numpy as np
import pytest

from mjlab_microduck import football_flat_hold as hold


def safe_frame():
    return dict(time_s=0., warning_count=0, forbidden_contacts=[], root_height_m=.12,
        tilt_rad=0., root_speed_mps=0., hinge_speed_rad_s=[0.]*14,
        applied_torque_nm=[0.]*14, outside_hard_limit=False,
        support_normal_n={'left': 1., 'right': 1.})


def test_delay_exactly_three_steps_owns_inputs_and_resets():
    initial = np.zeros(14); delay = hold.FixedDelay(initial)
    initial[:] = 99
    for _ in range(3): assert np.array_equal(delay.push(np.ones(14)), np.zeros(14))
    assert np.array_equal(delay.push(np.ones(14)), np.ones(14))
    delay.reset()
    assert np.array_equal(delay.push(np.ones(14)), np.zeros(14))
    with pytest.raises(ValueError): delay.push(np.full(14, np.nan))


@pytest.mark.parametrize('key,value,reason', [
    ('warning_count', 1, 'mujoco-warning'),
    ('forbidden_contacts', [['head','floor']], 'forbidden-contact'),
    ('root_height_m', .079, 'root-height'),
    ('tilt_rad', .351, 'tilt'),
    ('root_speed_mps', 1.001, 'root-speed'),
    ('hinge_speed_rad_s', [10.01]*14, 'hinge-speed'),
    ('applied_torque_nm', [-.361]*14, 'applied-torque'),
    ('outside_hard_limit', True, 'hard-joint-limit'),
])
def test_each_abort_gate(key, value, reason):
    frame = safe_frame(); frame[key] = value
    assert hold.stop_reasons(frame) == [reason]


def test_support_grace_and_proposed_torque_are_independent():
    frame = safe_frame(); frame['support_normal_n']['left'] = 0
    assert not hold.stop_reasons(frame)
    frame['time_s'] = .1
    assert hold.stop_reasons(frame) == ['lost-foot-support']
    assert hold.stop_reasons(safe_frame(), [.361]*14) == ['proposed-torque']
    with pytest.raises(ValueError): hold.stop_reasons(safe_frame(), [np.nan]*14)


def test_floor_geometry_is_unassisted_and_unstepped():
    model, data, actuator = hold.bam.build_component_fixture(flat_floor=True)
    before = data.qpos.copy()
    gaps = hold.place_on_floor(model, data)
    assert all(0 <= g <= .0003 for g in gaps)
    root = int(model.joint('trunk_base_freejoint').qposadr[0])
    before[root+2] = data.qpos[root+2]
    assert np.array_equal(before, data.qpos)
    assert data.time == 0 and model.neq == model.nmocap == 0
    frame = hold.observe(model, data, actuator)
    assert not hold.stop_reasons(frame)
    assert not np.any(data.ctrl)


def test_boundary_failure_stops_before_computation_or_step(monkeypatch):
    frame = safe_frame(); frame['tilt_rad'] = .4
    monkeypatch.setattr(hold, 'observe', lambda *args: deepcopy(frame))
    monkeypatch.setattr(hold.bam, 'compute_snapshot', lambda *a: pytest.fail('computed after abort'))
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('stepped after abort'))
    report = hold.run_hold()
    assert report['physics_steps'] == 0 and report['decision'] == 'diagnostic-abort'
    assert report['stop_reasons'] == ['tilt']
    assert report['commands'] == []


def test_proposed_torque_failure_is_retained_but_not_applied(monkeypatch):
    monkeypatch.setattr(hold.bam, 'compute_snapshot', lambda *a:
        dict(torque_nm=[.4]*14))
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('applied excessive torque'))
    report = hold.run_hold()
    assert report['physics_steps'] == 0
    assert report['stop_reasons'] == ['proposed-torque']
    assert len(report['commands']) == 1 and report['commands'][0]['applied'] is False


def test_nonfinite_boundary_retains_prefix_without_serializing_nan(monkeypatch):
    def invalid(*a): raise ValueError('nonfinite boundary')
    monkeypatch.setattr(hold, 'observe', invalid)
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('step after NaN'))
    report = hold.run_hold()
    assert report['decision'] == 'diagnostic-abort'
    assert report['error'] == 'nonfinite boundary'
    assert report['frames'] == [] and not report['final_boundary_observed']


def test_final_boundary_failure_not_mislabeled_completion(monkeypatch):
    # Two synthetic time advances, not a native rollout or performance result.
    monkeypatch.setattr(hold, 'STEPS', 2)
    monkeypatch.setattr(mujoco, 'mj_step', lambda m, d: setattr(d, 'time', d.time+hold.DT))
    original = hold.observe
    def observed(model, data, actuator):
        frame = original(model, data, actuator)
        if data.time >= .004: frame['tilt_rad'] = .4
        return frame
    monkeypatch.setattr(hold, 'observe', observed)
    report = hold.run_hold()
    assert report['physics_steps'] == 2 and len(report['frames']) == 3
    assert report['decision'] == 'diagnostic-abort'
    assert report['stop_reasons'] == ['tilt'] and report['final_boundary_observed']
    assert all(c['applied'] for c in report['commands'])
