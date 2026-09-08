from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck import football_margin_probe as probe


def test_tampered_input_refused_before_model_build(tmp_path, monkeypatch):
    p = tmp_path/'input.json'; p.write_text('{}')
    monkeypatch.setattr(probe, 'build_fixture', lambda **k: pytest.fail('unverified input used'))
    with pytest.raises(ValueError, match='input hash'): probe.read_pose(p)


def test_full_friction_utilization_uses_actual_not_reduced_cone():
    frames = np.tile(np.eye(3), (3, 1, 1))
    forces = [[4, 1, 1], [4, 0, 2], [4, 0, 0]]
    r = probe.full_friction_exposure(frames, np.ones((3, 2)), forces)
    assert [x['full_pyramid_utilization'] for x in r] == [.5, .5, 0.]
    assert r[0]['full_pyramid_slack_n'] == 2.


def test_zero_normal_is_not_a_supported_contact():
    r = probe.full_friction_exposure(np.tile(np.eye(3), (3, 1, 1)), np.ones((3, 2)), np.zeros((3, 3)))
    assert all(x['full_pyramid_utilization'] is None for x in r)


def test_invalid_friction_refused():
    with pytest.raises(ValueError):
        probe.full_friction_exposure(np.tile(np.eye(3), (3, 1, 1)), np.zeros((3, 2)), np.zeros((3, 3)))


def test_native_retained_pose_is_unchanged_and_no_search_or_dynamics_run(monkeypatch):
    path = Path(__file__).parents[1]/'artifacts/diagnostics/football-b0a-stance-local-e79aadf.json'
    if not path.is_file(): pytest.skip('retained Mac B0-A diagnostic not installed')
    def forbidden(*a, **k): pytest.fail('search or dynamics invoked')
    monkeypatch.setattr(probe.stance, 'adjust_pose', forbidden)
    for name in ('mj_step', 'mj_forward', 'mj_inverse', 'mj_fwdConstraint', 'mj_fwdActuation'):
        monkeypatch.setattr(mujoco, name, forbidden)
    raw = path.read_bytes()
    r = probe.run_probe(path)
    assert path.read_bytes() == raw
    assert r['interior_allocation_consistent']
    assert all(e['full_pyramid_utilization'] <= .5+1e-8 for e in r['full_friction_exposure'])
    assert len(r['named_ideal_torque_delta_nm']) == 14
    assert not r['pose_search_repeated'] and not r['contact_friction_modified']
    assert not r['motor_acceptance'] and not r['learned_balance']
    assert r['physics_steps'] == 0
