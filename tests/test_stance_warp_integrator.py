"""CPU Warp integration tests; these do not establish CUDA/robot contact parity."""

from dataclasses import replace

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import pytest
import torch
import warp as wp

from mjlab_microduck import stance_warp_integrator as integration


def fixture(*, damping=False):
    # Component-only model: no collision/robot/BAM; explicit damping-disabled
    # Euler exercises the stock integration kernels on Warp's CPU backend.
    flag = '' if damping else '<flag eulerdamp="disable"/>'
    native = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep="0.002" integrator="Euler">{flag}</option>
      <worldbody><body><joint type="free" damping=".1"/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>''')
    nd = mujoco.MjData(native)
    mujoco.mj_forward(native, nd)
    with wp.ScopedDevice('cpu'):
        model = mjwarp.put_model(native)
        data = mjwarp.put_data(native, nd, nworld=2, nconmax=4, njmax=8)
    # Explicit synthetic freshly-solved accelerations, not a contact rollout.
    wp.to_torch(data.qvel)[:] = torch.tensor([[.1, .2, .3, .1, 0, 0]]*2)
    wp.to_torch(data.qacc)[:] = torch.tensor([[1., 2., 3., .5, 0, 0]]*2)
    if damping:
        # Also exercise the real forward solve and implicit damping path. The
        # synthetic qacc above is replaced by the actual solved acceleration.
        with wp.ScopedDevice('cpu'): mjwarp.forward(model, data)
    return model, data


def state(data):
    return {name: getattr(data, name).numpy().copy() for name in integration.STATE_FIELDS}


@pytest.mark.parametrize('damping', [False, True])
def test_live_row_matches_stock_euler_and_closed_row_is_bitwise_unchanged(damping):
    model, data = fixture(damping=damping)
    runtime = integration.EulerCandidateCommit(model, data)
    original = state(data)
    with wp.ScopedDevice('cpu'):
        reference = replace(data, **{name: wp.clone(getattr(data, name))
                                   for name in integration.STATE_FIELDS})
        integration.forward.euler(model, reference)
    expected = state(reference)
    runtime.integrate(torch.tensor([True, False]))
    result = state(data)
    for name in integration.STATE_FIELDS:
        assert np.array_equal(result[name][0], expected[name][0]), name
        assert np.array_equal(result[name][1], original[name][1]), name
    assert result['time'][0] == pytest.approx(.002)
    if not damping:
        assert result['qpos'][0, 0] == pytest.approx(.002*(.1+.002))
    for _ in range(3): runtime.integrate(torch.tensor([True, False]))
    assert data.time.numpy()[0] == pytest.approx(.008)
    for name in integration.STATE_FIELDS:
        assert np.array_equal(getattr(data, name).numpy()[1], original[name][1])


def test_fully_closed_batch_does_not_call_integrator(monkeypatch):
    model, data = fixture(); runtime = integration.EulerCandidateCommit(model, data)
    before = state(data)
    monkeypatch.setattr(integration.forward, 'euler', lambda *a: pytest.fail('closed batch integrated'))
    runtime.integrate(torch.zeros(2, dtype=torch.bool))
    assert all(np.array_equal(state(data)[n], before[n]) for n in before)


def test_nonfinite_candidate_commits_nothing_and_faults_instance(monkeypatch):
    model, data = fixture(); runtime = integration.EulerCandidateCommit(model, data)
    before = state(data)
    def invalid(model, proposal):
        wp.to_torch(proposal.qpos).fill_(float('nan'))
        wp.to_torch(proposal.time).fill_(999)
    monkeypatch.setattr(integration.forward, 'euler', invalid)
    with pytest.raises(ValueError, match='nothing committed'):
        runtime.integrate(torch.tensor([True, False]))
    assert all(np.array_equal(state(data)[n], before[n]) for n in before)
    with pytest.raises(RuntimeError, match='faulted'):
        runtime.integrate(torch.tensor([True, False]))


@pytest.mark.parametrize('mask', [torch.ones(2), torch.ones(3, dtype=torch.bool)])
def test_invalid_live_mask_refused(mask):
    model, data = fixture(); runtime = integration.EulerCandidateCommit(model, data)
    with pytest.raises(ValueError, match='live mask'): runtime.integrate(mask)


def test_replaced_buffers_require_fresh_binding():
    model, data = fixture(); runtime = integration.EulerCandidateCommit(model, data)
    with wp.ScopedDevice('cpu'): data.qvel = wp.clone(data.qvel)
    with pytest.raises(ValueError, match='replaced'): runtime.integrate(torch.tensor([True, False]))


def test_unsupported_integrator_refused():
    model, data = fixture(); model.opt.integrator = integration.IntegratorType.RK4
    with pytest.raises(ValueError, match='Euler plant'): integration.EulerCandidateCommit(model, data)


def test_runtime_write_set_pin_cannot_be_silently_changed(monkeypatch):
    monkeypatch.setitem(integration.AUDITED_SOURCE, 'forward', '0'*64)
    with pytest.raises(ValueError, match='write-set audit'): integration.check_runtime()


def test_candidate_and_committed_state_never_alias():
    model, data = fixture(); runtime = integration.EulerCandidateCommit(model, data)
    for name in integration.STATE_FIELDS:
        assert getattr(runtime.candidate, name).ptr != getattr(data, name).ptr
    before = state(data)
    runtime.proposals['time'].fill_(123)
    assert np.array_equal(state(data)['time'], before['time'])


def test_nonfinite_committed_state_faults_before_integrating(monkeypatch):
    model, data = fixture(); runtime = integration.EulerCandidateCommit(model, data)
    wp.to_torch(data.qvel)[1, 0] = float('nan')
    monkeypatch.setattr(integration.forward, 'euler', lambda *a: pytest.fail('bad input integrated'))
    with pytest.raises(ValueError, match='nonfinite committed'):
        runtime.integrate(torch.tensor([True, False]))
