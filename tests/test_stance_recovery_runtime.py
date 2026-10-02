"""Focused D1 pulse adapter checks; synthetic counters are control-flow fixtures."""

import pytest
import torch

from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck.stance_recovery_runtime import RecoveryRuntime
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.mark.parametrize('cases', [
    ['zero-wrench'], ['+x', '-x'], ['+y', '-y'],
    ['zero-wrench', '+x', '-x', '+y', '-y'],
])
def test_contract_case_vectors_and_exact_schedule_edges(cases):
    checked = contract.cases_checked(cases)
    nbody, body_id = 9, 2
    for step in (0, 499, 500, 509, 510, contract.TOTAL_STEPS):
        for accepted in ([True]*len(checked), [False]*len(checked)):
            xfrc, qfrc = contract.expected_wrenches(checked, [step]*len(checked), accepted, nbody, body_id)
            active = 500 <= step <= 509
            for i, case in enumerate(checked):
                wanted = contract.force.expected_wrench(case if active and accepted[i] else 'zero-wrench', nbody, body_id)
                assert xfrc[i] == wanted[0][0]
                assert qfrc[i] == wanted[1][0] == [0]*contract.force.NV
            assert contract.validate_wrenches(xfrc, qfrc, checked, [step]*len(checked), accepted,
                                              nbody, body_id)


@pytest.mark.parametrize('bad', [[], ['+x', '+x'], ['unknown'], ['+x']*6])
def test_case_binding_rejects_empty_duplicate_unknown_and_oversized(bad):
    with pytest.raises(ValueError):
        RecoveryRuntime(bad, device='cpu')


def supported_refresh(env):
    refresh = env._refresh
    def supported(rows):
        refresh(rows)
        env.state.support[rows] = 1.
    env._refresh = supported


def exact_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            exact_tree(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            exact_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def pulse_fixture(cases=('+x', '-x')):
    torch.manual_seed(619)
    env = RecoveryRuntime(list(cases), device='cpu')
    # Exercise an actual nominal tick so all worlds have genuine BAM fields;
    # subsequent synthetic counters only select the pulse control-flow window.
    env.step(torch.zeros(len(cases), 10))
    # Synthetic episode counters/time put this focused control-flow fixture at
    # the declared window; this is not a full 500-step physical rollout.
    env.steps.fill_(contract.ONSET_STEP)
    env._view('time').fill_(contract.ONSET_STEP*.002)
    supported_refresh(env)
    # Pre-step state checks run before the post-step refresh hook.
    env.state.support.fill_(1.)
    return env


@pytest.mark.parametrize('phase', ['forced-solve', 'pre-capture', 'integrated-capture'])
def test_owned_pulse_is_cleared_on_every_failed_phase_before_next_solve(phase, monkeypatch):
    env = pulse_fixture(('+x',))
    calls = 0
    if phase == 'forced-solve':
        def fail(*_):
            raise ValueError('synthetic owned forced solve failure')
        monkeypatch.setattr(env, '_pulse_forward', fail)
    else:
        original = env._phase_capture
        def fail():
            nonlocal calls
            calls += 1
            if calls == (1 if phase == 'pre-capture' else 2):
                raise ValueError('synthetic owned phase capture failure')
            return original()
        monkeypatch.setattr(env, '_phase_capture', fail)
    with pytest.raises(ValueError, match='synthetic owned'):
        env.step_with_pulse(torch.zeros(1, 10))
    assert not env._view('xfrc_applied').any() and not env._view('qfrc_applied').any()
    assert env.faulted
    with pytest.raises(RuntimeError, match='job closeout'):
        env.reset(torch.ones(1, dtype=torch.bool))


@pytest.mark.parametrize('damage', ['body', 'cases', 'graph'])
def test_runtime_declared_body_cases_and_eager_path_are_frozen_before_control(damage):
    env = RecoveryRuntime(['+x'], device='cpu')
    before = env._control_snapshot()
    if damage == 'body':
        env.binding['body_id'] += 1
    elif damage == 'cases':
        env.cases = ('-x',)
    else:
        env.forward_graph = object()  # Explicit API refusal, no graph or physics executed.
    with pytest.raises(ValueError, match='frozen recovery|declared eager'):
        env.step_with_pulse(torch.zeros(1, 10))
    assert env.faulted
    assert all(torch.equal(v, env._control_snapshot()[k]) for k, v in before.items())


def test_ten_real_cpu_substeps_deliver_only_scheduled_pulse_and_capture_full_phases():
    env = pulse_fixture()
    before = {name: env._view(name).clone() for name in ('qpos', 'qvel', 'time', 'qacc_warmstart')}
    result = env.step_with_pulse(torch.zeros(2, 10), capture_control=True)
    assert result['executed_steps'].tolist() == [10, 10]
    assert len(result['boundaries']) == 11
    assert len(result['pulse_evidence']) == 10
    assert 'control_evidence' in result
    assert env.steps.tolist() == [510, 510]
    assert torch.allclose(env._view('time'), torch.full((2,), 1.02))
    for i, row in enumerate(result['pulse_evidence']):
        assert row['before_steps'].tolist() == [500+i, 500+i]
        assert row['accepted'].tolist() == [True, True]
        assert set(row) == {'before_steps', 'accepted', 'pre_xfrc', 'pre_qfrc',
                            'post_xfrc', 'post_qfrc', 'phases'}
        assert not row['post_xfrc'].any()
        assert not row['pre_qfrc'].any() and not row['post_qfrc'].any()
        if i == 0:
            assert set(row['phases']) == {'forced_pre', 'integrated', 'unforced_post'}
            for phase in row['phases'].values():
                assert set(phase) == {'solved', 'inputs', 'motor_fields'}
                assert set(phase['inputs']) == {'ctrl', 'xfrc_applied', 'qfrc_applied'}
                assert set(phase['motor_fields']) == {'dof_frictionloss', 'dof_damping'}
            phases = row['phases']
            for field in ('qpos', 'qvel', 'time', 'qacc_warmstart'):
                assert torch.equal(phases['forced_pre']['solved']['kinematics'][field],
                                   before[field])
            for field in ('qpos', 'qvel', 'time', 'qacc_warmstart'):
                assert torch.equal(phases['integrated']['solved']['kinematics'][field],
                                   phases['unforced_post']['solved']['kinematics'][field])
            for group in ('dynamics', 'solver', 'constraints', 'contacts'):
                assert exact_tree(phases['integrated']['solved'][group],
                                  phases['forced_pre']['solved'][group])
            body_id = env.plant_binding['body_id']
            assert phases['forced_pre']['inputs']['xfrc_applied'][0, body_id, 0] == 2
            assert phases['forced_pre']['inputs']['xfrc_applied'][1, body_id, 0] == -2
            assert not phases['unforced_post']['inputs']['xfrc_applied'].any()
        else:
            assert set(row['phases']) == {'forced_pre', 'integrated', 'unforced_post'}
            for name in ('forced_pre', 'integrated'):
                assert torch.equal(row['phases'][name]['inputs']['xfrc_applied'], row['pre_xfrc'])
                assert not row['phases'][name]['inputs']['qfrc_applied'].any()
            assert not row['phases']['unforced_post']['inputs']['xfrc_applied'].any()
    assert not env._view('xfrc_applied').any() and not env._view('qfrc_applied').any()


def test_runtime_phase_capture_starts_and_ends_at_exact_window_edges():
    env = pulse_fixture(('+x',))
    env.steps.fill_(contract.ONSET_STEP-1)
    env._view('time').fill_((contract.ONSET_STEP-1)*.002)
    result = env.step_with_pulse(torch.zeros(1, 10))
    assert [row['before_steps'].item() for row in result['pulse_evidence']] == list(range(499, 509))
    assert result['pulse_evidence'][0]['phases'] is None
    assert all(row['phases'] is not None for row in result['pulse_evidence'][1:])
    assert not result['pulse_evidence'][0]['pre_xfrc'].any()
    assert not result['pulse_evidence'][0]['post_xfrc'].any()
    assert result['pulse_evidence'][1]['pre_xfrc'][0, env.plant_binding['body_id'], 0] == 2
    second = env.step_with_pulse(torch.zeros(1, 10))
    assert len(second['pulse_evidence']) == 10
    assert second['pulse_evidence'][0]['before_steps'].item() == 509
    assert all(row['before_steps'].item() >= 509 for row in second['pulse_evidence'])
    assert second['pulse_evidence'][0]['pre_xfrc'][0, env.plant_binding['body_id'], 0] == 2
    assert second['pulse_evidence'][0]['phases'] is not None
    assert all(row['phases'] is None and not row['pre_xfrc'].any()
               for row in second['pulse_evidence'][1:])


@pytest.mark.parametrize('field', ['xfrc_applied', 'qfrc_applied'])
def test_external_force_input_faults_before_bam_or_fifo_changes(field):
    env = pulse_fixture()
    applied = env._view(field)
    applied[0].reshape(-1)[0] = .001
    queue = env.delay.queue.clone(); correction = env.delay.correction.clone()
    previous = env.motor.previous.clone(); steps = env.steps.clone()
    with pytest.raises(ValueError, match='zero applied-force arrays at entry'):
        env.step_with_pulse(torch.ones(2, 10))
    assert env.faulted and torch.equal(env.delay.queue, queue)
    assert torch.equal(env.delay.correction, correction)
    assert torch.equal(env.motor.previous, previous) and torch.equal(env.steps, steps)
    applied.zero_()
    with pytest.raises(RuntimeError, match='job closeout'):
        env.reset(torch.ones(2, dtype=torch.bool))


def test_integration_failure_clears_force_arrays_and_permanently_faults(monkeypatch):
    env = pulse_fixture()
    monkeypatch.setattr(env.integrator, 'integrate', lambda *_: (_ for _ in ()).throw(RuntimeError('synthetic Euler fault')))
    with pytest.raises(RuntimeError, match='synthetic Euler fault'):
        env.step_with_pulse(torch.zeros(2, 10))
    assert env.faulted
    assert not env._view('xfrc_applied').any() and not env._view('qfrc_applied').any()
    with pytest.raises(RuntimeError, match='job closeout'):
        env.reset(torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize('field', ['xfrc_applied', 'qfrc_applied'])
def test_installation_failure_also_clears_owned_force_arrays(field, monkeypatch):
    env = pulse_fixture(('+x',))
    pointer = env._view(field).data_ptr()
    original = torch.Tensor.copy_
    def fail_after_copy(destination, source, *args, **kwargs):
        result = original(destination, source, *args, **kwargs)
        if destination.data_ptr() == pointer:
            raise RuntimeError('synthetic pulse installation fault')
        return result
    monkeypatch.setattr(torch.Tensor, 'copy_', fail_after_copy)
    with pytest.raises(RuntimeError, match='synthetic pulse installation fault'):
        env.step_with_pulse(torch.zeros(1, 10))
    assert env.faulted
    assert not env._view('xfrc_applied').any() and not env._view('qfrc_applied').any()


def test_first_terminal_closed_row_gets_no_pulse_or_fifo_advance_while_sibling_runs():
    env = pulse_fixture()
    env.state.tilt[0] = .4
    frozen_physics = {name: env._view(name)[0].clone() for name in
                      ('qpos', 'qvel', 'time', 'qacc_warmstart', 'qacc', 'ctrl')}
    frozen_control = env._control_snapshot()
    result = env.step_with_pulse(torch.zeros(2, 10))
    assert result['executed_steps'].tolist() == [0, 10]
    assert env.steps.tolist() == [500, 510]
    assert result['terminal_records'][0]['physics_step'] == 500
    for name, value in frozen_physics.items():
        assert torch.equal(env._view(name)[0], value), name
    current_control = env._control_snapshot()
    for name, value in frozen_control.items():
        assert torch.equal(current_control[name][0], value[0]), name
    for row in result['pulse_evidence']:
        assert row['accepted'].tolist() == [False, True]
        assert not row['pre_xfrc'][0].any() and not row['pre_qfrc'][0].any()


def test_torque_gate_rejected_row_receives_no_scheduled_pulse(monkeypatch):
    env = pulse_fixture()
    def reject_first(command, live):
        torque = torch.zeros_like(env.motor.previous)
        torque[0].fill_(.37)
        rejected = live & (torque.abs().amax(dim=1) > .36)
        return dict(torque_nm=torque, accepted=live & ~rejected, rejected=rejected)
    monkeypatch.setattr(env.motor, 'compute', reject_first)
    result = env.step_with_pulse(torch.zeros(2, 10))
    first = result['pulse_evidence'][0]
    assert first['accepted'].tolist() == [False, True]
    assert not first['pre_xfrc'][0].any() and not first['pre_qfrc'][0].any()
    assert result['executed_steps'].tolist() == [0, 10]


def test_zero_window_explicit_collector_matches_nominal_short_step():
    recovery = RecoveryRuntime(['zero-wrench'], device='cpu')
    nominal = WarpStanceRuntime(1, device='cpu')
    actions = torch.zeros(1, 10)
    actual = recovery.step_with_pulse(actions)
    reference = nominal.step(actions)
    assert actual['executed_steps'].tolist() == reference['executed_steps'].tolist() == [10]
    assert torch.equal(recovery._view('qpos'), nominal._view('qpos'))
    assert torch.equal(recovery._view('qvel'), nominal._view('qvel'))
    assert exact_tree(recovery._control_snapshot(), nominal._control_snapshot())
    assert torch.equal(actual['reward'], reference['reward'])
    for left, right in zip(actual['boundaries'], reference['boundaries']):
        assert torch.equal(left['qpos'], right['qpos'])
        assert torch.equal(left['qvel'], right['qvel'])
        assert exact_tree(left['observation'], right['observation'])
    assert [r['phases'] for r in actual['pulse_evidence']] == [None]*10
