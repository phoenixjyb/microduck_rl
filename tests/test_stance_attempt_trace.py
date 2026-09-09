"""Synthetic full trajectories and short real CPU integration, not policy scores."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.stance_transition import PhysicsState


def binding(n=2):
    return dict(protocol=trace.PROTOCOL, source='a'*40, runtime_sha256='b'*64,
        checkpoint_sha256='c'*64, launch_sha256='d'*64, checkpoint_iteration=511,
        evaluation_seed=541, worlds=n, capture_device='cpu')


def frame(steps):
    n = len(steps)
    state = PhysicsState(torch.zeros(n), torch.zeros(n, 3), torch.full((n,), .12),
        torch.ones(n, 2), torch.zeros(n, 14), torch.zeros(n, 14),
        torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
    qpos = torch.zeros(n, 21); qpos[:, 2] = .12; qpos[:, 3] = 1
    actor = torch.zeros(n, 44); actor[:, 2] = -1
    return dict(physics_steps=torch.tensor(steps), qpos=qpos, qvel=torch.zeros(n, 20),
        soft_limit_mask=torch.zeros(n, 14, dtype=torch.bool), state=asdict(state),
        observation=dict(actor=actor, critic=torch.cat((actor, torch.zeros(n, 6)), 1)))


def terminal(f, row, *, proposed=None):
    # Deliberately synthetic contact fixture, never retained as actual rollout.
    failed = f['physics_steps'][row] < 2500 or proposed is not None
    return dict(world_id=row, physics_step=int(f['physics_steps'][row]),
        terminated=bool(failed), timed_out=not bool(failed),
        qpos=f['qpos'][row].tolist(), qvel=f['qvel'][row].tolist(),
        state={k: v[row].tolist() for k, v in f['state'].items()},
        observation={k: v[row].tolist() for k, v in f['observation'].items()},
        contacts={'worldid': []}, rejected_proposed_torque_nm=proposed,
        trajectory_continuity_validated=False, checkpoint_admitted=False)


def tick(start=0, n=2):
    frames = [frame([i]*n) for i in range(start, start+11)]
    done = start == 2490
    return dict(boundaries=frames, terminal_records=[terminal(frames[-1], i) if done else None for i in range(n)],
        reward=torch.zeros(n), terminated=torch.zeros(n, dtype=torch.bool),
        timed_out=torch.full((n,), done), live=torch.full((n,), not done),
        episode_steps=torch.full((n,), start+10), executed_steps=torch.full((n,), 10))


def append(recorder, result):
    recorder.append(result, recorder.last['observation']['actor'], torch.zeros(recorder.n, 10))


@pytest.fixture(scope='module')
def full_payload():
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0]))
    for start in range(0, 2500, 10): append(recorder, tick(start))
    return recorder.payload()


def test_complete_synthetic_first_attempts_score_without_admission(full_payload):
    result = trace.replay(full_payload, binding())
    assert result['complete_attempts'] == result['numerical_passes'] == 2
    assert all(a['last_physics_step'] == 2500 for a in result['attempts'])
    assert not result['provenance_validated'] and not result['checkpoint_admitted']
    assert not result['learned_stance_accepted'] and not result['physical_motion_authorized']


def test_partial_prefix_is_not_a_full_attempt_and_owned():
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0])); result = tick()
    append(recorder, result)
    result['boundaries'][-1]['qpos'][0, 0] = 9
    payload = recorder.payload(); payload['initial']['qpos'][0, 0] = 9
    assert recorder.initial['qpos'][0, 0] == 0 and recorder.last['qpos'][0, 0] == 0
    score = trace.replay(recorder.payload(), binding())
    assert score['complete_attempts'] == score['numerical_passes'] == 0
    assert all(a['last_physics_step'] == 10 for a in score['attempts'])


@pytest.mark.parametrize('damage', ['skip', 'reset', 'duplicate', 'nan', 'warning',
    'counter', 'reward', 'input', 'initial_physics', 'missing_terminal', 'fake_terminal'])
def test_bad_tick_fails_atomically_and_cannot_reset(damage):
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0])); result = tick()
    policy_input = recorder.last['observation']['actor'].clone()
    if damage == 'skip': result['boundaries'][2]['physics_steps'][0] += 1
    elif damage == 'reset': result['boundaries'][2]['physics_steps'][0] = 0
    elif damage == 'duplicate': result['boundaries'][2] = deepcopy(result['boundaries'][1])
    elif damage == 'nan': result['boundaries'][2]['qpos'][0, 0] = float('nan')
    elif damage == 'warning': result['boundaries'][2]['state']['warning'][0] = True
    elif damage == 'counter': result['executed_steps'][0] = 9
    elif damage == 'reward': result['reward'][0] = float('inf')
    elif damage == 'input': policy_input[0, 0] = .1
    elif damage == 'initial_physics': result['boundaries'][0]['qpos'][0, 0] = .1
    elif damage == 'missing_terminal': result['boundaries'][-1]['state']['tilt'][0] = .4
    elif damage == 'fake_terminal': result['terminal_records'][0] = terminal(result['boundaries'][-1], 0)
    with pytest.raises(ValueError): recorder.append(result, policy_input, torch.zeros(2, 10))
    assert not recorder.ticks and recorder.faulted
    with pytest.raises(ValueError, match='faulted'): recorder.payload()
    with pytest.raises(ValueError, match='faulted'): append(recorder, tick())


def isolated_tick(*, proposed=False):
    result = tick()
    for f in result['boundaries'][1:]:
        f['physics_steps'][0] = 0 if proposed else 1
        if not proposed: f['state']['tilt'][0] = .4
    f = result['boundaries'][-1]
    result.update(terminal_records=[terminal(f, 0, proposed=[.37]*14 if proposed else None), None],
        terminated=torch.tensor([True, False]), live=torch.tensor([False, True]),
        episode_steps=torch.tensor([0 if proposed else 1, 10]),
        executed_steps=torch.tensor([0 if proposed else 1, 10]), reward=torch.tensor([-2., 0.]))
    return result


@pytest.mark.parametrize('proposed', [False, True])
def test_first_physical_or_pre_step_torque_failure_retained(proposed):
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0])); first = isolated_tick(proposed=proposed)
    append(recorder, first)
    second = tick(10)
    for f in second['boundaries']:
        for k in ('physics_steps', 'qpos', 'qvel', 'soft_limit_mask'): f[k][0] = recorder.last[k][0]
        for group in ('state', 'observation'):
            for k in f[group]: f[group][k][0] = recorder.last[group][k][0]
    second.update(terminal_records=deepcopy(recorder.terminals), episode_steps=second['boundaries'][-1]['physics_steps'].clone(),
        executed_steps=torch.tensor([0, 10]), live=torch.tensor([False, True]))
    append(recorder, second)
    scored = trace.replay(recorder.payload(), binding())
    assert scored['complete_attempts'] == 1 and scored['numerical_passes'] == 0
    assert scored['attempts'][0]['last_physics_step'] == (0 if proposed else 1)
    assert scored['attempts'][1]['last_physics_step'] == 20


@pytest.mark.parametrize('damage', ['terminal_qpos', 'terminal_state', 'terminal_observation',
    'terminal_contact_world', 'terminal_step', 'after_terminal'])
def test_terminal_evidence_cannot_diverge_or_resume(damage):
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0])); result = isolated_tick()
    r = result['terminal_records'][0]
    if damage == 'terminal_qpos': r['qpos'][0] = 4
    elif damage == 'terminal_state': r['state']['tilt'] = .2
    elif damage == 'terminal_observation': r['observation']['actor'][0] = .2
    elif damage == 'terminal_contact_world': r['contacts']['worldid'] = [1]
    elif damage == 'terminal_step': r['physics_step'] = 2
    elif damage == 'after_terminal': result['boundaries'][3]['physics_steps'][0] = 2
    with pytest.raises(ValueError): append(recorder, result)


@pytest.mark.parametrize('key,value', [('source', 'bad'), ('runtime_sha256', 'x'*64),
    ('checkpoint_iteration', 500), ('evaluation_seed', 523), ('worlds', True), ('capture_device', 'cuda')])
def test_binding_refuses_wrong_experiment(key, value):
    b = binding(); b[key] = value
    with pytest.raises(ValueError): trace.FirstAttemptTrace(b, frame([0, 0]))


def test_replay_requires_expected_binding_and_never_claims_cpu_as_cuda():
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0]))
    other = {**binding(), 'source': 'e'*40}
    with pytest.raises(ValueError, match='binding mismatch'): trace.replay(recorder.payload(), other)
    other = {**binding(), 'capture_device': 'cuda:0'}
    with pytest.raises(ValueError, match='observed capture device'): trace.FirstAttemptTrace(other, frame([0, 0]))


@pytest.mark.parametrize('damage', ['changed_terminal', 'closed_reward', 'closed_observation', 'closed_state'])
def test_closed_world_cannot_change_on_a_later_tick(damage):
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0]))
    append(recorder, isolated_tick())
    second = tick(10)
    for f in second['boundaries']:
        for key in ('physics_steps', 'qpos', 'qvel', 'soft_limit_mask'): f[key][0] = recorder.last[key][0]
        for group in ('state', 'observation'):
            for key in f[group]: f[group][key][0] = recorder.last[group][key][0]
    second.update(terminal_records=deepcopy(recorder.terminals),
        episode_steps=torch.tensor([1, 20]), executed_steps=torch.tensor([0, 10]), live=torch.tensor([False, True]))
    if damage == 'changed_terminal': second['terminal_records'][0]['contacts']['worldid'] = [0]
    elif damage == 'closed_reward': second['reward'][0] = .1
    elif damage == 'closed_observation':
        second['boundaries'][0]['observation']['actor'][0, -1] = .1
        second['boundaries'][0]['observation']['critic'][0, 43] = .1
    elif damage == 'closed_state': second['boundaries'][2]['qvel'][0, 3] = .1
    with pytest.raises(ValueError): append(recorder, second)
    assert len(recorder.ticks) == 1


def test_lost_tick_and_false_end_receipt_are_rejected(full_payload):
    damaged = deepcopy(full_payload); del damaged['ticks'][3]
    with pytest.raises(ValueError, match='counter continuity'): trace.replay(damaged, binding())
    damaged = deepcopy(full_payload); damaged['ticks'][-1]['terminal_records'] = [None, None]
    with pytest.raises(ValueError): trace.replay(damaged, binding())


def test_hidden_earlier_failure_cannot_be_averaged_into_recovery():
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0])); result = tick()
    result['boundaries'][2]['state']['tilt'][0] = .4
    with pytest.raises(ValueError, match='first terminal'): append(recorder, result)


def test_serialized_hash_binding_and_cpu_replay(tmp_path):
    recorder = trace.FirstAttemptTrace(binding(), frame([0, 0])); append(recorder, isolated_tick())
    raw, receipt = trace.encode(recorder.payload(), binding())
    assert receipt['bytes'] == len(raw)
    assert trace.verify(raw, receipt['sha256'], binding()) == receipt['score']
    with pytest.raises(ValueError, match='hash mismatch'):
        trace.verify(raw+b'corrupt', receipt['sha256'], binding())
    with pytest.raises(ValueError, match='binding mismatch'):
        trace.verify(raw, receipt['sha256'], {**binding(), 'checkpoint_sha256': 'e'*64})
    assert not torch.cuda.is_initialized()


def test_size_and_digest_checked_before_tensor_deserialization(monkeypatch):
    monkeypatch.setattr(trace, 'MAX_TRACE_BYTES', 4)
    def forbidden(*args, **kwargs): pytest.fail('must reject before torch.load')
    monkeypatch.setattr(trace.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='bounded'): trace.verify(b'large', 'a'*64, binding())
    with pytest.raises(ValueError, match='hash mismatch'): trace.verify(b'tiny', 'a'*64, binding())


def test_short_real_warp_cpu_policy_input_before_correction_and_first_terminal():
    import mjlab
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(2, device='cpu')
    recorder = trace.FirstAttemptTrace(binding(), env.snapshot())
    actions = torch.ones(2, 10)
    policy_input = env.observations()['actor']
    recorder.append(env.step(actions), policy_input, actions)
    assert not torch.equal(recorder.ticks[0]['actor_input'], recorder.ticks[0]['boundaries'][0]['observation']['actor'])
    # Actual CPU integration followed by an explicitly synthetic tilt injection.
    integrate = env.integrator.integrate
    def inject(live):
        integrate(live)
        env._view('qpos')[0, 3:7] = torch.tensor([.9800666, 0, .1986693, 0])
    env.integrator.integrate = inject
    policy_input = env.observations()['actor']
    recorder.append(env.step(actions), policy_input, actions)
    scored = trace.replay(recorder.payload(), binding())
    assert scored['complete_attempts'] == 1
    assert scored['attempts'][0]['last_physics_step'] == 11
    assert scored['attempts'][1]['last_physics_step'] == 20
    assert not torch.cuda.is_initialized()
