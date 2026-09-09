"""Short CPU control-path fixtures, not PPO or learned stance evidence."""
from copy import deepcopy
from hashlib import sha256
import math
import pytest
import torch
import mjlab

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
from test_stance_attempt_trace import binding


def capture(env, actions):
    recorder = trace.FirstAttemptTrace(binding(), env.snapshot())
    evidence = dict(protocol=control.PROTOCOL, binding=binding(), ticks=[])
    for action in actions:
        obs = env.observations()['actor']
        result = env.step(action, capture_control=True)
        recorder.append(result, obs, action)
        evidence['ticks'].append(trace.owned(result['control_evidence']))
    payload = recorder.payload(); trace.replay(payload, binding())
    return evidence, payload, plant.describe(env.native)


@pytest.fixture(scope='module')
def sample():
    return capture(WarpStanceRuntime(2, device='cpu'), [torch.ones(2, 10)*3, -torch.ones(2, 10)])


def test_clipped_actions_slew_fixed_delay_and_roundtrip(sample):
    evidence, payload, spec = sample
    raw, receipt = control.encode(evidence, payload, spec)
    assert control.verify(raw, sha256(raw).hexdigest(), payload, spec) == receipt
    assert receipt['motor_proposals_checked'] == 20 and not receipt['bam_outputs_recomputed']
    first = evidence['ticks'][0]
    assert torch.allclose(first['after_action']['correction'], torch.full((2, 10), .02))
    for p in first['proposals'][:3]:
        assert torch.equal(p['command']['position_target'], first['initial']['target'])
    assert torch.equal(first['proposals'][3]['command']['position_target'], first['after_action']['target'])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['binding', 'missing_tick', 'reset', 'slew', 'head', 'delay',
    'queue', 'pos', 'vel', 'effort', 'target_velocity', 'accepted', 'rejected', 'live', 'counter',
    'history', 'ctrl', 'voltage', 'gain', 'friction', 'free_dof', 'final', 'missing_proposal', 'extra_proposal', 'applied_torque'])
def test_corrupt_control_path_fails(sample, damage):
    evidence, payload, spec = deepcopy(sample)
    tick = evidence['ticks'][0]; p = tick['proposals'][0]
    if damage == 'binding': evidence['binding']['source'] = 'b'*40
    elif damage == 'missing_tick': evidence['ticks'].pop()
    elif damage == 'reset': evidence['ticks'][1]['initial']['previous'].zero_()
    elif damage == 'slew': tick['after_action']['correction'][0, 0] += .02
    elif damage == 'head': tick['after_action']['target'][0, 0] += .1
    elif damage == 'delay': p['command']['position_target'][0, 4] += .1
    elif damage == 'queue': p['committed']['queue'][0, 0, 4] += .1
    elif damage in ('pos', 'vel'): p['command'][damage][0, 0] += .1
    elif damage == 'effort': p['command']['effort_target'][0, 0] = .1
    elif damage == 'target_velocity': p['command']['velocity_target'][0, 0] = .1
    elif damage in ('accepted', 'rejected', 'live'): p[damage][0] = ~p[damage][0]
    elif damage == 'counter': p['before_steps'][0] += 1
    elif damage == 'history': p['committed']['previous'][0, 0] += .01
    elif damage == 'ctrl': p['committed']['ctrl'][0, 0] += .01
    elif damage == 'voltage': p['committed']['voltage'][0, 0] -= .1
    elif damage == 'gain': p['committed']['kp'][0, 0] -= 1
    elif damage == 'friction': p['committed']['friction'][0, 6] = -.1
    elif damage == 'free_dof': p['committed']['damping'][0, 0] = .1
    elif damage == 'final': tick['final']['queue'][0, 0, 0] += .1
    elif damage == 'missing_proposal': tick['proposals'].pop()
    elif damage == 'applied_torque': payload['ticks'][0]['boundaries'][1]['state']['torque'][0, 0] += .01
    else: tick['proposals'].append(deepcopy(p))
    with pytest.raises(ValueError): control.replay(evidence, payload, spec)


def test_hash_checked_before_deserialization(sample, monkeypatch):
    evidence, payload, spec = sample
    raw, _ = control.encode(evidence, payload, spec)
    monkeypatch.setattr(torch, 'load', lambda *_a, **_k: pytest.fail('loaded before hash'))
    with pytest.raises(ValueError, match='byte identity'): control.verify(raw+b'x', sha256(raw).hexdigest(), payload, spec)


def test_replay_stays_on_cpu_under_another_default_device(sample):
    evidence, payload, spec = sample
    with torch.device('meta'):
        assert control.replay(evidence, payload, spec)['motor_proposals_checked'] == 20


@pytest.mark.parametrize('all_rows', [False, True])
def test_rejected_proposal_does_not_advance_fifo_or_motor_history(monkeypatch, all_rows):
    env = WarpStanceRuntime(2, device='cpu')
    cls = type(env.actuator); compute = cls.compute
    def excess(self, cmd):
        torque = compute(self, cmd)
        if all_rows: torque[:] = .37
        torque[0] = .37
        return torque
    monkeypatch.setattr(cls, 'compute', excess)
    actions = [torch.ones(2, 10)] if all_rows else [torch.ones(2, 10), -torch.ones(2, 10)]
    evidence, payload, spec = capture(env, actions)
    control.replay(evidence, payload, spec)
    p = evidence['ticks'][0]['proposals'][0]
    assert p['rejected'][0]
    assert torch.equal(p['committed']['queue'][0], evidence['ticks'][0]['initial']['queue'][0])
    assert not p['committed']['previous'][0].any()
    bad = deepcopy(evidence); bad['ticks'][0]['proposals'][0]['committed']['friction'][0, 7] += .1
    with pytest.raises(ValueError, match='unaccepted friction'): control.replay(bad, payload, spec)


def test_first_physical_failure_freezes_control_on_later_ticks(monkeypatch):
    env = WarpStanceRuntime(2, device='cpu'); integrate = env.integrator.integrate
    calls = 0
    def tipped(rows):
        nonlocal calls
        integrate(rows); calls += 1
        if calls == 1: env._view('qpos')[0, 3:7] = torch.tensor([math.cos(.2), 0, math.sin(.2), 0])
    monkeypatch.setattr(env.integrator, 'integrate', tipped)
    evidence, payload, spec = capture(env, [torch.ones(2, 10), -torch.ones(2, 10)])
    control.replay(evidence, payload, spec)
    assert payload['ticks'][1]['executed_steps'].tolist() == [0, 10]
    for key in control.SHAPES:
        assert torch.equal(evidence['ticks'][0]['final'][key][0], evidence['ticks'][1]['final'][key][0])
    bad = deepcopy(evidence); bad['ticks'][1]['after_action']['correction'][0, 0] += .01
    with pytest.raises(ValueError): control.replay(bad, payload, spec)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: equal_tree(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    else: assert a == b


def test_capture_is_opt_in_owned_and_does_not_change_cpu_physics():
    a, b = WarpStanceRuntime(2, device='cpu'), WarpStanceRuntime(2, device='cpu')
    action = torch.ones(2, 10)
    plain = trace.owned(a.step(action)); captured = b.step(action, capture_control=True)
    evidence = captured.pop('control_evidence')
    equal_tree(plain, trace.owned(captured))
    evidence['final']['queue'].fill_(99)
    assert not (b.delay.queue == 99).any()
