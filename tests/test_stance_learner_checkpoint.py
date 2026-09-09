from copy import deepcopy
from hashlib import sha256
import io
import pytest
import torch

from mjlab_microduck import stance_learner_checkpoint as codec
from mjlab_microduck import stance_checkpoint as weights
from mjlab_microduck.stance_ppo import CpuStanceLearner
from test_stance_ppo import SyntheticEnv, collect

BINDING = dict(source='a'*40, runtime_sha256='b'*64, fixture_launch_sha256='c'*64)


@pytest.fixture(scope='module')
def saved():
    learner = CpuStanceLearner(); env = SyntheticEnv(); collect(learner, env); learner.update()
    return codec.encode(learner, BINDING), env, learner


def test_restored_synthetic_continuation_matches_model_adam_and_rng(saved):
    raw, env, first = saved; digest = sha256(raw).hexdigest(); before = torch.get_rng_state().clone()
    second, receipt = codec.load_fixture(raw, digest, BINDING)
    assert receipt['optimizer_restored'] and not receipt['simulation_resume_authorized']
    collect(first, deepcopy(env)); collect(second, deepcopy(env))
    assert torch.equal(first.storage.actions, second.storage.actions)
    a, b = first.update(), second.update()
    assert a == b
    assert weights.state_hash(weights.states_of(first.actor, first.critic)) == weights.state_hash(weights.states_of(second.actor, second.critic))
    for i, state in first.algorithm.optimizer.state_dict()['state'].items():
        for k, v in state.items(): assert torch.equal(v, second.algorithm.optimizer.state_dict()['state'][i][k])
    assert torch.equal(first.rng, second.rng) and torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_only_empty_iteration_boundary_is_saveable():
    learner = CpuStanceLearner(); learner.collect_one(SyntheticEnv())
    with pytest.raises(ValueError, match='empty iteration'): codec.encode(learner, BINDING)
    learner.faulted = True
    with pytest.raises(ValueError, match='closeout'): codec.encode(learner, BINDING)


def test_restored_learner_refuses_simulator_continuation(saved):
    raw = saved[0]
    learner, _ = codec.load_fixture(raw, sha256(raw).hexdigest(), BINDING)
    env = SyntheticEnv(); env.synthetic_ppo_fixture = False
    with pytest.raises(ValueError, match='cannot resume simulator'): learner.collect_one(env)


@pytest.mark.parametrize('damage', ['lr', 'moment', 'negative_second', 'step', 'missing_state', 'rng',
    'worlds', 'updates', 'seed', 'initial', 'config', 'resume_flag', 'binding', 'weights', 'extra'])
def test_rehashed_malformed_checkpoint_refused(saved, damage):
    raw = saved[0]; value = torch.load(io.BytesIO(raw), weights_only=True)
    state = next(iter(value['optimizer']['state'].values()))
    if damage == 'lr': value['optimizer']['param_groups'][0]['lr'] = .01
    elif damage == 'moment': state['exp_avg'].flatten()[0] = float('nan')
    elif damage == 'negative_second': state['exp_avg_sq'].flatten()[0] = -.1
    elif damage == 'step': state['step'] += 1
    elif damage == 'missing_state': value['optimizer']['state'].clear()
    elif damage == 'rng': value['torch_cpu_rng'] = torch.zeros(3, dtype=torch.uint8)
    elif damage == 'worlds': value['worlds'] = 64
    elif damage == 'updates': value['updates'] = 4
    elif damage == 'seed': value['seed'] = 521
    elif damage == 'initial': value['initial_state_sha256'] = '0'*64
    elif damage == 'config': value['config']['gamma'] = .5
    elif damage == 'resume_flag': value['simulation_resume_authorized'] = True
    elif damage == 'binding': value['binding']['source'] = 'd'*40
    elif damage == 'weights': next(iter(value['states']['actor'].values())).flatten()[0] = float('nan')
    else: value['extra'] = True
    buffer = io.BytesIO(); torch.save(value, buffer); changed = buffer.getvalue()
    with pytest.raises(ValueError): codec.load_fixture(changed, sha256(changed).hexdigest(), BINDING)


def test_byte_hash_precedes_any_deserialization(saved, monkeypatch):
    raw = saved[0]
    monkeypatch.setattr(torch, 'load', lambda *_a, **_kw: pytest.fail('unexpected load'))
    with pytest.raises(ValueError, match='byte identity'): codec.load_fixture(raw, '0'*64, BINDING)
