"""Synthetic evaluation exports only: no PPO updates or accepted policy."""
from copy import deepcopy
from hashlib import sha256
import io
import pytest
import torch
from mjlab_microduck import stance_checkpoint as cp


def identity(purpose='pilot', iteration=128):
    seed = 521 if purpose == 'pilot' else 523
    a, c = cp.fresh_models(seed)
    return dict(protocol=cp.PROTOCOL, source='a'*40, runtime_sha256='b'*64,
        training_launch_sha256='c'*64, purpose=purpose, training_seed=seed,
        worlds=512 if purpose == 'pilot' else 64, iteration=iteration,
        initial_state_sha256=cp.state_hash(cp.states_of(a, c)), architecture=deepcopy(cp.ARCHITECTURE))


@pytest.fixture
def exported():
    a, c = cp.fresh_models(521); meta = identity()
    return cp.encode(a, c, meta), meta


def test_fresh_models_declared_layout_and_rng_isolation(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('CPU initialization must not seed CUDA')
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    before = torch.random.get_rng_state().clone()
    a, c = cp.fresh_models(521)
    assert torch.equal(before, torch.random.get_rng_state())
    assert a.obs_dim == 44 and c.obs_dim == 50
    assert isinstance(a.obs_normalizer, torch.nn.Identity)
    assert c.distribution is None
    assert torch.equal(a.distribution.std_param, torch.full((10,), .3))
    assert cp.state_hash(cp.states_of(a, c)) == '27e98087a3043f8a4b4fdc9b81958260db849cd4dbfec5dc5cb6a8810e0b7e8b'
    assert not torch.cuda.is_initialized()


def test_exact_strict_restore_inference_and_no_resume(exported):
    raw, meta = exported
    actor, receipt = cp.load_evaluation(raw, sha256(raw).hexdigest(), meta)
    original, _ = cp.fresh_models(521)
    obs = torch.arange(88, dtype=torch.float32).reshape(2, 44)/100
    assert torch.equal(cp.infer(actor, obs), cp.infer(original, obs))
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())
    assert receipt['strict_actor_restore'] and receipt['strict_critic_weights_checked']
    assert not receipt['optimizer_restored'] and not receipt['normalization_restored'] and not receipt['checkpoint_admitted']


@pytest.mark.parametrize('damage', ['missing', 'extra', 'legacy_shape', 'nan', 'dtype', 'std_zero', 'critic', 'metadata', 'optimizer'])
def test_reject_malformed_models_without_partial_restore(exported, damage):
    raw, meta = exported
    value = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    actor = value['states']['actor']
    if damage == 'missing': actor.pop('mlp.0.bias')
    elif damage == 'extra': actor['obs_normalizer.mean'] = torch.zeros(44)
    elif damage == 'legacy_shape': actor['mlp.0.weight'] = torch.zeros(128, 61)
    elif damage == 'nan': actor['mlp.0.bias'][0] = float('nan')
    elif damage == 'dtype': actor['mlp.0.bias'] = actor['mlp.0.bias'].double()
    elif damage == 'std_zero': actor['distribution.std_param'][0] = 0
    elif damage == 'critic': value['states']['critic']['mlp.0.weight'] = torch.zeros(128, 44)
    elif damage == 'metadata': value['identity']['iteration'] = 256
    elif damage == 'optimizer': value['optimizer'] = {}
    output = io.BytesIO(); torch.save(value, output); altered = output.getvalue()
    with pytest.raises(ValueError): cp.load_evaluation(altered, sha256(altered).hexdigest(), meta)


@pytest.mark.parametrize('key,value', [('source', 'x'*40), ('iteration', 500),
    ('training_seed', 523), ('worlds', 64), ('initial_state_sha256', 'e'*64)])
def test_wrong_experiment_identity_refused(exported, key, value):
    raw, meta = exported; meta = deepcopy(meta); meta[key] = value
    with pytest.raises(ValueError): cp.load_evaluation(raw, sha256(raw).hexdigest(), meta)


def test_smoke_cannot_be_promoted_and_initial_checkpoint_is_not_evaluation():
    a, c = cp.fresh_models(523); meta = identity('smoke', 15)
    raw = cp.encode(a, c, meta)
    with pytest.raises(ValueError, match='only declared pilot'): cp.load_evaluation(raw, sha256(raw).hexdigest(), meta)
    a, c = cp.fresh_models(521); meta = identity(iteration=-1); raw = cp.encode(a, c, meta)
    with pytest.raises(ValueError, match='only declared pilot'): cp.load_evaluation(raw, sha256(raw).hexdigest(), meta)


def test_hash_and_size_before_deserialization(exported, monkeypatch):
    raw, meta = exported
    def forbidden(*args, **kwargs): pytest.fail('must not deserialize')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='hash mismatch'): cp.load_evaluation(raw, '0'*64, meta)
    monkeypatch.setattr(cp, 'LIMIT', 1)
    with pytest.raises(ValueError, match='bounded'): cp.load_evaluation(raw, sha256(raw).hexdigest(), meta)


def test_same_keys_with_changed_activation_cannot_export():
    a, c = cp.fresh_models(521); a.mlp[1] = torch.nn.ReLU()
    with pytest.raises(ValueError, match='layout'): cp.encode(a, c, identity())


def test_bad_observation_or_output_refused():
    a, _ = cp.fresh_models(521)
    with pytest.raises(ValueError): cp.infer(a, torch.zeros(2, 61))
    with pytest.raises(ValueError): cp.infer(a, torch.full((2, 44), float('nan')))
    with torch.no_grad(): a.mlp[6].bias.fill_(float('inf'))
    with pytest.raises(ValueError, match='actor output'): cp.infer(a, torch.zeros(2, 44))


def test_fresh_models_force_cpu_without_changing_global_default_device():
    with torch.device('meta'):
        a, c = cp.fresh_models(521)
        assert torch.empty(1).device.type == 'meta'
    assert all(p.device.type == 'cpu' for m in (a, c) for p in m.parameters())


def test_non_float32_global_default_is_refused(monkeypatch):
    monkeypatch.setattr(torch, 'get_default_dtype', lambda: torch.float64)
    with pytest.raises(ValueError, match='float32'): cp.fresh_models(521)
