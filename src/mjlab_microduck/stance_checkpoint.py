"""Fresh B1-N models and strict evaluation exports; no optimizer or resume path."""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import distribution, version
import io
import re

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_attempt_trace import CHECKPOINTS

PROTOCOL = 'football-b1n-evaluation-checkpoint-v1'
LIMIT = 8*1024*1024
PINS = {
    'models/mlp_model.py': '6219ebb3ed4df036dae7ff1c30d0fbb169eaaa659e44a659dd757a292124476b',
    'modules/mlp.py': 'bad934b364b26cd47b6d1612e00ace107a425ae4d272c0cd0c82cbfc57bbcdbc',
    'modules/distribution.py': '4631eae1939dcd6b065d79c3c0128a88ba2c6126d08f46944c8795682a208a1b',
}
ARCHITECTURE = dict(actor_dim=44, critic_dim=50, action_dim=10,
    hidden_dims=[128, 128, 64], activation='elu', obs_normalization=False,
    gaussian_std_type='scalar', initial_std=.3, evaluation_output='deterministic-mean')


def runtime_check():
    require(version('rsl-rl-lib') == '5.0.1', 'reviewed RSL version')
    root = distribution('rsl-rl-lib').locate_file('rsl_rl')
    require({name: sha256((root/name).read_bytes()).hexdigest() for name in PINS} == PINS,
            'reviewed RSL model sources')


def fresh_models(seed):
    """Stock CPU initialization, isolated from caller CPU RNG; no optimizer."""
    require(type(seed) is int and seed in (521, 523), 'predeclared fresh initialization seed')
    require(torch.get_default_dtype() == torch.float32, 'declared float32 initialization')
    runtime_check()
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        # CPU export checks must not reseed any existing or later CUDA generator.
        torch.random.default_generator.manual_seed(seed)
        obs = TensorDict({'actor': torch.zeros(1, 44), 'critic': torch.zeros(1, 50)}, [1])
        groups = {'actor': ['actor'], 'critic': ['critic']}
        actor = MLPModel(obs, groups, 'actor', 10, hidden_dims=[128, 128, 64], activation='elu',
            obs_normalization=False, distribution_cfg=dict(
                class_name='rsl_rl.modules.distribution:GaussianDistribution', init_std=.3, std_type='scalar'))
        critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[128, 128, 64], activation='elu',
                          obs_normalization=False)
    return actor, critic


def state_hash(states):
    h = sha256()
    for group in ('actor', 'critic'):
        for name, value in sorted(states[group].items()):
            v = value.detach().cpu().contiguous()
            h.update((canonical([group, name, str(v.dtype), list(v.shape)])+'\n').encode())
            h.update(v.numpy().tobytes())
    return h.hexdigest()


def states_of(actor, critic):
    return {group: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for group, model in (('actor', actor), ('critic', critic))}


def validate_identity(identity, *, evaluation):
    require(set(identity) == {'protocol', 'source', 'runtime_sha256', 'training_launch_sha256',
        'purpose', 'training_seed', 'worlds', 'iteration', 'initial_state_sha256', 'architecture'},
        'exact checkpoint identity')
    require(identity['protocol'] == PROTOCOL and identity['architecture'] == ARCHITECTURE, 'declared stance architecture')
    for key in ('source', 'runtime_sha256', 'training_launch_sha256', 'initial_state_sha256'):
        size = 40 if key == 'source' else 64
        require(type(identity[key]) is str and re.fullmatch('[0-9a-f]{'+str(size)+'}', identity[key]) is not None,
                'checkpoint hash identity: '+key)
    require(identity['purpose'] in ('pilot', 'smoke'), 'checkpoint purpose')
    pilot = identity['purpose'] == 'pilot'
    require(type(identity['training_seed']) is int and identity['training_seed'] == (521 if pilot else 523)
            and type(identity['worlds']) is int and identity['worlds'] == (512 if pilot else 64), 'matched seed/worlds')
    require(type(identity['iteration']) is int and -1 <= identity['iteration'] < (512 if pilot else 16), 'bounded saved iteration')
    if evaluation:
        require(pilot and identity['iteration'] in CHECKPOINTS, 'only declared pilot checkpoints may be evaluated')
    actor, critic = fresh_models(identity['training_seed'])
    require(state_hash(states_of(actor, critic)) == identity['initial_state_sha256'], 'exact fresh initializer identity')
    return actor, critic


def validate_states(states, actor, critic):
    require(set(states) == {'actor', 'critic'}, 'both exact model groups')
    for group, model in (('actor', actor), ('critic', critic)):
        expected = model.state_dict(); values = states[group]
        require(set(values) == set(expected), 'no missing/extra/legacy/normalizer weights')
        for name, ref in expected.items():
            v = values[name]
            require(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.dtype == torch.float32
                    and v.shape == ref.shape and torch.isfinite(v).all(), 'finite exact checkpoint tensor: '+name)
        model.load_state_dict(values, strict=True)
    require((actor.distribution.std_param > 0).all(), 'positive finite Gaussian scale')


def encode(actor, critic, identity):
    """Export evaluation weights; this is deliberately not a resumable PPO save."""
    expected_actor, expected_critic = validate_identity(identity, evaluation=False)
    # Reject changed architecture even when parameter keys happen to match.
    for actual, expected in ((actor, expected_actor), (critic, expected_critic)):
        require(type(actual) is MLPModel and repr(actual) == repr(expected)
                and actual.obs_groups == expected.obs_groups and actual.obs_dim == expected.obs_dim,
                'exact stock stance model layout')
    states = states_of(actor, critic); validate_states(states, expected_actor, expected_critic)
    out = io.BytesIO(); torch.save(dict(identity=deepcopy(identity), states=states), out)
    raw = out.getvalue(); require(len(raw) <= LIMIT, 'bounded checkpoint export')
    return raw


def load_evaluation(raw, expected_sha256, expected_identity):
    """Hash before CPU weights-only loading; build new models, never partial load."""
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, 'bounded checkpoint bytes')
    require(sha256(raw).hexdigest() == expected_sha256, 'checkpoint byte hash mismatch')
    actor, critic = validate_identity(expected_identity, evaluation=True)
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(set(value) == {'identity', 'states'} and value['identity'] == expected_identity, 'checkpoint metadata mismatch')
    validate_states(value['states'], actor, critic)
    actor.eval().requires_grad_(False); critic.eval().requires_grad_(False)
    return actor, dict(protocol=PROTOCOL, checkpoint_sha256=expected_sha256,
        identity=deepcopy(expected_identity), loaded_state_sha256=state_hash(value['states']),
        strict_actor_restore=True, strict_critic_weights_checked=True,
        optimizer_restored=False, normalization_restored=False, checkpoint_admitted=False)


@torch.no_grad()
def infer(actor, observations):
    require(observations.ndim == 2 and observations.shape[1] == 44
            and observations.dtype == torch.float32 and torch.isfinite(observations).all(), 'finite actor inputs')
    result = actor(TensorDict({'actor': observations}, [len(observations)]), stochastic_output=False)
    require(result.shape == (len(observations), 10) and torch.isfinite(result).all(), 'finite deterministic actor output')
    return result.detach().clone()
