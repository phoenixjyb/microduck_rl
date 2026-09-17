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
from mjlab_microduck.stance_attempt_trace import (
    CHECKPOINTS, LEAN_CHECKPOINTS, LEAN_REPLICATION_CHECKPOINTS)

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

# Lean-lesson: a weight-initialized continuation, declared in
# docs/experiments/2026-09-16-stance-lean-lesson.md. The parent export is pinned by
# byte hash and iteration so only the reviewed frozen weights can initialize a run,
# and the resulting identity can never be mistaken for a fresh initializer.
LEAN_SEED, LEAN_WORLDS, LEAN_UPDATES = 571, 64, 256
# ``LEAN_CHECKPOINTS`` is imported from ``stance_attempt_trace``, which owns the
# protocol -> admitted-iteration map, so the evaluable iterations here and the
# iterations a lean trace may bind cannot drift apart.
LEAN_PARENT_SHA256 = '46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93'
LEAN_PARENT_SOURCE = 'c8f6b994a2991e400bf6478b967c30e9b618db6a'
LEAN_PARENT_FILE = 'model_127.pt'
LEAN_PARENT_ITERATION = 127
LEAN_PURPOSE = 'lean-lesson'
# Fresh-seed replication of the lean lesson, declared in
# docs/experiments/2026-09-17-stance-lean-replication.md. Same worlds, budget,
# common checkpoints and gates as the lesson; the learner seed is the single
# changed axis. Three bounded literals, never a range or a predicate: 577, 587
# and 593 were verified unused across src/, tests/ and docs/ before this landed.
LEAN_REPLICATION_SEEDS = (577, 587, 593)
LEAN_REPLICATION_PURPOSE = 'lean-replication'
FRESH_PURPOSES = ('pilot', 'smoke', 'eager-learning')
PURPOSES = FRESH_PURPOSES + (LEAN_PURPOSE, LEAN_REPLICATION_PURPOSE)
# purpose -> (admitted learner seeds, worlds, update budget). The seed slot is a
# tuple so a purpose can declare more than one without the allowlist widening to
# a range: the replication declares exactly its three literals, and every other
# purpose declares exactly one.
SCOPE = {'pilot': ((521,), 512, 512), 'smoke': ((523,), 64, 16),
         'eager-learning': ((563,), 64, 128),
         LEAN_PURPOSE: ((LEAN_SEED,), LEAN_WORLDS, LEAN_UPDATES),
         LEAN_REPLICATION_PURPOSE: (LEAN_REPLICATION_SEEDS, LEAN_WORLDS, LEAN_UPDATES)}
BASE_IDENTITY_KEYS = {'protocol', 'source', 'runtime_sha256', 'training_launch_sha256',
    'purpose', 'training_seed', 'worlds', 'iteration', 'initial_state_sha256', 'architecture'}
PARENT_IDENTITY_KEY = 'parent_checkpoint_sha256'
# Which iterations each evaluation path may admit. Keyed by purpose so that a
# purpose can never borrow another's evaluable iterations: the retained pilot
# evaluation path keeps rejecting eager-learning and lean-lesson exports, and the
# lesson and replication paths keep rejecting each other's.
EVALUABLE = {'pilot': CHECKPOINTS, LEAN_PURPOSE: LEAN_CHECKPOINTS,
             LEAN_REPLICATION_PURPOSE: LEAN_REPLICATION_CHECKPOINTS}
# The two purposes that start from reviewed frozen weights rather than a fresh
# initializer, and so carry the parent-export identity key.
WEIGHT_INITIALIZED_PURPOSES = (LEAN_PURPOSE, LEAN_REPLICATION_PURPOSE)
# Every seed the stock CPU initializer may be built at, composed from the declared
# per-purpose literals so it stays single-sourced. Still a bounded tuple: adding a
# seed means naming it above, never widening this to a range or a predicate.
FRESH_SEEDS = (521, 523, 563, LEAN_SEED) + LEAN_REPLICATION_SEEDS


def runtime_check():
    require(version('rsl-rl-lib') == '5.0.1', 'reviewed RSL version')
    root = distribution('rsl-rl-lib').locate_file('rsl_rl')
    require({name: sha256((root/name).read_bytes()).hexdigest() for name in PINS} == PINS,
            'reviewed RSL model sources')


def fresh_models(seed):
    """Stock CPU initialization, isolated from caller CPU RNG; no optimizer."""
    require(type(seed) is int and seed in FRESH_SEEDS, 'predeclared fresh initialization seed')
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
    require(type(identity) is dict, 'checkpoint identity mapping')
    purpose = identity.get('purpose')
    require(purpose in PURPOSES, 'checkpoint purpose')
    lean = purpose in WEIGHT_INITIALIZED_PURPOSES
    keys = BASE_IDENTITY_KEYS | ({PARENT_IDENTITY_KEY} if lean else set())
    require(set(identity) == keys, 'exact checkpoint identity')
    require(identity['protocol'] == PROTOCOL and identity['architecture'] == ARCHITECTURE, 'declared stance architecture')
    hashed = ('source', 'runtime_sha256', 'training_launch_sha256', 'initial_state_sha256')
    for key in hashed + ((PARENT_IDENTITY_KEY,) if lean else ()):
        size = 40 if key == 'source' else 64
        require(type(identity[key]) is str and re.fullmatch('[0-9a-f]{'+str(size)+'}', identity[key]) is not None,
                'checkpoint hash identity: '+key)
    seeds, worlds, updates = SCOPE[purpose]
    require(type(identity['training_seed']) is int and identity['training_seed'] in seeds
            and type(identity['worlds']) is int and identity['worlds'] == worlds, 'matched seed/worlds')
    require(type(identity['iteration']) is int and -1 <= identity['iteration'] < updates, 'bounded saved iteration')
    if evaluation:
        # `evaluation` names the exact purpose the calling evaluation path admits.
        # `True` is the retained pilot evaluation path. Naming a purpose keeps the
        # lean-lesson checkpoints evaluable by their own path while the pilot path
        # continues to refuse them, and vice versa.
        admitted = 'pilot' if evaluation is True else evaluation
        require(admitted in EVALUABLE and purpose == admitted
                and identity['iteration'] in EVALUABLE[admitted],
                'only declared '+str(admitted)+' checkpoints may be evaluated')
    actor, critic = fresh_models(identity['training_seed'])
    fresh = state_hash(states_of(actor, critic))
    if lean:
        require(identity[PARENT_IDENTITY_KEY] == LEAN_PARENT_SHA256, 'pinned lean-lesson parent export')
        # The whole point of the purpose: a weight-initialized run starts from
        # reviewed frozen weights, so its declared start must never coincide with
        # a fresh initializer.
        require(identity['initial_state_sha256'] != fresh,
                'a weight-initialized start must not equal a fresh initializer')
    else:
        require(fresh == identity['initial_state_sha256'], 'exact fresh initializer identity')
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
    return _load(raw, expected_sha256, expected_identity, evaluation=True)


def load_eager_diagnostic(raw, expected_sha256, expected_identity):
    """Separate initializer/final comparison; does not admit a pilot checkpoint."""
    require(expected_identity['purpose'] == 'eager-learning'
            and type(expected_identity['iteration']) is int
            and expected_identity['iteration'] in (-1, 127), 'only eager initializer/final diagnostic')
    return _load(raw, expected_sha256, expected_identity, evaluation=False)


def load_lean_lesson(raw, expected_sha256, expected_identity):
    """Lean-lesson exports; a weight-initialized continuation, never a resume."""
    require(expected_identity['purpose'] == LEAN_PURPOSE
            and expected_identity['iteration'] in (-1,)+LEAN_CHECKPOINTS,
            'only declared lean-lesson checkpoints')
    actor, info = _load(raw, expected_sha256, expected_identity, evaluation=False)
    return actor, dict(info, weight_initialized=True,
        parent_checkpoint_sha256=LEAN_PARENT_SHA256, optimizer_restored=False,
        simulator_restored=False, replay_restored=False)


def load_lean_evaluation(raw, expected_sha256, expected_identity):
    """Lean-lesson evaluation exports, on their own path.

    Deliberately separate from ``load_evaluation``: the retained pilot evaluation
    path must keep refusing lean-lesson exports, and this path must refuse pilot
    ones, so iteration labels are never aliased across purposes.
    """
    require(expected_identity['purpose'] == LEAN_PURPOSE, 'lean-lesson evaluation export')
    return _load(raw, expected_sha256, expected_identity, evaluation=LEAN_PURPOSE)


def load_lean_replication(raw, expected_sha256, expected_identity):
    """Lean replication exports; a weight-initialized continuation, never a resume."""
    require(expected_identity['purpose'] == LEAN_REPLICATION_PURPOSE
            and expected_identity['iteration'] in (-1,)+LEAN_REPLICATION_CHECKPOINTS,
            'only declared lean replication checkpoints')
    actor, info = _load(raw, expected_sha256, expected_identity, evaluation=False)
    return actor, dict(info, weight_initialized=True,
        parent_checkpoint_sha256=LEAN_PARENT_SHA256, optimizer_restored=False,
        simulator_restored=False, replay_restored=False)


def load_lean_replication_evaluation(raw, expected_sha256, expected_identity):
    """Lean replication evaluation exports, on their own path.

    Deliberately separate from both ``load_evaluation`` and
    ``load_lean_evaluation``. The replication's exports must not be readable
    through the lesson's evaluation path and the lesson's must not be readable
    here, so the two purposes can never alias each other's iteration labels. The
    admitted iterations are the same four, which is a property of the declared
    configurations rather than a shared name.
    """
    require(expected_identity['purpose'] == LEAN_REPLICATION_PURPOSE,
            'lean replication evaluation export')
    return _load(raw, expected_sha256, expected_identity, evaluation=LEAN_REPLICATION_PURPOSE)


def load_lean_parent(raw, expected_sha256, expected_identity):
    """Load the pinned frozen weight export that initializes a lean-lesson run.

    Returns *trainable* models. This is weight initialization, not a resume: no Adam
    state, rollout storage, simulator state or RNG state is restored. The caller must
    keep that distinction explicit in the run identity.
    """
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, 'bounded checkpoint bytes')
    require(expected_sha256 == LEAN_PARENT_SHA256, 'pinned lean-lesson parent export')
    require(sha256(raw).hexdigest() == expected_sha256, 'parent checkpoint byte hash mismatch')
    require(type(expected_identity) is dict and expected_identity.get('purpose') == 'eager-learning'
            and expected_identity.get('iteration') == LEAN_PARENT_ITERATION
            and expected_identity.get('source') == LEAN_PARENT_SOURCE,
            'frozen parent is the reviewed eager final export')
    actor, critic = validate_identity(expected_identity, evaluation=False)
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(set(value) == {'identity', 'states'} and value['identity'] == expected_identity,
            'parent checkpoint metadata mismatch')
    validate_states(value['states'], actor, critic)
    return dict(actor=actor, critic=critic, parent_checkpoint_sha256=expected_sha256,
        parent_state_sha256=state_hash(value['states']), weight_initialized=True,
        optimizer_restored=False, simulator_restored=False, replay_restored=False,
        normalization_restored=False, checkpoint_admitted=False)


def _load(raw, expected_sha256, expected_identity, *, evaluation):
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, 'bounded checkpoint bytes')
    require(sha256(raw).hexdigest() == expected_sha256, 'checkpoint byte hash mismatch')
    actor, critic = validate_identity(expected_identity, evaluation=evaluation)
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
