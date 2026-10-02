"""Hash-pinned CPU preparation of both D1 parent model groups, never a resume.

This does not install weights into a learner, create an optimizer or admit a
job. It is deliberately separate from the historical model-127 parent loader.
The caller still needs archive authentication and an independently qualified
new learner, schedule, checkpoint and evidence namespace.
"""
from copy import deepcopy
from hashlib import sha256
import io
import os

import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1d-parent-preparation-v1'
PARENT_STATE_SHA256 = 'e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329'
ARCHITECTURE = dict(actor_dim=44, critic_dim=50, action_dim=10,
    hidden_dims=[128, 128, 64], activation='elu', obs_normalization=False,
    gaussian_std_type='scalar', initial_std=.3, evaluation_output='deterministic-mean')


def expected_identity():
    """A fresh copy of the exact previously selected export identity."""
    return dict(protocol='football-b1n-evaluation-checkpoint-v1',
        source=baseline.TRAINING_SOURCE, purpose='lean-replication',
        training_seed=baseline.TRAINING_SEED, worlds=64, iteration=baseline.ITERATION,
        runtime_sha256='ef01ea829b5b69069cf732b6e252ee0e7c205e92e5f17f028753ab9ed08682cb',
        training_launch_sha256='8236806aa44c1d424b77e74942d1e01b6530702a55216186165856a870f48455',
        initial_state_sha256='0ca246873f1143c540ecdebb6e6bc80cb826f86b558dda9c3cff8c5694263144',
        parent_checkpoint_sha256='46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93',
        architecture=deepcopy(ARCHITECTURE))


def load_parent(raw):
    """Return new trainable CPU actor/critic from one exact immutable export.

    No caller-supplied identity/hash or late profile switch is accepted. Hash
    validation precedes deserialization and model construction. Both model
    groups are strictly checked; no existing policy is partially mutated.
    """
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CUDA-hidden uninitialized CPU parent preparation')
    require(type(raw) is bytes and 0 < len(raw) <= checkpoint.LIMIT, 'bounded parent export bytes')
    require(sha256(raw).hexdigest() == baseline.CHECKPOINT_SHA256, 'exact D1 parent byte hash')
    cpu_math = profile.checked_receipt()  # Inspect only; never set environment/math state.
    require(checkpoint.ARCHITECTURE == ARCHITECTURE, 'unchanged parent model architecture')
    identity = expected_identity()
    actor, critic = checkpoint.validate_identity(identity, evaluation='lean-replication')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(type(value) is dict and set(value) == {'identity', 'states'}
            and type(value['identity']) is dict
            and canonical(value['identity']) == canonical(identity), 'exact parent export metadata')
    require(type(value['states']) is dict and set(value['states']) == {'actor', 'critic'}
            and all(type(value['states'][group]) is dict for group in ('actor', 'critic')),
            'both typed parent model groups')
    checkpoint.validate_states(value['states'], actor, critic)
    state_sha = checkpoint.state_hash(checkpoint.states_of(actor, critic))
    require(state_sha == PARENT_STATE_SHA256, 'exact restored D1 actor and critic state')
    for model in (actor, critic):
        model.train().requires_grad_(True)
        require(all(p.device.type == 'cpu' and p.dtype == torch.float32
                    and p.requires_grad for p in model.parameters()), 'trainable CPU parent parameters')
    receipt = dict(protocol=PROTOCOL, parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        parent_identity=identity, parent_state_sha256=state_sha, cpu_math_profile=cpu_math,
        strict_actor_restore=True, strict_critic_restore=True, models_trainable=True,
        weights_only=True, fresh_optimizer_required=True, optimizer_created=False,
        optimizer_restored=False, simulator_restored=False, storage_restored=False,
        rng_restored=False, normalization_restored=False, historical_archive_authenticated_by_loader=False,
        installed_in_learner=False, execution_admitted=False, **baseline.FALSE_FLAGS)
    return dict(actor=actor, critic=critic, receipt=receipt)
