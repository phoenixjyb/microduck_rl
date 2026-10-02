"""Bind the exact recovery parent to fresh real CPU PPO objects, not a trainer.

No collection, update, reset, resume, export or job method is provided here.
The old initializer/checkpoint/learner allowlists remain unchanged. A future
learner needs separate schedule, transition, finite-gradient and job gates.
"""
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import distribution
import os

import torch
from rsl_rl.storage import RolloutStorage

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_lesson_plan as lesson
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck.stance_ppo import CONFIG, PINS, STEPS, FinitePPO, observations
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1d-recovery-policy-preparation-v1'
WORLDS = (2, 64)  # Small future throughput probe or proposed lesson, not admission.


def prepare_policy(raw, *, seed, worlds):
    """Return the real policy/storage/Adam wiring with no optimizer step.

    New learner seed/world count never relabel the historical parent identity.
    Models are loaded directly, not reconstructed with an old fresh seed.
    """
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CUDA-hidden CPU recovery policy preparation')
    require(type(seed) is int and seed in lesson.TRAINING_SEEDS,
            'only proposed recovery learner seeds')
    require(type(worlds) is int and worlds in WORLDS, 'only bounded recovery preparation worlds')
    before_rng = torch.random.get_rng_state().clone()
    loaded = parent.load_parent(raw)
    actor, critic = loaded['actor'], loaded['critic']
    root = distribution('rsl-rl-lib').locate_file('rsl_rl')
    require({name: sha256((root/name).read_bytes()).hexdigest() for name in PINS} == PINS,
            'unchanged reviewed PPO and storage source bytes')
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        obs = observations(dict(actor=torch.zeros(worlds, 44), critic=torch.zeros(worlds, 50)), worlds)
        storage = RolloutStorage('rl', worlds, STEPS, obs, (10,), device='cpu')
        algorithm = FinitePPO(actor, critic, storage, **CONFIG, device='cpu')
    require(algorithm.actor is actor and algorithm.critic is critic and algorithm.storage is storage,
            'real PPO samples and optimizes the exact restored model objects')
    actor_ids = [id(p) for p in actor.parameters()]
    critic_ids = [id(p) for p in critic.parameters()]
    expected_ids = actor_ids+critic_ids
    optimizer_ids = [id(p) for group in algorithm.optimizer.param_groups for p in group['params']]
    require(actor_ids and critic_ids and not set(actor_ids) & set(critic_ids)
            and len(expected_ids) == len(set(expected_ids))
            and len(optimizer_ids) == len(set(optimizer_ids))
            and set(optimizer_ids) == set(expected_ids), 'exact unique actor/critic optimizer parameter binding')
    require(type(algorithm.optimizer) is torch.optim.Adam and not algorithm.optimizer.state
            and all(group['lr'] == CONFIG['learning_rate'] for group in algorithm.optimizer.param_groups),
            'fresh empty declared Adam without restored moments')
    require(storage.step == 0 and storage.num_envs == worlds
            and storage.num_transitions_per_env == STEPS and storage.device == 'cpu',
            'fresh empty CPU rollout storage')
    state_sha = checkpoint.state_hash(checkpoint.states_of(actor, critic))
    require(state_sha == parent.PARENT_STATE_SHA256 == loaded['receipt']['parent_state_sha256'],
            'actual policy starts at the exact reviewed parent weights')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.requires_grad
                for p in (*actor.parameters(), *critic.parameters())), 'trainable float32 CPU policy groups')
    require(torch.equal(before_rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'preparation preserves caller CPU RNG and leaves CUDA uninitialized')
    rng = torch.Generator(device='cpu').manual_seed(seed).get_state()
    receipt = dict(protocol=PROTOCOL, learner_seed=seed, worlds=worlds,
        parent_preparation=deepcopy(loaded['receipt']), initial_state_sha256=state_sha,
        learner_rng_state_sha256=sha256(rng.numpy().tobytes()).hexdigest(),
        ppo_config=deepcopy(CONFIG), ppo_storage_source_sha256=dict(PINS),
        rollout_steps=STEPS, storage_step=0, optimizer_state_entries=0,
        policy_objects_wired=True, exact_optimizer_parameter_binding=True,
        actor_parameter_tensors=len(actor_ids), critic_parameter_tensors=len(critic_ids),
        global_cpu_rng_unchanged=True, optimizer_steps=0, simulator_created=False,
        learner_rng_connected_to_sampler=False,
        schedule_installed=False, transition_bridge_qualified=False,
        finite_optimizer_step_qualified=False, student_export_available=False,
        training_job_predeclared=False, execution_admitted=False,
        cuda_initialized=False, **baseline.FALSE_FLAGS)
    return dict(actor=actor, critic=critic, algorithm=algorithm, storage=storage,
                learner_rng_state=rng.clone(), receipt=receipt)
