"""CPU composition bridge from the exact recovery parent to scheduled PPO.

This is a bounded integration harness, not a CLI, trainer, checkpoint exporter,
or execution admission. It keeps the historical fresh-model allowlists intact.
The synthetic fixture seam is explicitly non-native and cannot be used unless
the caller opts in and the environment labels itself as synthetic.
"""
from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256
import math
import os

import torch
from rsl_rl.storage import RolloutStorage

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_policy_preparation as preparation
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_ppo import CONFIG, finite, observations, timeout_rewards

PROTOCOL = 'football-b1d-scheduled-recovery-ppo-bridge-v1'
HORIZON = 28
UPDATE_LIMIT = 1
MINI_BATCHES = 4
GAMMA = .99
LAMBDA = .95


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if type(left) is dict or type(right) is dict:
        return (type(left) is dict and type(right) is dict and left.keys() == right.keys()
                and all(_tree_equal(left[k], right[k]) for k in left))
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _finite_parameters(actor, critic):
    for group, model in (('actor', actor), ('critic', critic)):
        for name, parameter in model.named_parameters():
            require(parameter.device.type == 'cpu' and parameter.dtype == torch.float32
                    and parameter.requires_grad and torch.isfinite(parameter).all(),
                    'finite trainable CPU '+group+' parameter '+name)


class RecoveryFinitePPO(preparation.FinitePPO):
    """The reviewed finite GAE with its horizon bound to recovery storage."""

    @torch.no_grad()
    def compute_returns(self, obs):
        st = self.storage
        horizon = st.num_transitions_per_env
        require(horizon == HORIZON and st.step == horizon,
                'complete recovery rollout required for GAE')
        for name in ('rewards', 'values'):
            finite(getattr(st, name), (horizon, st.num_envs, 1), name)
        require(((st.dones == 0) | (st.dones == 1)).all(), 'binary recovery rollout terminals')
        last = self.critic(obs).detach()
        finite(last, (st.num_envs, 1), 'last recovery critic value')
        advantage = torch.zeros_like(last)
        for step in reversed(range(horizon)):
            next_value = last if step == horizon-1 else st.values[step+1]
            continuation = 1.-st.dones[step].float()
            delta = st.rewards[step]+self.gamma*continuation*next_value-st.values[step]
            advantage = delta+self.gamma*self.lam*continuation*advantage
            st.returns[step] = advantage+st.values[step]
        raw = st.returns-st.values
        finite(raw, (horizon, st.num_envs, 1), 'recovery raw advantages')
        st.advantages.copy_((raw-raw.mean())/(raw.std()+1e-8))
        finite(st.advantages, raw.shape, 'normalized recovery advantages')


class RecoveryPPOBridge:
    """One 28-step CPU PPO composition over fixed scheduled runtime rows.

    ``env`` must be the real ScheduledRecoveryRuntime unless the explicitly
    synthetic fixture seam is requested. Every failure faults the composition;
    it has no retry, resume, export, or job-admission path.
    """

    def __init__(self, raw_parent, env, declaration, *, seed, synthetic_fixture=False):
        require(type(synthetic_fixture) is bool, 'explicit synthetic fixture selector')
        require(torch.cuda.is_initialized() is False and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                'CUDA-hidden CPU recovery bridge')
        bound = schedule.checked(declaration)
        require(bound['split'] == 'training', 'held-out cells cannot enter a recovery learner')
        require(type(seed) is int and seed in (653, 659), 'only proposed recovery learner seeds')
        require(bound['worlds'] in (2, 64), 'only bounded recovery bridge worlds')
        require(type(env.n) is int and env.n == bound['worlds']
                and torch.device(env.device).type == 'cpu', 'matching fresh CPU scheduled runtime')
        require(callable(getattr(env, 'step_with_schedule', None))
                and callable(getattr(env, 'observations', None)) and callable(getattr(env, 'reset', None))
                and callable(getattr(env, '_control_snapshot', None)),
                'scheduled runtime transition API')
        require(type(getattr(env, 'schedule_declaration', None)) is dict
                and schedule.checked(env.schedule_declaration) == bound,
                'runtime owns the exact immutable row schedule')
        if synthetic_fixture:
            require(getattr(env, 'synthetic_ppo_fixture', False) is True,
                    'synthetic seam requires an explicitly labeled fake runtime')
        else:
            from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
            require(type(env) is ScheduledRecoveryRuntime, 'native bridge requires exact scheduled runtime type')
        require(torch.is_tensor(env.live) and env.live.dtype == torch.bool
                and env.live.shape == (bound['worlds'],) and env.live.all(),
                'fresh all-live scheduled episodes')
        require(torch.is_tensor(env.steps) and env.steps.dtype == torch.long
                and env.steps.shape == (bound['worlds'],) and not env.steps.any(),
                'fresh per-row episode clocks')

        prepared = preparation.prepare_policy(raw_parent, seed=seed, worlds=bound['worlds'])
        receipt = prepared['receipt']
        require(receipt['protocol'] == preparation.PROTOCOL
                and receipt['learner_seed'] == seed and receipt['worlds'] == bound['worlds']
                and receipt['initial_state_sha256'] == preparation.parent.PARENT_STATE_SHA256
                and receipt['parent_preparation']['parent_checkpoint_sha256'] == baseline.CHECKPOINT_SHA256
                and receipt['parent_preparation']['strict_actor_restore'] is True
                and receipt['parent_preparation']['strict_critic_restore'] is True
                and receipt['parent_preparation']['execution_admitted'] is False
                and receipt['learner_rng_connected_to_sampler'] is False
                and torch.is_tensor(prepared['learner_rng_state'])
                and prepared['learner_rng_state'].dtype == torch.uint8
                and prepared['learner_rng_state'].device.type == 'cpu'
                and sha256(prepared['learner_rng_state'].numpy().tobytes()).hexdigest()
                    == receipt['learner_rng_state_sha256']
                and receipt['optimizer_steps'] == 0 and receipt['storage_step'] == 0
                and receipt['optimizer_state_entries'] == 0,
                'exact prepared D1 parent and proposed learner identity')
        self.actor, self.critic = prepared['actor'], prepared['critic']
        self.state_schema = {group: {name: (tuple(value.shape), value.dtype)
            for name, value in model.state_dict().items()}
            for group, model in (('actor', self.actor), ('critic', self.critic))}
        self.initial_state_sha256 = receipt['initial_state_sha256']
        self.parent_receipt = deepcopy(receipt['parent_preparation'])
        self.learner_seed = seed
        self.worlds = bound['worlds']
        self.schedule = bound
        self.schedule_sha256 = schedule.binding_sha256(bound)
        self.env = env
        self.synthetic_fixture = synthetic_fixture
        self.private_rng_state = prepared['learner_rng_state'].detach().clone()
        self.initial_controls = deepcopy(env._control_snapshot())

        before_state = checkpoint.state_hash(checkpoint.states_of(self.actor, self.critic))
        require(before_state == self.initial_state_sha256, 'prepared parent weights before recovery PPO wiring')
        before_rng = torch.random.get_rng_state().clone()
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            obs = observations(dict(actor=torch.zeros(self.worlds, 44),
                                    critic=torch.zeros(self.worlds, 50)), self.worlds)
            self.storage = RolloutStorage('rl', self.worlds, HORIZON, obs, (10,), device='cpu')
            self.algorithm = RecoveryFinitePPO(self.actor, self.critic, self.storage,
                                               **CONFIG, device='cpu')
        require(torch.equal(before_rng, torch.random.get_rng_state()),
                'recovery storage and PPO wiring preserve caller CPU RNG')
        require(self.algorithm.actor is self.actor and self.algorithm.critic is self.critic
                and self.algorithm.storage is self.storage, 'same exact restored models in recovery PPO')
        self._check_optimizer_binding(empty=True)
        require(self.storage.step == 0 and self.storage.num_envs == self.worlds
                and self.storage.num_transitions_per_env == HORIZON,
                'fresh 28-step recovery storage')
        require(checkpoint.state_hash(checkpoint.states_of(self.actor, self.critic)) == before_state,
                'new recovery optimizer leaves parent weights unchanged')

        self.phase = 'empty'
        self.faulted = False
        self.completed_updates = 0
        self.optimizer_steps = 0
        self.terminal_events = []
        self.last_receipt = None
        self.algorithm.optimizer.register_step_pre_hook(self._gradient_gate)
        self.algorithm.optimizer.register_step_post_hook(self._moment_gate)

    @classmethod
    def from_parent(cls, raw_parent, env, declaration, *, seed, synthetic_fixture=False):
        """Load only through the exact-byte parent preparation path."""
        return cls(raw_parent, env, declaration, seed=seed, synthetic_fixture=synthetic_fixture)

    def _check_optimizer_binding(self, *, empty=False):
        require(type(self.algorithm.optimizer) is torch.optim.Adam
                and len(self.algorithm.optimizer.param_groups) == 1,
                'single fresh recovery Adam optimizer')
        group = self.algorithm.optimizer.param_groups[0]
        actor_ids = [id(p) for p in self.actor.parameters()]
        critic_ids = [id(p) for p in self.critic.parameters()]
        expected = actor_ids+critic_ids
        ids = [id(p) for p in group['params']]
        require(actor_ids and critic_ids and len(expected) == len(set(expected))
                and len(ids) == len(set(ids)) and set(ids) == set(expected),
                'unique recovery optimizer parameter union')
        require(group['lr'] == CONFIG['learning_rate'] and group['betas'] == (.9, .999)
                and group['eps'] == 1e-8 and group['weight_decay'] == 0,
                'fixed recovery Adam configuration')
        require(all(getattr(self.algorithm, key) == value
                    for key, value in CONFIG.items() if key != 'optimizer')
                and self.algorithm.num_mini_batches == MINI_BATCHES
                and self.algorithm.gamma == GAMMA and self.algorithm.lam == LAMBDA,
                'unchanged reviewed recovery PPO configuration')
        if empty:
            require(not self.algorithm.optimizer.state, 'fresh recovery Adam has no restored moments')
        else:
            for state in self.algorithm.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        require(torch.isfinite(value).all(), 'nonfinite recovery Adam state')
                    elif type(value) in (int, float):
                        require(math.isfinite(value), 'nonfinite recovery Adam scalar state')

    def _healthy(self):
        require(not self.faulted and self.phase != 'faulted', 'faulted recovery bridge requires closeout')
        try:
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
                    'recovery PPO bridge remains CPU-only')
            self._check_optimizer_binding(empty=self.completed_updates == 0)
            _finite_parameters(self.actor, self.critic)
            self._state_gate()
            if self.completed_updates:
                expected = {p for group in self.algorithm.optimizer.param_groups for p in group['params']}
                require(set(self.algorithm.optimizer.state) == expected,
                        'complete recovery Adam state after the declared update')
            require(schedule.checked(self.env.schedule_declaration) == self.schedule
                    and schedule.binding_sha256(self.env.schedule_declaration) == self.schedule_sha256,
                    'fixed recovery row schedule unchanged')
            if not self.synthetic_fixture:
                require(getattr(self.env, '_schedule_sha256', None) == self.schedule_sha256,
                        'native runtime schedule hash unchanged')
            if self.phase == 'empty':
                require(self.storage.step == 0, 'empty recovery phase/storage agreement')
            elif self.phase == 'collecting':
                require(0 < self.storage.step < HORIZON, 'collecting recovery phase/storage agreement')
            elif self.phase == 'full':
                require(self.storage.step == HORIZON, 'full recovery phase/storage agreement')
            else:
                raise ValueError('unknown recovery bridge phase')
        except Exception:
            self.faulted = True
            self.phase = 'faulted'
            raise

    @contextmanager
    def _rng_scope(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(self.private_rng_state)
            try:
                yield
            finally:
                self.private_rng_state = torch.random.get_rng_state().clone()

    def _state_gate(self):
        for group, model in (('actor', self.actor), ('critic', self.critic)):
            actual = model.state_dict()
            require(set(actual) == set(self.state_schema[group]), 'unchanged parent model state keys')
            for name, value in actual.items():
                require((tuple(value.shape), value.dtype) == self.state_schema[group][name]
                        and value.device.type == 'cpu' and torch.isfinite(value).all(),
                        'unchanged finite parent state schema '+group+'/'+name)
        require((self.actor.distribution.std_param > 0).all(), 'positive recovery Gaussian scale')

    def _gradient_gate(self, optimizer, args, kwargs):
        self._check_optimizer_binding(empty=False)
        _finite_parameters(self.actor, self.critic)
        self._state_gate()
        for group in optimizer.param_groups:
            for parameter in group['params']:
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                        'recovery optimizer requires finite complete gradients')

    def _moment_gate(self, optimizer, args, kwargs):
        _finite_parameters(self.actor, self.critic)
        self._state_gate()
        require(optimizer.state, 'recovery Adam step must initialize moments')
        for group in optimizer.param_groups:
            for parameter in group['params']:
                require(parameter in optimizer.state, 'missing recovery Adam parameter state')
                for value in optimizer.state[parameter].values():
                    if isinstance(value, torch.Tensor):
                        require(torch.isfinite(value).all(), 'nonfinite recovery Adam state')
                    elif type(value) in (int, float):
                        require(math.isfinite(value), 'nonfinite recovery Adam scalar state')
        self.optimizer_steps += 1

    @torch.no_grad()
    def collect_one(self, *, capture_control=False):
        self._healthy()
        try:
            require(type(capture_control) is bool and self.completed_updates < UPDATE_LIMIT
                    and self.phase in ('empty', 'collecting') and self.storage.step < HORIZON,
                    'recovery bridge collecting phase')
            require(self.env.live.all(), 'reset completed rows before next policy action')
            step = self.storage.step
            obs = observations(self.env.observations(), self.worlds)
            with self._rng_scope():
                sampled = self.algorithm.act(obs)
            finite(sampled, (self.worlds, 10), 'recovery raw sampled actions')
            for name in ('values', 'actions_log_prob'):
                require(torch.isfinite(getattr(self.algorithm.transition, name)).all(),
                        'finite recovery policy '+name)
            require(torch.equal(self.algorithm.transition.actions, sampled),
                    'PPO transition retains raw sampled actions')

            result = self.env.step_with_schedule(sampled, capture_control=capture_control)
            require(type(result) is dict and type(result.get('terminal_records')) is list
                    and len(result['terminal_records']) == self.worlds,
                    'pre-reset scheduled transition and terminal evidence')
            terminal_records = deepcopy(result['terminal_records'])
            terminal_obs = observations(result['observation'], self.worlds)
            terminal_values = self.critic(terminal_obs).detach()
            finite(terminal_values, (self.worlds, 1), 'recovery terminal critic values')
            terminated, timed_out = result['terminated'], result['timed_out']
            for name, flag in (('terminated', terminated), ('timed_out', timed_out)):
                require(torch.is_tensor(flag) and flag.shape == (self.worlds,)
                        and flag.dtype == torch.bool and flag.device.type == 'cpu',
                        'typed scheduled '+name+' mask')
            done = terminated | timed_out
            reward = timeout_rewards(result['reward'], terminated, timed_out, terminal_values)
            require(torch.equal(result['live'], ~done), 'scheduled live mask equals terminal complement')
            require(torch.equal(result['episode_steps'], self.env.steps),
                    'scheduled per-row episode clock binding')
            require(self.algorithm.gamma == GAMMA, 'timeout bootstrap matches fixed PPO gamma')

            for i, record in enumerate(terminal_records):
                require((type(record) is dict) if bool(done[i]) else record is None,
                        'exact terminal-record presence for every completed row')
                if bool(done[i]):
                    self.terminal_events.append(dict(storage_step=step, row=i,
                                                     record=deepcopy(record)))

            pre_reset = dict(steps=self.env.steps.clone(), live=self.env.live.clone(),
                             controls=deepcopy(self.env._control_snapshot()))
            if not self.synthetic_fixture:
                pre_reset['kinematics'] = {name: self.env._view(name).detach().clone()
                                          for name in ('qpos', 'qvel', 'time')}
            self.algorithm.process_env_step(terminal_obs, reward, done, {})
            require(torch.equal(self.storage.actions[step], sampled),
                    'rollout storage keeps unclipped sampled policy action')
            require(torch.equal(self.storage.observations['actor'][step], obs['actor'])
                    and torch.equal(self.storage.observations['critic'][step], obs['critic']),
                    'rollout storage keeps pre-action observations')

            reset_records = [None]*self.worlds
            if done.any():
                reset_records = self.env.reset(done)
                expected = [terminal_records[i] if bool(done[i]) else None for i in range(self.worlds)]
                require(_tree_equal(reset_records, expected),
                        'selective reset returns exact pre-reset terminal records')
                after_controls = self.env._control_snapshot()
                untouched = ~done
                require(torch.equal(self.env.steps[untouched], pre_reset['steps'][untouched])
                        and torch.equal(self.env.live[untouched], pre_reset['live'][untouched]),
                        'selective reset preserves untouched row clocks/live masks')
                if not self.synthetic_fixture:
                    for name, before in pre_reset['kinematics'].items():
                        require(torch.equal(before[untouched], self.env._view(name)[untouched]),
                                'selective reset preserves untouched physical '+name)
                for name, before in pre_reset['controls'].items():
                    after = after_controls[name]
                    if isinstance(before, torch.Tensor) and before.ndim > 0 and before.shape[0] == self.worlds:
                        require(torch.equal(before[untouched], after[untouched]),
                                'selective reset preserves untouched control '+name)
                        initial = self.initial_controls[name]
                        require(torch.equal(initial[done], after[done]),
                                'selective reset restores episode control '+name)
                    else:
                        require(_tree_equal(before, after), 'selective reset preserves control '+name)
                require(schedule.checked(self.env.schedule_declaration) == self.schedule
                        and torch.equal(self.env.steps[done], torch.zeros_like(self.env.steps[done]))
                        and self.env.live[done].all(),
                        'fixed row schedule restarts only with reset episode clock')
            self.phase = 'full' if self.storage.step == HORIZON else 'collecting'
            receipt = dict(storage_step=step, raw_actions=sampled.detach().clone(),
                reward=reward.detach().clone(), terminated=terminated.clone(), timed_out=timed_out.clone(),
                terminal_records=terminal_records, reset_records=deepcopy(reset_records),
                scheduled_runtime=not self.synthetic_fixture, **baseline.FALSE_FLAGS)
            self.last_receipt = receipt
            result['bridge_receipt'] = receipt
            return result
        except Exception:
            self.faulted = True
            self.phase = 'faulted'
            raise

    def update(self):
        self._healthy()
        try:
            require(self.completed_updates < UPDATE_LIMIT and self.phase == 'full'
                    and self.storage.step == HORIZON,
                    'complete recovery rollout before its single update')
            next_obs = observations(self.env.observations(), self.worlds)
            self.algorithm.compute_returns(next_obs)
            for name in ('returns', 'advantages'):
                require(torch.isfinite(getattr(self.storage, name)).all(),
                        'finite recovery '+name)
            before_steps = self.optimizer_steps
            with self._rng_scope():
                metrics = self.algorithm.update()
            require(type(metrics) is dict and metrics
                    and all(type(v) in (float, int) and math.isfinite(v) for v in metrics.values()),
                    'finite recovery PPO metrics')
            require(self.optimizer_steps-before_steps == CONFIG['num_learning_epochs']*MINI_BATCHES,
                    'all declared finite recovery optimizer steps completed')
            require(self.storage.step == 0, 'RSL clears full recovery storage after update')
            _finite_parameters(self.actor, self.critic)
            self._state_gate()
            self._check_optimizer_binding(empty=False)
            self.completed_updates += 1
            self.phase = 'empty'
            return dict(protocol=PROTOCOL, metrics=metrics, completed_updates=self.completed_updates,
                optimizer_steps=self.optimizer_steps,
                parent_initial_state_sha256=self.initial_state_sha256,
                updated_state_sha256=checkpoint.state_hash(checkpoint.states_of(self.actor, self.critic)),
                storage_step=self.storage.step, execution_admitted=False,
                student_export_available=False, **baseline.FALSE_FLAGS)
        except Exception:
            self.faulted = True
            self.phase = 'faulted'
            raise
