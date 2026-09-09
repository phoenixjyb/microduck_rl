"""Bounded CPU PPO adapter checks; no GPU/pilot CLI or deployment admission."""
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import distribution
import math

import mjlab  # Complete plugin discovery before taking the RSL subclass.
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck.first_attempt_smoke import require

STEPS = 24
CONFIG = dict(num_learning_epochs=5, num_mini_batches=4, clip_param=.2,
    gamma=.99, lam=.95, value_loss_coef=1., entropy_coef=.005,
    learning_rate=3e-4, max_grad_norm=1., optimizer='adam',
    use_clipped_value_loss=True, schedule='fixed', desired_kl=None,
    normalize_advantage_per_mini_batch=False)
PINS = {'algorithms/ppo.py': 'a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c',
        'storage/rollout_storage.py': 'ececf28e7412d24867066829cc443acae811f034f8e63975b90343947779918d'}


def finite(value, shape, label, *, dtype=torch.float32):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.shape == shape and value.dtype == dtype, 'CPU PPO layout: '+label)
    require(not value.is_floating_point() or torch.isfinite(value).all(), 'nonfinite PPO: '+label)


def observations(value, n):
    require(set(value) == {'actor', 'critic'}, 'exact PPO observation groups')
    for key, width in (('actor', 44), ('critic', 50)): finite(value[key], (n, width), key)
    require(torch.equal(value['actor'], value['critic'][:, :44]), 'PPO actor/critic prefix')
    return TensorDict({k: v.detach().clone() for k, v in value.items()}, [n], device='cpu')


def timeout_rewards(reward, terminated, timed_out, terminal_values):
    """Value bootstrap is a learner target, never an environment reward bonus."""
    n = len(reward)
    finite(reward, (n,), 'reward'); finite(terminal_values, (n, 1), 'terminal critic value')
    for k, v in (('terminated', terminated), ('timed_out', timed_out)): finite(v, (n,), k, dtype=torch.bool)
    require(not (terminated & timed_out).any(), 'exclusive failure/timeout flags')
    return reward + .99*torch.where(timed_out, terminal_values[:, 0], 0.)


class FinitePPO(PPO):
    """Local GAE override: do not inherit the historical nan_to_num patch."""

    @torch.no_grad()
    def compute_returns(self, obs):
        st = self.storage
        require(st.step == STEPS, 'complete rollout required for GAE')
        for name in ('rewards', 'values'):
            finite(getattr(st, name), (STEPS, st.num_envs, 1), name)
        require(((st.dones == 0) | (st.dones == 1)).all(), 'binary stored terminals')
        last = self.critic(obs).detach(); finite(last, (st.num_envs, 1), 'last critic value')
        advantage = torch.zeros_like(last)
        for step in reversed(range(STEPS)):
            next_value = last if step == STEPS-1 else st.values[step+1]
            continuation = 1.-st.dones[step].float()
            delta = st.rewards[step]+self.gamma*continuation*next_value-st.values[step]
            advantage = delta+self.gamma*self.lam*continuation*advantage
            st.returns[step] = advantage+st.values[step]
        raw = st.returns-st.values
        finite(raw, (STEPS, st.num_envs, 1), 'raw advantages')
        st.advantages.copy_((raw-raw.mean())/(raw.std()+1e-8))
        finite(st.advantages, raw.shape, 'normalized advantages')


class CpuStanceLearner:
    """Two-to-eight-world CPU integration harness, not a production trainer.

    The caller owns/reset-drives its environment and retains terminal evidence.
    This wrapper stores a transition before selectively resetting done rows.
    Learner RNG is isolated; no Python/NumPy/CUDA RNG or simulator state is saved.
    """

    UPDATE_LIMIT = 2

    def __init__(self, n=2):
        require(type(n) is int and 2 <= n <= 8, 'bounded CPU fixture worlds')
        self._initialize(n)

    def _initialize(self, n, *, seed=523):
        """Shared CPU optimizer implementation; subclasses declare their bounds."""
        checkpoint.runtime_check()
        root = distribution('rsl-rl-lib').locate_file('rsl_rl')
        require({k: sha256((root/k).read_bytes()).hexdigest() for k in PINS} == PINS, 'reviewed PPO/storage sources')
        self.n = n; self.seed = seed; self.faulted = False; self.updates = 0; self.phase = 'empty'
        self.restored_fixture_only = False
        self.actor, self.critic = checkpoint.fresh_models(self.seed)
        self.initial_hash = checkpoint.state_hash(checkpoint.states_of(self.actor, self.critic))
        self.rng = torch.Generator(device='cpu').manual_seed(self.seed).get_state()
        obs = observations(dict(actor=torch.zeros(n, 44, device='cpu'), critic=torch.zeros(n, 50, device='cpu')), n)
        self.storage = RolloutStorage('rl', n, STEPS, obs, (10,), device='cpu')
        self.algorithm = FinitePPO(self.actor, self.critic, self.storage, **CONFIG, device='cpu')
        self.algorithm.optimizer.register_step_pre_hook(self._gradient_gate)
        self.algorithm.optimizer.register_step_post_hook(self._moment_gate)

    def _healthy(self):
        require(not self.faulted, 'faulted PPO requires closeout')
        try:
            require(all(getattr(self.algorithm, k) == v for k, v in CONFIG.items() if k != 'optimizer'),
                    'declared PPO configuration changed')
            require(type(self.algorithm.optimizer) is torch.optim.Adam and all(
                group['lr'] == CONFIG['learning_rate'] for group in self.algorithm.optimizer.param_groups),
                'declared Adam optimizer')
        except Exception:
            self.faulted = True
            raise

    def _gradient_gate(self, optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), 'nonfinite/missing optimizer gradient')

    def _moment_gate(self, optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                require(torch.isfinite(parameter).all(), 'nonfinite updated parameter')
                for value in optimizer.state[parameter].values():
                    require(not isinstance(value, torch.Tensor) or torch.isfinite(value).all(), 'nonfinite Adam state')

    @contextmanager
    def _rng_scope(self):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng)
            try: yield
            finally: self.rng = torch.get_rng_state().clone()

    @torch.no_grad()
    def collect_one(self, env):
        """Return the pre-reset result so callers can retain terminal diagnostics."""
        self._healthy()
        try:
            require(self.updates < self.UPDATE_LIMIT and self.phase in ('empty', 'collecting') and self.storage.step < STEPS, 'collecting PPO phase')
            require(env.n == self.n and str(env.device) == 'cpu' and env.live.all(), 'live matching CPU environment')
            require(not self.restored_fixture_only or getattr(env, 'synthetic_ppo_fixture', False) is True,
                    'restored learner cannot resume simulator; synthetic fixture only')
            obs = observations(env.observations(), self.n)
            with self._rng_scope(): action = self.algorithm.act(obs)
            finite(action, (self.n, 10), 'sampled action')
            for name in ('values', 'actions_log_prob'):
                value = getattr(self.algorithm.transition, name)
                require(torch.isfinite(value).all(), 'nonfinite policy '+name)
            result = env.step(action)
            terminal_obs = observations(result['observation'], self.n)
            values = self.critic(terminal_obs).detach()
            reward = timeout_rewards(result['reward'], result['terminated'], result['timed_out'], values)
            done = result['terminated'] | result['timed_out']
            require(torch.equal(result['live'], ~done), 'every training world either live or terminal')
            # Deliberately omit stock time_outs: we already used V(terminal), not
            # its V(pre-action) bootstrap. Store raw sampled actions, not clips.
            self.algorithm.process_env_step(terminal_obs, reward, done, {})
            if done.any():
                retained = env.reset(done)
                require(retained == [r if bool(done[i]) else None for i, r in enumerate(result['terminal_records'])],
                        'terminal records retained across training reset')
            self.next_obs = observations(env.observations(), self.n)
            self.phase = 'full' if self.storage.step == STEPS else 'collecting'
            return result
        except Exception:
            self.faulted = True
            raise

    def update(self):
        self._healthy()
        try:
            require(self.phase == 'full', 'full rollout before optimizer update')
            self.algorithm.compute_returns(self.next_obs)
            with self._rng_scope(): metrics = self.algorithm.update()
            require(all(type(v) in (float, int) and math.isfinite(v) for v in metrics.values()), 'finite optimizer metrics')
            checkpoint.validate_states(checkpoint.states_of(self.actor, self.critic), *checkpoint.fresh_models(self.seed))
            self.updates += 1; self.phase = 'empty'
            return dict(metrics=metrics, completed_updates=self.updates, cpu_fixture_only=type(self) is CpuStanceLearner,
                        checkpoint_admitted=False, physical_motion_authorized=False)
        except Exception:
            self.faulted = True
            raise
