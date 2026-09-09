"""Synthetic CPU optimizer tests plus a short real CPU collection-only fixture."""
from copy import deepcopy
import pytest
import torch

from mjlab_microduck.stance_ppo import CpuStanceLearner, STEPS, timeout_rewards
from mjlab_microduck import stance_checkpoint as weights


class SyntheticEnv:
    """Deliberate stand-in dynamics; never a Duck acceptance rollout."""
    synthetic_ppo_fixture = True
    def __init__(self):
        self.n = 2; self.device = torch.device('cpu'); self.live = torch.ones(2, dtype=torch.bool)
        self.tick = 0; self.obs = torch.zeros(2, 50); self.obs[:, -1] = 2.
        self.terminals = [None, None]
    def observations(self): return dict(actor=self.obs[:, :44].clone(), critic=self.obs.clone())
    def step(self, action):
        self.tick += 1; self.obs[:, 0] = self.tick/100; self.obs[:, -1] = self.tick+2
        terminated = torch.tensor([self.tick % 5 == 0, False])
        timed_out = torch.tensor([False, self.tick % 7 == 0]); self.live = ~(terminated | timed_out)
        self.terminals = [None if bool(self.live[i]) else dict(tick=self.tick, row=i) for i in range(2)]
        return dict(observation=self.observations(), reward=torch.full((2,), .03), terminated=terminated,
            timed_out=timed_out, live=self.live.clone(), terminal_records=deepcopy(self.terminals))
    def reset(self, done):
        retained = [r if bool(done[i]) else None for i, r in enumerate(self.terminals)]
        self.obs[done, -1] = -10.; self.live[done] = True
        for i in done.nonzero().flatten().tolist(): self.terminals[i] = None
        return retained


def collect(learner, env):
    for _ in range(STEPS): learner.collect_one(env)


def test_timeout_targets_do_not_bootstrap_failures_or_modify_reward():
    reward = torch.tensor([1., 2., 3.]); original = reward.clone()
    result = timeout_rewards(reward, torch.tensor([False, True, False]),
        torch.tensor([True, False, False]), torch.tensor([[10.], [20.], [30.]]))
    assert torch.allclose(result, torch.tensor([10.9, 2., 3.]))
    assert torch.equal(reward, original)
    with pytest.raises(ValueError, match='exclusive'):
        timeout_rewards(reward, torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), torch.zeros(3, 1))


def test_actual_adapter_uses_terminal_value_before_reset(monkeypatch):
    learner = CpuStanceLearner(); env = SyntheticEnv(); env.tick = 6
    monkeypatch.setattr(learner.critic, 'forward', lambda obs, **kw: obs['critic'][:, -1:])
    result = learner.collect_one(env)
    assert result['timed_out'].tolist() == [False, True]
    assert learner.storage.values[0, 1, 0] == 2.  # pre-action value
    assert learner.storage.rewards[0, 1, 0] == pytest.approx(.03+.99*9.)
    assert result['observation']['critic'][1, -1] == 9.
    assert learner.next_obs['critic'][1, -1] == -10.  # reset value never bootstrapped
    assert learner.storage.dones[0, 1, 0] == 1


def test_raw_sampled_actions_and_pre_action_inputs_are_stored(monkeypatch):
    learner = CpuStanceLearner(); env = SyntheticEnv(); observed = []
    with torch.no_grad(): learner.actor.distribution.std_param.fill_(5.)
    step = env.step
    def record(action): observed.append(action.clone()); return step(action)
    monkeypatch.setattr(env, 'step', record)
    before = env.observations()['actor']; learner.collect_one(env)
    assert observed[0].abs().max() > 1
    assert torch.equal(learner.storage.actions[0], observed[0])
    assert torch.equal(learner.storage.observations['actor'][0], before)


def test_configuration_drift_faults_before_rollout():
    learner = CpuStanceLearner(); learner.algorithm.gamma = .5
    with pytest.raises(ValueError, match='configuration changed'): learner.collect_one(SyntheticEnv())
    assert learner.faulted and learner.storage.step == 0


def test_real_cpu_collection_does_not_launch_optimizer():
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    learner = CpuStanceLearner(); env = WarpStanceRuntime(2, device='cpu')
    before = weights.state_hash(weights.states_of(learner.actor, learner.critic))
    result = learner.collect_one(env)
    assert learner.storage.step == 1 and env.live.all()
    assert not result['optimizer_launched'] and not learner.algorithm.optimizer.state
    assert before == weights.state_hash(weights.states_of(learner.actor, learner.critic))
    assert not torch.cuda.is_initialized()


def test_synthetic_update_finite_and_caller_rng_unchanged():
    rng = torch.get_rng_state().clone(); learner = CpuStanceLearner(); env = SyntheticEnv()
    collect(learner, env)
    assert learner.storage.step == STEPS and learner.phase == 'full'
    old = learner.initial_hash; report = learner.update()
    assert report['completed_updates'] == 1 and report['cpu_fixture_only'] and not report['checkpoint_admitted']
    assert old != weights.state_hash(weights.states_of(learner.actor, learner.critic))
    assert learner.storage.step == 0 and learner.phase == 'empty'
    assert torch.equal(rng, torch.get_rng_state())
    assert all(v['step'] == 20 for v in learner.algorithm.optimizer.state.values())


def test_gae_uses_global_normalization_and_stops_at_done():
    learner = CpuStanceLearner(); collect(learner, SyntheticEnv())
    st = learner.storage
    learner.algorithm.compute_returns(learner.next_obs)
    last = learner.critic(learner.next_obs).detach(); advantage = torch.zeros(2, 1)
    expected = torch.empty_like(st.returns)
    for i in reversed(range(STEPS)):
        next_value = last if i == STEPS-1 else st.values[i+1]
        mask = 1-st.dones[i].float()
        advantage = st.rewards[i]+.99*mask*next_value-st.values[i]+.99*.95*mask*advantage
        expected[i] = advantage+st.values[i]
    assert torch.allclose(expected, st.returns)
    raw = expected-st.values
    assert torch.allclose(st.advantages, (raw-raw.mean())/(raw.std()+1e-8))
    assert st.returns[4, 0, 0] == pytest.approx(float(st.rewards[4, 0, 0]), abs=1e-6)


@pytest.mark.parametrize('damage', ['reward', 'observation', 'reset_ledger', 'partial_update', 'raw_advantage', 'gradient'])
def test_fail_closed_without_sanitizing_invalid_training_data(damage, monkeypatch):
    learner = CpuStanceLearner(); env = SyntheticEnv()
    if damage == 'observation': env.obs[0, 0] = float('nan')
    if damage == 'reward':
        step = env.step
        def bad(a):
            r = step(a); r['reward'][0] = float('nan'); return r
        monkeypatch.setattr(env, 'step', bad)
    if damage == 'reset_ledger':
        env.tick = 4; monkeypatch.setattr(env, 'reset', lambda _: [None, None])
    if damage in ('reward', 'observation', 'reset_ledger'):
        with pytest.raises(ValueError): learner.collect_one(env)
    else:
        if damage != 'partial_update': collect(learner, env)
        if damage == 'raw_advantage': learner.storage.rewards[0, 0] = float('nan')
        if damage == 'gradient':
            next(learner.actor.parameters()).register_hook(lambda grad: grad*float('nan'))
        with pytest.raises(ValueError): learner.update()
    assert learner.faulted and not learner.algorithm.optimizer.state
    with pytest.raises(ValueError, match='closeout'): learner.collect_one(env)


@pytest.mark.parametrize('n', [1, 9, 64, 512, True])
def test_cpu_harness_cannot_silently_become_a_pilot(n):
    with pytest.raises(ValueError, match='bounded CPU'): CpuStanceLearner(n)
