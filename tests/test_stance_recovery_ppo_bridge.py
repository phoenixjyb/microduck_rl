"""Synthetic CPU transition tests; no native runtime or recovery training claim."""
from copy import deepcopy
from hashlib import sha256
import pytest
import torch
from rsl_rl.storage import RolloutStorage

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_ppo_bridge as bridge_module
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_policy_preparation as preparation
from mjlab_microduck.stance_ppo import CONFIG, observations


SOURCE = "a" * 40
CELLS = ["zero-wrench", "+x-2n-20steps-t250"]
RAW_PARENT = b"synthetic parent bytes; never an authenticated archive"


class SyntheticScheduledEnv:
    """Deterministic API fixture; not the scheduled Warp runtime."""
    synthetic_ppo_fixture = True

    def __init__(self, declaration, *, terminal_at=None):
        self.schedule_declaration = schedule.checked(declaration)
        self.n = self.schedule_declaration["worlds"]
        self.device = torch.device("cpu")
        self.live = torch.ones(self.n, dtype=torch.bool)
        self.steps = torch.zeros(self.n, dtype=torch.long)
        self.tick = 0
        self.terminal_at = terminal_at or {}
        self.terminal = [None] * self.n
        self.obs = torch.zeros(self.n, 50)
        self.obs[:, -1] = 2.
        self.correction = torch.zeros(self.n, 10)
        self.targets = torch.zeros(self.n, 14)
        self.queue = torch.zeros(self.n, 3, 14)
        self.received_actions = []
        self.applied_actions = []
        self.pulse_steps = [[] for _ in range(self.n)]
        self.before_reset = None

    def observations(self):
        return {"actor": self.obs[:, :44].clone(), "critic": self.obs.clone()}

    def _control_snapshot(self):
        return {"correction": self.correction.clone(), "target": self.targets.clone(),
                "queue": self.queue.clone()}

    def step_with_schedule(self, actions, *, capture_control=False):
        assert type(capture_control) is bool
        schedule.checked(self.schedule_declaration)
        assert self.live.all()
        self.tick += 1
        self.received_actions.append(actions.clone())
        clipped = actions.clamp(-1, 1)
        self.applied_actions.append(clipped.clone())
        desired = .2 * clipped
        next_correction = (self.correction + (desired-self.correction).clamp(-.02, .02))
        self.correction.copy_(next_correction)
        self.targets[:, :10] = self.correction
        self.queue[:, 0] = self.targets
        for row, cell in enumerate(self.schedule_declaration["row_cells"]):
            if bool(self.live[row]):
                for physics_step in range(int(self.steps[row]), int(self.steps[row])+10):
                    if cell["onset_step"] <= physics_step < cell["onset_step"]+cell["duration_steps"]:
                        self.pulse_steps[row].append(physics_step)
        self.steps[self.live] += 10
        self.obs[:, 0] = self.tick / 100
        self.obs[:, -1] = self.tick + 2.
        terminated = torch.zeros(self.n, dtype=torch.bool)
        timed_out = torch.zeros_like(terminated)
        for row, (at_tick, kind, terminal_value) in self.terminal_at.items():
            if self.tick == at_tick:
                (terminated if kind == "terminated" else timed_out)[row] = True
                self.obs[row, -1] = terminal_value
                self.terminal[row] = {"row": row, "tick": self.tick, "kind": kind}
        done = terminated | timed_out
        self.live &= ~done
        records = deepcopy(self.terminal)
        result = dict(observation=self.observations(), reward=torch.full((self.n,), .03),
            terminated=terminated, timed_out=timed_out, live=self.live.clone(),
            episode_steps=self.steps.clone(), terminal_records=records, optimizer_launched=False,
            scheduled_pulse_evidence=deepcopy(self.pulse_steps))
        if capture_control:
            result["control_evidence"] = {"raw_actions": actions.clone(),
                "applied_actions": clipped.clone()}
        return result

    def reset(self, rows):
        self.before_reset = dict(steps=self.steps.clone(), live=self.live.clone(),
                                 controls=self._control_snapshot())
        retained = [deepcopy(self.terminal[i]) if bool(rows[i]) else None for i in range(self.n)]
        self.steps[rows] = 0
        self.live[rows] = True
        self.correction[rows] = 0
        self.targets[rows] = 0
        self.queue[rows] = 0
        self.obs[rows] = 0
        self.obs[rows, -1] = -10.
        for i in rows.nonzero().flatten().tolist():
            self.terminal[i] = None
        return retained


def _prepared(monkeypatch, *, seed, worlds):
    identity = parent.expected_identity()
    with torch.random.fork_rng(devices=[]):
        actor, critic = checkpoint.validate_identity(identity, evaluation="lean-replication")
    actor.train().requires_grad_(True)
    critic.train().requires_grad_(True)
    state_sha = checkpoint.state_hash(checkpoint.states_of(actor, critic))
    monkeypatch.setattr(parent, "PARENT_STATE_SHA256", state_sha)
    obs = observations(dict(actor=torch.zeros(worlds, 44), critic=torch.zeros(worlds, 50)), worlds)
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        storage = RolloutStorage("rl", worlds, preparation.STEPS, obs, (10,), device="cpu")
        algorithm = preparation.FinitePPO(actor, critic, storage, **CONFIG, device="cpu")
    parent_receipt = dict(parent_checkpoint_sha256=contract.CHECKPOINT_SHA256,
        parent_state_sha256=state_sha, strict_actor_restore=True, strict_critic_restore=True,
        execution_admitted=False, synthetic_fixture=True, **contract.FALSE_FLAGS)
    receipt = dict(protocol=preparation.PROTOCOL, learner_seed=seed, worlds=worlds,
        initial_state_sha256=state_sha, parent_preparation=parent_receipt,
        learner_rng_state_sha256=sha256(torch.Generator(device="cpu").manual_seed(seed).get_state().numpy().tobytes()).hexdigest(),
        learner_rng_connected_to_sampler=False, optimizer_steps=0, storage_step=0,
        optimizer_state_entries=0)
    return dict(actor=actor, critic=critic, algorithm=algorithm, storage=storage,
                learner_rng_state=torch.Generator(device="cpu").manual_seed(seed).get_state(),
                receipt=receipt)


def _make(monkeypatch, *, seed=653, worlds=2, terminal_at=None, env=None):
    row_cells = [CELLS[i % len(CELLS)] for i in range(worlds)]
    declaration = schedule.declaration(SOURCE, "dose", "training", row_cells)
    env = env or SyntheticScheduledEnv(declaration, terminal_at=terminal_at)
    prepared_calls = []
    def prepare(raw, *, seed, worlds):
        prepared_calls.append((raw, seed, worlds))
        return _prepared(monkeypatch, seed=seed, worlds=worlds)
    monkeypatch.setattr(bridge_module.preparation, "prepare_policy", prepare)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(bridge_module.torch.cuda, "is_initialized", lambda: False)
    learner = bridge_module.RecoveryPPOBridge.from_parent(
        RAW_PARENT, env, declaration, seed=seed, synthetic_fixture=True)
    assert prepared_calls == [(RAW_PARENT, seed, env.n)]
    return learner, env


@pytest.mark.parametrize("seed", [653, 659])
@pytest.mark.parametrize("worlds", [2, 64])
def test_preparation_keeps_new_seeds_out_of_old_allowlist(monkeypatch, seed, worlds):
    assert 653 not in checkpoint.FRESH_SEEDS and 659 not in checkpoint.FRESH_SEEDS
    learner, _ = _make(monkeypatch, seed=seed, worlds=worlds)
    assert learner.initial_state_sha256 == parent.PARENT_STATE_SHA256
    assert learner.worlds == worlds
    assert learner.storage.num_transitions_per_env == bridge_module.HORIZON == 28
    assert learner.storage.step == 0 and not learner.algorithm.optimizer.state
    assert learner.phase == "empty" and learner.completed_updates == 0
    assert torch.equal(learner.private_rng_state,
                       torch.Generator(device="cpu").manual_seed(seed).get_state())


def test_private_action_rng_is_repeatable_and_does_not_mutate_caller(monkeypatch):
    first, env1 = _make(monkeypatch)
    second, env2 = _make(monkeypatch)
    caller_rng = torch.Generator(device="cpu").manual_seed(88).get_state()
    torch.random.set_rng_state(caller_rng)
    first.collect_one()
    after_first = torch.random.get_rng_state().clone()
    action1 = env1.received_actions[0]
    private1 = first.private_rng_state.clone()
    assert torch.equal(after_first, caller_rng)
    assert not torch.equal(private1, torch.Generator(device="cpu").manual_seed(653).get_state())
    torch.random.set_rng_state(caller_rng)
    second.collect_one()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert torch.equal(env2.received_actions[0], action1)
    assert torch.equal(second.private_rng_state, private1)


def test_failed_sampling_keeps_advanced_rng_private_and_faults(monkeypatch):
    learner, _ = _make(monkeypatch)
    caller_rng = torch.Generator(device="cpu").manual_seed(220).get_state()
    torch.random.set_rng_state(caller_rng)
    def fail_after_draw(_obs):
        torch.rand(3)
        raise RuntimeError("synthetic sampler fault")
    monkeypatch.setattr(learner.algorithm, "act", fail_after_draw)
    with pytest.raises(RuntimeError, match="synthetic sampler fault"):
        learner.collect_one()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert not torch.equal(learner.private_rng_state,
                           torch.Generator(device="cpu").manual_seed(653).get_state())
    assert learner.faulted and learner.phase == "faulted"


def test_terminal_observation_timeout_bootstrap_is_added_once_before_selective_reset(monkeypatch):
    learner, env = _make(monkeypatch, terminal_at={1: (1, "timed_out", 9.)})
    monkeypatch.setattr(learner.critic, "forward", lambda obs, **kwargs: obs["critic"][:, -1:])
    before = env.observations()
    result = learner.collect_one()
    assert result["timed_out"].tolist() == [False, True]
    assert learner.storage.values[0, 1, 0] == before["critic"][1, -1] == 2.
    assert result["observation"]["critic"][1, -1] == 9.
    assert learner.storage.rewards[0, 1, 0] == pytest.approx(.03 + .99*9.)
    assert learner.storage.dones[0, 1, 0] == 1
    assert env.observations()["critic"][1, -1] == -10.
    assert result["bridge_receipt"]["terminal_records"][1] == {"row": 1, "tick": 1, "kind": "timed_out"}
    assert learner.terminal_events[0]["row"] == 1


def test_raw_sampled_actions_are_stored_while_runtime_clips_effect(monkeypatch):
    learner, env = _make(monkeypatch)
    with torch.no_grad():
        learner.actor.distribution.std_param.fill_(5.)
    result = learner.collect_one(capture_control=True)
    sampled = env.received_actions[0]
    assert sampled.abs().max() > 1.
    assert torch.equal(learner.storage.actions[0], sampled)
    assert torch.equal(result["control_evidence"]["raw_actions"], sampled)
    assert torch.equal(result["control_evidence"]["applied_actions"], sampled.clamp(-1, 1))
    assert torch.equal(env.applied_actions[0], sampled.clamp(-1, 1))


def test_partial_reset_retains_terminal_and_preserves_other_row_state(monkeypatch):
    learner, env = _make(monkeypatch, terminal_at={0: (1, "terminated", 4.)})
    result = learner.collect_one()
    assert result["terminated"].tolist() == [True, False]
    assert result["bridge_receipt"]["reset_records"] == [
        {"row": 0, "tick": 1, "kind": "terminated"}, None]
    assert env.steps.tolist() == [0, 10] and env.live.tolist() == [True, True]
    assert torch.equal(env.correction[1], env.before_reset["controls"]["correction"][1])
    assert torch.equal(env.steps[1], env.before_reset["steps"][1])
    assert learner.schedule["row_cells"] == schedule.checked(env.schedule_declaration)["row_cells"]
    assert learner.terminal_events[0]["record"] == result["terminal_records"][0]


def test_fixed_schedule_clock_reaches_pulse_only_after_full_28_step_buffer(monkeypatch):
    learner, env = _make(monkeypatch)
    for _ in range(24):
        learner.collect_one()
    assert learner.storage.step == 24 and learner.phase == "collecting"
    assert env.pulse_steps[1] == []
    for _ in range(4):
        learner.collect_one()
    assert learner.storage.step == bridge_module.HORIZON and learner.phase == "full"
    assert env.pulse_steps[1] == list(range(250, 270))
    assert env.pulse_steps[0] == []
    assert len(env.received_actions) == 28
    with pytest.raises(ValueError, match="recovery bridge collecting phase"):
        learner.collect_one()
    assert learner.faulted and learner.phase == "faulted"


def test_recovery_finite_gae_is_bound_to_28_step_storage(monkeypatch):
    learner, _ = _make(monkeypatch)
    for _ in range(bridge_module.HORIZON):
        learner.collect_one()
    learner.algorithm.compute_returns(observations(learner.env.observations(), learner.worlds))
    assert learner.storage.returns.shape == (28, 2, 1)
    assert torch.isfinite(learner.storage.returns).all()
    assert torch.isfinite(learner.storage.advantages).all()


def test_one_finite_update_advances_only_private_rng_and_keeps_parent_identity(monkeypatch):
    learner, _ = _make(monkeypatch)
    caller_rng = torch.Generator(device="cpu").manual_seed(991).get_state()
    torch.random.set_rng_state(caller_rng)
    for _ in range(bridge_module.HORIZON):
        learner.collect_one()
    start_private = learner.private_rng_state.clone()
    result = learner.update()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert not torch.equal(learner.private_rng_state, start_private)
    assert result["completed_updates"] == 1
    assert result["optimizer_steps"] == CONFIG["num_learning_epochs"]*CONFIG["num_mini_batches"]
    assert result["parent_initial_state_sha256"] == learner.initial_state_sha256
    assert result["updated_state_sha256"] != learner.initial_state_sha256
    assert learner.storage.step == 0 and learner.phase == "empty"
    assert len(learner.algorithm.optimizer.state) == sum(1 for _ in learner.actor.parameters()) + sum(
        1 for _ in learner.critic.parameters())
    for state in learner.algorithm.optimizer.state.values():
        assert all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                   for value in state.values())
    with pytest.raises(ValueError, match="single update"):
        learner.update()
    assert learner.faulted and learner.phase == "faulted"


def test_seeded_minibatch_update_is_repeatable_and_leaks_no_global_rng(monkeypatch):
    one, _ = _make(monkeypatch)
    two, _ = _make(monkeypatch)
    for learner in (one, two):
        for _ in range(bridge_module.HORIZON):
            learner.collect_one()
    caller_rng = torch.Generator(device="cpu").manual_seed(411).get_state()
    torch.random.set_rng_state(caller_rng)
    first = one.update()
    first_state = checkpoint.state_hash(checkpoint.states_of(one.actor, one.critic))
    first_private = one.private_rng_state.clone()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    torch.random.set_rng_state(caller_rng)
    second = two.update()
    second_state = checkpoint.state_hash(checkpoint.states_of(two.actor, two.critic))
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert first_state == second_state
    assert first_private.equal(two.private_rng_state)
    assert first["metrics"] == second["metrics"]


def test_nonfinite_adam_state_faults_after_the_single_update(monkeypatch):
    learner, _ = _make(monkeypatch)
    for _ in range(bridge_module.HORIZON):
        learner.collect_one()
    learner.update()
    parameter = next(learner.actor.parameters())
    learner.algorithm.optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite recovery Adam state"):
        learner.collect_one()
    assert learner.faulted and learner.phase == "faulted"


def test_incomplete_phase_configuration_and_nonfinite_grad_fault_without_retry(monkeypatch):
    partial, _ = _make(monkeypatch)
    for _ in range(3):
        partial.collect_one()
    with pytest.raises(ValueError, match="complete recovery rollout"):
        partial.update()
    assert partial.faulted and partial.phase == "faulted"

    drifted, _ = _make(monkeypatch)
    drifted.algorithm.gamma = .5
    with pytest.raises(ValueError, match="unchanged reviewed recovery PPO configuration"):
        drifted.collect_one()
    assert drifted.faulted and drifted.storage.step == 0

    bad_grad, _ = _make(monkeypatch)
    for _ in range(bridge_module.HORIZON):
        bad_grad.collect_one()
    next(bad_grad.actor.parameters()).register_hook(lambda grad: grad*float("nan"))
    with pytest.raises(ValueError, match="finite complete gradients"):
        bad_grad.update()
    assert bad_grad.faulted and bad_grad.phase == "faulted"
    assert not bad_grad.algorithm.optimizer.state


def test_bad_reward_and_runtime_exception_fault_without_softening_or_retry(monkeypatch):
    bad_reward, env = _make(monkeypatch)
    original = env.step_with_schedule
    def nan_reward(*args, **kwargs):
        result = original(*args, **kwargs)
        result["reward"][0] = float("nan")
        return result
    monkeypatch.setattr(env, "step_with_schedule", nan_reward)
    with pytest.raises(ValueError, match="nonfinite PPO"):
        bad_reward.collect_one()
    assert bad_reward.faulted and bad_reward.storage.step == 0

    failed, env2 = _make(monkeypatch)
    def raises(*args, **kwargs):
        raise RuntimeError("synthetic runtime fault")
    monkeypatch.setattr(env2, "step_with_schedule", raises)
    with pytest.raises(RuntimeError, match="synthetic runtime fault"):
        failed.collect_one()
    with pytest.raises(ValueError, match="faulted recovery bridge"):
        failed.collect_one()

    reset_failed, env3 = _make(monkeypatch, terminal_at={0: (1, "terminated", 4.)})
    monkeypatch.setattr(env3, "reset", lambda _rows: (_ for _ in ()).throw(RuntimeError("reset fault")))
    with pytest.raises(RuntimeError, match="reset fault"):
        reset_failed.collect_one()
    assert reset_failed.terminal_events == [{"storage_step": 0, "row": 0,
        "record": {"row": 0, "tick": 1, "kind": "terminated"}}]
    assert reset_failed.faulted and reset_failed.phase == "faulted"


def test_nonfinite_model_and_unlabeled_fake_runtime_refuse(monkeypatch):
    learner, _ = _make(monkeypatch)
    with torch.no_grad():
        next(learner.actor.parameters()).fill_(float("inf"))
    with pytest.raises(ValueError, match="finite trainable CPU actor parameter"):
        learner.collect_one()
    assert learner.faulted

    declaration = schedule.declaration(SOURCE, "dose", "training", CELLS)
    env = SyntheticScheduledEnv(declaration)
    monkeypatch.setattr(bridge_module.preparation, "prepare_policy", lambda *a, **kw: pytest.fail("must refuse first"))
    with pytest.raises(ValueError, match="exact scheduled runtime type"):
        bridge_module.RecoveryPPOBridge.from_parent(RAW_PARENT, env, declaration, seed=653)


def test_held_out_schedule_cannot_be_optimized(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    declaration = schedule.declaration(SOURCE, 'dose', 'held-out', CELLS)
    env = SyntheticScheduledEnv(declaration)
    with pytest.raises(ValueError, match='held-out cells cannot enter'):
        bridge_module.RecoveryPPOBridge(RAW_PARENT, env, declaration, seed=653, synthetic_fixture=True)


def test_parent_schema_and_positive_scale_remain_fixed(monkeypatch):
    learner, _ = _make(monkeypatch)
    learner.actor.register_buffer('unexpected_parent_buffer', torch.zeros(1))
    with pytest.raises(ValueError, match='unchanged parent model state keys'):
        learner.collect_one()
    assert learner.faulted
    learner, _ = _make(monkeypatch)
    with torch.no_grad(): learner.actor.distribution.std_param.fill_(0.)
    with pytest.raises(ValueError, match='positive recovery Gaussian scale'):
        learner.collect_one()
    assert learner.faulted


def test_done_without_terminal_record_cannot_be_stored_or_reset(monkeypatch):
    learner, env = _make(monkeypatch, terminal_at={0: (1, 'terminated', 4.)})
    original = env.step_with_schedule
    def missing(*args, **kwargs):
        result = original(*args, **kwargs); result['terminal_records'][0] = None
        return result
    monkeypatch.setattr(env, 'step_with_schedule', missing)
    with pytest.raises(ValueError, match='exact terminal-record presence'):
        learner.collect_one()
    assert learner.storage.step == 0 and env.steps.tolist() == [10, 10]
    assert learner.faulted
