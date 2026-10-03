"""SYNTHETIC_CPU_RSL checks for the CUDA transition source contract.

These tests use real stock CPU PPO/storage and a synthetic scheduled runtime.
The narrow private device/provenance seams below are test-only; no result is
CUDA, native physics, selective-reset or learning qualification.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_ppo
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_lesson_plan as lesson
from mjlab_microduck import stance_recovery_schedule as schedule

SYNTHETIC_BACKEND = "SYNTHETIC_CPU_RSL_NO_CUDA"
SOURCE = "a" * 40
LEASE_FD = 19
PARENT_HASH = preparation.parent.PARENT_STATE_SHA256
PRODUCTION_CHECK_POLICY = preparation._check_policy


class SyntheticPrivateScope:
    """CPU process-RNG substitute; does not represent a CUDA generator."""

    def __init__(self, seed, state, *, lease_fd):
        assert seed in preparation.SEEDS and lease_fd == LEASE_FD
        self.seed = seed
        self._state = state.detach().cpu().clone()
        self._active = False
        self._faulted = False
        self._scope_count = 0
        self._caller_cpu_preserved = None
        self.stall_private_state = False
        self.scope_count_override = 0
        self.cpu_preserved_override = None
        self.cuda_preserved_override = None

    @property
    def receipt(self):
        return {
            "source": SOURCE,
            "seed": self.seed,
            "device": "cuda:0",
            "scope_count": self._scope_count + self.scope_count_override,
            "scope_active": self._active,
            "faulted": self._faulted,
            "caller_cpu_rng_preserved": (
                self._caller_cpu_preserved
                if self.cpu_preserved_override is None
                else self.cpu_preserved_override
            ),
            # Synthetic only: no CUDA stream exists in these fixtures.
            "caller_cuda_rng_preserved": (
                True
                if self.cuda_preserved_override is None
                else self.cuda_preserved_override
            ),
        }

    @property
    def state(self):
        return self._state.detach().clone()

    def _verify_context(self):
        if self._faulted:
            raise ValueError("synthetic scope faulted")

    def scope(self):
        return _SyntheticScopeContext(self)


class _SyntheticScopeContext:
    def __init__(self, scope):
        self.scope = scope

    def __enter__(self):
        scope = self.scope
        scope._verify_context()
        if scope._active:
            scope._faulted = True
            raise ValueError("synthetic private scope reentry")
        scope._active = True
        self.cpu_before = torch.random.get_rng_state().clone()
        self.context = torch.random.fork_rng(devices=[])
        self.context.__enter__()
        torch.random.set_rng_state(scope._state)
        return None

    def __exit__(self, exc_type, exc, tb):
        scope = self.scope
        advanced = torch.random.get_rng_state().clone()
        if not scope.stall_private_state:
            scope._state = advanced
        scope._active = False
        if exc_type is not None:
            scope._faulted = True
        result = self.context.__exit__(exc_type, exc, tb)
        scope._caller_cpu_preserved = torch.equal(
            self.cpu_before, torch.random.get_rng_state()
        )
        if exc_type is None:
            scope._scope_count += 1
        return result


def _cpu_tensor(value, shape, label, *, dtype=torch.float32):
    assert torch.is_tensor(value), label
    assert tuple(value.shape) == shape, label
    assert value.dtype == dtype and value.device.type == "cpu", label
    assert not value.is_floating_point() or torch.isfinite(value).all(), label


def _cpu_observations(steps, *, terminal_marker=0):
    actor = torch.zeros(64, 44)
    actor[:, 0] = steps.float()
    critic = torch.zeros(64, 50)
    critic[:, :44] = actor
    critic[:, 44:] = float(terminal_marker)
    return {"actor": actor, "critic": critic}


class SyntheticScheduledRuntime:
    """Minimal test-only row runtime implementing the reviewed result schema."""

    synthetic_ppo_fixture = True

    def __init__(self, plan=None):
        cells = lesson.cells("gentle", held_out=False)
        cell_id = cells[0]["id"]
        self.schedule_declaration = schedule.declaration(
            SOURCE, "gentle", "training", [cell_id] * 64
        )
        self._schedule_sha256 = schedule.binding_sha256(self.schedule_declaration)
        self.device = torch.device("cuda:0")  # Declared profile only; tensors stay CPU.
        self.n = 64
        self.steps = torch.zeros(64, dtype=torch.long)
        self.live = torch.ones(64, dtype=torch.bool)
        self.faulted = False
        self.plan = plan or {}
        self.reset_drift = None
        self.storage = None
        self.algorithm = None
        self.calls = 0
        self.events = []
        self.actions_seen = []
        self.mutate_passed_actions = False
        self.terminal_records = [None] * 64
        self._terminal_observations = _cpu_observations(self.steps)
        self.initial_qpos = torch.arange(21, dtype=torch.float32) / 100
        self.physics = {
            "qpos": self.initial_qpos.expand(64, -1).clone(),
            "qvel": torch.zeros(64, 20),
            "time": torch.zeros(64),
        }
        self.initial_physics = deepcopy(self.physics)
        self.initial_control = {
            "rows": torch.arange(128, dtype=torch.float32).reshape(64, 2),
            "global": torch.tensor([7.0]),
        }
        self.control = deepcopy(self.initial_control)

    def _control_snapshot(self):
        return deepcopy(self.control)

    def _view(self, name):
        return self.physics[name]

    def observations(self):
        return {k: v.clone() for k, v in self._terminal_observations.items()}

    def step_with_schedule(self, actions, *, capture_control=False):
        self.calls += 1
        step_no = self.calls
        self.events.append(("physics", step_no))
        spec = self.plan.get(step_no, {})
        if spec.get("raise"):
            raise RuntimeError("synthetic physics")
        if self.mutate_passed_actions:
            actions.clamp_(-1, 1)
        self.actions_seen.append(actions.detach().clone())
        terminated = torch.zeros(64, dtype=torch.bool)
        timed_out = torch.zeros(64, dtype=torch.bool)
        executed = torch.full((64,), 10, dtype=torch.long)
        for row in spec.get("terminated", []):
            terminated[row] = True
            executed[row] = spec.get("terminated_executed", 3)
        for row in spec.get("timed_out", []):
            timed_out[row] = True
        done = terminated | timed_out
        self.steps += executed
        if done.any():
            self.steps[done] = (
                int(spec.get("timeout_clock", 20))
                if timed_out.any()
                else self.steps[done]
            )
        self.live = ~done
        self.physics["qpos"][:, 0] += 1
        self.physics["qvel"][:, 0] += 2
        self.physics["time"] += 0.02
        self.control["rows"] += 0.5

        self.terminal_records = [
            (
                {
                    "row": i,
                    "kind": "timeout" if timed_out[i] else "failure",
                    "call": step_no,
                }
                if bool(done[i])
                else None
            )
            for i in range(64)
        ]
        marker = float(100 + step_no)
        self._terminal_observations = _cpu_observations(
            self.steps, terminal_marker=marker
        )
        reward = torch.ones(64)
        reward[terminated] = -2.0
        result = {
            "reward": reward,
            "terminated": terminated,
            "timed_out": timed_out,
            "episode_steps": self.steps.clone(),
            "executed_steps": executed,
            "live": self.live.clone(),
            "term_sums": {"synthetic": reward.clone()},
            "observation": self.observations(),
            "terminal_records": deepcopy(self.terminal_records),
            "optimizer_launched": False,
            "boundaries": [],
            "scheduled_pulse_evidence": [],
        }
        self.events.append(("result", step_no))
        return result

    def reset(self, rows):
        self.events.append(("reset", self.storage.step, rows.detach().clone()))
        assert self.storage.step == self.calls, "transition must be stored before reset"
        assert self.algorithm.transition.actions is None, (
            "RSL transition must be consumed before reset"
        )
        records = [
            self.terminal_records[i] if bool(rows[i]) else None for i in range(64)
        ]
        self.steps[rows] = 0
        self.live[rows] = True
        for name, initial in self.initial_physics.items():
            self.physics[name][rows] = initial[rows]
        self.control["rows"][rows] = self.initial_control["rows"][rows]
        self.physics["qpos"][rows] = self.initial_qpos
        self.physics["qvel"][rows] = 0
        self.physics["time"][rows] = 0
        self.terminal_records = [
            None if bool(rows[i]) else self.terminal_records[i] for i in range(64)
        ]
        reset_obs = _cpu_observations(self.steps, terminal_marker=0)
        for key in self._terminal_observations:
            self._terminal_observations[key][rows] = reset_obs[key][rows]
        if self.reset_drift in ("qpos", "qvel", "time", "control", "observation"):
            untouched_row = int((~rows).nonzero()[0].item())
            if self.reset_drift == "qpos":
                self.physics["qpos"][untouched_row, 1] += 1
            elif self.reset_drift == "qvel":
                self.physics["qvel"][untouched_row, 1] += 1
            elif self.reset_drift == "time":
                self.physics["time"][untouched_row] += 1
            elif self.reset_drift == "control":
                self.control["rows"][untouched_row, 1] += 1
            else:
                self._terminal_observations["actor"][untouched_row, 2] += 1
                self._terminal_observations["critic"][untouched_row, 2] += 1
        elif self.reset_drift == "done-qpos":
            self.physics["qpos"][rows.nonzero()[0][0], 1] += 1
        elif self.reset_drift == "done-qvel":
            self.physics["qvel"][rows.nonzero()[0][0], 1] += 1
        elif self.reset_drift == "done-time":
            self.physics["time"][rows.nonzero()[0][0]] += 1
        return deepcopy(records)


@pytest.fixture
def synthetic_collector(monkeypatch):
    """Real stock CPU PPO/storage plus explicit synthetic CUDA contract seams."""
    lease_calls = []
    monkeypatch.setattr(
        transition.training_smoke, "inherited_lease", lambda fd: lease_calls.append(fd)
    )

    def validate_receipt(receipt, prepared):
        if receipt.get("source") != SOURCE:
            raise ValueError("synthetic prepared source mismatch")
        if receipt.get("learner_seed") != 653:
            raise ValueError("synthetic prepared seed mismatch")
        return SOURCE, 653, prepared["private_cuda_rng_state"].clone()

    monkeypatch.setattr(sampling, "_validate_receipt", validate_receipt)
    monkeypatch.setattr(sampling, "_check_cuda0_ready", lambda: None)
    monkeypatch.setattr(preparation, "_check_rsl_sources", lambda: None)
    monkeypatch.setattr(transition.checkpoint, "runtime_check", lambda: None)
    monkeypatch.setattr(
        transition.rng_scope, "CudaPrivateRngScope", SyntheticPrivateScope
    )
    monkeypatch.setattr(
        transition, "ScheduledRecoveryRuntime", SyntheticScheduledRuntime
    )
    monkeypatch.setattr(transition, "_tensor", _cpu_tensor)
    monkeypatch.setattr(
        transition.checkpoint, "state_hash", lambda _states: PARENT_HASH
    )

    actor, critic = checkpoint.fresh_models(seed=521)
    observations = TensorDict(
        {"actor": torch.zeros(64, 44), "critic": torch.zeros(64, 50)}, [64]
    )
    storage = RolloutStorage("rl", 64, 28, observations, (10,), device="cpu")
    algorithm = PPO(actor, critic, storage, **stance_ppo.CONFIG, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(653)
    private_state = generator.get_state().clone()
    prepared = {
        "actor": actor,
        "critic": critic,
        "algorithm": algorithm,
        "storage": storage,
        "receipt": {"source": SOURCE, "learner_seed": 653},
        "private_cuda_generator": SimpleNamespace(device="cuda:0"),
        "private_cuda_rng_state": private_state,
        "caller_rng_states": {},
    }

    def check_policy(a, c, algo, st):
        assert type(algo) is PPO
        assert algo.actor is a and algo.critic is c and algo.storage is st
        assert str(algo.device) == "cpu"
        assert type(algo.optimizer) is torch.optim.Adam
        assert len(algo.optimizer.param_groups) == 1
        assert set(algo.optimizer.param_groups[0]["params"]) == set(
            a.parameters()
        ) | set(c.parameters())
        assert not algo.optimizer.state

    def check_storage(st):
        assert type(st) is RolloutStorage and st.training_type == "rl"
        assert st.num_envs == 64 and st.num_transitions_per_env == 28
        assert st.actions_shape == (10,) and st.step < 28

    monkeypatch.setattr(preparation, "_check_policy", check_policy)
    monkeypatch.setattr(preparation, "_check_storage", check_storage)
    env = SyntheticScheduledRuntime()
    collector = transition.CudaTransitionCollector(prepared, env, lease_fd=LEASE_FD)
    env.storage, env.algorithm = storage, algorithm
    assert lease_calls == [LEASE_FD, LEASE_FD]
    assert SYNTHETIC_BACKEND == "SYNTHETIC_CPU_RSL_NO_CUDA"
    yield collector


def test_source_only_one_step_uses_stock_storage_and_no_update(synthetic_collector):
    collector = synthetic_collector
    result = collector.collect_one()
    assert result["receipt"]["optimizer_steps"] == 0
    assert result["receipt"]["native_terminal_qualified"] is False
    assert result["receipt"]["native_selective_reset_qualified"] is False
    assert all(result["receipt"][key] is False for key in preparation.FALSE_FLAGS)
    assert result["pre_action_observations"]["actor"].shape == (64, 44)
    assert result["pre_action_observations"]["critic"].shape == (64, 50)
    assert result["raw_actions"].shape == (64, 10)
    assert collector.storage.step == collector.transitions == 1
    assert collector.algorithm.transition.actions is None
    assert not collector.algorithm.optimizer.state
    assert torch.equal(collector.storage.actions[0], result["raw_actions"])
    assert torch.equal(
        collector.storage.observations["actor"][0],
        result["pre_action_observations"]["actor"],
    )
    assert torch.equal(
        collector.storage.observations["critic"][0],
        result["pre_action_observations"]["critic"],
    )
    assert not torch.equal(
        result["private_rng_state_before"], result["private_rng_state"]
    )
    scope_receipt = result["rng_scope_receipt"]
    assert scope_receipt["scope_count"] == 1
    assert scope_receipt["scope_active"] is False
    assert scope_receipt["faulted"] is False
    assert scope_receipt["caller_cpu_rng_preserved"] is True
    assert scope_receipt["caller_cuda_rng_preserved"] is True


@pytest.mark.parametrize(
    "corrupt",
    ["stalled-state", "wrong-count", "cpu-caller", "cuda-caller"],
)
def test_per_action_private_scope_receipt_is_required(synthetic_collector, corrupt):
    collector = synthetic_collector
    if corrupt == "stalled-state":
        collector.scope.stall_private_state = True
    elif corrupt == "wrong-count":
        collector.scope.scope_count_override = 1
    elif corrupt == "cpu-caller":
        collector.scope.cpu_preserved_override = False
    else:
        collector.scope.cuda_preserved_override = False
    with pytest.raises(ValueError, match="one advancing private draw"):
        collector.collect_one()
    assert collector.faulted and collector.storage.step == 0
    assert collector.env.calls == 0


def test_action_copy_prevents_runtime_alias_or_clipping(synthetic_collector):
    collector = synthetic_collector
    with torch.no_grad():
        collector.actor.distribution.std_param.fill_(2.0)
    collector.env.mutate_passed_actions = True
    result = collector.collect_one()
    assert (result["raw_actions"].abs() > 1).any()
    assert torch.equal(collector.storage.actions[0], result["raw_actions"])
    assert collector.algorithm.transition.actions is None
    assert torch.equal(
        collector.env.actions_seen[0].clamp(-1, 1), collector.env.actions_seen[0]
    )
    assert not torch.equal(collector.env.actions_seen[0], result["raw_actions"])


def test_failure_store_then_selective_reset_and_untouched_rows(synthetic_collector):
    collector = synthetic_collector
    collector.env.plan[1] = {"terminated": [4], "terminated_executed": 3}
    result = collector.collect_one()
    env = collector.env
    reset_events = [event for event in env.events if event[0] == "reset"]
    assert len(reset_events) == 1
    assert torch.where(reset_events[0][2])[0].tolist() == [4]
    assert result["terminated"][4] and not result["timed_out"][4]
    assert env.steps[4] == 0 and env.live.all()
    assert (env.steps[torch.arange(64) != 4] == 10).all()
    assert result["terminal_records"][4]["kind"] == "failure"
    assert result["reset_records"][4] == result["terminal_records"][4]
    assert torch.equal(env.control["rows"][4], env.initial_control["rows"][4])
    assert torch.equal(env.control["rows"][3], env.initial_control["rows"][3] + 0.5)
    assert torch.equal(
        env.physics["qpos"][3],
        env.initial_physics["qpos"][3]
        + torch.nn.functional.pad(torch.tensor([1.0]), (0, 20)),
    )
    assert torch.equal(env.physics["qpos"][4], env.initial_qpos)
    assert not env.physics["qvel"][4].any() and env.physics["time"][4] == 0
    assert torch.equal(
        result["terminal_observations"]["critic"][3],
        result["next_observations"]["critic"][3],
    )
    assert not torch.equal(
        result["terminal_observations"]["critic"][4],
        result["next_observations"]["critic"][4],
    )
    assert collector.storage.step == 1


def test_timeout_bootstrap_is_selected_pre_reset_and_once(
    synthetic_collector, monkeypatch
):
    collector = synthetic_collector
    # Make a natural timeout reachable in two synthetic steps while preserving
    # the production 10-substep bound. Row 0 fails/reset on step 1; rows 1..63
    # reach the shortened test-only threshold on step 2.
    monkeypatch.setattr(transition, "EPISODE_STEPS", 20)
    collector.env.plan[1] = {"terminated": [0], "terminated_executed": 3}
    collector.env.plan[2] = {"timed_out": list(range(1, 64)), "timeout_clock": 20}
    first = collector.collect_one()
    assert not first["timed_out"].any()

    calls, values = [], []
    original_forward = collector.critic.forward

    def counted_forward(obs, *args, **kwargs):
        calls.append({k: v.detach().clone() for k, v in obs.items()})
        value = original_forward(obs, *args, **kwargs)
        values.append(value.detach().clone())
        return value

    monkeypatch.setattr(collector.critic, "forward", counted_forward)
    result = collector.collect_one()
    assert result["timed_out"].sum().item() == 63
    assert len(calls) == 2  # PPO pre-action critic plus selected timeout bootstrap.
    assert calls[0]["actor"].shape == (64, 44)
    assert calls[1]["actor"].shape == (63, 44)
    assert torch.equal(
        calls[1]["critic"], result["terminal_observations"]["critic"][1:]
    )
    assert not torch.equal(
        calls[1]["critic"], result["next_observations"]["critic"][1:]
    )
    assert torch.equal(result["timeout_terminal_values"][0], torch.zeros(1))
    assert torch.equal(result["timeout_terminal_values"][1:], values[1])
    assert torch.allclose(
        result["learner_reward"],
        result["environment_reward"]
        + collector.algorithm.gamma * result["timeout_terminal_values"][:, 0],
    )
    assert (
        collector.storage.step == 2 and collector.algorithm.transition.actions is None
    )


def test_bad_lease_precedes_any_prepared_or_device_access(monkeypatch):
    calls = []

    def refuse(fd):
        calls.append(("lease", fd))
        raise ValueError("synthetic lease refusal")

    monkeypatch.setattr(transition.training_smoke, "inherited_lease", refuse)
    monkeypatch.setattr(
        sampling, "_validate_receipt", lambda *_: pytest.fail("before lease")
    )
    with pytest.raises(ValueError, match="lease refusal"):
        transition.CudaTransitionCollector({}, object(), lease_fd=LEASE_FD)
    assert calls == [("lease", LEASE_FD)]


def test_unpatched_production_policy_guard_rejects_cpu_prepared_objects(
    synthetic_collector, monkeypatch
):
    collector = synthetic_collector
    prepared = {
        "actor": collector.actor,
        "critic": collector.critic,
        "algorithm": collector.algorithm,
        "storage": collector.storage,
        "receipt": {"source": SOURCE, "learner_seed": 653},
        "private_cuda_generator": SimpleNamespace(device="cuda:0"),
        "private_cuda_rng_state": collector.scope.state,
        "caller_rng_states": {},
    }
    monkeypatch.setattr(
        sampling,
        "_validate_receipt",
        lambda _r, p: (SOURCE, 653, p["private_cuda_rng_state"]),
    )
    monkeypatch.setattr(sampling, "_check_cuda0_ready", lambda: None)
    monkeypatch.setattr(preparation, "_check_rsl_sources", lambda: None)
    monkeypatch.setattr(transition.checkpoint, "runtime_check", lambda: None)
    monkeypatch.setattr(preparation, "_check_policy", PRODUCTION_CHECK_POLICY)
    with pytest.raises(ValueError, match="stock PPO owns the exact transferred models"):
        transition.CudaTransitionCollector(prepared, collector.env, lease_fd=LEASE_FD)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda e: e.live.__setitem__(0, False), "all fresh CUDA64 rows live"),
        (lambda e: e.steps.__setitem__(0, 1), "fresh zero-clock CUDA64 runtime"),
    ],
)
def test_constructor_rejects_stale_or_closed_initial_rows(
    synthetic_collector, mutate, match
):
    collector = synthetic_collector
    env = SyntheticScheduledRuntime()
    mutate(env)
    with pytest.raises(ValueError, match=match):
        transition.CudaTransitionCollector(
            {
                "actor": collector.actor,
                "critic": collector.critic,
                "algorithm": collector.algorithm,
                "storage": collector.storage,
                "receipt": {"source": SOURCE, "learner_seed": 653},
                "private_cuda_generator": SimpleNamespace(device="cuda:0"),
                "private_cuda_rng_state": collector.scope.state,
                "caller_rng_states": {},
            },
            env,
            lease_fd=LEASE_FD,
        )


def _prepared_copy(collector, *, source=SOURCE):
    return {
        "actor": collector.actor,
        "critic": collector.critic,
        "algorithm": collector.algorithm,
        "storage": collector.storage,
        "receipt": {"source": source, "learner_seed": 653},
        "private_cuda_generator": SimpleNamespace(device="cuda:0"),
        "private_cuda_rng_state": collector.scope.state,
        "caller_rng_states": {},
    }


def test_constructor_rejects_prepared_source_mismatch(synthetic_collector):
    collector = synthetic_collector
    with pytest.raises(ValueError, match="prepared source mismatch"):
        transition.CudaTransitionCollector(
            _prepared_copy(collector, source="b" * 40),
            SyntheticScheduledRuntime(),
            lease_fd=LEASE_FD,
        )


def test_constructor_rejects_schedule_from_another_source(synthetic_collector):
    collector = synthetic_collector
    env = SyntheticScheduledRuntime()
    cells = lesson.cells("gentle", held_out=False)
    env.schedule_declaration = schedule.declaration(
        "b" * 40, "gentle", "training", [cells[0]["id"]] * 64
    )
    env._schedule_sha256 = schedule.binding_sha256(env.schedule_declaration)
    with pytest.raises(ValueError, match="exact source-bound CUDA64 row declaration"):
        transition.CudaTransitionCollector(
            _prepared_copy(collector), env, lease_fd=LEASE_FD
        )


@pytest.mark.parametrize("drift", ["parent", "fixed-schedule"])
def test_collection_rejects_parent_or_fixed_schedule_drift(
    synthetic_collector, monkeypatch, drift
):
    collector = synthetic_collector
    if drift == "parent":
        monkeypatch.setattr(transition.checkpoint, "state_hash", lambda _s: "f" * 64)
        expected = "unchanged actual D1 parent during no-update collection"
    else:
        cells = lesson.cells("gentle", held_out=False)
        other = cells[1]["id"]
        collector.env.schedule_declaration = schedule.declaration(
            SOURCE, "gentle", "training", [other] * 64
        )
        collector.env._schedule_sha256 = schedule.binding_sha256(
            collector.env.schedule_declaration
        )
        expected = "unchanged healthy scheduled runtime and fixed row map"
    with pytest.raises(ValueError, match=expected):
        collector.collect_one()
    assert collector.faulted and collector.env.calls == 0


@pytest.mark.parametrize(
    "corrupt,match",
    [
        ("prefix", "actor/critic observation prefix"),
        ("nonfinite", "finite CUDA0 transition actor observations"),
        ("dtype", "typed CUDA0 transition actor observations"),
    ],
)
def test_bad_observations_fault_before_sampling(synthetic_collector, corrupt, match):
    collector = synthetic_collector
    obs = collector.env.observations()
    if corrupt == "prefix":
        obs["critic"][0, 0] = 3
    elif corrupt == "nonfinite":
        obs["actor"][0, 0] = float("nan")
    else:
        obs["actor"] = obs["actor"].to(torch.int64)
    collector.env._terminal_observations = obs
    expected_error = AssertionError if corrupt in ("nonfinite", "dtype") else ValueError
    actual_match = "actor observations" if corrupt in ("nonfinite", "dtype") else match
    with pytest.raises(expected_error, match=actual_match):
        collector.collect_one()
    assert (
        collector.faulted and collector.env.calls == 0 and collector.storage.step == 0
    )


@pytest.mark.parametrize(
    "corrupt,match",
    [
        ("exclusive", "exclusive failure and timeout"),
        ("live", "pre-reset live"),
        ("clock", "pre-reset live"),
        ("record", "terminal record exactly"),
    ],
)
def test_malformed_runtime_result_faults_before_storage(
    synthetic_collector, corrupt, match
):
    collector = synthetic_collector
    original = collector.env.step_with_schedule

    def malformed(actions, *, capture_control=False):
        result = original(actions, capture_control=capture_control)
        if corrupt == "exclusive":
            result["terminated"][0] = result["timed_out"][0] = True
        elif corrupt == "live":
            result["live"][0] = False
        elif corrupt == "clock":
            result["episode_steps"][0] += 1
        else:
            result["terminal_records"][0] = None
        return result

    collector.env.step_with_schedule = malformed
    if corrupt == "record":
        collector.env.plan[1] = {"terminated": [0], "terminated_executed": 3}
    with pytest.raises(ValueError, match=match):
        collector.collect_one()
    assert collector.faulted and collector.storage.step == 0


@pytest.mark.parametrize(
    "corrupt,expected_error,match",
    [
        ("reward-nan", AssertionError, "reward"),
        ("timeout-dtype", AssertionError, "timed_out"),
        ("missing-live", KeyError, "live"),
    ],
)
def test_runtime_result_dtype_mask_and_finite_are_required(
    synthetic_collector, corrupt, expected_error, match
):
    collector = synthetic_collector
    original = collector.env.step_with_schedule

    def malformed(actions, *, capture_control=False):
        result = original(actions, capture_control=capture_control)
        if corrupt == "reward-nan":
            result["reward"][0] = float("nan")
        elif corrupt == "timeout-dtype":
            result["timed_out"] = result["timed_out"].to(torch.int64)
        else:
            result.pop("live")
        return result

    collector.env.step_with_schedule = malformed
    with pytest.raises(expected_error, match=match):
        collector.collect_one()
    assert collector.faulted and collector.storage.step == 0


@pytest.mark.parametrize(
    "drift,match",
    [
        ("qpos", "untouched physical qpos"),
        ("qvel", "untouched physical qvel"),
        ("time", "untouched physical time"),
        ("control", "selective reset preserves/restores control rows"),
        ("observation", "untouched observation rows preserved"),
        ("done-qpos", "reset physical rows restore exact initial qpos"),
        ("done-qvel", "reset physical rows restore exact initial qpos"),
        ("done-time", "reset physical rows restore exact initial qpos"),
    ],
)
def test_selective_reset_rejects_untouched_drift_and_bad_reset_physics(
    synthetic_collector, drift, match
):
    collector = synthetic_collector
    collector.env.plan[1] = {"terminated": [0], "terminated_executed": 3}
    collector.env.reset_drift = drift
    with pytest.raises(ValueError, match=match):
        collector.collect_one()
    assert collector.faulted and collector.storage.step == 1


def test_production_cuda_tensor_guard_refuses_cpu_without_seam():
    with pytest.raises(ValueError, match="typed CUDA0 transition actor observations"):
        transition._tensor(torch.zeros(64, 44), (64, 44), "actor observations")


@pytest.mark.parametrize("stage", ["act", "physics", "critic", "store", "reset"])
def test_failure_at_each_transition_stage_fault_latches_no_retry(
    synthetic_collector, monkeypatch, stage
):
    collector = synthetic_collector
    env, algorithm = collector.env, collector.algorithm
    if stage == "act":
        monkeypatch.setattr(
            collector.actor,
            "forward",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("synthetic act")),
        )
    elif stage == "physics":
        env.plan[1] = {"raise": True}
    elif stage == "critic":
        monkeypatch.setattr(transition, "EPISODE_STEPS", 10)
        env.plan[1] = {"timed_out": list(range(64)), "timeout_clock": 10}
        original_forward = collector.critic.forward
        count = 0

        def fail_terminal(obs, *args, **kwargs):
            nonlocal count
            count += 1
            if count > 1:
                raise RuntimeError("synthetic terminal critic")
            return original_forward(obs, *args, **kwargs)

        monkeypatch.setattr(collector.critic, "forward", fail_terminal)
    elif stage == "store":
        monkeypatch.setattr(transition, "_check_stock_step", lambda _algorithm: None)
        algorithm.process_env_step = lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("synthetic store")
        )
    else:
        env.plan[1] = {"terminated": [0], "terminated_executed": 3}

        def fail_reset(rows):
            assert collector.storage.step == 1
            raise RuntimeError("synthetic reset")

        env.reset = fail_reset

    with pytest.raises(RuntimeError, match="synthetic"):
        collector.collect_one()
    assert collector.faulted and collector.phase == "faulted"
    stored = collector.storage.step
    with pytest.raises(ValueError, match="faulted CUDA collector cannot retry"):
        collector.collect_one()
    assert collector.storage.step == stored
    assert env.calls == 0 if stage == "act" else env.calls == 1


def test_internal_transition_counter_rejects_injected_storage_step(synthetic_collector):
    collector = synthetic_collector
    collector.collect_one()
    collector.transitions = 0
    with pytest.raises(ValueError, match="bounded no-update CUDA collection phase"):
        collector.collect_one()
    assert collector.faulted


@pytest.mark.parametrize(
    "method,match",
    [
        ("act", "exact pinned stock PPO.act bound method"),
        ("process_env_step", "exact pinned stock PPO.process_env_step bound method"),
    ],
)
def test_bound_stock_sampler_and_store_methods_cannot_be_replaced(
    synthetic_collector, method, match
):
    collector = synthetic_collector
    setattr(collector.algorithm, method, lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match=match):
        collector.collect_one()
    assert (
        collector.faulted and collector.env.calls == 0 and collector.storage.step == 0
    )


def test_twenty_eight_step_bound_never_updates(synthetic_collector, monkeypatch):
    collector = synthetic_collector
    # The public pinned method guard forbids replacing `act`; count environment calls.
    for _ in range(28):
        collector.collect_one()
    assert collector.env.calls == 28
    assert collector.storage.step == collector.transitions == 28
    assert not collector.algorithm.optimizer.state
    with pytest.raises(ValueError, match="bounded no-update CUDA collection phase"):
        collector.collect_one()
    assert collector.storage.step == 28
    assert SYNTHETIC_BACKEND == "SYNTHETIC_CPU_RSL_NO_CUDA"
