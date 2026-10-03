"""SYNTHETIC_CPU_RSL tests for the CUDA shadow-sampling source contract.

These tests use stock CPU RSL objects and a CPU RNG substitute for CUDA0. They
exercise the bounded Python contract only and are not CUDA or sampler evidence.
"""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_ppo
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck import stance_recovery_cuda_rng_scope as rng_scope
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_cpu_replay_profile as cpu_profile

SYNTHETIC_BACKEND = "SYNTHETIC_CPU_RSL_NO_CUDA"
SOURCE = "a" * 40
LEASE_FD = 19


def digest_state(state):
    return sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _cpu_parent_receipt(source):
    receipt = {
        "protocol": parent.PROTOCOL,
        "parent_checkpoint_sha256": preparation.baseline.CHECKPOINT_SHA256,
        "parent_identity": parent.expected_identity(),
        "parent_state_sha256": parent.PARENT_STATE_SHA256,
        "cpu_math_profile": cpu_profile.expected_receipt(),
        "strict_actor_restore": True,
        "strict_critic_restore": True,
        "models_trainable": True,
        "weights_only": True,
        "fresh_optimizer_required": True,
        "optimizer_created": False,
        "optimizer_restored": False,
        "simulator_restored": False,
        "storage_restored": False,
        "rng_restored": False,
        "normalization_restored": False,
        "historical_archive_authenticated_by_loader": False,
        "installed_in_learner": False,
        "execution_admitted": False,
        **preparation.FALSE_FLAGS,
    }
    receipt_hash = preparation._cpu_parent_receipt_digest(receipt)
    binding = preparation.cpu_parent_binding(source, receipt_hash)
    return receipt, receipt_hash, binding


class SyntheticCpuScope:
    """Synthetic CPU state substitution; never represents a CUDA generator."""

    def __init__(self, seed, state, *, lease_fd):
        assert seed in preparation.SEEDS and lease_fd == LEASE_FD
        self._source = SOURCE
        self._state = state.detach().cpu().clone()
        self._count = 0
        self._active = False
        self._faulted = False
        self._cpu_ok = False
        self._cuda_ok = False

    @property
    def state(self):
        return self._state.detach().clone()

    @property
    def receipt(self):
        return {
            "source": self._source,
            "scope_count": self._count,
            "scope_active": self._active,
            "faulted": self._faulted,
            "caller_cpu_rng_preserved": self._cpu_ok,
            "caller_cuda_rng_preserved": self._cuda_ok,
            "private_state_sha256": digest_state(self._state),
        }

    @contextmanager
    def scope(self):
        cpu = torch.random.get_rng_state().clone()
        self._active = True
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(self._state)
                try:
                    yield
                finally:
                    self._state = torch.random.get_rng_state().clone()
        except BaseException:
            self._faulted = True
            raise
        finally:
            self._active = False
            self._cpu_ok = torch.equal(cpu, torch.random.get_rng_state())
            self._cuda_ok = True
        self._count += 1


@pytest.fixture
def synthetic_cpu_rsl(monkeypatch):
    """Build real CPU PPO/storage with narrowly mocked device/provenance seams."""
    original_generator = torch.Generator
    monkeypatch.setattr(sampling.training_smoke, "inherited_lease", lambda fd: None)
    monkeypatch.setattr(sampling, "_check_cuda0_ready", lambda: None)
    actor, critic = checkpoint.fresh_models(seed=521)
    input_obs = TensorDict(
        {
            "actor": torch.zeros(64, 44),
            "critic": torch.zeros(64, 50),
        },
        [64],
    )
    storage = RolloutStorage("rl", 64, 28, input_obs, (10,), device="cpu")
    algorithm = PPO(actor, critic, storage, **stance_ppo.CONFIG, device="cpu")
    private_generator = original_generator(device="cpu").manual_seed(653)
    private_state = private_generator.get_state().detach().cpu().clone()
    cpu_before = torch.random.get_rng_state().clone()
    caller_hash = digest_state(cpu_before)
    cuda_before = torch.tensor([1, 2, 3], dtype=torch.uint8)
    cuda_hash = digest_state(cuda_before)
    cpu_receipt, cpu_hash, caller_binding = _cpu_parent_receipt(SOURCE)
    states = checkpoint.states_of(actor, critic)
    state_hash = checkpoint.state_hash(states)
    source_identity = {
        "source": SOURCE,
        "execution_profile": deepcopy(preparation.execution.PROFILE),
    }
    receipt = {
        "protocol": preparation.PROTOCOL,
        "source": SOURCE,
        "source_identity": source_identity,
        "parent_checkpoint_sha256": preparation.baseline.CHECKPOINT_SHA256,
        "parent_identity": parent.expected_identity(),
        "parent_state_sha256": parent.PARENT_STATE_SHA256,
        "cpu_parent_receipt_sha256": cpu_hash,
        "caller_binding": caller_binding,
        "cpu_parent_receipt": cpu_receipt,
        "cpu_provenance_authentication_required": True,
        "native_cpu_provenance_authenticated_by_this_function": False,
        "learner_seed": 653,
        "worlds": 64,
        "horizon": 28,
        "device": "cuda:0",
        "actor_device": "cuda:0",
        "critic_device": "cuda:0",
        "state_sha256_before_transfer": parent.PARENT_STATE_SHA256,
        "state_sha256_after_transfer": parent.PARENT_STATE_SHA256,
        "actor_parameter_tensors": sum(1 for _ in actor.parameters()),
        "critic_parameter_tensors": sum(1 for _ in critic.parameters()),
        "ppo_config": deepcopy(stance_ppo.CONFIG),
        "ppo_source_sha256": deepcopy(stance_ppo.PINS),
        "storage_step": 0,
        "storage_empty": True,
        "optimizer_type": "Adam",
        "optimizer_state_entries": 0,
        "optimizer_empty": True,
        "private_cuda_generator_device": "cuda:0",
        "private_cuda_rng_state_sha256": digest_state(private_state),
        "private_cuda_rng_connected_to_sampler": False,
        "caller_cpu_rng_unchanged": True,
        "caller_cuda_rng_unchanged": True,
        "caller_rng_state_sha256": {
            "cpu_before": caller_hash,
            "cpu_after": caller_hash,
            "cuda_before": cuda_hash,
            "cuda_after": cuda_hash,
        },
        "cuda_initialized": True,
        "simulator_created": False,
        "rollout_collected": False,
        "optimizer_steps": 0,
        "training_update_performed": False,
        "finite_optimizer_step_qualified": False,
        "transition_bridge_qualified": False,
        "schedule_installed": False,
        "student_export_available": False,
        "training_job_predeclared": False,
        "execution_admitted": False,
        **preparation.FALSE_FLAGS,
    }
    assert state_hash != parent.PARENT_STATE_SHA256  # synthetic weights are not D1.
    prepared = {
        "actor": actor,
        "critic": critic,
        "algorithm": algorithm,
        "storage": storage,
        "private_cuda_generator": type(
            "SyntheticGenerator",
            (),
            {
                "device": "cuda:0",
                "get_state": lambda self: private_state.clone(),
            },
        )(),
        "private_cuda_rng_state": private_state,
        "caller_rng_states": {
            "cpu_before": cpu_before.clone(),
            "cpu_after": cpu_before.clone(),
            "cuda_before": cuda_before.clone(),
            "cuda_after": cuda_before.clone(),
        },
        "receipt": receipt,
    }

    def check_policy(a, c, algo, st):
        assert (
            type(algo) is PPO
            and algo.actor is a
            and algo.critic is c
            and algo.storage is st
        )
        assert str(algo.device) == "cpu" and not algo.optimizer.state
        assert algo.transition.actions is None and st.step == 0

    def check_storage(st):
        assert st.training_type == "rl" and st.num_envs == 64
        assert st.num_transitions_per_env == 28 and st.step == 0
        for name in (
            "actions",
            "rewards",
            "dones",
            "values",
            "actions_log_prob",
            "returns",
            "advantages",
        ):
            assert not torch.count_nonzero(getattr(st, name))

    monkeypatch.setattr(preparation, "_check_policy", check_policy)
    monkeypatch.setattr(preparation, "_check_storage", check_storage)
    monkeypatch.setattr(
        sampling.checkpoint, "state_hash", lambda _states: parent.PARENT_STATE_SHA256
    )
    monkeypatch.setattr(rng_scope, "CudaPrivateRngScope", SyntheticCpuScope)
    monkeypatch.setattr(sampling, "_check_cuda0_ready", lambda: None)
    monkeypatch.setattr(
        sampling,
        "_caller_cuda_state",
        lambda: cuda_before.clone(),
    )
    monkeypatch.setattr(
        sampling, "_current_cuda_state", lambda: torch.random.get_rng_state().clone()
    )
    monkeypatch.setattr(
        sampling,
        "_shadow_observations",
        lambda: TensorDict(
            {
                "actor": torch.zeros(64, 44),
                "critic": torch.zeros(64, 50),
            },
            [64],
        ),
    )

    def cpu_float(value, shape, label, *, positive=False):
        assert torch.is_tensor(value) and tuple(value.shape) == shape, label
        assert value.dtype == torch.float32 and value.device.type == "cpu", label
        assert torch.isfinite(value).all(), label
        if positive:
            assert (value > 0).all(), label

    monkeypatch.setattr(sampling, "_require_cuda_float", cpu_float)

    def check_cpu_observations(observations):
        assert tuple(observations.batch_size) == (64,)
        actor_obs, critic_obs = observations["actor"], observations["critic"]
        assert actor_obs.shape == (64, 44) and critic_obs.shape == (64, 50)
        assert actor_obs.dtype == critic_obs.dtype == torch.float32
        assert torch.isfinite(actor_obs).all() and torch.isfinite(critic_obs).all()
        assert not torch.count_nonzero(actor_obs) and not torch.count_nonzero(
            critic_obs
        )
        assert torch.equal(actor_obs, critic_obs[:, :44])

    monkeypatch.setattr(sampling, "_check_shadow_observations", check_cpu_observations)
    monkeypatch.setattr(preparation, "_check_rsl_sources", lambda: None)
    yield prepared
    assert SYNTHETIC_BACKEND == "SYNTHETIC_CPU_RSL_NO_CUDA"


def test_synthetic_cpu_rsl_samples_exactly_28_and_keeps_no_rollout(
    monkeypatch, synthetic_cpu_rsl
):
    prepared = synthetic_cpu_rsl
    original_private = prepared["private_cuda_rng_state"].clone()
    original_forward = prepared["actor"].forward
    calls = []

    def counted_forward(*args, **kwargs):
        calls.append(args[0])
        return original_forward(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("shadow sampler must not process a transition or update")

    monkeypatch.setattr(prepared["actor"], "forward", counted_forward)
    monkeypatch.setattr(prepared["algorithm"], "process_env_step", forbidden)
    monkeypatch.setattr(prepared["algorithm"], "update", forbidden)
    monkeypatch.setattr(prepared["algorithm"].optimizer, "step", forbidden)
    result = sampling.sample_shadow(prepared, lease_fd=LEASE_FD)
    receipt = result["receipt"]
    assert len(calls) == 28
    assert tuple(result["actions"].shape) == (28, 64, 10)
    assert tuple(result["values"].shape) == (28, 64, 1)
    assert tuple(result["actions_log_prob"].shape) == (28, 64)
    assert tuple(result["distribution_mean"].shape) == (28, 64, 10)
    assert tuple(result["distribution_std"].shape) == (28, 64, 10)
    assert tuple(result["private_cuda_state_boundaries"][0].shape) == tuple(
        result["private_cuda_state_boundaries"][-1].shape
    )
    assert len(result["private_cuda_state_boundaries"]) == 29
    assert len(set(receipt["private_state_boundary_sha256"])) == 29
    result["private_cuda_state_boundaries"][0].zero_()
    assert torch.equal(prepared["private_cuda_rng_state"], original_private)
    assert torch.equal(prepared["private_cuda_generator"].get_state(), original_private)
    assert receipt["stock_sampler_calls"] == 28
    assert receipt["synthetic_zero_observations"] is True
    assert receipt["physical_observations"] is False
    assert receipt["private_cuda_generator_connected_to_sampler"] is False
    assert receipt["private_state_installed_around_stock_sampler"] is True
    assert "private_cuda_rng_connected_to_sampler" not in receipt
    assert receipt["native_cuda_sampling_qualified"] is False
    assert all(
        value is False
        for key, value in receipt.items()
        if key in preparation.FALSE_FLAGS
    )
    assert prepared["algorithm"].storage.step == 0
    assert prepared["algorithm"].transition.actions is None
    assert not prepared["algorithm"].optimizer.state
    assert result["caller_rng_states"]["cpu_before"].equal(
        result["caller_rng_states"]["cpu_after"]
    )
    assert result["caller_rng_states"]["cuda_before"].equal(
        result["caller_rng_states"]["cuda_after"]
    )
    assert receipt["caller_rng_state_sha256"] == {
        name: digest_state(state) for name, state in result["caller_rng_states"].items()
    }
    result["caller_rng_states"]["cpu_before"].zero_()
    assert torch.equal(
        prepared["caller_rng_states"]["cpu_before"],
        prepared["caller_rng_states"]["cpu_after"],
    )
    assert all(
        not torch.count_nonzero(getattr(prepared["storage"], name))
        for name in ("actions", "rewards", "dones", "values", "returns", "advantages")
    )


def test_bad_lease_precedes_device_or_prepared_object_access(
    monkeypatch, synthetic_cpu_rsl
):
    called = []

    def reject(_fd):
        called.append("lease")
        raise ValueError("synthetic lease refusal")

    monkeypatch.setattr(sampling.training_smoke, "inherited_lease", reject)
    monkeypatch.setattr(
        sampling, "_shadow_observations", lambda: pytest.fail("allocated before lease")
    )
    with pytest.raises(ValueError, match="lease refusal"):
        sampling.sample_shadow({}, lease_fd=LEASE_FD)
    assert called == ["lease"]


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r.update(worlds=63), "fixed CUDA0 64-world"),
        (lambda r: r.update(device="cuda:1"), "fixed CUDA0 64-world"),
        (lambda r: r.update(actor_parameter_tensors=0), "parameter counts match"),
        (lambda r: r.update(parent_state_sha256="f" * 64), "exact D1 parent"),
        (lambda r: r.update(private_cuda_rng_connected_to_sampler=True), "unconnected"),
        (lambda r: r.update(optimizer_steps=1), "non-learning preparation"),
        (
            lambda r: r.update(private_cuda_rng_state_sha256="f" * 64),
            "prepared private CUDA RNG state",
        ),
        (lambda r: r.update(recovery_accepted=True), "all recovery admission flags"),
    ],
)
def test_synthetic_malformed_receipt_or_state_refused(
    monkeypatch, synthetic_cpu_rsl, change, match
):
    prepared = synthetic_cpu_rsl
    prepared["receipt"] = deepcopy(prepared["receipt"])
    change(prepared["receipt"])
    with pytest.raises(ValueError, match=match):
        sampling.sample_shadow(prepared, lease_fd=LEASE_FD)


def test_synthetic_malformed_private_state_layout_refused(synthetic_cpu_rsl):
    prepared = synthetic_cpu_rsl
    prepared["private_cuda_rng_state"] = torch.empty((0,), dtype=torch.uint8)
    with pytest.raises(
        ValueError, match="exact detached prepared private CUDA RNG state"
    ):
        sampling.sample_shadow(prepared, lease_fd=LEASE_FD)


@pytest.mark.parametrize("extra", [False, True])
def test_synthetic_preparation_receipt_schema_is_exact(synthetic_cpu_rsl, extra):
    prepared = synthetic_cpu_rsl
    prepared["receipt"] = deepcopy(prepared["receipt"])
    if extra:
        prepared["receipt"]["unexpected"] = True
    else:
        prepared["receipt"].pop("protocol")
    with pytest.raises(ValueError, match="exact policy-preparation receipt schema"):
        sampling.sample_shadow(prepared, lease_fd=LEASE_FD)


def test_synthetic_instance_act_replacement_is_refused(synthetic_cpu_rsl, monkeypatch):
    algorithm = synthetic_cpu_rsl["algorithm"]
    monkeypatch.setattr(algorithm, "act", lambda _obs: torch.zeros(64, 10))
    with pytest.raises(ValueError, match="exact pinned stock PPO.act"):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_nonfinite_action_is_refused(monkeypatch, synthetic_cpu_rsl):
    actor = synthetic_cpu_rsl["actor"]
    monkeypatch.setattr(
        actor.mlp,
        "forward",
        lambda *_args, **_kwargs: torch.full((64, 10), float("nan")),
    )
    with pytest.raises(AssertionError):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_nonfinite_logprob_is_refused(monkeypatch, synthetic_cpu_rsl):
    actor = synthetic_cpu_rsl["actor"]
    monkeypatch.setattr(
        actor,
        "get_output_log_prob",
        lambda _actions: torch.full((64,), float("nan")),
    )
    with pytest.raises(AssertionError):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_nonpositive_distribution_std_is_refused(
    monkeypatch, synthetic_cpu_rsl
):
    actor = synthetic_cpu_rsl["actor"]
    with torch.no_grad():
        actor.distribution.std_param.zero_()
    monkeypatch.setattr(actor, "get_output_log_prob", lambda _actions: torch.zeros(64))
    with pytest.raises(AssertionError):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_private_state_must_advance_each_call(monkeypatch, synthetic_cpu_rsl):
    state = synthetic_cpu_rsl["private_cuda_rng_state"].clone()
    monkeypatch.setattr(sampling, "_current_cuda_state", lambda: state.clone())
    with pytest.raises(ValueError, match="advances on every stock sampler call"):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_observation_mutation_is_refused(monkeypatch, synthetic_cpu_rsl):
    actor = synthetic_cpu_rsl["actor"]
    original_forward = actor.forward

    def mutate_observations(obs, *args, **kwargs):
        obs["actor"][0, 0] = 1
        return original_forward(obs, *args, **kwargs)

    monkeypatch.setattr(actor, "forward", mutate_observations)
    with pytest.raises(AssertionError):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_storage_mutation_is_refused(monkeypatch, synthetic_cpu_rsl):
    actor = synthetic_cpu_rsl["actor"]
    original_forward = actor.forward

    def mutate_storage(*args, **kwargs):
        synthetic_cpu_rsl["storage"].actions[0, 0, 0] = 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(actor, "forward", mutate_storage)
    with pytest.raises(AssertionError):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_model_mutation_is_refused(monkeypatch, synthetic_cpu_rsl):
    actor = synthetic_cpu_rsl["actor"]
    original_forward = actor.forward
    changed = False

    def mutate_parameter(*args, **kwargs):
        nonlocal changed
        if not changed:
            with torch.no_grad():
                next(actor.parameters()).add_(1)
            changed = True
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(actor, "forward", mutate_parameter)
    hashes = iter((parent.PARENT_STATE_SHA256, "f" * 64))
    monkeypatch.setattr(sampling.checkpoint, "state_hash", lambda _state: next(hashes))
    with pytest.raises(ValueError, match="weights unchanged"):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_caller_cuda_rng_change_is_refused(monkeypatch, synthetic_cpu_rsl):
    original_state = synthetic_cpu_rsl["caller_rng_states"]["cuda_after"].clone()
    changed_state = original_state + 1
    calls = 0

    def drifting_state():
        nonlocal calls
        calls += 1
        return original_state.clone() if calls == 1 else changed_state.clone()

    monkeypatch.setattr(sampling, "_caller_cuda_state", drifting_state)
    with pytest.raises(ValueError, match="caller CPU and CUDA0 RNG streams unchanged"):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)


def test_synthetic_sampler_exception_propagates_and_clears_only_transition(
    monkeypatch, synthetic_cpu_rsl
):
    algorithm = synthetic_cpu_rsl["algorithm"]
    actor = synthetic_cpu_rsl["actor"]
    original_forward = actor.forward
    calls = 0

    def fail_on_fourth(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic act failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(actor, "forward", fail_on_fourth)
    with pytest.raises(RuntimeError, match="synthetic act failure"):
        sampling.sample_shadow(synthetic_cpu_rsl, lease_fd=LEASE_FD)
    assert calls == 4
    assert algorithm.transition.actions is None
    assert algorithm.storage.step == 0
    assert not algorithm.optimizer.state
