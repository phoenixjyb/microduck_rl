"""Bounded shadow sampling through the stock RSL CUDA PPO sampler.

This callable is intentionally not a rollout bridge: its fixed-zero inputs are
synthetic, it retains sampler outputs, and it never stores transitions, steps
an environment, computes returns, updates an optimizer, or admits training.
"""

from copy import deepcopy
from hashlib import sha256
import os
import re

import torch
from tensordict import TensorDict

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_ppo
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_cuda_rng_scope as rng_scope
from mjlab_microduck import stance_training_smoke as training_smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cuda64-shadow-sampling-v1"
WORLDS, HORIZON, DEVICE = 64, 28, "cuda:0"
FALSE_FLAGS = dict(preparation.FALSE_FLAGS)
_PREP_RECEIPT_KEYS = set(
    """protocol source source_identity parent_checkpoint_sha256 parent_identity
    parent_state_sha256 cpu_parent_receipt_sha256 caller_binding cpu_parent_receipt
    cpu_provenance_authentication_required native_cpu_provenance_authenticated_by_this_function
    learner_seed worlds horizon device actor_device critic_device state_sha256_before_transfer
    state_sha256_after_transfer actor_parameter_tensors critic_parameter_tensors ppo_config
    ppo_source_sha256 storage_step storage_empty optimizer_type optimizer_state_entries
    optimizer_empty private_cuda_generator_device private_cuda_rng_state_sha256
    private_cuda_rng_connected_to_sampler caller_cpu_rng_unchanged caller_cuda_rng_unchanged
    caller_rng_state_sha256 cuda_initialized simulator_created rollout_collected optimizer_steps
    training_update_performed finite_optimizer_step_qualified transition_bridge_qualified
    schedule_installed student_export_available training_job_predeclared execution_admitted""".split()
) | set(FALSE_FLAGS)


def _state_digest(state):
    return sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _validate_receipt(receipt, prepared):
    require(
        type(receipt) is dict and set(receipt) == _PREP_RECEIPT_KEYS,
        "exact policy-preparation receipt schema",
    )
    require(
        receipt["protocol"] == preparation.PROTOCOL, "exact policy-preparation protocol"
    )
    source = receipt["source"]
    require(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
        "exact prepared source revision",
    )
    require(
        type(receipt["source_identity"]) is dict
        and receipt["source_identity"].get("source") == source
        and receipt["source_identity"].get("execution_profile")
        == preparation.execution.PROFILE,
        "prepared source identity binding",
    )
    require(
        receipt["parent_checkpoint_sha256"] == preparation.baseline.CHECKPOINT_SHA256
        and receipt["parent_state_sha256"] == preparation.parent.PARENT_STATE_SHA256
        and canonical(receipt["parent_identity"])
        == canonical(preparation.parent.expected_identity()),
        "exact D1 parent identity and state",
    )
    require(
        receipt["cpu_provenance_authentication_required"] is True
        and receipt["native_cpu_provenance_authenticated_by_this_function"] is False,
        "caller-authenticated CPU preparation boundary retained",
    )
    binding = preparation.validate_cpu_parent_receipt(
        receipt["cpu_parent_receipt"],
        receipt["cpu_parent_receipt_sha256"],
        receipt["caller_binding"],
        source=source,
    )
    require(
        canonical(binding) == canonical(receipt["caller_binding"]),
        "exact source-bound CPU parent receipt binding",
    )
    require(
        receipt["learner_seed"] in preparation.SEEDS
        and type(receipt["learner_seed"]) is int
        and receipt["worlds"] == WORLDS
        and receipt["horizon"] == HORIZON
        and receipt["device"] == DEVICE
        and receipt["actor_device"] == DEVICE
        and receipt["critic_device"] == DEVICE,
        "fixed CUDA0 64-world 28-call prepared profile",
    )
    require(
        receipt["state_sha256_before_transfer"]
        == preparation.parent.PARENT_STATE_SHA256
        and receipt["state_sha256_after_transfer"]
        == preparation.parent.PARENT_STATE_SHA256,
        "unchanged D1 state across transfer",
    )
    require(
        receipt["ppo_config"] == stance_ppo.CONFIG
        and receipt["ppo_source_sha256"] == stance_ppo.PINS,
        "pinned stock RSL PPO configuration and sources",
    )
    require(
        receipt["storage_step"] == 0
        and receipt["storage_empty"] is True
        and receipt["optimizer_type"] == "Adam"
        and receipt["optimizer_state_entries"] == 0
        and receipt["optimizer_empty"] is True,
        "fresh empty stock storage and Adam before sampling",
    )
    require(
        receipt["private_cuda_generator_device"] == DEVICE
        and receipt["private_cuda_rng_connected_to_sampler"] is False,
        "separate unconnected private CUDA generator",
    )
    require(
        receipt["caller_cpu_rng_unchanged"] is True
        and receipt["caller_cuda_rng_unchanged"] is True
        and receipt["cuda_initialized"] is True,
        "prepared caller RNG and initialized CUDA evidence",
    )
    caller_hashes = receipt["caller_rng_state_sha256"]
    require(
        type(caller_hashes) is dict
        and set(caller_hashes)
        == {"cpu_before", "cpu_after", "cuda_before", "cuda_after"}
        and all(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in caller_hashes.values()
        )
        and caller_hashes["cpu_before"] == caller_hashes["cpu_after"]
        and caller_hashes["cuda_before"] == caller_hashes["cuda_after"],
        "exact unchanged caller RNG state hash bindings",
    )
    caller_states = prepared.get("caller_rng_states")
    require(
        type(caller_states) is dict
        and set(caller_states)
        == {"cpu_before", "cpu_after", "cuda_before", "cuda_after"}
        and all(
            torch.is_tensor(value)
            and value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.numel() > 0
            for value in caller_states.values()
        )
        and all(
            _state_digest(value) == caller_hashes[name]
            for name, value in caller_states.items()
        )
        and torch.equal(caller_states["cpu_before"], caller_states["cpu_after"])
        and torch.equal(caller_states["cuda_before"], caller_states["cuda_after"]),
        "typed retained preparation caller RNG states",
    )
    for name, expected in (
        ("simulator_created", False),
        ("rollout_collected", False),
        ("optimizer_steps", 0),
        ("training_update_performed", False),
        ("finite_optimizer_step_qualified", False),
        ("transition_bridge_qualified", False),
        ("schedule_installed", False),
        ("student_export_available", False),
        ("training_job_predeclared", False),
        ("execution_admitted", False),
    ):
        require(receipt[name] is expected, "non-learning preparation field " + name)
    require(
        all(receipt[name] is False for name in FALSE_FLAGS),
        "all recovery admission flags false before shadow sampling",
    )
    state = prepared.get("private_cuda_rng_state")
    require(
        torch.is_tensor(state)
        and state.device.type == "cpu"
        and state.dtype == torch.uint8
        and state.ndim == 1
        and state.numel() > 0
        and _state_digest(state) == receipt["private_cuda_rng_state_sha256"],
        "exact detached prepared private CUDA RNG state",
    )
    return source, receipt["learner_seed"], state.detach().clone()


def _transition_empty(transition):
    require(
        transition.observations is None
        and transition.actions is None
        and transition.rewards is None
        and transition.dones is None
        and transition.values is None
        and transition.actions_log_prob is None
        and transition.distribution_params is None
        and transition.privileged_actions is None
        and transition.hidden_states == (None, None),
        "fresh PPO transition with no prior sampler call",
    )


def _storage_snapshot(storage):
    result = {
        "observations": {
            key: value.detach().cpu().clone()
            for key, value in storage.observations.items()
        }
    }
    for name in (
        "actions",
        "rewards",
        "dones",
        "values",
        "actions_log_prob",
        "returns",
        "advantages",
    ):
        result[name] = getattr(storage, name).detach().cpu().clone()
    result["step"] = storage.step
    result["distribution_params"] = storage.distribution_params
    result["saved_hidden_state_a"] = storage.saved_hidden_state_a
    result["saved_hidden_state_c"] = storage.saved_hidden_state_c
    return result


def _same_snapshots(left, right):
    require(
        left["step"] == right["step"] == 0
        and left["distribution_params"] is right["distribution_params"] is None
        and left["saved_hidden_state_a"] is right["saved_hidden_state_a"] is None
        and left["saved_hidden_state_c"] is right["saved_hidden_state_c"] is None,
        "storage remains structurally empty",
    )
    require(
        set(left["observations"]) == set(right["observations"])
        and all(
            torch.equal(left["observations"][k], right["observations"][k])
            for k in left["observations"]
        ),
        "storage observations unchanged",
    )
    for name in (
        "actions",
        "rewards",
        "dones",
        "values",
        "actions_log_prob",
        "returns",
        "advantages",
    ):
        require(torch.equal(left[name], right[name]), "storage " + name + " unchanged")
        require(
            not torch.count_nonzero(right[name]), "storage " + name + " remains zero"
        )


def _shadow_observations():
    actor = torch.zeros(WORLDS, 44, dtype=torch.float32, device=DEVICE)
    critic = torch.zeros(WORLDS, 50, dtype=torch.float32, device=DEVICE)
    require(
        torch.equal(actor, critic[:, :44]), "synthetic actor/critic observation prefix"
    )
    return TensorDict({"actor": actor, "critic": critic}, [WORLDS], device=DEVICE)


def _check_shadow_observations(observations):
    require(
        type(observations) is TensorDict
        and tuple(observations.batch_size) == (WORLDS,),
        "exact synthetic 64-world TensorDict inputs",
    )
    require(
        set(observations.keys()) == {"actor", "critic"},
        "exact synthetic actor/critic observation groups",
    )
    actor, critic = observations["actor"], observations["critic"]
    for name, value, shape in (
        ("actor", actor, (WORLDS, 44)),
        ("critic", critic, (WORLDS, 50)),
    ):
        require(
            value.shape == shape
            and value.dtype == torch.float32
            and value.device.type == "cuda"
            and value.device.index == 0
            and torch.isfinite(value).all()
            and not torch.count_nonzero(value),
            "immutable finite zero CUDA0 float32 synthetic " + name + " input",
        )
    require(
        torch.equal(actor, critic[:, :44]), "synthetic actor/critic observation prefix"
    )


def _check_stock_act(algorithm):
    method = getattr(algorithm, "act", None)
    require(
        type(algorithm) is preparation.PPO
        and getattr(method, "__self__", None) is algorithm
        and getattr(method, "__func__", None) is preparation.PPO.act,
        "exact pinned stock PPO.act bound method",
    )


def _caller_cuda_state():
    return torch.cuda.get_rng_state(0).detach().cpu().clone()


def _current_cuda_state():
    return torch.cuda.get_rng_state(0).detach().cpu().clone()


def _check_cuda0_ready():
    require(
        torch.cuda.is_initialized() is True,
        "prepared CUDA runtime must already be initialized",
    )
    require(
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
        "exact single visible CUDA0 device",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "CUDA0 must be the only visible device from process start",
    )


def _require_cuda_float(value, shape, label, *, positive=False):
    require(
        torch.is_tensor(value)
        and tuple(value.shape) == shape
        and value.dtype == torch.float32
        and value.device.type == "cuda"
        and value.device.index == 0
        and torch.isfinite(value).all(),
        "finite CUDA0 float32 " + label,
    )
    if positive:
        require((value > 0).all(), "positive " + label)


@torch.no_grad()
def sample_shadow(prepared, *, lease_fd):
    """Retain 28 stock stochastic policy calls on synthetic zero observations.

    The output is a bounded sampler trace, not a transition bridge, rollout,
    training result, or CUDA qualification. CPU provenance remains the
    responsibility of the supervising caller.
    """
    # The ownership proof is intentionally the first operation and precedes
    # any CUDA query, allocation, or tensor placement.
    require(
        type(lease_fd) is int and lease_fd >= 0, "inherited shared GPU lease descriptor"
    )
    training_smoke.inherited_lease(lease_fd)
    require(type(prepared) is dict, "typed policy preparation result")
    require(
        set(prepared)
        == {
            "actor",
            "critic",
            "algorithm",
            "storage",
            "private_cuda_generator",
            "private_cuda_rng_state",
            "caller_rng_states",
            "receipt",
        },
        "exact policy preparation result fields",
    )
    receipt = deepcopy(prepared["receipt"])
    source, seed, private_state = _validate_receipt(receipt, prepared)
    _check_cuda0_ready()
    actor, critic, algorithm, storage = (
        prepared[k] for k in ("actor", "critic", "algorithm", "storage")
    )
    preparation._check_rsl_sources()
    checkpoint.runtime_check()
    preparation._check_policy(actor, critic, algorithm, storage)
    preparation._check_storage(storage)
    require(
        algorithm.actor is actor
        and algorithm.critic is critic
        and algorithm.storage is storage,
        "same prepared actor/critic/PPO/storage objects",
    )
    require(
        type(receipt["actor_parameter_tensors"]) is int
        and type(receipt["critic_parameter_tensors"]) is int
        and receipt["actor_parameter_tensors"] == sum(1 for _ in actor.parameters())
        and receipt["critic_parameter_tensors"] == sum(1 for _ in critic.parameters()),
        "prepared actor/critic parameter counts match live models",
    )
    _transition_empty(algorithm.transition)
    _check_stock_act(algorithm)
    model_state_before = checkpoint.states_of(actor, critic)
    before_hash = checkpoint.state_hash(model_state_before)
    require(
        before_hash == preparation.parent.PARENT_STATE_SHA256,
        "actual unchanged D1 actor/critic state before sampling",
    )
    storage_before = _storage_snapshot(storage)
    optimizer_before = deepcopy(algorithm.optimizer.state_dict())
    private_generator = prepared["private_cuda_generator"]
    require(
        str(private_generator.device) == DEVICE
        and torch.equal(private_generator.get_state().detach().cpu(), private_state),
        "same fresh private CUDA generator remains separate and unadvanced",
    )
    caller_cpu = torch.random.get_rng_state().detach().clone()
    caller_cuda = _caller_cuda_state()
    require(
        torch.equal(caller_cpu, prepared["caller_rng_states"]["cpu_after"])
        and torch.equal(caller_cuda, prepared["caller_rng_states"]["cuda_after"]),
        "current caller RNG streams match the preparation closeout states",
    )
    observations = _shadow_observations()
    _check_shadow_observations(observations)
    observation_snapshot = {
        key: value.detach().cpu().clone() for key, value in observations.items()
    }

    scope = rng_scope.CudaPrivateRngScope(seed, private_state, lease_fd=lease_fd)
    require(
        scope.receipt["source"] == source,
        "private RNG scope bound to exact prepared source",
    )
    state_boundaries = [scope.state]
    captures = {
        name: []
        for name in (
            "actor_observations",
            "critic_observations",
            "actions",
            "values",
            "actions_log_prob",
            "distribution_mean",
            "distribution_std",
        )
    }
    try:
        with scope.scope():
            for _step in range(HORIZON):
                _check_shadow_observations(observations)
                _check_stock_act(algorithm)
                action = algorithm.act(observations)
                _require_cuda_float(action, (WORLDS, 10), "raw actions")
                transition = algorithm.transition
                require(
                    torch.equal(transition.actions, action),
                    "stock transition retains unclipped raw action",
                )
                _require_cuda_float(transition.values, (WORLDS, 1), "critic values")
                _require_cuda_float(
                    transition.actions_log_prob, (WORLDS,), "action log probability"
                )
                params = transition.distribution_params
                require(
                    type(params) is tuple and len(params) == 2,
                    "stock Gaussian mean/std distribution parameters",
                )
                mean, std = params
                _require_cuda_float(mean, (WORLDS, 10), "distribution mean")
                _require_cuda_float(
                    std, (WORLDS, 10), "distribution standard deviation", positive=True
                )
                _check_shadow_observations(observations)
                require(
                    all(
                        torch.equal(
                            observations[key].detach().cpu(), observation_snapshot[key]
                        )
                        for key in observation_snapshot
                    ),
                    "stock sampler did not mutate zero shadow observations",
                )
                for name, value in (
                    ("actor_observations", observations["actor"]),
                    ("critic_observations", observations["critic"]),
                    ("actions", action),
                    ("values", transition.values),
                    ("actions_log_prob", transition.actions_log_prob),
                    ("distribution_mean", mean),
                    ("distribution_std", std),
                ):
                    captures[name].append(value.detach().cpu().clone())
                state_boundaries.append(_current_cuda_state())
                require(
                    not torch.equal(state_boundaries[-2], state_boundaries[-1]),
                    "private CUDA RNG advances on every stock sampler call",
                )
    finally:
        # Retention precedes clearing; never call process_env_step or storage.add.
        algorithm.transition.clear()

    require(
        len(captures["actions"]) == HORIZON and len(state_boundaries) == HORIZON + 1,
        "exact 28 stock calls and 29 private RNG state boundaries",
    )
    _check_stock_act(algorithm)
    _check_shadow_observations(observations)
    caller_cpu_after = torch.random.get_rng_state().detach().clone()
    caller_cuda_after = _caller_cuda_state()
    caller_cpu_unchanged = torch.equal(caller_cpu, caller_cpu_after)
    caller_cuda_unchanged = torch.equal(caller_cuda, caller_cuda_after)
    require(
        caller_cpu_unchanged and caller_cuda_unchanged,
        "caller CPU and CUDA0 RNG streams unchanged",
    )
    require(
        torch.equal(private_generator.get_state().detach().cpu(), private_state),
        "prepared private generator remains unconnected and unadvanced",
    )
    preparation._check_policy(actor, critic, algorithm, storage)
    preparation._check_storage(storage)
    storage_after = _storage_snapshot(storage)
    _same_snapshots(storage_before, storage_after)
    optimizer_after = deepcopy(algorithm.optimizer.state_dict())
    require(
        canonical(optimizer_before) == canonical(optimizer_after)
        and not algorithm.optimizer.state,
        "fresh Adam remains empty and unchanged",
    )
    model_state_after = checkpoint.states_of(actor, critic)
    after_hash = checkpoint.state_hash(model_state_after)
    require(
        after_hash == before_hash == preparation.parent.PARENT_STATE_SHA256,
        "actor/critic parent weights unchanged by shadow sampling",
    )
    scope_receipt = scope.receipt
    require(
        scope_receipt["scope_count"] == 1
        and scope_receipt["scope_active"] is False
        and scope_receipt["faulted"] is False
        and scope_receipt["caller_cpu_rng_preserved"] is True
        and scope_receipt["caller_cuda_rng_preserved"] is True
        and _state_digest(state_boundaries[-1])
        == scope_receipt["private_state_sha256"],
        "private scoped sampler state closed cleanly",
    )
    result_receipt = {
        "protocol": PROTOCOL,
        "source": source,
        "seed": seed,
        "worlds": WORLDS,
        "horizon": HORIZON,
        "device": DEVICE,
        "synthetic_zero_observations": True,
        "physical_observations": False,
        "stock_sampler_calls": HORIZON,
        "private_cuda_generator_connected_to_sampler": False,
        "private_state_installed_around_stock_sampler": True,
        "private_cuda_generator_advanced": False,
        "private_rng_scope_count": scope_receipt["scope_count"],
        "private_state_boundary_sha256": [_state_digest(v) for v in state_boundaries],
        "parent_state_sha256_before": before_hash,
        "parent_state_sha256_after": after_hash,
        "storage_step_before": storage_before["step"],
        "storage_step_after": storage_after["step"],
        "caller_cpu_rng_unchanged": caller_cpu_unchanged,
        "caller_cuda_rng_unchanged": caller_cuda_unchanged,
        "caller_rng_state_sha256": {
            "cpu_before": _state_digest(caller_cpu),
            "cpu_after": _state_digest(caller_cpu_after),
            "cuda_before": _state_digest(caller_cuda),
            "cuda_after": _state_digest(caller_cuda_after),
        },
        "history_attested_by_this_function": False,
        "native_cuda_sampling_qualified": False,
        "rollout_collected": False,
        "optimizer_steps": 0,
        "training_update_performed": False,
        "execution_admitted": False,
        **FALSE_FLAGS,
    }
    return {
        "receipt": result_receipt,
        **{name: torch.stack(values) for name, values in captures.items()},
        "private_cuda_state_boundaries": [
            v.detach().cpu().clone() for v in state_boundaries
        ],
        "model_state_before": model_state_before,
        "model_state_after": model_state_after,
        "optimizer_state_before": optimizer_before,
        "optimizer_state_after": optimizer_after,
        "caller_rng_states": {
            "cpu_before": caller_cpu.detach().clone(),
            "cpu_after": caller_cpu_after.detach().clone(),
            "cuda_before": caller_cuda.detach().cpu().clone(),
            "cuda_after": caller_cuda_after.detach().cpu().clone(),
        },
        "storage_before": storage_before,
        "storage_after": storage_after,
        "scope_receipt": deepcopy(scope_receipt),
    }
