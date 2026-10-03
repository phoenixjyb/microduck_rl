"""Prepare exact D1 weights on CUDA for a bounded 64-world PPO learner.

This sibling creates only fresh models, an empty stock PPO/Adam and empty
rollout storage. It does not create an environment, sample actions, update,
export, resume or admit training. CPU provenance must be authenticated by the
separate caller; a matching preparation receipt is not independent proof.
"""

import io
import os
import re
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import distribution, version

import mjlab  # noqa: F401 - Finish plugin registration before stock RSL PPO.
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_cpu_replay_profile as cpu_profile
from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_execution_profile as execution
from mjlab_microduck import stance_ppo
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_training_smoke as training_smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cuda64-recovery-policy-preparation-v1"
CPU_BINDING_PROTOCOL = "football-b1d-cuda64-cpu-parent-binding-v1"
WORLDS, HORIZON, DEVICE = 64, 28, "cuda:0"
SEEDS = (653, 659)
FALSE_FLAGS = dict(baseline.FALSE_FLAGS)


def _sha256_hex(value, label):
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "exact " + label + " SHA256",
    )
    return value


def _cpu_parent_receipt_digest(receipt):
    return sha256(canonical(receipt).encode("utf-8")).hexdigest()


def cpu_parent_binding(source, receipt_sha256):
    """Return the exact source binding expected from the CPU-side caller."""
    require(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
        "exact source revision",
    )
    _sha256_hex(receipt_sha256, "CPU parent preparation receipt")
    return {
        "protocol": CPU_BINDING_PROTOCOL,
        "source": source,
        "cpu_parent_receipt_sha256": receipt_sha256,
        "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
        "parent_state_sha256": parent.PARENT_STATE_SHA256,
        "parent_identity": parent.expected_identity(),
    }


def validate_cpu_parent_receipt(receipt, receipt_sha256, caller_binding, *, source):
    """Validate exact receipt shape and its caller-supplied canonical hash.

    This is an input consistency check, not CPU provenance authentication. The
    supervisor must authenticate the CPU preparation and bind that full evidence
    to this source before calling the CUDA child.
    """
    require(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
        "exact source revision",
    )
    _sha256_hex(receipt_sha256, "CPU parent preparation receipt")
    require(type(receipt) is dict, "typed CPU parent preparation receipt")
    require(
        _cpu_parent_receipt_digest(receipt) == receipt_sha256,
        "whole canonical CPU parent preparation receipt hash",
    )
    expected = cpu_parent_binding(source, receipt_sha256)
    require(
        type(caller_binding) is dict
        and canonical(caller_binding) == canonical(expected),
        "source-bound CPU parent caller binding",
    )

    expected_keys = {
        "protocol",
        "parent_checkpoint_sha256",
        "parent_identity",
        "parent_state_sha256",
        "cpu_math_profile",
        "strict_actor_restore",
        "strict_critic_restore",
        "models_trainable",
        "weights_only",
        "fresh_optimizer_required",
        "optimizer_created",
        "optimizer_restored",
        "simulator_restored",
        "storage_restored",
        "rng_restored",
        "normalization_restored",
        "historical_archive_authenticated_by_loader",
        "installed_in_learner",
        "execution_admitted",
    } | set(FALSE_FLAGS)
    require(
        set(receipt) == expected_keys, "exact CPU parent preparation receipt schema"
    )
    require(
        receipt["protocol"] == parent.PROTOCOL
        and receipt["parent_checkpoint_sha256"] == baseline.CHECKPOINT_SHA256
        and canonical(receipt["parent_identity"])
        == canonical(parent.expected_identity())
        and receipt["parent_state_sha256"] == parent.PARENT_STATE_SHA256,
        "exact frozen D1 CPU parent identity",
    )
    require(
        receipt["strict_actor_restore"] is True
        and receipt["strict_critic_restore"] is True
        and receipt["models_trainable"] is True
        and receipt["weights_only"] is True
        and receipt["fresh_optimizer_required"] is True,
        "strict CPU actor and critic preparation",
    )
    for name in (
        "optimizer_created",
        "optimizer_restored",
        "simulator_restored",
        "storage_restored",
        "rng_restored",
        "normalization_restored",
        "historical_archive_authenticated_by_loader",
        "installed_in_learner",
        "execution_admitted",
    ):
        require(
            receipt[name] is False, "non-restoring non-admitting CPU receipt " + name
        )
    require(
        all(receipt[name] is False for name in FALSE_FLAGS),
        "all capability flags remain false in CPU receipt",
    )
    cpu_profile.validate_receipt(receipt["cpu_math_profile"])
    return deepcopy(expected)


def _check_rsl_sources():
    require(version("rsl-rl-lib") == "5.0.1", "pinned stock RSL 5.0.1")
    root = distribution("rsl-rl-lib").locate_file("rsl_rl")
    actual = {
        name: sha256((root / name).read_bytes()).hexdigest() for name in stance_ppo.PINS
    }
    require(
        actual == stance_ppo.PINS, "exact reviewed PPO and rollout storage source pins"
    )


def _check_storage(storage):
    require(
        type(storage) is RolloutStorage
        and storage.training_type == "rl"
        and storage.device == DEVICE
        and storage.num_envs == WORLDS
        and storage.num_transitions_per_env == HORIZON
        and storage.actions_shape == (10,)
        and storage.step == 0
        and storage.distribution_params is None
        and storage.saved_hidden_state_a is None
        and storage.saved_hidden_state_c is None,
        "fresh empty 64-world CUDA 28-step RSL rollout storage",
    )
    expected = {
        "actions": (HORIZON, WORLDS, 10),
        "rewards": (HORIZON, WORLDS, 1),
        "dones": (HORIZON, WORLDS, 1),
        "values": (HORIZON, WORLDS, 1),
        "actions_log_prob": (HORIZON, WORLDS, 1),
        "returns": (HORIZON, WORLDS, 1),
        "advantages": (HORIZON, WORLDS, 1),
    }
    require(
        set(storage.observations.keys()) == {"actor", "critic"},
        "exact actor/critic rollout observation groups",
    )
    for name, width in (("actor", 44), ("critic", 50)):
        value = storage.observations[name]
        require(
            tuple(value.shape) == (HORIZON, WORLDS, width)
            and value.dtype == torch.float32,
            "exact CUDA observation storage " + name,
        )
        require(
            value.device.type == "cuda"
            and value.device.index == 0
            and torch.isfinite(value).all()
            and not torch.count_nonzero(value),
            "fresh finite zero CUDA observation storage " + name,
        )
    for name, shape in expected.items():
        value = getattr(storage, name)
        require(tuple(value.shape) == shape, "exact CUDA storage shape " + name)
        expected_dtype = torch.uint8 if name == "dones" else torch.float32
        require(value.dtype == expected_dtype, "exact CUDA storage dtype " + name)
        require(
            value.device.type == "cuda"
            and value.device.index == 0
            and torch.isfinite(value).all()
            and not torch.count_nonzero(value),
            "fresh finite zero CUDA storage " + name,
        )


def _check_policy(actor, critic, algorithm, storage):
    require(
        type(algorithm) is PPO
        and algorithm.actor is actor
        and algorithm.critic is critic
        and algorithm.storage is storage
        and algorithm.device == DEVICE
        and algorithm.rnd is None
        and algorithm.rnd_optimizer is None
        and algorithm.symmetry is None,
        "stock PPO owns the exact transferred models and storage",
    )
    require(
        all(
            getattr(algorithm, name) == value
            for name, value in stance_ppo.CONFIG.items()
            if name != "optimizer"
        ),
        "unchanged declared stock PPO configuration",
    )
    require(
        type(algorithm.optimizer) is torch.optim.Adam
        and not algorithm.optimizer.state
        and len(algorithm.optimizer.param_groups) == 1,
        "fresh empty stock Adam",
    )
    optimizer_group = algorithm.optimizer.param_groups[0]
    require(
        optimizer_group["lr"] == stance_ppo.CONFIG["learning_rate"]
        and optimizer_group["betas"] == (0.9, 0.999)
        and optimizer_group["eps"] == 1e-8
        and optimizer_group["weight_decay"] == 0,
        "stock Adam defaults and declared learning rate",
    )
    actor_ids = [id(parameter) for parameter in actor.parameters()]
    critic_ids = [id(parameter) for parameter in critic.parameters()]
    expected_ids = actor_ids + critic_ids
    optimizer_ids = [
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    ]
    require(
        actor_ids
        and critic_ids
        and not set(actor_ids) & set(critic_ids)
        and len(expected_ids) == len(set(expected_ids))
        and len(optimizer_ids) == len(set(optimizer_ids))
        and set(optimizer_ids) == set(expected_ids),
        "unique exact CUDA optimizer parameter union",
    )
    for group, model in (("actor", actor), ("critic", critic)):
        for name, parameter in model.named_parameters():
            require(
                parameter.device.type == "cuda"
                and parameter.device.index == 0
                and parameter.dtype == torch.float32
                and parameter.requires_grad
                and parameter.grad is None
                and torch.isfinite(parameter).all(),
                "finite trainable CUDA0 " + group + " parameter " + name,
            )
    scale = getattr(getattr(actor, "distribution", None), "std_param", None)
    require(
        torch.is_tensor(scale)
        and scale.device.type == "cuda"
        and torch.isfinite(scale).all()
        and (scale > 0).all(),
        "positive finite CUDA actor Gaussian scale",
    )
    for name, value in stance_ppo.CONFIG.items():
        if name == "optimizer":
            require(value == "adam", "declared stock Adam selection")
        elif name == "num_learning_epochs":
            require(getattr(algorithm, name) == 5, "exact five PPO epochs")
        elif name == "num_mini_batches":
            require(getattr(algorithm, name) == 4, "exact four PPO mini-batches")


def prepare_policy(
    raw_parent,
    *,
    source,
    lease_fd,
    cpu_parent_receipt,
    cpu_parent_receipt_sha256,
    caller_binding,
    seed,
    worlds=64,
):
    """Build fresh exact-parent CUDA0 PPO/storage objects after the shared lease.

    ``cpu_parent_receipt`` must be the exact typed result of the old CPU parent
    loader. ``caller_binding`` binds its whole canonical hash to ``source``. The
    calling supervisor still has to authenticate that CPU run and its artifacts.
    The returned CUDA generator is fresh and intentionally not connected to PPO.
    """
    # This ownership proof must precede CUDA availability queries, context creation,
    # RNG snapshots, and every device allocation.
    training_smoke.inherited_lease(lease_fd)

    require(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
        "exact source revision",
    )
    require(type(seed) is int and seed in SEEDS, "only proposed recovery learner seeds")
    require(
        type(worlds) is int and worlds == WORLDS,
        "CUDA learner preparation is fixed at 64 worlds",
    )
    require(
        execution.PROFILE.get("name") == execution.WSL,
        "exact authorized 10098 WSL execution profile",
    )
    _sha256_hex(cpu_parent_receipt_sha256, "CPU parent preparation receipt")
    validate_cpu_parent_receipt(
        cpu_parent_receipt, cpu_parent_receipt_sha256, caller_binding, source=source
    )

    require(
        type(raw_parent) is bytes and 0 < len(raw_parent) <= checkpoint.LIMIT,
        "bounded exact parent checkpoint bytes",
    )
    require(
        sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256,
        "exact D1 parent bytes before deserialization",
    )

    source_identity = host.identity(source)
    require(
        source_identity.get("source") == source
        and source_identity.get("execution_profile") == execution.PROFILE,
        "exact source and 10098 host identity",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "CUDA device 0 must be the only visible device",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "actual single visible CUDA device required",
    )

    _check_rsl_sources()
    checkpoint.runtime_check()

    # Snapshot after the lease/source/device gates and before model or storage
    # allocation. fork_rng restores caller default CPU/CUDA streams on every exit.
    caller_cpu_rng_before = torch.random.get_rng_state().clone()
    caller_cuda_rng_before = torch.cuda.get_rng_state(0).detach().cpu().clone()
    with torch.random.fork_rng(devices=[0]):
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "CUDA0 remains available before preparation allocations",
        )
        identity = parent.expected_identity()
        actor_cpu, critic_cpu = checkpoint.validate_identity(
            identity, evaluation="lean-replication"
        )
        value = torch.load(
            io.BytesIO(raw_parent), map_location="cpu", weights_only=True
        )
        require(
            type(value) is dict
            and set(value) == {"identity", "states"}
            and type(value["identity"]) is dict
            and canonical(value["identity"]) == canonical(identity),
            "exact hash-first D1 export metadata",
        )
        require(
            type(value["states"]) is dict
            and set(value["states"]) == {"actor", "critic"}
            and all(
                type(value["states"][group]) is dict for group in ("actor", "critic")
            ),
            "both complete CPU parent model groups",
        )
        checkpoint.validate_states(value["states"], actor_cpu, critic_cpu)
        state_before_transfer = checkpoint.state_hash(
            checkpoint.states_of(actor_cpu, critic_cpu)
        )
        require(
            state_before_transfer == parent.PARENT_STATE_SHA256,
            "exact restored D1 parent state before device transfer",
        )
        for model in (actor_cpu, critic_cpu):
            model.train().requires_grad_(True)
            require(
                all(
                    parameter.device.type == "cpu"
                    and parameter.dtype == torch.float32
                    and parameter.requires_grad
                    and torch.isfinite(parameter).all()
                    for parameter in model.parameters()
                ),
                "fresh finite trainable CPU parent before transfer",
            )

        device = torch.device(DEVICE)
        actor_cpu.to(device)
        critic_cpu.to(device)
        obs = TensorDict(
            {
                "actor": torch.zeros(WORLDS, 44, dtype=torch.float32, device=device),
                "critic": torch.zeros(WORLDS, 50, dtype=torch.float32, device=device),
            },
            [WORLDS],
            device=device,
        )
        storage = RolloutStorage("rl", WORLDS, HORIZON, obs, (10,), device=DEVICE)
        algorithm = PPO(
            actor_cpu, critic_cpu, storage, **stance_ppo.CONFIG, device=DEVICE
        )
        require(
            algorithm.actor is actor_cpu and algorithm.critic is critic_cpu,
            "stock PPO preserves the exact transferred parent model objects",
        )
        private_generator = torch.Generator(device=DEVICE)
        private_generator.manual_seed(seed)
        private_cuda_rng_state = private_generator.get_state().detach().cpu().clone()
        _check_storage(storage)
        _check_policy(actor_cpu, critic_cpu, algorithm, storage)
        state_after_transfer = checkpoint.state_hash(
            checkpoint.states_of(actor_cpu, critic_cpu)
        )
        require(
            state_after_transfer == state_before_transfer == parent.PARENT_STATE_SHA256,
            "exact parent state hash preserved across CUDA0 transfer",
        )

    caller_cpu_rng_after = torch.random.get_rng_state().clone()
    caller_cuda_rng_after = torch.cuda.get_rng_state(0).detach().cpu().clone()
    require(
        torch.equal(caller_cpu_rng_before, caller_cpu_rng_after)
        and torch.equal(caller_cuda_rng_before, caller_cuda_rng_after),
        "preparation preserves caller CPU and CUDA RNG states",
    )
    training_smoke.inherited_lease(lease_fd)
    require(
        host.identity(source) == source_identity,
        "exact source and host remain unchanged after device preparation",
    )
    require(
        private_cuda_rng_state.dtype == torch.uint8
        and private_cuda_rng_state.ndim == 1
        and private_cuda_rng_state.numel() > 0
        and sha256(private_cuda_rng_state.numpy().tobytes()).hexdigest(),
        "separately retained fresh private CUDA0 generator state",
    )

    receipt = dict(
        protocol=PROTOCOL,
        source=source,
        source_identity=deepcopy(source_identity),
        parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        parent_identity=parent.expected_identity(),
        parent_state_sha256=parent.PARENT_STATE_SHA256,
        cpu_parent_receipt_sha256=cpu_parent_receipt_sha256,
        caller_binding=deepcopy(caller_binding),
        cpu_parent_receipt=deepcopy(cpu_parent_receipt),
        cpu_provenance_authentication_required=True,
        native_cpu_provenance_authenticated_by_this_function=False,
        learner_seed=seed,
        worlds=WORLDS,
        horizon=HORIZON,
        device=DEVICE,
        actor_device=str(next(actor_cpu.parameters()).device),
        critic_device=str(next(critic_cpu.parameters()).device),
        state_sha256_before_transfer=state_before_transfer,
        state_sha256_after_transfer=state_after_transfer,
        actor_parameter_tensors=sum(1 for _ in actor_cpu.parameters()),
        critic_parameter_tensors=sum(1 for _ in critic_cpu.parameters()),
        ppo_config=deepcopy(stance_ppo.CONFIG),
        ppo_source_sha256=deepcopy(stance_ppo.PINS),
        storage_step=storage.step,
        storage_empty=True,
        optimizer_type="Adam",
        optimizer_state_entries=len(algorithm.optimizer.state),
        optimizer_empty=not algorithm.optimizer.state,
        private_cuda_generator_device=str(private_generator.device),
        private_cuda_rng_state_sha256=sha256(
            private_cuda_rng_state.numpy().tobytes()
        ).hexdigest(),
        private_cuda_rng_connected_to_sampler=False,
        caller_cpu_rng_unchanged=True,
        caller_cuda_rng_unchanged=True,
        caller_rng_state_sha256={
            "cpu_before": sha256(caller_cpu_rng_before.numpy().tobytes()).hexdigest(),
            "cpu_after": sha256(caller_cpu_rng_after.numpy().tobytes()).hexdigest(),
            "cuda_before": sha256(caller_cuda_rng_before.numpy().tobytes()).hexdigest(),
            "cuda_after": sha256(caller_cuda_rng_after.numpy().tobytes()).hexdigest(),
        },
        cuda_initialized=torch.cuda.is_initialized(),
        simulator_created=False,
        rollout_collected=False,
        optimizer_steps=0,
        training_update_performed=False,
        finite_optimizer_step_qualified=False,
        transition_bridge_qualified=False,
        schedule_installed=False,
        student_export_available=False,
        training_job_predeclared=False,
        execution_admitted=False,
        **FALSE_FLAGS,
    )
    return {
        "actor": actor_cpu,
        "critic": critic_cpu,
        "algorithm": algorithm,
        "storage": storage,
        "private_cuda_generator": private_generator,
        "private_cuda_rng_state": private_cuda_rng_state.clone(),
        "caller_rng_states": {
            "cpu_before": caller_cpu_rng_before.clone(),
            "cpu_after": caller_cpu_rng_after.clone(),
            "cuda_before": caller_cuda_rng_before.clone(),
            "cuda_after": caller_cuda_rng_after.clone(),
        },
        "receipt": receipt,
    }
