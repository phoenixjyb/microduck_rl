"""CPU consistency scoring for separately retained CUDA shadow samples.

This module authenticates no host, launch, service or CUDA execution. A native
supervisor must bind whole raw bytes, fresh provenance and isolated children.
Equality of two supplied records alone is never CUDA replay qualification.
"""

from copy import deepcopy
import math

import torch

from mjlab_microduck import stance_recovery_cuda_policy_probe as preparation_probe
from mjlab_microduck import stance_recovery_cuda_rng_scope as rng_scope
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cuda64-shadow-consistency-v1"
SHAPES = {
    "actor_observations": (28, 64, 44),
    "critic_observations": (28, 64, 50),
    "actions": (28, 64, 10),
    "values": (28, 64, 1),
    "actions_log_prob": (28, 64),
    "distribution_mean": (28, 64, 10),
    "distribution_std": (28, 64, 10),
}
EXTRA_FIELDS = {
    "receipt",
    "private_cuda_state_boundaries",
    "model_state_before",
    "model_state_after",
    "optimizer_state_before",
    "optimizer_state_after",
    "caller_rng_states",
    "storage_before",
    "storage_after",
    "scope_receipt",
}
FALSE_RECEIPT_FIELDS = {
    "physical_observations",
    "private_cuda_generator_connected_to_sampler",
    "private_cuda_generator_advanced",
    "history_attested_by_this_function",
    "native_cuda_sampling_qualified",
    "rollout_collected",
    "training_update_performed",
    "execution_admitted",
    *sampling.FALSE_FLAGS,
}
RECEIPT_FIELDS = FALSE_RECEIPT_FIELDS | {
    "protocol",
    "source",
    "seed",
    "worlds",
    "horizon",
    "device",
    "synthetic_zero_observations",
    "stock_sampler_calls",
    "private_state_installed_around_stock_sampler",
    "private_rng_scope_count",
    "private_state_boundary_sha256",
    "parent_state_sha256_before",
    "parent_state_sha256_after",
    "storage_step_before",
    "storage_step_after",
    "caller_cpu_rng_unchanged",
    "caller_cuda_rng_unchanged",
    "caller_rng_state_sha256",
    "optimizer_steps",
}


def equal_tree(left, right):
    """Typed recursive equality; no broadcasting or float tolerance."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device.type == right.device.type == "cpu"
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            equal_tree(left[k], right[k]) for k in left
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right)
        )
    return left == right


def _bytes_state(value, *, length=None):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.numel() > 0
        and (length is None or value.numel() == length),
        "raw CPU uint8 RNG state with unchanged layout",
    )
    return sampling._state_digest(value)


def score_sampling(result, prepared_snapshot, *, source, seed):
    """Check a hash-authenticated caller's CPU snapshots, not native execution."""
    preparation_probe._hex(source, 40, "sampler evaluator source")
    require(
        type(seed) is int and seed in sampling.preparation.SEEDS,
        "one declared shadow seed",
    )
    require(
        type(result) is dict and set(result) == set(SHAPES) | EXTRA_FIELDS,
        "exact retained shadow payload schema",
    )
    require(
        type(prepared_snapshot) is dict
        and set(prepared_snapshot)
        == {
            "metadata",
            "actor_critic_states",
            "optimizer_state_dict",
            "storage",
            "caller_rng_states",
            "private_cuda_rng_state",
        },
        "exact retained preparation payload schema",
    )
    receipt = result["receipt"]
    require(
        type(receipt) is dict and set(receipt) == RECEIPT_FIELDS,
        "exact non-admitting shadow receipt schema",
    )
    require(
        receipt["protocol"] == sampling.PROTOCOL
        and receipt["source"] == source
        and type(receipt["seed"]) is int
        and receipt["seed"] == seed
        and receipt["device"] == "cuda:0"
        and all(
            type(receipt[k]) is int and receipt[k] == n
            for k, n in (
                ("worlds", 64),
                ("horizon", 28),
                ("stock_sampler_calls", 28),
                ("private_rng_scope_count", 1),
                ("storage_step_before", 0),
                ("storage_step_after", 0),
                ("optimizer_steps", 0),
            )
        )
        and all(
            receipt[k] is True
            for k in (
                "synthetic_zero_observations",
                "private_state_installed_around_stock_sampler",
                "caller_cpu_rng_unchanged",
                "caller_cuda_rng_unchanged",
            )
        )
        and all(receipt[k] is False for k in FALSE_RECEIPT_FIELDS),
        "fixed synthetic sampling without learning or admission",
    )
    for name, shape in SHAPES.items():
        value = result[name]
        require(
            torch.is_tensor(value)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and tuple(value.shape) == shape
            and torch.isfinite(value).all(),
            "typed finite CPU sample " + name,
        )
    require(
        not torch.count_nonzero(result["actor_observations"])
        and not torch.count_nonzero(result["critic_observations"])
        and torch.equal(
            result["actor_observations"], result["critic_observations"][:, :, :44]
        ),
        "all retained observations are immutable synthetic zeros",
    )
    mean, std = result["distribution_mean"], result["distribution_std"]
    require((std > 0).all(), "positive retained Gaussian standard deviations")
    require(
        torch.equal(mean, mean[0:1].expand_as(mean))
        and torch.equal(std, std[0:1].expand_as(std))
        and torch.equal(
            result["values"], result["values"][0:1].expand_as(result["values"])
        ),
        "fixed-input distribution and critic stay constant across calls",
    )
    # CPU algebra is a consistency cross-check, not replay of CUDA kernels.
    expected = (
        -0.5 * ((result["actions"].double() - mean.double()) / std.double()).square()
        - std.double().log()
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    require(
        torch.allclose(
            result["actions_log_prob"].double(), expected, rtol=0, atol=2e-4
        ),
        "retained Gaussian log probability agrees with raw actions within declared CPU tolerance",
    )
    parent_hash = sampling.preparation.parent.PARENT_STATE_SHA256
    states = prepared_snapshot["actor_critic_states"]
    require(
        sampling.checkpoint.state_hash(states) == parent_hash,
        "retained prepared parent is exact D1",
    )
    for name in ("model_state_before", "model_state_after"):
        require(
            equal_tree(result[name], states), "actual model snapshots unchanged " + name
        )
    require(
        receipt["parent_state_sha256_before"]
        == receipt["parent_state_sha256_after"]
        == parent_hash,
        "receipt agrees with actual unchanged parent tensors",
    )
    prepared_storage = prepared_snapshot["storage"]
    preparation_probe._check_storage(prepared_storage)
    for name in ("storage_before", "storage_after"):
        require(
            type(result[name]) is dict
            and type(result[name].get("step")) is int
            and result[name]["step"] == 0
            and equal_tree(
                {k: v for k, v in result[name].items() if k != "step"}, prepared_storage
            ),
            "actual unchanged zero storage " + name,
        )
    optimizer = prepared_snapshot["optimizer_state_dict"]
    require(
        type(optimizer) is dict
        and optimizer.get("state") == {}
        and set(optimizer) == {"state", "param_groups"},
        "empty prepared Adam schema",
    )
    require(
        equal_tree(result["optimizer_state_before"], optimizer)
        and equal_tree(result["optimizer_state_after"], optimizer),
        "actual empty Adam unchanged across sampler",
    )
    caller = result["caller_rng_states"]
    require(
        type(caller) is dict
        and set(caller) == {"cpu_before", "cpu_after", "cuda_before", "cuda_after"},
        "four newly retained sampler caller RNG snapshots",
    )
    caller_hashes = {k: _bytes_state(v) for k, v in caller.items()}
    prepared_caller = prepared_snapshot["caller_rng_states"]
    require(
        torch.equal(caller["cpu_before"], caller["cpu_after"])
        and torch.equal(caller["cuda_before"], caller["cuda_after"])
        and torch.equal(caller["cpu_before"], prepared_caller["cpu_after"])
        and torch.equal(caller["cuda_before"], prepared_caller["cuda_after"])
        and caller_hashes == receipt["caller_rng_state_sha256"],
        "new sampler boundaries preserve actual prepared caller streams",
    )
    boundaries = result["private_cuda_state_boundaries"]
    initial = prepared_snapshot["private_cuda_rng_state"]
    _bytes_state(initial)
    require(
        type(boundaries) is list and len(boundaries) == 29,
        "exact 29 retained private CUDA boundaries",
    )
    boundary_hashes = [_bytes_state(v, length=initial.numel()) for v in boundaries]
    require(
        torch.equal(boundaries[0], initial)
        and all(not torch.equal(a, b) for a, b in zip(boundaries, boundaries[1:]))
        and len(set(boundary_hashes)) == 29
        and boundary_hashes == receipt["private_state_boundary_sha256"],
        "fresh seed-bound private stream advances on every retained draw",
    )
    scope = result["scope_receipt"]
    expected_scope = {
        "protocol": rng_scope.PROTOCOL,
        "source": source,
        "seed": seed,
        "device": "cuda:0",
        "state_bytes": initial.numel(),
        "initial_state_sha256": boundary_hashes[0],
        "private_state_sha256": boundary_hashes[-1],
        "scope_count": 1,
        "caller_cpu_rng_preserved": True,
        "caller_cuda_rng_preserved": True,
        "scope_active": False,
        "faulted": False,
        "native_cuda_rng_scope_qualified": False,
        "native_cuda_sampling_qualified": False,
        **sampling.FALSE_FLAGS,
    }
    require(
        equal_tree(scope, expected_scope),
        "actual scope boundary receipt closed without admission",
    )
    return {
        "protocol": PROTOCOL,
        "source": source,
        "seed": seed,
        "stock_sampler_calls": 28,
        "caller_rng_preserved": True,
        "private_boundaries_advanced": 29,
        "synthetic_inputs_only": True,
        "cpu_log_prob_absolute_tolerance": 2e-4,
        "cuda_math_replayed": False,
        "optimizer_steps": 0,
        **sampling.FALSE_FLAGS,
    }


def compare_pair(capture, replay, *, source, seed):
    """Compare all numerical sample/state outputs; caller streams are private per process.

    Each independently supplied caller stream must be preserved by scoring. Its
    OS-initialized starting bytes may differ between fresh processes and are not
    required to match. No other field or sampling RNG boundary is excluded.
    """
    require(
        type(capture) is dict
        and type(replay) is dict
        and set(capture) == set(replay) == {"sampling", "prepared_snapshot"},
        "exact two retained pair inputs",
    )
    scores = [
        score_sampling(v["sampling"], v["prepared_snapshot"], source=source, seed=seed)
        for v in (capture, replay)
    ]
    left, right = (deepcopy(v["sampling"]) for v in (capture, replay))
    for value in (left, right):
        value.pop("caller_rng_states")
        value["receipt"].pop("caller_rng_state_sha256")
    require(
        equal_tree(left, right),
        "all paired sample tensors and private/model/optimizer/storage states exactly match",
    )
    return {
        "protocol": PROTOCOL + ":pair-v1",
        "source": source,
        "seed": seed,
        "scores": scores,
        "paired_numerical_outputs_exact": True,
        "caller_streams_individually_preserved": True,
        "only_pair_exclusions": [
            "caller_rng_states",
            "receipt.caller_rng_state_sha256",
        ],
        "native_cuda_replay_authenticated_by_this_function": False,
        "optimizer_steps": 0,
        **sampling.FALSE_FLAGS,
    }
