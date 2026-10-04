"""CPU-owned populated RSL storage evidence for the CUDA64 source path.

This records what the stock rollout storage contains after no-update collection.
It does not attest model/optimizer immutability, CUDA execution, or native
physics, and it never computes returns, clears storage, or launches an update.
"""

from collections.abc import Mapping
import math
import os

import torch

from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_training_smoke as training
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cuda64-storage-evidence-source-v1"
WORLDS, HORIZON = 64, 28
TENSOR_BUDGET = 128 * 1024**2
NODE_BUDGET = 100_000
TENSOR_KEYS = {
    "observations",
    "actions",
    "rewards",
    "dones",
    "values",
    "actions_log_prob",
    "returns",
    "advantages",
    "distribution_params",
}
ARRAY_SHAPES = {
    "actions": (HORIZON, WORLDS, 10),
    "rewards": (HORIZON, WORLDS, 1),
    "dones": (HORIZON, WORLDS, 1),
    "values": (HORIZON, WORLDS, 1),
    "actions_log_prob": (HORIZON, WORLDS, 1),
    "returns": (HORIZON, WORLDS, 1),
    "advantages": (HORIZON, WORLDS, 1),
}


def _owned_tree(value, *, allow_cuda, clone):
    budget = {"nodes": 0, "bytes": 0}

    def visit(item, depth=0):
        budget["nodes"] += 1
        require(
            depth <= 24 and budget["nodes"] <= NODE_BUDGET,
            "bounded storage evidence tree",
        )
        if torch.is_tensor(item):
            require(
                item.layout == torch.strided
                and item.is_contiguous()
                and item.dtype in (torch.float32, torch.int64, torch.uint8, torch.bool),
                "dense supported storage evidence tensor",
            )
            require(
                item.device.type == "cpu"
                or (
                    allow_cuda and item.device.type == "cuda" and item.device.index == 0
                ),
                "CPU evidence or leased CUDA0 storage",
            )
            budget["bytes"] += item.numel() * item.element_size()
            require(
                budget["bytes"] <= TENSOR_BUDGET,
                "bounded storage evidence tensor bytes",
            )
            require(
                not item.is_floating_point() or torch.isfinite(item).all(),
                "finite storage evidence",
            )
            return item.detach().cpu().clone() if clone else item
        if isinstance(item, Mapping):
            require(
                all(type(key) is str for key in item), "string storage evidence keys"
            )
            return {key: visit(child, depth + 1) for key, child in item.items()}
        if type(item) in (list, tuple):
            copied = [visit(child, depth + 1) for child in item]
            return tuple(copied) if type(item) is tuple else copied
        require(
            item is None or type(item) in (str, bool, int, float),
            "safe storage evidence leaf",
        )
        require(
            type(item) is not float or math.isfinite(item),
            "finite storage evidence scalar",
        )
        return item

    return visit(value)


def _check_source_storage(storage):
    require(
        type(storage) is preparation.RolloutStorage
        and storage.training_type == "rl"
        and storage.device == "cuda:0"
        and storage.num_envs == WORLDS
        and storage.num_transitions_per_env == HORIZON
        and storage.actions_shape == (10,)
        and type(storage.step) is int
        and 0 <= storage.step <= HORIZON
        and storage.saved_hidden_state_a is None
        and storage.saved_hidden_state_c is None,
        "exact stock CUDA64 nonrecurrent rollout storage",
    )
    require(
        set(storage.observations.keys()) == {"actor", "critic"},
        "exact storage observation groups",
    )
    tensors = {
        "observations": {key: storage.observations[key] for key in ("actor", "critic")}
    }
    tensors.update({key: getattr(storage, key) for key in ARRAY_SHAPES})
    tensors["distribution_params"] = storage.distribution_params
    for name, width in (("actor", 44), ("critic", 50)):
        _source_tensor(
            tensors["observations"][name],
            (HORIZON, WORLDS, width),
            torch.float32,
            name,
        )
    for name, shape in ARRAY_SHAPES.items():
        _source_tensor(
            tensors[name],
            shape,
            torch.uint8 if name == "dones" else torch.float32,
            name,
        )
    require((tensors["dones"] <= 1).all(), "binary CUDA storage done bytes")
    params = tensors["distribution_params"]
    if storage.step == 0:
        require(params is None, "empty stock storage has no Gaussian parameters")
    else:
        require(
            type(params) in (tuple, list) and len(params) == 2,
            "populated stock storage has Gaussian parameter pair",
        )
        for item in params:
            _source_tensor(
                item, (HORIZON, WORLDS, 10), torch.float32, "distribution parameters"
            )
    return storage.step, tensors


def _source_tensor(value, shape, dtype, label):
    require(
        torch.is_tensor(value)
        and value.device.type == "cuda"
        and value.device.index == 0
        and value.layout == torch.strided
        and value.is_contiguous()
        and value.dtype == dtype
        and tuple(value.shape) == shape,
        "CUDA0 stock storage tensor layout: " + label,
    )
    if dtype == torch.float32:
        require(torch.isfinite(value).all(), "finite CUDA0 stock storage: " + label)


def capture(storage, *, lease_fd):
    """Snapshot exact stock storage only after checking the inherited lease."""
    require(type(lease_fd) is int and lease_fd >= 0, "inherited capture lease")
    training.inherited_lease(lease_fd)
    require(
        torch.cuda.is_initialized() is True
        and torch.cuda.device_count() == 1
        and torch.cuda.current_device() == 0,
        "already initialized isolated CUDA0 capture",
    )
    cursor, tensors = _check_source_storage(storage)
    owned = _owned_tree(tensors, allow_cuda=True, clone=True)
    return {
        "protocol": PROTOCOL,
        "worlds": WORLDS,
        "horizon": HORIZON,
        "cursor": cursor,
        "tensors": owned,
        **transition.FALSE_FLAGS,
    }


def _cpu_tensor(value, shape, dtype, label):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.is_contiguous()
        and value.dtype == dtype
        and tuple(value.shape) == shape,
        "storage evidence tensor layout: " + label,
    )
    if dtype == torch.float32:
        require(torch.isfinite(value).all(), "finite storage evidence: " + label)


def check(value, records):
    """Check a CPU-owned storage snapshot against retained transition records."""
    require(
        os_cuda_hidden() and not torch.cuda.is_initialized(),
        "CUDA-hidden storage evidence checker",
    )
    _owned_tree(value, allow_cuda=False, clone=False)
    require(
        type(value) is dict
        and set(value)
        == {"protocol", "worlds", "horizon", "cursor", "tensors"}
        | set(transition.FALSE_FLAGS)
        and value["protocol"] == PROTOCOL
        and value["worlds"] == WORLDS
        and value["horizon"] == HORIZON
        and type(value["cursor"]) is int
        and 0 <= value["cursor"] <= HORIZON
        and all(value[key] is False for key in transition.FALSE_FLAGS),
        "exact non-admitting CUDA64 storage envelope",
    )
    cursor, tensors = value["cursor"], value["tensors"]
    require(
        type(tensors) is dict and set(tensors) == TENSOR_KEYS,
        "exact storage tensor fields",
    )
    require(
        type(tensors["observations"]) is dict
        and set(tensors["observations"]) == {"actor", "critic"},
        "exact stored observation groups",
    )
    for name, width in (("actor", 44), ("critic", 50)):
        _cpu_tensor(
            tensors["observations"][name], (HORIZON, WORLDS, width), torch.float32, name
        )
    for key, shape in ARRAY_SHAPES.items():
        dtype = torch.uint8 if key == "dones" else torch.float32
        _cpu_tensor(tensors[key], shape, dtype, key)
    require((tensors["dones"] <= 1).all(), "binary stored done bytes")
    params = tensors["distribution_params"]
    if cursor == 0:
        require(params is None, "empty cursor has no Gaussian storage")
    else:
        require(
            type(params) in (tuple, list) and len(params) == 2,
            "stored Gaussian parameter pair",
        )
        for item in params:
            _cpu_tensor(
                item, (HORIZON, WORLDS, 10), torch.float32, "distribution parameters"
            )
    require(
        type(records) is list and len(records) == cursor,
        "storage cursor matches retained records",
    )
    require(
        not tensors["returns"].any() and not tensors["advantages"].any(),
        "returns and advantages remain zero",
    )
    for index, record in enumerate(records):
        require(type(record) is dict, "typed retained transition record")
        for key in (
            "pre_action_observations",
            "raw_actions",
            "pre_action_values",
            "raw_actions_log_prob",
            "learner_reward",
            "terminated",
            "timed_out",
            "distribution_params",
        ):
            require(key in record, "retained record field: " + key)
        for name in ("actor", "critic"):
            expected = record["pre_action_observations"][name]
            _cpu_tensor(
                expected,
                (WORLDS, 44 if name == "actor" else 50),
                torch.float32,
                "record " + name,
            )
            require(
                torch.equal(tensors["observations"][name][index], expected),
                "stored pre-action " + name,
            )
        _cpu_tensor(
            record["raw_actions_log_prob"],
            (WORLDS,),
            torch.float32,
            "record log probability",
        )
        _cpu_tensor(
            record["learner_reward"], (WORLDS,), torch.float32, "record learner reward"
        )
        expected_fields = (
            ("actions", record["raw_actions"], (WORLDS, 10)),
            ("values", record["pre_action_values"], (WORLDS, 1)),
            (
                "actions_log_prob",
                record["raw_actions_log_prob"].unsqueeze(-1),
                (WORLDS, 1),
            ),
            ("rewards", record["learner_reward"].unsqueeze(-1), (WORLDS, 1)),
        )
        for name, expected, shape in expected_fields:
            _cpu_tensor(expected, shape, torch.float32, "record " + name)
            require(torch.equal(tensors[name][index], expected), "stored raw " + name)
        _cpu_tensor(record["terminated"], (WORLDS,), torch.bool, "record terminated")
        _cpu_tensor(record["timed_out"], (WORLDS,), torch.bool, "record timed out")
        done = record["terminated"] | record["timed_out"]
        require(
            torch.equal(tensors["dones"][index, :, 0], done.to(torch.uint8)),
            "stored done",
        )
        pair = record["distribution_params"]
        require(
            type(pair) in (tuple, list) and len(pair) == 2,
            "record Gaussian parameter pair",
        )
        for stored, expected in zip(params, pair):
            _cpu_tensor(
                expected, (WORLDS, 10), torch.float32, "record Gaussian parameters"
            )
            require(torch.equal(stored[index], expected), "stored Gaussian parameters")

    def check_tail(item, label):
        if torch.is_tensor(item):
            require(
                not item[cursor:].any(), "unused storage tail remains zero: " + label
            )
        elif type(item) is dict:
            for child_key, child in item.items():
                check_tail(child, label + "." + child_key)
        elif type(item) in (tuple, list):
            for index, child in enumerate(item):
                check_tail(child, label + "." + str(index))

    for key, tensor in tensors.items():
        check_tail(tensor, key)
    return {
        "protocol": PROTOCOL,
        "cursor": cursor,
        "records_bound": cursor,
        "raw_unclipped_actions_bound": True,
        "zero_returns_and_advantages": True,
        "unused_tail_zero": True,
        **transition.FALSE_FLAGS,
    }


def os_cuda_hidden():
    return os.environ.get("CUDA_VISIBLE_DEVICES") == ""
