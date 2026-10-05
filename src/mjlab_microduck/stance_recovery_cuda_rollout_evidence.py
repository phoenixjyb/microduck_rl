"""Whole-byte CPU reader for one retained CUDA64 rollout attempt.

This module checks retained bytes and CPU-visible bindings only. It does not
launch a child, initialize CUDA, reproduce CUDA sampling/physics, or qualify a
rollout. In particular, a complete-looking record is not native evidence.
"""

from collections.abc import Mapping
from hashlib import sha256
import io
import math
import os
import re

import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_recovery_cuda_policy_probe as base
from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_record_replay as record_replay
from mjlab_microduck import stance_recovery_cuda_storage_evidence as storage_evidence
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_cuda_constructor_rng as constructor
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cuda64-rollout-body-v2"
ATTEMPTS = {"capture", "replay"}
TENSOR_BUDGET = archive.TENSOR_BUDGET
NODE_BUDGET = archive.NODE_BUDGET
BODY_KEYS = {
    "protocol",
    "source",
    "launch_sha256",
    "seed",
    "attempt",
    "initial_frame",
    "initial_control_state",
    "constructor_receipt",
    "archive",
    "storage",
    "model_state_after",
    "optimizer_state_after",
    "caller_rng_states",
    "private_cuda_state_final",
    "elapsed_seconds",
}
RNG_KEYS = {"cpu_before", "cpu_after", "cuda_before", "cuda_after"}
FALSE_FLAGS = dict(transition.FALSE_FLAGS)
SCORE_PROTOCOL = PROTOCOL + ":score-v1"
PAIR_PROTOCOL = PROTOCOL + ":pair-v1"


def _owned_tree(value, *, clone=True):
    """Bounded CPU-only tensor/tree ownership and finite-value guard."""
    budget = {"nodes": 0, "bytes": 0}

    def copy(item, depth=0):
        budget["nodes"] += 1
        require(
            depth <= 32 and budget["nodes"] <= NODE_BUDGET, "bounded rollout body tree"
        )
        if torch.is_tensor(item):
            require(
                item.layout == torch.strided
                and item.dtype
                in (
                    torch.float32,
                    torch.float64,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                    torch.bool,
                ),
                "dense supported rollout body tensor",
            )
            require(item.device.type == "cpu", "CPU-owned rollout body tensor")
            budget["bytes"] += item.numel() * item.element_size()
            require(
                budget["bytes"] <= TENSOR_BUDGET, "bounded rollout body tensor bytes"
            )
            require(
                not item.is_floating_point() or torch.isfinite(item).all(),
                "finite rollout body tensor",
            )
            return item.detach().clone() if clone else item
        if isinstance(item, Mapping):
            require(all(type(key) is str for key in item), "string rollout body keys")
            return {key: copy(child, depth + 1) for key, child in item.items()}
        if type(item) in (list, tuple):
            copied = [copy(child, depth + 1) for child in item]
            return tuple(copied) if type(item) is tuple else copied
        require(
            item is None or type(item) in (str, bool, int, float),
            "safe rollout body scalar",
        )
        require(
            type(item) is not float or math.isfinite(item), "finite rollout body scalar"
        )
        return item

    return copy(value)


def _raw_state(value, label):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.is_contiguous()
        and value.numel() > 0,
        "exact immutable raw uint8 state: " + label,
    )
    return value


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device == right.device
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if type(left) is dict or type(right) is dict:
        return (
            type(left) is type(right) is dict
            and set(left) == set(right)
            and all(_equal(left[key], right[key]) for key in left)
        )
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _schema(value, launch, launch_sha256, attempt):
    require(
        type(value) is dict
        and set(value) == BODY_KEYS | set(FALSE_FLAGS)
        and all(value[key] is False for key in FALSE_FLAGS),
        "exact non-admitting rollout body schema",
    )
    require(
        value["protocol"] == PROTOCOL
        and type(value["source"]) is str
        and value["source"] == launch["source"]
        and type(value["launch_sha256"]) is str
        and value["launch_sha256"] == launch_sha256
        and type(value["seed"]) is int
        and value["seed"] == 653
        and attempt in ATTEMPTS
        and value["attempt"] == attempt,
        "fixed source/launch/seed/attempt rollout binding",
    )
    require(
        type(value["elapsed_seconds"]) in (float, int)
        and type(value["elapsed_seconds"]) is not bool
        and math.isfinite(value["elapsed_seconds"])
        and 0 < value["elapsed_seconds"] < 480,
        "positive bounded rollout elapsed time",
    )


def _check_prepared_boundary(body, prepared, launch):
    require(
        type(prepared) is dict
        and set(prepared)
        == {
            "metadata",
            "actor_critic_states",
            "optimizer_state_dict",
            "storage",
            "caller_rng_states",
            "private_cuda_rng_state",
        },
        "exact separately retained preparation payload",
    )
    states = prepared["actor_critic_states"]
    require(
        type(states) is dict
        and set(states) == {"actor", "critic"}
        and _equal(body["model_state_after"], states)
        and checkpoint.state_hash(body["model_state_after"])
        == base.preparation.parent.PARENT_STATE_SHA256,
        "rollout model tensors remain exact frozen D1 parent",
    )
    optimizer = prepared["optimizer_state_dict"]
    require(
        type(optimizer) is dict
        and set(optimizer) == {"state", "param_groups"}
        and optimizer["state"] == {}
        and _equal(body["optimizer_state_after"], optimizer),
        "rollout optimizer remains exact empty prepared Adam",
    )
    caller = body["caller_rng_states"]
    prepared_caller = prepared["caller_rng_states"]
    require(
        type(caller) is dict
        and set(caller) == RNG_KEYS
        and type(prepared_caller) is dict
        and set(prepared_caller) == RNG_KEYS,
        "exact four caller RNG boundary states",
    )
    for name in RNG_KEYS:
        _raw_state(caller[name], "rollout caller " + name)
        _raw_state(prepared_caller[name], "prepared caller " + name)
    require(
        torch.equal(caller["cpu_before"], prepared_caller["cpu_after"])
        and torch.equal(caller["cpu_after"], prepared_caller["cpu_before"])
        and torch.equal(caller["cpu_after"], prepared_caller["cpu_after"])
        and torch.equal(prepared_caller["cpu_before"], prepared_caller["cpu_after"])
        and torch.equal(caller["cuda_before"], prepared_caller["cuda_after"])
        and torch.equal(caller["cuda_after"], caller["cuda_before"])
        and torch.equal(caller["cuda_after"], prepared_caller["cuda_after"])
        and torch.equal(prepared_caller["cuda_before"], prepared_caller["cuda_after"]),
        "rollout preserves the prepared caller CPU/CUDA streams exactly",
    )
    private_initial = _raw_state(
        prepared["private_cuda_rng_state"], "prepared private CUDA state"
    )
    private_final = _raw_state(
        body["private_cuda_state_final"], "final private CUDA state"
    )
    records = body["archive"]["records"]
    require(
        type(records) is list and len(records) > 0,
        "nonempty retained first-terminal record prefix",
    )
    first_private = _raw_state(
        records[0]["private_rng_state_before"], "first private CUDA state"
    )
    last_private = _raw_state(
        records[-1]["private_rng_state"], "last private CUDA state"
    )
    require(
        torch.equal(first_private, private_initial)
        and torch.equal(last_private, private_final),
        "rollout private CUDA stream endpoints bind to prepared and final states",
    )
    initial_controls = body["initial_control_state"]
    control.state(initial_controls, 64)
    first_record = records[0]
    require(
        _equal(initial_controls, first_record["initial_reset_controls"])
        and _equal(
            initial_controls,
            first_record["runtime_result_before_reset"]["control_evidence"]["initial"],
        ),
        "independently retained initial controls bind to first transition",
    )
    require(
        type(launch.get("schedule")) is dict
        and type(launch.get("compiled_plant")) is dict,
        "fixed launch schedule and compiled-plant declaration",
    )
    return records


def encode(value):
    """Serialize an already CPU-owned body and enforce the archive byte cap."""
    owned = _owned_tree(value)

    class BoundedBuffer(io.BytesIO):
        cap_exceeded = False

        def write(self, data):
            if self.tell() + len(data) > archive.LIMIT:
                self.cap_exceeded = True
                raise ValueError("bounded CUDA64 rollout body bytes")
            return super().write(data)

    buffer = BoundedBuffer()
    try:
        torch.save(owned, buffer)
    except Exception as error:
        if buffer.cap_exceeded:
            raise ValueError("bounded CUDA64 rollout body bytes") from error
        raise
    raw = buffer.getvalue()
    require(0 < len(raw) <= archive.LIMIT, "bounded CUDA64 rollout body bytes")
    return raw


def verify(
    raw,
    expected_sha256,
    prepared_raw,
    prepared_sha256,
    prepared_summary,
    launch,
    launch_sha,
    cpu_receipt,
    *,
    attempt,
):
    """Authenticate both whole artifacts before either weights-only CPU load."""
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden rollout evidence reader",
    )
    require(
        type(raw) is bytes and 0 < len(raw) <= archive.LIMIT,
        "bounded CUDA64 rollout body bytes",
    )
    require(
        type(prepared_raw) is bytes and 0 < len(prepared_raw) <= base.RAW_LIMIT,
        "bounded retained preparation bytes",
    )
    require(
        type(expected_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
        and sha256(raw).hexdigest() == expected_sha256,
        "whole rollout body hash before CPU load",
    )
    require(
        type(prepared_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", prepared_sha256) is not None
        and sha256(prepared_raw).hexdigest() == prepared_sha256,
        "whole preparation bytes hash before CPU load",
    )
    require(
        type(launch_sha) is str
        and re.fullmatch(r"[0-9a-f]{64}", launch_sha) is not None
        and type(launch) is dict
        and type(cpu_receipt) is dict,
        "typed launch digest and prepared CPU receipt",
    )
    require(
        type(prepared_summary) is dict
        and prepared_summary.get("protocol") == base.PROTOCOL + ":child-summary-v1"
        and prepared_summary.get("source") == launch.get("source")
        and prepared_summary.get("launch_sha256") == launch_sha
        and type(prepared_summary.get("seed")) is int
        and prepared_summary["seed"] == 653
        and prepared_summary.get("payload_sha256") == prepared_sha256
        and prepared_summary.get("payload_bytes") == len(prepared_raw),
        "truthful pre-rollout preparation child summary",
    )

    # This existing scorer performs a weights-only CPU load internally. Both raw
    # artifacts have already been hashed above before it can deserialize either.
    prepared_score = base._score_payload(
        prepared_raw,
        prepared_summary,
        launch,
        launch_sha,
        cpu_receipt,
        653,
    )
    prepared = torch.load(
        io.BytesIO(prepared_raw), map_location="cpu", weights_only=True
    )
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _owned_tree(value, clone=False)
    _schema(value, launch, launch_sha, attempt)
    records = _check_prepared_boundary(value, prepared, launch)
    constructor_score = constructor.check(
        value["constructor_receipt"],
        source=value["source"],
        prepared_caller=prepared["caller_rng_states"],
    )
    archive_score = archive.check(
        value["archive"], launch["schedule"], launch["compiled_plant"], seed=653
    )
    storage_score = storage_evidence.check(value["storage"], records)
    replay_score = record_replay.check(
        value["archive"],
        value["initial_frame"],
        launch["schedule"],
        launch["compiled_plant"],
        seed=653,
        cpu_profile=cpu_receipt["cpu_math_profile"],
    )
    # Replay invokes archive.check itself. Require both paths to agree on the
    # exact archive summary, rather than treating either as an admission result.
    require(
        replay_score["archive"] == archive_score,
        "archive and physical-record replay summaries agree",
    )
    terminal_prefix = bool(len(records) < 28 and records[-1]["terminated"].any())
    score = {
        "protocol": SCORE_PROTOCOL,
        "source": value["source"],
        "launch_sha256": launch_sha,
        "seed": 653,
        "attempt": attempt,
        "prepared_payload_sha256": prepared_sha256,
        "prepared_state_sha256": prepared_score["state_sha256"],
        "constructor": constructor_score,
        "archive": archive_score,
        "storage": storage_score,
        "record_replay": replay_score,
        "retained_first_terminal_prefix": terminal_prefix,
        "complete_28_call_gate": archive_score["complete_short_window"],
        "complete_nonzero_pulse_gate": archive_score["pulse"][
            "complete_nonzero_pulse_delivery"
        ],
        "whole_bytes_verified_before_load": True,
        "individually_cpu_validated": True,
        "native_rollout_qualified": False,
        "cuda_inference_reexecuted": False,
        "cuda_physics_reexecuted": False,
        "cuda_rng_reexecuted": False,
        "solver_reexecuted": False,
        "model_reexecuted": False,
        "optimizer_reexecuted": False,
        "physical_resimulation_performed": False,
        "training_admitted": False,
        **FALSE_FLAGS,
    }
    return value, score


def paired(left, right):
    """Compare the decoded bodies from two successful ``verify`` calls.

    This semantic comparison is diagnostic only; it does not qualify native
    collection or replay.
    """
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden rollout pair comparison",
    )
    _owned_tree(left, clone=False)
    _owned_tree(right, clone=False)
    require(
        type(left) is dict
        and type(right) is dict
        and set(left) == set(right) == BODY_KEYS | set(FALSE_FLAGS)
        and all(left[k] is False and right[k] is False for k in FALSE_FLAGS),
        "two individually verified exact rollout bodies",
    )
    bodies = (left, right)
    require(
        all(
            body["protocol"] == PROTOCOL
            and type(body["source"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", body["source"]) is not None
            and type(body["launch_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", body["launch_sha256"]) is not None
            and type(body["seed"]) is int
            and body["seed"] == 653
            and body["attempt"] in ATTEMPTS
            and type(body["elapsed_seconds"]) in (int, float)
            and type(body["elapsed_seconds"]) is not bool
            and math.isfinite(body["elapsed_seconds"])
            and 0 < body["elapsed_seconds"] < 480
            for body in bodies
        )
        and all(
            bodies[0][key] == bodies[1][key]
            for key in ("protocol", "source", "launch_sha256", "seed")
        )
        and {body["attempt"] for body in bodies} == ATTEMPTS,
        "paired attempts share source/seed/launch schema",
    )
    excluded = {
        "attempt",
        "elapsed_seconds",
        "caller_rng_states",
        "constructor_receipt.caller_states",
        "constructor_receipt.state_sha256.caller_endpoints",
    }
    compare_keys = {
        "initial_frame",
        "initial_control_state",
        "archive",
        "storage",
        "model_state_after",
        "optimizer_state_after",
        "private_cuda_state_final",
    }
    require(
        compare_keys <= set(bodies[0]) == set(bodies[1]),
        "exact pair semantic body fields",
    )
    require(
        all(_equal(bodies[0][key], bodies[1][key]) for key in compare_keys),
        "paired rollout semantic state exactness",
    )
    constructors = []
    for body in bodies:
        record = body["constructor_receipt"]
        constructor.check(
            record, source=body["source"], prepared_caller=body["caller_rng_states"]
        )
        constructors.append(
            {
                key: (
                    {k: v for k, v in child.items() if k in constructor.PRIVATE_KEYS}
                    if key == "state_sha256"
                    else child
                )
                for key, child in record.items()
                if key != "caller_states"
            }
        )
    require(
        _equal(*constructors),
        "paired constructor private streams and physical fields exact",
    )
    return {
        "protocol": PAIR_PROTOCOL,
        "source": bodies[0]["source"],
        "launch_sha256": bodies[0]["launch_sha256"],
        "seed": 653,
        "attempts": [bodies[0]["attempt"], bodies[1]["attempt"]],
        "semantic_fields_compared": sorted(compare_keys),
        "constructor_semantics_exact": True,
        "only_ignored_fields": sorted(excluded),
        "paired_semantics_exact": True,
        "native_pair_replay_qualified": False,
        "training_admitted": False,
        **FALSE_FLAGS,
    }
