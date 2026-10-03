"""SYNTHETIC CPU sample records, never CUDA capture/replay qualification."""

from copy import deepcopy
import io
import math
from pathlib import Path

import pytest
import torch

from mjlab_microduck import stance_recovery_cuda_shadow_evidence as evidence

SOURCE = "b" * 40


@pytest.fixture
def pair_input():
    native = evidence.preparation_probe.execution.ROOT
    repo = Path(evidence.__file__).resolve().parents[2]
    candidates = [
        native,
        *sorted(
            (repo / "artifacts/retained").glob("cuda64-reader-fixture-48653c58121d.*")
        ),
    ]
    relative = "artifacts/evaluations/stance-wsl-cuda64-policy-preparation-48653c58121d/seed-653.pt"
    for root in candidates:
        if (root / relative).is_file():
            raw = evidence.preparation_probe._read_file(
                root / relative, evidence.preparation_probe.RAW_LIMIT
            )
            assert (
                evidence.preparation_probe.digest(raw)
                == "aef5a5d0f838aad8e3fb172adeba42b71d692982639fecbcd17580d8097e7fe1"
            )
            prepared = torch.load(
                io.BytesIO(raw), map_location="cpu", weights_only=True
            )
            break
    else:
        if native == Path("/home/yanbo/work/microduck_rl-stance-replication-20260930"):
            pytest.fail("native host lacks exact closed preparation fixtures")
        pytest.skip(
            "optional authentic preparation tensors missing in external checkout"
        )
    # All sampler outputs and boundaries below are explicitly synthetic CPU data.
    # The old unchanged model/storage/Adam bytes do not authenticate these draws.
    generator = torch.Generator(device="cpu").manual_seed(653)
    values = {
        k: torch.zeros(shape, dtype=torch.float32)
        for k, shape in evidence.SHAPES.items()
    }
    values["distribution_mean"].fill_(0.15)
    values["distribution_std"].fill_(0.3)
    values["values"].fill_(0.1)
    values["actions"] = torch.randn(28, 64, 10, generator=generator) * 0.3 + 0.15
    mean, std = (
        values["distribution_mean"].double(),
        values["distribution_std"].double(),
    )
    values["actions_log_prob"] = (
        (
            -0.5 * ((values["actions"].double() - mean) / std).square()
            - std.log()
            - 0.5 * math.log(2 * math.pi)
        )
        .sum(-1)
        .float()
    )
    initial = prepared["private_cuda_rng_state"].clone()
    boundaries = [initial.clone() for _ in range(29)]
    for i, v in enumerate(boundaries):
        if i:
            v[-1] = (int(initial[-1]) + i) % 256
    hashes = [evidence.sampling._state_digest(v) for v in boundaries]
    caller = {k: v.clone() for k, v in prepared["caller_rng_states"].items()}
    caller_hashes = {k: evidence.sampling._state_digest(v) for k, v in caller.items()}
    parent_hash = evidence.sampling.preparation.parent.PARENT_STATE_SHA256
    receipt = {
        "protocol": evidence.sampling.PROTOCOL,
        "source": SOURCE,
        "seed": 653,
        "worlds": 64,
        "horizon": 28,
        "device": "cuda:0",
        "synthetic_zero_observations": True,
        "stock_sampler_calls": 28,
        "private_state_installed_around_stock_sampler": True,
        "private_rng_scope_count": 1,
        "private_state_boundary_sha256": hashes,
        "parent_state_sha256_before": parent_hash,
        "parent_state_sha256_after": parent_hash,
        "storage_step_before": 0,
        "storage_step_after": 0,
        "caller_cpu_rng_unchanged": True,
        "caller_cuda_rng_unchanged": True,
        "caller_rng_state_sha256": caller_hashes,
        "optimizer_steps": 0,
        **dict.fromkeys(evidence.FALSE_RECEIPT_FIELDS, False),
    }
    scope = {
        "protocol": evidence.rng_scope.PROTOCOL,
        "source": SOURCE,
        "seed": 653,
        "device": "cuda:0",
        "state_bytes": initial.numel(),
        "initial_state_sha256": hashes[0],
        "private_state_sha256": hashes[-1],
        "scope_count": 1,
        "caller_cpu_rng_preserved": True,
        "caller_cuda_rng_preserved": True,
        "scope_active": False,
        "faulted": False,
        "native_cuda_rng_scope_qualified": False,
        "native_cuda_sampling_qualified": False,
        **evidence.sampling.FALSE_FLAGS,
    }
    values.update(
        receipt=receipt,
        private_cuda_state_boundaries=boundaries,
        model_state_before=deepcopy(prepared["actor_critic_states"]),
        model_state_after=deepcopy(prepared["actor_critic_states"]),
        optimizer_state_before=deepcopy(prepared["optimizer_state_dict"]),
        optimizer_state_after=deepcopy(prepared["optimizer_state_dict"]),
        caller_rng_states=caller,
        storage_before={"step": 0, **deepcopy(prepared["storage"])},
        storage_after={"step": 0, **deepcopy(prepared["storage"])},
        scope_receipt=scope,
    )
    return {"sampling": values, "prepared_snapshot": prepared}


def score(value):
    return evidence.score_sampling(
        value["sampling"], value["prepared_snapshot"], source=SOURCE, seed=653
    )


def test_synthetic_cpu_record_scores_without_native_admission(pair_input):
    result = score(pair_input)
    assert result["cuda_math_replayed"] is False
    assert result["stock_sampler_calls"] == 28
    assert all(result[k] is False for k in evidence.sampling.FALSE_FLAGS)


def test_synthetic_pair_compares_all_numerical_outputs_not_native_replay(pair_input):
    result = evidence.compare_pair(
        pair_input, deepcopy(pair_input), source=SOURCE, seed=653
    )
    assert result["paired_numerical_outputs_exact"] is True
    assert result["native_cuda_replay_authenticated_by_this_function"] is False


@pytest.mark.parametrize("field", sorted(evidence.FALSE_RECEIPT_FIELDS))
def test_every_false_flag_refuses_admission(pair_input, field):
    pair_input["sampling"]["receipt"][field] = True
    with pytest.raises(ValueError, match="without learning or admission"):
        score(pair_input)


@pytest.mark.parametrize(
    "field",
    [
        "worlds",
        "horizon",
        "stock_sampler_calls",
        "private_rng_scope_count",
        "storage_step_before",
        "storage_step_after",
        "optimizer_steps",
    ],
)
def test_boolean_is_not_integer_counter(pair_input, field):
    pair_input["sampling"]["receipt"][field] = True
    with pytest.raises(ValueError, match="without learning or admission"):
        score(pair_input)


@pytest.mark.parametrize("field", list(evidence.SHAPES))
def test_nonfinite_tensor_refused(pair_input, field):
    pair_input["sampling"][field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="typed finite"):
        score(pair_input)


@pytest.mark.parametrize("value", [0, -0.1])
def test_nonpositive_gaussian_scale_refused(pair_input, value):
    pair_input["sampling"]["distribution_std"].fill_(value)
    with pytest.raises(ValueError, match="positive retained Gaussian"):
        score(pair_input)


def test_nonzero_input_refused(pair_input):
    pair_input["sampling"]["actor_observations"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="immutable synthetic zeros"):
        score(pair_input)


def test_mismatched_log_probability_refused(pair_input):
    pair_input["sampling"]["actions_log_prob"][0, 0] += 0.01
    with pytest.raises(ValueError, match="Gaussian log probability"):
        score(pair_input)


@pytest.mark.parametrize("field", ["distribution_mean", "distribution_std", "values"])
def test_changed_fixed_input_output_refused(pair_input, field):
    pair_input["sampling"][field][1, 0, 0] += 0.01
    with pytest.raises(ValueError, match="stay constant across calls"):
        score(pair_input)


@pytest.mark.parametrize("field", ["model_state_before", "model_state_after"])
def test_changed_actual_weight_refused(pair_input, field):
    values = pair_input["sampling"][field]["actor"]
    next(iter(values.values())).reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="actual model snapshots unchanged"):
        score(pair_input)


@pytest.mark.parametrize("field", ["storage_before", "storage_after"])
def test_changed_storage_refused(pair_input, field):
    pair_input["sampling"][field]["actions"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="unchanged zero storage"):
        score(pair_input)


@pytest.mark.parametrize("field", ["optimizer_state_before", "optimizer_state_after"])
def test_changed_adam_refused(pair_input, field):
    pair_input["sampling"][field]["state"][0] = {"step": 1}
    with pytest.raises(ValueError, match="empty Adam unchanged"):
        score(pair_input)


def test_caller_boundary_mutation_refused(pair_input):
    pair_input["sampling"]["caller_rng_states"]["cpu_after"][0] ^= 1
    with pytest.raises(ValueError, match="actual prepared caller streams"):
        score(pair_input)


def test_private_nonadvance_refused(pair_input):
    b = pair_input["sampling"]["private_cuda_state_boundaries"]
    b[2] = b[1].clone()
    with pytest.raises(ValueError, match="advances on every retained draw"):
        score(pair_input)


def test_scope_fault_cannot_borrow_outer_receipt(pair_input):
    pair_input["sampling"]["scope_receipt"]["faulted"] = True
    with pytest.raises(ValueError, match="scope boundary receipt"):
        score(pair_input)


def test_pair_excludes_only_individually_preserved_caller_streams(pair_input):
    replay = deepcopy(pair_input)
    for family in ("cpu", "cuda"):
        for suffix in ("before", "after"):
            key = family + "_" + suffix
            replay["sampling"]["caller_rng_states"][key][0] ^= 1
            replay["prepared_snapshot"]["caller_rng_states"][key][0] ^= 1
            replay["sampling"]["receipt"]["caller_rng_state_sha256"][key] = (
                evidence.sampling._state_digest(
                    replay["sampling"]["caller_rng_states"][key]
                )
            )
    assert evidence.compare_pair(pair_input, replay, source=SOURCE, seed=653)[
        "paired_numerical_outputs_exact"
    ]


def test_finite_self_consistent_but_different_draws_fail_pair(pair_input):
    replay = deepcopy(pair_input)
    sample = replay["sampling"]
    sample["actions"] *= -1
    mean, std = (
        sample["distribution_mean"].double(),
        sample["distribution_std"].double(),
    )
    sample["actions_log_prob"] = (
        (
            -0.5 * ((sample["actions"].double() - mean) / std).square()
            - std.log()
            - 0.5 * math.log(2 * math.pi)
        )
        .sum(-1)
        .float()
    )
    assert score(replay)["cuda_math_replayed"] is False
    with pytest.raises(ValueError, match="all paired sample tensors"):
        evidence.compare_pair(pair_input, replay, source=SOURCE, seed=653)


def test_extra_payload_and_receipt_fields_refused(pair_input):
    pair_input["sampling"]["unexpected"] = True
    with pytest.raises(ValueError, match="payload schema"):
        score(pair_input)
    del pair_input["sampling"]["unexpected"]
    pair_input["sampling"]["receipt"]["unexpected"] = True
    with pytest.raises(ValueError, match="receipt schema"):
        score(pair_input)
