"""Synthetic CUDA-hidden tests for the whole-byte rollout evidence reader."""

from copy import deepcopy
from hashlib import sha256
import io

import pytest
import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_cuda_policy_probe as base
from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_record_replay as record_replay
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck import stance_recovery_cuda_storage_evidence as storage_evidence
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_cuda_constructor_rng as constructor
from test_stance_recovery_cuda_record_archive import (
    _control_state,
    _declaration,
    _plant,
    _record,
    _records,
)

SOURCE = "b" * 40
LAUNCH_SHA = "c" * 64
PARENT_HASH = "d" * 64


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


def _launch():
    return {
        "source": SOURCE,
        "schedule": _declaration(),
        "compiled_plant": _plant(),
    }


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if type(left) is dict or type(right) is dict:
        return (
            type(left) is type(right) is dict
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _prepared_and_body(*, count=1):
    records = _records(count, _declaration())
    prepared_private = records[0]["private_rng_state_before"].clone()
    states = {
        "actor": {"weight": torch.tensor([1.0], dtype=torch.float32)},
        "critic": {"weight": torch.tensor([2.0], dtype=torch.float32)},
    }
    optimizer = {"state": {}, "param_groups": []}
    caller = {
        "cpu_before": torch.random.get_rng_state().clone(),
        "cpu_after": torch.random.get_rng_state().clone(),
        "cuda_before": torch.tensor([3, 4], dtype=torch.uint8),
        "cuda_after": torch.tensor([3, 4], dtype=torch.uint8),
    }
    prepared = {
        "metadata": {},
        "actor_critic_states": deepcopy(states),
        "optimizer_state_dict": deepcopy(optimizer),
        "storage": {},
        "caller_rng_states": deepcopy(caller),
        "private_cuda_rng_state": prepared_private,
    }
    body_archive = {
        "protocol": archive.PROTOCOL,
        "declaration": _declaration(),
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": records,
        **transition.FALSE_FLAGS,
    }
    body = {
        "protocol": evidence.PROTOCOL,
        "source": SOURCE,
        "launch_sha256": LAUNCH_SHA,
        "seed": 653,
        "attempt": "capture",
        "initial_frame": {"fixture": torch.tensor([5], dtype=torch.int64)},
        "initial_control_state": deepcopy(_control_state()),
        "constructor_receipt": _constructor_receipt(caller),
        "archive": body_archive,
        "storage": {"fixture": torch.tensor([6], dtype=torch.int64)},
        "model_state_after": deepcopy(states),
        "optimizer_state_after": deepcopy(optimizer),
        "caller_rng_states": deepcopy(caller),
        "private_cuda_state_final": records[-1]["private_rng_state"].clone(),
        "elapsed_seconds": 12.5,
        **transition.FALSE_FLAGS,
    }
    prep_buffer = io.BytesIO()
    torch.save(prepared, prep_buffer)
    prep_raw = prep_buffer.getvalue()
    prep_hash = sha256(prep_raw).hexdigest()
    summary = {
        "protocol": base.PROTOCOL + ":child-summary-v1",
        "source": SOURCE,
        "launch_sha256": LAUNCH_SHA,
        "seed": 653,
        "payload_sha256": prep_hash,
        "payload_bytes": len(prep_raw),
    }
    raw = evidence.encode(body)
    return body, raw, prepared, prep_raw, prep_hash, summary


def _constructor_receipt(caller):
    # CPU seed-bound fields; CUDA endpoints remain explicitly synthetic bytes.
    private = {
        "cpu_start": torch.Generator(device="cpu")
        .manual_seed(constructor.CPU_SEED)
        .get_state(),
        "cpu_end": torch.Generator(device="cpu")
        .manual_seed(constructor.CPU_SEED)
        .get_state(),
        "cuda_start": torch.tensor([7, 8], dtype=torch.uint8),
        "cuda_end": torch.tensor([9, 10], dtype=torch.uint8),
    }
    return dict(
        protocol=constructor.PROTOCOL,
        source=SOURCE,
        cpu_seed=constructor.CPU_SEED,
        cuda_seed=constructor.CUDA_SEED,
        worlds=64,
        status="success",
        constructor_calls=1,
        faulted=False,
        caller_states=deepcopy(caller),
        private_states=private,
        state_sha256={
            key: constructor._digest(value)
            for group in (caller, private)
            for key, value in group.items()
        },
        nominal_parameters={
            key: torch.full((64, 1), value)
            for key, value in constructor.NOMINAL.items()
        },
        caller_cpu_preserved=True,
        caller_cuda_preserved=True,
        error_type=None,
        error=None,
        **constructor.FLAGS,
    )


def _install_synthetic_scoring(monkeypatch, *, archive_validator=None):
    events = []

    def score_payload(*args):
        events.append("preparation-score-load")
        assert sha256(args[0]).hexdigest() == args[1]["payload_sha256"]
        assert args[1]["source"] == args[2]["source"] == SOURCE
        assert args[-1] == 653
        return {"state_sha256": PARENT_HASH}

    monkeypatch.setattr(base, "_score_payload", score_payload)
    monkeypatch.setattr(checkpoint, "state_hash", lambda _state: PARENT_HASH)
    monkeypatch.setattr(base.preparation.parent, "PARENT_STATE_SHA256", PARENT_HASH)

    def check_archive(value, declaration, compiled, *, seed):
        events.append("archive-check")
        if archive_validator is not None:
            return archive_validator(value, declaration, compiled, seed=seed)
        assert value["declaration"] == declaration
        assert value["compiled_plant"] == compiled
        assert seed == 653
        return {
            "complete_short_window": False,
            "pulse": {"complete_nonzero_pulse_delivery": False},
            "records": len(value["records"]),
        }

    monkeypatch.setattr(archive, "check", check_archive)
    monkeypatch.setattr(
        storage_evidence,
        "check",
        lambda _storage, _records: {"protocol": "synthetic-storage", "ok": True},
    )

    def replay_check(value, initial, declaration, compiled, *, seed, cpu_profile):
        events.append("record-replay-check")
        assert initial == {"fixture": torch.tensor([5], dtype=torch.int64)}
        assert declaration == _launch()["schedule"]
        assert compiled == _launch()["compiled_plant"]
        assert seed == 653 and cpu_profile == {"synthetic": True}
        score = check_archive(value, declaration, compiled, seed=seed)
        return {"archive": score, "cpu_only": True}

    monkeypatch.setattr(record_replay, "check", replay_check)
    return events


def _verify_args(bundle, body=None, raw=None):
    original_body, original_raw, _prepared, prep_raw, prep_hash, summary = bundle
    return (
        original_raw if raw is None else raw,
        sha256(original_raw).hexdigest(),
        prep_raw,
        prep_hash,
        summary,
        _launch(),
        LAUNCH_SHA,
        {"cpu_math_profile": {"synthetic": True}},
    )


def test_round_trip_keeps_short_terminal_prefix_diagnostic_non_admitting(monkeypatch):
    bundle = _prepared_and_body(count=1)
    body, raw = bundle[0], bundle[1]
    # The archive checker is an explicit synthetic seam here; the retained row
    # models a first-terminal prefix without claiming native execution.
    body["archive"]["records"][-1]["terminated"][:] = True
    before_verify = deepcopy(body)
    raw = evidence.encode(body)
    events = _install_synthetic_scoring(monkeypatch)
    args = list(_verify_args(bundle, body=body, raw=raw))
    args[1] = sha256(raw).hexdigest()
    args[7] = {"cpu_math_profile": {"synthetic": True}}
    decoded, score = evidence.verify(*args, attempt="capture")

    assert decoded["attempt"] == "capture"
    assert score["retained_first_terminal_prefix"] is True
    assert score["complete_28_call_gate"] is False
    assert score["complete_nonzero_pulse_gate"] is False
    assert score["native_rollout_qualified"] is False
    assert score["cuda_physics_reexecuted"] is False
    assert score["solver_reexecuted"] is False
    assert events[:2] == ["preparation-score-load", "archive-check"]
    assert _tree_equal(decoded, before_verify)
    assert _tree_equal(body, before_verify)


@pytest.mark.parametrize("bad", ["body", "prepared"])
def test_both_whole_hashes_precede_any_weights_only_load(monkeypatch, bad):
    bundle = _prepared_and_body()
    events = _install_synthetic_scoring(monkeypatch)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail("weights-only load before both hash checks"),
    )
    args = list(_verify_args(bundle))
    if bad == "body":
        args[1] = "0" * 64
        expected = "whole rollout body hash"
    else:
        args[3] = "0" * 64
        expected = "whole preparation bytes hash"
    with pytest.raises(ValueError, match=expected):
        evidence.verify(*args, attempt="capture")
    assert events == []


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "protocol",
            "football-b1d-cuda64-rollout-body-v1",
            "source/launch/seed/attempt",
        ),
        ("source", "f" * 40, "source/launch/seed/attempt"),
        ("launch_sha256", "e" * 64, "source/launch/seed/attempt"),
        ("seed", 659, "source/launch/seed/attempt"),
        ("attempt", "other", "source/launch/seed/attempt"),
        ("elapsed_seconds", 480.0, "positive bounded"),
        ("training_update_performed", True, "exact non-admitting"),
    ],
)
def test_body_binding_and_non_admission_schema_damage_refused(
    monkeypatch, field, value, match
):
    bundle = _prepared_and_body()
    body = deepcopy(bundle[0])
    body[field] = value
    raw = evidence.encode(body)
    _install_synthetic_scoring(monkeypatch)
    args = list(_verify_args(bundle, raw=raw))
    args[1] = sha256(raw).hexdigest()
    with pytest.raises(ValueError, match=match):
        evidence.verify(*args, attempt="capture")


def test_archive_cursor_damage_reaches_real_archive_validator(monkeypatch):
    declaration, compiled = _declaration(), _plant()
    records = [
        _record(1, declaration)
    ]  # Wrong cursor: first retained step is not zero.
    body, raw, _prepared, prep_raw, prep_hash, summary = _prepared_and_body()
    body["archive"] = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": compiled,
        "learner_seed": 653,
        "records": records,
        **transition.FALSE_FLAGS,
    }
    body["private_cuda_state_final"] = records[-1]["private_rng_state"].clone()
    prepared = torch.load(io.BytesIO(prep_raw), map_location="cpu", weights_only=True)
    prepared["private_cuda_rng_state"] = records[0]["private_rng_state_before"].clone()
    prep_buffer = io.BytesIO()
    torch.save(prepared, prep_buffer)
    prep_raw = prep_buffer.getvalue()
    prep_hash = sha256(prep_raw).hexdigest()
    summary = {**summary, "payload_sha256": prep_hash, "payload_bytes": len(prep_raw)}
    raw = evidence.encode(body)
    events = _install_synthetic_scoring(monkeypatch, archive_validator=archive.check)
    # Restore the actual function after saving it as callback.
    real_archive_check = archive.check
    monkeypatch.setattr(archive, "check", real_archive_check)
    args = (
        raw,
        sha256(raw).hexdigest(),
        prep_raw,
        prep_hash,
        summary,
        _launch(),
        LAUNCH_SHA,
        {"cpu_math_profile": {"synthetic": True}},
    )
    with pytest.raises(ValueError):
        evidence.verify(*args, attempt="capture")
    assert "archive-check" in events


@pytest.mark.parametrize(
    "damage,match",
    [
        ("model", "exact frozen D1 parent"),
        ("optimizer", "exact empty prepared Adam"),
        ("caller-cpu", "prepared caller CPU/CUDA streams"),
        ("caller-cuda", "prepared caller CPU/CUDA streams"),
        ("private-initial", "private CUDA stream endpoints"),
        ("private-final", "private CUDA stream endpoints"),
        ("initial-control", "bind to first transition"),
    ],
)
def test_model_optimizer_and_rng_boundary_damage_refused(monkeypatch, damage, match):
    bundle = _prepared_and_body()
    body = deepcopy(bundle[0])
    if damage == "model":
        body["model_state_after"]["actor"]["weight"][0] += 1
    elif damage == "optimizer":
        body["optimizer_state_after"]["param_groups"].append({"lr": 0.1})
    elif damage == "caller-cpu":
        body["caller_rng_states"]["cpu_after"][0] ^= 1
    elif damage == "caller-cuda":
        body["caller_rng_states"]["cuda_before"][0] ^= 1
    elif damage == "private-initial":
        body["archive"]["records"][0]["private_rng_state_before"][0] ^= 1
    elif damage == "private-final":
        body["private_cuda_state_final"][0] ^= 1
    else:
        body["initial_control_state"]["ctrl"][0, 0] = 0.1
    raw = evidence.encode(body)
    _install_synthetic_scoring(monkeypatch)
    args = list(_verify_args(bundle, raw=raw))
    args[1] = sha256(raw).hexdigest()
    with pytest.raises(ValueError, match=match):
        evidence.verify(*args, attempt="capture")


def test_pair_ignores_only_per_process_rng_attempt_and_elapsed_after_scoring():
    caller = dict(
        cpu_before=torch.random.get_rng_state().clone(),
        cpu_after=torch.random.get_rng_state().clone(),
        cuda_before=torch.tensor([3, 4], dtype=torch.uint8),
        cuda_after=torch.tensor([3, 4], dtype=torch.uint8),
    )
    left_body = {
        "protocol": evidence.PROTOCOL,
        "source": SOURCE,
        "launch_sha256": LAUNCH_SHA,
        "seed": 653,
        "attempt": "capture",
        "elapsed_seconds": 4.0,
        "caller_rng_states": caller,
        "constructor_receipt": _constructor_receipt(caller),
        "initial_frame": {"qpos": torch.zeros(1)},
        "initial_control_state": {"ctrl": torch.zeros(1)},
        "archive": {"steps": [0, 1]},
        "storage": {"step": 1},
        "model_state_after": {"weight": torch.ones(1)},
        "optimizer_state_after": {"state": {}},
        "private_cuda_state_final": torch.tensor([5], dtype=torch.uint8),
    }
    right_body = deepcopy(left_body)
    right_body.update(attempt="replay", elapsed_seconds=5.0)
    right_body["caller_rng_states"]["cpu_before"][0] = 9
    right_body["caller_rng_states"]["cpu_after"][0] = 9
    right_body["constructor_receipt"] = _constructor_receipt(
        right_body["caller_rng_states"]
    )
    left_body.update(evidence.FALSE_FLAGS)
    right_body.update(evidence.FALSE_FLAGS)
    result = evidence.paired(left_body, right_body)
    assert result["paired_semantics_exact"] is True
    assert set(result["only_ignored_fields"]) == {
        "attempt",
        "caller_rng_states",
        "elapsed_seconds",
        "constructor_receipt.caller_states",
        "constructor_receipt.state_sha256.caller_endpoints",
    }
    right_body["initial_frame"]["qpos"] = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="paired rollout semantic"):
        evidence.paired(left_body, right_body)
    right_body["initial_frame"]["qpos"] = torch.zeros(1, dtype=torch.float32)
    right_body["storage"]["step"] = 2
    with pytest.raises(ValueError, match="paired rollout semantic"):
        evidence.paired(left_body, right_body)


def test_reader_refuses_visible_cuda_before_any_deserialization(monkeypatch):
    bundle = _prepared_and_body()
    _install_synthetic_scoring(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail("visible CUDA must refuse before deserialization"),
    )
    with pytest.raises(ValueError, match="CUDA-hidden rollout evidence reader"):
        evidence.verify(*_verify_args(bundle), attempt="capture")


def test_pair_refuses_visible_cuda(monkeypatch):
    left = {"not": "a body"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA-hidden rollout pair comparison"):
        evidence.paired(left, left)


def test_pair_refuses_changed_constructor_private_endpoint_even_with_matching_hash():
    left = _prepared_and_body()[0]
    right = deepcopy(left)
    right["attempt"] = "replay"
    ctor = right["constructor_receipt"]
    ctor["private_states"]["cuda_end"][0] ^= 1
    ctor["state_sha256"]["cuda_end"] = constructor._digest(
        ctor["private_states"]["cuda_end"]
    )
    with pytest.raises(ValueError, match="paired constructor private streams"):
        evidence.paired(left, right)


def test_encode_honors_archive_limit_during_serialization(monkeypatch):
    monkeypatch.setattr(archive, "LIMIT", 128)
    with pytest.raises(ValueError, match="bounded CUDA64 rollout body bytes"):
        evidence.encode({"payload": torch.zeros(4096, dtype=torch.float32)})
