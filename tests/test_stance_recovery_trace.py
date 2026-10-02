"""Short CPU collection and independent-contract tests for D1 recovery traces."""

from copy import deepcopy
import gc
from hashlib import sha256

import pytest
import torch
from torch import nn

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_runtime as runtime
from mjlab_microduck import stance_recovery_trace as recovery_trace
from mjlab_microduck import stance_attempt_trace as nominal_trace


SOURCE = "a" * 40
TRAINING_SOURCE = "be2d59661af293b0d67ae20d2e16db50514cce14"
CHECKPOINT_SHA256 = "2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5"
EXPECTED_CPU_PROFILE = profile.expected_receipt()


@pytest.fixture(autouse=True)
def release_finished_native_test_fixtures():
    """Bound native fixture lifetime after each test and its monkeypatch undo."""
    yield
    gc.collect()


def frozen_checkpoint_identity():
    return {
        "protocol": "football-b1n-evaluation-checkpoint-v1",
        "source": TRAINING_SOURCE,
        "runtime_sha256": "ef01ea829b5b69069cf732b6e252ee0e7c205e92e5f17f028753ab9ed08682cb",
        "training_launch_sha256": "8236806aa44c1d424b77e74942d1e01b6530702a55216186165856a870f48455",
        "purpose": "lean-replication",
        "training_seed": 577,
        "worlds": 64,
        "iteration": 255,
        "initial_state_sha256": "0ca246873f1143c540ecdebb6e6bc80cb826f86b558dda9c3cff8c5694263144",
        "parent_checkpoint_sha256": "46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93",
        "architecture": deepcopy(checkpoint.ARCHITECTURE),
    }


class ZeroActor(nn.Module):
    """Deterministic frozen CPU stand-in; it is not the pinned checkpoint."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, observations, *, stochastic_output=False):
        actor_input = observations["actor"]
        return actor_input.new_zeros((len(actor_input), 10)) + self.bias


def install_cpu_profile_seam(monkeypatch):
    expected = deepcopy(EXPECTED_CPU_PROFILE)
    monkeypatch.setattr(profile, "checked_receipt", lambda: deepcopy(expected))
    # Exercise the trace's archived-profile schema check without inspecting this
    # test process's initial-exec environment or CPU runtime.
    monkeypatch.setattr(profile, "check_recorded", lambda receipt: profile.validate_receipt(receipt))


def make_env(monkeypatch, cases=("+x",)):
    install_cpu_profile_seam(monkeypatch)
    env = runtime.RecoveryRuntime(list(cases), device="cpu")
    declaration = contract.declaration(SOURCE, env.binding)
    return env, declaration, frozen_checkpoint_identity(), ZeroActor()


def collect_prefix(monkeypatch, *, policy_tick_limit=1):
    env, declaration, identity, actor = make_env(monkeypatch)
    value = recovery_trace.collect(env, actor, declaration, identity,
        deadline_monotonic=__import__("time").monotonic()+60,
        policy_tick_limit=policy_tick_limit)
    return env, declaration, identity, actor, value


def test_real_cpu_frozen_actor_collection_retains_bounded_pre_pulse_prefix(monkeypatch):
    env, declaration, identity, actor, value = collect_prefix(monkeypatch, policy_tick_limit=1)

    assert env.device.type == "cpu" and env.forward_graph is None
    assert not actor.bias.requires_grad
    assert value["protocol"] == contract.TRACE_PROTOCOL
    assert value["declaration"] == declaration
    assert value["backend"] == {"torch_device": "cpu", "warp_is_cuda": False}
    assert value["collection"]["policy_ticks"] == 1
    assert value["collection"]["stop_reason"] == "policy-tick-limit"
    assert value["collection"]["elapsed_seconds"] > 0
    assert len(value["payload"]["ticks"]) == 1
    assert value["payload"]["ticks"][0]["actions"].shape == (1, 10)
    assert not value["payload"]["ticks"][0]["actions"].any()
    assert value["payload"]["binding"]["checkpoint_identity"] == identity
    assert all(value[name] is False for name in contract.FALSE_FLAGS)

    pulse_result = recovery_trace.check_pulses(value)
    assert pulse_result["checked_physics_steps"] == 10
    assert pulse_result["pulse_window_steps"] == 0
    assert pulse_result["delivered_nonzero_pulse_steps"] == 0
    assert pulse_result["complete_pulse_delivery"] is False
    assert len(value["pulse_evidence"]) == 1
    for row in value["pulse_evidence"][0]:
        assert not row["pre_xfrc"].any() and not row["post_xfrc"].any()
        assert not row["pre_qfrc"].any() and not row["post_qfrc"].any()
        assert row["phases"] is None
    controls = value["control_evidence"]["ticks"][0]["proposals"]
    assert len(controls) == 10
    assert all(proposal["accepted"].tolist() == [True] for proposal in controls)
    assert all(row["accepted"].tolist() == proposal["accepted"].tolist()
               for row, proposal in zip(value["pulse_evidence"][0], controls))


def test_collected_pulse_ledger_is_per_tick_across_multiple_policy_ticks(monkeypatch):
    _env, _declaration, _identity, _actor, value = collect_prefix(
        monkeypatch, policy_tick_limit=2)
    assert value["collection"]["policy_ticks"] == 2
    assert len(value["payload"]["ticks"]) == 2
    assert [len(entries) for entries in value["pulse_evidence"]] == [10, 10]
    receipt = recovery_trace.check_pulses(value)
    assert receipt["checked_physics_steps"] == 20
    assert receipt["pulse_window_steps"] == 0
    assert receipt["delivered_nonzero_pulse_steps"] == 0


def test_nominal_trace_rejects_recovery_binding_and_recovery_recorder_requires_mode(monkeypatch):
    env, declaration, identity, _actor, value = collect_prefix(monkeypatch)
    binding = value["payload"]["binding"]
    initial = value["payload"]["initial"]

    with pytest.raises(ValueError, match="exact trace binding fields"):
        nominal_trace.FirstAttemptTrace(binding, initial)

    capture = recovery_trace.RecoveryTrace(binding, initial, declaration, replay=False)
    assert capture._tensor_device == "cpu"
    replay = recovery_trace.RecoveryTrace(binding, initial, declaration, replay=True)
    assert replay._tensor_device == "cpu"
    with pytest.raises(ValueError, match="explicit recorded recovery replay mode"):
        recovery_trace.RecoveryTrace(binding, initial, declaration, replay=1)


def test_recovery_trace_append_checks_continuity_and_faults_permanently(monkeypatch):
    _env, declaration, _identity, _actor, value = collect_prefix(monkeypatch)
    binding = value["payload"]["binding"]
    tick = value["payload"]["ticks"][0]
    recorder = recovery_trace.RecoveryTrace(binding, value["payload"]["initial"], declaration)
    result = {key: deepcopy(part) for key, part in tick.items()
              if key not in ("actor_input", "actions")}
    recorder.append(result, tick["actor_input"], tick["actions"])
    assert len(recorder.ticks) == 1

    damaged = deepcopy(result)
    damaged["boundaries"][0]["physics_steps"][0] += 1
    recorder = recovery_trace.RecoveryTrace(binding, value["payload"]["initial"], declaration)
    with pytest.raises(ValueError, match="tick boundary counter continuity"):
        recorder.append(damaged, tick["actor_input"], tick["actions"])
    assert recorder.faulted is True
    with pytest.raises(ValueError, match="faulted trace requires closeout"):
        recorder.append(result, tick["actor_input"], tick["actions"])


@pytest.mark.parametrize("damage", ["foreign-declaration", "foreign-source", "unknown-case", "wrong-profile"])
def test_recovery_trace_refuses_foreign_declaration_binding_or_profile(monkeypatch, damage):
    _env, declaration, _identity, _actor, value = collect_prefix(monkeypatch)
    binding = deepcopy(value["payload"]["binding"])
    changed_declaration = deepcopy(declaration)
    if damage == "foreign-declaration":
        changed_declaration["total_physics_steps"] += 1
    elif damage == "foreign-source":
        binding["source"] = "c" * 40
    elif damage == "unknown-case":
        binding["case"] = "diagonal"
    else:
        binding["cpu_math_profile"]["settings"]["OMP_NUM_THREADS"] = "2"

    with pytest.raises(ValueError):
        recovery_trace.RecoveryTrace(binding, value["payload"]["initial"], changed_declaration)


def synthetic_pulse_value(monkeypatch):
    """One real CPU pulse tick at synthetic counter 500; control-flow only."""
    env, declaration, _identity, _actor = make_env(monkeypatch, cases=("+x",))
    env.steps.fill_(contract.ONSET_STEP)
    env._view("time").fill_(contract.ONSET_STEP*contract.force.DT)
    refresh = env._refresh

    def supported(rows):
        refresh(rows)
        env.state.support[rows] = 1.

    env._refresh = supported
    env.state.support.fill_(1.)
    result = env.step_with_pulse(torch.zeros(1, 10), capture_control=True)
    # This is an isolated pulse-window control-flow fixture, not a physical
    # 500-step trajectory or first-attempt replay.
    value = dict(declaration=declaration,
        payload=dict(binding=contract.binding(SOURCE, env.binding, "+x", "cpu",
            frozen_checkpoint_identity(), EXPECTED_CPU_PROFILE),
            ticks=[dict(boundaries=result["boundaries"])]),
        pulse_evidence=[result["pulse_evidence"]],
        control_evidence=dict(ticks=[result["control_evidence"]]))
    return value


def test_check_pulses_accepts_real_synthetic_window_phases_only_inside_window(monkeypatch):
    value = synthetic_pulse_value(monkeypatch)
    receipt = recovery_trace.check_pulses(value)

    assert receipt["checked_physics_steps"] == 10
    assert receipt["pulse_window_steps"] == contract.PULSE_STEPS
    assert receipt["delivered_nonzero_pulse_steps"] == contract.PULSE_STEPS
    assert receipt["complete_pulse_delivery"] is True
    entries = value["pulse_evidence"][0]
    assert [int(row["before_steps"][0]) for row in entries] == list(range(500, 510))
    body_id = value["declaration"]["plant"]["body_id"]
    for row in entries:
        assert row["phases"] is not None
        assert row["pre_xfrc"][0, body_id, 0] == 2
        assert not row["post_xfrc"].any()
        assert not row["pre_qfrc"].any() and not row["post_qfrc"].any()
        assert row["phases"]["forced_pre"]["inputs"]["xfrc_applied"][0, body_id, 0] == 2
        assert not row["phases"]["unforced_post"]["inputs"]["xfrc_applied"].any()


@pytest.mark.parametrize("damage", ["missing-event", "wrong-force", "post-leak", "missing-phase"])
def test_check_pulses_rejects_missing_or_mismatched_force_evidence(monkeypatch, damage):
    value = synthetic_pulse_value(monkeypatch)
    changed = deepcopy(value)
    row = changed["pulse_evidence"][0][0]
    if damage == "missing-event":
        changed["pulse_evidence"][0].pop()
    elif damage == "wrong-force":
        row["pre_xfrc"][0, 2, 0] = 1
    elif damage == "post-leak":
        row["post_xfrc"].copy_(row["pre_xfrc"])
    else:
        row["phases"].pop("unforced_post")

    with pytest.raises(ValueError):
        recovery_trace.check_pulses(changed)


def test_score_and_encoded_artifact_replay_actor_control_pulses_without_admission(monkeypatch):
    _env, declaration, _identity, actor, value = collect_prefix(monkeypatch)
    loaded = []

    def load_checkpoint(raw, expected_sha256, identity):
        loaded.append((raw, expected_sha256, identity))
        return actor, {"strict_actor_restore": True}

    monkeypatch.setattr(checkpoint, "load_lean_replication_evaluation", load_checkpoint)
    result = recovery_trace.score(value, b"synthetic bytes behind local test seam", declaration)
    assert result["case"] == "+x"
    assert result["strict_actor_restore"] is True
    assert result["actor_replay_max_abs_error"] == 0
    assert result["pulse"]["checked_physics_steps"] == 10
    assert all(result[key] is False for key in contract.FALSE_FLAGS)
    assert len(loaded) == 1 and loaded[0][1] == CHECKPOINT_SHA256
    assert loaded[0][2] == frozen_checkpoint_identity()

    raw = recovery_trace.encode(value)
    assert isinstance(raw, bytes) and 0 < len(raw) <= contract.CAPTURE_LIMIT
    encoded_result = recovery_trace.verify(raw, sha256(raw).hexdigest(), b"synthetic bytes behind local test seam", declaration)
    assert encoded_result == result


def test_score_rejects_actor_changed_action_and_admission_or_binding_before_checkpoint_load(monkeypatch):
    _env, declaration, _identity, actor, value = collect_prefix(monkeypatch)
    loaded = []

    def load_checkpoint(*args):
        loaded.append(args)
        return actor, {"strict_actor_restore": True}

    monkeypatch.setattr(checkpoint, "load_lean_replication_evaluation", load_checkpoint)
    changed_action = deepcopy(value)
    changed_action["payload"]["ticks"][0]["actions"][0, 0] = .1
    with pytest.raises(ValueError, match="exact portable frozen actor replay"):
        recovery_trace.score(changed_action, b"local-test-seam", declaration)
    assert loaded

    loaded.clear()
    admitted = deepcopy(value)
    admitted["recovery_accepted"] = True
    with pytest.raises(ValueError, match="recovery evidence never admits"):
        recovery_trace.score(admitted, b"local-test-seam", declaration)
    assert loaded == []

    malformed_binding = deepcopy(value)
    malformed_binding["payload"]["binding"]["case"] = "unknown"
    with pytest.raises(ValueError):
        recovery_trace.score(malformed_binding, b"local-test-seam", declaration)
    assert loaded == []


def test_hash_mismatch_is_rejected_before_tensor_load(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("tensor load preceded hash check"))
    raw = b"untrusted serialized bytes"
    with pytest.raises(ValueError, match="byte hash before tensor loading"):
        recovery_trace.verify(raw, "0"*64, b"unused checkpoint", {"unused": True})
