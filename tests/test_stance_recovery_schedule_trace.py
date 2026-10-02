"""Independent CPU scheduled-trace checks; synthetic pins are test-only seams."""

from copy import deepcopy
from hashlib import sha256
import io
import time

import pytest
import torch

from mjlab_microduck import stance_attempt_trace as legacy_trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_runtime as runtime
from mjlab_microduck import stance_recovery_schedule_trace as trace


SOURCE = "a" * 40
SYNTHETIC_PROFILE = profile.expected_receipt()


@pytest.fixture(scope="module")
def synthetic_parent_bytes():
    """Valid tensor/identity schema with no claim to historical artifact bytes."""
    identity = trace.parent.expected_identity()
    actor, critic = checkpoint.validate_identity(identity, evaluation="lean-replication")
    stream = io.BytesIO()
    torch.save(dict(identity=identity, states=checkpoint.states_of(actor, critic)), stream)
    return stream.getvalue()


def setup_test_pins(monkeypatch, checkpoint_raw):
    """Explicitly synthetic local test seam; never archive or profile proof."""
    monkeypatch.setattr(baseline, "CHECKPOINT_SHA256", sha256(checkpoint_raw).hexdigest())
    monkeypatch.setattr(trace.profile, "checked_receipt", lambda: deepcopy(SYNTHETIC_PROFILE))
    monkeypatch.setattr(trace.profile, "check_recorded", lambda receipt: _require_test_profile(receipt))


def _require_test_profile(receipt):
    if receipt != SYNTHETIC_PROFILE:
        raise ValueError("synthetic test profile receipt mismatch")


def collect_partial(monkeypatch, checkpoint_raw):
    setup_test_pins(monkeypatch, checkpoint_raw)
    declaration = schedule.declaration(SOURCE, "gentle", "held-out", ["zero-wrench"])
    env = runtime.ScheduledRecoveryRuntime(declaration, device="cpu")
    value = trace.collect(env, checkpoint_raw, deadline_monotonic=time.monotonic()+60,
                          policy_tick_limit=1)
    return env, declaration, checkpoint_raw, value


def synthetic_pulse_value(monkeypatch, *, mutate=None):
    """Ten actual CPU substeps selected by a synthetic held-out episode clock."""
    monkeypatch.setattr(trace.profile, "checked_receipt", lambda: deepcopy(SYNTHETIC_PROFILE))
    cell = "diagonal-+--2n-10steps-t375"
    declaration = schedule.declaration(SOURCE, "gentle", "held-out", [cell])
    env = runtime.ScheduledRecoveryRuntime(declaration, device="cpu")
    # This is a bounded pulse-adapter control-flow fixture, not a full 375-step
    # first attempt or provenance-bearing trajectory.
    env.steps.fill_(375)
    env._view("time").fill_(.75)
    env.state.support.fill_(1.)
    result = env.step_with_schedule(torch.zeros(1, 10), capture_control=True)
    tick = {key: result[key] for key in legacy_trace.TICK_KEYS if key in result}
    tick.update(actor_input=torch.zeros(1, 44), actions=torch.zeros(1, 10))
    payload = {"ticks": [tick]}
    value = dict(declaration=declaration, compiled_plant=deepcopy(env.binding), payload=payload,
        control_evidence={"ticks": [result["control_evidence"]]},
        scheduled_pulse_evidence=[result["scheduled_pulse_evidence"]])
    if mutate is not None:
        mutate(value)
    return env, value


def test_scheduled_trace_has_separate_protocol_and_does_not_widen_legacy_allowlists():
    assert trace.PROTOCOL == "football-b1d-cpu-scheduled-first-attempt-v1"
    assert trace.PROTOCOL not in legacy_trace.ITERATIONS
    assert legacy_trace.ITERATIONS[legacy_trace.PROTOCOL] == (128, 256, 384, 511)
    assert legacy_trace.ITERATIONS[legacy_trace.LEAN_REPLICATION_PROTOCOL] == (64, 128, 192, 255)
    assert checkpoint.EVALUABLE["pilot"] == (128, 256, 384, 511)
    assert checkpoint.EVALUABLE["lean-replication"] == (64, 128, 192, 255)


def test_binding_is_one_heldout_world_exact_schedule_plant_parent_and_profile():
    declaration = schedule.declaration(SOURCE, "gentle", "held-out", ["zero-wrench"])
    plant = {"selected_plant": {"plant": "test-only"}, "nbody": 2,
        "body_names": ["world", "trunk_base"], "body_name": "trunk_base", "body_id": 1}
    result = trace.binding(declaration, plant, SYNTHETIC_PROFILE)
    assert result["protocol"] == trace.PROTOCOL and result["source"] == SOURCE
    assert result["worlds"] == 1 and result["capture_device"] == "cpu"
    assert result["schedule_sha256"] == schedule.binding_sha256(declaration)
    assert result["plant_sha256"] == sha256(trace.canonical(plant).encode()).hexdigest()
    assert result["checkpoint_sha256"] == baseline.CHECKPOINT_SHA256
    assert result["checkpoint_identity"] == trace.parent.expected_identity()
    assert result["evaluation_seed"] == trace.EVALUATION_SEED


@pytest.mark.parametrize("bad", [
    ("training", ["zero-wrench"]), ("held-out", ["+x-2n-10steps-t500", "zero-wrench"]),
])
def test_binding_refuses_training_split_or_multiworld_trace(bad):
    split, ids = bad
    declaration = schedule.declaration(SOURCE, "dose", split, ids)
    plant = {"selected_plant": {}, "nbody": 2, "body_names": ["world", "trunk_base"],
             "body_name": "trunk_base", "body_id": 1}
    with pytest.raises(ValueError, match="one explicitly held-out CPU scheduled attempt"):
        trace.binding(declaration, plant, SYNTHETIC_PROFILE)


def test_real_cpu_one_tick_collect_score_and_hash_verify_remain_partial_and_nonadmitting(
        monkeypatch, synthetic_parent_bytes):
    env, declaration, checkpoint_raw, value = collect_partial(monkeypatch, synthetic_parent_bytes)
    assert value["collection"]["policy_ticks"] == 1
    assert value["collection"]["stop_reason"] == "policy-tick-limit"
    assert value["backend"] == {"torch_device": "cpu", "warp_is_cuda": False}
    assert value["payload"]["ticks"][0]["boundaries"][-1]["physics_steps"].tolist() == [10]
    receipt = trace.score(value, checkpoint_raw, declaration, env.binding)
    gates = receipt["numerical_diagnostic"]["gates"]
    assert gates["full_duration"] is False and gates["complete_first_attempt"] is False
    assert receipt["numerical_diagnostic"]["candidate_pass"] is False
    assert receipt["pulse"]["checked_physics_steps"] == 10
    assert receipt["pulse"]["pulse_window_steps"] == 0
    assert receipt["pulse"]["complete_pulse_delivery"] is False
    assert receipt["whole_trajectory_physics_resimulated"] is False
    assert all(receipt[key] is False for key in baseline.FALSE_FLAGS)

    raw = trace.encode(value)
    verified = trace.verify(raw, sha256(raw).hexdigest(), checkpoint_raw, declaration, env.binding)
    assert verified["numerical_diagnostic"]["gates"]["full_duration"] is False
    assert verified["numerical_diagnostic"]["candidate_pass"] is False
    assert all(verified[key] is False for key in baseline.FALSE_FLAGS)


def test_partial_score_refuses_actor_action_mutation_and_admission_flags(monkeypatch, synthetic_parent_bytes):
    env, declaration, checkpoint_raw, value = collect_partial(monkeypatch, synthetic_parent_bytes)
    changed = deepcopy(value)
    changed["payload"]["ticks"][0]["actions"][0, 0] += .01
    with pytest.raises(ValueError, match="exact scheduled actor replay"):
        trace.score(changed, checkpoint_raw, declaration, env.binding)
    changed = deepcopy(value)
    changed["control_evidence"]["ticks"][0]["initial"]["ctrl"][0, 0] += .01
    with pytest.raises(ValueError, match="control evidence mismatch"):
        trace.score(changed, checkpoint_raw, declaration, env.binding)
    changed = deepcopy(value); changed["execution_admitted"] = True
    with pytest.raises(ValueError, match="exact CPU scheduled artifact schema|exact non-admitting CPU scheduled artifact"):
        trace.score(changed, checkpoint_raw, declaration, env.binding)
    changed = deepcopy(value); changed["training_admitted"] = True
    with pytest.raises(ValueError, match="exact CPU scheduled artifact schema|exact non-admitting CPU scheduled artifact"):
        trace.score(changed, checkpoint_raw, declaration, env.binding)


@pytest.mark.parametrize("damage", ["tick-count", "elapsed", "stop-reason", "backend", "plant"])
def test_partial_trace_rejects_malformed_collection_binding_or_backend(
        monkeypatch, synthetic_parent_bytes, damage):
    env, declaration, checkpoint_raw, value = collect_partial(monkeypatch, synthetic_parent_bytes)
    changed = deepcopy(value)
    if damage == "tick-count": changed["collection"]["policy_ticks"] = 2
    elif damage == "elapsed": changed["collection"]["elapsed_seconds"] = float("nan")
    elif damage == "stop-reason": changed["collection"]["stop_reason"] = "all-first-attempts-complete"
    elif damage == "backend": changed["backend"]["warp_is_cuda"] = True
    else: changed["compiled_plant"]["body_id"] += 1
    with pytest.raises(ValueError):
        trace.score(changed, checkpoint_raw, declaration, env.binding)


def test_verify_authenticates_whole_bytes_before_torch_load(monkeypatch, synthetic_parent_bytes):
    env, declaration, checkpoint_raw, value = collect_partial(monkeypatch, synthetic_parent_bytes)
    raw = trace.encode(value)
    monkeypatch.setattr(trace.torch, "load", lambda *_a, **_k: pytest.fail("hash must precede tensor load"))
    with pytest.raises(ValueError, match="whole-byte hash before tensor loading"):
        trace.verify(raw, "0"*64, checkpoint_raw, declaration, env.binding)


@pytest.mark.parametrize("damage", ["recorded-force", "post-leak", "schedule-hash", "plant-hash",
                                     "missing-phases", "malformed-phase", "extra-pulse-field"])
def test_independent_scheduled_pulse_checker_rejects_force_phase_and_binding_mutations(monkeypatch, damage):
    def mutate(value):
        row = value["scheduled_pulse_evidence"][0][0]
        if damage == "recorded-force": row["pre_xfrc"][0, value["compiled_plant"]["body_id"], 0] += 1.
        elif damage == "post-leak": row["post_qfrc"][0, 0] = 1.
        elif damage == "schedule-hash": row["schedule_sha256"] = "0"*64
        elif damage == "plant-hash": row["binding_sha256"] = "0"*64
        elif damage == "missing-phases": row["phases"] = None
        elif damage == "malformed-phase": row["phases"]["forced_pre"]["inputs"].pop("xfrc_applied")
        else: row["unexpected"] = True
    env, value = synthetic_pulse_value(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError):
        trace.check_pulses(value)


def test_synthetic_counter_heldout_four_newton_substep_checks_all_three_cpu_phases(monkeypatch):
    env, value = synthetic_pulse_value(monkeypatch)
    receipt = trace.check_pulses(value)
    assert receipt["checked_physics_steps"] == 1
    assert receipt["pulse_window_steps"] == 1 and receipt["delivered_nonzero_pulse_steps"] == 1
    assert receipt["complete_pulse_delivery"] is False
    assert receipt["complete_phase_checks"] is False
    assert receipt["recorded_phase_checks_valid"] is True
    assert receipt["exact_full_force_arrays_checked"] is True
    assert receipt["unforced_post_arrays_checked"] is True
    row = value["scheduled_pulse_evidence"][0][0]
    assert row["phases"] is not None
    force = row["phases"]["forced_pre"]["inputs"]["xfrc_applied"]
    trunk = env.binding["body_id"]
    assert force[0, trunk].tolist() == pytest.approx([1.4142135381698608, -1.4142135381698608, 0., 0., 0., 0.])


def _prefix_fixture():
    ticks, controls = [], []
    for tick_index in range(52):
        start = tick_index*10
        boundaries = [dict(physics_steps=torch.tensor([start+i], dtype=torch.int64),
            qpos=torch.tensor([[float(start+i)]], dtype=torch.float32)) for i in range(11)]
        proposals = [dict(before_steps=torch.tensor([start+i], dtype=torch.int64),
            torque=torch.tensor([[float(start+i)]], dtype=torch.float32)) for i in range(10)]
        ticks.append(dict(actor_input=torch.full((1, 44), float(tick_index)),
            actions=torch.full((1, 10), float(tick_index)), boundaries=boundaries))
        controls.append(dict(initial={"queue": torch.tensor([tick_index])},
            after_action={"correction": torch.full((1, 10), float(tick_index))}, proposals=proposals))
    return dict(payload=dict(initial={"state": torch.tensor([1.], dtype=torch.float32)}, ticks=ticks),
        control_evidence={"ticks": controls})


@pytest.mark.parametrize("step", [505, 509])
def test_mid_policy_tick_prefix_includes_onset_actor_boundaries_and_control_proposals(step):
    value = _prefix_fixture()
    original = trace.prefix_hash(value, step)
    changed = deepcopy(value)
    changed["payload"]["ticks"][50]["actor_input"][0, 0] = 2.
    assert trace.prefix_hash(changed, step) != original
    changed = deepcopy(value)
    changed["control_evidence"]["ticks"][50]["proposals"][step-500]["torque"][0, 0] = 2.
    assert trace.prefix_hash(changed, step) != original
    changed = deepcopy(value)
    changed["payload"]["ticks"][50]["boundaries"][step-500]["qpos"][0, 0] += 1.
    assert trace.prefix_hash(changed, step) != original
    changed = deepcopy(value)
    changed["control_evidence"]["ticks"][50]["after_action"]["correction"][0, 0] += 1.
    assert trace.prefix_hash(changed, step) != original


@pytest.mark.parametrize("step", [505, 509])
def test_mid_tick_prefix_excludes_later_within_tick_boundary_and_next_tick(step):
    value = _prefix_fixture()
    original = trace.prefix_hash(value, step)
    changed = deepcopy(value)
    changed["payload"]["ticks"][50]["boundaries"][step-500+1]["qpos"][0, 0] += 5.
    # The next boundary after the pre-force proposal is intentionally outside
    # the prefix, as are following post-force/after-window observations.
    assert trace.prefix_hash(changed, step) == original
    changed = deepcopy(value)
    changed["payload"]["ticks"][51]["actor_input"][0, 0] += 7.
    assert trace.prefix_hash(changed, step) == original


def test_prefix_hash_rejects_out_of_range_and_incomplete_midtick_boundary():
    value = _prefix_fixture()
    for step in (-1, 2500, True, 505.0):
        with pytest.raises(ValueError, match="bounded scheduled prefix step"):
            trace.prefix_hash(value, step)
    changed = _prefix_fixture(); changed["control_evidence"]["ticks"][50]["proposals"][5]["before_steps"][0] = 504
    with pytest.raises(ValueError, match="exact scheduled pre-force prefix clock"):
        trace.prefix_hash(changed, 505)
