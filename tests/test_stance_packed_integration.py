"""CPU-only integration contract for the opt-in packed stance predicate.

These checks cover the short CPU runtime path only; they are not CUDA,
trajectory-parity, training, or runtime-equivalence evidence.
"""

from copy import deepcopy

import pytest
import torch

import mjlab_microduck.stance_warp_runtime as runtime
from mjlab_microduck import stance_solved_field_check as checker
from mjlab_microduck import stance_solved_field_integration as integration


@pytest.fixture(scope="module")
def cpu_integration():
    return integration.cases("cpu")


@pytest.mark.parametrize("bad_mode", [None, "Packed", "packed ", "other", True])
def test_unknown_solved_field_mode_is_rejected_before_native_construction(monkeypatch, bad_mode):
    def forbidden_construction(*_args, **_kwargs):
        pytest.fail("invalid checker mode reached native construction")

    monkeypatch.setattr(runtime, "build_entity", forbidden_construction)
    with pytest.raises((ValueError, TypeError), match="solved_field_check|legacy|packed"):
        runtime.WarpStanceRuntime(1, device="cpu", solved_field_check=bad_mode)


def test_runtime_defaults_to_legacy_checker_on_cpu():
    env = runtime.WarpStanceRuntime(1, device="cpu")
    try:
        assert env.solved_field_check == "legacy"
        assert env.forward_graph is None
        assert not env.faulted
    finally:
        # No simulator teardown or external state; release owned references.
        del env


def test_runtime_checker_modes_keep_exact_field_inventory_and_fail_closed(monkeypatch):
    env = runtime.WarpStanceRuntime(1, device="cpu", solved_field_check="legacy")
    try:
        names = []
        view = env._view
        monkeypatch.setattr(env, "_view", lambda name: (names.append(name), view(name))[1])
        env._check_solved_fields()
        assert names == list(checker.FIELDS)

        with pytest.raises(AttributeError):
            env.solved_field_check = "packed"
        assert env.solved_field_check == "legacy"
        packed = runtime.WarpStanceRuntime(1, device="cpu", solved_field_check="packed")
        names.clear()
        packed_view = packed._view
        monkeypatch.setattr(packed, "_view", lambda name: (names.append(name), packed_view(name))[1])
        packed._check_solved_fields()
        assert names == list(checker.FIELDS) and packed.forward_graph is None
        with pytest.raises(AttributeError):
            packed.solved_field_check = "legacy"

        env._solved_field_check = "corrupted-mode"  # Deliberate private-state corruption fixture.
        with pytest.raises(ValueError, match="explicit legacy or packed"):
            env._check_solved_fields()
    finally:
        del env


def test_cpu_short_cases_complete_and_receipt_remains_non_admitting(cpu_integration):
    payload, receipt = cpu_integration
    assert payload["normal"]["steps"] == [20, 20]
    assert len(payload["normal"]["boundaries"]) == 21
    assert [b["physics_steps"] for b in payload["normal"]["boundaries"]] == [
        [i, i] for i in range(21)
    ]
    assert payload["isolation"]["before_reset_steps"] == [1, 20]
    assert payload["isolation"]["after_reset_steps"] == [0, 20]
    assert payload["isolation"]["first_terminal"]["physics_step"] == 1
    assert payload["isolation"]["first_terminal"]["terminated"] is True
    assert payload["isolation"]["frozen_fields_equal"] is True
    assert payload["isolation"]["reset_sibling_fields_equal"] is True
    assert [b["physics_steps"] for b in payload["isolation"]["first_boundaries"]] == [
        [0, 0], *[[1, i] for i in range(1, 11)]
    ]
    assert [b["physics_steps"] for b in payload["isolation"]["next_boundaries"]] == [
        [1, i] for i in range(10, 21)
    ]
    assert payload["backend"]["torch_device"] == "cpu"
    assert payload["backend"]["warp_is_cuda"] is False
    assert receipt["solved_field_check"] == "packed"
    assert receipt["runtimes"] == [
        {"worlds": 2, "checked_forwards": 41},
        {"worlds": 2, "checked_forwards": 42},
    ]
    assert receipt["same_input_predicates_agree"] is True
    assert receipt["input_bits_unchanged"] is True
    assert receipt["forward_graph"] is False
    assert receipt["timing_qualified"] is False
    assert receipt["learned_stance"] is False
    assert receipt["football_balance"] is False
    assert receipt["physical_motion_authorized"] is False
    assert receipt["separate_trajectory_equivalence_claimed"] is False
    assert receipt["policy_inferences"] == receipt["optimizer_updates"] == 0
    assert payload["optimizer_steps"] == 0
    assert payload["learned_stance"] is False
    assert payload["football_balance"] is False
    assert payload["complete_trajectory_evaluation"] is False
    assert payload["physical_motion_authorized"] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", [
    "device", "mode", "checker_hash", "field_order", "forward_count",
    "bool_counter", "fault_field", "fault_order", "fault_hash", "faults_agree",
    "input_mutation", "trajectory_claim", "graph", "timing", "inference_count",
    "optimizer_count", "learned", "football", "motion",
])
def test_integration_receipt_tampering_fails_closed(cpu_integration, damage):
    _payload, receipt = cpu_integration
    altered = deepcopy(receipt)
    if damage == "device": altered["device"] = "cuda:0"
    elif damage == "mode": altered["solved_field_check"] = "legacy"
    elif damage == "checker_hash": altered["checker_sha256"] = "0" * 64
    elif damage == "field_order": altered["fields"].reverse()
    elif damage == "forward_count": altered["runtimes"][0]["checked_forwards"] += 1
    elif damage == "bool_counter": altered["runtimes"][0]["checked_forwards"] = True
    elif damage == "fault_field": altered["faults"][0]["field"] = "qvel"
    elif damage == "fault_order": altered["faults"].reverse()
    elif damage == "fault_hash": altered["faults_sha256"] = "0" * 64
    elif damage == "faults_agree": altered["same_input_predicates_agree"] = False
    elif damage == "input_mutation": altered["input_bits_unchanged"] = False
    elif damage == "trajectory_claim": altered["separate_trajectory_equivalence_claimed"] = True
    elif damage == "graph": altered["forward_graph"] = True
    elif damage == "timing": altered["timing_qualified"] = True
    elif damage == "inference_count": altered["policy_inferences"] = 1
    elif damage == "optimizer_count": altered["optimizer_updates"] = True
    elif damage == "learned": altered["learned_stance"] = True
    elif damage == "football": altered["football_balance"] = True
    elif damage == "motion": altered["physical_motion_authorized"] = True
    with pytest.raises((ValueError, TypeError, KeyError)):
        integration.validate(altered, "cpu")
