"""CPU-only structural checks for frozen five-case recovery comparison."""

from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_matrix as matrix


SOURCE = "a" * 40
DECLARATION = {"source": SOURCE, "plant": {"selected_plant": "synthetic-bound-plant"}}
RAW = b"synthetic authenticated recovery capture"
CHECKPOINT_RAW = b"checkpoint bytes behind scoring seam"


def numerical_score(case, *, candidate_pass=True):
    return dict(protocol=contract.PROTOCOL, case=case,
        numerical_diagnostic=dict(candidate_pass=candidate_pass, gates=dict(
            complete_first_attempt=True, full_duration=True, no_hard_failure=True,
            displacement=True, soft_limit=True, final_tilt=True, final_speed=True,
            final_height=True, foot_support=True)),
        pulse=dict(checked_physics_steps=contract.TOTAL_STEPS,
            pulse_window_steps=contract.PULSE_STEPS, complete_pulse_delivery=True,
            complete_pulse_window_phase_checks=True, exact_scheduled_forces_checked=True,
            unforced_post_arrays_checked=True),
        actor_replay_max_abs_error=0., whole_trajectory_physics_resimulated=False,
        **contract.FALSE_FLAGS)


def make_case_value(case):
    ticks = []
    controls = []
    for tick_index in range(52):
        step = tick_index * 10
        ticks.append(dict(actor_input=torch.full((44,), tick_index, dtype=torch.float32),
            actions=torch.full((10,), tick_index / 100, dtype=torch.float32),
            boundaries=[dict(physics_steps=torch.tensor([step], dtype=torch.long),
                             root=torch.tensor([float(tick_index), 0., .2], dtype=torch.float32))]))
        proposal = dict(before_steps=torch.tensor([step], dtype=torch.long),
            accepted=torch.tensor([True]), torque=torch.full((10,), tick_index, dtype=torch.float32))
        controls.append(dict(initial=dict(queue=torch.tensor([tick_index], dtype=torch.long)),
            after_action=dict(correction=torch.tensor([tick_index / 10], dtype=torch.float32)),
            proposals=[proposal]))
    return dict(payload=dict(initial=dict(state=torch.tensor([0., 0., .2], dtype=torch.float32)),
                             ticks=ticks),
        control_evidence=dict(ticks=controls), case=case)


def checked_fixture(monkeypatch, *, case="zero-wrench", value=None, score=None):
    events = []
    value = make_case_value(case) if value is None else value
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(matrix.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(matrix.torch, "load", lambda stream, **kwargs: (
        events.append(("load", kwargs, stream.read())), value)[1])
    monkeypatch.setattr(matrix.evidence, "score", lambda payload, checkpoint, declaration: (
        events.append(("score", checkpoint, declaration)), score or numerical_score(case))[1])
    return events, value


def receipt(case, *, prefix="c" * 64, candidate_pass=True):
    return dict(protocol=matrix.PROTOCOL, source=SOURCE, case=case,
        declaration_sha256="d" * 64, checkpoint_sha256=contract.CHECKPOINT_SHA256,
        evaluation_seed=contract.EVALUATION_SEED, capture_sha256="e" * 64, capture_bytes=42,
        prefix_sha256=prefix, matched_through_physics_step=contract.ONSET_STEP,
        first_pre_force_action_included=True, post_push_actions_compared=False,
        score=numerical_score(case, candidate_pass=candidate_pass),
        whole_trajectory_physics_resimulated=False, **contract.FALSE_FLAGS)


def test_prefix_hash_is_typed_shape_sensitive_and_bit_exact():
    value = {"x": torch.tensor([[1., -0.]], dtype=torch.float32), "n": (True, 1, 1.0)}
    assert matrix.prefix_hash(value) == matrix.prefix_hash(deepcopy(value))
    assert matrix.prefix_hash(value) != matrix.prefix_hash({**value, "x": value["x"].reshape(2)})
    assert matrix.prefix_hash(value) != matrix.prefix_hash({**value,
        "x": value["x"].to(torch.long)})
    assert matrix.prefix_hash({"v": True}) != matrix.prefix_hash({"v": 1})


@pytest.mark.parametrize("value", [torch.tensor([float("nan")]), torch.tensor([float("inf")]), float("nan")])
def test_prefix_hash_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError, match="finite"):
        matrix.prefix_hash({"bad": value})


def test_checked_case_requires_cpu_hidden_before_hash_or_loading(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(matrix.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(matrix, "sha256", lambda: pytest.fail("CPU gate must precede hash"))
    monkeypatch.setattr(matrix.torch, "load", lambda *_args, **_kwargs: pytest.fail("CPU gate must precede load"))
    with pytest.raises(ValueError, match="CPU-only independent five-case checker"):
        matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)


def test_checked_case_authenticates_bytes_before_cpu_tensor_load(monkeypatch):
    events, _value = checked_fixture(monkeypatch)
    with pytest.raises(ValueError, match="whole case byte hash before tensor loading"):
        matrix.checked_case(RAW, "0" * 64, CHECKPOINT_RAW, DECLARATION)
    assert events == []


def test_checked_case_scores_before_computing_matched_prefix_hash(monkeypatch):
    events, _value = checked_fixture(monkeypatch)
    original = matrix.prefix_hash
    monkeypatch.setattr(matrix, "prefix_hash", lambda prefix: (
        events.append(("prefix", None)), original(prefix))[1])
    checked = matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)
    assert events[0][0] == "load"
    assert events[0][1] == {"map_location": "cpu", "weights_only": True}
    assert events[0][2] == RAW
    assert events[1] == ("score", CHECKPOINT_RAW, DECLARATION)
    assert events[2][0] == "prefix"
    assert checked["matched_through_physics_step"] == 500
    assert checked["first_pre_force_action_included"] is True
    assert checked["post_push_actions_compared"] is False


@pytest.mark.parametrize("which", ["boundary", "proposal"])
def test_checked_case_rejects_any_onset_other_than_pre_force_step_500(monkeypatch, which):
    events, value = checked_fixture(monkeypatch)
    if which == "boundary":
        value["payload"]["ticks"][matrix.PREFIX_TICKS]["boundaries"][0]["physics_steps"][0] = 499
    else:
        value["control_evidence"]["ticks"][matrix.PREFIX_TICKS]["proposals"][0]["before_steps"][0] = 501
    monkeypatch.setattr(matrix, "prefix_hash", lambda *_: pytest.fail("invalid onset reached prefix digest"))
    with pytest.raises(ValueError, match="matched exact pre-force onset"):
        matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)
    assert [event[0] for event in events] == ["load", "score"]

def test_prefix_includes_initial_unforced_actions_controls_and_pre_force_onset(monkeypatch):
    events, original = checked_fixture(monkeypatch)
    original_receipt = matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)

    def changed(path):
        value = deepcopy(original)
        if path == "initial":
            value["payload"]["initial"]["state"][0] += 1
        elif path == "unforced-action":
            value["payload"]["ticks"][49]["actions"][0] += 1
        elif path == "unforced-control":
            value["control_evidence"]["ticks"][49]["initial"]["queue"][0] += 1
        elif path == "onset-input":
            value["payload"]["ticks"][50]["actor_input"][0] += 1
        elif path == "onset-action":
            value["payload"]["ticks"][50]["actions"][0] += 1
        elif path == "onset-state":
            value["payload"]["ticks"][50]["boundaries"][0]["root"][0] += 1
        elif path == "onset-after-action-control":
            value["control_evidence"]["ticks"][50]["after_action"]["correction"][0] += 1
        elif path == "first-proposal":
            value["control_evidence"]["ticks"][50]["proposals"][0]["torque"][0] += 1
        return value

    for path in ("initial", "unforced-action", "unforced-control", "onset-input", "onset-action",
                 "onset-state", "onset-after-action-control", "first-proposal"):
        _events, value = checked_fixture(monkeypatch, value=changed(path))
        changed_receipt = matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)
        assert changed_receipt["prefix_sha256"] != original_receipt["prefix_sha256"], path


def test_reactive_actions_after_first_pulse_step_do_not_change_matched_prefix(monkeypatch):
    _events, original = checked_fixture(monkeypatch)
    before = matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)
    reactive = deepcopy(original)
    reactive["payload"]["ticks"][51]["actor_input"].add_(3.)
    reactive["payload"]["ticks"][51]["actions"].add_(4.)
    reactive["control_evidence"]["ticks"][51]["proposals"][0]["torque"].add_(5.)
    _events, _value = checked_fixture(monkeypatch, value=reactive)
    after = matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)
    assert before["prefix_sha256"] == after["prefix_sha256"]


def test_compare_accepts_five_ordered_case_passes_without_admission():
    receipts = [receipt(case) for case in contract.CASE_NAMES]
    result = matrix.compare(receipts)
    assert result["decision"] == "frozen-five-case-baseline-passed"
    assert result["cases"] == list(contract.CASE_NAMES)
    assert result["zero_wrench_control_passed"] is True
    assert result["all_prefixes_identical"] is True
    assert result["held_out_randomized_recovery_verified"] is False
    assert result["whole_trajectory_physics_resimulated"] is False
    assert all(result[flag] is False for flag in contract.FALSE_FLAGS)


@pytest.mark.parametrize("bad_cases", [
    list(reversed(contract.CASE_NAMES)),
    list(contract.CASE_NAMES[:-1]) + [contract.CASE_NAMES[0]],
    list(contract.CASE_NAMES[:-1]),
])
def test_compare_rejects_wrong_order_duplicates_and_missing_case(bad_cases):
    with pytest.raises(ValueError, match="all five ordered"):
        matrix.compare([receipt(case) for case in bad_cases])


@pytest.mark.parametrize("field", [
    "protocol", "source", "declaration_sha256", "checkpoint_sha256", "evaluation_seed",
    "prefix_sha256", "matched_through_physics_step", "first_pre_force_action_included",
    "post_push_actions_compared", "whole_trajectory_physics_resimulated",
])
def test_compare_rejects_mismatch_in_shared_binding_or_prefix(field):
    receipts = [receipt(case) for case in contract.CASE_NAMES]
    receipts[-1][field] = "mismatch" if isinstance(receipts[-1][field], str) else not receipts[-1][field]
    with pytest.raises(ValueError, match="identical binding and unforced state/action prefixes"):
        matrix.compare(receipts)


@pytest.mark.parametrize("where,flag", [
    (where, flag) for where in ("receipt", "score") for flag in contract.FALSE_FLAGS
])
def test_compare_rejects_every_admission_flag(where, flag):
    receipts = [receipt(case) for case in contract.CASE_NAMES]
    receipts[2][flag] = True if where == "receipt" else receipts[2][flag]
    if where == "score":
        receipts[2]["score"][flag] = True
    with pytest.raises(ValueError):
        matrix.compare(receipts)


@pytest.mark.parametrize("damage", ["partial", "candidate-type", "pulse-steps", "pulse-window",
    "pulse-phases", "force-check", "post-force-clear", "resimulation", "actor-error", "score-flag"])
def test_compare_rejects_partial_or_unverified_whole_case_score(damage):
    receipts = [receipt(case) for case in contract.CASE_NAMES]
    score = receipts[3]["score"]
    if damage == "partial":
        score["numerical_diagnostic"]["gates"]["full_duration"] = False
    elif damage == "candidate-type":
        score["numerical_diagnostic"]["candidate_pass"] = 1
    elif damage == "pulse-steps":
        score["pulse"]["checked_physics_steps"] -= 1
    elif damage == "pulse-window":
        score["pulse"]["pulse_window_steps"] -= 1
    elif damage == "pulse-phases":
        score["pulse"]["complete_pulse_window_phase_checks"] = False
    elif damage == "force-check":
        score["pulse"]["exact_scheduled_forces_checked"] = False
    elif damage == "post-force-clear":
        score["pulse"]["unforced_post_arrays_checked"] = False
    elif damage == "resimulation":
        score["whole_trajectory_physics_resimulated"] = True
    elif damage == "actor-error":
        score["actor_replay_max_abs_error"] = .001
    else:
        score["training_admitted"] = True
    with pytest.raises(ValueError, match="whole-case verified prefix receipt"):
        matrix.compare(receipts)


def test_compare_rejects_non_delivered_pulse():
    receipts = [receipt(case) for case in contract.CASE_NAMES]
    receipts[1]["score"]["pulse"]["complete_pulse_delivery"] = False
    with pytest.raises(ValueError, match="whole-case verified prefix receipt"):
        matrix.compare(receipts)


def test_one_numeric_case_failure_rejects_baseline_without_learning_admission():
    receipts = [receipt(case) for case in contract.CASE_NAMES]
    receipts[2]["score"]["numerical_diagnostic"]["candidate_pass"] = False
    result = matrix.compare(receipts)
    assert result["decision"] == "frozen-five-case-baseline-rejected"
    assert result["zero_wrench_control_passed"] is True
    assert all(result[flag] is False for flag in contract.FALSE_FLAGS)


def test_checked_case_refuses_non_full_duration_before_prefix_digest(monkeypatch):
    events, _value = checked_fixture(monkeypatch, score=numerical_score("zero-wrench"))
    monkeypatch.setattr(matrix.evidence, "score", lambda *_args: (
        events.append(("score", None)),
        {**numerical_score("zero-wrench"), "numerical_diagnostic": {
            "gates": {"full_duration": False}}})[1])
    monkeypatch.setattr(matrix, "prefix_hash", lambda *_: pytest.fail("partial case reached prefix hash"))
    with pytest.raises(ValueError, match="full-length case"):
        matrix.checked_case(RAW, sha256(RAW).hexdigest(), CHECKPOINT_RAW, DECLARATION)
    assert [event[0] for event in events] == ["load", "score"]
