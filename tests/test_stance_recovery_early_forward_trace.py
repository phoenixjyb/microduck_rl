"""Actual CPU early-forward instrumentation; no solver replacement or GPU claim."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import gc

import pytest
import torch

from mjlab_microduck import stance_recovery_early_forward_trace as early
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

SOURCE = "a" * 40
CELL_IDS = ["zero-wrench", "zero-wrench"]


def _tree_equal(left, right):
    """Compare runtime snapshot dataclasses without invoking tensor truthiness."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if is_dataclass(left) or is_dataclass(right):
        return (
            is_dataclass(left)
            and is_dataclass(right)
            and type(left) is type(right)
            and all(
                _tree_equal(getattr(left, field.name), getattr(right, field.name))
                for field in fields(left)
            )
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


@pytest.fixture(scope="module")
def actual_case():
    """Capture three real CPU ticks, plus the same uninstrumented CPU run."""
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    caller_rng = torch.random.get_rng_state().clone()

    def fresh_runtime():
        with torch.random.fork_rng(devices=[]):
            torch.random.manual_seed(977)
            return ScheduledRecoveryRuntime(
                declaration, device="cpu", solved_field_check="packed"
            )

    observed = fresh_runtime()
    reference = fresh_runtime()
    initial = observed.snapshot()
    assert _tree_equal(initial, reference.snapshot())

    observer = early.EarlyForwardTrace(observed)
    with observer:
        observed_result = observed.step_with_schedule(
            torch.zeros(2, 10), capture_control=True
        )
    retained = observer.capture()
    reference_result = reference.step_with_schedule(
        torch.zeros(2, 10), capture_control=True
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return {
        "declaration": declaration,
        "observer": observer,
        "observed": observed,
        "reference": reference,
        "retained": retained,
        "observed_result": observed_result,
        "reference_result": reference_result,
        "first_record": {"runtime_result_before_reset": observed_result},
    }


@pytest.fixture(autouse=True)
def cleanup_runtime_objects():
    yield
    gc.collect()


def test_actual_two_world_trace_keeps_uninstrumented_trajectory_and_controls(
    actual_case,
):
    observed = actual_case["observed"]
    reference = actual_case["reference"]
    observer = actual_case["observer"]
    assert _tree_equal(actual_case["observed_result"], actual_case["reference_result"])
    assert _tree_equal(observed._control_snapshot(), reference._control_snapshot())
    assert _tree_equal(observed.snapshot(), reference.snapshot())
    assert observer.capture()["status"] == "complete"
    assert len(observer.events) == 2 * early.STEPS == 6
    assert "_forward" not in observed.__dict__
    assert "_scheduled_forward" not in observed.__dict__
    assert observed._forward.__func__ is WarpStanceRuntime._forward
    assert (
        observed._scheduled_forward.__func__
        is ScheduledRecoveryRuntime._scheduled_forward
    )


def test_trace_records_exact_six_ordered_pre_and_post_forward_events(actual_case):
    events = actual_case["retained"]["events"]
    assert [(item["phase"], item["step"]) for item in events] == [
        ("scheduled-pre", 0),
        ("unforced-post", 0),
        ("scheduled-pre", 1),
        ("unforced-post", 1),
        ("scheduled-pre", 2),
        ("unforced-post", 2),
    ]
    assert len(events) == 6
    for event in events:
        assert set(event["inputs"]) == set(early.INPUTS)
        assert event["inputs"]["ctrl"].shape == (2, 14)
        assert event["inputs"]["xfrc_applied"].shape == (
            2,
            actual_case["observed"].native.nbody,
            6,
        )
        assert event["inputs"]["qfrc_applied"].shape == (2, 20)
        assert not event["inputs"]["xfrc_applied"].any()
        assert not event["inputs"]["qfrc_applied"].any()
        assert event["inputs"]["qM"].shape == (2, 20, 20)
        assert event["qM"].shape == (2, 20, 20)
        assert event["inputs"]["qM"].dtype == event["qM"].dtype == torch.float32


def test_cpu_checker_binds_actual_events_to_normalized_first_runtime_record(
    actual_case,
):
    result = early.check(
        actual_case["retained"],
        actual_case["declaration"],
        actual_case["first_record"],
    )
    assert result["protocol"] == early.PROTOCOL + ":score"
    assert result["source"] == SOURCE
    assert result["worlds"] == 2
    assert result["steps"] == 3 and result["events"] == 6
    assert all(result[key] is False for key in early.FLAGS)


def test_exact_trace_comparison_is_nonadmitting_and_has_no_earliest_difference(
    actual_case,
):
    trace = actual_case["retained"]
    first = actual_case["first_record"]
    result = early.compare(
        trace, deepcopy(trace), actual_case["declaration"], (first, deepcopy(first))
    )
    assert result["exact"] is True
    assert result["earliest_differing_event"] is None
    assert len(result["events"]) == 6
    assert all(result[key] is False for key in early.FLAGS)


def test_finite_mass_matrix_change_is_reported_without_qualification(actual_case):
    left = deepcopy(actual_case["retained"])
    right = deepcopy(actual_case["retained"])
    right["events"][0]["qM"][0, 0, 0] += 0.125
    records = (actual_case["first_record"], actual_case["first_record"])
    result = early.compare(left, right, actual_case["declaration"], records)
    event = result["events"][0]
    assert result["exact"] is False
    assert result["earliest_differing_event"] == 0
    assert event["mass_matrix_after_exact"] is False
    assert event["differing_leaves"] > 0
    assert all(result[key] is False for key in early.FLAGS)


def test_contact_row_reordering_is_only_an_unordered_diagnostic(actual_case):
    left = deepcopy(actual_case["retained"])
    right = deepcopy(actual_case["retained"])
    # The first early forwards can be contact-free; by the fourth event the
    # actual CPU trajectory has a populated, nontrivial contact table.
    event_index = next(
        index
        for index, event in enumerate(right["events"])
        if event["solved"]["contacts"]["worldid"].numel() > 1
    )
    contacts = right["events"][event_index]["solved"]["contacts"]
    rows = contacts["worldid"].numel()
    assert rows > 1
    permutation = torch.arange(rows - 1, -1, -1)
    for value in contacts.values():
        if torch.is_tensor(value):
            assert value.shape[0] == rows
            value.copy_(value.index_select(0, permutation))
    records = (actual_case["first_record"], actual_case["first_record"])
    result = early.compare(left, right, actual_case["declaration"], records)
    event = result["events"][event_index]
    assert result["exact"] is False
    assert result["earliest_differing_event"] == event_index
    assert event["exact"] is False
    assert event["contact_multiset_ignoring_addresses_exact"] is True
    assert all(result[key] is False for key in early.FLAGS)


@pytest.mark.parametrize("damage", ["committed-control", "control-order"])
def test_checker_rejects_committed_control_or_order_mutation(actual_case, damage):
    trace = deepcopy(actual_case["retained"])
    first = deepcopy(actual_case["first_record"])
    if damage == "committed-control":
        first["runtime_result_before_reset"]["control_evidence"]["proposals"][0][
            "committed"
        ]["ctrl"][0, 0] += 0.25
    else:
        trace["control_ids"][0], trace["control_ids"][1] = (
            trace["control_ids"][1],
            trace["control_ids"][0],
        )
        first["runtime_result_before_reset"]["control_evidence"]["proposals"][0][
            "committed"
        ]["ctrl"] = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    with pytest.raises(ValueError, match="committed actuator order"):
        early.check(trace, actual_case["declaration"], first)


def test_checker_refuses_late_foreign_hook_without_replacing_it(actual_case):
    env = ScheduledRecoveryRuntime(
        actual_case["declaration"], device="cpu", solved_field_check="packed"
    )
    observer = early.EarlyForwardTrace(env)

    def foreign_hook():
        return None

    env._forward = foreign_hook
    with pytest.raises(
        ValueError, match="forward hooks are still exclusively available"
    ):
        observer.__enter__()
    assert env.__dict__["_forward"] is foreign_hook
    del env._forward


def test_cuda_hidden_guard_precedes_trace_or_declaration_reads(monkeypatch):
    class DoNotRead(dict):
        def __getitem__(self, key):
            raise AssertionError("trace accessed before CUDA-hidden guard")

        def items(self):
            raise AssertionError("trace accessed before CUDA-hidden guard")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA state read before guard")),
    )
    with pytest.raises(ValueError, match="CUDA-hidden trace reader"):
        early.check(DoNotRead(), DoNotRead(), DoNotRead())


def test_trace_clock_cannot_shift_while_preserving_input_output_equality(actual_case):
    value = deepcopy(actual_case["retained"])
    value["events"][0]["inputs"]["time"].add_(1.0)
    value["events"][0]["solved"]["kinematics"]["time"].add_(1.0)
    with pytest.raises(ValueError, match="phase clock binds to declared physics step"):
        early.check(value, actual_case["declaration"], actual_case["first_record"])


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "phase-order",
        "step-order",
        "kinematics",
        "force",
        "qfrc",
        "qM-input",
        "qM-solved",
        "false-flag",
        "boundary",
    ],
)
def test_cpu_checker_rejects_malformed_trace_or_runtime_binding(actual_case, damage):
    trace = deepcopy(actual_case["retained"])
    first = deepcopy(actual_case["first_record"])
    if damage == "extra":
        trace["unexpected"] = True
    elif damage == "phase-order":
        trace["events"][0]["phase"] = "unforced-post"
    elif damage == "step-order":
        trace["events"][2]["step"] = 3
    elif damage == "kinematics":
        trace["events"][0]["inputs"]["qpos"][0, 0] += 0.1
    elif damage == "force":
        trace["events"][0]["inputs"]["xfrc_applied"][0, 0, 0] = 1.0
    elif damage == "qfrc":
        trace["events"][1]["inputs"]["qfrc_applied"][0, 0] = 1.0
    elif damage == "qM-input":
        trace["events"][0]["inputs"]["qM"] = torch.zeros(2, 20, 19)
    elif damage == "qM-solved":
        trace["events"][0]["qM"][0, 0, 0] = float("nan")
    elif damage == "false-flag":
        trace["native_trace_qualified"] = True
    else:
        first["runtime_result_before_reset"]["boundaries"][1]["qvel"][0, 0] += 0.1
    with pytest.raises(ValueError):
        early.check(trace, actual_case["declaration"], first)


def test_fault_restores_exact_inherited_hooks_and_marks_trace_faulted(
    actual_case, monkeypatch
):
    declaration = actual_case["declaration"]
    env = ScheduledRecoveryRuntime(
        declaration, device="cpu", solved_field_check="packed"
    )
    observer = early.EarlyForwardTrace(env)

    def fail_after_inherited_call(_env):
        raise RuntimeError("SYNTHETIC observational output failure")

    monkeypatch.setattr(early.forward, "output", fail_after_inherited_call)
    with pytest.raises(RuntimeError, match="observational output failure"):
        with observer:
            env.step_with_schedule(torch.zeros(2, 10), capture_control=True)
    captured = observer.capture()
    assert captured["status"] == "faulted"
    assert observer.faulted is True
    assert "_forward" not in env.__dict__
    assert "_scheduled_forward" not in env.__dict__
    assert env._forward.__func__ is WarpStanceRuntime._forward
    assert (
        env._scheduled_forward.__func__ is ScheduledRecoveryRuntime._scheduled_forward
    )


def test_trace_is_one_shot_and_refuses_preexisting_borrowed_method_replacement(
    actual_case,
):
    observer = actual_case["observer"]
    with pytest.raises(ValueError, match="one-shot forward observation lifetime"):
        observer.__enter__()

    env = ScheduledRecoveryRuntime(
        actual_case["declaration"], device="cpu", solved_field_check="packed"
    )
    env._forward = lambda: None
    with pytest.raises(ValueError):
        early.EarlyForwardTrace(env)
    del env._forward


def test_fault_capture_and_incomplete_trace_are_nonadmitting(actual_case):
    faulted = deepcopy(actual_case["retained"])
    faulted["status"] = "faulted"
    faulted["native_kernel_cause_proven"] = False
    with pytest.raises(ValueError, match="exact complete non-admitting early trace"):
        early.check(faulted, actual_case["declaration"], actual_case["first_record"])
    incomplete = deepcopy(actual_case["retained"])
    incomplete["events"] = incomplete["events"][:-1]
    with pytest.raises(ValueError):
        early.check(incomplete, actual_case["declaration"], actual_case["first_record"])
