"""Constraint identity diagnostics are comparison evidence, never qualification."""

import gc

import pytest
import torch

from mjlab_microduck import stance_constraint_identity as identity
from mjlab_microduck import stance_recovery_early_forward_trace as early
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime

SOURCE = "b" * 40
CELL_IDS = ["zero-wrench", "zero-wrench"]
FIELDS = ("J", "D", "aref", "force", "state")


def _rows(keys):
    """Build one world's active constraint table with unique `(type, id)` keys."""
    types = torch.tensor([key[0] for key in keys], dtype=torch.int32)
    ids = torch.tensor([key[1] for key in keys], dtype=torch.int32)
    count = len(keys)
    return {
        "type": types,
        "id": ids,
        "J": torch.arange(count * 20, dtype=torch.float32).reshape(count, 20),
        "D": torch.arange(count, dtype=torch.float32) + 0.5,
        "aref": torch.arange(count, dtype=torch.float32) + 1.5,
        "force": torch.arange(count, dtype=torch.float32) + 2.5,
        "state": torch.arange(count, dtype=torch.int32) % 3,
    }


def _clone_table(table):
    return {key: value.clone() for key, value in table.items()}


def _keys(result):
    return {(item["key"][0], item["key"][1]): item["count"] for item in result}


@pytest.fixture(scope="module")
def actual_cpu_pair():
    """Two identical actual CPU traces; no mocked runtime or device bypass."""
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)

    def capture():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            env = ScheduledRecoveryRuntime(
                declaration, device="cpu", solved_field_check="packed"
            )
        observer = early.EarlyForwardTrace(env)
        with observer:
            result = env.step_with_schedule(torch.zeros(2, 10), capture_control=True)
        return observer.capture(), {"runtime_result_before_reset": result}

    first = capture()
    second = capture()
    return declaration, first, second


@pytest.fixture(autouse=True)
def cleanup_runtime_objects():
    yield
    gc.collect()


def test_unique_key_rows_compare_exactly_without_mutating_inputs():
    left = _rows([(1, 2), (1, 5), (3, 7)])
    right = _clone_table(left)
    before = _clone_table(left), _clone_table(right)
    result = identity._rows(left, right)
    assert result["raw_order_exact"] is True
    assert result["key_sets_exact"] is True
    assert result["comparison_complete"] is True
    assert result["duplicate_identities"]["left"] == []
    assert result["duplicate_identities"]["right"] == []
    assert result["duplicate_identities"]["left_count"] == 0
    assert result["duplicate_identities"]["right_count"] == 0
    assert result["comparison_complete"] is True
    assert all(result[name] is False for name in identity.FLAGS)
    for name in FIELDS:
        assert result["fields"][name]["exact"] is True
        assert result["fields"][name]["compared_unique_rows"] == 3
        assert result["fields"][name]["differing_unique_rows"] == 0
    for actual, saved in zip((left, right), before, strict=True):
        for name in actual:
            assert torch.equal(actual[name], saved[name])


def test_permutation_is_exact_by_identity_but_not_raw_order():
    left = _rows([(1, 2), (1, 5), (3, 7)])
    right = {name: value[[2, 0, 1]].clone() for name, value in left.items()}
    result = identity._rows(left, right)
    assert result["raw_order_exact"] is False
    assert result["key_sets_exact"] is True
    assert result["comparison_complete"] is True
    assert all(result["fields"][name]["exact"] is True for name in FIELDS)
    assert len(result["raw_identity_prefix"]["left"]) == 3
    assert (
        result["raw_identity_order_sha256"]["left"]
        != result["raw_identity_order_sha256"]["right"]
    )


@pytest.mark.parametrize("field", ["force", "J"])
def test_float_field_differences_are_aligned_by_identity(field):
    left = _rows([(1, 2), (1, 5), (3, 7)])
    right = {name: value[[2, 0, 1]].clone() for name, value in left.items()}
    if field == "J":
        right[field][1, 0] += 0.25
    else:
        right[field][1] += 0.25
    result = identity._rows(left, right)
    summary = result["fields"][field]
    assert result["comparison_complete"] is True
    assert summary["exact"] is False
    assert summary["compared_unique_rows"] == 3
    assert summary["differing_unique_rows"] == 1
    assert summary["max_abs_delta"] == pytest.approx(0.25)
    assert summary["first_differing_keys"] == [[1, 2]]


def test_missing_identity_is_incomplete_and_does_not_pair_by_position():
    left = _rows([(1, 2), (1, 5), (3, 7)])
    right = _rows([(1, 2), (3, 7)])
    result = identity._rows(left, right)
    assert result["key_sets_exact"] is False
    assert result["comparison_complete"] is False
    for name in FIELDS:
        assert result["fields"][name]["exact"] is False
        assert result["fields"][name]["compared_unique_rows"] == 2


def test_duplicate_keys_are_explicitly_ambiguous_and_unique_rows_still_compared():
    left = _rows([(1, 2), (1, 2), (3, 7)])
    right = _rows([(1, 2), (1, 2), (3, 7)])
    right["force"][2] += 0.75
    result = identity._rows(left, right)
    assert result["key_sets_exact"] is True
    assert result["comparison_complete"] is False
    assert result["duplicate_identities"]["left"] == [{"key": [1, 2], "count": 2}]
    assert result["duplicate_identities"]["right"] == [{"key": [1, 2], "count": 2}]
    assert result["duplicate_identities"]["left_count"] == 1
    assert result["duplicate_identities"]["right_count"] == 1
    assert result["duplicate_identities"]["left_truncated"] is False
    assert result["duplicate_identities"]["right_truncated"] is False
    assert result["fields"]["force"]["exact"] is None
    assert result["fields"]["force"]["compared_unique_rows"] == 1
    assert result["fields"]["force"]["differing_unique_rows"] == 1


def test_mixed_duplicate_and_unique_differences_preserve_ambiguity():
    left = _rows([(1, 2), (1, 2), (3, 7), (4, 9)])
    right = _rows([(1, 2), (1, 2), (3, 7), (4, 10)])
    result = identity._rows(left, right)
    assert result["key_sets_exact"] is False
    assert result["comparison_complete"] is False
    assert result["fields"]["J"]["exact"] is None
    assert result["fields"]["J"]["compared_unique_rows"] == 1


def test_public_compare_uses_actual_checked_traces_and_stays_nonadmitting(
    actual_cpu_pair,
):
    declaration, (left, left_record), (right, right_record) = actual_cpu_pair
    result = identity.compare(left, right, declaration, (left_record, right_record))
    assert result["protocol"] == identity.PROTOCOL
    assert result["source"] == SOURCE
    assert len(result["events"]) == early.STEPS * 2
    assert all(result[flag] is False for flag in identity.FLAGS)
    assert result["worlds"] in (2, 64)
    assert result["earliest_raw_differing_event"] is None
    for event in result["events"]:
        assert len(event["worlds"]) == result["worlds"] == 2
        assert all(
            world["friction_rows"]["identity_rows"]["left"] == 14
            for world in event["worlds"]
        )
        assert all(
            world["friction_rows"]["comparison_complete"] for world in event["worlds"]
        )


def test_cuda_hidden_compare_guard_precedes_trace_validation(monkeypatch):
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
    with pytest.raises(ValueError, match="CUDA-hidden"):
        identity.compare(DoNotRead(), DoNotRead(), DoNotRead(), (DoNotRead(),) * 2)
