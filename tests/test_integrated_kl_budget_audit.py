import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mjlab_microduck.integrated_kl_budget_audit import audit_rows


MIN_LR = 1e-5
MAX_LR = 1e-2


def adaptive_lr(value, current, target):
    if value > target * 2:
        return max(MIN_LR, current / 1.5)
    if 0 < value < target / 2:
        return min(MAX_LR, current * 1.5)
    return current


def make_row(update, target, means, *, start_lr=MIN_LR, sample_delta=.01):
    rates = [start_lr]
    for value in means[:-1]:
        rates.append(adaptive_lr(value, rates[-1], target))
    final_lr = adaptive_lr(means[-1], rates[-1], target)
    applied = rates[1:] + [final_lr]
    return {
        "update": update,
        "optimizer_minibatches": 20,
        "desired_kl": target,
        "kl_mean_min": min(means),
        "kl_mean_max": max(means),
        "kl_mean_mean": sum(means) / len(means),
        "kl_sample_min": min(means) - sample_delta,
        "kl_sample_max": max(means) + sample_delta,
        "kl_high_count": sum(value > target * 2 for value in means),
        "kl_low_positive_count": sum(0 < value < target / 2 for value in means),
        "lr_before_min": min(rates),
        "lr_before_max": max(rates),
        "lr_after_update": final_lr,
        "minibatch_kl_means": means,
        "minibatch_lr_before": rates,
        "minibatch_lr_applied_inferred": applied,
    }


def valid_rows(count=2, target=.01):
    patterns = [
        [-.001, 0., target / 2, target * 2, target * 2.1] * 4,
        [target * .25, target * 3, 0., -.002, target / 2] * 4,
    ]
    rows = []
    start_lr = MIN_LR
    for i in range(count):
        row = make_row(i + 1, target, patterns[i % len(patterns)], start_lr=start_lr)
        rows.append(row)
        start_lr = row["lr_after_update"]
    return rows


def test_audits_mixed_branches_and_separates_minibatches_from_end_update_floor():
    rows = valid_rows(2)
    summary = audit_rows(rows, expected_updates=2, desired_kl=.01)
    assert summary["status"] == "telemetry-audit-only"
    assert summary["policy_acceptance"] is False
    assert summary["optimizer_minibatches"] == 40
    assert summary["kl_high_count"] == sum(row["kl_high_count"] for row in rows)
    assert summary["kl_low_positive_count"] == sum(row["kl_low_positive_count"] for row in rows)
    assert summary["minibatch_lr_applied_floor_count"] > summary["end_update_lr_floor_count"]
    assert summary["end_update_lr_floor_count"] == 2
    assert summary["end_update_lr_floor_updates"] == [1, 2]


@pytest.mark.parametrize("target", [.01, .02])
def test_threshold_boundaries_zero_and_negative_kl_do_not_adjust(target):
    means = [-.1, 0., target / 2, target * 2, target / 2 - 1e-9,
             target * 2 + 1e-9] * 3 + [-.02, 0.]
    rows = [make_row(1, target, means)]
    result = audit_rows(rows, expected_updates=1, desired_kl=target)
    assert result["kl_high_count"] == 3
    assert result["kl_low_positive_count"] == 3


def test_aggregates_all_minibatches_and_end_update_floor_counts():
    patterns = [
        [-.001, 0., .005, .04, .042] * 4,
        [.005, .06, 0., -.002, .01] * 3 + [.005, .06, 0., -.002, .05],
        [.005, .06, 0., -.002, .01] * 3 + [.005, .06, 0., -.002, .05],
    ]
    rows = []
    start_lr = MIN_LR
    for index, means in enumerate(patterns, 1):
        row = make_row(index, .02, means, start_lr=start_lr)
        rows.append(row)
        start_lr = row["lr_after_update"]
    summary = audit_rows(rows, expected_updates=3, desired_kl=.02)
    assert summary["optimizer_minibatches"] == 60
    assert summary["end_update_lr_floor_count"] == 3
    assert summary["end_update_lr_floor_updates"] == [1, 2, 3]
    assert summary["minibatch_lr_applied_floor_count"] > summary["end_update_lr_floor_count"]


def test_accepts_float32_constant_mean_reduction_rounding():
    # Native torch.float32: full((20,), .03).mean() is slightly below min/max.
    means = [0.029999999329447746] * 20
    row = make_row(1, .01, means)
    row["kl_mean_mean"] = 0.029999997466802597
    result = audit_rows([row], expected_updates=1, desired_kl=.01)
    assert result["kl_high_count"] == 20


def test_float32_summary_tolerance_does_not_relax_controller_thresholds():
    row = make_row(1, .01, [.02] * 20)
    row["kl_high_count"] = 20
    with pytest.raises(ValueError, match="high-KL count"):
        audit_rows([row], expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("bad", [True, False, "20", None])
def test_rejects_noninteger_update_count_argument(bad):
    with pytest.raises(ValueError):
        audit_rows([], expected_updates=bad, desired_kl=.01)


def test_rejects_float_equal_integer_counts():
    with pytest.raises(ValueError):
        audit_rows([], expected_updates=1.0, desired_kl=.01)
    rows = valid_rows(1)
    rows[0]["update"] = 1.0
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)
    rows = valid_rows(1)
    rows[0]["optimizer_minibatches"] = 20.0
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)
    rows = valid_rows(1)
    rows[0]["kl_high_count"] = float(rows[0]["kl_high_count"])
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("target", [0., .03, True, float("nan"), float("inf")])
def test_rejects_invalid_requested_target(target):
    with pytest.raises(ValueError):
        audit_rows([], expected_updates=1, desired_kl=target)


@pytest.mark.parametrize("mutation", ["missing", "short", "long", "bool", "nan", "inf"])
@pytest.mark.parametrize("key", ["minibatch_kl_means", "minibatch_lr_before",
                                  "minibatch_lr_applied_inferred"])
def test_rejects_malformed_or_nonfinite_arrays(mutation, key):
    rows = valid_rows(1)
    row = rows[0]
    if mutation == "missing":
        row.pop(key)
    elif mutation == "short":
        row[key] = row[key][:-1]
    elif mutation == "long":
        row[key] = row[key] + [row[key][-1]]
    elif mutation == "bool":
        row[key][2] = True
    elif mutation == "nan":
        row[key][2] = float("nan")
    else:
        row[key][2] = float("inf")
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("mutation", ["missing", "bool", "nonfinite", "wrong"])
@pytest.mark.parametrize("key", ["kl_mean_min", "kl_mean_max", "kl_mean_mean",
                                  "kl_sample_min", "kl_sample_max", "lr_before_min",
                                  "lr_before_max", "lr_after_update"])
def test_rejects_malformed_numeric_summaries(mutation, key):
    rows = valid_rows(1)
    if mutation == "missing":
        rows[0].pop(key)
    elif mutation == "bool":
        rows[0][key] = True
    elif mutation == "nonfinite":
        rows[0][key] = float("nan")
    else:
        rows[0][key] = -123.0 if key == "kl_sample_max" else 123.0
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("key,value", [
    ("update", True), ("update", 2), ("optimizer_minibatches", True),
    ("optimizer_minibatches", 19), ("desired_kl", .02),
    ("kl_high_count", True), ("kl_high_count", 99),
    ("kl_low_positive_count", False), ("kl_low_positive_count", -1),
])
def test_rejects_bad_order_counts_and_declared_target(key, value):
    rows = valid_rows(1)
    rows[0][key] = value
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("key", ["update", "optimizer_minibatches", "kl_high_count",
                                 "kl_low_positive_count"])
def test_rejects_missing_counts(key):
    rows = valid_rows(1)
    rows[0].pop(key)
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("key", ["optimizer_minibatches", "kl_high_count", "kl_low_positive_count"])
def test_rejects_nonfinite_counts(key, value):
    rows = valid_rows(1)
    rows[0][key] = value
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("mutation", ["summary", "sample_extrema", "pre_lr_extrema"])
def test_rejects_inconsistent_summaries(mutation):
    rows = valid_rows(1)
    if mutation == "summary":
        rows[0]["kl_mean_mean"] += .001
    elif mutation == "sample_extrema":
        rows[0]["kl_sample_min"] = max(rows[0]["minibatch_kl_means"]) + 1.
    else:
        rows[0]["lr_before_min"] *= 2.
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


def test_rejects_learning_rate_outside_bounds_and_inconsistent_transition():
    rows = valid_rows(1)
    rows[0]["minibatch_lr_before"][1] = 1e-6
    rows[0]["lr_before_min"] = 1e-6
    with pytest.raises(ValueError, match="learning rates"):
        audit_rows(rows, expected_updates=1, desired_kl=.01)

    rows = valid_rows(1)
    rates = rows[0]["minibatch_lr_before"]
    rates[1] = rates[0] * 2.
    rows[0]["lr_before_min"], rows[0]["lr_before_max"] = min(rates), max(rates)
    rows[0]["minibatch_lr_applied_inferred"] = rates[1:] + [rows[0]["lr_after_update"]]
    with pytest.raises(ValueError, match="transition"):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_missing_or_extra_rows(count):
    rows = valid_rows(count)
    with pytest.raises(ValueError):
        audit_rows(rows, expected_updates=1, desired_kl=.01)


def test_rejects_out_of_order_rows():
    rows = valid_rows(2)
    rows.reverse()
    with pytest.raises(ValueError, match="ordered 1..N"):
        audit_rows(rows, expected_updates=2, desired_kl=.01)


@pytest.mark.parametrize("mutation", ["first", "continuity"])
def test_rejects_initial_or_cross_update_lr_discontinuity(mutation):
    rows = valid_rows(2)
    if mutation == "first":
        rows[0]["minibatch_lr_before"][0] = 1.5e-5
        rows[0]["lr_before_min"] = min(rows[0]["minibatch_lr_before"])
        with pytest.raises(ValueError, match="restored parent LR"):
            audit_rows(rows, expected_updates=2, desired_kl=.01)
    else:
        rows[1] = make_row(2, .01, rows[1]["minibatch_kl_means"], start_lr=1.5e-5)
        with pytest.raises(ValueError, match="continue prior update"):
            audit_rows(rows, expected_updates=2, desired_kl=.01)


def test_cli_is_deterministic_and_hashes_raw_input(tmp_path):
    rows = valid_rows(2)
    raw = "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows).encode()
    path = tmp_path / "updates.jsonl"
    path.write_bytes(raw)
    script = Path(__file__).parents[1] / "src/mjlab_microduck/integrated_kl_budget_audit.py"
    command = [sys.executable, str(script), str(path), "--expected-updates", "2", "--desired-kl", ".01"]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    result = json.loads(first.stdout)
    assert result["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["status"] == "telemetry-audit-only" and result["policy_acceptance"] is False


def test_explicit_nonfloor_parent_lr_preserves_controller_and_default_guard():
    inherited = 5.062500000000001e-05
    row = make_row(1, .01, [0.] * 20, start_lr=inherited)
    with pytest.raises(ValueError, match="restored parent LR"):
        audit_rows([row], expected_updates=1, desired_kl=.01)
    result = audit_rows([row], expected_updates=1, desired_kl=.01, initial_lr=inherited)
    assert result["initial_lr"] == result["lr_before_min"] == result["lr_applied_max"] == inherited
    assert result["minibatch_lr_applied_floor_count"] == 0


@pytest.mark.parametrize("initial", [True, None, 1e-6, .1, float("nan"), float("inf")])
def test_invalid_explicit_initial_lr_is_rejected(initial):
    with pytest.raises(ValueError):
        audit_rows(valid_rows(1), expected_updates=1, desired_kl=.01, initial_lr=initial)


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_requires_explicit_nonfloor_parent_identity(tmp_path, explicit):
    inherited = 5.062500000000001e-05
    raw = (json.dumps(make_row(1, .01, [.01]*20, start_lr=inherited)) + "\n").encode()
    path = tmp_path / "updates.jsonl"
    path.write_bytes(raw)
    script = Path(__file__).parents[1] / "src/mjlab_microduck/integrated_kl_budget_audit.py"
    command = [sys.executable, str(script), str(path), "--expected-updates", "1", "--desired-kl", ".01"]
    if explicit: command.extend(["--initial-lr", repr(inherited)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == (0 if explicit else 2)
    if explicit:
        report = json.loads(result.stdout)
        assert report["initial_lr"] == inherited
        assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    else: assert "restored parent LR" in result.stderr

