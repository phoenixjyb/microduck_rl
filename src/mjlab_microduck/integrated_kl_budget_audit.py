"""Audit retained adaptive-KL update telemetry without loading the simulator.

Usage: python integrated_kl_budget_audit.py updates.jsonl \
    --expected-updates 1000 --desired-kl 0.01

The report audits optimizer telemetry only. It does not establish policy or
simulator acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


_MIN_LR = 1e-5
_MAX_LR = 1e-2
_MINIBATCHES_PER_UPDATE = 20
_SUMMARY_REL_TOL = 1e-6
_SUMMARY_ABS_TOL = 1e-8
_LR_REL_TOL = 1e-12
_LR_ABS_TOL = 1e-15


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be a non-bool number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{label} must be a non-bool integer")
    return value


def _numeric_array(row: dict[str, Any], key: str) -> list[float]:
    raw = row.get(key)
    _require(isinstance(raw, list) and len(raw) == _MINIBATCHES_PER_UPDATE,
             f"{key} must contain exactly {_MINIBATCHES_PER_UPDATE} values")
    return [_number(value, f"{key}[{index}]") for index, value in enumerate(raw)]


def _close(actual: float, expected: float, *, rel: float, abs_: float) -> bool:
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_)


def _adaptive_lr(kl_mean: float, current_lr: float, desired_kl: float) -> float:
    """Mirror rsl_rl 5.0.1 PPO's per-minibatch adaptive-LR branch."""
    if kl_mean > desired_kl * 2.0:
        return max(_MIN_LR, current_lr / 1.5)
    if 0.0 < kl_mean < desired_kl / 2.0:
        return min(_MAX_LR, current_lr * 1.5)
    return current_lr


def _audit_row(row: Any, index: int, desired_kl: float) -> dict[str, Any]:
    _require(isinstance(row, dict), f"row {index} must be an object")
    update = _integer(row.get("update"), f"row {index} update")
    _require(update == index, f"updates must be ordered 1..N (row {index} has {update})")
    _require(_integer(row.get("optimizer_minibatches"), f"row {index} optimizer_minibatches")
             == _MINIBATCHES_PER_UPDATE,
             f"row {index} must report {_MINIBATCHES_PER_UPDATE} optimizer minibatches")
    row_target = _number(row.get("desired_kl"), f"row {index} desired_kl")
    _require(row_target == desired_kl, f"row {index} desired_kl does not match requested target")

    means = _numeric_array(row, "minibatch_kl_means")
    before = _numeric_array(row, "minibatch_lr_before")
    inferred = _numeric_array(row, "minibatch_lr_applied_inferred")
    final_lr = _number(row.get("lr_after_update"), f"row {index} lr_after_update")
    rates = before + [final_lr]
    _require(all(_MIN_LR <= rate <= _MAX_LR for rate in rates),
             f"row {index} learning rates must be within [{_MIN_LR}, {_MAX_LR}]")

    applied = before[1:] + [final_lr]
    for minibatch, (observed, recorded) in enumerate(zip(applied, inferred, strict=True)):
        _require(_close(recorded, observed, rel=_LR_REL_TOL, abs_=_LR_ABS_TOL),
                 f"row {index} inferred applied LR {minibatch} disagrees with pre-LR/final-LR")
    for minibatch, (kl_mean, current_lr, next_lr) in enumerate(
        zip(means, before, applied, strict=True)
    ):
        expected_lr = _adaptive_lr(kl_mean, current_lr, desired_kl)
        _require(_close(next_lr, expected_lr, rel=_LR_REL_TOL, abs_=_LR_ABS_TOL),
                 f"row {index} adaptive LR transition {minibatch} is inconsistent with KL")

    mean_min = _number(row.get("kl_mean_min"), f"row {index} kl_mean_min")
    mean_max = _number(row.get("kl_mean_max"), f"row {index} kl_mean_max")
    mean_avg = _number(row.get("kl_mean_mean"), f"row {index} kl_mean_mean")
    # Float32 reduction may place a constant-array mean one ULP outside its
    # exact extrema. This logging tolerance never changes KL branch thresholds.
    _require(mean_min <= mean_max
             and (mean_min <= mean_avg or _close(mean_avg, mean_min,
                  rel=_SUMMARY_REL_TOL, abs_=_SUMMARY_ABS_TOL))
             and (mean_avg <= mean_max or _close(mean_avg, mean_max,
                  rel=_SUMMARY_REL_TOL, abs_=_SUMMARY_ABS_TOL)),
             f"row {index} KL mean summary extrema are unordered")
    _require(_close(mean_min, min(means), rel=_SUMMARY_REL_TOL, abs_=_SUMMARY_ABS_TOL)
             and _close(mean_max, max(means), rel=_SUMMARY_REL_TOL, abs_=_SUMMARY_ABS_TOL)
             and _close(mean_avg, math.fsum(means) / len(means),
                        rel=_SUMMARY_REL_TOL, abs_=_SUMMARY_ABS_TOL),
             f"row {index} KL mean summary disagrees with minibatch means")

    sample_min = _number(row.get("kl_sample_min"), f"row {index} kl_sample_min")
    sample_max = _number(row.get("kl_sample_max"), f"row {index} kl_sample_max")
    _require(sample_min <= sample_max and sample_min <= min(means) <= max(means) <= sample_max,
             f"row {index} sample KL extrema must encompass minibatch means")

    high_count = _integer(row.get("kl_high_count"), f"row {index} kl_high_count")
    low_count = _integer(row.get("kl_low_positive_count"),
                         f"row {index} kl_low_positive_count")
    _require(high_count == sum(value > desired_kl * 2.0 for value in means),
             f"row {index} high-KL count disagrees with minibatch means")
    _require(low_count == sum(0.0 < value < desired_kl / 2.0 for value in means),
             f"row {index} low-positive-KL count disagrees with minibatch means")

    before_min = _number(row.get("lr_before_min"), f"row {index} lr_before_min")
    before_max = _number(row.get("lr_before_max"), f"row {index} lr_before_max")
    _require(before_min == min(before) and before_max == max(before),
             f"row {index} pre-LR extrema disagree with minibatch rates")

    return {
        "update": update,
        "kl_means": means,
        "lr_before": before,
        "lr_applied": applied,
        "lr_after_update": final_lr,
        "high_count": high_count,
        "low_positive_count": low_count,
    }


def audit_rows(rows: Any, *, expected_updates: int, desired_kl: float,
               initial_lr: float = _MIN_LR) -> dict[str, Any]:
    """Validate complete per-update telemetry and return a deterministic summary."""
    expected_updates = _integer(expected_updates, "expected_updates")
    _require(expected_updates > 0, "expected_updates must be positive")
    desired_kl = _number(desired_kl, "desired_kl")
    _require(desired_kl in (0.01, 0.02), "desired_kl must be 0.01 or 0.02")
    initial_lr = _number(initial_lr, "initial_lr")
    _require(_MIN_LR <= initial_lr <= _MAX_LR, "initial_lr must be within PPO LR bounds")
    _require(isinstance(rows, list) and len(rows) == expected_updates,
             f"expected exactly {expected_updates} update rows")
    _require(isinstance(rows[0], dict), "row 1 must be an object")
    first_rates = _numeric_array(rows[0], "minibatch_lr_before")
    _require(first_rates[0] == initial_lr,
             "first update must start at the predeclared restored parent LR")

    audited = [_audit_row(row, index, desired_kl) for index, row in enumerate(rows, 1)]
    for previous, current in zip(audited, audited[1:]):
        _require(_close(current["lr_before"][0], previous["lr_after_update"],
                        rel=_LR_REL_TOL, abs_=_LR_ABS_TOL),
                 f"update {current['update']} first pre-LR must continue prior update LR")

    kl_means = [value for row in audited for value in row["kl_means"]]
    lr_before = [value for row in audited for value in row["lr_before"]]
    lr_applied = [value for row in audited for value in row["lr_applied"]]
    end_lrs = [row["lr_after_update"] for row in audited]
    floor_updates = [row["update"] for row in audited if row["lr_after_update"] == _MIN_LR]
    applied_floor_count = sum(rate == _MIN_LR for rate in lr_applied)

    return {
        "status": "telemetry-audit-only",
        "policy_acceptance": False,
        "expected_updates": expected_updates,
        "desired_kl": desired_kl,
        "initial_lr": initial_lr,
        "optimizer_minibatches": len(kl_means),
        "kl_mean_min": min(kl_means),
        "kl_mean_max": max(kl_means),
        "kl_mean_mean": math.fsum(kl_means) / len(kl_means),
        "kl_high_count": sum(row["high_count"] for row in audited),
        "kl_low_positive_count": sum(row["low_positive_count"] for row in audited),
        "lr_before_min": min(lr_before),
        "lr_before_max": max(lr_before),
        "lr_applied_min": min(lr_applied),
        "lr_applied_max": max(lr_applied),
        "minibatch_lr_applied_floor_count": applied_floor_count,
        "minibatch_lr_applied_floor_fraction": applied_floor_count / len(lr_applied),
        "end_update_lr_floor_count": len(floor_updates),
        "end_update_lr_floor_updates": floor_updates,
        "end_update_lr_min": min(end_lrs),
        "end_update_lr_max": max(end_lrs),
    }


def _read_jsonl(raw: bytes) -> list[dict[str, Any]]:
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("input must be UTF-8 JSONL") from exc
    rows = []
    for line_number, line in enumerate(content.splitlines(), 1):
        _require(bool(line.strip()), f"blank JSONL line at {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="retained updates.jsonl")
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--desired-kl", type=float, required=True)
    parser.add_argument("--initial-lr", type=float, default=_MIN_LR,
                        help="exact predeclared parent Adam LR; never resets the optimizer")
    args = parser.parse_args(argv)
    try:
        raw = args.input_path.read_bytes()
        result = audit_rows(_read_jsonl(raw), expected_updates=args.expected_updates,
                            desired_kl=args.desired_kl, initial_lr=args.initial_lr)
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"telemetry audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
