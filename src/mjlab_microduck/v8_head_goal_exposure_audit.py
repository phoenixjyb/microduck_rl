"""CPU-only audit of retained v8 full-foundation head-goal exposure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from . import v8_head_goal_schedule as schedule


PROTOCOL = "v8-head-goal-exposure-audit-v1"
WORLDS = 256
UPDATES = 1000
STEPS_PER_UPDATE = 24
TOTAL_STEPS = UPDATES * STEPS_PER_UPDATE
PARENT_ADAM_STEP = 60_000
ADAM_STEPS_PER_UPDATE = 20
MAX_LINE_BYTES = 1 << 20
_STAGES = {"control": (24_000,), "staged": (12_000, 6_000, 6_000)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    return value


def _finite_tree(value: Any, label: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")


def audit_rows(rows: Iterable[dict], arm: str) -> dict:
    """Validate exactly 1000 update rows and summarize actual stage exposure."""
    _require(type(arm) is str and arm in schedule.ARMS, "known head-goal audit arm")
    stages = len(schedule.FULL_SCHEDULES[arm])
    accum = [dict(action_steps=0, command_sample_count=0,
                  command_min=[None] * 4, command_max=[None] * 4,
                  command_abs_sum=[0.0] * 4, command_nonzero_count=[0] * 4,
                  command_outside_previous_range_count=[0] * 4)
             for _ in range(stages)]

    iterator = iter(rows)
    for update in range(1, UPDATES + 1):
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"expected exactly {UPDATES} updates; stopped at {update - 1}") from exc
        _require(isinstance(row, dict), f"update row {update} must be an object")
        _finite_tree(row, f"update row {update}")
        _require(_integer(row.get("update"), f"update row {update} update") == update,
                 f"updates must be contiguous 1..{UPDATES}")
        _require(row.get("arm") == arm, f"update {update} arm must match requested {arm}")
        telemetry = row.get("head_goal_telemetry")
        _require(isinstance(telemetry, Mapping), f"update {update} head_goal_telemetry is required")
        schedule.validate_window(telemetry, arm, False, WORLDS,
                                 (update - 1) * STEPS_PER_UPDATE, STEPS_PER_UPDATE)
        step = update * STEPS_PER_UPDATE
        _require(_integer(row.get("local_common_step_counter"),
                          f"update {update} local_common_step_counter") == step,
                 f"update {update} local counter must equal 24 * update")
        adam_steps = row.get("optimizer_adam_steps")
        _require(type(adam_steps) is list and len(adam_steps) == 1
                 and type(adam_steps[0]) is int
                 and adam_steps[0] == PARENT_ADAM_STEP + ADAM_STEPS_PER_UPDATE * update,
                 f"update {update} Adam step must continue from 60000 by 20")
        expected_ranges = [list(pair) for pair in schedule.ranges_at(arm, step)]
        _require(row.get("head_goal_ranges_at_update") == expected_ranges,
                 f"update {update} report ranges must match the live schedule")

        for entry in telemetry["step_records"]:
            stat = accum[entry["stage"]]
            stat["action_steps"] += 1
            stat["command_sample_count"] += WORLDS
            for component in range(4):
                low, high = entry["command_min"][component], entry["command_max"][component]
                old_low, old_high = stat["command_min"][component], stat["command_max"][component]
                stat["command_min"][component] = low if old_low is None else min(old_low, low)
                stat["command_max"][component] = high if old_high is None else max(old_high, high)
                stat["command_abs_sum"][component] += entry["command_abs_mean"][component] * WORLDS
                stat["command_nonzero_count"][component] += entry["command_nonzero_count"][component]
                stat["command_outside_previous_range_count"][component] += \
                    entry["command_outside_previous_range_count"][component]

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"expected exactly {UPDATES} updates; found an extra row")

    expected_steps = _STAGES[arm]
    by_stage = {}
    for index, (stat, expected) in enumerate(zip(accum, expected_steps, strict=True)):
        _require(stat["action_steps"] == expected,
                 f"stage {index} actual action exposure must equal {expected}")
        _require(all(count > 0 for count in stat["command_nonzero_count"]),
                 f"stage {index} needs actual nonzero samples in every head component")
        by_stage[str(index)] = {
            "action_steps": stat["action_steps"],
            "command_sample_count": stat["command_sample_count"],
            "command_min": stat["command_min"],
            "command_max": stat["command_max"],
            "command_abs_mean": [value / stat["command_sample_count"]
                                 for value in stat["command_abs_sum"]],
            "command_nonzero_count": stat["command_nonzero_count"],
            "command_outside_previous_range_count":
                stat["command_outside_previous_range_count"],
        }
    if arm == "staged":
        for stage in (1, 2):
            counts = by_stage[str(stage)]["command_outside_previous_range_count"]
            _require(all(count > 0 for count in counts),
                     f"staged widening {stage} needs actual exposure in every head component")

    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "worlds": WORLDS,
        "updates": UPDATES,
        "steps": TOTAL_STEPS,
        "command_sample_count": TOTAL_STEPS * WORLDS,
        "stage_action_steps": {str(index): count for index, count in enumerate(expected_steps)},
        "by_stage": by_stage,
        "exposure_only": True,
        "policy_acceptance": False,
        "qualified": False,
        "physical_motion_authorized": False,
    }


def _json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is forbidden")


def _file_rows(path: Path, metadata: dict):
    with path.open("rb") as stream:
        for index in range(1, UPDATES + 2):
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            _require(len(line) <= MAX_LINE_BYTES,
                     f"updates.jsonl line {index} exceeds {MAX_LINE_BYTES} bytes")
            metadata["digest"].update(line)
            metadata["bytes"] += len(line)
            try:
                yield json.loads(line, parse_constant=_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON on updates.jsonl line {index}") from exc
        if stream.read(1):
            # Consume only enough to establish that input exceeds the 1000-row bound.
            raise ValueError(f"updates.jsonl exceeds {UPDATES} rows or has trailing data")


def _audit_file(path: Path, arm: str) -> dict:
    _require(not path.is_symlink(), "updates input must not be a symlink")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"cannot inspect updates input: {exc}") from exc
    _require(stat.S_ISREG(mode), "updates input must be a regular file")
    metadata = {"digest": hashlib.sha256(), "bytes": 0}
    report = audit_rows(_file_rows(path, metadata), arm)
    report["updates_file_sha256"] = metadata["digest"].hexdigest()
    report["updates_file_bytes"] = metadata["bytes"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", required=True, type=Path, help="retained updates.jsonl")
    parser.add_argument("--arm", required=True, choices=schedule.ARMS)
    args = parser.parse_args(argv)
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
             "set CUDA_VISIBLE_DEVICES to the empty string for this CPU audit")
    import torch
    _require(not torch.cuda.is_initialized(), "CUDA must not be initialized for this audit")
    report = _audit_file(args.updates, args.arm)
    json.dump(report, sys.stdout, sort_keys=True, separators=(",", ":"), allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
