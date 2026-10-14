from __future__ import annotations

from itertools import chain, islice
import math

import pytest

from mjlab_microduck import v8_head_goal_exposure_audit as audit
from mjlab_microduck import v8_head_goal_schedule as schedule


def _window(arm: str, update: int, *, outside: bool = True) -> dict:
    start = (update - 1) * 24
    records = []
    for offset in range(24):
        step = start + offset
        stage = schedule.stage_index(arm, step)
        ranges = schedule.ranges_at(arm, step)
        previous = None if stage == 0 else schedule.ranges_at(
            arm, schedule.FULL_SCHEDULES[arm][stage][0] - 1)
        lows, highs, cross, means = [], [], [], []
        for component, (low, high) in enumerate(ranges):
            sample_low, sample_high = low / 2, high / 2
            crossed = 0
            if outside and previous is not None:
                prior_low, prior_high = previous[component]
                if high > prior_high:
                    sample_high = prior_high + (high - prior_high) / 2
                    crossed = 1
                elif low < prior_low:
                    sample_low = prior_low + (low - prior_low) / 2
                    crossed = 1
            elif previous is not None:
                prior_low, prior_high = previous[component]
                sample_low = max(sample_low, prior_low / 2)
                sample_high = min(sample_high, prior_high / 2)
            lows.append(sample_low)
            highs.append(sample_high)
            cross.append(crossed)
            means.append(max(abs(sample_low), abs(sample_high)) / 2)
        records.append({
            "before_step": step, "completed_step": step + 1,
            "schedule_kind": "full-training", "arm": arm, "stage": stage,
            "ranges": [list(pair) for pair in ranges],
            "previous_ranges": None if previous is None else [list(pair) for pair in previous],
            "worlds": audit.WORLDS,
            "command_min": lows, "command_max": highs,
            "command_abs_mean": means,
            "command_nonzero_count": [audit.WORLDS] * 4,
            "command_outside_previous_range_count": cross,
        })
    counts = {str(stage): sum(r["stage"] == stage for r in records)
              for stage in range(len(schedule.FULL_SCHEDULES[arm]))}
    by_stage_ranges = {}
    for record in records:
        by_stage_ranges[str(record["stage"])] = record["ranges"]
    summary = {
        "arm": arm, "probe": False, "schedule_kind": "full-training",
        "steps": 24, "worlds": audit.WORLDS,
        "first_before_step": start, "last_completed_step": start + 24,
        "stage_action_steps": counts, "ranges_by_stage": by_stage_ranges,
        "command_min": [min(r["command_min"][i] for r in records) for i in range(4)],
        "command_max": [max(r["command_max"][i] for r in records) for i in range(4)],
        "command_abs_mean": [math.fsum(r["command_abs_mean"][i] for r in records) / 24
                             for i in range(4)],
        "command_nonzero_count": [24 * audit.WORLDS] * 4,
        "command_outside_previous_range_count": [
            sum(r["command_outside_previous_range_count"][i] for r in records)
            for i in range(4)],
        "command_sample_count": 24 * audit.WORLDS,
        "step_records": records,
    }
    return summary


def _rows(arm: str, *, mutate=None):
    for update in range(1, audit.UPDATES + 1):
        row = {
            "update": update,
            "arm": arm,
            "local_common_step_counter": 24 * update,
            "optimizer_adam_steps": [60_000 + 20 * update],
            "head_goal_telemetry": _window(arm, update),
            "head_goal_ranges_at_update": [list(pair) for pair in schedule.ranges_at(arm, 24 * update)],
            "diagnostic": None,
        }
        if mutate is not None:
            mutate(row, update)
        yield row


@pytest.mark.parametrize(("arm", "steps"), [
    ("control", {"0": 24_000}),
    ("staged", {"0": 12_000, "1": 6_000, "2": 6_000}),
])
def test_full_foundation_audit_reports_actual_exposure_only(arm, steps):
    report = audit.audit_rows(_rows(arm), arm)
    assert report["protocol"] == "v8-head-goal-exposure-audit-v1"
    assert report["worlds"] == 256 and report["updates"] == 1000
    assert report["steps"] == 24_000 and report["command_sample_count"] == 6_144_000
    assert report["stage_action_steps"] == steps
    assert report["policy_acceptance"] is False
    assert report["physical_motion_authorized"] is False and report["qualified"] is False
    assert all(stats["command_sample_count"] == stats["action_steps"] * 256
               for stats in report["by_stage"].values())
    if arm == "staged":
        assert report["by_stage"]["2"]["action_steps"] == 6000
        assert all(value > 0 for value in
                   report["by_stage"]["1"]["command_outside_previous_range_count"])
        assert all(value > 0 for value in
                   report["by_stage"]["2"]["command_outside_previous_range_count"])


@pytest.mark.parametrize("fault", ["counter", "duplicate", "arm", "adam", "adam_float",
                                    "adam_bool", "range", "kind", "nan"])
def test_audit_rejects_corrupt_rows_and_windows(fault):
    def mutate(row, update):
        if update != 2:
            return
        if fault == "counter":
            row["local_common_step_counter"] += 1
        elif fault == "duplicate":
            row["update"] = 1
        elif fault == "arm":
            row["arm"] = "control"
        elif fault == "adam":
            row["optimizer_adam_steps"] = [60_000 + 20 * update - 1]
        elif fault == "adam_float":
            row["optimizer_adam_steps"] = [float(60_000 + 20 * update)]
        elif fault == "adam_bool":
            row["optimizer_adam_steps"] = [True]
        elif fault == "range":
            row["head_goal_ranges_at_update"][0][1] += .1
        elif fault == "kind":
            row["head_goal_telemetry"]["probe"] = True
        elif fault == "nan":
            row["head_goal_telemetry"]["step_records"][0]["command_abs_mean"][0] = float("nan")

    with pytest.raises(ValueError):
        audit.audit_rows(_rows("staged", mutate=mutate), "staged")


def test_audit_rejects_missing_and_extra_rows():
    with pytest.raises(ValueError, match="stopped at 999"):
        audit.audit_rows(islice(_rows("control"), 999), "control")
    with pytest.raises(ValueError, match="extra row"):
        audit.audit_rows(chain(_rows("control"), ({},)), "control")


def test_staged_audit_requires_real_per_component_widening_exposure():
    def no_exposure(row, update):
        if 501 <= update <= 750 or update >= 751:
            row["head_goal_telemetry"] = _window("staged", update, outside=False)

    with pytest.raises(ValueError, match="widening"):
        audit.audit_rows(_rows("staged", mutate=no_exposure), "staged")


def test_audit_rejects_a_stage_with_no_actual_nonzero_commands(monkeypatch):
    monkeypatch.setattr(audit, "UPDATES", 2)
    monkeypatch.setattr(audit, "TOTAL_STEPS", 48)
    monkeypatch.setattr(audit, "_STAGES", {"control": (48,), "staged": (24, 12, 12)})

    def no_nonzero(row, _update):
        telemetry = row["head_goal_telemetry"]
        for record in telemetry["step_records"]:
            record["command_min"] = [0.0] * 4
            record["command_max"] = [0.0] * 4
            record["command_abs_mean"] = [0.0] * 4
            record["command_nonzero_count"] = [0] * 4
        telemetry["command_min"] = [0.0] * 4
        telemetry["command_max"] = [0.0] * 4
        telemetry["command_abs_mean"] = [0.0] * 4
        telemetry["command_nonzero_count"] = [0] * 4

    with pytest.raises(ValueError, match="nonzero samples"):
        audit.audit_rows(_rows("control", mutate=no_nonzero), "control")


def test_jsonl_reader_bounds_lines_and_rejects_symlinks(tmp_path):
    from hashlib import sha256

    huge = tmp_path / "huge.jsonl"
    huge.write_bytes(b" " * (audit.MAX_LINE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        list(audit._file_rows(huge, {"digest": sha256(), "bytes": 0}))
    link = tmp_path / "link.jsonl"
    link.symlink_to(huge)
    with pytest.raises(ValueError, match="symlink"):
        audit._audit_file(link, "control")

