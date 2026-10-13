"""Prospective walking/idle nonregression against the sealed cautious contrast."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat

from . import integrated_turn_foundation as base
from . import community_policy_inspection as inspection

BASELINE_SOURCE = "934bdff47a63d458f571189fe5db155fcd4fd8d0"
BASELINE_CHECKPOINT_SHA = "820c86e28b3d05d8bd292b63777b78cfa0b4657d69cc028ad17f8afe1ab91738"
BASELINE_SUMMARY_SHA = "5c8542012dd31a42e6482300290de20ea1a90b68735a720f94f7c7b2f20f0b32"
BASELINE_PATH = Path("/home/converge/.local/state/v8-cautious-lr-934bdff4-20261013-handoff/cautious-summary.json")
MOVING_MEAN_MAE_ALLOWANCE_MPS = .02


def load_baseline():
    info = BASELINE_PATH.lstat()
    base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and info.st_size <= 128*1024, "owned compact sealed contrast summary")
    payload = inspection._read_payload(BASELINE_PATH)
    base.require(hashlib.sha256(payload).hexdigest() == BASELINE_SUMMARY_SHA,
                 "exact sealed cautious summary bytes")
    value = json.loads(payload)
    base.require(value["source"] == BASELINE_SOURCE
                 and value["evaluations"]["final-eval"]["checkpoint_sha256"] == BASELINE_CHECKPOINT_SHA,
                 "sealed cautious source and final999 identity")
    return value


def compare(rows, baseline):
    expected = [(seed, speed, yaw) for seed in base.SEEDS for speed, yaw in base.CASES]
    base.require(type(rows) is list and len(rows) == len(expected), "complete retention matrix")
    for row, identity in zip(rows, expected, strict=True):
        base.require((row["seed"], row["speed"], row["yaw"]) == identity
                     and row["complete"] is True and row["worlds"] == 8 and row["steps"] == 240
                     and row["terminal_worlds"] == [], "ordered complete first episodes for retention")
        for field in ("speed_mae", "yaw_mae"):
            values = row[field]
            base.require(type(values) is list and len(values) == 8
                         and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values),
                         "finite nonnegative eight-world retention errors")
    base.require(baseline["source"] == BASELINE_SOURCE
                 and baseline["evaluations"]["final-eval"]["checkpoint_sha256"] == BASELINE_CHECKPOINT_SHA,
                 "retention contrast identity")
    groups = baseline["evaluations"]["final-eval"]["groups"]
    base.require([tuple(g["command"]) for g in groups] == list(base.CASES),
                 "ordered compact control groups")
    reports, failures = [], []
    moving_control_passes = moving_candidate_passes = 0
    for command, control in zip(base.CASES, groups, strict=True):
        speed,yaw = command
        selected = [row for row in rows if (row["speed"],row["yaw"]) == command]
        errors = [value for row in selected for value in row["speed_mae"]]
        mean = math.fsum(errors)/24
        limit = .03 if speed == 0 else .10
        speed_passes = sum(value <= limit for value in errors)
        yaw_passes = sum(value <= .10 for row in selected for value in row["yaw_mae"])
        base.require(type(control["speed_mae"]) in (int,float)
                     and math.isfinite(control["speed_mae"]) and control["speed_mae"] >= 0
                     and type(control["speed_passes"]) is int and 0 <= control["speed_passes"] <= 24,
                     "finite bounded control speed measurements")
        if speed != 0:
            moving_control_passes += control["speed_passes"]
            moving_candidate_passes += speed_passes
            if mean > control["speed_mae"] + MOVING_MEAN_MAE_ALLOWANCE_MPS:
                failures.append([speed,yaw,"moving-mean-speed-mae-regression"])
        elif speed_passes != 24 or yaw_passes != 24:
            failures.append([speed,yaw,"idle-speed-or-yaw-retention"])
        reports.append(dict(command=list(command),candidate_speed_mae_mean=mean,
            control_speed_mae_mean=control["speed_mae"],candidate_speed_passes=speed_passes,
            control_speed_passes=control["speed_passes"],candidate_yaw_passes=yaw_passes))
    if moving_candidate_passes < moving_control_passes:
        failures.append(["moving-speed-pass-count-regression"])
    return dict(protocol="v8-steady-steering-retention-v1",
        decision="walking-idle-retained" if not failures else "walking-idle-not-retained",
        baseline_source=BASELINE_SOURCE,baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
        baseline_summary_sha256=BASELINE_SUMMARY_SHA,
        moving_mean_speed_mae_allowance_mps=MOVING_MEAN_MAE_ALLOWANCE_MPS,
        moving_candidate_speed_passes=moving_candidate_passes,
        moving_control_speed_passes=moving_control_passes,groups=reports,failures=failures,
        policy_acceptance=False,physical_motion_authorized=False)
