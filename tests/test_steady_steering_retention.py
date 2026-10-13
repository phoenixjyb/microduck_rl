from copy import deepcopy

import pytest

from mjlab_microduck import steady_steering_retention as retention
from mjlab_microduck import integrated_turn_foundation as base


def fixture():
    groups = [dict(command=list(command),speed_mae=.05 if command[0] else .002,
                   speed_passes=24,yaw_passes=24) for command in base.CASES]
    baseline = dict(source=retention.BASELINE_SOURCE,evaluations={"final-eval":dict(
        checkpoint_sha256=retention.BASELINE_CHECKPOINT_SHA,groups=groups)})
    rows = [dict(seed=seed,speed=speed,yaw=yaw,complete=True,worlds=8,steps=240,
        terminal_worlds=[],speed_mae=[.05 if speed else .002]*8,yaw_mae=[.01]*8)
        for seed in base.SEEDS for speed,yaw in base.CASES]
    return rows,baseline


def test_improved_or_unchanged_speed_retains_without_promoting_policy():
    rows,baseline = fixture()
    report = retention.compare(rows,baseline)
    assert report["decision"] == "walking-idle-retained"
    assert report["moving_candidate_speed_passes"] == 72
    assert report["baseline_summary_sha256"] == retention.BASELINE_SUMMARY_SHA
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False


@pytest.mark.parametrize("fault,reason", [("slower","moving-mean-speed-mae-regression"),
    ("lost-speed-world","moving-speed-pass-count-regression"),
    ("idle-speed","idle-speed-or-yaw-retention"),("idle-yaw","idle-speed-or-yaw-retention")])
def test_steering_gains_cannot_hide_speed_or_idle_loss(fault,reason):
    rows,baseline = fixture()
    if fault == "slower":
        for row in rows:
            if row["speed"] == .3: row["speed_mae"] = [.09]*8
    elif fault == "lost-speed-world": rows[1]["speed_mae"][0] = .101
    elif fault == "idle-speed": rows[0]["speed_mae"][0] = .031
    else: rows[0]["yaw_mae"][0] = .101
    report = retention.compare(rows,baseline)
    assert report["decision"] == "walking-idle-not-retained"
    assert any(reason in failure for failure in report["failures"])


@pytest.mark.parametrize("fault", ["missing","order","terminal","nan","bool","negative","control-sha"])
def test_incomplete_or_spoofed_evidence_refuses(fault):
    rows,baseline = fixture()
    if fault == "missing": rows.pop()
    elif fault == "order": rows.reverse()
    elif fault == "terminal": rows[0]["complete"] = False
    elif fault == "control-sha": baseline["evaluations"]["final-eval"]["checkpoint_sha256"] = "0"*64
    else: rows[1]["speed_mae"][0] = {"nan":float("nan"),"bool":True,"negative":-.1}[fault]
    with pytest.raises(ValueError): retention.compare(rows,baseline)


def test_retention_reads_no_trace_inputs_and_does_not_mutate():
    rows,baseline = fixture()
    before = deepcopy((rows,baseline))
    retention.compare(rows,baseline)
    assert (rows,baseline) == before
