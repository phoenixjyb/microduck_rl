from copy import deepcopy
import hashlib
import json

import pytest

from mjlab_microduck import projected_heading_retention as retention
from mjlab_microduck import steady_steering_retention as original
from mjlab_microduck import integrated_turn_foundation as base


def fixture(monkeypatch):
    groups = [dict(command=list(command), speed_mae=.05 if command[0] else .002,
                   speed_passes=24, yaw_passes=24) for command in base.CASES]
    new = dict(source=retention.BASELINE_SOURCE, evaluations={"final-eval": dict(
        checkpoint_sha256=retention.BASELINE_CHECKPOINT_SHA, groups=groups)})
    old = deepcopy(new)
    old["source"] = original.BASELINE_SOURCE
    old["evaluations"]["final-eval"]["checkpoint_sha256"] = original.BASELINE_CHECKPOINT_SHA
    monkeypatch.setattr(original, "load_baseline", lambda: old)
    rows = [dict(seed=seed, speed=speed, yaw=yaw, complete=True, worlds=8, steps=240,
        terminal_worlds=[], speed_mae=[.05 if speed else .002]*8, yaw_mae=[.01]*8)
        for seed in base.SEEDS for speed,yaw in base.CASES]
    return rows,new,old


def test_both_controls_bound_and_no_input_mutation(monkeypatch):
    rows,new,old = fixture(monkeypatch)
    before = deepcopy((rows,new,old))
    report = retention.compare(rows,new)
    assert report["decision"] == "walking-idle-retained"
    assert report["baseline_source"] == retention.BASELINE_SOURCE
    assert report["original_cautious_retention"]["baseline_source"] == original.BASELINE_SOURCE
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False
    assert (rows,new,old) == before


@pytest.mark.parametrize("control", ["new","old"])
def test_either_control_can_reject_without_relaxing_bars(monkeypatch,control):
    rows,new,old = fixture(monkeypatch)
    target = new if control == "new" else old
    target["evaluations"]["final-eval"]["groups"][1]["speed_mae"] = .01
    report = retention.compare(rows,new)
    assert report["decision"] == "walking-idle-not-retained"
    assert any("moving-mean-speed-mae-regression" in failure for failure in report["failures"])
    if control == "old": assert report["failures"][0][0] == "original-cautious"


def test_spoofed_control_refuses(monkeypatch):
    rows,new,_ = fixture(monkeypatch)
    new["source"] = "0"*40
    with pytest.raises(ValueError): retention.compare(rows,new)


def test_loader_authenticates_bytes_and_both_controls(monkeypatch,tmp_path):
    _,new,_ = fixture(monkeypatch)
    path = tmp_path/"summary.json"
    payload = json.dumps(new).encode()
    path.write_bytes(payload)
    monkeypatch.setattr(retention,"BASELINE_PATH",path)
    monkeypatch.setattr(retention,"BASELINE_SUMMARY_SHA",hashlib.sha256(payload).hexdigest())
    assert retention.load_baseline() == new
    path.write_bytes(payload+b" ")
    with pytest.raises(ValueError,match="summary bytes"): retention.load_baseline()


def test_loader_refuses_symlink(monkeypatch,tmp_path):
    path = tmp_path/"summary.json"
    path.symlink_to(tmp_path/"missing")
    monkeypatch.setattr(retention,"BASELINE_PATH",path)
    with pytest.raises(ValueError,match="owned compact"): retention.load_baseline()
