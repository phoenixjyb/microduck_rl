import datetime as dt
import json
from types import SimpleNamespace as NS

import pytest

from mjlab_microduck import foundation_motor_timing as timing


def test_historical_equivalence_ignores_only_source_and_added_motor_trace():
    old = dict(source="old", groups={"load": .1}, route_trace=[1, 2])
    new = dict(old, source="new", motor_trace={"raw": True})
    timing.compare_historical(new, old)
    new["route_trace"] = [1, 2.00000001]
    with pytest.raises(ValueError): timing.compare_historical(new, old)


def test_changed_original_manifest_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "ORIGINAL", tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable F1-M identity"): timing.verify_original()


@pytest.mark.parametrize("failure", [None, "safety", "replay", "child"])
def test_sequential_stop_retains_evidence_without_training(tmp_path, monkeypatch, failure):
    output = tmp_path / "new"
    monkeypatch.setattr(timing, "OUTPUT", output)
    monkeypatch.setattr(timing, "verify_source", lambda s: None)
    monkeypatch.setattr(timing.exp, "LAST_START", dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    originals = {a: dict(safety_failures=[], payload=1) for a in timing.ARMS}
    monkeypatch.setattr(timing, "verify_original", lambda: originals)
    monkeypatch.setattr(timing.exp.training, "runtime_identity", lambda: {})
    monkeypatch.setattr(timing, "wait_idle", lambda: {})
    def forbidden(*a, **k): raise AssertionError("no optimizer work")
    monkeypatch.setattr(timing.exp.training, "train", forbidden)
    monkeypatch.setattr(timing, "audit_motor_trace", lambda r: dict(decision="timing-measurement-only"))
    calls=[]
    def child(arm, source, end):
        calls.append(arm)
        if failure=="child": raise ValueError("child failed")
        report = dict(originals[arm], motor_trace={})
        if failure=="safety": report["safety_failures"]=["absolute-torque"]
        if failure=="replay": report["payload"]=2
        timing.write_new(output / f"{arm}.json", report)
    monkeypatch.setattr(timing, "run_child", child)
    if failure in ("replay", "child"):
        with pytest.raises(ValueError): timing.campaign("a"*40)
    else: timing.campaign("a"*40)
    decision=json.loads((output / "decision.json").read_text())
    assert calls==(["parent"] if failure else list(timing.ARMS))
    assert decision["optimizer_updates"]==0 and not decision["curriculum_promotion"]
    assert decision["decision"]==("runtime-failure-stop" if failure in ("replay","child") else
                                    "safety-stop" if failure else "timing-measurement-only")
    manifest=json.loads((output / "manifest.json").read_text())
    for name,row in manifest["files"].items(): assert timing.sha256(output/name)==row["sha256"]
    with pytest.raises(FileExistsError): timing.campaign("a"*40)


def test_timing_child_bounded_and_idle_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "OUTPUT", tmp_path)
    monkeypatch.setattr(timing.exp, "DEADLINE", dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    calls=[]
    monkeypatch.setattr(timing, "wait_idle", lambda: calls.append("idle") or {})
    monkeypatch.setattr(timing.subprocess, "run", lambda *a, **k: calls.append(k["timeout"]) or NS(returncode=0))
    timing.run_child("parent","a"*40,timing.time.monotonic()+500)
    assert calls==["idle",120]
    with pytest.raises(ValueError): timing.run_child("control","a"*40,timing.time.monotonic()+150)
    assert calls==["idle",120]
