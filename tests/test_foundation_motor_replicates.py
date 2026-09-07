"""Sequential, immutable collection lifecycle; never learned behavior proof."""

import copy
import datetime as dt
import json
from types import SimpleNamespace as NS

import pytest

from mjlab_microduck import foundation_motor_replicates as runner
from test_motor_measurement_contract import records, SOURCE


@pytest.mark.parametrize("failure",[None,"first","later","child","fingerprint"])
def test_fixed_order_first_failure_stop_preserved_manifest_and_no_optimizer(tmp_path,monkeypatch,failure):
    output=tmp_path/"new"
    rows=records(); calls=[]; verified=[]
    baseline=copy.deepcopy(rows[0]["fingerprint"])
    monkeypatch.setattr(runner,"OUTPUT",output)
    monkeypatch.setattr(runner,"verify_source",lambda s:None)
    monkeypatch.setattr(runner,"verify_history",lambda:verified.append(True) or dict(baseline_fingerprint=copy.deepcopy(baseline)))
    monkeypatch.setattr(runner.exp.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(runner.exp,"LAST_START",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(runner.exp,"DEADLINE",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(runner,"wait_idle",lambda:{})
    def forbidden(*a,**k):raise AssertionError("no optimizer")
    monkeypatch.setattr(runner.exp.training,"train",forbidden)
    def child(case,source,end):
        calls.append(case)
        if failure=="child":raise ValueError("child failed")
        row=rows[len(calls)-1]
        if failure=="first" or failure=="later" and case=="motor-1":
            row["report"].update(safety_failures=["all-legacy-torque"],classification="safety-or-coverage-stop")
        if failure=="fingerprint": row["fingerprint"]["after"]["threads"]=99
        runner.write_new(output/f"{case}.json",row["report"])
        runner.write_new(output/f"{case}-fingerprint.json",row["fingerprint"])
    monkeypatch.setattr(runner,"run_child",child)
    if failure in ("child","fingerprint"):
        with pytest.raises(ValueError):runner.campaign(SOURCE)
    else:runner.campaign(SOURCE)
    decision=json.loads((output/"decision.json").read_text())
    expected={None:"descriptive-replicates-only","first":"absolute-safety-stop","later":"absolute-safety-stop",
              "child":"runtime-failure-stop","fingerprint":"runtime-failure-stop"}
    assert decision["decision"]==expected[failure] and decision["optimizer_updates"]==0
    count={None:6,"first":1,"later":3,"child":1,"fingerprint":1}[failure]
    assert calls==list(runner.contract.ORDER[:count])
    assert all(decision[k] is False for k in runner.contract.CLAIMS)
    manifest=json.loads((output/"manifest.json").read_text())
    assert set(manifest["files"])=={p.name for p in output.iterdir() if p.name!="manifest.json"}
    for name,row in manifest["files"].items():assert runner.sha256(output/name)==row["sha256"]
    with pytest.raises(FileExistsError):runner.campaign(SOURCE)


def test_idle_gate_timeout_startup_hash_and_budget_precede_child(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"OUTPUT",tmp_path)
    monkeypatch.setattr(runner.exp,"DEADLINE",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    order=[]
    monkeypatch.setattr(runner,"wait_idle",lambda:order.append("idle") or {})
    monkeypatch.setattr(runner.subprocess,"run",lambda argv,**kw:order.append((argv,kw)) or NS(returncode=0))
    runner.run_child("parent-1",SOURCE,runner.time.monotonic()+500)
    assert order[0]=="idle" and order[1][1]["timeout"]==120
    assert order[1][1]["env"]["PYTHONHASHSEED"]=="503"
    assert order[1][0][-1]=="parent-1"
    with pytest.raises(ValueError):runner.run_child("control-1",SOURCE,runner.time.monotonic()+120)
    assert len(order)==2


def test_late_start_refused_before_directory_or_history_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"OUTPUT",tmp_path/"new")
    monkeypatch.setattr(runner,"verify_source",lambda s:None)
    monkeypatch.setattr(runner.exp,"LAST_START",dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    def forbidden():raise AssertionError("must fail before history")
    monkeypatch.setattr(runner,"verify_history",forbidden)
    with pytest.raises(ValueError):runner.campaign(SOURCE)
    assert not runner.OUTPUT.exists()


def test_child_retains_raw_report_before_validation_failure_and_only_records(tmp_path,monkeypatch):
    row=records()[0]; calls=[]
    monkeypatch.setattr(runner,"OUTPUT",tmp_path)
    monkeypatch.setattr(runner,"verify_source",lambda s:None)
    monkeypatch.setattr(runner,"verify_history",lambda:dict(baseline_fingerprint=row["fingerprint"]))
    monkeypatch.setattr(runner.exp.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(runner.exp,"check_host",lambda:{})
    monkeypatch.setattr(runner.exp,"preserved",lambda:{})
    monkeypatch.setattr(runner.exp,"DEADLINE",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(runner.exp,"checkpoint",lambda *a:("fixed.pt","wrong-expected-hash"))
    snapshots=iter((row["fingerprint"]["before"],row["fingerprint"]["after"]))
    monkeypatch.setattr(runner.replay,"fingerprint",lambda:next(snapshots))
    def record(**kwargs):calls.append(kwargs);return row["report"]
    monkeypatch.setattr(runner.exp,"run_control",record)
    def forbidden(*a,**k):raise AssertionError("no optimizer")
    monkeypatch.setattr(runner.exp.training,"train",forbidden)
    runner.write_new(tmp_path/"launch.json",dict(source=SOURCE,cases=list(runner.contract.ORDER)))
    with pytest.raises(ValueError,match="recorded fixed checkpoint"):runner.record_child("parent-1",SOURCE)
    assert (tmp_path/"parent-1.json").is_file() and (tmp_path/"parent-1-fingerprint.json").is_file()
    assert len(calls)==1 and calls[0]["retain_motor_trace"] and calls[0]["retain_route_trace"]
    assert calls[0]["seed"]==503 and calls[0]["checkpoint"]=="fixed.pt"
