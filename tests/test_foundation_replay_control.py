import copy
import datetime as dt
import json
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import foundation_replay_control as replay
from test_foundation_motor_experiment import good


def report():
    r = good(); r["source"] = "a"*40
    return r


def states():
    return {mode: {phase: dict(setting="same", python_hash_probe=i)
                   for phase in ("before", "after")} for i, mode in enumerate(replay.MODES)}


def test_fingerprint_reads_without_rng_or_cuda_initialization_changes():
    rng=torch.get_rng_state().clone(); cuda=torch.cuda.is_initialized()
    result=replay.fingerprint()
    assert torch.equal(rng,torch.get_rng_state()) and torch.cuda.is_initialized()==cuda
    assert result["python_hash_probe"]==hash("microduck-repeatability-probe-v1")
    assert set(result["environment"])==set(replay.ENV_KEYS)


def test_full_exact_comparison_and_uncontrolled_hash_probe_are_separate():
    a, b = report(), report()
    result=replay.compare_pair(a,b,states())
    assert result["decision"]=="recording-disabled-exact-match-in-this-pair"
    assert not result["python_hash_probe_equal"] and not result["third_case_admitted"]
    b["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"]+=.001
    result=replay.compare_pair(a,b,states())
    assert result["decision"]=="recording-disabled-same-source-divergence" and result["difference_count"]==1
    assert result["differences"][0]["path"]=="/groups/settled/pre_reset_joint_p99/left_knee"
    assert not result["specific_numerical_cause_established"] and not result["policy_acceptance"]


@pytest.mark.parametrize("what", ["source", "recording", "settings", "unsafe"])
def test_invalid_or_changed_control_is_not_a_repeatability_conclusion(what):
    a,b,s=report(),report(),states()
    if what=="source": b["source"]="b"*40
    if what=="recording": b["motor_trace"]={}
    if what=="settings": s["second"]["before"]["setting"]="changed"
    if what=="unsafe":
        b.update(safety_failures=["terminal-including-startup"],classification="safety-or-coverage-stop")
    with pytest.raises(ValueError): replay.compare_pair(a,b,s)


@pytest.mark.parametrize("failure", [None,"first","second","child"])
def test_only_two_sequential_cases_first_unsafe_stops_and_no_optimizer(tmp_path,monkeypatch,failure):
    output=tmp_path/"new"
    monkeypatch.setattr(replay,"OUTPUT",output)
    monkeypatch.setattr(replay,"verify_source",lambda s:None)
    monkeypatch.setattr(replay,"verify_history",lambda:{})
    monkeypatch.setattr(replay.exp.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(replay.exp,"LAST_START",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(replay,"wait_idle",lambda:{})
    def forbidden(*a,**k): raise AssertionError("no optimizer work")
    monkeypatch.setattr(replay.exp.training,"train",forbidden)
    calls=[]
    def child(mode,source,end):
        calls.append(mode)
        if failure=="child": raise ValueError("child failed")
        r=report()
        if failure==mode:r.update(safety_failures=["absolute-torque"],classification="safety-or-coverage-stop")
        replay.write_new(output/f"{mode}.json",r)
        replay.write_new(output/f"{mode}-fingerprint.json",states()[mode])
    monkeypatch.setattr(replay,"run_child",child)
    if failure=="child":
        with pytest.raises(ValueError): replay.campaign("a"*40)
    else: replay.campaign("a"*40)
    d=json.loads((output/"decision.json").read_text())
    assert calls==(["first"] if failure in ("first","child") else ["first","second"])
    assert d["optimizer_updates"]==0 and not d["third_case_admitted"]
    assert d["decision"]==("runtime-failure-stop" if failure=="child" else "absolute-safety-stop" if failure else
                            "recording-disabled-exact-match-in-this-pair")
    m=json.loads((output/"manifest.json").read_text())
    assert set(m["files"])=={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"}
    for n,r in m["files"].items(): assert replay.sha256(output/n)==r["sha256"]
    with pytest.raises(FileExistsError): replay.campaign("a"*40)


def test_idle_gate_precedes_child_and_budget_expiry_prevents_it(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,"OUTPUT",tmp_path)
    monkeypatch.setattr(replay.exp,"DEADLINE",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    calls=[]
    monkeypatch.setattr(replay,"wait_idle",lambda:calls.append("idle") or {})
    monkeypatch.setattr(replay.subprocess,"run",lambda *a,**k:calls.append(k["timeout"]) or NS(returncode=0))
    replay.run_child("first","a"*40,replay.time.monotonic()+300)
    assert calls==["idle",120]
    with pytest.raises(ValueError):replay.run_child("second","a"*40,replay.time.monotonic()+120)
    assert calls==["idle",120]
