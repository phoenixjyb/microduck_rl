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


def hash_states():
    s = states()
    for mode in replay.MODES:
        for phase in ("before", "after"):
            s[mode][phase].update(environment={"PYTHONHASHSEED": "503"}, python_hash_probe=50342)
    return s


@pytest.mark.parametrize("failure", [None, "unset", "probe", "after_probe"])
def test_startup_hash_control_requires_effective_identical_entry_and_exit_hash(failure):
    s = hash_states()
    if failure == "unset": s["first"]["before"]["environment"]["PYTHONHASHSEED"] = None
    if failure == "probe": s["second"]["before"]["python_hash_probe"] += 1
    if failure == "after_probe": s["first"]["after"]["python_hash_probe"] += 1
    if failure:
        with pytest.raises(ValueError, match="startup hash503"):
            replay.compare_pair(report(), report(), s, require_startup_hash=True)
    else:
        a, b = report(), report()
        d = replay.compare_pair(a,b,s,require_startup_hash=True)
        assert d["decision"] == "startup-hash-controlled-exact-match-in-this-pair"
        assert d["python_hash_probe_equal"] and not d["policy_acceptance"]
        b["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"] += .001
        assert replay.compare_pair(a,b,s,require_startup_hash=True)["decision"] == "startup-hash-controlled-divergence"


def test_child_start_environment_changes_only_python_hash_and_not_parent(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    parent = dict(replay.os.environ)
    plain, controlled = replay.child_environment(), replay.child_environment(True)
    assert set(plain) ^ set(controlled) == {"PYTHONHASHSEED"}
    assert controlled.pop("PYTHONHASHSEED") == "503" and plain == controlled
    assert dict(replay.os.environ) == parent


def test_real_fresh_python_children_agree_on_public_hash_probe():
    argv=[replay.sys.executable,"-c","print(hash('microduck-repeatability-probe-v1'))"]
    values=[replay.subprocess.check_output(argv,env=replay.child_environment(True),text=True,timeout=10)
            for _ in range(2)]
    assert values[0] == values[1]


@pytest.mark.parametrize("failure", [None,"payload","extra","decision"])
def test_closed_off_control_is_hash_and_coverage_protected(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(replay,"OUTPUT",tmp_path)
    for name in ["decision.json"]+[f"case-{i}.json" for i in range(9)]:
        replay.write_new(tmp_path/name,{"value":0})
    files={p.name:dict(sha256=replay.sha256(p),bytes=p.stat().st_size) for p in tmp_path.iterdir()}
    replay.write_new(tmp_path/"manifest.json",dict(files=files))
    monkeypatch.setattr(replay,"OFF_MANIFEST",replay.sha256(tmp_path/"manifest.json"))
    monkeypatch.setattr(replay,"OFF_DECISION",replay.sha256(tmp_path/"decision.json"))
    if failure=="payload": (tmp_path/"case-0.json").write_text('{"value":1}')
    if failure=="extra": replay.write_new(tmp_path/"extra.json",{})
    if failure=="decision": (tmp_path/"decision.json").write_text('{"value":1}')
    if failure:
        with pytest.raises(ValueError): replay.verify_off_control()
    else: replay.verify_off_control()


def test_startup_flag_and_environment_reach_fresh_child_only_in_new_directory(tmp_path,monkeypatch):
    old, new = tmp_path/"old", tmp_path/"new"
    old.mkdir(); new.mkdir(); (old/"closed").write_text("preserve")
    monkeypatch.setattr(replay,"OUTPUT",old); monkeypatch.setattr(replay,"HASH_OUTPUT",new)
    monkeypatch.setattr(replay.exp,"DEADLINE",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(replay,"wait_idle",lambda:{})
    seen=[]
    monkeypatch.setattr(replay.subprocess,"run",lambda argv,**kw: seen.append((argv,kw)) or NS(returncode=0))
    replay.run_child("first","a"*40,replay.time.monotonic()+300,startup_hash=True)
    argv, kw = seen[0]
    assert argv[-1] == "--startup-hash" and kw["env"]["PYTHONHASHSEED"] == "503" and kw["timeout"] == 120
    assert sorted(p.name for p in old.iterdir()) == ["closed"]
    assert (new/"first-idle.json").is_file() and (new/"first.log").is_file()


def test_startup_campaign_preserves_closed_control_and_limits_two_cases(tmp_path,monkeypatch):
    output=tmp_path/"new"
    monkeypatch.setattr(replay,"HASH_OUTPUT",output)
    monkeypatch.setattr(replay,"verify_source",lambda s:None)
    monkeypatch.setattr(replay,"verify_history",lambda:{})
    verified=[]
    monkeypatch.setattr(replay,"verify_off_control",lambda:verified.append(True))
    monkeypatch.setattr(replay.exp.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(replay.exp,"LAST_START",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(replay,"wait_idle",lambda:{})
    def forbidden(*a,**k): raise AssertionError("no optimizer")
    monkeypatch.setattr(replay.exp.training,"train",forbidden)
    calls=[]
    def child(mode,source,end,*,startup_hash=False):
        assert startup_hash
        calls.append(mode)
        replay.write_new(output/f"{mode}.json",report())
        replay.write_new(output/f"{mode}-fingerprint.json",hash_states()[mode])
    monkeypatch.setattr(replay,"run_child",child)
    replay.campaign("a"*40,startup_hash=True)
    decision=json.loads((output/"decision.json").read_text())
    launch=json.loads((output/"launch.json").read_text())
    assert calls==["first","second"] and len(verified)==2
    assert decision["protocol"]==replay.HASH_PROTOCOL and decision["optimizer_updates"]==0
    assert launch["startup_hash_control"] and launch["child_numerical_environment"]["PYTHONHASHSEED"]=="503"
    with pytest.raises(FileExistsError): replay.campaign("a"*40,startup_hash=True)
