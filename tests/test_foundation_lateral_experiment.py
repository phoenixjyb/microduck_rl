"""Bounded lateral-training contracts; synthetic checks are not skill evidence."""

from dataclasses import asdict
from types import SimpleNamespace as NS
import datetime as dt
import json
import math

import pytest
import torch

from mjlab_microduck import foundation_lateral_experiment as exp
from mjlab_microduck import foundation_pilot as training
from test_heading_hold_diagnostic import report


def test_lateral_cost_yaw_frame_sign_scale_and_no_mutation():
    yaw=torch.tensor([0.,0.,math.pi/2,math.pi/2],dtype=torch.float64)
    q=torch.stack(((yaw/2).cos(),yaw*0,yaw*0,(yaw/2).sin()),-1)
    v=torch.tensor([[.3,.1,2.],[.3,-.1,-2.],[-.1,.3,5.],[0.,.3,0.]],dtype=torch.float64)
    env=NS(scene={"robot":NS(data=NS(root_link_quat_w=q,root_link_lin_vel_w=v))})
    before=q.clone(),v.clone()
    torch.testing.assert_close(exp.lateral_velocity_cost(env),torch.tensor([1.,1.,1.,0.],dtype=torch.float64))
    assert torch.equal(q,before[0]) and torch.equal(v,before[1])
    v[0,1]=float("nan")
    with pytest.raises(ValueError):exp.lateral_velocity_cost(env)
    v[0,1]=1e200
    with pytest.raises(ValueError):exp.lateral_velocity_cost(env)


@pytest.mark.parametrize("mode",exp.MODES)
def test_configs_change_only_one_cost_weight_with_historical_contracts_preserved(mode):
    a,aa=exp.prepare_config(mode,arm="control");b,ba=exp.prepare_config(mode,arm="lateral")
    assert a.rewards["lateral_velocity_cost"].weight==0
    assert b.rewards["lateral_velocity_cost"].weight==-.5
    assert b.rewards["lateral_velocity_cost"].func.__module__=="mjlab_microduck.lateral_cost"
    b.rewards["lateral_velocity_cost"].weight=0;ba.run_name=aa.run_name
    assert asdict(a)==asdict(b) and asdict(aa)==asdict(ba)
    original,agent=training.prepare_config(mode)
    del a.rewards["lateral_velocity_cost"]
    a.rewards["track_linear_velocity"].params["std"]=math.sqrt(.15)
    assert a.seed==aa.seed==(461 if mode=="smoke" else 463)
    assert aa.max_iterations==(10 if mode=="smoke" else 500)
    a.seed=original.seed;aa.seed=agent.seed
    aa.experiment_name,aa.run_name=agent.experiment_name,agent.run_name
    assert asdict(a)==asdict(original) and asdict(aa)==asdict(agent)


def test_explicit_resume_preserves_optimizer_and_environment_time():
    runner=NS(device="cpu",current_learning_iteration=8498,
        env=NS(unwrapped=NS(common_step_counter=204000)),
        alg=NS(learning_rate=.1,optimizer=NS(param_groups=[{"lr":.0001}])) )
    runner.load=lambda *a,**k:None
    training.restore_parent(runner,"model_8498.pt",parent_iteration=8498,parent_step=204000)
    assert runner.current_learning_iteration==8499 and runner.alg.learning_rate==.0001
    assert runner.env.unwrapped.common_step_counter==204000
    with pytest.raises(ValueError):training.restore_parent(runner,"bad.pt",parent_iteration=8498,parent_step=204000)


def good(seed=467):
    r=report(True);r.update(protocol=exp.PROTOCOL,seed=seed)
    r["groups"]["settled"]["pre_reset_joint_p99"]={joint:.4 for joint in exp.JOINTS}
    return r


def test_retains_absolute_paired_per_joint_and_power_gates():
    p,c,l=good(),good(),good()
    assert exp.paired_decision(p,c,l)["failures"]==[]
    l["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"]+=.03
    assert "joint-torque-nonregression:left_knee" in exp.paired_decision(p,c,l)["failures"]
    assert "matched-control:joint-torque-nonregression:left_knee" in exp.paired_decision(p,c,l)["failures"]
    l=good();l["groups"]["settled"]["pre_reset_mechanical_abs_power_mean_w"]*=1.06
    assert "pre_reset_mechanical_abs_power_mean_w-nonregression" in exp.paired_decision(p,c,l)["failures"]
    del l["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"]
    with pytest.raises(ValueError):exp.paired_decision(p,c,l)


def test_heading_trace_identity_is_explicit_for_new_protocol():
    r=good()
    exp.validate_controller_trace(r,True,protocol=exp.PROTOCOL,seeds=exp.SEEDS,checkpoint_sha=exp.MODEL_SHA)
    with pytest.raises(ValueError):exp.validate_controller_trace(r,True)
    with pytest.raises(ValueError):
        exp.validate_controller_trace(r,True,protocol=exp.PROTOCOL,seeds=exp.SEEDS,checkpoint_sha="wrong")


def test_child_timeout_and_insufficient_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path);monkeypatch.setattr(exp,"check_host",lambda:{})
    calls=[]
    def run(args,**kw):calls.append((args,kw));return NS(returncode=0)
    monkeypatch.setattr(exp.subprocess,"run",run)
    exp.run_child("pilot","lateral","a"*40,exp.time.monotonic()+1000)
    assert calls[0][0][1:3]==["-m","mjlab_microduck.foundation_lateral_experiment"]
    assert calls[0][1]["timeout"]==900
    with pytest.raises(ValueError):exp.run_child("pilot","control","a"*40,exp.time.monotonic()+900)
    assert len(calls)==1


@pytest.mark.parametrize("failure",[None,"parent","control","lateral","runtime","initial"])
def test_campaign_reference_first_no_overlap_first_failure_and_hash_manifest(tmp_path,monkeypatch,failure):
    output=tmp_path/"campaign"
    monkeypatch.setattr(exp,"OUTPUT",output)
    monkeypatch.setattr(exp,"LAST_START",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(exp,"verify_source",lambda s:None)
    monkeypatch.setattr(exp.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(exp,"check_host",lambda:{})
    monkeypatch.setattr(exp,"preserved",lambda:{"unchanged":True})
    monkeypatch.setattr(exp,"checkpoint",lambda arm,source:(None,exp.MODEL_SHA))
    calls=[]
    def child(kind,arm,source,end,*,seed=467):
        calls.append((kind,arm,seed))
        if failure=="runtime":raise ValueError("runtime")
        if kind in exp.MODES:
            path=output/f"{arm}-{kind}";path.mkdir()
            (path/"initial.pt").write_bytes(b"mismatch" if failure=="initial" and arm=="lateral" else b"paired")
            exp.write_new(path/"result.json",dict(source=source,updates=10,wall_seconds=1))
        else:
            r=good(seed)
            if arm==failure:
                if failure=="lateral":r["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"]+=.03
                else:r.update(safety_failures=["terminal-including-startup"],classification="safety-or-coverage-stop")
            exp.write_new(output/f"{arm}-s{seed}.json",r)
    monkeypatch.setattr(exp,"run_child",child)
    if failure in ("runtime","initial"):
        with pytest.raises(ValueError):exp.campaign("a"*40)
    else:exp.campaign("a"*40)
    d=json.loads((output/"decision.json").read_text())
    expected={None:"single-seed-diagnostic-support-only","parent":"reference-safety-stop",
        "control":"reference-safety-stop","lateral":"numerical-gate-stop",
        "runtime":"runtime-failure-stop","initial":"runtime-failure-stop"}
    assert d["decision"]==expected[failure] and not d["obstacles_admitted"] and not d["hop_validated"]
    assert calls[0]==("evaluate","parent",467)
    if failure=="parent":assert len(calls)==1
    if failure=="initial":assert all(kind!="pilot" for kind,_,_ in calls)
    if failure=="lateral":assert calls[-1]==("evaluate","lateral",467)
    if failure is None:assert len(calls)==13 and len(d["reports"])==9
    manifest=json.loads((output/"manifest.json").read_text())
    assert set(manifest["files"])=={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"}
    for name,row in manifest["files"].items():
        assert exp.sha256(output/name)==row["sha256"] and (output/name).stat().st_size==row["bytes"]
    with pytest.raises(FileExistsError):exp.campaign("a"*40)
