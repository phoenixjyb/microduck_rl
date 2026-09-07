"""Single-axis configuration, bounded execution, and unchanged numerical gates."""
from dataclasses import asdict
from types import SimpleNamespace as NS
import datetime as dt
import json

import pytest
import torch

from mjlab_microduck import foundation_yaw_experiment as exp
from test_foundation_motor_experiment import good as motor_good


def good(seed=503):
    r=motor_good(seed);r.update(protocol=exp.PROTOCOL,source="a"*40)
    return r


@pytest.mark.parametrize("mode",exp.MODES)
def test_only_yaw_range_changes_and_old_rewards_are_preserved(mode):
    a,aa=exp.prepare_config(mode,arm="control")
    b,ba=exp.prepare_config(mode,arm="yaw")
    assert a.commands["twist"].ranges.ang_vel_z==(0.,0.)
    assert b.commands["twist"].ranges.ang_vel_z==(-.35,.35)
    assert a.rewards[exp.TERM].weight==b.rewards[exp.TERM].weight==-.1
    b.commands["twist"].ranges.ang_vel_z=(0.,0.);ba.run_name=aa.run_name
    assert asdict(a)==asdict(b) and asdict(aa)==asdict(ba)
    old,agent=exp.base.prepare_config(mode,arm="motor")
    aa.run_name,aa.experiment_name=agent.run_name,agent.experiment_name
    assert asdict(a)==asdict(old) and asdict(aa)==asdict(agent)


def test_old_per_joint_and_speed_gates_remain():
    p,c,n=good(),good(),good()
    assert not exp.paired_decision(p,c,n)["failures"]
    n["groups"]["settled"]["pre_reset_joint_p99"]["head_roll"]+=.03
    r=exp.paired_decision(p,c,n)
    assert r["failures"] and not r["policy_acceptance"]


@pytest.mark.parametrize("failure",[None,"parent","control","yaw","runtime","initial","activity"])
def test_first_failure_fixed_order_and_immutable_manifest(tmp_path,monkeypatch,failure):
    output=tmp_path/"new"
    monkeypatch.setattr(exp,"OUTPUT",output)
    for key in ("LAST_START","DEADLINE"):
        monkeypatch.setattr(exp.base,key,dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(exp,"verify_source",lambda s:None)
    monkeypatch.setattr(exp,"verify_history",lambda:{})
    monkeypatch.setattr(exp.base.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(exp,"wait_idle",lambda:{})
    monkeypatch.setattr(exp,"checkpoint",lambda a,s:(None,exp.base.MODEL_SHA))
    monkeypatch.setattr(exp,"audit_motor_trace",lambda r:{})
    calls=[]
    def child(kind,arm,source,end,*,seed=503):
        calls.append((kind,arm,seed))
        if failure=="runtime":raise ValueError("runtime")
        if kind in exp.MODES:
            p=output/f"{arm}-{kind}";p.mkdir()
            (p/"initial.pt").write_bytes(b"different" if failure=="initial" and arm=="yaw" else b"same")
            exp.write_new(p/"fingerprint.json",{})
        else:
            r=good(seed)
            if arm==failure:
                if arm=="yaw":r["groups"]["settled"]["pre_reset_joint_p99"]["head_roll"]+=.03
                else:r.update(safety_failures=["terminal-including-startup"],classification="safety-or-coverage-stop")
            exp.write_new(output/f"{arm}-s{seed}.json",r)
            exp.write_new(output/f"{arm}-s{seed}-fingerprint.json",{})
    def training(mode,arm,source):
        if failure=="activity":raise ValueError("inactive neck")
        return dict(wall_seconds=1)
    monkeypatch.setattr(exp,"run_child",child)
    monkeypatch.setattr(exp,"validate_training",training)
    if failure in ("runtime","initial","activity"):
        with pytest.raises(ValueError):exp.campaign("a"*40)
    else:exp.campaign("a"*40)
    d=json.loads((output/"decision.json").read_text())
    expected={None:"single-seed-diagnostic-support-only","parent":"reference-safety-stop",
              "control":"reference-safety-stop","yaw":"numerical-gate-stop"}
    assert d["decision"]==expected.get(failure,"runtime-failure-stop") and not d["policy_acceptance"]
    if failure=="parent":assert len(calls)==1
    if failure in ("initial","activity"):assert all(c[0]!="pilot" for c in calls)
    if failure in ("yaw","control"):assert calls[-1]==("evaluate",failure,503)
    if failure is None:assert len(calls)==13
    manifest=json.loads((output/"manifest.json").read_text())["files"]
    assert set(manifest)=={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"}
    for name,row in manifest.items():assert exp.sha256(output/name)==row["sha256"]
    with pytest.raises(FileExistsError):exp.campaign("a"*40)


@pytest.mark.parametrize("kind,seed,expected,timeout",[("smoke",503,"491",180),("pilot",503,"499",900),("evaluate",509,"509",120)])
def test_child_start_hash_and_caps(tmp_path,monkeypatch,kind,seed,expected,timeout):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    monkeypatch.setattr(exp.base,"DEADLINE",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(exp,"wait_idle",lambda:{})
    calls=[]
    monkeypatch.setattr(exp.subprocess,"run",lambda *a,**k:calls.append(k) or NS(returncode=0))
    exp.run_child(kind,"yaw","a"*40,exp.time.monotonic()+1100,seed=seed)
    assert calls[0]["env"]["PYTHONHASHSEED"]==expected and calls[0]["timeout"]==timeout
    with pytest.raises(ValueError):exp.run_child(kind,"control","a"*40,exp.time.monotonic()+timeout,seed=seed)
    assert len(calls)==1


def test_training_child_attaches_read_only_guards_and_original_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    monkeypatch.setattr(exp,"verify_source",lambda s:None)
    monkeypatch.setattr(exp,"verify_history",lambda:None)
    monkeypatch.setattr(exp.base.training,"runtime_identity",lambda:None)
    monkeypatch.setattr(exp.base,"check_host",lambda:None)
    monkeypatch.setattr(exp.history.replay,"fingerprint",lambda:dict(environment=dict(PYTHONHASHSEED="499")))
    calls=[]
    def train(mode,path,**kw):
        calls.append(mode)
        assert kw['parent']==exp.base.MODEL and kw['parent_iteration']==8498 and kw['parent_step']==204000
        assert kw['reward_observer'].weight==-.1 and kw['command_validator'].yaw_range==(-.35,.35)
        cfg,_=kw['config_factory'](mode)
        assert cfg.commands['twist'].ranges.ang_vel_z==(-.35,.35)
        return dict(status='training-complete-not-accepted')
    monkeypatch.setattr(exp.base.training,'train',train)
    exp.child('pilot','yaw',503,'a'*40)
    assert calls==['pilot'] and (tmp_path/'yaw-pilot/fingerprint.json').exists()


@pytest.mark.parametrize("corrupt",[None,"counter","missing","nonfinite","weight","final"])
def test_retained_training_checkpoints_and_live_activity(tmp_path,monkeypatch,corrupt):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    monkeypatch.setattr(exp,"validate_command_evidence",lambda *a: None)
    p=tmp_path/"yaw-smoke";p.mkdir()
    for label in (8500,8508):
        payload=dict(iter=label,infos=dict(env_state=dict(common_step_counter=204000+24*(label-8498))),
            actor_state_dict=dict(w=torch.ones(1)),critic_state_dict={},optimizer_state_dict={})
        if label==8500:
            if corrupt=="counter":payload["infos"]["env_state"]["common_step_counter"]+=1
            if corrupt=="nonfinite":payload["actor_state_dict"]["w"][0]=float("nan")
            if corrupt=="missing":continue
        torch.save(payload,p/f"model_{label}.pt")
    final=p/"model_8508.pt"
    exp.write_new(p/"result.json",dict(source="a"*40,status="training-complete-not-accepted",updates=10,
        parent_step=204000,common_step=204240,final_checkpoint=str(final),
        final_sha256="bad" if corrupt=="final" else exp.sha256(final),
        reward_observer=dict(protocol="f1n-live-neck-reward-v1",control_steps=240,neck_weight=-.1,
                             positive_samples=1,raw_neck_cost_mean=.01)))
    rows=[dict(control_steps=24*i,common_step=204000+24*i,neck_weight=-.1,
               motor_weight=-2. if corrupt=="weight" else -4.,lateral_weight=-.5,samples=6144) for i in range(1,11)]
    (p/"neck-reward-activity.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if corrupt:
        with pytest.raises(ValueError):exp.validate_training("smoke","yaw","a"*40)
    else:assert exp.validate_training("smoke","yaw","a"*40)["updates"]==10
