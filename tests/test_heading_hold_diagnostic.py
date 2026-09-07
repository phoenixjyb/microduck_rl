"""CPU heading controller and fresh-command integration, not gait acceptance."""

import math
import datetime as dt
import json
import sys
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.heading_hold_diagnostic import HeadingHold, compare, validate_controller_trace, PROTOCOL, MODEL_SHA
from test_command_delivery import cpu_manager
from test_foundation_trace_audit import example


def test_sign_wrap_slew_and_fixed_forward_command():
    c=HeadingHold(True)
    heading=torch.tensor([.8,-.8,2*math.pi-.2,0.])
    out=c.command(heading,0)
    torch.testing.assert_close(out[:,2],torch.tensor([-.02,.02,.02,0.]))
    assert torch.equal(out[:,:2],torch.tensor([.3,0.]).expand(4,2))
    previous=out[:,2]
    for step in range(1,400):
        out=c.command(heading,step)
        assert bool((out[:,2].abs()<=.35+1e-6).all())
        assert bool(((out[:,2]-previous).abs()<=.020001).all())
        previous=out[:,2]
    with pytest.raises(ValueError):c.command(heading,400)


def test_off_is_exact_constant_and_invalid_data_does_not_advance_state():
    c=HeadingHold(False)
    with pytest.raises(ValueError):c.command(torch.tensor([float("nan")]),0)
    assert c.next_step==0 and c.previous is None
    assert c.command(torch.tensor([1.]),0)[0,2]==0
    with pytest.raises(ValueError):c.command(torch.tensor([1.]),0)


def test_adapter_updates_same_step_actor_only_without_resampling(cpu_manager):
    env,manager=cpu_manager
    env.observation_manager=manager
    obs=TensorDict(manager.compute(update_history=True),batch_size=[2])
    original=obs.clone();rng=torch.get_rng_state().clone()
    result=HeadingHold(True).prepare(obs,env,0,torch.tensor([.5,-.5]))
    assert torch.equal(result.consumed,result.issued)
    assert not torch.equal(result.cached,result.consumed)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(original["actor"],obs["actor"])
    assert torch.equal(result.observations["actor"][:,:48],obs["actor"][:,:48])
    assert torch.equal(result.observations["actor"][:,51:],obs["actor"][:,51:])


def report(enabled):
    r=example()
    r.update(protocol=PROTOCOL,seed=443,checkpoint_sha256=MODEL_SHA)
    c=HeadingHold(enabled)
    commands=torch.stack([c.command(torch.zeros(8),i) for i in range(400)]).tolist()
    r["command_adapter"]=c.provenance()
    r["command_trace"]=dict(issued=commands,actor_input=commands,cached=commands,inference_and_simulation_steps=400)
    r["groups"]["settled"]["pre_reset_joint_p99"]={"left_knee":.4}
    return r


def test_trace_controller_provenance_and_per_joint_gate():
    off,on=report(False),report(True)
    assert compare(off,on)["failures"]==[]
    on["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"]+=.03
    assert "joint-torque-nonregression:left_knee" in compare(off,on)["failures"]
    assert not compare(off,on)["hop_validated"]


@pytest.mark.parametrize("mutate",[
    lambda r:r["command_adapter"].update(kp=2.),
    lambda r:r["command_trace"].update(inference_and_simulation_steps=399),
    lambda r:r["command_trace"]["issued"][10][0].__setitem__(2,.1),
    lambda r:r.update(checkpoint_sha256="wrong"),
    lambda r:r.update(seed=431),
])
def test_invalid_controller_trace_rejected(mutate):
    r=report(True);mutate(r)
    with pytest.raises(ValueError):validate_controller_trace(r,True)


@pytest.mark.parametrize("failure", [None, "reference", "numerical", "runtime"])
def test_campaign_sequential_first_failure_stop_and_manifest(tmp_path, monkeypatch, failure):
    from mjlab_microduck import heading_hold_diagnostic as diagnostic
    output=tmp_path/"diagnostic"
    monkeypatch.setattr(diagnostic,"OUTPUT",output)
    monkeypatch.setattr(diagnostic,"LAST_START",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(diagnostic,"verify_source",lambda source:None)
    monkeypatch.setattr(diagnostic,"runtime_identity",lambda:{})
    monkeypatch.setattr(diagnostic,"check_host",lambda:{"idle":True})
    monkeypatch.setattr(diagnostic,"preserved",lambda:{"unchanged":True})
    monkeypatch.setattr(sys,"argv",["diagnostic","--source","a"*40])
    calls=[]
    def child(args, **kwargs):
        arm,seed=args[-3],int(args[-1])
        calls.append((arm,seed))
        assert kwargs["timeout"]==90 and kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="0"
        if failure=="runtime": return SimpleNamespace(returncode=1)
        result=report(arm=="on");result["seed"]=seed
        if failure=="reference": result["safety_failures"]=["mock safety stop"]
        if failure=="numerical" and arm=="on":
            result["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"]+=.03
        diagnostic.write_new(output/f"{arm}-s{seed}.json",result)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(diagnostic.subprocess,"run",child)
    if failure=="runtime":
        with pytest.raises(ValueError):diagnostic.main()
    else:diagnostic.main()
    expected={None:6,"reference":1,"numerical":2,"runtime":1}[failure]
    assert calls==[(arm,seed) for seed in diagnostic.SEEDS for arm in ("off","on")][:expected]
    decision=json.loads((output/"decision.json").read_text())
    assert decision["decision"]=={None:"heading-diagnostic-support-only","reference":"reference-safety-stop",
                                 "numerical":"numerical-gate-stop","runtime":"runtime-failure-stop"}[failure]
    assert not decision["obstacles_admitted"] and not decision["hop_validated"]
    manifest=json.loads((output/"manifest.json").read_text())
    assert set(manifest["files"])=={p.name for p in output.iterdir() if p.name!="manifest.json"}
    for name,identity in manifest["files"].items():
        assert diagnostic.sha256(output/name)==identity["sha256"]
        assert (output/name).stat().st_size==identity["bytes"]
