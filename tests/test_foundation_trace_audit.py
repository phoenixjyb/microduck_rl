"""Adversarial retained trace checks; no simulated learning claims."""

import pytest
import torch
import hashlib
import json
from pathlib import Path

from mjlab_microduck.foundation_trace_audit import audit_trace
from mjlab_microduck.speed_response_control import summarize
from test_speed_response_control import inputs


def example():
    data=inputs()
    report=summarize(**data)
    v=data["velocities"]
    p=torch.zeros(400,8,2,dtype=torch.float64)
    p[:,:,0]=torch.arange(400,dtype=torch.float64)[:,None]*.006
    report["route_trace"]=dict(protocol="initial-route-pre-control-v1",
        position_columns=["route_forward_m","cross_route_m"],
        velocity_columns=["body_forward_mps","route_forward_mps","cross_route_mps","heading_rad"],
        position=p.tolist(),velocity=v.tolist(),last_sample_cross_route_m=[0.]*8,
        max_abs_cross_route_m=[0.]*8,signed_cross_route_velocity_mean_mps=[0.]*8)
    return report


def test_reproduces_means_windows_and_trace_accounting_without_gpu():
    result=audit_trace(example())
    assert result==dict(trace_summary_verified=True, stable_response_reproduced=True,
                       sampled_steps=400, policy_acceptance=False)


@pytest.mark.parametrize("mutate", [
    lambda r:r["route_trace"]["position"].pop(),
    lambda r:r["route_trace"]["position"][0][0].__setitem__(1,.01),
    lambda r:r["route_trace"]["velocity"][123][1].__setitem__(1,.1),
    lambda r:r["route_trace"]["velocity"][123][1].__setitem__(2,float("nan")),
    lambda r:r["route_trace"]["last_sample_cross_route_m"].__setitem__(0,.01),
    lambda r:r["groups"]["settled"].__setitem__("body_forward_mean",.29),
    lambda r:r["stable_route_response"]["environments"][0].__setitem__("stable_recovery_latency_s",.52),
])
def test_contradictory_or_incomplete_trace_fails_closed(mutate):
    r=example();mutate(r)
    with pytest.raises((ValueError,RuntimeError)): audit_trace(r)


def test_retained_f1r_manifest_traces_and_failed_pair_when_available():
    root=Path(__file__).resolve().parents[1]/"artifacts"
    options=[root/k/"f1r-width-paired-s421-v1" for k in ("diagnostics","experiments")]
    base=next((p for p in options if (p/"manifest.json").exists()),None)
    if base is None: pytest.skip("separately retained F1-R artifacts unavailable")
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(base/"manifest.json")=="ead4f9b6cd4ef9dcde184f5f70485da9a9c223efc80226afb86f71c183c3f753"
    assert digest(base/"decision.json")=="fde5ce1a0659a1bbbc759215c7548c182a7f61adae25a8b94f60cfbb92db8440"
    manifest=json.loads((base/"manifest.json").read_text())
    assert len(manifest["files"])==74
    assert sum(v["bytes"] for v in manifest["files"].values())==159024034
    for name, expected in manifest["files"].items():
        path=base/name
        assert path.stat().st_size==expected["bytes"] and digest(path)==expected["sha256"]
    decision=json.loads((base/"decision.json").read_text())
    assert decision["source"]=="77a1d612495c484b42a0c58702d55eae163bff1b"
    assert decision["decision"]=="numerical-gate-stop"
    assert [(r["arm"],r["seed"]) for r in decision["reports"]]==[
        ("parent",431),("parent",433),("parent",439),("control",431),("narrow",431)]
    reports={}
    for entry in decision["reports"]:
        path=base/f'{entry["arm"]}-s{entry["seed"]}.json'
        assert digest(path)==entry["sha256"]
        r=json.loads(path.read_text())
        assert audit_trace(r)["stable_response_reproduced"] is True
        assert not r["safety_failures"] and r["sample_steps"]==400
        reports[entry["arm"],entry["seed"]]=r
    from mjlab_microduck.foundation_reward_experiment import paired_decision
    pair=paired_decision(*(reports[arm,431] for arm in ("parent","control","narrow")))
    assert decision["pairs"]==[pair] and decision["failures"]==pair["failures"]
    assert "matched-control:torque-nonregression" in pair["failures"]
    assert not any(decision[k] for k in ("policy_acceptance","f2_admitted","obstacles_admitted","physical_motion_authorized"))
    assert len(list(base.glob("*-s*.json")))==5
