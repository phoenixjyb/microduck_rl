"""Matched reward experiment contracts; synthetic checks are not gait evidence."""

from dataclasses import asdict
from types import SimpleNamespace as NS
import datetime as dt
import json
import math

import pytest
import torch

from mjlab_microduck import foundation_reward_experiment as exp
from mjlab_microduck import foundation_pilot as old
from mjlab_microduck.speed_response_control import route_position_rows
from test_foundation_pilot import report


@pytest.mark.parametrize("mode", exp.MODES)
def test_matched_configs_differ_only_in_reward_width_and_run_label(mode):
    a, aa = exp.prepare_config(mode, arm="control")
    b, ba = exp.prepare_config(mode, arm="narrow")
    assert a.seed == b.seed == (419 if mode == "smoke" else 421)
    assert aa.max_iterations == ba.max_iterations == (10 if mode == "smoke" else 500)
    assert a.rewards["track_linear_velocity"].weight == b.rewards["track_linear_velocity"].weight == 4.5
    assert a.rewards["track_linear_velocity"].params["std"] == math.sqrt(.15)
    assert b.rewards["track_linear_velocity"].params["std"] == math.sqrt(.05)
    b.rewards["track_linear_velocity"].params["std"] = math.sqrt(.15)
    ba.run_name = aa.run_name
    assert asdict(a) == asdict(b) and asdict(aa) == asdict(ba)
    original, agent = old.prepare_config(mode)
    a.seed = original.seed; aa.seed = agent.seed
    aa.experiment_name, aa.run_name = agent.experiment_name, agent.run_name
    assert asdict(a) == asdict(original) and asdict(aa) == asdict(agent)


def good_report(seed=431):
    r = report()
    r.update(protocol=exp.PROTOCOL, seed=seed)
    r["groups"]["settled"].update(body_forward_mean=.3, route_forward_mean=.3,
        pre_reset_squared_utilization_mean=.03, pre_reset_mechanical_abs_power_mean_w=.1)
    return r


def test_numerical_gate_and_control_nonregression_both_remain_closed():
    p, c, n = good_report(), good_report(), good_report()
    assert not exp.paired_decision(p,c,n)["policy_acceptance"]
    n["groups"]["settled"]["heading_abs_max"] = .3
    assert "heading-drift" in exp.paired_decision(p,c,n)["failures"]
    n = good_report()
    c["groups"]["settled"]["legacy_torque_p99"] = .45
    assert "matched-control:torque-nonregression" in exp.paired_decision(p,c,n)["failures"]


def test_control_performance_failure_is_retained_not_hidden_or_used_to_skip_treatment():
    p, c, n = good_report(), good_report(), good_report()
    c["groups"]["settled"]["heading_abs_max"] = .3
    d = exp.paired_decision(p,c,n)
    assert "heading-drift" in d["control_failures"] and not d["failures"]
    assert not d["policy_acceptance"] and d["independent_training_seeds"] == 1


def test_wrong_seed_and_nonfinite_effects_refused():
    p, c, n = good_report(), good_report(), good_report(433)
    with pytest.raises(ValueError): exp.paired_decision(p,c,n)
    n = good_report(); n["groups"]["settled"]["body_forward_mean"] = float("nan")
    with pytest.raises(ValueError): exp.paired_decision(p,c,n)


def test_route_position_is_translation_and_rotation_invariant_not_velocity_integral():
    origin = torch.tensor([[10., -5., 0.]], dtype=torch.float64)
    pos = origin+torch.tensor([[.2, -.1, .7]], dtype=torch.float64)
    route = torch.tensor([[1., 0.]], dtype=torch.float64)
    expected = route_position_rows(pos, origin, route)
    torch.testing.assert_close(expected, torch.tensor([[.2,-.1]], dtype=torch.float64))
    angle = .7
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
        [math.sin(angle), math.cos(angle), 0.], [0.,0.,1.]], dtype=torch.float64)
    actual = route_position_rows(pos@rotation.T, origin@rotation.T, route@rotation[:2,:2].T)
    torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(route_position_rows(origin, origin, route), torch.zeros_like(expected))
    with pytest.raises(ValueError): route_position_rows(pos, origin, route*2)


def test_child_has_fixed_module_timeout_and_refuses_insufficient_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "check_host", lambda: {})
    calls=[]
    def run(command, **kwargs):
        calls.append((command,kwargs))
        return NS(returncode=0)
    monkeypatch.setattr(exp.subprocess, "run", run)
    exp.run_child("pilot", "control", "a"*40, exp.time.monotonic()+1000)
    command,kw = calls[0]
    assert command[1:3] == ["-m", "mjlab_microduck.foundation_reward_experiment"]
    assert kw["timeout"] == 900 and kw["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    with pytest.raises(ValueError): exp.run_child("pilot", "narrow", "a"*40, exp.time.monotonic()+100)
    assert len(calls)==1


@pytest.mark.parametrize("failure", ["narrow", "parent", "control", "none"])
def test_ordered_campaign_stops_at_first_applicable_failure(tmp_path, monkeypatch, failure):
    output = tmp_path/"experiment"
    monkeypatch.setattr(exp, "OUTPUT", output)
    monkeypatch.setattr(exp, "LAST_START", dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=1))
    monkeypatch.setattr(exp, "verify_source", lambda s: None)
    monkeypatch.setattr(exp.training, "runtime_identity", lambda: {})
    monkeypatch.setattr(exp, "check_host", lambda: {})
    actual_hash = exp.sha256
    monkeypatch.setattr(exp, "sha256", lambda p: exp.ACTOR_SHA256 if p==exp.ACTOR else actual_hash(p))
    calls=[]
    def child(kind, arm, source, end, *, seed=431):
        calls.append((kind,arm,seed))
        if kind in exp.MODES:
            path = output/f"{arm}-{kind}"; path.mkdir()
            (path/"initial.pt").write_bytes(b"same mock restored state")
            (path/"result.json").write_text(json.dumps(dict(updates=10, source=source, wall_seconds=1)))
        else:
            r=good_report(seed)
            if arm==failure:
                if failure=="narrow": r["groups"]["settled"]["heading_abs_max"] = .3
                else:
                    r["safety_failures"]=["terminal-including-startup"]
                    r["classification"]="safety-or-coverage-stop"
            (output/f"{arm}-s{seed}.json").write_text(json.dumps(r))
    monkeypatch.setattr(exp, "run_child", child)
    exp.campaign("a"*40)
    d=json.loads((output/"decision.json").read_text())
    assert calls[:4]==[("smoke","control",431),("smoke","narrow",431),
                      ("pilot","control",431),("pilot","narrow",431)]
    assert len(d["reports"])==dict(narrow=5,parent=1,control=4,none=9)[failure]
    assert d["decision"]==dict(narrow="numerical-gate-stop",parent="reference-safety-stop",
        control="reference-safety-stop",none="single-seed-diagnostic-support-only")[failure]
    assert not any(d[k] for k in ("policy_acceptance","f2_admitted","obstacles_admitted","physical_motion_authorized"))
    assert (output/"manifest.json").is_file()
    with pytest.raises(FileExistsError): exp.campaign("a"*40)
