"""F1-M contracts, not learned behavior or physical acceptance."""

from dataclasses import asdict
from types import SimpleNamespace as NS
import datetime as dt
import json

import pytest
import torch

from mjlab_microduck import foundation_motor_experiment as exp
from test_foundation_lateral_experiment import good as lateral_good


def good(seed=503):
    report = lateral_good()
    report.update(protocol=exp.PROTOCOL, seed=seed)
    return report


@pytest.mark.parametrize("mode", exp.MODES)
def test_only_motor_weight_and_its_live_curriculum_change(mode):
    a, aa = exp.prepare_config(mode, arm="control")
    b, ba = exp.prepare_config(mode, arm="motor")
    assert a.rewards["motor_torque_load"].weight == -2
    assert b.rewards["motor_torque_load"].weight == -4
    assert a.rewards["lateral_velocity_cost"].weight == b.rewards["lateral_velocity_cost"].weight == -.5
    for cfg, weight in ((a, -2), (b, -4)):
        stage = cfg.curriculum["motor_torque_load_weight"]
        assert stage.params["weight_stages"] == [dict(step=0, weight=weight)]
        for step in (0, 1, exp.PARENT_STEP, exp.PARENT_STEP+240, 216000):
            live = NS(weight=weight)
            env = NS(common_step_counter=step, reward_manager=NS(get_term_cfg=lambda _: live))
            stage.func(env, torch.tensor([0]), **stage.params)
            assert live.weight == weight
    b.rewards["motor_torque_load"].weight = -2
    b.curriculum["motor_torque_load_weight"].params["weight_stages"] = [dict(step=0, weight=-2.)]
    ba.run_name = aa.run_name
    assert asdict(a) == asdict(b) and asdict(aa) == asdict(ba)
    historical, agent = exp.lateral.prepare_config(mode, arm="lateral")
    assert a.seed == aa.seed == (491 if mode == "smoke" else 499)
    a.seed, aa.seed = historical.seed, agent.seed
    aa.experiment_name, aa.run_name = agent.experiment_name, agent.run_name
    assert asdict(a) == asdict(historical) and asdict(aa) == asdict(agent)


def test_all_old_gates_survive_new_protocol_and_motor_tradeoff_is_descriptive():
    p, c, m = good(), good(), good()
    assert not exp.paired_decision(p, c, m)["failures"]
    m["groups"]["settled"]["pre_reset_squared_utilization_mean"] *= 1.06
    result = exp.paired_decision(p, c, m)
    assert "pre_reset_squared_utilization_mean-nonregression" in result["failures"]
    assert "matched-control:pre_reset_squared_utilization_mean-nonregression" in result["failures"]
    assert not result["policy_acceptance"]
    m = good()
    m.update(safety_failures=["all-legacy-torque"], classification="safety-or-coverage-stop")
    for key in ("body_mean_in_band_all_envs", "route_mean_in_band_all_envs", "stable_route_window_all_envs"):
        del m[key]
    assert "all-legacy-torque" in exp.paired_decision(p, c, m)["failures"]
    p["safety_failures"] = ["unsafe-reference"]
    with pytest.raises(ValueError): exp.paired_decision(p, c, m)


def test_child_idle_gate_recorded_and_deadline_refuses_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "DEADLINE", dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    order = []
    monkeypatch.setattr(exp, "wait_idle", lambda: order.append("idle") or {"samples": [0, 0]})
    monkeypatch.setattr(exp.subprocess, "run", lambda *a, **k: order.append((a, k)) or NS(returncode=0))
    exp.run_child("pilot", "motor", "a"*40, exp.time.monotonic()+1100)
    assert order[0] == "idle" and order[1][1]["timeout"] == 900
    assert json.loads((tmp_path / "motor-pilot-idle.json").read_text())["samples"] == [0, 0]
    monkeypatch.setattr(exp, "DEADLINE", dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=900))
    with pytest.raises(ValueError): exp.run_child("pilot", "control", "a"*40, exp.time.monotonic()+1100)
    assert len(order) == 2


@pytest.mark.parametrize("failure", [None, "parent", "control", "motor", "runtime", "initial", "smoke"])
def test_order_first_failure_no_retry_and_immutable_manifest(tmp_path, monkeypatch, failure):
    output = tmp_path / "campaign"
    monkeypatch.setattr(exp, "OUTPUT", output)
    monkeypatch.setattr(exp, "LAST_START", dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(exp, "verify_source", lambda s: None)
    monkeypatch.setattr(exp.training, "runtime_identity", lambda: {})
    monkeypatch.setattr(exp, "wait_idle", lambda: {})
    monkeypatch.setattr(exp, "preserved", lambda: {"unchanged": True})
    monkeypatch.setattr(exp, "checkpoint", lambda arm, source: (None, exp.MODEL_SHA))
    calls = []
    def child(kind, arm, source, end, *, seed=503):
        calls.append((kind, arm, seed))
        if failure == "runtime": raise ValueError("runtime")
        if kind in exp.MODES:
            path = output / f"{arm}-{kind}"; path.mkdir()
            (path / "initial.pt").write_bytes(b"different" if failure == "initial" and arm == "motor" else b"same")
            exp.write_new(path / "result.json", dict(status="training-complete-not-accepted", source=source,
                updates=9 if failure == "smoke" else 10, wall_seconds=1, common_step=exp.PARENT_STEP+240))
        else:
            r = good(seed)
            if arm == failure:
                if failure == "motor": r["groups"]["settled"]["pre_reset_joint_p99"]["left_knee"] += .03
                else: r.update(safety_failures=["terminal-including-startup"], classification="safety-or-coverage-stop")
            exp.write_new(output / f"{arm}-s{seed}.json", r)
    monkeypatch.setattr(exp, "run_child", child)
    if failure in ("runtime", "initial", "smoke"):
        with pytest.raises(ValueError): exp.campaign("a"*40)
    else: exp.campaign("a"*40)
    decision = json.loads((output / "decision.json").read_text())
    expected = {None: "single-seed-diagnostic-support-only", "parent": "reference-safety-stop",
        "control": "reference-safety-stop", "motor": "numerical-gate-stop", "runtime": "runtime-failure-stop",
        "initial": "runtime-failure-stop", "smoke": "runtime-failure-stop"}
    assert decision["decision"] == expected[failure] and not decision["policy_acceptance"]
    assert calls[0] == ("evaluate", "parent", 503)
    if failure == "parent": assert len(calls) == 1
    if failure in ("initial", "smoke"): assert all(c[0] != "pilot" for c in calls)
    if failure == "control": assert calls[-1] == ("evaluate", "control", 503)
    if failure == "motor": assert calls[-1] == ("evaluate", "motor", 503)
    if failure is None: assert len(calls) == 13 and len(decision["reports"]) == 9
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {str(p.relative_to(output)) for p in output.rglob("*")
                                     if p.is_file() and p.name != "manifest.json"}
    for name, row in manifest["files"].items():
        assert exp.sha256(output / name) == row["sha256"]
        assert (output / name).stat().st_size == row["bytes"]
    with pytest.raises(FileExistsError): exp.campaign("a"*40)


def test_campaign_refuses_late_start_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "verify_source", lambda s: None)
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "uncreated")
    monkeypatch.setattr(exp, "LAST_START", dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    with pytest.raises(ValueError): exp.campaign("a"*40)
    assert not exp.OUTPUT.exists()
