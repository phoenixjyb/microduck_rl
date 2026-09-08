"""Synthetic trace arithmetic and fail-closed coverage, never policy rollout."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from mjlab_microduck import foundation_command_map as exp
from mjlab_microduck import speed_response_control as historical


def trace(speed=.1, steps=400):
    velocity = torch.zeros((steps, 8, 4), dtype=torch.float64)
    velocity[:, :, :2] = speed
    command = torch.zeros((steps, 8, 3), dtype=torch.float64)
    command[:, :, 0] = speed
    motors = torch.full((steps, 8, 14), .12, dtype=torch.float64)
    position = torch.zeros((steps, 8, 2), dtype=torch.float64)
    position[:, :, 0] = torch.arange(steps)[:, None]*.02*speed
    return exp.Trace(velocity, position, command, command.clone(), motors,
                     torch.ones_like(motors), motors.clone(), torch.ones_like(motors),
                     torch.zeros((steps, 8), dtype=torch.bool), tuple(exp.JOINTS))


def cell(speed=.1):
    return exp.Cell(503, speed, "original")


@pytest.mark.parametrize("speed", exp.SPEEDS)
def test_each_bin_scores_its_actual_command_and_grants_no_authority(speed):
    t = trace(speed)
    before = t.velocity.clone()
    rng = torch.get_rng_state().clone()
    report = exp.score(cell(speed), t)
    assert report["classification"] == "descriptive-cell-within-checks"
    assert not report["safety_failures"] and not report["performance_failures"]
    assert report["stable_route_response"]["nominal_speed_mps"] == speed
    assert report["stable_route_response"]["environments"][0]["stable_recovery_latency_s"] == .5
    assert all(report[key] is False for key in exp._no_authority())
    assert torch.equal(t.velocity, before) and torch.equal(torch.get_rng_state(), rng)


def test_old_point_three_contract_is_not_mutated():
    p = Path(historical.__file__)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    exp.score(cell(.2), trace(.2))
    assert historical.SPEED == .3 and historical.PROTOCOL == "frozen-straight-speed-s383-v1"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


def test_shared_point_three_statistics_match_unchanged_historical_arithmetic():
    t = trace(.3)
    generator = torch.Generator().manual_seed(907)
    t.pre_force.copy_(torch.rand(t.pre_force.shape, generator=generator)*.3)
    t.legacy_force.copy_(t.pre_force)
    r = exp.score(cell(.3), t)
    old = historical.summarize(t.velocity,t.issued,t.legacy_force,t.legacy_speed,
                               t.pre_force,t.pre_speed,[],r['stable_route_response'])
    for group in ('all','settled'):
        for key in set(r['groups'][group]) & set(old['groups'][group]):
            assert r['groups'][group][key] == old['groups'][group][key], key
    assert old['speed_mps'] == .3
    assert old['classification'] == 'straight-response-within-both-criteria'


@pytest.mark.parametrize("args", [(True,.1,"original"),(503,0.,"original"),
                                  (503,.15,"original"),(503,float('nan'),"original"),
                                  (503,.1,"yaw"),(999,.1,"original")])
def test_reject_undeclared_cells(args):
    with pytest.raises(ValueError): exp.Cell(*args)


@pytest.mark.parametrize("field", ["velocity", "position", "issued", "consumed",
                                   "legacy_force", "legacy_speed", "pre_force", "pre_speed"])
def test_reject_nonfinite_anywhere_including_startup(field):
    t = trace(); getattr(t, field)[0, 0, 0] = float('nan')
    with pytest.raises(ValueError): exp.score(cell(), t)


def test_invalid_shapes_dtypes_joint_order_and_overflow_fail_closed():
    t = trace()
    for changed in [replace(t, position=t.position[:-1]), replace(t, issued=t.issued.int()),
                    replace(t, dones=t.dones.double()), replace(t, joint_names=tuple(reversed(t.joint_names))),
                    replace(t, pre_force=t.pre_force*1e200)]:
        with pytest.raises(ValueError): exp.score(cell(), changed)


def test_one_slow_environment_cannot_hide_in_pooled_mean():
    t = trace(.3); t.velocity[:, 0, :2] = .25
    assert abs(t.velocity[:, :, 0].mean().item()-.3) < .03
    report = exp.score(cell(.3), t)
    assert report["classification"] == "descriptive-performance-miss"
    assert "body_forward_per_env_mean-outside-band" in report["performance_failures"]
    assert "route_forward_per_env_mean-outside-band" in report["performance_failures"]
    assert "stable-speed-window-missed" in report["performance_failures"]


def test_band_boundary_and_exact_sampled_stability_window():
    t = trace(.1); t.velocity[:, :, :2] = .13
    assert not exp.score(cell(),t)["performance_failures"]
    t = trace(.1); t.velocity[100:176,:,1] = 0.
    assert exp.score(cell(),t)["stable_route_response"]["counts"]["window-missed"] == 8
    t.velocity[175,:,1] = .1  # Samples175..200 span .50s and finish at2.00s.
    assert exp.score(cell(),t)["stable_route_response"]["counts"]["recovered-in-window"] == 8


def test_terminal_stops_all_environments_and_never_accepts_post_reset_samples():
    t = trace(steps=101); t.dones[-1, 2] = True
    r = exp.score(cell(),t)
    assert r["classification"] == "safety-or-coverage-stop"
    assert "terminal-including-startup" in r["safety_failures"]
    assert "incomplete-control" in r["safety_failures"]
    t = trace(); t.dones[20, 2] = True
    with pytest.raises(ValueError,match="after first terminal"): exp.score(cell(),t)


def test_terminal_at_final_step_is_still_failure_and_short_startup_is_censored():
    t = trace(); t.dones[-1,0] = True
    assert "terminal-including-startup" in exp.score(cell(),t)["safety_failures"]
    r = exp.score(cell(),trace(steps=60))
    assert "settled" not in r["groups"]
    assert r["stable_route_response"]["counts"]["not-observed"] == 8
    assert r["classification"] == "safety-or-coverage-stop"


@pytest.mark.parametrize("field,label", [("legacy_force","legacy-torque"),
                                        ("legacy_speed","legacy-rated-speed"),
                                        ("pre_speed","pre-reset-rated-speed")])
def test_motor_failures_include_startup_and_named_joint_metrics(field,label):
    t = trace(); getattr(t,field)[:100] = 20.
    r = exp.score(cell(),t)
    assert "all-"+label in r["safety_failures"]
    assert "settled-"+label not in r["safety_failures"]
    assert set(r["groups"]["all"]["pre_reset_joint_p99"]) == set(exp.JOINTS)


def test_heading_controller_rule_delivery_and_lateral_motion_are_distinct():
    t = trace(); t.velocity[:,:,3] = .3
    # Controller slews -.02 at a time to -.30.
    t.issued[:,:,2] = -torch.arange(1,401).double().clamp(max=15)[:,None]*.02
    t.consumed.copy_(t.issued)
    t.velocity[:,:,2] = .06
    r = exp.score(cell(),t)
    assert not r["safety_failures"]
    assert {"heading-drift","cross-route-motion"} <= set(r["performance_failures"])
    t.consumed[0,0,2] = 0.
    assert "issued-consumed-mismatch" in exp.score(cell(),t)["safety_failures"]
    t.consumed.copy_(t.issued); t.issued[:,:,0] = .3; t.consumed.copy_(t.issued)
    assert "heading-or-speed-command-mismatch" in exp.score(cell(),t)["safety_failures"]


def test_exact_schedule_prefix_and_continue_performance_not_safety_failure():
    planned = exp.schedule()
    assert len(planned) == len(set(planned)) == 18
    assert planned[:3] == (cell(), exp.Cell(503,.1,"narrow"), cell(.2))
    first = trace(); first.velocity[:,:,0] = 0.
    prefix = [(planned[0],first),(planned[1],trace())]
    r = exp.summarize_prefix(prefix)
    assert r["decision"] == "incomplete-map" and len(r["unexecuted"]) == 16
    first.dones[-1,0] = True
    assert exp.summarize_prefix(prefix[:1])["decision"] == "safety-or-coverage-stop"
    with pytest.raises(ValueError,match="after safety stop"): exp.summarize_prefix(prefix)
    with pytest.raises(ValueError,match="order"): exp.summarize_prefix(list(reversed(prefix)))
    with pytest.raises(ValueError): exp.summarize_prefix([])


def test_complete_map_remains_descriptive_not_admission():
    r = exp.summarize_prefix([(c,trace(c.speed_mps)) for c in exp.schedule()])
    assert r["decision"] == "complete-descriptive-map" and not r["unexecuted"]
    assert r["development_seeds"] == [503,509,521]
    assert all(r[key] is False for key in exp._no_authority())
