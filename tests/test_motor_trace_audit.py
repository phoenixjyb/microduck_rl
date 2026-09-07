"""Adversarial raw-motor timing/accounting tests; no simulator acceptance."""

import copy
import pytest
import torch

from mjlab_microduck.motor_trace_audit import pack_motor_trace, audit_motor_trace, JOINTS
from mjlab_microduck.speed_response_control import summarize
from test_speed_response_control import inputs


def fixture():
    data = inputs()
    # Known single-joint impulse at a known environment/control interval.
    data["pre_force"][175, 2, 3] = .48
    report = summarize(**data)
    report["motor_trace"] = pack_motor_trace(data["pre_force"], data["pre_speed"])
    report["route_trace"] = dict(protocol="initial-route-pre-control-v1",
        position_columns=["route_forward_m", "cross_route_m"],
        velocity_columns=["body_forward_mps", "route_forward_mps", "cross_route_mps", "heading_rad"],
        position=torch.zeros(400, 8, 2).tolist(), velocity=data["velocities"].tolist(),
        last_sample_cross_route_m=[0.]*8, max_abs_cross_route_m=[0.]*8,
        signed_cross_route_velocity_mean_mps=[0.]*8)
    return report


def test_recovers_motor_summary_peak_environment_and_full_window_coverage():
    report = fixture(); before = copy.deepcopy(report)
    result = audit_motor_trace(report)
    assert result["raw_motor_summary_verified"] and result["decision"] == "timing-measurement-only"
    row = result["joints"][JOINTS[3]]
    assert row["peak"]["step"] == 175 and row["peak"]["environment"] == 2
    assert row["peak"]["utilization"] == pytest.approx(.8)
    assert row["slow_samples"] == 0 and row["squared_load_when_route_below_027"] is None
    assert [b["first_step"] for b in row["one_second_bins"]] == [100,150,200,250,300,350]
    assert row["one_second_bins"][1]["squared_utilization_mean"] > row["one_second_bins"][0]["squared_utilization_mean"]
    assert sum(r["settled_squared_load_share"] for r in result["joints"].values()) == pytest.approx(1.)
    assert not result["contact_phase_measured"] and not result["causal_effect_established"]
    assert report == before


@pytest.mark.parametrize("mutation", ["shape", "nan", "unit", "timing", "joint_order", "summary", "joint_summary", "steps"])
def test_rejects_corrupt_raw_or_summary_evidence(mutation):
    r = fixture(); t = r["motor_trace"]
    if mutation == "shape": t["force_nm"].pop()
    if mutation == "nan": t["speed_rad_s"][0][0][0] = float("nan")
    if mutation == "unit": t["units"]["force"] = "mNm"
    if mutation == "timing": t["timing"]["force"] = "post-reset"
    if mutation == "joint_order": t["joint_columns"].reverse()
    if mutation == "summary": r["groups"]["all"]["pre_reset_squared_utilization_mean"] += .01
    if mutation == "joint_summary": r["groups"]["settled"]["pre_reset_joint_p99"][JOINTS[3]] += .01
    if mutation == "steps": r["groups"]["settled"]["steps"] = 299
    with pytest.raises(ValueError): audit_motor_trace(r)


def test_pack_refuses_unbounded_and_nonfinite_samples():
    with pytest.raises(ValueError): pack_motor_trace(torch.zeros(401,8,14), torch.zeros(401,8,14))
    with pytest.raises(ValueError): pack_motor_trace(torch.full((1,8,14),float("inf")), torch.zeros(1,8,14))


def test_no_timing_diagnosis_for_unsafe_report_even_if_raw_history_complete():
    r=fixture(); r["safety_failures"]=["all-legacy-torque"]
    result=audit_motor_trace(r)
    assert result["decision"] == "incomplete-or-unsafe-no-timing-diagnosis"
    assert "joints" not in result and not result["policy_acceptance"]
