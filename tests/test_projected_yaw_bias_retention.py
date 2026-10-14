from copy import deepcopy
from mjlab_microduck import projected_yaw_bias_retention as retention
from mjlab_microduck import projected_heading_retention as reference


def test_both_prior_bars_delegate_without_modifying_decision(monkeypatch):
    baseline = {"source": retention.BASELINE_SOURCE}
    expected = dict(protocol="v8-projected-heading-retention-v1",
        decision="walking-idle-not-retained", failures=[["original-cautious","loss"]],
        original_cautious_retention={"decision":"walking-idle-not-retained"},
        policy_acceptance=False, physical_motion_authorized=False)
    calls=[]
    monkeypatch.setattr(reference,"load_baseline",lambda: baseline)
    monkeypatch.setattr(reference,"compare",lambda rows,value: (
        calls.append((rows,value)) or deepcopy(expected)))
    assert retention.load_baseline() is baseline
    report=retention.compare([],baseline)
    assert calls==[([],baseline)]
    assert report=={**expected,"protocol":"v8-projected-yaw-bias-retention-v1"}
    assert expected["protocol"]=="v8-projected-heading-retention-v1"
