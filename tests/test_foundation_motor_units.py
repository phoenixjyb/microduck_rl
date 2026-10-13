"""Lock the legacy utilization gate while making raw force units explicit."""
import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as foundation


def accepted_rows(torque_p99):
    return [dict(seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240,
                 complete=True, terminal_worlds=[], torque_p99=torque_p99,
                 rated_speed_exceed_fraction=0., speed_mae=[.02]*8, yaw_mae=[.08]*8)
            for seed in foundation.SEEDS for speed, yaw in foundation.CASES]


def test_motor_force_summary_preserves_utilization_and_adds_raw_nm():
    force_nm = torch.tensor([-.12, .24, -.36, .30], dtype=torch.float64)
    summary = foundation.motor_force_summary(force_nm)

    # torch.quantile's linear interpolation at .99 over abs forces is .3582 Nm.
    assert summary["abs_force_p99_nm"] == pytest.approx(.3582)
    assert summary["torque_p99_reference_nm"] == .60
    assert summary["torque_p99"] == pytest.approx(.597)
    assert summary["torque_p99"] * summary["torque_p99_reference_nm"] == pytest.approx(
        summary["abs_force_p99_nm"])


def test_legacy_motor_gate_remains_point_six_utilization():
    # At exactly .60 utilization, the existing gate still passes; the next
    # represented value above it fails. No Nm reinterpretation is introduced.
    assert foundation.decision(accepted_rows(.60))["decision"] == "foundation-ready-for-next-experiment"
    result = foundation.decision(accepted_rows(.60001))
    assert result["decision"] == "foundation-not-ready"
    assert result["failures"][0][-1] == "motor-envelope"

