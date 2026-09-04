import pytest
import torch

from mjlab_microduck.hop_evaluation import h1_decision, summarize_hop_trace


def _trace(*, fall_at=None, drift=0.0, bottomed=False):
    steps_per_cycle = 4
    cycles = 2
    steps = steps_per_cycle * cycles
    num_envs = 2
    z = torch.full((steps + 1, num_envs), 0.15)
    xy = torch.zeros((steps + 1, num_envs, 2))
    xy[:, :, 0] = torch.linspace(0.0, drift, steps + 1)[:, None]
    airborne = torch.zeros((steps + 1, num_envs), dtype=torch.bool)
    # Env 0 performs one 2 cm hop and lands in each cycle. Env 1 tucks its feet
    # without raising its body and must not count as a hop.
    for start in (0, 4):
        airborne[start + 2 : start + 4] = True
        z[start + 3, 0] = 0.17
    dones = torch.zeros((steps, num_envs), dtype=torch.bool)
    falls = torch.zeros_like(dones)
    if fall_at is not None:
        dones[fall_at, 0] = True
        falls[fall_at, 0] = True
        # Simulate an excellent automatic-reset episode afterward. It must be ignored.
        z[fall_at + 2 :, 0] = 0.25
    finite = torch.ones_like(dones)
    compression = torch.zeros((steps, num_envs, 2))
    if bottomed:
        compression[..., 0] = 0.95
    force = torch.ones((steps, num_envs, 2)) * 5.0
    return summarize_hop_trace(
        base_z=z,
        base_xy=xy,
        both_airborne=airborne,
        dones=dones,
        falls=falls,
        nan_terminations=torch.zeros_like(dones),
        finite=finite,
        spring_compression_ratio=compression,
        landing_force=force,
        steps_per_cycle=steps_per_cycle,
        cycles=cycles,
    )


def test_trace_counts_body_rise_and_landing_but_rejects_stationary_tuck():
    result = _trace()
    assert result["planned_cycles"] == 4
    assert result["qualified_cycles"] == 2
    assert result["landed_qualified_cycles"] == 2
    assert result["cycle_success_fraction"] == pytest.approx(0.5)
    assert result["landing_fraction"] == 1.0
    assert result["episode_passes"] == 1
    assert result["cycle_rise_peak_m"] == pytest.approx(0.02)


def test_trace_ignores_automatic_reset_after_first_done():
    result = _trace(fall_at=2)
    assert result["fall_events"] == 1
    assert result["completed_cycles"] == 2  # only env 1 completes both cycles
    assert result["cycle_rise_peak_m"] < 0.1
    assert result["episode_passes"] == 0


def test_trace_reports_drift_and_bottoming():
    result = _trace(drift=0.2, bottomed=True)
    assert result["max_drift_peak_m"] == pytest.approx(0.2)
    assert result["episode_passes"] == 0
    assert result["spring_bottomed_fraction"] == 0.5


def _passing_case():
    return {
        "cycle_success_fraction": 0.95,
        "landing_fraction": 0.99,
        "episode_pass_fraction": 0.85,
        "cycle_rise_p50_m": 0.03,
        "fall_events": 0,
        "nan_termination_events": 0,
        "nonfinite_episodes": 0,
        "cycle_rise_peak_m": 0.08,
        "max_drift_p95_m": 0.08,
        "spring_bottomed_fraction": 0.005,
        "motor_speed_rated_exceed_fraction": 0.0,
        "motor_torque_utilization_p99": 0.8,
        "motor_torque_near_stall_fraction": 0.001,
    }


def test_h1_decision_accepts_only_when_every_seed_passes_every_gate():
    assert h1_decision([_passing_case(), _passing_case()])["decision"] == "accepted"
    failed = _passing_case()
    failed["fall_events"] = 1
    decision = h1_decision([_passing_case(), failed])
    assert decision["decision"] == "rejected"
    assert next(g for g in decision["gates"] if g["name"] == "falls")["status"] == "fail"
    assert decision["physical_motion_authorized"] is False
