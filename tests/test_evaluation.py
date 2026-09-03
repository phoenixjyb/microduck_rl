import pytest
import torch

from mjlab_microduck.evaluation import (
    aggregate_command_cases,
    aggregate_speed_cases,
    parse_float_list,
    parse_int_list,
    valid_action_deltas,
)


def _case(speed, observed, error, ends, scale=1.0):
    return {
        "commanded_speed_mps": speed,
        "commanded_yaw_rate_rps": 0.3,
        "observed_speed_mean_mps": observed,
        "tracking_error_mean_mps": error,
        "observed_yaw_rate_mean_rps": 0.25,
        "yaw_tracking_error_mean_rps": 0.05,
        "episode_ends": ends,
        "non_timeout_ends": ends,
        "fall_events": 0,
        "nan_termination_events": 0,
        "nonfinite_steps": 0,
        "motor_speed_utilization_p99": 0.5 * scale,
        "motor_speed_rated_exceed_fraction": 0.01 * scale,
        "motor_torque_utilization_p99": 0.4 * scale,
        "motor_torque_near_stall_fraction": 0.02 * scale,
        "motor_mechanical_power_abs_mean_w": 3.0 * scale,
        "motor_thermal_load_proxy_mean": 0.2 * scale,
        "action_abs_p99": 0.8 * scale,
        "action_rate_abs_p99": 0.1 * scale,
    }


def test_parse_lists():
    assert parse_float_list("0.5, 1.0") == (0.5, 1.0)
    assert parse_int_list("41, 42") == (41, 42)


@pytest.mark.parametrize("value", ["", " , ", "nan", "inf"])
def test_parse_float_list_rejects_empty_or_non_finite_values(value):
    with pytest.raises(ValueError):
        parse_float_list(value)


def test_aggregate_speed_cases_groups_seeds_and_preserves_speed_order():
    cases = [
        _case(1.0, 0.8, 0.2, 2, scale=1.0),
        _case(0.5, 0.4, 0.1, 0, scale=1.0),
        _case(1.0, 1.0, 0.1, 3, scale=2.0),
    ]

    aggregates = aggregate_speed_cases(cases)

    assert [item["commanded_speed_mps"] for item in aggregates] == [0.5, 1.0]
    one_mps = aggregates[1]
    assert one_mps["seed_count"] == 2
    assert one_mps["observed_speed_mean_mps"] == pytest.approx(0.9)
    assert one_mps["observed_speed_std_mps"] == pytest.approx(0.1)
    assert one_mps["tracking_error_mean_mps"] == pytest.approx(0.15)
    assert one_mps["episode_ends_total"] == 5
    assert one_mps["non_timeout_ends_total"] == 5
    assert one_mps["motor_speed_utilization_p99_mean"] == pytest.approx(0.75)
    assert one_mps["motor_mechanical_power_abs_mean_w"] == pytest.approx(4.5)
    assert one_mps["action_abs_p99_mean"] == pytest.approx(1.2)
    assert one_mps["action_rate_abs_p99_mean"] == pytest.approx(0.15)


def test_aggregate_command_cases_groups_speed_yaw_and_safety_metrics():
    first = _case(0.3, 0.25, 0.05, 0, scale=1.0)
    second = _case(0.3, 0.27, 0.03, 0, scale=2.0)
    second["fall_events"] = 1
    aggregates = aggregate_command_cases([first, second])

    assert len(aggregates) == 1
    item = aggregates[0]
    assert item["commanded_speed_mps"] == 0.3
    assert item["commanded_yaw_rate_rps"] == 0.3
    assert item["seed_count"] == 2
    assert item["observed_speed_mean_mps"] == pytest.approx(0.26)
    assert item["observed_yaw_rate_mean_rps"] == pytest.approx(0.25)
    assert item["fall_events_total"] == 1
    assert item["motor_torque_utilization_p99_mean"] == pytest.approx(0.6)


def test_valid_action_deltas_excludes_first_action_after_reset():
    current = torch.tensor([[0.5, -0.5], [0.4, 0.6], [0.1, 0.2]])
    previous = torch.tensor([[0.2, -0.1], [0.9, 0.1], [0.0, 0.5]])
    previous_dones = torch.tensor([False, True, False])

    assert torch.allclose(
        valid_action_deltas(current, previous, previous_dones),
        torch.tensor([0.3, 0.4, 0.1, 0.3]),
    )


@pytest.mark.parametrize(
    "current,previous,dones",
    [
        (torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, dtype=torch.bool)),
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3, dtype=torch.bool)),
    ],
)
def test_valid_action_deltas_rejects_shape_mismatch(current, previous, dones):
    with pytest.raises(ValueError):
        valid_action_deltas(current, previous, dones)
