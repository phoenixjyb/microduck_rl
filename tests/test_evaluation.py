import pytest

from mjlab_microduck.evaluation import (
    aggregate_speed_cases,
    parse_float_list,
    parse_int_list,
)


def _case(speed, observed, error, ends, scale=1.0):
    return {
        "commanded_speed_mps": speed,
        "observed_speed_mean_mps": observed,
        "tracking_error_mean_mps": error,
        "episode_ends": ends,
        "non_timeout_ends": ends,
        "motor_speed_utilization_p99": 0.5 * scale,
        "motor_speed_rated_exceed_fraction": 0.01 * scale,
        "motor_torque_utilization_p99": 0.4 * scale,
        "motor_torque_near_stall_fraction": 0.02 * scale,
        "motor_mechanical_power_abs_mean_w": 3.0 * scale,
        "motor_thermal_load_proxy_mean": 0.2 * scale,
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
