"""Tests for the perception-independent obstacle observation contract."""

import math

import pytest
import torch

from mjlab_microduck.tasks.obstacle_observation import (
    OBSTACLE_OBSERVATION_DIM,
    OBSTACLE_OBSERVATION_FIELDS,
    ObstacleObservationLimits,
    ObstacleSensorModel,
    encode_obstacle_observation,
    encode_perturbed_obstacle_observation,
    encode_relative_obstacle_observation,
)


def _encode(**overrides):
    values = {
        "range_m": torch.tensor([1.0]),
        "bearing_rad": torch.tensor([math.pi / 2]),
        "width_m": torch.tensor([0.25]),
        "height_m": torch.tensor([0.125]),
        "closing_rate_mps": torch.tensor([1.0]),
        "valid": torch.tensor([True]),
    }
    values.update(overrides)
    return encode_obstacle_observation(**values)


def test_v1_layout_and_normalization_are_explicit():
    assert OBSTACLE_OBSERVATION_FIELDS == (
        "range",
        "bearing_sin",
        "bearing_cos",
        "width",
        "height",
        "closing_rate",
        "valid",
    )
    assert OBSTACLE_OBSERVATION_DIM == 7
    torch.testing.assert_close(
        _encode(),
        torch.tensor([[0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_bearing_encoding_is_continuous_across_angle_wrap():
    eps = 1e-5
    out = _encode(bearing_rad=torch.tensor([math.pi - eps, -math.pi + eps]))
    torch.testing.assert_close(out[0, 1:3], out[1, 1:3], atol=3e-5, rtol=0.0)


def test_physical_values_are_clamped_to_contract_limits():
    out = _encode(
        range_m=torch.tensor([-0.5, 3.0]),
        bearing_rad=torch.tensor([0.0, 0.0]),
        width_m=torch.tensor([-0.1, 2.0]),
        height_m=torch.tensor([-0.1, 2.0]),
        closing_rate_mps=torch.tensor([-3.0, 3.0]),
    )
    torch.testing.assert_close(out[:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out[:, 3], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out[:, 4], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out[:, 5], torch.tensor([-1.0, 1.0]))


def test_invalid_or_nonfinite_estimates_cannot_leak_stale_geometry():
    invalid = _encode(valid=torch.tensor([False]))
    nonfinite = _encode(range_m=torch.tensor([float("nan")]))
    assert torch.equal(invalid, torch.zeros_like(invalid))
    assert torch.equal(nonfinite, torch.zeros_like(nonfinite))


def test_inputs_broadcast_over_environment_batch():
    out = _encode(
        range_m=torch.tensor([0.5, 1.0, 1.5]),
        valid=torch.tensor(True),
    )
    assert out.shape == (3, OBSTACLE_OBSERVATION_DIM)
    torch.testing.assert_close(out[:, 0], torch.tensor([0.25, 0.5, 0.75]))


def test_limits_reject_zero_or_negative_scales():
    with pytest.raises(ValueError, match="must all be positive"):
        ObstacleObservationLimits(max_range_m=0.0)


def test_relative_state_maps_surface_range_bearing_and_closing_rate():
    out = encode_relative_obstacle_observation(
        relative_position_m=torch.tensor([[1.0, 0.0, 0.1]]),
        relative_velocity_mps=torch.tensor([[-0.5, 0.0, 0.0]]),
        width_m=torch.tensor([0.2]),
        height_m=torch.tensor([0.1]),
        valid=torch.tensor([True]),
    )
    # Surface range is 1.0 - width/2 = 0.9 m.  The obstacle approaches at
    # 0.5 m/s, so the normalized closing-rate channel is +0.25.
    expected = torch.tensor([[0.45, 0.0, 1.0, 0.4, 0.4, 0.25, 1.0]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


def test_relative_state_uses_robot_frame_bearing_and_signed_closing_rate():
    out = encode_relative_obstacle_observation(
        relative_position_m=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        relative_velocity_mps=torch.tensor([[0.0, -0.2], [0.2, 0.0]]),
        width_m=torch.tensor([0.0, 0.0]),
        height_m=torch.tensor([0.1, 0.1]),
        valid=torch.tensor([True, True]),
    )
    # Left-side obstacle: bearing +pi/2, approaching.  Forward obstacle:
    # bearing zero, receding.
    torch.testing.assert_close(out[:, 1], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[:, 2], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[:, 5], torch.tensor([0.1, -0.1]), atol=1e-6, rtol=0.0)


def test_relative_state_clamps_surface_range_inside_obstacle():
    out = encode_relative_obstacle_observation(
        relative_position_m=torch.tensor([[0.05, 0.0]]),
        relative_velocity_mps=torch.zeros(1, 2),
        width_m=torch.tensor([0.2]),
        height_m=torch.tensor([0.1]),
        valid=torch.tensor([True]),
    )
    assert float(out[0, 0]) == 0.0


def test_relative_state_requires_planar_vectors():
    with pytest.raises(ValueError, match="position needs x and y"):
        encode_relative_obstacle_observation(
            relative_position_m=torch.tensor([1.0]),
            relative_velocity_mps=torch.tensor([0.0, 0.0]),
            width_m=torch.tensor(0.1),
            height_m=torch.tensor(0.1),
            valid=torch.tensor(True),
        )


def test_zero_sensor_model_preserves_exact_observation():
    kwargs = {
        "range_m": torch.tensor([1.0]),
        "bearing_rad": torch.tensor([0.25]),
        "width_m": torch.tensor([0.2]),
        "height_m": torch.tensor([0.1]),
        "closing_rate_mps": torch.tensor([0.4]),
        "valid": torch.tensor([True]),
    }
    exact = encode_obstacle_observation(**kwargs)
    perturbed = encode_perturbed_obstacle_observation(
        **kwargs,
        sensor_model=ObstacleSensorModel(),
        uniform_noise_samples=torch.zeros(1, 5),
        dropout_samples=torch.ones(1),
    )
    torch.testing.assert_close(perturbed, exact)


def test_sensor_noise_samples_map_to_documented_physical_bounds():
    out = encode_perturbed_obstacle_observation(
        range_m=torch.tensor([1.0]),
        bearing_rad=torch.tensor([0.0]),
        width_m=torch.tensor([0.2]),
        height_m=torch.tensor([0.1]),
        closing_rate_mps=torch.tensor([0.0]),
        valid=torch.tensor([True]),
        sensor_model=ObstacleSensorModel(
            range_noise_m=0.2,
            bearing_noise_rad=0.1,
            width_noise_m=0.05,
            height_noise_m=0.02,
            closing_rate_noise_mps=0.4,
        ),
        uniform_noise_samples=torch.tensor([[1.0, 0.0, 0.5, 1.0, 0.0]]),
        dropout_samples=torch.ones(1),
    )
    expected = torch.tensor(
        [[0.6, -math.sin(0.1), math.cos(0.1), 0.4, 0.48, -0.2, 1.0]]
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


def test_sensor_dropout_zeros_every_channel():
    out = encode_perturbed_obstacle_observation(
        range_m=torch.tensor([1.0]),
        bearing_rad=torch.tensor([0.0]),
        width_m=torch.tensor([0.2]),
        height_m=torch.tensor([0.1]),
        closing_rate_mps=torch.tensor([0.0]),
        valid=torch.tensor([True]),
        sensor_model=ObstacleSensorModel(dropout_probability=0.2),
        uniform_noise_samples=torch.zeros(1, 5),
        dropout_samples=torch.tensor([0.1]),
    )
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"range_noise_m": -0.1}, "noise bounds"),
        ({"dropout_probability": -0.1}, "dropout_probability"),
        ({"dropout_probability": 1.1}, "dropout_probability"),
    ],
)
def test_sensor_model_rejects_invalid_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ObstacleSensorModel(**kwargs)


def test_sensor_model_rejects_out_of_range_replay_samples():
    with pytest.raises(ValueError, match="uniform_noise_samples"):
        encode_perturbed_obstacle_observation(
            range_m=torch.tensor([1.0]),
            bearing_rad=torch.tensor([0.0]),
            width_m=torch.tensor([0.2]),
            height_m=torch.tensor([0.1]),
            closing_rate_mps=torch.tensor([0.0]),
            valid=torch.tensor([True]),
            sensor_model=ObstacleSensorModel(),
            uniform_noise_samples=torch.tensor([[1.1, 0.0, 0.0, 0.0, 0.0]]),
            dropout_samples=torch.ones(1),
        )
