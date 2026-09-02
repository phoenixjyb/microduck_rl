"""Tests for the perception-independent obstacle observation contract."""

import math

import pytest
import torch

from mjlab_microduck.tasks.obstacle_observation import (
    OBSTACLE_OBSERVATION_DIM,
    OBSTACLE_OBSERVATION_FIELDS,
    ObstacleObservationLimits,
    encode_obstacle_observation,
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
