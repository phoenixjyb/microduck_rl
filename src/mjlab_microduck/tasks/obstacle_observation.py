"""Policy-facing contract for externally estimated obstacle geometry.

This module deliberately contains no camera, depth, or detector code.  The
locomotion policy consumes a compact estimate produced elsewhere; simulation
can provide the same fields from ground truth and perturb them later with a
separate sensor model.
"""

from dataclasses import dataclass

import torch


OBSTACLE_OBSERVATION_FIELDS = (
    "range",
    "bearing_sin",
    "bearing_cos",
    "width",
    "height",
    "closing_rate",
    "valid",
)
OBSTACLE_OBSERVATION_DIM = len(OBSTACLE_OBSERVATION_FIELDS)


@dataclass(frozen=True)
class ObstacleObservationLimits:
    """Physical scales used by the v1 obstacle observation contract."""

    max_range_m: float = 2.0
    max_width_m: float = 0.50
    max_height_m: float = 0.25
    max_closing_rate_mps: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.max_range_m,
            self.max_width_m,
            self.max_height_m,
            self.max_closing_rate_mps,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("obstacle observation limits must all be positive")


DEFAULT_OBSTACLE_OBSERVATION_LIMITS = ObstacleObservationLimits()


def encode_obstacle_observation(
    range_m: torch.Tensor,
    bearing_rad: torch.Tensor,
    width_m: torch.Tensor,
    height_m: torch.Tensor,
    closing_rate_mps: torch.Tensor,
    valid: torch.Tensor,
    *,
    limits: ObstacleObservationLimits = DEFAULT_OBSTACLE_OBSERVATION_LIMITS,
) -> torch.Tensor:
    """Encode one nearest-obstacle estimate per environment into the v1 layout.

    Inputs may have any mutually broadcastable batch shape.  Positive closing
    rate means the obstacle and robot are approaching.  Non-finite inputs are
    treated as an invalid estimate.  Invalid rows are all zero, including the
    validity channel, so stale geometry cannot leak into the policy.
    """
    range_m, bearing_rad, width_m, height_m, closing_rate_mps = (
        torch.broadcast_tensors(
            range_m,
            bearing_rad,
            width_m,
            height_m,
            closing_rate_mps,
        )
    )
    valid = torch.broadcast_to(valid.to(device=range_m.device, dtype=torch.bool), range_m.shape)

    finite = (
        torch.isfinite(range_m)
        & torch.isfinite(bearing_rad)
        & torch.isfinite(width_m)
        & torch.isfinite(height_m)
        & torch.isfinite(closing_rate_mps)
    )
    valid = valid & finite

    encoded = torch.stack(
        (
            torch.nan_to_num(range_m / limits.max_range_m).clamp(0.0, 1.0),
            torch.nan_to_num(torch.sin(bearing_rad)),
            torch.nan_to_num(torch.cos(bearing_rad)),
            torch.nan_to_num(width_m / limits.max_width_m).clamp(0.0, 1.0),
            torch.nan_to_num(height_m / limits.max_height_m).clamp(0.0, 1.0),
            torch.nan_to_num(closing_rate_mps / limits.max_closing_rate_mps).clamp(
                -1.0, 1.0
            ),
        ),
        dim=-1,
    )
    encoded = torch.where(valid.unsqueeze(-1), encoded, torch.zeros_like(encoded))
    return torch.cat((encoded, valid.to(dtype=encoded.dtype).unsqueeze(-1)), dim=-1)


def encode_relative_obstacle_observation(
    relative_position_m: torch.Tensor,
    relative_velocity_mps: torch.Tensor,
    width_m: torch.Tensor,
    height_m: torch.Tensor,
    valid: torch.Tensor,
    *,
    limits: ObstacleObservationLimits = DEFAULT_OBSTACLE_OBSERVATION_LIMITS,
) -> torch.Tensor:
    """Encode simulated relative state without introducing perception code.

    Position and velocity are expressed in the robot base frame and need at
    least planar ``x, y`` components.  The simulated surface range uses a
    conservative circular footprint with radius ``width_m / 2``.  Positive
    closing rate means decreasing center distance.

    A scene adapter can call this function after transforming simulator state
    into the base frame.  Noise, latency, dropout, and field-of-view masking
    belong between that adapter and this deterministic geometry encoder.
    """
    if relative_position_m.shape[-1] < 2:
        raise ValueError("relative obstacle position needs x and y components")
    if relative_velocity_mps.shape[-1] < 2:
        raise ValueError("relative obstacle velocity needs x and y components")

    relative_position_xy = relative_position_m[..., :2]
    relative_velocity_xy = relative_velocity_mps[..., :2]
    center_range_m = torch.linalg.vector_norm(relative_position_xy, dim=-1)
    direction = relative_position_xy / center_range_m.clamp_min(1e-8).unsqueeze(-1)
    closing_rate_mps = -(relative_velocity_xy * direction).sum(dim=-1)
    closing_rate_mps = torch.where(
        center_range_m > 1e-8,
        closing_rate_mps,
        torch.zeros_like(closing_rate_mps),
    )
    bearing_rad = torch.atan2(
        relative_position_xy[..., 1], relative_position_xy[..., 0]
    )
    surface_range_m = (center_range_m - width_m / 2.0).clamp_min(0.0)

    return encode_obstacle_observation(
        range_m=surface_range_m,
        bearing_rad=bearing_rad,
        width_m=width_m,
        height_m=height_m,
        closing_rate_mps=closing_rate_mps,
        valid=valid,
        limits=limits,
    )
