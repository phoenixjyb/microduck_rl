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
