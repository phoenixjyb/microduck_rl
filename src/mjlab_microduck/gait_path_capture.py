"""Opt-in aligned root-path channels, without changing the retained v1 gait trace."""
from __future__ import annotations

import torch

from . import gait_yaw_capture as gait
from . import integrated_turn_foundation as base

PROTOCOL = "microduck-gait-path-trace-v1"


def capture(robot, gait_sample):
    """Reuse the same pre-action gait sample and clone the measured root position."""
    position = robot.data.root_link_pos_w
    base.require(isinstance(position, torch.Tensor) and position.shape == (8, 3)
                 and base.finite_tree(position), "finite eight-world root world-position layout")
    base.require("root_position_w" not in gait_sample, "path sample must extend the original gait sample")
    return {**gait_sample, "root_position_w": position.detach().cpu().clone()}


def trace(samples, checkpoint_sha256):
    base.require(bool(samples) and all("root_position_w" in sample for sample in samples),
                 "aligned path samples required")
    result = gait.trace(samples, checkpoint_sha256)
    result.update(protocol=PROTOCOL, position_frame="world",
                  position_measurement="root-link world position, not integrated velocity")
    return result
