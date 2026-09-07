"""Privileged simulation reward, independent of actor observations or commands."""

import torch
from mjlab_microduck.first_attempt_smoke import require


def lateral_velocity_cost(env):
    """Squared yaw-aligned lateral speed / (0.1m/s)^2; never sensor filtering."""
    data = env.scene["robot"].data
    q, v = data.root_link_quat_w, data.root_link_lin_vel_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2].square()+q[:,3].square()))
    lateral = -v[:,0]*yaw.sin()+v[:,1]*yaw.cos()
    # Refuse corruption before the installed manager can sanitize a NaN reward.
    cost = (lateral/.1).square()
    require(bool(torch.isfinite(cost).all()), "finite raw lateral reward")
    return cost
