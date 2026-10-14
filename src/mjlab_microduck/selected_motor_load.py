"""Strict motor-load reward for an explicit set of named actuators."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def selected_motor_torque_load_cost(
    env: ManagerBasedRlEnv,
    actuator_names: tuple[str, ...],
    rated_stall_torque_nm: float = 0.60,
    soft_limit_fraction: float = 0.70,
    over_limit_gain: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a strict, per-environment load cost for selected actuators.

    The cost is the mean of ``u**2 + over_limit_gain * relu(u-soft)**2``,
    where ``u = abs(actuator_force) / rated_stall_torque_nm``. Selector names
    are resolved with the entity API and then checked against the exact request
    so regex-like matches cannot silently widen the selected set.

    Invalid configuration, selection, or force data raises ``ValueError``.
    This function does not alter the environment, actuator state, or force data.
    """
    rated = _finite_scalar(rated_stall_torque_nm, "rated_stall_torque_nm")
    soft = _finite_scalar(soft_limit_fraction, "soft_limit_fraction")
    gain = _finite_scalar(over_limit_gain, "over_limit_gain")
    if rated <= 0.0:
        raise ValueError("rated_stall_torque_nm must be positive")
    if not 0.0 < soft < 1.0:
        raise ValueError("soft_limit_fraction must be between 0 and 1")
    if gain < 0.0:
        raise ValueError("over_limit_gain must be non-negative")
    if not isinstance(actuator_names, tuple) or not actuator_names:
        raise ValueError("actuator_names must be a non-empty tuple of exact names")
    if any(not isinstance(name, str) or not name for name in actuator_names):
        raise ValueError("actuator_names must contain non-empty strings")
    if len(set(actuator_names)) != len(actuator_names):
        raise ValueError("actuator_names must be unique")

    asset = env.scene[asset_cfg.name]
    try:
        actuator_ids, resolved_names = asset.find_actuators(
            actuator_names, preserve_order=True
        )
    except Exception as exc:
        raise ValueError("could not resolve requested actuator names") from exc

    try:
        actuator_ids = tuple(actuator_ids)
        resolved_names = tuple(resolved_names)
    except TypeError as exc:
        raise ValueError("actuator selector returned malformed results") from exc
    if (
        len(actuator_ids) != len(actuator_names)
        or len(resolved_names) != len(actuator_names)
    ):
        raise ValueError("resolved actuator index count does not match requested names")
    if resolved_names != actuator_names:
        raise ValueError(
            "resolved actuator names must exactly match actuator_names in order"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in actuator_ids
    ):
        raise ValueError("resolved actuator indices must be integers")
    actuator_ids = tuple(int(index) for index in actuator_ids)
    if len(set(actuator_ids)) != len(actuator_ids):
        raise ValueError("resolved actuator indices must be unique")

    actuator_force = asset.data.actuator_force
    if not isinstance(actuator_force, torch.Tensor) or actuator_force.ndim != 2:
        raise ValueError("actuator_force must be a rank-2 tensor")
    num_envs = getattr(env, "num_envs", None)
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or actuator_force.shape[0] != num_envs
    ):
        raise ValueError("actuator_force must have shape (num_envs, num_actuators)")
    if actuator_force.shape[1] == 0:
        raise ValueError("actuator_force must contain actuator columns")
    if any(index < 0 or index >= actuator_force.shape[1] for index in actuator_ids):
        raise ValueError("resolved actuator index is out of bounds")
    if not actuator_force.is_floating_point():
        raise ValueError("actuator_force must use a floating-point dtype")
    if not bool(torch.isfinite(actuator_force).all()):
        raise ValueError("actuator_force contains non-finite values")

    selected_force = actuator_force[:, actuator_ids]
    utilization = selected_force.abs() / rated
    overload = torch.relu(utilization - soft)
    cost = (utilization.square() + gain * overload.square()).mean(dim=1)
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("selected actuator load cost is non-finite")
    return cost
