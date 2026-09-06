"""Bounded high-level command teacher for obstacle negotiation.

The controller in this module never emits joint targets.  It produces only a
forward-speed command and a yaw-rate command for an independently loaded,
frozen locomotion actor.  Its obstacle input is the documented seven-channel
external geometry contract; no camera or simulator-only identifier is part of
the policy-facing observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from mjlab_microduck.recovery_control import RecoveryAccelerationCfg, cap_recovery_acceleration

from mjlab_microduck.tasks.obstacle_observation import (
    DEFAULT_OBSTACLE_OBSERVATION_LIMITS,
    OBSTACLE_OBSERVATION_DIM,
)


class ObstaclePhase(IntEnum):
    APPROACH = 0
    INTERACTION = 1
    RECOVERY = 2


SUPERVISOR_PHASE_DIM = len(ObstaclePhase)
SUPERVISOR_OBSERVATION_DIM = 17


@dataclass(frozen=True)
class ObstacleTeacherCfg:
    """Physical limits and deterministic HC1 teacher gains."""

    max_forward_speed_mps: float = 0.8
    max_yaw_rate_rps: float = 0.6
    interaction_entry_m: float = 0.90
    interaction_time_to_contact_s: float = 3.0
    passed_margin_m: float = 0.0
    bypass_clearance_m: float = 0.42
    bypass_lookahead_m: float = 0.30
    route_lookahead_m: float = 0.60
    interaction_speed_scale: float = 1.0
    min_interaction_speed_mps: float = 0.30
    max_interaction_speed_mps: float = 0.30
    yaw_gain: float = 2.0
    max_speed_delta_per_update_mps: float = 0.08
    max_yaw_delta_per_update_rps: float = 0.20
    centered_deadband_m: float = 0.03

    def __post_init__(self) -> None:
        positive = (
            self.max_forward_speed_mps,
            self.max_yaw_rate_rps,
            self.interaction_entry_m,
            self.interaction_time_to_contact_s,
            self.bypass_clearance_m,
            self.bypass_lookahead_m,
            self.route_lookahead_m,
            self.interaction_speed_scale,
            self.min_interaction_speed_mps,
            self.max_interaction_speed_mps,
            self.yaw_gain,
            self.max_speed_delta_per_update_mps,
            self.max_yaw_delta_per_update_rps,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("teacher limits and gains must be positive")
        if self.min_interaction_speed_mps > self.max_interaction_speed_mps:
            raise ValueError("interaction speed bounds must be ordered")
        if self.max_interaction_speed_mps > self.max_forward_speed_mps:
            raise ValueError("interaction speed exceeds the forward command limit")
        if self.centered_deadband_m < 0.0:
            raise ValueError("centered deadband must be non-negative")
        if self.passed_margin_m < 0.0:
            raise ValueError("passed margin must be non-negative")


@dataclass
class ObstacleTeacherState:
    """Per-environment maneuver state retained by the execution layer."""

    phase: torch.Tensor
    bypass_side: torch.Tensor
    preferred_side: torch.Tensor
    previous_command: torch.Tensor


def clone_teacher_state(state: ObstacleTeacherState) -> ObstacleTeacherState:
    """Clone mutable supervisor state for side-effect-free counterfactual labels."""
    return ObstacleTeacherState(
        phase=state.phase.clone(),
        bypass_side=state.bypass_side.clone(),
        preferred_side=state.preferred_side.clone(),
        previous_command=state.previous_command.clone(),
    )


def make_teacher_state(
    num_envs: int,
    *,
    device: torch.device | str,
    nominal_speed_mps: float | torch.Tensor,
) -> ObstacleTeacherState:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    nominal = torch.as_tensor(nominal_speed_mps, device=device, dtype=torch.float32)
    if nominal.ndim == 0:
        nominal = nominal.expand(num_envs)
    if nominal.shape != (num_envs,):
        raise ValueError("nominal_speed_mps must be scalar or have shape (N,)")
    if bool((nominal < 0.0).any()):
        raise ValueError("nominal_speed_mps must be non-negative")
    preferred_side = torch.ones(num_envs, device=device)
    preferred_side[1::2] = -1.0
    previous_command = torch.zeros(num_envs, 2, device=device)
    previous_command[:, 0] = nominal
    return ObstacleTeacherState(
        phase=torch.full(
            (num_envs,), int(ObstaclePhase.APPROACH), dtype=torch.long, device=device
        ),
        bypass_side=preferred_side.clone(),
        preferred_side=preferred_side,
        previous_command=previous_command,
    )


def reset_teacher_state(
    state: ObstacleTeacherState,
    reset_mask: torch.Tensor,
    *,
    nominal_speed_mps: float | torch.Tensor,
) -> None:
    reset_mask = reset_mask.to(device=state.phase.device, dtype=torch.bool)
    nominal = torch.as_tensor(
        nominal_speed_mps,
        device=state.phase.device,
        dtype=state.previous_command.dtype,
    )
    if nominal.ndim == 0:
        nominal = nominal.expand(state.phase.shape[0])
    if nominal.shape != state.phase.shape:
        raise ValueError("nominal_speed_mps must be scalar or match teacher state")
    if bool((nominal < 0.0).any()):
        raise ValueError("nominal_speed_mps must be non-negative")
    state.phase[reset_mask] = int(ObstaclePhase.APPROACH)
    state.bypass_side[reset_mask] = state.preferred_side[reset_mask]
    state.previous_command[reset_mask, 0] = nominal[reset_mask]
    state.previous_command[reset_mask, 1] = 0.0


def _decoded_obstacle_geometry(
    obstacle_observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if obstacle_observation.ndim != 2 or obstacle_observation.shape[1] != OBSTACLE_OBSERVATION_DIM:
        raise ValueError(
            f"obstacle_observation must have shape (N, {OBSTACLE_OBSERVATION_DIM})"
        )
    limits = DEFAULT_OBSTACLE_OBSERVATION_LIMITS
    surface_range = obstacle_observation[:, 0] * limits.max_range_m
    bearing_sin = obstacle_observation[:, 1]
    bearing_cos = obstacle_observation[:, 2]
    width = obstacle_observation[:, 3] * limits.max_width_m
    center_range = surface_range + width / 2.0
    relative_x = center_range * bearing_cos
    relative_y = center_range * bearing_sin
    closing_rate = (
        obstacle_observation[:, 5]
        * DEFAULT_OBSTACLE_OBSERVATION_LIMITS.max_closing_rate_mps
    )
    valid = obstacle_observation[:, 6] > 0.5
    return relative_x, relative_y, width, closing_rate, valid


def _clamp_delta(
    target: torch.Tensor, previous: torch.Tensor, maximum_delta: float
) -> torch.Tensor:
    return previous + (target - previous).clamp(-maximum_delta, maximum_delta)


def advance_obstacle_state(
    obstacle_observation: torch.Tensor,
    route_lateral_error_m: torch.Tensor,
    route_heading_error_rad: torch.Tensor,
    state: ObstacleTeacherState,
    *,
    cfg: ObstacleTeacherCfg = ObstacleTeacherCfg(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance route-controller phase/side state without choosing a command."""
    relative_x, relative_y, _, closing_rate, valid = _decoded_obstacle_geometry(
        obstacle_observation
    )
    cos_heading = torch.cos(route_heading_error_rad)
    sin_heading = torch.sin(route_heading_error_rad)
    obstacle_route_x = cos_heading * relative_x - sin_heading * relative_y
    obstacle_route_y = (
        route_lateral_error_m
        + sin_heading * relative_x
        + cos_heading * relative_y
    )

    approach = state.phase == int(ObstaclePhase.APPROACH)
    surface_range = obstacle_observation[:, 0] * (
        DEFAULT_OBSTACLE_OBSERVATION_LIMITS.max_range_m
    )
    time_to_contact = surface_range / closing_rate.clamp_min(0.05)
    imminent_contact = (closing_rate > 0.05) & (
        time_to_contact <= cfg.interaction_time_to_contact_s
    )
    enter_interaction = approach & valid & (
        (obstacle_route_x <= cfg.interaction_entry_m) | imminent_contact
    )
    obstacle_left = relative_y > cfg.centered_deadband_m
    obstacle_right = relative_y < -cfg.centered_deadband_m
    selected_side = torch.where(
        obstacle_left,
        -torch.ones_like(relative_y),
        torch.where(obstacle_right, torch.ones_like(relative_y), state.preferred_side),
    )
    state.bypass_side[enter_interaction] = selected_side[enter_interaction]
    state.phase[enter_interaction] = int(ObstaclePhase.INTERACTION)

    interaction = state.phase == int(ObstaclePhase.INTERACTION)
    enter_recovery = interaction & valid & (
        obstacle_route_x <= -cfg.passed_margin_m
    )
    state.phase[enter_recovery] = int(ObstaclePhase.RECOVERY)
    return relative_x, relative_y, obstacle_route_x, obstacle_route_y, valid


def apply_bounded_supervisor_command(
    desired_command: torch.Tensor,
    obstacle_observation: torch.Tensor,
    state: ObstacleTeacherState,
    *,
    cfg: ObstacleTeacherCfg = ObstacleTeacherCfg(),
    recovery_cfg: RecoveryAccelerationCfg | None = None,
    update_dt_s: float | None = None,
) -> torch.Tensor:
    """Apply execution-layer limits and fail-safe behavior to any supervisor."""
    if desired_command.shape != state.previous_command.shape:
        raise ValueError("desired supervisor command must have shape (N, 2)")
    if recovery_cfg is not None and not bool(torch.isfinite(desired_command).all()):
        raise ValueError("nonfinite recovery supervisor command")
    desired_speed = desired_command[:, 0].clamp(0.0, cfg.max_forward_speed_mps)
    desired_yaw = desired_command[:, 1].clamp(
        -cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps
    )
    command = torch.stack(
        (
            _clamp_delta(
                desired_speed,
                state.previous_command[:, 0],
                cfg.max_speed_delta_per_update_mps,
            ),
            _clamp_delta(
                desired_yaw,
                state.previous_command[:, 1],
                cfg.max_yaw_delta_per_update_rps,
            ),
        ),
        dim=-1,
    )
    valid = obstacle_observation[:, 6] > 0.5
    unsafe_invalid = (state.phase != int(ObstaclePhase.RECOVERY)) & ~valid
    if recovery_cfg is not None:
        command = cap_recovery_acceleration(
            command, state.previous_command, state.phase,
            cfg=recovery_cfg, update_dt_s=update_dt_s,
        )
    command[unsafe_invalid] = 0.0
    state.previous_command.copy_(command)
    return command


def teacher_command(
    obstacle_observation: torch.Tensor,
    nominal_speed_mps: torch.Tensor,
    route_lateral_error_m: torch.Tensor,
    route_heading_error_rad: torch.Tensor,
    state: ObstacleTeacherState,
    *,
    cfg: ObstacleTeacherCfg = ObstacleTeacherCfg(),
    recovery_cfg: RecoveryAccelerationCfg | None = None,
    update_dt_s: float | None = None,
) -> torch.Tensor:
    """Advance the HC1 teacher and return ``[forward_mps, yaw_rps]``.

    Speed tracking is preserved in approach and recovery.  Only interaction
    may deliberately slow down.  Invalid geometry stops immediately in the
    approach/interaction phases; recovery remains route-state driven after a
    positively observed pass.
    """
    num_envs = obstacle_observation.shape[0]
    one_d_inputs = (
        nominal_speed_mps,
        route_lateral_error_m,
        route_heading_error_rad,
    )
    if any(value.shape != (num_envs,) for value in one_d_inputs):
        raise ValueError("teacher scalar inputs must each have shape (N,)")
    if state.phase.shape != (num_envs,) or state.previous_command.shape != (num_envs, 2):
        raise ValueError("teacher state shape does not match observation batch")

    _, _, obstacle_route_x, obstacle_route_y, _ = advance_obstacle_state(
        obstacle_observation,
        route_lateral_error_m,
        route_heading_error_rad,
        state,
        cfg=cfg,
    )

    phase = state.phase
    nominal = nominal_speed_mps.clamp(0.0, cfg.max_forward_speed_mps)

    bypass_target_y = obstacle_route_y + state.bypass_side * cfg.bypass_clearance_m
    bypass_target_x = (obstacle_route_x + cfg.bypass_lookahead_m).clamp_min(0.05)
    bypass_heading_route = torch.atan2(
        bypass_target_y - route_lateral_error_m, bypass_target_x
    )
    bypass_heading_error = torch.atan2(
        torch.sin(bypass_heading_route - route_heading_error_rad),
        torch.cos(bypass_heading_route - route_heading_error_rad),
    )

    route_heading_target = torch.atan2(
        -route_lateral_error_m,
        torch.full_like(route_lateral_error_m, cfg.route_lookahead_m),
    )
    route_heading_delta = torch.atan2(
        torch.sin(route_heading_target - route_heading_error_rad),
        torch.cos(route_heading_target - route_heading_error_rad),
    )

    in_interaction = phase == int(ObstaclePhase.INTERACTION)
    desired_speed = nominal.clone()
    interaction_speed = (nominal * cfg.interaction_speed_scale).clamp(
        cfg.min_interaction_speed_mps, cfg.max_interaction_speed_mps
    )
    desired_speed[in_interaction] = interaction_speed[in_interaction]
    desired_yaw = cfg.yaw_gain * torch.where(
        in_interaction, bypass_heading_error, route_heading_delta
    )
    desired_yaw = desired_yaw.clamp(-cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps)

    desired_command = torch.stack((desired_speed, desired_yaw), dim=-1)
    return apply_bounded_supervisor_command(
        desired_command,
        obstacle_observation,
        state,
        cfg=cfg,
        recovery_cfg=recovery_cfg,
        update_dt_s=update_dt_s,
    )


def supervisor_observation(
    obstacle_observation: torch.Tensor,
    nominal_speed_mps: torch.Tensor,
    route_lateral_error_m: torch.Tensor,
    route_heading_error_rad: torch.Tensor,
    measured_forward_speed_mps: torch.Tensor,
    state: ObstacleTeacherState,
    *,
    cfg: ObstacleTeacherCfg = ObstacleTeacherCfg(),
    previous_command: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the normalized HC2 imitation/RL supervisor observation."""
    num_envs = obstacle_observation.shape[0]
    if obstacle_observation.shape != (num_envs, OBSTACLE_OBSERVATION_DIM):
        raise ValueError("invalid obstacle observation shape")
    phase_one_hot = torch.nn.functional.one_hot(
        state.phase, num_classes=SUPERVISOR_PHASE_DIM
    ).to(dtype=obstacle_observation.dtype)
    previous = state.previous_command if previous_command is None else previous_command
    if previous.shape != (num_envs, 2):
        raise ValueError("previous supervisor command must have shape (N, 2)")
    result = torch.cat(
        (
            (nominal_speed_mps / cfg.max_forward_speed_mps).clamp(0.0, 1.0).unsqueeze(-1),
            obstacle_observation,
            (route_lateral_error_m / 0.75).clamp(-1.0, 1.0).unsqueeze(-1),
            (route_heading_error_rad / torch.pi).clamp(-1.0, 1.0).unsqueeze(-1),
            (measured_forward_speed_mps / cfg.max_forward_speed_mps)
            .clamp(-1.0, 1.0)
            .unsqueeze(-1),
            (previous[:, 0] / cfg.max_forward_speed_mps)
            .clamp(0.0, 1.0)
            .unsqueeze(-1),
            (previous[:, 1] / cfg.max_yaw_rate_rps)
            .clamp(-1.0, 1.0)
            .unsqueeze(-1),
            phase_one_hot,
            state.bypass_side.unsqueeze(-1),
        ),
        dim=-1,
    )
    if result.shape != (num_envs, SUPERVISOR_OBSERVATION_DIM):
        raise RuntimeError(f"unexpected supervisor observation shape {result.shape}")
    return result
