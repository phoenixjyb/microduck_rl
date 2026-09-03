"""Pure helpers shared by checkpoint evaluation tooling."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import pstdev

import torch


def fix_velocity_commands(env_cfg, speed: float, yaw_rate: float = 0.0) -> None:
    """Pin deterministic body commands without a heading-controller override."""
    commands = env_cfg.commands
    twist = commands["twist"]
    twist.resampling_time_range = (1.0e6, 1.0e6)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    # Fixed ranges already select a positive forward command.  Enabling the
    # upstream "forward-only" sampler would additionally clamp vx to 0.3 m/s
    # and force both lateral velocity and yaw rate to zero.
    twist.rel_forward_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = (speed, speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (yaw_rate, yaw_rate)

    for name in ("head_pose", "body_pose"):
        command = commands[name]
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.ranges = tuple((0.0, 0.0) for _ in command.ranges)
        command.zero_command_prob = 1.0


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float")
    if not all(math.isfinite(item) for item in values):
        raise ValueError("all values must be finite")
    return values


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated integer")
    return values


def aggregate_speed_cases(cases: Iterable[dict]) -> list[dict]:
    """Aggregate per-seed case dictionaries by commanded speed."""
    grouped: dict[float, list[dict]] = {}
    for case in cases:
        grouped.setdefault(float(case["commanded_speed_mps"]), []).append(case)

    aggregates = []
    for speed, group in sorted(grouped.items()):
        aggregates.append(
            {
                "commanded_speed_mps": speed,
                "seed_count": len(group),
                "observed_speed_mean_mps": _mean(
                    case["observed_speed_mean_mps"] for case in group
                ),
                "observed_speed_std_mps": _std(
                    case["observed_speed_mean_mps"] for case in group
                ),
                "tracking_error_mean_mps": _mean(
                    case["tracking_error_mean_mps"] for case in group
                ),
                "tracking_error_std_mps": _std(
                    case["tracking_error_mean_mps"] for case in group
                ),
                "episode_ends_total": sum(case["episode_ends"] for case in group),
                "non_timeout_ends_total": sum(
                    case["non_timeout_ends"] for case in group
                ),
                "motor_speed_utilization_p99_mean": _mean(
                    case["motor_speed_utilization_p99"] for case in group
                ),
                "motor_speed_rated_exceed_fraction_mean": _mean(
                    case["motor_speed_rated_exceed_fraction"] for case in group
                ),
                "motor_torque_utilization_p99_mean": _mean(
                    case["motor_torque_utilization_p99"] for case in group
                ),
                "motor_torque_near_stall_fraction_mean": _mean(
                    case["motor_torque_near_stall_fraction"] for case in group
                ),
                "motor_mechanical_power_abs_mean_w": _mean(
                    case["motor_mechanical_power_abs_mean_w"] for case in group
                ),
                "motor_thermal_load_proxy_mean": _mean(
                    case["motor_thermal_load_proxy_mean"] for case in group
                ),
                "action_abs_p99_mean": _mean(
                    case["action_abs_p99"] for case in group
                ),
                "action_rate_abs_p99_mean": _mean(
                    case["action_rate_abs_p99"] for case in group
                ),
            }
        )
    return aggregates


def aggregate_command_cases(cases: Iterable[dict]) -> list[dict]:
    """Aggregate command-matrix cases by forward speed and yaw rate."""
    grouped: dict[tuple[float, float], list[dict]] = {}
    for case in cases:
        key = (
            float(case["commanded_speed_mps"]),
            float(case["commanded_yaw_rate_rps"]),
        )
        grouped.setdefault(key, []).append(case)

    aggregates = []
    for (speed, yaw_rate), group in sorted(grouped.items()):
        aggregates.append(
            {
                "commanded_speed_mps": speed,
                "commanded_yaw_rate_rps": yaw_rate,
                "seed_count": len(group),
                "observed_speed_mean_mps": _mean(
                    case["observed_speed_mean_mps"] for case in group
                ),
                "speed_tracking_error_mean_mps": _mean(
                    case["tracking_error_mean_mps"] for case in group
                ),
                "observed_yaw_rate_mean_rps": _mean(
                    case["observed_yaw_rate_mean_rps"] for case in group
                ),
                "yaw_tracking_error_mean_rps": _mean(
                    case["yaw_tracking_error_mean_rps"] for case in group
                ),
                "fall_events_total": sum(case["fall_events"] for case in group),
                "nan_termination_events_total": sum(
                    case["nan_termination_events"] for case in group
                ),
                "nonfinite_steps_total": sum(
                    case["nonfinite_steps"] for case in group
                ),
                "motor_torque_utilization_p99_mean": _mean(
                    case["motor_torque_utilization_p99"] for case in group
                ),
                "motor_torque_near_stall_fraction_mean": _mean(
                    case["motor_torque_near_stall_fraction"] for case in group
                ),
                "motor_thermal_load_proxy_mean": _mean(
                    case["motor_thermal_load_proxy_mean"] for case in group
                ),
                "action_abs_p99_mean": _mean(
                    case["action_abs_p99"] for case in group
                ),
                "action_rate_abs_p99_mean": _mean(
                    case["action_rate_abs_p99"] for case in group
                ),
            }
        )
    return aggregates


def valid_action_deltas(
    current_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_dones: torch.Tensor,
) -> torch.Tensor:
    """Return absolute action deltas, excluding first actions after resets."""
    if current_actions.shape != previous_actions.shape:
        raise ValueError("current and previous action shapes must match")
    if previous_dones.shape != current_actions.shape[:1]:
        raise ValueError("previous_dones must have one value per environment")
    return (current_actions - previous_actions).abs()[~previous_dones].flatten()


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return sum(materialized) / len(materialized)


def _std(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot calculate deviation of an empty sequence")
    return pstdev(materialized)
