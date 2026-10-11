"""Offline descriptive summaries for aligned gait/contact/head/yaw traces.

This module deliberately has no simulator, torch, or NumPy dependency. Its
statistics describe co-occurrence and correlation; they are not admission
criteria or causal evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


PROTOCOL = "microduck-gait-yaw-trace-v1"
SUMMARY_PROTOCOL = "microduck-gait-yaw-summary-v1"
STEP_DT = 0.02
STARTUP_STEPS = 60
STEPS = 240
WORLDS = 8
EVALUATION_SEEDS = (839, 853, 857)
EVALUATION_CASES = ((0.0, 0.0), (0.30, 0.0), (0.20, -0.20), (0.20, 0.20))
CONTACT_COLUMNS = ("left_foot_collision", "right_foot_collision")
FOOT_SITES = ("left_foot", "right_foot")
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
TIMING = "pre-action state before policy/step"
FORCE_TIMING = "last-physics-derived-contact-at-pre-action"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _positive_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _contact_count(value: object) -> bool:
    if type(value) is int:
        return value >= 0
    return type(value) is float and math.isfinite(value) and value >= 0.0 and value.is_integer()


def _validate_array(value: object, shape: tuple[int, ...], name: str, predicate) -> None:
    _require(type(value) is list, f"{name} is a JSON array")
    _require(len(value) == shape[0], f"{name} has declared leading dimension")
    if len(shape) == 1:
        _require(all(predicate(item) for item in value), f"{name} has valid scalar values")
        return
    for row in value:
        _validate_array(row, shape[1:], name, predicate)


def _flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _mean(values: list[float]) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    return 0.0 if scale == 0.0 else scale * (math.fsum(value / scale for value in values) / len(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    scale_x = max(abs(value) for value in xs)
    scale_y = max(abs(value) for value in ys)
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    normalized_x = [value / scale_x for value in xs]
    normalized_y = [value / scale_y for value in ys]
    mean_x, mean_y = math.fsum(normalized_x) / len(xs), math.fsum(normalized_y) / len(ys)
    centered_x = [value - mean_x for value in normalized_x]
    centered_y = [value - mean_y for value in normalized_y]
    sum_x = math.fsum(value * value for value in centered_x)
    sum_y = math.fsum(value * value for value in centered_y)
    if sum_x <= 0.0 or sum_y <= 0.0:
        return None
    correlation = math.fsum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / math.sqrt(sum_x * sum_y)
    return max(-1.0, min(1.0, correlation))


def _rotate_wxyz(quaternion: list[float], vector: list[float]) -> list[float]:
    """Rotate body-frame vector by q=(w,x,y,z), matching body_yaw_trace."""
    w, x, y, z = quaternion
    vx, vy, vz = vector
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return [vx + 2.0 * (w * cx + dx),
            vy + 2.0 * (w * cy + dy),
            vz + 2.0 * (w * cz + dz)]


def _finite_summary(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_summary(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_summary(item) for item in value)
    return value is None or type(value) in (str, bool) or _number(value)


def summarize_trace(case: Mapping, trace: Mapping) -> dict:
    """Validate a complete fixed evaluation trace and return descriptive stats.

    Per-world counts/fractions exclude the first ``STARTUP_STEPS``. All arrays
    are retained by the caller; this function only reads them and returns a new
    summary. No policy gate is evaluated here.
    """
    _require(isinstance(case, Mapping) and isinstance(trace, Mapping), "case and trace mappings")
    _require(case.get("complete") is True and case.get("terminal_worlds") == [],
             "complete terminal-free fixed case")
    _require(type(case.get("worlds")) is int and case["worlds"] == WORLDS
             and type(case.get("steps")) is int and case["steps"] == STEPS,
             "fixed eight-world 240-step case")
    speed, target_yaw = case.get("speed"), case.get("yaw")
    _require(_number(speed) and _number(target_yaw)
             and (speed, target_yaw) in EVALUATION_CASES,
             "declared fixed held-out case command")
    _require(type(case.get("seed")) is int and case["seed"] in EVALUATION_SEEDS,
             "declared fixed held-out evaluation seed")
    digest = case.get("checkpoint_sha256")
    _require(type(digest) is str and len(digest) == 64
             and all(character in "0123456789abcdef" for character in digest),
             "lowercase case checkpoint SHA256")
    case_yaw_mae = case.get("yaw_mae")
    _validate_array(case_yaw_mae, (WORLDS,), "case yaw_mae", _number)

    _require(trace.get("protocol") == PROTOCOL
             and trace.get("step_dt") == STEP_DT
             and trace.get("startup_steps") == STARTUP_STEPS
             and trace.get("steps") == STEPS and trace.get("worlds") == WORLDS
             and trace.get("checkpoint_sha256") == digest
             and trace.get("timing") == TIMING
             and trace.get("policy_acceptance") is False
             and trace.get("physical_motion_authorized") is False,
             "fixed trace identity and non-acceptance metadata")
    _require(trace.get("contact_columns") == list(CONTACT_COLUMNS)
             and trace.get("foot_sites") == list(FOOT_SITES)
             and trace.get("head_joints") == list(HEAD_JOINTS)
             and trace.get("force_timing") == FORCE_TIMING
             and trace.get("force_frame") == "world"
             and trace.get("terrain") == "stationary-flat",
             "fixed sensor columns and force/terrain metadata")

    shapes = {
        "command": (STEPS, WORLDS, 3),
        "actor_command": (STEPS, WORLDS, 3),
        "angular_b": (STEPS, WORLDS, 3),
        "angular_w": (STEPS, WORLDS, 3),
        "quat_wxyz": (STEPS, WORLDS, 4),
        "episode_length": (STEPS, WORLDS),
        "physics_time": (STEPS, WORLDS),
        "contact_found": (STEPS, WORLDS, 2),
        "contact_force_w": (STEPS, WORLDS, 2, 3),
        "current_air_time": (STEPS, WORLDS, 2),
        "current_contact_time": (STEPS, WORLDS, 2),
        "foot_velocity_w": (STEPS, WORLDS, 2, 3),
        "head_position_offset": (STEPS, WORLDS, 4),
        "head_velocity": (STEPS, WORLDS, 4),
        "head_command": (STEPS, WORLDS, 4),
        "actor_head_command": (STEPS, WORLDS, 4),
    }
    for name, shape in shapes.items():
        predicate = (_positive_int if name == "episode_length" else
                     _contact_count if name == "contact_found" else _number)
        _validate_array(trace.get(name), shape, name, predicate)

    for name in ("physics_time", "current_air_time", "current_contact_time"):
        _require(all(value >= 0.0 for value in _flatten(trace[name])),
                 f"{name} is nonnegative")

    command, actor_command = trace["command"], trace["actor_command"]
    for step in range(STEPS):
        for world in range(WORLDS):
            for actual, observed in zip(command[step][world], actor_command[step][world], strict=True):
                _require(actual == observed, "actor twist equals manager command")
            _require(all(abs(actual - expected) <= 1e-6 for actual, expected in
                         zip(command[step][world], (speed, 0.0, target_yaw), strict=True)),
                     "trace twist matches fixed evaluation case")
            for actual, observed in zip(trace["head_command"][step][world],
                                         trace["actor_head_command"][step][world], strict=True):
                _require(actual == observed, "actor head command equals manager command")
            quaternion = trace["quat_wxyz"][step][world]
            quaternion_norm = math.hypot(*quaternion)
            _require(abs(quaternion_norm - 1.0) <= 1e-5,
                     "body quaternion is unit length")
            rotated_angular = _rotate_wxyz(quaternion, trace["angular_b"][step][world])
            _require(all(_number(value) for value in rotated_angular)
                     and all(abs(actual - expected) <= 1e-5 for actual, expected in
                             zip(rotated_angular, trace["angular_w"][step][world], strict=True)),
                     "body/world angular velocity frames agree")

    episode_length, physics_time = trace["episode_length"], trace["physics_time"]
    for world in range(WORLDS):
        for step in range(1, STEPS):
            _require(episode_length[step][world] == episode_length[step - 1][world] + 1,
                     "episode length increments without reset segments")
            delta = physics_time[step][world] - physics_time[step - 1][world]
            _require(delta > 0.0 and abs(delta - STEP_DT) <= 1e-6,
                     "monotonic physics time advances by fixed step")

    settled = range(STARTUP_STEPS, STEPS)
    residual = [[trace["angular_b"][step][world][2] - target_yaw
                 for world in range(WORLDS)] for step in settled]
    recomputed_yaw_mae = [_mean([abs(residual[t][world]) for t in range(len(residual))])
                           for world in range(WORLDS)]
    _require(all(abs(actual - expected) <= 1e-6
                 for actual, expected in zip(recomputed_yaw_mae, case_yaw_mae, strict=True)),
             "trace yaw MAE matches retained case within 1e-6")

    pattern_names = ("no-contact", "one-contact", "two-contacts")
    counts = {name: [0] * WORLDS for name in pattern_names}
    flight_counts = [0] * WORLDS
    timer_disagreements = [0] * WORLDS
    total_foot_frames = len(settled) * 2
    slip_sums = [[0.0] * WORLDS for _ in FOOT_SITES]
    slip_maxima: list[list[float | None]] = [[None] * WORLDS for _ in FOOT_SITES]
    slip_counts = [[0] * WORLDS for _ in FOOT_SITES]
    residual_by_pattern: dict[str, list[float]] = {name: [] for name in pattern_names}
    residual_by_pattern_world: dict[str, list[list[float]]] = {
        name: [[] for _ in range(WORLDS)] for name in pattern_names
    }
    pooled_head_abs_velocities: list[list[float]] = [[] for _ in HEAD_JOINTS]
    world_residuals: list[list[float]] = [[] for _ in range(WORLDS)]
    world_head_velocities: list[list[list[float]]] = [
        [[] for _ in HEAD_JOINTS] for _ in range(WORLDS)
    ]
    world_head_abs_velocities: list[list[list[float]]] = [
        [[] for _ in HEAD_JOINTS] for _ in range(WORLDS)
    ]
    pooled_residuals: list[float] = []
    pooled_head_velocities: list[list[float]] = [[] for _ in HEAD_JOINTS]

    for step in settled:
        for world in range(WORLDS):
            found = trace["contact_found"][step][world]
            air = trace["current_air_time"][step][world]
            touching = [value > 0 for value in found]
            in_air = [value > 0.0 for value in air]
            contact_count = sum(touching)
            pattern = pattern_names[contact_count]
            counts[pattern][world] += 1
            if not any(touching) and all(in_air):
                flight_counts[world] += 1
            timer_disagreements[world] += sum(
                (not contact_present) != (air_time > 0.0)
                for contact_present, air_time in zip(touching, air, strict=True)
            )
            yaw_residual = residual[step - STARTUP_STEPS][world]
            _require(_number(yaw_residual), "finite signed yaw residual")
            residual_by_pattern[pattern].append(abs(yaw_residual))
            residual_by_pattern_world[pattern][world].append(abs(yaw_residual))
            pooled_residuals.append(yaw_residual)
            world_residuals[world].append(yaw_residual)
            for joint, velocity in enumerate(trace["head_velocity"][step][world]):
                pooled_head_velocities[joint].append(velocity)
                pooled_head_abs_velocities[joint].append(abs(velocity))
                world_head_velocities[world][joint].append(velocity)
                world_head_abs_velocities[world][joint].append(abs(velocity))
            for foot in range(2):
                if not touching[foot]:
                    continue
                vx, vy = trace["foot_velocity_w"][step][world][foot][:2]
                slip = math.hypot(vx, vy)
                _require(_number(slip), "finite horizontal foot slip speed")
                slip_sums[foot][world] += slip
                slip_counts[foot][world] += 1
                prior_max = slip_maxima[foot][world]
                slip_maxima[foot][world] = slip if prior_max is None else max(prior_max, slip)

    def fractions(values: list[int], denominator: int) -> list[float]:
        return [value / denominator for value in values]

    slip_means = {
        foot_name: [slip_sums[foot][world] / slip_counts[foot][world]
                    if slip_counts[foot][world] else None for world in range(WORLDS)]
        for foot, foot_name in enumerate(FOOT_SITES)
    }
    per_world_contact_count = {name: values for name, values in counts.items()}
    per_world_contact_fraction = {
        name: fractions(values, STEPS - STARTUP_STEPS) for name, values in counts.items()
    }
    conditional_yaw_mae = {
        name: (_mean(values) if values else None)
        for name, values in residual_by_pattern.items()
    }
    conditional_yaw_mae_per_world = {
        name: [_mean(values) if values else None for values in world_values]
        for name, world_values in residual_by_pattern_world.items()
    }
    yaw_correlations = {
        joint_name: _pearson(pooled_residuals, pooled_head_velocities[joint])
        for joint, joint_name in enumerate(HEAD_JOINTS)
    }
    yaw_correlations_per_world = {
        joint_name: [_pearson(world_residuals[world], world_head_velocities[world][joint])
                     for world in range(WORLDS)]
        for joint, joint_name in enumerate(HEAD_JOINTS)
    }
    mean_abs_head_velocity_per_world = {
        joint_name: [_mean(world_head_abs_velocities[world][joint]) for world in range(WORLDS)]
        for joint, joint_name in enumerate(HEAD_JOINTS)
    }

    summary = {
        "protocol": SUMMARY_PROTOCOL,
        "trace_protocol": PROTOCOL,
        "checkpoint_sha256": digest,
        "seed": case.get("seed"),
        "speed": speed,
        "yaw": target_yaw,
        "steps": STEPS,
        "startup_steps_excluded": STARTUP_STEPS,
        "worlds": WORLDS,
        "contact_columns": list(CONTACT_COLUMNS),
        "foot_sites": list(FOOT_SITES),
        "head_joints": list(HEAD_JOINTS),
        "timing": TIMING,
        "correlation_scope_pooled": "pooled post-startup world-steps; may include between-world covariance",
        "per_world_correlation_scope": "within-world post-startup time samples; descriptive only",
        "contact_pattern_count_per_world": per_world_contact_count,
        "contact_pattern_fraction_per_world": per_world_contact_fraction,
        "double_flight_fraction_per_world": fractions(flight_counts, STEPS - STARTUP_STEPS),
        "ambiguous_air_timer_fraction_per_world": fractions(timer_disagreements, total_foot_frames),
        "air_timer_disagreement_basis": (
            "foot-step where contact_found==0 disagrees with current_air_time>0"
        ),
        "contact_masked_foot_slip_mean_mps_per_world": slip_means,
        "foot_slip_basis": "named foot-site world-XY speed when sampled contact_found>0; not contact-patch slip",
        "flight_fraction_basis": "post-startup pre-action samples, not substep-resolved flight duration",
        "contact_masked_foot_slip_max_mps_per_world": {
            foot_name: slip_maxima[foot] for foot, foot_name in enumerate(FOOT_SITES)
        },
        "yaw_residual_mae_by_contact_pattern_pooled": conditional_yaw_mae,
        "yaw_residual_mae_by_contact_pattern_per_world": conditional_yaw_mae_per_world,
        "mean_abs_head_velocity_rad_s_pooled": {
            name: _mean(pooled_head_abs_velocities[index])
            for index, name in enumerate(HEAD_JOINTS)
        },
        "mean_abs_head_velocity_rad_s_per_world": mean_abs_head_velocity_per_world,
        "signed_yaw_residual_head_velocity_pearson_pooled": yaw_correlations,
        "signed_yaw_residual_head_velocity_pearson_per_world": yaw_correlations_per_world,
        "recomputed_yaw_mae_per_world": recomputed_yaw_mae,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
    }
    _require(_finite_summary(summary), "finite descriptive gait/yaw summary")
    return summary
