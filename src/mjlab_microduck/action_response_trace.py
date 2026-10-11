"""CPU-only descriptive audit for aligned raw-action and joint-response traces.

This module uses only the standard library. It preserves the distinction
between raw policy output, the previously applied target, measured qpos/qvel,
and lagged generalized constraints. Statistics are descriptive, not causal or
policy-admission evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mjlab_microduck.body_yaw_trace import summarize_trace as summarize_body_yaw_trace


PROTOCOL = "microduck-action-response-trace-v1"
SUMMARY_PROTOCOL = "microduck-action-response-descriptive-audit-v1"
STEP_DT = 0.02
STEPS = 240
WORLDS = 8
STARTUP_STEPS = 60
ACTION_WIDTH = 14
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
EVALUATION_SEEDS = (839, 853, 857)
EVALUATION_CASES = ((0.0, 0.0), (0.30, 0.0), (0.20, -0.20), (0.20, 0.20))
STATE_TIMING = "pre-action state before policy/step"
ACTION_TIMING = "raw policy output after inference before step"
TARGET_TIMING = "previous applied target at pre-action boundary"
HISTORY_TIMING = (
    "actor_previous_action and raw_previous_action are the pre-policy "
    "previous-step action history; first sample may follow restored/reset state"
)
CONSTRAINT_TIMING = "last-physics-derived generalized constraint at pre-action boundary"
ACTION_INTERPRETATION = "raw unclipped actor output; magnitude alone is not saturation"
COMMAND_TOLERANCE = 1e-6
PHYSICS_DT_TOLERANCE = 1e-6
TARGET_ARITHMETIC_TOLERANCE = 2e-6
BODY_YAW_SUMMARY_TOLERANCE = 1e-6
LIMIT_MARGIN = 0.15
STOP_MARGIN = 0.05


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str) -> float:
    _require(type(value) in (int, float), f"{name} must be a non-bool number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _array(value: Any, shape: tuple[int, ...], name: str) -> Any:
    _require(type(value) is list and len(value) == shape[0],
             f"{name} must have shape {shape}")
    if len(shape) == 1:
        return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]
    return [_array(item, shape[1:], f"{name}[{index}]") for index, item in enumerate(value)]


def _mean(values: list[float]) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0.0:
        return 0.0
    return scale * (math.fsum(value / scale for value in values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty vector")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = [value / scale for value in values]
    center = math.fsum(normalized) / len(normalized)
    deviation = math.sqrt(math.fsum((value - center) ** 2 for value in normalized) / len(values))
    result = scale * deviation
    _require(math.isfinite(result), "nonfinite standard deviation summary")
    return result


def _finite_difference(left: float, right: float, name: str) -> float:
    result = left - right
    _require(math.isfinite(result), f"nonfinite {name} arithmetic")
    return result


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    scale_x = max(abs(value) for value in xs)
    scale_y = max(abs(value) for value in ys)
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    normalized_x = [value / scale_x for value in xs]
    normalized_y = [value / scale_y for value in ys]
    mean_x = math.fsum(normalized_x) / len(xs)
    mean_y = math.fsum(normalized_y) / len(ys)
    centered_x = [value - mean_x for value in normalized_x]
    centered_y = [value - mean_y for value in normalized_y]
    sum_x = math.fsum(value * value for value in centered_x)
    sum_y = math.fsum(value * value for value in centered_y)
    if sum_x == 0.0 or sum_y == 0.0:
        return None
    result = math.fsum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / math.sqrt(sum_x * sum_y)
    _require(math.isfinite(result), "nonfinite correlation summary")
    return max(-1.0, min(1.0, result))


def _case_identity(case: Mapping[str, Any], trace: Mapping[str, Any]) -> tuple[str, int, float, float]:
    _require(case.get("complete") is True, "case must be complete")
    _require(case.get("terminal_worlds") == [], "case must have no terminal worlds")
    _require(_integer(case.get("steps"), "case.steps") == STEPS
             and _integer(case.get("worlds"), "case.worlds") == WORLDS,
             "case must contain exactly 240 steps and 8 worlds")
    seed = _integer(case.get("seed"), "case.seed")
    _require(seed in EVALUATION_SEEDS, "case seed is not in the held-out set")
    speed, yaw = _number(case.get("speed"), "case.speed"), _number(case.get("yaw"), "case.yaw")
    _require((speed, yaw) in EVALUATION_CASES, "case command is not a held-out command")
    digest = case.get("checkpoint_sha256")
    _require(type(digest) is str and len(digest) == 64
             and all(character in "0123456789abcdef" for character in digest),
             "case checkpoint_sha256 must be lowercase SHA-256")
    _require(trace.get("checkpoint_sha256") == digest, "trace checkpoint hash must match case")
    return digest, seed, speed, yaw


def _validate_metadata(case: Mapping[str, Any], trace: Mapping[str, Any]) -> tuple[str, int, float, float]:
    digest, seed, speed, yaw = _case_identity(case, trace)
    _require(trace.get("protocol") == PROTOCOL, "unsupported action-response protocol")
    _require(_number(trace.get("step_dt"), "trace.step_dt") == STEP_DT,
             "trace step_dt must be 0.02 seconds")
    for key, expected in (("steps", STEPS), ("worlds", WORLDS), ("startup_steps", STARTUP_STEPS)):
        _require(_integer(trace.get(key), f"trace.{key}") == expected,
                 f"trace {key} changed")
    _require(trace.get("timing") == STATE_TIMING
             and trace.get("action_timing") == ACTION_TIMING
             and trace.get("target_timing") == TARGET_TIMING
             and trace.get("action_history_timing") == HISTORY_TIMING
             and trace.get("constraint_timing") == CONSTRAINT_TIMING,
             "action/target/constraint timing identity changed")
    _require(trace.get("action_interpretation") == ACTION_INTERPRETATION,
             "raw unclipped action interpretation changed")
    _require(trace.get("policy_acceptance") is False
             and trace.get("physical_motion_authorized") is False
             and trace.get("simulator_qualified") is False,
             "trace must remain diagnostic-only and non-authorizing")
    _require(trace.get("constraint_includes_contacts_and_limits") is True
             and trace.get("hard_stop_force_isolated") is False,
             "constraint-force scope metadata changed")
    _require(trace.get("joint_columns") == list(JOINTS), "14 walking-joint column order changed")
    target_ids = trace.get("target_ids")
    _require(type(target_ids) is list and len(target_ids) == ACTION_WIDTH
             and all(type(value) is int and value >= 0 for value in target_ids)
             and len(set(target_ids)) == ACTION_WIDTH,
             "target_ids must be 14 unique nonnegative integer ids")
    return digest, seed, speed, yaw


def _validate_body_yaw(case: Mapping[str, Any], trace: Mapping[str, Any],
                       arrays: dict[str, Any], digest: str, seed: int,
                       speed: float, yaw: float) -> dict[str, Any]:
    body_case = {
        "complete": True, "steps": STEPS, "worlds": WORLDS,
        "terminal_worlds": [], "seed": seed, "checkpoint_sha256": digest,
        "speed": speed, "yaw": yaw, "yaw_mae": case.get("yaw_mae"),
    }
    body_trace = {
        "protocol": "body-yaw-orientation-trace-v1",
        "timing": STATE_TIMING,
        "step_dt": STEP_DT,
        "steps": STEPS,
        "worlds": WORLDS,
        "startup_steps": STARTUP_STEPS,
        "checkpoint_sha256": digest,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
        **{key: arrays[key] for key in ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")},
    }
    return summarize_body_yaw_trace(body_case, body_trace)


def _world_joint_summary(world: int, joint: int, arrays: dict[str, Any],
                         yaw: float) -> dict[str, Any]:
    settled = range(STARTUP_STEPS, STEPS)
    raw = [arrays["raw_action"][step][world][joint] for step in settled]
    qpos = [arrays["joint_position"][step][world][joint] for step in settled]
    qvel = [arrays["joint_velocity"][step][world][joint] for step in settled]
    target = [arrays["previous_applied_target"][step][world][joint] for step in settled]
    bias = arrays["encoder_bias"][STARTUP_STEPS][world][joint]
    lo, hi = arrays["hard_limits"][STARTUP_STEPS][world][joint]
    constraint = [arrays["generalized_constraint_force"][step][world][joint] for step in settled]
    yaw_residual = [arrays["angular_b"][step][world][2] - yaw for step in settled]

    deltas = [abs(_finite_difference(right, left, "raw action delta"))
              for left, right in zip(raw, raw[1:], strict=False)]
    margins = [min(_finite_difference(value, lo, "lower hard-limit margin"),
                   _finite_difference(hi, value, "upper hard-limit margin")) for value in qpos]
    costs = []
    for value in qpos:
        lower_cost = max(_finite_difference(lo + LIMIT_MARGIN, value, "lower proximity cost"), 0.0)
        upper_cost = max(_finite_difference(value, hi - LIMIT_MARGIN, "upper proximity cost"), 0.0)
        cost = lower_cost + upper_cost
        _require(math.isfinite(cost), "nonfinite hypothetical limit proximity cost")
        costs.append(cost)
    target_errors = [abs(_finite_difference(target_value, qpos_value, "lagged target qpos error"))
                     for target_value, qpos_value in zip(target, qpos, strict=True)]

    correlations = {
        "yaw_residual_vs_raw_action": _pearson(yaw_residual, raw),
        "yaw_residual_vs_joint_velocity": _pearson(yaw_residual, qvel),
        "yaw_residual_vs_lagged_constraint": _pearson(yaw_residual, constraint),
    }
    return {
        "world": world,
        "joint": JOINTS[joint],
        "raw_action_mean": _mean(raw),
        "raw_action_std": _std(raw),
        "raw_action_min": min(raw),
        "raw_action_max": max(raw),
        "raw_action_mean_abs_delta": _mean(deltas),
        "joint_position_min": min(qpos),
        "joint_position_max": max(qpos),
        "joint_position_std": _std(qpos),
        "joint_velocity_abs_mean": _mean([abs(value) for value in qvel]),
        "lagged_target_qpos_mae": _mean(target_errors),
        "previous_target_outside_hard_range_fraction": _mean([
            1.0 if value < lo or value > hi else 0.0 for value in target
        ]),
        "qpos_within_0_15_rad_hard_limit_margin_fraction": _mean([
            1.0 if margin < LIMIT_MARGIN else 0.0 for margin in margins
        ]),
        "hypothetical_joint_limit_proximity_cost_mean": _mean(costs),
        "qpos_within_0_05_rad_hard_stop_margin_fraction": _mean([
            1.0 if margin <= STOP_MARGIN else 0.0 for margin in margins
        ]),
        "minimum_hard_limit_margin_rad": min(margins),
        "generalized_constraint_force_abs_mean": _mean([abs(value) for value in constraint]),
        "yaw_residual_correlations": correlations,
        "encoder_bias": bias,
    }


def summarize_trace(case: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a retained full trace and summarize settled action/joint response.

    Correlations describe within-world co-occurrence only. Raw actor magnitude
    above one is retained as a value and is never called saturation. Constraint
    forces are solver-lagged and include contacts and limits, not isolated hard
    stops. The hypothetical proximity cost is per joint and is not the actual
    summed reward term.
    """
    _require(isinstance(case, Mapping) and isinstance(trace, Mapping), "case and trace mappings")
    digest, seed, speed, yaw = _validate_metadata(case, trace)
    shapes = {
        "command": (STEPS, WORLDS, 3), "actor_command": (STEPS, WORLDS, 3),
        "quat_wxyz": (STEPS, WORLDS, 4), "angular_b": (STEPS, WORLDS, 3),
        "angular_w": (STEPS, WORLDS, 3),
        "actor_previous_action": (STEPS, WORLDS, ACTION_WIDTH),
        "raw_previous_action": (STEPS, WORLDS, ACTION_WIDTH),
        "raw_action": (STEPS, WORLDS, ACTION_WIDTH),
        "joint_position": (STEPS, WORLDS, ACTION_WIDTH),
        "joint_velocity": (STEPS, WORLDS, ACTION_WIDTH),
        "previous_applied_target": (STEPS, WORLDS, ACTION_WIDTH),
        "encoder_bias": (STEPS, WORLDS, ACTION_WIDTH),
        "action_offset": (STEPS, WORLDS, ACTION_WIDTH),
        "hard_limits": (STEPS, WORLDS, ACTION_WIDTH, 2),
        "generalized_constraint_force": (STEPS, WORLDS, ACTION_WIDTH),
        "episode_length": (STEPS, WORLDS),
        "physics_time": (STEPS, WORLDS),
    }
    arrays: dict[str, Any] = {}
    for name, shape in shapes.items():
        if name == "episode_length":
            raw = trace.get(name)
            _require(type(raw) is list and len(raw) == STEPS, f"{name} must have shape {shape}")
            rows = []
            for step, row in enumerate(raw):
                _require(type(row) is list and len(row) == WORLDS,
                         f"{name}[{step}] must have shape {(WORLDS,)}")
                rows.append([_integer(value, f"{name}[{step}][{world}]")
                             for world, value in enumerate(row)])
            arrays[name] = rows
        else:
            arrays[name] = _array(trace.get(name), shape, name)
    common = trace.get("common_step_counter")
    _require(type(common) is list and len(common) == STEPS,
             "common_step_counter must be a 240-element integer vector")
    counters = [_integer(value, f"common_step_counter[{step}]") for step, value in enumerate(common)]
    _require(all(right == left + 1 for left, right in zip(counters, counters[1:])),
             "common-step counter must increment by one")

    limits = arrays["hard_limits"]
    for step in range(STEPS):
        for world in range(WORLDS):
            _require(all(hi > lo for lo, hi in limits[step][world]),
                     "hard limits must be strictly ordered")
    for name in ("action_offset", "encoder_bias", "hard_limits"):
        baseline = arrays[name][0]
        _require(all(arrays[name][step] == baseline for step in range(1, STEPS)),
                 f"{name} must remain unchanged throughout the trace")

    command = arrays["command"]
    actor_command = arrays["actor_command"]
    expected_command = (speed, 0.0, yaw)
    for step in range(STEPS):
        for world in range(WORLDS):
            _require(all(actor_command[step][world][axis] == command[step][world][axis]
                         for axis in range(3)),
                     f"actor command differs from manager command at step {step}, world {world}")
            _require(all(abs(command[step][world][axis] - expected_command[axis]) <= COMMAND_TOLERANCE
                         for axis in range(3)),
                     f"held-out command changed at step {step}, world {world}")

    for step in range(STEPS):
        for world in range(WORLDS):
            if step:
                _require(arrays["episode_length"][step][world]
                         == arrays["episode_length"][step - 1][world] + 1,
                         "episode clock must increment without reset")
                delta = _finite_difference(arrays["physics_time"][step][world],
                                           arrays["physics_time"][step - 1][world],
                                           "physics clock delta")
                _require(delta > 0 and abs(delta - STEP_DT) <= PHYSICS_DT_TOLERANCE,
                         "physics clock must advance by 0.02 s")
            _require(arrays["physics_time"][step][world] >= 0.0,
                     "physics clocks must be nonnegative")

    for step, following in enumerate(range(1, STEPS)):
        for world in range(WORLDS):
            _require(arrays["raw_previous_action"][step][world]
                     == arrays["actor_previous_action"][step][world],
                     "raw previous action must equal actor previous-action history")
            _require(arrays["actor_previous_action"][following][world]
                     == arrays["raw_action"][step][world]
                     and arrays["raw_previous_action"][following][world]
                     == arrays["raw_action"][step][world],
                     "both next-step histories must equal current raw action")
            expected_target = [
                arrays["raw_action"][step][world][joint]
                + arrays["action_offset"][following][world][joint]
                - arrays["encoder_bias"][following][world][joint]
                for joint in range(ACTION_WIDTH)
            ]
            _require(all(math.isfinite(value) for value in expected_target),
                     "nonfinite raw action target arithmetic")
            _require(all(abs(actual - expected) <= TARGET_ARITHMETIC_TOLERANCE
                         for actual, expected in zip(
                             arrays["previous_applied_target"][following][world], expected_target, strict=True)),
                     "next previous-applied target differs from action + offset - encoder bias")

    body_yaw = _validate_body_yaw(case, trace, arrays, digest, seed, speed, yaw)
    per_world = []
    for world in range(WORLDS):
        per_world.append({
            "world": world,
            "yaw_residual_convention": "body angular z minus held-out command yaw",
            "joints": [_world_joint_summary(world, joint, arrays, yaw)
                       for joint in range(ACTION_WIDTH)],
        })
    result = {
        "protocol": SUMMARY_PROTOCOL,
        "status": "descriptive-audit-only",
        "checkpoint_sha256": digest,
        "seed": seed,
        "speed": speed,
        "yaw": yaw,
        "steps": STEPS,
        "worlds_count": WORLDS,
        "startup_steps": STARTUP_STEPS,
        "settled_steps": STEPS - STARTUP_STEPS,
        "step_dt": STEP_DT,
        "joint_columns": list(JOINTS),
        "target_ids": trace["target_ids"].copy(),
        "timing": STATE_TIMING,
        "action_timing": ACTION_TIMING,
        "target_timing": TARGET_TIMING,
        "constraint_timing": CONSTRAINT_TIMING,
        "raw_action_magnitude_above_one_is_not_saturation": True,
        "lagged_constraint_includes_contacts_and_limits": True,
        "hard_stop_force_is_isolated": False,
        "proximity_cost_margin_rad": LIMIT_MARGIN,
        "proximity_cost_is_hypothetical_per_joint_not_actual_summed_reward": True,
        "stop_margin_rad": STOP_MARGIN,
        "correlations_are_descriptive_not_causal": True,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
        "body_yaw_trace_audit": body_yaw,
        "worlds": per_world,
    }
    _require(_finite_summary(result), "summary contains nonfinite or unsupported values")
    return result


def _finite_summary(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_summary(item) for item in value.values())
    if type(value) is list:
        return all(_finite_summary(item) for item in value)
    if value is None or type(value) in (str, bool):
        return True
    return type(value) in (int, float) and math.isfinite(value)
