"""CPU-only descriptive audit for a retained body-yaw orientation trace.

The trace is sampled before policy inference and environment stepping. This
module checks frame/identity arithmetic and reports descriptive time-series
summaries; it does not establish causality or change policy gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


PROTOCOL = "body-yaw-orientation-trace-v1"
TIMING = "pre-action state before policy/step"
STEPS = 240
WORLDS = 8
STARTUP_STEPS = 60
STEP_DT = 0.02
IDENTITY_TOLERANCE = 1e-5
COMMAND_TOLERANCE = 1e-6
SUMMARY_TOLERANCE = 1e-6
HEADING_AXIS_MIN_NORM_SQ = 1e-4
HELD_OUT_COMMANDS = ((0., 0.), (.30, 0.), (.20, -.20), (.20, .20))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{name} must be a non-bool number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    return value


def _array(value: Any, dims: tuple[int, ...], name: str) -> Any:
    _require(type(value) is list and len(value) == dims[0],
             f"{name} must have shape {dims}")
    if len(dims) == 1:
        return [_number(item, f"{name}[{i}]") for i, item in enumerate(value)]
    return [_array(item, dims[1:], f"{name}[{i}]") for i, item in enumerate(value)]


def _rotate_wxyz(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by unit quaternion q=(w,x,y,z), body to world."""
    w, x, y, z = q
    vx, vy, vz = v
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return [vx + 2. * (w * cx + dx),
            vy + 2. * (w * cy + dy),
            vz + 2. * (w * cz + dz)]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sign_transitions(values: list[float]) -> int:
    """Count adjacent nonzero sign changes; exact zeros are skipped."""
    signs = [1 if value > 0. else -1 for value in values if value != 0.]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def _lag1_covariance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    return sum((left - mean) * (right - mean)
               for left, right in zip(values, values[1:])) / (len(values) - 1)


def summarize_trace(case: Any, trace: Any) -> dict[str, Any]:
    """Validate and summarize one hash-bound 8-world, 240-step trace.

    Residuals are measured-minus-command. Heading rate is the instantaneous
    derivative of the azimuth of the body x-axis projected into world XY;
    samples with near-zero projected-axis norm are undefined, not substituted
    into the body-frame yaw tracking metric.
    """
    _require(isinstance(case, dict), "case must be an object")
    _require(isinstance(trace, dict), "trace must be an object")
    _require(case.get("complete") is True, "case must be complete")
    _require(_integer(case.get("steps"), "case.steps") == STEPS,
             f"case must contain exactly {STEPS} steps")
    _require(_integer(case.get("worlds"), "case.worlds") == WORLDS,
             f"case must contain exactly {WORLDS} worlds")
    _require(type(case.get("terminal_worlds")) is list and not case["terminal_worlds"],
             "case must have no terminal worlds")
    seed = _integer(case.get("seed"), "case.seed")
    _require(seed in (839, 853, 857), "case seed is not held-out")
    digest = case.get("checkpoint_sha256")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(char in "0123456789abcdef" for char in digest),
             "case checkpoint_sha256 must be lowercase SHA-256")
    _require(trace.get("checkpoint_sha256") == digest,
             "trace checkpoint SHA must match case")
    _require(trace.get("protocol") == PROTOCOL, "unsupported trace protocol")
    _require(trace.get("timing") == TIMING, "trace timing identity changed")
    _require(_integer(trace.get("steps"), "trace.steps") == STEPS
             and _integer(trace.get("worlds"), "trace.worlds") == WORLDS
             and _integer(trace.get("startup_steps"), "trace.startup_steps") == STARTUP_STEPS,
             "trace dimensions/window changed")
    step_dt = _number(trace.get("step_dt"), "trace.step_dt")
    _require(step_dt == STEP_DT, "trace step_dt must be 0.02 seconds")
    _require(trace.get("policy_acceptance") is False
             and trace.get("physical_motion_authorized") is False,
             "trace must remain diagnostic-only and non-authorizing")

    arrays = {
        "command": _array(trace.get("command"), (STEPS, WORLDS, 3), "command"),
        "actor_command": _array(trace.get("actor_command"), (STEPS, WORLDS, 3), "actor_command"),
        "quat_wxyz": _array(trace.get("quat_wxyz"), (STEPS, WORLDS, 4), "quat_wxyz"),
        "angular_b": _array(trace.get("angular_b"), (STEPS, WORLDS, 3), "angular_b"),
        "angular_w": _array(trace.get("angular_w"), (STEPS, WORLDS, 3), "angular_w"),
    }
    speed = _number(case.get("speed"), "case.speed")
    yaw = _number(case.get("yaw"), "case.yaw")
    _require((speed, yaw) in HELD_OUT_COMMANDS,
             "case command is not one of the declared held-out commands")
    expected_command = [speed, 0., yaw]
    command, actor, quats = arrays["command"], arrays["actor_command"], arrays["quat_wxyz"]
    angular_b, angular_w = arrays["angular_b"], arrays["angular_w"]
    for step in range(STEPS):
        for world in range(WORLDS):
            for axis in range(3):
                _require(abs(command[step][world][axis] - expected_command[axis]) <= COMMAND_TOLERANCE,
                         f"command changed at step {step}, world {world}, axis {axis}")
                _require(actor[step][world][axis] == command[step][world][axis],
                         f"actor command mismatch at step {step}, world {world}, axis {axis}")
            norm = math.sqrt(sum(component * component for component in quats[step][world]))
            _require(abs(norm - 1.) <= IDENTITY_TOLERANCE,
                     f"quaternion is not unit at step {step}, world {world}")
            rotated = _rotate_wxyz(quats[step][world], angular_b[step][world])
            _require(all(abs(rotated[axis] - angular_w[step][world][axis]) <= IDENTITY_TOLERANCE
                         for axis in range(3)),
                     f"body/world angular rotation mismatch at step {step}, world {world}")

    yaw_mae = case.get("yaw_mae")
    _require(type(yaw_mae) is list and len(yaw_mae) == WORLDS,
             "case.yaw_mae must contain 8 world values")
    stored_mae = [_number(value, f"case.yaw_mae[{world}]") for world, value in enumerate(yaw_mae)]
    _require(all(value >= 0. for value in stored_mae), "case yaw_mae must be nonnegative")

    reports = []
    for world in range(WORLDS):
        residual = [angular_b[step][world][2] - yaw
                    for step in range(STARTUP_STEPS, STEPS)]
        body_mae = _mean([abs(value) for value in residual])
        _require(abs(body_mae - stored_mae[world]) <= SUMMARY_TOLERANCE,
                 f"recomputed body yaw MAE mismatch in world {world}")

        world_vertical_error = [angular_w[step][world][2] - yaw
                                for step in range(STARTUP_STEPS, STEPS)]
        heading_rates = []
        for step in range(STARTUP_STEPS, STEPS):
            q, omega = quats[step][world], angular_w[step][world]
            axis = _rotate_wxyz(q, [1., 0., 0.])
            axis_dot = _cross(omega, axis)
            denominator = axis[0] ** 2 + axis[1] ** 2
            heading_rates.append(None if denominator < HEADING_AXIS_MIN_NORM_SQ else
                                  (axis[0] * axis_dot[1] - axis[1] * axis_dot[0]) / denominator)
        defined = [rate for rate in heading_rates if rate is not None]
        heading_errors = [rate - yaw for rate in defined]
        reports.append({
            "world": world,
            "body_yaw_mae": body_mae,
            "body_yaw_signed_bias": _mean(residual),
            "body_yaw_residual_std": _std(residual),
            "body_yaw_residual_sign_transitions": _sign_transitions(residual),
            "body_yaw_residual_lag1_covariance": _lag1_covariance(residual),
            "world_vertical_rate_mae": _mean([abs(value) for value in world_vertical_error]),
            "projected_body_x_heading_rate_mean": _mean(defined) if defined else None,
            "projected_body_x_heading_rate_mae": _mean([abs(value) for value in heading_errors]) if defined else None,
            "projected_body_x_heading_rate_defined_samples": len(defined),
            "projected_body_x_heading_rate_undefined_samples": len(heading_rates) - len(defined),
        })

    return {
        "protocol": "body-yaw-trace-descriptive-audit-v1",
        "status": "descriptive-audit-only",
        "checkpoint_sha256": digest,
        "seed": seed,
        "speed": speed,
        "yaw": yaw,
        "steps": STEPS,
        "startup_steps": STARTUP_STEPS,
        "settled_steps": STEPS - STARTUP_STEPS,
        "step_dt": step_dt,
        "body_yaw_summary_tolerance": SUMMARY_TOLERANCE,
        "body_yaw_summary_tolerance_is_acceptance_threshold": False,
        "heading_axis_min_norm_sq": HEADING_AXIS_MIN_NORM_SQ,
        "heading_rate_is_body_yaw_gate_substitute": False,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
        "worlds": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json", type=Path,
                        help="JSON object with the retained case and trace objects")
    args = parser.parse_args(argv)
    try:
        raw = args.trace_json.read_bytes()
        payload = json.loads(raw)
        _require(isinstance(payload, dict), "input must be an object")
        result = summarize_trace(payload.get("case"), payload.get("trace"))
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"body yaw trace audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
