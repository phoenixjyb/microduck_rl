"""Descriptive, CPU-only summaries of retained signed body-response cases.

This module validates the shape and arithmetic consistency of an existing
evaluation row. It does not load the simulator or alter evaluation gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


PROTOCOL = "body-response-descriptive-audit-v1"
SOURCE_DIAGNOSTIC_PROTOCOL = "signed-body-response-and-joint-load-v1"
WORLD_COUNT = 8
CASE_STEPS = 240
STARTUP_STEPS = 60
SETTLED_STEPS = CASE_STEPS - STARTUP_STEPS
# Allows only small representation/reduction roundoff in descriptive summary
# identities. This is not a tracking tolerance or policy-acceptance threshold.
SUMMARY_TOLERANCE = 1e-6
_JOINT_ARRAYS = (
    "abs_force_p99_nm_by_joint",
    "mean_signed_force_nm_by_joint",
    "soft_limit_fraction_by_joint",
    "hard_stop_proximity_fraction_by_joint",
    "hard_range_violation_fraction_by_joint",
    "min_distance_to_hard_stop_rad_by_joint",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be a non-bool number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _integer(value: Any, label: str) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    return value


def _world_values(mapping: dict[str, Any], key: str, *, nullable: bool = False) -> list[float] | None:
    value = mapping.get(key)
    if nullable and value is None:
        return None
    _require(isinstance(value, list) and len(value) == WORLD_COUNT,
             f"{key} must contain {WORLD_COUNT} world values")
    return [_number(item, f"{key}[{index}]") for index, item in enumerate(value)]


def _joint_values(mapping: dict[str, Any], key: str, count: int) -> list[float]:
    value = mapping.get(key)
    _require(isinstance(value, list) and len(value) == count,
             f"{key} must match joint_columns")
    return [_number(item, f"{key}[{index}]") for index, item in enumerate(value)]


def summarize_case(row: Any) -> dict[str, Any]:
    """Validate and summarize one complete, retained 8-world evaluation case.

    Jensen gaps here mean only ``mean absolute residual - absolute mean
    residual``. They do not reconstruct temporal ordering, oscillation
    frequency, or its cause.
    """
    _require(isinstance(row, dict), "case row must be an object")
    _require(row.get("complete") is True, "case must be a complete first episode")
    _require(_integer(row.get("worlds"), "worlds") == WORLD_COUNT,
             f"case must contain exactly {WORLD_COUNT} worlds")
    _require(_integer(row.get("steps"), "steps") == CASE_STEPS,
             f"case must contain exactly {CASE_STEPS} steps")
    terminals = row.get("terminal_worlds")
    _require(type(terminals) is list and not terminals,
             "complete first episode must have no terminal worlds")
    seed = _integer(row.get("seed"), "seed")
    _require(seed >= 0, "seed must be nonnegative")
    checkpoint = row.get("checkpoint_sha256")
    _require(isinstance(checkpoint, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint),
             "checkpoint_sha256 must be a lowercase SHA-256 digest")
    speed = _number(row.get("speed"), "speed")
    yaw = _number(row.get("yaw"), "yaw")

    speed_mae = _world_values(row, "speed_mae")
    yaw_mae = _world_values(row, "yaw_mae")
    assert speed_mae is not None and yaw_mae is not None
    _require(all(value >= 0 for value in speed_mae + yaw_mae),
             "speed_mae and yaw_mae must be nonnegative")

    diag = row.get("response_diagnostics")
    _require(isinstance(diag, dict), "response_diagnostics must be an object")
    _require(diag.get("protocol") == SOURCE_DIAGNOSTIC_PROTOCOL,
             "unsupported response_diagnostics protocol")
    _require(_integer(diag.get("steps"), "response_diagnostics.steps") == CASE_STEPS
             and _integer(diag.get("startup_steps"), "response_diagnostics.startup_steps") == STARTUP_STEPS
             and _integer(diag.get("settled_steps"), "response_diagnostics.settled_steps") == SETTLED_STEPS,
             "response diagnostic window must be 240 steps with 60 startup and 180 settled")

    names = diag.get("joint_columns")
    _require(isinstance(names, list) and bool(names)
             and all(isinstance(name, str) and name for name in names)
             and len(names) == len(set(names)),
             "joint_columns must contain unique nonempty names")
    joint_arrays = {key: _joint_values(diag, key, len(names)) for key in _JOINT_ARRAYS}
    for key in ("soft_limit_fraction_by_joint", "hard_stop_proximity_fraction_by_joint",
                "hard_range_violation_fraction_by_joint"):
        _require(all(0 <= value <= 1 for value in joint_arrays[key]),
                 f"{key} values must be fractions in [0, 1]")
    _require(diag.get("positions_not_synchronous_with_force") is True,
             "response diagnostic timing identity is missing")
    _require(diag.get("velocity_and_position_timing") ==
             "pre-action body-frame state; startup excluded only for response"
             and diag.get("force_timing") ==
             "post-decimation pre-reset control-step stream; all steps included",
             "response diagnostic sampling frames/timing changed")
    _require(diag.get("policy_acceptance") is False
             and diag.get("physical_motion_authorized") is False,
             "response diagnostic must not claim policy or physical acceptance")
    _require(all(value >= 0 for value in joint_arrays["abs_force_p99_nm_by_joint"]),
             "absolute joint-force p99 values must be nonnegative")

    mean_vx = _world_values(diag, "mean_body_vx_per_world")
    mean_yaw = _world_values(diag, "mean_body_yaw_per_world")
    mean_abs_yaw = _world_values(diag, "mean_abs_body_yaw_per_world")
    sign_fraction = _world_values(diag, "yaw_correct_sign_fraction_per_world", nullable=True)
    assert mean_vx is not None and mean_yaw is not None and mean_abs_yaw is not None
    if yaw == 0:
        _require("yaw_correct_sign_fraction_per_world" in diag and sign_fraction is None,
                 "zero-yaw case must not claim a signed response fraction")
    else:
        _require(sign_fraction is not None,
                 "nonzero-yaw case must include signed response fractions")
        _require(all(0 <= value <= 1 for value in sign_fraction),
                 "yaw_correct_sign_fraction_per_world values must be in [0, 1]")
    _require(all(value >= 0 for value in mean_abs_yaw),
             "mean_abs_body_yaw_per_world must be nonnegative")

    worlds = []
    for index in range(WORLD_COUNT):
        yaw_bias = mean_yaw[index] - yaw
        yaw_lower_bound = abs(yaw_bias)
        _require(yaw_mae[index] + SUMMARY_TOLERANCE >= yaw_lower_bound,
                 f"world {index} yaw_mae is below the absolute mean residual")
        _require(mean_abs_yaw[index] + SUMMARY_TOLERANCE >= abs(mean_yaw[index]),
                 f"world {index} mean absolute yaw is below absolute mean yaw")
        _require(mean_abs_yaw[index] <= abs(yaw) + yaw_mae[index] + SUMMARY_TOLERANCE,
                 f"world {index} mean absolute yaw exceeds command plus yaw_mae")
        _require(yaw_mae[index] <= abs(yaw) + mean_abs_yaw[index] + SUMMARY_TOLERANCE,
                 f"world {index} yaw_mae exceeds command plus mean absolute yaw")

        speed_bias = mean_vx[index] - speed
        speed_lower_bound = abs(speed_bias)
        _require(speed_mae[index] + SUMMARY_TOLERANCE >= speed_lower_bound,
                 f"world {index} speed_mae is below the absolute mean residual")

        worlds.append({
            "world": index,
            "mean_body_yaw": mean_yaw[index],
            "yaw_signed_bias": yaw_bias,
            "yaw_abs_bias_lower_bound": yaw_lower_bound,
            "yaw_mae": yaw_mae[index],
            "yaw_jensen_gap": yaw_mae[index] - yaw_lower_bound,
            "yaw_correct_sign_fraction": None if sign_fraction is None else sign_fraction[index],
            "mean_abs_body_yaw": mean_abs_yaw[index],
            "mean_body_vx": mean_vx[index],
            "speed_signed_bias": speed_bias,
            "speed_abs_bias_lower_bound": speed_lower_bound,
            "speed_mae": speed_mae[index],
            "speed_jensen_gap": speed_mae[index] - speed_lower_bound,
        })

    return {
        "protocol": PROTOCOL,
        "status": "descriptive-audit-only",
        "policy_acceptance": False,
        "physical_motion_authorized": False,
        "case": {"seed": seed, "speed": speed, "yaw": yaw,
                 "checkpoint_sha256": checkpoint, "worlds": WORLD_COUNT,
                 "steps": CASE_STEPS},
        "window": {"startup_steps": STARTUP_STEPS, "settled_steps": SETTLED_STEPS},
        "summary_validation_tolerance": SUMMARY_TOLERANCE,
        "summary_validation_tolerance_is_acceptance_threshold": False,
        "joint_columns": names,
        "worlds": worlds,
        "interpretation_limit": (
            "Jensen gaps are absolute residual beyond constant mean bias; they do not identify "
            "temporal ordering, oscillation frequency, or its cause."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_json", type=Path, help="one retained evaluation case JSON")
    args = parser.parse_args(argv)
    try:
        raw = args.case_json.read_bytes()
        row = json.loads(raw)
        result = summarize_case(row)
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"body response audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
