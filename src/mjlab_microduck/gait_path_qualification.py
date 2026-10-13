"""CPU-only, gait-aware world-path qualification for retained MicroDuck traces.

This prospective qualification adds measured route and support-cycle checks to
the existing held-out cases. Instantaneous body-frame yaw remains a retained
diagnostic and is not substituted for projected-body-x path heading.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from mjlab_microduck import body_yaw_trace, gait_yaw_trace


PROTOCOL = "microduck-gait-path-qualification-v1"
TRACE_PROTOCOL = "microduck-gait-path-trace-v1"
STEPS = gait_yaw_trace.STEPS
WORLDS = gait_yaw_trace.WORLDS
STARTUP_STEPS = gait_yaw_trace.STARTUP_STEPS
STEP_DT = gait_yaw_trace.STEP_DT
MIN_CYCLES = 3
MIN_CYCLE_S = 0.2
MAX_CYCLE_S = 1.5
DEBOUNCE_FRAMES = 2
HEADING_AXIS_MIN_NORM_SQ = 1e-4
MOVING_XY_ERROR_MPS_LIMIT = 0.1
IDLE_XY_ERROR_MPS_LIMIT = 0.03
HEADING_ERROR_RAD_S_LIMIT = 0.1
TORQUE_UTIL_P99_LIMIT = 0.60
RATED_SPEED_EXCEED_FRACTION_LIMIT = 0.0
MIN_MOTOR_HINGE_MARGIN_RAD = -0.005
RIGHT_HIP_YAW_NEAR_STOP_FRACTION_LIMIT = 0.01
MOTOR_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
IDENTITIES = tuple((seed, *command) for seed in gait_yaw_trace.EVALUATION_SEEDS
                   for command in gait_yaw_trace.EVALUATION_CASES)


def _thresholds() -> dict:
    return dict(min_complete_cycles=MIN_CYCLES,
                cycle_duration_s=[MIN_CYCLE_S, MAX_CYCLE_S],
                contact_debounce_frames=DEBOUNCE_FRAMES,
                moving_xy_error_mps=MOVING_XY_ERROR_MPS_LIMIT,
                idle_xy_error_mps=IDLE_XY_ERROR_MPS_LIMIT,
                heading_error_rad_s=HEADING_ERROR_RAD_S_LIMIT,
                projected_body_x_min_norm_sq=HEADING_AXIS_MIN_NORM_SQ)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_array(value: object, shape: tuple[int, ...], name: str) -> None:
    _require(type(value) is list and len(value) == shape[0],
             f"{name} must have shape {shape}")
    if len(shape) == 1:
        _require(all(_number(item) for item in value), f"{name} must be finite numbers")
    else:
        for row in value:
            _validate_array(row, shape[1:], name)


def _rotate_wxyz(quaternion: list[float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (vx + 2.0 * (w * cx + dx),
            vy + 2.0 * (w * cy + dy),
            vz + 2.0 * (w * cz + dz))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _heading(quaternion: list[float]) -> float:
    axis_x, axis_y, _ = _rotate_wxyz(quaternion, (1.0, 0.0, 0.0))
    _require(axis_x * axis_x + axis_y * axis_y >= HEADING_AXIS_MIN_NORM_SQ,
             "projected body-x heading is undefined")
    return math.atan2(axis_y, axis_x)


def _arc_displacement(speed: float, yaw_rate: float, duration: float) -> tuple[float, float]:
    if yaw_rate == 0.0:
        return speed * duration, 0.0
    return (speed / yaw_rate * math.sin(yaw_rate * duration),
            speed / yaw_rate * (1.0 - math.cos(yaw_rate * duration)))


def _path_error(trace: Mapping, world: int, start: int, end: int,
                speed: float, yaw_rate: float) -> dict:
    duration = trace["physics_time"][end][world] - trace["physics_time"][start][world]
    _require(duration > 0.0, "path interval duration must be positive")
    headings = [_heading(trace["quat_wxyz"][step][world]) for step in range(start, end + 1)]
    unwrapped_headings = [headings[0]]
    for before, after in zip(headings, headings[1:]):
        unwrapped_headings.append(unwrapped_headings[-1] + _wrap(after - before))
    start_heading = unwrapped_headings[0]
    heading_delta = unwrapped_headings[-1] - start_heading
    start_xy = trace["root_position_w"][start][world][:2]
    xy_errors = []
    heading_errors = []
    start_time = trace["physics_time"][start][world]
    for offset, step in enumerate(range(start, end + 1)):
        elapsed = trace["physics_time"][step][world] - start_time
        local_x, local_y = _arc_displacement(speed, yaw_rate, elapsed)
        expected_x = math.cos(start_heading) * local_x - math.sin(start_heading) * local_y
        expected_y = math.sin(start_heading) * local_x + math.cos(start_heading) * local_y
        actual_xy = trace["root_position_w"][step][world][:2]
        xy_errors.append(math.hypot(actual_xy[0] - start_xy[0] - expected_x,
                                    actual_xy[1] - start_xy[1] - expected_y))
        heading_errors.append((unwrapped_headings[offset] - start_heading) - yaw_rate * elapsed)
    xy_error_m = xy_errors[-1]
    heading_error_rad = heading_errors[-1]
    max_xy_error_m = max(xy_errors)
    max_heading_error_rad = max(abs(value) for value in heading_errors)
    return dict(start_step=start, end_step=end, duration_s=duration,
                xy_error_m=xy_error_m, xy_error_mps=xy_error_m / duration,
                max_xy_error_m=max_xy_error_m, max_xy_error_mps=max_xy_error_m / duration,
                heading_delta_rad=heading_delta,
                heading_delta_wrapped_rad=_wrap(heading_delta),
                expected_heading_delta_rad=yaw_rate * duration,
                heading_error_rad=heading_error_rad,
                heading_error_rad_s=abs(heading_error_rad) / duration,
                max_heading_error_rad=max_heading_error_rad,
                max_heading_error_rad_s=max_heading_error_rad / duration)


def _debounced_rising_edges(values: list[float], first_step: int) -> list[int]:
    """Return first sample of stable positive runs preceded by stable flight."""
    touching = [value > 0.0 for value in values]
    runs: list[tuple[bool, int, int]] = []
    begin = 0
    for index in range(1, len(touching) + 1):
        if index == len(touching) or touching[index] != touching[begin]:
            runs.append((touching[begin], begin, index - begin))
            begin = index
    edges = []
    for previous, current in zip(runs, runs[1:]):
        previous_contact, _, previous_length = previous
        current_contact, current_start, current_length = current
        if (not previous_contact and previous_length >= DEBOUNCE_FRAMES
                and current_contact and current_length >= DEBOUNCE_FRAMES):
            edges.append(first_step + current_start)
    return edges


def _support_timer_failures(trace: Mapping, world: int) -> list[str]:
    failures = []
    for step in range(1, STEPS):
        for foot in range(2):
            touching = trace["contact_found"][step][world][foot] > 0.0
            air_time = trace["current_air_time"][step][world][foot]
            contact_time = trace["current_contact_time"][step][world][foot]
            if touching and not (contact_time > 0.0 and air_time == 0.0):
                failures.append(f"contact-timer-disagreement-step-{step}-foot-{foot}")
            elif not touching and not (air_time > 0.0 and contact_time == 0.0):
                failures.append(f"air-timer-disagreement-step-{step}-foot-{foot}")
    return failures


def _validate_trace(case: Mapping, trace: Mapping) -> None:
    # The prospective trace is the old gait trace plus root position. Validate
    # both historical identities through their existing read-only validators.
    _require(isinstance(case, Mapping) and isinstance(trace, Mapping),
             "case and path trace must be mappings")
    _require(trace.get("protocol") == TRACE_PROTOCOL,
             "unsupported gait path trace protocol")
    gait_trace = dict(trace)
    gait_trace["protocol"] = gait_yaw_trace.PROTOCOL
    gait_yaw_trace.summarize_trace(case, gait_trace)
    body_trace = {
        "protocol": body_yaw_trace.PROTOCOL,
        "step_dt": trace.get("step_dt"),
        "startup_steps": trace.get("startup_steps"),
        "steps": trace.get("steps"),
        "worlds": trace.get("worlds"),
        "checkpoint_sha256": trace.get("checkpoint_sha256"),
        "timing": trace.get("timing"),
        "policy_acceptance": trace.get("policy_acceptance"),
        "physical_motion_authorized": trace.get("physical_motion_authorized"),
        **{key: trace[key] for key in
           ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")},
    }
    body_yaw_trace.summarize_trace(case, body_trace)
    _require(trace.get("step_dt") == STEP_DT and trace.get("startup_steps") == STARTUP_STEPS
             and trace.get("steps") == STEPS and trace.get("worlds") == WORLDS
             and trace.get("checkpoint_sha256") == case.get("checkpoint_sha256")
             and trace.get("timing") == gait_yaw_trace.TIMING
             and trace.get("contact_columns") == list(gait_yaw_trace.CONTACT_COLUMNS)
             and trace.get("foot_sites") == list(gait_yaw_trace.FOOT_SITES)
             and trace.get("head_joints") == list(gait_yaw_trace.HEAD_JOINTS)
             and trace.get("force_timing") == gait_yaw_trace.FORCE_TIMING
             and trace.get("force_frame") == "world"
             and trace.get("terrain") == "stationary-flat"
             and trace.get("position_frame") == "world"
             and trace.get("position_measurement") ==
             "root-link world position, not integrated velocity"
             and trace.get("policy_acceptance") is False
             and trace.get("physical_motion_authorized") is False,
             "path trace metadata must match the retained gait trace")
    _validate_array(trace.get("root_position_w"), (STEPS, WORLDS, 3), "root_position_w")
    for step in range(STEPS):
        for world in range(WORLDS):
            quaternion = trace["quat_wxyz"][step][world]
            norm = math.hypot(*quaternion)
            _require(abs(norm - 1.0) <= 1e-5, "path trace quaternion must be unit length")
            axis_x, axis_y, _ = _rotate_wxyz(quaternion, (1.0, 0.0, 0.0))
            _require(axis_x * axis_x + axis_y * axis_y >= HEADING_AXIS_MIN_NORM_SQ,
                     "projected body-x heading is undefined")


def _world_report(case: Mapping, trace: Mapping, world: int) -> dict:
    speed, yaw_rate = case["speed"], case["yaw"]
    idle = speed == 0.0 and yaw_rate == 0.0
    failures = _support_timer_failures(trace, world)
    left_edges = _debounced_rising_edges(
        [trace["contact_found"][step][world][0] for step in range(STEPS)], 0)
    right_edges = _debounced_rising_edges(
        [trace["contact_found"][step][world][1] for step in range(STEPS)], 0)
    left_edges = [edge for edge in left_edges if edge >= STARTUP_STEPS]
    right_edges = [edge for edge in right_edges if edge >= STARTUP_STEPS]
    cycles = []
    if idle:
        cycle_status = "not-applicable-stationary-command"
    else:
        cycle_status = "qualified"
        if len(left_edges) < MIN_CYCLES + 1:
            failures.append("fewer-than-three-complete-left-cycles")
        for start, end in zip(left_edges, left_edges[1:]):
            duration = trace["physics_time"][end][world] - trace["physics_time"][start][world]
            interior_right = [edge for edge in right_edges if start < edge < end]
            pair_failures = []
            if not MIN_CYCLE_S <= duration <= MAX_CYCLE_S:
                pair_failures.append("cycle-duration-out-of-range")
            if len(interior_right) != 1:
                pair_failures.append("cycle-must-contain-exactly-one-right-touchdown")
            error = _path_error(trace, world, start, end, speed, yaw_rate)
            if (error["xy_error_mps"] > MOVING_XY_ERROR_MPS_LIMIT
                    or error["max_xy_error_mps"] > MOVING_XY_ERROR_MPS_LIMIT):
                pair_failures.append("cycle-xy-error-rate")
            if (error["heading_error_rad_s"] > HEADING_ERROR_RAD_S_LIMIT
                    or error["max_heading_error_rad_s"] > HEADING_ERROR_RAD_S_LIMIT):
                pair_failures.append("cycle-heading-error-rate")
            cycles.append({**error, "right_touchdown_steps": interior_right,
                           "qualified": not pair_failures, "failures": pair_failures})
            failures.extend(f"{reason}-steps-{start}-{end}" for reason in pair_failures)
        if any(not item["qualified"] for item in cycles):
            cycle_status = "rejected-unsupported-cycle"
        valid_cycle_count = sum(item["qualified"] for item in cycles)
        if valid_cycle_count < MIN_CYCLES and "fewer-than-three-complete-left-cycles" not in failures:
            failures.append("fewer-than-three-valid-cycles")
    if idle:
        valid_cycle_count = 0

    full = _path_error(trace, world, STARTUP_STEPS, STEPS - 1, speed, yaw_rate)
    xy_limit = IDLE_XY_ERROR_MPS_LIMIT if idle else MOVING_XY_ERROR_MPS_LIMIT
    if full["xy_error_mps"] > xy_limit or full["max_xy_error_mps"] > xy_limit:
        failures.append("full-settled-xy-error-rate")
    if (full["heading_error_rad_s"] > HEADING_ERROR_RAD_S_LIMIT
            or full["max_heading_error_rad_s"] > HEADING_ERROR_RAD_S_LIMIT):
        failures.append("full-settled-heading-error-rate")
    if failures and cycle_status == "qualified":
        cycle_status = "rejected-support-or-path"
    return dict(world=world, idle=idle, cycle_status=cycle_status, left_touchdown_steps=left_edges,
                right_touchdown_steps=right_edges, cycle_count=len(cycles),
                valid_cycle_count=valid_cycle_count, cycles=cycles,
                full_settled=full, qualified=not failures, failures=failures)


def _safety_gates(case: Mapping) -> dict:
    failures = []
    torque = case.get("torque_p99")
    rated = case.get("rated_speed_exceed_fraction")
    _require(_number(torque) and _number(rated), "finite motor utilization and rated-speed metrics")
    _require(torque >= 0.0 and 0.0 <= rated <= 1.0,
             "motor utilization and rated-speed fraction are nonnegative bounded metrics")
    if torque > TORQUE_UTIL_P99_LIMIT:
        failures.append("torque-utilization-p99")
    if rated != RATED_SPEED_EXCEED_FRACTION_LIMIT:
        failures.append("rated-speed-exceedance")
    response = case.get("response_diagnostics")
    _require(isinstance(response, Mapping), "motor response diagnostics are required")
    columns = response.get("joint_columns")
    margins = response.get("min_distance_to_hard_stop_rad_by_joint")
    near_stop = response.get("hard_stop_proximity_fraction_by_joint")
    _require(type(columns) is list and tuple(columns) == MOTOR_JOINTS
             and type(margins) is list and len(margins) == len(MOTOR_JOINTS)
             and all(_number(value) for value in margins)
             and type(near_stop) is list and len(near_stop) == len(MOTOR_JOINTS)
             and all(_number(value) and 0.0 <= value <= 1.0 for value in near_stop),
             "named 14-joint hard-stop response diagnostics are required")
    minimum_margin = min(margins)
    right_yaw_index = MOTOR_JOINTS.index("right_hip_yaw")
    right_yaw_near_stop = near_stop[right_yaw_index]
    if minimum_margin < MIN_MOTOR_HINGE_MARGIN_RAD:
        failures.append("motor-hinge-margin")
    if right_yaw_near_stop > RIGHT_HIP_YAW_NEAR_STOP_FRACTION_LIMIT:
        failures.append("right-hip-yaw-near-stop")
    return dict(qualified=not failures, failures=failures,
                torque_utilization_p99=torque, rated_speed_exceed_fraction=rated,
                minimum_motor_hinge_margin_rad=minimum_margin,
                right_hip_yaw_near_stop_fraction=right_yaw_near_stop,
                no_fall=case.get("complete") is True and case.get("terminal_worlds") == [])


def summarize_case(case: Mapping, trace: Mapping) -> dict:
    """Validate and summarize one 8-world path trace without changing inputs."""
    _validate_trace(case, trace)
    speed_mae = case.get("speed_mae")
    _validate_array(speed_mae, (WORLDS,), "case speed_mae")
    _require(all(value >= 0.0 for value in speed_mae), "case speed_mae values must be nonnegative")
    idle = case["speed"] == 0.0 and case["yaw"] == 0.0
    safety = _safety_gates(case)
    if not safety["no_fall"]:
        safety["failures"].append("fall-or-terminal")
        safety["qualified"] = False
    if idle:
        yaw_mae = case["yaw_mae"]
        if any(value > 0.1 for value in yaw_mae):
            safety["failures"].append("idle-body-yaw-mae")
            safety["qualified"] = False
        if any(value > 0.03 for value in speed_mae):
            safety["failures"].append("idle-speed-mae")
            safety["qualified"] = False
    worlds = [_world_report(case, trace, world) for world in range(WORLDS)]
    failures = []
    for item in worlds:
        failures.extend([f"world-{item['world']}:{reason}" for reason in item["failures"]])
    failures.extend(safety["failures"])
    return dict(protocol=PROTOCOL, trace_protocol=TRACE_PROTOCOL,
                status="route-qualified-simulation-only" if not failures
                else "route-not-qualified",
                seed=case["seed"], speed=case["speed"], yaw=case["yaw"],
                checkpoint_sha256=case["checkpoint_sha256"], qualified=not failures,
                worlds=worlds, safety=safety, failures=failures,
                original_yaw_mae_diagnostic=list(case["yaw_mae"]),
                thresholds=_thresholds(),
                policy_acceptance=False, physical_motion_authorized=False)


def decision(rows: list[Mapping]) -> dict:
    """Aggregate the exact 12 held-out case summaries in their declared order."""
    _require(type(rows) is list and len(rows) == len(IDENTITIES),
             "exactly twelve ordered held-out route summaries are required")
    failures = []
    checkpoint_sha256 = None
    for row, (seed, speed, yaw) in zip(rows, IDENTITIES, strict=True):
        _require(isinstance(row, Mapping)
                 and (row.get("seed"), row.get("speed"), row.get("yaw")) == (seed, speed, yaw),
                 "ordered held-out route identities")
        _require(row.get("protocol") == PROTOCOL
                 and row.get("policy_acceptance") is False
                 and row.get("physical_motion_authorized") is False
                 and row.get("status") in ("route-qualified-simulation-only", "route-not-qualified")
                 and type(row.get("worlds")) is list and len(row["worlds"]) == WORLDS
                 and row.get("thresholds") == _thresholds()
                 and _finite_summary(row), "route thresholds and finite non-authorizing summary")
        digest = row.get("checkpoint_sha256")
        _require(type(digest) is str and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 "lowercase full checkpoint SHA256 required")
        if checkpoint_sha256 is None:
            checkpoint_sha256 = digest
        _require(digest == checkpoint_sha256, "all route summaries must use one checkpoint")
        _require(type(row.get("safety")) is dict and type(row["safety"].get("qualified")) is bool
                 and type(row["safety"].get("no_fall")) is bool
                 and type(row["safety"].get("failures")) is list
                 and all(type(item) is dict and item.get("world") == world
                         and type(item.get("qualified")) is bool
                         and type(item.get("failures")) is list
                         for world, item in enumerate(row["worlds"])),
                 "complete per-world and safety gate summaries")
        _require(type(row.get("qualified")) is bool and type(row.get("failures")) is list
                 and all(type(reason) is str for reason in row["failures"]),
                 "top-level qualification and failure summary required")
        _require(row["safety"]["no_fall"]
                 and row["safety"]["qualified"] == (not row["safety"].get("failures"))
                 and all(item["qualified"] == (not item.get("failures"))
                         for item in row["worlds"]),
                 "safety and world pass flags must match their failure lists")
        expected_qualified = row["safety"]["qualified"] and all(
            item["qualified"] for item in row["worlds"])
        _require(row["qualified"] is expected_qualified
                 and expected_qualified == (not row["failures"])
                 and (row["status"] == "route-qualified-simulation-only") == expected_qualified,
                 "route status must match every world and safety gate")
        if row["status"] != "route-qualified-simulation-only":
            failures.append([seed, speed, yaw, "route-not-qualified"])
    return dict(decision="route-qualified-simulation-only" if not failures else "route-not-qualified",
                failures=failures, cases=len(rows), policy_acceptance=False,
                physical_motion_authorized=False)


def _finite_summary(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_summary(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_summary(item) for item in value)
    return value is None or type(value) in (str, bool) or _number(value)
