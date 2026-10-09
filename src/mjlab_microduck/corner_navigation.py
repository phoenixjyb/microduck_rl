"""Unwired moving-corner teacher; commands, not joint control or safety proof.

Pose is externally supplied in a fixed local XY route frame whose origin is
the start. The locomotion actor's observations and weights are not changed.
Command slew bounds are not bounds on measured robot acceleration or torque.
"""

from dataclasses import dataclass, fields
from enum import Enum
import math


def _number(value):
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        return False


def wrap_angle(angle):
    if not _number(angle):
        raise ValueError("finite angle required")
    return math.atan2(math.sin(angle), math.cos(angle))


class Phase(str, Enum):
    APPROACH = "approach"
    TURN = "turn"
    RESUME = "resume"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class CornerCfg:
    # Proposed diagnostic limits, not tuned or physically calibrated values.
    nominal_speed_mps: float = 0.30
    start_heading_rad: float = 0.0
    turn_rad: float = math.pi / 6
    entry_length_m: float = 1.0
    exit_length_m: float = 1.0
    turn_radius_m: float = 1.0
    heading_lookahead_rad: float = 0.05
    route_tolerance_m: float = 0.15
    heading_tolerance_rad: float = math.radians(5)
    tilt_limit_rad: float = math.radians(5)
    settled_yaw_rate_rps: float = 0.05
    speed_tolerance_mps: float = 0.03
    dwell_s: float = 0.50
    recovery_deadline_s: float = 2.0
    timeout_s: float = 20.0
    max_yaw_rate_rps: float = 0.30
    yaw_gain: float = 1.5
    lateral_gain: float = 1.0
    max_speed_slew_mps2: float = 0.20
    max_yaw_slew_rps2: float = 0.60
    initial_dt_s: float = 0.02
    max_update_gap_s: float = 0.05
    max_observation_age_s: float = 0.05

    def __post_init__(self):
        if not all(_number(getattr(self, f.name)) for f in fields(self)):
            raise ValueError("finite numeric configuration required")
        if any(getattr(self, f.name) <= 0 for f in fields(self)
               if f.name not in ("start_heading_rad", "turn_rad")):
            raise ValueError("positive limits required")
        if not (0 < abs(self.turn_rad) <= math.pi / 2
                and self.nominal_speed_mps <= 0.30
                and self.max_yaw_rate_rps <= 0.30
                and self.heading_lookahead_rad <= min(abs(self.turn_rad), 0.10)
                and self.nominal_speed_mps / self.turn_radius_m <= self.max_yaw_rate_rps
                and self.route_tolerance_m < self.exit_length_m
                and self.speed_tolerance_mps < self.nominal_speed_mps
                and self.initial_dt_s <= self.max_update_gap_s
                and abs(self.start_heading_rad) <= math.pi
                and self.max_speed_slew_mps2 <= 0.20
                and self.max_yaw_slew_rps2 <= 0.60
                and self.yaw_gain <= 10 and self.lateral_gain <= 10
                and self.max_update_gap_s <= 0.05
                and self.max_observation_age_s <= 0.05
                and max(self.entry_length_m, self.exit_length_m, self.turn_radius_m) <= 5
                and self.route_tolerance_m <= 0.15
                and self.heading_tolerance_rad <= math.radians(5)
                and self.tilt_limit_rad <= math.radians(5)
                and self.speed_tolerance_mps <= 0.03
                and self.settled_yaw_rate_rps <= 0.05
                and self.dwell_s >= 0.50 and self.recovery_deadline_s <= 2
                and (self.entry_length_m + self.exit_length_m
                     + self.turn_radius_m * abs(self.turn_rad)) / self.nominal_speed_mps
                     + self.dwell_s < self.timeout_s <= 20
                and self.dwell_s < self.recovery_deadline_s < self.timeout_s):
            raise ValueError("unsupported corner envelope")


@dataclass(frozen=True)
class CornerObservation:
    timestamp_s: float
    x_m: float
    y_m: float
    heading_rad: float
    velocity_x_mps: float
    velocity_y_mps: float
    yaw_rate_rps: float
    tilt_rad: float
    grounded: bool
    fallen: bool = False
    collision: bool = False


@dataclass(frozen=True)
class CornerCommand:
    phase: Phase
    forward_mps: float
    yaw_rps: float
    failure: str | None


class CornerTeacher:
    """Single first-attempt teacher. New instance means an explicit new attempt.

    Repeated/backward observations, missing data, excessive update gaps, lost
    support, collision/fall, route escape and timeout fail the attempt permanently.
    Failure requests a slew-limited zero command; an independent safety layer
    must handle actual stopping, unsupported/airborne states and motor limits.
    ``DONE`` is teacher sequence completion, never a scored policy acceptance.
    """

    def __init__(self, cfg: CornerCfg):
        if type(cfg) is not CornerCfg:
            raise ValueError("explicit CornerCfg required")
        self.cfg = cfg
        self.phase = Phase.APPROACH
        self.failure = None
        self._last_now = self._last_observation = self._start = self._dwell = None
        self._recovery_start = None
        self.recovered_at_s = None
        self._forward = self._yaw = 0.0

    def _fail(self, reason):
        self.phase = Phase.FAILED
        self.failure = reason
        self._dwell = None

    def _held(self, condition, now):
        if not condition:
            self._dwell = None
            return False
        if self._dwell is None:
            self._dwell = now
        return now - self._dwell >= self.cfg.dwell_s - 1e-12

    def _transition(self, phase):
        self.phase = phase
        self._dwell = None

    def update(self, observation: CornerObservation | None, *, now_s: float):
        if not _number(now_s) or now_s < 0 or (
            self._last_now is not None and now_s <= self._last_now
        ):
            raise ValueError("strictly increasing nonnegative controller time required")
        c = self.cfg
        dt = c.initial_dt_s if self._last_now is None else now_s - self._last_now
        if self._start is None:
            self._start = now_s
        self._last_now = now_s
        if self.phase not in (Phase.FAILED, Phase.DONE):
            numeric = ("timestamp_s", "x_m", "y_m", "heading_rad", "velocity_x_mps",
                       "velocity_y_mps", "yaw_rate_rps", "tilt_rad")
            valid = (type(observation) is CornerObservation
                     and all(_number(getattr(observation, n)) for n in numeric)
                     and all(type(getattr(observation, n)) is bool
                             for n in ("grounded", "fallen", "collision")))
            if dt > c.max_update_gap_s + 1e-12:
                self._fail("controller-update-gap")
            elif not valid:
                self._fail("missing-or-invalid-observation")
            elif not (0 <= observation.timestamp_s <= now_s
                      and now_s - observation.timestamp_s <= c.max_observation_age_s + 1e-12):
                self._fail("stale-or-future-observation")
            elif self._last_observation is not None and observation.timestamp_s <= self._last_observation:
                self._fail("non-increasing-observation")
            elif (abs(observation.x_m) > 20 or abs(observation.y_m) > 20
                  or abs(observation.heading_rad) > math.pi
                  or abs(observation.velocity_x_mps) > 5 or abs(observation.velocity_y_mps) > 5
                  or abs(observation.yaw_rate_rps) > 10 or not 0 <= observation.tilt_rad <= math.pi / 2):
                self._fail("out-of-envelope-observation")
            elif observation.fallen or observation.collision or not observation.grounded:
                self._fail("fall-collision-or-support-loss")
            elif observation.tilt_rad > c.tilt_limit_rad:
                self._fail("tilt-limit")
            elif now_s - self._start >= c.timeout_s:
                self._fail("attempt-timeout")
            else:
                self._last_observation = observation.timestamp_s
                self._advance(observation, now_s)
        # A long missing interval does not enlarge permitted command deltas.
        slew_dt = min(dt, c.max_update_gap_s)
        forward, yaw = self._targets(observation)
        self._forward += max(-c.max_speed_slew_mps2 * slew_dt,
                             min(c.max_speed_slew_mps2 * slew_dt, forward - self._forward))
        self._yaw += max(-c.max_yaw_slew_rps2 * slew_dt,
                         min(c.max_yaw_slew_rps2 * slew_dt, yaw - self._yaw))
        return CornerCommand(self.phase, self._forward, self._yaw, self.failure)

    def _geometry(self, o):
        c = self.cfg
        heading = c.start_heading_rad if self.phase == Phase.APPROACH else c.start_heading_rad + c.turn_rad
        x, y = o.x_m, o.y_m
        if self.phase != Phase.APPROACH:
            end_x, end_y = self.turn_endpoint()
            x -= end_x
            y -= end_y
        along = x * math.cos(heading) + y * math.sin(heading)
        lateral = -x * math.sin(heading) + y * math.cos(heading)
        return along, lateral, wrap_angle(heading - o.heading_rad)

    def turn_endpoint(self):
        """End of the declared circular arc, in externally supplied local XY."""
        c = self.cfg
        sign = math.copysign(1.0, c.turn_rad)
        along = c.entry_length_m + c.turn_radius_m * math.sin(abs(c.turn_rad))
        lateral = sign * c.turn_radius_m * (1 - math.cos(c.turn_rad))
        return (along * math.cos(c.start_heading_rad) - lateral * math.sin(c.start_heading_rad),
                along * math.sin(c.start_heading_rad) + lateral * math.cos(c.start_heading_rad))

    def _arc(self, o):
        c = self.cfg
        sign = math.copysign(1.0, c.turn_rad)
        cx = c.entry_length_m * math.cos(c.start_heading_rad) - sign * c.turn_radius_m * math.sin(c.start_heading_rad)
        cy = c.entry_length_m * math.sin(c.start_heading_rad) + sign * c.turn_radius_m * math.cos(c.start_heading_rad)
        radial = math.hypot(o.x_m - cx, o.y_m - cy) - c.turn_radius_m
        progress = sign * wrap_angle(math.atan2(o.y_m - cy, o.x_m - cx)
                                     - (c.start_heading_rad - sign * math.pi / 2))
        return progress, radial

    def _advance(self, o, now):
        c = self.cfg
        along, lateral, error = self._geometry(o)
        if self.phase in (Phase.APPROACH, Phase.RESUME) and abs(lateral) > c.route_tolerance_m:
            self._fail("route-escape")
        elif self.phase == Phase.APPROACH:
            if along < -c.route_tolerance_m or along > c.entry_length_m + c.route_tolerance_m:
                self._fail("approach-position-escape")
            elif along >= c.entry_length_m - 1e-12:
                self._transition(Phase.TURN)
        elif self.phase == Phase.TURN:
            progress, radial = self._arc(o)
            if abs(radial) > c.route_tolerance_m or progress < -c.heading_tolerance_rad:
                self._fail("turn-route-escape")
            elif progress > abs(c.turn_rad) + c.heading_tolerance_rad:
                self._fail("turn-overshoot-without-alignment")
            elif progress >= abs(c.turn_rad) and abs(error) <= c.heading_tolerance_rad:
                self._transition(Phase.RESUME)
                self._recovery_start = now
        elif self.phase == Phase.RESUME:
            if along > c.exit_length_m + c.route_tolerance_m:
                self._fail("exit-overshoot-without-recovery")
                return
            # Route-projected measured speed, not command or body speed alone.
            route_speed = self._speed(o, c.start_heading_rad + c.turn_rad)
            stable = self._held(
                abs(error) <= c.heading_tolerance_rad
                and abs(route_speed - c.nominal_speed_mps) <= c.speed_tolerance_mps
                and abs(o.yaw_rate_rps) <= c.settled_yaw_rate_rps, now)
            if (self.recovered_at_s is None and stable
                    and now - self._recovery_start <= c.recovery_deadline_s + 1e-12):
                self.recovered_at_s = now
            if (self.recovered_at_s is None
                    and now - self._recovery_start >= c.recovery_deadline_s - 1e-12):
                self._fail("nominal-speed-recovery-deadline")
            elif along >= c.exit_length_m and stable:
                self._transition(Phase.DONE)

    def _targets(self, o):
        if self.phase in (Phase.FAILED, Phase.DONE):
            return 0.0, 0.0
        _, lateral, error = self._geometry(o)
        yaw = self.cfg.yaw_gain * error
        if self.phase == Phase.TURN:
            progress, radial = self._arc(o)
            sign = math.copysign(1.0, self.cfg.turn_rad)
            target = self.cfg.start_heading_rad + sign * min(abs(self.cfg.turn_rad),
                         max(0.0, progress) + self.cfg.heading_lookahead_rad)
            yaw = (sign * max(0.0, self._speed(o, o.heading_rad)) / self.cfg.turn_radius_m
                   + self.cfg.yaw_gain * wrap_angle(target - o.heading_rad)
                   + sign * self.cfg.lateral_gain * radial)
        else:
            yaw -= self.cfg.lateral_gain * lateral
        yaw = max(-self.cfg.max_yaw_rate_rps, min(self.cfg.max_yaw_rate_rps, yaw))
        return self.cfg.nominal_speed_mps, yaw

    @staticmethod
    def _speed(o, heading):
        return o.velocity_x_mps * math.cos(heading) + o.velocity_y_mps * math.sin(heading)
