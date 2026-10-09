"""Synthetic moving-corner observations: no actor, native runtime or physics."""

from dataclasses import replace, fields
import math
import subprocess
import sys

import pytest

from mjlab_microduck.corner_navigation import (
    CornerCfg, CornerObservation, CornerTeacher, Phase, wrap_angle,
)


def observe(t, **changes):
    return replace(CornerObservation(t, 0., 0., 0., .30, 0., 0., 0., True), **changes)


def arc_pose(cfg, progress):
    sign = math.copysign(1., cfg.turn_rad)
    local_x = cfg.entry_length_m + cfg.turn_radius_m * math.sin(progress)
    local_y = sign * cfg.turn_radius_m * (1 - math.cos(progress))
    heading = wrap_angle(cfg.start_heading_rad + sign * progress)
    x = local_x * math.cos(cfg.start_heading_rad) - local_y * math.sin(cfg.start_heading_rad)
    y = local_x * math.sin(cfg.start_heading_rad) + local_y * math.cos(cfg.start_heading_rad)
    return dict(x_m=x, y_m=y, heading_rad=heading,
                velocity_x_mps=.30 * math.cos(heading), velocity_y_mps=.30 * math.sin(heading))


def reach_resume(teacher):
    c = teacher.cfg
    # Synthetic geometry sequence; teleportation is not claimed as dynamics.
    command = teacher.update(observe(0, **arc_pose(c, 0)), now_s=0)
    assert command.phase == Phase.TURN
    for i in range(1, 101):
        angle = abs(c.turn_rad) * i / 100
        command = teacher.update(observe(i * .02, **arc_pose(c, angle)), now_s=i * .02)
    if command.phase == Phase.TURN:  # exact trig endpoint rounding
        command = teacher.update(observe(2.02, **arc_pose(c, abs(c.turn_rad) + 1e-10)), now_s=2.02)
    assert command.phase == Phase.RESUME
    return teacher._last_now


def resumed_observation(teacher, t, *, along=.0, speed=.30, **changes):
    c = teacher.cfg
    x, y = teacher.turn_endpoint()
    heading = wrap_angle(c.start_heading_rad + c.turn_rad)
    base = dict(x_m=x + along * math.cos(heading), y_m=y + along * math.sin(heading),
                heading_rad=heading, velocity_x_mps=speed * math.cos(heading),
                velocity_y_mps=speed * math.sin(heading))
    base.update(changes)
    return observe(t, **base)


@pytest.mark.parametrize("degrees", [-90, -60, -30, 30, 60, 90])
@pytest.mark.parametrize("heading", [0., math.pi - .1, -math.pi + .1])
def test_mirrored_and_rotated_corner_then_measured_speed_dwell(degrees, heading):
    teacher = CornerTeacher(CornerCfg(turn_rad=math.radians(degrees), start_heading_rad=heading))
    t = reach_resume(teacher)
    for i in range(1, 26):
        now = t + .02 * i
        out = teacher.update(resumed_observation(teacher, now), now_s=now)
        assert teacher.recovered_at_s is None  # 25 samples cover only .48s
        assert out.phase == Phase.RESUME
    now = t + .52
    teacher.update(resumed_observation(teacher, now), now_s=now)
    assert teacher.recovered_at_s == pytest.approx(now)
    now += .02
    out = teacher.update(resumed_observation(teacher, now, along=1.), now_s=now)
    assert out.phase == Phase.DONE


def test_commands_are_moving_and_slew_bounded_not_joint_targets():
    teacher = CornerTeacher(CornerCfg())
    previous = (0., 0.)
    for i in range(120):
        now = i * .02
        o = observe(now, **arc_pose(teacher.cfg, min(.5, i * .003)))
        out = teacher.update(o, now_s=now)
        assert 0 <= out.forward_mps <= .30
        assert abs(out.yaw_rps) <= .30
        assert abs(out.forward_mps - previous[0]) <= .004 + 1e-12
        assert abs(out.yaw_rps - previous[1]) <= .012 + 1e-12
        previous = (out.forward_mps, out.yaw_rps)
    assert out.forward_mps == pytest.approx(.30)
    assert out.yaw_rps > 0


def test_undertracking_point21_is_not_recovery():
    teacher = CornerTeacher(CornerCfg())
    t = reach_resume(teacher)
    for i in range(1, 102):
        now = t + .02 * i
        out = teacher.update(resumed_observation(teacher, now, speed=.21), now_s=now)
        if out.phase == Phase.FAILED:
            break
    assert out.failure == "nominal-speed-recovery-deadline"
    assert teacher.recovered_at_s is None


def test_projection_uses_both_world_velocity_components():
    teacher = CornerTeacher(CornerCfg(turn_rad=math.pi / 2))
    t = reach_resume(teacher)
    # 0.30 along old x is not 0.30 along new y.
    for i in range(1, 30):
        now = t + i * .02
        teacher.update(resumed_observation(teacher, now, velocity_x_mps=.30, velocity_y_mps=0.), now_s=now)
    assert teacher.recovered_at_s is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_turn_command_sign_matches_route_not_just_replayed_heading(sign):
    teacher = CornerTeacher(CornerCfg(turn_rad=sign * math.pi / 6))
    out = teacher.update(observe(0, **arc_pose(teacher.cfg, 0)), now_s=0)
    assert out.phase == Phase.TURN and out.yaw_rps * sign > 0
    assert out.forward_mps > 0  # not an in-place turn


def test_timeout_route_radial_escape_and_turn_overshoot():
    teacher = CornerTeacher(CornerCfg())
    for i in range(1001):
        out = teacher.update(observe(i * .02), now_s=i * .02)
    assert out.failure == "attempt-timeout"
    teacher = CornerTeacher(CornerCfg())
    teacher.update(observe(0, **arc_pose(teacher.cfg, 0)), now_s=0)
    out = teacher.update(observe(.02, x_m=1., y_m=-.2), now_s=.02)
    assert out.failure == "turn-route-escape"
    teacher = CornerTeacher(CornerCfg())
    teacher.update(observe(0, **arc_pose(teacher.cfg, 0)), now_s=0)
    out = teacher.update(observe(.02, **arc_pose(teacher.cfg, math.pi / 6 + .2)), now_s=.02)
    assert out.failure == "turn-overshoot-without-alignment"


def test_huge_integer_is_invalid_instead_of_overflowing_validator():
    with pytest.raises(ValueError):
        CornerCfg(entry_length_m=10 ** 1000)
    teacher = CornerTeacher(CornerCfg())
    out = teacher.update(observe(0, x_m=10 ** 1000), now_s=0)
    assert out.failure == "missing-or-invalid-observation"


@pytest.mark.parametrize("late", [False, True])
def test_exact_first_recovery_deadline_cannot_be_evaded_by_late_dwell(late):
    teacher = CornerTeacher(CornerCfg())
    start = reach_resume(teacher)
    for i in range(1, 102):
        now = start + i * .02
        # First good sample at1.50 or1.52, giving a completed span at2.00 or2.02.
        speed = .30 if i >= (76 if late else 75) else .21
        out = teacher.update(resumed_observation(teacher, now, speed=speed), now_s=now)
        if i == 100:
            break
    if late:
        assert out.failure == "nominal-speed-recovery-deadline"
        assert teacher.recovered_at_s is None
    else:
        assert out.phase == Phase.RESUME
        assert teacher.recovered_at_s == pytest.approx(start + 2.)


@pytest.mark.parametrize("sign", [-1, 1])
def test_ideal_kinematic_fixture_walks_from_origin_to_exit(sign):
    # Perfect command following with no mass, contacts, actuators or policy.
    # This checks route-control math only, not MicroDuck simulation performance.
    teacher = CornerTeacher(CornerCfg(turn_rad=sign * math.pi / 6))
    x = y = heading = forward = yaw = 0.
    seen = set()
    for i in range(1000):
        now = i * .02
        o = observe(now, x_m=x, y_m=y, heading_rad=heading,
                    velocity_x_mps=forward * math.cos(heading),
                    velocity_y_mps=forward * math.sin(heading), yaw_rate_rps=yaw)
        out = teacher.update(o, now_s=now)
        seen.add(out.phase)
        assert out.phase != Phase.FAILED, out.failure
        if out.phase == Phase.DONE:
            break
        forward, yaw = out.forward_mps, out.yaw_rps
        heading = wrap_angle(heading + yaw * .02)
        x += forward * math.cos(heading) * .02
        y += forward * math.sin(heading) * .02
    assert out.phase == Phase.DONE
    assert seen == {Phase.APPROACH, Phase.TURN, Phase.RESUME, Phase.DONE}


def test_bad_sample_breaks_dwell_and_later_loss_cannot_finish_immediately():
    teacher = CornerTeacher(CornerCfg())
    t = reach_resume(teacher)
    for i in range(1, 27):
        now = t + i * .02
        teacher.update(resumed_observation(teacher, now), now_s=now)
    assert teacher.recovered_at_s is not None
    now += .02
    teacher.update(resumed_observation(teacher, now, speed=.21), now_s=now)
    now += .02
    out = teacher.update(resumed_observation(teacher, now, along=1.), now_s=now)
    assert out.phase == Phase.RESUME


@pytest.mark.parametrize("change,reason", [
    ({"collision": True}, "fall-collision-or-support-loss"),
    ({"fallen": True}, "fall-collision-or-support-loss"),
    ({"grounded": False}, "fall-collision-or-support-loss"),
    ({"tilt_rad": .1}, "tilt-limit"),
    ({"y_m": .2}, "route-escape"),
    ({"x_m": 2.}, "approach-position-escape"),
    ({"heading_rad": float("nan")}, "missing-or-invalid-observation"),
    ({"collision": 1}, "missing-or-invalid-observation"),
    ({"velocity_y_mps": 6.}, "out-of-envelope-observation"),
    ({"timestamp_s": -.1}, "stale-or-future-observation"),
    ({"timestamp_s": .1}, "stale-or-future-observation"),
])
def test_first_attempt_failures_are_latched(change, reason):
    teacher = CornerTeacher(CornerCfg())
    out = teacher.update(observe(0, **change), now_s=0)
    assert out.phase == Phase.FAILED and out.failure == reason
    out = teacher.update(observe(.02), now_s=.02)
    assert out.phase == Phase.FAILED and out.forward_mps == out.yaw_rps == 0


def test_missing_stale_duplicate_and_gap_fail_without_reset_or_large_slew():
    for kind in ("missing", "stale", "duplicate", "gap"):
        teacher = CornerTeacher(CornerCfg())
        for i in range(80):
            teacher.update(observe(i * .02), now_s=i * .02)
        old = teacher._forward
        now = 1.60 if kind != "gap" else 10.
        o = None if kind == "missing" else observe(now if kind == "gap" else 1.5 if kind == "stale" else 1.58)
        out = teacher.update(o, now_s=now)
        assert out.phase == Phase.FAILED
        assert 0 < old - out.forward_mps <= .01 + 1e-12


@pytest.mark.parametrize("field", [f.name for f in fields(CornerCfg)])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_bad_config_numbers_rejected(field, value):
    with pytest.raises(ValueError):
        CornerCfg(**{field: value})


@pytest.mark.parametrize("changes", [{"turn_rad": 0}, {"turn_rad": math.pi},
    {"nominal_speed_mps": .31}, {"turn_radius_m": .5}, {"max_yaw_rate_rps": .6},
    {"route_tolerance_m": .2}, {"dwell_s": .49}, {"recovery_deadline_s": 3},
    {"timeout_s": 21}, {"max_update_gap_s": .1}, {"entry_length_m": 10},
    {"max_speed_slew_mps2": .3}, {"max_yaw_slew_rps2": 1}, {"tilt_limit_rad": .2},
    {"heading_lookahead_rad": 100}, {"timeout_s": 2.01}])
def test_unsupported_envelope_rejected(changes):
    with pytest.raises(ValueError):
        CornerCfg(**changes)


def test_reversed_controller_time_raises_without_state_change():
    teacher = CornerTeacher(CornerCfg())
    teacher.update(observe(.1), now_s=.1)
    before = teacher.__dict__.copy()
    with pytest.raises(ValueError):
        teacher.update(observe(.1), now_s=.1)
    assert teacher.__dict__ == before


def test_module_imports_no_simulator_or_tensor_runtime():
    code = "import sys; import mjlab_microduck.corner_navigation; assert not {'torch','warp','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
