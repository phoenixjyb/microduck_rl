"""Command-fixing tests for the retained checkpoint evaluator."""

from types import SimpleNamespace

from mjlab_microduck.evaluation import fix_velocity_commands


def test_fix_commands_disables_heading_override_and_pins_twist():
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-3.14, 3.14),
    )
    twist = SimpleNamespace(
        resampling_time_range=(1.0, 2.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.3,
        rel_world_envs=0.4,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.2,
        init_velocity_prob=0.5,
        heading_command=True,
        ranges=ranges,
    )
    pose = lambda: SimpleNamespace(  # noqa: E731
        resampling_time_range=(1.0, 2.0),
        ranges=((-1.0, 1.0),),
        zero_command_prob=0.0,
    )
    cfg = SimpleNamespace(
        commands={"twist": twist, "head_pose": pose(), "body_pose": pose()}
    )

    fix_velocity_commands(cfg, speed=0.3, yaw_rate=-0.6)

    assert twist.heading_command is False
    assert twist.ranges.heading is None
    assert twist.ranges.lin_vel_x == (0.3, 0.3)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (-0.6, -0.6)
    assert twist.rel_heading_envs == 0.0
    assert twist.rel_forward_envs == 0.0
