"""CPU command-config checks for bounded stock-walking checkpoint recordings."""

import sys
from pathlib import Path
from runpy import run_path

import pytest

from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401  Register the task configs.
_recorder = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "record_checkpoint.py"
    )
)
_fix_commands = _recorder["_fix_commands"]
_video_name_prefix = _recorder["_video_name_prefix"]
parse_args = _recorder["parse_args"]


TASK = "Mjlab-Velocity-Flat-Backlash-MicroDuck"


@pytest.mark.parametrize(
    ("speed", "yaw_rate"),
    ((0.0, 0.0), (0.3, 0.0), (0.2, 0.2), (0.2, -0.2)),
    ids=("idle", "straight", "turn-left", "turn-right"),
)
def test_recording_pins_real_play_config_to_requested_command(speed, yaw_rate):
    cfg = load_env_cfg(TASK, play=True)

    _fix_commands(cfg, speed, yaw_rate)

    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (speed, speed)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (yaw_rate, yaw_rate)
    assert twist.heading_command is False
    assert twist.ranges.heading is None
    assert twist.rel_forward_envs == 0.0
    assert twist.rel_turn_in_place_envs == 0.0
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_heading_envs == 0.0
    assert twist.rel_world_envs == 0.0
    assert twist.init_velocity_prob == 0.0

    for name in ("head_pose", "body_pose"):
        command = cfg.commands[name]
        assert command.ranges == tuple((0.0, 0.0) for _ in command.ranges)
        assert command.zero_command_prob == 1.0


def test_straight_video_filename_is_unchanged_and_turn_filename_has_signed_yaw():
    assert _video_name_prefix(0.3, 0.0) == "microduck-run-0.30mps"
    assert _video_name_prefix(0.2, 0.2) == "microduck-run-0.20mps-yaw+0.20radps"
    assert _video_name_prefix(0.2, -0.2) == "microduck-run-0.20mps-yaw-0.20radps"


@pytest.mark.parametrize(
    ("option", "value"),
    (("--speed", "nan"), ("--speed", "inf"), ("--yaw-rate", "-inf")),
)
def test_nonfinite_recording_commands_are_rejected(monkeypatch, option, value):
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_checkpoint.py", "unused.pt", "--output-dir", "out", option, value],
    )

    with pytest.raises(SystemExit):
        parse_args()
