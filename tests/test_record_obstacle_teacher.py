import pytest

from mjlab_microduck.hierarchical_obstacle_rollout import (
    recording_controller_stage,
    recording_stem,
)


def test_recording_stem_identifies_speed_and_obstacle_position():
    assert recording_stem(0.5, 1.15, -0.27) == (
        "microduck-hc1-0.50mps-x1.15m-y-0.27m"
    )


def test_recording_stem_distinguishes_learned_supervisor():
    assert recording_stem(
        0.8, 1.4, 0.0, controller_stage="hc2"
    ) == "microduck-hc2-0.80mps-x1.40m-y+0.00m"


def test_recording_stem_identifies_lateral_gated_hybrid():
    assert recording_stem(
        0.5, 1.15, 0.12, controller_stage="hc4lh"
    ) == "microduck-hc4lh-0.50mps-x1.15m-y+0.12m"


@pytest.mark.parametrize(
    ("supervisor_stage", "expected"),
    [
        (None, "hc1"),
        ("HC2-behavioral-cloning", "hc2"),
        ("HC4L-lateral-behavioral-cloning", "hc4l"),
        ("HC4LH-lateral-gated-supervisor", "hc4lh"),
        ("HC4R-near-range-behavioral-cloning", "hc4r"),
        ("HC4R2-student-state-correction-BC", "hc4r2"),
        ("HC4R2H-range-speed-gated-supervisor", "hc4r2h"),
        ("HC4R2L-episode-latched-supervisor", "hc4r2l"),
    ],
)
def test_recording_controller_stage(supervisor_stage, expected):
    assert recording_controller_stage(supervisor_stage) == expected


def test_recording_controller_stage_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unsupported supervisor stage"):
        recording_controller_stage("unknown")
