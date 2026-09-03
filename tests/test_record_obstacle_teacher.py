from mjlab_microduck.hierarchical_obstacle_rollout import recording_stem


def test_recording_stem_identifies_speed_and_obstacle_position():
    assert recording_stem(0.5, 1.15, -0.27) == (
        "microduck-hc1-0.50mps-x1.15m-y-0.27m"
    )


def test_recording_stem_distinguishes_learned_supervisor():
    assert recording_stem(
        0.8, 1.4, 0.0, controller_stage="hc2"
    ) == "microduck-hc2-0.80mps-x1.40m-y+0.00m"
