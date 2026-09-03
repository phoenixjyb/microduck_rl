from mjlab_microduck.hierarchical_obstacle_rollout import recording_stem


def test_recording_stem_identifies_speed_and_obstacle_position():
    assert recording_stem(0.5, 1.15, -0.27) == (
        "microduck-hc1-0.50mps-x1.15m-y-0.27m"
    )
