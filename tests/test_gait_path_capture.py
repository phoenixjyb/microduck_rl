from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import gait_path_capture as capture


def test_position_is_detached_measured_nonaliasing_and_existing_samples_unchanged():
    position = torch.arange(24., requires_grad=True).reshape(8, 3)
    robot = NS(data=NS(root_link_pos_w=position))
    original = {"episode_length": torch.zeros(8, dtype=torch.long),
                "physics_time": torch.zeros(8, dtype=torch.float64)}
    rng = torch.random.get_rng_state().clone()
    sampled = capture.capture(robot, original)
    assert "root_position_w" not in original
    assert sampled["physics_time"] is original["physics_time"]
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert not sampled["root_position_w"].requires_grad
    with torch.no_grad(): position.zero_()
    assert sampled["root_position_w"][7].tolist() == [21., 22., 23.]
    trace = capture.trace([sampled, sampled], "a"*64)
    assert trace["protocol"] == "microduck-gait-path-trace-v1"
    assert trace["position_frame"] == "world"
    assert trace["root_position_w"][0][7] == [21., 22., 23.]
    assert type(trace["episode_length"][0][0]) is int
    assert trace["policy_acceptance"] is trace["physical_motion_authorized"] is False


@pytest.mark.parametrize("fault", ["shape", "nan", "collision", "not-tensor"])
def test_invalid_or_ambiguous_position_refuses(fault):
    position = torch.zeros(8, 3)
    original = {}
    if fault == "shape": position = position[:7]
    elif fault == "nan": position[0, 0] = float("nan")
    elif fault == "collision": original["root_position_w"] = position
    else: position = [[0., 0., 0.]]*8
    with pytest.raises(ValueError):
        capture.capture(NS(data=NS(root_link_pos_w=position)), original)
