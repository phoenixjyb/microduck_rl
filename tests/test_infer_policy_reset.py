import numpy as np
import pytest

from scripts.infer_policy import PolicyInference


class _Session:
    def __init__(self, prefix):
        self._inputs = [type("IO", (), {"name": f"{prefix}_obs"})()]
        self._outputs = [type("IO", (), {"name": f"{prefix}_action"})()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def _policy(*, walking=True, standing=True, sitstand=False, velocity=(0.0, 0.0, 0.0)):
    policy = PolicyInference.__new__(PolicyInference)
    policy.last_action = np.ones(2, dtype=np.float32)
    policy.action_buffer = [np.ones(2, dtype=np.float32) for _ in range(3)]
    policy.buffer_index = 2
    policy.ground_pick_mode = True
    policy.ground_pick_phase = 0.4
    policy.sit_mode = True
    policy.slope_mode = True
    policy.behavior_mode = "roulade"
    policy.behavior_time_left = 1.2
    policy.head_mode = True
    policy.body_pose_mode = True
    policy.head_offset = np.ones(4, dtype=np.float32)
    policy.body_cmd = np.ones(6, dtype=np.float32)
    policy.walking_session = _Session("walking") if walking else None
    policy.standing_session = _Session("standing") if standing else None
    policy.sit_session = _Session("sitstand") if sitstand else None
    policy.is_sitstand = sitstand
    policy.vel_cmd = np.asarray(velocity, dtype=np.float32)
    policy.switch_threshold = 0.05
    policy.new_cmd_obs = True
    policy.current_policy = "roulade"
    policy.ort_session = _Session("roulade")
    policy.input_name = "stale_obs"
    policy.output_name = "stale_action"
    policy.command = np.ones(13, dtype=np.float32)
    return policy


@pytest.mark.parametrize(
    ("velocity", "expected_policy"),
    [((0.0, 0.0, 0.0), "standing"), ((0.2, 0.0, 0.0), "walking")],
)
def test_reset_clears_latched_modes_and_restores_matching_base_session(velocity, expected_policy):
    policy = _policy(velocity=velocity)

    policy.reset_state()

    assert policy.current_policy == expected_policy
    assert policy.ort_session is getattr(policy, f"{expected_policy}_session")
    assert (policy.input_name, policy.output_name) == (
        f"{expected_policy}_obs", f"{expected_policy}_action"
    )
    assert not policy.ground_pick_mode and policy.ground_pick_phase == 0.0
    assert not policy.sit_mode and not policy.slope_mode
    assert policy.behavior_mode is None and policy.behavior_time_left == 0.0
    assert not policy.head_mode and not policy.body_pose_mode
    np.testing.assert_array_equal(policy.last_action, np.zeros(2))
    assert all(not buffer.any() for buffer in policy.action_buffer)
    assert policy.buffer_index == 0
    np.testing.assert_array_equal(policy.head_offset, np.zeros(4))
    np.testing.assert_array_equal(policy.body_cmd, np.zeros(6))
    np.testing.assert_allclose(policy.vel_cmd, velocity)
    np.testing.assert_allclose(policy.command[:3], velocity)


def test_reset_keeps_sitstand_only_session_and_stand_flag():
    policy = _policy(walking=False, standing=False, sitstand=True)

    policy.reset_state()

    assert policy.current_policy == "sit"
    assert policy.ort_session is policy.sit_session
    assert (policy.input_name, policy.output_name) == ("sitstand_obs", "sitstand_action")
    assert policy.command[0] == 0.0
    assert policy.behavior_mode is None and not policy.slope_mode
