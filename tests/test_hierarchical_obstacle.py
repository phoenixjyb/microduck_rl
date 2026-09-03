import pytest
import torch

from mjlab_microduck.hierarchical_obstacle import (
    ObstaclePhase,
    ObstacleTeacherCfg,
    SUPERVISOR_OBSERVATION_DIM,
    apply_bounded_supervisor_command,
    clone_teacher_state,
    make_teacher_state,
    reset_teacher_state,
    supervisor_observation,
    teacher_command,
)
from mjlab_microduck.tasks.obstacle_observation import encode_obstacle_observation


def _observation(range_m, bearing_rad=0.0, closing_rate_mps=0.3, valid=True):
    return encode_obstacle_observation(
        range_m=torch.tensor([range_m]),
        bearing_rad=torch.tensor([bearing_rad]),
        width_m=torch.tensor([0.2]),
        height_m=torch.tensor([0.1]),
        closing_rate_mps=torch.tensor([closing_rate_mps]),
        valid=torch.tensor([valid]),
    )


def _step(observation, state, nominal=0.5, lateral=0.0, heading=0.0):
    return teacher_command(
        observation,
        torch.tensor([nominal]),
        torch.tensor([lateral]),
        torch.tensor([heading]),
        state,
    )


def test_teacher_tracks_nominal_speed_before_obstacle():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.5)
    command = _step(_observation(1.2), state)
    assert state.phase.item() == ObstaclePhase.APPROACH
    assert torch.allclose(command, torch.tensor([[0.5, 0.0]]))


def test_teacher_can_slow_and_turn_only_inside_interaction():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.8)
    command = _step(_observation(0.4, bearing_rad=0.2), state, nominal=0.8)
    assert state.phase.item() == ObstaclePhase.INTERACTION
    assert command[0, 0] < 0.8
    assert command[0, 1] < 0.0


def test_teacher_enters_early_when_time_to_contact_is_short():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.8)
    _step(_observation(1.0, closing_rate_mps=0.5), state, nominal=0.8)
    assert state.phase.item() == ObstaclePhase.INTERACTION


def test_teacher_recovers_nominal_speed_after_obstacle_passes():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.5)
    _step(_observation(0.4), state)
    state.previous_command[:] = torch.tensor([[0.3, 0.0]])
    command = _step(_observation(0.3, bearing_rad=torch.pi), state, lateral=0.2)
    assert state.phase.item() == ObstaclePhase.RECOVERY
    assert command[0, 0] > 0.3
    assert command[0, 1] < 0.0


def test_teacher_phase_transition_uses_route_not_body_forward_axis():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.5)
    state.phase[:] = int(ObstaclePhase.INTERACTION)
    _step(
        _observation(0.3, bearing_rad=torch.pi / 2),
        state,
        heading=torch.pi / 2,
    )
    assert state.phase.item() == ObstaclePhase.RECOVERY


def test_invalid_geometry_is_immediate_stop_until_recovery():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.5)
    command = _step(_observation(0.4, valid=False), state)
    assert torch.equal(command, torch.zeros(1, 2))
    state.phase[:] = int(ObstaclePhase.RECOVERY)
    state.previous_command.zero_()
    command = _step(_observation(0.4, valid=False), state)
    assert command[0, 0] > 0.0


def test_execution_layer_clamps_arbitrary_supervisor_output():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.3)
    command = apply_bounded_supervisor_command(
        torch.tensor([[10.0, -10.0]]), _observation(1.0), state
    )
    assert torch.allclose(command, torch.tensor([[0.38, -0.2]]))


def test_teacher_commands_obey_clamps_and_slew_limits():
    cfg = ObstacleTeacherCfg()
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.8)
    command = _step(_observation(0.1), state, nominal=0.8)
    assert 0.0 <= command[0, 0] <= cfg.max_forward_speed_mps
    assert command[0, 1].abs() <= cfg.max_yaw_delta_per_update_rps
    assert command[0, 1].abs() <= cfg.max_yaw_rate_rps


def test_teacher_rejects_negative_recovery_margin():
    with pytest.raises(ValueError, match="passed margin"):
        ObstacleTeacherCfg(passed_margin_m=-0.01)


def test_supervisor_observation_includes_phase_and_bypass_state():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.5)
    obstacle = _observation(1.0)
    observation = supervisor_observation(
        obstacle,
        torch.tensor([0.5]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.4]),
        state,
    )
    assert observation.shape == (1, SUPERVISOR_OBSERVATION_DIM)
    assert torch.equal(observation[0, -4:-1], torch.tensor([1.0, 0.0, 0.0]))


def test_supervisor_observation_can_retain_pre_action_command():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=0.5)
    previous = torch.tensor([[0.4, -0.3]])
    state.previous_command[:] = torch.tensor([[0.2, 0.6]])
    observation = supervisor_observation(
        _observation(1.0),
        torch.tensor([0.5]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.4]),
        state,
        previous_command=previous,
    )
    assert torch.allclose(observation[0, 11:13], torch.tensor([0.5, -0.5]))


def test_reset_teacher_state_restores_approach_and_nominal_command():
    state = make_teacher_state(2, device="cpu", nominal_speed_mps=0.5)
    state.phase[:] = int(ObstaclePhase.RECOVERY)
    state.previous_command.zero_()
    reset_teacher_state(state, torch.tensor([True, False]), nominal_speed_mps=0.3)
    assert state.phase.tolist() == [ObstaclePhase.APPROACH, ObstaclePhase.RECOVERY]
    assert torch.equal(state.previous_command[0], torch.tensor([0.3, 0.0]))
    assert torch.equal(state.previous_command[1], torch.tensor([0.0, 0.0]))


def test_teacher_state_accepts_per_environment_nominal_speeds():
    nominal = torch.tensor([0.3, 0.5, 0.8])
    state = make_teacher_state(3, device="cpu", nominal_speed_mps=nominal)
    torch.testing.assert_close(state.previous_command[:, 0], nominal)
    state.previous_command.zero_()
    reset_teacher_state(
        state,
        torch.tensor([True, False, True]),
        nominal_speed_mps=nominal,
    )
    torch.testing.assert_close(
        state.previous_command[:, 0], torch.tensor([0.3, 0.0, 0.8])
    )


def test_cloned_teacher_state_is_independent_for_counterfactual_labels():
    state = make_teacher_state(2, device="cpu", nominal_speed_mps=0.4)
    cloned = clone_teacher_state(state)
    cloned.phase[0] = int(ObstaclePhase.RECOVERY)
    cloned.bypass_side[0] = -cloned.bypass_side[0]
    cloned.previous_command[0] = torch.tensor([0.1, 0.2])

    assert state.phase[0] == int(ObstaclePhase.APPROACH)
    assert cloned.bypass_side[0] != state.bypass_side[0]
    torch.testing.assert_close(state.previous_command[0], torch.tensor([0.4, 0.0]))
