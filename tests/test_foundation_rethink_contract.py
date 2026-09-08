"""CPU configuration findings, not runtime equivalence or policy acceptance."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mjlab_microduck.foundation_yaw_experiment import prepare_config
from mjlab_microduck.speed_response_control import prepare_config as evaluation_config


def test_selected_actor_action_and_timing_contracts_match():
    train, train_agent = prepare_config("pilot", arm="control")
    evaluate, eval_agent = evaluation_config()
    assert train.observations == evaluate.observations
    assert train.actions == evaluate.actions
    assert train.sim == evaluate.sim
    assert train.decimation == evaluate.decimation == 4
    assert train_agent.clip_actions is eval_agent.clip_actions is None
    assert train.observations["actor"].enable_corruption is True


def test_pushes_are_training_only_in_this_selected_comparison():
    train, _ = prepare_config("pilot", arm="control")
    evaluate, _ = evaluation_config()
    assert set(train.events) - set(evaluate.events) == {"push_robot"}
    assert not set(evaluate.events) - set(train.events)
    for name, term in evaluate.events.items():
        assert train.events[name] == term
    push = train.events["push_robot"]
    assert push.interval_range_s == (3.0, 6.0)
    assert push.params["velocity_range"] == {"x": (-.3, .3), "y": (-.3, .3)}


@pytest.mark.parametrize("name,extent", [("com_range", .015), ("head_com_range", .01)])
def test_completed_training_curriculum_changes_live_range_not_saved_config(name, extent):
    train, _ = prepare_config("pilot", arm="control")
    evaluate, _ = evaluation_config()
    assert not evaluate.curriculum
    term = train.curriculum[name]
    event_name = term.params["event_name"]
    assert train.events[event_name].params["ranges"] == (-.003, .003)
    assert evaluate.events[event_name].params["ranges"] == (-.003, .003)
    # Synthetic manager ownership, invoking the actual curriculum function.
    # No environment reset, randomization draw, robot or simulator is executed.
    live = deepcopy(train.events)
    env = SimpleNamespace(common_step_counter=204000,
                          event_manager=SimpleNamespace(get_term_cfg=live.__getitem__))
    result = term.func(env, None, **term.params)
    assert result.item() == pytest.approx(extent)
    assert live[event_name].params["ranges"] == (-extent, extent)
    assert train.events[event_name].params["ranges"] == (-.003, .003)


def test_fixed_training_is_not_transition_or_stop_practice():
    train, agent = prepare_config("pilot", arm="control")
    twist = train.commands["twist"]
    assert twist.ranges.lin_vel_x == (.3, .3)
    assert twist.ranges.lin_vel_y == twist.ranges.ang_vel_z == (0., 0.)
    assert twist.rel_standing_envs == 0.
    assert twist.resampling_time_range == (1e6, 1e6)
    assert agent.max_iterations == 500 and agent.num_steps_per_env == 24
    assert train.scene.num_envs == 256
    # 3,072,000 transitions; 240 s per lane, not 500 seconds of experience.
    assert agent.max_iterations * agent.num_steps_per_env * train.scene.num_envs == 3072000
