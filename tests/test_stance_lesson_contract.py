import numpy as np
import pytest

from mjlab_microduck import stance_lesson_contract as lesson


def test_actor_layout_units_and_no_privileged_fields():
    obs = lesson.actor_observation([0, 0, -1], [5, 0, 0], np.full(14, .5),
        np.full(14, 10), np.full(10, .2))
    assert obs.shape == (44,)
    assert np.array_equal(obs[:6], [0, 0, -1, 1, 0, 0])
    assert np.all(obs[6:] == 1)
    critic = lesson.critic_observation(obs, [1, 2, 3], .12, [3, 4])
    assert critic.shape == (50,) and np.array_equal(critic[:44], obs)
    assert np.allclose(critic[44:], [1, 2, 3, 0, .3, .4])


def test_action_rate_cap_head_freeze_and_input_ownership():
    nominal = np.zeros(14); previous = np.zeros(10); action = np.full(10, 99.)
    target, realized = lesson.limited_targets(action, previous, nominal, np.tile([-1., 1.], (14, 1)))
    assert np.allclose(realized, .02)
    assert np.all(target[5:9] == 0)
    assert np.allclose(target[list(lesson.LEG_IDS)], .02)
    assert not np.any(nominal) and not np.any(previous) and np.all(action == 99)
    for _ in range(20):
        target, realized = lesson.limited_targets(action, realized, nominal, np.tile([-1., 1.], (14, 1)))
    assert np.allclose(realized, .2)


def test_soft_range_intersection_and_reversal_obey_rate_limit():
    nominal = np.zeros(14); ranges = np.tile([-.1, .1], (14, 1))
    previous = np.full(10, .085)
    _, realized = lesson.limited_targets(np.ones(10), previous, nominal, ranges)
    assert np.allclose(realized, .09)
    _, reversed_ = lesson.limited_targets(-np.ones(10), realized, nominal, ranges)
    assert np.allclose(reversed_, .07)


@pytest.mark.parametrize('bad', [np.full(10, np.nan), np.zeros(9)])
def test_invalid_action_rejected(bad):
    with pytest.raises(ValueError):
        lesson.limited_targets(bad, np.zeros(10), np.zeros(14), np.tile([-1., 1.], (14, 1)))


def test_invalid_previous_state_is_not_silently_clipped():
    with pytest.raises(ValueError, match='previous'):
        lesson.limited_targets(np.zeros(10), np.full(10, .3), np.zeros(14), np.tile([-1., 1.], (14, 1)))


def test_observation_requires_unit_gravity_and_finite_signals():
    with pytest.raises(ValueError, match='unit'):
        lesson.actor_observation([0, 0, 0], np.zeros(3), np.zeros(14), np.zeros(14), np.zeros(10))
    with pytest.raises(ValueError, match='finite'):
        lesson.critic_observation(np.zeros(44), np.zeros(3), np.nan, [0, 0])
    with pytest.raises(ValueError, match='nonnegative'):
        lesson.critic_observation(np.zeros(44), np.zeros(3), .12, [-1, 0])


def test_actual_robot_joint_order_and_nominal_soft_range_match_contract():
    from mjlab_microduck.football_bam_probe import build_component_fixture
    model, data, actuator = build_component_fixture()
    assert tuple(actuator.target_names) == lesson.JOINTS
    ids = [model.joint(n).id for n in lesson.JOINTS]
    nominal = data.qpos[model.jnt_qposadr[ids]]
    target, correction = lesson.limited_targets(np.zeros(10), np.zeros(10), nominal,
                                                model.jnt_range[ids])
    assert np.array_equal(target, nominal) and not np.any(correction)
