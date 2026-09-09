"""Exact float32 predicate/accounting fixtures, not rollout or hardware evidence."""
import pytest
import torch

from mjlab_microduck.stance_transition import physical_failures, StanceTransition
from test_stance_transition import state


def neighbors(value):
    center = torch.tensor(value, dtype=torch.float32)
    return torch.stack((torch.nextafter(center, torch.tensor(-float('inf'))), center,
                        torch.nextafter(center, torch.tensor(float('inf')))))


def checked(frame, expected, step=0):
    frame.validate(len(expected), torch.device('cpu'))
    counters = torch.full((len(expected),), step, dtype=torch.long)
    assert physical_failures(frame, counters).tolist() == expected


@pytest.mark.parametrize('field,bound,expected', [
    ('tilt', .35, [False, False, True, False]),
    ('height', .08, [True, False, False, False]),
])
def test_scalar_adjacent_float_stops(field, bound, expected):
    frame = state(4); getattr(frame, field)[:3] = neighbors(bound)
    checked(frame, expected)


@pytest.mark.parametrize('field,bound', [('torque', .36), ('joint_velocity', 10.)])
@pytest.mark.parametrize('joint', range(14))
@pytest.mark.parametrize('sign', [-1, 1])
def test_every_motor_magnitude_at_adjacent_floats(field, bound, joint, sign):
    frame = state(4); getattr(frame, field)[:3, joint] = sign*neighbors(bound)
    checked(frame, [False, False, True, False])


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', [-1, 1])
def test_each_root_velocity_axis_uses_strict_norm_stop(axis, sign):
    frame = state(4); frame.root_velocity[:3, axis] = sign*neighbors(1.)
    checked(frame, [False, False, True, False])


@pytest.mark.parametrize('sign', [-1, 1])
def test_full_three_dimensional_root_speed_not_componentwise_or_planar(sign):
    frame = state(4)
    frame.root_velocity[:3] = sign*neighbors(1.)[:, None]*torch.tensor([.36, .48, .8])
    # Verify the realized float32 norms really are the adjacent values; this
    # fixture does not derive its expected predicate from the predicate itself.
    assert torch.equal(torch.linalg.vector_norm(frame.root_velocity[:3], dim=1), neighbors(1.))
    assert (frame.root_velocity.abs() < 1.).all()
    assert (torch.linalg.vector_norm(frame.root_velocity[:, :2], dim=1) < 1.).all()
    checked(frame, [False, False, True, False])


@pytest.mark.parametrize('foot', [0, 1])
@pytest.mark.parametrize('step', [49, 50, 51])
def test_each_foot_inclusive_support_boundary_and_integer_grace(foot, step):
    frame = state(4); frame.support[:3, foot] = neighbors(.01)
    checked(frame, [False]*4 if step < 50 else [True, True, False, False], step)


@pytest.mark.parametrize('foot', [0, 1])
def test_pre_and_post_step_support_grace_use_their_own_integer_boundary(foot):
    frame = state(4); frame.support[:3, foot] = neighbors(.01)
    tick = StanceTransition(torch.full((4,), 49, dtype=torch.long), torch.zeros(4, 10), torch.zeros(4, 10))
    assert not tick.reject_pre_step_state(frame).any()
    tick.advance(frame)
    r = tick.result()
    assert r['terminated'].tolist() == [True, True, False, False]
    assert r['episode_steps'].tolist() == [50]*4 and r['executed_steps'].tolist() == [1]*4
    frozen = r['reward'][:2].clone()
    for _ in range(9): tick.advance(state(4))
    r = tick.result()
    assert r['episode_steps'].tolist() == [50, 50, 59, 59]
    assert r['executed_steps'].tolist() == [1, 1, 10, 10]
    assert torch.equal(r['reward'][:2], frozen)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('joint', [0, 13])
def test_adjacent_proposed_torque_rejected_before_any_physics(sign, joint):
    tick = StanceTransition(torch.zeros(4, dtype=torch.long), torch.zeros(4, 10), torch.zeros(4, 10))
    torque = torch.zeros(4, 14); torque[:3, joint] = sign*neighbors(.36)
    assert tick.reject_proposed_torque(torque).tolist() == [False, False, True, False]
    assert not tick.reject_proposed_torque(torque).any()
    assert tick.applied_calls == 0 and not tick.executed_steps.any()
    for _ in range(10): tick.advance(state(4))
    r = tick.result()
    assert r['executed_steps'].tolist() == [10, 10, 0, 10]
    assert r['reward'][2] == -2 and not r['timed_out'].any()


@pytest.mark.parametrize('mode', ['pre', 'post'])
def test_adjacent_failure_and_exact_boundary_at_final_step(mode):
    frame = state(4); frame.tilt[:3] = neighbors(.35)
    tick = StanceTransition(torch.full((4,), 2499, dtype=torch.long), torch.zeros(4, 10), torch.zeros(4, 10))
    if mode == 'pre':
        assert tick.reject_pre_step_state(frame).tolist() == [False, False, True, False]
        frame = state(4)
    tick.advance(frame)
    r = tick.result()
    assert r['terminated'].tolist() == [False, False, True, False]
    assert r['timed_out'].tolist() == [True, True, False, True]
    assert r['executed_steps'].tolist() == [1, 1, 0 if mode == 'pre' else 1, 1]
    reward = r['reward'].clone(); steps = r['episode_steps'].clone()
    tick.advance(state(4))
    assert torch.equal(tick.result()['reward'], reward)
    assert torch.equal(tick.result()['episode_steps'], steps)
    assert not tick.live.any()
