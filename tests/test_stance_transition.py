import pytest
import torch

from mjlab_microduck.stance_transition import PhysicsState, StanceTransition


def state(n=2):
    return PhysicsState(tilt=torch.zeros(n), root_velocity=torch.zeros(n, 3),
        height=torch.full((n,), .12), support=torch.ones(n, 2), torque=torch.zeros(n, 14),
        joint_velocity=torch.zeros(n, 14), hard_limit=torch.zeros(n, dtype=torch.bool),
        forbidden_contact=torch.zeros(n, dtype=torch.bool), warning=torch.zeros(n, dtype=torch.bool))


def transition(counters=(0, 0)):
    return StanceTransition(torch.tensor(counters), torch.zeros(len(counters), 10),
                            torch.zeros(len(counters), 10))


def test_nominal_reward_is_sum_over_ten_physics_steps():
    tick = transition()
    for _ in range(10): tick.advance(state())
    r = tick.result()
    assert torch.allclose(r['reward'], torch.full((2,), .09))  # .02*(2+1+1+.5)
    assert r['executed_steps'].tolist() == [10, 10]
    assert r['live'].tolist() == [True, True]
    assert torch.allclose(sum(r['term_sums'].values()), r['reward'])
    with pytest.raises(ValueError, match='ten'): tick.advance(state())
    with pytest.raises(ValueError, match='closed'): tick.reject_proposed_torque(torch.zeros(2, 14))


def test_early_failure_cost_once_and_no_later_reward_or_episode_time():
    tick = transition(); tick.advance(state())
    bad = state(); bad.tilt[0] = .4
    tick.advance(bad)
    frozen_reward = tick.result()['reward'][0].item()
    for _ in range(8): tick.advance(state())
    r = tick.result()
    assert r['executed_steps'].tolist() == [2, 10]
    assert r['episode_steps'].tolist() == [2, 10]
    assert r['terminated'].tolist() == [True, False]
    assert r['reward'][0].item() == frozen_reward
    assert r['reward'][0].item() > -2 and r['reward'][0].item() < -1.98
    assert r['reward'][1].item() == pytest.approx(.09)


def test_pre_step_motor_rejection_does_not_count_physics_or_repeat_penalty():
    tick = transition(); torque = torch.zeros(2, 14); torque[0, 3] = .37
    assert tick.reject_proposed_torque(torque).tolist() == [True, False]
    assert tick.reject_proposed_torque(torque).tolist() == [False, False]
    for _ in range(10): tick.advance(state())
    r = tick.result()
    assert r['executed_steps'].tolist() == [0, 10]
    assert r['reward'][0].item() == -2


def test_failure_precedes_simultaneous_timeout_and_final_boundary_is_checked():
    tick = transition((2499, 2499)); bad = state(); bad.torque[0, 0] = .37
    tick.advance(bad)
    r = tick.result()
    assert r['terminated'].tolist() == [True, False]
    assert r['timed_out'].tolist() == [False, True]
    tick.advance(state())
    assert tick.result()['executed_steps'].tolist() == [1, 1]
    assert torch.equal(tick.result()['reward'], r['reward'])


def test_support_grace_uses_exact_integer_physics_counter():
    tick = transition((48, 49)); unsupported = state(); unsupported.support[:] = 0
    tick.advance(unsupported)
    assert tick.result()['terminated'].tolist() == [False, True]


@pytest.mark.parametrize('field,value', [('height', .079), ('tilt', .351),
    ('root_velocity', 1.01), ('joint_velocity', 10.01), ('torque', -.361),
    ('hard_limit', True), ('forbidden_contact', True)])
def test_all_physics_stops(field, value):
    tick = transition(); bad = state(); getattr(bad, field)[0] = value
    tick.advance(bad)
    assert tick.result()['terminated'].tolist() == [True, False]


@pytest.mark.parametrize('field', ['tilt', 'root_velocity', 'height', 'support', 'torque', 'joint_velocity'])
def test_nonfinite_aborts_job_before_mutating_accounting(field):
    tick = transition(); bad = state(); getattr(bad, field)[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'): tick.advance(bad)
    assert tick.applied_calls == 0 and not tick.reward.any()


def test_solver_warning_and_invalid_corrections_are_job_failures():
    bad = state(); bad.warning[0] = True
    with pytest.raises(ValueError, match='warning'): transition().advance(bad)
    with pytest.raises(ValueError, match='bounded'):
        StanceTransition(torch.tensor([0]), torch.full((1, 10), .21), torch.zeros(1, 10))


def test_output_and_input_tensors_are_owned():
    steps = torch.tensor([0, 0]); correction = torch.zeros(2, 10)
    tick = StanceTransition(steps, correction, correction)
    steps[:] = 200; correction[:] = .1
    tick.advance(state()); r = tick.result(); r['reward'][:] = 100
    assert tick.episode_steps.tolist() == [1, 1]
    assert not tick.correction.any() and not tick.change.any()
    assert tick.reward[0].item() == pytest.approx(.009)


def test_reward_accounting_does_not_retain_actor_autograd_graph():
    correction = torch.zeros(2, 10, requires_grad=True)
    tick = StanceTransition(torch.tensor([0, 0]), correction, correction)
    tick.advance(state())
    assert not tick.reward.requires_grad and not tick.correction.requires_grad
