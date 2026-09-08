from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microduck import football_bam_probe as fixture
from mjlab_microduck import stance_lesson_contract as reference
from mjlab_microduck.stance_control_state import BamStateCommit, StanceActionDelay


def batched_fixture():
    model, data, actuator = fixture.build_component_fixture(flat_floor=True)
    # Two owned CPU tensor worlds built from the same actual native snapshot;
    # no simulator step, GPU initialization or source fixture modification.
    def repeat(value):
        if isinstance(value, torch.Tensor): return value.repeat((2,)+(1,)*(value.ndim-1))
        return SimpleNamespace(**{k: repeat(v) for k, v in vars(value).items()})
    actuator._num_envs = 2
    for name in ('kp_scale', 'kd_scale', 'friction_scale', 'vin_tensor', 'vin_drop_gain', '_prev_motor_torque'):
        setattr(actuator, name, repeat(getattr(actuator, name)))
    actuator._data = repeat(actuator._data)
    actuator._mjwarp_model = repeat(actuator._mjwarp_model)
    pos = torch.tensor([[data.qpos[model.joint(n).qposadr[0]] for n in reference.JOINTS]]*2)
    # Native qpos values infer float64; the training BAM inputs are float32.
    pos = pos.float()
    command = ActuatorCmd(position_target=pos+.01, velocity_target=torch.zeros_like(pos),
        effort_target=torch.zeros_like(pos), pos=pos, vel=torch.zeros_like(pos))
    return model, actuator, command


def snapshot(runtime):
    return dict(previous=runtime.previous.clone(), voltage=runtime.voltage.clone(), kp=runtime.kp.clone(),
                **{k: v.clone() for k, v in runtime.fields.items()})


def test_bam_live_rows_match_actual_compute_and_closed_history_fields_stay_exact():
    _, actuator, command = batched_fixture()
    actuator._prev_motor_torque[1] = .02
    actuator._mjwarp_model.dof_frictionloss[1] = .123
    runtime = BamStateCommit(actuator); before = snapshot(runtime)
    _, direct, _ = batched_fixture()
    direct._prev_motor_torque[1] = .02
    direct._mjwarp_model.dof_frictionloss[1] = .123
    for _ in range(2):
        expected = direct.compute(command)
        result = runtime.compute(command, torch.tensor([True, False]))
        assert torch.equal(result['torque_nm'][0], expected[0])
        assert result['accepted'].tolist() == [True, False]
        assert result['rejected'].tolist() == [False, False]
        assert torch.equal(runtime.previous[0], direct._prev_motor_torque[0])
        for name, value in runtime.fields.items():
            assert torch.equal(value[0], getattr(direct._mjwarp_model, name)[0])
        assert torch.equal(runtime.voltage[0], direct._bam_model.actuator.vin[0])
        assert all(torch.equal(snapshot(runtime)[name][1], value[1]) for name, value in before.items())


def test_excessive_row_does_not_commit_any_motor_state(monkeypatch):
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    before = snapshot(runtime)
    original = type(actuator._bam_model.actuator).compute_torque
    def excessive(self, *args):
        torque = original(self, *args); torque[1] = .37
        return torque
    monkeypatch.setattr(type(actuator._bam_model.actuator), 'compute_torque', excessive)
    result = runtime.compute(command, torch.tensor([True, True]))
    assert result['accepted'].tolist() == [True, False]
    assert result['rejected'].tolist() == [False, True]
    assert all(torch.equal(snapshot(runtime)[name][1], value[1]) for name, value in before.items())


def test_nonfinite_proposal_faults_without_committing_and_cannot_reset(monkeypatch):
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    before = snapshot(runtime)
    monkeypatch.setattr(type(actuator._bam_model.actuator), 'compute_torque',
                        lambda *a: torch.full((2, 14), float('nan')))
    with pytest.raises(ValueError, match='proposed torque'):
        runtime.compute(command, torch.tensor([True, False]))
    assert all(torch.equal(snapshot(runtime)[name], value) for name, value in before.items())
    with pytest.raises(RuntimeError, match='faulted'): runtime.reset(torch.tensor([True, True]))


def test_selective_motor_reset_preserves_sibling_and_uncontrolled_dofs():
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    runtime.compute(command, torch.tensor([True, True]))
    for value in runtime.fields.values(): value[:, :6] = .123
    before = snapshot(runtime); runtime.reset(torch.tensor([True, False]))
    after = snapshot(runtime)
    assert not runtime.previous[0].any() and runtime.voltage[0] == 7.5
    for name, value in runtime.fields.items():
        assert not value[0, runtime.ids].any()
        assert torch.equal(value[0, :6], before[name][0, :6])
    assert all(torch.equal(after[name][1], value[1]) for name, value in before.items())


def test_closed_batch_does_not_compute_and_returned_torque_is_owned(monkeypatch):
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    result = runtime.compute(command, torch.tensor([True, True])); original = runtime.previous.clone()
    result['torque_nm'].fill_(999); assert torch.equal(runtime.previous, original)
    monkeypatch.setattr(type(actuator), 'compute', lambda *a: pytest.fail('closed batch computed'))
    runtime.compute(command, torch.tensor([False, False]))


def test_unexpanded_friction_or_replaced_history_refused():
    _, actuator, command = batched_fixture()
    saved = actuator._mjwarp_model.dof_damping
    actuator._mjwarp_model.dof_damping = saved[:1].expand(2, -1)
    with pytest.raises(ValueError, match='expanded'): BamStateCommit(actuator)
    actuator._mjwarp_model.dof_damping = saved
    runtime = BamStateCommit(actuator); actuator._prev_motor_torque = runtime.previous.clone()
    with pytest.raises(ValueError, match='replaced'): runtime.compute(command, torch.tensor([True, True]))


def action_fixture():
    nominal = torch.zeros(14, dtype=torch.float64)
    ranges = torch.tensor([[-1., 1.]]*14, dtype=torch.float64)
    return nominal, ranges, StanceActionDelay(nominal, ranges, 2)


def test_batched_action_limiter_matches_numpy_reference_and_preserves_closed_row():
    nominal, ranges, state = action_fixture(); rng = np.random.default_rng(17)
    previous = np.zeros(10)
    for _ in range(30):
        actions = rng.normal(size=(2, 10))
        target, previous = reference.limited_targets(actions[0], previous, nominal.numpy(), ranges.numpy())
        correction, change = state.set_actions(torch.from_numpy(actions), torch.tensor([True, False]))
        assert np.allclose(state.target[0].numpy(), target, rtol=0, atol=1e-15)
        assert not correction[1].any() and not change[1].any()
        assert not state.target[1].any()
        assert not state.target[:, 5:9].any()
        assert change.abs().max() <= .02


def test_delay_advances_only_accepted_rows_and_reset_is_selective():
    nominal, _, state = action_fixture()
    state.set_actions(torch.ones((2, 10), dtype=torch.float64), torch.tensor([True, True]))
    for _ in range(3):
        assert torch.equal(state.peek(), nominal.expand(2, 14))
        state.advance(torch.tensor([True, False]))
    assert torch.equal(state.peek()[0], state.target[0])
    assert torch.equal(state.peek()[1], nominal)
    before = state.queue.clone(); state.reset(torch.tensor([False, True]))
    assert torch.equal(state.queue[0], before[0])
    assert not state.correction[1].any() and not state.target[1].any()
    result = state.peek(); result.fill_(999); assert not (state.queue == 999).any()


def test_invalid_action_faults_without_updating_state():
    _, _, state = action_fixture(); before = state.target.clone()
    with pytest.raises(ValueError): state.set_actions(torch.full((2, 10), float('nan')), torch.tensor([True, True]))
    assert torch.equal(state.target, before)
    with pytest.raises(RuntimeError, match='faulted'): state.reset(torch.tensor([True, True]))


def test_position_only_commands_required():
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    command.effort_target.fill_(.1)
    with pytest.raises(ValueError, match='position-only'):
        runtime.compute(command, torch.tensor([True, True]))


def test_invalid_friction_candidate_never_reaches_owned_model_fields(monkeypatch):
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    before = snapshot(runtime)
    monkeypatch.setattr(type(actuator), '_compute_friction_budget',
                        lambda *a: torch.full((2, 14), float('inf')))
    with pytest.raises(ValueError, match='proposed dof_frictionloss'):
        runtime.compute(command, torch.tensor([True, True]))
    assert all(torch.equal(snapshot(runtime)[name], value) for name, value in before.items())


def test_rejected_motor_row_never_shifts_its_delay_queue(monkeypatch):
    model, actuator, command = batched_fixture(); motor = BamStateCommit(actuator)
    ranges = torch.tensor(model.jnt_range[[model.joint(n).id for n in reference.JOINTS]], dtype=torch.float32)
    delay = StanceActionDelay(command.pos[0], ranges, 2)
    delay.set_actions(torch.ones((2, 10)), torch.tensor([True, True]))
    initial = delay.queue.clone()
    original = type(actuator._bam_model.actuator).compute_torque
    def excessive(self, *args):
        torque = original(self, *args); torque[1] = .37
        return torque
    monkeypatch.setattr(type(actuator._bam_model.actuator), 'compute_torque', excessive)
    live = torch.tensor([True, True])
    for _ in range(3):
        command.position_target = delay.peek()
        result = motor.compute(command, live)
        delay.advance(result['accepted'])
        live &= ~result['rejected']
    assert live.tolist() == [True, False]
    assert torch.equal(delay.queue[1], initial[1])
    assert torch.equal(delay.peek()[0], delay.target[0])


def test_nominal_motor_settings_cannot_be_changed_silently():
    _, actuator, _ = batched_fixture(); actuator.vin_tensor[0] = 8.
    with pytest.raises(ValueError, match='nominal vin_tensor'): BamStateCommit(actuator)


def test_post_binding_randomization_is_refused_before_any_commit():
    _, actuator, command = batched_fixture(); runtime = BamStateCommit(actuator)
    before = snapshot(runtime); actuator.friction_scale[0] = .9
    with pytest.raises(ValueError, match='nominal parameter changed'):
        runtime.compute(command, torch.tensor([True, True]))
    assert all(torch.equal(snapshot(runtime)[name], value) for name, value in before.items())
