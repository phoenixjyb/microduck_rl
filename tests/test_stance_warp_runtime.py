"""Short real Duck Warp CPU integration; not CUDA, full hold or learned balance."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
from mjlab_microduck.stance_native_runtime import NativeStanceRuntime
from mjlab_microduck.stance_warp_integrator import STATE_FIELDS
from mjlab_microduck.stance_transition import StanceTransition


@pytest.fixture
def env():
    return WarpStanceRuntime(2, device='cpu')


def test_initial_entity_binding_and_observation_match_native_reference(env):
    ref = NativeStanceRuntime()
    assert env.actuator.entity is env.entity
    assert not env.actuator.has_delay and env.actuator._delay_buffer is None
    assert torch.equal(env.actuator._dof_ids, env.dofs)
    assert env.motor.fields['dof_frictionloss'].shape == (2, 20)
    for key, shape in (('actor', (2, 44)), ('critic', (2, 50))):
        assert env.observations()[key].shape == shape
        assert np.allclose(env.observations()[key].numpy(), ref.observations()[key][None], atol=1e-5)
    assert np.allclose(env.initial_qpos.numpy(), ref.initial_qpos, atol=1e-7)
    for name in ('actuator_gear', 'actuator_forcerange', 'dof_armature', 'geom_condim', 'geom_friction'):
        assert np.array_equal(getattr(env.native, name), getattr(ref.model, name)), name
    assert env.data.naccdmax == env.data.naconmax


def test_ten_actual_substeps_one_fixed_delay_and_finite_reward(env, monkeypatch):
    compute = env.motor.compute; commands = []
    def observed(command, live):
        commands.append(command.position_target.clone())
        return compute(command, live)
    monkeypatch.setattr(env.motor, 'compute', observed)
    result = env.step(torch.ones(2, 10))
    assert result['executed_steps'].tolist() == [10, 10]
    assert len(result['boundaries']) == 11 and len(commands) == 10
    assert torch.allclose(env._view('time'), torch.full((2,), .02))
    for target in commands[:3]: assert torch.equal(target, env.nominal.expand(2, 14))
    assert torch.equal(commands[3], env.delay.target)
    assert torch.allclose(result['observation']['actor'][:, -10:], torch.full((2, 10), .1))
    assert not result['terminated'].any() and not result['timed_out'].any()
    assert torch.allclose(result['reward'], sum(result['term_sums'].values()))
    assert all(torch.isfinite(b['observation']['critic']).all() for b in result['boundaries'])
    assert not result['optimizer_launched']
    assert env.motor.previous.abs().max() <= .36


def freeze_fields(env, row=0):
    result = {name: env._view(name)[row].clone() for name in (*STATE_FIELDS, 'ctrl')}
    result.update(queue=env.delay.queue[row].clone(), correction=env.delay.correction[row].clone(),
                  target=env.delay.target[row].clone(), history=env.motor.previous[row].clone(),
                  voltage=env.motor.voltage[row].clone(), kp=env.motor.kp[row].clone())
    result.update({k: v[row].clone() for k, v in env.motor.fields.items()})
    result.update({k: v[row].clone() for k, v in env.observations().items()})
    return result


def test_first_terminal_frozen_across_ticks_while_sibling_advances(env, monkeypatch):
    original = env.integrator.integrate; calls = []
    def fail_first(live):
        original(live); calls.append(live.clone())
        if len(calls) == 1:
            # Deliberate synthetic failure following a real full-robot Euler step.
            env._view('qpos')[0, 3:7] = torch.tensor([np.cos(.2), 0, np.sin(.2), 0])
    monkeypatch.setattr(env.integrator, 'integrate', fail_first)
    result = env.step(torch.zeros(2, 10))
    assert result['executed_steps'].tolist() == [1, 10]
    terminal = result['terminal_records'][0]
    assert terminal['physics_step'] == 1 and terminal['terminated']
    assert terminal['state']['tilt'] == pytest.approx(.4, abs=1e-5)
    assert all(w == 0 for w in terminal['contacts']['worldid'])
    before = freeze_fields(env)
    result2 = env.step(torch.ones(2, 10))
    assert result2['executed_steps'].tolist() == [0, 10]
    assert env.steps.tolist() == [1, 20]
    assert result2['reward'][0] == 0 and not result2['terminated'][0]
    assert result2['terminal_records'][0] == terminal
    for key, value in freeze_fields(env).items(): assert torch.equal(value, before[key]), key
    result2['terminal_records'][0]['qpos'][0] = 999
    assert env.terminal[0] == terminal


def test_selective_reset_returns_terminal_and_preserves_sibling(env, monkeypatch):
    # Inject one excessive proposal through the real BAM staging wrapper.
    cls = type(env.actuator); compute = cls.compute
    def excess(self, command):
        torque = compute(self, command); torque[0] = .37
        return torque
    monkeypatch.setattr(cls, 'compute', excess)
    result = env.step(torch.ones(2, 10))
    assert result['executed_steps'].tolist() == [0, 10]
    assert result['terminal_records'][0]['rejected_proposed_torque_nm'] is not None
    assert not env._view('ctrl')[0].any() and not env.motor.previous[0].any()
    assert torch.equal(env.delay.queue[0], env.nominal.expand(3, 14))
    sibling = freeze_fields(env, 1)
    retained = env.reset(torch.tensor([True, False]))
    assert retained[0] == result['terminal_records'][0] and retained[1] is None
    assert env.terminal[0] is None and env.live.all()
    assert env.steps.tolist() == [0, 10]
    assert torch.equal(env._view('qpos')[0], env.initial_qpos)
    for name in ('qvel', 'ctrl', 'time', 'qacc_warmstart'): assert not env._view(name)[0].any()
    assert not env.delay.correction[0].any()
    for name, value in freeze_fields(env, 1).items(): assert torch.equal(value, sibling[name]), name


def test_invalid_input_fault_cannot_be_erased_by_reset(env):
    before = env._view('qpos').clone()
    with pytest.raises(ValueError, match='actions'): env.step(torch.full((2, 10), float('nan')))
    assert torch.equal(before, env._view('qpos'))
    with pytest.raises(RuntimeError, match='job closeout'): env.reset(torch.ones(2, dtype=torch.bool))


def test_rotated_nonzero_velocity_observations_use_exact_body_frame(env):
    # Deliberate state injection checks kinematic conventions, not policy success.
    q = [np.cos(.1), 0, np.sin(.1), 0]
    env._view('qpos')[:, 3:7] = torch.tensor(q, dtype=torch.float32)
    env._view('qvel')[:, :6] = torch.tensor([.1, -.2, .3, .2, -.3, .4])
    env._forward(); env._refresh(env.live)
    nd = mujoco.MjData(env.native)
    nd.qpos[:] = env._view('qpos')[0].numpy()
    nd.qvel[:] = env._view('qvel')[0].numpy()
    mujoco.mj_forward(env.native, nd)
    local = np.zeros(6)
    root = int(env.native.jnt_bodyid[env.native.joint('trunk_base_freejoint').id])
    mujoco.mj_objectVelocity(env.native, nd, mujoco.mjtObj.mjOBJ_XBODY, root, local, 1)
    rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, nd.qpos[3:7])
    observation = env.observations()
    assert np.allclose(observation['actor'][0, :3].numpy(), rotation.reshape(3, 3).T @ [0, 0, -1], atol=1e-6)
    assert np.allclose(observation['actor'][0, 3:6].numpy(), local[:3]/5, atol=1e-6)
    assert torch.equal(observation['critic'][0, 44:47], env._view('qvel')[0, :3])
    assert env.state.tilt[0] == pytest.approx(.2, abs=1e-6)


def test_solved_nonfinite_output_faults_before_euler_and_cannot_reset(env, monkeypatch):
    import mjlab_microduck.stance_warp_runtime as module
    forward = module.mjwarp.forward
    def nonfinite(model, data):
        forward(model, data)
        env._view('qacc')[0, 0] = float('nan')
    monkeypatch.setattr(module.mjwarp, 'forward', nonfinite)
    monkeypatch.setattr(env.integrator, 'integrate', lambda *_: pytest.fail('invalid solve integrated'))
    with pytest.raises(ValueError, match='nonfinite solved'): env.step(torch.zeros(2, 10))
    assert not env._view('time').any() and not env.steps.any()
    with pytest.raises(RuntimeError, match='job closeout'): env.reset(torch.ones(2, dtype=torch.bool))


def test_mismatched_physics_clock_faults_the_job(env, monkeypatch):
    integrate = env.integrator.integrate
    def wrong_clock(rows):
        integrate(rows); env._view('time')[0] += .002
    monkeypatch.setattr(env.integrator, 'integrate', wrong_clock)
    with pytest.raises(ValueError, match='physical clock'): env.step(torch.zeros(2, 10))
    assert env.faulted


@pytest.mark.parametrize('mask', [torch.tensor([True]), torch.tensor([1, 0])])
def test_transition_rejects_malformed_initial_mask(mask):
    with pytest.raises(ValueError, match='initial live mask'):
        StanceTransition(torch.zeros(2, dtype=torch.long), torch.zeros(2, 10), torch.zeros(2, 10), initial_live=mask)


def test_prior_closed_world_not_recharged_or_counted_at_next_policy_tick():
    from test_stance_transition import state
    tick = StanceTransition(torch.tensor([2500, 0]), torch.zeros(2, 10), torch.zeros(2, 10),
                            initial_live=torch.tensor([False, True]))
    frame = state(); frame.tilt[0] = .4
    for _ in range(10): tick.advance(frame)
    result = tick.result()
    assert result['episode_steps'].tolist() == [2500, 10]
    assert result['executed_steps'].tolist() == [0, 10]
    assert result['reward'][0] == 0 and not result['terminated'].any()
    assert result['live'].tolist() == [False, True]


def test_timeout_exact_boundary_is_not_failure_and_cannot_advance(env):
    # Deliberately synthetic near-timeout counters, not a completed 5s rollout.
    env.steps[:] = 2499
    env._view('time')[:] = 4.998
    env.state.support[:] = 1.  # avoid synthetic unsupported late-reset boundary
    # Advance from the exact initial pose with synthetic support control-flow input.
    refresh = env._refresh
    def supported(rows):
        refresh(rows); env.state.support[rows] = 1.
    env._refresh = supported
    result = env.step(torch.zeros(2, 10))
    assert result['executed_steps'].tolist() == [1, 1]
    assert result['timed_out'].all() and not result['terminated'].any()
    assert all(r['physics_step'] == 2500 for r in result['terminal_records'])
    before = freeze_fields(env)
    with pytest.raises(RuntimeError, match='all stance worlds closed'): env.step(torch.zeros(2, 10))
    for key, value in freeze_fields(env).items(): assert torch.equal(value, before[key])
