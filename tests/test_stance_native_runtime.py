import mujoco
import numpy as np
import pytest

from mjlab_microduck import stance_native_runtime as runtime


def test_angular_velocity_uses_regular_body_not_rotated_inertia_axes():
    env = runtime.NativeStanceRuntime()
    env.data.qpos[3:7] = [np.cos(.1), 0, np.sin(.1), 0]
    env.data.qvel[:6] = [.1, -.2, .3, .2, -.3, .4]
    mujoco.mj_forward(env.model, env.data)
    regular = env.data.xmat[env.root_body_id].reshape(3, 3).T @ env.data.cvel[env.root_body_id, :3]
    inertial = env.data.ximat[env.root_body_id].reshape(3, 3).T @ env.data.cvel[env.root_body_id, :3]
    assert not np.allclose(regular, inertial)
    assert np.allclose(env.observations()['actor'][3:6], regular/5, atol=1e-12)


def test_one_native_policy_tick_has_ten_steps_and_delayed_bounded_targets():
    env = runtime.NativeStanceRuntime()
    initial = env.observations()
    assert initial['actor'].shape == (44,) and initial['critic'].shape == (50,)
    result = env.step(np.ones(10))
    assert result['executed_physics_steps'] == 10
    assert env.data.time == pytest.approx(.02) and env.steps == 10
    trace = env.trace(); assert len(trace['frames']) == 11 and len(trace['commands']) == 10
    for command in trace['commands'][:3]:
        assert np.allclose(command['delayed_target_rad'], env.nominal)
    realized = np.asarray(trace['commands'][3]['delayed_target_rad'])-env.nominal
    assert np.allclose(realized[list(runtime.lesson.LEG_IDS)], .02)
    assert np.array_equal(realized[5:9], np.zeros(4))
    assert all(c['applied'] for c in trace['commands'])
    assert not result['terminated'] and not result['timed_out']
    assert not trace['policy_training'] and not trace['gpu_parity']


def test_excessive_proposed_torque_never_reaches_mujoco_and_terminal_is_frozen(monkeypatch):
    env = runtime.NativeStanceRuntime()
    before = env.data.qpos.copy()
    monkeypatch.setattr(runtime.bam, 'compute_snapshot', lambda *a: dict(torque_nm=[.37]*14))
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('unsafe physics call'))
    result = env.step(np.zeros(10))
    assert result['reward'] == -2 and result['terminated']
    assert result['executed_physics_steps'] == 0
    assert env.data.time == 0 and np.array_equal(before, env.data.qpos)
    assert not np.any(env.data.ctrl)
    assert env.trace()['commands'][0]['applied'] is False
    terminal = env.trace()
    with pytest.raises(RuntimeError, match='closed'): env.step(np.zeros(10))
    assert env.trace() == terminal
    env.observations()
    assert np.array_equal(before, env.data.qpos) and env.data.time == 0


def test_post_step_first_failure_breaks_decimation_and_is_not_reset(monkeypatch):
    env = runtime.NativeStanceRuntime()
    original = mujoco.mj_step; calls = []
    def tilt_after_first_step(model, data):
        original(model, data); calls.append(float(data.time))
        # Inject an explicit synthetic tilt after a real native integration step.
        data.qpos[3:7] = [np.cos(.4/2), 0, np.sin(.4/2), 0]
    monkeypatch.setattr(mujoco, 'mj_step', tilt_after_first_step)
    result = env.step(np.zeros(10))
    assert calls == pytest.approx([.002])
    assert result['terminated'] and result['executed_physics_steps'] == 1
    assert result['terminal_frame']['tilt_rad'] == pytest.approx(.4)
    assert len(env.trace()['frames']) == 2 and env.data.time == pytest.approx(.002)


def test_explicit_reset_clears_motor_delay_control_and_reward_episode_state():
    env = runtime.NativeStanceRuntime(); original_qpos = env.data.qpos.copy()
    env.step(np.ones(10))
    env.reset()
    assert env.steps == 0 and env.data.time == 0 and not env.done
    assert np.array_equal(env.data.qpos, original_qpos)
    assert not np.any(env.data.qvel) and not np.any(env.data.ctrl)
    assert not env.actuator._prev_motor_torque.any()
    assert not np.any(env.model.dof_frictionloss[env.dofs])
    assert not np.any(env.model.dof_damping[env.dofs])
    env.step(-np.ones(10))
    assert all(np.allclose(c['delayed_target_rad'], env.nominal)
               for c in env.trace()['commands'][:3])


def test_independent_world_does_not_advance_a_closed_sibling(monkeypatch):
    closed = runtime.NativeStanceRuntime(); live = runtime.NativeStanceRuntime()
    original = runtime.bam.compute_snapshot
    def excessive(model, data, actuator, target):
        if data is closed.data: return dict(torque_nm=[.37]*14)
        return original(model, data, actuator, target)
    monkeypatch.setattr(runtime.bam, 'compute_snapshot', excessive)
    closed.step(np.zeros(10)); terminal = closed.trace()
    live.step(np.zeros(10))
    assert live.steps == 10 and closed.steps == 0 and closed.trace() == terminal


def test_nonfinite_action_faults_job_and_cannot_be_reset_or_stepped(monkeypatch):
    env = runtime.NativeStanceRuntime()
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('invalid action applied'))
    with pytest.raises(ValueError): env.step(np.full(10, np.nan))
    assert env.faulted and env.steps == 0
    with pytest.raises(RuntimeError, match='faulted'): env.reset()
    with pytest.raises(RuntimeError, match='closed'): env.step(np.zeros(10))


def test_invalid_pre_step_state_terminates_without_computation_or_integration(monkeypatch):
    env = runtime.NativeStanceRuntime()
    env.data.qpos[3:7] = [np.cos(.4/2), 0, np.sin(.4/2), 0]
    mujoco.mj_forward(env.model, env.data)
    monkeypatch.setattr(runtime.bam, 'compute_snapshot', lambda *a: pytest.fail('computed after invalid boundary'))
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('stepped after invalid boundary'))
    result = env.step(np.zeros(10))
    assert result['terminated'] and result['reward'] == -2
    assert result['executed_physics_steps'] == 0 and env.data.time == 0
    assert env.trace()['commands'] == []


def test_returned_observation_and_trace_do_not_mutate_runtime():
    env = runtime.NativeStanceRuntime()
    obs = env.observations(); obs['actor'][:] = 100
    trace = env.trace(); trace['frames'][0]['qpos'][0] = 100
    assert env.data.qpos[0] == 0 and env.observations()['actor'][0] == 0
