"""Analytic world-projected heading-rate EMA reward contract."""

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


class _Env:
    def __init__(self, worlds=2, dtype=torch.float32):
        self.num_envs = worlds
        self.step_dt = .02
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((worlds,), 10, dtype=torch.long)
        self.command = torch.zeros((worlds, 3), dtype=dtype)
        self.quat = torch.zeros((worlds, 4), dtype=dtype)
        self.quat[:, 0] = 1
        self.angular = torch.zeros((worlds, 3), dtype=dtype)
        self.command_manager = SimpleNamespace(get_command=lambda _name: self.command)
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_quat_w=self.quat, root_link_ang_vel_b=self.angular))}


def _step(env, *, yaw=None, angular=None, quat=None, episode=None):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    if yaw is not None:
        env.command[:, 2] = yaw
    if angular is not None:
        env.angular.copy_(torch.as_tensor(angular, dtype=env.angular.dtype).reshape_as(env.angular))
    if quat is not None:
        env.quat.copy_(torch.as_tensor(quat, dtype=env.quat.dtype).reshape_as(env.quat))
    if episode is not None:
        env.episode_length_buf.copy_(torch.as_tensor(episode, dtype=torch.long))
    return mdp.projected_yaw_bias_cost(env)


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def _axis_quat(axis, angle):
    axis = torch.as_tensor(axis, dtype=angle.dtype, device=angle.device)
    axis = axis / torch.linalg.vector_norm(axis)
    return torch.cat((torch.cos(angle / 2).reshape(1), axis * torch.sin(angle / 2)))


def _heading(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (x*y + w*z), 1 - 2 * (y*y + z*z))


def test_upright_projected_yaw_matches_body_yaw_ema_across_resets_commands_and_rollback():
    projected, body = _Env(), _Env()
    samples = [(.2, .0, 11), (.2, -.1, 12), (.2, .05, 1), (-.3, .1, 2),
               (-.3, -.2, 3), (-.3, .4, 4)]
    for command, measured, episode in samples:
        projected.command[:, 2] = body.command[:, 2] = command
        projected.angular[:, 2] = body.angular[:, 2] = measured
        projected.episode_length_buf.fill_(episode)
        body.episode_length_buf.fill_(episode)
        if episode == 1:
            projected.common_step_counter = body.common_step_counter = 0
        else:
            projected.common_step_counter += 1
            body.common_step_counter += 1
        a = mdp.projected_yaw_bias_cost(projected)
        b = mdp.body_yaw_bias_cost(body)
        assert torch.equal(a, b)
    projected.common_step_counter = body.common_step_counter = 0
    assert torch.equal(mdp.projected_yaw_bias_cost(projected), mdp.body_yaw_bias_cost(body))


def test_tilted_analytic_heading_rate_differs_from_body_z_and_matches_q_integration():
    env = _Env(1, torch.float64)
    roll = _axis_quat([1, 0, 0], torch.tensor(.7, dtype=torch.float64))
    pitch = _axis_quat([0, 1, 0], torch.tensor(-.45, dtype=torch.float64))
    env.quat[0] = _quat_mul(roll, pitch)
    env.angular[0] = torch.tensor([.8, -.35, .42], dtype=torch.float64)
    mdp.projected_yaw_bias_cost(env)
    rate = env._microduck_projected_yaw_bias_state["projected_yaw_rate_rad_s"][0]
    assert abs(rate.item() - env.angular[0, 2].item()) > .05

    eps = 1.0e-6
    omega = env.angular[0]
    # qdot = 1/2 q * [0, omega_body]. Use symmetric quaternion integration.
    delta = torch.cat((torch.zeros(1, dtype=torch.float64), omega)) * (eps / 2)
    q_plus = env.quat[0] + _quat_mul(env.quat[0], delta)
    q_minus = env.quat[0] - _quat_mul(env.quat[0], delta)
    q_plus = q_plus / torch.linalg.vector_norm(q_plus)
    q_minus = q_minus / torch.linalg.vector_norm(q_minus)
    dh = torch.atan2(torch.sin(_heading(q_plus)-_heading(q_minus)),
                     torch.cos(_heading(q_plus)-_heading(q_minus))) / (2*eps)
    assert rate.item() == pytest.approx(dh.item(), abs=2e-8)


def test_global_world_z_yaw_rotation_leaves_projected_rate_unchanged():
    q = _quat_mul(_axis_quat([0, 0, 1], torch.tensor(1.1)),
                  _quat_mul(_axis_quat([1, 0, 0], torch.tensor(.8)),
                            _axis_quat([0, 1, 0], torch.tensor(.3))))
    env_a, env_b = _Env(1), _Env(1)
    env_a.quat[0] = q
    env_b.quat[0] = _quat_mul(_axis_quat([0, 0, 1], torch.tensor(-2.2)), q)
    env_a.angular[0] = env_b.angular[0] = torch.tensor([.4, -.7, .2])
    mdp.projected_yaw_bias_cost(env_a)
    mdp.projected_yaw_bias_cost(env_b)
    assert torch.allclose(env_a._microduck_projected_yaw_bias_state["projected_yaw_rate_rad_s"],
                          env_b._microduck_projected_yaw_bias_state["projected_yaw_rate_rad_s"],
                          atol=2e-6, rtol=0)


@pytest.mark.parametrize("command", [.4, -.4])
def test_signed_bias_tracks_and_alternating_residual_cancels(command):
    env = _Env(1)
    env.command[0, 2] = command
    for _ in range(80):
        out = _step(env)
    assert out.item() == pytest.approx(abs(command) * (1 - .98**80), abs=2e-6)
    assert env._microduck_projected_yaw_bias_state["signed_ema"].item() == pytest.approx(
        command * (1 - .98**80), abs=2e-6)
    env = _Env(1)
    for i in range(200):
        _step(env, angular=[[0, 0, 1 if i % 2 == 0 else -1]])
    assert env._microduck_projected_yaw_bias_state["cost"].item() < .011


def test_invalid_projection_finite_cost_and_fresh_recovery_ema():
    env = _Env(1)
    env.command[0, 2] = .5
    # Body X points nearly vertical: 90-degree pitch is below the XY threshold.
    invalid = _axis_quat([0, 1, 0], torch.tensor(torch.pi / 2))
    out = _step(env, quat=invalid)
    state = env._microduck_projected_yaw_bias_state
    assert out.item() == 1.0 and not state["projection_valid"].item()
    assert state["signed_residual"].item() == state["signed_ema"].item() == 0.0
    valid = torch.tensor([1., 0., 0., 0.])
    out = _step(env, quat=valid)
    assert out.item() == pytest.approx(.02 * .5)
    assert env._microduck_projected_yaw_bias_state["projection_valid"].item()


def test_no_preaction_priming_cumulative_target_or_non_yaw_command_reset():
    env = _Env(1)
    env.command[0] = torch.tensor([0., 0., .3])
    first = mdp.projected_yaw_bias_cost(env)
    assert first.item() == pytest.approx(.006)
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env.command[0, 0] = 2.0  # only command yaw changes clear EMA history
    second = mdp.projected_yaw_bias_cost(env)
    assert second.item() == pytest.approx(.01188)
    assert "target_heading" not in env._microduck_projected_yaw_bias_state


def test_same_step_cache_is_idempotent_alias_free_and_compares_all_measurements():
    env = _Env(1)
    env.command[0, 2] = .2
    first = mdp.projected_yaw_bias_cost(env)
    expected = first.item()
    first.zero_()
    env._microduck_projected_yaw_bias_state["signed_ema"].zero_()
    again = mdp.projected_yaw_bias_cost(env)
    assert again.item() == pytest.approx(expected)
    assert again.data_ptr() != env._microduck_projected_yaw_bias_internal["cost"].data_ptr()
    env.angular[0, 0] = .1
    with pytest.raises(RuntimeError, match="inputs changed"):
        mdp.projected_yaw_bias_cost(env)


def test_invalid_inputs_and_layout_fail_before_private_cache_mutation():
    env = _Env(1)
    env.quat[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        mdp.projected_yaw_bias_cost(env)
    assert not hasattr(env, "_microduck_projected_yaw_bias_internal")
    env.quat[0] = torch.tensor([2., 0., 0., 0.])
    with pytest.raises(ValueError, match="unit normalized"):
        mdp.projected_yaw_bias_cost(env)
    env.quat[0] = torch.tensor([1., 0., 0., 0.])
    env.angular = torch.zeros((1, 2))
    env.scene["robot"].data.root_link_ang_vel_b = env.angular
    with pytest.raises(ValueError, match="shape"):
        mdp.projected_yaw_bias_cost(env)
    env.angular = torch.zeros((1, 3))
    env.scene["robot"].data.root_link_ang_vel_b = env.angular
    env.episode_length_buf = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="episode_length_buf"):
        mdp.projected_yaw_bias_cost(env)
    assert not hasattr(env, "_microduck_projected_yaw_bias_internal")


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_tau_rejected(tau):
    with pytest.raises(ValueError):
        mdp.projected_yaw_bias_cost(_Env(1), tau_s=tau)


def test_counter_skip_config_mismatch_and_step_dt_drift_rejected():
    env = _Env(1)
    mdp.projected_yaw_bias_cost(env)
    with pytest.raises(ValueError, match="incompatible config"):
        mdp.projected_yaw_bias_cost(env, tau_s=.5)
    env.common_step_counter = 2
    with pytest.raises(RuntimeError, match="skipped forward"):
        mdp.projected_yaw_bias_cost(env)


def test_initialized_layout_metadata_and_nonfinite_measured_input_rejected():
    env = _Env(1)
    mdp.projected_yaw_bias_cost(env)
    env.episode_length_buf = env.episode_length_buf.to(torch.int32)
    with pytest.raises(ValueError, match="incompatible config"):
        mdp.projected_yaw_bias_cost(env)
    fresh = _Env(1)
    fresh.angular[0, 1] = float("inf")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        mdp.projected_yaw_bias_cost(fresh)
    assert not hasattr(fresh, "_microduck_projected_yaw_bias_internal")


