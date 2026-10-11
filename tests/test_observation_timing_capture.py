from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import observation_timing_capture as capture
from mjlab_microduck.tasks import mdp


# Independent literals from the raw actor terms in the actual play factory.
TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
         "command", "head_command", "body_command")
DIMS = ((3,), (3,), (14,), (14,), (14,), (3,), (4,), (6,))


def fixture_capture_state():
    from mjlab.utils.buffers import DelayBuffer
    from mjlab.utils.noise import UniformNoiseCfg

    generator = torch.Generator(device="cpu").manual_seed(83)
    delay = DelayBuffer(min_lag=0, max_lag=1, batch_size=8, device="cpu",
                        per_env=True, hold_prob=0., update_period=64,
                        per_env_phase=True, generator=generator)
    old = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 100.
    newest = old + .2
    delay._buffer.append(old)
    delay._buffer.append(newest)
    lag = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    delay.set_lags(lag)
    valid = torch.minimum(delay.current_lags, delay._buffer.current_length - 1).clamp_min(0)
    selected = delay._buffer[valid]
    command = torch.tensor([.3, 0., .2]).repeat(8, 1)
    actor = torch.zeros(8, 61)
    actor[:, :3] = selected
    actor[:, 48:51] = command
    term = NS(func=mdp.base_ang_vel_imu_misaligned, params={"max_angle_deg": 6.},
              scale=None, clip=None, history_length=0, delay_min_lag=0,
              delay_max_lag=1, delay_update_period=64, delay_per_env=True,
              delay_per_env_phase=True, delay_hold_prob=0.,
              noise=UniformNoiseCfg(n_min=-.03, n_max=.03, operation="add"))
    manager = NS(
        active_terms={"actor": TERMS}, group_obs_term_dim={"actor": DIMS},
        group_obs_concatenate={"actor": True}, cfg={"actor": NS(enable_corruption=True)},
        _obs_buffer={"actor": actor},
        _group_obs_term_delay_buffer={"actor": {"base_ang_vel": delay}},
        get_term_cfg=lambda group, name: term,
    )
    manager.compute_calls = 0
    manager.compute = lambda *args, **kwargs: setattr(manager, "compute_calls", manager.compute_calls + 1)
    quat = torch.tensor([.98, .1, .12, .08]).repeat(8, 1)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    body = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10.
    time = torch.arange(8, dtype=torch.float32) * .02 + 1.
    robot = NS(data=NS(root_link_ang_vel_b=body, data=NS(time=time)))
    actuator = NS(calls=0)
    actuator.update = lambda *args, **kwargs: setattr(actuator, "calls", actuator.calls + 1)
    sensor = NS(calls=0)
    sensor.compute = lambda *args, **kwargs: setattr(sensor, "calls", sensor.calls + 1)
    env = NS(
        num_envs=8, step_dt=.02, physics_dt=.005,
        cfg=NS(decimation=4), observation_manager=manager,
        command_manager=NS(get_command=lambda name: command), scene={"robot": robot},
        actuator_manager=actuator, sensor_manager=sensor,
        common_step_counter=21, _sim_step_counter=84,
        episode_length_buf=torch.arange(8, dtype=torch.long) + 10,
        _imu_misalign_quat=quat,
    )
    return NS(env=env, robot=robot, actor=actor, command=command, delay=delay,
              term=term, manager=manager, old=old, newest=newest, selected=selected)


def test_capture_pre_reads_actual_cpu_delay_buffer_without_compute_rng_or_mutation():
    f = fixture_capture_state()
    rng = torch.random.get_rng_state().clone()
    actor = f.actor.clone()
    command = f.command.clone()
    buffer = f.delay._buffer.buffer.clone()
    lag = f.delay.current_lags.clone()
    steps = f.delay._step_count.clone()
    length = f.delay._buffer.current_length.clone()

    result = capture.capture_pre(f.env, f.robot, f.actor)

    assert torch.equal(result["actor_angular"], f.delay._buffer[lag])
    assert torch.equal(result["selected_noisy_angular"], f.delay._buffer[lag])
    assert torch.equal(result["newest_noisy_angular"], f.delay._buffer[torch.zeros_like(lag)])
    assert result["sampled_lag"].tolist() == lag.tolist()
    assert result["effective_lag"].tolist() == lag.tolist()
    assert torch.equal(result["history_length"], length)
    assert torch.equal(result["lag_step_count"], steps)
    assert result["common_step"] == 21 and result["sim_step"] == 84
    assert result["episode_length"].dtype == torch.int64
    assert result["imu_quat_wxyz"].shape == (8, 4)
    assert torch.equal(f.actor, actor) and torch.equal(f.command, command)
    assert torch.equal(f.delay._buffer.buffer, buffer)
    assert torch.equal(f.delay.current_lags, lag) and torch.equal(f.delay._step_count, steps)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert f.manager.compute_calls == 0
    assert f.env.actuator_manager.calls == f.env.sensor_manager.calls == 0

    # Every retained field is a detached clone, not a view into manager/history state.
    expected = result["selected_noisy_angular"].clone()
    f.actor[:, :3].add_(1.)
    f.delay._buffer._buffer[f.delay._buffer._pointer].add_(1.)
    f.command.zero_()
    assert torch.equal(result["selected_noisy_angular"], expected)


@pytest.mark.parametrize("fault", [
    "layout", "dimension", "concat", "corruption", "term-func", "term-params",
    "scale", "clip", "history", "lag-config", "noise", "delay-config",
    "cache-mismatch", "nonfinite", "command-shape", "lag-dtype", "no-history",
])
def test_capture_pre_rejects_observation_or_delay_contract_drift(fault):
    f = fixture_capture_state()
    if fault == "layout":
        f.manager.active_terms["actor"] = tuple(reversed(TERMS))
    elif fault == "dimension":
        f.manager.group_obs_term_dim["actor"] = DIMS[:-1]
    elif fault == "concat":
        f.manager.group_obs_concatenate["actor"] = False
    elif fault == "corruption":
        f.manager.cfg["actor"].enable_corruption = False
    elif fault == "term-func":
        f.term.func = lambda *_args: None
    elif fault == "term-params":
        f.term.params["max_angle_deg"] = 5.
    elif fault == "scale":
        f.term.scale = 1.
    elif fault == "clip":
        f.term.clip = (-1., 1.)
    elif fault == "history":
        f.term.history_length = 2
    elif fault == "lag-config":
        f.term.delay_update_period = 32
    elif fault == "noise":
        f.term.noise.n_max = .04
    elif fault == "delay-config":
        f.delay.update_period = 32
    elif fault == "cache-mismatch":
        f.actor[0, 1] += .1
    elif fault == "nonfinite":
        f.actor[0, 0] = float("nan")
    elif fault == "command-shape":
        f.command_manager = f.env.command_manager
        f.env.command_manager.get_command = lambda _name: torch.zeros(8, 2)
    elif fault == "lag-dtype":
        f.delay._current_lags = f.delay.current_lags.float()
    else:
        from mjlab.utils.buffers import CircularBuffer
        f.delay._buffer = CircularBuffer(max_len=2, batch_size=8, device="cpu")
    with pytest.raises(ValueError):
        capture.capture_pre(f.env, f.robot, f.actor)


def test_capture_reward_and_attach_post_clone_terminal_clocks_before_reset(monkeypatch):
    from mjlab_microduck import angular_reward_diagnostics as angular

    env = NS(common_step_counter=21, _sim_step_counter=84, cfg=NS(decimation=4),
             scene={"robot": NS(data=NS(data=NS(time=torch.arange(8, dtype=torch.float32) * .02 + 1.)))},
             reset_buf=torch.tensor([False, False, True, False, False, False, False, False]))
    snapshot = torch.arange(40, dtype=torch.float32).reshape(8, 5)
    env._microduck_angular_reward_snapshot = snapshot.clone()
    calls = []

    def angular_capture(received):
        calls.append(received)
        return torch.zeros(8)

    monkeypatch.setattr(angular, "capture", angular_capture)
    assert torch.equal(capture.capture_reward(env), torch.zeros(8))
    post = env._microduck_observation_reward_snapshot
    assert calls == [env] and post["common_step"] == 21 and post["sim_step"] == 84
    assert post["terminal"].tolist() == env.reset_buf.tolist()
    assert torch.equal(post["angular_reward"], snapshot)

    pre = {"common_step": 20, "sim_step": 80}
    env.common_step_counter = 21
    env._sim_step_counter = 84
    pair = capture.attach_post(env, pre)
    assert pair["pre"] is pre and pair["post"] is post
    assert env._microduck_observation_reward_snapshot is None
    with pytest.raises(ValueError, match="previous observation reward clock"):
        env._microduck_observation_reward_snapshot = post
        capture.capture_reward(env)


def make_synthetic_trace(steps=61, *, terminal_at=None):
    from mjlab.utils.lab_api.math import quat_apply

    samples = []
    worlds = torch.arange(8, dtype=torch.float64)
    q = torch.tensor([.98, .2, .0, .0], dtype=torch.float64)
    q = q / q.norm()
    quat = q.repeat(8, 1)
    noise = torch.stack((worlds * .001, worlds * -.0005, worlds * .002), dim=-1)
    for step in range(steps):
        body = torch.stack((.1 + worlds * .001, -.05 + worlds * .002,
                            .2 + step * .003 + worlds * .001), dim=-1)
        rotated = quat_apply(quat, body)
        newest = rotated + noise
        prev_body = body if step == 0 else torch.stack((.1 + worlds * .001, -.05 + worlds * .002,
                                                        .2 + (step - 1) * .003 + worlds * .001), dim=-1)
        previous_noisy = quat_apply(quat, prev_body) + noise
        lag = (torch.arange(8, dtype=torch.long) + step) % 2
        history_length = torch.full((8,), min(step + 1, 2), dtype=torch.long)
        effective_lag = torch.minimum(lag, history_length - 1)
        selected = torch.where(effective_lag[:, None] == 0, newest, previous_noisy)
        pre_time = (torch.full((8,), 1., dtype=torch.float64) if step == 0 else
                    samples[-1]["post"]["physics_time"].clone())
        terminal = torch.zeros(8, dtype=torch.bool)
        if terminal_at == step:
            terminal[2] = True
        next_body = torch.stack((.1 + worlds * .001, -.05 + worlds * .002,
                                 .2 + (step + 1) * .003 + worlds * .001), dim=-1)
        post_time = pre_time + .02
        pre = dict(
            common_step=100 + step, sim_step=400 + step * 4,
            physics_time=pre_time, episode_length=torch.full((8,), step, dtype=torch.long),
            body_angular=body, actor_angular=selected, command=torch.tensor([.3, 0., 0.]).repeat(8, 1).double(),
            newest_noisy_angular=newest, selected_noisy_angular=selected, imu_quat_wxyz=quat,
            sampled_lag=lag, effective_lag=effective_lag, history_length=history_length,
            lag_step_count=torch.full((8,), step + 2, dtype=torch.long),
        )
        post = dict(common_step=101 + step, sim_step=404 + step * 4,
                    physics_time=post_time, terminal=terminal,
                    angular_reward=torch.cat((next_body, torch.zeros(8, 2)), dim=-1))
        samples.append(dict(pre=pre, post=post))
    return capture.serialize(samples, "a" * 64)


def test_summary_decomposes_mount_noise_and_lag_and_accepts_terminal_before_reset():
    trace = make_synthetic_trace(61, terminal_at=60)
    result = capture.summarize(trace)
    assert result["settled_steps"] == 1
    metrics = result["per_world"]
    assert metrics["lag_one_fraction"] == pytest.approx([.0, 1., 0., 1., 0., 1., 0., 1.])
    assert metrics["fresh_noise_yaw_mae"] == pytest.approx([0., .002, .004, .006, .008, .010, .012, .014])
    assert metrics["mounting_yaw_difference_mae"][0] > 0.
    assert metrics["delay_yaw_difference_mae"][1] > 0.
    assert metrics["reward_phase_to_next_body_yaw_difference_mae"] is None
    complete = capture.summarize(make_synthetic_trace(62))
    assert complete["per_world"]["reward_phase_to_next_body_yaw_difference_mae"] == pytest.approx([0.] * 8)
    assert result["policy_acceptance"] is result["physical_motion_authorized"] is False


@pytest.mark.parametrize("fault", [
    "clock", "sim-clock", "physics-clock", "nonconsecutive", "early-terminal",
    "lag", "actor", "quaternion", "noise", "bad-shape", "episode-clock", "lag-clock",
])
def test_summary_rejects_clock_lag_noise_and_mounting_drift(fault):
    trace = make_synthetic_trace(62)
    row = trace["samples"][60]
    if fault == "clock":
        row["post"]["common_step"] += 1
    elif fault == "sim-clock":
        row["post"]["sim_step"] += 1
    elif fault == "physics-clock":
        row["post"]["physics_time"][0] += .005
    elif fault == "nonconsecutive":
        trace["samples"][61]["pre"]["physics_time"][0] += .005
    elif fault == "early-terminal":
        trace["samples"][2]["post"]["terminal"][0] = True
    elif fault == "lag":
        row["pre"]["effective_lag"][0] = 1 - row["pre"]["sampled_lag"][0]
    elif fault == "actor":
        row["pre"]["actor_angular"][0][2] += .1
    elif fault == "quaternion":
        row["pre"]["imu_quat_wxyz"][0][0] = 2.
    elif fault == "noise":
        row["pre"]["newest_noisy_angular"][0][0] += .1
    elif fault == "episode-clock":
        row["pre"]["episode_length"][0] = -1
    elif fault == "lag-clock":
        row["pre"]["lag_step_count"][0] += 1
    else:
        row["pre"]["body_angular"][0] = [1., 2.]
    with pytest.raises(ValueError):
        capture.summarize(trace)
