"""Read-only actor/reward timing trace; no sensor or policy recomputation."""

import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import angular_reward_diagnostics as angular

PROTOCOL = "actor-angular-observation-timing-v1"
TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
         "command", "head_command", "body_command")
DIMS = ((3,), (3,), (14,), (14,), (14,), (3,), (4,), (6,))


def clone(value):
    return value.detach().cpu().clone()


def capture_pre(env, robot, actor):
    """Read cached manager output before checkpoint normalization and inference."""
    from mjlab.utils.noise import UniformNoiseCfg
    from mjlab_microduck.tasks import mdp

    manager = env.observation_manager
    base.require(env.num_envs == 8 and env.step_dt == .02 and env.physics_dt == .005 and env.cfg.decimation == 4
                 and tuple(manager.active_terms["actor"]) == TERMS
                 and tuple(manager.group_obs_term_dim["actor"]) == DIMS
                 and manager.group_obs_concatenate["actor"]
                 and manager.cfg["actor"].enable_corruption,
                 "pinned raw 61D corrupted actor layout")
    term = manager.get_term_cfg("actor", "base_ang_vel")
    base.require(term.func is mdp.base_ang_vel_imu_misaligned
                 and term.params == {"max_angle_deg": 6.}
                 and term.scale is None and term.clip is None and term.history_length == 0
                 and (term.delay_min_lag, term.delay_max_lag, term.delay_update_period) == (0, 1, 64)
                 and term.delay_per_env and term.delay_per_env_phase and term.delay_hold_prob == 0.
                 and isinstance(term.noise, UniformNoiseCfg)
                 and (term.noise.n_min, term.noise.n_max, term.noise.operation) == (-.03, .03, "add"),
                 "pinned effective angular observation config")
    base.require(actor.shape == (8, 61) and base.finite_tree(actor)
                 and torch.equal(actor, manager._obs_buffer["actor"]),
                 "already-computed raw actor cache")
    delay = manager._group_obs_term_delay_buffer["actor"]["base_ang_vel"]
    base.require((delay.min_lag, delay.max_lag, delay.update_period) == (0, 1, 64)
                 and delay.per_env and delay.per_env_phase and delay.hold_prob == 0.,
                 "pinned live delay buffer config")
    history = delay._buffer
    lag, length = delay.current_lags, history.current_length
    base.require(lag.shape == length.shape == (8,) and lag.dtype == length.dtype == torch.int64
                 and bool(((lag >= 0) & (lag <= 1) & (length >= 1) & (length <= 2)).all()),
                 "known initialized delay history")
    valid = torch.minimum(lag, length - 1).clamp_min(0)
    # Indexed reads do not advance lag state or consume RNG; compute() would.
    selected, newest = history[valid], history[torch.zeros_like(valid)]
    q = getattr(env, "_imu_misalign_quat", None)
    command = env.command_manager.get_command("twist")
    body = robot.data.root_link_ang_vel_b
    time = robot.data.data.time[:]
    base.require(isinstance(q, torch.Tensor) and q.shape == (8, 4)
                 and command.shape == body.shape == selected.shape == newest.shape == (8, 3)
                 and time.shape == (8,) and base.finite_tree((q, command, body, selected, newest, time))
                 and torch.allclose(q.square().sum(-1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=0)
                 and torch.equal(actor[:, :3], selected)
                 and torch.equal(actor[:, 48:51], command),
                 "actual selected angular cache and existing mounting rotation")
    base.require(type(env.common_step_counter) is int and env.common_step_counter >= 0
                 and type(env._sim_step_counter) is int and env._sim_step_counter >= 0
                 and env.episode_length_buf.shape == (8,)
                 and env.episode_length_buf.dtype == torch.int64
                 and bool((env.episode_length_buf >= 0).all()) and bool((time >= 0).all()),
                 "integer nonnegative pre-action clocks")
    return dict(common_step=env.common_step_counter, sim_step=env._sim_step_counter,
        physics_time=clone(time), episode_length=clone(env.episode_length_buf),
        body_angular=clone(body), actor_angular=clone(actor[:, :3]), command=clone(command),
        newest_noisy_angular=clone(newest), selected_noisy_angular=clone(selected),
        imu_quat_wxyz=clone(q), sampled_lag=clone(lag), effective_lag=clone(valid),
        history_length=clone(length), lag_step_count=clone(delay._step_count))


def capture_reward(env):
    """Metrics hook after reward, before reset/final forward/next observation."""
    base.require(getattr(env, "_microduck_observation_reward_snapshot", None) is None,
                 "previous observation reward clock consumed")
    result = angular.capture(env)
    env._microduck_observation_reward_snapshot = dict(
        common_step=env.common_step_counter, sim_step=env._sim_step_counter,
        physics_time=clone(env.scene["robot"].data.data.time[:]),
        terminal=clone(env.reset_buf.bool()),
        angular_reward=clone(env._microduck_angular_reward_snapshot))
    return result


def attach_post(env, pre):
    post = env._microduck_observation_reward_snapshot
    base.require(isinstance(post, dict) and post["common_step"] == pre["common_step"] + 1
                 and post["sim_step"] == pre["sim_step"] + env.cfg.decimation,
                 "paired pre-action and pre-reset reward clocks")
    env._microduck_observation_reward_snapshot = None
    return dict(pre=pre, post=post)


def serialize(samples, checkpoint_sha256):
    base.require(type(samples) is list and 0 < len(samples) <= 240,
                 "bounded first-episode observation trace")
    return dict(protocol=PROTOCOL, checkpoint_sha256=checkpoint_sha256,
        step_dt=.02, startup_steps=60, steps=len(samples), worlds=8,
        actor_view="raw manager output before checkpoint observation normalizer",
        pre_timing="pre-action post-forward state; cached noisy rotated delayed body angular velocity",
        post_timing="post-reward pre-reset pre-final-forward derived angular state may be one physics substep stale",
        samples=[{phase: {key: value.tolist() if isinstance(value, torch.Tensor) else value
                         for key, value in sample[phase].items()} for phase in ("pre", "post")}
                 for sample in samples],
        policy_acceptance=False, physical_motion_authorized=False)


def summarize(trace):
    from mjlab.utils.lab_api.math import quat_apply

    base.require(trace["protocol"] == PROTOCOL and trace["step_dt"] == .02
                 and trace["startup_steps"] == 60 and trace["worlds"] == 8
                 and trace["steps"] == len(trace["samples"]) and 0 < trace["steps"] <= 240,
                 "pinned observation timing trace")
    base.require(trace["policy_acceptance"] is False and trace["physical_motion_authorized"] is False,
                 "non-admitting observation trace")
    rows = trace["samples"]
    for i, row in enumerate(rows):
        pre, post = row["pre"], row["post"]
        base.require(all(type(phase[key]) is int and phase[key] >= 0
                         for phase in (pre, post) for key in ("common_step", "sim_step"))
                     and len(post["terminal"]) == 8 and all(type(v) is bool for v in post["terminal"]),
                     "integer clocks and boolean terminal worlds")
        for key, shape in (("body_angular", (8, 3)), ("actor_angular", (8, 3)),
                           ("newest_noisy_angular", (8, 3)), ("selected_noisy_angular", (8, 3)),
                           ("imu_quat_wxyz", (8, 4)), ("command", (8, 3)),
                           ("physics_time", (8,)), ("episode_length", (8,)),
                           ("sampled_lag", (8,)), ("effective_lag", (8,)),
                           ("history_length", (8,)), ("lag_step_count", (8,))):
            value = torch.tensor(pre[key], dtype=torch.float64)
            base.require(value.shape == shape and base.finite_tree(value), "finite retained " + key)
        for key, shape in (("physics_time", (8,)), ("angular_reward", (8, 5))):
            value = torch.tensor(post[key], dtype=torch.float64)
            base.require(value.shape == shape and base.finite_tree(value), "finite retained reward " + key)
        lag, length = torch.tensor(pre["sampled_lag"]), torch.tensor(pre["history_length"])
        valid = torch.tensor(pre["effective_lag"])
        base.require(all(type(v) is int and v >= 0 for key in ("episode_length", "lag_step_count")
                         for v in pre[key]), "retained nonnegative integer observation clocks")
        base.require(lag.dtype == length.dtype == valid.dtype == torch.int64
                     and bool(((lag >= 0) & (lag <= 1) & (length >= 1) & (length <= 2)).all())
                     and torch.equal(valid, torch.minimum(lag, length-1).clamp_min(0)),
                     "retained effective lag matches available history")
        q = torch.tensor(pre["imu_quat_wxyz"], dtype=torch.float64)
        base.require(torch.allclose(q.square().sum(-1), torch.ones(8, dtype=torch.float64), atol=1e-5, rtol=0),
                     "retained unit mounting quaternion")
        pre_time, post_time = (torch.tensor(phase["physics_time"], dtype=torch.float64) for phase in (pre, post))
        base.require(torch.allclose(post_time-pre_time, torch.full((8,), .02, dtype=torch.float64), atol=1e-5, rtol=0),
                     "one control step between pre-action and pre-reset physics clocks")
        base.require(post["common_step"] == pre["common_step"] + 1
                     and post["sim_step"] == pre["sim_step"] + 4,
                     "serialized paired clock ordering")
        if i:
            old = rows[i-1]["post"]
            base.require(pre["physics_time"] == old["physics_time"], "consecutive physics-clock identity")
            base.require(not any(old["terminal"])
                         and pre["common_step"] == old["common_step"]
                         and pre["sim_step"] == old["sim_step"],
                         "consecutive first-episode clock identity")
        base.require(not any(post["terminal"]) or i == len(rows)-1,
                     "stop at first terminal reward, never pair reset observations")
        actor = torch.tensor(pre["actor_angular"], dtype=torch.float64)
        selected = torch.tensor(pre["selected_noisy_angular"], dtype=torch.float64)
        base.require(actor.shape == selected.shape == (8, 3) and torch.equal(actor, selected),
                     "retained cache exactly equals selected history")
        newest = torch.tensor(pre["newest_noisy_angular"], dtype=torch.float64)
        base.require(torch.equal(selected[valid == 0], newest[valid == 0]),
                     "zero-lag history is the current noisy observation")
        reward = torch.tensor(post["angular_reward"], dtype=torch.float64)
        command = torch.tensor(pre["command"], dtype=torch.float64)
        base.require(torch.equal(reward[:, 3], command[:, 2]),
                     "reward retains the same action-step yaw command")
        if i:
            previous = rows[i-1]["pre"]
            previous_newest = torch.tensor(previous["newest_noisy_angular"], dtype=torch.float64)
            base.require(torch.equal(selected[valid == 1], previous_newest[valid == 1]),
                         "one-lag history is the preceding sampled observation")
            base.require(pre["imu_quat_wxyz"] == previous["imu_quat_wxyz"]
                         and all(b == a + 1 for a, b in zip(previous["lag_step_count"], pre["lag_step_count"], strict=True))
                         and all(b == a + 1 for a, b in zip(previous["episode_length"], pre["episode_length"], strict=True)),
                         "unchanged mounting view and single observation/episode clock advance")
    tensors = {key: torch.tensor([row["pre"][key] for row in rows], dtype=torch.float64)
               for key in ("body_angular", "actor_angular", "newest_noisy_angular",
                           "imu_quat_wxyz", "effective_lag")}
    base.require(base.finite_tree(tensors), "finite timing tensors")
    rotated = quat_apply(tensors["imu_quat_wxyz"], tensors["body_angular"])
    noise = tensors["newest_noisy_angular"] - rotated
    base.require(float(noise.abs().max()) <= .03001,
                 "fresh angular cache matches known mounting rotation plus bounded noise")
    settled = {key: value[60:] for key, value in tensors.items()}
    if not len(settled["body_angular"]):
        return dict(settled_steps=0, per_world=None, policy_acceptance=False,
                    physical_motion_authorized=False)
    body, actor, fresh = (settled[key] for key in
                         ("body_angular", "actor_angular", "newest_noisy_angular"))
    metric = lambda value: value.abs().mean(0).tolist()
    next_errors = []
    for i in range(60, len(rows)-1):
        # Only matching nonterminal physics/common-step clocks are compared.
        old, nxt = rows[i]["post"], rows[i+1]["pre"]
        base.require(old["physics_time"] == nxt["physics_time"], "paired next-observation physics clocks")
        next_errors.append(torch.tensor(nxt["body_angular"], dtype=torch.float64)[:, 2] -
                           torch.tensor(old["angular_reward"], dtype=torch.float64)[:, 2])
    return dict(settled_steps=len(body),
        per_world=dict(actor_vs_current_body_yaw_mae=metric(actor[..., 2]-body[..., 2]),
            delay_yaw_difference_mae=metric(actor[..., 2]-fresh[..., 2]),
            mounting_yaw_difference_mae=metric(rotated[60:, :, 2]-body[..., 2]),
            fresh_noise_yaw_mae=metric(noise[60:, :, 2]),
            lag_one_fraction=(settled["effective_lag"] == 1).double().mean(0).tolist(),
            reward_phase_to_next_body_yaw_difference_mae=(metric(torch.stack(next_errors)) if next_errors else None)),
        interpretation="descriptive decomposition only; not policy or mechanical causality and no new gate",
        policy_acceptance=False, physical_motion_authorized=False)
