from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_angular_latency_refinement as latency
from mjlab_microduck import integrated_training_budget_refinement as parent
from mjlab_microduck import observation_timing_capture as capture
from mjlab_microduck.tasks import mdp


def test_recipe_has_only_angular_max_lag_change_and_preserves_parent_identity():
    assert latency.PARENT == parent.PARENT and latency.PARENT_SHA == parent.PARENT_SHA
    assert latency.PARENT_LEARNING_RATE == parent.PARENT_LEARNING_RATE
    assert latency.warm_load is parent.warm_load
    assert latency.evaluate_case is parent.evaluate_case and latency.decision is parent.decision
    assert latency.ARMS == ("control", "fresh")
    assert latency.TRAINING_CAP_SECONDS == {"control": 2100, "fresh": 2100}
    assert latency.MODES == {
        "smoke-control": (64, 5, 1193), "smoke-fresh": (64, 5, 1193),
        "benchmark-control": (256, 10, 1201), "benchmark-fresh": (256, 10, 1201),
        "control": (256, 1000, 1213), "fresh": (256, 1000, 1213),
    }
    for phase, worlds, updates, seed in (("smoke-", 64, 5, 1193),
                                         ("benchmark-", 256, 10, 1201),
                                         ("", 256, 1000, 1213)):
        control, control_agent = latency.prepare_config(phase + "control")
        fresh, fresh_agent = latency.prepare_config(phase + "fresh")
        angular = fresh.observations["actor"].terms["base_ang_vel"]
        assert control.observations["actor"].terms["base_ang_vel"].delay_max_lag == 1
        assert angular.delay_min_lag == 0 and angular.delay_max_lag == 0
        angular.delay_max_lag = 1
        assert repr(control) == repr(fresh)
        assert control.seed == fresh.seed == control_agent.seed == fresh_agent.seed == seed
        assert control.scene.num_envs == fresh.scene.num_envs == worlds
        assert control_agent.max_iterations == fresh_agent.max_iterations == updates
        assert control_agent.experiment_name == fresh_agent.experiment_name == latency.PROTOCOL
        assert control_agent.run_name == phase + "control"
        assert fresh_agent.run_name == phase + "fresh"
        assert control_agent.save_interval == fresh_agent.save_interval == 250
        assert not control.curriculum and not fresh.curriculum


def make_env(mode="control", worlds=4, *, lag=None):
    from mjlab.utils.buffers import DelayBuffer
    from mjlab.utils.noise import UniformNoiseCfg
    from mjlab.envs import mdp as base_mdp

    is_control = mode == "control" or mode.endswith("-control")
    max_angular = 1 if is_control else 0
    generator = torch.Generator(device="cpu").manual_seed(701)
    command = torch.tensor([.3, 0., .2]).repeat(worlds, 1)
    actor = torch.zeros(worlds, 61)
    actor[:, 48:51] = command

    def delay_buffer(min_lag, max_lag, period):
        value = DelayBuffer(min_lag=min_lag, max_lag=max_lag, batch_size=worlds,
                            device="cpu", per_env=True, hold_prob=0.,
                            update_period=period, per_env_phase=True, generator=generator)
        x = torch.arange(worlds * 3, dtype=torch.float32).reshape(worlds, 3) / 100.
        value._buffer.append(x)
        value._buffer.append(x + .2)
        if min_lag == max_lag:
            value.set_lags(torch.full((worlds,), min_lag, dtype=torch.long))
        elif lag is not None:
            value.set_lags(torch.as_tensor(lag, dtype=torch.long))
        else:
            value.set_lags(torch.arange(worlds, dtype=torch.long) % 2)
        return value

    angular_term = NS(
        func=mdp.base_ang_vel_imu_misaligned, params={"max_angle_deg": 6.},
        scale=None, clip=None, history_length=0, delay_min_lag=0,
        delay_max_lag=max_angular, delay_update_period=64, delay_per_env=True,
        delay_per_env_phase=True, delay_hold_prob=0.,
        noise=UniformNoiseCfg(n_min=-.03, n_max=.03, operation="add"),
    )
    gravity_term = NS(delay_min_lag=0, delay_max_lag=1, delay_update_period=64,
                      delay_per_env=True, delay_per_env_phase=True,
                      delay_hold_prob=0., history_length=0,
                      func=mdp.projected_gravity_imu_misaligned,
                      params={"max_angle_deg": 6.}, scale=None, clip=None,
                      noise=UniformNoiseCfg(n_min=-.01, n_max=.01, operation="add"))
    velocity_term = NS(delay_min_lag=1, delay_max_lag=1, delay_update_period=0,
                       delay_per_env=True, delay_per_env_phase=True,
                       delay_hold_prob=0., history_length=0,
                       func=base_mdp.joint_vel_rel,
                       params={"asset_cfg": NS(name="robot", joint_names=(r"^(?!passive_).*",))},
                       scale=None, clip=None,
                       noise=UniformNoiseCfg(n_min=-.25, n_max=.25, operation="add"))
    terms = {"base_ang_vel": angular_term, "projected_gravity": gravity_term,
             "joint_vel": velocity_term}
    delays = {
        "projected_gravity": delay_buffer(0, 1, 64),
        "joint_vel": delay_buffer(1, 1, 0),
    }
    if max_angular:
        angular_delay = delay_buffer(0, 1, 64)
        selected = angular_delay._buffer[torch.minimum(
            angular_delay.current_lags, angular_delay._buffer.current_length - 1)]
        actor[:, :3] = selected
        delays["base_ang_vel"] = angular_delay
    else:
        actor[:, :3] = torch.arange(worlds * 3, dtype=torch.float32).reshape(worlds, 3) / 37.
        angular_delay = None

    manager = NS(
        active_terms={"actor": list(capture.TERMS)},
        group_obs_term_dim={"actor": list(capture.DIMS)},
        group_obs_concatenate={"actor": True},
        cfg={"actor": NS(enable_corruption=True)},
        _obs_buffer={"actor": actor},
        _group_obs_term_delay_buffer={"actor": delays},
        get_term_cfg=lambda group, name: terms[name],
        compute_calls=0,
    )
    manager.compute = lambda *args, **kwargs: setattr(manager, "compute_calls", manager.compute_calls + 1)
    env = NS(num_envs=worlds, common_step_counter=0, observation_manager=manager,
             command_manager=NS(get_command=lambda name: command),
             _latency_command=command, _latency_actor=actor)
    return NS(env=env, actor=actor, command=command, terms=terms, delays=delays,
              angular_delay=angular_delay)


def test_live_control_reads_real_selected_delay_and_fresh_is_absent_buffer_without_side_effects():
    for mode in ("control", "fresh"):
        fixture = make_env(mode, 8, lag=[0, 1, 0, 1, 1, 0, 1, 0])
        state = torch.random.get_rng_state().clone()
        actor = fixture.actor.clone()
        command = fixture.command.clone()
        snapshots = {name: (delay.current_lags.clone(), delay._step_count.clone(),
                            delay._buffer.buffer.clone())
                     for name, delay in fixture.delays.items()}

        counts = latency.inspect_latency(fixture.env, fixture.actor, mode)

        if mode == "control":
            assert counts == {"lag0": 4, "lag1": 4}
            assert fixture.angular_delay is fixture.delays["base_ang_vel"]
        else:
            assert counts == {"lag0": 8, "lag1": 0}
            assert fixture.angular_delay is None
            assert "base_ang_vel" not in fixture.delays
        assert torch.equal(fixture.actor, actor) and torch.equal(fixture.command, command)
        assert torch.equal(torch.random.get_rng_state(), state)
        assert fixture.env.observation_manager.compute_calls == 0
        for name, delay in fixture.delays.items():
            assert torch.equal(delay.current_lags, snapshots[name][0])
            assert torch.equal(delay._step_count, snapshots[name][1])
            assert torch.equal(delay._buffer.buffer, snapshots[name][2])


@pytest.mark.parametrize("fault", [
    "layout", "cache-shape", "cache-missing", "cache-mismatch", "nonfinite",
    "command-mismatch", "angular-function", "angular-noise", "angular-flags",
    "angular-delay-config", "control-buffer-missing", "fresh-buffer-present",
    "control-buffer-flags", "gravity-cfg", "gravity-buffer", "velocity-cfg",
    "velocity-buffer",
])
def test_live_angular_latency_rejects_effective_config_cache_or_buffer_drift(fault):
    mode = "fresh" if fault == "fresh-buffer-present" else "control"
    fixture = make_env(mode, 4)
    env = fixture.env
    manager = env.observation_manager
    if fault == "layout":
        manager.active_terms["actor"].reverse()
    elif fault == "cache-shape":
        manager._obs_buffer["actor"] = torch.zeros(4, 60)
    elif fault == "cache-missing":
        manager._obs_buffer = None
    elif fault == "cache-mismatch":
        manager._obs_buffer["actor"][0, 0] += .1
    elif fault == "nonfinite":
        manager._obs_buffer["actor"][0, 0] = float("nan")
    elif fault == "command-mismatch":
        env._latency_command[0, 2] += .1
    elif fault == "angular-function":
        fixture.terms["base_ang_vel"].func = lambda *_a, **_k: None
    elif fault == "angular-noise":
        fixture.terms["base_ang_vel"].noise.n_max = .04
    elif fault == "angular-flags":
        fixture.terms["base_ang_vel"].delay_per_env = False
    elif fault == "angular-delay-config":
        fixture.terms["base_ang_vel"].delay_update_period = 32
    elif fault == "control-buffer-missing":
        manager._group_obs_term_delay_buffer["actor"].pop("base_ang_vel")
    elif fault == "fresh-buffer-present":
        manager._group_obs_term_delay_buffer["actor"]["base_ang_vel"] = fixture.delays["projected_gravity"]
    elif fault == "control-buffer-flags":
        fixture.angular_delay.per_env_phase = False
    elif fault == "gravity-cfg":
        fixture.terms["projected_gravity"].delay_max_lag = 0
    elif fault == "gravity-buffer":
        fixture.delays["projected_gravity"].update_period = 32
    elif fault == "velocity-cfg":
        fixture.terms["joint_vel"].delay_min_lag = 0
    else:
        fixture.delays["joint_vel"]._current_lags.zero_()
    with pytest.raises(ValueError):
        latency.inspect_latency(env, manager._obs_buffer["actor"] if manager._obs_buffer else None, mode)


def test_observer_counts_pre_action_lags_across_two_complete_updates(monkeypatch):
    monkeypatch.setattr(parent.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(parent.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(parent.MotorObserver, "update", lambda *args: {})
    for mode, expected in (("control", {"lag0": 48, "lag1": 48}),
                           ("fresh", {"lag0": 96, "lag1": 0})):
        fixture = make_env(mode, 4, lag=[0, 1, 0, 1])
        env = fixture.env
        observer = latency.MotorObserver(mode)
        assert observer.mode == "control" and observer.latency_mode == mode
        for update_index in range(2):
            for _ in range(24):
                if mode == "control":
                    fixture.angular_delay.set_lags(torch.tensor([0, 1, 0, 1]))
                    lag = fixture.angular_delay.current_lags
                    length = fixture.angular_delay._buffer.current_length
                    effective = torch.minimum(lag, length - 1).clamp_min(0)
                    env.observation_manager._obs_buffer["actor"][:, :3] = fixture.angular_delay._buffer[effective]
                observer.before(env)
                # Mutation after before must not change the already captured row.
                if mode == "control":
                    fixture.angular_delay.set_lags(torch.zeros(4, dtype=torch.long))
                env.common_step_counter += 1
                observer.step(env, None, {})
            row = observer.update(env)
            assert row["angular_effective_lag_counts"] == expected
            cumulative = {key: value * (update_index + 1) for key, value in expected.items()}
            assert row["angular_effective_lag_cumulative"] == cumulative
            assert row["angular_delay_buffer_present"] is (mode == "control")
            assert observer.pending_latency is None and observer.latency_steps == 0
            assert observer.latency_counts == {"lag0": 0, "lag1": 0}


def test_observer_rejects_missing_duplicate_skipped_and_incomplete_latency_samples(monkeypatch):
    monkeypatch.setattr(parent.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(parent.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(parent.MotorObserver, "update", lambda *args: {})
    fixture = make_env("fresh", 2)
    env = fixture.env
    observer = latency.MotorObserver("fresh")
    with pytest.raises(ValueError):
        observer.step(env, None, {})
    observer.before(env)
    with pytest.raises(ValueError):
        observer.before(env)
    env.common_step_counter += 2
    with pytest.raises(ValueError):
        observer.step(env, None, {})

    # After a consumed step, skipping the next before() must not let a later
    # observation be recorded under the wrong action clock.
    fixture = make_env("fresh", 2)
    env = fixture.env
    observer = latency.MotorObserver("fresh")
    observer.before(env)
    env.common_step_counter += 1
    observer.step(env, None, {})
    env.common_step_counter += 1
    with pytest.raises(ValueError, match="consecutive pre-action"):
        observer.before(env)

    fixture = make_env("fresh", 2)
    env = fixture.env
    observer = latency.MotorObserver("fresh")
    for _ in range(23):
        observer.before(env)
        env.common_step_counter += 1
        observer.step(env, None, {})
    with pytest.raises(ValueError):
        observer.update(env)


@pytest.mark.parametrize("mode", ["undeclared", "smoke-unknown", "benchmark-unknown"])
def test_unknown_mode_fails_closed(mode):
    with pytest.raises(ValueError):
        latency.prepare_config(mode)
    with pytest.raises(ValueError):
        latency.MotorObserver(mode)
