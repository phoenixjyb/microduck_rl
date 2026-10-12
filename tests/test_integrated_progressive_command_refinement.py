from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.utils.buffers import DelayBuffer

from mjlab_microduck import integrated_progressive_command_refinement as command
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_yaw_bias_refinement as bias


def test_modes_stage_boundaries_and_parent_aliases():
    assert command.PROTOCOL == "integrated-progressive-command-continuation-v1"
    assert command.ARMS == ("control", "progressive")
    assert command.MODES == {
        "smoke-control": (64, 5, 1231), "smoke-progressive": (64, 5, 1231),
        "benchmark-control": (256, 10, 1237), "benchmark-progressive": (256, 10, 1237),
        "control": (256, 1000, 1249), "progressive": (256, 1000, 1249),
    }
    assert command.TRAINING_CAP_SECONDS == {"control": 2100, "progressive": 2100}
    assert command.PARENT == previous.PARENT and command.PARENT_SHA == previous.PARENT_SHA
    assert command.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE
    assert command.warm_load is previous.warm_load
    assert command.evaluate_case is previous.evaluate_case
    assert command.decision is previous.decision
    assert command.stage_at_step(0, command.FULL_STAGES) == command.FULL_STAGES[0]
    for update, stage in ((249, 0), (250, 1), (499, 1), (500, 2),
                          (749, 2), (750, 3)):
        assert command.stage_at_step(update * 24, command.FULL_STAGES) == command.FULL_STAGES[stage]
    assert command.stage_at_step(5999, command.FULL_STAGES) == command.FULL_STAGES[0]
    assert command.stage_at_step(6000, command.FULL_STAGES) == command.FULL_STAGES[1]
    assert command.stage_at_step(11999, command.FULL_STAGES) == command.FULL_STAGES[1]
    assert command.stage_at_step(12000, command.FULL_STAGES) == command.FULL_STAGES[2]
    assert command.stage_at_step(17999, command.FULL_STAGES) == command.FULL_STAGES[2]
    assert command.stage_at_step(18000, command.FULL_STAGES) == command.FULL_STAGES[3]
    assert tuple(command.stage_at_step(update * 24, command.SMOKE_STAGES)
                 for update in range(4)) == command.SMOKE_STAGES


@pytest.mark.parametrize("prefix", ["smoke-", "benchmark-", ""])
def test_only_two_command_settings_and_run_name_differ(prefix):
    control, control_agent = command.prepare_config(prefix + "control")
    progressive, progressive_agent = command.prepare_config(prefix + "progressive")
    assert control.commands["twist"].ranges.ang_vel_z == (-.5, .5)
    assert control.commands["twist"].rel_forward_envs == .25
    stage = command.stages(prefix + "progressive")[0]
    assert progressive.commands["twist"].ranges.ang_vel_z == (-stage[1], stage[1])
    assert progressive.commands["twist"].rel_forward_envs == stage[2]
    progressive.commands["twist"].ranges.ang_vel_z = control.commands["twist"].ranges.ang_vel_z
    progressive.commands["twist"].rel_forward_envs = control.commands["twist"].rel_forward_envs
    assert repr(control) == repr(progressive)
    assert control_agent.max_iterations == progressive_agent.max_iterations
    assert control_agent.seed == progressive_agent.seed
    assert control_agent.experiment_name == progressive_agent.experiment_name == command.PROTOCOL
    progressive_agent.run_name = control_agent.run_name
    assert control_agent == progressive_agent
    assert control.rewards[previous.previous.TERM].weight == 0.
    assert control.rewards["motor_load"].weight == -1.
    assert control.actions["joint_pos"].clip is None
    assert not control.curriculum


def fake_env(worlds=2, yaw=.04):
    patterns = ((0., 0., 0.), (.3, 0., yaw), (.3, 0., -yaw), (.3, 0., yaw))
    command_tensor = torch.tensor([patterns[i % len(patterns)] for i in range(worlds)],
                                  dtype=torch.float32)
    actor = torch.zeros(worlds, 61)
    actor[:, 48:51] = command_tensor
    twist_cfg = NS(ranges=NS(lin_vel_x=(.2, .4), lin_vel_y=(0., 0.),
                             ang_vel_z=(-.05, .05)),
                   resampling_time_range=(3., 5.), rel_standing_envs=.15,
                   rel_forward_envs=.65, rel_turn_in_place_envs=0.)
    term = NS(cfg=twist_cfg, _resample_command=lambda *args: None)
    delay = DelayBuffer(min_lag=1, max_lag=2, batch_size=worlds, device="cpu",
                        update_period=8, per_env_phase=True)
    delay.append(torch.ones(worlds, 2))
    delay.append(torch.full((worlds, 2), 2.))
    manager = NS(_obs_buffer={"actor": actor, "critic": torch.ones(worlds, 4)},
                 _group_obs_term_delay_buffer={"actor": {"joint_vel": delay}},
                 compute=lambda *args: None)
    env = NS(num_envs=worlds, common_step_counter=0, observation_manager=manager,
             cfg=NS(commands={"twist": NS(ranges=NS(ang_vel_z=(-.05, .05)),
                                         rel_forward_envs=.65)}),
             command_manager=NS(get_term=lambda name: term,
                                get_command=lambda name: command_tensor))
    return env, term, command_tensor


def patch_inherited_reward_callbacks(monkeypatch):
    monkeypatch.setattr(bias.MotorObserver, "before", lambda self, env: None)
    monkeypatch.setattr(bias.MotorObserver, "step", lambda self, env, sample, extras: None)
    monkeypatch.setattr(bias.MotorObserver, "update", lambda self, env: {})


def test_live_stage_update_keeps_old_command_cache_buffers_rng_and_dwell(monkeypatch):
    patch_inherited_reward_callbacks(monkeypatch)
    env, term, commands = fake_env(worlds=4)
    observer = command.MotorObserver("smoke-progressive")
    initial_command = commands.clone()
    cache = {key: value.clone() for key, value in env.observation_manager._obs_buffer.items()}
    delay = env.observation_manager._group_obs_term_delay_buffer["actor"]["joint_vel"]
    delay_identity = id(delay)
    delay_state = (delay._current_lags.clone(), delay._step_count.clone(),
                   delay._phase_offsets.clone(), delay._buffer._pointer,
                   delay._buffer._num_pushes.clone(), delay._buffer._buffer.clone())
    rng = torch.random.get_rng_state().clone()
    def unexpected(*args, **kwargs):
        raise AssertionError("observer must not compute observations or resample commands")
    monkeypatch.setattr(delay, "compute", unexpected)
    monkeypatch.setattr(delay, "append", unexpected)
    monkeypatch.setattr(term, "_resample_command", unexpected)
    monkeypatch.setattr(env.observation_manager, "compute", unexpected)
    rows = []
    for update in range(5):
        for _ in range(24):
            observer.before(env)
            assert torch.equal(commands, initial_command)  # widening never rewrites cached commands
            env.common_step_counter += 1
            observer.step(env, None, {})
        rows.append(observer.update(env))
        assert env.cfg.commands["twist"].ranges.ang_vel_z == (-.05, .05)
        assert env.cfg.commands["twist"].rel_forward_envs == .65
        assert observer.pending_clock is None
        assert observer.update_action_steps == 0
    assert [row["command_curriculum_stage"] for row in rows] == [
        list(command.SMOKE_STAGES[i]) for i in (0, 1, 2, 3, 3)]
    assert rows[0]["command_curriculum_stage_counts"] == {"0": 4 * 24, "1": 0, "2": 0, "3": 0}
    assert rows[1]["command_curriculum_stage_counts"] == {"0": 0, "1": 4 * 24, "2": 0, "3": 0}
    assert rows[3]["command_curriculum_stage_counts"] == {"0": 0, "1": 0, "2": 0, "3": 4 * 24}
    assert rows[-1]["command_curriculum_cumulative_action_steps"] == 120
    assert rows[-1]["command_curriculum_cumulative_counts"] == {"0": 96, "1": 96, "2": 96, "3": 192}
    assert rows[-1]["command_exposure_counts"]["idle"] == 24
    assert rows[-1]["command_exposure_counts"]["positive-low"] == 48
    assert rows[-1]["command_exposure_counts"]["negative-low"] == 24
    assert rows[-1]["command_exposure_cumulative"]["positive-low"] == 240
    assert rows[-1]["command_curriculum_configured_forward_fraction"] == .25
    assert all(row["command_curriculum_observed_max_abs_yaw"] == pytest.approx(.04)
               for row in rows)
    assert torch.equal(env.observation_manager._obs_buffer["actor"], cache["actor"])
    assert torch.equal(env.observation_manager._obs_buffer["critic"], cache["critic"])
    live_delay = env.observation_manager._group_obs_term_delay_buffer["actor"]["joint_vel"]
    assert id(live_delay) == delay_identity
    assert torch.equal(live_delay._current_lags, delay_state[0])
    assert torch.equal(live_delay._step_count, delay_state[1])
    assert torch.equal(live_delay._phase_offsets, delay_state[2])
    assert live_delay._buffer._pointer == delay_state[3]
    assert torch.equal(live_delay._buffer._num_pushes, delay_state[4])
    assert torch.equal(live_delay._buffer._buffer, delay_state[5])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(commands, initial_command)


@pytest.mark.parametrize("mutation", ["yaw", "forward", "cache", "clock"])
def test_rejects_live_cfg_cache_or_clock_drift(monkeypatch, mutation):
    patch_inherited_reward_callbacks(monkeypatch)
    env, term, commands = fake_env()
    observer = command.MotorObserver("smoke-progressive")
    if mutation == "yaw":
        term.cfg.ranges.ang_vel_z = (-.1, .1)
    elif mutation == "forward":
        term.cfg.rel_forward_envs = .5
    elif mutation == "cache":
        env.observation_manager._obs_buffer["actor"][0, 48] = .2
    elif mutation == "clock":
        env.common_step_counter = 1
    with pytest.raises(ValueError):
        observer.before(env)


def test_rejects_wrong_mode_invalid_stage_command_support_and_skipped_clock(monkeypatch):
    patch_inherited_reward_callbacks(monkeypatch)
    with pytest.raises(ValueError):
        command.MotorObserver("unknown")
    with pytest.raises(ValueError):
        command.prepare_config("unknown")
    with pytest.raises(ValueError):
        command.stage_at_step(-1, command.FULL_STAGES)
    with pytest.raises(ValueError):
        command.stage_at_step(0, ((0, .5, .25), (0, .6, .2)))
    with pytest.raises(ValueError):
        command.stage_at_step(0, ((250, .1, .5),))

    env, _, _ = fake_env(yaw=.08)
    observer = command.MotorObserver("smoke-progressive")
    with pytest.raises(ValueError):
        observer.before(env)

    env, _, _ = fake_env()
    observer = command.MotorObserver("smoke-progressive")
    observer.before(env)
    env.common_step_counter += 2
    with pytest.raises(ValueError):
        observer.step(env, None, {})


def test_forward_config_and_yaw_ceiling_change_only_at_common_step_boundaries(monkeypatch):
    patch_inherited_reward_callbacks(monkeypatch)
    env, term, _ = fake_env()
    observer = command.MotorObserver("smoke-progressive")
    for step in range(25):
        observer.before(env)
        expected = command.SMOKE_STAGES[0 if step < 24 else 1]
        assert term.cfg.ranges.ang_vel_z == (-expected[1], expected[1])
        assert term.cfg.rel_forward_envs == expected[2]
        env.common_step_counter += 1
        observer.step(env, None, {})


@pytest.mark.parametrize("forward_fraction", [.65, .25])
def test_pinned_velocity_sampler_resamples_forward_bucket_and_signed_yaw(forward_fraction):
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    worlds = 20000
    sampler = UniformVelocityCommand.__new__(UniformVelocityCommand)
    sampler.cfg = NS(heading_command=False, init_velocity_prob=0.,
                     rel_heading_envs=0., rel_standing_envs=.15,
                     rel_world_envs=0., rel_forward_envs=forward_fraction,
                     ranges=NS(lin_vel_x=(.2, .4), lin_vel_y=(0., 0.),
                               ang_vel_z=(-.5, .5), heading=None))
    sampler._env = NS(device="cpu", num_envs=worlds)
    sampler.vel_command_b = torch.zeros(worlds, 3)
    sampler.vel_command_w = torch.zeros(worlds, 3)
    sampler.heading_target = torch.zeros(worlds)
    sampler.is_heading_env = torch.zeros(worlds, dtype=torch.bool)
    sampler.is_standing_env = torch.zeros(worlds, dtype=torch.bool)
    sampler.is_world_env = torch.zeros(worlds, dtype=torch.bool)
    sampler.is_forward_env = torch.zeros(worlds, dtype=torch.bool)
    sampler._resample_command(torch.arange(worlds))
    forward_mask = sampler.is_forward_env.clone()
    sampler._update_command()
    commands = sampler.vel_command_b
    active_forward = forward_mask & ~sampler.is_standing_env
    assert abs(float(forward_mask.float().mean()) - forward_fraction) < .02
    assert bool((commands[active_forward, 0] >= .3).all())
    assert bool((commands[active_forward, 1:] == 0.).all())
    assert bool((commands[:, 1] == 0.).all())
    assert bool((commands[:, 2] < 0.).any()) and bool((commands[:, 2] > 0.).any())
    idle = sampler.is_standing_env
    assert bool((commands[idle] == 0.).all())
