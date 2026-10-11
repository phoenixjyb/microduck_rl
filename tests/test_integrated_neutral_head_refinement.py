from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_neutral_head_refinement as neutral
from mjlab_microduck import integrated_training_budget_refinement as previous


HEAD_RANGES = ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))
HEAD_RESAMPLING = (2., 5.)


def attach_commands(env, mode="neutral"):
    from test_integrated_yaw_bias_refinement import fake_env as inherited_fake_env

    if env is None:
        env = inherited_fake_env("control")
    env.num_envs = 4
    env.device = "cpu"
    env.episode_length_buf = torch.full((4,), 10, dtype=torch.long)
    commands = {
        "twist": torch.tensor([[.3, 0., .2]] * 4),
        "head_pose": torch.tensor([[0., 0., 0., 0.], [.1, 0., 0., 0.],
                                   [0., 0., 0., 0.], [0., 0., -.1, 0.]]),
        "body_pose": torch.tensor([[0., .1, 0., 0., 0., 0.]] * 4),
    }
    inherited_get_command = env.command_manager.get_command

    def get_command(name):
        return commands[name] if name in commands else inherited_get_command(name)

    head_term = NS(
        cfg=NS(zero_command_prob=neutral.zero_probability(mode),
              ranges=HEAD_RANGES, resampling_time_range=HEAD_RESAMPLING),
        command_counter=torch.ones(4, dtype=torch.long))
    env.command_manager.get_command = get_command
    env.command_manager.get_term = lambda name: head_term if name == "head_pose" else None
    actor = torch.zeros(4, 61)
    observations = NS(_obs_buffer={"actor": actor}, compute_calls=0)

    def compute(*args, **kwargs):
        observations.compute_calls += 1
        raise AssertionError("neutral-head telemetry must not recompute observations")

    observations.compute = compute
    env.observation_manager = observations
    env._neutral_head_commands = commands
    env._neutral_head_term = head_term
    env._neutral_head_actor = actor
    sync_actor(env)
    return env


def sync_actor(env):
    actor = env._neutral_head_actor
    actor.zero_()
    actor[:, 48:51].copy_(env.command_manager.get_command("twist"))
    actor[:, 51:55].copy_(env.command_manager.get_command("head_pose"))
    actor[:, 55:61].copy_(env.command_manager.get_command("body_pose"))


def test_recipe_changes_only_head_zero_atom_and_preserves_inherited_training_contract():
    assert neutral.PROTOCOL == "integrated-neutral-head-exposure-continuation-v1"
    assert neutral.ARMS == ("control", "neutral")
    assert neutral.PARENT == previous.PARENT and neutral.PARENT_SHA == previous.PARENT_SHA
    assert neutral.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE
    assert neutral.warm_load is previous.warm_load
    assert neutral.evaluate_case is previous.evaluate_case
    assert neutral.decision is previous.decision
    assert neutral.EXPOSURE_BINS is previous.EXPOSURE_BINS
    assert neutral.HEAD_RANGES == HEAD_RANGES and neutral.HEAD_RESAMPLING == HEAD_RESAMPLING
    assert neutral.ZERO_PROBABILITY == .25
    assert neutral.MODES == {
        "smoke-control": (64, 5, 1103), "smoke-neutral": (64, 5, 1103),
        "benchmark-control": (256, 10, 1109), "benchmark-neutral": (256, 10, 1109),
        "control": (256, 1000, 1117), "neutral": (256, 1000, 1117),
    }

    for control_mode, neutral_mode in (
            ("smoke-control", "smoke-neutral"),
            ("benchmark-control", "benchmark-neutral"),
            ("control", "neutral")):
        control, control_agent = neutral.prepare_config(control_mode)
        treatment, treatment_agent = neutral.prepare_config(neutral_mode)
        c_head = control.commands["head_pose"]
        t_head = treatment.commands["head_pose"]
        assert tuple(c_head.ranges) == tuple(t_head.ranges) == HEAD_RANGES
        assert c_head.resampling_time_range == t_head.resampling_time_range == HEAD_RESAMPLING
        assert c_head.zero_command_prob == 0.
        assert t_head.zero_command_prob == .25
        assert control.seed == treatment.seed == control_agent.seed == treatment_agent.seed
        assert control.scene.num_envs == treatment.scene.num_envs
        assert set(control.rewards) == set(treatment.rewards)
        assert control.curriculum == treatment.curriculum == {}
        assert control.actions["joint_pos"].clip is treatment.actions["joint_pos"].clip is None
        assert control_agent.algorithm.symmetry_cfg is treatment_agent.algorithm.symmetry_cfg is None
        t_head.zero_command_prob = c_head.zero_command_prob
        treatment_agent.run_name = control_agent.run_name
        assert repr(control) == repr(treatment)
        assert control_agent == treatment_agent

        inherited, inherited_agent = previous.prepare_config("control")
        treatment_head = treatment.commands["head_pose"]
        treatment_head.zero_command_prob = 0.
        treatment.seed = inherited.seed
        treatment.scene.num_envs = treatment.scene.terrain.num_envs = inherited.scene.num_envs
        assert repr(treatment) == repr(inherited)
        for key in ("seed", "max_iterations", "experiment_name", "run_name"):
            setattr(treatment_agent, key, getattr(inherited_agent, key))
        assert treatment_agent == inherited_agent

    assert neutral.CAP_SECONDS == previous.CAP_SECONDS == 2100
    assert neutral.TRAINING_CAP_SECONDS == {"control": 2100, "neutral": 2100}
    for mode, (worlds, updates, seed) in neutral.MODES.items():
        cfg, agent = neutral.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed and agent.max_iterations == updates
        assert agent.save_interval == 250
        assert cfg.commands["head_pose"].zero_command_prob == neutral.zero_probability(mode)
        assert agent.algorithm.symmetry_cfg is None


@pytest.mark.parametrize("mutation", ["twist", "head", "body", "shape", "nonfinite"])
def test_before_uses_validated_cached_actor_command_slots_without_recomputing(mutation, monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env = attach_commands(None)
    if mutation == "twist":
        env._neutral_head_actor[0, 48] += .1
    elif mutation == "head":
        env._neutral_head_actor[0, 51] += .1
    elif mutation == "body":
        env._neutral_head_actor[0, 55] += .1
    elif mutation == "shape":
        env.observation_manager._obs_buffer["actor"] = torch.zeros(4, 60)
    else:
        env._neutral_head_actor[0, 55] = float("nan")
    observer = neutral.MotorObserver("neutral")
    with pytest.raises(ValueError):
        observer.before(env)
    assert env.observation_manager.compute_calls == 0


@pytest.mark.parametrize("mutation", [
    "buffer-none", "actor-key", "actor-none", "actor-type", "probability", "ranges",
    "resampling", "counter-shape", "counter-dtype", "counter-zero",
])
def test_before_rejects_missing_cache_and_sampler_contract_drift(mutation, monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env = attach_commands(None)
    if mutation == "buffer-none":
        env.observation_manager._obs_buffer = None
    elif mutation == "actor-key":
        env.observation_manager._obs_buffer.pop("actor")
    elif mutation == "actor-none":
        env.observation_manager._obs_buffer["actor"] = None
    elif mutation == "actor-type":
        env.observation_manager._obs_buffer["actor"] = []
    elif mutation == "probability":
        env._neutral_head_term.cfg.zero_command_prob = 0.
    elif mutation == "ranges":
        env._neutral_head_term.cfg.ranges = ((-.38, .39),) + HEAD_RANGES[1:]
    elif mutation == "resampling":
        env._neutral_head_term.cfg.resampling_time_range = (1., 2.)
    elif mutation == "counter-shape":
        env._neutral_head_term.command_counter = torch.ones(3, dtype=torch.long)
    elif mutation == "counter-dtype":
        env._neutral_head_term.command_counter = torch.ones(4, dtype=torch.int32)
    else:
        env._neutral_head_term.command_counter = torch.tensor([1, 1, 0, 1])

    with pytest.raises(ValueError):
        neutral.MotorObserver("neutral").before(env)
    assert env.observation_manager.compute_calls == 0


def test_full_24_step_cached_dwell_counts_pre_action_state_and_reset_segments(monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "update", lambda *args: {})
    env = attach_commands(None)
    observer = neutral.MotorObserver("neutral")
    initial_rng = torch.random.get_rng_state().clone()

    for step in range(24):
        commands = env._neutral_head_commands
        term = env._neutral_head_term
        # The first action contains both sampled states. Subsequent actions use
        # the zero atom, including new zero-to-zero segments after counter changes
        # and an episode reset with an unchanged counter.
        commands["twist"].copy_(torch.tensor([[.3, 0., .2]] * 4))
        commands["head_pose"].zero_()
        if step == 0:
            commands["head_pose"][1, 0] = .1
            commands["head_pose"][3, 2] = -.1
            term.command_counter.copy_(torch.tensor([1, 1, 1, 1]))
            env.episode_length_buf.fill_(10)
        elif step == 1:
            term.command_counter.copy_(torch.tensor([99, 1, 1, 2]))
            env.episode_length_buf.copy_(torch.tensor([10, 0, 10, 10]))
        elif step == 2:
            term.command_counter.copy_(torch.tensor([99, 1, 1, 2]))
            env.episode_length_buf.copy_(torch.tensor([10, 1, 0, 10]))
        else:
            term.command_counter.copy_(torch.tensor([99, 1, 1, 2]))
            env.episode_length_buf.fill_(10 + step)
        sync_actor(env)
        actor_before = env._neutral_head_actor.clone()
        commands_before = {name: value.clone() for name, value in commands.items()}
        rng_before = torch.random.get_rng_state().clone()
        observer.before(env)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert torch.equal(env._neutral_head_actor, actor_before)
        assert all(torch.equal(commands[name], value) for name, value in commands_before.items())
        assert env.observation_manager.compute_calls == 0
        pending = observer.neutral_head_pending
        assert pending is not None and pending[0] == env.common_step_counter
        if step == 0:
            assert pending[2] == {"zero": 2, "nonzero": 2}
        elif step == 1:
            assert pending[2] == {"zero": 3, "nonzero": 0}
        elif step == 2:
            assert pending[2] == {"zero": 1, "nonzero": 0}
        else:
            assert pending[2] == {"zero": 0, "nonzero": 0}

        # Simulate a post-action resample/reset that changes all live buffers.
        # The already captured step must remain assigned to the old command.
        commands["head_pose"].fill_(.2)
        commands["body_pose"].fill_(-.1)
        commands["twist"][:, 2] = -.2
        term.command_counter.add_(1)
        sync_actor(env)
        env.episode_length_buf.zero_()
        env.common_step_counter += 1
        observer.step(env, None, {})
        assert observer.neutral_head_pending is None

    assert torch.equal(initial_rng, torch.random.get_rng_state())
    row = observer.update(env)
    assert row["neutral_head_zero_probability"] == .25
    assert row["neutral_head_exposure_protocol"] == "cached-actor-head-twist-dwell-v1"
    assert row["neutral_head_counts"]["positive-target:zero"] == 94
    assert row["neutral_head_counts"]["positive-target:nonzero"] == 2
    assert sum(row["neutral_head_counts"].values()) == 24 * env.num_envs
    assert row["neutral_head_segments"] == {"zero": 6, "nonzero": 2}
    assert row["neutral_head_segments_cumulative"] == {"zero": 6, "nonzero": 2}
    assert sum(row["neutral_head_cumulative"].values()) == 24 * env.num_envs
    assert observer.neutral_head_steps == 0
    assert sum(observer.neutral_head_counts.values()) == 0
    assert env.observation_manager.compute_calls == 0

    # A second complete update starts with empty per-update bins, while totals
    # and reset/resampling segment counts remain cumulative across the reset.
    for step in range(24):
        commands = env._neutral_head_commands
        term = env._neutral_head_term
        commands["twist"].copy_(torch.tensor([[.3, 0., .2]] * 4))
        commands["head_pose"].zero_()
        commands["body_pose"].zero_()
        term.command_counter.copy_(torch.tensor([100, 2, 2, 3]))
        env.episode_length_buf.fill_(10 + step)
        sync_actor(env)
        actor_before = env._neutral_head_actor.clone()
        commands_before = {name: value.clone() for name, value in commands.items()}
        rng_before = torch.random.get_rng_state().clone()
        observer.before(env)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert torch.equal(env._neutral_head_actor, actor_before)
        assert all(torch.equal(commands[name], value) for name, value in commands_before.items())
        assert env.observation_manager.compute_calls == 0
        pending = observer.neutral_head_pending
        assert pending is not None and pending[2] == {
            "zero": 4 if step == 0 else 0, "nonzero": 0,
        }
        commands["head_pose"].fill_(.2)
        commands["body_pose"].fill_(.1)
        term.command_counter.add_(1)
        sync_actor(env)
        env.episode_length_buf.zero_()
        env.common_step_counter += 1
        observer.step(env, None, {})

    second = observer.update(env)
    assert second["neutral_head_counts"]["positive-target:zero"] == 96
    assert sum(second["neutral_head_counts"].values()) == 96
    assert second["neutral_head_segments"] == {"zero": 4, "nonzero": 0}
    assert second["neutral_head_cumulative"]["positive-target:zero"] == 190
    assert second["neutral_head_cumulative"]["positive-target:nonzero"] == 2
    assert sum(second["neutral_head_cumulative"].values()) == 192
    assert second["neutral_head_segments_cumulative"] == {"zero": 10, "nonzero": 2}
    assert observer.neutral_head_steps == 0
    assert sum(observer.neutral_head_counts.values()) == 0
    assert env.observation_manager.compute_calls == 0


def test_full_inherited_reward_and_optimizer_update_path_preserves_dwell(monkeypatch):
    from test_integrated_decoupled_yaw_precision_refinement import (
        capture_other_pre_reset_terms, fake_optimizer,
    )
    from test_integrated_yaw_bias_refinement import fake_env as inherited_fake_env
    from mjlab_microduck import integrated_yaw_bias_refinement as bias
    from mjlab_microduck import integrated_yaw_precision_refinement as angular

    env = attach_commands(inherited_fake_env("control"), "neutral")
    runner, _, _ = fake_optimizer(monkeypatch, env)
    observer = neutral.MotorObserver("neutral")
    precision_term = env.reward_manager.get_term_cfg(angular.TERM)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))

    for step in range(24):
        env._neutral_head_commands["twist"].copy_(torch.tensor([[.3, 0., .2]] * 4))
        env._neutral_head_commands["head_pose"].zero_()
        if step % 2:
            env._neutral_head_commands["head_pose"][1, 0] = .1
            env._neutral_head_commands["head_pose"][3, 2] = -.1
        env._neutral_head_commands["body_pose"].zero_()
        env._neutral_head_term.command_counter.fill_(step + 1)
        env.episode_length_buf.fill_(10 + step)
        sync_actor(env)
        observer.before(env)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        capture_other_pre_reset_terms(env, precision_term)
        angular.capture_angular_metric(env)
        bias.capture_bias_metric(env)
        # The observer must use only its cached actor snapshot; simulate reset
        # and command resampling before after-step telemetry consumes that copy.
        env._neutral_head_commands["head_pose"].zero_()
        env._neutral_head_commands["body_pose"].fill_(.1)
        env._neutral_head_term.command_counter.add_(1)
        sync_actor(env)
        env.scene["robot"].data.root_link_ang_vel_b.zero_()
        env.episode_length_buf.zero_()
        observer.step(env, sample, {})

    for _ in range(20):
        runner.alg.actor.get_kl_divergence(None)
    row = observer.update(env)
    assert row["optimizer_minibatches"] == 20
    assert row["yaw_variance_min"] == row["yaw_variance_max"] == .5
    assert row["yaw_bias_weight"] == row["weighted_yaw_bias_reward_mean"] == 0.
    assert row["command_exposure_counts"]["positive-target"] == 96
    assert sum(row["command_exposure_counts"].values()) == 96
    assert row["command_exposure_cumulative"] == row["command_exposure_counts"]
    assert sum(row["neutral_head_counts"].values()) == 96
    assert sum(row["neutral_head_cumulative"].values()) == 96
    assert env.observation_manager.compute_calls == 0


def test_missing_duplicate_skipped_exposure_and_invalid_modes_fail_closed(monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "step", lambda *args: None)
    monkeypatch.setattr(previous.MotorObserver, "update", lambda *args: {})
    env = attach_commands(None, "control")
    observer = neutral.MotorObserver("control")
    with pytest.raises(ValueError, match="one complete action step"):
        observer.step(env, None, {})
    observer.before(env)
    with pytest.raises(ValueError, match="previous neutral-head exposure"):
        observer.before(env)
    env.common_step_counter += 2
    with pytest.raises(ValueError, match="one complete action step"):
        observer.step(env, None, {})

    duplicate = neutral.MotorObserver("control")
    duplicate.before(env)
    env.common_step_counter += 1
    duplicate.step(env, None, {})
    with pytest.raises(ValueError, match="one complete action step"):
        duplicate.step(env, None, {})

    incomplete = neutral.MotorObserver("control")
    for _ in range(23):
        sync_actor(env)
        incomplete.before(env)
        env.common_step_counter += 1
        incomplete.step(env, None, {})
    with pytest.raises(ValueError, match="complete actor-visible neutral-head dwell"):
        incomplete.update(env)

    for call in (neutral.zero_probability, neutral.prepare_config, neutral.MotorObserver):
        with pytest.raises(ValueError):
            call("undeclared")
