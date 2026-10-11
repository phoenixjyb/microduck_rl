from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_low_yaw_rehearsal_refinement as rehearsal
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor


def fake_env(mode="rehearsal", command=None):
    worlds = 4
    if command is None:
        command = torch.tensor([[.3, 0., .2]] * worlds)
    support = rehearsal.yaw_support(mode)
    term = NS(cfg=NS(
        ranges=NS(lin_vel_x=(.2, .4), lin_vel_y=(0., 0.), ang_vel_z=support),
        resampling_time_range=(3., 5.), rel_standing_envs=.15,
        rel_forward_envs=.25, rel_turn_in_place_envs=0.,
    ))
    actor = torch.zeros(worlds, 61)
    actor[:, 48:51].copy_(command)
    observations = NS(_obs_buffer={"actor": actor}, compute_calls=0)

    def compute(*_args, **_kwargs):
        observations.compute_calls += 1
        raise AssertionError("low-yaw exposure must not recompute observations")

    observations.compute = compute
    env = NS(num_envs=worlds, common_step_counter=0, device="cpu",
             command_manager=NS(get_term=lambda name: term if name == "twist" else None,
                                get_command=lambda name: command if name == "twist" else None),
             observation_manager=observations)
    env._low_yaw_command = command
    env._low_yaw_actor = actor
    env._low_yaw_term = term
    env._microduck_optimizer_audit = NS(alg=NS(entropy_coef=.01))
    return env


def sync_actor(env):
    env._low_yaw_actor[:, 48:51].copy_(env._low_yaw_command)


def test_matched_recipe_changes_only_training_yaw_support_and_preserves_parent_contract():
    assert rehearsal.PROTOCOL == "integrated-low-yaw-rehearsal-continuation-v1"
    assert rehearsal.ARMS == ("control", "rehearsal")
    assert rehearsal.MODES == {
        "smoke-control": (64, 5, 1163), "smoke-rehearsal": (64, 5, 1163),
        "benchmark-control": (256, 10, 1171), "benchmark-rehearsal": (256, 10, 1171),
        "control": (256, 500, 1181), "rehearsal": (256, 500, 1181),
    }
    assert rehearsal.CAP_SECONDS == 900
    assert rehearsal.TRAINING_CAP_SECONDS == {"control": 900, "rehearsal": 900}
    assert rehearsal.PARENT == previous.PARENT and rehearsal.PARENT_SHA == previous.PARENT_SHA
    assert rehearsal.PARENT_LEARNING_RATE == previous.PARENT_LEARNING_RATE
    assert rehearsal.warm_load is previous.warm_load
    assert rehearsal.evaluate_case is previous.evaluate_case
    assert rehearsal.decision is previous.decision
    assert rehearsal.EXPOSURE_BINS == previous.EXPOSURE_BINS

    for control_mode, rehearsal_mode in (
        ("smoke-control", "smoke-rehearsal"),
        ("benchmark-control", "benchmark-rehearsal"),
        ("control", "rehearsal"),
    ):
        control, control_agent = rehearsal.prepare_config(control_mode)
        narrow, narrow_agent = rehearsal.prepare_config(rehearsal_mode)
        assert control.commands["twist"].ranges.ang_vel_z == (-.5, .5)
        assert narrow.commands["twist"].ranges.ang_vel_z == (-.25, .25)
        assert control.commands["twist"].ranges.lin_vel_x == narrow.commands["twist"].ranges.lin_vel_x == (.2, .4)
        assert control.commands["twist"].ranges.lin_vel_y == narrow.commands["twist"].ranges.lin_vel_y == (0., 0.)
        assert control.commands["twist"].resampling_time_range == narrow.commands["twist"].resampling_time_range == (3., 5.)
        assert control.commands["twist"].rel_standing_envs == narrow.commands["twist"].rel_standing_envs == .15
        assert control.commands["twist"].rel_forward_envs == narrow.commands["twist"].rel_forward_envs == .25
        assert control.commands["twist"].rel_turn_in_place_envs == narrow.commands["twist"].rel_turn_in_place_envs == 0.
        assert control.seed == narrow.seed == control_agent.seed == narrow_agent.seed
        assert control.scene.num_envs == narrow.scene.num_envs
        assert control.rewards == narrow.rewards
        assert control.curriculum == narrow.curriculum == {}
        assert control.actions["joint_pos"] == narrow.actions["joint_pos"]
        assert control.actions["joint_pos"].clip is None and control.actions["joint_pos"].scale == 1.
        assert control_agent.algorithm.entropy_coef == narrow_agent.algorithm.entropy_coef == .01
        assert control_agent.algorithm.desired_kl == narrow_agent.algorithm.desired_kl == .01
        assert control.commands["head_pose"].zero_command_prob == narrow.commands["head_pose"].zero_command_prob == 0.
        assert control.commands["head_pose"].ranges == narrow.commands["head_pose"].ranges

        # Normalize the single yaw-support difference, then compare the entire
        # configuration and runner against the matched control arm.
        narrow.commands["twist"].ranges.ang_vel_z = (-.5, .5)
        narrow_agent.run_name = control_agent.run_name
        assert repr(control) == repr(narrow)
        assert control_agent == narrow_agent

        inherited, inherited_agent = previous.prepare_config("control")
        narrow.seed = inherited.seed
        narrow.scene.num_envs = narrow.scene.terrain.num_envs = inherited.scene.num_envs
        assert repr(narrow) == repr(inherited)
        for key in ("seed", "max_iterations", "experiment_name", "run_name"):
            setattr(narrow_agent, key, getattr(inherited_agent, key))
        assert narrow_agent == inherited_agent

    for mode, (worlds, updates, seed) in rehearsal.MODES.items():
        cfg, agent = rehearsal.prepare_config(mode)
        assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
        assert cfg.seed == agent.seed == seed and agent.max_iterations == updates
        assert agent.save_interval == 250
        assert cfg.commands["twist"].ranges.ang_vel_z == rehearsal.yaw_support(mode)
        assert cfg.commands["head_pose"].zero_command_prob == 0.
        assert cfg.actions["joint_pos"].clip is None
        assert agent.algorithm.entropy_coef == .01
    with pytest.raises(ValueError):
        rehearsal.yaw_support("undeclared")

    plan = motor.evaluation_plan(rehearsal)
    assert {arm: contract["final_iteration"] for arm, contract in plan["contracts"].items()} == {
        "control": 499, "rehearsal": 499,
    }
    assert plan["final_iterations"] == {"control": 499, "rehearsal": 499}
    assert rehearsal.prepare_config("control")[1].save_interval == 250
    assert (0, 250, 499) == (0, *range(250, 500, 250), 499)


def test_previous_command_exposure_keeps_exact_low_and_target_boundaries():
    command = torch.tensor([
        [.3, 0., .15], [.3, 0., .25], [.3, 0., .2501],
        [.3, 0., -.15], [.3, 0., -.25], [.3, 0., -.2501],
    ])
    control = previous.command_exposure(command)
    assert control["positive-low"] == control["negative-low"] == 1
    assert control["positive-target"] == control["negative-target"] == 1
    assert control["positive-high"] == control["negative-high"] == 1
    boundary = command.clone()
    boundary[:, 2] = torch.tensor([.15, .25, 0., -.15, -.25, 0.])
    bins = previous.command_exposure(boundary)
    assert bins["positive-low"] == bins["negative-low"] == 1
    assert bins["positive-target"] == bins["negative-target"] == 1
    assert bins["positive-high"] == bins["negative-high"] == 0


def test_before_reads_cached_actor_and_live_support_without_rng_or_observation_recompute(monkeypatch):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env = fake_env("rehearsal", torch.tensor([[.3, 0., .25]] * 4))
    observer = rehearsal.MotorObserver("rehearsal")
    command_before = env._low_yaw_command.clone()
    actor_before = env._low_yaw_actor.clone()
    rng_before = torch.random.get_rng_state().clone()
    observer.before(env)
    assert torch.equal(env._low_yaw_command, command_before)
    assert torch.equal(env._low_yaw_actor, actor_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert env.observation_manager.compute_calls == 0


@pytest.mark.parametrize("fault", [
    "shape", "missing-cache", "non-tensor", "misaligned", "nonfinite",
    "outside-low", "outside-high", "cfg-yaw", "cfg-linear", "cfg-resampling",
    "cfg-standing", "cfg-forward", "cfg-turn", "entropy",
])
def test_rehearsal_before_fails_closed_on_live_command_cache_and_sampler_drift(monkeypatch, fault):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env = fake_env("rehearsal")
    term = env._low_yaw_term
    if fault == "shape":
        env.observation_manager._obs_buffer["actor"] = torch.zeros(4, 60)
    elif fault == "missing-cache":
        env.observation_manager._obs_buffer.pop("actor")
    elif fault == "non-tensor":
        env.observation_manager._obs_buffer["actor"] = []
    elif fault == "misaligned":
        env._low_yaw_actor[0, 48] += .1
    elif fault == "nonfinite":
        env._low_yaw_command[0, 2] = float("nan")
        sync_actor(env)
    elif fault == "outside-low":
        env._low_yaw_command[:, 2] = -.2501
        sync_actor(env)
    elif fault == "outside-high":
        env._low_yaw_command[:, 2] = .2501
        sync_actor(env)
    elif fault == "cfg-yaw":
        term.cfg.ranges.ang_vel_z = (-.5, .5)
    elif fault == "cfg-linear":
        term.cfg.ranges.lin_vel_x = (.1, .4)
    elif fault == "cfg-resampling":
        term.cfg.resampling_time_range = (2., 5.)
    elif fault == "cfg-standing":
        term.cfg.rel_standing_envs = .2
    elif fault == "cfg-forward":
        term.cfg.rel_forward_envs = .2
    elif fault == "cfg-turn":
        term.cfg.rel_turn_in_place_envs = .1
    else:
        env._microduck_optimizer_audit.alg.entropy_coef = 0.
    with pytest.raises(ValueError):
        rehearsal.MotorObserver("rehearsal").before(env)
    assert env.observation_manager.compute_calls == 0


@pytest.mark.parametrize("mode,yaw", [("control", .5), ("control", -.5),
                                      ("rehearsal", .25), ("rehearsal", -.25)])
def test_control_keeps_wide_support_and_rehearsal_accepts_exact_narrow_edge(monkeypatch, mode, yaw):
    monkeypatch.setattr(previous.MotorObserver, "before", lambda *args: None)
    env = fake_env(mode, torch.tensor([[.3, 0., yaw]] * 4))
    rehearsal.MotorObserver(mode).before(env)


def test_full_24_step_observer_preserves_pre_action_counts_and_rejects_high_rehearsal_bins(monkeypatch):
    inherited_observer = previous.MotorObserver.__mro__[1]
    monkeypatch.setattr(inherited_observer, "before", lambda *args: None)
    monkeypatch.setattr(inherited_observer, "step", lambda *args: None)
    monkeypatch.setattr(inherited_observer, "update", lambda *args: {})
    env = fake_env("rehearsal")
    observer = rehearsal.MotorObserver("rehearsal")
    for step in range(24):
        env._low_yaw_command[:, 0] = .3
        env._low_yaw_command[:, 2] = torch.tensor([.15, .25, -.15, -.25])
        sync_actor(env)
        command_before = env._low_yaw_command.clone()
        actor_before = env._low_yaw_actor.clone()
        rng_before = torch.random.get_rng_state().clone()
        observer.before(env)
        assert torch.equal(env._low_yaw_command, command_before)
        assert torch.equal(env._low_yaw_actor, actor_before)
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        env._low_yaw_command.zero_()  # post-action resampling cannot rewrite pending bins
        env.common_step_counter += 1
        observer.step(env, None, {})
    row = observer.update(env)
    assert row["command_exposure_counts"]["positive-low"] == 24
    assert row["command_exposure_counts"]["positive-target"] == 24
    assert row["command_exposure_counts"]["negative-low"] == 24
    assert row["command_exposure_counts"]["negative-target"] == 24
    assert row["command_exposure_counts"]["positive-high"] == 0
    assert row["command_exposure_counts"]["negative-high"] == 0
    assert row["training_yaw_support"] == [-.25, .25]
    assert observer.exposure_steps == 0 and not any(observer.exposure_counts.values())


@pytest.mark.parametrize("mode", ["control", "rehearsal"])
def test_control_observer_allows_high_exposure_and_reports_wide_support(monkeypatch, mode):
    inherited_observer = previous.MotorObserver.__mro__[1]
    monkeypatch.setattr(inherited_observer, "before", lambda *args: None)
    monkeypatch.setattr(inherited_observer, "step", lambda *args: None)
    monkeypatch.setattr(inherited_observer, "update", lambda *args: {})
    env = fake_env("control", torch.tensor([[.3, 0., .5]] * 4))
    if mode == "rehearsal":
        with pytest.raises(ValueError):
            rehearsal.MotorObserver(mode).before(env)
        return
    observer = rehearsal.MotorObserver("control")
    for _ in range(24):
        sync_actor(env)
        observer.before(env)
        env.common_step_counter += 1
        observer.step(env, None, {})
    row = observer.update(env)
    assert row["command_exposure_counts"]["positive-high"] == 4 * 24
    assert row["command_exposure_counts"]["negative-high"] == 0
    assert row["training_yaw_support"] == [-.5, .5]
