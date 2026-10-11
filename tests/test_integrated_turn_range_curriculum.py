from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_turn_range_curriculum as turn
from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_joint_limit_refinement as repair
from mjlab_microduck import integrated_tracking_refinement as tracking


def test_only_initial_yaw_sampling_differs_and_parent_is_control(monkeypatch):
    cfg, agent = turn.prepare_config("control")
    other, oa = turn.prepare_config("curriculum")
    assert cfg.commands["twist"].ranges.ang_vel_z == (-.5, .5)
    assert other.commands["twist"].ranges.ang_vel_z == (-.05, .05)
    other.commands["twist"].ranges.ang_vel_z = cfg.commands["twist"].ranges.ang_vel_z
    assert repr(other) == repr(cfg)
    oa.run_name = agent.run_name
    assert oa == agent and cfg.seed == agent.seed == 953
    assert cfg.rewards["motor_load"].weight == -1.
    assert cfg.rewards["joint_limit"].weight == -4.
    assert cfg.rewards["command_error"].weight == -.5
    assert cfg.rewards["forward_error"].weight == 0.
    assert cfg.actions["joint_pos"].clip is None and cfg.curriculum == {}
    assert agent.max_iterations == 1000 and agent.save_interval == 250
    assert turn.evaluate_case is forward.evaluate_case and turn.decision is forward.decision
    calls = []
    monkeypatch.setattr(motor, "warm_load", lambda *a, **k: calls.append((a, k)))
    turn.warm_load("runner", "env")
    assert calls == [(("runner", "env", turn.PARENT), dict(parent_sha=turn.PARENT_SHA,
                     parent_iteration=999, parent_steps=24000))]
    assert turn.PARENT.parent.name == "control"
    assert turn.PARENT_SHA == "604700b0d50ee5f11558cf826a1c50fc4283bae801a80f7d6a8f90dff1a14132"


def fake_env(bound=.05):
    from test_integrated_joint_limit_refinement import fake_env as original
    env = original()
    terms = {k: env.reward_manager.get_term_cfg(k) for k in env.reward_manager.active_terms}
    terms["forward_error"] = NS(weight=0., params={"command_name": "twist"})
    env.reward_manager.get_term_cfg = terms.__getitem__
    env.reward_manager.active_terms.append("forward_error")
    env.reward_manager._step_reward = torch.zeros(4, 4)
    term = NS(cfg=NS(ranges=NS(ang_vel_z=(-bound, bound))))
    command = torch.tensor([[.2, 0., bound], [.3, 0., -bound], [0., 0., 0.], [.4, 0., 0.]])
    env.command_manager.get_term = lambda _: term
    env.command_manager.get_command = lambda _: command
    # A distinct cfg proves the live manager is the mutation target.
    env.cfg = NS(commands={"twist": NS(ranges=NS(ang_vel_z=(-bound, bound)))})
    return env, term, command


def test_all_modes_and_explicit_strict_step_boundaries():
    for mode, (worlds, updates, seed) in turn.MODES.items():
        cfg, agent = turn.prepare_config(mode)
        assert cfg.scene.num_envs == worlds and agent.max_iterations == updates and agent.seed == seed
        bound = turn.stages(mode)[0][1]
        assert cfg.commands["twist"].ranges.ang_vel_z == (-bound, bound)
    assert turn.stages("smoke-curriculum") == ((0, .05), (1, .10), (2, .20))
    assert turn.stages("benchmark-curriculum") == ((0, .20),)
    for step, bound in ((0, .05), (6000, .05), (6001, .1), (12000, .1), (12001, .2)):
        assert motor.motor_weight(step, turn.YAW_STAGES) == bound


@pytest.mark.parametrize("mode,bound", [("control", .5), ("curriculum", .05)])
def test_live_sampling_stages_preserve_commands_rewards_and_exact_exposure(mode, bound):
    env, term, command = fake_env(bound)
    observer = turn.MotorObserver(mode)
    sample = NS(force_nm=torch.zeros(4, 14), mean_cost=torch.zeros(4))
    q = env.scene["robot"].data.joint_pos.clone()
    original_command = command.clone()
    for start in (0, 6000, 12000):
        env.common_step_counter = start
        expected = {}
        for _ in range(24):
            env.scene["robot"].data.joint_pos.copy_(q)
            observer.before(env)
            active = term.cfg.ranges.ang_vel_z[1]
            expected[str(active)] = expected.get(str(active), 0)+1
            assert torch.equal(command, original_command)  # no resampling/rewrite
            assert env.cfg.commands["twist"].ranges.ang_vel_z == (-bound, bound)
            assert env.reward_manager.get_term_cfg("command_error").weight == -.5
            assert env.reward_manager.get_term_cfg("forward_error").weight == 0.
            tracking.capture_tracking_metric(env); repair.capture_limit_metric(env)
            forward.capture_forward_metric(env)
            env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*-.5
            env.reward_manager._step_reward[:, 2] = env._microduck_limit_snapshot[0]*-4.
            observer.step(env, sample, {})
            env.common_step_counter += 1
        row = observer.update(env)
        assert row["yaw_range_step_counts"] == expected
        assert sum(row["yaw_range_cumulative_step_counts"].values()) == 24*(1+(0, 6000, 12000).index(start))
        assert row["observed_yaw_command_abs_max"] == pytest.approx(bound)
        if mode == "curriculum" and start == 6000: assert expected == {"0.05": 1, "0.1": 23}
        if mode == "curriculum" and start == 12000: assert expected == {"0.1": 1, "0.2": 23}


@pytest.mark.parametrize("fault", ["wrong-manager-range", "command-outside", "nan", "reverse-clock"])
def test_live_range_violations_fail_closed(fault):
    env, term, command = fake_env()
    observer = turn.MotorObserver("curriculum")
    if fault == "wrong-manager-range": term.cfg.ranges.ang_vel_z = (-.5, .5)
    if fault == "command-outside": command[0, 2] = .051
    if fault == "nan": command[0, 2] = float("nan")
    if fault == "reverse-clock":
        env.common_step_counter = 6001
        observer.before(env)
        env.common_step_counter = 0
    with pytest.raises(ValueError): observer.before(env)
