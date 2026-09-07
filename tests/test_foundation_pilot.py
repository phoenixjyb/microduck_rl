"""F1 source contracts: these tests are not training or simulation acceptance."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck import foundation_pilot as f1
from mjlab_microduck.foundation_evaluation import candidate_failures, SEEDS
from mjlab_microduck.tasks import mdp


@pytest.mark.parametrize("mode", ["smoke", "pilot"])
def test_task_changes_only_commands_completed_curricula_and_instrumentation(mode):
    original = load_env_cfg(f1.TASK)
    cfg, agent = f1.prepare_config(mode)
    assert asdict(cfg.observations["actor"]) == asdict(original.observations["actor"])
    assert asdict(cfg.actions["joint_pos"]) == asdict(original.actions["joint_pos"])
    assert asdict(cfg.sim) == asdict(original.sim)
    assert cfg.events == original.events
    assert cfg.scene.entities == original.scene.entities
    assert cfg.terminations == original.terminations
    for name, reward in cfg.rewards.items():
        expected = deepcopy(original.rewards[name])
        if name == "motor_torque_load": expected.weight = -2.
        if name == "action_rate_l2": expected.weight = -1.
        assert reward == expected
    for term in cfg.curriculum.values():
        for stages in term.params.values():
            if isinstance(stages, list): assert len(stages) == 1 and stages[0]["step"] == 0
    assert cfg.commands["twist"].ranges.lin_vel_x == (.3, .3)
    assert cfg.commands["twist"].rel_standing_envs == 0
    assert cfg.commands["twist"].heading_command is False
    assert all(c.zero_command_prob == 1 for n,c in cfg.commands.items() if n != "twist")
    assert agent.algorithm == load_rl_cfg(f1.TASK).algorithm
    assert cfg.scene.num_envs == 256 and agent.num_steps_per_env == 24
    assert not agent.upload_model


def test_curriculum_does_not_reopen_after_parent_resume():
    cfg, _ = f1.prepare_config("pilot")
    for step in (0, f1.PARENT_STEP, f1.PARENT_STEP+500*24):
        env = NS(common_step_counter=step, reward_manager=NS(_term_names=["motor_torque_load"],
                 _term_cfgs=[deepcopy(cfg.rewards["motor_torque_load"])]))
        # Test the real curriculum function with the actual manager interface.
        env.reward_manager.get_term_cfg = lambda name: env.reward_manager._term_cfgs[0]
        env.reward_manager.set_term_cfg = lambda name, value: env.reward_manager._term_cfgs.__setitem__(0,value)
        mdp.reward_weight(env, None, **cfg.curriculum["motor_torque_load_weight"].params)
        assert env.reward_manager._term_cfgs[0].weight == -2.


def test_resume_keeps_adam_and_common_step_but_advances_update_label():
    calls = []
    runner = NS(device="cpu", current_learning_iteration=7998,
        env=NS(unwrapped=NS(common_step_counter=192000)),
        alg=NS(learning_rate=.001, optimizer=NS(param_groups=[{"lr": .00011390625}])) )
    runner.load = lambda *a, **k: calls.append((a,k))
    f1.restore_parent(runner, "/retained/model_7998.pt")
    assert calls[0][1] == dict(strict=True, map_location="cpu")
    assert runner.current_learning_iteration == 7999
    assert runner.env.unwrapped.common_step_counter == 192000
    assert runner.alg.learning_rate == .00011390625


@pytest.mark.parametrize("key,value", [("fall_fraction", .51), ("pre_reset_torque_p99", .86),
    ("rated_speed_exceed_fraction", .011), ("fall_fraction", float("nan"))])
def test_training_gross_guards(key,value):
    row = dict(fall_fraction=0., pre_reset_torque_p99=.6, rated_speed_exceed_fraction=0.)
    row[key] = value
    assert f1.training_failures(row)


def test_finite_gradient_hook_is_identity_and_rejects_nan():
    g = torch.ones(3)
    assert f1.finite_gradient(g) is g
    with pytest.raises(ValueError): f1.finite_gradient(torch.tensor([float("nan")]))


def report():
    return dict(protocol=f1.PROTOCOL, seed=SEEDS[0], num_envs=8, safety_failures=[],
        classification="straight-response-within-both-criteria", groups={"settled":dict(
        heading_abs_max=.1, cross_route_abs_per_env_mean=[.02]*8, legacy_torque_p99=.5,
        cross_route_abs_mean=.02, body_forward_per_env_mean=[.3]*8, route_forward_per_env_mean=[.3]*8)})


@pytest.mark.parametrize("key,value", [("heading_abs_max", .3),
    ("cross_route_abs_per_env_mean", [.06]+[.02]*7), ("legacy_torque_p99", .53),
    ("body_forward_per_env_mean", [.21]+[.3]*7)])
def test_gate_cannot_hide_bad_env_or_motor_in_pool(key,value):
    parent, candidate = report(), report()
    assert candidate_failures(candidate,parent) == []
    candidate["groups"]["settled"][key] = value
    assert candidate_failures(candidate,parent)
