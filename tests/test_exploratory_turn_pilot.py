"""CPU source checks only: these do not assert learning or simulator admission."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import exploratory_turn_pilot as pilot


@pytest.mark.parametrize("mode", ["smoke", "pilot"])
def test_minimal_config_preserves_plant_policy_rewards_optimizer(mode):
    original, old_agent = pilot.base.prepare_config(mode)
    cfg, agent = pilot.prepare_config(mode)
    for key in ("observations", "actions", "rewards", "curriculum", "events", "terminations"):
        assert getattr(cfg, key) == getattr(original, key)
    assert cfg.scene == original.scene
    expected_sim = deepcopy(original.sim)
    expected_sim.nan_guard.enabled = True
    assert cfg.sim == expected_sim
    assert agent.algorithm == old_agent.algorithm
    assert agent.actor == old_agent.actor and agent.critic == old_agent.critic
    assert cfg.scene.num_envs == 256 and agent.num_steps_per_env == 24
    assert (agent.seed, agent.max_iterations) == pilot.MODES[mode][:2]
    assert agent.save_interval == 10 and not agent.upload_model
    assert cfg.commands["twist"].resampling_time_range == (3., 5.)
    assert cfg.commands["twist"].ranges.lin_vel_x == (.15, .30)
    assert cfg.commands["twist"].ranges.ang_vel_z == (-.20, .20)
    for name in ("head_pose", "body_pose"):
        assert cfg.commands[name] == original.commands[name]


def commands():
    c = torch.zeros(256, 3)
    c[:, 0] = .20
    c[:128, 2] = -.10
    c[128:, 2] = .10
    return c


@pytest.mark.parametrize("column,value", [(0, .14), (0, .31), (1, .01), (2, .21), (0, float("nan"))])
def test_command_bounds_reject_bad_delivery(column, value):
    c = commands(); c[0, column] = value
    with pytest.raises(ValueError): pilot.validate_commands(c)


def test_observer_observes_both_signs_and_resampling_without_mutation():
    c = commands()
    env = NS(command_manager=NS(get_command=lambda name: c))
    checker = pilot.CommandObserver(host_check=lambda: None)
    for step in range(2):
        c[:, 0] = .20 + step*.01
        obs = {"actor": torch.zeros(256, 61), "critic": torch.zeros(256, 64)}
        obs["actor"][:, 48:51] = c
        before = deepcopy(obs)
        checker.before_actor(env, obs)
        checker.after_actor(env, obs, torch.zeros(256, 14))
        checker.before_step(env, c)
        checker.after_step(env, torch.zeros(256, dtype=torch.bool))
        assert all(torch.equal(obs[k], before[k]) for k in obs)
    result = checker.finish(2)
    assert result["changed_world_commands"] == 256 and result["actor_command_equal"]
    assert not result["commands_modified"]


def test_stale_actor_input_rejected():
    c = commands(); env = NS(command_manager=NS(get_command=lambda name: c))
    with pytest.raises(ValueError, match="fresh raw actor command"):
        pilot.CommandObserver().before_actor(env, {"actor": torch.zeros(256, 61)})


def matrix(error=.1):
    return [dict(seed=719, speed_mps=speed, yaw_rad_s=yaw, worlds=8, steps=240,
        complete=True, terminal_worlds=[], torque_p99=.5, rated_speed_exceed_fraction=0.,
        settled_speed_mae_per_world=[error]*8, settled_yaw_mae_per_world=[.02]*8)
        for speed,yaw in pilot.CASES]


def test_improvement_is_never_acceptance():
    result = pilot.compare(matrix(), matrix(.09))
    assert result["decision"] == "exploratory-improvement"
    assert not result["policy_acceptance"] and not result["physical_motion_authorized"]


@pytest.mark.parametrize("kind", ["fall", "motor", "regression", "speed"])
def test_bad_world_or_safety_is_not_hidden_by_average(kind):
    parent, candidate = matrix(), matrix(.09)
    if kind == "fall": candidate[0].update(complete=False, steps=100, terminal_worlds=[0])
    if kind == "motor": candidate[0]["torque_p99"] = .61
    if kind == "regression": candidate[0]["settled_speed_mae_per_world"][0] = .2
    if kind == "speed": candidate[0]["rated_speed_exceed_fraction"] = .001
    assert pilot.compare(parent,candidate)["decision"] == "no-demonstrated-improvement"


def test_missing_and_nonfinite_matrix_rejected():
    with pytest.raises(ValueError): pilot.compare([], [])
    c = matrix(); c[0]["settled_speed_mae_per_world"][0] = float("nan")
    with pytest.raises(ValueError): pilot.compare(matrix(), c)


def test_shared_gpu_guard_is_nonmutating_and_retains_dino(monkeypatch):
    calls = []
    def fake(*args):
        calls.append(args)
        if "--query-gpu=uuid,memory.used,memory.free,temperature.gpu" in args:
            return pilot.GPU + ", 1000, 15000, 44"
        if "--query-compute-apps=pid,process_name,used_memory" in args:
            return "1592, /home/converge/Tonghao/VLM/grounding_dino_cpp_dev/build_worker/grounding_dino_cpp_worker, 946"
        return "inactive"
    monkeypatch.setattr(pilot, "read", fake)
    report = pilot.shared_host()
    assert report["processes"][0]["pid"] == 1592
    assert len(report["protected_services"]) == 4
    assert all("stop" not in call and "start" not in call for call in calls)
