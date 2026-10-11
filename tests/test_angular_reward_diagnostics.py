import math
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import angular_reward_diagnostics as audit


def fake_env():
    angular = torch.tensor([[0., 0., .2], [2., 0., .2], [0., 0., -.2], [0., 0., 0.]])
    command = torch.tensor([[0., 0., .2]]*4)
    term = NS(weight=2., params=dict(std=math.sqrt(.5), asset_cfg=NS(name="robot"), command_name="twist"))
    value = 2*torch.exp(-((command[:, 2]-angular[:, 2]).square()+angular[:, :2].square().sum(-1))/.5)
    return NS(num_envs=4, device="cpu", scene={"robot": NS(data=NS(root_link_ang_vel_b=angular))},
        command_manager=NS(get_command=lambda _: command), reward_manager=NS(active_terms=[audit.TERM],
        get_term_cfg=lambda _: term, _step_reward=value[:, None]))


def test_actual_reward_capture_survives_reset_and_exposes_xy_suppression():
    env = fake_env()
    assert audit.capture(env).tolist() == [0.]*4
    saved = env._microduck_angular_reward_snapshot
    env.scene["robot"].data.root_link_ang_vel_b.zero_()
    env.command_manager.get_command("").zero_()
    row = audit.summarize(saved[None].repeat(3, 1, 1), startup=1)
    d = row["per_world"]
    assert row["settled_steps"] == 2 and not row["policy_acceptance"]
    assert "not PPO" in row["gradient_kind"]
    assert d["yaw_mae"] == pytest.approx([0., 0., .4, .2])
    assert d["roll_pitch_multiplicative_factor_mean"] == pytest.approx([1., math.exp(-8), 1., 1.])
    assert d["yaw_only_above_point8_full_below_point1_fraction"] == [0., 1., 0., 0.]
    assert d["yaw_only_weighted_reward_mean"][1] == pytest.approx(2.)
    assert d["actual_weighted_reward_mean"][1] == pytest.approx(2*math.exp(-8))
    assert audit.summarize(saved[None], startup=60)["per_world"] is None
    with pytest.raises(ValueError, match="consumed"): audit.capture(env)


@pytest.mark.parametrize("fault", ["nan-angular", "nan-command", "wrong-reward", "zero-std", "wrong-weight"])
def test_raw_capture_fails_closed(fault):
    env = fake_env()
    if fault == "nan-angular": env.scene["robot"].data.root_link_ang_vel_b[0, 0] = float("nan")
    if fault == "nan-command": env.command_manager.get_command("")[0, 2] = float("nan")
    if fault == "wrong-reward": env.reward_manager._step_reward.zero_()
    if fault == "zero-std": env.reward_manager.get_term_cfg("").params["std"] = 0
    if fault == "wrong-weight": env.reward_manager.get_term_cfg("").weight = -2
    with pytest.raises(ValueError): audit.capture(env)


def test_gradient_and_variation_are_descriptive_not_bias_only():
    env = fake_env(); audit.capture(env)
    s = env._microduck_angular_reward_snapshot[None].repeat(2, 1, 1)
    s[0, 0, 2], s[1, 0, 2] = 0., .4  # equal/opposite yaw errors
    s[:, 0, 4] = 2*math.exp(-.04/.5)
    d = audit.summarize(s, startup=0)["per_world"]
    assert d["yaw_bias_abs"][0] == pytest.approx(0., abs=1e-7)
    assert d["yaw_mae"][0] == pytest.approx(.2)
    assert d["full_abs_yaw_gradient_mean"][0] == pytest.approx(2*.4/.5*math.exp(-.04/.5))
    assert d["full_abs_yaw_gradient_mean"][0] == pytest.approx(d["yaw_only_abs_yaw_gradient_mean"][0])


@pytest.mark.parametrize("fault", ["shape", "nan", "wrong-actual", "wrong-std", "bad-startup"])
def test_summary_checks_provenance_values_and_parameters(fault):
    env = fake_env(); audit.capture(env)
    s = env._microduck_angular_reward_snapshot[None]
    kwargs = dict(startup=0)
    if fault == "shape": s = s[..., :4]
    if fault == "nan": s[0, 0, 0] = float("nan")
    if fault == "wrong-actual": s[0, 0, 4] = 0.
    if fault == "wrong-std": kwargs["std"] = .5
    if fault == "bad-startup": kwargs["startup"] = -1
    with pytest.raises(ValueError): audit.summarize(s, **kwargs)


def test_audit_is_only_the_declared_rejected_finals_and_does_not_replace_reward():
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck import integrated_angular_reward_audit as run
    from mjlab_microduck import integrated_turn_foundation as base
    cfg = load_env_cfg(base.TASK, play=True)
    term = cfg.rewards[audit.TERM]
    assert term.weight == 2. and term.params["std"] == math.sqrt(.5)
    assert term.params["command_name"] == "twist" and "asset_cfg" not in term.params
    env = fake_env()
    env.reward_manager.get_term_cfg("").params.pop("asset_cfg")
    # Exercise the real upstream callable's omitted-asset default too.
    value = term.func(env, **term.params)
    assert torch.allclose(value*term.weight, env.reward_manager._step_reward[:, 0])
    audit.capture(env)
    assert tuple(run.CHECKPOINTS) == ("control", "curriculum") and run.SEED == 839
    assert run.EVALUATION_SHA == "71bcfbe46ad8919123222067e2a26f9737967e187950838811efff8b08d5bf94"
    assert len(run.CHECKPOINTS) == 2 and all(len(v) == 64 for v in run.CHECKPOINTS.values())
