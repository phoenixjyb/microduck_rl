import copy

import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck import integrated_turn_foundation as foundation
from mjlab.tasks.registry import load_env_cfg


def test_scoped_recipe_preserves_upstream_curricula_and_rewards():
    original = load_env_cfg(foundation.TASK)
    cfg, agent = foundation.prepare_config("foundation")
    command = cfg.commands["twist"]
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 256
    assert cfg.seed == agent.seed == 827
    assert agent.max_iterations == 1500 and agent.save_interval == 250
    assert agent.logger == "tensorboard" and agent.upload_model is False
    assert set(cfg.rewards) == set(original.rewards)
    assert set(cfg.curriculum) == set(original.curriculum)
    assert {k: v.weight for k, v in cfg.rewards.items()} == {k: v.weight for k, v in original.rewards.items()}
    assert command.ranges.lin_vel_x == (.20, .40)
    assert command.ranges.lin_vel_y == (0., 0.)
    assert command.ranges.ang_vel_z == (-.50, .50)
    assert command.rel_forward_envs == .25 and command.rel_turn_in_place_envs == 0.
    assert command.rel_standing_envs == original.commands["twist"].rel_standing_envs == .02
    assert not command.heading_command and command.rel_world_envs == 0.
    assert "microduck_motor_step_stream" in cfg.metrics
    assert "velocity_command_ranges" not in cfg.curriculum
    assert original.commands["twist"].ranges.lin_vel_x == (-.4, .4)


def test_sampler_bounds_accept_idle_and_both_turns_but_not_reverse_or_spin():
    foundation.validate_commands(torch.tensor([[0., 0., 0.], [.3, 0., 0.], [.2, 0., -.5], [.4, 0., .5]]))
    for row in ([-.2, 0., 0.], [0., 0., .2], [.3, .1, 0.], [.3, 0., .6], [.3, 0., float("nan")]):
        with pytest.raises(ValueError):
            foundation.validate_commands(torch.tensor([row]))


def accepted_rows():
    return [dict(seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240,
                 complete=True, terminal_worlds=[], torque_p99=.5,
                 rated_speed_exceed_fraction=0., speed_mae=[.02]*8, yaw_mae=[.08]*8)
            for seed in foundation.SEEDS for speed, yaw in foundation.CASES]


def test_gate_requires_all_worlds_and_does_not_promote_physical_or_policy_acceptance():
    result = foundation.decision(accepted_rows())
    assert result["decision"] == "foundation-ready-for-next-experiment"
    assert not result["policy_acceptance"] and not result["simulator_qualified"]
    assert not result["physical_motion_authorized"]


@pytest.mark.parametrize("key,value,gate", [
    ("complete", False, "first-episode-terminal"),
    ("torque_p99", .60001, "motor-envelope"),
    ("rated_speed_exceed_fraction", .000001, "motor-envelope"),
    ("speed_mae", [.02]*7+[.031], "speed-or-yaw-tracking"),
    ("yaw_mae", [.02]*7+[.101], "speed-or-yaw-tracking"),
])
def test_single_world_or_case_failure_rejects_foundation(key, value, gate):
    rows = accepted_rows()
    rows[0][key] = value
    result = foundation.decision(rows)
    assert result["decision"] == "foundation-not-ready"
    assert result["failures"][0][-1] == gate


def test_gate_rejects_partial_reordered_nan_or_reset_laundered_reports():
    rows = accepted_rows()
    variants = [rows[:-1], list(reversed(rows))]
    for change in ({"torque_p99": float("nan")}, {"steps": 239}, {"terminal_worlds": [0]},
                   {"yaw_mae": [0.]*7}, {"speed_mae": [-.1]*8}):
        changed = copy.deepcopy(rows)
        changed[0].update(change)
        variants.append(changed)
    for variant in variants:
        with pytest.raises(ValueError):
            foundation.decision(variant)


def test_finite_checks_cover_tensordict_and_nested_adam():
    assert foundation.finite_tree(TensorDict({"actor": torch.ones(1, 61)}, batch_size=[1]))
    assert not foundation.finite_tree({"optimizer": {"exp_avg": torch.tensor(float("inf"))}})


def test_predeclared_budgets_cases_and_checkpoint_selection():
    assert foundation.MODES == {"smoke": (64, 5, 821), "benchmark": (256, 10, 823),
                                "foundation": (256, 1500, 827)}
    assert foundation.EVAL_CHECKPOINTS == (750, 1499)
    assert foundation.FOUNDATION_CAP_SECONDS == 2700
    assert foundation.CASES == ((0., 0.), (.30, 0.), (.20, -.20), (.20, .20))
