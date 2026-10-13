import math

import pytest
import torch

from mjlab_microduck import integrated_motor_range_yaw_only as original
from mjlab_microduck import integrated_motor_range_yaw_only_replication as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks import mdp
from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity


def _same_recipe_config(mode):
    cfg, agent = recipe.prepare_config(mode)
    reference, reference_agent = original.prepare_config(mode)

    # The replication changes only the experiment protocol and the declared
    # seed. Normalize those intentional identity fields before comparing the
    # complete config and learner settings.
    reference.seed = cfg.seed
    reference_agent.seed = agent.seed
    reference_agent.experiment_name = agent.experiment_name
    assert repr(cfg) == repr(reference)
    assert agent == reference_agent
    assert cfg.seed == agent.seed == recipe.MODES[mode][2]
    assert agent.experiment_name == recipe.PROTOCOL
    assert agent.run_name == mode
    assert agent.max_iterations == recipe.MODES[mode][1]
    assert agent.save_interval == 250
    return cfg, agent


def test_replication_modes_use_new_seeds_but_same_bounded_protocol():
    assert recipe.PROTOCOL == "integrated-motor-range-yaw-only-replication-v1"
    assert recipe.ARMS == original.ARMS == ("control", "yawonly")
    assert recipe.MODES == {
        "smoke-control": (64, 5, 1471),
        "smoke-yawonly": (64, 5, 1471),
        "benchmark-control": (256, 10, 1477),
        "benchmark-yawonly": (256, 10, 1477),
        "control": (256, 1000, 1481),
        "yawonly": (256, 1000, 1481),
    }
    assert recipe.CAP_SECONDS == original.CAP_SECONDS == 2100
    assert recipe.REFERENCE_SOURCE == "d71c30285ff3acbdfee12615548eaf2ea1a10975"
    assert recipe.REFERENCE_TRAINING_SEED == 1447 and recipe.TRAINING_SEED == 1481

    for mode in recipe.MODES:
        _same_recipe_config(mode)

    for invalid in ("", "smoke", "not-a-mode", "benchmark-control-extra"):
        with pytest.raises(ValueError):
            recipe.yaw_only(invalid)
        with pytest.raises(ValueError):
            recipe.prepare_config(invalid)


def test_replication_keeps_exact_original_parent_counters_and_runtime_helpers():
    assert recipe.base is original.base
    assert recipe.previous is original.previous
    assert recipe.PARENT == original.PARENT
    assert recipe.PARENT.parent.name == "control"
    assert recipe.PARENT.name == "model_999.pt"
    assert recipe.PARENT_SHA == original.PARENT_SHA == (
        "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
    )
    assert recipe.PARENT_NORMALIZER_COUNT == original.PARENT_NORMALIZER_COUNT == 43_008_000
    assert recipe.PARENT_ADAM_STEP == original.PARENT_ADAM_STEP == 140_000
    assert recipe.warm_load is original.warm_load
    assert recipe.evaluate_case is original.evaluate_case
    assert recipe.decision is original.decision
    assert recipe.MotorObserver is original.MotorObserver
    assert recipe.TERM is original.TERM == "track_angular_velocity"
    assert recipe.VARIANCE == original.VARIANCE == .5
    assert recipe.WEIGHT == original.WEIGHT == 2.


def test_matched_arms_differ_only_by_yaw_tracking_function_and_keep_contracts():
    for control_mode, treatment_mode in (
        ("smoke-control", "smoke-yawonly"),
        ("benchmark-control", "benchmark-yawonly"),
        ("control", "yawonly"),
    ):
        control, control_agent = _same_recipe_config(control_mode)
        treatment, treatment_agent = _same_recipe_config(treatment_mode)
        cterm = control.rewards[recipe.TERM]
        tterm = treatment.rewards[recipe.TERM]
        assert cterm.func is track_angular_velocity
        assert tterm.func is mdp.yaw_only_body_angular_tracking
        assert cterm.weight == tterm.weight == recipe.WEIGHT
        assert cterm.params == tterm.params == dict(std=math.sqrt(.5), command_name="twist")

        # Configs and learner parameters match after accounting solely for the
        # reward function and the per-arm mode label.
        tterm.func = cterm.func
        treatment_agent.run_name = control_agent.run_name
        assert repr(control) == repr(treatment)
        assert control_agent == treatment_agent

        parent_cfg, parent_agent = original.prepare_config(control_mode)
        assert control.observations == treatment.observations == parent_cfg.observations
        assert control.actions == treatment.actions == parent_cfg.actions
        assert control.commands == treatment.commands == parent_cfg.commands
        assert control.events == treatment.events == parent_cfg.events
        assert control.curriculum == treatment.curriculum == parent_cfg.curriculum == {}
        assert control_agent.actor == treatment_agent.actor == parent_agent.actor
        assert control_agent.clip_actions is treatment_agent.clip_actions is None


def test_shared_parent_safety_observation_action_and_final_evaluation_gates():
    cfg, agent = _same_recipe_config("control")
    treatment, treatment_agent = _same_recipe_config("yawonly")
    parent, parent_agent = original.prepare_config("control")

    # The paired experiment remains the same 61D actor and raw 14-action task.
    actor_terms = cfg.observations["actor"].terms
    assert "head_command" in actor_terms and "body_command" in actor_terms
    assert tuple(actor_terms["joint_pos"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert tuple(actor_terms["joint_vel"].params["asset_cfg"].joint_names) == (r"^(?!passive_).*",)
    assert sum((14, 14, 3, 3, 14, 3, 4, 6)) == 61
    assert cfg.actions == treatment.actions == parent.actions
    action = cfg.actions["joint_pos"]
    assert action.scale == 1. and action.clip is None
    assert tuple(action.actuator_names) == (".*",)
    assert agent.clip_actions is treatment_agent.clip_actions is parent_agent.clip_actions is None
    assert agent.actor.obs_normalization is True

    for candidate in (cfg, treatment):
        assert candidate.rewards["track_angular_velocity"].weight == 2.
        assert candidate.rewards["body_ang_vel"].weight == -.05
        assert candidate.rewards["angular_momentum"].weight == -.02
        assert candidate.rewards["motor_load"].weight == -1.
        assert candidate.rewards["motor_load"].params == dict(
            rated_stall_torque_nm=.60, soft_limit_fraction=.70, over_limit_gain=4.,
        )
        assert candidate.rewards["joint_limit"].weight == -4.
        assert candidate.rewards["joint_limit"].params["margin"] == .15
        assert candidate.rewards["joint_limit"].params["asset_cfg"].joint_names == (
            "^(left|right)_hip_yaw$",
        )
        assert candidate.rewards["forward_error"].weight == 0.

    assert recipe.warm_load is original.warm_load
    assert recipe.base.SEEDS == original.base.SEEDS
    assert recipe.base.CASES == original.base.CASES
    assert len(recipe.base.SEEDS) * len(recipe.base.CASES) == 12
    plan = motor.evaluation_plan(recipe)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert {arm: item["final_iteration"] for arm, item in plan["contracts"].items()} == {
        "control": 999, "yawonly": 999,
    }


@pytest.mark.parametrize("mode", ["smoke-control", "smoke-yawonly"])
def test_reused_observer_keeps_positive_yaw_only_expectation_and_telemetry(mode):
    assert recipe.MotorObserver is original.MotorObserver
    observer = recipe.MotorObserver(mode)
    yaw = torch.tensor([0., .25, 1.])
    xy = torch.tensor([0., 2., 5.])
    actual = observer.expected_reward(yaw, xy)
    if mode.endswith("yawonly"):
        expected_reward = recipe.WEIGHT * torch.exp(-yaw / recipe.VARIANCE)
    else:
        expected_reward = recipe.WEIGHT * torch.exp(-(yaw + xy) / recipe.VARIANCE)
    assert torch.allclose(actual, expected_reward)
    assert bool((actual > 0.).all() & (actual <= recipe.WEIGHT).all())
    assert observer.update.__func__ is original.MotorObserver.update
    assert observer.yaw_only_tracking is mode.endswith("yawonly")

