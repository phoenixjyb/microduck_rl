"""Explicit price opt-in preserves historical scalar-weight protocols."""
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import v8_gait_scale_contrast as driver
from mjlab_microduck import v8_motor_headroom as headroom
from mjlab_microduck import v8_steering_budget as budget


def test_historical_weights_and_observers_remain_minus_one():
    before = (driver.WEIGHT, headroom.WEIGHT, budget.WEIGHT)
    for experiment in (None, driver, headroom, budget):
        arms = driver.ARMS if experiment is None else experiment.ARMS
        for arm in arms:
            assert driver.bias_weight(experiment, arm) == -1.
    assert driver.Observer("smoke", "control").bias_weight == -1.
    assert headroom.Observer("smoke", "control").bias_weight == -1.
    assert budget.Observer("smoke", "extended").bias_weight == -1.
    assert before == (driver.WEIGHT, headroom.WEIGHT, budget.WEIGHT)


@pytest.mark.parametrize("value", [0., .5, True, float("nan"), float("inf"), "-0.5"])
def test_nonnegative_or_ambiguous_prices_are_rejected(value):
    experiment = SimpleNamespace(ARMS=("control", "reduced"), WEIGHT=-1.,
                                 WEIGHTS={"control":-1., "reduced":value})
    with pytest.raises(ValueError, match="strictly negative"):
        driver.bias_weight(experiment, "control")  # checks the whole mapping
    with pytest.raises(ValueError, match="strictly negative"):
        driver.Observer("smoke", "control", bias_weight=value)


def test_mapping_is_complete_and_does_not_change_global_weight():
    experiment = SimpleNamespace(ARMS=("control", "reduced"), WEIGHT=-1.,
                                 WEIGHTS={"control":-1., "reduced":-.5})
    assert driver.bias_weight(experiment, "reduced") == -.5
    assert driver.WEIGHT == -1.
    del experiment.WEIGHTS["control"]
    with pytest.raises(ValueError, match="complete explicit"):
        driver.bias_weight(experiment, "reduced")


def test_actual_reduced_reward_snapshot_is_verified_without_changing_legacy_price(monkeypatch):
    from mjlab_microduck.tasks import mdp
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager
    cost = torch.tensor([.1, .2])
    def cached(env, **kwargs):
        env._microduck_body_yaw_bias_state = dict(step=env.common_step_counter,
            signed_residual=torch.tensor([.1, -.2]), signed_ema=torch.tensor([.1, -.2]), cost=cost.clone())
        return cost.clone()
    monkeypatch.setattr(mdp, "body_yaw_bias_cost", cached)
    env = SimpleNamespace(common_step_counter=8, num_envs=2, device="cpu", max_episode_length_s=20.)
    env.reward_manager = RewardManager({"yaw_bias":RewardTermCfg(func=cached, weight=-.5,
        params={"command_name":"twist", "tau_s":.5})}, env, scale_by_dt=False)
    env.reward_manager.compute(.02)
    driver.capture_metric(env,"gait-scale",weight=-.5)
    assert torch.equal(env._microduck_yaw_bias_snapshot["weighted_reward"], -.5*cost)
    env._microduck_yaw_bias_snapshot = None
    with pytest.raises(ValueError, match="live contrast reward"):
        driver.capture_metric(env,"gait-scale")
    assert driver.WEIGHT == -1.
