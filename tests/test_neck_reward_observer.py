import copy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import foundation_pilot as training
from mjlab_microduck.neck_reward_observer import NeckRewardObserver, JOINTS, TERM, mdp


def environment(weight=-.2):
    terms={TERM:NS(weight=weight,func=mdp.neck_action_rate_l2),
           "motor_torque_load":NS(weight=-4.),"lateral_velocity_cost":NS(weight=-.5)}
    manager=NS(get_term_cfg=terms.__getitem__,active_terms=[TERM],_step_reward=torch.zeros(256,1),
               _episode_sums={TERM:torch.zeros(256)})
    return NS(scene={"robot":NS(joint_names=JOINTS)},reward_manager=manager,
              action_manager=NS(action=torch.zeros(256,14)),common_step_counter=204000,
              num_envs=256,device="cpu")


def consume(env, observer):
    before=env.action_manager.action.clone()
    observer.before_reward(env)
    value=mdp.neck_action_rate_l2(env)
    env.reward_manager._step_reward[:,0]=value*observer.weight*.02/.02
    observer.after_reward(env)
    assert torch.equal(before,env.action_manager.action)


@pytest.mark.parametrize("weight",[-.1,-.2])
def test_original_cost_and_cache_semantics_are_unchanged_and_logged(tmp_path,weight):
    env=environment(weight); observer=NeckRewardObserver(weight,tmp_path/"activity.jsonl")
    original=environment(weight)
    for i in range(24):
        env.action_manager.action.fill_(i*.01);original.action_manager.action.fill_(i*.01)
        env.common_step_counter+=1
        consume(env,observer)
        expected=mdp.neck_action_rate_l2(original)*weight
        torch.testing.assert_close(env.reward_manager._step_reward[:,0],expected)
        assert torch.equal(env._prev_neck_actions,original._prev_neck_actions)
    result=observer.finish(24)
    assert result["positive_samples"]==23*256 and result["neck_weight"]==weight
    row=json.loads((tmp_path/"activity.jsonl").read_text())
    assert row["common_step"]==204024 and row["neck_weight"]==weight and row["raw_neck_cost_mean"]>0
    assert not result["policy_acceptance"]


@pytest.mark.parametrize("bad",["mapping","shape","nan","weight","motor","function","cache"])
def test_refuses_silent_miswiring_before_original_reward(bad):
    env=environment(); observer=NeckRewardObserver(-.2)
    if bad=="mapping":env.scene["robot"].joint_names=JOINTS[::-1]
    if bad=="shape":env.action_manager.action=torch.zeros(256,15)
    if bad=="nan":env.action_manager.action[0,5]=float("nan")
    if bad=="weight":env.reward_manager.get_term_cfg(TERM).weight=-.1
    if bad=="motor":env.reward_manager.get_term_cfg("motor_torque_load").weight=-2.
    if bad=="function":env.reward_manager.get_term_cfg(TERM).func=lambda _:0
    if bad=="cache":env._prev_neck_actions=torch.full((256,4),float("nan"))
    with pytest.raises(ValueError):observer.before_reward(env)


def test_consumed_reward_and_inactive_term_cannot_be_faked():
    env=environment();o=NeckRewardObserver(-.2)
    for _ in range(24):consume(env,o)
    with pytest.raises(ValueError,match="not silently disabled"):o.finish(24)
    env.action_manager.action.fill_(1);o.before_reward(env)
    with pytest.raises(ValueError,match="consumed weighted"):o.after_reward(env)


def test_optional_training_hook_order_and_default_result_identity(monkeypatch):
    env=environment();expected=torch.ones(256);calls=[]
    monkeypatch.setattr(mdp,"_orig_reward_compute",lambda manager,dt:calls.append("compute") or expected)
    assert training.checked_reward_compute(env,.02) is expected and calls==["compute"]
    observer=NS(before_reward=lambda env:calls.append("before"),after_reward=lambda env:calls.append("after"))
    calls.clear()
    assert training.checked_reward_compute(env,.02,observer) is expected and calls==["before","compute","after"]
    expected[0]=float("nan")
    with pytest.raises(ValueError):training.checked_reward_compute(env,.02)


def test_real_rigid_model_joint_mapping():
    from mjlab.entity import Entity
    from mjlab_microduck.foundation_motor_experiment import prepare_config
    cfg,_=prepare_config("smoke",arm="motor")
    robot=Entity(cfg.scene.entities["robot"])
    assert robot.joint_names==JOINTS
