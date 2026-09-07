import copy
import json
from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.managers.observation_manager import ObservationManager

from mjlab_microduck import foundation_pilot as training
from mjlab_microduck.foundation_yaw_experiment import prepare_config
from mjlab_microduck.command_delivery import TERMS, DIMENSIONS
from mjlab_microduck.yaw_command_observer import YawCommandObserver, validate_command_evidence


def sensor(env,name):return env.sensors[name]


def environment(arm):
    cfg,_=prepare_config("smoke",arm=arm)
    data=NS(root_link_lin_vel_b=torch.zeros(256,3),root_link_ang_vel_b=torch.zeros(256,3))
    env=NS(num_envs=256,device="cpu",step_dt=.02,scene={"robot":NS(data=data)},common_step_counter=204000)
    term=cfg.commands["twist"].build(env)
    commands={"twist":term.command,"head_pose":torch.zeros(256,4),"body_pose":torch.zeros(256,6)}
    env.command_manager=NS(get_term=lambda name:term,get_command=commands.__getitem__)
    env.reward_manager=NS(get_term_cfg=lambda name:cfg.rewards[name])
    env.sensors={name:torch.zeros(256,shape[0]) for name,shape in zip(TERMS,DIMENSIONS)}
    group=copy.deepcopy(cfg.observations["actor"])
    for name in TERMS[:5]:group.terms[name].func=sensor;group.terms[name].params={"name":name}
    env.observation_manager=ObservationManager({"actor":group},env)
    term.reset(torch.arange(256));term.compute(.02)
    return env,term,cfg


def step(env,term,observer,reset=False):
    obs=env.observation_manager.compute(update_history=True)
    algorithm=NS(transition=NS());actions=torch.zeros(256,14)
    def act(received):algorithm.transition.observations=received;return actions
    rng=torch.get_rng_state().clone();before=obs["actor"].clone()
    assert training.checked_training_act(algorithm,env,obs,act,observer) is actions
    assert torch.equal(before,obs["actor"]) and torch.equal(rng,torch.get_rng_state())
    training.check_training_command(env,observer)
    dones=torch.zeros(256,dtype=torch.bool)
    if reset:dones[0]=True;term.reset(torch.tensor([0]))
    term.compute(.02);env.common_step_counter+=1
    observer.after_step(env,dones)


@pytest.mark.parametrize("arm",["control","yaw"])
def test_original_sampler_actual_actor_input_and_episode_reset(tmp_path,arm):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(491);env,term,cfg=environment(arm)
        bounds=cfg.commands['twist'].ranges.ang_vel_z
        assert not {'standing_envs','velocity_command_ranges','head_pose_range','body_pose_range'} & set(cfg.curriculum)
        if arm=='yaw':assert bool((term.command[:,2]>0).any() and (term.command[:,2]<0).any())
        else:assert bool((term.command[:,2]==0).all())
        observer=YawCommandObserver(bounds,tmp_path/'command-activity.jsonl')
        for i in range(24):step(env,term,observer,reset=i in (5,14))
        result=observer.finish(24)
        (tmp_path/'result.json').write_text(json.dumps(dict(command_validator=result)))
        assert validate_command_evidence(tmp_path,1,bounds)==result
        assert not result['commands_modified'] and not result['policy_acceptance']
        row=json.loads((tmp_path/'command-activity.jsonl').read_text())
        assert row['reset_samples']==2 and row['actor_command_equal'] and row['common_step']==204024
        row['positive']+=1
        (tmp_path/'command-activity.jsonl').write_text(json.dumps(row))
        with pytest.raises(ValueError):validate_command_evidence(tmp_path,1,bounds)


@pytest.mark.parametrize('bad',['range','mode','head','weight','forward','stale','normalization'])
def test_miswired_live_input_fails_before_actor(bad):
    env,term,cfg=environment('yaw');o=YawCommandObserver((-.35,.35))
    obs=env.observation_manager.compute(update_history=True)
    if bad=='range':term.cfg.ranges.ang_vel_z=(-1.,1.)
    if bad=='mode':term.cfg.rel_forward_envs=1.
    if bad=='head':env.command_manager.get_command('head_pose')[0,0]=1.
    if bad=='weight':cfg.rewards['motor_torque_load'].weight=-2.
    if bad=='forward':term.command[0,0]=.2
    if bad=='stale':term.command[0,2]*=-1
    if bad=='normalization':obs['actor'][:,48:51]*=2
    with pytest.raises(ValueError):o.before_actor(env,obs)


def test_nonreset_command_changes_and_missing_action_are_rejected():
    env,term,_=environment('yaw');o=YawCommandObserver((-.35,.35))
    with pytest.raises(ValueError):o.before_step(env,term.command)
    obs=env.observation_manager.compute(update_history=True);o.before_actor(env,obs)
    o.after_actor(env,obs,torch.zeros(256,14));o.before_step(env,term.command)
    term.command[0,2]*=-1;env.common_step_counter+=1
    with pytest.raises(ValueError,match='only on episode reset'):o.after_step(env,torch.zeros(256,dtype=torch.bool))


def test_default_fixed_command_guard_unchanged_and_transition_must_match():
    env,term,_=environment('control');training.check_training_command(env)
    term.command[0,2]=.1
    with pytest.raises(ValueError,match='fixed training'):training.check_training_command(env)
    env,term,_=environment('yaw');o=YawCommandObserver((-.35,.35))
    obs=env.observation_manager.compute(update_history=True);o.before_actor(env,obs)
    wrong={'actor':obs['actor'].clone()};wrong['actor'][0,50]*=-1
    with pytest.raises(ValueError,match='transition'):o.after_actor(env,wrong,torch.zeros(256,14))


def test_one_sided_or_unexecuted_support_is_not_accepted():
    env,term,_=environment('yaw');term.command[:,2].abs_();o=YawCommandObserver((-.35,.35))
    for _ in range(24):step(env,term,o)
    with pytest.raises(ValueError,match='both yaw signs'):o.finish(24)
    with pytest.raises(ValueError):YawCommandObserver((0.,0.)).finish(24)
