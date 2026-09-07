"""Read-only evidence at the PPO actor boundary and episode-reset boundary."""

import json
import os

import torch

from mjlab_microduck.command_delivery import fresh_actor_twist
from mjlab_microduck.first_attempt_smoke import require, canonical
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly


class YawCommandObserver:
    def __init__(self, yaw_range, path=None):
        require(yaw_range in ((0.,0.),(-.35,.35)),"predeclared yaw range")
        self.yaw_range,self.path=yaw_range,path
        self.steps,self.positive,self.negative,self.zero=0,0,0,0
        self.pending=None;self.acted=False;self.stepping=False;self.window=[]

    def validate_command(self, env):
        term=env.command_manager.get_term("twist");cfg=term.cfg
        require(type(term) is VelocityCommandCommandOnly and not cfg.heading_command
                and cfg.ranges.heading is None and cfg.resampling_time_range==(1e6,1e6)
                and cfg.ranges.lin_vel_x==(.3,.3) and cfg.ranges.lin_vel_y==(0.,0.)
                and cfg.ranges.ang_vel_z==self.yaw_range,"unchanged live command sampler/ranges")
        for key in ("rel_standing_envs","rel_heading_envs","rel_world_envs","rel_forward_envs",
                    "rel_turn_in_place_envs","init_velocity_prob"):
            require(getattr(cfg,key)==0.,"disabled command mode "+key)
        c=env.command_manager.get_command("twist")
        require(c.shape==(256,3) and bool(torch.isfinite(c).all())
                and torch.equal(c[:,:2],c.new_tensor([.3,0.]).expand(256,2))
                and bool(((c[:,2]>=self.yaw_range[0])&(c[:,2]<=self.yaw_range[1])).all()),
                "actual fixed-forward bounded-yaw command")
        for name in ("head_pose","body_pose"):
            head=env.command_manager.get_command(name)
            require(bool(torch.isfinite(head).all()) and bool((head==0).all()),"unchanged neutral "+name)
        for name,weight in (("neck_action_rate_l2",-.1),("motor_torque_load",-4.),("lateral_velocity_cost",-.5)):
            require(env.reward_manager.get_term_cfg(name).weight==weight,"unchanged live reward "+name)
        return c

    def before_actor(self, env, observations):
        require(self.pending is None and not self.acted and not self.stepping,"one actor per control step")
        c=self.validate_command(env);actor=observations["actor"]
        verified=fresh_actor_twist(actor,c,env.observation_manager)
        require(torch.equal(actor,verified),"sampled command reached raw actor input without refresh")
        self.pending=c.detach().clone();self.before_counter=int(env.common_step_counter)

    def after_actor(self, env, stored_observations, actions):
        require(self.pending is not None and not self.acted and actions.shape==(256,14)
                and bool(torch.isfinite(actions).all())
                and torch.equal(stored_observations["actor"][:,48:51],self.pending),
                "PPO action sampled and original transition command retained")
        self.acted=True

    def before_step(self, env, command):
        require(self.acted and not self.stepping and torch.equal(command,self.pending),
                "same consumed command at physics entry")
        self.stepping=True

    def after_step(self, env, dones):
        require(self.stepping and dones.shape==(256,) and dones.dtype==torch.bool
                and int(env.common_step_counter)==self.before_counter+1,"ordered completed control step")
        current=self.validate_command(env)
        require(torch.equal(current[~dones],self.pending[~dones]),"commands change only on episode reset")
        yaw=self.pending[:,2];positive=int((yaw>0).sum());negative=int((yaw<0).sum());zero=int((yaw==0).sum())
        self.positive+=positive;self.negative+=negative;self.zero+=zero;self.steps+=1
        self.window.append((float(yaw.min()),float(yaw.max()),positive,negative,zero,int(dones.sum())))
        self.pending=None;self.acted=False;self.stepping=False
        if self.steps%24==0:
            row=dict(control_steps=self.steps,common_step=int(env.common_step_counter),samples=6144,
                yaw_range=list(self.yaw_range),yaw_min=min(x[0] for x in self.window),yaw_max=max(x[1] for x in self.window),
                positive=sum(x[2] for x in self.window),negative=sum(x[3] for x in self.window),
                zero=sum(x[4] for x in self.window),reset_samples=sum(x[5] for x in self.window),
                actor_command_equal=True,nonreset_command_unchanged=True,
                neck_weight=-.1,motor_weight=-4.,lateral_weight=-.5)
            if self.path is not None:
                with self.path.open("a") as handle:
                    handle.write(json.dumps(row,allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
            self.window.clear()

    def finish(self, expected_steps):
        require(self.steps==expected_steps and self.pending is None and not self.window
                and not self.acted and not self.stepping,"complete actor/physics command evidence")
        require(self.positive+self.negative+self.zero==256*self.steps,"all command samples accounted")
        if self.yaw_range==(0.,0.):require(self.zero==256*self.steps,"zero-yaw control")
        else:require(self.positive>0 and self.negative>0,"both yaw signs actually reached actor")
        return dict(protocol="f1y-live-command-v1",control_steps=self.steps,yaw_range=list(self.yaw_range),
            positive=self.positive,negative=self.negative,zero=self.zero,
            actor_command_equal=True,nonreset_command_unchanged=True,
            commands_modified=False,policy_acceptance=False)


def validate_command_evidence(path, updates, yaw_range):
    result=json.loads((path/"result.json").read_text())["command_validator"]
    rows=[json.loads(line) for line in (path/"command-activity.jsonl").read_text().splitlines()]
    canonical([result,rows])
    require(result["protocol"]=="f1y-live-command-v1" and result["control_steps"]==24*updates
            and result["yaw_range"]==list(yaw_range) and result["actor_command_equal"] is True
            and result["nonreset_command_unchanged"] is True and result["commands_modified"] is False
            and result["policy_acceptance"] is False and len(rows)==updates,"complete command result")
    for i,row in enumerate(rows,1):
        require(row["control_steps"]==24*i and row["common_step"]==204000+24*i
                and row["yaw_range"]==list(yaw_range) and row["samples"]==6144
                and yaw_range[0]<=row["yaw_min"]<=row["yaw_max"]<=yaw_range[1]
                and row["actor_command_equal"] is True and row["nonreset_command_unchanged"] is True
                and row["neck_weight"]==-.1 and row["motor_weight"]==-4. and row["lateral_weight"]==-.5,
                "actual per-update command/reward evidence")
        require(all(type(row[k]) is int and 0<=row[k]<=6144 for k in ("positive","negative","zero","reset_samples"))
                and row["positive"]+row["negative"]+row["zero"]==6144,"complete update sign counts")
    require(all(result[k]==sum(row[k] for row in rows) for k in ("positive","negative","zero")),"reconciled sign totals")
    if yaw_range==(0.,0.):require(result["zero"]==6144*updates,"actual zero-yaw control")
    else:require(result["positive"]>0 and result["negative"]>0,"actual positive and negative yaw support")
    return result
