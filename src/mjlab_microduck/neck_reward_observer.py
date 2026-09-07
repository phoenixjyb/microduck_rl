"""Read-only evidence that the existing neck action-rate cost is live."""

import json
import os

import torch

from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.motor_audit_smoke import JOINTS
from mjlab_microduck.tasks import mdp

TERM = "neck_action_rate_l2"
NECK = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


class NeckRewardObserver:
    def __init__(self, weight, path=None):
        require(weight in (-.1,-.2), "predeclared neck weight")
        self.weight, self.path = weight, path
        self.steps, self.positive_samples, self.raw_sum = 0, 0, 0.
        self.window, self.expected = [], None

    def before_reward(self, env):
        require(self.expected is None, "one before/after reward observer pair")
        require(tuple(env.scene["robot"].joint_names) == JOINTS and JOINTS[5:9] == NECK,
                "named rigid robot action mapping")
        require(env.reward_manager.get_term_cfg(TERM).func is mdp.neck_action_rate_l2,
                "unchanged neck cost implementation")
        require(env.reward_manager.get_term_cfg(TERM).weight == self.weight
                and env.reward_manager.get_term_cfg("motor_torque_load").weight == -4.
                and env.reward_manager.get_term_cfg("lateral_velocity_cost").weight == -.5,
                "live predeclared neck/motor/lateral weights")
        actions = env.action_manager.action
        require(actions.shape == (256,14) and bool(torch.isfinite(actions).all()), "finite original 14D actions")
        neck = actions[:,5:9]
        previous = getattr(env,"_prev_neck_actions",None)
        if previous is None:
            self.expected = torch.zeros(256,device=actions.device,dtype=actions.dtype)
        else:
            require(previous.shape == neck.shape and bool(torch.isfinite(previous).all()), "finite original neck cache")
            self.expected = (neck-previous).square().sum(1)

    def after_reward(self, env):
        require(self.expected is not None, "reward observed after before hook")
        manager=env.reward_manager
        index=manager.active_terms.index(TERM)
        weighted=manager._step_reward[:,index]
        require(bool(torch.isfinite(weighted).all())
                and bool(torch.allclose(weighted,self.expected*self.weight,atol=1e-6,rtol=1e-6)),
                "consumed weighted neck reward matches unchanged cost")
        raw=float(self.expected.mean()); positive=int((self.expected>0).sum())
        self.steps+=1; self.raw_sum+=raw; self.positive_samples+=positive
        self.window.append((raw,float(weighted.mean()),positive))
        self.expected=None
        if self.steps % 24 == 0:
            row=dict(control_steps=self.steps,common_step=int(env.common_step_counter),
                neck_weight=self.weight,motor_weight=-4.,lateral_weight=-.5,
                raw_neck_cost_mean=sum(r[0] for r in self.window)/24,
                consumed_weighted_neck_mean=sum(r[1] for r in self.window)/24,
                positive_samples=sum(r[2] for r in self.window),samples=24*256)
            if self.path is not None:
                with self.path.open("a") as handle:
                    handle.write(json.dumps(row,allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
            self.window.clear()

    def finish(self, expected_steps):
        require(self.steps == expected_steps and not self.window and self.expected is None,
                "complete neck reward observation coverage")
        require(self.positive_samples > 0 and self.raw_sum > 0, "neck reward was active, not silently disabled")
        return dict(protocol="f1n-live-neck-reward-v1",control_steps=self.steps,neck_weight=self.weight,
                    positive_samples=self.positive_samples,raw_neck_cost_mean=self.raw_sum/self.steps,
                    original_cost_and_cache_unchanged=True,policy_acceptance=False)
