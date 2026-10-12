"""Matched angular aggregation diagnostic, retaining bilateral/motor shaping."""
from __future__ import annotations

import math
import sys

import torch
from mjlab_microduck import integrated_motor_range_forward as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-motor-range-yaw-decoupling-v1"
ARMS = ("control", "decoupled")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke",64,5,1423),("benchmark",256,10,1427))
         for arm in ARMS}
MODES.update(control=(256,1000,1429),decoupled=(256,1000,1429))
CAP_SECONDS = 2100
PARENT = base.ROOT / "artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt"
PARENT_SHA = "936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6"
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = 43008000, 140000
TERM = "track_angular_velocity"
VARIANCE, WEIGHT = .5, 2.
evaluate_case, decision = previous.evaluate_case, previous.decision


def decoupled(mode):
    base.require(mode in MODES, "declared motor range yaw mode")
    return mode == "decoupled" or mode.endswith("-decoupled")


def prepare_config(mode):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity
    from mjlab_microduck.tasks import mdp

    treatment = decoupled(mode)
    # Preserve the actual control parent's zero forward term and read-only
    # metric. Never inherit the rejected forward arm's -.25 objective.
    cfg, agent = previous.prepare_config("control")
    worlds,updates,seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    term = cfg.rewards[TERM]
    base.require(term.func is track_angular_velocity and term.weight == WEIGHT
                 and term.params == dict(std=math.sqrt(VARIANCE),command_name="twist"),
                 "unaltered stock angular control")
    if treatment:
        term.func = mdp.decoupled_body_angular_tracking
        term.params = dict(command_name="twist",yaw_variance=VARIANCE,roll_pitch_variance=VARIANCE)
    cfg.metrics["motor_range_angular_snapshot"] = MetricsTermCfg(func=capture_angular_metric,per_substep=False)
    base.require(cfg.rewards["forward_error"].weight == 0. and not cfg.curriculum
                 and cfg.rewards["motor_load"].weight == -1.
                 and cfg.rewards["joint_limit"].weight == -4., "unchanged parent safety and no forward pressure")
    agent.max_iterations,agent.save_interval = updates,250
    agent.experiment_name,agent.run_name = PROTOCOL,mode
    return cfg,agent


def capture_angular_metric(env):
    from mjlab_microduck.tasks.mdp import body_angular_tracking_errors
    base.require(getattr(env,"_microduck_yaw_aggregation_snapshot",None) is None,
                 "previous angular snapshot consumed")
    term = env.reward_manager.get_term_cfg(TERM)
    yaw,xy = body_angular_tracking_errors(env,term.params["command_name"])
    actual = env.reward_manager._step_reward[:,env.reward_manager.active_terms.index(TERM)]
    env._microduck_yaw_aggregation_snapshot = (yaw.detach().clone(),xy.detach().clone(),
                                               actual.detach().clone(),env.common_step_counter)
    return torch.zeros(env.num_envs,device=env.device)


def warm_load(runner,env):
    receipt = motor.warm_load(runner,env,PARENT,parent_sha=PARENT_SHA,parent_iteration=999,parent_steps=24000)
    receipt.update(previous.restored_counters(runner,normalizer_count=PARENT_NORMALIZER_COUNT,adam_step=PARENT_ADAM_STEP))
    previous.safety.fresh.kl.install_optimizer_capture(runner,env,initial_lr=receipt["learning_rate"])
    return receipt


class MotorObserver(previous.MotorObserver):
    def __init__(self,mode):
        self.treatment = decoupled(mode)
        super().__init__("smoke-control" if mode.startswith("smoke-") else "benchmark-control")
        cfg,_ = prepare_config(mode)
        self.expected_func = cfg.rewards[TERM].func
        self.expected_params = dict(cfg.rewards[TERM].params)
        self.yaw_costs,self.xy_costs,self.angular_rewards = [],[],[]
        self.negative_samples,self.sample_count = 0,0

    def before(self,env):
        super().before(env)
        term = env.reward_manager.get_term_cfg(TERM)
        base.require(term.func is self.expected_func and term.params == self.expected_params and term.weight == WEIGHT,
                     "fixed live angular aggregation")
        base.require(getattr(env,"_microduck_yaw_aggregation_snapshot",None) is None,
                     "previous angular snapshot consumed")

    def step(self,env,sample,extras):
        super().step(env,sample,extras)
        snapshot = getattr(env,"_microduck_yaw_aggregation_snapshot",None)
        base.require(isinstance(snapshot,tuple) and len(snapshot) == 4
                     and type(snapshot[3]) is int and snapshot[3] == env.common_step_counter,
                     "same-step pre-reset angular snapshot")
        yaw,xy,actual,_ = snapshot
        base.require(all(isinstance(v,torch.Tensor) and v.shape == (env.num_envs,) for v in (yaw,xy,actual))
                     and base.finite_tree(snapshot) and bool((yaw >= 0).all() & (xy >= 0).all()),
                     "finite pre-reset angular components")
        expected = (WEIGHT*(torch.exp(-yaw/VARIANCE)+torch.exp(-xy/VARIANCE)-1.) if self.treatment
                    else WEIGHT*torch.exp(-(yaw+xy)/VARIANCE))
        base.require(bool(((actual >= (-WEIGHT if self.treatment else 0.)) & (actual <= WEIGHT)).all())
                     and torch.allclose(actual,expected,atol=1e-6,rtol=1e-5),
                     "actual signed angular reward matches live aggregation")
        self.yaw_costs.append(float(yaw.mean()));self.xy_costs.append(float(xy.mean()))
        self.angular_rewards.append(float(actual.mean()))
        self.negative_samples += int((actual < 0).sum());self.sample_count += actual.numel()
        env._microduck_yaw_aggregation_snapshot = None

    def update(self,env):
        row = super().update(env)
        base.require(len(self.yaw_costs) == len(self.xy_costs) == len(self.angular_rewards) == 24
                     and self.sample_count == env.num_envs*24,"complete angular update telemetry")
        row.update(angular_aggregation="additive_decoupled" if self.treatment else "stock_multiplicative_product",
            yaw_variance=VARIANCE,roll_pitch_variance=VARIANCE,angular_tracking_weight=WEIGHT,
            yaw_error_sq_mean=sum(self.yaw_costs)/24,roll_pitch_rate_sq_mean=sum(self.xy_costs)/24,
            weighted_angular_reward_mean=sum(self.angular_rewards)/24,
            negative_reward_sample_count=self.negative_samples,angular_reward_sample_count=self.sample_count,
            negative_reward_fraction=self.negative_samples/self.sample_count)
        self.yaw_costs.clear();self.xy_costs.clear();self.angular_rewards.clear()
        self.negative_samples,self.sample_count = 0,0
        return row


if __name__ == "__main__": motor.main(sys.modules[__name__])
