"""Direct linear-tracking mass contrast with full smoothness and fixed safety."""
from __future__ import annotations

import math
import sys
import torch
from mjlab_microduck import integrated_motor_range_action_release as action
from mjlab_microduck import integrated_motor_refinement as motor

base, previous = action.base, action.previous
PROTOCOL = "integrated-motor-range-linear-boost-v1"
ARMS = ("control", "linearboost")
MODES = {f"{phase}-{arm}": (worlds,updates,seed)
         for phase,worlds,updates,seed in (("smoke",64,5,1523),("benchmark",256,10,1531))
         for arm in ARMS}
MODES.update(control=(256,1000,1543),linearboost=(256,1000,1543))
CAP_SECONDS = 2100
PARENT, PARENT_SHA = action.PARENT, action.PARENT_SHA
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = action.PARENT_NORMALIZER_COUNT, action.PARENT_ADAM_STEP
TERM, VARIANCE, WEIGHT = action.TERM, action.VARIANCE, action.WEIGHT
warm_load, evaluate_case, decision = action.warm_load, action.evaluate_case, action.decision
LINEAR_TERM = "track_linear_velocity"
GROUPS = action.GROUPS


def linear_weight(mode):
    base.require(mode in MODES,"declared linear-boost mode")
    return 2.5 if mode == "linearboost" or mode.endswith("-linearboost") else 2.


def action_mode(mode):
    linear_weight(mode)
    return "smoke-control" if mode.startswith("smoke-") else "benchmark-control" if mode.startswith("benchmark-") else "control"


def prepare_config(mode):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp
    weight = linear_weight(mode)
    cfg,agent = action.prepare_config("control")
    worlds,updates,seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards[LINEAR_TERM].weight = weight
    cfg.metrics["linear_tracking_training_snapshot"] = MetricsTermCfg(
        func=mdp.capture_linear_tracking_training_metric,per_substep=False)
    agent.max_iterations,agent.save_interval = updates,250
    agent.experiment_name,agent.run_name = PROTOCOL,mode
    return cfg,agent


def validate_action_row(row,mode,worlds,update):
    action.validate_action_row(row,action_mode(mode),worlds,update)


def validate_linear_row(row,mode,worlds,update):
    weight = linear_weight(mode)
    base.require(type(update) is int and 1 <= update <= MODES[mode][1]
                 and type(worlds) is int and worlds == MODES[mode][0],"declared linear tracking row budget")
    base.require(row["linear_tracking_weight"] == weight and row["linear_tracking_variance"] == .1
                 and row["linear_tracking_sample_count"] == worlds*24,"fixed linear tracking weight variance and sample count")
    counts = row["command_exposure_counts"]
    expected = dict(idle=counts["idle"],straight=counts["straight"],
                    negative_yaw=sum(v for k,v in counts.items() if k.startswith("negative-")),
                    positive_yaw=sum(v for k,v in counts.items() if k.startswith("positive-")))
    groups = row["linear_tracking_command_strata"]
    base.require(set(groups) == set(GROUPS) and sum(expected.values()) == worlds*24,"complete linear tracking strata")
    for name,g in groups.items():
        n,raw,reward = g["sample_count"],g["raw_sum"],g["weighted_reward_sum"]
        base.require(type(n) is int and n == expected[name] and n >= 0 and base.finite_tree(g)
                     and 0 <= raw <= n and 0 <= reward <= n*weight
                     and math.isclose(reward,weight*raw,rel_tol=1e-5,abs_tol=1e-6),"finite positive linear tracking stratum")
        base.require((n == 0 and g["raw_mean"] is g["weighted_reward_mean"] is None)
                     or (n > 0 and g["raw_mean"] == raw/n and g["weighted_reward_mean"] == reward/n),
                     "linear tracking stratum means")
    total = worlds*24
    base.require(math.isclose(sum(g["raw_sum"] for g in groups.values())/total,row["linear_tracking_raw_mean"],rel_tol=1e-8,abs_tol=1e-8)
                 and math.isclose(sum(g["weighted_reward_sum"] for g in groups.values())/total,
                                  row["weighted_linear_tracking_reward_mean"],rel_tol=1e-8,abs_tol=1e-8),
                 "linear tracking group and full means agree")


class MotorObserver(action.MotorObserver):
    def __init__(self,mode):
        self.linear_mode,self.linear_weight = mode,linear_weight(mode)
        super().__init__(action_mode(mode))
        cfg,_ = prepare_config(mode)
        self.linear_func = cfg.rewards[LINEAR_TERM].func
        self.linear_params = dict(cfg.rewards[LINEAR_TERM].params)
        self.linear_raw,self.linear_rewards = [],[]
        self.linear_groups = {g:dict(sample_count=0,raw_sum=0.,weighted_reward_sum=0.) for g in GROUPS}

    def before(self,env):
        term = env.reward_manager.get_term_cfg(LINEAR_TERM)
        base.require(self.linear_weight == linear_weight(self.linear_mode)
                     and term.func is self.linear_func and term.params == self.linear_params
                     and term.weight == self.linear_weight,"fixed live linear tracking contract")
        base.require(getattr(env,"_microduck_linear_tracking_snapshot",None) is None,"previous linear tracking snapshot consumed")
        super().before(env)

    def step(self,env,sample,extras):
        masks = self.pending_action_masks
        super().step(env,sample,extras)
        snapshot = getattr(env,"_microduck_linear_tracking_snapshot",None)
        base.require(isinstance(snapshot,tuple) and len(snapshot) == 4 and type(snapshot[3]) is int
                     and snapshot[3] == env.common_step_counter,"same-step pre-reset linear tracking snapshot")
        raw,actual,weight,_ = snapshot
        base.require(masks is not None and weight == self.linear_weight and base.finite_tree(snapshot)
                     and all(isinstance(v,torch.Tensor) and v.shape == (env.num_envs,) for v in (raw,actual))
                     and bool(((raw >= 0)&(raw <= 1)&(actual >= 0)&(actual <= weight)).all())
                     and torch.allclose(actual,raw*weight,atol=1e-6,rtol=1e-5),"actual positive linear tracking matches fixed weight")
        self.linear_raw.append(float(raw.double().mean()))
        self.linear_rewards.append(float(actual.double().mean()))
        for name,mask in masks.items():
            g = self.linear_groups[name]
            g["sample_count"] += int(mask.sum())
            g["raw_sum"] += float(raw[mask].double().sum())
            g["weighted_reward_sum"] += float(actual[mask].double().sum())
        env._microduck_linear_tracking_snapshot = None

    def update(self,env):
        row = super().update(env)
        base.require(len(self.linear_raw) == len(self.linear_rewards) == 24,"complete linear tracking update telemetry")
        row.update(linear_tracking_weight=self.linear_weight,linear_tracking_variance=.1,
            linear_tracking_sample_count=env.num_envs*24,
            linear_tracking_raw_mean=sum(self.linear_raw)/24,
            weighted_linear_tracking_reward_mean=sum(self.linear_rewards)/24,
            linear_tracking_command_strata={g:{**v,"raw_mean":v["raw_sum"]/v["sample_count"] if v["sample_count"] else None,
                "weighted_reward_mean":v["weighted_reward_sum"]/v["sample_count"] if v["sample_count"] else None}
                for g,v in self.linear_groups.items()})
        validate_linear_row(row,self.linear_mode,env.num_envs,env.common_step_counter//24)
        self.linear_raw.clear();self.linear_rewards.clear()
        self.linear_groups = {g:dict(sample_count=0,raw_sum=0.,weighted_reward_sum=0.) for g in GROUPS}
        return row


if __name__ == "__main__": motor.main(sys.modules[__name__])
