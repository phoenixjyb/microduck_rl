"""Matched discovery/recovery action-cost schedule; preserve motor/range gates."""
from __future__ import annotations

import sys
import math
import torch

from mjlab_microduck import integrated_motor_range_yaw_only as original
from mjlab_microduck import integrated_motor_refinement as motor

base, previous = original.base, original.previous
PROTOCOL = "integrated-motor-range-action-release-v1"
ARMS = ("control", "release")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke",64,5,1501),("benchmark",256,10,1511))
         for arm in ARMS}
MODES.update(control=(256,1000,1517),release=(256,1000,1517))
CAP_SECONDS = 2100
PARENT, PARENT_SHA = original.PARENT, original.PARENT_SHA
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = original.PARENT_NORMALIZER_COUNT, original.PARENT_ADAM_STEP
TERM, VARIANCE, WEIGHT = original.TERM, original.VARIANCE, original.WEIGHT
warm_load, evaluate_case, decision = original.warm_load, original.evaluate_case, original.decision
ACTION_TERM = "action_rate_l2"
GROUPS = ("idle", "straight", "negative_yaw", "positive_yaw")
TRAINING_STAGES = ((0,-.1),(250,-.4),(500,-1.))
SMOKE_STAGES = ((0,-.1),(1,-.4),(3,-1.))


def schedule(mode):
    base.require(mode in MODES, "declared action-release mode")
    if mode == "control" or mode.endswith("-control") or mode.startswith("benchmark-"):
        return ((0,-1.),)
    return SMOKE_STAGES if mode.startswith("smoke-") else TRAINING_STAGES


def prepare_config(mode):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp
    stages = schedule(mode)
    cfg, agent = original.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards[ACTION_TERM].weight = stages[0][1]
    cfg.metrics["action_rate_training_snapshot"] = MetricsTermCfg(
        func=mdp.capture_action_rate_training_metric,per_substep=False)
    agent.max_iterations, agent.save_interval = updates,250
    agent.experiment_name, agent.run_name = PROTOCOL,mode
    return cfg, agent


def command_masks(command):
    previous.safety.fresh.budget.command_exposure(command)  # validate full inherited support
    wz = command[:,2]
    idle = (command == 0).all(-1)
    masks = dict(zip(GROUPS,(idle,(wz == 0)&~idle,wz < 0,wz > 0),strict=True))
    base.require(sum(int(v.sum()) for v in masks.values()) == command.shape[0],
                 "exclusive action-rate command groups")
    return masks


def validate_action_row(row,mode,worlds,update):
    """Independent row-level reconciliation for preflight and retained output."""
    stages = schedule(mode)
    base.require(type(update) is int and 1 <= update <= MODES[mode][1]
                 and type(worlds) is int and worlds == MODES[mode][0],"declared action-rate row budget")
    weights = [motor.motor_weight(step,stages) for step in range((update-1)*24,update*24)]
    low,high = min(weights),max(weights)
    base.require(row["action_rate_schedule"] == [list(s) for s in stages]
                 and row["action_rate_weight_min"] == low and row["action_rate_weight_max"] == high
                 and row["curriculum"]["action_rate_weight"] == weights[-1],"actual action-rate stage weights")
    groups = row["action_rate_command_strata"]
    counts = row["command_exposure_counts"]
    expected = dict(idle=counts["idle"],straight=counts["straight"],
                    negative_yaw=sum(v for k,v in counts.items() if k.startswith("negative-")),
                    positive_yaw=sum(v for k,v in counts.items() if k.startswith("positive-")))
    total = worlds*24
    base.require(set(groups) == set(GROUPS) and type(row["action_rate_sample_count"]) is int
                 and row["action_rate_sample_count"] == total and sum(expected.values()) == total,
                 "complete action-rate command strata")
    for name,g in groups.items():
        n,cost,reward = g["sample_count"],g["cost_sum"],g["weighted_reward_sum"]
        base.require(type(n) is int and n == expected[name] and n >= 0
                     and base.finite_tree(g) and cost >= 0 and reward <= 0,"finite signed action-rate stratum")
        tolerance = 1e-6*max(1.,abs(cost*low),abs(cost*high))
        base.require(cost*low-tolerance <= reward <= cost*high+tolerance,"weighted action-rate stratum bounds")
        base.require((n == 0 and cost == reward == 0 and g["cost_mean"] is g["weighted_reward_mean"] is None)
                     or (n > 0 and g["cost_mean"] == cost/n and g["weighted_reward_mean"] == reward/n),
                     "action-rate stratum means")
    base.require(math.isclose(sum(g["cost_sum"] for g in groups.values())/total,row["action_rate_cost_mean"],rel_tol=1e-8,abs_tol=1e-8)
                 and math.isclose(sum(g["weighted_reward_sum"] for g in groups.values())/total,
                                  row["weighted_action_rate_reward_mean"],rel_tol=1e-8,abs_tol=1e-8),
                 "action-rate group and full means agree")


class MotorObserver(original.MotorObserver):
    def __init__(self,mode):
        self.release_mode = mode
        self.action_schedule = schedule(mode)
        super().__init__("smoke-control" if mode.startswith("smoke-") else "benchmark-control")
        cfg,_ = prepare_config(mode)
        self.action_func = cfg.rewards[ACTION_TERM].func
        self.action_params = dict(cfg.rewards[ACTION_TERM].params)
        self.last_action_weight = cfg.rewards[ACTION_TERM].weight
        self.action_costs, self.action_rewards, self.action_weights = [], [], []
        self.action_groups = {g:dict(sample_count=0,cost_sum=0.,weighted_reward_sum=0.) for g in GROUPS}
        self.pending_action_masks = None

    def before(self,env):
        base.require(self.action_schedule == schedule(self.release_mode),"unchanged action-release schedule")
        base.require(self.pending_action_masks is None
                     and getattr(env,"_microduck_action_rate_snapshot",None) is None,
                     "previous action-rate snapshot consumed")
        term = env.reward_manager.get_term_cfg(ACTION_TERM)
        base.require(term.func is self.action_func and term.params == self.action_params
                     and term.weight == self.last_action_weight,
                     "unchanged live raw action-rate function")
        self.current_action_weight = motor.motor_weight(env.common_step_counter,self.action_schedule)
        term.weight = self.current_action_weight  # mutate live manager, not env.cfg
        self.last_action_weight = self.current_action_weight
        super().before(env)
        self.pending_action_masks = command_masks(env.command_manager.get_command("twist").detach().clone())

    def step(self,env,sample,extras):
        super().step(env,sample,extras)
        snapshot = getattr(env,"_microduck_action_rate_snapshot",None)
        base.require(isinstance(snapshot,tuple) and len(snapshot) == 4
                     and type(snapshot[3]) is int and snapshot[3] == env.common_step_counter,
                     "same-step pre-reset action-rate snapshot")
        cost,actual,weight,_ = snapshot
        base.require(self.pending_action_masks is not None
                     and all(isinstance(v,torch.Tensor) and v.shape == (env.num_envs,) for v in (cost,actual))
                     and base.finite_tree(snapshot) and weight == self.current_action_weight
                     and bool((cost >= 0).all() & (actual <= 0).all())
                     and torch.allclose(actual,cost*weight,atol=1e-6,rtol=1e-5),
                     "actual negative action-rate reward matches live schedule")
        self.action_costs.append(float(cost.double().mean()))
        self.action_rewards.append(float(actual.double().mean()))
        self.action_weights.append(weight)
        for name,mask in self.pending_action_masks.items():
            g = self.action_groups[name]
            g["sample_count"] += int(mask.sum())
            g["cost_sum"] += float(cost[mask].double().sum())
            g["weighted_reward_sum"] += float(actual[mask].double().sum())
        self.pending_action_masks = None
        env._microduck_action_rate_snapshot = None

    def update(self,env):
        row = super().update(env)
        base.require(self.pending_action_masks is None
                     and len(self.action_costs) == len(self.action_rewards) == len(self.action_weights) == 24,
                     "complete 24-step action-rate telemetry")
        counts = row["command_exposure_counts"]
        expected = dict(idle=counts["idle"],straight=counts["straight"],
                        negative_yaw=sum(v for k,v in counts.items() if k.startswith("negative-")),
                        positive_yaw=sum(v for k,v in counts.items() if k.startswith("positive-")))
        base.require({g:v["sample_count"] for g,v in self.action_groups.items()} == expected
                     and sum(expected.values()) == env.num_envs*24,"aligned action-rate command dwell")
        row.update(action_rate_schedule=[list(s) for s in self.action_schedule],
                   action_rate_weight_min=min(self.action_weights),action_rate_weight_max=max(self.action_weights),
                   action_rate_cost_mean=sum(self.action_costs)/24,
                   weighted_action_rate_reward_mean=sum(self.action_rewards)/24,
                   action_rate_sample_count=env.num_envs*24,
                   action_rate_command_strata={g:{**v,
                       "cost_mean":v["cost_sum"]/v["sample_count"] if v["sample_count"] else None,
                       "weighted_reward_mean":v["weighted_reward_sum"]/v["sample_count"] if v["sample_count"] else None}
                       for g,v in self.action_groups.items()})
        self.action_costs.clear();self.action_rewards.clear();self.action_weights.clear()
        self.action_groups = {g:dict(sample_count=0,cost_sum=0.,weighted_reward_sum=0.) for g in GROUPS}
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
