"""Fresh stock-curriculum baseline with staged motor and bilateral qpos costs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from mjlab_microduck import integrated_fresh_foundation as fresh
from mjlab_microduck import integrated_motor_refinement as motor

base = fresh.base
PROTOCOL = "integrated-fresh-motor-range-foundation-v1"
MODES = dict(fresh.MODES)
FINAL_CHECKPOINT = fresh.FINAL_CHECKPOINT
REQUIRED_COVERAGE = fresh.REQUIRED_COVERAGE
EVAL_SEEDS, EVAL_CASES = fresh.EVAL_SEEDS, fresh.EVAL_CASES
evaluate_case, decision = fresh.evaluate_case, fresh.decision
FOUNDATION_CAP_SECONDS = 10800
FOUNDATION_PROJECTION_LIMIT_SECONDS = 10740
JOINTS = ("left_hip_yaw", "right_hip_yaw")
MARGIN = .15
MOTOR_STAGES = ((0, 0.), (250, -.25), (1000, -.50), (2000, -1.))
LIMIT_STAGES = ((0, -.25), (250, -1.), (1000, -2.), (2000, -4.))


def schedules(mode):
    base.require(mode in MODES, "declared fresh motor/range mode")
    if mode == "smoke":
        return (((0, 0.), (1, -.25), (2, -.50), (3, -1.)),
                ((0, -.25), (1, -1.), (2, -2.), (3, -4.)))
    if mode == "benchmark":
        return ((0, -1.),), ((0, -4.),)
    return MOTOR_STAGES, LIMIT_STAGES


def capture_limit_metric(env):
    """Freeze both named qpos columns after reward computation, before reset."""
    from mjlab_microduck.tasks.mdp import joint_pos_limit_proximity

    term = env.reward_manager.get_term_cfg("joint_limit")
    selector = term.params["asset_cfg"]
    robot = env.scene[selector.name]
    names = tuple(robot.joint_names[index] for index in selector.joint_ids)
    q = robot.data.joint_pos[:, selector.joint_ids]
    bounds = robot.data.joint_pos_limits[:, selector.joint_ids]
    cost = joint_pos_limit_proximity(env, **term.params)
    distance = torch.minimum(q-bounds[..., 0], bounds[..., 1]-q)
    columns = (bounds[..., 0]+MARGIN-q).clamp_min(0) + (q-bounds[..., 1]+MARGIN).clamp_min(0)
    base.require(len(names) == 2 and set(names) == set(JOINTS)
                 and q.shape == distance.shape == (env.num_envs, 2)
                 and bounds.shape == (env.num_envs, 2, 2)
                 and cost.shape == (env.num_envs,)
                 and base.finite_tree((q, bounds, cost, distance))
                 and bool((bounds[..., 1]-bounds[..., 0] > 2*MARGIN).all())
                 and bool((cost >= 0).all())
                 and torch.allclose(cost, columns.sum(-1), atol=1e-6, rtol=1e-5),
                 "finite bilateral named pre-reset qpos cost")
    env._microduck_fresh_limit_snapshot = (cost.detach().clone(), distance.detach().clone(),
                                          names, columns.detach().clone())
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg, SceneEntityCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    cfg, agent = fresh.prepare_config(mode)
    motor_stages, limit_stages = schedules(mode)
    cfg.rewards["motor_load"] = RewardTermCfg(func=mdp.motor_torque_load_cost,
        weight=motor_stages[0][1], params=dict(rated_stall_torque_nm=.60,
                                             soft_limit_fraction=.70, over_limit_gain=4.))
    cfg.rewards["joint_limit"] = RewardTermCfg(func=mdp.joint_pos_limit_proximity,
        weight=limit_stages[0][1], params=dict(margin=MARGIN,
        asset_cfg=SceneEntityCfg("robot", joint_names=("^(left|right)_hip_yaw$",))))
    cfg.metrics["fresh_bilateral_limit_cost"] = MetricsTermCfg(
        func=capture_limit_metric, per_substep=False)
    agent.experiment_name = PROTOCOL
    return cfg, agent


class FreshObserver(fresh.FreshObserver):
    def __init__(self, mode):
        super().__init__(mode)
        self.motor = motor.MotorObserver("control")
        self.motor.schedule, self.limit_schedule = schedules(mode)
        self.costs, self.rewards, self.weights, self.exposures, self.overruns = [], [], [], [], []
        self.column_costs = []

    def before_action(self, env):
        super().before_action(env)
        self.motor.before(env)
        self.limit_weight = motor.motor_weight(env.common_step_counter, self.limit_schedule)
        env.reward_manager.get_term_cfg("joint_limit").weight = self.limit_weight
        env._microduck_fresh_limit_snapshot = None

    def after_step(self, env, sample, extras):
        super().after_step(env, sample, extras)
        self.motor.step(env, sample, extras)
        # Stream load telemetry is deliberately uncapped, but the reward caps
        # utilization at2. Report the actual raw reward cost alongside its
        # weighted value; rare >2 samples must not silently change its meaning.
        utilization = (sample.force_nm.abs().float()/.60).clamp(max=2.)
        reward_cost = (utilization.square()+4.*(utilization-.70).clamp_min(0).square()).mean(1)
        self.motor.costs[-1] = float(reward_cost.mean())
        snapshot = env._microduck_fresh_limit_snapshot
        base.require(isinstance(snapshot, tuple) and len(snapshot) == 4,
                     "pre-reset bilateral limit snapshot present")
        cost, distance, names, columns = snapshot
        index = env.reward_manager.active_terms.index("joint_limit")
        weighted = env.reward_manager._step_reward[:, index]
        base.require(cost.shape == weighted.shape == (env.num_envs,)
                     and distance.shape == columns.shape == (env.num_envs, 2)
                     and len(names) == 2 and set(names) == set(JOINTS)
                     and base.finite_tree((cost, distance, weighted, columns))
                     and bool((columns >= 0).all())
                     and torch.allclose(cost, columns.sum(-1), atol=1e-6, rtol=1e-5)
                     and bool((cost >= 0).all()) and bool((weighted <= 0).all())
                     and torch.allclose(weighted, cost*self.limit_weight, atol=1e-6, rtol=1e-5),
                     "bilateral limit reward matches pre-reset cost including terminal worlds")
        self.costs.append(float(cost.mean()))
        self.rewards.append(float(weighted.mean()))
        self.weights.append(self.limit_weight)
        self.column_costs.append({name: float(columns[:, names.index(name)].mean()) for name in JOINTS})
        self.exposures.append({name: float((distance[:, names.index(name)] <= .05).double().mean())
                               for name in JOINTS})
        self.overruns.append({name: float((-distance[:, names.index(name)]).clamp_min(0).max())
                              for name in JOINTS})
        env._microduck_fresh_limit_snapshot = None

    def after_update(self, env):
        row = super().after_update(env)
        base.require(all(len(values) == 24 for values in
                         (self.costs, self.rewards, self.weights, self.exposures, self.overruns, self.column_costs)),
                     "complete bilateral limit update telemetry")
        row.update(self.motor.update(env))
        row.update(limit_weight_min=min(self.weights), limit_weight_max=max(self.weights),
            limit_cost_mean=sum(self.costs)/24, weighted_limit_reward_mean=sum(self.rewards)/24,
            hip_limit_cost_mean={name: sum(item[name] for item in self.column_costs)/24 for name in JOINTS},
            hip_stop_exposure_mean={name: sum(item[name] for item in self.exposures)/24 for name in JOINTS},
            hip_max_range_overrun_rad={name: max(item[name] for item in self.overruns) for name in JOINTS})
        for values in (self.costs, self.rewards, self.weights, self.exposures, self.overruns, self.column_costs):
            values.clear()
        return row


benchmark_projection_seconds = fresh.benchmark_projection_seconds


def projection_admitted(elapsed_s):
    return benchmark_projection_seconds(elapsed_s) < FOUNDATION_PROJECTION_LIMIT_SECONDS


def validate_training_receipt(root, mode, source, receipt):
    return fresh.validate_training_receipt(root, mode, source, receipt,
        protocol=PROTOCOL, cap_seconds=FOUNDATION_CAP_SECONDS)


def validate_evaluation_receipt(root, source, receipt):
    validate_training_receipt(root, "foundation", source, receipt)
    return Path(root) / "foundation" / "model_5999.pt"


if __name__ == "__main__":
    fresh.main(sys.modules[__name__])
