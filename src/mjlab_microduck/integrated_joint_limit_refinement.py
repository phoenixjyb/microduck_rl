"""Matched qpos-side limit repair; preserve BAM targets and original gates."""

from __future__ import annotations

import sys

import torch

from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_tracking_refinement as tracking

base = motor.base
PROTOCOL = "integrated-qpos-limit-continuation-v1"
ARMS = ("control", "limit")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 887), ("benchmark", 256, 10, 889))
         for arm in ARMS}
MODES.update(control=(256, 1000, 907), limit=(256, 1000, 907))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/tracking-refinement-6dbc0613-20261010/control/model_999.pt"
PARENT_SHA = "b7a6601c479ef6805d4fc68236deff1805e6997bea4286e5b0f69b7b3255deec"
JOINT = "right_hip_yaw"
MARGIN = .15
LIMIT_STAGES = ((0, -1.), (250, -2.), (500, -4.))


def stages(mode):
    if mode.endswith("control"):
        return ((0, 0.),)
    if mode.startswith("smoke"):
        return ((0, -1.), (1, -2.), (2, -4.))
    if mode.startswith("benchmark"):
        return ((0, -4.),)
    return LIMIT_STAGES


def capture_limit_metric(env):
    # Metrics run after rewards and before autoreset. Use the manager-resolved
    # selector, not compiled IDs or a presumed canonical servo column.
    from mjlab_microduck.tasks.mdp import joint_pos_limit_proximity
    term = env.reward_manager.get_term_cfg("joint_limit")
    asset_cfg = term.params["asset_cfg"]
    data = env.scene[asset_cfg.name].data
    q = data.joint_pos[:, asset_cfg.joint_ids]
    bounds = data.joint_pos_limits[:, asset_cfg.joint_ids]
    cost = joint_pos_limit_proximity(env, **term.params)
    distance = torch.minimum(q-bounds[..., 0], bounds[..., 1]-q)
    base.require(q.shape == (env.num_envs, 1) and base.finite_tree((q, bounds, cost, distance))
                 and bool((bounds[..., 1] > bounds[..., 0]).all()) and bool((cost >= 0).all()),
                 "finite named pre-reset qpos cost")
    env._microduck_limit_snapshot = (cost.detach().clone(), distance.detach().clone())
    return torch.zeros(env.num_envs, device=env.device)


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg, SceneEntityCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    cfg, agent = tracking.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards["joint_limit"] = RewardTermCfg(func=mdp.joint_pos_limit_proximity,
        weight=stages(mode)[0][1], params=dict(margin=MARGIN,
        asset_cfg=SceneEntityCfg("robot", joint_names=("^right_hip_yaw$",))))
    cfg.metrics["microduck_limit_cost"] = MetricsTermCfg(func=capture_limit_metric, per_substep=False)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    return motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                           parent_iteration=999, parent_steps=24000)


class MotorObserver(tracking.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.limit_schedule = stages(mode)
        self.limit_costs, self.limit_rewards, self.limit_weights = [], [], []
        self.limit_exposures, self.limit_overruns = [], []

    def before(self, env):
        super().before(env)
        self.limit_weight = motor.motor_weight(env.common_step_counter, self.limit_schedule)
        env.reward_manager.get_term_cfg("joint_limit").weight = self.limit_weight
        env._microduck_limit_snapshot = None

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        snapshot = env._microduck_limit_snapshot
        base.require(isinstance(snapshot, tuple) and len(snapshot) == 2, "pre-reset limit snapshot")
        cost, distance = snapshot
        base.require(cost.shape == sample.mean_cost.shape and distance.shape == (len(cost), 1)
                     and base.finite_tree(snapshot) and bool((cost >= 0).all()), "finite limit cost")
        idx = env.reward_manager.active_terms.index("joint_limit")
        weighted = env.reward_manager._step_reward[:, idx]
        base.require(base.finite_tree(weighted) and bool((weighted <= 0).all())
                     and torch.allclose(weighted, cost*self.limit_weight, atol=1e-6, rtol=1e-5),
                     "limit reward matches pre-reset qpos cost")
        self.limit_costs.append(float(cost.mean()))
        self.limit_rewards.append(float(weighted.mean()))
        self.limit_weights.append(self.limit_weight)
        self.limit_exposures.append(float((distance <= .05).double().mean()))
        self.limit_overruns.append(float((-distance).clamp_min(0).max()))
        env._microduck_limit_snapshot = None

    def update(self, env):
        row = super().update(env)
        row.update(limit_weight_min=min(self.limit_weights), limit_weight_max=max(self.limit_weights),
            limit_cost_mean=sum(self.limit_costs)/24, weighted_limit_reward_mean=sum(self.limit_rewards)/24,
            right_hip_stop_exposure_mean=sum(self.limit_exposures)/24,
            right_hip_max_range_overrun_rad=max(self.limit_overruns))
        for values in (self.limit_costs, self.limit_rewards, self.limit_weights,
                       self.limit_exposures, self.limit_overruns):
            values.clear()
        return row


def evaluate_case(*args):
    return base.evaluate_case(*args, diagnostics=True)


def decision(rows):
    original = base.decision(rows)
    failures = list(original["failures"])
    for row in rows:
        d = row["response_diagnostics"]
        base.require(d["protocol"] == "signed-body-response-and-joint-load-v1"
                     and d["hard_stop_margin_rad"] == .05
                     and len(d["joint_columns"]) == 14 and len(set(d["joint_columns"])) == 14
                     and len(d["min_distance_to_hard_stop_rad_by_joint"]) == 14
                     and len(d["hard_stop_proximity_fraction_by_joint"]) == 14
                     and all(0 <= v <= 1 for v in d["hard_stop_proximity_fraction_by_joint"])
                     and d["steps"] == row["steps"], "named position coverage")
        index = d["joint_columns"].index(JOINT)
        if (min(d["min_distance_to_hard_stop_rad_by_joint"]) < -.005
                or d["hard_stop_proximity_fraction_by_joint"][index] > .01):
            failures.append([row["seed"], row["speed"], row["yaw"], "joint-range-or-stop-exposure"])
    return dict(original_foundation_decision=original, failures=failures,
        decision="foundation-ready-for-next-experiment" if not failures else "foundation-not-ready",
        policy_acceptance=False, physical_motion_authorized=False)


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
