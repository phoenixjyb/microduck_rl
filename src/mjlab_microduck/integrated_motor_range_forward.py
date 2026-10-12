"""One-factor forward tracking diagnostic from the unaccepted safe foundation.

Preserve the parent's final stock curriculum state, motor and bilateral qpos
objectives. Only the constant forward residual weight differs between arms.
"""

from __future__ import annotations

import copy
import sys

import torch

from mjlab_microduck import integrated_fresh_motor_range as safety
from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_motor_refinement as motor

base = motor.base
PROTOCOL = "integrated-motor-range-forward-diagnostic-v1"
ARMS = ("control", "forward")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1381), ("benchmark", 256, 10, 1399))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1409), forward=(256, 1000, 1409))
CAP_SECONDS = 2100
PARENT = base.ROOT / "artifacts/training/fresh-motor-range-d0cba9c0-20261012/foundation/model_5999.pt"
PARENT_SHA = "811b38b530e508d08967dba0ff194cf14fddb35011b9a4878a04f71a5b804190"
PARENT_STEPS = 144000
PARENT_NORMALIZER_COUNT = 36864000
PARENT_ADAM_STEP = 120000
FORWARD_WEIGHT = -.25
evaluate_case, decision = safety.evaluate_case, safety.decision


def forward_weight(mode):
    base.require(mode in MODES, "declared safe forward mode")
    return 0. if mode == "control" or mode.endswith("-control") else FORWARD_WEIGHT


def freeze_parent_curricula(cfg):
    """Pin every source stage actually reached by the 144000-step parent."""
    expected = {"action_rate_weight", "standing_envs", "head_pose_range", "body_pose_range",
                "com_range", "head_com_range", "head_pose_bias_weight"}
    base.require(set(cfg.curriculum) == expected, "known seven parent curricula")
    for term in cfg.curriculum.values():
        params = term.params
        keys = [key for key in params if key.endswith("_stages")]
        base.require(len(keys) == 1, "single source stage declaration")
        key = keys[0]
        stages = params[key]
        base.require(stages and all(stage["step"] < PARENT_STEPS for stage in stages)
                     and [stage["step"] for stage in stages] == sorted(stage["step"] for stage in stages),
                     "parent reached every ordered source stage")
        stage = stages[-1]
        if key == "weight_stages":
            cfg.rewards[params["reward_name"]].weight = stage["weight"]
        elif key == "standing_stages":
            cfg.commands[params["command_name"]].rel_standing_envs = stage["rel_standing_envs"]
        elif "command_name" in params:
            cfg.commands[params["command_name"]].ranges = copy.deepcopy(stage["ranges"])
        else:
            cfg.events[params["event_name"]].params["ranges"] = (-stage["range"], stage["range"])
    cfg.curriculum.clear()
    base.require(cfg.rewards["action_rate_l2"].weight == -1.
                 and cfg.rewards["head_pose_bias"].weight == 3.
                 and cfg.commands["twist"].rel_standing_envs == .25
                 and cfg.commands["head_pose"].ranges == ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-.31, .31))
                 and cfg.commands["body_pose"].ranges == ((-.005, .005),)*3 + ((-.05, .05),)*3
                 and cfg.events["randomize_com"].params["ranges"] == (-.015, .015)
                 and cfg.events["randomize_head_com"].params["ranges"] == (-.01, .01),
                 "exact final parent curriculum configuration")


def prepare_config(mode):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab_microduck.tasks import mdp

    weight = forward_weight(mode)
    cfg, agent = safety.prepare_config("benchmark")
    freeze_parent_curricula(cfg)
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards["forward_error"] = RewardTermCfg(func=mdp.body_forward_tracking_cost,
        weight=weight, params={"command_name": "twist"})
    cfg.metrics["motor_range_forward_cost"] = MetricsTermCfg(
        func=forward.capture_forward_metric, per_substep=False)
    agent.max_iterations, agent.save_interval = updates, 250
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def restored_counters(runner):
    counts = {}
    for name in ("actor", "critic"):
        model = getattr(runner.alg, name)
        matches = [module.count for module in model.modules()
                   if isinstance(getattr(module, "count", None), torch.Tensor)
                   and all(isinstance(getattr(module, key, None), torch.Tensor)
                           for key in ("_mean", "_var", "_std"))]
        base.require(len(matches) == 1 and matches[0].ndim == 0
                     and matches[0].dtype == torch.int64
                     and matches[0].item() == PARENT_NORMALIZER_COUNT,
                     "exact restored " + name + " normalizer count")
        counts[name + "_normalizer_count"] = matches[0].item()
    states = list(runner.alg.optimizer.state.values())
    base.require(len(states) == 17 and all(isinstance(state.get("step"), torch.Tensor)
                 and state["step"].numel() == 1 and state["step"].item() == PARENT_ADAM_STEP
                 for state in states), "exact restored seventeen Adam counters")
    return dict(**counts, adam_state_entries=17, adam_step=PARENT_ADAM_STEP)


def warm_load(runner, env):
    receipt = motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                             parent_iteration=5999, parent_steps=PARENT_STEPS)
    receipt.update(restored_counters(runner))
    safety.fresh.kl.install_optimizer_capture(runner, env, initial_lr=receipt["learning_rate"])
    return receipt


class MotorObserver(safety.FreshObserver):
    """Reuse bilateral snapshots, capped motor cost and action/KL telemetry."""

    def __init__(self, mode):
        self.forward_weight = forward_weight(mode)
        # The shared update observer uses only this mode's world count. Both
        # continuation phases have 256 worlds; smoke alone has 64. Never patch
        # its module-global mode table or reinterpret it as a fresh learner.
        super().__init__("smoke" if mode.startswith("smoke-") else "benchmark")
        self.motor.schedule, self.limit_schedule = ((0, -1.),), ((0, -4.),)
        self.forward_costs, self.forward_rewards = [], []

    def before(self, env):
        self.before_action(env)
        env.reward_manager.get_term_cfg("forward_error").weight = self.forward_weight
        env._microduck_forward_cost = None

    def step(self, env, sample, extras):
        self.after_step(env, sample, extras)
        cost = env._microduck_forward_cost
        base.require(isinstance(cost, torch.Tensor) and cost.shape == (env.num_envs,)
                     and base.finite_tree(cost) and bool((cost >= 0).all()),
                     "finite pre-reset forward cost")
        index = env.reward_manager.active_terms.index("forward_error")
        weighted = env.reward_manager._step_reward[:, index]
        base.require(base.finite_tree(weighted) and bool((weighted <= 0).all())
                     and torch.allclose(weighted, cost*self.forward_weight, atol=1e-6, rtol=1e-5),
                     "forward reward matches pre-reset cost")
        self.forward_costs.append(float(cost.mean()))
        self.forward_rewards.append(float(weighted.mean()))
        env._microduck_forward_cost = None

    def update(self, env):
        row = self.after_update(env)
        base.require(len(self.forward_costs) == len(self.forward_rewards) == 24,
                     "complete forward update telemetry")
        row.update(forward_weight_min=self.forward_weight, forward_weight_max=self.forward_weight,
                   forward_cost_mean=sum(self.forward_costs)/24,
                   weighted_forward_reward_mean=sum(self.forward_rewards)/24)
        self.forward_costs.clear(); self.forward_rewards.clear()
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
