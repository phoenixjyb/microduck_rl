"""Yaw-axis precision contrast inside the simpler motor-aware head lesson."""
from __future__ import annotations

import sys
import torch

from mjlab_microduck import integrated_motor_range_head_lesson as head
from mjlab_microduck import integrated_motor_refinement as motor

base, previous = head.base, head.previous
PROTOCOL = "integrated-quiet-head-yaw-focus-v1"
ARMS = ("control", "yawfocus")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke",64,5,1637),("benchmark",256,10,1657))
         for arm in ARMS}
MODES.update(control=(256,1000,1663),yawfocus=(256,1000,1663))
CAP_SECONDS = 2100
PARENT, PARENT_SHA = head.PARENT, head.PARENT_SHA
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = head.PARENT_NORMALIZER_COUNT, head.PARENT_ADAM_STEP
TERM, VARIANCE, WEIGHT = head.TERM, head.VARIANCE, head.WEIGHT
warm_load, decision = head.warm_load, head.decision


def yaw_variance(mode):
    base.require(mode in MODES,"declared quiet-head yaw-focus mode")
    return .1 if mode == "yawfocus" or mode.endswith("-yawfocus") else .5


def reference_mode(mode):
    yaw_variance(mode)
    return "smoke-quiethead" if mode.startswith("smoke-") else "benchmark-quiethead" if mode.startswith("benchmark-") else "quiethead"


def head_ranges(mode):
    return head.head_ranges(reference_mode(mode))


def prepare_config(mode):
    from mjlab_microduck.tasks import mdp
    cfg,agent = head.prepare_config(reference_mode(mode))
    worlds,updates,seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    if yaw_variance(mode) == .1:
        # Reuse the tested primary MDP product; keep coefficient and XY width.
        cfg.rewards[TERM].func = mdp.factorized_body_angular_tracking
        cfg.rewards[TERM].params = dict(command_name="twist",yaw_variance=.1,roll_pitch_variance=.5)
    agent.max_iterations,agent.save_interval = updates,250
    agent.experiment_name,agent.run_name = PROTOCOL,mode
    return cfg,agent


def validate_action_row(row,mode,worlds,update):
    head.validate_action_row(row,reference_mode(mode),worlds,update)


def validate_linear_row(row,mode,worlds,update):
    head.validate_linear_row(row,reference_mode(mode),worlds,update)


def validate_head_row(row,mode,worlds,update):
    head.validate_head_row(row,reference_mode(mode),worlds,update)


def validate_yaw_row(row,mode,worlds,update):
    variance = yaw_variance(mode)
    base.require(type(update) is int and 1 <= update <= MODES[mode][1]
                 and type(worlds) is int and worlds == MODES[mode][0],"declared yaw-focus row budget")
    label = "axis_precision_multiplicative_product" if variance == .1 else "stock_multiplicative_product"
    base.require(row["yaw_variance"] == variance and row["roll_pitch_variance"] == .5
                 and row["angular_tracking_weight"] == 2. and row["angular_aggregation"] == label
                 and row["roll_pitch_tracking_included"] is True
                 and row["angular_reward_sample_count"] == worlds*24
                 and row["negative_reward_sample_count"] == row["negative_reward_fraction"] == 0
                 and base.finite_tree([row[k] for k in ("weighted_angular_reward_mean","yaw_error_sq_mean","roll_pitch_rate_sq_mean")])
                 and 0 <= row["weighted_angular_reward_mean"] <= 2.
                 and row["yaw_error_sq_mean"] >= 0 and row["roll_pitch_rate_sq_mean"] >= 0
                 and row["body_ang_vel_weight"] == -.05 and row["angular_momentum_weight"] == -.02,
                 "actual fixed positive yaw-axis precision contract")


class MotorObserver(head.MotorObserver):
    def __init__(self,mode):
        self.focus_mode = mode
        super().__init__(reference_mode(mode))
        cfg,_ = prepare_config(mode)
        self.expected_func = cfg.rewards[TERM].func
        self.expected_params = dict(cfg.rewards[TERM].params)

    def expected_reward(self,yaw,xy):
        if yaw_variance(self.focus_mode) == .1:
            return 2.*torch.exp(-yaw/.1-xy/.5)
        return super().expected_reward(yaw,xy)

    def update(self,env):
        row = super().update(env)
        variance = yaw_variance(self.focus_mode)
        row.update(yaw_variance=variance,angular_aggregation="axis_precision_multiplicative_product"
                   if variance == .1 else "stock_multiplicative_product")
        validate_yaw_row(row,self.focus_mode,env.num_envs,env.common_step_counter//24)
        return row


def evaluate_case(checkpoint,seed,speed,yaw,health):
    # More read-only telemetry in both arms, not new commands or a relaxed gate.
    return base.evaluate_case(checkpoint,seed,speed,yaw,health,diagnostics=True,yaw_trace=True,gait_trace=True)


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
