"""One angular-axis inclusion factor; preserve separate stability penalties."""
from __future__ import annotations

import sys
import torch
from mjlab_microduck import integrated_motor_range_yaw as angular
from mjlab_microduck import integrated_motor_refinement as motor

base = angular.base
previous = angular.previous
PROTOCOL = "integrated-motor-range-yaw-only-v1"
ARMS = ("control","yawonly")
MODES = {f"{phase}-{arm}": (worlds,updates,seed)
         for phase,worlds,updates,seed in (("smoke",64,5,1433),("benchmark",256,10,1439))
         for arm in ARMS}
MODES.update(control=(256,1000,1447),yawonly=(256,1000,1447))
CAP_SECONDS = 2100
PARENT,PARENT_SHA = angular.PARENT,angular.PARENT_SHA
PARENT_NORMALIZER_COUNT,PARENT_ADAM_STEP = angular.PARENT_NORMALIZER_COUNT,angular.PARENT_ADAM_STEP
TERM,VARIANCE,WEIGHT = angular.TERM,angular.VARIANCE,angular.WEIGHT
warm_load,evaluate_case,decision = angular.warm_load,angular.evaluate_case,angular.decision


def yaw_only(mode):
    base.require(mode in MODES,"declared yaw-only mode")
    return mode == "yawonly" or mode.endswith("-yawonly")


def prepare_config(mode):
    from mjlab_microduck.tasks import mdp
    treatment = yaw_only(mode)
    cfg,agent = angular.prepare_config("control")
    worlds,updates,seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    if treatment:
        cfg.rewards[TERM].func = mdp.yaw_only_body_angular_tracking
    # Different signals/frames remain unchanged; no assertion of redundancy.
    base.require(cfg.rewards["body_ang_vel"].weight == -.05
                 and cfg.rewards["angular_momentum"].weight == -.02
                 and cfg.rewards["motor_load"].weight == -1.
                 and cfg.rewards["joint_limit"].weight == -4.
                 and cfg.rewards["forward_error"].weight == 0.,"separate stability and safety retained")
    agent.max_iterations,agent.save_interval = updates,250
    agent.experiment_name,agent.run_name = PROTOCOL,mode
    return cfg,agent


class MotorObserver(angular.MotorObserver):
    def __init__(self,mode):
        self.yaw_only_tracking = yaw_only(mode)
        super().__init__("smoke-control" if mode.startswith("smoke-") else "benchmark-control")
        cfg,_ = prepare_config(mode)
        self.expected_func = cfg.rewards[TERM].func
        self.expected_params = dict(cfg.rewards[TERM].params)

    def before(self,env):
        super().before(env)
        base.require(env.reward_manager.get_term_cfg("body_ang_vel").weight == -.05
                     and env.reward_manager.get_term_cfg("angular_momentum").weight == -.02,
                     "fixed live separate angular stability weights")

    def expected_reward(self,yaw,xy):
        if self.yaw_only_tracking:
            return WEIGHT*torch.exp(-yaw/VARIANCE)
        return super().expected_reward(yaw,xy)

    def update(self,env):
        row = super().update(env)
        row.update(angular_aggregation="yaw_only_tracking" if self.yaw_only_tracking else "stock_multiplicative_product",
                   roll_pitch_tracking_included=not self.yaw_only_tracking,
                   roll_pitch_variance=None if self.yaw_only_tracking else VARIANCE,
                   body_ang_vel_weight=env.reward_manager.get_term_cfg("body_ang_vel").weight,
                   angular_momentum_weight=env.reward_manager.get_term_cfg("angular_momentum").weight)
        return row


if __name__ == "__main__": motor.main(sys.modules[__name__])
