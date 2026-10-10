"""Recover command tracking while retaining the measured joint-limit repair.

The parent passed the range/motor subsets only. It is not an accepted policy.
"""

from __future__ import annotations

import sys

from mjlab_microduck import integrated_joint_limit_refinement as repair
from mjlab_microduck import integrated_motor_refinement as motor

base = repair.base
PROTOCOL = "integrated-range-retaining-tracking-continuation-v1"
ARMS = ("control", "tracking")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 911), ("benchmark", 256, 10, 913))
         for arm in ARMS}
MODES.update(control=(256, 1000, 919), tracking=(256, 1000, 919))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/qpos-limit-1689e874-20261011/limit/model_999.pt"
PARENT_SHA = "212a567984014f90054cbd7137e7560d8c76a00496a3ca15958407b9de746e6b"
TRACKING_STAGES = ((0, -.125), (250, -.25), (500, -.5))


def stages(mode):
    if mode.endswith("control"):
        return ((0, 0.),)
    if mode.startswith("smoke"):
        return ((0, -.125), (1, -.25), (2, -.5))
    if mode.startswith("benchmark"):
        return ((0, -.5),)
    return TRACKING_STAGES


def prepare_config(mode):
    cfg, agent = repair.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.rewards["joint_limit"].weight = -4.
    cfg.rewards["command_error"].weight = stages(mode)[0][1]
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    return motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                           parent_iteration=999, parent_steps=24000)


class MotorObserver(repair.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.limit_schedule = ((0, -4.),)
        self.tracking_schedule = stages(mode)

    def before(self, env):
        super().before(env)
        self.tracking_weight = motor.motor_weight(env.common_step_counter, self.tracking_schedule)
        env.reward_manager.get_term_cfg("command_error").weight = self.tracking_weight


# The stricter joint repair gates are retained, not reverted to the old gate.
evaluate_case = repair.evaluate_case
decision = repair.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
