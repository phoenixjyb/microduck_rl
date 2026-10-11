"""Matched easier-to-harder yaw sampling; no additional reward pressure.

This tests a whole command curriculum under the unchanged held-out gate.
Neither the parent nor this recipe authorizes skill or physical promotion.
"""

from __future__ import annotations

from collections import Counter
import sys

from mjlab_microduck import integrated_forward_tracking_refinement as forward
from mjlab_microduck import integrated_motor_refinement as motor

base = forward.base
PROTOCOL = "integrated-gentle-turn-range-curriculum-v1"
ARMS = ("control", "curriculum")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 947), ("benchmark", 256, 10, 949))
         for arm in ARMS}
MODES.update(control=(256, 1000, 953), curriculum=(256, 1000, 953))
CAP_SECONDS = motor.CAP_SECONDS
PARENT = base.ROOT / "artifacts/training/forward-tracking-a09a9c0a-20261011/control/model_999.pt"
PARENT_SHA = "604700b0d50ee5f11558cf826a1c50fc4283bae801a80f7d6a8f90dff1a14132"
YAW_STAGES = ((0, .05), (250, .10), (500, .20))


def stages(mode):
    if mode.endswith("control"):
        return ((0, .50),)
    if mode.startswith("smoke"):
        return ((0, .05), (1, .10), (2, .20))
    if mode.startswith("benchmark"):
        return ((0, .20),)
    return YAW_STAGES


def prepare_config(mode):
    cfg, agent = forward.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    bound = stages(mode)[0][1]
    cfg.commands["twist"].ranges.ang_vel_z = (-bound, bound)
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


def warm_load(runner, env):
    return motor.warm_load(runner, env, PARENT, parent_sha=PARENT_SHA,
                           parent_iteration=999, parent_steps=24000)


class MotorObserver(forward.MotorObserver):
    def __init__(self, mode):
        super().__init__("control")
        self.yaw_schedule = stages(mode)
        self.previous_bound = self.yaw_schedule[0][1]
        self.yaw_bounds, self.command_maxima = [], []
        self.cumulative_counts = Counter()

    def before(self, env):
        super().before(env)
        term = env.command_manager.get_term("twist")
        bound = motor.motor_weight(env.common_step_counter, self.yaw_schedule)
        base.require(term.cfg.ranges.ang_vel_z == (-self.previous_bound, self.previous_bound)
                     and self.previous_bound <= bound <= .5, "known monotonic live yaw range")
        # Change only the live sampling range. Existing commands stay inside
        # an expanding bound; do not force-resample or rewrite actor input.
        term.cfg.ranges.ang_vel_z = (-bound, bound)
        self.previous_bound = bound
        command = env.command_manager.get_command("twist")
        base.validate_commands(command)
        maximum = float(command[:, 2].abs().max())
        base.require(maximum <= bound+1e-6, "sampled yaw inside declared live range")
        self.yaw_bounds.append(bound)
        self.command_maxima.append(maximum)
        self.cumulative_counts[str(bound)] += 1

    def update(self, env):
        row = super().update(env)
        base.require(len(self.yaw_bounds) == len(self.command_maxima) == 24, "full yaw stage update")
        row.update(yaw_range_min=min(self.yaw_bounds), yaw_range_max=max(self.yaw_bounds),
                   observed_yaw_command_abs_max=max(self.command_maxima),
                   yaw_range_step_counts=dict(Counter(map(str, self.yaw_bounds))),
                   yaw_range_cumulative_step_counts=dict(self.cumulative_counts))
        self.yaw_bounds.clear(); self.command_maxima.clear()
        return row


evaluate_case = forward.evaluate_case
decision = forward.decision


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
