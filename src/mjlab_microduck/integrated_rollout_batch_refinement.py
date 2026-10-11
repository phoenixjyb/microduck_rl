"""Matched fixed-update rollout-batch comparison on the unchanged recipe."""

import sys

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor

base = previous.base
PROTOCOL = "integrated-rollout-batch-continuation-v1"
ARMS = ("control", "batch")
TRAINING_WORLDS = {"control": 256, "batch": 1024}
MODES = {
    "smoke-control": (64, 5, 1073),
    "smoke-batch": (64, 5, 1073),
    "benchmark-control": (256, 10, 1079),
    "benchmark-batch": (1024, 10, 1079),
    "control": (256, 1000, 1087),
    "batch": (1024, 1000, 1087),
}
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {"control": 2100, "batch": 4500}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE
EXPOSURE_BINS = previous.EXPOSURE_BINS


def prepare_config(mode):
    """Use the established duration-control recipe; vary only declared fields."""
    base.require(type(mode) is str and mode in MODES, "known rollout-batch mode")
    worlds, updates, seed = MODES[mode]
    arm = mode.split("-")[-1]
    if mode.startswith("smoke-"):
        base.require(worlds == 64, "paired smoke uses 64 worlds")
    elif mode.startswith("benchmark-"):
        base.require(worlds == TRAINING_WORLDS[arm], "benchmark worlds match declared arm")
    else:
        base.require(worlds == TRAINING_WORLDS[arm], "full-run world count matches declared arm")

    cfg, agent = previous.prepare_config("control")
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
evaluate_case = previous.evaluate_case
decision = previous.decision


class MotorObserver(previous.MotorObserver):
    """Reuse identical duration telemetry while accepting only this recipe's modes."""

    def __init__(self, mode):
        base.require(type(mode) is str and mode in MODES, "known rollout-batch mode")
        super().__init__("control")


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
