"""Second matched training seed; no objective, parent or evaluator changes."""
from __future__ import annotations

import sys

from mjlab_microduck import integrated_motor_range_yaw_only as original
from mjlab_microduck import integrated_motor_refinement as motor

base = original.base
# Retain the same verifier's safety path, rather than adding an alias layer.
previous = original.previous
PROTOCOL = "integrated-motor-range-yaw-only-replication-v1"
REFERENCE_SOURCE = "d71c30285ff3acbdfee12615548eaf2ea1a10975"
REFERENCE_TRAINING_SEED, TRAINING_SEED = 1447, 1481
ARMS = original.ARMS
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1471),
                                             ("benchmark", 256, 10, 1477))
         for arm in ARMS}
MODES.update(control=(256, 1000, TRAINING_SEED), yawonly=(256, 1000, TRAINING_SEED))
CAP_SECONDS = original.CAP_SECONDS
PARENT, PARENT_SHA = original.PARENT, original.PARENT_SHA
PARENT_NORMALIZER_COUNT, PARENT_ADAM_STEP = original.PARENT_NORMALIZER_COUNT, original.PARENT_ADAM_STEP
TERM, VARIANCE, WEIGHT = original.TERM, original.VARIANCE, original.WEIGHT
MotorObserver = original.MotorObserver
warm_load, evaluate_case, decision = original.warm_load, original.evaluate_case, original.decision
yaw_only = original.yaw_only


def prepare_config(mode):
    base.require(mode in MODES, "declared yaw-only replication mode")
    cfg, agent = original.prepare_config(mode)
    worlds, updates, seed = MODES[mode]
    base.require(cfg.scene.num_envs == cfg.scene.terrain.num_envs == worlds
                 and agent.max_iterations == updates, "identical replication budget")
    cfg.seed = agent.seed = seed
    agent.experiment_name = PROTOCOL
    return cfg, agent


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
