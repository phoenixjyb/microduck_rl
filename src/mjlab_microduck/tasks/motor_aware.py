"""Motor-aware fine-tuning variant for the rigid Run policy.

This stage resumes the Stage 1 locomotion checkpoint with the same actor and
observation contract, freezes already-completed auxiliary curricula, narrows
the forward command ladder to the measured useful range, and gradually adds a
motor-load cost.  It is a prerequisite for hop/obstacle work, not a physical
safety certification.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.run import (
    MicroduckRunRlCfg,
    XL330_M288_RATED_STALL_TORQUE_NM_6V,
)

MOTOR_SOFT_LIMIT_FRACTION = 0.70
MOTOR_OVER_LIMIT_GAIN = 4.0

# Fine-tuning steps are environment steps (iteration * 24).
MOTOR_AWARE_VELOCITY_STAGES = [
    {"step": 0, "lin_vel_range": 0.50, "ang_vel_range": 1.0},
    {"step": 750 * 24, "lin_vel_range": 0.65, "ang_vel_range": 1.0},
    {"step": 1500 * 24, "lin_vel_range": 0.80, "ang_vel_range": 1.0},
]

MOTOR_COST_WEIGHT_STAGES = [
    {"step": 0, "weight": -0.25},
    {"step": 250 * 24, "weight": -0.75},
    {"step": 750 * 24, "weight": -1.25},
    {"step": 1500 * 24, "weight": -2.00},
]


def _freeze_other_step_curricula(
    cfg: ManagerBasedRlEnvCfg, excluded: frozenset[str]
) -> None:
    """Collapse completed stage lists to their final value at step zero."""
    for name, term in cfg.curriculum.items():
        if name in excluded:
            continue
        for key, value in list(term.params.items()):
            if (
                isinstance(value, list)
                and value
                and isinstance(value[0], dict)
                and "step" in value[0]
            ):
                final = dict(value[-1])
                final["step"] = 0
                term.params[key] = [final]


def make_motor_aware_run_variant(
    cfg: ManagerBasedRlEnvCfg,
) -> ManagerBasedRlEnvCfg:
    """Add the bounded motor-aware fine-tuning objective to a Run config."""
    _freeze_other_step_curricula(cfg, frozenset({"velocity_command_ranges"}))

    velocity = cfg.curriculum["velocity_command_ranges"].params
    velocity["velocity_stages"] = [
        dict(stage) for stage in MOTOR_AWARE_VELOCITY_STAGES
    ]
    velocity["forward_only"] = True
    velocity["update_lin_vel_y"] = False

    cfg.rewards["motor_torque_load"] = RewardTermCfg(
        func=microduck_mdp.motor_torque_load_cost,
        weight=MOTOR_COST_WEIGHT_STAGES[0]["weight"],
        params={
            "rated_stall_torque_nm": XL330_M288_RATED_STALL_TORQUE_NM_6V,
            "soft_limit_fraction": MOTOR_SOFT_LIMIT_FRACTION,
            "over_limit_gain": MOTOR_OVER_LIMIT_GAIN,
        },
    )
    cfg.curriculum["motor_torque_load_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "motor_torque_load",
            "weight_stages": [dict(stage) for stage in MOTOR_COST_WEIGHT_STAGES],
        },
    )
    return cfg


MicroduckMotorAwareRunRlCfg = replace(
    MicroduckRunRlCfg,
    actor=deepcopy(MicroduckRunRlCfg.actor),
    critic=deepcopy(MicroduckRunRlCfg.critic),
    algorithm=deepcopy(MicroduckRunRlCfg.algorithm),
    experiment_name="run_motor_aware",
    run_name="stage2_motor_envelope",
)
