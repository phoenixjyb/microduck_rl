from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_velocity_hostile_env_cfg import (
    MicroduckHostileRlCfg,
    make_microduck_velocity_hostile_env_cfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant
from .run import make_run_variant, MicroduckRunRlCfg
from .athletics_velocity_env_cfg import (
    make_microduck_velocity_env_cfg as make_athletics_velocity_env_cfg,
)
from .motor_aware import make_motor_aware_run_variant, MicroduckMotorAwareRunRlCfg
from .obstacle_avoidance import (
    MicroduckObstacleAssistedRlCfg,
    MicroduckObstacleAssistedOutcomeRlCfg,
    MicroduckObstacleAssistedPhaseSpeedRlCfg,
    MicroduckObstacleAvoidanceRlCfg,
    make_obstacle_assisted_variant,
    make_obstacle_assisted_outcome_variant,
    make_obstacle_assisted_phase_speed_variant,
    make_obstacle_avoidance_variant,
)
from .sprung import SWEEP_ARMS, make_sprung_variant, sprung_rl_cfg, ARM_TASK_SUFFIX
from .hop import (
    HOP_ARMS,
    HOP_ARM_SUFFIX,
    h1p_rl_cfg,
    h1s_rl_cfg,
    h1t_rl_cfg,
    hop_rl_cfg,
    make_h1p_variant,
    make_h1s_variant,
    make_h1t_variant,
    make_hop_variant,
)
from mjlab_microduck.robot.sprung_foot import H_ADD, PAD_MASS, TRAVEL

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Run task — rigid running baseline (Phase 1 of the sprung-leg campaign).
# Control for the later sprung comparison; see
# docs/superpowers/specs/2026-08-17-sprung-running-design.md
register_mjlab_task(
    task_id="Mjlab-Run-Flat-MicroDuck",
    env_cfg=make_run_variant(make_athletics_velocity_env_cfg()),
    play_env_cfg=make_run_variant(make_athletics_velocity_env_cfg(play=True)),
    rl_cfg=MicroduckRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Run task registered: Mjlab-Run-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Run-Rough-MicroDuck",
    env_cfg=make_run_variant(make_athletics_velocity_env_cfg(rough=True)),
    play_env_cfg=make_run_variant(
        make_athletics_velocity_env_cfg(play=True, rough=True)
    ),
    rl_cfg=MicroduckRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Run task registered: Mjlab-Run-Rough-MicroDuck")

# Motor-aware Stage 2 fine-tune.  Observation/action dimensions are unchanged
# from Run, so the retained Stage 1 checkpoint can be resumed directly.
register_mjlab_task(
    task_id="Mjlab-Run-MotorAware-Flat-MicroDuck",
    env_cfg=make_motor_aware_run_variant(
        make_run_variant(make_athletics_velocity_env_cfg())
    ),
    play_env_cfg=make_motor_aware_run_variant(
        make_run_variant(make_athletics_velocity_env_cfg(play=True))
    ),
    rl_cfg=MicroduckMotorAwareRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Motor-aware Run task registered: Mjlab-Run-MotorAware-Flat-MicroDuck")

# Single-box obstacle curriculum. The actor is 7D wider than Stage 2; training
# remains gated on a reviewed first-layer warm-start migration.
register_mjlab_task(
    task_id="Mjlab-Run-Obstacle-Flat-MicroDuck",
    env_cfg=make_obstacle_avoidance_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg())
        )
    ),
    play_env_cfg=make_obstacle_avoidance_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg(play=True))
        ),
        play=True,
    ),
    rl_cfg=MicroduckObstacleAvoidanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Obstacle Run task registered: Mjlab-Run-Obstacle-Flat-MicroDuck")

# OA0 scaffold keeps the O1 actor contract but supplies a signed lateral hint,
# fixed 0.30 m/s command, route-return success, and bounded attempt horizon.
register_mjlab_task(
    task_id="Mjlab-Run-Obstacle-Assisted-Flat-MicroDuck",
    env_cfg=make_obstacle_assisted_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg())
        )
    ),
    play_env_cfg=make_obstacle_assisted_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg(play=True))
        ),
        play=True,
    ),
    rl_cfg=MicroduckObstacleAssistedRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print(
    "✓ Assisted obstacle Run task registered: "
    "Mjlab-Run-Obstacle-Assisted-Flat-MicroDuck"
)

# OA0R changes one axis from OA0: a dt-normalized terminal outcome impulse.
register_mjlab_task(
    task_id="Mjlab-Run-Obstacle-Assisted-Outcome-Flat-MicroDuck",
    env_cfg=make_obstacle_assisted_outcome_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg())
        )
    ),
    play_env_cfg=make_obstacle_assisted_outcome_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg(play=True))
        ),
        play=True,
    ),
    rl_cfg=MicroduckObstacleAssistedOutcomeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print(
    "✓ Outcome-balanced obstacle Run task registered: "
    "Mjlab-Run-Obstacle-Assisted-Outcome-Flat-MicroDuck"
)

# OA0P preserves the OA0R command and safety contract while suspending normal
# linear-speed shaping only inside the obstacle interaction zone.
register_mjlab_task(
    task_id="Mjlab-Run-Obstacle-Assisted-PhaseSpeed-Flat-MicroDuck",
    env_cfg=make_obstacle_assisted_phase_speed_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg())
        )
    ),
    play_env_cfg=make_obstacle_assisted_phase_speed_variant(
        make_motor_aware_run_variant(
            make_run_variant(make_athletics_velocity_env_cfg(play=True))
        ),
        play=True,
    ),
    rl_cfg=MicroduckObstacleAssistedPhaseSpeedRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print(
    "✓ Phase-aware-speed obstacle Run task registered: "
    "Mjlab-Run-Obstacle-Assisted-PhaseSpeed-Flat-MicroDuck"
)

# Sprung-foot stiffness sweep — Phase 2. See
# docs/superpowers/specs/2026-08-20-sprung-foot-design.md
for _label, _k, _travel, _pad_mass in SWEEP_ARMS:
    _tid = f"Mjlab-Run-Flat-Sprung-{ARM_TASK_SUFFIX[_label]}-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=make_sprung_variant(
            make_run_variant(make_athletics_velocity_env_cfg()),
            stiffness=_k,
            travel=_travel,
            pad_mass=_pad_mass,
        ),
        play_env_cfg=make_sprung_variant(
            make_run_variant(make_athletics_velocity_env_cfg(play=True)),
            stiffness=_k,
            travel=_travel,
            pad_mass=_pad_mass,
        ),
        rl_cfg=sprung_rl_cfg(_label),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Sprung task registered: {_tid}")

# Periodic hop on the sprung foot — Phase 4. See
# docs/superpowers/specs/2026-08-24-sprung-hop-design.md
#
# NOTE: stiffness=_k is passed into make_hop_variant (not just
# make_sprung_variant) for every arm below. make_hop_variant registers
# hop_energy_monitor with a stiffness used to compute stored spring energy,
# defaulting to 3900.0 -- left unpassed, the k2500 arm's
# Metrics/hop_spring_energy_* would read 56% high, and the spec requires
# reading the spring instruments before any hop-height number.
#
# NOTE 2: make_hop_variant NO LONGER TAKES h_add, and the two-call sync problem
# this note used to describe is gone with it. It used to need h_add to shift an
# ABSOLUTE hop-height target, which had to stay in step with the h_add
# make_sprung_variant uses to shift the com_height_target band -- overriding one
# and not the other desynchronised them silently, with no error. Both hop height
# rewards now measure RISE ABOVE TAKEOFF HEIGHT, which is invariant to how tall
# the robot stands, so h_add has exactly one consumer again: the CoM band, owned
# by make_sprung_variant. It is still passed explicitly there for visibility.
for _label, _k, _travel, _pad in HOP_ARMS:
    _tid = f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[_label]}-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=make_sprung_variant(
            make_hop_variant(make_athletics_velocity_env_cfg(), stiffness=_k),
            stiffness=_k, travel=_travel, pad_mass=_pad, h_add=H_ADD,
        ),
        play_env_cfg=make_sprung_variant(
            make_hop_variant(make_athletics_velocity_env_cfg(play=True), stiffness=_k),
            stiffness=_k, travel=_travel, pad_mass=_pad, h_add=H_ADD,
        ),
        rl_cfg=hop_rl_cfg(_label),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Hop task registered: {_tid}")

# Rejected H1 follow-up: K3900 mechanics and immutable H1 evaluator, with only
# a motor-load cost and progressive 20/30/40 mm training envelope added.
_h1p_tid = "Mjlab-Hop-H1P-Flat-Sprung-K3900-MicroDuck"
register_mjlab_task(
    task_id=_h1p_tid,
    env_cfg=make_sprung_variant(
        make_h1p_variant(
            make_hop_variant(make_athletics_velocity_env_cfg(), stiffness=3900.0)
        ),
        stiffness=3900.0,
        travel=TRAVEL,
        pad_mass=PAD_MASS,
        h_add=H_ADD,
    ),
    play_env_cfg=make_sprung_variant(
        make_h1p_variant(
            make_hop_variant(
                make_athletics_velocity_env_cfg(play=True), stiffness=3900.0
            )
        ),
        stiffness=3900.0,
        travel=TRAVEL,
        pad_mass=PAD_MASS,
        h_add=H_ADD,
    ),
    rl_cfg=h1p_rl_cfg(),
    runner_cls=MicroduckOnPolicyRunner,
)
print(f"✓ H1-P hop task registered: {_h1p_tid}")

# Source-only stability revision after H1-P: reactivate the inherited planar
# stillness reward without changing observations, mechanics, or H1 gates.
_h1s_tid = "Mjlab-Hop-H1S-Flat-Sprung-K3900-MicroDuck"
register_mjlab_task(
    task_id=_h1s_tid,
    env_cfg=make_sprung_variant(
        make_h1s_variant(
            make_h1p_variant(
                make_hop_variant(make_athletics_velocity_env_cfg(), stiffness=3900.0)
            )
        ),
        stiffness=3900.0,
        travel=TRAVEL,
        pad_mass=PAD_MASS,
        h_add=H_ADD,
    ),
    play_env_cfg=make_sprung_variant(
        make_h1s_variant(
            make_h1p_variant(
                make_hop_variant(
                    make_athletics_velocity_env_cfg(play=True), stiffness=3900.0
                )
            )
        ),
        stiffness=3900.0,
        travel=TRAVEL,
        pad_mass=PAD_MASS,
        h_add=H_ADD,
    ),
    rl_cfg=h1s_rl_cfg(),
    runner_cls=MicroduckOnPolicyRunner,
)
print(f"✓ H1-S hop task registered: {_h1s_tid}")

# Bounded planar-translation revision after the H1-S suppression result. Compose
# directly from H1-P so the rejected positive stillness reward is not inherited.
_h1t_tid = "Mjlab-Hop-H1T-Flat-Sprung-K3900-MicroDuck"
register_mjlab_task(
    task_id=_h1t_tid,
    env_cfg=make_sprung_variant(
        make_h1t_variant(
            make_h1p_variant(
                make_hop_variant(make_athletics_velocity_env_cfg(), stiffness=3900.0)
            )
        ),
        stiffness=3900.0,
        travel=TRAVEL,
        pad_mass=PAD_MASS,
        h_add=H_ADD,
    ),
    play_env_cfg=make_sprung_variant(
        make_h1t_variant(
            make_h1p_variant(
                make_hop_variant(
                    make_athletics_velocity_env_cfg(play=True), stiffness=3900.0
                )
            )
        ),
        stiffness=3900.0,
        travel=TRAVEL,
        pad_mass=PAD_MASS,
        h_add=H_ADD,
    ),
    rl_cfg=h1t_rl_cfg(),
    runner_cls=MicroduckOnPolicyRunner,
)
print(f"✓ H1-T hop task registered: {_h1t_tid}")

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Hostile terrain (2026-08-29). Finetune* variants are meant to resume from the shipped walk's
# checkpoint (wandb yr25mna4, model_9999.pt); Scratch* start from random weights.
for _name, _kw in {
    "FinetuneBase": dict(finetune=True, feet=False, track=False),   # A: ladder only, reward unchanged
    "FinetuneFeet": dict(finetune=True, feet=True, track=False),    # B: + foot 3.5 cm, hip roll loose
    "ScratchFeet": dict(finetune=False, feet=True, track=False),    # C: same env as B, from scratch
    "FinetuneFeetTrack": dict(finetune=True, feet=True, track=True),  # D: B + stricter speed tracking
    "FinetuneFeetProgress": dict(finetune=True, feet=True, progress=True),  # E: B + anti-circling (run A lesson)
    "ScratchFeetProgress": dict(finetune=False, feet=True, progress=True),  # C': from scratch, same env as E
    "V2Scratch": dict(finetune=False, v2=True),      # 2026-08-30 menu v2, spawn anywhere, no-progress demotion, strict tracking
    "V2Finetune": dict(finetune=True, v2=True),
}.items():
    register_mjlab_task(
        task_id=f"Mjlab-Hostile-{_name}-MicroDuck",
        env_cfg=make_microduck_velocity_hostile_env_cfg(**_kw),
        play_env_cfg=make_microduck_velocity_hostile_env_cfg(play=True, **_kw),
        rl_cfg=MicroduckHostileRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
