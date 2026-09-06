# Recovery-control and streaming motor primitives: source chunk 1

Date: 2026-09-06. Parent: `c17643732e7a80148314d2cf7cbfa8ef14050219`.
Scope: opt-in source implementation and CPU validation only. No new simulation
campaign, training job, model, dataset or video is authorized by this change.

## Motivation and hypothesis, not a cleared policy

The [seed-373 audit](2026-09-06-motor-audit-smoke.md) showed four clean first
attempts, yet recovery had higher pooled torque utilization and right-hip-pitch
load than the whole-run statistic suggested. This is a single-case observation,
not proof that command acceleration caused the load or U4's earlier collision.
U4/U3/O3a rejection, seed-359 recorder failure and unresolved seed-367
repeatability remain unchanged. The next A/B should use accepted HC4-R2/HC4-LH
specialist baselines within their retained envelopes; U4 is not promoted.

Hypothesis: limiting **positive forward-command acceleration during recovery**
may reduce load without harming avoidance or timely speed recovery. Merely
slowing the duck indefinitely is not success. We do not retrain the gait to
test this hypothesis, expand the actor observation, or add raw perception.

## Opt-in recovery acceleration cap

`recovery_control.py` defines `RecoveryAccelerationCfg`; the prototype rate is
**0.20 m/s²**. This is a source-test starting point, not a tuned or physically
calibrated motor limit. Existing `ObstacleTeacherCfg` defaults and serialized
legacy configuration are unchanged. The default recovery option is **None**.

The existing execution layer first applies its speed/yaw absolute and slew
limits. Only when the new option is enabled, it additionally caps recovery
speed at `previous_forward_command + acceleration * command_update_dt`.
The cap cannot relax any existing command limit. Approach and interaction,
yaw commands, and ordinary braking are unchanged. Invalid-observation immediate
stop outside positively observed recovery remains unchanged. Raw nonfinite
commands and invalid timing/state fail before command history is mutated.

Timing is the actual supervisor update period: five control steps times
`env.step_dt`, **0.1 s** in the retained setup, not the 0.005 s physics step.
This introduces no extra state: the existing previous-command tensor and phase
are sufficient, so clone/reset behavior remains intact. Command dimensions,
17-dimensional supervisor observation and frozen locomotion inputs are unchanged.

For a held target, the command converges rather than asymptotically remaining
slow: 0.3 to 0.5 m/s takes 1 s at the prototype cap and current legacy limits.
CPU tests check this at 0.05, 0.1 and 0.2 s command periods. This is a
**command-level guarantee under a held target**, not a guarantee of measured
robot acceleration or physical target-speed recovery. A learned supervisor can
still request too little speed. No deadline forces an acceleration jump.
The current nominal-speed targets and approach/recovery evaluation gates stay
unchanged; the next experiment must predeclare a measured recovery-time window
and assess it without relaxing existing criteria. There is no new reward mask
that exempts recovery from speed tracking, and no interaction-speed target.

The diagnostic CLI exposes `--recovery-acceleration-mps2`. It requires
`--first-attempt-only --motor-measurement-audit`, so the existing audit bounds
(64 environments, 1000 control steps, 12 cases) also apply. Dataset collection,
trajectory recording and sensor perturbations are refused for this mode. The
same configuration reaches both teacher and learned-controller execution paths.
Each case retains its cap configuration and actual update period. Reports use
the distinct `RecoveryAcceleration-diagnostic-rollout` stage, retain the source
controller stage, and set policy/motion admission false. Existing U4 gates
reject that changed execution identity even if numerical metrics appear good.
Legacy motor measurements, thresholds and first-attempt accounting stay intact.

## Constant-retention pre-reset motor stream

`motor_step_stream.py` provides `MotorStepStream` and an opt-in zero-valued
logging metric. Production callers construct it via `from_robot`, reusing the
audited unique named direct-unit-gear hinge mapping. Up to 256 environments
and 64 motor columns are supported. There is **one pending control-step
snapshot**, not a growing trajectory list or the first-attempt-only collector.

Required lifecycle for an auto-reset environment:

1. Install the metric before environment construction, then attach the stream
   as `env._microduck_motor_step_stream` after obtaining the robot layout.
2. Before every low-level control step, call `begin(sequential_step, phase)`.
3. The existing post-decimation metrics callback clones motor force, mapped
   velocity and terminal identity before reset/final forward.
4. After step returns, call `consume(terminated | truncated)`. The returned
   sample still belongs to the original episode; only then does the stream
   advance terminal environments' generation counters for the next step.

Every environment participates again on the next step, including previously
terminal ones. Missing/duplicate/unconsumed samples, skipped steps, invalid
phase/layout or mismatched terminal identity fail closed. Snapshots, phase and
episode identities are detached from mutable simulator/reset buffers. Mutating
a returned sample cannot corrupt the next sample's generation. Manual resets
are not inferred; they require a new stream at a clean boundary. Installation
rejects non-auto-reset configuration.

For each joint, the detached float64 cost primitive is
`u² + gain * max(u - soft_limit, 0)²`, where `u = abs(force)/stall_reference`.
Defaults reuse the motor-aware model references: 0.60 Nm, soft fraction 0.70,
gain 4.0. Returned fields include the per-joint vector, its per-environment
mean, joint names, phase, episode generation and terminal identity. No phase
is silently discarded, including terminal steps. No reward sign/weight is
applied here. NaN/Infinity or overflow in the derived cost raises, rather than
becoming a zero reward or advancing episode accounting.

As with the audit, force is last-physics-substep derived force with one
integration lag; velocity is integrated state at capture. These are not all-
substep peaks, synchronized mechanical power, measured temperature or a
continuous-duty hardware rating. Old results cannot be rescored into acceptance
by switching to this stream.

Important integration boundary: mjlab's ordinary reward manager has already
run before this metrics hook. The stream is for a future **external supervisor
reward computed after the low-level step**; it is not a new mjlab reward term.
This source chunk does not modify the PPO trainer, optimizer, checkpoint,
reward weights or policy parameters. A later trainer adapter must consume every
substep and correctly aggregate terminal contributions within each supervisor
action window. No training-integrated or GPU non-interference claim is made.

## Validation and next boundary

CPU tests exercise the cap, held-target convergence, unchanged non-recovery/
braking/yaw behavior, invalid-input refusal, phase transitions, state reset/
clone, default report compatibility, opt-in provenance and old-gate refusal.
Stream tests execute the **actual installed mjlab step method** with synthetic
physics/reset managers over repeated autoresets; terminal snapshots survive
both zero and large replacement forces. Mock observations/actions/rewards and
torch RNG remain unchanged. Tests also cover real robot mapping, finite costs,
episode identity, constant retention over 300 steps and lifecycle failures.
This is CPU evidence, not a physical or real MuJoCo/Warp rollout comparison.

Precommit validation: **448 focused CPU tests passed locally** with CUDA hidden
(7.70 s), including the unchanged motor-audit, repeatability, recorder,
supervisor-PPO and U4/U3/U1 gate suites. Two instances of the existing real-spec
actuator/site-pattern warning were emitted; both tests independently verified
the 14 named unit-gear hinge mappings. Diff whitespace checks passed.

Next authorized-design boundary: predeclare a small frozen-controller A/B with
baseline repeats, exact seeds/cases, recovery-time and motor criteria, runtime
budget, failure stops and artifact retention. Check 100.100 idle and protected
services before any separately approved GPU run. Only evidence supporting that
experiment should lead to a separate recovery-speed PPO adapter/pilot. Preserve
both protected services and unrelated workloads; do not touch 100.98.
