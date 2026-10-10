# Integrated walking and moving-turn foundation

Owner approved the next bounded training round after the source integration.
This is exploratory simulation learning, not simulator qualification or robot
acceptance. The historical motor-aware parent remains in its original plant
and is not loaded into this changed upstream plant. No old/new-plant causal
comparison or preservation of an old learned hop skill is claimed.

## Predeclaration, before GPU execution

Entry point: `python -m mjlab_microduck.integrated_turn_foundation
{smoke,benchmark,foundation,evaluate} --source <exact-clean-commit>
--run-id <unique-id>`. Commit and push this declaration and its tested code
before execution. Use the separate native integration worktree and retained
frozen research venv with explicit integration-source PYTHONPATH. No dependency,
driver, framework or runtime deployment changes.

Fresh PPO on `Mjlab-Velocity-Flat-MicroDuck`, 61D actor/14 motor actions, 24
control steps/update. Keep the upstream rewards, BAM, DR, observation noise,
posture/CoM/standing/smoothing curricula and learner settings. Change only twist
sampling: forward0.20–0.40m/s, lateral0, yaw±0.50rad/s, resample3–5s, explicit
straight bucket0.25, no turn-in-place, world-frame or heading-control override,
no initial velocity injection. Retain upstream standing probability/schedule,
which supplies exact idle commands. Straight/standing fractions are not promised
to be exclusive: audit the commands actually consumed by the actor and physics.
Head/body slots retain upstream sampling; no yaw-reward fix or command remap.

Sequential independent fresh starts, never from each other's checkpoints:

| Phase | Worlds | Seed | PPO updates | External hard cap |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 64 | 821 | 5 | 180s |
| Timing benchmark | 256 | 823 | 10 | 180s |
| Foundation | 256 | 827 | 1500 | 1800s |

Foundation requires both same-source prior successes and measured benchmark
wall time ×150×1.5 <1740s. Refuse the launch otherwise; no implicit budget
extension. Save initial weights, every250th iteration and final1499 atomically
with file and directory fsync. This is9,216,000 foundation transitions, not a
guarantee that a small-world budget learns a usable gait. A curriculum stage at
1500×24 has no subsequent consolidation in this run; do not extrapolate its
endpoint to the original upstream full recipe.

Retain the existing owned cooperative GPU0 lock. Allow only exact observed
GroundingDINO PID1592/binary and the current Duck process on native GPU UUID
`GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`. Require protected AI Mission services
inactive in both user/system scopes. Aggregate usage≤6GiB, free≥8GiB, temperature
<75C; Torch allocator cap20% does not cap Warp. Check host before/after each
phase and every25 PPO updates. The user services have CPU200%, RAM8GiB,
TasksMax64, Nice10 and KillMode=control-group. Preserve unrelated workloads.

Require finite actor/critic inputs, actions and manager rewards every control
step, exact actor/manager/physics twist equality, finite optimizer losses each
update, finite models/Adam every25 updates and at durable closeout, and zero
NaN terminations. Reward-manager sanitization is not raw reward-term proof.
Log actual idle/straight/negative-yaw/positive-yaw coverage and per-update falls, torque p99,
soft-limit exposure, rated-speed exceedance, absolute mechanical power and
squared-utilization thermal proxy using the retained pre-reset motor stream.
This is post-decimation sampling with its documented force lag, not substep
peak or hardware temperature certification. Fresh-policy gross abort guards:
torque utilization p99>1.5 or rated-speed exceedance fraction>0.10. Random-policy
falls do not abort early learning; they remain explicit and fail held-out gates.
These exploratory abort bounds do not relax the acceptance envelope below.

## Fixed held-out evaluations and deterministic next-stage decision

After complete training, evaluate checkpoints750 and1499 sequentially under
the same new plant: seeds839,853,857; eight worlds; first240 steps/case; exclude
only the first60 startup steps from tracking errors, never from terminals or
motor gates. Cases: idle0/0, straight0.30/0, moving0.20/-0.20,
moving0.20/+0.20 (m/s,rad/s). These are signed body-yaw cases; no unverified
physical left/right naming is imposed. All commands are pinned with override buckets
disabled, posture neutral, push event removed and curricula cleared. Use the
official runner/normalizer and pre-reset motor sample. Stop the entire case on
first terminal; reset episodes cannot launder stability. That is24 fixed cases
and a maximum46,080 evaluation transitions.

Only final1499 decides readiness, with no best-checkpoint fishing. Every final
case must finish240 steps without terminal; every world's speed MAE≤0.10m/s
(idle≤0.03m/s), yaw MAE≤0.10rad/s, case torque utilization p99≤0.60 and zero
rated-speed exceedance. A stationary duck therefore cannot pass the moving-turn
tracking gate. Midpoint750 is descriptive same-plant progress evidence, not a
causal command contrast and not a fallback selection. Report failures and exact
checkpoint/results hashes. Readiness authorizes planning the next experiment,
not an automatic job, obstacle/H2 admission, video or physical motion.

Next, only if this foundation passes: predeclare a matched control versus
mirrored-turn refinement from this exact new-plant parent, with restoration and
curriculum state explicitly fixed. Keep old research/evaluation gates unchanged.
No raw perception, obstacle geometry, hop/football revision, Hub/W&B downloads,
policy publication or protected-service stop/restart is part of this round.

## Source checks before launch

Focused Mac checks:52 passed (new foundation, integration, motor-step stream
and evaluation helpers); one existing actuator-selector warning. Syntax and
whitespace checks passed. Independent read-only review confirmed actor-command
ordering, metric-before-reset timing and checkpoint indices. Coverage names use
yaw signs instead of assuming hardware left/right. Hard runtime bounds are
external systemd user-service properties, to be checked live at launch.
