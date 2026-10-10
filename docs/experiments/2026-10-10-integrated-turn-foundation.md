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
| Foundation | 256 | 827 | 1500 | 2700s |

Foundation requires both same-source prior successes and measured benchmark
wall time ×150×1.5 <2640s. Refuse the launch otherwise; no implicit budget
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

## Timing revision before foundation execution

Initial source `8154e54e3de06bd6df7e3dbe6e218fb1af4e68d4` passed52 native CPU
tests in6.18s. Its64-world smoke completed5 updates in5.544s; its256-world
benchmark completed10 in9.703s. Both had finite final model/Adam state and
nonzero actual idle, straight and both signed-yaw coverage. Benchmark projection
9.703×150×1.5≈2183s exceeds the originally declared1740s admission ceiling.
No foundation ran under that declaration. Before any foundation launch, revise
only the external cap from1800s to2700s (admission ceiling2640s); retain all
1500 updates, seeds, rewards, curricula and held-out gates. Commit/push the
revision and rerun the same-source smoke/benchmark checks. Retain the initial
timing evidence, rather than overwriting it.

## Retained launch, training/evaluation still pending

Executed source: `791f27c4cbd19b93ca04778890664cda55afc772`, clean detached
native integration worktree. Native focused checks again passed52 tests in6.09s.
The revised-source smoke and benchmark passed; benchmark wall time9.698s gives
a conservative2182s projection, within the revised2640s admission ceiling.
Both official runner-generated ONNX exports passed CPU-only ONNX Runtime
shape61→14 and finite-inference checks. No environment was synced or changed.

Retained run: `artifacts/training/turn-foundation-791f27c4-20261010` in
`/home/converge/work/microduck_rl-upstream-20261010`. The user service
`microduck-turn-791f27c4-foundation-eval.service` started at2026-10-10
11:27:37 Asia/Shanghai. Live properties confirm RAM8GiB, CPU200%, TasksMax64,
Nice10, KillMode=control-group and an overall3000s cap. Its explicit GNU
timeout children independently enforce2700s foundation and180s evaluation
(15s kill grace). Evaluation runs sequentially only after successful complete
training; no further training or publication follows. RemainAfterExit retains
the terminal service evidence; active/exited later is not a running learner.

Launch sample:57 updates, finite losses, no NaN terminations; GPU1710MiB/57C,
Duck746MiB alongside unchanged DINO1592/946MiB. At240 updates/5760 control
steps (1,474,560 transitions), losses remained finite, no NaN terminations,
106 falls in the last6144 transitions, torque utilization p99≈1.0675 and
soft-limit exposure≈0.2834. These fresh stochastic rollouts do not meet the
held-out motor/stability gates. Do not infer learned competence from an
increasing aggregate reward. All four protected service scopes were inactive
at the sampled checks. Existing baseline/checkpoints and unrelated workloads
were untouched; no physical motion is authorized.

Only small preflight/launch receipts are mirrored on Mac under
`artifacts/evaluations/turn-foundation-791f27c4-20261010`; native/Mac hashes match:

| Receipt | SHA256 |
| --- | --- |
| Smoke result | `3f884f995838b98d1ca19e2dce36f067259a954969b691c7be6ed568c38e8b37` |
| Benchmark result | `e60a38d70b75f4ebe2251cd5aa8c865c771a0945d39fc27cdb03c1937227345d` |
| Foundation launch | `76b593751e9cffcb76cde2aa41429d0a98ffe18679539edf6c2702e9dc3f8830` |

Operational note: an initial launcher used `--wait` on a retained CPU unit;
CPU tests finished but the wait did not advance to its next phase. Read-only
service inspection established MainPID0/active-exited and no GPU job. Only
the owner's local SSH wait process was terminated; the finished unit stayed
retained. Subsequent phases were launched individually with successful
terminal-state checks. No unrelated process or service was stopped.

Handoff: inspect this exact retained service, `foundation/result.json` and
`evaluate/result.json` before declaring completion or advancing. The only
readiness decision is final1499 under the predeclared matrix; training and
numerical acceptance are pending at this launch record.

Durable checkpoint250 was independently loaded on CPU while training continued:
SHA256 `f16e4396389029f2b6b7ca0a53b49989e7ff74a32df486abddb1742a7537a037`.
Full saved model/Adam payload was finite; all eight actor and eight critic MLP
tensors differed from the fresh initial checkpoint. All17 Adam entries had
advanced5020 steps (251 updates×5 epochs×4 minibatches), and saved common step
was6024. This establishes actual learning and durable progress, not a usable
policy. At the next inspected sample,413 updates were complete, still with
finite losses and zero NaN terminations. Final training/evaluation remained
pending; no checkpoint selection or follow-on task was changed.

## Completed result, retained before continuation

The retained service finished successfully (MainPID0, Resultsuccess,
ExecMainStatus0, active/exited). All1500 updates completed in968.202s,
36,000 control steps and9,216,000 world transitions. Final model/Adam state
was finite; NaN terminations remainedzero. Final checkpoint1499 SHA256:
`b4c51e072b4b21c865ab4108c477ed4c27f26a129577e7ad17a14b1e6250dd90`.
Training result SHA256:
`516d841541670270fe69e434d9a0d56c1a81d0d60786fe7b97077594bb0cfd0b`.
Evaluation result SHA256:
`defc2f0b56a0e044fd779cb3f8c3788338dfa559c030e42d93c1b692ea4f34e9`.
Small receipts were copied to Mac and their hashes verified; checkpoint bytes
stay on the native host.

All24 checkpoint/case/seed rows finished240 steps without terminals. Final1499
was nevertheless `foundation-not-ready`: all12 final cases failed tracking and
all12 failed the motor envelope. Mean speed/yaw MAE across seeds/worlds:

| Case (m/s,rad/s) | Speed MAE (m/s) | Yaw MAE (rad/s) |
| --- | ---: | ---: |
| Idle0/0 | .0577 | .4200 |
| Straight.30/0 | .1119 | .3953 |
| Moving.20/-.20 | .0517 | .4717 |
| Moving.20/+.20 | .0491 | .4481 |

All final cases had torque utilization p99≈1.06754 and zero rated-speed
exceedance. Thresholds are per world/case, not these aggregated means.
The final checkpoint reduced tracking errors versus750 descriptively, but
raised torque p99;750 is not selected as a fallback. Falls declined from15,247
in the first100 updates to276 in the last100, equal614,400-transition windows.
This is an evolving-curriculum trend, not causal or physical validation.

Closeout inspection: GPU961MiB/43C, only unchanged DINO1592/946MiB;
all four protected service scopes inactive. No learner remained. Since the
foundation did not pass, the originally gated mirrored-turn/new-skill stage
remainsclosed. The separately declared
[motor-cost continuation](2026-10-10-integrated-motor-refinement.md) repairs
the failed locomotion foundation; it does not treat this checkpoint as accepted.
