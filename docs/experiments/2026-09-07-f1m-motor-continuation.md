# F1-M: motor-load contrast with the lateral objective retained

Status: **completed and numerically rejected at the first pair**. Renewed user authorization on September7:
continue bounded coding/training through **September8 07:00 Asia/Shanghai**.
This is a new experiment, not a retry or reinterpretation of rejected F1-L.

## Hypothesis and controls

F1-L seed467 reduced settled absolute cross-route speed from0.073549 to
0.062914m/s against its matched continuation, but pooled torque rose from
0.552639 to0.600343, squared load rose15.2% and mechanical power rose19.0%.
Its absolute torque and speed/direction gates failed. Preserve all original
reports, manifests, decisions and checkpoints unchanged.

Test whether stronger **existing** motor-load shaping can retain the lateral
gain without paying that motor cost. Both arms independently restore the same
unaccepted F1-R narrow model8498, SHA256
`7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`.
Do not resume the torque-rejected F1-L model8998 or call the research parent
an accepted gait. Restore actor, critic, normalizers, Adam and common step204000;
first new update8499, final8998 and common step216000 after500updates.

Both arms keep lateral velocity cost-0.5, linear tracking weight4.5/variance0.05,
action-rate weight-1, rigid feet, fixed(0.30,0,0) training commands, PPO, physics,
randomization, terminations, 61D actor and frozen completed curricula unchanged.
The **only treatment variable** is motor_torque_load reward weight:
control-2, treatment-4. Its initial manager config and collapsed step-zero
curriculum must agree; otherwise reset silently undoes the treatment. No new
motor clamp, mechanics change, action override or actor observation is added.
The reward remains a normalized squared-load plus soft-limit hinge proxy,
not a calibrated motor temperature/current model or physical safety proof.

## Fixed budget, identity and execution order

Module `mjlab_microduck.foundation_motor_experiment`, protocol/output
`artifacts/experiments/f1m-motor-paired-s499-v1`.
Host100.100 only, exact clean/pushed `feat/athletics-obstacle-curriculum`,
frozen dependencies and original walking/narrow/rejected-hop hash sentinels.
No file reuse and no changes to an active job.

1. Evaluate parent sequentially at seeds503,509,521; stop on the first unsafe
   reference before optimizer work. Its speed deficits remain descriptive.
2. Two independent10-update smokes, seed491, one per arm. Require finite
   successful source-matched results, matching restored-state hashes and
   measured smoke time consistent with the pilot budget.
3. Two independent500-update pilots, seed499, 256env,24steps/update, save50.
   Both resume model8498, not smoke output. Require matching initial states.
4. Evaluate control then treatment at503,509,521, final checkpoint only.
   Stop at the first unsafe control or failed candidate pair. If the control
   fails, the remaining comparison is unanswered; do not rerun seeds to find
   a favorable reference. At most9cases/72first episodes, no best-checkpoint search.

One sequential retained user service, hard3300seconds. Child limits are180s
per smoke,900s per pilot,90s per evaluation, with40s reserve before a child.
Latest launch before September8 06:00 Shanghai; no child may overrun the07:00 cutoff.
Expected pilots total about10minutes from prior timings; hard limits remain
authoritative. Durable saves at every50updates and final; retain partial data
on failure. No automatic campaign restart. Stable-idle check before **every**
child retains two zero-utilization samples1second apart within10seconds;
any competing compute PID/protected service, memory>=100MiB or GPU>=80C fails
closed. Only residual utilization with empty/cool GPU may settle. Never stop
another workload to make a gate pass.

## Unchanged numerical and retention gates

Use the unchanged F1 first-attempt400step/8env/0.02s protocol,100startup and
300settled samples. Parent/control/treatment all use the same frozen heading
hold ON and fresh-command/route-trace audit. No heading controller in training.
Preserve zero terminals/nonfinite/rated-speed exposure; all/settled pooled
torque p99<=0.60; each environment body/route mean0.30+/-0.03m/s; stable0.5s
speed window within2s; heading<=0.25rad; each environment mean absolute lateral
speed<=0.05m/s. Preserve original per-env speed, heading, lateral and pooled
torque nonregression margins against safe parent and matched control. Require
every named joint p99<=reference+0.02 and squared load/mechanical power<=1.05x
both references. No rounding a failure into a pass.

Report the signed motor/sideways changes even when admission fails. A lower
motor cost alone is not success if it loses speed, balance or lateral control.
All three passing pairs give only single-training-seed diagnostic support;
independent-seed confirmation and integrated retention remain required.
Training fall/torque/rated-speed gross guards are unchanged, not equivalent to
deterministic acceptance. TensorBoard and per-update JSONL retain finite
reward/optimizer values, falls, motor cost, squared thermal proxy, soft-limit
exposure, pre-reset torque and sampled GPU temperature. Verify actual active
motor weights-2/-4 and active lateral cost in both arms.

## Overnight continuation policy

See [the bounded overnight curriculum](2026-09-07-overnight-curriculum.md).
A stopped experiment stays closed: retain/hash/mirror its evidence, diagnose
read-only, test and commit/push the result before declaring a distinct revision.
The renewed overnight authorization permits the next justified bounded revision,
not an unlogged retry, relaxed gate, seed hunt or automatic stage promotion.
No obstacle training, H2, video, raw perception or physical motion is admitted
by F1-M. Protected AI services stay inactive; do not restore them at the cutoff.

## Prelaunch verification

Local focused CPU regression:727 passed, no skips, two existing actuator/site
pattern warnings,15.22seconds. Coverage includes live motor-weight curriculum
at reset/resume/final times, single-axis config equality, unchanged historical
evaluator defaults and safety-stop handling, fresh heading commands, motor
stream and per-joint/power gates, idle telemetry retention, first-failure stop,
budget refusal, paired initial states and complete immutable manifest coverage.
Document links resolve and `git diff --check` passes. GPU/remote runtime results
are separate live checks, not implied by these local tests.

## Retained completion: September7 22:13 Shanghai

Source `fbc0b73ca1a61016ea8e4002f1af65e24990e892`; user service
`microduck-rl-f1m-fbc0b73-s499.service` ran22:01:28–22:13:29 and exited normally.
Normal service completion is **not** numerical acceptance. Remote prelaunch
regression also passed727 tests in15.35s (same two warnings, no skips).

Both10-update smokes and both500-update pilots completed. All26 post-update
checkpoints and4initial snapshots were verified finite with exact iteration
and common-step counters. Every initial snapshot matches the original narrow
parent's actor, critic, normalizers and Adam, with the declared next-iteration
label8499 and common step204000. All four initial file hashes are
`4eab6fcb933e5244bf29a546a4fb663bb4ba201a32966277cf75f2ff4c84d6d3`.

| Training arm | Pilot seconds | Fall windows | Max fall fraction/window | Final model8998 SHA256 |
| --- | ---: | ---: | ---: | --- |
| Control motor weight-2 | 293.383 | 7 | 1/256 | `cf63952d0975b120f14f92b7cb9b7d5f8b2d6f5e46b5db8f214c63b1c77df83a` |
| Treatment motor weight-4 | 292.348 | 8 | 1/256 | `dcef0f75a3b3bccd0f026de34510aca793f888d10ff8aa4569e74ee4201a18fb` |

Each pilot retains500 finite values for75 TensorBoard scalar tags. Actual
motor-weight curriculum telemetry is-2 throughout control and-4 throughout
treatment; lateral cost is active in both (final episode metrics-0.214891 and
-0.226036). No gross training guard fired. Maximum pooled training torque p99
was0.734897/0.722518; maximum rated-speed sample fraction0.000116257/0.000104632;
maximum sampled GPU temperature58C in both. These stochastic training values
are not deterministic evaluation acceptance or continuous physical telemetry.

All three parent references and control/treatment seed503 completed400steps
with no terminal or absolute safety failure. The first treatment pair failed
17 labels, including speed, lateral motion and motor nonregression. Seeds509
and521 were **not** evaluated for the trained arms after the terminal decision.
Five cases/40first episodes were evaluated, not the nine-case maximum.

| Settled seed503 metric | Parent8498 | Matched control8998 | Motor treatment8998 |
| --- | ---: | ---: | ---: |
| Body forward mean m/s | 0.265486 | 0.266530 | 0.263566 |
| Route forward mean m/s | 0.264222 | 0.265489 | 0.262488 |
| Absolute cross-route speed m/s | 0.076588 | 0.061455 | 0.059304 |
| Maximum absolute heading rad | 0.220607 | 0.173266 | 0.149070 |
| Pooled torque utilization p99 | 0.555508 | 0.577563 | 0.577160 |
| Mean squared utilization | 0.039697 | 0.042347 | 0.040973 |
| Mean absolute mechanical power/sample W | 0.137257 | 0.155006 | 0.150813 |
| Stable speed windows | 0/8 | 1/8 | 1/8 |

Versus matched control, treatment reduced lateral motion3.50%, squared load
3.24% and power2.70%, but reduced route speed by0.003000m/s and did not improve
stable windows. Every treatment environment still exceeds0.05m/s mean absolute
lateral speed (range0.054552–0.065314). Two route means are below0.25m/s.
The pooled torque p99 is below the absolute0.60 gate but exceeds the frozen
parent's+0.02 margin; power is9.88% above the parent, beyond the5% margin.

Named-joint diagnosis: against matched control the left knee p99 rises
0.594742→0.639553 and head roll0.299829→0.353293, both above the+0.02 margin.
Against the parent, left hip yaw/pitch, head yaw/roll, right hip yaw and right
knee regress. Other joints improve; this is load redistribution, not merely
one global scale change. These are pooled named-joint quantiles, not time- or
contact-phase-resolved evidence. Original reports retain raw route/command
traces but not raw per-step motor history, so they cannot establish when those
joint peaks coincide with slowing or lateral motion. Do not infer a gait-phase
cause from these quantiles alone.

Read-only closeout reproduced the exact pair/failure decision on CPU, audited
all five heading-command traces and reconstructed their route-speed summaries
and stable windows from raw route samples. All83 payload hashes and byte counts
match on remote and local mirror;163,963,281 payload bytes plus manifest.
Local mirror: `artifacts/diagnostics/f1m-motor-paired-s499-v1`.

- Manifest SHA256: `cf0954316a0523f54104e4ddf0b1982f4e3dde9ef53e5352c75aad0883c0a788`.
- Decision SHA256: `1c5bb1f83bbd86a2a8eea8a9d65e71bdad43e0f8e041df4f484018e07aecfe7f`.
- Parent503 report: `36d1ebd9fc86afa4039c29287010beb12d626b97be09d80199f2a4ab0b1d3deb`.
- Control503 report: `ca6cc0ab870618f242a6b71e497b37e0cd177961d9c138373e25e256c89e177e`.
- Treatment503 report: `b54f7b20b897b622042af15add4783047e05efd3e91846d352485ba3d6b7f22f`.

At22:18 GPU was idle (no compute PID,0%,45C,12MiB) and both protected system
services were inactive. Original walking/narrow/rejected-hop sentinels remain
unchanged. **No promotion or continuation of this closed seed sweep.** Next:
predeclare a short motor-timing measurement of these already absolute-safe
seed503 cases before guessing another reward change. That measurement must
write separate evidence, add no optimizer updates, and retain existing gates.
