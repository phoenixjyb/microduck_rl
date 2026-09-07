# F1-M: motor-load contrast with the lateral objective retained

Status: predeclared, not yet run. Renewed user authorization on September7:
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
