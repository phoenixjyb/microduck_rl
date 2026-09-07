# F1-N: predeclared matched neck-action smoothing pilot

Status: **closed, rejected at the first evaluation pair**. Both pilots completed;
this consumed the second paired-pilot slot. Current launched experiments:2/3.

## Evidence and falsifiable hypothesis

The [six-case descriptive motor dataset](2026-09-07-f1m-descriptive-motor.md)
is closed at source `c0ec38453ff18ee1100a1f3eb92e70ca37357521`;
manifest `44d43f73564e73e67d4e5a456487c4129eff44cd20b128ca051ab3b211f1795e`.
All cases were absolute-safe, but both motor-policy performance readouts failed.
Head-roll torque and mean squared load are elevated in both motor repeats,
in every settled one-second bin, with higher RMS joint speed and adjacent
speed differences. Left-knee torque also regresses relative to control.

Hypothesis: moderately increasing the existing neck-action-rate penalty may
reduce sustained head activity and its motor cost without impairing locomotion.
Joint-velocity variation is not action chatter proof; we did not record policy
action targets in that dataset. This is a testable development hypothesis,
not a causal diagnosis or a promise to repair knee/speed failures.

## Exactly one treatment axis

| Setting | Matched control | Neck treatment |
| --- | --- | --- |
| Existing `neck_action_rate_l2` weight | -0.1 | -0.2 |
| Existing `motor_torque_load` weight and live stage | -4 | -4 |
| Lateral cost weight | -0.5 | -0.5 |
| Global action-rate weight | -1 | -1 |
| Linear tracking weight / variance | 4.5 / 0.05 | 4.5 / 0.05 |

Reuse the existing neck cost implementation unchanged, with its four action
indices5:9 verified against named model joints. Do not change its reset/cache
semantics in this experiment. Do not freeze the head, change head/body pose
commands or tracking rewards, add observations, change actor dimensions,
alter physics, or combine this with a speed/motor-weight change.

Both arms restore **all** learned state independently from the same unaccepted
narrow parent8498, SHA
`7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`,
including Adam state, learning rate and environment step204000. No arm
continues from the other, and the rejected F1-M8998 is not promoted to parent.
The new control has the same reward settings as the old F1-M motor arm, but
is trained fresh for the matched comparison; no cross-run bit-exact claim.

## Fixed budget, seeds and selection

- Intended module `mjlab_microduck.foundation_neck_experiment`, protocol/output
  `f1n-neck-rate-paired-s499-v1` under `artifacts/experiments/`.
- Two10-update smokes at seed491, then two500-update pilots at seed499;
  256 environments,24 rollout steps/update, checkpoint cadence50.
- Seeds491/499 are deliberately reused development seeds, not a replacement
  seed search or independent confirmation. Fix child-start PYTHONHASHSEED to
  the training seed in both matched children. Preserve other backend settings.
- Identical initial snapshots must hash identically. First update8499, fixed
  final8998, final environment step216000. No best-checkpoint selection,
  extension, retry, added third arm or early-checkpoint evaluation.
- Verify the neck term exists, indexes the intended joints, has its declared
  live weight throughout, and contributes finite nonzero reward during both
  smokes/pilots. Preserve existing motor/lateral activity and checkpoint checks.
- Keep all gross stochastic-training guards, original deterministic evaluation
  gates and immutable gait/hop/obstacle checkpoint sentinels unchanged.

## Unchanged numerical evaluation

Record fresh parent references at503/509/521 before training; stop on any
unsafe reference. After successful smokes/pilots, evaluate matched control
then neck final8998 at503, then509, then521 **only while all previous gates
pass**. Same400steps/8env, 100 startup/300 settled,0.3m/s, heading hold,
same command delivery, mechanics and original evaluator logic.

Raw motor/route/command recording may retain already-collected tensors after
rollout, with the validated within-rollout audit. This does not replace the
original performance gate or demand historical replay identity. Use effective
child-start hash equal to the evaluation seed and capture complete process
fingerprints; do not alter numerical backend flags between matched children.

All absolute, per-environment speed/window/heading/lateral, pooled torque,
every named-joint nonregression, squared-load and mechanical-power gates must
pass relative to parent and matched control as originally defined. Head-roll
improvement alone is insufficient. Keep every failed label and stop at the
first failing pair. No favorable repeat, rounding, threshold relaxation or
extra evaluation after a rejection. A full development-seed pass still does
not establish independent-seed confirmation, hopping or obstacle retention.

## Operational and implementation gate

Implement a bounded, reviewable runner and focused tests before committing,
pushing and launching. Test that the two configs differ only in the declared
neck weight (and arm label), including live curriculum values; verify action
index mapping, identical parent restoration, nonzero term activity, finite
metrics, fixed final selection, first-failure behavior and unchanged gates.

One retained user service on100.100 only. Smokes cap180s each, pilots900s each,
evaluation children120s each, total service3600s with30s stop timeout and at
least60s internal closeout reserve. The entire cap plus90s must fit before
07:00 Shanghai September8; no new GPU launch after06:00. Require clean pushed
exact feature branch, idle GPU and protected SYSTEM services inactive. Do not
touch100.98/FilmBrain or any unrelated workload. Preserve all earlier closed
manifests, new descriptive data and checkpoint hashes before/after the run.

On closure, reconcile counters/hashes and the exact decision, mirror payloads,
test, commit and push concise evidence. Do not start a third experiment merely
because time remains; it needs evidence and a separate predeclaration. Even a
successful second pilot plus one independent confirmation would not supply
the required three independent training seeds for final promotion tonight.
No physical motion, raw perception, H2, video or harder obstacle training.

## Implementation gate evidence

`foundation_neck_experiment.py` implements the fixed sequence and immutable
manifest, with child-start hash seeds and matched backend fingerprints.
`neck_reward_observer.py` reads the original action cache before compute and
checks the reward manager's consumed weighted value afterward. It changes no
action, cache, reset, reward function, or RNG state. It retains every24-step
window, checks live neck/motor/lateral weights, and requires aggregate nonzero
activity. Consumed-buffer arithmetic uses1e-6 tensor reconstruction precision,
not a relaxed performance threshold. The inspected installed reward-manager
file is pinned by SHA256 before launch. Checkpoint cadence, saved environment
counters and finite actor/critic/optimizer tensors are verified before evaluation.

Local CPU regression:850 passed,3 existing actuator/site mapping warnings.
Tests cover the single config axis, original cache/mapping, consumed reward,
fixed checkpoint/counter corruption, child hash/caps, first-failure stop,
immutable manifests and original motor/speed/route gates. Remote frozen-runtime,
CPU tests and clean idle launch gates remain separate requirements.

## Closure: September8 00:55:40 Shanghai

Source `4317802cb09a3464b2db805ce4bef801abecd825`, user service
`microduck-rl-f1n-4317802-s499.service`, started00:42:57, normal exit0.
Remote84 focused CPU tests, frozen dependency hashes, old evidence/sentinels,
clean source and two idle samples passed before launch. On closure MainPID0,
active/exited (retained service, not running), GPU0%,45C,12MiB and no compute
PID; both protected SYSTEM services inactive. No service restart or new rollout.

All101 payloads (179,111,100 bytes), plus manifest, mirrored to
`artifacts/diagnostics/f1n-neck-rate-paired-s499-v1` on the Mac. Complete payload
inventory/hash verification, all26 post-update checkpoints plus4 initial
snapshots, finite learned/Adam tensors, checkpoint cadence and environment
counters verified on both hosts. The exact original pair decision was
independently recomputed on CPU on both hosts: `numerical-gate-stop`,22 failed
labels. All three parent cases and the control/neck503 cases were400-step,
absolute-safe rollouts. No candidate509/521 cases were run after rejection.

- Manifest SHA256: `8e6fcb53477ee6edb6508d55678073540b59d3ade61981bb9437a6ce636b0bcf`
- Decision SHA256: `ce37f29a36a078522bd09fc497d62d541b4fcaa3a58523fbb2349f13cd603196`
- Control8998: `3d895dac137576c3a7a60aed1728edf31e1ec8e78259e71d1ed0f46518c8e6c7`
- Neck8998: `e04660c311a1bdd5de27c1640f0ef0ec309f176ce4c30beb27b59a572ac8a1cb`
- All four identical initial snapshots: `2ba27fa7c3f403b6eacd48a2a102da9252f0b8aa7dfe2a91d6dec63bce9729fa`

Smokes completed10 updates each (~6.92s); pilots500 each (299.81s/298.48s),
final environment counter216000. Every rollout/reward window was finite;
neck weights remained-0.1/-0.2, motor-4, lateral-0.5. Both pilots had maximum
per-update fall fraction0.0078125, temperature58C and no gross guard failure.
Training pooled torque peaks0.726261/0.728269 are below the gross training
guard, not evidence of passing the stricter held-out limit.

| Settled seed503 metric | Parent | Control | Neck treatment |
| --- | --- | --- | --- |
| Route speed m/s | 0.264810 | 0.258435 | 0.258513 |
| Absolute lateral speed m/s | 0.076618 | 0.056387 | 0.054643 |
| Maximum heading error rad | 0.231430 | 0.161820 | 0.125040 |
| Pooled torque utilization p99 | 0.562742 | 0.541279 | 0.566971 |
| Mean squared utilization | 0.039734 | 0.039624 | 0.040386 |
| Mean absolute mechanical power W/motor | 0.137111 | 0.143608 | 0.147600 |
| Head-roll utilization p99 | 0.288640 | 0.362075 | 0.359958 |
| Left-hip-pitch utilization p99 | 0.662164 | 0.631318 | 0.720799 |
| Right-knee utilization p99 | 0.662799 | 0.655608 | 0.685223 |

Values above are rounded for display only; all decisions use retained full
precision and per-environment/joint gates. The22 labels include insufficient
speed and recovery window, lateral motion, multiple joint regressions,
pooled torque versus control, and mechanical power versus parent. Control
also fails its speed/lateral comparison; neither arm is admitted.

The live raw neck action-change cost averaged0.179033 control versus0.170752
treatment, with final100-update means0.178495/0.171470. Thus the term was live
and the sampled action cost decreased; acceptable head/body motor behavior
did not follow. This one development seed cannot establish a reliable causal
effect. Do not raise this penalty again or promote head-roll-only improvement.

## Read-only command-domain diagnosis and next step

The pinned training YAML and guarded training loop use only body commands
`[0.3,0,0]`. In all five retained held-out cases, all2400 settled actor inputs
have nonzero yaw from the unchanged heading controller. Forward/lateral
commands remain within their training bounds. At seed503, absolute mean yaw
is0.059199/0.041417/0.039590 rad/s for parent/control/neck; parent521 reaches
-0.273181 rad/s. `foundation_command_coverage.py` verifies the closed manifest,
reads YAML without executing Python tags, validates original command delivery,
and compares native float32 command ranges without a deadband.

This is a **continuation command-domain gap**, not proof that the earlier
parent was never trained on yaw, nor proof that the gap caused this rejection.
No new physics/optimizer steps were used in this audit. The bounded next
hypothesis is [F1-Y command support](2026-09-08-f1y-command-support.md), not
another neck/motor-weight search. It remains unimplemented/unlaunched and must
pass its implementation gate before consuming the last slot.

Closeout CPU regression:858 passed,3 existing actuator/site warnings. The
command audit's canonical JSON SHA256 is
`3240dbd02754a341ba2494528322aff45cc38740ad35645d4718b6b463f69053`
on both Mac and100.100. Reproduce with `audit_retained(root)` from
`mjlab_microduck.foundation_command_coverage`, then SHA256 of UTF-8
`first_attempt_smoke.canonical(result)`; this is a read-only result, not a
replacement for either original manifest or decision. Post-push verification
at126cf5083ff93e80bf42ab5a0b919e3abab3503f passed39 focused CPU tests, exactly
reproduced this canonical hash on100.100, and reverified all older pinned
evidence/sentinels. No GPU work was launched during closeout.
