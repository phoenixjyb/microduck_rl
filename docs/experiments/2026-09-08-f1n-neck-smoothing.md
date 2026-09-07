# F1-N: predeclared matched neck-action smoothing pilot

Status: predeclared, **not implemented or launched**. This reserves the second
paired-pilot slot of the current window; a launch consumes that slot even if
it stops early. Current launched paired experiments:1/3.

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
