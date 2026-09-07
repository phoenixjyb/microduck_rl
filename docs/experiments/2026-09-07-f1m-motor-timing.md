# F1-M motor timing: evaluation-only measurement predeclaration

Status: **closed at the parent replay-identity failure; no paired timing conclusion**. This is the
next bounded diagnostic under the September8 07:00 Shanghai work window.
F1-M remains rejected. Paired pilot count stays1/3; optimizer updates here=0.

## Question and evidence boundary

In the retained seed503 pair, doubling motor-load weight lowers pooled load
and power modestly, but increases left-knee p99 by0.044811 and head-roll p99 by
0.053464 versus the matched control. The pooled route speed falls by0.003m/s.
Named-joint aggregate quantiles do not identify when/where the load moves.
Before selecting another reward weight, record the underlying motor samples
for the same parent/control/treatment seed503 cases and inspect concentration
by environment and control interval. This is not contact-phase identification
or proof that a particular joint causes the speed deficit.

## Immutable references and unchanged execution

- Original F1-M source: `fbc0b73ca1a61016ea8e4002f1af65e24990e892`.
- Original directory: `artifacts/experiments/f1m-motor-paired-s499-v1`.
- Manifest SHA256: `cf0954316a0523f54104e4ddf0b1982f4e3dde9ef53e5352c75aad0883c0a788`.
- Decision SHA256: `1c5bb1f83bbd86a2a8eea8a9d65e71bdad43e0f8e041df4f484018e07aecfe7f`.
- Checkpoints: original narrow8498, F1-M control8998 and motor8998; hashes
  are pinned through the original manifest/results and recomputed before use.
- New directory/protocol: `artifacts/experiments/f1m-motor-timing-s503-v1`.
- New module: `mjlab_microduck.foundation_motor_timing`.

Before launching, verify all83 original payload hashes/bytes and the original
deterministic rejection. Numerical entry gate: all three retained503 reports
have full400step/8env coverage and no absolute safety failures. Their failed
speed/lateral/nonregression gates remain failed; this gate admits only a
bounded measurement of existing policies, not training or skill promotion.

Run parent then control then motor, seed503 only, each in its own sequential
child process. Keep the original F1-M task, heading controller, physics,
random state initialization, commands, actor61D and400step protocol unchanged.
Retain raw route/command traces and add raw pre-reset motor force/speed arrays
of shape400x8x14. These arrays are already collected by the evaluator for its
aggregate statistics; the option only serializes them after the rollout.
Historical callers default to no new motor trace. No reward, reset, actuator,
controller or actor-observation change and no resampling are involved.

Retain force in Nm and speed in rad/s, explicit joint column names and timing:
force is last-physics-substep-derived with one-integration lag; speed is the
integrated velocity at the post-decimation/pre-reset capture. Route velocity
is sampled pre-control. Associations within that control interval are not
simultaneous physics measurements, calibrated electrical power, contact-phase
labels or motor temperature readings. No new physical safety assertion.

## Validation and refusal rules

Before each child, require two idle GPU samples under the existing bounded
idle checker and both protected SYSTEM services inactive. Exact clean pushed
branch only,100.100 only; preserve100.98 and unrelated workloads. One retained
user service, hard600seconds,120seconds/child,40seconds closeout reserve. No
new GPU work after06:00 Shanghai; no child may extend beyond07:00.

Write each new raw report before any downstream audit can fail. Refuse further
cases on an absolute safety failure or **any bit-exact replay disagreement**
with the corresponding original report (excluding only source and the added
motor_trace field). Do not silently tolerate, average, retry or select away
a changed replay. A mismatch is separately retained evidence to diagnose.
Preserve original F1-M files/decision untouched before and after this job.

On matching safe replays, independently reconstruct every pre-reset motor
aggregate and all14 named-joint p99s from the new raw motor arrays. Recheck
route summaries/windows and command delivery from raw traces. Report per joint:
settled p99 by environment, soft-limit fraction, share of squared load,
peak control step/environment, fixed one-second bins for steps100–399, and
load conditional on pre-control route speed below0.27m/s versus other samples.
An empty condition produces null, not a fabricated zero. All associations
remain descriptive; no contact-phase or causal conclusion is admitted.

All three valid measurements yield `timing-measurement-only`, never policy
acceptance. Mirror/hashes/test/commit the closeout before choosing paired
pilot2. A new training hypothesis must cite the measured load distribution,
retain all original numerical gates, and change only one declared variable.
No larger seed sweep, video, obstacle promotion, hopping promotion, raw
perception or physical motion follows automatically from this measurement.

## Prelaunch checks

Local focused regression:749 passed, no skips, two existing actuator/site
pattern warnings,15.84s. Tests include raw-versus-summary reconstruction,
known peak step/environment, units/joint order/timing/coverage corruption,
empty conditional bins, zero-optimizer orchestration, exact replay mismatch
refusal, first-failure retention and terminal pre-reset capture with the new
option both enabled and disabled. `git diff --check` passes.

## Retained failure and read-only diagnosis

Source `dbbeb8c8e708dd99968d6dde6330531baa5e2d1b`. Remote focused regression
also passed749 tests,15.85s, same two existing warnings and no skips. Service
`microduck-rl-f1m-timing-dbbeb8c-s503.service` ran22:33:14–22:33:42 Shanghai,
September7, then exited1 at the declared gate:

`ValueError: motor instrumentation replay differs from retained case`

The parent completed400steps without a terminal/absolute safety failure and
its raw report was saved. The historical-equivalence check then stopped the
campaign. **Control and treatment were not run.** No optimizer update or policy
change occurred. Do not resume this directory, relax the equality criterion,
or interpret the safe parent as a valid three-arm timing measurement.

Read-only comparison confirms matching checkpoint/seed, controller constants,
motor-stream metadata and sentinel hashes. The first three pre-control velocity
samples (steps0–2) match exactly. First divergence: step3, environment1,
body-forward speed0.017756961286067963 versus0.01775696873664856m/s
(7.45e-9m/s). First position difference is step4, environment2. Differences
grow during the trajectory: maximum absolute discrepancies are0.048875m/s
body forward,0.042570m/s route forward,0.077583m/s cross-route,0.057006rad
heading and0.030276m position across either position column. Initial tiny
differences followed by amplification are consistent with floating-point or
solver divergence, but do **not** establish that cause or rule out initialization
or execution-path effects. The original process did not retain initial actor
input/action or complete backend-state fingerprints.

| Settled parent503 metric | Original F1-M | Timing replay |
| --- | ---: | ---: |
| Body forward mean m/s | 0.265486180 | 0.265508459 |
| Route forward mean m/s | 0.264222319 | 0.264257576 |
| Absolute lateral mean m/s | 0.076587537 | 0.076657114 |
| Pooled torque utilization p99 | 0.555508137 | 0.554849505 |
| Squared utilization mean | 0.039696987 | 0.039728345 |
| Absolute mechanical power/sample W | 0.137257287 | 0.137113368 |
| Stable windows | 0/8 | 0/8 |

The new raw force/speed arrays independently reproduce their **own** stored
motor aggregates and all named-joint quantiles; raw route/command audits also
pass. This establishes internal accounting only, not equivalence to the old
trajectory. Similar aggregate means do not override the failed exact-replay
gate, and one cross-source replay does not measure same-source repeatability.

Five payloads,4,677,248bytes plus manifest are mirrored to
`artifacts/diagnostics/f1m-motor-timing-s503-v1`, all hashes/bytes verified.
Original F1-M manifest/decision and all checkpoint sentinels remain unchanged.

- New manifest SHA256: `001ed5d5d9060a687a206af50d11f00cf26019b62d50d9f34b331a5beee51037`.
- New decision SHA256: `2661a4c0b7e7e6c4d7547d603905b69a0ca7fe7b3a09f24ed351114e44da5160`.
- New parent report SHA256: `e2cdf82a21fa582d91a0b0ddb097395bef475231090b76db2a453e694c484fb4`.

GPU returned idle,0%,46C,12MiB with no compute PID; protected system services
remain inactive. No service restart or backend/seed/precision fix was made.

Next bounded work is an execution-path/repeatability design, **not another
reward adjustment**: inspect initialization/seeding/backend code read-only,
then predeclare a same-source, recording-disabled parent503 versus parent503
control in fresh processes if justified. Preserve first-attempt absolute
safety gates and stop if the first control is unsafe. Exact same-source
divergence would rule out motor-history serialization as a necessary cause;
it would not by itself identify the numerical source. A repeated OFF case is
a new measurement-control experiment with its own source/budget/output, not a
retry of this failed comparison. Do not run a third ON case or paired pilot2
without separately predeclaring why it is needed and its stopping rule.
