# F1-M motor timing: evaluation-only measurement predeclaration

Status: instrumentation implemented; measurement not yet run. This is the
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
