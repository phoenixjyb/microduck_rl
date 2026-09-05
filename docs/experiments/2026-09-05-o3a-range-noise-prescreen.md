# O3a compact-range-noise pre-screen

Date: 2026-09-05

Parent: `c8026b1781f92eec0d0eadcdcba3e1bc74200207`

Decision: **predeclare a bounded HC4-LH sensitivity pre-screen; do not retrain
or promote a noisy-sensor capability yet**

## Purpose and unchanged authority

HC4-LH and HC4-R2 remain the accepted exact-geometry specialists after the
HC4-U1/HC4-U2 unified behavioral-cloning line closed on local fixed-attempt
regressions. O3a begins the next independent capability axis: sensitivity to
range error in the compact externally supplied obstacle estimate.

This first slice changes no policy weights, locomotion actor, command bounds,
box geometry, collision geometry, termination, attempt horizon, or accepted
specialist envelope. The supervisor still consumes the seven-field compact
contract `range, bearing_sin, bearing_cos, width, height, closing_rate, valid`;
it never consumes camera pixels. Range perturbation is applied only to that
policy-facing contract. Simulator physics, collision/pass classification, and
visibility stay on exact ground truth. Physical motion is not authorized.

The noise magnitude below is a provisional simulation stress level, not a
measured claim about a real sensor. Consequently a pass can justify a broader
measured-noise campaign, but cannot complete O3a or establish real-perception
readiness.

## Frozen perturbation protocol

Protocol identity: `compact-range-uniform-v1`.

- Range error is independent bounded uniform noise in `[-0.02, +0.02]` m.
- A new sample is drawn for every environment at each supervisor update; the
  supervisor updates every five simulator steps.
- Bearing, width, height, closing rate, and validity are unchanged.
- Encoded range remains clipped by the existing v1 observation limits.
- Noise uses a dedicated CPU Torch generator with seed
  `physics_seed + 3000011`. It does not consume or alter the simulator's
  global random stream. The full model, distribution, bound, seed rule, and
  update interval must be retained in the JSON report.
- Exact baseline runs set the range bound to zero and preserve legacy rollout
  behavior byte-for-byte at the observation boundary.

Supplying a dedicated stream matters for the paired comparison: a noisy run
must not change future environment resets merely because it consumed extra
random values.

## HC4-LH seed-271 pre-screen

The first runtime screen stays inside HC4-LH's accepted exact-geometry
envelope and uses HC4-E1's fixed first-attempt denominator.

| Field | Frozen value |
|---|---|
| Locomotion actor | retained motor-aware `model_7998.pt` |
| Supervisor | accepted HC4-LH checkpoint SHA-256 `0b2608080671c5df85d8c9f900d68b6a6f298ec820eb1c6ba75afc948337505a` |
| Nominal speed | 0.50 m/s |
| Obstacle forward position | 1.15 m |
| Obstacle lateral positions | -0.08, 0.00, +0.08 m |
| Physics seed | 271 |
| Noise seed | 3000282 |
| Environments | 64 per cell |
| Step ceiling | 700 |
| Evaluation window | `first-terminal-attempt-per-environment-v1` |
| Compared conditions | exact baseline and `compact-range-uniform-v1` |

The exact baseline runs first. The noisy condition may run only after the
baseline completes with the exact actor/supervisor identities, 64 completed
attempts per cell, zero unresolved/hard/other terminal events, finite metrics,
zero falls/NaNs/non-finite steps, and zero rated motor-speed exceedance.

## Frozen decision gate

The noisy condition passes this pre-screen only if every cell and the pooled
matrix satisfy all of the following against the matched exact baseline:

- expected attempts = completed attempts = 64 per cell and unresolved = 0;
- zero hard failures, other terminal events, falls, NaN terminations,
  non-finite steps, and rated motor-speed exceedance;
- no increase in collision count, per cell or pooled;
- fixed-denominator clean-pass loss no greater than 3/64 per cell and no
  greater than five percentage points pooled;
- approach and recovery route speed each no more than 0.03 m/s below baseline
  per cell and no more than 0.01 m/s below baseline when sample-weighted across
  the matrix;
- motor torque-utilization p99 no greater than 0.60 in any cell.

Interaction-phase speed remains deliberately ungated: the duck may slow or
brake while avoiding the obstacle, but must track its nominal speed before and
after the maneuver.

A deterministic comparison artifact must name both report hashes and list
every failed gate. Only a complete `continue` decision allows a separately
predeclared HC4-R2 near-range sensitivity screen. A stop decision closes this
noise level for diagnosis; it does not authorize retraining, a post-hoc gate
change, bearing noise, latency, dropout, MP4 creation, raw perception, or
physical motion.

## Retained implementation and wiring evidence

Commit `a2d3962514dbc1160d16bb6a0e4a4967c3150a46` adds the dedicated replay
stream, range-only rollout wiring, report provenance, and deterministic O3a
gate. The focused obstacle suite passed 74 tests locally and on 100.100. A
four-environment CPU comparison at the centered cell completed all first
attempts with four clean passages in both conditions and no collision,
timeout, fall, NaN, non-finite, hard, other, or unresolved event.

- exact CPU report SHA-256:
  `08b1c474a375258214d16c4feca504e4dd903bbfc876d0da6fc3596f6f0b2e5a`;
- noisy CPU report SHA-256:
  `67268c521bac7fdeda86f849693202e2012d4297d488cf05dd0b987eb037f9be`.

The noisy report names only `range` as perturbed, retains the two-centimeter
bound, physics seed 271, noise seed 3000282, and exact ground-truth outcomes.
This is wiring evidence only and does not count toward the 192-attempt gate.

## Retained HC4-LH seed-271 decision

The exclusive-CUDA exact and noisy matrices both completed 192/192 first
attempts. They recorded zero timeout, fall, NaN, non-finite, hard, other, or
unresolved events and zero rated motor-speed exceedance.

| Condition | Clean | Collision | Timeout | Max torque p99 |
|---|---:|---:|---:|---:|
| Exact baseline | 191 | 1 | 0 | 0.5905 |
| Range error `[-0.02, +0.02]` m | 192 | 0 | 0 | 0.5879 |

The sole baseline collision was centered. The noisy matrix had no collision
in any cell, so clean count changed by +1 and collision count by -1 without a
timeout trade. Sample-weighted approach speed changed by +0.00073 m/s and
recovery speed by -0.00207 m/s. Every per-cell phase-speed delta stayed inside
the -0.03 m/s bound, and every deterministic check passed.

Retained SHA-256 evidence under
`artifacts/evaluations/o3a-a2d3962-hc4lh-s271-*` on 100.100:

- exact baseline report:
  `97adf0d98842ab7e4c8fe3375e453459819e49bb5c90b24f2e9b31cf65482f48`;
- noisy report:
  `a0712b0d91978889ed4f5adf4fe11e51959ce74022b6188afa6c1f699edfa902`;
- deterministic decision:
  `921c5a309f4c57169a65e9f6fabf63fef1900065d4b9edc75092f5fcdfcd0636`.

Decision: `continue_hc4r2_predeclaration`. This single matched seed establishes
bounded sensitivity evidence, not a statistical claim that range noise is
beneficial. It authorizes only a separately frozen HC4-R2 near-range
pre-screen. O3a completion, measured-sensor acceptance, retraining, later
sensor axes, video, raw perception, and physical motion remain closed.

## HC4-R2 seed-277 predeclaration

Parent: `e44c95130f261354bb9d64c827cc22fc6bc95a47`.

The second specialist uses the byte-exact HC4-R2 checkpoint SHA-256
`c4ba5925de7144373c94145b57b5e7a7ae3e1fc89bc7c2c3203f8724bdebf1b7`.
Its paired matrix remains entirely inside the accepted near-range envelope:

| Field | Frozen value |
|---|---|
| Nominal speeds | 0.30 and 0.40 m/s |
| Obstacle forward position | 0.90 m |
| Obstacle lateral positions | -0.08, 0.00, +0.08 m |
| Physics seed | 277 |
| Noise seed | 3000288 |
| Environments | 64 per cell |
| Step ceiling | 700 |
| Evaluation window | `first-terminal-attempt-per-environment-v1` |
| Compared conditions | exact baseline and the unchanged `compact-range-uniform-v1` |

The six-cell exact baseline runs first and must pass the same identity,
completion, finite-state, terminal-integrity, and rated-motor-speed admission
checks as the HC4-LH baseline. The noisy half then uses the unchanged
independent bounded uniform error in `[-0.02, +0.02]` m at each five-step
supervisor update. All other compact fields and all ground-truth outcomes stay
exact.

The decision gates are unchanged: no per-cell or pooled collision increase;
clean-pass loss no greater than 3/64 per cell and five percentage points
pooled; approach/recovery speed delta at least -0.03 m/s per cell and -0.01
m/s sample-weighted; torque-utilization p99 at most 0.60 per cell; and zero
unresolved, hard, other-terminal, fall, NaN, non-finite, or rated-speed event.

Only a deterministic `continue_multi_seed_predeclaration` decision allows a
later document to freeze multiple independent seeds. This seed-277 screen
cannot by itself complete O3a, authorize a larger noise level, measure a real
sensor, create video, introduce raw perception, or authorize physical motion.

## Retained HC4-R2 seed-277 decision

Commit `218266e382869c5a99f531da161888d080eae26c` extends the deterministic
gate to the frozen HC4-R2 matrix. Its six gate tests passed locally and on
100.100 before runtime evaluation.

The exclusive-CUDA exact and noisy conditions each completed all 384 first
attempts as clean passages. Both had zero collision, timeout, fall, NaN,
non-finite, hard, other-terminal, unresolved, or rated motor-speed event. The
noisy matrix's maximum torque-utilization p99 was 0.5679.

Range noise changed sample-weighted approach speed by -0.00545 m/s and
recovery speed by +0.00002 m/s. The largest individual approach delta was
-0.00987 m/s at 0.30 m/s and +0.08 m lateral; every per-cell phase delta
remained inside the -0.03 m/s bound. Every deterministic check passed.

Retained SHA-256 evidence under
`artifacts/evaluations/o3a-218266e-hc4r2-s277-*` on 100.100:

- exact baseline report:
  `e83a59edf50b8567c0f2a40b750eb354d334734d7a4a4757ac9b1f70a9d65a22`;
- noisy report:
  `90f0afcbede29d566cffc28661461a0127dbf1693d1264f7b2c6e2059246e932`;
- deterministic decision:
  `5a0435d74876467fbaffdb4d150c7bb733664790a163c679c9b790c21231f8ef`.

Decision: `continue_multi_seed_predeclaration`. Together, the two specialist
pre-screens cover 576 noisy first attempts without a noisy-condition collision
or timeout. They remain independent single-seed results and do not establish
a three-seed campaign result or measured sensor readiness.

## Three-seed continuation predeclaration

Parent: `2928b0586d6bd54b1a787ec7b6a3ad613d6fbe97`.

The only additional physics seeds are 281 and 283. Together with retained
HC4-LH seed 271 and HC4-R2 seed 277, this gives each specialist three
independent seeds. Noise seeds remain mechanically derived by adding 3000011:
3000292 and 3000294 for the two new physics seeds.

For each new seed, HC4-LH repeats its frozen three-cell 0.50 x 1.15 m matrix
and HC4-R2 repeats its frozen six-cell 0.30/0.40 x 0.90 m matrix. Every cell
uses 64 environments, a 700-step ceiling, the fixed first-attempt protocol,
and the same exact-versus-`[-0.02, +0.02]` m comparison. Policy checkpoints,
box geometry, update rate, separate noise stream, command authority, exact
ground-truth outcomes, and all numerical gates remain unchanged.

Execution is sequential and fail-closed:

1. run and admit the exact baseline for one specialist/seed;
2. run its noisy condition only after that baseline passes;
3. produce the deterministic per-seed decision;
4. stop the affected specialist immediately if any check fails;
5. continue to the next seed only after a complete per-seed pass.

The final deterministic campaign artifact must name all six baseline/noisy
report pairs, their hashes, their per-seed decision hashes, and the exact
source commit. Campaign acceptance requires every retained per-seed gate to
pass without gate relaxation. It reports pooled outcomes and sample-weighted
phase-speed deltas descriptively but cannot use pooling to hide a local fail.

A complete pass establishes only three-seed simulation sensitivity to bounded
two-centimeter uniform range error inside the two accepted specialist
envelopes. Because this distribution is provisional rather than measured,
even campaign acceptance does not complete O3a. The next gate would be a
sensor-calibration artifact defining a measured range-error distribution.
Failure stops the affected line for diagnosis. No result authorizes
retraining, bearing noise, latency, dropout, MP4 creation, raw perception, or
physical motion.
