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
