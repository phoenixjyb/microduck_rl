# Explanatory motion decomposition of the closed map

Purpose: diagnose the [failed performance criteria](2026-09-09-foundation-command-map-results.md)
using retained CPU traces only, before selecting another learning change.
No new inference, reset, simulation, optimizer, video or acceptance test is run.
The original map and its numerical gates remain immutable.

Implementation: [foundation_motion_decomposition.py](../../src/mjlab_microduck/foundation_motion_decomposition.py),
with [synthetic signal tests](../../tests/test_foundation_motion_decomposition.py).
The input campaign result is pinned to SHA256
`8dbc0f8fa5f41a3e35764b1e335003033f5cb72811fccc61bd14781efd7cce5d`.
All18 cells must pass the existing exact reader, manifest/journal reconciliation
and unchanged original-score comparison. Partial or safety-stopped traces do
not enter this explanatory protocol. Output is exclusive and outside the raw map.

## Fixed analysis, declared before execution

Report each environment separately in startup steps0–99 and settled100–399:

- Forward mean error, RMS error and demeaned RMS. Bias squared plus variance
  must equal mean squared error in synthetic tests; smoothing cannot erase bias.
- Signed lateral mean, absolute mean, demeaned RMS and the demeaned fraction
  of lateral mean-square energy. Demeaned variation includes transients as well
  as oscillation; it is not automatically gait sway or a harmless component.
- Dominant positive-frequency bin and its power fraction after demeaning and
  applying a symmetric Hann window. Resolution is0.5 Hz in startup and1/6 Hz
  in settled data. Flat signals report no peak. Foot-contact phase was not
  recorded in this map, so a spectral peak does not prove gait phase or cause.
- A fixed26-sample arithmetic mean, valid windows only, separately inside each
  phase:0.50 s between first/last timestamps,75 startup/275 settled windows.
  No padding, phase mixing, threshold selection or passing/acceptance score.
- Each named joint's normalized torque p99, soft-limit exposure, absolute
  mechanical-power proxy and peak control-interval index/environment. Torque
  normalization remains the0.6 Nm reference. Velocity is pre-control; motor
  values belong to the corresponding pre-reset interval, not an instantaneous
  simultaneous measurement. Neither these proxies nor spectral peaks establish
  calibrated thermal safety or causality.

Use the original Linux host for authoritative exact reconciliation. A read-only
Mac check of cell00 found two squared-utilization mean reductions differing by
3.469446951953614e-18, although retained file hashes match. Do not relax the old
reader's equality test to accommodate cross-architecture arithmetic. Its Linux
closeout already passed exact reconciliation; the new CLI requires the original
machine identity and CUDA hidden.

Results and their exact output hash will be appended after the tested CPU run.
No training revision is selected by this analysis design alone.
