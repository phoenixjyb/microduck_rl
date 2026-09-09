# B1-N faster-runtime qualification: review draft, not admission

The [same-input forward diagnosis](2026-09-09-stance-forward-repeatability.md)
shows eager and captured execution both vary numerically before integration.
It does not show that every difference is harmless, that graph execution is
equivalent, or that the duck learned to stand. The original throughput rejection
and all archived numerical decisions remain unchanged.

This draft separates implementation correctness from learned-skill evaluation.
It proposes the evidence needed to return to training, not an executable launch
plan or a new tolerance-based pass. **No new GPU job is predeclared here.**

## Three separate questions

1. **Does the wrapper preserve the program?** Exact allocation/static bindings,
   complete restored inputs, BAM/FIFO updates, active masks, captured call order,
   physical stop predicates and first-failure freeze behavior must remain intact.
   Numeric noise cannot excuse a stale pointer, skipped motor update, changed
   stop threshold, leaked reset, unsupported contact, or missing evidence.
2. **Does the optimized runtime add unacceptable numerical error?** Measure eager
   repetition separately from eager-versus-graph differences, in physical units,
   at fixed inputs and over complete loaded-contact trajectories. A difference
   that also occurs in eager runs is context, not automatic permission to ignore
   it. Conversely, exact float identity is not by itself a physical quality metric.
3. **Does a trained policy meet its held-out skill gate?** Runtime qualification
   cannot answer this. A separately admitted optimizer-inclusive timing smoke
   must precede the original pilot; the existing five-second stance success gate
   still applies afterward. Preserve all earlier hopping and obstacle policies.

## Proposed evidence layout

| Layer | Required coverage | What remains exact |
| --- | --- | --- |
| CPU contract fixtures | Branches of every stop; first failure; FIFO; reset; active masks | Predicate output, executed-step accounting and frozen-state ownership |
| Restored-input physics | Both 64 and 512 worlds; friction-only, bilateral loaded support and unilateral support inputs | Complete input/static binding and discrete fixture identities |
| Closed-loop trajectories | Frozen policy and external commands; complete 2,500-step/5-second horizon or retained first stop | Protocol, seeds, initial state, event definitions and evidence completeness |
| Optimizer-inclusive smoke | The intended 512-world shape, real PPO update and checkpoint/evidence cost | Optimizer configuration, retained source, finite/capacity checks and time cap |

Friction DOF rows may be matched by unique DOF ID **only in a derived explanation**.
The new CPU report deliberately rejects other constraint types, duplicate/missing
DOF identities and extra rows. It is not a general contact-row matcher. Loaded
contact comparison needs a separately tested correspondence rule preserving
world, geom pair, contact frame, position and dimensionality; duplicate or
ambiguous matches must remain explicit, not silently sorted together. Keep every
original ordered table, including excluded contacts and its active-row addresses.

Before another runtime trial, freeze a manifest of actual input snapshots,
policy weights, held-out scenario seeds, batch sizes, repetitions and execution
order. Balance/interleave eager and graph repetitions to reduce order effects.
Keep sample construction separate from acceptance scoring. Select repeat count
from the declared inference plan and cost budget, not by running until a pass.
No resimulation, simulator mutation or graph admission is performed by the CPU
row-alignment report.

## Continuous quantities and decisions must not be conflated

Generalized acceleration DOFs 0-2 use m/s^2; DOFs 3-19 use rad/s^2.
Generalized forces use N for DOFs 0-2 and N*m for DOFs 3-19 in this pinned
free-root plus fourteen-hinge plant. Retain per-coordinate/world locations of
maxima. Do not pool these into a single unit, RMS score or undifferentiated
`allclose` check. Loaded-trajectory reports additionally need root translation,
orientation angle, linear/angular velocity, joint position/velocity, motor torque,
left/right normal support, and signed distance to each existing stop boundary.

Floating-point representation differences, numerical magnitude, structural
mismatches and integer/boolean event changes remain separate report fields.
Signed zero is a bit difference even when its absolute numerical error is zero.
For strict synthetic boundary fixtures, check below/equal/above behavior using
the actual production predicate; do not blur the boundary with a tolerance.
For continuous trajectories, retain stop reason and first-stop step per world,
along with the preceding physical margins. Do not erase an earlier failure by
comparing only the last common frame or by resetting a fallen world.

Near a discontinuity, a tiny continuous error can change a stop decision. Such
worlds are reported explicitly; they cannot simply be excluded after observing
the candidate. Statistical outcome comparison, if proposed, needs preregistered
absolute performance floors, noninferiority margins, confidence method and a
minimum effective sample count. Pairwise combinations sharing one run are **not**
independent samples; neither are cloned worlds with identical initialization.
Repeated batches must not be presented as thousands of independent training seeds.

## Required budget worksheet before launch

Every proposed numerical allowance must identify its physical quantity and unit,
engineering source, safety/accuracy rationale, applicable input domain, and
behavior when the eager baseline itself exceeds it. Candidate sources include
an explicitly allocated simulation accuracy budget, independently justified
control-resolution requirements, and a separately verified numerical error
analysis. The actual XL330 motor envelope and existing stop limits constrain
the experiment; they do not by themselves establish a permissible integration
error. We have no real-robot calibration data yet.

Do not choose a multiple of the observed 0.4025 N prefix support difference, the
new sub-micro generalized-acceleration differences, or a fitted eager percentile
as the engineering allowance. The baseline and candidate must both satisfy an
independent absolute budget. If they do not, report the unresolved numerical
accuracy rather than promoting the candidate because it resembles the baseline.

This worksheet is intentionally **not filled with guessed acceptance numbers**.
The current raw evidence constrains the investigation but does not supply an
independent physical error budget. Needed before any tolerance-based admission:

- Reviewed continuous budgets and discrete/event acceptance rules.
- Tested correspondence for loaded contacts and near-boundary fixtures.
- Frozen held-out loaded-input/trajectory manifest and replication plan.
- A bounded executable evaluator that preserves raw evidence and fails closed.
- A separate full-cost timing decision before enabling the optimizer smoke.

## Immediate next implementation and time boundary

The smallest useful next CPU task is to audit and, where missing, test the exact
below/equal/above stop predicates and their first-failure freeze semantics. Those
tests can improve wrapper qualification without choosing new tolerances or
spending GPU time. Do not alter thresholds or physics to make the fixtures pass;
diagnose a failure read-only before proposing a scoped correction.

The authorized work window ends **September 9, 2026 at 14:00 Shanghai**.
No new Duck work starts at or after that time. Retain completed evidence and
confirm no owned compute job is running. Leave protected system services and
100.98 untouched. This draft does not extend that window or authorize physical
motion, video, raw perception, a new pilot, or a changed curriculum success gate.

## CPU boundary-contract implementation

The next CPU task above is now implemented in
`tests/test_stance_stop_boundaries.py` and two additional Warp-runtime tests.
No production implementation, threshold, reward, action, observation or policy
file was changed. The 80 new predicate/accounting cases cover:

- The immediately adjacent float32 values below, equal to and above the tilt
  and height limits. The equality cases remain allowed for these strict stops.
- All fourteen torque and joint-velocity components, both signs, at adjacent
  **magnitudes**. Equality is allowed; the next greater magnitude is rejected.
- Each signed root-velocity axis and a three-nonzero-component vector whose
  realized float32 norms are exactly adjacent to 1 m/s. This distinguishes the
  full 3D norm from planar speed or a componentwise maximum.
- Each foot's support at adjacent values around 0.01 N and integer steps 49,
  50 and 51. Support equal to 0.01 N is rejected beginning at step 50, not before.
- Pre-step support checks at step 49 versus the refreshed post-step-50 check,
  torque-proposal rejection without a physics count, one-time failure penalties,
  and failure-versus-timeout precedence at the final episode boundary.

Two additional real CPU Warp wrapper cases inject a cached height immediately
below its limit or support exactly at its step-50 limit. The failed world executes
zero physical substeps. Its physical state, motor state, FIFO and observations
remain frozen across the next policy tick, its terminal record is preserved, and
the independent live sibling executes twenty actual CPU substeps. These are
explicit synthetic boundary/control-flow injections, **not** naturally occurring
low-height/support trajectories, a completed hold, CUDA evidence or learned
balance. Existing post-step-failure, selective-reset and timeout tests are reused
instead of duplicating those contracts.

The combined new and existing transition/Warp-runtime selection passed **115
tests in 7.41s** on the Mac with CUDA hidden. Assertions check exact booleans,
counters and frozen tensors, not a new numeric tolerance. `git diff --check`
passed. Exact-source Linux regressions are the remaining delivery check.

At pushed test source `e706cf53d37cf3605dae3fe838f8db674881e833`, all **598 Linux
stance, GPU-idle and foundation-campaign tests passed in 66.15s**, without skips
or deselection and with CUDA hidden. This adds 82 cases to the previous 516-test
selection: 80 predicate/accounting tests and two synthetic-boundary real CPU
wrapper cases. The diff from the previous evidence tip contains only tests and
this document; the entire production `src` tree is unchanged. No failed
predicate, threshold change or runtime fix was needed.

This CPU boundary task is complete. It strengthens wrapper qualification only;
it does not close the independent numerical-budget, loaded-contact, full
trajectory, optimizer-timing or learned-skill gates above. Do not rerun the
completed GPU probes or enable graph training as a consequence of these tests.
Any remaining work before 14:00 must be bounded, justified CPU review or evidence
closeout, not an invented acceptance tolerance or an unpredeclared GPU job.
