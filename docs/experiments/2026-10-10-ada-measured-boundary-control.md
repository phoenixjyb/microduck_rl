# New measured boundary and identical-bank solver control preparation

Base `7ca75aca57613cc803a42f62362d61a67fb9ec58`, exact branch
`feat/athletics-obstacle-curriculum`. This follows the exactly received
[saved prepared-pose response](2026-10-10-ada-saved-pose-solver-response.md).
The present change implements **import-inert packet primitives and synthetic
tests only**. There is no executable launcher, GPU capture or identical-bank
CPU solve in this change. Simulator/PPO admission remains closed. Libraries,
driver, plant, motor inputs, solver options and existing gates stay frozen.

## Why a new measurement is necessary

The historical authenticated Ada packet retains seven prepared state inputs,
347 model arrays and solved outcomes, but not every actual measured collision
pose or the complete immediately-before-solve Data bank. The completed CPU
response supplies eleven prepared poses and generates a new CPU manifold. Its
small descriptive residuals do not establish identical GPU/CPU solver inputs.
Do not reconstruct historical GPU inputs from solved forces or accelerations.

A future measured run starts from those authenticated seven state/motor/model
inputs but computes fixture Warp **GPU kinematics exactly once**. Its eleven
actual before-collision poses become the reference for that new run only. This
is a new measured replay, not a recovered historical Ada boundary. Allocation's
internal native kinematics remains counted separately. There is no integrator.

## Predeclared measurement and control semantics

Use the unchanged ordinary forward stage order, through solve, retaining the
same five complete boundaries as the preceding response: before collision,
after collision, after construction/transmission, immediately before solve
after factorize=True acceleration, and after solve. Keep position
factorize=False semantics and all ordinary sensor/energy/force stage ordering;
stop before acceleration sensors/integration. The helper prepends one fixture
kinematics call to the already tested stage-order helper; it is not a launcher.

Capture every allocated Data array at all five boundaries: 114 per bank, 570
raw leaves, including inactive contact/constraint capacity. Require actual
complete dtype, logical shape, stride, expanded NumPy shape/dtype, raw length,
finite values, and exact pose/state stability. Maintain the existing contact
candidate/address and six-field solver write fences. Reconstruct resultants
under the established ordered arithmetic only; no contact normalization,
spatial pairing, altered forces, rounding or tolerance-based acceptance.

Bind all 347 model arrays separately by exact bytes/layout against the prior
prepared bank. Additionally serialize the **complete** actual Model/Data
dataclass/container/scalar topology and values: options, enums, callbacks,
solver dispatch statics and all six Data capacities, not a selected whitelist.
Floating static values preserve signed zero; nonfinite/callable/unsupported
objects fail closed. Portable identities exclude pointers and device identities;
those remain separate within-process/device checks. A manifest must equal a
fresh complete actual walk: a self-consistent tree that omits real scalar fields
is insufficient. Array bytes are not supplied by the scalar serializer.

The frozen two-world capacities are nworld2, naconmax256, naccdmax256, njmax512,
njmax_pad512 and njmax_nnz10240. Canonical layout comes from the authenticated
array report, never from an unchecked consumer-supplied layout. Unexpected
static/backend differences must be diagnosed, not silently excluded or repaired.

For a separately authorized CPU control, allocate fresh CPU Model/Data and
prove the same complete model bytes and statics. Restore **all 114 actual GPU
before-solve Data fields** exactly, preflighting every layout, finite leaf and
complete Model/Data statics before the first write. Verify every post-copy raw
byte/layout, then call only the frozen public solver once. Do not recompute
kinematics, collision, construction, factorization, sensors, actuation or forces.
The frozen public solve allocates fresh SolverContext; no persistent GPU context
is reconstructed. Its warmstart copy direction stays destination qacc from
source qacc_warmstart (or qacc_smooth when disabled).

Both branches must respect the unchanged six-output write fence. Compare all
six complete outputs, reporting exact hashes, byte equality, counts of scalar
elements with differing bytes and ordered CPU-minus-GPU maximum absolute differences over
the full allocated capacity. Validate all three complete input/output banks,
not just active solver slices. No RMS/BLAS reduction, tolerance, solver tuning
or admission inference. Identical inputs do not imply identical CPU/GPU kernel
execution; retain that distinction explicitly.

## Current implementation boundary

`ada_measured_boundary_packet.py` implements complete portable static binding,
new measured pose-reference checking, stage-order composition, exact complete
CPU Data restore and full solver-bank comparison. It imports no NumPy/Torch/
Warp/MuJoCo runtime at module import and has no CLI or runtime allocator.
Mocks exercise the runtime helper contracts without executing physics.

The present primitives are **not sufficient execution authority**. Still needed:
closed source-bound CPU/GPU runners, complete model/runtime/topology bindings,
strict retained packet receiver, cache/device/source provenance, resource and
service receipts, watchdog/lease and foreign-workload guards, independent
review, and an exact-source predeclaration for each launch. No hidden third
GPU profile has been added to the existing two-profile supervisor.

Future GPU diagnostic must inherit the existing single-Duck advisory lease,
owner/child/service deadlines and GPU occupancy/temperature limits. Preserve
Grounding DINO/FilmBrain and inactive protected services. CPU control gets its
own explicit hidden-CUDA service and fresh private cache. Never overlap Duck
GPU runs. Do not begin either launch unless it can finish safely before
07:00 Asia/Shanghai; reserve ten minutes for closeout. No PPO, raw perception,
policy promotion, service restoration or physical robot motion is authorized
by this preparation.

## Source-only verification plan

Run the six-file suite on Mac, review independently, commit/push the exact
feature source, clean-fast-forward native100.100, then run the same contract
suite in a CPU-only user service. Cap60s/MemoryMax2G/CPUQuota200%/TasksMax64,
Nice10/LimitFSIZE16M/LimitCORE0/Restart=no/KillMode=control-group/
TimeoutStopSec10/Type=exec/RemainAfterExit=yes, thread pools1, literal
CUDA_VISIBLE_DEVICES empty. This suite initializes no simulator physics.
Retain JUnit and actual service receipts under the unique source8 prefix in
`artifacts/tools/ada-measured-boundary/`; require equal complete testcase
multisets and zero failures/errors/skips. Physics needs a later separate gate.

## Preparation review

Owner six-file suite passed408 cases in9.20s after the final schema correction;
git diff whitespace checks passed. Independent Luna read-only review passed345
focused cases in5.04s and found the prior complete-static/capacity/layout gaps
closed. Its one remaining output-label issue was corrected before freezing:
`differing_scalar_elements` counts scalar elements whose bytes differ, not byte
offsets. The source-only imports/AST checks and tests do not execute physics.
Native at17:27 UTC retained clean base7ca75aca, Dino PID1592/946MiB as the
only GPU compute owner,43C/0%, and both protected services inactive in system
and user scopes. This snapshot is not authority for a later GPU launch.
