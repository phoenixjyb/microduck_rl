# B1-N forward-graph throughput probe

This follows the [completed disposable smoke](2026-09-09-stance-training-smoke.md).
The 30-minute full-pilot gate stays closed. No existing weights are loaded or
overwritten, and there is no optimizer, video or physical-motion path here.

## Hypothesis and smallest candidate

A local CPU cProfile of three warmed two-world collection ticks attributed
2.131s of 2.357s instrumented wall time to the 60 forward calls, including
1.872s within the stock forward implementation. This is CPU profiling evidence,
not a CUDA timing estimate; profiler overhead is substantial and this sample
does not measure checkpoint writing. Repeated collision/solver kernel launch
dispatch is therefore the first candidate to measure on GPU.

The opt-in `ForwardGraph` captures the same `mujoco_warp.forward(model, data)`
using the pinned mjlab pattern of scoped capture with garbage collection
suspended. It requires CUDA memory-pool and conditional-graph support, binds
every model/data array identity, pointer, layout and static configuration,
and rejects replacement before replay. Tensor contents may change in place;
model/data allocation and scalar configuration may not. Unknown binding types
fail closed. Capture must leave those bindings unchanged.

`WarpStanceRuntime` remains eager by default. Only explicit graph enablement
selects the candidate. The pre/post Torch/Warp synchronization, all solved-field
finite checks, contact decoding, BAM/motor/limit checks, Euler candidate/commit,
reward accumulation, terminal recording and selective reset are untouched.
The graph does not capture a policy tick or bypass any substep decision.

## One bounded differential measurement

One retained user service executes six cases sequentially: eager64, eager64
repeat, graph64, eager512, eager512 repeat, graph512. Eager repeatability must
pass before a difference can be attributed to the graph. Baseline timing is the
mean of its two runs. Each starts a fresh nominal plant at seed523; two zero-action
warmup ticks precede a full reset. It then executes 24 ticks using identical
seeded raw Gaussian action streams within each world-count pair. There is no
actor or learner in this probe. One explicitly synthetic world0 tilt is injected
after its first real Euler step, then left frozen through the following policy
tick before selective reset. Other worlds must continue unchanged by that reset.

Each tick hashes the complete returned physical boundaries, observations,
rewards, component sums, terminal/contact records, final motor/delay/control
state and committed qpos/qvel/time/warmstart. Reset-return records are also
hashed. Matching pairs must have bit-identical values, shapes and types for
every tick; no tolerance, rounding, skipped frame or changed acceptance threshold
may hide a difference. This is short differential software evidence, not a full
stance performance trajectory or independent re-simulation from the hashes.
Retain raw CPU tensor data for the first four ticks of every case (covering the
synthetic terminal, frozen tick and reset), with strict hashes and weights-only
CPU deserialization during supervisor verification. These prefixes aid diagnosis
if a comparison fails; they are not complete held-out trajectory bundles.

Step and reset wall times include existing synchronization and CPU result copies.
Hashing and one extra per-case cProfile tick are excluded from those times.
Profiles retain the 25 largest cumulative call costs. Timings do not include PPO
updates or durable per-tick training journals; extrapolating collection alone
cannot admit the full pilot. The fixed order is exploratory rather than a
randomized multi-repeat performance benchmark.

The source/runtime/asset/compiled-plant-bound CPU plan must be tested and pushed
before launch. Supply its independent SHA256 to the supervisor. Require the
exact `microduck-stance-throughput-SOURCE12.service` main PID, active state,
180-second RuntimeMaxSec and control-group kill policy. The child has its own
120-second watchdog and inherited shared GPU lease. Preserve the 600-second
closeout reserve before September10 07:30 Shanghai. Parent stays CPU-only;
only the child receives CUDA0 through an allowlisted environment. Monitor GPU
ownership, temperature below80 C, backend warnings and the two inactive protected
system services. Never change unrelated services or repeat the old smoke.

Retain each completed case before proceeding and a final recomputed decision,
log, telemetry and file hashes in `artifacts/evaluations/stance-throughput-SOURCE12`.
Refuse existing output; diagnose failures read-only, with no automatic retry.
An exact match admits only the short optimization comparison; even a faster
candidate needs a separately predeclared full training smoke before a pilot.

## Validation

The initial 27 focused CPU tests passed in23.35s. Real CPU tests verify immutable
bindings, pointer/static-option rejection, preserved graph-dispatch safety checks,
and repeatable first-terminal/freeze/reset hashes. The fake CPU graph-dispatch
test is explicitly not CUDA graph evidence. Matrix and supervision tests use
synthetic fixtures. Full regressions and exact-source Linux checks follow before
any actual GPU graph execution.

A broad Mac regression attempt reported456 passes, two30-second subprocess
import timeouts and the existing single deselected Mac process-group test.
The observed Mac load average was49.99; no assertion or numerical failure was
reported in those two subprocess checks. Both passed when rerun serially in
21.34s without increasing their limits. The six-case/raw-prefix focused suite
also passed16 tests in66.29s. Final binding-owner checks and the full Linux
suite must pass before the source-bound GPU comparison; no timeout is waived.
