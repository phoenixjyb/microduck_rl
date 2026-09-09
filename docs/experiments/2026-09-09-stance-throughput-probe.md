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

## Executed result: faster collection, differential rejected

The final focused suite passed 16 tests in 8.24s. At exact pushed source
`f2ff27e434f4add09bba040bb69d6d7fe2927b26`, all 460 Linux stance, GPU-idle and
foundation-campaign tests passed in 46.72s, without skips or deselection.
Only then was `microduck-stance-throughput-f2ff27e434f4.service` launched.
Its sequential child completed successfully in 62.018s, with peak sampled GPU
temperature 55 C. Both protected system services remained inactive. Successful
process completion is not numerical acceptance: the recomputed decision is
`differential-rejected`, with training and full-pilot timing admission false.

| Worlds | Eager mean, seconds | Graph, seconds | Collection speedup | Eager repeat exact | Graph exact gate |
| --- | --- | --- | --- | --- | --- |
| 64 | 5.105381 | 2.761106 | 1.8490x | false | false |
| 512 | 14.280239 | 3.005673 | 4.7511x | false | false |

Each time measures 24 policy ticks, not an end-to-end training update. At 512
worlds, graph collection alone extrapolates to 1,538.904s (25.648 minutes) for
512 updates. Optimizer, journal and checkpoint overhead remain unmeasured;
this does not establish that a full pilot fits its 30-minute cap. The separate
instrumented GPU tick attributed 0.708s to eager `_forward` versus 0.086s for
graph at 512 worlds, consistent with reduced forward dispatch cost, but subject
to profiler overhead and the exploratory fixed case order.

### Read-only rejection diagnosis

After validating artifact hashes, weights-only CPU inspection of the four
retained ticks in every case found identical initial qpos/qvel, physics-step
counts and soft-limit masks within each pair. However, both eager/eager and
eager/graph comparisons already diverged numerically at the first post-Euler
boundary. At 512 worlds, eager/eager maximum first-boundary qpos difference was
`1.824743264475237e-12`, versus `2.0656412778180533e-12` for eager/graph.
These are real numerical differences, not merely serialized-file differences.

Maximum absolute differences over the complete four-tick retained prefixes:

| Worlds / comparison | qpos (mixed coordinates) | qvel (mixed coordinates/s) | Foot support (N) | Torque (Nm) | Actor observation | Critic observation |
| --- | --- | --- | --- | --- | --- | --- |
| 64 eager/eager | 3.79980e-5 | 0.00397968 | 0.0353193 | 0.000194165 | 0.000397967 | 0.00353193 |
| 64 eager/graph | 3.79980e-5 | 0.00398065 | 0.0353069 | 0.000194211 | 0.000398065 | 0.00353071 |
| 512 eager/eager | 0.000141054 | 0.00995786 | 0.402495 | 0.000485705 | 0.000995786 | 0.0402495 |
| 512 eager/graph | 5.07534e-5 | 0.00787467 | 0.171662 | 0.000383772 | 0.000787467 | 0.0171662 |

Across these prefixes, checked discrete fields were exact in all four pairs:
terminated, timed_out, live, executed_steps, episode_steps, boundary
physics_steps and soft_limit_mask, plus hard_limit, forbidden_contact and
warning flags. This limited agreement does not establish full-trajectory
contact-record equality, safety equivalence or learned capability. CPU diagnosis
left CUDA uninitialized. Eager numerical nonrepeatability is observed; its root
cause is not established, and this sample cannot establish graph equivalence or
identify a graph-specific regression. No tolerance was introduced and the
rejected decision remains unchanged.

### Retained evidence and next gate

`artifacts/evaluations/stance-throughput-f2ff27e434f4/` is retained on 100.100
and the Mac: 33 report-hashed files plus the report itself, 34 files totaling
63,619,159 bytes. This includes all six case records, 24 raw prefix files,
launch metadata, child log and deterministic decision. Both copies were
independently checked against exact inventory and every file hash.

| Evidence | SHA256 |
| --- | --- |
| launch.json | `d2503736fd4d0bbff2115e70b68d7d3c840fa39d8309525728321e1780025c5f` |
| decision.json | `ff7c37cc1f75a494f59da6aa089060687ea37100e914117c3bb94e5a254fbc58` |
| report.json | `b9c1ab3b2148a851389e87dc91c9a0dfda00734e11390d0e25ef955b99832373` |

Next, characterize baseline repeatability and the first divergence using the
immutable prefixes, then predeclare an independently justified equivalence
protocol before another GPU comparison. Do not choose tolerances merely to pass
these observed values or relax physical stop thresholds. Only accepted software
comparison evidence may lead to a separately predeclared optimized training
smoke that measures optimizer and durable-evidence overhead at the planned batch.
Graph remains opt-in and unused by normal training. No new weights were trained,
no prior skill policy was changed, and no pilot, video or physical motion was run.

The next chunk provides [hash-bound CPU prefix diagnosis and the same-input
forward isolation predeclaration](2026-09-09-stance-forward-repeatability.md).
The original rejected comparison above is unchanged.
