# B1-N forward repeatability: diagnosis and next isolation test

The [six-case throughput probe](2026-09-09-stance-throughput-probe.md) remains
`differential-rejected`. Its faster graph candidate is not enabled in training.
The initial chunk added an offline, tested diagnosis of immutable evidence and
specified the next bounded experiment. The v2 implementation and completed
forward-only GPU isolation are recorded below. No weights were trained.

## Reproducible CPU diagnosis

`python -m mjlab_microduck.stance_prefix_diagnosis` requires an independently
supplied source-report SHA256. It verifies the exact 34-file inventory, every
file hash, the original deterministic decision, every prefix descriptor and
captured-value hash, then deserializes with weights-only loading onto CPU.
Frames are checked against the existing physical-state schema; nonfinite values
anywhere, unsupported types and unverified files fail closed. No simulator is
run. Comparison preserves ordered contact records, signed-zero bit differences,
numerical differences, discrete values and structural mismatches separately.
No tolerance, sorting of contact rows or rounding can turn a failure into a pass.

The derived output must be fresh and outside the immutable input directory.
It never changes the original decision and always reports diagnostic-only,
no graph equivalence, no training admission and no physical-motion authority.
All per-field maxima retain their concrete tensor index and ordered path;
chronological boundary divergence is separate from dictionary field order.
This is a comparison report, not a simulator-resume state or causal proof.

Run from the repository root with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python \
  -m mjlab_microduck.stance_prefix_diagnosis \
  --input artifacts/evaluations/stance-throughput-f2ff27e434f4 \
  --report-sha256 b9c1ab3b2148a851389e87dc91c9a0dfda00734e11390d0e25ef955b99832373 \
  --output artifacts/evaluations/stance-throughput-f2ff27e434f4-prefix-diagnosis-v1.json
```

The actual Mac result is 1,076,349 bytes, SHA256
`898403098bb6435419a3dbcc871553dd472493b905fdea01c29ca0dc9d8ecb0b`.
Its original report and all original raw files remain unchanged. Source-bound
Linux regeneration produced the identical bytes and hash, as recorded below.

Observed in all four pairs (64 eager/eager, 64 eager/graph, 512 eager/eager,
512 eager/graph):

- The complete initial retained boundary is bit-identical, not just qpos/qvel.
- The first differing retained boundary is tick 0, boundary 1, immediately after
  the first integrated step and its post-step solve. Differing fields are qpos,
  qvel, root/joint velocity and actor/critic observations. The pre-Euler solved
  acceleration is **not retained**, so this does not attribute the discrepancy
  to Euler itself rather than its input solve.
- All compared integer/boolean/string/None values and structures match across
  the four complete retained ticks, including terminal contact metadata. Floating
  forces and states still differ. No metadata permutation was observed here.
- Four ticks cover at most 40 physics steps (80 ms); the support-loss stop becomes
  active at step 50. The injected world-0 tilt tests a synthetic terminal, not
  a natural fall or a successful standing policy. Later trajectory behavior,
  support-loss behavior and full five-second acceptance remain untested here.

## Source inspection: hypotheses, not a proven root cause

The installed, pinned MuJoCo Warp source contains these relevant operations:

- `constraint.py`, `_friction_dof`: active friction rows receive indices through
  `wp.atomic_add(nefc_out, worldid, 1)` (line 1394 in this exact source).
- `solver.py`, `update_gradient_grad`: the gradient squared norm is accumulated
  with `wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)` (line 2256).
- The dense constraint-force accumulation at lines 1999-2008 is a sequential
  row loop. Do not cite the sparse atomic force path as this plant's dense path.

These are candidate assembly/reduction sites to examine. Their existence alone
does not prove that a race, contact order, reduction order or graph implementation
caused this run's divergence. The retained prefix lacks the necessary pre-Euler
solver/constraint state. Do not patch installed kernels or alter solver options
on this evidence alone.

| Installed source | SHA256 |
| --- | --- |
| mujoco_warp/_src/constraint.py | `b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53` |
| mujoco_warp/_src/solver.py | `bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a` |
| mujoco_warp/_src/forward.py | `c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3` |

## Predeclared next experiment: same-input forward only

The original v1 design below was not launched. CPU implementation inspection
found reset-only `nefc=nf=0`: it would test unconstrained dynamics. The v2
amendment and earlier user-requested cutoff are declared below before GPU launch.

Protocol: `football-b1n-same-input-forward-v1`. Implementation and its CPU tests
are still required; this document is **not a launch-ready source-bound plan**.
Before any launch, bind the reviewed implementation commit, runtime hashes,
compiled plant, source-snapshot hashes, output path and service to the plan.

Use one owned runtime at a time, first 64 worlds then 512, seed 523, unchanged
nominal plant and zero controls. There are no policy ticks, BAM updates, Euler
steps, optimizer steps, trajectory resets between trials, or checkpoint inputs.
After normal runtime initialization, retain a complete model/data array-content
snapshot and bind every array pointer, layout and scalar/static option. This
includes warmstart, model damping/friction and scratch contents, not just qpos.
Before **each** measured forward call, restore those same contents in place,
synchronize both streams and require an exact input-content hash. Graph capture
must preserve the bound allocations; restore again after capture before any
measurement. Do not assume capture is a no-op for dynamic contents. Unknown
fields or an unbounded snapshot fail before trials; do not omit them to fit.

At each world count run eight calls in the fixed order E, G, G, E, G, E, E, G
(four eager and four graph), with identical restored inputs. Eager calls the
unchanged stock forward; graph calls the same complete captured forward. Keep
the existing fresh finite, capacity and active-contact checks after each call.
Never substitute a separately split phase graph for the candidate under test.

Retain each input hash and post-call raw output before proceeding:

- qpos/qvel/time and qacc_warmstart, to check no integration/input drift;
- qacc, qacc_smooth, qfrc_bias, qfrc_actuator, qfrc_constraint and solver_niter;
- active contact records and decoded forces, preserving original ordering;
- nefc and each world's active constraint type/id/J/D/aref/force/state rows,
  so a row-order difference is distinguishable from changed solved values.

Bind the actual fields against the pinned runtime during implementation. Retain
raw arrays and their hashes, not only maxima. Keep inactive scratch bytes in the
input snapshot; do not mislabel inactive scratch as an active physical quantity.
The snapshot covers owned model/data, not the entire CUDA process or ephemeral
solver allocations. Both arms use the same objects to avoid confounding fresh
allocation differences with graph dispatch.

Compare all six within-eager pairs, all six within-graph pairs and all sixteen
cross-arm pairs at each batch size, with no numerical tolerances. If any input
hash differs, classify `invalid-input-control` and stop. If same-input eager
outputs vary, classify `forward-baseline-nonrepeatable`: integration and learning
are then unnecessary for variation to occur, but the precise kernel cause still
needs evidence. If eager repeats match and graph differs, classify
`candidate-difference-requires-diagnosis`. If all outputs match, classify only
`same-input-forward-repeatable-in-this-sample`. No classification admits graph
training, restores the rejected trajectory result or validates motor safety.

Use one retained sequential user service, the existing shared GPU lease,
120-second child watchdog, 180-second service cap, 80 C stop threshold and
600-second closeout reserve before September 10 07:30 Shanghai. Require idle
GPU0 and both protected system services inactive before and throughout the job.
Cap each batch's input snapshot at 256 MiB and the retained experiment at 1 GiB;
abort cleanly rather than trimming evidence. Preserve unrelated workloads.
No automatic retry, raw perception, MP4, installed-library patch or physical
motion. Diagnose any failure read-only first.

## Path back to training

The immediate aim is to separate same-input forward variation from accumulated
trajectory divergence, not to make a strict test pass by changing its tolerance.
An eventual numerical-equivalence protocol must have independently justified
engineering-unit error budgets, replicated held-out inputs, unchanged discrete
stops and complete trajectory coverage. The 0.4025 N observed support difference
is not itself an acceptable tolerance. No such tolerance-based gate is declared
or passed by this document. After accepted optimization evidence, run a separate
optimized training smoke with full optimizer and evidence costs before deciding
whether the original 512-world, 512-update pilot fits its 30-minute cap.

## Validation and delivery

The initial 18 diagnostic CPU tests passed in 17.22s. After adding explicit
missing-boundary coverage, the combined 87 diagnostic, throughput, attempt-trace
and contact-evidence tests passed in 31.04s. These include bit-versus-numerical
comparison, nonfinite rejection, contact ordering, chronological divergence,
hash-before-deserialization, input immutability, deterministic regeneration and
no-admission behavior. Synthetic bundles are test fixtures, not physical evidence.
Markdown HTML rendering, table structure, local links and `git diff --check`
passed.

At pushed source `8feabd98e46047d0092e610349fb1b806758ed27`, all 479 Linux stance,
GPU-idle and foundation-campaign tests passed in 48.48s without skips or
deselection. The 19 new tests are CPU diagnostic checks, not CUDA evidence.
Independent regeneration on 100.100, with CUDA hidden, produced the identical
1,076,349-byte diagnosis and SHA256 shown above. The analyzer's own source SHA256
is `8533a247c111419fdfa17572024bba62eb3a0691f5ff41c7dc88166c273b3c8b`.
All three inspected installed source hashes also matched on 100.100.
The report is retained outside the immutable probe directory on both machines.

Delivery state: CPU diagnosis complete and cross-machine reproducible;
same-input GPU isolation specified but not implemented or launched. The numerical
root cause, graph equivalence, optimized training smoke, full pilot timing and
learned stance capability remain open. No weights or protected services were
changed by this chunk.

## Implementation amendment before GPU launch (v2)

`stance_forward_probe.py` implements `football-b1n-same-input-forward-v2`.
The user has replaced the previous working window with **September 9 14:00
Shanghai (06:00 UTC)**. This module enforces that earlier deadline independently
of the older shared helper, retaining a 180-second service budget and 600-second
closeout reserve. The existing continuation schedule has the same new cutoff.

CPU evidence at two worlds showed reset `nefc=[0,0]`, `nf=[0,0]`, and zero foot
support. One genuine BAM proposal for the delayed nominal target, zero velocity
and effort targets, was accepted with exactly zero torque. After committing that
control and solving forward once, `nefc=nf=[14,14]`, physics steps remained zero
and time remained zero. This preparation is now mandatory before the snapshot:
exactly one motor preparation per batch, no policy tick or Euler call. Trials
then restore the same fully prepared model/data contents, with **no further BAM
updates**. Every recorded output must contain 14 friction rows per world.
This exercises the pre-Euler friction solve, not a loaded-contact trajectory or
stance performance. v1 was superseded before launch, not rescored as passing.

The first CPU snapshot test rejected a metadata mismatch: `wp.clone` packs the
storage strides of singleton broadcast axes (for example `model.body_mass`),
although logical bytes were unchanged. Snapshot metadata now records the original
simulator allocation layout, while the packed CPU backup supplies its exact
logical bytes. The complete original allocation/static signature is checked
before every restore. No byte, layout or input-hash check is waived.

The launch plan binds the source, frozen dependency trees, compiled plant,
snapshot schema and hash protocol. CUDA-specific scratch snapshots cannot have
known hashes before GPU allocation: the child must exclusively retain and hash
the actual full snapshot **before any measured call**, and independently check
that same hash after every restore. The CPU supervisor rehashes the retained
snapshot and raw outputs, checks every receipt and recomputes all 28 pairs per
batch. Preparation is documented separately from the eight measured calls.

All 461 owned model/data arrays are enumerated in the current two-world CPU
binding (15,011,239 logical bytes). The 256 MiB per-snapshot and 1 GiB total limits
remain unchanged and are checked on actual allocated sizes, not extrapolated
from two worlds. Outputs retain active rows and contacts in original order.
The graph is opt-in only inside this disposable diagnostic. Existing runtime
defaults, installed libraries, policies, rewards and physical stop thresholds
remain unchanged. The post-run numerical classifications and no-admission
boundaries are the same as v1.

Implementation validation initially exposed the stride mismatch (one failure and
six dependent fixture errors), then the empty-constraint nonfinite-test fixture
(one failure). Read-only CPU inspection preceded both changes. After v2 motor
preparation, all 11 initial focused tests passed in 9.13s. The earlier-deadline
test and broader exact-source validation must pass before launch.

The completed 16-test forward-probe suite plus throughput, prefix-diagnosis and
runtime regressions passed all 63 checks in 15.90s on the Mac. Input-hash/order
tampering, missing raw files, active nonfinite values, empty-friction trials,
storage ceilings and the earlier cutoff all fail closed. Exact-source Linux
regressions remain the mandatory next check before the v2 GPU service.

## Completed v2 same-input isolation

Implementation source: `ee3e856286e2228edb821ce1e5a89f4940c674c8` on
`feat/athletics-obstacle-curriculum`. Before launch, the exact-source Linux
stance, GPU-idle and foundation-campaign suite passed **495 tests in 54.78s**,
without skips or deselection, with CUDA hidden. The preceding pending statements
are historical prelaunch records, not the current execution state.

The retained user service `microduck-stance-forward-ee3e856286e2.service`
completed with `Result=success`, `ExecMainStatus=0`, and `MainPID=0`.
Its child took 13.662719s; peak sampled GPU temperature was 48 C. The service's
`active/exited` state is its retained completion record, not a running workload.
The post-run GPU check found no compute PIDs, 0% utilization, 12 MiB allocated
and 45 C. Both protected system services remained inactive and were not changed.

Retained on both the Mac and 100.100:
`artifacts/evaluations/stance-forward-ee3e856286e2/` contains **23 files,
105,383,394 bytes**. Exact inventory and every raw file SHA256 were verified on
both machines; the Mac inspection did not initialize CUDA. The inventory includes
both complete input snapshots, sixteen raw output tables, both batch reports,
launch metadata, child log and the final report. No original evidence was edited.

| Evidence | SHA256 |
| --- | --- |
| `launch.json` | `66af91aadd2f8b19d7c3de0dad03e2beb4a57c3836d8c6f0cdff575ac8515ea7` |
| `report.json` | `8d9222a8d02fb1a56d16de4fab44cc19918a42331c6f91bf3e0429dc24865de9` |
| 64-world input logical-value hash | `9799e01cce3d82759a82f2a764ad893d2db504d2f556df91e916e535031d009b` |
| 512-world input logical-value hash | `d9c0815108768e166fc820fe6e89d40971fe823f25c68e4c8a0beade49348ac4` |

The input snapshots contain 21,308,455 and 66,810,919 logical array bytes,
respectively. Their value hashes are not their serialized `.pt` file hashes;
the latter are separately bound by `report.json.files` and batch descriptors.

Both batches classified **`forward-baseline-nonrepeatable`**. At each size, zero
of six eager/eager, zero of six graph/graph and zero of sixteen cross-mode pairs
were bit-identical. Only the `constraints` and `dynamics` groups differed.
Positions, velocities, time, warmstart, ordered contacts and solver counters
were unchanged. All worlds had 14 friction rows and one solver iteration.
There was no integration or optimizer step. This establishes that variation can
occur within forward dynamics, without trajectory accumulation or learning.

Maximum absolute differences across all 28 pairs per batch:

| Field | 64 worlds | 512 worlds |
| --- | ---: | ---: |
| `qacc` | 4.036924110550899e-7 | 6.820483804403921e-7 |
| `qacc_smooth` | 7.397179615509231e-7 | 1.155451514023298e-6 |
| `qfrc_bias` | 1.862645149230957e-9 | 3.725290298461914e-9 |
| `qfrc_actuator` | 0 | 0 |
| `qfrc_constraint` | 1.1326322102434006e-9 | 1.4861247787933962e-9 |

These are generalized-coordinate vectors with mixed translational and rotational
components, not a single shared physical unit or an acceptance tolerance.

### Secondary CPU row-alignment inspection

The original raw comparison preserves row order. A separate read-only inspection
verified that every table has exactly 14 unique DOF IDs of friction type 1, then
sorted copies by DOF ID. Comparing calls 1 through 7 against call 0, the aligned
`type`, `id`, `J`, `D`, `aref` and `state` fields matched exactly at both sizes.
Thus the raw Jacobian and diagonal-weight discrepancies in this sample were row
permutations, not changed coefficients. Aligned constraint forces still differed:
140 scalar values at 64 worlds (maximum 9.895175789864652e-10), and 2,378 at 512
worlds (maximum 1.4861247787933962e-9). These counts aggregate seven comparisons;
they are neither independent trials nor counts of failing worlds.

The alignment is explanatory only: it does **not** modify the recorded gate,
sort simulator inputs, waive bitwise mismatches, or establish equivalence. Bias
and smooth-acceleration differences also occur upstream of the constraint solve,
so constraint row reordering alone is not a complete causal explanation.

Pinned `smooth.py` SHA256
`63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f`
contains `_cfrc_backward` at line 1219 and an atomic parent-force accumulation at
line 1233. `rne` calls this backward pass before `_qfrc_bias` (lines 1273-1274).
That is a concrete upstream candidate, **not proof that this particular kernel
caused all observed differences**. Installed library source was not patched.

Next bounded work is a tested, hash-bound CPU report for this secondary
alignment and a reviewable numerical-equivalence protocol. Such a protocol must
separate row identity, continuous physical quantities, discrete stop decisions,
full loaded-contact trajectories and eager baseline variability. It must not set
tolerances merely to pass these observed numbers. No further GPU probe is
required to preserve or explain the current result. Any later GPU experiment
requires its own predeclaration, idle lease and enough time before the **September
9 14:00 Shanghai** cutoff. Graph training, optimized optimizer-smoke timing, full
pilot admission and learned football balance all remain unaccepted.

Evidence-closeout validation: all 16 forward-probe tests passed again on the Mac
in 11.05s with CUDA hidden. Markdown rendering and local-link checks passed (the
new section has two tables and eleven rows; the complete document has three
tables and fifteen rows). `git diff --check` passed. A fresh remote readback
still showed a clean implementation worktree, the completed service with no PID,
no GPU compute process, and both protected system services inactive.

## Reproducible secondary row report

`python -m mjlab_microduck.stance_forward_diagnosis` now implements the read-only
inspection. The completed-report path added to `stance_forward_probe.verify`
accepts an independent report hash, verifies the complete manifest before any
tensor load, rechecks descriptors, input value hashes and replay controls, and
recomputes the original decision without removing or rewriting `report.json`.
The existing pre-report live-supervisor verification path is unchanged.

The derived report retains all seven call-0 comparisons at each batch size:
raw row differences, differences after unique-DOF alignment, bit-versus-numeric
distinctions, and per-coordinate dynamics maxima with world/DOF provenance and
separate translational/rotational units. It rejects ambiguous or non-friction
rows rather than attempting general contact matching. It always returns
`diagnostic-only`, with graph equivalence, training admission and physical-motion
authority false. No simulator or optimizer runs during this analysis.

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python \
  -m mjlab_microduck.stance_forward_diagnosis \
  --input artifacts/evaluations/stance-forward-ee3e856286e2 \
  --report-sha256 8d9222a8d02fb1a56d16de4fab44cc19918a42331c6f91bf3e0429dc24865de9 \
  --output artifacts/evaluations/stance-forward-ee3e856286e2-row-diagnosis-v1.json
```

The Mac generated **141,740 bytes**, SHA256
`82246f34a066dda2792b8c4ce5d28aeb89d72209bba920dcf0df22cfa3296eee`.
Both batches reproduce the force-only aligned differences documented above:
140 and 2,378 differing scalar values, respectively. Original raw decisions
remain unchanged. Exact-source Linux tests and independent regeneration are
required before claiming cross-machine delivery of this new report.

The [qualification review draft](2026-09-09-stance-equivalence-review.md) separates
wrapper contracts, forward arithmetic, loaded trajectories and trained-skill
acceptance. It does not invent tolerances from observed errors or authorize a
new GPU trial. The next CPU-only step is coverage of exact stop-boundary and
first-failure-freeze contracts, while the independent numerical budget remains
an explicit prerequisite for a future tolerance-based gate.

Local implementation checks: the initial combined forward suites passed 37
tests in 21.05s. After correcting the secondary pair label to `cross` whenever
the eager reference is compared with a graph call, the five-suite regression
selection passed **87 tests in 22.27s**. New coverage includes all-input hash
checks before loading, report semantic revalidation, missing/extra/symlink files,
unique-row and finite checks, signed zero, immutable original decisions,
coordinate units, deterministic regeneration and safe output placement.
Both Markdown documents rendered with balanced tables and valid local links;
`git diff --check` passed. These are CPU checks, not new runtime acceptance.

At pushed implementation source `a67d9f4304c6846ace404550c85686893e160ebf`, the
exact-source Linux stance/GPU-idle/foundation selection passed **516 tests in
64.20s**, without skips or deselection. Independent CPU-only regeneration on
100.100 produced the identical **141,740-byte** report and SHA256
`82246f34a066dda2792b8c4ce5d28aeb89d72209bba920dcf0df22cfa3296eee`.
All source evidence was rehashed before and after analysis on each machine.
The new derived report is durable outside the immutable input directory on both
hosts; it was generated independently, not copied as a substitute for validation.
The remote checkout remained clean, GPU0 had no compute PIDs (0% utilization,
12 MiB, 45 C), and both protected system services remained inactive.

The row-diagnosis chunk is complete. No new policy, GPU run, tolerance-based
admission or single-kernel causal claim resulted. Read-only review of existing
tests found broad stop coverage and the step-49/50 support-grace check, but no
systematic adjacent-float below/equal/above sweep of every continuous stop.
That focused CPU boundary coverage is the next bounded task; do not rerun this
completed diagnosis or mutate its evidence to obtain another outcome.
