# October 6 whole-tree CRB schedule investigation

## Scope and starting evidence

Continue only within the authorized window ending **2026-10-06 08:00
Asia/Shanghai (00:00 UTC)**. Native work is restricted to 100.98 via
`gw98-direct`; 100.100, packages, drivers, installed kernel source, optimizer,
raw perception, physical motion and protected-service restores are excluded.
The original full-window replay remains rejected. Preserve its exact gates and
all retained evidence; a source planner or isolated control cannot admit training.

Start from clean fork branch `feat/athletics-obstacle-curriculum` at
`f98a694033a5402e4c9afb2f87f1ff21c7112a6f`. The preceding
[serial control](2026-10-06-crb-serial-schedule-control.md) at execution source
`c9708cf68fdb812733f7a9fddd336c36a8241d26` produced 32 byte-identical outputs
matching the independently predeclared CPU candidate. The concurrent control
produced seven distinct outputs from the same derived fixture. That fixture
contains root inertia and three completed-forward child composite inertias;
other bodies are synthetic zeros. It is not actual kernel-entry capture.

The next source-only slice permits exactly three new paths: this declaration,
`src/mjlab_microduck/stance_crb_level_plan.py` and
`tests/test_stance_crb_level_plan.py`. All previous source leaves stay unchanged.
The native checkout remains at C970 until any separately declared source sync.
No new native run is authorized by this planner's source tests alone.

## Fixed topology and schedule

Use the same pinned `_crb_accumulate` kernel contract: read the body's parent,
return if that parent is zero, otherwise atomically add the body's current
ten-component composite inertia to its parent's working value. CRB input and
output alias the same working array. Children must be accumulated before parents.

The exact seven-field canonical topology JSON plus newline has SHA256
`2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19`:
64 worlds, 16 bodies, `nq=21`, `nv=20`, `nu=14`, parent vector
`[0,0,1,2,3,4,5,1,7,8,9,1,11,12,13,14]`, and these reversed levels:

| Original level | Fixed launch groups | Nonzero parent targets |
| --- | --- | --- |
| `[6,15]` | `[6,15]` | `[5,14]` |
| `[5,10,14]` | `[5,10,14]` | `[4,9,13]` |
| `[4,9,13]` | `[4,9,13]` | `[3,8,12]` |
| `[3,8,12]` | `[3,8,12]` | `[2,7,11]` |
| `[2,7,11]` | `[2]`, `[7]`, `[11]` | `[1]`, `[1]`, `[1]` |
| `[1]` | `[1]` | none; parent-zero no-op |
| `[0]` | `[0]` | none; parent-zero no-op |

Preserve every original level boundary, stable body-ID order and both no-ops.
Only level four requires splitting: its three children target the same parent.
Nine launches are minimal **within those level boundaries**, because that level
requires three distinct single-writer rounds. Do not claim global optimality
across fused/reordered levels, deterministic execution of other kernels, or
unchanged timing. Unique logical write destinations are a schedule property,
not observation of actual atomic order.

## Bounded planner and acceptance

Before hashing or traversing, require a plain dictionary with exactly seven
fields, plain integers (not booleans), exact dimensions, a 16-element parent
list and seven nonempty level lists with at most 16 IDs each. Then authenticate
the canonical whole-topology hash and reuse the frozen exact-topology validator.
No input-dependent unbounded search or alternate topology is permitted.

Return detached, owned lists and explicit witnesses for all 16 bodies occurring
once, child-before-parent order, unchanged original boundaries, stable sibling
order, unique nonzero parent targets in each group, retained no-op groups, and
the per-level lower bound. Check the exact nine-group output independently in
tests. Reject extra/missing fields, altered hashes, dimensions/parents/levels,
booleans, oversized lists and duplicate/missing IDs. Test that caller mutation
cannot change an already returned plan and output mutation cannot change input.

Actual launch execution, actual atomic-order observation, original-pair
acceptance, cause proof, full-window qualification, training authorization and
physical acceptance remain false. Source tests establish only the planner's
contract. A future whole-tree control requires a separate declaration, exact
source/pinned runtime, authenticated whole-cinert fixture provenance, CPU
prediction, cooperative lease, caps, independent checking and retained results.

## Results

The source-only planner passed **24 focused CUDA-hidden tests**. An independent
read-only worker repeated that file successfully. Owner integration with the
unchanged serial checker/supervisor passed **45 tests in 6.28 s**; Ruff lint and
format checks and `git diff --check` passed. The routing skill separated the
bounded implementation from independent review and owner integration.

No whole-tree native control, fresh compilation, array-alias witness, replay
qualification or training result is claimed by these checks. The native source
and completed serial artifacts remain unchanged at C970.

## Separately predeclared full-tree fixture/control

Freeze the next source slice at planner commit
`a3b149d4fef78c1bf1a23a1b44699fabdb599a1a`. Permit only this declaration plus
the new pure full-tree fixture/checker and supervisor, and their two focused
test files. The planner and all earlier implementation/test leaves remain frozen.
Native source sync starts only from clean C970 and uses the exact reviewed
source bundle; native tests must qualify the new source before a fixture or run.

### Fifty predecessor files and CPU-only fixture

Authenticate the unchanged 38 predecessor files from the earlier control, plus
all seven serial-run files, three serial-test files, the serial owner packet and
its separate CPU closeout terminal: **50 predecessor files**. Bind both packet
hashes before JSON decoding:

- Serial owner: `186dea37ef643c510bf3285a5c2a9785dd91c1152a64587fc95fec085db9af02`.
- CPU closeout terminal: `f431fc6f3677fcba16ba530351b2fd970f4d8cf276e2ed76f42978a3cbf7062d`.

Historical terminal invocations and source leaves remain historical evidence;
their already cleared services are neither restarted nor required live. The
serial document in F98 is an evidence-only successor, not the C970 execution
document. Verify historical leaves against the exact historical commit, and
new leaves against the new committed source separately.

The new fixture mode is CUDA-hidden CPU-only. After whole-byte authentication,
reuse the frozen original body/preparation/constructor/early-trace validators
and weights-only CPU loading. Reproduce exactly the original strict rejection
`paired rollout semantic state exactness`; do not weaken it or replace original
trajectory binding with a trace-only projection. Recompile the topology on CPU
without a physics step and bind its seven-field launch-topology hash.

Extract event zero, `scheduled-pre`, step zero, from both authenticated original
traces. Require every complete `persistent.cinert` snapshot to be finite CPU
float32 `[64,16,10]`, exactly **40,960 bytes**; require the two whole byte streams
identical, including signed zero. Retain both complete `persistent.crb` snapshots.
Label the shared baseline **derived-completed-forward-full-cinert**: this is
not a newly captured actual kernel-entry array. If the paired baseline differs,
stop; never silently choose one or treat two inputs as identical.

Predict the complete CRB matrix from a full baseline clone using the nine fixed
groups. Each nonzero-parent addition uses the evolving child composite, not the
original child baseline, with float32 rounding after every addition. Retain both
parent-zero no-ops. Require all **9,600 non-root scalars** (15 bodies, including
world body zero) exactly match both original completed CRB observations; report
the **640 root scalars** separately, without choosing an observed root as the
prediction. Retain full baseline, both observed CRBs, full prediction, metadata
and whole-byte hashes before any CUDA child.

Reauthenticate predecessor inputs after extraction and check caller CPU RNG,
Torch CUDA state, current source/runtime/host, service ownership, FilmBrain,
protected services and idle GPU. The pure codec does not authenticate the
original trace files; the supervisor supplies that provenance before calling it.

### Fixed paired GPU batch and numerical interpretation

Only after successful retained new CPU tests and fixture terminal evidence:
run one separately capped isolated child, original seven levels for 32 repeats
**then** the fixed nine groups for 32 repeats. No adaptive order, repeated search
or favorable-output retry. All **512** installed `_crb_accumulate` calls use the
same full working array as input/output and the same Warp stream. Reset the
complete baseline once per repeat; snapshot only after all that repeat's levels.
Keep both no-op levels. Do not synchronize, observe intermediate stages or read
back between repeats/schedules. One end-of-batch synchronization precedes all
64 complete output readbacks.

Retain two separate **1,310,720-byte** output streams and equal before/after
caller CPU RNG states. No model, actor, optimizer, rollout storage or random
kernel is constructed/launched; Warp alone owns CUDA in the numerical child.
Torch CUDA remains uninitialized. Source/runtime/installed-kernel identity,
actual owner/child PID and parent/invocation, aliasing, stream, dimensions,
reset/snapshot/readback order and call counts are mandatory receipt/mock binds.

The CPU checker validates its own fixture/prediction first. Require every output
non-root scalar to match the full CPU prediction, not the unaccumulated baseline.
Compare all serial output scalars against the one fixed prediction by uint32 bits.
Serial acceptance is `stable-exact-candidate` only for this isolated control.
Concurrent root variation or a constant alternative to the fixed candidate is
retained numerical evidence, not discarded. Retain per-repeat hashes and exact
mismatch counts, bounded first mismatches, differing root cells, and maximum
pairwise same-cell repeat delta. Nonfinite or unexpected non-root data is a real
protocol failure requiring read-only diagnosis.

This paired batch changes schedule and timing. It cannot prove actual atomic
order or original trajectory cause, qualify joint-space mass matrices/other
forward kernels, waive the original full-window gate, or authorize training.
All inherited and new qualification/physical flags stay false.

### Resource and completion gates

Use only fresh source-tagged `microduck-crb-fulltree-{tests,fixture,run,sync}`
units and output directories. Tests/fixture: **300 s** each; supervised run:
**600 s**; GPU child: **240 s**. Tests/fixture/run: **6 GiB cgroup memory**,
CPUQuota 200%, Nice 10, KillMode control-group, RemainAfterExit yes. Sync:
**120 s / 256 MiB / CPUQuota 100%**. Do not use RLIMIT_AS as a resident-memory
substitute. Reserve a 300-s independent closeout and 60-s margin plus every
remaining declared mode's full cap before the 08:00 cutoff.

The owner must freeze the actual positive count of the **51-file** CUDA-hidden
suite after review. Require zero skips and exact log/receipt/report counts and
digests; no placeholder count permits native admission. Acquire/inherit the
unchanged cooperative FilmBrain GPU lease. Require no overlapping Duck service,
idle GPU before/after, no foreign compute PID during the child, temperature
below 75 C, used VRAM at most 5 GiB and free at least 6 GiB. Preserve FilmBrain
PIDs/restart counts and both protected-service namespaces inactive; do not alter
unrelated workloads or old jobs.

Each log is capped at 1 MiB, JSON at 2 MiB, each full-output binary at 2 MiB and
new declared inventories at 12 MiB. Retain partial outputs on any failure. Before
clearing only completed owned units, independently authenticate whole inputs,
source/runtime, predictions, outputs, RNG and terminal caps/PID zero/success/
zero restarts/exact invocations; preserve all artifacts and logs. At 08:00,
start no new work and audit the durable final state without protected restores.

This second phase is predeclared, not yet native-qualified or executed.

### Independent original-trace forecast before native admission

The owner authenticated both 6,193,895-byte original early traces by their
already retained whole SHA-256 values before inspecting their first-event
tensors. CPU-only, weights-only loads and an independent NumPy implementation
with literal parent IDs/groups produced these fixed full-tree anchors:

- Shared full-cinert baseline: `eec453717d592814c0b15d4b44acd4a8836d5acb941699ac5e5ed27b96805368`.
- Capture complete CRB: `0b7702eb8afaa3fc316467aac167f4a802f29a6ea5299a25651d6e3fd815c056`.
- Replay complete CRB: `ddf1f87364a696af9c40eb041d1efae8407e3e2f7a5a04f3351ed272838c7cfe`.
- Fixed nine-group complete prediction: `5a5cac252692294113c560b0139ec72670cf9dd15143acf349ebe46818e4a1f4`.
- Fixed 32-repeat complete-output prediction: `a638d20a4485a0e0e30343f652a16e9ef0ddd4139d648671ecbcc1bc5f2fc8ad`.

Both originals match all 9,600 predicted non-root scalars exactly. Against the
fixed prediction, capture differs at seven root scalars and replay at five;
each maximum absolute root delta is `9.313225746154785e-10`. Caller CPU RNG
is unchanged and Torch CUDA is uninitialized. This independently checks the
first full matrices only, not the original full trajectory or native execution.
The native CPU fixture must still reproduce the unchanged strict rejection
and all source/body/constructor/trace binding before any GPU child.

The full-cinert baseline is intentionally different from the earlier sparse
root-control baseline (`554aa8...`); those fixtures must not be interchanged.
Freeze the new baseline and complete-prediction hashes above in the new control.
The owner arithmetic implementation also passed a separate hand-built `1/1024`
coefficient check plus root-only tolerance/non-root rejection checks, without
importing the new repository predictor.

After review hardening, the exact 51-file suite collects **1,522 tests**. This
positive count is frozen in the new supervisor; require all 1,522 passes and
zero skips for new native tests. The focused new planner/fixture/supervisor
slice passed 86 tests before the full final regression. Reader schemas reject
nonpositive/nonfinite/bool elapsed times and incomplete WSL monitor records.
The report distinguishes whether a monitor poll actually observed the child
GPU PID; an idle poll is not represented as observed GPU ownership.

Final owner regression: **1,522 passed in 232.45 s**, CUDA hidden with pinned
single-thread CPU/ATEN/MKL settings. Ruff lint, format check and whitespace
checks pass. Independent review cleared the arithmetic, predecessor-binding,
input/launch/alias/stream and hardened reporting paths; the owner also validated
the actual 12-file historical serial packet against its pinned hashes and C970
source leaves. These are source/CPU checks only; native admission remains open.

## Completed native fixed-input control

Execution source was clean `d923275040b3538f6ed48b867f1b1e6bc42304d3`
on the exact fork feature branch, on 100.98 only. The newly retained native
51-file suite passed **1,522 tests in 116.05 s**, zero skips. The CPU fixture
and independent CPU owner both freshly re-scored the authenticated original
pair and reproduced `paired rollout semantic state exactness`; that pair is
still rejected. All 50 predecessor files were authenticated afresh.

The one predeclared GPU batch completed successfully in **13.400 s** with
512 installed-kernel launches. Serial outputs were `stable-exact-candidate`:
all 327,680 scalars matched the independently frozen prediction, with one
complete snapshot and zero root variability. Concurrent outputs had **five
complete variants**, 186 root-scalar mismatches against the fixed prediction,
four varying root cells, and maximum pairwise same-cell root delta
`4.656612873077393e-10`. Every non-root output scalar was exact in both banks.
No favorable-output retry or intermediate observer was added.

The actual child PID was 3188116; one of three monitor samples observed that
PID owning CUDA. Peak sampled temperature was **31 C**, peak used VRAM
**942 MiB**. Two idle post-samples showed no compute PID and 661 MiB used.
FilmBrain remained active at PIDs 521 and 298048, zero restarts; both protected
services remained inactive in both system and user namespaces. 100.100 and
installed package/kernel files were not changed.

### Durable artifact anchors

All paths below are relative to native root
`/home/yanbo/work/microduck_rl-stance-replication-20260930`:

| Artifact | Whole SHA-256 |
| --- | --- |
| `artifacts/evaluations/stance-crb-fulltree-run-d923275040b3/report.json` | `e3f70f68a49132db05e23907fc22dc857e142dcb90f8473d0cac1c0c2184e4b2` |
| Run `concurrent.bin` (1,310,720 bytes) | `44115f6bc6e3e68f2f15c29d4820907160056308103d528695ab7037cf9c0a63` |
| Run `serial.bin` (1,310,720 bytes) | `a638d20a4485a0e0e30343f652a16e9ef0ddd4139d648671ecbcc1bc5f2fc8ad` |
| `artifacts/evaluations/stance-crb-fulltree-fixture-d923275040b3/declaration.json` | `563c5be1b1df29c0ed95b5ffdf00d5949499d847858ca2c507ae103a56e742f4` |
| Fixture `report.json` | `8c2f62678c1314c1153b2ac250c7fb8e04018213001a2b88c19202b1d9ad72d1` |
| `artifacts/tools/stance-crb-fulltree-tests-d923275040b3/receipt.json` | `d4186da1c03e4f9e77f94e7b6bb9a57103efb6a1381f395a5176b2d3b0a6b765` |
| `artifacts/tools/stance-crb-fulltree-owner-terminal-d923275040b3.json` (141,834 bytes) | `718b61f3d08e61a2e27410fe8b8b62ae3be52a9a74dad0034e4998bc074ae707` |
| `artifacts/tools/stance-crb-fulltree-closeout-terminal-d923275040b3.json` (1,389 bytes) | `4806842ddf6b664858875b632246d40df384ef72564f99852b96a1ff1d93dc2f` |

The independently capped native owner reauthenticated all inputs, source/runtime,
full bank bytes, literal NumPy arithmetic, RNG, retained GPU monitor and completed
service invocations. Its post-exit packet confirms PID zero, success, zero
restarts, original caps and resident-memory peak **1,600,233,472 bytes** below
6 GiB. Exact invocations for the `microduck-crb-fulltree-*-d923275040b3.service`
units are:

- Sync: `c76c64903ee84b09b1a290a2e6cd72d1`.
- Tests: `8cf05493d0ec41c5babde4924d05d8b1`.
- Fixture: `75286b04bbc84471ae8387541853fa8a`.
- Run: `f9779454708f4e9c97ed2714cf5f9d89`.
- Independent closeout: `5f50f3f71f8e4aa198dcc96e83ab1371`.

The initial terminal-recorder SSH invocation started in `/home/yanbo` and
refused with `ValueError: exact Linux worktree` before publishing a terminal
file. Read-only checks verified successful owner exit and unchanged owner/output
hashes. Only the unchanged CPU recorder was rerun with the exact working
directory; no GPU batch, source or retained publication was rerun/overwritten.

The Mac receiver authenticated both whole owner/terminal packets before decoding,
all **20** new artifact files and committed D923 source leaves, and independently
reproduced complete prediction/output-bank statistics. The hardened 5,897-byte
result is retained at
`/private/tmp/microduck-crb-fulltree-20261006.yz5aYX/mac-verification-v2.json`,
SHA-256 `20b5bcf1ca84c1f20f41f8cabd0e95090c1f16c49b9aa60050a06de6f7fe0b59`.
It also cross-binds publication/terminal invocations, the exact original-trace
forecast wrapper and the receiver/predictor implementation hashes. The initial
successful v1 result is preserved separately; no result was overwritten.
Mac checking is CPU/transport evidence, not a fresh live native GPU attestation
or full original-trajectory scoring on Mac.

This establishes that the fixed-input serial schedule removes observed root
variation in this bounded batch. It changes timing/schedule and uses a derived
completed-forward full-cinert fixture, not actual original kernel-entry capture.
Original-cause proof, actual atomic order, full-runtime/full-window replay,
training, learned stance and physical acceptance remain **unqualified**.
The next useful question is a separately predeclared process-local runtime
schedule control covering the constructor's first forward as well as subsequent
eager forwards. Do not edit installed libraries or reuse this isolated acceptance
as original replay admission.

After whole-evidence verification, only the five completed source-tagged units
were stopped/cleared. The retained 4,292-byte
`artifacts/tools/stance-crb-fulltree-clearance-d923275040b3.json`, SHA-256
`0d495a521888a9a49e90eac31ea7ac2e6fd7696c5e40db691decf3e2945aaac4`,
confirms each inactive/PID zero, all 20 artifact files unchanged, two idle
GPU samples and unchanged FilmBrain/protected state. The lease and all journals
and evidence remain. The Mac v2 result and its receiver source were also copied
to `artifacts/tools/stance-crb-fulltree-mac-receiver-d923275040b3/` on 100.98;
the Mac temporary copy is not the only retained copy.
