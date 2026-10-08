# Four-stage caller adapter preparation

## Boundary

New protocol `microduck-caller-cost-dispatch-oct9-v1`, based at exact feature
branch commit `7aec9cebc2178be909ba119a0fa2cb504b264695`. Only the new
`stance_solver_cost_dispatch.py`, its tests and this note are in scope.
Historical capture/receiver protocols and source fences remain unchanged.
No GPU job, training, video, raw perception or physical motion is admitted by
this preparation. Existing FilmBrain and protected service states are preserved.

## Observed contract

`CostDispatchObserver` delegates one unchanged frozen `_update_constraint`
invocation with its default `track_changes=False`. It brackets exactly four
original launches with the existing complete 24-field `CostStageCapture`:

| Stage | Frozen line | Dimensions | Factory specialization |
| --- | --- | --- | --- |
| init_cost | 2156 | integer 64 | top-level kernel |
| efc | 2180 | (64, 512) | track_changes=False |
| dense | 2197 | (64, 20) | top-level, nonsparse branch |
| gauss | 2214 | (64, 1) | nv=20, dofs_per_thread=50 |

The adapter compiles source bytes without executing/importing them and compares
the loaded CPython caller, top-level kernels, nested factory kernels and cache
wrappers against those frozen code objects. It then holds their exact identities,
defaults, globals, closure types/values, cache dictionary and two specialized
cache entries. The whole installed solver and cache utility source hashes are
checked throughout; no factory is called to generate a replacement kernel.

Four exact `LoadedModuleBinding` objects are required. Init and dense share the
same top-level module, executable and retained artifact; EFC and Gauss use two
distinct unique modules. Every binding rechecks its artifact bytes, module cache,
device/context, metadata, kernel adjoint and forward hook. These checks observe
objects supplied by the owner: they do **not** prove the loader consumed those
files or identify driver-resident bytes. A synthetic fixture can satisfy them.
Held Warp entries are not independently authenticated against runtime hashes
here. The declared base is a protocol label, not live checkout evidence; both
runtime and checkout identity remain owner checks.

Every launch must come directly from the held caller frame at its original line,
with the held model/data/context, False track_changes, exact stage kernel,
dimensions, input/output identities/order and unchanged eager forward defaults.
Gauss additionally requires the single-thread recipe in the caller locals.
Same thread, sm120 current device, explicit current stream and no active Tape
are required; the launch `record_tape` default remains True.

All 24 bank allocations and three extra arguments have held contiguous,
nonoverlapping shape/dtype/device/pointer spans. Extras are contact friction
(8192 vec5 values), changed EFC IDs (64 by 512 int32) and counts (64 int32).
They are excluded from raw stage packets, explicitly listed in the receipt,
and must not be treated as byte-captured. The eventual owner must preserve the
sealed original packet's no-active-elliptic-row domain; friction is not read
on that admitted branch and changed IDs/count are not accessed when changes
are disabled. The independent bank receiver still owns numerical/domain checks.

## Ownership and failure

Bindings and observer state are sealed; binding/capture maps are read-only.
The adapter uses the **same** nonblocking global hook lock as the earlier dense
observer. One attempt consumes the observer, including lock contention or
preflight refusal. Manual/reentrant/duplicate launches, mutated factories,
closures, cache entries, kernels, layouts, runtime entries or streams refuse a
successful receipt. On exception it retires its own hook, releases the lock and
retains complete pre-failure snapshots. A foreign hook is preserved and fails
the receipt. This is trusted in-process integrity, not a hostile-Python sandbox.

One capture path performs the eight before/after snapshots; there is no second
dense snapshot timeline. Same-stream readbacks synchronize and change timing,
so EFC atomic ordering is not qualified by the result. No EFC row-cost oracle
or runtime-cause assertion is introduced.

## Checks and remaining native gate

New tests execute the real frozen Python caller and cache wrapper code with
synthetic arrays, kernels, executable objects and ELF-shaped fixture bytes.
They do not import/initialize CUDA or load/execute binaries. Tests cover four
call sites, complete raw capture, staged boundary argument refusal, closure/cache
and kernel mutations, recursion, failure retention, hook contention/foreign
replacement, sealed bindings and inert import. Run these alongside the retained
scratch/replay/Gauss/cost-stage tests on Mac and WSL with CUDA hidden.

Before native work, implement the separately fenced owner: whole runtime pins,
fresh isolated process/private caches, source/compile/CUBIN/metadata/SASS
bindings for all three actual modules and four entry points, explicit load
provenance, original plus nonzero-control restoration, independent whole raw
inventory/receiver, inherited FilmBrain lease, fresh deadline/capacity checks,
resource caps and external cgroup retirement. No old source fence is widened.
This adapter is not that owner. Every qualification flag remains false, and
explicit load provenance, native GPU execution, capture origin and external
retirement authentication remain false in its receipt. Training is still gated
on broader retained simulation evidence, not this CPU preparation.

## Retained paired CPU closeout

Executed source: `a931612436ca4708cbdf4bdc8c61bf4d51d7863c`, clean exact feature
branch on Mac and WSL. Fourteen focused files were run with CUDA hidden: the
thirteen retained scratch/replay/Gauss/cost-stage files plus the new adapter
tests. Both XMLs independently verify 633 tests with zero failures, errors or
skips; this is **not** the full repository suite. The 93 new adapter tests use
the frozen Python caller with synthetic CPU objects, not native GPU execution.

- Mac: 633 passed in 62.68 seconds. XML 907172 bytes, SHA256
  `7bfc291f97984699cffb351a3488a5c56b88e808bd02b454c2a0177b1eb685e4`.
- WSL: 633 passed in 46.15 seconds. XML 907176 bytes, SHA256
  `451f21b94c4b38125fdc7c8877704fd9d76b4ea38ef6685c0f55e0c2a22611e9`.

Retained paths under `artifacts/tools/caller-cost-dispatch-preparation/` are
`a9316124-mac-tests.xml` (Mac) and `a9316124-wsl-tests.xml` (WSL and verified
whole-byte copy on Mac). The source bundle SHA256 is
`83eb873bf1ab1a106ed347bd3b949298df5145c0aa5da8ccf9f1ae2c6115c1cc`.

The WSL CPU unit `microduck-cost-dispatch-cpu-a9316124.service`, invocation
`b94d98d5fe0b495f99eed303ddac3ed3`, finished successfully with exit status 0,
MainPID 0, empty cgroup and zero restarts. Its retained active/exited state
is not a running workload. Verified caps: 150-second startup timeout, 4 GiB RAM,
200% CPU, Nice 10, 64 tasks, 16 MiB/file, no restart, control-group termination.
No GPU lease was acquired or GPU job launched. No Duck unit remained running.

Afterward the frozen environment alias, interpreter, package/library, Warp and
MuJoCo-Warp source-tree and disassembler pins reverified unchanged. FilmBrain
observatory and video services remained active at PIDs 521 and 298048, zero
restarts. Both protected AI mission services remained inactive in user and
system scopes. Point-in-time GPU state was 8145 MiB used, 16017 MiB free,
2% utilization and 34 C: shared capacity, not exclusive ownership.

The bounded read-only Luna review found no concrete defect in this slice and
confirmed that held runtime objects and declared base are not independent
runtime/checkout authentication. The reviewer ran no tests or native work.
All qualification and physical/training gates remain false. The next slice is
the separately tested multi-module explicit-load owner and capped collector,
not training or policy promotion.
