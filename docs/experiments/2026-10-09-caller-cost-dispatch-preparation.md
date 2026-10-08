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
