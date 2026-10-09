# Selected gradient runtime API boundary

Base `e7b12f6d07f74a388b926b4252b9ba9d4f045e77`, exact branch
`feat/athletics-obstacle-curriculum`. This separate three-path fence adds only
the gradient runtime-boundary module, its test file and this document. Previous
source fences, executable loaders, observers, receivers and evidence stay intact.

This is a bounded prerequisite discovered while preparing the native gradient
collector following [the two-entry loader](2026-10-09-gradient-executable-preparation.md).
That loader and the stopped observer hold current Warp functions at construction;
they detect later replacement but do not authenticate a preinstalled function
shim. This preparation closes that gap for the **selected Python bodies**. It
does not complete the owner/child, receiver or pristine whole-runtime prerequisite.
No new native attempt, optimizer, training, MP4 or physical motion is included.
All six qualification flags remain false.

## Exact scope

Before allocation or loader/observer construction, `WarpApiGuard` verifies four
whole installed source files (`warp/__init__.py`, `_src/context.py`, `_src/types.py`,
`_src/build.py`) at canonical Python 3.12 environment paths, their module names,
`sys.modules` identities and package exports. Source lengths and SHA256 values
are literal pins for Warp 1.12.0, not values learned from the current files.
The native owner must still authenticate the complete frozen toolchain source
tree, package versions, native libraries, runtime origin and process lifecycle.

Compile the pinned sources without executing them, extract qualified CPython
code objects, and compare 24 plain functions/methods:

- `launch`, `copy`, `empty`, `get_device`, `get_stream`, `synchronize_stream`;
- eight Module constructor/hash/compiler/loader/generated-name methods;
- ModuleExec constructor and kernel hooks, ModuleBuilder constructor and codegen;
- Kernel constructor and mangled name;
- array constructor and `numpy`, plus `build_cuda` and `load_cuda`.

Require source-equivalent code, canonical function globals, native Python
function descriptors, matching names, no closure, decorator/wrapper attributes
or keyword defaults, and exact typed default **values** parsed from source.
Only the literal defaults and the pinned `float`/`typing.Any` names are allowed;
no default expression is executed. The four mutable empty launch-list defaults
must remain empty. Hold exact function/code/default-tuple identities afterwards.
Hold class identities/shapes and the selected public function, array, Module
and Kernel aliases. Refuse changed source bytes, descriptors, code (including
an equal-but-distinct later code clone), defaults, aliases or modules.

For direct global names referenced by those selected bodies, hold resolved
namespace, object and Python helper-code identities across subsequent checks.
Reject module-global shadows of builtins, including a shadow with the same
builtin object; the pinned Warp `types.bool` is an explicit non-builtin case.
Check CPython builtin type/name/module metadata, including `_io.open`, without
claiming authentication of their C implementation bytes. The initialized
runtime object is held unchanged, never silently adopted after replacement.

## Limits that must remain visible

This is trusted in-process integrity, not a hostile-interpreter sandbox. A
source-equivalent function clone or equal-but-distinct initial default values
can pass initial authentication: original function/default object creation
provenance is not established. The receiver must not reinterpret that as proof.

Direct globals are held **after** authentication; unselected helper bodies,
mutable object internals and transitive module attributes are not authenticated
at construction. A preinstalled replacement of an unselected helper such as
`context.init` can pass, and an explicit test documents that limitation. This
guard therefore cannot alone satisfy a complete pristine-runtime prerequisite.
Independent initialization, transitive dependencies used by the future collector,
native library identity and fresh isolated process/cache evidence remain required.

Construction/check/record invoke no Warp runtime API. They neither initialize
CUDA nor allocate arrays, obtain a device, compile, load, launch or read back.
Construction can be checked with an uninitialized CPU frontend. For a future
initialized native child, the owner must independently establish initialization
origin before construction and keep the held runtime stable afterwards.

The receipt explicitly records `transitive_dependencies_authenticated=false`,
`initialized_runtime_origin_authenticated=false`, `native_execution_observed=false`
and all six qualification flags false. Source binding is a separate exact
three-path fence, not a widening or reuse of an older phase's fence.

## Validation and next execution gate

Run the prior 23 declared files plus the new runtime-boundary tests with CUDA
hidden and pre-import `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `NUMEXPR_NUM_THREADS` set to 1. Use the exact worktree `PYTHONPATH` on WSL;
the shared virtual environment's editable project points to an older checkout.
Retain source identity and exact test multiset; preserve the established CPU
limits and old ABI assertions/timeouts. No package, driver or service repair is
part of these tests.

Negative tests cover preinstalled selected API/compiler/readback shims,
source/default/code/namespace/alias/descriptor changes, builtin shadows,
runtime/class replacement, mutable defaults and immutable held bindings.
Fresh CPU subprocesses prove inert import and use a profiler to forbid runtime
API execution during authentication. Positive limitation tests prevent an
accidental stronger origin or transitive-dependency claim.

The next implementation remains a separately fenced owner/child and independent
whole-inventory receiver: authenticate retained parent Gauss banks and historical
packet, restore all 26 typed fields, explicitly compile/load the two gradient
entries, observe only the stopped two-launch prefix and retain all eight paired
boundary banks. Reproduce all arithmetic/write-set checks after complete byte
authentication, with external same-invocation retirement and fresh resource,
cache, tool, protected-workload and lease evidence. No native attempt may bypass
the still-pending initialization/transitive-runtime authentication boundary.

## Retained paired CPU closeout

Execution source `d9763cf65fcc401eaf1e2264f11951bc964e915c`, tree
`4abeeb7bd5505b616f0ac9002cb86fb9ae5d211c`: all 994 committed plain-file
blobs matched before and after both checks. This section is a later document
closeout, not tests relabeled at a newer source commit.

All 1199 cases across the declared 24 files passed with zero failures, errors
or skips on both hosts, including the 72 new cases. Mac used 24 serial isolated
per-file CPU processes with unchanged 120-second per-file bounds and Nice 10,
168.54 seconds total. WSL used one aggregate suite, 112.95 seconds. Independent
verification matches the exact case multiset; it does not claim identical
process partition, operating system or Python patch version.

WSL unit `microduck-gradient-runtime-cpu-d9763cf65fcc.service`, invocation
`30b76b216e7241008966fd35adb40d0b`, retired successfully with MainPID 0,
ExecMainStatus 0, no restarts, empty ControlGroup and its original cgroup absent.
The retained oneshot service is active/exited, not a running CPU/GPU process.
Caps remained 300-second start / 10-second stop, 4 GiB, 200% CPU, 64 tasks,
Nice 10, 16 MiB file, no core, control-group kill and no restart. Peak sampled
tasks were 14, with every `pids.events` sample `max 0`; MemoryPeak was
1737793536 bytes. CUDA stayed hidden and all four pre-import thread limits were 1.

Whole frozen runtime bindings matched before and after. FilmBrain observatory
PID 521 and video-playground PID 298048 stayed active with zero restarts.
Both protected AI mission services stayed inactive in both system and user
scopes. The independent checker also validates the retained runtime declaration
against the unchanged predecessor's literal source/library/tool pins.

Artifacts are retained on both hosts under `artifacts/tools/gradient-runtime-boundary/`.
Independent Mac verification authenticates 54 inventoried files / 4291235 bytes,
the closeout and four helper sources, all source leaves, exact case multiset,
service caps and same-invocation retirement. Its copied paired receipt is
byte-matched on WSL. All six qualification flags and native admission remain false.

At 13:49 Asia/Shanghai, sequential Windows and WSL samples showed 8259/8249 MiB
used, 15903/15913 MiB free, 35 C and 2% utilization; a Windows 3D engine was
active. These are not simultaneous samples, proof of an idle GPU, process
attribution, exclusive lease or reservation for a later native attempt.

Pre-commit development runs exposed a test-only class-name cleanup bug and
an overly strict builtin-origin assumption (`open` belongs to `_io`). The
working JUnit records remain locally retained outside the accepted source-bound
inventory. Both were corrected before the accepted commit; no installed source,
package, driver or unrelated process was changed. A read-only Luna review also
prompted same-object builtin-shadow and public Kernel-alias coverage, plus
explicit positive tests preserving the source-equivalent-object limitation.
The final review found no correctness blocker within this selected scope.

The initial SSH banner timeout and GitHub connection reset were retried after
read-only connectivity checks; no network service was restarted. The feature
source was pushed to the fork and fast-forwarded on WSL without rewriting history.

| Evidence | SHA256 |
| --- | --- |
| `d9763cf65fcc-mac-tests.xml` | `3728cf645d0fad9c8f2b306befa26f5e4320e1b52cca1ac5628eb75f2c128290` |
| `d9763cf65fcc-wsl-tests.xml` | `1d1ac95585e0ecc724cf0239eea0920c6a1ae649c4bf12c92ec7f22a6cd29cc2` |
| `d9763cf65fcc-wsl-report.json` | `ce098c539a034dfc283eeabc9522a782d9ff9ab7631a7da8ca0bbaf4257a9c2c` |
| `d9763cf65fcc-wsl-closeout.json` | `e9061afaa42315f611a90ab951904a6fc809292503384bdf672578443959ffdf` |
| `d9763cf65fcc-paired-check.json` | `b77bdb1872a8666462a6222fe7b583322970830ed47be71adb7caa93f5e92847` |
