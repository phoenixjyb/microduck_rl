# Entered-Python bootstrap observer

Base `20c00062c7d81dd4185cdb61e25a3a2efc68faca`, branch
`feat/athletics-obstacle-curriculum`. This separate three-path preparation
contains one observer, its CPU tests and this document. It leaves every older
collector, source fence, loader, solver, observer and receiver unchanged.

The [collector preparation](2026-10-09-gradient-native-preparation.md) verified
selected API functions, `init` and 17 Runtime methods. An unselected replacement
of `Device.__init__` or `build.init_kernel_cache` could still be called. This
observer makes a narrowly stronger claim: entered Warp Python bodies and direct
standard-library Python callees are checked at call entry during one held init
call. It does not establish complete runtime origin or resume Duck training.

## Enforced scope

Before initialization, authenticate all 460 Python/compiler-header leaves,
9292904 bytes, against literal Warp source-tree SHA256
`4aa3c865b7e523e1c0bef175f80ed51b00543569246d524914e78c333cdf9e6c`.
The coordinator source also has a literal whole-file pin. Compile relevant
sources without executing them, recursively collect nested code objects and
compare each entered code object to that graph before the callee's first
bytecode. Require canonical module globals, installed source paths and held
module identity. Whole Warp source bytes are checked again before and after.
Resolve absolute entered-code filenames to that same installed file, permitting
WSL's retained shared-venv alias without reinstalling or weakening the whole-tree
byte pin. Reject synthetic, relative and missing code filenames.

For direct Python callees leaving Warp, require a closed standard-library
module allowlist and code equivalence to the current installed source. Record
those whole-file hashes and check them again afterwards. These are observed
environment-derived anchors, not literal cross-host Python pins. CPython's
`collections.abc` functions live in the canonical `_collections_abc` module
with a frozen code filename; that exact alias is checked explicitly. Indirect
standard-library calls remain a trust boundary.

Observe only the main thread, with no existing profile/trace hooks on admission.
Hold the original CPython profile API metadata and the bootstrap coordinator
method. Use a one-shot latch, a 10-second observational deadline, at most 20000
Python call events and 256 unique observed frame rows. Existing external
supervision is still required: a profile deadline cannot interrupt a hung C
call. No hook is installed during kernels, simulation or optimization.

Reject calls to the held thread-hook setters and foreign CPython profile/trace
mutation; require thread-hook identities and empty state again afterwards.
Clear only the observer's own hook.
If CPython removes it after an observer exception, retain failure and refuse a
receipt. If another hook replaces it, preserve that hook and refuse success.
Constructor failure, source mismatch, bounds failure and retry cannot yield a
successful receipt. Staging/readback and the original private stopped-gradient
sentinel are not changed by this preparation.

## Limits and execution gate

This is integrity checking in a trusted fresh interpreter, not a hostile-process
sandbox. Source-equivalent object clones and altered callable defaults may pass;
callable creation, descriptor/default bindings and standard-library origin are
not authenticated. The stdlib hashes describe current installed source, not
provenance or a preinstallation integrity claim. Mutable values and native C
calls can still have invalid behavior. Native libraries, compiler and driver
bytes, C callbacks, other threads, indirect external Python helpers, unentered
branches and GPU-only initialization paths are not authenticated here. All six
qualification flags, whole-transitive authentication and GPU-route observation
remain false. Profiling deliberately changes initialization timing.

CPU tests use fresh processes, CUDA hidden, all four pre-import thread limits
set to 1, private cache paths and a 30-second per-process timeout. They test
successful CPU initialization, helper shims before/after observer construction,
canonical-globals and external-globals replacements, direct stdlib shims,
foreign hooks, source/bounds/coordinator failures, cleanup, one-shot behavior,
sealed bindings and inert import. Positive boundary tests retain an indirect
stdlib helper shim and a source-equivalent callable with altered defaults;
their receipts still explicitly deny transitive/default/origin authentication.
CPU initialization is not CUDA evidence.

At the start of this turn, 100.98 was clean at the base commit and FilmBrain
services were active at PIDs 521 / 298048 with zero restarts. WSL telemetry
showed 12999 MiB used, above the retained 12288 MiB GPU admission ceiling,
11163 MiB free, 42 C and 27 percent utilization. No Duck GPU job is admitted
on that sample, and unrelated workloads must not be stopped to meet this gate.

The declared CPU suite is the previous 26 files plus the new observer tests.
After paired CPU evidence, a separate reviewed integration must wire the
observer into a newly fenced collector and receiver, retain its record, and
establish the GPU-specific initialization route under fresh resource admission.
Do not widen or silently override any older fence. No native attempt, optimizer,
policy, video, raw perception or physical motion is part of this preparation.

## Retained paired CPU closeout

Execution source `c8f07cde950afcee639b53cd23b2a84d717dd9c4`, tree
`fc4e5bca5b9b9c365c41de6bec35e0e329d62206`: all 1002 committed plain-file
blobs matched before and after the checks. This section is a later documentation
closeout; the execution evidence is not relabeled at its newer commit.

All 1349 cases across the declared 27 files passed on Mac and WSL, with zero
failures, errors or skips, including 34 new observer cases. Mac used 27 serial
isolated CPU processes, Nice 10 and a 120-second per-file timeout, 176.06 seconds
total. WSL used one aggregate suite, 142.89 seconds pytest time / 143.68 seconds
owner-measured subprocess time. Independent reception on each host reproduced
the exact testcase multiset, not identical process partition or Python patch
version. Both produced the same paired-check bytes and SHA256.

WSL unit `microduck-bootstrap-observer-cpu-c8f07cde950a.service`, invocation
`8e5f263f831944cf9dbb437aff2cca4d`, finished successfully: MainPID 0,
ExecMainStatus 0, no restarts, empty ControlGroup and absent original cgroup.
Active/exited retains the oneshot record; it is not a running Duck job. The
unchanged CPU caps were 300-second start / 10-second stop, 4 GiB RAM, 200% CPU,
64 tasks, Nice 10, 16 MiB file, no core/restart and control-group kill. Peak
sampled tasks were 14, every `pids.events` sample was `max 0`, and MemoryPeak
was 3552206848 bytes. CUDA stayed hidden and all four pre-import thread limits
were 1. Frozen runtime and protected service snapshots matched before and after,
and again during external retirement. The FilmBrain PIDs and zero restart
counts remained unchanged; both AI mission services remained inactive in both
scopes. No environment, package, driver or unrelated workload was changed.

Separate fresh, 30-second-supervised CUDA-hidden receipt processes retained
22 unique observed rows / 868 Python call events on Mac and 21 rows / 778 events
on WSL. Both removed the owned hook and retained explicit false transitive,
stdlib-origin, callable-default, native-C, other-thread and GPU-route claims,
plus the six false qualification flags. The WSL CUDA error 100 under an empty
`CUDA_VISIBLE_DEVICES` is not an exposed-GPU compatibility test or a diagnosed
driver fault. Private CPU cache contents are not native-qualification evidence.

Artifacts are retained on both hosts under `artifacts/tools/bootstrap-observer/`.
The independent checker authenticated 62 inventoried files / 16973463 bytes,
plus closeout and five helper anchors, both whole source inventories, receipts,
exact cases and same-invocation retirement. At 18:28 Asia/Shanghai, sequential
Windows/WSL samples showed 22916 / 22932 MiB used, 1246 / 1230 MiB free,
38 / 37 C and 4 / 3 percent utilization. Both exceed the 12288 MiB used-memory
ceiling and fail the free-reserve gate. A Windows 3D engine was active; these
samples are not simultaneous counters or an idle/reserved-GPU claim.

The earlier execution at `34db18c938dea33461edc5d40ec2ab4ed5ab35e5` passed
1348 Mac cases but failed four WSL success-path cases: literal code-path text
did not equal the canonical shared-venv file path. Read-only inspection showed
both resolve to the same unchanged installed Warp file. The fix restored strict
absolute-path resolution, normalized missing/synthetic paths to refusal, and
added an alias regression case. The failed pair remains retained separately;
it is not accepted evidence and required no installation repair. Luna's review
also made the owner add explicit indirect-helper/default limitations and thread
hook checks rather than claiming complete transitive authentication.

The next bounded task is separately fenced collector/receiver integration and
review of the remaining initialization dependencies. Native admission still
requires closure of that reviewed boundary and fresh capacity/lease checks.
No optimizer, newly learned capability, MP4, CUDA qualification or physical
result is claimed by these CPU receipts.

| Evidence | SHA256 |
| --- | --- |
| `c8f07cde950a-mac-tests.xml` | `39116ff5d097cba947842b687253f2bab6e8932ac2e0ad80247dbba4fe5590fd` |
| `c8f07cde950a-wsl-tests.xml` | `194fb4cfbede9dcca2a793e2546bf1ee4059b82eb638b32c7b9482b226226a02` |
| `c8f07cde950a-mac-receipt.json` | `93de97d91e92f96468c3c01ebff9d5382fe0d17e66df80c8a66fb2213905e121` |
| `c8f07cde950a-wsl-receipt.json` | `9a887c9eb55752dcb858933701ceea18fe6e42752c7a18632981beebd3e9d632` |
| `c8f07cde950a-wsl-report.json` | `8a4ca940cf6b55f0441c1134a50007b97f01e2cf4e28f0d63a0a43433a3d1ab6` |
| `c8f07cde950a-wsl-closeout.json` | `489f85f08cca3bca922b3cbcd05e95f0cf347f7e335ace6e82c972a9905dcf60` |
| `c8f07cde950a-paired-check.json` | `be0c73a6e1b3ab9f9ea5cdd272c242feae9bc0315a82f400be4c8d5aafe4842d` |
