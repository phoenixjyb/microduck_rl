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
