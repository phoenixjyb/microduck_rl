# Stopped initialization-gradient adapter preparation

Base `870d933dc69e7bc3070e2e3b3c645f31c7f2b99c`, exact branch
`feat/athletics-obstacle-curriculum`. Separate three-path fence: new
`stance_solver_gradient_dispatch.py`, its test file, and this document. All
previous contracts, source fences, receivers, installed runtimes and evidence
stay unchanged. Protocol `microduck-stopped-gradient-dispatch-oct9-v1`.

This implements one part of the next native gate described in
[gradient-prefix preparation](2026-10-09-gradient-prefix-preparation.md): an
original-caller observer with complete typed readback. It is **not the native
collector, executable loader, independent native receiver or GPU admission**.
No GPU job, optimizer, policy inference, MP4, physical motion, perception or
service change is included. All six qualification flags remain false.

## Explicit stopped boundary

Authenticate the complete frozen installed solver SHA256
`bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`
and loaded CPython code compiled from those bytes. Hold the original caller,
globals, defaults/closure and both kernel functions. Invoke unchanged
`_update_gradient` directly; observe its original launches at lines 2925 and
2927 with exact kernel, dimension types, argument identities/order and eager
defaults. No incremental caller or enclosing initialization/iteration runs.

After the second launch and its complete after-readback, the wrapper raises
a private per-observer `BaseException` instance. Only that exact instance,
after both completed stages, is accepted. The caller never resumes to its next
branch at line 2934. This is a **call-site stop before caller resumption**, not
observation of that next line, normal caller completion or a naturally generated
initialization transition. No tracing hook is installed. Existing or newly
installed trace/profile instrumentation refuses the attempt and is preserved.

Unexpected/reentrant/indirect/third launches, changed call sites, substituted
code, premature/unrelated stop signals, normal return, launch/copy/sync failure,
foreign hook replacement and buffer/stream/device mutation cannot yield a
successful adapter receipt. Use the existing shared launch-hook lock. Restore
`wp.launch` only while it is still the owned wrapper; preserve a foreign
replacement, refuse success and release the lock on every acquired path.

## Complete readback and evidence limits

Require exactly two supplied `LoadedModuleBinding` objects for the two kernels,
sharing one module/executable/artifact, sm120 device, block 256, and same explicit
current stream. Those object checks are not compilation/load provenance or
driver-resident byte proof. A new two-entry loader is still required; do not
patch the older four-entry cost executable's graph or globals.

The held launch/readback functions are the objects present at construction.
The future owner must independently authenticate the pristine pinned Warp API
and exclude preinstalled shims **before** constructing this observer. Inert
import and synthetic fixture passes do not establish that runtime binding.

Capture all 26 fields in the unchanged prefix packet order through typed,
nonoverlapping pinned CPU staging. Hold allocations, pointers, types, shapes,
runtime entries and guard function. Copy on the declared stream with pre/post
synchronization, serialize canonical complete little-endian bytes and retain
whole hashes. Two stages produce four 3735368-byte banks per arm; the future
paired collector would retain eight banks / 29882944 bytes. Readback changes
execution timing and is not a timing-equivalence claim. This capture helper
does not authenticate an arbitrary callback as a kernel.

The observer checks dense nv20 / 64 worlds / njmax512. Frozen model options,
parent-byte restoration, lease, deadlines, capacity, external inventory and
retirement must be authenticated separately by the future owner. Numerical
interpretation still belongs to the unchanged conditional CPU prefix contract.
Hessian, search, line search, convergence, complete solver window, motor/plant
acceptance and learned skills remain pending.

## CPU checks and next gate

Run `stance_solver_gradient_dispatch.TESTS` with CUDA hidden on Mac and WSL.
Tests use the original authenticated CPython caller but synthetic kernel objects,
ELF fixtures and NumPy-backed arrays; no Warp/device package is imported by the
production module. The happy-path fake model deliberately has no `.opt.solver`, so
resuming to line 2934 would fail. Exercise failed hooks/capture/restoration,
sealed bindings, trace/profile preservation, exact sites and stop identity.
The separate source-binding helper checks the new three-path fence and all
committed plain-file blobs; it never widens the previous source fences.

Independent Luna review found no blocker in the stop/capture/restore flow and
flagged the pristine-API owner prerequisite above. It made no edits or test
claims. Owner keeps that requirement outside this preparation's acceptance.

Before a native run: separately fence and implement the new two-entry
loader/SASS checks, owner/child with authenticated predecessor restoration and
26-field staging, and independent whole-inventory receiver. Commit and push,
rebind paired prerequisites at the exact execution source, verify a clean WSL
fast-forward and frozen runtime, obtain fresh shared Windows/WSL capacity and
lease checks, then run one capped user-service child. Preserve FilmBrain and all
unrelated workloads. Never substitute these CPU fixtures for native capture or
claim curriculum training admission.
