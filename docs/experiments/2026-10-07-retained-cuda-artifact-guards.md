# Retained CUDA artifact guards: source preparation only

## New scope after the completed 13:00 window

The user requested the next chunk after the previous window closed. Start from
clean fork branch `feat/athletics-obstacle-curriculum` at
`84a71126e02c7c8f637480cbcd7cada30a2ee795`; do not reopen or amend the expired
declarations. Only100.98 (`gw98-direct`) is in scope. Read-only inspection at
2026-10-07 22:34 Shanghai found the native lean checkout clean at that same
revision, no running Duck unit and an empty GPU compute-process list. The RTX
PRO4000 Blackwell was30°C with664MiB display memory. FilmBrain services retained
PIDs521/298048 and zero restarts; protected AI mission user services were inactive.
These are point-in-time observations, not continuous occupancy monitoring.

This slice implements a small prerequisite to the
[isolated CUDA append component](2026-10-07-dense-friction-prefix-packet-audit.md#next-round-boundary).
It does **not** implement or execute that numerical probe. Preserve frozen
packages, drivers, shared caches, leases, historical worktrees, failed units and
all artifacts. No service restoration, simulation, GPU load/compile/launch,
policy/optimizer/storage, perception, video or physical motion is performed.

## Implemented contract

`stance_cuda_artifact_binding.py` has no Warp import, loader call or launcher.
Protocol `microduck-retained-cuda-artifact-binding-oct7-v1` provides three guards:

1. `retain_artifact`: caller-supplied **external** whole-length/SHA256 anchors for
   existing PTX/CUBIN and metadata files. Reject symlinks (including parents),
   nonregular files, aliases, changed inodes/whole bytes and oversized files.
   Authenticate before JSON decoding; reject duplicate/nonfinite JSON and
   non-integer, negative or excessive shared-memory metadata. Binary cap8MiB,
   metadata cap256KiB. The extension labels a retained input format; it does not
   validate CUDA instructions or establish the origin of the artifact.
2. `assert_fresh_module`: reject any preloaded executable for the actual CUDA
   context, including other block dimensions, and prior failed builds. Require
   eager CUDA, a plain nonzero context, bounded explicit block dimension and
   `strip_hash=False`. Clear no cache. This check belongs immediately before the
   future supervisor's authenticated, held explicit-load call.
3. `bind_loaded_module`/`assert_unchanged`: bind the observed module/executable,
   source/options hash, context/device/architecture, cache key, kernel adjoint,
   forward hook object/handle, mangled symbol and authenticated shared-memory
   metadata. Recheck artifact files and mutable observed objects at each record
   boundary. The original MuJoCo-Warp kernel inherits `enable_backward=False`
   from its module; the candidate overrides it. Check the actual merged options,
   not a nonexistent original-kernel override.

Independent Luna review of frozen Warp1.12 loader source confirmed its early
cache return and hook/metadata contract, then found a format-label consistency
gap in the new guard. Owner review fixed that gap and added inherited-option
handling, held adjoint/device bindings, strict metadata and an explicit
`loaded_binary_bytes_observed=False` limit. The frozen `context.py` remains
408818 bytes / SHA256
`eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`.

These checks are **not** a sandbox against arbitrary Python replacement and are
not authentication of the runtime classes, callable source or process. A mock
can satisfy the observed-object contract, as the tests deliberately demonstrate.
The caller must authenticate this source and its held entry points as well as
the real frozen runtime. The record deliberately leaves loaded-binary/JIT,
native-execution, training and physical flags false.

## Explicit load and numerical gates still pending

The artifact hash and opaque module handle in a record **do not prove that the
driver consumed those bytes**. The future supervisor must witness the held
frozen `Module.load` invocation with exact binary/metadata paths, `output_arch`
and block dimension, prove the fresh-cache check applied to that same call, and
tie its return object to the binding. Retain source/options, compiler/toolchain,
device identity/architecture and actual artifact bytes before/after. Warp exposes
no driver-JIT machine-code bytes; a PTX hash must never be called their digest.

Next executable work is a separate bounded CUDA producer plus independent
CUDA-specific receiver, not a widened CPU protocol. Reuse the ten literal prefix
inputs, original once and two fresh ascending candidates:30 launches maximum.
Before launching: exact-source CPU tests, clean native source, existing shared
GPU lease and idle GPU, fresh isolated capped child, authenticated source/load
witness, stream/63-buffer/launch bindings and complete before/after packets.
Declare exact source, output root, service/process/log/time/memory caps and
independent whole-artifact authentication before GPU execution. If any binding
cannot be observed, retain the gap and stop; do not substitute a claimed hash.

The receiver must recompute counts/addresses, prefix and poison preservation,
addressed original equivalence and complete candidate replay under a new CUDA
envelope. Overflow cases remain negative. Even a passing synthetic component
would not qualify the simulator: a separately declared paired runtime check and
full-window learned-policy gates remain afterward. Earlier C3/BAM GPU negatives
are unchanged. Do not launch a new training campaign from these source tests.

## Validation and delivery

Initial54 focused file/mock tests passed in2.07s on Mac with CUDA hidden; after
the format-label review fix,55 focused tests passed in0.11s. Owner review also
closed Python float/integer equality aliasing in loaded metadata; the final
56-test scope passes and is being checked again in the full regression. Ruff
lint and formatting passed. Tests cover external anchors, JSON, symlinks/inode
replacement, cache refusal, block/context guards, artifact changes, executable
and hook replacement, handle/hash/metadata drift, inherited backward options and
resource-type bounds. No mocked load is counted as a native result.

The prior82-file/2508-test regression passed in155.70s before the last
metadata-type check. Its report remains intact at
`artifacts/tools/cuda-artifact-guard-final-mac-oct7`; do not relabel that execution
as the final source. The complete prior81-file CPU scope plus this final test
file is being validated again. Exact-source native CPU checks and the final
fork revision will be recorded below. This document grants no native
qualification or new GPU execution.
