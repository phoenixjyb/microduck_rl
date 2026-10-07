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

## Completed source/CPU closeout

Source commit `9f7f803a8376dd17da6a5f08950e556fdeabe52b` was pushed to the exact
fork feature branch through the owner's existing authenticated SSH route.
The HTTPS push first failed because an invalid `GH_TOKEN` overrides the saved
account. Read-only authentication checks identified that conflict and proved
the existing SSH key authenticates as `phoenixjyb`; no token, account, remote
configuration or credential was changed. `git ls-remote` independently confirmed
the published source SHA.

The final component is12943 bytes / SHA256
`8d605178b026d58ff071da2838bea8ecc43eb0564969dcc0721e0e8e0f1e10f3`;
tests are10745 bytes /
`f29735b3ec6b21fe0ee9379576045dbc4cdf1d7b30610569888be90bac99ebab`.
The full final82-file Mac CPU regression passed **2509 tests in174.07s**, zero
errors/failures/skips. Source/test bytes were checked unchanged before/after
and match the committed files. This run began before the source commit was
created; it is not a whole-tree invocation-provenance claim. Ruff and relative
documentation-target checks passed.

Mac reports are fsynced at `artifacts/tools/cuda-artifact-guard-release-mac-oct7`:
JUnit459308 bytes / `93eb65eaeee115e9d3036abd228f743f4cb1189f11fbfd96c2025e9e7e7db7e9`,
log2833 bytes / `c8abb9ea4aa62c29f4ca6b3b643348357d4306c83f257caad8f2f3dbb8322f46`,
source bindings4190 bytes / `015b5290c3b998300e5dee30c1dc8237a34fbdfaced6e584e6337b056c1434da`,
verification527 bytes / `a8ec3a9e91a4c3ca807b59bf73c0b2189d57f32b346391031e8db71b2d084e5f`.
Earlier54/55-test drafts and both broader draft reports were preserved.

The clean lean100.98 worktree fast-forwarded from the base using a verified
bundle (SHA256 `16d956ea75d0608717aca0e0454002bdd37bca8b27badaaea78292a1efc20864`).
Unit `microduck-cuda-artifact-cpu-9f7f803a.service`, invocation
`8bb61003144e45b5a551372507042448`, passed **56 focused tests in0.32s** at
22:49:36–22:49:37 Shanghai with empty CUDA visibility. It retained success/exit0,
MainPID0, no restarts, exited state and a deallocated cgroup. Actual caps were
45s overall/30s subprocess,1GiB memory,100% CPU,32 tasks, Nice10 and8MiB file
limit. Reported peak memory9175040 bytes is the unit's accounting counter, not
independent memory instrumentation. The waiting local `systemd-run --wait` SSH
client was terminated **after** the unit completed because `RemainAfterExit=yes`
retains active/exited state; the unit itself was not stopped or reset.

Native raw evidence is retained on both hosts at
`artifacts/tools/cuda-artifact-guard-native-9f7f803a`: JUnit8074 bytes /
`f21fcf71a6bbe894ca2dccaa2a8773c8411e921bfba5078cfee85c2b20c2e87e`, receipt780
bytes / `1f06ce2c596e16f42a84b980bd4ee0d490210bf79cde5df7397c301761df3ca9`.
The separately attempted journal log is empty: stdout was observed through
`--pipe`, not retained by journald. Do not treat the empty file as a pytest
stdout log. Full JUnit and terminal unit state establish the bounded CPU test
result; no broader native regression was run in this slice.

Mac authenticated the whole native artifacts before XML/JSON decoding, checked
56/zero omissions and matched both complete source/test files to the final Mac
bindings. Independent proof659 bytes /
`bf52468b9bf4b1a4390e72f4b36afcf25bfa7d5290b3f9f3ef038a86fc06af9d`
is retained separately at
`artifacts/tools/cuda-artifact-guard-independent-9f7f803a/verification.json`.
Frozen package versions and Warp context source stayed unchanged. No running
Duck GPU process was observed; FilmBrain PIDs521/298048 and inactive protected
services stayed unchanged. The existing WAN lease remains a regular empty file
at inode35886 and was neither acquired nor replaced by these CPU tests.

**Remaining delivery gate:** build/test/predeclare the actual bounded CUDA
producer and independent receiver before any of its30 synthetic launches.
Neither a loaded CUDA artifact nor a learned Duck capability was obtained here.
