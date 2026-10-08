# Dense solver executable preparation, October 8

Status: bounded compile/explicit-load integration implemented and Mac CPU-tested. **No native
solver probe or policy training is admitted or reported by this slice.**

## Scope and reason

The [dispatch and disassembly preparation](2026-10-08-dense-solver-probe-preparation.md)
already passed same-source CPU checks on Mac and frozen WSL. The next missing
integration is the bridge from the literal installed solver kernel to a fresh
compiled artifact and an explicitly loaded forward hook. A directory of older
friction/contact binaries cannot supply that bridge. The historical
[solver-init capture](2026-10-08-passive-solver-initialization.md) records
post-initialization state: it is not a pristine input for a new solver call.

This slice starts at `afa86bf720a8a867a11687eceb79fbe99ccdb1af` on
`feat/athletics-obstacle-curriculum`. Its three owned paths are:

- `src/mjlab_microduck/stance_solver_executable.py`
- `tests/test_stance_solver_executable.py`
- this document

Do not widen a historical source fence, change solver arithmetic, clone a
kernel into a new namespace, reorder rows, modify tolerances, or reuse an
expired owner cutoff. Independent read-only review identified these boundaries
before implementation. The owner reviews the returned code and checks.

## Integration contract

Import initializes no CUDA package or device. The future fresh child supplies
the authenticated installed runtime, current sm120 device, frozen solver,
private cache and a unique absolute `compiled-dense-solver` directory under its
owned run. It also owns the source/environment/compiler/library pins, lease,
resource supervisor, time budget and telemetry. This helper does not substitute
for those obligations and has no CLI or subprocess/disassembler runner.

One preparation object must:

1. Bind the installed whole solver source and literal target/module, original
   runtime entry objects/code, options/configuration, actual device/context and
   source/options hash. Refuse preloaded executables or failed builds.
2. Compile the existing target module once without loading it. Retain exactly
   the generated CUDA source, metadata and CUBIN, with bounded whole bytes and
   stable canonical regular-file identities. Other functions in the module are
   not relabeled as the selected target.
3. Select the literal dense target and require complete offline SASS encoding
   bytes to match its retained CUBIN span before any load.
4. Recheck a fresh module immediately before the held explicit `Module.load`
   entry. Supply the exact retained binary/metadata paths, architecture and
   block dimension; authenticate the returned cache entry and forward hook.
5. Recheck all bindings after each phase. Partial or failed attempts cannot
   produce a successful receipt or be silently retried. Preserve produced
   artifacts and foreign cache state rather than deleting or resetting them.

The compiler may generate multiple functions in the solver's existing module;
the offline selector must still admit exactly one literal target variant.
This is compile-input and observed Python/object evidence, **not** an inspection
of driver-loaded machine-code bytes, an actual solver dispatch, a numerical
cause, or a qualification. All admission flags remain false.
It is not a sandbox against arbitrary Python instrumentation or external
filesystem writers: canonical-path and retained-file checks do not eliminate
the loader's external file-replacement race. Preparation does not freeze every
name in the solver's mutable globals; a future dispatch observer must bind its
actual caller/dependencies separately.

## Focused source verification

Owner-reviewed six-file CPU integration passed **256 tests in 75.98 s on Mac**,
with CUDA hidden and numerical thread settings one. Complete JUnit contains
zero failures/errors/skips and remains below the unchanged 8 MiB native file
cap. This includes 46 new synthetic preparation tests plus 210 existing
dispatch, disassembly, static-target, artifact and ELF-scope checks. The tests
delegate only mock compiler/load entries, not Warp, NVRTC or a driver.

Tests cover one-shot order and partial failure, exact generated leaves,
symlinks/size caps, unchanged files/source/configuration, source-code namespaces
and defaults, distinct function objects with identical code, legitimate bound
method reaccess, mismatched target symbols, SASS mismatch before load, wrong
load/hook returns, foreign cache/failed-build state, detached receipts and a
fresh import without device packages. Source mutations use temporary copies
only; the real installed solver's complete bytes are asserted before and after
every new case. The real Mac source still matches its original frozen hash.
Syntax, local documentation links and whitespace checks passed. A full
repository suite was not run. Same committed-source WSL checks remain the next
delivery step, not native compilation or GPU dispatch qualification.

### Native harness failures retained before acceptance

The first source-`0b0f8795fc31` CPU service failed at collection with six import
errors: its pytest subprocess resolved the shared environment's older editable
checkout instead of this worktree. The parent-only `sys.path` setting did not
propagate to the subprocess. A new retained run explicitly sets `PYTHONPATH` to
this exact worktree's `src` and asserts the imported package path; no environment
reinstallation is needed or authorized.

That second run reached all cases but yielded **255 passed, one failed**: the
oversized-binary fixture wrote 8 MiB plus one byte and hit the service's unchanged
`LimitFSIZE=8M`, raising `OSError: [Errno 27] File too large` before the helper's
refusal assertion. This is not a passing WSL suite. Both failed units and raw
outputs remain retained without reset. The test-only repair exercises a real
1,025-byte leaf against a 1,024-byte reader cap, then synthetic `fstat` metadata
against the asserted actual 8 MiB production cap. It still requires rejection
before any load. Production source, input caps, service limits and qualification
gates are unchanged; the repaired exact-source suite must pass before closeout.

## Host check and remaining native gate

The renewed read-only connection to `gw98-direct` reached `DESKTOP-HNKBDR1` and
the clean exact base branch. Frozen versions and interpreter matched their
existing pins. All Duck user-service MainPIDs were zero. FilmBrain observatory
and video playground remained active at PIDs 521/298048 with zero restarts;
the protected AI services remained inactive in both scopes.

The initial Linux GPU snapshot was **31 C, 0% utilization, 6,432 MiB used /
17,730 MiB free**. This is not a shared-GPU lease or Windows-idleness claim.
No remote workload was stopped, no WSL cache/environment/driver was changed,
and no Duck CUDA process was started by this preparation.

### Test-fixture failure and exact Mac restoration

Owner review caught a synthetic source-mutation fixture targeting the live Mac
installed `solver.py` rather than a temporary copy. Test work was interrupted
before acceptance. Read-only examination identified exactly two appended LF
bytes: the modified 102,690-byte leaf hashed to
`8a9c816450e6d7982eb01803947841c38f498a287f27ab599fc43cd4316a4963`;
removing exactly those two bytes yields the original 102,688-byte frozen leaf,
SHA256 `bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
The owner restored that exact file with a two-line patch and verified its
complete length/hash. Its filesystem modification timestamp changed; byte
restoration is not a claim that the environment was never touched.

The assertion that a package upgrade explained the developer Mac source
difference was not established; owner inspection identified these trailing-byte
test writes, not evidence of a new dependency installation.
No bypass of the production whole-source pin is accepted. The fixture must copy
the verified original bytes beneath its temporary prefix, mutate only that
copy, and check the real installed source before/after. Failed pre-repair test
outputs cannot count as passing evidence. No remote solver or package was edited.

A native probe still needs its own reviewed owner/child protocol, exact tested
source closure, fresh private-cache proof, Windows and WSL capacity checks
throughout execution, held existing FilmBrain lease, bounded tool execution,
meaningful scratch model/data/context setup, and raw packets immediately around
the actual target dispatch. Capture must bracket the literal target launch
inside the caller: `_update_constraint` first recomputes EFC forces, so copies
before the whole caller would not represent the target's actual inputs.
The existing guarded `_update_constraint` call
mutates solver outputs/bookkeeping and must use child-owned scratch data, not
shared simulator state. Merely allocating zero-filled arrays or reporting a
load handle does not explain the historical divergence.

Only after those separate checks should one bounded dispatch run. Successful
collection would remain a diagnostic, not permission for a learner, obstacle
curriculum promotion, hopping, football, raw perception or physical motion.
