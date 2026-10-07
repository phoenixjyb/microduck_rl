# Dense friction-row CPU component fixture

## Question and exclusions

The [authenticated BAM row-address audit](2026-10-07-bam-row-address-audit.md) found a force-preserving row→DOF permutation at proposal 1 and real addressed force-bit changes beginning at proposal 2. That does not establish a cause. This new **standalone CPU component fixture** asks only whether a single writer per world can build ascending-DOF dense friction rows with the original kernel's complete per-address float32 fields and count/capacity semantics.

The [fixture module](../../src/mjlab_microduck/stance_friction_row_cpu_fixture.py) does not intercept or replace a runtime controller. It uses no robot model, simulation forward/step, solver, integration, contact, tendon, sparse constraints, CUDA kernel, graph, actor, learner, storage, optimizer, video, perception or physical device. No package, driver or existing cache is upgraded. Every result keeps cause/native/full-window/training/physical qualification false. The original negative repeat gate is unchanged.

## Frozen component and ownership

Allowed CPU environments are Darwin/arm64/Python 3.12.12 and Linux/x86_64/Python 3.12.13, Warp 1.12.0 and MuJoCo-Warp 3.8.1. Every invocation requires literal empty `CUDA_VISIBLE_DEVICES` and explicitly allocates/launches on `cpu`. The complete 69-file installed Python tree is SHA256 `188d58bfac6ab54e51adfc80a6c504d782aec43856cbb15e54b4109f0ce30a1d`; `constraint.py` is `b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53`. Canonical tree hashing includes a trailing newline. Original kernel/helper, candidate kernel and launch object/code identities plus source bytes are checked before and after execution. The matrix requires one unchanged binding across its eight cases. In-process object IDs are evidence, not portable identities across processes or hosts.

Each call copies one caller-independent NumPy input snapshot, clones it into separate fresh original/candidate CPU buffers, and executes each kernel once. The candidate iterates DOFs 0–19 in one thread per world. It retains the original positive-friction criterion, model-field broadcast modulo, `nf` then `nefc` increments, dense Jacobian zero-and-unit-column, and exact original `_efc_row` helper/arguments. Overflow still increments complete counters; the candidate continues subsequent DOFs instead of exiting its sole world thread. A report with overflow cannot qualify, even if visible subsets happen to match.

Inputs are finite bounded plain contiguous float32 arrays with 1–4 worlds, 20 DOFs and row capacity 1–32. Reports retain every input shape/uint32 bit, both complete counters, every visible row's original position/DOF/type/Jacobian/position/margin/D/velocity/reference acceleration/friction-loss bits, and per-field addressed comparisons preserving multiplicity and signed zero. Invalid or nonfinite constructed rows are rejected. Positive input DOFs independently determine expected counts and candidate addresses; equal malformed outputs cannot pass merely by agreeing with each other.

## Predeclared CPU matrix

Protocol: `microduck-dense-friction-row-cpu-fixture-oct7-v1:eight-case-matrix`. The CLI is `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m mjlab_microduck.stance_friction_row_cpu_fixture --output <new-existing-real-directory>/matrix.json`. It anchors every directory component with no-follow directory descriptors, disallows parent traversal, writes a bounded complete canonical report exclusively relative to the held parent descriptor, fsyncs file and parent directory, and refuses existing/symlink outputs before launches. A failed matrix expectation is retained as a negative report and a nonzero command result; it is not silently omitted.

| Case | Worlds / parameter rows / capacity | Coverage and expected component decision |
| --- | --- | --- |
| empty | 1 / 1 / 32 | No active rows; exact zero counts |
| broadcast | 3 / 1 / 32 | DOFs 1, 6, 19; full addressed fields exact |
| per-world | 4 / 4 / 32 | Distinct positive DOF sets; full addressed fields exact |
| mixed-broadcast | 4 / independently 1 or 4 / 32 | Independent field modulo, timesteps and REFSAFE clamp; exact |
| all-dofs | 1 / 1 / 20 | All 20 DOFs exactly fill capacity; exact |
| signed-zero | 1 / 1 / 32 | DOF 6 velocity retains `0x80000000`; exact |
| direct-solref | 2 / 1 / 32 | Negative stiffness/damping parameter form; exact |
| overflow | 2 / 2 / 2 | Five proposals per world, two visible rows; deliberately negative, no qualification |

The non-overflow cases must have complete expected counts/addresses, ascending candidate order and identical complete addressed row fields. Overflow must retain complete proposal counts, ascending visible candidate addresses and the negative decision. `component_expectations_met` refers only to these predeclared CPU checks. It is not native CUDA admission. Cross-host comparison must keep platform/source/receipt evidence separate and compare the complete input and row fields; do not discard a difference or describe CPU arithmetic as a runtime solver result.

## Validation and retained evidence

Owner review and independent read-only review found no concrete dense-row semantic defect. The initial 19 focused tests and the expanded 31-test matrix/CLI suite passed. Independent review identified ancestor-symlink redirection in the CLI; the owner replaced path-based writing with held no-follow directory descriptors. Ancestor/traversal rejection, held-parent redirection during computation, oversized-payload rejection before creation and retained expectation-failure tests are included. The hardened **36 focused tests passed in 1.72s**, with Ruff lint/format checks passing. The final **79-file/2373-test Mac CPU regression passed in 168.52s**, with zero errors, failures or skips. Retained at `artifacts/tools/friction-row-final-mac-cpu-oct7`: JUnit 438928 bytes / SHA256 `2d3e9cf73c130250e1eb4646913a3d7b51707a262616b2a1fa3e4ce34a43a1fc`; log 2673 bytes / `5094d91eb8f1addddef89b23c8f0b15f012663f56fc26958b1c902e1473766c4`; verification 3965 bytes / `575cec2b55c69e2656e45362284c78387f5ac601b297619463418833080c1891`. Earlier 79-file/2356- and 2368-test artifacts precede the added matrix or path tests and are not the final source closeout.

The initial Mac matrix met all eight expectations: `artifacts/tools/dense-friction-cpu-matrix-oct7/mac-matrix.json`, 89821 bytes, SHA256 `3a4987da9266d3f8828593258f3d66e85e9aecc863a7225a422b771a4c149713`. It is retained **draft** component evidence before CLI path hardening, not final source/native closeout. Final evidence must identify the exact tested source revision, complete file/test counts, zero skips, report sizes/hashes and native CPU reproduction. No new GPU job is predeclared here.

The hardened final Mac matrix also met all eight expectations, retained at `artifacts/tools/dense-friction-cpu-matrix-final-oct7/mac-matrix.json`: 89821 bytes / SHA256 `1f752ba2dadd1d3f735cce4e27e17ae92d4f93e0a37129a5bdf561c2bf1ef52e`. Its fixture module bytes are SHA256 `8e96c5077f248aad06cacab4c3a45f17cd6b3c35c09c7a46457281916e8046d9`. Whole report hashes contain process-local identities and cannot be assumed portable; compare actual complete component inputs/outputs separately.

The next authorized reproduction is CPU-only on the clean exact pushed feature tip in `/home/yanbo/work/microduck_rl-com-entry-20261006`, using the frozen environment and literal empty CUDA visibility. Use a unique retained user unit `microduck-friction-cpu-tests-<source12>.service`, at most 360 seconds overall / 300 seconds for the complete 79-file/2373-test pytest subprocess, MemoryMax 6GiB, CPUQuota 200%, Nice10, TasksMax64, no restart, control-group cleanup, 10-second stop timeout and retained exited state. Capture the exact source/tree/whole tracked-file hashes, actual service invocation/PID, full test list/log/JUnit, complete authenticated row audit and complete eight-case matrix in `artifacts/tools/friction-cpu-reproduction-<source12>`. Independently authenticate whole output bytes before JSON/XML decoding on the Mac, compare all row-audit bytes and all component input/output bits separately from process/platform metadata, retain both reports on both hosts and fsync them. Require at least 60 seconds of cutoff reserve. This is a new CPU-only protocol, not reuse or expansion of the prior BAM/RNE producer fences. It admits no CUDA execution.

## Follow-up gate

Even a passing matrix warrants only review of a separately predeclared fresh bounded CUDA **component** probe. That would require a clean pushed source, fresh full native CPU proof, exact original/candidate source and compiled-kernel bindings, complete dense row packets/counts, negative overflow handling, one owned child under the existing shared GPU lease, resource/deadline caps and independent authenticated replay. It must not patch the retained robot runtime or reuse a previous run namespace. Runtime replacement would still require its own bounded input/output and full-window gates; training remains blocked until those gates pass. No CUDA component job is authorized by this document.

Current work ends at **2026-10-07 13:00 Asia/Shanghai**. Only 100.98 is in scope. Preserve FilmBrain, inactive protected AI-mission services, old failed units, frozen environments and all original artifacts; never restore services or perform physical motion in this slice.
