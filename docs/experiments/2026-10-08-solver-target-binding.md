# Dense solver static target contract, October 8

This separate CPU-only contract prepares the missing solver-target evidence.
It does not change any historical receiver, simulator, row order, learner,
GPU admission rule or numerical tolerance. See the retained
[binary-scope rejection](2026-10-08-solver-binary-scope.md).

## Fixed source contract

[The new checker](../../src/mjlab_microduck/stance_solver_target_binding.py)
accepts only the complete frozen MuJoCo Warp 3.8.1 `solver.py` bytes with SHA256
`bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
Without importing the installed solver, it checks the literal target definition
`mujoco_warp._src.solver.update_constraint_init_qfrc_constraint_dense`:
decorator, typed parameter order, early return, ascending row reduction and
output write must have the exact AST. The exact direct `if m.is_sparse` branch
in `_update_constraint` must contain the sparse launch and the dense `else`
launch with dimensions `(d.nworld, m.nv)`, inputs
`[d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done]`, and output
`[d.qfrc_constraint]`. Duplicate definitions or other target references are
rejected. Structural matching ignores formatting, but the admission API first
requires the whole frozen file hash. The lower-level structural helper alone
does not authenticate source bytes.

The only decision is `static-dense-solver-contract-only-not-native`. Actual
dispatch, binary binding and instruction observation are false, as are all
native/full-window/runtime-cause/training/physical flags. This is a source-byte
contract, not proof of the path/package from which a caller obtained those
bytes, nor a security sandbox against hostile Python instrumentation.

## Still required before a native solver diagnostic

Keep any new collector separate from the expired historical probes. Authenticate
its clean exact branch and committed bytes, frozen package/interpreter and
resolved installed solver source path. Bind the literal kernel's module, actual
sm120 device/context, private fresh cache and `ModuleExec`, complete generated
source/metadata/CUBIN, explicit load and forward hook. Observe the target actually
dispatched by the unchanged dense branch with the declared dimensions and data
arguments. Compile-only identity and sibling friction/contact artifacts are
insufficient. Reuse the existing
[artifact guards](../../src/mjlab_microduck/stance_cuda_artifact_binding.py) and
reviewed explicit-load patterns; do not expand an old source fence.

Pin an existing disassembler's whole bytes, version/help and selected command.
[NVIDIA documents](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)
`cuobjdump --dump-sass --function <exact-symbol> <cubin>`; reject output that
contains another function or omits the declared target. A string match for FMA
in a whole module is not target-instruction evidence. No tool has been installed
or run, no instructions observed, and no compiled-solver arithmetic claim is made.
Bound output, time and resources; preserve all workloads and require the
separate native protocol's GPU gate before execution. Even this evidence would
not accept the unresolved full-window numerical result or authorize training.

## Current shared-GPU refusal and tool inventory

At source `50f9331e79b4fc14f33cbd5abde75cb5700fece3`, the separate CPU-only
two-sample precheck retained its first observation at **14:40:41 Shanghai**:
Windows **70 C / 0%**, WSL **66 C / 34%**, each **5,911 MiB used /
18,251 MiB free**. Queries are sequential, not simultaneous. The unchanged
`shared thermal guard` refused admission before a second observation or any
CUDA child/service. A preceding cooler snapshot did not reserve Windows demand.
FilmBrain remained active at PIDs 521 and 298048 with zero restarts, and both
protected services stayed inactive in user and system scopes. No retry loop,
service stop, environment change or learning run occurred.

Retained precheck on Mac and WSL:
`artifacts/tools/shared-cuda-precheck-50f9331e-oct8.json`, SHA256
`b8a29277c8afe41adef3e541695411e3723b25546cc84b88c53d6ba65835361b`.

A bounded read-only Windows inventory checked PATH and immediate subdirectories
(including their `bin` directories) under the NVIDIA CUDA Toolkit and NVIDIA
Corporation roots for `cuobjdump.exe` and `nvdisasm.exe`. The toolkit root was
absent, the Corporation root present, and no tool found. Together with the
earlier WSL path checks, this is a missing resolved tool, **not an exhaustive
installation inventory**. No binaries, driver or packages were installed.

## Focused source checks

[Tests](../../tests/test_stance_solver_target_binding.py) reject changed
arithmetic, decorator, signature, row bounds, early return, dimensions,
arguments/order, outputs, sparse/dense placement, extra references, duplicate
definitions and bad source lengths/types. A fresh subprocess proves importing
the checker does not import Torch, Warp or MuJoCo. The installed file is read as
bytes, not imported. All **87 checks** passed on Mac (23 new static-target tests
plus 64 unchanged historical scope tests). Independent read-only review found
no correctness blocker and required the installed-path/package provenance to
remain a separate future collector obligation. It did not rerun the tests.

Same-source WSL CPU verification and retained source-bound reports are the next
delivery step. No GPU execution or new Duck skill is claimed.
