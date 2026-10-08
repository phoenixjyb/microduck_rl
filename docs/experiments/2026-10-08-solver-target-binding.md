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

## Same-source CPU closeout

Executed source `b1f0a2a14f00e2f1fdcf46c8286568ba725b9b71`, checker SHA256
`00b00db6599d746c64f218c0b5a917ea66e46f622e4225cb59ab71edf946c989`.
Mac passed 87 checks in 0.52 s; WSL passed the same 87 in 0.47 s, zero
failures/errors/skips. WSL unit
`microduck-solver-static-target-cpu-b1f0a2a14f00.service`, invocation
`9c3c9cbb55f3471b8741ccf66691dea0`, exited zero with MainPID zero and no
restarts. Its ceilings were 90 s, 2 GiB RAM, 100% CPU, Nice 10, 64 tasks and
4 MiB/file, not measured peaks. CUDA was hidden for both CPU runs.

Each replay binds the clean exact execution source and whole committed checker,
reads its resolved installed 3.8.1 solver path without importing the package,
and emits the same **974-byte canonical static contract**, SHA256
`53d41f7cae73ee85ab5186bd06b4121fd6ff65318e3e8e30ef6b4dc7d8fda1a8`.
Path provenance is separately recorded; the static API alone does not prove it.
No device package was imported in either replay. Native flags remain false.

Retained leaves copied between Mac and WSL and rechecked by whole hash:

| Evidence | SHA256 |
| --- | --- |
| Mac JUnit | `0554e0baa1c8961217042e632afa1e78ed1a8c54d007fa73a934878ca937336d` |
| Mac source/path-bound replay | `f2d0420bafc137b9f1c497d2af637aba080843f22ba80b9cc1f97e186de68b5f` |
| WSL JUnit | `1655cdb7c53579bf2aac99df27f1489825b34c4137caa384e42ca9f2569aed77` |
| WSL pytest log | `88c7225da7f027530fc050810b788d001dea4ba8bbfa0883f931ae2e8326b03f` |
| WSL source/path-bound replay | `41006321f2a8db43760c4b41fc239878019c7bc26cae3e1fd1f486e126fd3462` |
| WSL service/environment closeout | `d6688de2f606477500e0d359db8fe8ec3c63bdec364cc8067f77e85a3db93745` |
| Windows bounded tool inventory | `ca62f26cabc33ad7b6ec4db7db4dc6583c24f9770ae06dc8be43344fd64433eb` |

The CPU namespaces are `artifacts/tools/solver-static-target-mac-oct8` and
`artifacts/tools/solver-static-target-native-b1f0a2a14f00`; the Windows inventory
is `artifacts/tools/windows-cuda-tool-inventory-oct8-50f9331e.json`.
The first bundle fetch requested a branch ref not carried in that bundle;
read-only inspection identified its `HEAD` ref. Fetching that ref and checking
the exact source permitted a clean fast-forward before starting the CPU unit.
No test service or GPU child had started on the failed fetch.

Closeout verified every retained Duck unit's MainPID zero, FilmBrain unchanged
at PIDs 521/298048 with zero restarts, protected services inactive in both
scopes, and frozen interpreter/packages unchanged. Python syntax, local doc
links and whitespace checks passed. The full repository suite and native
solver execution were not run. No GPU execution or new Duck skill is claimed.
