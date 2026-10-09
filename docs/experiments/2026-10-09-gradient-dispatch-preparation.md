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

## Paired CPU closeout and WSL resource diagnosis

Execution source `34591828b78c105c932e5a56978ce734551845fe`, tree
`e59af4c6c3f85d67c484be825d82ee8dd802ea8b`: all 986 committed plain-file blobs
matched before and after execution. This evidence section is a later
documentation-only closeout, not a relabeling of those tests at a new revision.
The separate three-path fence remains unchanged.

Mac passed all 1014 tests across the declared 21 files in 113.67 seconds. WSL
passed the exact same case multiset, including all 87 new adapter cases, in
82.32 seconds with zero failures, errors or skips. Before that successful run:

- The first WSL launcher imported the shared environment's old editable
  checkout. Retain its failure; correct only process-local `PYTHONPATH` to this
  worktree's `src`, without reinstalling or changing the environment alias.
- The first full WSL run passed 1012 tests but timed out in two unchanged
  60-second CPU ABI subprocess tests. The focused retry also failed (one timeout,
  one child SIGABRT / `libc++abi: terminating`). CUDA error 100 with CUDA hidden
  was not interpreted as proof of a GPU fault.
- Both exact original ABI bodies, extracted without modifying their code,
  subsequently completed alone in 5.93 and 5.69 seconds. These isolated runs
  were diagnostics, not substitutes for passing the pytest suite.
- A reproduced original module-materialization pytest test reached the existing
  64-task cgroup limit: owner 1 thread, pytest 20 threads and child 43 threads.
  `pids.events` changed from `max 0` to `max 1`; the unchanged test timed out.
  This directly establishes task pressure in the reproduced CPU recipe, not
  which individual library pool caused each earlier failure or any CUDA cause.

The successful repeat sets only process-local `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` **before
imports**, plus the corrected `PYTHONPATH` and empty `CUDA_VISIBLE_DEVICES`.
The child inherits those limits. No source, assertions, test timeout, package,
driver or shared service setting changed. The original 4 GiB memory, 200% CPU,
64 tasks, Nice 10, 16 MiB file, control-group kill and no-restart limits stayed
in place; the owner service is bounded to 300 seconds with 10-second stop and
core dumps disabled. The two focused cases passed in 10.57 seconds before the
full 1014-test run. Every sampled task event stayed `max 0`; the sampled peak
was 14 tasks. MemoryPeak was 3507884032 bytes, below the unchanged memory cap.
The combined thread-budget intervention removed the reproduced failure; it
does not isolate a specific numerical library or prove native runtime behavior.

Successful unit `microduck-gradient-dispatch-thread-bounded-34591828.service`,
invocation `7638a744164749b38f682c7de9ef391f`, retired with MainPID 0,
Result success, ExecMainStatus 0 and an empty ControlGroup; its original cgroup
is absent. The external closeout verifies the original invocation and absent
cgroup of all seven owned CPU/diagnostic units, retaining failed statuses as
failed. Frozen runtime bindings and protected services matched before/after.
FilmBrain PIDs 521 and 298048 remained active with zero restarts; both protected
AI mission services remained inactive in system and user scopes.

At closeout, sequential Windows/WSL GPU views reported 8210/8209 MiB used,
15952/15953 MiB free, 2% utilization and 35/34 C. A Windows 3D engine remained
active. These are dated shared-device snapshots, not simultaneous counters,
an idle-GPU claim, a lease or reserved future capacity.

Retained files live on Mac and WSL under
`artifacts/tools/gradient-dispatch-preparation/`. Independent Mac verification
checked 23 inventoried plain files / 4545005 bytes, whole hashes, exact CPU test
multisets, source bindings and retired-unit records. Three diagnosed private
CPU cache directories were empty and preserved, not treated as result files.
The paired check and its helper are additionally retained on both hosts.

| Retained evidence | SHA256 |
| --- | --- |
| `34591828-mac-tests.xml` | `397a8d05357e08984477d9848169014f1b9adfc3f5b2e2191c023a8171d7afe2` |
| `34591828-wsl-tests.xml` (original failure) | `88f9b58e2448ee1e3fb180ae21e7d18b89463e339625d3251743a87c40ae8f97` |
| `34591828-wsl-pids-probe.json` | `561472308ad8ca595409dbf96c87e66d9c03fe462427e200f87c06ff4331e7c0` |
| `34591828-wsl-thread-bounded-tests.xml` | `e3f367e3edc30e0a4cc990d1d0940924d995a39655774192cda3fa2cf27c9b1b` |
| `34591828-wsl-thread-bounded-report.json` | `53c34cb120cdeac3964471f085ba6643d2dce4fd2f8cf1bfcf73ba6c38abea74` |
| `34591828-wsl-thread-bounded-closeout.json` | `bdc6bf006304412a49f84648cb46d0966b4e1076b38c15a3f5b877613d55e886` |
| `34591828-paired-cpu-check.json` | `a96c53a99a66e371a7b5c9fde07b5f069bbfa3d12b3dfd2ee3514ca012ac30c2` |

CPU prerequisites for this adapter are now closed. The paired receipt explicitly
keeps all six qualification flags false. No new native gradient capture,
executable provenance, full solver window, simulation-training admission,
learned skill or physical acceptance follows from this result. The next task
remains the separately fenced two-entry loader, bounded collector owner/child
and independent whole-inventory receiver described above; do not bypass those
steps or alter previous contracts to admit a GPU run.
