# Dense solver dispatch packets, October 8

Status: same-source CPU-tested packet staging and exact dispatch integration on
Mac and frozen WSL. **No native
collector, GPU dispatch, numerical qualification or policy training is reported.**

## Bounded next slice

Base `bf2dfd6f132d78bb780bf0e919d1ba18e70a6eb2`, exact branch
`feat/athletics-obstacle-curriculum`. Owned delta is this document,
[`stance_solver_packets.py`](../../src/mjlab_microduck/stance_solver_packets.py),
its test, and the narrow integration in
[`stance_solver_dispatch_guard.py`](../../src/mjlab_microduck/stance_solver_dispatch_guard.py).
The [explicit executable preparation](2026-10-08-dense-solver-executable-preparation.md),
installed solver, arithmetic, row order, tolerances, historical captures/source
fences/deadlines and training gates are unchanged.

Independent read-only review confirmed the capture must be inside the literal
target branch of the existing launch wrapper. `_update_constraint` first
enqueues the EFC-force writer, then the dense target, then Gauss-cost work.
Capturing outside the whole caller would observe the wrong force/output phase.

## Capture contract

`DenseSolverDispatchGuard.run(capture=...)` accepts only the exact
`DenseSolverPacketCapture` class. Existing no-capture callers retain their
behavior and receipt schema. Guard construction/run/launch/receipt remain on
one held thread. All existing source/frame/argument/default/stream/executable
and buffer-identity checks still apply.

After authenticating the literal target launch, before delegating it:

1. Synchronize the held stream to retire preceding EFC work.
2. Copy the complete five buffers to distinct owned pinned CPU staging arrays,
   explicitly passing that same stream to each held Warp copy entry.
3. Synchronize that stream again, then serialize only CPU array views.

Immediately after the original target launch, before the wrapper returns and
before later Gauss work is enqueued, repeat those steps for the post packet.
Runtime entry objects/code, source/staging layouts and pointers are rechecked
throughout. GPU `array.numpy()` convenience copies/default-stream readback and
device-wide synchronizations are not used by this helper. The frozen runtime
and copy implementation still require independent owner authentication.

| Full field, in wire order | Shape | Wire dtype | Bytes per phase |
| --- | --- | --- | ---: |
| `nefc` | 64 | `<i4` | 256 |
| `J` | 64 x 512 x 20 | `<f4` | 2,621,440 |
| `force` | 64 x 512 | `<f4` | 131,072 |
| `done` | 64 | `\|b1` | 64 |
| `qfrc_constraint` | 64 x 20 | `<f4` | 5,120 |

Each immutable raw phase is **2,757,952 bytes**, pair **5,515,904 bytes**, below
the fixed 8 MiB pair cap. Include inactive rows, padding and preexisting output
sentinels. Retain signed-zero, NaN payload, infinity and boolean carrier bits;
do not finite-filter, cast numeric values, normalize or serialize number lists.
The detached manifest has fixed shape/dtype/order/offset/size, whole and field
SHA256, stream handle, timing-change statement and unchanged false admission
flags. A changed-input-bits indicator is diagnostic, not numerical acceptance.

Failed sync, copy, readback, target/remainder dispatch, ownership or binding
checks consume the attempt and prevent a successful pair/guard receipt. Owned
hooks are restored; foreign hooks are preserved. Complete pre-packet bytes may
remain available for failure diagnosis without implying a complete pair.
Retained bytes are rehashed before receipt or retrieval. Python instrumentation
can still tamper with mutable state; this is not a hostile-code sandbox.

The helper has no CLI, subprocess, archive writer, supervisor or admission
decision. Native artifacts must separately retain both raw files plus the
guard/executable/source/environment identities, not just this manifest. A mock
capture receipt cannot establish native dispatch or actual driver-loaded bytes.

## Source verification

The seven-file integration scope (new packet tests plus the preceding six-file
executable/guard/disassembly/target/artifact/ELF scope) passed **294 tests in
69.47 s on Mac**, CUDA hidden, numerical threads one, zero failures/errors/skips
in bounded complete XML. There are 38 new packet cases. Synthetic tests execute
the real frozen caller Python code, but use fake launches, streams, arrays,
allocation/copy entries and CPU NumPy; they do not load Warp or execute CUDA.

Tests prove capture sees the preceding EFC update and precedes subsequent Gauss
work; done-world sentinels, inactive/special/boolean raw bits, exact raw lengths
and hashes, detached receipts, single-use/class/thread/stream ownership,
runtime/staging/retained-byte mutations, pre/post failure, hook preservation and
post-fault pre-packet retention. An initial focused run had three synthetic
fixture attribute failures (missing replacement-buffer/device attributes);
fixtures were corrected without changing refusal boundaries. Its output is not
accepted evidence. The corrected focused scope passed 75 cases before four
additional cases and the final 294-case integration run.

Mac XML: `artifacts/tools/dense-solver-packets-mac-oct8/integration.xml`.
A full repository suite has not been run. The matching capped WSL checks below
complete source delivery closeout; neither CPU suite is GPU admission.

### Same-source WSL closeout

Executed source `230b91bdd3a3de6cb2da5220e424c82999f09e66` was committed and
pushed to the fork feature branch, then installed on clean exact WSL by a
whole-hash-verified fast-forward bundle. Service
`microduck-solver-packets-prep-cpu-230b91bdd3a3.service`, invocation
`d146f0fb1b294880a08f1d8127288d66`, succeeded with exit 0, MainPID 0 and no
restarts: **294 passed in 9.89 s**, complete XML with zero failures/errors/skips.
Limits remained 90 s, 2 GiB memory, CPU quota 100%, Nice 10, 64 tasks, 8 MiB per
file, control-group cleanup and no restart. CUDA was hidden, `PYTHONPATH` named
the exact source tree, and all four numerical thread settings stayed one.

| Retained whole file | Bytes | SHA256 |
| --- | ---: | --- |
| Mac integration XML | 309,019 | `e0e83e01dc975e433721df6b3f68f551c763e91785a2c3e2233d95be79c658be` |
| WSL integration XML | 309,022 | `548fbf09377116d6043f2d9d743375249a7c8fb92363b92bcc42014111a6e2fa` |
| WSL service journal | 1,148 | `71f413d527d056d83b682a68f0931aebf1ac6ff531fcf59444e7c823d710d191` |

Native directory `artifacts/tools/dense-solver-packets-native-230b91bdd3a3`
has closeout JSON SHA256
`26e80b2085f01cf1e1376cbc97c0a6cc2ac356c04e5fe3366dc7e3f08d9bba27`.
Its authenticated Mac copy and every indexed artifact match. The source,
frozen package/interpreter versions, before/after installed solver hash,
service settings, runner hash and workload snapshot are retained. Both installed
solver leaves still match the original frozen whole hash. No environment,
driver, package, historical cache/capture, lease or unrelated service was changed.

At `2026-10-08T16:56:54+08:00`, sequential Windows/WSL NVIDIA samples showed
0% utilization, 6,923 MiB used / 17,239 MiB free, temperatures 32/31 C, unchanged
GPU UUID and driver. Windows active 3D-engine counters were 12% and 1%, without
NVIDIA process attribution here; samples do not establish idle ownership.
Every Duck user-service MainPID was zero. FilmBrain kept PIDs 521/298048 and zero
restarts, and protected services remained inactive in both scopes. No native
compile/load/dispatch, lease acquisition or Duck GPU process occurred.

## Host and next actual native gate

The renewed read-only `.98` check found the clean exact base, unchanged frozen
packages/solver and preserved FilmBrain PIDs 521/298048 with zero restarts.
Protected services stayed inactive in both scopes. The Linux GPU snapshot read
33 C, 0% utilization, 8,023 MiB used / 16,139 MiB free. This is not an exclusive
lease, Windows ownership attribution or authorization to overlap workloads.

The full native runner is still missing. It must assemble the reviewed
source/environment/compiler/library closure, held existing Wan lease, fresh
private cache and explicit executable load, current Windows/WSL telemetry,
process-tree/resource/deadline supervisor, bounded pinned disassembler call,
and meaningful child-owned scratch model/data/context. Historical post-init
captures are not pristine target inputs: the actual target packets above must
be captured after its preceding EFC writer. Do not replace meaningful state
with empty/zero arrays and infer a cause, or relabel older friction binaries as
solver evidence. The scratch caller mutates state and cannot use shared data.

Once that separate runner passes its CPU/source/admission prerequisites, one
bounded diagnostic can collect actual raw packets for independent replay and
numerical interpretation. Training remains paused; no actor/perception,
obstacle/hopping/football promotion or physical motion follows from this slice.
