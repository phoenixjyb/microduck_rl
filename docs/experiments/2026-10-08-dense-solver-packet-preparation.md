# Dense solver dispatch packets, October 8

Status: CPU-tested packet staging and exact dispatch integration. **No native
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
A full repository suite has not been run. Same committed-source capped WSL
checks are required before delivery closeout; neither CPU suite is GPU admission.

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
