# Caller cost-stage protocol preparation

## Authority and source boundary

New preparatory protocol, based at `02a5c605417fac8268d8d9baf826a140010af3e2`.
Only the new `stance_solver_cost_stages.py`, its test file and this note change.
Historical capture/receiver protocols, existing source fences, libraries,
drivers, GPU leases and other workloads remain untouched.

This chunk implements raw-bank layout, literal synthetic controls, ordered
same-stream readback and a pure CPU receiver. It deliberately does not implement
a GPU owner/runner or authenticate a dispatch callable as a kernel. No native
job, full simulation, learning, raw perception, video or physical motion is
admitted. All qualification flags remain false, even if every packet check passes.

## Predeclared bank and control

Each snapshot is the complete 24-field scratch restoration bank, in unchanged
`RESTORE_ORDER`, with shapes/dtypes from the frozen 66-field initialization ABI.
One bank is 3729992 bytes. Two arms, each with eight snapshots, use 59679872 bytes
below the 64 MiB packet budget. Padding bytes are retained, not normalized.

Both arms begin with a sealed authenticated historical original forward-4 bank.
The control changes only four Gauss drivers and three scalar destinations:

- Every DOF: `Ma=1`, smooth force `0`, acceleration `1`, smooth acceleration `0`.
- World w: Gauss canary `(-1)^w * (4096+w)`, cost `1024+w`, prior cost `-2048-w`.

These are synthetic solver controls, not physical plant states or policy inputs.
The Gauss contribution must be exactly 10 in every world; adding 10 to the
immediate pre-Gauss cost must change each destination and match binary32 RNE.
This nonvacuity check prevents a huge existing cost from hiding a skipped add.

## Four ordered stages

Capture complete before/after banks for `init_cost`, `efc`, `dense`, then `gauss`.
Every after-bank must equal the following before-bank exactly. Check all fields
outside each stage's write set as complete bytes, including inactive padding:

| Stage | Allowed writes | Numerical check |
| --- | --- | --- |
| Setup | Gauss, cost, prior cost | Gauss/cost positive-zero reset; prior cost bitwise copy of incoming cost |
| EFC (`track_changes=False`) | force, state, cost | Inactive force/state tails preserved; cost finite/nonnegative; no atomic row-cost oracle |
| Dense | constraint force | Existing ascending-row float32 product/sum hypothesis and 2e-5 absolute/relative tolerances |
| Gauss (20 DOFs, one thread/world) | Gauss, cost | Existing exact-rational two-hypothesis consistency bound; cost is RNE(immediate pre-stage cost + captured contribution) |

Require all worlds active, bounded positive row extents and no active elliptic
rows. The omitted contact-friction field is only safe under that row guard.
Both arms must retain the same EFC force/state and dense-force output banks;
EFC cost itself may differ because atomic accumulation order is not qualified.

The receiver authenticates all sixteen complete raw hashes against externally
supplied anchors before decoding any bank. The future collector must obtain
those anchors from an independently authenticated complete inventory. Passing
invented packets with matching invented hashes is not capture-origin evidence.

## Readback ownership and failure behavior

`CostStageCapture` holds source allocation identities/pointers, pinned CPU
staging arrays, copy/sync/CPU-view entry points, the owner guard and an explicit
same-device stream. Each bracket synchronizes preceding work, copies every
field on that stream, synchronizes, then reads only the CPU staging arrays.
No GPU convenience readback, implicit stream, kernel replacement or launch hook
is installed here. Readbacks change timing and do not establish production
atomic ordering. A failed/duplicate/out-of-order attempt cannot be retried; any
complete pre-failure snapshot remains recoverable but has no successful receipt.

The supplied owner guard must bind the frozen runtime and model/context. A
separate future caller adapter must bind the four actual original call sites,
kernel/factory code and closure values, dimensions, argument identities,
track_changes=False, exact executable objects and explicit module artifacts.
The capture helper alone does not prove those facts.
Binding attributes are sealed; runtime entries, captured phase bytes and their
capture-time hashes are exposed only through read-only directories. Snapshot
insertion is internal and only allowed during the correct active bracket;
this is observed-object integrity for trusted in-process code, not a sandbox
against arbitrary Python reflection or instrumentation. Every capture receipt
also explicitly leaves origin, executable binding, GPU dispatch and external
retirement authentication false.

## Verification and native gate

Focused synthetic tests cover positive/nonzero controls, reset/copy exactness,
stage continuity, disallowed writes, padding/tail preservation, hash-before-decode,
domain/finite-value refusal, cost-add nonvacuity, same-stream CPU-only readbacks,
entry/array mutation, one-shot ordering and recoverable failure snapshots. Run
these plus the retained scratch/replay/Gauss contracts on Mac and WSL with
`CUDA_VISIBLE_DEVICES=''` at the same exact source revision.

Before any GPU invocation, the missing native adapter/owner must be implemented
and tested with its own exact source fence, paired CPU evidence, fresh deadlines,
shared-capacity/service checks, inherited FilmBrain lease, resource caps, exact
module bindings, exclusive artifact paths and independent cgroup retirement.
Do not widen a historical fence to run new code. Do not train on synthetic banks.

## Retained CPU closeout

Executed source: `ba58241c04044ce9297754b9ee6e555d6efa0a96`, clean exact feature
branch on both Mac and 100.98. Thirteen focused files were run with CUDA hidden:
the eleven retained scratch/replay contracts, the Gauss arithmetic tests, and
the new cost-stage tests. This is not the full repository suite.

- Mac: 540 passed, zero failures/errors/skips, 22.01 seconds. XML 895011 bytes,
  SHA256 `152651745cf9ce9c39853733159f2309c43a93510e4753e43ae3499d991dbd4c`.
- WSL: 540 passed, zero failures/errors/skips, 20.49 seconds. XML 895015 bytes,
  SHA256 `36c1782260f0b7e16a15281adee7e606677effdf9896ff3e1dbb661c686c2ec6`.
- The 68 new tests used synthetic packets and CPU mock array/readback objects.
  Their successful nonzero control is not a native GPU result.

Retained XMLs on Mac and WSL:
`artifacts/tools/caller-cost-stage-preparation/ba58241c-mac-tests.xml` and
`ba58241c-wsl-tests.xml` (the Mac XML is local; the WSL XML is retained on both).
The source bundle SHA256 is
`bfe392ca154442192c6c0e51cfacfd50613c59fb6b96c4d4817e0901a5d5820f`.

The WSL capped CPU unit `microduck-cost-stages-cpu-ba58241c.service` used
invocation `888494327e84458f93a59084b200fefb`; it finished with `Result=success`,
`ExecMainStatus=0`, `MainPID=0`, empty cgroup and zero restarts. Its retained
`active/exited` state is not a running workload. Caps were 150 seconds, 4 GiB
RAM, 200% CPU quota, Nice 10, 64 tasks, 16 MiB/file, no restart and control-group
termination. No GPU lease was acquired and no GPU workload was launched.

The read-only Luna review prompted sealing binding attributes and making both
capture bytes and capture-time hashes read-only. Regression tests now refuse
coordinated binding or packet/hash reassignment; explicit false provenance
fields prevent a capture receipt from being mistaken for executable/dispatch
authentication. A final read-only review found no remaining concrete contract
defect; the reviewer did not run tests or native work.

After the CPU run, frozen interpreter/alias, packages, runtime libraries, Warp
and MuJoCo-Warp source trees and the disassembler pins reverified unchanged.
Both FilmBrain services remained active at PIDs 521 and 298048, zero restarts.
Both protected AI mission services remained inactive in user and system scopes.
No Duck unit was running. The WSL point-in-time GPU reading was 8023 MiB used,
16139 MiB free, 0% utilization and 31 C: shared capacity, not exclusive-idle
ownership. The next required work remains the separately tested native
caller/executable adapter and capped owner, followed by independently received
native stage evidence. Simulation/training acceptance has not advanced here.
