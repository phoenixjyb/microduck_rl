# October 6 constructor/eager-forward CRB schedule control

## Question and boundary

The [completed full-tree fixed-input control](2026-10-06-crb-full-tree-control.md)
produced five concurrent outputs and one exact serial output bank. That result
does not qualify a runtime trajectory. The next question is whether changing
only the conflicting CRB launch schedule affects repeatability of the **actual
constructor's first forward and subsequent unforced eager forwards**.

This is a separately predeclared no-integration numerical diagnostic. Its CUDA
children must not instantiate an actor, critic, learner, rollout storage or
optimizer, or call `step`, `step_with_schedule`, an integrator, auto-reset or a
graph constructor. The unchanged historical CPU evidence checker and existing
source tests may construct CPU reference modules to verify earlier authenticated
model snapshots; they do not run a new policy rollout or optimizer update.
No perception, physical motion, video, package/driver changes, installed-source
edits or protected-service restores are permitted. The original full-window
pair and its strict rejection remain immutable and rejected.

Use 100.98 only via `gw98-direct`, native root
`/home/yanbo/work/microduck_rl-stance-replication-20260930`, exact
`feat/athletics-obstacle-curriculum`. Stop starting new work at **2026-10-06
08:00 Asia/Shanghai (00:00 UTC)**. Native source still starts at
`d923275040b3538f6ed48b867f1b1e6bc42304d3`; local declaration baseline is
`9d2162d71d4c869d6d9b2481b663b3cb07042dd9`. Permit only this new declaration,
the new runtime-dispatch adapter, the new bounded supervisor and their two
focused test files. All earlier source/test leaves remain byte-for-byte frozen;
the two intervening full-tree documentation commits are evidence-only.

## Process-local adapter

Use a one-shot exclusive main-thread scope, never an installed-library edit.
Temporarily wrap only the exact pinned `mujoco_warp._src.smooth.crb` reference.
While that original function executes, temporarily wrap the exact original
`warp.launch` reference. Other forward kernels run outside that launch wrapper.
Within CRB, all non-accumulation launches, including dense qM construction,
are passed through once with untouched arguments. The original CRB copy and
qM zeroing remain inside the unchanged installed function.
Pin the effective installed forward dispatch too: its `smooth` reference must
be that exact pinned module, and its original forward/fwd_position functions
and public `mjwarp.forward` alias must be unchanged before arming the scope.
The installed forward source hash is
`c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3`.
Do not rely on the separate public `mjwarp.crb` alias as the caller path.

Before its first actual CRB call, bind the actual model/data identity, dense
16-body topology, parents, seven reversed levels, dimensions and device.
Validate parent/level ID contents before each of the four original CRB calls;
identity and pointer checks alone do not rule out an in-place topology change.
Allocate three singleton IDs once on the same device before the first call.
These pre-CRB ID readbacks and the startup allocation change timing and are
explicit instrumentation, not a hidden claim of uninstrumented equivalence.
Both modes use the same validation and allocation path. Do not copy, observe
or synchronize intermediate CRB states between accumulation launches.

The concurrent mode forwards all seven original accumulation calls unchanged.
The serial mode replaces only level `[2,7,11]` with `[2]`, `[7]`, `[11]`, in
that order. Each call uses the installed `_crb_accumulate` kernel and the current
implicit Warp device/stream, with the same actual evolving CRB array aliased
as input and output. Preserve every other original level and both parent-zero
no-ops. Verify exact launch arguments, actual model/array identities, level
order, layout/device and seven original/nine serial accumulation launches
per CRB call.

Coverage begins before the exact `CudaConstructorRng.construct` invocation.
The first CRB call therefore belongs to the inherited constructor/reset forward;
after construction bind the exact returned `ScheduledRecoveryRuntime` instance
to the already observed model/data before any caller forward. Do not substitute
a factory/subclass, change runtime method identities, retarget the old constructor
contract or skip initial-forward coverage.

Reject foreign models/data/threads, nested CRB calls, CUDA capture, changed hooks,
changed array identity/layout/device, extra/missing levels, reused scopes or a
swallowed fault. Restore owned references and release the lock on every exit;
never overwrite an unrelated replacement silently. Receipts contain bounded
detached counters and explicit non-qualification flags, not mutable array aliases.

## Fixed four-child order and observations

After reviewed source tests, fresh native source/identity/pins, authenticated
predecessors and idle/lease gates, run one sequential retained service with four
fresh isolated CUDA0 children, in exactly this order:

1. Concurrent capture.
2. Concurrent replay.
3. Serial capture.
4. Serial replay.

Each child uses 64 worlds and the unchanged source-bound recovery schedule:
32 zero-wrench rows plus 32 declared dose rows. No dose is executed: no physics
or schedule counters advance. Initialize caller CPU/CUDA generators to fixed
declared seeds 673/677 before the frozen private-generator constructor. Keep
both caller streams unchanged across construction and all observations. Preserve
the frozen constructor's private seed/layout and fixed nominal motor checks.

Observe four completed-forward frames per child: the constructor output, then
three calls to the exact inherited unforced `_forward` without integration.
Record the ten persistent fields from `EarlyInertiaTrace.SHAPES`, plus qpos,
qvel, time, qacc_warmstart, ctrl and both applied-force arrays, as complete
finite owned CPU float32 tensors. Bind actual floor/foot/control/plant metadata
and the frozen constructor receipt. Applied forces, ctrl and velocity remain
zero; qpos/time/qacc_warmstart and physics counters must not change. Record raw
whole field hashes and keep byte-exact signed-zero-sensitive comparisons.
Here the physics-counter invariant is exactly `env.steps == 0` throughout and
unchanged `data.time == 0`; no integrator is constructed by this diagnostic
beyond the ordinary runtime-owned object, and none is called. Do not assert
forward-derived constraint/contact/solver counters stay fixed. Require each
child's closed scope to report four topology validations, exactly four CRB
invocations, 28 original level
requests, one constructor invocation and three caller forwards: actual
accumulation launches are 28 concurrent or 36 serial, with four dense-qM calls.

These are repeated forwards at unchanged kinematics, **not a complete scratch
array restore** and not the original policy's scheduled three-step trace. Do
not claim all forward inputs identical from kinematics alone. Compare paired
constructor frames and all four corresponding frames within each mode; report
each named field separately. Compare cinert/cdof/subtree_com across modes before
interpreting a CRB difference. Retain any alternative or varying finite output;
no tolerance, row canonicalization, favorable-output retry or gate relaxation.
The comparison inventory is fixed at 28 field-wise pairs: eight capture/replay
pairs within modes, eight concurrent/serial pairs, and twelve within-child
pairs comparing each constructor frame to that child's three later forwards.
Require the private constructor CPU/CUDA endpoints to agree across all four
children, in addition to the preserved caller streams; no CPU CUDA-RNG
re-execution is claimed.
Nonfinite data, ownership/protocol/capacity failure is a real failure, not an
allowed numerical negative outcome.

Even exact serial frames in this bounded sample do not establish original cause,
full-window replay, uninstrumented determinism, learned stance or training
admission. Other reduction kernels may still vary. A negative concurrent sample
cannot prove determinism either. All original admission/physical flags stay false.

Read-only inspection of the pinned installed smooth source identifies two
distinct, unchanged reduction paths: `_subtree_com_acc` (line 477, invoked by
`com_pos` at line 602) precedes CRB and feeds subtree_com/cinert/cdof;
`_cfrc_backward` (line 1219, invoked by `_rne_cfrc_backward`) feeds the
inverse-dynamics bias path. Both use parent-targeted atomic additions over
body-tree levels. This is source-level risk evidence, not observation of their
actual execution order or a demonstrated cause. Neither path is changed by
this CRB-only control. Their body-zero refusal differs from CRB's parent-zero
refusal: body 1 contributes to parent/world 0 in those paths but is a CRB no-op.
Do not silently reuse the CRB no-op witness for a future CoM/RNE adapter.

## Provenance, resources and closeout

Authenticate the completed full-tree owner and post-exit terminal by whole SHA
before decoding, all 20 new run/fixture/test files, the Mac v2 receiver result,
the completed-unit clearance and the inherited 50 predecessor files. Preserve
historical D923 source/terminal invocations as historical evidence, not live-unit
requirements. Reproduce the original strict pair rejection in CPU preflight.
Frozen full-tree anchors are owner `718b61f3d08e61a2e27410fe8b8b62ae3be52a9a74dad0034e4998bc074ae707`,
terminal `4806842ddf6b664858875b632246d40df384ef72564f99852b96a1ff1d93dc2f`,
Mac v2 `20b5bcf1ca84c1f20f41f8cabd0e95090c1f16c49b9aa60050a06de6f7fe0b59`,
clearance `0d495a521888a9a49e90eac31ea7ac2e6fd7696c5e40db691decf3e2945aaac4`.

Do not admit native execution until the new positive complete CUDA-hidden test
count is frozen after review, passes with zero skips, and all implementation
contracts above are tested. Native source sync is a guarded clean fast-forward
from D923 to the exact pushed source; all inputs/runtime/source are rechecked
before and after each mode.

The frozen complete suite contains **1,579 tests in 53 files**. Source regressions
include the real writer/reader owner-PID bridge, at-least-once observation of the
actual child GPU PID, the 180-second child-monitor bound, and content validation
before the single exclusive final report publication. A failure of late content
or context checks must retain a failed report, never a passed publication.
Consumers rehash the complete stage-specific campaign inventory, excluding the
current report and later stages, and require byte-exact snapshot agreement.

Predeclare separate source-tagged sync (120 s/256 MiB/CPUQuota 100%), tests and
CPU preflight (300 s each), run (1,200 s), and independent CPU closeout (300 s)
services. Each of four children has a hard 180-s cap, own process group and
inherited cooperative lease. Tests/preflight/run/closeout use 6 GiB cgroup
memory, CPUQuota 200%, Nice 10, KillMode control-group and RemainAfterExit yes.
No RLIMIT_AS substitute. Reserve every remaining mode's full cap plus at least
60 s before 08:00. Logs <=1 MiB, JSON and each child payload <=2 MiB, declared
new inventory <=16 MiB. Keep partial artifacts on failure.

The unchanged cooperative FilmBrain lock must cover every CPU/native mode.
Require idle GPU before/after; during a child permit only its actual observed
PID, temperature below 75 C, used VRAM <=5 GiB and free >=6 GiB. Preserve both
FilmBrain PIDs/restarts and both protected services inactive in system/user
namespaces; never overlap Duck work with another GPU workload.

Independent CPU closeout must authenticate complete bytes before weights-only
loading, validate tensor/receipt/counter/metadata schemas and raw numerical
comparisons, caller/private RNG, exact source/pinned runtime and terminal caps,
PID zero, zero restarts and invocations. Retain whole artifact hashes, including
negative numerical evidence, before clearing only completed owned units. At
cutoff begin no new Duck work and audit durable state without protected restores.

The supervisor's separate CPU closeout is supplemented by an owner post-exit
terminal recorder with externally observed whole report hashes and a standalone
CPU receiver using a literal 17-field/28-pair NumPy comparison, not the producer
comparator. The receiver authenticates all 21 new campaign files before decoding
the four bounded payloads. Neither an internally computed report digest nor
agreement of child metadata alone is external authentication. Retain these small
tool/packet hashes separately; do not download the old 95-MiB body traces to Mac.

## Status

The adapter and supervisor have received independent source review. The complete
frozen 53-file suite passed 1,579 tests in 228.44 s with zero skips on Mac with
CUDA hidden. The final focused adapter/supervisor suite passed 57 tests. Fresh native
qualification and runtime numerical results remain pending. No runtime-control
acceptance or training authorization is claimed.
