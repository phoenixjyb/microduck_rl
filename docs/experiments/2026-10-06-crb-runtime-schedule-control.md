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
08:00 Asia/Shanghai (00:00 UTC)**. Native source initially started at
`d923275040b3538f6ed48b867f1b1e6bc42304d3`; the follow-up sync starts only
at retained failed-preflight source `5cbd18ce3cff0c51461b0061ac885ef4b8f024d7`.
The local declaration baseline is
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
from the predeclared retained head to the exact pushed source; all inputs/runtime/source are rechecked
before and after each mode.

The frozen follow-up suite contains **1,581 tests in 53 files**. Source regressions
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

### Retained native test failure and bounded test correction

The first native suite at `dab4f3258e10e79fa3e44629161a76705b8b359d`
finished with **1 failed, 1,578 passed in 123.36 s**, zero skips. Its retained
tests invocation is `8f9c04a96a3c47118a1dd4a89b7aa2c3`, terminal PID 0,
exit status 1, zero restarts and peak resident memory 5,219,262,464 bytes.
The failed report SHA is
`9c7fe2de2aef50fb1cf11d033065b552bec48ec2f0e2c3b7ec33257043b9b88f`;
the pytest log SHA is
`c7ecd58ed72bbf2182d906baa5420486471847f298330ae6f658720a56b7cfa7`.
Both remain immutable at `artifacts/tools/stance-crb-runtime-tests-dab4f3258e10/`.
No preflight or CUDA child was started at that source.

Read-only diagnosis found that the foreign-launch test replaced `warp.launch`
before model/actuator construction. That suppresses unrelated library array
initialization and makes friction validation depend on uninitialized memory:
the native run raised `nonnegative matching-dtype friction fields` before the
intended `unmodified Warp launch before CRB` guard. A Mac pass was not evidence
that this fixture was correctly isolated. Correct only the fixture: construct
and bind the normal CPU runtime first, then install the foreign launch and call
the controlled CRB entry directly. Retain the foreign replacement and fault
assertions, plus constructor coverage/counter checks. Do not change the adapter,
production validation, numerical gates or dependencies. Re-run the unchanged
1,579-test inventory under a fresh exact-source namespace before preflight.

The concrete path is `WarpStanceRuntime` calling `expand_model_fields` for
`dof_frictionloss` and `dof_damping` before creating `BamStateCommit`.
Installed `mjlab/sim/randomization.py:28-49` allocates each expanded destination
and fills it through `wp.launch(repeat_array_kernel, ...)`; the premature stub
skips that fill. The installed file SHA matches on Mac and native WSL:
`35d0bddcd6ef3b0317cb89987e0c42bbf1862fcd4beb795c1327ade814c2a64c`.

The corrected adapter suite passed all 27 tests; the corrected complete 53-file
CUDA-hidden Mac suite passed all 1,579 tests in 269.83 s with zero skips. Ruff
check/format and `git diff --check` passed. Fresh native tests and preflight are
still required; the adapter implementation is byte-for-byte unchanged.

### Retained CPU preflight failure and module-routing correction

At `5cbd18ce3cff0c51461b0061ac885ef4b8f024d7`, the fresh native suite
passed all 1,579 tests in 126.48 s with zero skips. Its report SHA is
`f148c9f54902be4f301f0b5737f8a41bdd37a1a427c1c3cac3fd7d7eb5c5ee83`.
The next CPU preflight failed with
`AttributeError: module 'mjlab_microduck.stance_recovery_cuda_inertia_probe' has no attribute 'order'`.
The retained preflight report SHA is
`7f6eec85a1989417d85c88ed2f2e2d6dcf5f5ab2a6b106c6d554655f78ef8b88`,
invocation `bff43f70a93545679437a70a39884d93`, terminal PID 0, exit status 1,
zero restarts and peak memory 560,226,304 bytes. No CUDA child was started.

Read-only inspection found two calls routing through nonexistent `prior.order`.
The actual authenticated reader is the existing `stance_inertia_order_probe`,
also imported by the frozen full-tree supervisor. Import that module explicitly
in the new supervisor; keep its source/byte authentication and the unchanged
`prior._score_inputs` / `prior._strict_pair` numerical checks. Add two wrapper
regressions that exercise the real routing with mocked byte I/O rather than
mocking the wrappers themselves. The resulting complete suite has 1,581 tests;
it must pass with zero skips under a new exact-source namespace. Preserve the
successful native tests and failed preflight at the old namespace, and do not
reuse their passed status as qualification of the correction.

The corrected complete 53-file CUDA-hidden Mac suite passed **1,581 tests in
448.78 s**, zero skips; the focused adapter/supervisor suite passed 59 tests.
Ruff check/format and `git diff --check` passed. The native test cap remains
300 s, based on the measured native suite; the longer Mac duration is not a
native runtime qualification. Fresh native tests and CPU preflight are pending.

### Independent receiver schema correction

Owner review found that the private receiver and its synthetic fixture had both
represented `metadata.schedule` as a body-level list, whereas the real producer
uses the complete source-bound recovery schedule dictionary. Preserve those
original tools unchanged; they were not used to qualify native payloads. The
separate `receiver-1581` tools now validate the literal 64-row dictionary against
the preflight declaration, metadata reference and all four child payloads.
An independent read-only review cleared this delta. A CPU-only check confirmed
canonical equality with the actual frozen `prior.declaration(source)` schedule.

The corrected 21-artifact synthetic test passes all 28 comparisons / 476 field
pairs, retains the finite CRB negative case, and rejects bad hashes, missing
anchors, incorrect counts/pair labels and absent/list/wrong-source schedules.
Synthetic evidence is not native evidence. Whole private tool hashes are:

- receiver `d633491a0016ecebe51146f19bd35ac13bef66ba33121f609407c0375dd273f9`;
- synthetic builder `c84f02dd9bc86fbe2a722490cb5f805a92414c71e5dd20c7aa8d0a1f165db5a4`;
- independent math `f64b373cc0eb9b016ff12dc7e29b09a9cd01145b8f3f287cb879f902b741c944`;
- owner terminal recorder `ca01fbd6de63adcee8fb171aa82cc2a8e59a2cf396a1cec97d5c40e6e59bede2`.

The receiver remains CUDA-hidden, authenticates all bytes before decoding, and
cannot admit training or overturn the original full-rollout rejection.

### Retained runtime and CPU comparison failure; CPU-only recovery declaration

At exact native source `2ecee471f7b999822ed19defd7e2d1c7ad09df07`,
all 1,581 CUDA-hidden tests passed in 126.11 s with zero skips, CPU preflight
passed, and all four predeclared CUDA children completed successfully. Their
four payloads, receipts and logs are immutable; do not repeat the GPU runs to
repair a CPU reader. The original full-rollout rejection remains unchanged.

The CPU closeout exited 1 before writing a receipt or numerical result:
`_load_and_compare_run` passes the real retained Torch frames to
`_field_comparison`, whose NumPy-only `a.view(np.uint32)` raises `TypeError`.
The existing numerical tests exercised NumPy frames and missed this boundary.
Retain the failed invocation `8531be13b48a4d4f8c63bf0d6e6023b6`, owner PID
3251146, terminal PID 0, zero restarts and measured peak 560,336,896 bytes.
Whole observed report hashes are:

- tests `70fc98e0e9ad83703ca92854cccff8b6ebbf489ce7793d8be58b7589853d46e3`;
- preflight `6c470f72725e7f1cd99ce64c66ccb9b0a264e7cf896c4cd52b0ea4f111340d3a`;
- runtime `dae635a1dcd68be5790395a7fb876c3f50a7d810460db5e7c96a17863e3f431a`;
- failed closeout `02202ab30181629ea2dac4b5305643b972c15dceb994f1719031261b239ea0d8`.

Predeclare a separate **partial-evidence CPU recovery**, not successful native
closeout: authenticate exactly 20 retained files (three tests, two preflight,
14 runtime, and the failed closeout report), explicitly reject an unexpected
closeout receipt, and preserve the exact failed report and terminal state.
Keep the native source at the runtime revision while recording fresh source,
host/package, idle GPU, protected/FilmBrain state, service invocations/caps and
the four actual observed child PID/resource histories. A separately reviewed
CPU-hidden reader must retain all payload/schema/RNG/byte authentication and
independently compute the same 28 pairs / 476 field comparisons; it must not
invent a producer comparison or convert the failed closeout into success.
No tolerance, GPU retry, numerical admission or training authorization follows.

The short source-sync service has `MemoryAccounting=yes` but its removed
cgroup leaves `MemoryPeak=[not set]`, `ControlGroup=''` and
`MemoryCurrent=[not set]`. Preserve those observations and explicitly mark the
sync peak unavailable. The sync-only exception requires exact source/invocation,
120-s/256-MiB/100% caps and successful PID-zero terminal state; all four test/
numerical services still require positive measured peaks below their 6-GiB caps.

The bounded comparator correction validates real retained CPU float32 Torch
frames with the existing exact shape/contiguity/finiteness contract before
converting to NumPy, retaining raw-bit and signed-zero checks. Eight additional
regressions cover 28 real-Torch pairs / 476 fields, finite upstream/CRB negatives,
signed zero, and nonfinite/dtype/device/layout/shape rejection. The complete
53-file CUDA-hidden Mac suite passed **1,589 tests in 207.65 s**, zero skips;
the focused adapter/supervisor suite passed 67 tests in 25.69 s. Independent
source review, Ruff check/format and `git diff --check` passed. These checks do
not retroactively change the frozen native revision or its 1,581-test receipt.

The separate partial-owner recorder authenticates all 20 files (8,114,151 bytes)
and all five terminal units, retaining the one failed CPU closeout. Its whole
tool SHA is `25bcb30c1876990a09d051cb8c649708b3b35bd641356b5ca4b7d2813b12823b`;
the 22 terminal schema regressions passed without producer imports or CUDA.
The native 20,732-byte packet has independently observed whole SHA
`dc2aa2d344fcad58d2054a2238c15d0d3678724852160834dc5a2a29bef8151a`.
Its four monitor histories observed actual child PIDs 3248986, 3249394, 3249842
and 3250255, each in four samples; used VRAM was at most 973 MiB and temperature
at most 33 C. Protected namespaces remained inactive, FilmBrain PIDs/restarts
unchanged and the GPU idle at closure. No successful original closeout is
implied by this packet.

### Independently authenticated partial numerical result

The separate CPU-hidden reader authenticated all 20 files before decoding and
independently compared all **28 pairs / 476 fields**. Its first actual read
rejected a private-fixture assumption: the archived constructor anchor's source
is original `65abe930caa16476e4d8ece44f937492da47ce2e`, not the fresh
runtime source. Read-only inspection of `_constructor_anchor(old_ctor)` and the
authenticated declaration confirmed this distinction. Bind that exact historical
source, keep fresh payload constructor receipts bound to the runtime revision,
and add a complete synthetic negative substituting the fresh source for the
old anchor. Do not accept an arbitrary source or regenerate CUDA RNG on CPU.
The corrected synthetic test and actual reader then passed; the preserved CPU
closeout still failed and has no receipt.

Whole reader / fixture / independent-math hashes are respectively
`de1aeabb7316a525a18a1d041f1798f9a369185e25b08117b2bfddc412b6742a`,
`4124e39c4c1799e297d97bfb9cc41518e950baa4d39b0a965c61e71bd96c6319`,
and `f64b373cc0eb9b016ff12dc7e29b09a9cd01145b8f3f287cb879f902b741c944`.
The 133,091-byte result whole SHA is
`5d77483122dc390a53c5d277828f9d8eb7196856b0d950e2445bf9694ab4b167`.
Its protocol is `microduck-independent-runtime-partial-evidence-receiver-v1`,
status `authenticated-partial-artifacts-and-independent-math`,
`closeout_succeeded=false` and `all_exact_raw_bits=false`.

| Comparison family | Pairs | Bitwise-negative pairs | Negative field pairs |
| --- | ---: | ---: | ---: |
| Concurrent capture/replay | 4 | 3 | 7 |
| Serial capture/replay | 4 | 1 | 2 |
| Capture concurrent/serial | 4 | 4 | 26 |
| Replay concurrent/serial | 4 | 4 | 26 |
| Constructor/subsequent forwards | 12 | 10 | 68 |

Serial CRB has exact paired CRB, qM and qLD in all four frames, but the paired
constructor still differs in `qfrc_bias` / `qfrc_smooth` (one scalar each,
maximum 1.3969838619232178e-9). Both serial children also have a later forward
whose `subtree_com`, `cinert` and `cdof` differ from their own constructor
(8 / 91 / 39 raw-bit scalar mismatches; maxima 1.4901161193847656e-8 /
3.026798367500305e-9 / 1.4901161193847656e-8). This is evidence against
CRB-only runtime exactness, not proof of another kernel's cause: unchanged
kinematics is not a full scratch reset, and the adapter changes timing.
The original full rollout remains rejected; no training admission follows.

Retain the result, literal reader/fixture/math tools, 20-file anchor manifest,
anchor extractor and terminal schema regressions under native
`artifacts/tools/stance-crb-runtime-partial-mac-2ecee471f7b9/`.
The next source-only slice may plan CoM/RNE reductions with their actual
body-zero activity rules; do not reuse CRB's parent-zero no-op assumption,
perform a new GPU replay or claim original cause from this result.
