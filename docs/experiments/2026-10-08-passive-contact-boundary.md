# Passive contact allocation and constraint-completion diagnostic

## Question, source boundary and retained dependencies

The complete one-tick run at
`321a72a6b7d3d64808dafdbc36309052495cc85b` was a retained numerical negative,
independently replayed byte for byte on Mac. Its bit-addressed diagnosis at
`ae6814d77c5e4b8672b2fa4a57ad2849e33df7c4` first locates differences in the
substep2 solved load/EFC snapshot, followed by a motor friction budget/input
delta at forward5, then physical velocity and RNE/CoM differences at forward6.
This does not prove a contact-allocation cause.

New question: where do the **first captured raw differences** appear relative
to the actual frozen contact-init call and the completion of constraint
construction? There is no new kernel intervention beyond the previously
reviewed ascending dense-friction writer. Passive observation may change timing;
therefore an observed repeat or negative is localization evidence under this
instrumentation, not replacement of the earlier result or runtime acceptance.

This separately owned experiment starts at base
`0ad275806cb64383535fa2338f307fc91dccea99`. The exact changed-path fence is
the three new `stance_contact_boundary_{control,probe,receiver}.py` modules,
their three tests, and this document: seven paths only. Do not edit or enlarge
any historical experiment's source fence, protocol or admission. Work only
on `feat/athletics-obstacle-curriculum` in the agreed Mac/100.98 worktrees;
never use100.100 or modify FilmBrain/protected services.

Retain and authenticate the accepted synthetic dependency and the complete
negative's external raw inventory/independent receiver anchors before any
interpretation. Preserve both failed attempts and all earlier raw/private
caches. Bind every committed source blob, frozen package/library/toolchain,
loaded executable and actual process/service/GPU lease. Installed packages,
drivers, shared caches, environment aliases and unrelated work remain unchanged.

## Literal recipe and passive capture

Three fresh sequential arms: `original`, `candidate0`, `candidate1`. Each has
64 worlds, the unchanged constructor forward plus exactly one zero-action
ten-substep nominal tick, and unchanged serial CoM/CRB/RNE and BAM controls.
Caller CPU673/CUDA677 streams are initialized once; private CPU977/CUDA983
forks preserve them. Keep all21 friction entries and ten accepted proposals
per arm. No state restore, second tick, reset, terminal, graph, actor,
optimizer, learner storage, perception, video or physical motion is permitted.

Within the same owned constraint/launch window, observe contact-init and
complete constraint construction for forwards0–6 only: seven points per arm,
21 points total. Forward0 is constructor, odd forwards are pre-integration
solves and positive even forwards are post-integration solves. Forward4 is
poststep1; the next BAM proposal is substep2, before forward5.

The literal frozen `_efc_contact_init` call has18 ordered inputs and six
outputs, launch dimension8192, dense512-row capacity and10240 nnz capacity:

1. Model arrays: body_weldid, body_dofnum, body_dofadr, dof_parentid,
   geom_bodyid, flex_vertadr, flex_vertbodyid.
2. Scalars: njmax512, njmax_nnz10240; then data.nacon.
3. Contact arrays: dist, dim, includemargin, worldid, geom, flex, vert, type.
4. Outputs: nefc, contact.efc_address, efc.id, J_rownnz, J_rowadr, and the
   actual temporary efc_nnz from that same make_constraint call.

Pin the actual plain-int pyramidal cone0 and dense mode, the cached factory
callable and unwrapped code, and the returned kernel/function/code/Warp-module
identities. Independently pin its Python namespace and globals. The actual
captured specialization is `IS_ELLIPTIC=False, IS_SPARSE=False`; require those
plain Boolean closure values unchanged before compilation and through dispatch.
Recognize only that kernel identity; forward its original call unchanged.
Before/after capture includes all actual arrays and inactive capacity, the
temporary nnz pre-state, exact layouts/pointers/strides/dtypes and raw bytes.
The additional contact geomcollisionid, position and frame are explicitly
labelled **context**, never substituted for or folded into the ABI. Contact
indices alone cannot identify a physical contact when a geom pair has several.

After frozen make_constraint returns, synchronize the same stream and capture
all16 EFC fields, constraint counters and physical qpos/qvel/qacc/
qacc_warmstart/ctrl. Label this boundary **constraint construction complete,
before solver**. Force, state, Ma and Jqvel are prior/stale solver storage at
this boundary, not newly solved outputs. Keep the unchanged later BAM load
snapshot and complete recipe packets to distinguish that phase.

All snapshots preserve inactive bytes and exact row/contact order. Empty
dense scratch may have actual null pointers; retain those literally only
with zero span/bytes. Do not filter, sort, normalize, zero uninitialized
capacity, reject opaque inactive float bit patterns as active-state NaNs, or
rewrite any recorded numerical packet. Require immutable inputs/context
across the allocation call, stable storage and non-overlapping nonempty
carriers, one contact call per sampled forward and exactly seven completed
captures per arm. Foreign/reentrant hooks, code replacement, changed ABI,
storage, capacity, cone or sparse mode must refuse without overwriting them.

## Resource bounds and evidence gate

An allocation-only CPU check at base0ad27580 confirms cone0, naconmax8192,
contact address shape[8192,4], and all16 EFC fields totalling4068352 bytes.
It ran no physics forward or integrator; it does not establish CUDA behavior.
Proof `artifacts/tools/contact-boundary-cpu-layout-0ad275806cb6/layouts.json`:
11167 bytes, SHA256
`b88d299d60d5a46769ba7dbf1a3696212bf025cdf6d283c6865305759eea747c`.

Preflight actual native layouts, not only these CPU sizes. New captures are
limited to80MiB per arm /240MiB total, three primary packets per sampled point,
16MiB per leaf. Entire raw run is bounded to1024 leaves /768MiB, including
receipts, generated source/CUBIN/metadata and the complete private Warp cache.
Original raw baseline526737887 bytes plus the new cap remains below768MiB.
Use fresh explicit loads for original friction, candidate friction and cached
contact-init at sm120, PCH disabled only in the private process. No loaded/JIT
machine-code observation claim is made.

Before execution, require focused source tests and the separately frozen94-file
exact-source CPU suites on both hosts, zero omissions and complete source/
environment closure. CPU pytest children alone receive literal process-local
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and
NUMEXPR_NUM_THREADS=1. Retain these exact four settings in both receipts and
refuse absent/changed settings. Do not retune an already imported owner or CUDA
process, install packages, widen service task limits or reuse an older receipt.
The separately declared runtime host-thread-budget revision below is not part
of the earlier CPU-only repair or the logging-only trial. The reviewed
collection is2944 tests before that new revision; its ordered-scope
SHA256 is `253d755fc09f0b5286695a85303353ff6349e7f4bb88cbd3b97dea5b1872fb6a`.
Native execution remains blocked until both same-source receipts and the
completed native CPU user-service witness are authenticated. Do not reuse older
CPU receipts. Source-only focused checks are not CUDA execution evidence.

One retained user service only: parent CUDA hidden, child CUDA0, unchanged
exclusive existing lease FD inherited by the child, idle GPU before and after,
actual PID/PPID and invocation witnesses, FilmBrain identities unchanged and
protected services inactive. Caps: service600s, child540s, memory6GiB,
CPU200%, tasks64, nice10, no restart, process-group cleanup and16MiB log/file
limit. CPU prerequisite cap660s/test600s uses64MiB file limit. Refuse launch
without900s remaining before2026-10-08 07:30 Asia/Shanghai. Start no work
at the cutoff, preserve durable evidence and do not restore protected services.

The independent receiver must authenticate every whole leaf before decoding,
validate complete ownership/source/compiler/recipe/capture evidence, then
report raw candidate repeat and the earliest captured differing boundary.
Its decision is only `passive-contact-boundary-capture-complete`; the recipe
repeat Boolean is diagnostic, never promotion of the historical runtime result.
Distinguish whole-capacity raw differences from authenticated active contact
prefixes and per-world EFC extents. Empty dense scratch has no active span;
stale/prior solver fields remain labelled even within active row extents.
Do not call a matching sampled boundary a causal proof. All five qualification
flags remain false, regardless of the outcome. No longer rollout or learner
is authorized by this diagnostic; further action requires its retained review.

## Status

Implementation and predeclared capture/receiver contracts have been reviewed.
The initial84 focused control/owner/receiver tests passed, including integrated three-arm
fake-observer receipts, opaque inactive float bits, compiler-role records,
packet corruption and separate Warp/Python module identities. The unchanged
prior91 files plus these three files collect2931 tests with no collection error.
At d669eec1278020dad9b4474eeea363f0ca91b711, the Mac full suite passed2931
tests/zero omissions in187.25s. Its319-byte inventory SHA256 is
`d395a8f372475ed9f8fd0e5fa7463947a97da8e0912b11e957e1f5c519c3638a`.
The WSL CPU service invocation3828a98dd1bc4effa7e84c1490484f35 failed with
`subprocess.TimeoutExpired` after600s, PID0/status1; it reached the64-task cap
and stalled after126 tests at the first real CPU physics test. Exact cause was
not captured; this is consistent with lazy thread-pool pressure, not a proven
kernel defect. With the four child-only caps, the exact stalled test passed in
5.93s under unchanged64-task/6GiB/CPU200% limits. No GPU work was launched.
Retain the failed directory and closeout
`artifacts/tools/contact-boundary-cpu-failure-d669eec12780`: its554-byte inventory
SHA256 is `5561314a7b489187b94ef4914714c59943463e2299a358eb49c5c7f6511721b7`.
The bounded repair adds nine CPU settings guards; all93 focused tests pass.
The unchanged ordered94-file scope recollects2940 tests. Both full same-source
suites must be rerun. This is source-only evidence. Those new full runs are pending.

At06065bf56edca8c8c21418a2a48d25d33d82bc59 those2940-test full suites passed:
WSL179.65s and Mac315.17s, zero omissions. Their319-byte inventory SHA256s are
`76348f2239ffb96f5e4b638b7cc16d0948526554278750c9f470c850f45a5b6c`
and `4ecb2c247c139c58bbe1f7d659122bb8a0e0302e075c0440dc9987b577402ba4`.
The subsequent native attempt invocationdfbfba0753d240959dd628dbb413ef77
failed: child3998746 SIGABRT/status-6, owner3998663 status1, both nowPID0.
Its23-byte child log says `libc++abi: terminating`. All three generated
source/metadata/CUBIN triplets exist, but there is no child receipt or arm
packet: compilation, explicit loading, post-load checks and constructor/first
forward are not distinguished. The authenticated kernel log records
`cgroup: fork rejected by pids controller` for this exact service immediately
before signal6: task/thread creation exhaustion is a documented coincident
condition and strongly supported proximate explanation. The responsible native
library/pool and Python call site remain unknown; do not claim a contact cause.
Preserve18 raw leaves/2827130 bytes and
`artifacts/tools/contact-boundary-gpu-failure-06065bf56edc`:
547-byte inventory SHA256
`4ee348a15a4959de921e6421d82798e9d3aed6f65b901be93125f268619caa4f`,
2378-byte raw-inventory SHA256
`eede365df7abb84ecb6141ae9ebcce5ffb11cad2d7bafab93d1652eb91c9be58`.
Seven owner samples saw only the owned child,30–32C; GPU idle afterward,
FilmBrain identities unchanged and protected services inactive.

Next bounded attempt adds child-start PYTHONFAULTHANDLER=1 and flushed bounded
stderr phase JSON around the existing import/init, serial compile/load/hooks,
factory check and unchanged recipe-case calls. Observe only four thread-env
values and `/proc/self/status` Threads; do not change their values, service
caps, physics/control/seed recipe, legacy modules or raw packets. Fatal-signal
logging provides Python frames, not a native C++ root-cause claim. The four new
guard cases bring the unchanged94-file scope to2944; both same-source full
CPU prerequisites must be rerun before any retry. There is still no completed
native contact-boundary run; the earlier complete numerical negative remains
the latest real-model result. All qualification flags remain false.

The97 focused tests pass and the94-file scope collects2944 tests for this
logging-only revision. The bounded unchanged-thread retry is for localization,
not blind promotion; if it identifies resource exhaustion at a native call,
predeclare a separately bound host-thread-budget repair rather than silently
changing this recipe or widening the service cap.

At743cd9871b1b2bb30e6d93140bcb4c865f13dfd1, both2944-test full suites passed,
zero omissions: WSL177.05s and Mac337.97s. Their319-byte inventory SHA256s are
`1ce7fe5a4f3392d783a7d643d230bd32b8aedb4c5baa9be895ae4b9bc7313620`
and `3f2a93135c6925405d3cc5f30d2ca17c247eab976d5651980a4d4b273ebaa037`.
The native run invocation68f42a25611145578ac25ce35e74e2c6 completed all three
explicit compile/load/hooks/bind sequences, then stalled in the original
recipe-case entry. Its33 flushed phase records end at `recipe-case-start` for
original; there are no arm captures or child receipt. The owner closed it at
the540s child timeout (child4008530/status-15, owner4008490/status1); the
retained service is failed/PID0, not active or successful.

The independent active witness observed owner1/child44 threads, service
pids.current45/pids.peak64/pids.max64 and pids.events max2. The child main
thread was waiting in futex; all four observed runtime thread-env values were
null. This narrows the resource failure to constructor entry after explicit
loads, not to one of those three loads. It does not identify a particular
native pool or establish a contact/physics root cause. Retain
`artifacts/tools/contact-boundary-active-thread-audit-743cd9871b1b.json`:
1233 bytes, SHA256
`cca0f62bfb007a89302b23e72acf58c8e34891bd6e2313712f9f8703191a1f50`.
The completed failure closeout
`artifacts/tools/contact-boundary-gpu-failure-743cd9871b1b` has667-byte inventory
SHA256 `d1a7f6b86136c6cb6212f40cf00e1dd1694bbca6db73274a0dc2be4244b73137`
and2382-byte raw-inventory SHA256
`9e17cc88b5140328a3741a76c77e3d708251b47d08ac8e9aa560ba2ddc23d6f9`.
All18 raw leaves/3037416 bytes and every closeout leaf are whole-byte
authenticated on Mac and retained on both hosts. The258 owner samples cover
539.715s, at most32C, with only the owned CUDA child. GPU idle afterward;
FilmBrain identities unchanged and protected services inactive.

## Separately predeclared runtime host-thread budget

The two resource failures justify one bounded resource-control trial, not a
task-limit increase or a claim of identical numerical conditions. Start the
CUDA-hidden owner Python process with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1 and NUMEXPR_NUM_THREADS=1, explicitly in this user
service only. Set the same four literal settings in the separately launched
CUDA child before interpreter startup. Keep the CPU-test map separate from
these owner and child maps. Change no machine-wide/service-global settings,
packages, driver, shared cache, environment alias or protected workload.

Before either process imports NumPy/Torch/Warp, require its literal role map
and a single observed native thread. Validate the owner before importing the
NumPy-bearing receiver. Child validation follows whole-byte declaration
authentication and precedes Torch. Retain exact role/PID/settings/count
snapshots at owner pre-import/after-child and child pre-import/after-Warp-init/
after-recipe. Derive child snapshots from their already written phase records.
Bind all maps into declaration, child and owner receipts, and require the
authenticated child log's exact40-phase sequence with unchanged maps,
owned PID and bounded thread counts. Receiver tests must reject changed maps,
roles, process identities, counts, schema, phase order and missing/extra logs.

This trial changes the observed runtime CPU environment relative to743; it
must not be described as an identical-environment retry or retroactively
ascribed to321/060. Physics/control/arm order, caller/private seeds, dedicated
stream, kernel sources/options, capture windows and all existing resource caps
remain unchanged. In particular retain tasks64, one nominal tick, no learner,
no longer rollout, no physical motion and all five flags false. Recollect and
freeze the exact94-file count, run focused checks and both same-source full CPU
suites, and authenticate their inventories before any new native launch.
An eventual completed capture remains only diagnostic evidence requiring
independent whole-byte replay; it does not qualify training or establish a
learned Duck capability.

The integrated revision passes127 focused control/owner/receiver tests. The
unchanged ordered94-file scope collects2974 tests (30 additional budget/log
guard cases), and that exact count is frozen in the producer. Ruff checks,
format checks and diff whitespace checks pass. Both full same-source suites
are pending; no GPU trial has been launched for this revision.

At2f99d1ec95dfd2f1ba82755a5483f619ed9a4b7c, both2974-test full suites passed,
zero omissions: WSL178.17s and Mac239.48s. Their319-byte inventory SHA256s are
`9adba22f2c7694e7d7a605cf58e036fd8cdd342a45e03d86fd8573fd2cd14ba6`
and `99dada3215434913e7ab7e85f17438daf75493bda32d0ee44d8f648eb0e84411`.
The native CPU terminal is PID0/exited/success0, invocation
810a956d6d534461803a7df84de40d42. Every CPU proof leaf is independently whole-
authenticated on both hosts before decoding; full source closure/count/maps
correspond. The CPU peak4744609792 bytes remains below6GiB.

The ensuing native run invocationf34f2d681fe24b7da1c92a0987f3058f reached the
original constructor's first contact-init observer and refused with exact child
error `ValueError: literal array shape and dtype contact-before.context.frame`.
Owner4024616/child4024637 both exited (owner1/child1); service failed/PID0.
All33 recorded phases have the four literal settings at1 and thread counts1–4.
The earlier stalled constructor progressed to a normal Python guard failure;
this does not establish the earlier native pool's identity or a physics cause.
Seven samples cover12.984s, at most32C, with only the owned CUDA child; GPU
idle afterward, FilmBrain identities unchanged and protected services inactive.
Retain40 raw leaves/14219071 bytes and closeout
`artifacts/tools/contact-boundary-gpu-failure-2f99d1ec95df` on both hosts:
550-byte inventory SHA256
`634b31bb52c5e20d8c51381b882fd90e5cebe6a4d5a7b5080e09058f0d998488`,
6864-byte raw-inventory SHA256
`fedf888d5165805779c0c5fa9b4ea21b09dac5590894c2543d1a3f0560f3238d`.
All raw/closeout leaves are whole-authenticated on Mac. No arm completed and
there is no child receipt or contact-boundary qualification.

## Original matrix-carrier layout correction

Read-only diagnosis found a source-contract error, not changed installation:
the externally anchored CPU allocation proof above already recorded actual
`contact.frame` as logical[8192], host[8192,3,3], stride[36], Warp `mat33f`.
The frozen Contact type uses8192 matrix elements. make_constraint later builds
a local `[8192,3]` vec3 reinterpretation for its Jacobian; our separate context
points at the original data.contact.frame, not that local view. The observer
specification and fake fixture incorrectly copied the latter logical shape.
The independent receiver also incorrectly required that extra logical axis.

Correct only those logical specs/fake fixtures and independently require the
original mat33 logical rank and dtype in the receiver. Keep host/raw shape,
all294912 bytes, pointer/span/context/storage checks and opaque inactive bits
unchanged. Explicitly reject the same-pointer vec3 reinterpretation rather
than accepting a widened shape allowlist. A real CPU allocation-only test
checks the original matrix metadata and rejects the local view, with no forward
or integrator. Synthetic receiver cases accept that matrix carrier and reject
wrong rank, reinterpretation, scalar dtype, stride, host shape or byte count.
No kernel dispatch, seed, control, resource cap or runtime thread map changes.

The revised focused suite passes135 tests; the unchanged94-file collection is
2982, frozen in the producer. Both full same-source CPU suites must pass again
before any bounded native retry. All qualification flags remain false, and no
learner or longer rollout is admitted by this layout repair.

## Completed CUDA child, refused numerical adapter

At0450bca9b0d735ddf1696d42f8dc50948253e055, both2982-test suites passed,
zero omissions: WSL179.54s and Mac215.92s. Their319-byte inventory SHA256s
are `00d1022419c9d1e81d02e881bf65eb76c2c43fdc8aa313e04eecb926ea6e93d7`
and `fc0cd15f0e18fb0bf07baad265ee78cf5b2d1984b8d96f925e58f21e4f326e4d`.
Native CPU invocation902e139470ef493e96f27a0a9f352a1a is PID0/exited/success0.

CUDA invocation2ba95b9e1e26455792deed94780f1898 completed all three arms
and all40 phases (owner4033711, child4033779, child exit0). The owner then
refused `ValueError: exact entry fields original.forward0` in the unchanged
six-field numerical auditor; owner exit1/service failed/PID0. This is not a
successful service or accepted receiver result. The contact observer supplies
six numerical fields plus layouts/identities/stream for ownership cross-binding;
passing those enriched rows directly to the strict numerical component was an
adapter error. No numerical/physics failure is inferred from that schema error.

Retain478 raw leaves/657303161 bytes and the failed closeout
`artifacts/tools/contact-boundary-gpu-failure-0450bca9b0d7`:551-byte inventory
SHA256 `cf3be7422ee39d83b91e0a332e962172eff6b3125e24a4c3abb61f56906a1e47`,
69450-byte raw-inventory SHA256
`1cb31c236b89836c84a946bdaf7fabf6faf9e3985a6b3a486945cc69b2a7061f`.
All phase maps remain at1, maximum recorded phase threads4. Seventeen owner
samples cover34.347s, maximum32C, with only the owned child. GPU idle afterward;
FilmBrain identities unchanged and protected services inactive. All five flags
remain false. Keep this failed run immutable rather than retroactively calling
it a successful run under a newer receiver.

## Strict numerical-view adapter predeclaration

Change only the new receiver adapter after all three contact-boundary ownership
checks, aggregate cap and boundary comparison. Require exactly three arms,
21 rows per arm and the exact nine enriched fields; unknown/missing fields
refuse. Build new row dictionaries referencing only the unchanged numerical
component's six fields. Keep original metadata and all carrier/byte objects
unmodified for contact cross-binding; perform no normalization or decoding in
the adapter. Do not modify the historical numerical module or any old schema,
admission, source fence, cutoff or protocol.

Actual small numerical fixtures exercise all21 rows through the unchanged
auditor, metadata retention, raw-object identity, pre/post-audit immutability,
exact-schema refusal and malformed-carrier rejection. The focused suite passes
151 tests; the unchanged94-file scope collects2998, frozen in the producer.
Both same-source full CPU prerequisites are required again before a fresh
bounded GPU trial. Preserve the same recipe, private seeds, thread maps, lease,
capture windows and resource caps. A successful capture would still require
independent whole-byte Mac replay and would not admit a learner, longer rollout,
video or physical motion.

## Successful passive capture, independently replayed negative

At9de88d19f3ee43f93eaba1cd9319eeca60713de9, both2998-test suites passed,
zero omissions: WSL180.24s and Mac386.77s. Their319-byte inventory SHA256s
are `d27c6f58ba19d05572c159fad31f35932c39ebfd17d1c15adc3534d2e3dc6c28`
and `23ed776a14e8b781a1fd98d97133d5aed510df817de4ebc753bbe84021498d72`.
Both entire source closures/counts/thread maps and every CPU proof leaf were
whole-authenticated on both hosts. Native CPU invocation
a75dbdde153040f393f2279bab5cbe45 is PID0/exited/success0.

The GPU owner completed successfully, invocation
f8d9893ffbb7455496dfc6f0316fa194, owner4044266/child4044287. Seventeen samples,
maximum32C, only the owned child; GPU idle after exit. Retain477 raw leaves /
657294134 bytes on both hosts. Closeout
`artifacts/tools/contact-boundary-tick-closeout-9de88d19f3ee` has69343-byte
inventory SHA256
`84fe5ee296cce8022ade62a4378b6a027c7b1eaedbd18887ff635bfc61467196`,
1035016-byte native and independent Mac receiver SHA256
`434f569bb45606339b743a15c3d39da97ee50cc949caa675543e360cacf315f5`,
and4746-byte Mac replay SHA256
`7896adc0fe1c9c7f7966813d62ad26b66b7bab5f6e7fe2cf9a5f176d988c9288`.
Whole transport (1042 bytes) SHA256
`5b291165b7fc65edd80af42ea763b6881fb037b526bb21581e396bdd44bf8b71`;
whole archive (22678658 bytes) SHA256
`407b3513dc15bd0693e4507df9542a744cf0ed4f840e54fa00e101d814ac0f9a`.
Every raw leaf was authenticated before decoding; Mac independently reproduced
the complete native receiver byte for byte, then its proof was retained and
whole-authenticated back on100.98. The source stayed unchanged through replay.

Decision is `passive-contact-boundary-capture-complete`, with candidate recipe
repeat **false**. All63 friction calls are structurally exact, no overflow;
the aggregate numerical repeat predicate is nevertheless false. The first
sampled active and whole-capacity difference is forward4/contact-before,
contact.dist slot7: little-endian words c9eb1bb7 versus2b0a1cb7. Forward0–3
sampled boundaries are exact. At forward4/complete, first raw field difference
is efc.id, world0 row14 (303 versus308); at forward5/complete it is data.qacc,
and at forward6/complete data.qpos. The first friction input divergence remains
forward5. Five original/candidate entries match exactly before later unmatched
trajectory entries. Keep the old321 negative distinct; instrumentation and
the separately declared runtime thread settings do not make them identical
experimental conditions. All five flags remain false. No learner/longer
rollout/video/physical motion follows this negative.

## Source-only payload and raw-coordinate side views

Read-only whole-authenticated inspection of the independently replayed run
found the same512 active raw input/context payload tuples, with no duplicates,
at forward4–6, but changed recorded order (500/508/492 uniquely linked payloads
move slots respectively). This means the first contact.dist slot difference
does not itself establish a changed contact value for a matched payload. It
does not prove physical-contact identity, equality of constraint rows, solver
equivalence, an allocation cause or a training-qualified runtime.

Predeclare two pure, separate receiver helpers; do not wire them into or alter
the historical capture report/decision. Validate the complete fixed carrier
schemas before taking active side views. Count exact input/context byte tuples,
not hashes or float equality, retain insertion order, leave duplicate tuples
explicitly unpaired, and report unmatched multiplicity and model-input equality
separately. Never sort/rewrite packets, compare outputs as though they were
inputs, replace contact identifiers, or promote the earlier negative.

The original coordinate helper used the unchanged active comparator and added host-shape
row-major coordinates, contact/per-world counts and a declared stale-storage
label. Its historical force/state/Ma/Jqvel label is retained in the old artifact,
but the Jqvel phase classification was wrong; see the correction below. No label
may call these fields newly solved by the current solver. Coordinates describe storage,
not identity. Caller whole-authentication and the retained independent replay
are prerequisites; these pure helpers confer no provenance themselves.

Focused197 tests pass, including all input/context fields, duplicate and
multiplicity guards, signed-zero/NaN bits, inactive-only changes, matrix slots,
EFC world/row/component coordinates, count mismatches and malformed banks.
The unchanged94-file scope now collects3044; this exact count is frozen.
Both full CPU suites and source-bound side-view retention are still required.
This is a source-only diagnostic revision, not a fresh CUDA trial or admission.

### Completed source-only retention at0696cb888d28

Both3044-test suites passed with zero omissions: Mac214.12s and WSL180.74s.
The319-byte inventory SHA256s are respectively
`3fac796f356db20afeb4463e5ed2f4b5dafaea100f442e943eb31532eaeb67d4`
and `d2d8799bbc2bc19875bef9976e85fd825b7b327f6cc9bf3c0949ba8a7fb7868e`.
Every CPU proof leaf/source closure/thread map was authenticated on both hosts.
Native invocation475d5c86f3c74cca989cab006a05e8d2 closed PID0/success0.

Independent source-only retention on both hosts produced identical literal
bytes at `artifacts/tools/contact-boundary-payload-view-0696cb888d28`:
diagnosis251170 bytes SHA256
`1e903a407f7a0dec692e1adf1e4ee8544c7af5ec70c36dbcbc7af97055391f1f`,
inventory112 bytes SHA256
`af92eeef8fe11920bcecc2e94a3c4f2bd5539134150cb4a10fdc2839be4ddba0`.
The first differing slot7 is world3 versus world18; it is not a comparison of
the same linked byte payload. Payload matching omits friction, solref,
solreffriction and solimp because those contact parameters were not captured.
Do not turn the captured subset equality into full contact-parameter or solver
equivalence. Preserve this artifact and its old phase label unchanged.

## Prospective phase correction and observed-row side view

The frozen installed constraint.py is identical on both hosts, SHA256
`b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53`.
Its dense construction clears Jqvel at3059 and launches the dense Jacobian
kernel with J/Jqvel outputs at3080–3096;2340–2343 computes and accumulates
J*qvel. Jqvel is therefore recomputed construction data before the current
solver, not stale prior-solver storage. Force/state/Ma keep their prior-storage
classification. Do not rewrite STALE_PRIOR_FIELDS, old capture declarations,
old reports or their decisions. New coordinate-view v2 retains the historical
tag separately and corrects the prospective tag with the source SHA.

Predeclare a pure output-observed contact-row subset, not a full eligibility
classifier. Start with completed per-world active rows of frictionless/pyramidal
type5/6 and require valid active contact IDs, matching world/CONSTRAINT bit,
condim1/3, complete contiguous address blocks and exact id/type backlinks.
Require matching bounded contact-after/complete nefc, no duplicate, missing,
extra or out-of-prefix row links. Read no address slots for contacts without
observed completed contact rows: an unused address can retain prior bytes,
including positive numbers or minus one. Never substitute a host float margin
predicate for GPU eligibility or label unobserved contacts inactive/overflow.
Leave all unused/inactive bytes intact. Dense sparse-J metadata is not a row
offset. Keep force/state/Ma out of construction-row comparisons.

Source init1909–2030 allocates rows with per-world atomic addition and records
contact offsets/IDs. The collision writer reserves global contact slots with
an atomic addition (collision_core.py213), but this source observation does
not prove which runtime invocation caused the observed order or numerical
negative. Observed offsets are storage associations, not physical identity,
fresh-allocation evidence, complete contact equality or solver equivalence.
All five flags remain false. No CUDA trial, learner, longer rollout, video or
physical motion is authorized by these helpers.

Focused source tests cover the historical Jqvel correction, untouched raw banks,
condim1/3 across worlds, preceding non-contact rows, arbitrary unused addresses,
orphan/extra/missing/backlink/type/world/extent/overflow failures. The unchanged
94-file scope now collects3069, frozen in the producer. New same-source CPU
proofs and separately rooted source-only retention are required again.

### Completed row retention at2ad1299b090e

Both3069-test suites passed without omissions: Mac326.95s, WSL184.49s.
The319-byte inventory SHA256s are
`1e10c5a3c02f7eebe49ddc69907e41e5d27d5fdb5237523cc89680966f0b633e`
and `4b4549039e41b926998b8ab928dcf27a285145934c840a720221d3b38124c92a`.
Every proof leaf and the exact source/count/thread closure was authenticated
on both hosts. Native CPU invocationc5e1467230624f169d81d9849a9d129d closed
PID0/exited/success0, reported service MemoryPeak5009879040 bytes, within6GiB.

At `artifacts/tools/contact-boundary-payload-view-2ad1299b090e`, independent
source-only reports on both hosts are literal-byte identical:640145-byte
diagnosis SHA256
`75ee9e03d8649c7d8c3d50149d31c4fec2f8e33de2effdd4795b42a2988d9f22`,
112-byte inventory SHA256
`cb91cc34c8abfc0c04d9c69e9ff157f2dcd9ed6e09d6e1102e37328c3398409c`.
External whole anchors were checked before JSON decode and literal byte equality.
The7783-byte CPU-only driver matches both hosts, SHA256
`39bd1712fe94248fa1a1670ca81c4fd395b1a363c965cb3298e83cdb58133d86`;
the84093-byte installed construction source also matches the pinned SHA.

Forwards0–3 have no observed contact rows. At each forward4/5/6, both arms have
2048 valid observed row backlinks and512 unique captured payload links. Of
those links,421/416/412 respectively occupy different per-world row offsets.
These are associations in the retained capture, not a replay, newly accepted
runtime, physical-contact match or proven cause. The old numerical negative
and every qualification flag remain unchanged.

## Matched construction-field subset predeclaration

Add a pure side view over the same caller-authenticated same-arm/forward
packets. Require captured model metadata exact, then pair only unique raw
input/context payload links with observed row blocks on both sides. Align by
local within-contact row ordinal, retaining both original absolute row/byte
offsets. Compare literal bytes for J,pos,margin,D,vel,aref,frictionloss,Jqvel;
use id/type only as backlink guards. Do not rewrite IDs, sort raw rows, compare
force/state/Ma, use sparse metadata as dense offsets or classify unobserved
contacts. Report compared/differing raw-word counts and first differences for
each field; zero compared words means null exact, never vacuous acceptance.
Leave duplicate, unmatched and unobserved links explicitly outside the compared
subset. Missing friction/solver parameters still prohibit a full contact
equality or solver-equivalence conclusion even if all compared words match.

Focused tests cover reordered unique contacts and shifted row offsets across
two worlds, all eight captured fields, separate first-delta offsets/components,
signed zero/NaN payload bits, null empty/duplicate/unobserved comparisons, model
mismatch, malformed backlinks and raw immutability. The unchanged94-file scope
collects3088, frozen in the producer. Separately report raw qpos/qvel/ctrl
equality and explicitly do not assert equality of all construction drivers.
Require fresh exact-source full CPU proofs
on both hosts and new separately rooted authenticated retention. No new CUDA
trial or training/longer-rollout/video/physical admission follows this view.
