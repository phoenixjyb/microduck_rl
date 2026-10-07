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
