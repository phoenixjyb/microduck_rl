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
refuse absent/changed settings. Do not mutate the owner environment or CUDA
recipe, install packages, widen service task limits or reuse an older receipt.
The reviewed collection is2940 tests; its ordered-scope
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
No native run exists for this experiment; the earlier numerical negative remains
the latest real-model result. Native execution remains blocked on prerequisites.
