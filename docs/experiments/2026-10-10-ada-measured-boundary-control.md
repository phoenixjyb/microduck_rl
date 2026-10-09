# New measured boundary and identical-bank solver control preparation

Base `7ca75aca57613cc803a42f62362d61a67fb9ec58`, exact branch
`feat/athletics-obstacle-curriculum`. This follows the exactly received
[saved prepared-pose response](2026-10-10-ada-saved-pose-solver-response.md).
The first change implemented **import-inert packet primitives and synthetic
tests only**. Subsequent sections separately predeclare allocation and measured
capture arms; no identical-bank CPU solve is authorized by their launchers.
Simulator/PPO admission remains closed. Libraries,
driver, plant, motor inputs, solver options and existing gates stay frozen.

## Why a new measurement is necessary

The historical authenticated Ada packet retains seven prepared state inputs,
347 model arrays and solved outcomes, but not every actual measured collision
pose or the complete immediately-before-solve Data bank. The completed CPU
response supplies eleven prepared poses and generates a new CPU manifold. Its
small descriptive residuals do not establish identical GPU/CPU solver inputs.
Do not reconstruct historical GPU inputs from solved forces or accelerations.

A future measured run starts from those authenticated seven state/motor/model
inputs but computes fixture Warp **GPU kinematics exactly once**. Its eleven
actual before-collision poses become the reference for that new run only. This
is a new measured replay, not a recovered historical Ada boundary. Allocation's
internal native kinematics remains counted separately. There is no integrator.

## Predeclared measurement and control semantics

Use the unchanged ordinary forward stage order, through solve, retaining the
same five complete boundaries as the preceding response: before collision,
after collision, after construction/transmission, immediately before solve
after factorize=True acceleration, and after solve. Keep position
factorize=False semantics and all ordinary sensor/energy/force stage ordering;
stop before acceleration sensors/integration. The helper prepends one fixture
kinematics call to the already tested stage-order helper; it is not a launcher.

Capture every allocated Data array at all five boundaries: 114 per bank, 570
raw leaves, including inactive contact/constraint capacity. Require actual
complete dtype, logical shape, stride, expanded NumPy shape/dtype, raw length,
finite values, and exact pose/state stability. Maintain the existing contact
candidate/address and six-field solver write fences. Reconstruct resultants
under the established ordered arithmetic only; no contact normalization,
spatial pairing, altered forces, rounding or tolerance-based acceptance.

Bind all 347 model arrays separately by exact bytes/layout against the prior
prepared bank. Additionally serialize the **complete** actual Model/Data
dataclass/container/scalar topology and values: options, enums, callbacks,
solver dispatch statics and all six Data capacities, not a selected whitelist.
Floating static values preserve signed zero; nonfinite/callable/unsupported
objects fail closed. Portable identities exclude pointers and device identities;
those remain separate within-process/device checks. A manifest must equal a
fresh complete actual walk: a self-consistent tree that omits real scalar fields
is insufficient. Array bytes are not supplied by the scalar serializer.

The frozen two-world capacities are nworld2, naconmax256, naccdmax256, njmax512,
njmax_pad512 and njmax_nnz10240. Canonical layout comes from the authenticated
array report, never from an unchecked consumer-supplied layout. Unexpected
static/backend differences must be diagnosed, not silently excluded or repaired.

For a separately authorized CPU control, allocate fresh CPU Model/Data and
prove the same complete model bytes and statics. Restore **all 114 actual GPU
before-solve Data fields** exactly, preflighting every layout, finite leaf and
complete Model/Data statics before the first write. Verify every post-copy raw
byte/layout, then call only the frozen public solver once. Do not recompute
kinematics, collision, construction, factorization, sensors, actuation or forces.
The frozen public solve allocates fresh SolverContext; no persistent GPU context
is reconstructed. Its warmstart copy direction stays destination qacc from
source qacc_warmstart (or qacc_smooth when disabled).

Both branches must respect the unchanged six-output write fence. Compare all
six complete outputs, reporting exact hashes, byte equality, counts of scalar
elements with differing bytes and ordered CPU-minus-GPU maximum absolute differences over
the full allocated capacity. Validate all three complete input/output banks,
not just active solver slices. No RMS/BLAS reduction, tolerance, solver tuning
or admission inference. Identical inputs do not imply identical CPU/GPU kernel
execution; retain that distinction explicitly.

## Initial primitives implementation boundary

`ada_measured_boundary_packet.py` implements complete portable static binding,
new measured pose-reference checking, stage-order composition, exact complete
CPU Data restore and full solver-bank comparison. It imports no NumPy/Torch/
Warp/MuJoCo runtime at module import and has no CLI or runtime allocator.
Mocks exercise the runtime helper contracts without executing physics.

The primitives alone are **not sufficient execution authority**. At this initial
stage the following remained needed: closed source-bound CPU/GPU runners, complete model/runtime/topology bindings,
strict retained packet receiver, cache/device/source provenance, resource and
service receipts, watchdog/lease and foreign-workload guards, independent
review, and an exact-source predeclaration for each launch. At the primitives
stage no third GPU profile had been added to the two-profile supervisor.
The later allocation and GPU predeclaration sections below supersede that
source-only status only for their respective bounded arms; the CPU identical-
bank solver launcher remains unimplemented at this revision.

Future GPU diagnostic must inherit the existing single-Duck advisory lease,
owner/child/service deadlines and GPU occupancy/temperature limits. Preserve
Grounding DINO/FilmBrain and inactive protected services. CPU control gets its
own explicit hidden-CUDA service and fresh private cache. Never overlap Duck
GPU runs. Do not begin either launch unless it can finish safely before
07:00 Asia/Shanghai; reserve ten minutes for closeout. No PPO, raw perception,
policy promotion, service restoration or physical robot motion is authorized
by this preparation.

## Source-only verification plan

Run the six-file suite on Mac, review independently, commit/push the exact
feature source, clean-fast-forward native100.100, then run the same contract
suite in a CPU-only user service. Cap60s/MemoryMax2G/CPUQuota200%/TasksMax64,
Nice10/LimitFSIZE16M/LimitCORE0/Restart=no/KillMode=control-group/
TimeoutStopSec10/Type=exec/RemainAfterExit=yes, thread pools1, literal
CUDA_VISIBLE_DEVICES empty. This suite initializes no simulator physics.
Retain JUnit and actual service receipts under the unique source8 prefix in
`artifacts/tools/ada-measured-boundary/`; require equal complete testcase
multisets and zero failures/errors/skips. Physics needs a later separate gate.

## Preparation review

Owner six-file suite passed408 cases in9.20s after the final schema correction;
git diff whitespace checks passed. Independent Luna read-only review passed345
focused cases in5.04s and found the prior complete-static/capacity/layout gaps
closed. Its one remaining output-label issue was corrected before freezing:
`differing_scalar_elements` counts scalar elements whose bytes differ, not byte
offsets. The source-only imports/AST checks and tests do not execute physics.
Native at17:27 UTC retained clean base7ca75aca, Dino PID1592/946MiB as the
only GPU compute owner,43C/0%, and both protected services inactive in system
and user scopes. This snapshot is not authority for a later GPU launch.

## Retained source-only verification

Source `a8fa2ef9517e7bfaa0159740f0ae5e9204f96ce3` was pushed to the fork and
clean-fast-forwarded on native100.100. Committed-source Mac408/408 (6.67s) and
native408/408 (2.57s) JUnit testcase multisets match exactly, with zero
failures/errors/skips. Native unit
`microduck-ada-response-contracts-a8fa2ef9.service`, invocation
`1a5ad9bc07db42f0a5b7a778ce991137`, completed in3.133567s with the declared
60s/2G caps, status0/Result success/MainPID0/empty ControlGroup, active/exited.
No simulator allocation, collision, solve or GPU Duck job ran in this suite.

Both hosts retain these files in `artifacts/tools/ada-measured-boundary/`:

| Evidence | SHA256 |
| --- | --- |
| `a8fa2ef9-mac-tests.xml` | `cf5c24e46e8a7b2d0ec5af06fcf40a49ea6635e51affd3364124cd46bb569c35` |
| `a8fa2ef9-linux-tests.xml` | `cbd3fe78c2bef749c4658eca01700748e0ae696139dfe9f420d1b1e944edeeaa` |
| `a8fa2ef9-native-services.json` | `fbdea309d42825afd1d64c00157da614e37dc976a2905d2b8f4beb5a4b51f5f7` |

Read-only service receipt helper remains the previously retained
`ada-saved-pose-response/retain_response_services.py`, SHA256
`8a6104530e3b4940762b9372c71be4751beb6c9dbe0d298b98639349452bb060`.
At17:29:59 UTC it verified both protected services inactive in both scopes,
Dino PID1592/946MiB only,44C/0%,961MiB total used. Source/test evidence only;
the new measurement and identical-bank control remain unexecuted.

## Narrow actual CPU allocation/static validation predeclaration

At clean base56f5e8a2, add `ada_measured_model_probe.py` before implementing the
GPU capture runner. Independent frozen-source review found that the complete
walker supports the actual retained Model/Data shapes, including NumPy-typed
counts, `geom_pair_type_count`, nested BlockDim/TileSet/Option/Callback and
array tuples. Preserve NumPy scalar dtype/raw bytes; do not coerce them into
Python integers. Numeric Option arrays remain part of the separate347-array
binding. `put_model` applies its existing tolerance floor unconditionally, not
only on GPU; do not change it or mislabel it a backend-only mutation.

This CPU probe compiles the same actual plant, performs one put_model and
one make_data with explicit nworld2/nconmax128/naconmax256/naccdmax256/
njmax512/njmax_nnz10240, and expands/copies the same two motor fields. Frozen
`mjlab/sim/randomization.py:42-48` launches its repeat kernel **twice**, once
per field. Retain these allocation/model-expansion launches and actual passive
CPU executable/cache metadata, not a zero-kernel claim. Allocation's one
internal native kinematics call remains counted separately. Copy only the
original seven state fields; do not supply any prepared geometry poses or call
fixture kinematics, forward, collision, constraint construction, factorization,
solver, sensors or integration.

Require exact347 model bytes/layouts, complete114 CPU Data fields, every actual
array on CPU, zero collision/EFC/solver counters, unchanged unsupported-topology
and callback guards, and complete actual static serialization bound to a fresh
walk. The portable receiver additionally checks every declared dataclass field
against frozen types.py AST declaration order; omitted real scalar subtrees
cannot pass by editing parent field lists. It independently checks all114 raw
leaves, original state bytes, capacities, source/service/flag/cache envelopes.
Portable reception does not allocate a fresh current Model/Data or establish
runtime binary identity. Native execution performs the actual static walk;
receiver schema/source checks and actual captured values remain distinct.

Bind the model-expansion Python source to its wheel RECORD and exact SHA256
`35d0bddcd6ef3b0317cb89987e0c42bbf1862fcd4beb795c1327ade814c2a64c`;
retain the unchanged MuJoCo-Warp/Warp stage-source audit digest separately.
No library/version/model/driver or simulator acceptance changes.

The executed `expand_model_fields` function's module name/source file and the
imported module file must resolve to that verified wheel file before its call.
Require one completed expansion call with exactly model/2/two named fields.
`source_expected_model_expansion_launches=2` is an inference from that frozen
source and completed matched expansion, **not** a dynamic launch trace or a
ModuleExec hook count. Post-run passive executable inventory must consist of
the one repeat kernel entry already retained in exactly received CPU response
05216efe: module repeat_array_kernel_39317a34/block_dim1, source/options hash
`64511f00215be403f7a05e8069fbed3ff8b6b02bb3b0a9560b56c2ee9f1f1847`,
one kernel hook, the two retained zero-smem metadata entries, CPU device, and
opaque handle explicitly **not** treated as loaded binary identity. Unexpected
modules/options/metadata fail closed and are diagnosed; do not loosen this
inventory after a failure. Retain actual cache files/hashes separately.

Frozen schema checking is annotation-backed, including nested mandatory
dataclass kinds, typed tuples, Warp array paths, integer/bool/float NumPy dtype
families, actual enum types and absent callbacks. Replacing a typed dataclass
with an arbitrary dictionary or turning a Warp array into a scalar is refused.

After focused tests, independent review, committed source push, clean native
sync and matching native contract checks, execute exactly one CPU probe as
`microduck-ada-model-probe-<source8>.service`. Use the preceding response
180s/6G capped service properties and exactly its five environment entries:
literal CUDA_VISIBLE_DEVICES empty and four thread pools1. Source-bound CLI
requires the actual service PID/invocation/caps, frozen native machine/venv,
clean exact branch, protected services inactive before/after, and launch before
22:50 UTC. Retain outputs in `artifacts/tools/ada-measured-boundary/` as
`<source8>-linux-model-probe.json`, matching `.npz`, with absent private cache
`<source8>-cpu-model-cache` under that same parent. Authenticate input directory
`artifacts/evaluations/ada-duck-contact-0ce8c9d0a820` by the historical report
and all prior file hashes. No new fixture physics or GPU Duck run under this
predeclaration. Require exact native reception with cache rehash and portable
Mac reception; failures are diagnosed read-only before any repair.

Preparation checks: owner seven-file suite473/473 in3.57s and whitespace checks
passed. Independent Luna read-only final review passed65 probe cases in0.71s
and found no material source/test blocker for the allocation-only CPU arm.
That review executed neither allocation nor physics. It confirmed both limits:
two expansion launches are source-inferred, and portable reception checks the
retained typed static manifest rather than remeasuring live scalar values.
The earlier negative expansion-source test initially aliased its expected dict;
the fixture was corrected with deepcopy before final passing tests. No runtime
probe has run yet. Native contract suite uses the existing60s/2G limits and
unique source8 artifacts; the actual allocation arm separately uses180s/6G.

## Retained actual CPU allocation/static packet

Execution source `03a8be9209597b158468ed2901089ba9358ced3b` was pushed and
clean-fast-forwarded on native100.100 before launch. Committed-source seven-file
tests passed473/473 on Mac (3.34s) and native (3.17s); complete testcase multisets
match with zero failures/errors/skips. Test service invocation
`bac7a27bf38842d7bde3ddfbc68774e0` succeeded in3.732743s under60s/2G caps.

CPU allocation unit `microduck-ada-model-probe-03a8be92.service`, invocation
`5fb279ea08154a81ac1a644f2359167d`, succeeded in5.934013s under every declared
180s/6G cap. Its terminal state is MainPID0/status0/Result success, active/exited,
empty ControlGroup. Actual source/schema/byte checks succeeded: all347 model
arrays,114 Data arrays,461 complete static array paths and184 static/container
nodes. Static manifest SHA256 is
`8180631aae0b6d04130f140737c79cef65349b2b38ca9227134d637718fcf78e`.
All collision/EFC/solver counters are zero. The only held CPU executable is the
predeclared repeat kernel with matching source/options hash/metadata, not an
unexpected dynamics module. Its three private cache files total5122 bytes.
Loaded binary bytes remain unbound; the two repeat launches remain source
inference, not an instrumented runtime count.

Native exact receiver with **every cache file rehashed** passed, and portable
Mac reception passed independently with no Warp/Torch/MuJoCo runtime imports.
Independent Luna artifact review also passed portable source/schema/all114-leaf/
461-path/184-node reception, exact473-case XML comparison, helper/service hashes,
and every recorded cap/exit. It did not query native/cache state; the owner
performed that separate read-only native cache check. Actual runtime scalar
values are retained measurements checked against a fresh walk during execution;
portable reception does not independently remeasure them.

Both hosts retain these files under `artifacts/tools/ada-measured-boundary/`.
The fresh private CPU cache remains only on native to avoid duplicate storage.

| Evidence | SHA256 |
| --- | --- |
| `03a8be92-linux-model-probe.json` (98821 bytes) | `469558cbbf19d11c4b7c2d14619dac6ceecba089d02dabbbbb3a6083fcb25eb1` |
| `03a8be92-linux-model-probe.npz` (232864 bytes) | `d997420e6449b4f57848bc7393d1a512d22d3ae262232c7879e1ab240989c80c` |
| `03a8be92-mac-tests.xml` | `c0f9d16ee817a62596f25b1a229f1f4759a6d4345e11cf9d0cfc42316c085bfa` |
| `03a8be92-linux-tests.xml` | `649750cfd1fc99e39232ffc9921999ac50e16b04f580b0c534858cfb000a9a2d` |
| `03a8be92-contract-service.json` | `9eb10612d1101ec25ae2dd39a6e994ecb25509572d3a2b9f7f66bdd40f45720b` |
| `03a8be92-native-services.json` | `e7343fe8e0a84ded8ba19a56bfbd6de55d0bd6cac0b257a78668648b054925b9` |
| read-only `retain_model_probe_services.py` | `f8aa0a4a0024984c3fdde56e165bf7b2bb9b4a3d47fb1ddb19dd833dd0dbd263` |

No Duck GPU process or active CPU process remains from these units. Protected
mission services stayed inactive in both scopes and Dino remained the sole
GPU compute owner. No fixture dynamics or solver control was executed. The
next execution gate, after its source/review/native-contract checks, is the closed supervised **new measured GPU boundary capture**,
followed separately by a same-GPU-before-solve-bank CPU solver-only control.
The allocation/static result does not reopen simulator/PPO admission.

## Closed new GPU measured-boundary arm predeclaration

From clean base33ccaac, `ada_measured_gpu_boundary.py` adds an explicit **third
closed diagnostic profile** to the existing stdlib supervisor. The old two
profiles retain identical bounds, flags and behavior. The shared topology guard
still defaults to CPU-only; only this reviewed runner explicitly selects Ada
cuda:0/arch89/ordinal0, then checks every actual461 array's device separately.
The named third profile/unit prefix is closed, not arbitrary plug-in authority.

Require the retained CPU allocation JSON and NPZ exact hashes above and full
portable reception before construction. Allocate one fresh matched plant,
put_model and make_data on selected Ada, with all six explicit Data capacities.
Allocation performs its one internal native kinematics; execute the verified
wheel helper once to expand only two motor fields, copy their exact retained
bytes and the original seven state fields. Do not supply old poses, recompute
BAM, invoke native fixture forward, ordinary public forward or integration.
Before fixture physics, bind all347 model array bytes/layouts and the complete
actual static manifest. Require exact **8180631a...fcf78e** CPU/GPU static
equality; do not omit/normalize/rewrite backend differences on failure.

Then execute one new Warp fixture kinematics followed by the already tested
frozen stage sequence through one solve. Retain five114-field raw banks plus
separate full statics, canonical expanded layout and ordered descriptive
analysis. The eleven actual before-collision poses are this new run's reference
and must remain byte-exact at all later boundaries; they are not a recovered
historical GPU pose bundle. Check original seven inputs, complete model bytes,
all statics, actual pointers/device/layout, absent callbacks and default options
at every boundary. Check contact counters/candidates immediately after collision,
bounded fourteen-friction-row/dim3 construction, no early solve, candidate
stability and the unchanged six-output solver write fence. Stop before
acceleration sensors. CPU identical-bank control is **not** executed here.

The receiver independently authenticates every retained file, separate child
and launch receipt, source tree/modules/lock, frozen wheel source audits and
expansion helper, actual service PID/invocation/caps, exact known advisory lease,
protected service and foreign-owner receipts, complete telemetry/cap checks,
all570 arrays, layout/finiteness/state/pose/contact/write fences, full static
schema/equality and recomputed ordered analysis. Private cache bytes and
already-held CUDA ModuleExec source/options metadata are retained separately;
never trigger extra build/load/hash calls or infer loaded binary bytes,
executed-kernel identity, launch count, simulator admission or training permission.

All source/mock checks, independent review, exact committed push, clean native
fast-forward and matching native CPU contract checks must pass **before** one
GPU measurement launch. Use unit `microduck-ada-measured-boundary-<source12>.service`
with180s/6G, owner150s/child120s and20s owner closeout reserve, CPUQuota200%,
Tasks64/Nice10/LimitFSIZE16M/LimitCORE0/Restart=no/KillMode=control-group/
TimeoutStopSec10/Type=exec/RemainAfterExit=yes. Parent literalCUDAempty/thread
pools1; child inherits only the existing held lease, CUDA0/disabled CUDA cache,
five fresh private cache directories and64MiB Torch allocator budget. The
unchanged shared GPU gates remain≥10240MiB free,≤12288MiB total used,
≤2048MiB aggregate growth, temperature<65C and utilization≤85%. Recheck foreign
PID1592/DINO and predeclared lease device66306/inode11419619/empty/owned regular
file before launch. Both protected services remain inactive in both scopes.
On demand, capacity, source, service, lease, telemetry or child failure, kill
only the owned child and diagnose retained evidence read-only; never loosen
guards, stop DINO, reinstall libraries/drivers, or restore protected services.

Unique fresh evidence root is `artifacts/evaluations/ada-measured-boundary-<source12>`;
retain launch.json, child.log, child.json, measured-stages.npz, measured-statics.json,
measured-layout.json, measured-analysis.json and report.json. Cache stays native;
portable packet is copied to Mac and received independently. Launch only before
22:50 UTC with safe deadline margin. After success, a separately reviewed/source-
bound CPU-only solver control may restore the authentic **new GPU** before-solve
bank. No PPO, raw perception, physical motion or learned-skill claims arise here.

Source preparation: owner nine-file CPU-hidden suite passed593 cases with one
explicitly deselected actual-physics test; zero failures/errors/skips. It covers
the seven existing allocation/boundary/source suites, new measured-profile
synthetics and unchanged supervisor cleanup contracts. Whitespace checks pass.
Independent Luna final read-only review passed188 focused cases in9.19s and
found no material source/test blocker after the direct pre-physics347-model-
hash comparison and documentation status corrections. Neither review nor the
source/mock suite ran allocation, simulator physics, GPU work or services.
Execution-source SHA and matching committed-source/native JUnit receipts will
be resolved and retained before any launch; dirty-source execution is forbidden.

## Retained actual new measured GPU boundary

Execution source **`daebeb6ee97b1ce0c4a665331306bcb78a94204c`** was reviewed,
committed/pushed to the exact fork feature branch and clean-fast-forwarded on
native100.100 before launch. Its matching source nine-file suites passed593
cases on Mac (19.28s) and native, with identical complete testcase multisets,
zero failures/errors/skips and one explicitly deselected actual-physics test.
Native contract unit invocation `fa7715c58c1041b39c8a8d552fa2751a` succeeded
in5.179066s under all60s/2G limits, PID0/empty ControlGroup.

Prelaunch exact-source/source-audit/allocation reception and available exclusive
advisory lease check passed CPU-hidden with no simulator runtime imports or GPU
calls. Snapshot: Dino PID1592 sole foreign owner,961MiB used/15232MiB free,
44C/0%, declared empty owned lease device66306/inode11419619, protected services
inactive in both scopes. This receipt preceded the actual capped owner launch.

Unit `microduck-ada-measured-boundary-daebeb6ee97b.service`, invocation
`1138238649d14adeabb11c86533e8470`, completed successfully under the predeclared
180s/6G caps. Actual terminal duration34.737338s (supervisor34.687609s),
MainPID0/status0/Result success/empty ControlGroup, active/exited rather than
a live worker. Peak captured GPU used1262MiB/temperature50C/utilization6%;
foreign owner, protected services and existing lease stayed unchanged. The
watchdog and capacity gates were not loosened. No other workload was stopped.

Native full receiver with every fresh private cache leaf rehashed passed, then
portable Mac reception passed independently in a fresh CPU-hidden process
without Warp/Torch/MuJoCo runtime imports. All570 raw stage leaves,114 canonical
layouts,461 static array paths and184 scalar/container nodes were received.
The complete actual static manifest equals the retained CPU allocation digest
8180631a...fcf78e; its JSON file hash below is distinct from that semantic digest.
All347 model bytes/layouts match the predecessor and CPU allocation. The actual
eleven before-collision poses remain byte-exact throughout this new run, as do
the original seven state fields. All six allowed solver output fields changed;
every other Data field is byte-exact across solve.

| Boundary | nacon / ncollision | nf | nefc | solver_niter |
| --- | --- | --- | --- | --- |
| Before collision | 0 / 0 | [0,0] | [0,0] | [0,0] |
| After collision | 8 / 4 | [0,0] | [0,0] | [0,0] |
| After construction | 8 / 4 | [14,14] | [14,46] | [0,0] |
| Before solve | 8 / 4 | [14,14] | [14,46] | [0,0] |
| After solve | 8 / 4 | [14,14] | [14,46] | [1,4] |

The new GPU solve's twenty final fields retained in the historical packet are
all byte-equal to their old GPU counterparts. This is a **subset descriptive
agreement**, not recovery/authentication of the historical missing full pose or
before-solve bank. Both feet again have four distinct generated contacts;
ordered vertical resultants are3.2832817435264587N and3.284973382949829N. Their
existing native comparison remains a generated-manifold comparison, not an
identical-input CPU solver control. The new complete GPU before-solve bank now
makes that separate control possible without guessing historical inputs.

Native private cache contains84 files/19224432 bytes;25 already-held CUDA
ModuleExec metadata rows were retained. Cache file hashes are not loaded-object
byte identity; module source/options/opaque handles are not executed-kernel
proof or dynamic launch counts. All runtime/solver/simulator/training/physical
qualification flags remain false. No integrator, optimizer, BAM proposal,
CPU identical-bank solve, policy training or physical motion ran in this arm.

Native packet root is `artifacts/evaluations/ada-measured-boundary-daebeb6ee97b`;
the eight portable files also reside on Mac under
`artifacts/tools/ada-measured-boundary/ada-measured-boundary-daebeb6ee97b`.
Private cache stays native only. Source tests/prelaunch/terminal helper receipts
are retained on both hosts under `artifacts/tools/ada-measured-boundary/`.

| Retained evidence | SHA256 |
| --- | --- |
| report.json (100413 bytes) | `5146e17ace38e5b0e2897cb0a3febf1202b66d9d69b757ac0398e2c581d267de` |
| child.json (81247 bytes) | `d6c3b4b0524fa6a7fb6c29411afc125e1ffbdca9eb4060afa69673ebd97fde86` |
| child.log | `697c5618834554c19cd3b4a16848b068b2c5689d8e845cfbec4b7bca08a18c4e` |
| launch.json | `18f2759cd63df139b552a15ccf28706125e59ec08ac2b2f0b5f35cbe18d3a6f4` |
| measured-stages.npz (1180648 bytes) | `b73b220693f77bf4c33de50bd7b5d5bd63805bb50f6d91b5d54581e8712cd195` |
| measured-statics.json | `5d2a23d4b17e289033c4a9532970b0b4bd2ffce5232783ca444ca318c9f7738e` |
| measured-layout.json | `334d7d3e125e47d9dae37470b79443875415799bbd195329b7403125a273955c` |
| measured-analysis.json | `1ec22c0ca60cc1fe21bb68680b38e7561e29adb74a60592d557d5be52d474c32` |
| daebeb6e-mac-tests.xml | `3abb9acb5143358b9469a1dd50279dcb4a76ab85c5bc71fbc4607a85fe1a94dd` |
| daebeb6e-linux-tests.xml | `b69d71df69be1d286c4bc9e250eaf5df85f41eaaa4c5fad4a6c0f346146504b5` |
| daebeb6e-contract-service.json | `7a3d55813fdd0b7690d8f4ae302d2cebc5bca17475d1ebb47e9c32050b6c6fed` |
| daebeb6e-prelaunch.json | `fbb607158251bfbdf2ebfb88a19fa84a305a58b8c1354a122459293fd8b4cced` |
| daebeb6e-gpu-service.json | `2ac2c91901b380e7d35efc879ae253b39fe6cdd98942deeba201a64aa60cc1e2` |
| retain_measured_boundary_services.py | `4b4e3c51e1f0ffe606646d13e8b2257dd7624eff20596bced796022d21560805` |

Next: separately implement/review/test/predeclare a CPU-only solver control,
binding this exact report/full packet, restoring every before-solve array and
proving all model/static/byte bindings before one public solve. Never recompute
kinematics, collision, constraints, factorization, sensors, actuation or forces
in that control. Original simulator/PPO admission remains closed.

Independent Luna actual-artifact review passed portable reception in a fresh
CUDA-hidden process with no simulator runtime imports. It independently checked
the eight-file/source/predecessor/allocation packet,570/461/184 inventories,
all eleven new poses, stage counters and six-only output fence, matching593-case
XMLs, terminal service/helper/report/NPZ hashes and20/20 old final-field byte
agreement. It found no packet blocker and preserved the historical-boundary,
same-bank CPU-control, training and physical limits. Native cache was not
queried by that review; the owner's separate native84-file rehash is the
cache evidence. This closes **diagnostic packet reception only**.

## CPU identical-new-GPU-bank solver-only arm predeclaration

From clean base941133c7, `ada_same_bank_solver.py` adds a separate CPU-hidden
runner/receiver, not another GPU profile. Pin new GPU execution source
daebeb6ee97b1ce0c4a665331306bcb78a94204c and exact report hash
5146e17ace38e5b0e2897cb0a3febf1202b66d9d69b757ac0398e2c581d267de.
Authenticate and fully receive its eight-file/570-leaf packet before any CPU
allocation; bind every payload, canonical layout, full static manifest,
predecessor/model/motor/state input and stage source. This is the new actual
measured GPU bank, never reconstructed historical inputs.

Allocate one fresh CPU plant/put_model/make_data with unchanged explicit
capacities. Count its one internal native kinematics separately. Execute the
verified frozen helper once to expand only two motor fields (two source-inferred
repeat launches), copy exact retained motor bytes, and bind all347 CPU model
arrays and all184 scalar/container nodes to the new GPU/CPU allocation packet
before restore. Require every461 actual Model/Data array on CPU, absent
callbacks and default unchanged topology/solver/options. No BAM proposal.

Restore **all114** GPU immediately-before-solve Data leaves using the tested
complete restore helper. Preflight every source/target layout, dtype, stride,
expanded shape, raw length, finite decoded value, full statics and source
inventory before the first copy; then prove every actual post-copy byte/layout.
Retain that complete restored-before bank and a114-field hash receipt. Check
all model/static/device/pointer bindings before the call. The passive CPU
inventory before allocation must be empty; immediately before restore and
after restore it must be only the exactly received repeat executable. This
checks absence of hidden pre-solver stage kernels, not dynamic launch counts.

Then call the frozen **public solver exactly once**, which source-allocates a
fresh SolverContext. Restore no persistent internal GPU context. Do not call
fixture kinematics, collision, construction, pre-solve mass factorization,
sensor/force/actuation stages, ordinary forward or integration. The solver's
own unchanged Newton/Hessian factorization and line-search algebra remain part
of that authorized public solve; the zero pre-solve mass-factorization claim
does not claim absence of the solver's internal algebra.

After sync, retain all114 CPU after-solve leaves. Recheck full model/static/
pointer/device/layout bindings and original iteration bounds; reject any write
outside the unchanged six solver outputs. Compare complete GPU after-solve and
CPU after-solve outputs on the exact same restored input bank, with full hashes,
byte equality, differing scalar-element counts and ordered CPU-minus-GPU
maximum absolute differences. No RMS/BLAS, tolerance, row normalization,
manifold reconstruction, solver tuning, gate relaxation or admission inference.
Both228 serialized branches and all114 restore hashes must receive exactly.

Passive post-solve CPU metadata is restricted to11 predeclared Newton/support/
repeat modules listed in `ALLOWED_MODULES`; collision, construction, sensors,
passive/force and forward stage modules are excluded. Required repeat/solver
rows must exist. Bind source/options hash, block dimension, device, metadata and
false loaded-binary flag to the subset of the exactly received prior CPU
response at source05216efe9d33a80d7a839eaa20b7a57ea05dbe86, JSON hash
af380438ef6da87e1e77f7eefcb5cdc1da8724fab757a8b9d2836c2f0a8386e9 and NPZ hash
b3c52c1463863389ba8161ac9fca0bd080b020e01b48920405904024c183dc34.
Current opaque handles remain opaque; positive hook counts may be lower than
that full-stage response, never higher and never treated as launch counts.
No build/load/hash calls may be triggered just to inspect provenance. Exact
native private cache hashes are retained separately, not loaded-code identity.

Before launch require focused tests, independent source review, committed push,
clean native fast-forward and matching CPU contract tests. Use
`microduck-ada-same-bank-solver-<source8>.service` with the existing180s/6G CPU
response caps: Type=exec/RemainAfterExit=yes/CPUQuota200%/Tasks64/Nice10/
LimitFSIZE16M/LimitCORE0/Restart=no/KillMode=control-group/TimeoutStopSec10 and
exactly five environment entries, CUDA_VISIBLE_DEVICES empty/four thread pools1.
The closed CLI checks actual source-bound PID/invocation/caps, clean exact
native machine/venv/source, frozen six distributions including BAM1.0.1,
pre/post installed stage/helper/source hashes and unchanged protected services
and Dino owner. Fresh private CPU cache uses actual `use_precompiled_headers=False`;
all enumerated Warp devices CPU/Torch CUDA uninitialized before and after.
No GPU allocation is permitted. Launch only before22:50 UTC, preserving the
ten-minute closeout reserve. Diagnose any failure read-only before changes.

Closed input is native `artifacts/evaluations/ada-measured-boundary-daebeb6ee97b`.
Fresh unique outputs under `artifacts/tools/ada-measured-boundary/` are
`<source8>-linux-same-bank-solver.json`, matching NPZ, and absent private
`<source8>-cpu-same-bank-cache`. Native receiver must rehash every cache leaf;
portable Mac receiver must authenticate full new GPU references, all228 branch
leaves, restored actual-bank receipt, model/statics/layout/source/caps/counters,
both complete solver fences and recompute every comparison. Retain terminal
service/exit and copied evidence separately. No PPO, learned-skill promotion,
raw perception, physical motion or protected-service restoration is authorized.
All original simulator/solver/physical qualification flags stay false even if
the two solver branches numerically agree.

Independent source review found the initial portable receiver did not require
its own checkout to be the execution revision. Before any launch, add a portable
clean exact HEAD/feature-branch gate and directly compare the current receiver
module bytes to that revision's blob, with negative tests for wrong HEAD,
branch, dirt, module bytes and malformed source. Full receipt must run in a
fresh process at the clean execution revision before editing closeout docs;
later historical reproduction must use that exact clean revision, not a newer
checker checkout. Native machine/interpreter checks remain additional runner
requirements, not portable Mac requirements.

Preparation check after that repair: all ten retained CPU-hidden contract
modules passed **665 tests**, with the single real-physics test explicitly
deselected (13.33s). The new solver-control file contributes72 synthetic/source
tests, not physics acceptance. Independent Luna source review verified the
repair and bounded restore/one-solve path; its focused289-test suite passed
without runtime allocation or service operations. No remaining source/mock
blocker was found. `git diff --check` passed. Commit this predeclaration before
native tests or runtime launch; successful contracts do not predict solver
agreement or establish simulator admission.

## Retained CPU identical-bank solver result

Executed clean source d75ef0cde0be3198e2f0372c4d5f9fe0aa17825d after fork push
and verified native fast-forward. Committed-source suites passed665 cases on
both hosts (Mac33.61s, native5.45s), identical testcase multisets and no
failure/error/skip; the sole actual-physics test remained explicitly deselected.
Native contract unit invocation bd0dd5055d384581ac7bdd9ce223fe56 completed in
6.012610s under60s/2G caps, with successful exit/MainPID0/empty control group.
These are contract checks, not runtime admission.

Then `microduck-ada-same-bank-solver-d75ef0cd.service`, invocation
0a812a9c49414ead9f91b4a6d195d29c, completed in **16.883411s**, exit0/Resultsuccess/
MainPID0/empty control group under the exact predeclared180s/6G CPU caps and
five-entry hidden-CUDA/single-thread environment. Source/machine/venv/versions,
four protected-service scope states and sole foreign Dino PID1592 stayed bound.
Terminal GPU occupancy remained961MiB used/15232MiB free,45C/0%; no Duck CUDA
allocation occurred. The expected hidden-CUDA Warp startup error100 did not
prevent CPU execution; actual runtime device enumeration and all461 actual
array devices remained CPU, with Torch CUDA uninitialized.

All114 actual restored-before leaves equal the authenticated NEW GPU
before-solve bank byte-for-byte, with all114 restore hashes. Both complete
114-field branches received (228 leaves; full canonical allocated capacities,
including padded slots). Model347/static184/device461/pointer bindings and
complete fences passed. The sole public solve changed exactly the six permitted
Data outputs in both arms; no other Data field changed. Counters remain
nacon8/ncollision4, nf[14,14], nefc[14,46], ne/nl[0,0], solver_niter[1,4].

| Complete solver output | Max absolute CPU minus GPU | Differing scalar elements | Byte equal |
| --- | ---: | ---: | --- |
| efc/Ma | 1.4901161193847656e-8 | 24 | no |
| efc/force | 7.450580596923828e-8 | 48 | no |
| efc/state | 0 | 0 | yes |
| qacc | 1.9073486328125e-6 | 39 | no |
| qfrc_constraint | 4.76837158203125e-7 | 31 | no |
| solver_niter | 0 | 0 | yes |

These ordered full-buffer differences use the frozen comparison convention;
they are not a tolerance pass. In this single two-world fixture, CPU versus GPU
Warp solve on the identical new bank produces much smaller differences than
the retained native-MuJoCo versus Warp full-forward discrepancy. That narrows
the next investigation toward upstream state/contact/constraint construction;
it does not establish exclusive causation, general repeatability, correctness
of either solve, native solver equivalence or acceptance of contact behavior.
No library/plant/solver tuning, row normalization or gate change was made.

Native fresh-process receiver rehashed all30 private cache files (479216B),
separate from ten held CPU ModuleExec rows. The pre-solve inventory was exactly
the retained repeat row; post-solve metadata stayed within the eleven-module
predeclared whitelist (support was not loaded). All source/options/meta rows
passed the prior exact CPU provenance reference; hook counts are not launch
counts and loaded binary identity remains unbound. Full pure portable receipt
passed on Mac at the clean execution revision in a fresh process without
Warp/Torch/MuJoCo imports. Independent Luna re-received the same full packet,
hashes, counters, fences, comparison, service limits/terminal state and test
inventories and found no blocker. That review did not rehash native cache;
the owner's separate native check is the cache evidence.

Exact retained files under `artifacts/tools/ada-measured-boundary/`, copied to
both hosts (private CPU cache stays native):

- d75ef0cd-linux-same-bank-solver.json (143694B):
  b0287a679ad297dfb40a71a6c03a2e99a3227d298da109edcdde3b9e131c0339
- d75ef0cd-linux-same-bank-solver.npz (472546B):
  8193be85083cb50e031a3fa95238daa3d6f7eeb2fa36f60c93e5074e3406cb40
- d75ef0cd-mac-tests.xml:
  f5b0b22c594ddf31209a774af3b73897136090632b1732a423fd9b79199f12d6
- d75ef0cd-linux-tests.xml:
  a185dec764fd609d18acb9c2153a136ed332005010266e37ef72d3eba031448f
- d75ef0cd-contract-service.json (3625B):
  7e7e40dd75ee0c87fe906e74786d399d82b9031416fae5e209b17222ada4e24a
- d75ef0cd-solver-service.json (10584B):
  3ad868c73bc37fa599a0fd292fd562a8a582cdddca8177900a75c98628103f50
- retain_same_bank_solver_service.py:
  91d9dd085f0cc05a3584554211d5a32be479185a41137f823e0d5f8853d6fb07
- d75ef0cd.bundle, verified incremental source bundle from941133c7:
  67c295e3b61a4a52cd05e3b2fc77430c72ec8f318cd126c3ed329ca2b3d6a923

Decision remains **complete-new-ada-bank-cpu-solver-control-collected-not-admission**.
This closes the separately predeclared same-bank diagnostic, not the historical
live GPU boundary. Original solver/simulator/training/physical flags remain
false. Next bounded work is a source-backed upstream discrepancy diagnosis and
an evidence-justified predeclaration, not PPO, video or hardware motion.

Closeout documentation check verified all eight new artifact hashes against
local bytes and identical native copies, the existing relative Markdown link,
and whitespace. No runtime/source code changed after the665-case tested
execution revision. Commit/push this retained result separately from that
immutable execution source.

## Predeclared measured constraint-input arithmetic audit

Next is a pure, CPU-hidden upstream arithmetic diagnostic, not another physics
run or a simulator-admission decision. Source
`src/mjlab_microduck/ada_constraint_input_audit.py` and its synthetic test module
must be committed, independently reviewed, pushed and clean-synced before a
retained execution. Preserve the frozen plant, libraries, original gates,
inactive protected mission services and sole foreign Dino workload. Deadline
remains2026-10-10 07:00 Asia/Shanghai; launch cutoff06:50. No PPO, video, hardware,
raw perception or new collision/constraint/solver/integration call is authorized
by this audit. Import/help are runtime-inert.

The exact new measured GPU report is source
daebeb6ee97b1ce0c4a665331306bcb78a94204c, SHA256
5146e17ace38e5b0e2897cb0a3febf1202b66d9d69b757ac0398e2c581d267de.
Its full570-leaf packet, complete static/source bindings and original predecessor
are re-received, not replaced by a hand-picked subset. Historical numeric
**Model** bytes are usable only after every one of347 raw model-array hashes
matches both the authenticated predecessor manifest and the actual new GPU
model hash map. This is an explicit full-model binding already enforced by the
measured allocation/runtime/receiver chain. Historical **Data**, old poses,
old contact tables and old EFC parameters must not supply GPU scalar recipes.
The complete predecessor prepared-inputs file remains anchored by the prior
receiver (SHA256
3d454f4d61e1af4667bc84acfac054724f0589c13964937455d8a49a32d83f61).

After that full binding, decode only four model inputs using their exact
canonical raw layouts: timestep, impratio_invsqrt, body_invweight0 and geom_bodyid.
Use the NEW before-solve Data for all eight dim3 contacts and32 own-contact
pyramid rows. Verify actual world/geom/body topology, unique contiguous row
addresses, row type/id and exact vel/Jqvel linkage. Each contact must carry the
frozen ContactType.CONSTRAINT bit1; an independent SENSOR bit is permitted,
while zero/sensor-only flags are refused. This closed arm requires
disableflags0, positive solref, positive friction/body inverse-weight sum and
solimp power2; refuse other arms rather than extend coverage after seeing a
result. Recompute D, aref, vel, pos, margin and frictionloss from the installed
frozen `_efc_contact_update`/`_efc_row` formulas. Preserve all32 captured values
and complete recipe-minus-captured leaves; report all six max-absolute residuals
without a pass tolerance.

Source arithmetic uses widened FP32 inputs and source-ordered Python float64
products/sums, not an emulation of Warp FP32 fusion, pow or loaded machine code.
The installed MuJoCo-Warp constraint.py RECORD/source hash is
b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53;
the existing full stage source audit is rehashed. Numeric constants are pinned
to plain bounded MuJoCo3.10.0 headers and their wheel RECORD:

- mjmodel.h (63289B):
  bcd51b20cb29b6aac7c8b9e1cf348f569b7fc2239f9226b844737c73511d90c8
- mjtype.h (27906B):
  ec580ce2a4ef0c1f6a3e68b3c5b5eeaf61d03413827453e2d0a87c3bd92d16e4

Separately retain complete unmasked nominal row-sum proxies H=J^T D J,
h=J^T D aref and c=0.5 aref^T D aref. Reduce original rows in ascending index
order with explicit float64 arithmetic, no BLAS/RMS/averaging. Partition every
active row exactly once into world-specific friction or ordered-foot-pair bins;
check own contact type/id/address linkage. Native inputs are the authenticated
native post-solve active rows. GPU inputs are the NEW measured J/D/aref with
before/after-solve byte equality; final states are descriptive histograms only.
Retain full20x20 matrices/vectors/scalars, both row counts and GPU-minus-native
aggregate differences. The four bin keys are grouping metadata, **not physical
point correspondences**: native has two points/foot and GPU four. Apply no state
mask and claim neither actual solver Hessian reconstruction nor response/cause
isolation. Native contact scalar parameters are unavailable, so no native
parameter-recipe reconstruction or per-point cross-backend comparison is made.

Reference only: frozen upstream
[MuJoCo native constraint source](https://raw.githubusercontent.com/google-deepmind/mujoco/3.10.0/src/engine/engine_core_constraint.c)
and [MuJoCo-Warp tagged constraint source](https://raw.githubusercontent.com/google-deepmind/mujoco_warp/v3.8.1/mujoco_warp/_src/constraint.py).
The installed RECORD-bound wheel source, not an assumption that the web tag has
identical bytes, is the execution arithmetic authority.

Retained output is one exclusive, fsynced, bounded256KiB canonical JSON on each
host, named `<execution-source8>-{mac,linux}-constraint-input.json` under
`artifacts/tools/ada-measured-boundary/`. Portable receiver requires clean exact
branch/HEAD/current module bytes, re-receives both complete input packets and
recomputes every output leaf, source/header/layout/model binding and false flag.
Fresh-process receipt and byte comparison across hosts are required; any
difference is reported, not normalized or hidden. Native execution is one CPU
user service `microduck-ada-constraint-input-<source8>.service`: Typeexec,
RemainAfterExityes, RuntimeMaxSec90, MemoryMax6G, CPUQuota200%, TasksMax64,
Nice10, LimitFSIZE16M, LimitCORE0, Restartno, KillModecontrol-group,
TimeoutStopSec10, exact repository/venv interpreter. Exactly five environment
entries: CUDA_VISIBLE_DEVICES empty and OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS1.
Verify exact machine/source/clean tree, inactive protected services in both
scopes and unchanged Dino inventory before and after. Retain invocation,
successful exit/MainPID0/empty group, elapsed time, caps, environment, result
hash and terminal GPU/protected-service evidence separately. No GPU allocation
is needed, and do not manipulate the existing GPU advisory lease.

Run all eleven retained CPU contract modules on both committed-source hosts
with the single real-physics test explicitly deselected. Compare complete
JUnit testcase multisets and failure/error/skip counts; native contract service
retains the previous60s/2G resource profile. Source/test preparation currently
passes88 new synthetic/header tests after adding explicit aggregate linkage and
checking the fixed floating-point reduction convention. Add independent review
and the full suite evidence before retained execution. Stop on any genuine
receiver/byte/provenance failure and diagnose read-only.

Independent Luna review found that the original synthetic fixture could enter
the recipe without the CONSTRAINT contact bit, while the frozen update kernel
would skip that contact. Actual measured contacts all carry bit1, but the
pre-execution guard was incomplete. Added the int32/eight-entry bit guard,
corrected the fixture, and added zero/sensor-only/wrong-dtype refusals plus the
valid combined-bit case. No runtime packet or acceptance gate was changed.
Complete internal wiring tests also exercise all570 new-stage leaves, the full
347-model binding,32 own-contact recipes and unlike native/GPU row counts with
only the outer authenticated-packet fixtures mocked. Poisoning historical
non-state/non-Model bytes has no effect; changing any new J/D/aref solver input
afterward is refused. An initial test-only bare-pose-name KeyError was corrected
to the actual `/data/` keys; all four new wiring cases then passed.

Final preparation suite: all eleven CPU-hidden contract modules passed753 tests,
the sole actual-physics test explicitly deselected (34.70s). Fresh independent
Luna review re-ran88 focused cases (3.12s) and found no remaining source/test
blocker after the two guards and fixture correction. Relative local link and
`git diff --check` passed. These are contract/source checks only; commit this
predeclaration before clean committed-source tests and retained execution.

Decision is fixed to
**measured-constraint-input-arithmetic-not-cause-isolation-or-admission**.
All solver, simulator, training, compiled-binary and physical qualification flags
remain false regardless of arithmetic residual size. A subsequent physics
control would need its own evidence-backed, tested predeclaration.

## Retained measured constraint-input arithmetic result

Executed clean committed source bd2815fb0d99640c531fa4a28be2919ea4b8961f after
fork push and hash-verified native fast-forward. Both committed-source suites
passed753 cases, no failure/error/skip, identical complete testcase multisets;
the sole actual-physics test was explicitly deselected (Mac24.64s, native5.82s).
The first native launch used the erroneous single token `not_actual_duck...`
instead of the intended pytest expression `not actual_duck...`: it selected no
tests, deselected all754, exited5, and ran no physics. Read-only journal/XML
diagnosis preceded retry. Its failed service/243B zero-case XML remain retained,
not reset/restarted/overwritten or counted as a successful suite. Corrected
retry used a separate `microduck-ada-response-contracts-bd2815fb-r1.service`
and fresh XML under the unchanged60s/2G CPU contract caps.

Failed-selector invocation7f5165f3608447a1be1dc98829475a2b elapsed0.456349s;
successful retry invocation066d0d19064d4243863fbadd2475521b elapsed6.387348s,
exit0/MainPID0/empty group. The separate selection-bound receipt pins both
terminal receipts and immutable helper bytes, checks exact full eleven-file
argv, binds native XML hashes to local copies, and verifies the negative
754-deselected summary versus the positive753-pass/1-deselect summary and full
Mac testcase inventory. Independent review requested this stronger binding;
an exit5/partial-command check alone would not prove the selection diagnosis.
The arithmetic packet itself does not depend on the failed selection attempt.

Then `microduck-ada-constraint-input-bd2815fb.service`, invocation
8f0035c5fcd64e6d88fce8106d064e95, finished in **1.271054s**, successful exit0/
MainPID0/empty control group under90s/6G and the exact five-entry hidden-CUDA,
single-thread environment. This service performed pure packet/source arithmetic
only. No physics/runtime/device allocation, collision, constraint construction,
solver, integration or optimizer call was made. Native terminal occupancy was
961MiB used/15232MiB free,44C/0%; sole foreign Dino PID1592 stayed at946MiB and
both protected services stayed inactive in user and system scopes. No advisory
GPU lease change was needed.

Both96,096B output JSONs are **byte identical**. Fresh-process full reception
passed independently on Mac and native; Warp/Torch/MuJoCo/MuJoCo-Warp remained
absent from imported modules. All347 numeric Model bindings, the new complete
five-by114 Data bank,32 own-address contact recipes, four native and four GPU
aggregate bins, full result leaves, numeric headers/RECORD and source audit
recomputed. Independent Luna re-received both packets and checked the separate
selection receipt/local XMLs, finding no remaining scoped blocker. That review
did not run SSH/services; actual native terminal/remote XML binding is the
owner's separate read-only evidence.

| Complete captured GPU contact-row parameter | Maximum absolute float64 recipe minus captured FP32 |
| --- | ---: |
| D | 1.2963849310709818e-7 |
| aref | 3.127977976635776e-8 |
| vel | 0 |
| pos | 0 |
| margin | 0 |
| frictionloss | 0 |

These are source-arithmetic residuals, not a tolerance gate or compiled-kernel
identity proof. Every captured value and residual remains in the packet. The
closed positive-ref/power2 recipe reproduces the new measured contact-row inputs
closely under the predeclared widened-arithmetic convention; it does not
reconstruct the missing native contact scalar parameters or contact Jacobian.

Selected entries of the full, unmasked aggregate nominal proxy differences:

| Ordered world1 floor/foot bin | Native/GPU original row counts | GPU minus native H[2,2] | GPU minus native h[2] |
| --- | ---: | ---: | ---: |
| geom0/29 | 8/16 | 3.0183082526704785 | -0.1808022540619001 |
| geom0/79 | 8/16 | 3.018975883554497 | -0.18073954588214802 |

Both world-specific motor-friction bins have14/14 rows and zero differences
at these selected base-z entries. Full20x20 matrices,20-component vectors,
constants and final-state histograms remain retained. This exposes substantial
differences in the unlike contact aggregate terms despite the small same-bank
CPU/GPU solver differences. It is evidence to investigate upstream contact
construction, not to infer exclusive cause: no point matching, row averaging,
state masking, actual Newton matrix or physical-equivalence claim was made.

Exact retained files under `artifacts/tools/ada-measured-boundary/`, on both
hosts; no old evidence or failed attempt was overwritten:

- bd2815fb-mac-constraint-input.json and bd2815fb-linux-constraint-input.json
  (96096B each):
  1d7ded608c976ddc599dfab31cfd4eebdda58ccea3d338ac81bfbbe79d824e56
- bd2815fb-mac-tests.xml (102371B):
  43aa106e1d0d9bdf790ab1a34e512e30523fc2ed793a6420898e8c2e9706f9dd
- bd2815fb-linux-tests.xml, failed selection (243B):
  4501ab09284e9484b5c68703fbb4436e6e82d954fe9f6431767a5354d8ddd4e3
- bd2815fb-linux-r1-tests.xml (102376B):
  bf0947451711fe9689ce761092878dd23d65998258f26b205b6b3a3806d8256b
- bd2815fb-contract-selection-failure-service.json (4177B):
  a811f1cbf8df1c3e35def8035a88f1b633463c71416a448b4a547675d88a3649
- bd2815fb-contract-r1-service.json (4885B):
  6b2df283d7483ac407cb51d60f7ef1af5160a3c422c496d838964c192a98b22e
- bd2815fb-contract-selection-bound.json (200553B):
  87b99a18b32c49618356d440d39b0fcc43953b79a7403c4dfa3aaba61b1ba333
- bd2815fb-constraint-input-service.json (3950B):
  22f51833b215a7562bfb0d7675e82b2ed9e5191388668241778d0666eeaf094f
- retain_constraint_input_service.py (5967B):
  dfcb988081a92974b10b745cb9d412a04981a93d7a81cedab75a838829bd27ec
- retain_constraint_contract_selection.py (5131B):
  49cb7e5671407412a63301b786056592225d43babf2485c15ed93d17caea165e
- bd2815fb.bundle, verified incremental source bundle fromee05ee65 (17857B):
  512f3d6f4c5a6aea41f14c6c60f6508626398420797cd4dcd045f0341c0d3d2e

Decision remains
**measured-constraint-input-arithmetic-not-cause-isolation-or-admission**;
all original qualification/training/physical flags remain false. This bounded
diagnostic is complete; the authorized until07:00 campaign continues. Next
useful control is an evidence-backed predeclaration to recover full native
contact-construction inputs or compare a separately controlled upstream arm,
not PPO or hardware motion. Reserve the existing06:50 launch cutoff.

Closeout documentation check matched all twelve artifact sizes/hashes against
both hosts, the existing relative Markdown link and whitespace. Native user
running-service inventory contains no Duck service; unrelated CPU/parser,
grounding and media services remain preserved. Source/runtime code has not
changed after the753-case execution revision. Retain this documentation commit
separately from the clean immutable execution source; do not reinterpret a
future wrong-HEAD receipt refusal as a reason to loosen its source contract.

## Current native full-forward input capture predeclaration (before execution)

The preceding GPU-only arithmetic recipe does not reconstruct the historical
native Model, contact parameters or native solver KBIP inputs. The next bounded
control captures **current native** parameters at the exact seven retained
float32 states, widened to float64 in the same way as the original native arm.
It cannot retroactively authenticate an unavailable historical complete Model
or establish that public output equality makes all inputs physically equivalent.

Execution source will be the clean feature-branch commit containing
`ada_native_constraint_capture.py`, its tests, this predeclaration and
`ada_native_constraint_schema.json`. The frozen public class getter inventory
was obtained by importing MuJoCo without constructing a Model/Data or invoking
physics:627 entries (474 arrays,3 byte strings,106 integers,10 floats,31 methods,
3 nested objects); schema42029B SHA256
12ee7d544b19c4d4c6fa8308e39c5611f317ba17693c056d337348c6292626ff.
Methods are not invoked, stat/vis are explicitly excluded from reflection, and
the **complete current serialized MJB** is retained independently. Schema names,
types, scalar/exclusion inventories and both-world nonmotor byte equality are
checked; this is not a claim that introspection alone proves historical identity.

Only the exact Linux machine0c79e415429b4933a400159bfa79a34d, checkout
`/home/converge/work/microduck_rl-athletics-obstacle-curriculum`, branch
`feat/athletics-obstacle-curriculum`, frozen source-root Python3.12.12 venv,
unchanged five package versions, BAM1.0.1 and driver595.91.07 are in scope.
The GPU remains shared with Grounding DINO PID1592; this arm exposes no CUDA
device and does not enter or replace the existing advisory GPU lock.
Both system/user scopes of the two protected mission services must stay inactive.
No changes to100.98, FilmBrain, the XML/assets, installed libraries or gates.

One retained **CPU-only user service** will use a unique source-prefix output,
`RuntimeMaxSec=90`, `MemoryMax=6G`, `CPUQuota=200%`, `TasksMax=64`, `Nice=10`,
`LimitFSIZE=64M`, `LimitCORE=0`, `Restart=no`, `RemainAfterExit=yes`,
`KillMode=control-group`, `TimeoutStopSec=10`. Its five literal environment
assignments are `CUDA_VISIBLE_DEVICES=` and OMP/MKL/OPENBLAS/NUMEXPR threads1.
Run only after exact committed source synchronization, focused pure contracts,
independent review and a fresh host/workload check, before22:50UTC.

Construction is two `build_entity().compile()` calls with a static selected
plant/assets binding; **never** describe/reference/place_on_floor. Only recorded
motor frictionloss/damping fields are installed, from the authenticated original
motor bank, and seven saved states are restored. Exactly two public `mj_forward`
calls perform native collision, constraint construction and solving; **zero
integration** does not mean zero solver work. A fresh process installs
fail-closed Warp concrete-array, launch/load/Module.load guards before dynamic
plant imports, and rejects Entity/mjlab BAM runtime initialize/compute paths.
Static actuator/BAM parameter constructors and their imports are explicitly
allowed. No Warp runtime arrays, kernels, BAM torque proposal, optimizer,
reset/placement, GPU allocation or policy execution are permitted. All eight
native global callbacks must be absent and no Warp ModuleExec may be held.

Retain all public numeric top-level Model/Option fields, native full MJB
before/after byte fences, all contact fields (including current post-constraint
mu/H), all public EFC construction/solver vectors selected in the protocol,
dense-J metadata, KBIP and diagA (not a fictional diagApprox), all4000 solver
statistic slots, counters, warnings, seven states and all20 original output
fields. Contact/EFC IDs are own-run slots only, not cross-backend point IDs.
`mj_contactForce` is called only for included contacts (bounded≤256 total);
uninstantiated contacts retain explicitly labelled zero-force markers.
Actual contacts/rows may differ from historical0/4 and14/30; bounded structural
differences, warnings, nonfinite computed values and exact-byte disagreements
are retained as diagnostic failure signals, not tolerated or discarded.
Model infinity sentinels remain raw; NaNs/unknown types, overflow, model-fence
drift, hidden runtime calls or authority/resource violations stop the arm.

Each world is capped at128 contacts and512 EFC rows. Public model arrays include
native mesh BVHs absent from the old Warp array inventory: the authenticated
old model input alone is14808415B before NPZ headers. Therefore this arm uses
one compressed numeric NPZ (raw arrays<144MiB, compressed file<64MiB), each
MJB<64MiB, report≤256KiB, and a192MiB ceiling for the three large retained
leaves. Reception bounds complete ZIP expanded byte counts before decoding;
compression changes storage only, never dtype, order or raw numeric hashes.
Mac available capacity was76516728KiB before this source revision. Writes are
exclusive/fsynced and never overwrite historical or failed evidence. Source,
schema, native four headers/Linux extension/shared library and wheel RECORD are
bound by bytes. Installed-file provenance is **not** loaded-machine-code identity.
The portable receiver creates no native Model/Data or physics calls: it binds
all raw bytes/layouts, fixed inventories, model option/scalar exclusions, own
contact/EFC linkage, current versus historical fields/active rows/counters and
failure signals. Retain receipt/service/evidence hashes and commit the closeout.
All existing qualification/training/learned-skill/physical flags remain false;
no PPO, Locked/H2 promotion, video or physical motion follows this capture alone.

### Native size-guard stop and exact shared-storage revision

Execution revision a7b2449d5b2616ed45688449d466830778f1f928 is preserved in the
fork and native history. Its12 CPU suites passed830 cases on each host (Mac
38.09s, native6.55s; one actual-physics test deselected); the complete830
classname/name multisets agree with zero errors, failures or skips. Native
contracts unit invocation2230f58482fa4cc4b1f8d288b87aae5b finished successfully,
PID0 and empty cgroup, with the declared60s/2GiB/16MiB limits.

The first capture unit, invocation1b8e0faa3c8c47be8ebaa09d7b5824d6, exited1
after5.676774s under its original90s/6GiB/64MiB file limit. Exact error:
`ValueError: bounded current MJB: 80760771`.
Its journal records compiled world0 MJB80760771B and public numeric
Model/Option80759191B. The guard is immediately before MJB save, MjData
allocation and the first forward call: no new forward, contact-force decoding,
integration or GPU simulation occurred. There is no partial JSON/NPZ/MJB at
the output prefix. Preserve the failed unit, source and journal; do not reset,
restart or replace it. This is a diagnosed file-budget shortfall, not evidence
of a failed physics solve. Warp's CUDA-error100 line is expected with literal
empty CUDA_VISIBLE_DEVICES; the sole NVIDIA process remained DINO1592/946MiB.

Read-only diagnosis retained the whole terminal service receipt:
`artifacts/tools/ada-measured-boundary/a7b2449d-size-failure-service.json`
(9269B), SHA256
06b07bb4d217393d80ea63010f1184c21e55e6a3463b2eac01a9b70aa8b982cf.
Its helper binds exact source/branch/machine, argv, environment, all service
limits, timestamps, failed exit, empty cgroup, absent partial output paths and
the four inactive protected-service scopes. Source/test evidence stays separate
from service evidence, and neither is training admission.

The next **new source/unit/output prefix** changes storage only. Keep two fresh
compiled native models, two public forwards and all prior call/state/physics
protocols unchanged. Before packing, compare every common public model array
by dtype, shape and exact bytes. Store those475 identical fields once, with a
fixed source-bound shared-path inventory/hash; retain both actual world-specific
frictionloss/damping fields independently (authenticated old frictionloss
differs by world, damping is identical). The portable receiver reconstructs
only the logical byte namespace for checking; it does not create a Model,
recompute a parameter or invent a physical match. Keep both complete MJBs,
which remain world-specific, with independent before/after byte fences/hashes.

The revised compressed numeric bank has<96MiB stored raw array bytes,<64MiB
serialized bytes,≤2048 ZIP entries, each expanded entry<96MiB and aggregate
ZIP expansion<97MiB before decode. Each complete MJB is<128MiB, so the three
large-leaf budget is<320MiB. Increase **only the new capture unit's**
LimitFSIZE to128MiB; retain RuntimeMax90s, MemoryMax6GiB and all other limits,
five literal CPU environment values, protected services and workload checks.
Current native free disk250757680KiB and Mac free76516728KiB comfortably
exceed this bounded retained evidence budget. No installed library, plant,
acceptance threshold, GPU lease or old artifact is changed. Re-test/review,
commit/push and exact-clean native synchronization precede the second attempt.

### Seven-slot native warning-layout diagnosis and bounded correction

Shared-storage execution revision d88ebba68cb6e4005396c4ad2d8bd6e93a1f7090
passed837 pure cases on both hosts (Mac57.92s, native6.66s; one actual-physics
test deselected), with identical837 classname/name multisets and zero
failures/errors/skips. Native contracts invocation and terminal service are
retained independently in `d88ebba6-contracts-service.json` (4985B), SHA256
387fc45f542d4b88345dd325103f19eb4f76e8420f406ef8ebbe743008cb3550.
Mac JUnit115267B SHA256
ffeef2c4d7f544d53254ebdef9e72ecf104d5048a82062e4c9b7c3c859fb398a;
native JUnit115272B SHA256
c970e44ecd771cb07bdb33f2c8b6548bca1f383502705b62d8a913c011f0bc56.

The second capture invocation cc680ea7ef14491bb9879c19d11ab707 exited1
after6.260802s, with the declared90s/6GiB/128MiB limits, PID0 and empty
cgroup. Exact error: `ValueError: all solver/warning counter layouts`.
Both world compilations report MJB80760771B and public numeric80759191B.
Unlike the first size-guard stop, source control flow shows this stop occurs
after both public native forwards and complete model fences, while checking
the summary, **before any output leaves are written**. Preserve this failed
source/unit/journal separately; no partial JSON/NPZ/MJB exists at its prefix.

Read-only diagnosis inspected the installed frozen three headers and imported
only the warning enum (no Model/Data/physics). MuJoCo3.10.0 defines exactly
seven warning entries, with contiguous IDs0..6 and mjNWARNING=7. The code and
synthetic fixture incorrectly assumed eight. Header solver dimensions remain
mjNISLAND=20 and mjNSOLVER=200. This is a diagnosed checker layout bug; it is
not a NaN/fall/physics failure or proof that the underlying solve is accepted.
Receipt `d88ebba6-layout-failure-service.json` (127800B), SHA256
04b3f9a45b1db5e72df89684e371a529bafbd73b2695a1b055b9093710bc50e5,
retains the full terminal service, absent output paths, three header bytes/
hashes, enum, four inactive protected scopes and unchanged DINO1592/946MiB.
The read-only remote search command encountered unavailable rg and exited127;
the separate header/enum receipt succeeded without installing anything.

The next new immutable source/unit/output prefix corrects only the warning
array schema to7, checks the exact ordered warning declaration and20/200
dimensions against the already pinned installed headers **before any Model**,
and checks the Python enum before compilation/physics. Wrong layouts now name
the actual failing field, shape and dtype. Add installed-header tests without
Model/physics, reject header drift and reject synthetic eight-slot arrays;
do not pad/truncate actual data. All other states, plant, callbacks, two-forward
protocol, own-row linkage, exact comparisons, storage caps,90s/6GiB/128MiB
user-service limits and false qualification flags remain unchanged. Re-test,
independent review, commit/push and clean exact native synchronization precede
the third attempt. Preserve both earlier failed units and their evidence.

### Successful current native capture and complete portable reception

Execution revision f8540785a4ff5ee96d8186962c1934a9a15c0bb7 passed843 pure
cases on both hosts (Mac17.96s, native6.72s; the same actual-physics case
deselected). All843 classname/name multisets match with zero failures/errors/
skips. Independent source review passed90 focused cases in2.03s and found no
blocker. Native contracts receipt4984B SHA256
b66c59fdc3482cdf152c54fba7edc52e310b484c14d8d697b441a3a94d6bbd71;
Mac JUnit116191B SHA256
072074d6dd71bdd67b3174dcfc021fce877e1db3c106058d0fad7b76e24f5226;
native JUnit116196B SHA256
8d52ccc99763192451d9fcdf13533f4a62527b438f58c648a843de5f0a6ed2ce.

The third unique capture unit completed with exit0 after10.869572s,
invocation9768630dcbe94ffc92e6ea3d52327ca4, PID0 and empty cgroup. Actual
service receipt8173B SHA256
88b7ccce8a79b326cc8b032224bcf4c4d0ac14fbab5c5ad099d76408e521607e
binds the exact argv, clean execution source,90s/6GiB/128MiB limits, literal
CPU-only environment, full journal and four inactive protected scopes.
No Warp array/kernel/ModuleExec or Entity/BAM runtime call occurred. Exactly
two native forwards and four included-contact force decodes ran; integration
and optimizer counts remain0. Static BAM parameter construction was allowed.
DINO remained PID1592/946MiB; final GPU45C,0%,961MiB used. No service restart,
lease replacement,100.98/FilmBrain change or physical motion occurred.

Both current model numeric/scalar fences and complete MJB before/after byte
fences match. All20 output fields and all24 historical active-bank leaves match
by dtype, shape and raw bytes, as do the original six counter fields. Seven
restored states are unchanged, both warning sums0 and all computed arrays
finite. Native ncon0/4,nefc14/30,nf14/14,niter1/4; dense nJ280/600,nisland1/1.
These results reproduce the historical **outputs**, not an unavailable
historical complete Model, compiled-device identity, common physical contact
points or training admission.

Complete leaves retained under `artifacts/tools/ada-measured-boundary/` on
both hosts (prefix `f8540785-linux-native-constraint`):

| Leaf | Bytes | SHA256 |
| --- | ---: | --- |
| .json | 134653 | 52d76742ab534af2434d4dd698124f485c15a3e2ff3ee6e7171e93d5485961f4 |
| .npz | 33205295 | 35792c896118d7ebd62f0e5356e116c520ec004b0c254abd184c85c565afe950 |
| -world0.mjb | 80760771 | 1522cace1c5582f1364145396791683d9c4127be880d3df4bf42ca63d9de9a58 |
| -world1.mjb | 80760771 | a95db5076919780704e7b3d4fd0f9566c6974f5891af7e34dbdb5f512c453be7 |

Stored619 numeric leaves total81125319B; logical1094-leaf two-world namespace
161884190B includes shared aliases, not another physical copy. Distinct MJBs
remain independent. Fresh portable receivers on both hosts imported no
MuJoCo/Warp/Torch runtime and made no native/GPU call, verified every complete
leaf, and produced identical1644B receipts, SHA256
91007ebdd68bd3175781cf635fce3cf68aed4c37d1a5ffaa3b67731788f830a1.
The ignored exclusive-write receiver helper SHA256 is
35a4ffa45a794e113cdc6bab33ed8129fa886a80eedf0cd9924e1c2303320da9;
service helper v3 SHA256
4076adc38a2653529b4621f4f3d31848b2b6e88c4668fac6edaa5af9b8dc73a1.
An early Mac receive, attempted while SCP was still writing the NPZ, correctly
refused `plain bounded full capture bank`; it wrote no receipt. Complete
post-transfer reception passed; no contract was loosened to bypass that refusal.

The next bounded step is a separately tested, source-bound **pure arithmetic**
comparison using this complete current capture and the already measured GPU
bank. Keep both backends' own contact slots/manifolds; native4 versus Warp8
does not become point correspondence or a row-normalization prescription.
No further Model/forward/solver/GPU call or PPO follows the capture alone.

## Current native row-recipe arithmetic predeclaration

The next immutable source contains `ada_native_constraint_recipe.py` and its
pure tests. Use only the complete current f854 capture (report SHA above), the
exact authenticated original eleven-file packet and the measured daeb GPU
packet. No Model/Data, physics, Warp runtime, solver or device is allocated.
This control is separate from the earlier GPU-only recipe; preserve that
earlier source/result and its native-input-unavailable limitation.

The native scalar recipe is independently checked against the public
[MuJoCo3.10.0 constraint source](https://raw.githubusercontent.com/google-deepmind/mujoco/28009f9105cd92784b7b0b30c0605a5e29107a77/src/engine/engine_core_constraint.c).
Git tag3.10.0 resolved to commit28009f9105cd92784b7b0b30c0605a5e29107a77.
Retain the exact102904B source as
`mujoco-28009f91-engine-core-constraint.c`, SHA256
71bcfcc6e3518846ee3b65e6835b14565aacf4027ea3c522b26498ea9ccc58cd.
This is a source-level arithmetic reference, not proof that a particular
loaded wheel binary was compiled from it or that Python emulates native SIMD
or Warp FP32 fusion/rounding. Existing numeric headers/installed Warp sources
remain byte/RECORD-bound independently.

Cover all44 native EFC rows exactly once:28 DOF-friction rows with their own
dof IDs, captured Model inverse weights/reference/impedance/frictionloss and
zero position/margin;16 rigid isotropic dim3 pyramidal contact rows linked to
their own four included contact slots. Derive velocity by ordered J*qvel over
the captured J and unchanged saved qvel. Native contact inputs are captured
geometry/body mapping, inverse weights, friction, reference, impedance,
distance and inclusion margin. Native R/D/KBIP/aref/final solver state are
comparison outputs, **never recipe inputs**. Keep direct negative-reference
motor-friction and positive contact-reference branches, REFSAFE, native
impedance derivative, contact regularization and post-impedance adjusted diagA
distinct. Unsupported row types/branches/layouts fail closed.

Retain numeric differences and exact FP64 byte disagreements for every row;
these are descriptions, not new acceptance tolerances. Native data is a
post-forward capture, not a newly replayed before-solve stage. Keep measured
GPU32 contact rows/eight contact slots separate, with its existing source
recipe and before/after construction-input byte fences. Do not pair physical
points, normalize row counts, mask active states or reconstruct an actual
solver Hessian. Eight selected current native Model arrays have explicit
cast-to-measured-dtype byte descriptors after all347 historical Model arrays
are bound to the actual GPU bank; those descriptors do not prove full models
or physical contact inputs equal. All previous qualification flags stay false.

Before arithmetic execution: focused/full CPU tests, independent review,
commit/push and exact-clean native synchronization. Both hosts use a fresh
source-prefix JSON output≤256KiB and complete deterministic recomputation.
The native contracts unit is60s/2GiB/16MiB; the pure arithmetic unit is
120s/2GiB/16MiB. Both are retained CPU-only user units with Type=exec,
RemainAfterExit=yes, CPUQuota200%, TasksMax64, Nice10, LimitCORE0, Restart=no,
KillMode=control-group and TimeoutStop10s. Five literal CUDA-empty/thread1
environment assignments, inactive protected services, unchanged DINO, no GPU
lease entry and the22:50UTC launch cutoff remain mandatory. No physical
motion, policy training, installed-library/plant/gate change or raw perception.

### Current native recipe execution and independent closeout

Execution source ec81e2bb2ffd25f6e847036cb0deb1ab1836478d passed the same891
pure cases on Mac20.92s/native6.78s; one explicitly actual-physics test was
deselected. Classname/name multisets match exactly, with zero failures/errors/
skips. Independent source review passed48 focused cases in5.01s; independent
post-execution reception recomputed both reports without importing a simulator
runtime and found no blocker. JUnit leaves under the retained tools directory:

| Leaf | Bytes | SHA256 |
| --- | ---: | --- |
| ec81e2bb-mac-tests.xml | 123432 | bf2bbcffeef62a4a7e7635e1ed3553a9ef687845061b1fab3f6217a5fa29e80e |
| ec81e2bb-linux-tests.xml | 123437 | 42943f11ca0ee47d59d6daa167cf02e0495752d1aa7c10cd042c58b1e406a614 |
| ec81e2bb-contracts-service.json | 5129 | 1109e45daa6ac8fdbeff0564d4f43275f00bd01115f644c237176585e83ea777 |
| ec81e2bb-arithmetic-service.json | 4837 | b943ccfade42949cab05cf91f9fcb0a10109752957673d184e2272890ed3eac6 |

The native arithmetic unit completed successfully in2.682441s, invocation
ffba8650e3e248479ad823b90a098fc4, exit0/PID0/empty cgroup. The receipt binds
its exact source, argv,120s/2GiB limits, all five environment assignments,
unchanged DINO1592/946MiB and four inactive protected service scopes. The
read-only receipt helper `retain_native_recipe_service.py` SHA256
cb04e9e48d6e78970e555c5f488a726bd36e8c0192bf2707b4351a2e861c2a00
is retained with the results; no service was restarted or foreign workload
changed. The two host reports `ec81e2bb-{mac,linux}-native-recipe.json` are
byte-identical79729B, SHA256
674c6d7779bad3daef7c54aa761745d26df03cf44051753811f3dfcaf05490ae.
All above leaves and the helper are retained on both hosts.

All44 native rows have exact FP64 byte agreement for each of R,D,diagA,aref,
vel,pos,margin,frictionloss,K,B,I,P; every numerical residual and every byte
disagreement count is0. The16 selected Model precision descriptors all report
cast-to-measured-GPU-dtype byte agreement. Their largest native-minus-widened
GPU differences include body_invweight0 5.20583537308994e-05 (including its
rotational component) and dof_invweight0 2.953653176973603e-05. Those are
descriptive rounding brackets, not tolerances or complete Model identity.

The report independently retains each backend's different contact inputs:
native four included contacts/two per foot versus measured GPU eight/four
per foot, each with its own distances, slots and row addresses. Recipe
agreement establishes consistency of the captured native construction
arithmetic; it does not make these unlike manifolds corresponding physical
points, isolate a single cause, prove loaded machine-code identity, or open
solver/simulator/training/physical admission. All qualification flags remain
false. A further row-family intervention, if developed, must be separately
predeclared, preserve the complete measured surrounding context, explicitly
bind every changed/held field and index translation, and remain a hybrid
diagnostic rather than a native end-to-end replay.

## Pure native row-family hybrid packet predeclaration

The next immutable source adds `ada_native_row_family_plan.py` and synthetic
tests. This is **packet preparation only**; its CLI does not import a simulator,
allocate Model/Data, execute physics, or authorize a subsequent solver. Require
the exact ec81 native arithmetic report SHA256674c6d7779bad3daef7c54aa761745d26df03cf44051753811f3dfcaf05490ae
and independently recompute it before encoding. Bind the same complete native
f854 capture, original eleven-file packet, measured daeb GPU bank and pinned C
reference. Preserve all347 measured GPU Model hashes and the complete184
portable static manifest through references rather than duplicate storage.

Encode a complete114-field Data bank with exactly31 declared overrides and83
byte-held measured before-solve leaves. Override ne,nf,nl,nefc and global nacon;
native per-world ncon0/4 remains explicitly **descriptive**, because frozen
Warp Data has no ncon array. Override dense EFC J,D,aref,frictionloss,type,id,
pos,margin,vel and Jqvel (the latter from native efc_vel). Override every field
of the16-field Warp Contact table. Explicit float64-to-float32 casts are not
an emulation of an FP32 constructor. Unsupported shape/type/finite/capacity,
row kind, DOF ID, contact ID, address or rigid-foot layout fails closed.

Concatenate native contacts in their own world/slot order, not a GPU contact
matching: world0 supplies0 slots, world1 supplies4 slots. Each native slot owns
four contiguous pyramid rows14..29; global ids0..3 are linked consistently in
type/id/address/dim/worldid. Clear all overridden EFC inactive tails to0.
Inactive contact geometry/flex/vert/world/address/collisionid indices are-1;
other inactive contact fields are0. Native supplies twelve compatible contact
fields; own global worldid and four-address vectors are explicit translations.
Two fields are explicitly synthetic: active type1=CONSTRAINT, and
geomcollisionid=-1 throughout. No collision provenance is invented. This bank
must not enter collision, constraint construction, sensors, force decoders,
ordinary forward, integration or optimizers. It is not a complete native
forward-context or physically point-matched replay.

Hold all83 surrounding Data leaves byte-exact, including qM/factors/smooth
forces and acceleration, warmstart, saved states, dense empty sparse metadata,
and the measured GPU before-solve solver outputs. Never transplant native
post-solve qacc, force, state, b or solver_niter. Retain every114 leaf's layout,
before/encoded hash and changed/held policy, all own contact links and an
exclusive full raw NPZ. A portable receiver independently regenerates every
leaf and descriptor and rejects omitted, added, duplicated, changed or
scope-forged packets. The sparse/elliptic or other solver read-set has **not**
been proved by packet preparation; actual_solver_read_set_proved and
subsequent_solver_execution_authorized remain false. A runtime intervention
would need a separate reviewed read-set/branch analysis and exact-source
predeclaration, fresh CPU cache, complete actual bindings and one public solve.

Before real packet encoding, run focused/full CPU synthetic tests and
independent source review, commit/push and synchronize the exact-clean native
branch. Initial uncommitted tests caught an incorrectly proposed ncon target;
inspection of the authenticated canonical layout established that it does not
exist, so the proposal was corrected rather than fabricating an extra field.
An additional test initially did not damage the one-element nacon shape; the
fixture was corrected to actually change that shape. No real packet, runtime
or service was executed during either failing test attempt.

Owner focused suite passed212 cases in8.55s; the complete14-file pure suite
passed964 cases with one actual-physics case deselected in82.07s. Independent
review passed73 focused cases in7.54s and found no packet-layer blocker.
The reviewer checked frozen solver/type/constraint sources: subsequent runtime
scope must be dense/Newton/pyramidal, expected friction/pyramid EFC kinds only,
and authentic empty dense sparse metadata. Contact.type and geomcollisionid
are not read by the frozen solver Python source; elliptic paths consume other
contact fields, so they cannot be silently admitted. These source observations
do not prove compiled-code behavior or authorize a solver in this packet arm.

Both hosts use fresh source-prefix JSON≤256KiB plus complete NPZ<2MiB in the
retained tools directory. Native contracts use60s/2GiB/16MiB limits; pure
encoding uses120s/2GiB/16MiB. All prior Type=exec/RemainAfterExit=yes,
CPUQuota200%/TasksMax64/Nice10/LimitCORE0/Restart=no/KillMode=control-group/
TimeoutStop10s limits, five literal CUDA-empty/thread1 assignments, protected
services/DINO preservation and22:50UTC launch cutoff apply. No new native,
Warp, GPU, optimizer or integration call occurs. All admission flags stayfalse.

### Complete native row-family packet execution

Execution source c9b37b5496fc4b809c65fb832263c78f3f17bbdd was committed,
pushed and clean-fast-forwarded to native before encoding. Its committed
14-file contract suite passed964 cases on both hosts, with the same one
actual-physics test deselected. Complete classname/name multisets match and
there are zero failures/errors/skips. Mac JUnit elapsed69.782s; native pytest
7.31s, unit7.902789s. Exact retained test/service leaves:

| Leaf | Bytes | SHA256 |
| --- | ---: | --- |
| c9b37b54-mac-tests.xml | 133864 | f4e816eb593f736303b4e7290ca04b39e4dfacbeac6fd10c57cb93ba200f9bff |
| c9b37b54-linux-tests.xml | 133869 | 8c4c6da725bf34c9fe60b4c884a55a98ef67ce22593ff8d522171546e046c0b9 |
| c9b37b54-contracts-service.json | 5337 | 521e05893097ef30828ef118232ab155ecd36661093680669e87605d61d7f207 |
| c9b37b54-packet-service.json | 5285 | 4abfec7f0b63ad67407c586f3e4eabf3556c6584c4c3f804738a1f3fe1e20f73 |

The120s/2GiB native pure encoding unit completed successfully in4.917796s,
invocationd690f4ae36614f85a4ac73c58f776063, exit0/PID0/empty cgroup. Its
receipt binds source/argv/resource/environment/journal and unchanged
DINO1592/946MiB plus four inactive protected service scopes. GPU43C/0%,
961MiB used/15232MiB free; no Duck GPU process, lease change, service restart,
100.98/FilmBrain change or physics call. The read-only service helper
`retain_native_family_service.py` SHA256
e02faa15707e4da0f99eae339e442c2d5e2feba046edb6cc5d77497ee22e4ba8
is retained with these receipts.

Both host packets completely recompute all114 leaves, layouts and descriptors.
They agree exactly except the explicit payload **basename**. Both NPZs are
byte-identical35838B, SHA256
9a885cd219762de5345139438cb7f3b7e9581bbcbecefaa3f386bb9272023285.
Mac report `c9b37b54-mac-native-family.json`84848B SHA256
547173200dc303189fc63c9658177bbe8b49b4f393e4e966ab8a474d9c6c1998;
native `c9b37b54-linux-native-family.json`84850B SHA256
585d728f420cc1ef7bc4a9bcf3ab9d9b3682cfeb8c8f96d97b4d0df26b6bfcb2.
All31 declared overrides are materialized, but only23 actually differ in
bytes from the authentic GPU before-solve bank; eight declared overrides
happen to be byte-equal. Every one of the83 held leaves stays byte-exact.
Counts remain native_ncon0/4 (description only), ne/nl0/0,nf14/14,nefc14/30,
and actual global nacon4. Four own linked rigid contact slots cover native
rows14..29; no GPU physical point was paired or discarded by a matching rule.

Fresh portable receivers on both hosts recomputed the same native packet
without importing a simulator runtime or making physics/device calls. Their
identical790B receipts `c9b37b54-{mac,linux}-family-reception.json` have SHA256
576154bed0ed49ae2722f5be4e2721e9334484f26a4772a7051d008f5ffe6007.
Receiver helper `receive_native_family.py` SHA256
9bd5af086f67d69045663c8316cc22fce8a825db4d0507cb45f67ab2ebbb10ed
and all reports/NPZs/JUnits/service receipts are durable on both hosts.
Independent post-execution review recomputed both packets and verified all
114 leaves, both964-case inventories, terminal receipts and helper hashes;
it found no blocker for this packet-only interpretation.
This closes only pure packet preparation. No solver, native phase replay,
machine-code read-set, single-cause, simulator/training or physical admission
is established. The next meaningful bounded experiment is a separately
reviewed **one-call CPU solver-only joint row-family intervention**, holding
the authentic GPU mass/force/warmstart context and all Model/statics fixed;
do not claim it as a full native solver replay or launch PPO from this packet.

### Predeclared native-family CPU solver-only hybrid control

This separately authorized arm is one fresh CPU Warp **public solver call**
using the complete c9b37b54 native-family packet. It is a joint31-field input
intervention, not a single-variable ablation or end-to-end native replay.
Hold all83 measured GPU surrounding Data fields, all347 Model arrays and
all184 statics byte-exact. Native F64 J/D/aref/contact fields have already been
explicitly cast to F32; the native capture is post-forward, not a recovered
native before-solve phase. Never transplant native final outputs, change the
plant/libraries/gates, pair unlike contact manifolds, or imply FP32 constructor
emulation. Synthetic contact.type and geomcollisionid may go to this guarded
solver path only, never collision/construction/sensors/force decoding.

Execution must use exact-clean committed/pushed/synchronized feature source,
unchanged frozen packages and audited sources, CUDA_VISIBLE_DEVICES empty,
CPU-only enumerated Warp devices, uninitialized Torch CUDA, a fresh private
CPU cache and a unique retained user unit. The actual pre/post branch guard
requires dense/Newton/pyramidal with zero native disable/enable flags and
noslip iterations, only friction/pyramid active EFC kinds, authentic empty
dense sparse metadata, ne/nl0/0,nf14/14,nefc14/30,nacon4 and fixed capacities.
This is a Python source-path guard, **not** compiled machine-code read-set
proof. Existing frozen source review finds contact dereferences in the
elliptic branch; sparse/elliptic and other regimes are excluded.

Allow one put_model allocation-native kinematics call and the existing two
Model expansion launches; no fixture kinematics, native forward, collision,
constraint construction, mass factorization, sensor, force decoder, ordinary
forward, integration or optimizer call. Restore all114 actual Data arrays
before solve; retain full114 before and full114 after, finite layouts/bytes,
pointer/Model/static fences, and require only the six declared solver outputs
to change. Counters must remain within the unchanged iteration cap.

Immutable input anchors are c9b37b54-linux-native-family JSON
585d728f420cc1ef7bc4a9bcf3ab9d9b3682cfeb8c8f96d97b4d0df26b6bfcb2
and NPZ9a885cd219762de5345139438cb7f3b7e9581bbcbecefaa3f386bb9272023285;
prior complete same-context CPU solver JSON
b0287a679ad297dfb40a71a6c03a2e99a3227d298da109edcdde3b9e131c0339
and NPZ8193be85083cb50e031a3fa95238daa3d6f7eeb2fa36f60c93e5074e3406cb40.
Bind each archival module to its unchanged historical execution blob;
recompute every plan descriptor and114 payload leaves, plus every228 baseline
leaf, restore/write fence, output comparison and counter. Do not modify or
bypass older live exact-source receivers. The new receiver has explicit
immutable archival bindings and must run in a **separate fresh process**:
pure plan recomputation deliberately refuses an imported simulator runtime.

Primary retained observations are all six full allocated solver outputs
against both measured GPU and prior same-context CPU, with exact hashes,
byte equality, differing scalar counts and ordered widened max differences.
Also retain full2x20 hybrid F32-minus-current-native F64 qacc and
qfrc_constraint differences per world. No tolerance, survivor rule, admission
threshold or contact matching is introduced. Even improvement is only
evidence about this hybrid family within the held measured context; no
exclusive cause, native phase identity, binary identity, simulator/PPO or
physical qualification follows. The deterministic decision remains
`native-family-cpu-hybrid-response-collected-not-cause-isolation-or-admission`.

Before execution: focused synthetic tests, complete15-file pure suite,
independent review, exact-source commit/push/native fast-forward and native
contract run. Retain terminal receipt, actual private CPU cache byte inventory
and independent full receptions on both hosts. Native contracts60s/2GiB;
solver180s/6GiB, fresh report≤256KiB/full NPZ<2MiB/cache≤96MiB.
Use inherited16MiB file cap, CPUQuota200%,TasksMax64,Nice10,LimitCORE0,
Typeexec,RemainAfterExityes,Restartno,KillModecontrol-group,TimeoutStop10s
and five literal CUDA-empty/thread1 assignments. Preserve DINO1592/946MiB
and four inactive protected service scopes. No GPU Duck workload is launched;
the existing advisory lease inode is left untouched. Launch before22:50UTC
and retain/close out by23:00UTC. Any failure is first diagnosed read-only;
failed source-prefix evidence is never overwritten or restarted.

The initial uncommitted synthetic test run passed75 and failed one success
fixture because its repeat executable opaque handle lacked the frozen
required prefix. That fixture was corrected, not the production guard;
the focused suite then passed76 in8.13s. No real solver ran during either
attempt. A pre-execution source check also removed an invalid same-process
self-reception, preserving the pure receiver's no-runtime-import boundary.
Independent review then passed76 focused cases in10.17s and suggested
explicit Warp option checks and a clearer reference label. Both were applied:
disableflags/enableflags are checked on both actual Models, native noslip stays
zero (frozen Warp Option has no noslip member), and
`original_gpu_bank_cpu_reference` explicitly names the differing-input prior
CPU result. Added negative option tests bring the focused suite to78;
independent rerun78/11.18s. The revised complete15-file pure suite passed1042
with one actual-physics case deselected in63.97s (earlier1040/61.24s before
the extra guards). Independent source review found no runner/receiver blocker.
Terminal helpers require canonical source-prefix destinations and the exact
ordered15-file test argv/JUnit destination or exact solver argv, not only
command fragments. Helpers are retained with raw source hashes alongside
the execution evidence; these receipts do not promote the diagnostic scope.

### Native-family hybrid solver execution and reception

Execution source baab71808c2bca04892c4b61907e675408cbb0d6 was committed,
pushed, verified at the fork and clean-fast-forwarded to native before any
real solve. The incremental source bundle SHA256 is
3df303b918fdc47b846f9de56365974cf6a7d0a07509d01547bca4a1e860fc07.
Committed15-file contracts passed1042 on both hosts with one actual-physics
test deselected; all1042 classname/name identities match and there are zero
failures/errors/skips. Mac JUnit54.930s/native8.678s (pytest55.10s/8.71s).
Native contract unit9.308973s, invocation460cfab632f7441ca4230ee09daed3f2,
exit0/PID0/empty cgroup. Exact retained leaves on both hosts:

| Leaf | Bytes | SHA256 |
| --- | ---: | --- |
| baab7180-mac-tests.xml | 144105 | f97993c555c51501c2cdc4c1a178097b09d5beb5cc43a8838ba243497818e3de |
| baab7180-linux-tests.xml | 144110 | 0f4d6a3e195c4f1dc76a57819d50b6af99e44bd5576efcf6b5cc952a86a957d4 |
| baab7180-contracts-service.json | 9466 | 3c70063a6a9498a192318e75ad201fbcb0b77227b85100c097806c4184f4913f |
| baab7180-linux-native-family-solver.json | 146380 | bdc34b3f800187145eca2e0078644f6c44ba3bc009883e77e6bcd04e13d3825a |
| baab7180-linux-native-family-solver.npz | 472546 | a3b37560f7185a6cdfb4b23d9148353c069d5330f8812e9967840b3121e547cd |
| baab7180-solver-service.json | 13219 | d6e77795402eb83b51b3e95964811336187d125a1c91a6a18a5e5a46b89465d3 |
| baab7180-linux-family-solver-reception.json | 37675 | b3fe6e0483bdfc8dad48ebc6875715ee1a313ee9dd7b264534498d3fcfc92479 |
| baab7180-mac-family-solver-reception.json | 37676 | ceb743fe155a1fc68565cc4a70eaa466a7136d3f49c6fd7d4226b51e48dbae02 |

The original terminal helper rejected its contracts receipt before writing
output: systemd `show ExecStart` flattens argv without quotes, while the
launch's `-k` value is one argument containing a space. Read-only `systemctl
cat` showed the correct retained quoted unit and all1042 tests had passed.
No source/service/test command was changed or restarted. Preserve original
helper SHA3d908a64401649069be237f579b101485dc9af003fcaa8c2f0b68acb25954f05;
new helper `retain_native_family_solver_service_v2.py` SHA256
2f6270e44751de29619438377eb52c5efcf241ad0450d635d6c86a95ae4b0b34
binds exact transient FragmentPath/Transient, reset-plus-single quoted
ExecStart parsed against the complete expected argv, flattened display and
full retained unit text/hash. Its synthetic success and three malformed-argv
refusals passed; independent helper review found no blocker. Both terminal
and receiver helpers explicitly reject optimized Python without relying on
assert (both `python -O` probes refused before evidence access).
Receiver helper SHA8636b647092904887fc017646be0dbffe26889d50549808055131635c63dcbd8.

The one-call CPU solver unit completed successfully in19.042561s, invocation
e07742a63f8a4528b58d9b54b1c16923, exit0/PID0/empty cgroup. Warp printed its
expected no-visible-CUDA-device initialization warning with CUDA hidden;
enumerated devices and all10 retained passive executables were CPU-only,
Torch CUDA remained uninitialized, and no Duck GPU process was created.
All347 Model arrays,184 statics,114 restored input leaves, actual layouts,
pointer bindings and post-solve fences passed. All six permitted outputs
changed from the restored pre-solve bank; no other leaf changed.
Counts remained ne/nl0/0,nf14/14,nefc14/30,nacon4,ncollision4;
solver_niter1/4 matches both post-solve references. The fresh native CPU cache
contains30 files totaling479232B, individually byte-bound on native. Mac
checks all report/payload/reference bytes and cache metadata, but does not
claim native cache-file inspection. The two complete reception JSONs differ
only in that explicit `private_cache_actual_bytes_checked` boolean.

Full allocated-output observations (hybrid minus each differing-input
reference; no point matching, force/state row pairing or tolerance gate):

| Output | Max abs vs measured GPU | Max abs vs original-GPU-bank CPU |
| --- | ---: | ---: |
| qacc | 2.6640543937683105 | 2.664053797721863 |
| qfrc_constraint | 0.3180689811706543 | 0.3180685043334961 |
| efc.Ma | 0.3180685043334961 | 0.3180685043334961 |
| efc.force | 0.6017815750092268 | 0.6017815731465816 |
| efc.state | 2 | 2 |
| solver_niter | 0 (byte-equal) | 0 (byte-equal) |

Against the complete **current native F64** generalized result, hybrid F32
qacc max absolute difference is2.4373025519796076e-6; world0/world1 maxima
2.4373025519796076e-6/2.3734929035512664e-6. qfrc_constraint max is
4.425324231505101e-7; world0/world1 maxima
3.857108815878065e-9/4.425324231505101e-7. This is an observed reduction
relative to the large original constraint-family discrepancy, not a selected
admission tolerance. Independent fresh-process review regenerated all228
leaves, restore/comparisons/counters/flags and descriptor/data hashes with no
runtime imports; JUnit/receipts/helpers also passed independent checks.
Four protected service scopes stayed inactive and the sole GPU owner was
unchanged DINO1592/946MiB (GPU43C/0%,961MiB used/15232MiB free).
The advisory lease inode was not changed; unrelated services and100.98/
FilmBrain remained untouched.

Interpretation: this complete native-family input intervention, while holding
the measured surrounding context, produces a near-native generalized result;
the earlier CPU/Ada same-input control was also close. Constraint-family
construction is therefore the next diagnostic target for this fixture.
This does **not** isolate contact count, contact manifold, J, regularization,
reference acceleration, precision or any one cause, prove native before-solve
identity, or qualify compiled kernels/the simulator/PPO/hardware. Keep all
admission flags false and do not launch training from this observation.
Next inspect/redeclare a bounded **own-manifold constraint-recipe audit**:
retain each generator's own contacts/rows, compare complete source-bound
regularization/reference-acceleration recipes without physical point pairing,
and do not invent a row-matched single-input swap or loosen the existing gates.

### Predeclared own-manifold like-input scalar recipe algebra

Purpose: distinguish the two **source algebra recipes on declared like
inputs** from differences in generated contact geometry/J/counts. This is
pure arithmetic, not another runtime intervention, and does not replace
actual native or Warp constraint construction. Keep own native16 contact
rows (four contacts) and own measured GPU32 rows (eight contacts) separate.
Cover every respective own slot/address once, retaining complete six
comparable captured scalars (D,aref,vel,pos,margin,frictionloss), captured-dtype
scalar hashes (JSON scalar recast to its declared dtype), declared inputs,
both source recipes and differences. Do not
pair physical points, normalize by counts, discard contacts or compare force/
state as physically corresponding rows. This arm does not cross-compare
the56 DOF-friction rows; native full44-row C validation remains separate.

Restrict to captured positive-standard-solref, power2, isotropic rigid dim3
pyramidal contacts, with frozen no-override/REFSAFE branch. Source recipes
are ordered Python F64: existing RECORD-bound Warp `_efc_contact_update`/
`_efc_row` algebra and exact3.10.0 upstream C functions already bound by the
ec81 report. On native own F64 inputs use its actual captured impratio and
explicit `1/sqrt(impratio)` for hypothetical Warp source algebra. On GPU
own F32 inputs widen values and explicitly derive hypothetical C
`impratio=1/(invsqrt*invsqrt)` from the actual stored Model ratio. Retain the
conversion convention and ratio reconstruction difference on every row.
This is **not** an actual native option on the GPU-contact arm, ABI/parameter
byte identity, FP32 rounding/fusion emulation or compiled-code agreement.

Bind ec81 report SHA674c6d7779bad3daef7c54aa761745d26df03cf44051753811f3dfcaf05490ae,
its unchanged historical module and complete independently regenerated
recipe descriptors. Reauthenticate native full capture/C-source and measured
GPU packets; bind all347 Model arrays before decoding numeric inputs.
The accepted baab hybrid report SHAbdc34b3f800187145eca2e0078644f6c44ba3bc009883e77e6bcd04e13d3825a
and unchanged execution module are motivation-only anchors; no new computed
result is inferred from a partial solver packet. Existing strict live
receivers stay untouched. New pure reception requires current exact-clean
execution source and recomputes every output descriptor, row, input conversion,
captured scalar hash and arithmetic result.

No equality tolerance or admission decision is introduced: retain all
differences even if zero. Synthetic tests deliberately show flat-zero-width
source branches can disagree (C averages impedance; Warp clamps width and
saturates), and mismatched ratio inputs yield differences. Those tests
demonstrate why ordinary-case agreement cannot be promoted to equivalence
outside the observed domain. Direct/negative-reference and other contact
types are excluded, not silently admitted. The deterministic decision is
`own-manifold-like-input-source-algebra-not-compiled-construction-or-admission`.
All runtime/new physics/optimizer/integration counts remain zero; every
solver/simulator/training/physical/binary/cause flag stays false.

Before actual arithmetic: focused/full16-file pure tests, independent source
review, exact-source commit/push/native clean fast-forward and native
contracts. Native contracts60s/2GiB and algebra120s/2GiB, source-prefix fresh
JSON≤256KiB under the retained tools dir; inherited16MiB file cap and all
Typeexec/RemainAfterExityes/CPUQuota200%/TasksMax64/Nice10/LimitCORE0/
Restartno/KillModecontrol-group/TimeoutStop10s/five literal CPU-hidden/thread1
assignments apply. Retain exact transient argv/resource/env/terminal receipt
and pure complete receptions/reports on both hosts. DINO and four inactive
protected scopes remain untouched; no lease mutation, GPU workload, training,
physical motion or raw perception. Launch before22:50UTC, close out by23:00.
Initial focused synthetic suite passed76 in3.81s, with no real analysis or
simulator execution; source inspection confirmed the retained own address
partitions independently, without cross-generator point matching.
The complete16-file pure suite passed1118 with one actual-physics case
deselected in58.20s; independent focused review passed76 in1.06s and found
no source/semantic blocker. Independent predeclaration/helper spotcheck also
passed. Terminal and receiver helper SHA256s are respectively
96b987ba2b984b358e3545231eacf7eff196ce6f9a66b67211356557bb4f72b4
and26a64f8895502302ff9a584905703715305eca02f369e873d1f78eb5f7f02725;
terminal parsing reuses the immutable2f6270e4 v2 quoted-argv reader.

### Own-manifold algebra result at f960d590

Execution source `f960d5905550a280e4350ce740c4f0a53c51f54f` was committed,
pushed and fast-forwarded exactly into the clean native feature branch before
execution. Its incremental source bundle SHA256 is
3039e334e440c20784f28a565e206c79ed38896748cacbd9290435096c11b58f.
Committed Mac/native16-file pure suites passed1118 with the one actual-physics
case deselected in49.60s/8.91s. All1118 ordered testcase identities match;
neither JUnit has a failure, error or skip. JUnit byte hashes are
2d1d120baf68e796c2f4763fa8c7f5419bb082feb1923595ffd2afba54480766
(Mac154263B) and86e5f967fc9846fc29031176a4fe22686c74fa09f4882a78d83dee12ef1b379d
(native154268B).

The retained contracts unit `microduck-own-recipe-contracts-f960d590.service`
finished exit0 in9.510238s, invocation85167b2ec1c64b8085ec897ca63a5ace;
the pure arithmetic unit `microduck-own-recipe-algebra-f960d590.service`
finished exit0 in3.619191s, invocationb31f4780ceb542e4993464cbeeeea57e.
Both have PID0/empty cgroup and the exact predeclared argv, environments,
transient definitions and caps. The9769B contracts-service receipt SHA256 is
f93d47c04d67af1581c77e33e41ece7157997c5fe6d6494eb4b00962ceb69cb8;
the7329B algebra-service receipt SHA256 is
d4361b3fa5d81dcc23f2d0d8e28748483736a094839a46cd569a9d1f95b073a8.
Both receipts retain four inactive protected scopes and unchanged DINO1592/
946MiB. No Duck GPU allocation, lease mutation or simulator call occurred.

Both `f960d590-{mac,linux}-own-manifold-recipe.json` files are byte-identical,
146436B, SHA256060e392640e9a5d33dd42c218e63d016dcf0ef79d7c96ec08e1cd3226f5d51d6.
Fresh independent complete reception on each host regenerated every row/input/
conversion/captured-dtype scalar hash/source descriptor without importing
Torch/Warp/MuJoCo/MuJoCoWarp. Both828B `f960d590-{mac,linux}-own-recipe-reception.json`
receipts are byte-identical, SHA256
d3252e28bbc7729194a82dc93c273a2a5eb9572f73f69ef336a88070e2e0da99.
All source/JSON/JUnit/service/reception evidence is retained under the existing
tools directory; no old packet, cache, helper or service was replaced.

For every own native16 and measured GPU32 contact row, the ordered F64 C and
Warp source recipes have exactly zero differences across D,aref,vel,pos,margin
and frictionloss. The explicit ratio-reconstruction difference is also zero
for these captured parameters. Both recipes exactly reproduce all six native
captured fields; against widened measured GPU fields, maximum absolute
differences are D1.2963849310709818e-7 and aref3.127977976635776e-8, with the
other four fields zero. Those latter differences are retained observations,
not an FP32 rounding/fusion explanation or an acceptance tolerance.

The deterministic decision remains
`own-manifold-like-input-source-algebra-not-compiled-construction-or-admission`.
The complete native-family CPU intervention and this ordinary-domain algebra
agreement justify looking next at source-bound contact/constraint generation
and phase identity, not reward tuning or PPO. They do not isolate count,
geometry, J, recipe arithmetic, precision or a single cause; synthetic excluded
branch disagreements remain relevant. No cross-generator physical point pairing,
compiled binary identity, native before-solve replay, simulator qualification,
training/policy promotion or physical motion is established.
Independent postexecution review regenerated the complete Linux packet in a
fresh CPU-hidden process, checked both raw report/receipt byte identities,
all48 row partitions and arithmetic residuals,1118-case test inventories,
both exact capped terminal units and unchanged protected/DINO scopes. Its
focused no-cache/no-bytecode suite passed76 in1.90s; no blocker was found.

### Predeclared pure native before-solve phase preflight

Next source: `ada_native_phase_plan.py`, with no simulator import/execution.
Bind the unchanged f960 source/report and independently regenerate its entire
48-row packet; reauthenticate the f854 full native capture, two current MJBs,
all logical public numeric Model arrays and seven F64 input states. Require
the captured closed model/option domain, including no plugins/activation/
flex/equality/mocap/userdata, zero enable/disable flags, dense Newton and no-slip0.
The full627 schema/MJB fences remain prospective runtime requirements, not a
claim of full native Data coverage or historical-model identity.

Bind the exact3.10 upstream
[forward source](https://raw.githubusercontent.com/google-deepmind/mujoco/28009f9105cd92784b7b0b30c0605a5e29107a77/src/engine/engine_forward.c),
64281B, SHA256cba19332e7dc7b110e158cb12c0ff9ff247b1b7e9338b2bbb88cedbf1b9887ed.
Retain seven lexical function-body hashes and the complete forward-call order;
this is not a transitive machine-code read-set proof. Proposed explicit calls
are fwdPosition,sensorPos,fwdVelocity,sensorVel,fwdActuation,fwdAcceleration,
capture before fwdConstraint, then fwdConstraint and capture again. The zero
enableflags domain excludes energy/sleep branches; callbacks must be absent.
No sensorAcc is proposed after the second boundary. Do not claim full sensor
state or output equality to ordinary forward, or assume inputs survive solve.

The plan retains closed projected20-field/EFC/contact/counter/statistic
inventories and complete model/state manifests, but executes none of them.
Future fresh native2-Model/2-Data execution would need separately committed,
reviewed code, installed provenance, all runtime shape/dtype/byte/bounds guards,
state/Model/MJB fences and a180s/6GiB supervised cap. This preflight explicitly
sets `execution_ready=false` and authorizes no runtime launch. No point pairing,
library/plant/reward/gate change, binary/cause/solver/simulator/training/physical
admission is established.

Preflight execution itself is pure CPU-hidden, four literal thread1 settings,
source-prefix fresh JSON≤256KiB, separately committed source before arithmetic,
fresh full independent reception on both hosts, and a120s/2GiB user unit with
the existing exact caps/environment/terminal evidence. Focused42 tests passed
in0.17s. One initial receiver test failed because its mock captured a mutable
expected dict subsequently mutated by the test; read-only diagnosis confirmed
the fixture issue and freezing that test reference resolved it without changing
receiver logic. The real frozen-source lexical check passed; no real retained
packet analysis or prospective physics execution has occurred yet.
Independent preexecution review passed42 focused tests in0.11s and the exact
frozen C lexical audit, with no simulator imports or retained-packet analysis.
Its clarity correction is explicit: prospective fwdConstraint executes the
native solver in both worlds; current zero solver calls describe this pure
preflight only. Internal solver-dispatch counts still require runtime evidence,
and the prospective whole pipeline is not a solver-only intervention.
The full17-file suite passed1160, one actual-physics case deselected, in13.33s.
Terminal/receiver helper SHA256s are respectively
fbbb19c37552f7289f7a33a872b65cdba9e9c91a0372f22c0883a65eed615e97
andc3f85477feb1aa1b266ad2a6a6250b1e61923a6402398d046092bb5dcc7b18ce;
their exact-unit quoted-argv parser reuses immutable2f6270e4, and optimized
Python is refused before evidence access. The contracts unit selects all17
files in the declared order and expects1160 cases plus one physics deselection.

### Native phase preflight result at8382487e

Exact execution source8382487e5dea6142d9300e617781cfa20b3d7b62 was pushed and
clean-fast-forwarded into native. Source bundle SHA256
552a8f05b249a49f25340f8716e8c80edeb034220dcd97bc5c7278a98c43b296.
Committed Mac/native17-file suites passed1160 plus one physics deselection in
15.96s/8.95s, with identical ordered testcase IDs and no failure/error/skip.
JUnit SHA256s are8771b7d3fee2bb645b4f8903b464ca3dc2abef615c75c7e8a1fcb56a856f9b90
(Mac159782B) and2f0344e9f0c28625451b31d5854d35e06ed51d82fceca940efaf0758d3cce235
(native159787B).

The contracts unit `microduck-native-phase-contracts-8382487e.service` finished
exit0 in9.554214s, invocation61aeeadf2c84449b886490e1be6ec524; its10051B terminal
receipt SHA256 isdf807273ad323a57240f1969b8ba28c46d82bdf773e2f5f4fdc308f687f400d3.
The pure plan unit `microduck-native-phase-plan-8382487e.service` finished exit0
in5.197887s, invocationb4608dde534b499cb35054034e02f376; its7058B terminal receipt
SHA256 is774d0d6e96141f4f562b5872a3a87757949831546a271484cd9fc9cd3cd49908.
Both have PID0/empty cgroup, the exact declared argv/limits/env/transient units,
four protected scopes inactive and unchanged DINO1592/946MiB.

Both168336B `8382487e-{mac,linux}-native-phase-plan.json` reports are identical,
SHA2561c1980250956120ad9589b706bf71053e8d72fd14d03ac2fea0819f7856eeb85.
They bind954 logical native Model array descriptors, seven input-state fields,
the full native payload descriptor hash and both full current MJBs; this does
not claim a complete native Data snapshot or historical model identity.
Both469B `8382487e-{mac,linux}-native-phase-reception.json` receipts are identical,
SHA256d857bf2cc9ecc039fd8e1c85d952f0003f3f99a3275c03db2c213c2322193880.
Fresh independent review regenerated the entire plan and raw references with
no Torch/Warp/MuJoCo/MuJoCoWarp imports, checked all manifests/guards, test case
identities, exact source/helper hashes and both terminal units. No blocker.

All evidence is retained in the existing tools directory on both hosts.
Decision remains `source-bound-native-phase-plan-not-executed-or-admitted`:
no Model/Data allocation, native phase/collision/construction/solver call,
integration or optimizer step occurred. The prospective solver distinction is
explicit, while `execution_ready=false` and all admission flags remain false.
Next prepare/test a closed before/after projection contract before implementing
or launching the separately reviewed real native phase collector.

### Predeclared native phase projection contract and retained compatibility

Add pure `ada_native_phase_projection.py` before any real collector. Each
world's closed68-leaf projection contains the declared20 output/state fields,
all EFC/contact leaves, counters, warnings and all solver statistics, retaining
correctly shaped zero-size world0 contact leaves. Require exact F64/int32/int64
layouts, finite values, all seven independently supplied state bytes, the
declared native0/4-contact and14/30-row fixture counters, bounded iteration/
matrix/island counts, warning0 and every own contact type/id/address partition.
These are projected-array checks, not full native Data coverage or producer
authentication; even synthetic arrays can satisfy them. No physical contact
matching or admission follows. All68 leaves must survive comparison, including
solver-written outputs; any nonzero difference is retained without tolerance.

Read-only inspection of the SHA-authenticated f854 bank showed its dense J
uses the full flat array, while sparse row metadata getters are zero, not a
populated20-per-row table. The initial uncommitted projection assumption was
corrected before any real compatibility analysis: retain all sparse metadata
bytes/descriptors and differences but do not interpret them as dense storage
indices. Dedicated tests exercise each retained metadata leaf. No captured
packet, underlying model, library or gate was modified.

The separately source-bound compatibility arm reauthenticates/regenerates the
entire838 phase plan, all f854 capture/payload/MJB references and original state
inputs. Decode all68 leaves per world from **retained post-forward only** data
into owned copies, then validate each against independently restored F64 input
states. Retain complete136 descriptors/raw hashes and the referenced full954
Model-array inventory count. Do not relabel this as before-solve, manufacture
a pre-boundary, compare before/after, execute native calls, or infer producer/
compiled/cause/simulator/training/physical identity. The deterministic decision
is `retained-post-forward-projection-compatible-not-before-solve-capture`;
all runtime/capture/admission flags and `execution_ready` remain false.

Before this pure compatibility analysis: synthetic192 focused tests, independent
review, full18-file suite, commit/push/exact clean native fast-forward and native
contracts60s/2GiB. Compatibility120s/2GiB, outputJSON≤256KiB, fresh source-prefix
names, existing exact transient argv/env/caps/terminal and both fresh full
receivers apply. All launches precede22:50UTC; unchanged protected/DINO scopes,
no GPU lease mutation. Latest focused192 tests passed0.65s. Helper SHA256s are
d35f034ee3e2218a0942b38cfcae89e1c30c03b6f48ced9579a9423f5c2ff9b0
(terminal) and7bee3843635c50de8116ae8522bb0a059fa6111569dd591ebf192f36e3b5b7b1
(receiver); py_compile passed. No retained compatibility analysis or real native
capture ran before this predeclaration.
The complete18-file suite passed1352 with one actual-physics deselection in
23.44s. Independent source/document/helper review passed192 focused synthetic
tests in0.39s with no cache/bytecode, found no blocker and performed no actual
analysis or simulator execution. Exact source/commit/native contracts precede
the separately retained compatibility packet.
