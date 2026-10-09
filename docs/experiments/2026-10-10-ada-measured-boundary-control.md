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
