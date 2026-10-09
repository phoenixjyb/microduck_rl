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
