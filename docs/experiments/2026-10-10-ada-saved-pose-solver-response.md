# Saved prepared-pose solver-response follow-up

Base `6fe37f27cd4ea1009a78502a8f29b60cb4bdc7fe`, exact branch
`feat/athletics-obstacle-curriculum`. This separates the next response arm from
the completed [collision-only intervention](2026-10-10-ada-saved-pose-collision.md).
The source audit and implementation-primitives sections retain their historical
gates. The closed CLI/receiver added below still requires exact-source execution
checks before its first capped response run. No library, plant, driver, motor,
acceptance gate, curriculum or training protocol changes.

## Source audit and dependency decision

`ada_solver_stage_audit.py` checks all69 installed MuJoCo-Warp Python files
and three Warp loader/config files against wheel RECORD. Six relevant files
also match explicitly frozen SHA256s. AST receipts preserve exact lexical
call order, arguments, line references and conservative model/data attribute
references. Conditional and unreachable calls remain labeled **lexical**, not
claimed executed. This inventory is not a transitive kernel read/write proof.
The audit imports no NumPy, Torch, Warp or MuJoCo runtime and executes no physics.

The frozen primary source establishes:

- `forward.py:596-619`: `fwd_position` starts with kinematics, then com_pos,
  camlight, flex, tendon, crb, tendon_armature; conditional factor_m and collision;
  constraint construction; disabled island path; transmission. Thus public
  fwd_position/forward cannot preserve a supplied geometry intervention.
- `forward.py:1230-1257`: ordinary forward passes factorize=False to position,
  then runs velocity, optional control callback, actuation, acceleration with
  factorize=True, solver and sensors. Position and acceleration factorization
  choices must not be silently changed.
- `smooth.py:602-632`: com_pos consumes supplied body/inertial/joint poses and
  writes subtree_com, cinert and cdof, not those supplied poses.
- `constraint.py:2503+`: construction resets counts and constructs friction,
  limits and contact rows. Its outputs include contact.efc_address; candidate
  addresses-1 are retained before construction, not required unchanged afterward.
- `solver.py:3341-3387`: `wp.copy(d.qacc, d.qacc_warmstart)` copies the **source
  warmstart into destination qacc**, because Warp `copy(dest, src)` is defined
  at `warp/_src/context.py:8708+`. When warmstart is disabled the source is
  qacc_smooth. Forward solve does not integrate or itself update qacc_warmstart;
  the `_advance` integration path does, and is excluded.

The source audit explicitly checks the two copy argument lists, not an assumed
arrow direction. The current retained plant has zero equalities, activations
and mocap worlds, seven sites, and14 joint transmissions. Site poses are not
among the earlier nine-field collision bundle. Do not generalize that bundle
to every dynamics dependency or initialize fixture sites from allocation poses.

## Proposed smallest bounded arm, not yet execution-ready

Use the existing authenticated predecessor source/report/files and exact
float32 seven-state and motor inputs. Fresh native plant, all347 model-array
bindings and unchanged dense default options must match the prior arm.
Fresh Warp CPU data allocation still counts its one internal native kinematics
call separately. No fixture native/Warp kinematics, integration, BAM recompute,
constraint/force injection, altered solver settings or model repair.

In addition to the nine saved collision poses, supply **site_xpos/site_xmat**
from the same authenticated prepared bank, with complete logical layout and
byte checks. This is an eleven-field prepared-pose dynamics intervention,
not a retroactive claim that the eleven measured Ada GPU-boundary arrays were
captured. Require rigid/no-SDF/no-contactfilter, all other callbacks absent,
no equalities/mocap/activations/tendons/body transmissions and the actual
joint-only transmission topology. Unsupported paths fail before physics.

Keep the frozen position-stage order, omitting only fixture kinematics and
the factor_m branch excluded by ordinary forward's factorize=False. Do not
move construction after velocity merely because a different ordering could
work for this zero-velocity fixture. Keep camlight/flex/tendon/tendon_armature
calls in order even when topology makes them no-ops. Recompute com_pos, crb,
transmission and normal velocity/passive/bias/actuation/acceleration outputs
from the supplied inputs; do not reuse preceding GPU-derived mass or forces.
Sensors/energy must either be executed in original order with supplied site
poses or be proven force-independent under explicit guards before exclusion.
The exact implementation and focused tests are a further review gate.

Retain complete actual model/data byte/layout inventories, including all
allocated Contact and Constraint fields, at these boundaries:

1. Before collision, after nonkinematic position prerequisites.
2. After collision, before make_constraint: all16 generated candidate fields,
   complete eleven actual poses, unchanged state and zero EFC/solver counters.
3. After make_constraint/transmission: actual EFC addresses/counts and rows.
4. Immediately before solve, after velocity/actuation/acceleration: full Data
   arrays, mass factors, qacc_smooth, qfrc_smooth, warmstart, every constraint
   input and all actual candidate fields, not a hand-selected subset alone.
5. After solve: full Data arrays, complete active row force/state, generalized
   qfrc_constraint and qacc, solver counters and complete generated contacts.

Synchronize and check all eleven pose bytes, seven state bytes, model bindings,
capacities/options and callback/static identities at each applicable boundary.
Preserve pre-constraint and post-constraint candidate tables separately.
Only EFC addresses may change in the contact struct after construction; report
and fail on unexpected changes in the other candidate fields. Reconstruct
J.T@force and ordered-pair world resultants over all active rows/contacts, using
existing arithmetic conventions with no rounding, spatial pairing, contact-count
normalization, imposed force, tolerance or acceptance promotion.

This produces a **new CPU-generated manifold followed by its CPU solver
response**. It is neither integrated state replay nor an Ada-versus-CPU
same-manifold solver control. The retained Ada post-solve table lacks
includemargin and complete measured GPU collision-boundary poses. Residuals
against historical Ada resultants can be descriptive only. A causal downstream
solver control needs branching from an identical complete pre-solver bank;
do not reconstruct a purported GPU input manifold from its solved outcomes.

## Execution gates retained

No response physics has run under this document. Before a runner may execute:
implement the closed CLI/receiver and complete boundary checks; test them on
Mac, independently review, commit/push the exact feature source, fast-forward
the clean native worktree and run its focused CPU checks. Predeclare its exact
source, absent output/cache paths, invocation and resource limits first.

Future native response cap: CPU-only user service, literal CUDA_VISIBLE_DEVICES
empty, one thread per pool, Type=exec/RemainAfterExit, RuntimeMaxSec180,
MemoryMax6G, CPUQuota200%, TasksMax64, Nice10, LimitFSIZE16M, LimitCORE0,
Restart=no, KillMode=control-group, TimeoutStopSec10. Fresh private CPU cache
and passive existing-executable provenance follow the preceding arm;
loaded_binary_bytes_bound remains false. Stop before any integrator/step.
No GPU Duck job, PPO, raw perception, policy promotion or physical motion.
Preserve Grounding DINO and inactive protected services. The07:00 Shanghai
deadline and ten-minute closeout reserve remain unchanged.

Source-audit CLI only: `python -m mjlab_microduck.ada_solver_stage_audit
--output <absent-absolute-json> --source <exact-clean-SHA>` with CUDA hidden.
It needs no runtime cache because no runtime is loaded. JSON source bindings
and both-host audit results will be retained separately from future physics.

Audit/test service cap60s, MemoryMax2G; CPUQuota200%, TasksMax64, Nice10,
LimitFSIZE16M, LimitCORE0, Restart=no, KillMode=control-group, TimeoutStopSec10,
Type=exec/RemainAfterExit and thread-pool1. These limits apply only to this
read-only source audit and its three-file contract suite, not response physics.
Output JSON is41838 bytes before exclusive serialization/source additions.

## Preparation checks

Mac focused audit11/11 and combined audit/saved-collision/metadata191/191
passed (combined1.93s). Import audit asserts none of NumPy/Torch/Warp/MuJoCo/
MuJoCo-Warp runtime modules loaded. Changed/missing source inventories, ambiguous
functions, lexical branch interpretation and the no-physics AST fence are tested.
Independent Luna review passed11/11 in1.28s and confirmed the source order,
deferred factorization, extra site/state guards and interpretation limits.
The review initially reversed Warp copy arguments; the owner checked the
installed primary API and the reviewer corrected this before integration.
The audit now records and checks the actual source/destination argument lists.
No runtime library or installed source was changed. Neither source review nor
these tests authorize solver execution or training.

## Retained installed-source audit execution

Exact source `ef8e522aa72eb2f3f192bcb999ae2f21cdea565b` was pushed and
fast-forwarded cleanly on native100.100. Both committed-source three-file
suites passed191 cases, Mac1.72s and native1.54s, with identical complete
testcase multisets. Both audit JSONs are **byte-identical** and independently
recomputed from installed source, including the module SHA and closed decision.
This demonstrates the source-audit packet, not a CPU/GPU runtime or solver result.
Independent Luna reception matched every installed-source audit field, both
191-case zero-failure/error/skip XMLs, the module/helper/receipt hashes and
declared stage/copy/site/manifold boundaries. It used the received packets,
not direct host/service access; owner separately rehashed all six native files.

Native unit `microduck-ada-solver-stage-audit-ef8e522a.service`, invocation
`0da9193be29a46c7ad5db376a8704c00`, finished successfully in2.210055s,
all declared60s/2G limits and literal CPU-hidden/thread1 environment checked.
MainPID0/status0/Result success, empty ControlGroup, active/exited because
RemainAfterExit is retained. Grounding DINO PID1592 was the only GPU compute
owner and all four system/user protected mission states remained inactive.
No collision, constraint construction, solver, integration or Duck GPU process
was launched by this audit. No response runner or simulator gate is accepted.

Both hosts retain these files under `artifacts/tools/ada-solver-stage-audit/`:

| Evidence | SHA256 |
| --- | --- |
| `ef8e522a-mac-audit.json` and `ef8e522a-linux-audit.json` | `17a1e064e0afa2cc68b2c63a8d66e2842f800c73207d3e54c5c3d2275686f765` |
| `ef8e522a-mac-tests.xml` | `e19679a249d84510463226dfaff7b20ff4a84763cef844ab2bded280589fbe78` |
| `ef8e522a-linux-tests.xml` | `b03c340ba6577b072223959ae8f6181b1f25d95ecf6a605669c15c842f466e73` |
| `ef8e522a-native-service.json` | `28985253a147a8023f8edf1db64978708a1af19bbb12228eab23f455d3ee815d` |
| read-only `retain_source_audit_services.py` | `6efa1b7a4e9adb06292d6b560aaba7fd676b9efa03bb0d7e1649fc254348bfb9` |

Next implementation is the guarded eleven-pose CPU response runner and its
closed receiver, with focused tests/review before any separately frozen
source execution. The scientific/authority boundaries above remain unchanged.

## Staged response implementation primitives, not runtime execution

At base `bdf0beb7fea4894e8eb687352664042b0da9aa46`, added
`ada_saved_pose_response.py` and its focused tests. At that primitive revision
the module had **no main or standalone CLI**. It imports no simulator runtime until its explicit run
primitive is called. That primitive has not been called by the owner/reviewer
or tests. Launch/reception/source-binding/cache gates must be implemented and
reviewed separately before any response physics.

The prepared code implements all eleven pose binders; explicit no-flex/
equality/mocap/activation/tendon/body-transmission/custom-callback guards;
joint-only14 transmissions and seven sites; dense/CPU/default dispatch checks;
exact frozen staged calls and deferred factorization. It stops immediately
after solve, deliberately before acceleration sensors as well as integration.
Position/velocity sensors retain ordinary source order with supplied sites and
default energy flags0. All seven callbacks must be absent.

Each of five capture boundaries retains owned raw bytes and complete logical
and expanded NumPy layouts for **all114 Data arrays**, not just active rows.
Every stage checks unchanged347 model-array bytes, within-process static/
pointer/options binding, all eleven prepared poses and seven state inputs.
Only construction-time EFC addresses may differ in generated Contact records.
The before/after solver write fence compares every Data array and allows changes
only in qacc, qfrc_constraint, solver_niter, efc.Ma, efc.force and efc.state,
matching the frozen solver's declared Data outputs. Other bytes—including
warmstart, candidate addresses, Jacobians, constraint inputs and mass factors—
must remain exact. Capacities, constraint coverage/counts and solver iteration
limits fail closed; they do not impose a new numerical acceptance tolerance.

Pure mock/NumPy fixtures passed48 focused cases and239 combined cases with
the audit/collision/metadata suites (combined4.44s on Mac). These test exact
ordering/factorization, all extra site layouts/hashes/finiteness, state signed
zero/one-bit changes, callback/topology/CPU refusals, complete Data snapshots,
candidate address stage semantics and solver-input write refusals. The first
AST fence test incorrectly handled subscript-rooted calls; it was corrected
to inspect the called attribute directly. No physics was run during that
test failure or repair.

Independent Luna source/API review passed48/48 in0.43s, verified that every
guarded model/data/callback attribute exists in the frozen types, and found
the stage order and six-output fence consistent with the frozen source. The
review explicitly notes these are **synthetic contract tests**, not proof of
actual native Data layout, CPU solver behavior, CUDA identity or admission.
Next implement the closed receiver and exact-source CLI; then re-review and
freeze the full arm before any capped response invocation.

The primitives were committed/pushed as
`7f31329c9dfa592b93efaad322be77e25b3779b0` and fast-forwarded cleanly on
native100.100. Committed-source four-file tests passed **239/239 on both
hosts**, Mac4.59s and native1.65s (CLI timing); full testcase multisets match
with zero failures/errors/skips. The native synthetic-test service reused
the source-audit60s/2G/CPU200%/thread1/Tasks64/Nice10/FSIZE16M/CORE0/
Restart=no/KillMode control-group/Stop10s caps; no response primitive ran.
Invocation `75dc998423614c349f561b59d121f7ba` finished in2.215121s,
MainPID0/status0/Result success/empty ControlGroup, active/exited. The helper
again verified inactive protected services and Grounding DINO as sole GPU
compute owner. No CPU response or GPU Duck workload has been launched.

Both hosts retain these files under `artifacts/tools/ada-saved-pose-response/`:

| Evidence | SHA256 |
| --- | --- |
| `7f31329c-mac-tests.xml` | `fbf1bc29287e00f06fc7cacc205155a18ca8e88e66430437dc7c07ef8898e011` |
| `7f31329c-linux-tests.xml` | `ac16bb54550f7fbbd26b0a5fb684f327f3e38fc100b6e070d79503d8a6c581a2` |
| `7f31329c-native-service.json` | `7825a2c6291cce2000fda4c11549b23afae61dddeeeb85d2b3d2b6967ca70ff4` |
| read-only `retain_primitive_test_service.py` | `959a714cf638622a15fafac2167bab1c87a5c3b9e8296b895bcbe1f68189a0e7` |

The receiver must decode and independently validate all five complete raw
stage banks, not trust these mocked tests as actual boundary evidence. Add
source-bound exclusive CLI/private-cache receipts, final complete-row/wrench
arithmetic and negative reception tests before a new physics predeclaration.

## Closed response CLI/receiver predeclaration

At base `c33d464cc25a8ed6f3fbba4c79b6338fba0a4b54`, added a closed
CLI and pure byte-bank receiver for the staged primitives. The receiver derives
expanded component layouts independently from all114 authenticated logical Data
layouts and checks the exact **570-array** five-stage inventory. Every pose/state
byte is rechecked in every bank. It recomputes actual counters, all14 friction
rows per world, complete contiguous dim3 four-row contact addresses and raw-slot
links, candidate stability and the six-output solver write fence. No active row
may be omitted or shared by two contact groups.

Diagnostic force decoding is host arithmetic over the new solved EFC rows,
not another kernel call or imposed force. Float32 pair/add rounding follows
frozen `support._decode_pyramid`; float64 all-row reconstruction and ordered-pair
world wrench sums then apply. V1 used the historical BLAS reduction; the v2 repair
below uses explicit ascending scalar row-order/no-BLAS reconstruction.
Historical comparisons are explicitly
**current minus historical** and never establish identical solver inputs,
physical point identity or acceptance. No tolerance is introduced.

The exact-source native CLI requires literal CUDA hidden, this machine/root,
clean feature branch and committed module bytes; frozen package/source bindings;
absent canonical output/NPZ/private-cache paths; actual running user service with
all declared180s/6G caps, RemainAfterExit=yes, PID/invocation and the exact five
CPU/thread environment tokens. It refuses after **2026-10-09 22:50 UTC** to
preserve the ten-minute closeout reserve. Output uses exclusive-create and a
bounded finite JSON/NPZ. Native reception must supply the private cache to rehash
every cache file; portable Mac reception checks its declared inventory only,
without duplicating native cache storage. Both modes explicitly leave loaded
binary bytes unbound.

Predeclared first response invocation, after exact-source tests/review/push and
clean native fast-forward:

```text
unit: microduck-ada-response-<source8>.service
python: /home/converge/work/microduck_rl-athletics-obstacle-curriculum/.venv/bin/python
module: mjlab_microduck.ada_saved_pose_response
--source <exact committed 40-hex source>
--input /home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/evaluations/ada-duck-contact-0ce8c9d0a820
--output /home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/tools/ada-saved-pose-response/<source8>-linux-response.json
--cache /home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/tools/ada-saved-pose-response/<source8>-private-cache
```

Caps are the already declared180s/6G CPU-only caps above. Run one fresh retained
unit, never repair/retry it while active, and diagnose any failure read-only
before changes. No GPU lease or new GPU process is needed for this CPU arm;
foreign Grounding DINO and inactive protected mission services are preserved.
Successful collection still means only an artificial eleven-prepared-pose,
CPU-generated-manifold response, **not simulator admission or PPO permission**.

Pure response tests passed **123/123** on Mac in0.79s. Independent Luna
read-only source/API/receiver review repeated123/123 in0.81s, checked all114
actual predecessor layouts and the frozen float32 decoder source. It identified
the optional-cache reception distinction, now explicit in API/docs. No runtime
physics was called during these tests or review. The five-file response,
source-audit, saved-collision and metadata regression suite passed314/314 on
Mac in2.67s; `git diff --check` passed. Native committed-source tests,
first response execution and retained result hashes remain separate next gates.

## First response execution and portable arithmetic diagnosis

Source `c1634dfd956815380e4214d2ccda13b92d70354b` was pushed and
fast-forwarded cleanly on native100.100. Its committed-source five-file suite
passed314/314 on both hosts, Mac2.60s; full testcase multisets match with no
failures/errors/skips. The Linux contract unit invocation
`c6f65f6b88864787a278a2a6b964af78` completed in2.705104s under60s/2G caps.

CPU response unit `microduck-ada-response-c1634dfd.service`, invocation
`74532b078709493788e5c455c21a9a4b`, succeeded in37.599239s under all180s/6G
caps. MainPID0/status0/Result success/empty ControlGroup, active/exited.
Native closed reception independently rehashed all72 private-cache files
(4812174 bytes) and recomputed all570 stage leaves.24 already-held CPU
ModuleExec receipts remain passive metadata, not loaded-binary byte proof.
The CUDA-hidden initialization log emits Warp CUDA error100 before choosing
the CPU-only device; it is not a service failure or a CUDA execution claim.

Actual stages: initially zero candidates/EFC/solver work; collision ncollision4,
nacon8 with zero EFC/solver counts; construction and pre-solve nf[14,14],
nefc[14,46], ne/nl0; final solver_niter[1,4]. Only the six predeclared Data
outputs changed across solve. Every prepared pose/state/model binding stayed
exact. Candidate metadata residuals are the earlier collision-arm residuals:
dist1.127773430198431e-10m, pos1.862645149230957e-09m; frame/friction0.

Each foot has four generated contacts. New summed vertical diagnostic forces
are3.2832815647125244N and3.284973293542862N. Relative to retained Ada,
vertical deltas are-1.7881393432617188e-7N and-8.940696716308594e-8N;
qacc max residual7.62939453125e-6. These are descriptive different-input
comparisons, not same-manifold controls, solver acceptance or skill evidence.

The **Mac v1 receiver failed closed**, despite native v1 reception passing:
`ValueError: every actual stage invariant and resultant recomputed`. Read-only
owner and independent Luna diagnosis found exactly six differing scalar leaves,
all Mac-minus-native **+3.469446951953614e-18**: world1 DOFs4 and8 of
contact_generalized_force, total_generalized_force and reconstructed_minus_stored.
Every other analysis field and the complete saved arrays/layouts/counters agree.
The shared historical helper uses float64 NumPy `J.T @ force`, whose BLAS
reduction tree is backend-dependent. This is a reception/arithmetic portability
failure, not a failed solve or divergent physics. Do not call v1 portable reception
accepted; preserve the original packet/source unchanged.

Both hosts retain these v1 artifacts in `artifacts/tools/ada-saved-pose-response/`:

| Evidence | SHA256 |
| --- | --- |
| `c1634dfd-linux-response.json` | `1144f106916d138104073d5d6cc7166e877670f8eef29bc49771d203d95d8f71` |
| `c1634dfd-linux-response.npz` (1180648 bytes) | `b3c52c1463863389ba8161ac9fca0bd080b020e01b48920405904024c183dc34` |
| `c1634dfd-mac-tests.xml` | `6137052ffc09020a8464178b0298c4c0bfe545be7b199aa95241c2b3ae42967a` |
| `c1634dfd-linux-tests.xml` | `c57c111c8081800ccfeee5b3af3033683d6170b85815e10db22864f755c84791` |
| `c1634dfd-contract-test-service.json` | `a1e70dca9ee971288817e865c90314497fd2d5c7067cf8b7e671695e35d36c7a` |
| `c1634dfd-native-services.json` | `fbc9d7da315a656538e41b9178bd3650577793f45e911c269cead8e5e927d89a` |
| read-only `retain_response_services.py` | `8a6104530e3b4940762b9372c71be4751beb6c9dbe0d298b98639349452bb060` |

## Minimal response v2 arithmetic repair predeclaration

At base c1634dfd, leave historical/shared arithmetic and all physics unchanged.
The response-local generalized reconstruction now casts each float32 operand to
Python IEEE float64, computes each product explicitly and adds it in increasing
raw EFC-row order, separately for friction/contact/all-row sums. Subtraction from
stored generalized force is then explicit scalar float64. No BLAS matrix reduction,
reassociation, tolerance, normalization or row omission. The stated reduction
convention and response protocol advance to v2; this does not change any simulator
gate. An adversarial2**60,1,-2**60 cancellation fixture requires the declared
ascending-row sum0, distinguishing a regrouped1. Full packet test remains exact.
Independent Luna read-only repair review passed124 focused tests in0.82s and
verified explicit separate binary64 products/additions and unchanged historical
helper. The owner corrected the older paragraph's reduction description as
requested. Five-file regression passed315 cases; no new response has run yet.

After focused tests, independent review, exact new source push/clean native sync
and native contract tests, run a fresh **v2** CPU service using the identical
predeclared180s/6G invocation/path template above with the new source8. Retain
fresh output/cache paths; do not overwrite v1 or reuse its compiled cache.
Success requires native reception **with cache** plus exact portable Mac
reception from the new packet. A numerical residual reduction alone is not a
passed reception gate. Preserve Dino/protected services and all diagnostic
interpretation limits. No solver tolerance, new GPU/PPO work or physical motion.

## Retained v2 exact reception

Source `05216efe9d33a80d7a839eaa20b7a57ea05dbe86` was pushed and
fast-forwarded cleanly on native100.100. The five-file committed-source suite
passed **315/315 on both hosts**, Mac2.64s; full testcase multisets match with
no failures/errors/skips. Contract unit invocation
`de63a72a44544333a10e7dafed804acf` completed in2.714729s under60s/2G caps.

Response unit `microduck-ada-response-05216efe.service`, invocation
`50a4b911d8644a29984df3201d1613db`, succeeded in37.778253s under all180s/6G
caps. MainPID0/status0/Result success/empty ControlGroup, active/exited.
Native reception independently recomputed the complete packet and rehashed
all72 private-cache files (4812174 bytes). Portable Mac reception now passes
**exactly**, including all numeric analysis fields, not a tolerance-based check.
The24 passive CPU executable receipts still do not bind loaded binary bytes.
Independent Luna portable artifact reception also passed: all source/layout/
570-array/arithmetic bindings, exact v1 leaf equality, both315-case XMLs and
service/helper hashes. Its scope excludes direct host queries/native-cache
rehashing; the owner performed those native checks separately and verified
all seven native artifact hashes against the retained Mac copies.

All570 saved Data stage leaves are **byte-identical to v1**; even the complete
NPZ hash is identical. Thus the tested revision changes the response-local
reduction convention, not the collision, construction, solve, state or motor
inputs. Counters, eight candidates, model/pose/state bindings and the six changed
solver outputs remain those listed above. Ordered full-row reconstructed-minus-
stored generalized maxima are[0,4.973262548446655e-7]; they are descriptive
float64-versus-stored-float32 residuals, not a new acceptance tolerance.
Historical Ada qacc residual remains7.62939453125e-6. No skill, integrated
trajectory, GPU-origin, same-manifold solver or simulator qualification follows.

Both hosts retain the v2 files under `artifacts/tools/ada-saved-pose-response/`;
native private-cache bytes remain only on native to avoid duplicate storage:

| Evidence | SHA256 |
| --- | --- |
| `05216efe-linux-response.json` (245711 bytes) | `af380438ef6da87e1e77f7eefcb5cdc1da8724fab757a8b9d2836c2f0a8386e9` |
| `05216efe-linux-response.npz` (1180648 bytes) | `b3c52c1463863389ba8161ac9fca0bd080b020e01b48920405904024c183dc34` |
| `05216efe-mac-tests.xml` | `3e1865c978428999aae716b0039d4edd3dcf21d5cfb8a07e229bd01c75b4e18e` |
| `05216efe-linux-tests.xml` | `7e79bf408787624cba41b13455367c05036385fff3f0ff23353f19c4abc4b89d` |
| `05216efe-contract-test-service.json` | `68ab894ad123406b3910bcdc1fe26e9fca3192ef763777d38e6b96086cebd779` |
| `05216efe-native-services.json` | `88141ada9e65bf30fe83317408c1d4dfc7f0e997e8bad2f694258732417769d2` |
| read-only `retain_response_services.py` | `8a6104530e3b4940762b9372c71be4751beb6c9dbe0d298b98639349452bb060` |

Both protected mission services remain inactive in system and user scopes.
Grounding DINO PID1592 is the only GPU compute owner (946MiB); no Duck GPU
job or active Duck CPU process remains from these units. The driver/library/
plant/acceptance gates are untouched. Next prepare a separately bounded,
source-tested **complete measured Ada collision/pre-solver boundary capture**
and identical-bank control, rather than reconstructing GPU inputs from solved
outputs or promoting small response residuals. Any GPU execution still needs
its own occupancy/lease/cap/deadline proof and numerical-gate review first.

## CPU cache-option erratum and future preparation

A later source review found the three CPU replay CLIs assigned a dynamic
`wp.config.enable_precompiled_headers` attribute. The frozen Warp option is
actually **use_precompiled_headers** (`warp/config.py:241`). The former
attribute was a no-op, so earlier CPU PCH-disabled wording does not establish
that actual option state. Frozen `warp/_src/build.py:117-139` CPU compilation
does not consume that option; the CUDA builder does (`build.py:97`). This
does not explain the v1 BLAS reduction mismatch, change retained CPU raw
results/cache hashes, or invalidate exact v2 reception. Existing GPU supervisor
already sets the correct option. Preserve all historical artifacts unchanged.

Future CPU replay/saved-collision/response CLIs now share a tested helper that
requires the real boolean option, sets use_precompiled_headers=False and
verifies it together with the private cache path. Missing/phantom/nonboolean
options fail closed. The change does not alter frozen libraries or physics.
The [new measured-boundary preparation](2026-10-10-ada-measured-boundary-control.md)
documents the next source-only primitives and remaining runtime launch gates.
