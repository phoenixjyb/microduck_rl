# Separately fenced native positive EFC transition preparation

## Scope and protocol boundary

Base `a6d65d9b90650900a6c736913a967caf99ae5638`, exact branch
`feat/athletics-obstacle-curriculum`; the new fence contains only
`stance_solver_efc_probe.py`, `stance_solver_efc_receiver.py`, their focused
test file and this document. Protocols are
`microduck-efc-transition-probe-oct9-v1` and
`microduck-efc-transition-receiver-oct9-v1`. No old source fence, receiver,
installed runtime, driver or retained artifact is modified or monkeypatched.

The initial preparation slice built a collector/receiver and tested it with
hidden CUDA, without launching the native GPU test. The separately authorized
execution and closeout are recorded below. The older cost-only receiver
intentionally
requires paired force/state/dense equality. The new receiver instead validates
each arm independently, allowing the positive control's intended changes.
No broader solver window, motor/plant acceptance, training, video, raw
perception, actor expansion, learned skill or physical motion is authorized by
a preparation result. All six qualification flags remain false.

## Immutable native interfaces and typed staging

Reuse `CostModuleExecutable` without modifying its implementation: three
fresh module groups, shared init/dense plus unique EFC and Gauss specializations,
four offline complete CUBIN/function-span SASS checks, explicit loads, actual
returned ModuleExec/cache/hook identities and unchanged frontend/loader checks.
Reuse a fresh one-shot `CostDispatchObserver` for each sequential arm. The
unchanged `_update_constraint(False)` calls the four pinned sites in order;
complete eight-boundary packets are saved per arm, including failure boundaries
when available. Same-stream readbacks deliberately change timing. Loaded
driver-resident bytes and atomic order are not observed.

Restore the sealed complete historical 24-field bank separately before each
arm. For control only, derive all thirteen overrides through the unchanged
`stance_solver_efc_transition.control_fields(packet)`:

- Seven inherited Gauss driver/cost canaries, unchanged.
- Active-row Jaref, D, frictionloss, force and **int32 state** overrides,
  preserving inactive tails and original row routing/types/counts.
- The complete dense-force output canary.

The new typed copy path uses canonical `<f4` or `<i4` CPU views according to
the frozen `SPECS`, not one float32 cast for all fields. Validate the complete
destination bank and stage all thirteen pinned host arrays before the first
H2D copy. All host buffers must be mutually disjoint and disjoint from all
destination spans. Check exact whole staging bytes, logical and host shapes,
dtypes, pointers, stream handle and held copy/sync/numpy/guard entries before,
between and after copies. Never call numpy on a CUDA array. A preflight failure
in the late state field produces no control copies. A later failure cannot
produce a successful staging receipt; it does not promise rollback.

The child records the typed staging receipt only in control; reference staging
is null. Actual complete `init_cost.before` packets, not this receipt alone,
must match the sealed historical bank plus exactly thirteen overrides.

## Admission and closeout remain separate gates

The native owner can run only from the clean exact WSL worktree
`/home/yanbo/work/microduck_rl-com-entry-20261006`, on the exact committed
feature source. It hashes all committed blobs and checks the four-path fence.
It requires matching nineteen-file Mac/native CPU XML evidence bound to that
exact source before reading/acquiring the existing FilmBrain lock. Documentation
closeout commits change source identity too: rebind CPU evidence at the final
launch revision, never relabel an earlier report.

Keep the frozen `mjlab==1.3.0`, Torch 2.9.1, Warp 1.12.0, MuJoCo 3.10.0,
MuJoCo Warp 3.8.1, Triton 3.5.1 stack. Authenticate the exact runtime libraries,
source trees, installed solver/cache utility and disassembler before spawning
an inherited-lease fresh CUDA child; private caches must be absent before CUDA
initialization. Preserve compiled plant descriptor and dense 64-world recipe.

Caps are inherited unchanged: service 480 seconds, child 240 seconds, probe
25 seconds, cleanup 5 seconds, closeout reserve 120 seconds, margin 60 seconds;
6 GiB system memory, 200 percent CPU, Nice=10, TasksMax=64, 16 MiB file limit,
KillMode=control-group, no restart. Fresh deadline must include all reserves.
Shared GPU gates: combined usage at most 12288 MiB, reserve at least 10240 MiB,
growth at most 2048 MiB, temperature at most 65 C and utilization at most
85 percent on both Windows and WSL. Recheck the same protected services and
FilmBrain PIDs/restart counts during supervision. Do not claim exclusive GPU
ownership; hold the existing lease or refuse admission. Do not stop workloads.

The external receiver authenticates the complete retained leaf inventory before
any record decode. It requires exact protocol/control/restoration/staging,
runtime, observed module/hook/artifact binds, all four offline checks, bounded
telemetry, successful joined child and externally obtained same-invocation
retired unit/absent-cgroup evidence. It authenticates all sixteen banks before
decoding any, binds both exact initial banks, then reuses unchanged stage and
row-cost analyzers. Require zero reference row mismatches and the control's
2944 force/state replacements, exact dyadic aggregate costs, dense changes in
every world and Gauss=10. No cross-arm output equality is imposed. Synthetic
fixtures can satisfy these contracts; this is not independent native proof.
The source declaration has a closed commit/tree/branch/leaf schema, unique
canonical paths, complete digest formats and all four reviewed paths. Generated
native inode/time records must have five valid integer fields and size equal
to authenticated artifact bytes. Their inode/timestamps remain observed native
metadata: copies on Mac have different identities and cannot independently
prove the native inode. The unchanged executable checks actual native identities
before and after explicit load; receiving declarations does not reconstruct a
Git tree or read driver-resident machine code.

## Focused verification and next execution gate

Run the nineteen files in `stance_solver_efc_probe.TESTS` on Mac and WSL with
`CUDA_VISIBLE_DEVICES=''`; retain XMLs in
`artifacts/tools/efc-native-preparation/`. Require matched collection and zero
failures/errors/skips. Tests cover late-field typed preflight, aliases, code and
stream drift, all sixteen hashes before decode, independently received changing
arms, full mock native envelopes, external retirement substitutions, staging
receipt tampering, unchanged constructor contracts, inert imports and no runtime
patches. This is not the whole repository suite or CUDA execution.

Before the actual native test: bind paired evidence to the exact final source,
recheck clean host/runtime/services/lease/capacity and absence of other Duck jobs,
declare a fresh deadline and unique capped user unit, then execute the two arms
sequentially. After completion, collect external retirement and the complete
native inventory, receive it independently on WSL and Mac, retain exact hashes
and document the result. A failed gate is diagnosed read-only before changes.
Broader solver-window and plant/motor validation still precede curriculum jobs.

## Retained CPU preparation evidence

Tested source `a97d20921e3875eb4e5d85cdd3cca5c8a7913ac1`, tree
`e50a82f8112517dda1e172d2fd3ce4910da7f0cb`, was committed and pushed before
the paired checks. The verified incremental bundle fast-forwarded the clean
exact WSL worktree. Its actual native source-binding function verified all
980 committed blobs, the new four-path fence and thirteen derived controls;
this read-only check acquired no GPU lease and imported no CUDA runtime.

**857 tests in nineteen focused files passed on both Mac and WSL**, including
68 new staging/receiver cases, with CUDA hidden and zero failures/errors/skips.
Mac pytest reported 115.51 seconds and WSL 69.00 seconds. Not the full suite,
GPU execution, plant stability or learned-policy evidence. The read-only Luna
interface/review slice confirmed typed int32 staging and independent-arm gates;
it prompted closed source-leaf validation and generated-identity size/schema
guards with reanchored tampering cases. Owner reviewed and integrated the fixes.

Paired XMLs retained on both machines under
`artifacts/tools/efc-native-preparation/`:

- `a97d2092-mac-tests.xml`: 938689 bytes, SHA256
  `a6def315bee666371e86797492e48bd71d8c0faf066dc964dd2748fc76ed62aa`.
- `a97d2092-wsl-tests.xml`: 938692 bytes, SHA256
  `d308ccdbb9d64817417c598fce0147da97c437ea2823f129088652ebad4ef318`.
- `a97d2092-source.bundle`: 21302 bytes, SHA256
  `a8c9852180078fe0c4f41b4f4483a696a7cdee578ef7ab2e4f2a44d6a9de8759`.

The CPU-only unit `microduck-efc-native-prep-cpu-a97d2092.service`, invocation
`db99f74ea03f43e99ed5ee43b2633153`, retired successfully: MainPID=0,
ExecMainStatus=0, Result=success, NRestarts=0, empty ControlGroup and absent
original kernel cgroup. Its active/exited state is retained metadata, not a
running job. Checked CPU caps were 300 seconds, 4 GiB memory, 200 percent CPU,
Nice=10, TasksMax=64, LimitFSIZE=16777216, KillMode=control-group, no restart.
At this CPU-preparation closeout no Duck user unit remained running; the native
collector itself had not run.

The initial runtime/source-tree/library/tool verification passed unchanged.
Before and after validation, FilmBrain observatory PID 521 and video playground
PID 298048 stayed active with NRestarts=0. Both protected AI mission services
stayed inactive in user and system scopes; no unrelated service was changed.
Post-check Windows and WSL each reported the pinned GPU/driver, 8195 MiB used,
15967 MiB free, 2 percent utilization and 34 C. These are point-in-time separate
counters, not admission for a future launch.

At this CPU-preparation closeout the native run was still pending. That
documentation-only closeout changed source identity; regenerate/rebind paired
CPU prerequisites at the final launch
revision, then recheck the fresh runtime, services, existing lease and capacity.
Never substitute this earlier test source into a later owner's declaration.

## Native execution and independent closeout, 2026-10-09

The user separately authorized the capped positive EFC diagnostic on 100.98.
Launch source was `7717b4cbef3aad06de6c3bf33463076f067e4253`, tree
`79cc1bc242ee1c19e1816a122c414af3aea32cc1`, with a clean exact feature branch
on both machines. The native source check verified all 980 committed blobs.
The runtime/library/source/tool checks passed unchanged. No code or runtime
was edited during this execution or closeout; this addition is documentation only.

Paired CPU prerequisites were rerun at that exact launch source: **857 tests
in nineteen files passed on each host**, with CUDA hidden and no errors,
failures or skips. Mac took 101.18 seconds; WSL took 71.13 seconds. Files are
retained on both machines under `artifacts/tools/efc-native-preparation/`:

- `7717b4cb-mac-tests.xml`: 938689 bytes, SHA256
  `96f4c6d87e77a58298b25bf865e59a2b8405f7aac9f96f39ad289a62a7fd4a23`.
- `7717b4cb-wsl-tests.xml`: 938692 bytes, SHA256
  `7a30390777cb1751d67bb030ea6d3e50941e562dd1c44c38f853ca9078c0ffe6`.
- `7717b4cb-cpu-evidence.json`: 1527 bytes, SHA256
  `e63acf4976fbe374f1a715d46128fb59ef0ab0694a972e6d858393ded5ab12d9`.

CPU unit `microduck-efc-launch-cpu-7717b4cb.service`, invocation
`2d71d7b51d254b8eb7f81312845e6fa4`, finished with MainPID=0,
Result=success, ExecMainStatus=0, NRestarts=0 and empty ControlGroup. Its
original kernel cgroup was independently observed absent. TimeoutStartUSec
was 5min (Type=oneshot); memory/CPU/task/file/kill/restart caps matched those
in the preparation CPU check. The whole post-state observation is retained below.

Native unit `microduck-efc-transition-7717b4cbef3a.service`, invocation
`1a27fe59e3b340ceb6313a15ffea2a19`, started at 09:37:35 Asia/Shanghai.
Owner PID was 336775; joined child PID was 336811, returncode=0, with
22.656911396421492 seconds of supervised child elapsed time. All declared
native caps remained unchanged. The final external systemd observation had
MainPID=0, Result=success, ExecMainStatus=0, NRestarts=0, empty ControlGroup,
active/exited, and the same invocation. The original kernel cgroup
`/user.slice/user-1000.slice/user@1000.service/app.slice/microduck-efc-transition-7717b4cbef3a.service`
was absent, not merely inferred from the child supervision receipt.

Complete native output is retained on both machines at
`artifacts/evaluations/efc-transition-7717b4cbef3a/`: **46 regular files,
64133906 bytes**, including all private cache leaves, three generated module
groups, four offline SASS outputs, logs/records, and all sixteen boundary banks.
Banks are 3729992 bytes each, totaling 59679872 bytes. No symlinks, omitted
leaves or `failure.json` were accepted. External inventory and retirement,
plus both independently recomputed receivers, are retained on both hosts at
`artifacts/tools/efc-transition-closeout-7717b4cbef3a/`:

- `inventory.json` SHA256
  `bac5c1f3958504649db5d443640e47782b8b16110d6be7a38e63102afcfcee4f`.
- Native `declaration.json` SHA256
  `02a5134f2903084cc1e53ad10a37e74844cb9b179e5467f7b2eba3a40d1f8932`.
- Native `receipt.json` SHA256
  `16e36fcd94eb2a5209a756519a10613e0edc08f8e1936235c2b15aba5b173817`.
- `retirement.json` SHA256
  `5222b93af19c5826d5f133b873601597af3dffc6eed52183a74d70c4010ebeba`.
- `wsl-receiver.json` and `mac-receiver.json`: identical 119001 bytes,
  SHA256 `bb493d31cdc47d47248e851629a9c4761e754346a78098c2360eddcb2fc38f7b`.
- `post-state.json`: 1805 bytes, SHA256
  `21d8b77deb93cf9faee4643684bc3151fede1619f9ee10c66634cfe1abdbc69b`.

Both receivers authenticated complete files before decoding and checked all
sixteen packet hashes before their interpretation. WSL used
`stance_solver_efc_probe.retained.historical_packet()`; Mac used
`stance_solver_efc_transition.historical(Path.cwd())` over the already retained
historical files. Each called `stance_solver_efc_receiver.receive` with the
local copied run directory and the same externally obtained native inventory
and retirement. The Mac replay is an independent pure recomputation, not a
second live observation of the native cgroup or native inode identities.

| Retained check | Reference | Positive control |
| --- | --- | --- |
| Worlds / active EFC rows | 64 / 2944 | 64 / 2944 |
| Force / state mismatches | 0 / 0 | 0 / 0 |
| Force / state replacements | 0 / 0 | 2944 / 2944 |
| Cost outside declared interval | 0 | 0 |
| Dense DOFs / mismatches | 1280 / 0 | 1280 / 0 |
| Maximum dense absolute error | 4.789326339960098e-7 | 0 |
| Gauss outside declared bound | 0 | 0 |

Control covered all five row branches: both friction-linear signs,
friction-quadratic, unilateral-quadratic and unilateral-satisfied. Every
control world changed dense force as required; all 64 aggregate EFC costs
equaled the exact dyadic model (range 24.5–25.75), and every control Gauss
was 10. Reference Gauss was 3.963656371842135e-15. Both arms preserved
inactive tails, adjacent boundary continuity, setup reset and prior-cost
transfer; only stage-declared fields changed. Cross-arm output equality was
intentionally not required.

Thirteen retained telemetry samples passed shared-capacity gates. Across
Windows/WSL counters, peak combined usage was 8459 MiB, minimum free was
15703 MiB, peak sampled utilization 3 percent and temperature 36 C. Samples
are not simultaneous and do not prove transient peaks absent. FilmBrain
observatory PID 521 and video playground PID 298048 stayed active with
NRestarts=0; protected AI mission services stayed inactive in both scopes.
The external post-state found no running Duck unit. No unrelated workload
was stopped, restarted or changed, and no exclusive GPU ownership was claimed.

Decision: `positive-efc-native-diagnostic-received-not-qualification`.
This establishes the retained two-arm diagnostic under its declared arithmetic
models and observed loader/caller/artifact envelope. It does **not** read
driver-resident code, identify atomic order/device rounding, prove runtime
cause or qualify a complete solver window, motor/plant, policy or robot.
All six qualification flags remain false. No training or video was launched.

Closeout verification: all documented artifact hashes, XML counts, complete
inventory counts/bytes and identical receiver bytes were checked against the
retained files. The focused 68-case probe/receiver test file passed again on
Mac (12.45 seconds) and WSL (8.89 seconds), with CUDA hidden; `git diff --check`
passed. The full repository suite was not rerun. These later checks do not
relabel the source-bound launch XMLs or grant admission to another run.

Next bounded gate: predeclare a separately fenced enclosing solver-window
diagnostic with independent expected outputs and explicit rejection criteria,
before any new native execution. Then complete plant/motor acceptance before
resuming curriculum jobs. Do not extend this four-stage positive control result
into full-step or learned-capability acceptance.
