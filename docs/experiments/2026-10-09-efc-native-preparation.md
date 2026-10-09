# Separately fenced native positive EFC transition preparation

## Scope and protocol boundary

Base `a6d65d9b90650900a6c736913a967caf99ae5638`, exact branch
`feat/athletics-obstacle-curriculum`; the new fence contains only
`stance_solver_efc_probe.py`, `stance_solver_efc_receiver.py`, their focused
test file and this document. Protocols are
`microduck-efc-transition-probe-oct9-v1` and
`microduck-efc-transition-receiver-oct9-v1`. No old source fence, receiver,
installed runtime, driver or retained artifact is modified or monkeypatched.

This slice prepares a collector/receiver and tests it with hidden CUDA. It
does not launch the native GPU test. The older cost-only receiver intentionally
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
No Duck user unit remains running; the native collector itself has not run.

The initial runtime/source-tree/library/tool verification passed unchanged.
Before and after validation, FilmBrain observatory PID 521 and video playground
PID 298048 stayed active with NRestarts=0. Both protected AI mission services
stayed inactive in user and system scopes; no unrelated service was changed.
Post-check Windows and WSL each reported the pinned GPU/driver, 8195 MiB used,
15967 MiB free, 2 percent utilization and 34 C. These are point-in-time separate
counters, not admission for a future launch.

The native run is still pending. This documentation-only closeout changes
source identity; regenerate/rebind paired CPU prerequisites at the final launch
revision, then recheck the fresh runtime, services, existing lease and capacity.
Never substitute this earlier test source into a later owner's declaration.
