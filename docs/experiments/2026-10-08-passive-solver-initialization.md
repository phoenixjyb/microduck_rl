# Passive solver-initialization diagnostic, 2026-10-08

Status: bounded native capture completed and independently replayed on Mac;
candidate repetition remains negative. No training admission. This is one
bounded diagnostic, not a new learning campaign.

## Question and retained baseline

The separately retained contact-boundary capture at source
`9de88d19f3ee43f93eaba1cd9319eeca60713de9` completed but candidate repetition
was negative. Its whole raw inventory SHA256 is
`84fe5ee296cce8022ade62a4378b6a027c7b1eaedbd18887ff635bfc61467196`;
the native and independently recomputed Mac receiver SHA256 is
`434f569bb45606339b743a15c3d39da97ee50cc949caa675543e360cacf315f5`.
The full-row interpretation at tested source `d04d069507beeb0f762cc62b527c30a64a49107a`
covered all 2944 forward-four active rows: 2048 uniquely payload-linked contact
rows and 896 authenticated same-offset noncontact rows. All 79488 words of the
eight captured construction fields matched, but 1986/2048 linked post-forward
force words differed. See the [retained analysis](2026-10-08-passive-contact-boundary.md).
This establishes a captured phase boundary, not all solver inputs, physical
contact identity, or a CUDA cause.

## Predeclared owner and scope

New owner protocol: `microduck-solver-init-tick-oct8-v1`.
New receiver protocol: `microduck-solver-init-receiver-oct8-v1`.
New control protocol: `microduck-solver-init-control-oct8-v1`.
Base: `9814ceb8bbeaf656630dd28b97f38e190207a307`; exact branch:
`feat/athletics-obstacle-curriculum`. The source delta consists only of the
three `stance_solver_init_*` modules, their three tests, and this document.
Historical owner protocols, source fences, deadlines, captures and caches are
not modified or reinterpreted as admission.

Authorized host: `gw98-direct`, exact lean root
`/home/yanbo/work/microduck_rl-com-entry-20261006` on 100.98. Absolute owner
cutoff: **2026-10-08 11:00 Asia/Shanghai**, `2026-10-08T03:00:00Z`.
Reserve the entire child/closeout budget before starting. CPU owner 660 s,
CPU suite 600 s, GPU owner 600 s, child 540 s, closeout 240 s, extra margin 60 s.
One inherited exclusive existing Wan GPU lease, one child, no concurrent GPU
workload. Preserve both FilmBrain user services; both protected AI mission
services remain inactive. Never change packages, drivers, aliases or old caches.
CPU and owner imports hide CUDA. All four declared numerical thread settings
are one at process startup. Limits remain 6 GiB, CPU 200%, 64 tasks, nice 10,
restart no, control-group kill, file cap 16 MiB GPU/64 MiB CPU.

## Observation boundary and limitations

Retain the same fresh three-arm constructor plus one zero-action nominal
ten-substep tick: original, candidate0, candidate1. The existing ascending
friction intervention and contact capture are unchanged. For each arm, call
the frozen `solver.init_context(model, data, context, grad=True)` exactly once
at each of its 21 scheduled entries. Only at forward four, after the original
call returns and before solver search/iteration, synchronize and copy the full
declared initialized carriers. Do not invoke a solver/search/iteration from
the observer or restore state. Pin init and solve function/code/module identity,
model/data/context, thread, device/stream and installed source.
Authenticate the actual `_solve -> solve body -> event_scope -> forward body ->
forward event_scope` caller chain. Direct hook or public-solver calls outside
that chain are refused before dispatch. Release frame references before copying.

Frozen solver.py SHA256:
`bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
Frozen forward.py SHA256:
`c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3`;
event-scope warp_util.py SHA256:
`b0d156ddbced4848cd6cbcd70977f1e96c230fce0d746a6acfa911549afb447c`.
Dense 64 worlds, nv=nv_pad=20, njmax=512, contact capacity 8192. Raw scalar
option arrays have host shape `[1]`; no guessed per-world option replication.
Full fields and host/logical layouts are literal exported control specifications.
The measured ABI has 66 fields and 5,375,152 raw bytes per arm, including
initialized `gauss` and `prev_cost`. Singleton broadcast stride zero and empty
canonical Warp C-strides are retained explicitly; nonempty layouts are not relaxed.
Each new snapshot is bounded to 8 MiB, three arms to 24 MiB, and the whole
capture retains the existing 1024-leaf / 768-MiB cap. Preserve inactive/padding,
signed-zero, NaN payloads, and boolean carrier bytes. New path per arm:
`solver-init/{arm}/forward-04.initialized.bin`.

Compare world/DOF carriers only at their original offsets. An exact raw EFC bank
comparison is storage evidence, not semantic contact-row identity. Row-level
claims require authenticated captured payload links, current EFC address and
ID/type backlinks, with unchanged same-offset noncontact identifiers. Empty or
excluded coverage is null/inconclusive, never vacuous acceptance. A partial
capture does not prove complete solver-driving inputs. Added copies/synchrony
change timing; results concern this instrumented recipe, not an uninstrumented
causal mechanism.

## Admission and closeout

Before GPU launch, freeze the complete 97-file CPU collection and exact count,
run it on both Mac and native against the same committed source with zero
skips/errors/failures, authenticate each whole receipt/log/JUnit/inventory,
verify the clean exact branch and source closure, idle GPU and unchanged services.
Use distinct source-derived user services and fresh artifact paths.

Authenticate every raw leaf before interpreting bytes, independently replay on
Mac, retain both reports and hashes, and commit/push the reviewed evidence.
All five admission flags stay false even if the observed initialization matches.
A negative result remains retained evidence. Do not start long training, learner
updates, actor changes, H2/Locked, video, raw perception or physical motion.

## Implementation checks and result

The exact 97-file collection is frozen at **3206 tests**, with file-list SHA256
`c905171b3cd0e1f9f1a2b498deb53aef8f222089589856fa723df1e3d63b0074`.
It contains all 94 prior files plus the three new test files. Tests include real
installed CPU allocation of the plant, data and SolverContext without executing
forward or solve, actual decorated-function constructor binding, adversarial
direct-call/caller checks, hook ownership/closeout, field caps, layouts, singleton
broadcasts, empty allocations, signed-zero/NaN/boolean bytes, exact recipes,
whole hashes and optimized-Python refusal. CPU ABI consistency is not CUDA proof.

The focused three-file suite passed **63 tests in 5.37 s** on Mac; Ruff and
diff whitespace checks passed. Independent read-only review found no remaining
code blocker, conditional on the full same-source prerequisites below.
A failed CPU guard prevents GPU launch; a failed native diagnostic is diagnosed
read-only and preserved before any subsequent revision.

## Executed source and CPU prerequisites

Executed source on both clean feature-branch worktrees:
`e45a59c412bfe6bcf2de751f98159aa7fe0db463`. This closeout document is a later
documentation-only revision, not the source executed by the native child.
Full collection passed with zero failures, errors or skips: **Mac 3206 tests in
258.22 s**, **native 3206 tests in 199.48 s**. Each whole inventory, receipt,
pytest log and JUnit file was authenticated on both hosts before GPU launch.

Retained CPU roots on both hosts, relative to their exact worktree:

- `artifacts/tools/solver-init-tick-mac-tests-e45a59c412bf`;
  inventory SHA256 `cf187f06ff363ca7c3763bd9e5ee8125d28f2350c282f72eb98972a08ac26f60`.
- `artifacts/tools/solver-init-tick-tests-e45a59c412bf`;
  inventory SHA256 `7df590e1a90769cc2326bfe784da60895c7c5169da3347e8ae3ff76b4168d05b`.

Native CPU service `microduck-solver-init-tick-tests-e45a59c412bf.service`,
invocation `fa431c4d117446349da48619b89c46a4`, owner PID 4125964, completed
with status zero. Peak cgroup memory was 4,722,155,520 bytes, below 6 GiB.
Both CPU checks used CUDA-hidden startup and the unchanged frozen environment.

## Native capture and independent closeout

Service `microduck-solver-init-tick-run-e45a59c412bf.service` completed with
`Result=success`, `ExecMainStatus=0`, `MainPID=0`, `SubState=exited`; invocation
`65ea298815ea4ad399dad46b4d141ca9`, owner PID 4131680, child PID 4131701.
Journal span: **2026-10-08 09:29:00–09:29:46 Asia/Shanghai**. There were 18
owner samples, maximum GPU temperature **32 °C**. The post-run GPU had no
compute process, 0% utilization, 659 MiB occupancy and 32 °C. FilmBrain service
PIDs 521 and 298048 remained active; protected AI mission services remained
inactive in both user and system scopes. No package, driver, environment alias,
historical cache or unrelated workload was changed.

The receiver authenticated all 63 scheduled friction entries, 30 BAM proposals
and 21 original initialization calls per arm. Each of the three initialized
snapshots contains the declared 66 fields / 5,375,152 bytes. Raw root on both
hosts: `artifacts/evaluations/solver-init-tick-run-e45a59c412bf`; closeout root:
`artifacts/tools/solver-init-tick-closeout-e45a59c412bf`. The complete retained
raw inventory has **480 leaves / 673,600,544 bytes**. Private compilation caches
were not transported or relabeled as independent compilation proof.

Whole-file anchors (SHA256):

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Raw inventory `inventory.json` | 69,779 | `8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625` |
| Native and Mac receiver | 1,053,121 | `c91d812df1278d1cfe208951d8dad6f23f5240639989d56407a6ca38ed08f999` |
| Native `terminal.txt` | 104 | `93cc0f1c095330b42a6de95606aa87a07c22758041e51c223ce12f3580a7a3af` |
| `transport.json` | 1,042 | `3a1b9d67102fe6abc59378bf21791c4153289439771fe2664f6f0e2d04e8fbf3` |
| `solver-init-closed-e45a59c412bf.tar.gz` | 23,015,138 | `45023d740749585de0ff50f6a612695ba3d4f4fc9c3685aa6e21fe642189d64b` |
| Mac `mac-replay.json` | 4,741 | `9d1258625b04ed1ad2563a77093de209ef6ca294197854ef2e920d9c4386f3e7` |

Mac authenticated the external transport anchor, compressed archive, exact
member set and every whole raw leaf before interpretation. It independently
ran the pure receiver while retaining literal native compiler provenance roots;
the recomputed receiver was **byte-identical** to the native report. The Mac
receiver and replay proof were copied back to native and their whole hashes
rechecked. This is independent byte replay, not a second GPU execution.

## Numerical result and next gate

Decision: `passive-solver-init-capture-complete`;
`recipe_candidate_repeat_exact=false`. At forward four, after initialization
and before solver search, all 2944 active rows are covered for the eight-field
construction subset: 2048 uniquely payload-linked contact rows and 896
authenticated same-offset noncontact rows. All **79,488 compared words match**.
This does not equate literal EFC storage ordering with semantic contact identity.
The raw contact/EFC carrier banks differ, including some row addresses and IDs.

At original world/DOF offsets, full-carrier 32-bit word comparison gives:

| Initialized field | Compared words | Different words |
| --- | ---: | ---: |
| `data.qpos` | 1,344 | 0 |
| `data.qvel` | 1,280 | 0 |
| `data.qM` | 25,600 | 0 |
| `data.qacc`, `data.qacc_warmstart`, `data.qacc_smooth`, `data.qfrc_smooth` | 1,280 each | 0 each |
| `data.qfrc_constraint` | 1,280 | 477 |
| `context.cost` | 64 | 52 |
| `context.grad` | 1,280 | 477 |
| `context.grad_dot` | 64 | 52 |
| `context.Mgrad` | 1,280 | 1,234 |
| `context.h` | 25,600 | 4,645 |

These counts use authenticated packed offsets and raw bytes, with no float
tolerance, canonicalization or state repair. Full raw EFC force/state/Ma bank
booleans remain storage-only; initialized row-semantic force comparison is not
implemented. `context.Jaref` raw ordering also differs and is not a world/DOF
comparison. The underlying frozen `init_context` invokes `_update_constraint`
and `_update_gradient` before returning. The observed mismatch therefore exists
at this captured pre-search boundary, not only after iterative solver search.
It does **not** prove which initialization operation introduced it: ordering,
uncaptured drivers, copied timing and other alternatives are not excluded.

Next proposed bounded diagnostic, not launched here: bracket initialization's
constraint and gradient stages using fresh isolated packets, after reviewing
their full input/output contracts and snapshot budgets. Add authenticated
backlink-safe comparisons for active `Jaref` / initialized force/state; keep
world/DOF reductions separate from row carriers. Capture or explicitly exclude
the missing contact parameters (`friction`, `solref`, `solreffriction`, `solimp`)
before claiming equal initialization inputs. Preserve original invocation count,
recipe, source/device/stream binding, CPU prerequisites and independent replay.
Do not sort/rewrite solver inputs or relax the repeatability gate on this result.

All five flags remain false: native qualification, full-window qualification,
runtime-cause proof, training authorization and physical acceptance. No learner,
long rollout, curriculum promotion, video or physical robot motion was run.

Closeout checks: the focused three-file suite passed again (**63 tests in
76.58 s**); Ruff, diff whitespace and local Markdown link/section checks passed.
Only this experiment document changed after the executed source. The complete
CPU suites were not rerun for the documentation-only closeout.
