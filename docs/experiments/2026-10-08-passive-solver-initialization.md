# Passive solver-initialization diagnostic, 2026-10-08

Status: implementation and CPU review in progress; no native result or training
admission. This is one bounded diagnostic, not a new learning campaign.

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

Full same-source Mac/native CPU receipts and the native result are pending.
The focused three-file suite passed **63 tests in 5.37 s** on Mac; Ruff and
diff whitespace checks passed. Independent read-only review found no remaining
code blocker, conditional on the full same-source prerequisites below.
A failed CPU guard prevents GPU launch; a failed native diagnostic is diagnosed
read-only and preserved before any subsequent revision.
