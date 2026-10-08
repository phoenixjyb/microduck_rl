# Initialized solver row side view, 2026-10-08

Status: CPU-only view completed on Mac and native at the same committed source;
initialized row-local fields match, while aggregate buffers differ.
No GPU job, learner update or new physics capture is authorized by this checker.

## Scope and evidence boundary

Base `4011fdc5e277c06e3bd4975707f5edfcd0073feb`, feature branch
`feat/athletics-obstacle-curriculum`. Exactly three new paths: this document,
`stance_solver_init_row_view.py`, and its matching test file. Historical source,
owners, receivers, fences, cutoffs, captures and packages are unchanged.

Reuse the [initialized capture](2026-10-08-passive-solver-initialization.md)
at source `e45a59c412bfe6bcf2de751f98159aa7fe0db463`: 480 whole leaves /
673,600,544 bytes, externally fixed raw inventory SHA256
`8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625` and
receiver SHA256
`c91d812df1278d1cfe208951d8dad6f23f5240639989d56407a6ca38ed08f999`.
Reconstruct the historical whole source binding from Git objects, authenticate
every raw leaf, and recompute the complete existing receiver before decoding
the new side view. Bind analysis separately to its clean committed source.
Run with CUDA hidden and no GPU imports on both Mac and WSL; compare canonical
output byte-for-byte. Keep generated evidence under a fresh ignored tools root.
The inherited numerical receiver uses NumPy on CPU; this is not a
standard-library-only checker. Torch, Warp and MuJoCo are not imported.

## Predeclared comparisons

At initialized forward four, compare **active** `context.Jaref`, `efc.force`
and `efc.state`. Pair contacts only through unique captured payload bytes and
per-arm current address/id/type backlinks, then local row ordinal. Pair other
supported rows only by equal id/type words at the same original world/row.
Never search/rewrite/sort row banks. Duplicate, unmatched and unobserved payloads
remain excluded. Done worlds are excluded. Report active, covered and uncovered
counts; partial coverage is inconclusive; no comparisons mean null, not true.
Inactive capacity is not initialized-row evidence. Preserve raw NaN payloads,
signed zeros and integer state words without numeric tolerance.

Keep world/DOF, matrix and scalar fields in a **separate original-offset full
carrier view**, including `efc.Ma`, constraint force, cost, gradient and hessian
buffers. Whole-carrier comparison does not establish fresh writes of every
padding word. Match neither these aggregates nor matrices using contact rows.

These are associations of observed storage, not physical contact identity,
complete initialization driver equality, CUDA cause or solver equivalence.
Uncaptured contact parameters stay explicitly listed. All five qualification,
training and physical-acceptance flags remain false regardless of equality.

## Frozen source review and next capture boundary

The frozen `solver.py` SHA256 is
`bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
`init_context` lines 3301–3337 reset solver bookkeeping, initialize active
`Jaref`, compute `Ma`, call `_update_constraint`, then `_update_gradient`.
The retained packet is therefore post-constraint **and** post-gradient.

For this nv=20 dense Newton/pyramidal recipe, `Jaref` is written with one
thread per row; inactive rows are untouched. `_update_constraint` updates active
force/state, per-world cost, dense constraint-force reductions and Gauss cost.
`_update_gradient` computes gradient, the dense hessian and Cholesky-preconditioned
gradient. Row-local outputs and reductions must be evaluated separately.
These source facts identify useful boundaries, not the numerical cause.

Only after reviewing the retained row view should a new passive capture be
predeclared. The smallest candidate brackets pre-constraint and post-constraint /
pre-gradient, retaining the existing post-gradient endpoint. Two additional full
5.375-MB snapshots would exceed the 8-MiB-per-arm budget; use reviewed compact
driver/output subsets with same-forward layout and row anchors, or explicitly
predeclare a new bounded budget. Relevant missing parameters must be captured or
source-supported exclusions, not silently assumed equal. No such GPU run is
launched by this work.

## Checks and retained outcome

The new 34-test suite covers original-offset payload permutation, broken
backlinks, duplicate payloads, same-offset noncontact exclusions, done/empty/
inactive coverage, signed-zero/NaN/state bytes, literal ABI and packed offsets,
whole-file anchors, nested historical Git blobs, source-root refusal and CUDA /
optimized-Python guards. Combined with the existing initialized receiver suite,
48 focused tests passed. Ruff and local link/whitespace checks passed. Independent
read-only review also passed all 34 new tests and found no remaining blocker;
the owner reviewed its findings and diff. A combined-short-option review concern
was disproved by a live recursive 918-blob Git-tree check; explicit flags and a
nested-file fixture now make that contract clearer.

Executed analysis source on both hosts:
`e1f6b96189da4b2e1dffc9f22269a9c9d6fe8864`. This outcome document is a later
documentation-only closeout, not the source used to execute the checker.
The final precommit combined suite passed 48 tests in 21.27 s. Same-source CPU
owners again passed 48 tests per host, zero failures/errors/skips, then each
authenticated all raw leaves and recomputed the complete historical receiver.
The old complete 3206-test suites were not rerun: this is an additive pure side
view, with no GPU launch or change to the already executed owners/receivers.

Native retained service: `microduck-initialized-row-view-cpu-e1f6b96189da.service`,
invocation `7958b595c2f74a37857e89c0dc16be6c`, owner PID 4140306. Journal span
**2026-10-08 09:58:22–09:58:32 Asia/Shanghai**. Terminal: `MainPID=0`,
`Result=success`, `ExecMainStatus=0`, `SubState=exited`. CPU owner budget 420 s,
test child 90 s, replay child 300 s, memory ceiling 6 GiB, CPU 200%, tasks 64,
nice 10, restart no, control-group kill; CUDA hidden and all four numerical
thread environment settings one. Observed owner spans: Mac 34.239538 s,
native 8.840212 s. No GPU workload was started.

Local native cgroup `MemoryPeak` was reported as **not set** after exit; do not
claim a measured peak. The enforced memory ceiling is a separate fact.

Retained directories, relative to each exact worktree:

- Each host's own original: `artifacts/tools/initialized-row-view-cpu-e1f6b96189da`.
- Native proof copied to Mac: `artifacts/tools/initialized-row-view-native-e1f6b96189da`.
- Mac proof copied to native: `artifacts/tools/initialized-row-view-mac-e1f6b96189da`.

Each directory retains the canonical report, whole-anchoring proof, pytest log,
JUnit and replay stderr. Reports are **byte-identical** across hosts. Whole
proofs and their exact leaf sets were authenticated after transfer.

| Whole artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Mac/native `report.json` | 185,413 | `6ba82c0c066df014ddff8f7bb7b8c2c5c5af648e6a1ee3233196d0e88c85aa92` |
| Mac `proof.json` | 1,038 | `271cb6923a690ecd00a542832b4e62eca9a86112e2d81ae6a04f2be58ee8ee41` |
| Native `proof.json` | 1,072 | `f6559bca741d54f9d0538b04d59d4bb141bdc48a2e77aee3c46441ce9711f91b` |

## Retained numerical result

The initialized view independently reconstructs **complete active storage-row
coverage: 2944/2944 in each arm**, zero uncovered rows, zero ambiguous/unmatched/
unobserved payload-link groups and no excluded done worlds.

| Active initialized field | Contact words | Noncontact words | Different words |
| --- | ---: | ---: | ---: |
| `context.Jaref` | 2,048 | 896 | 0 |
| `efc.force` | 2,048 | 896 | 0 |
| `efc.state` | 2,048 | 896 | 0 |

These 8,832 comparisons use the separately retained **original** row offsets;
literal row banks still have different order. At original world/DOF offsets,
`efc.Ma`, qM, positions/velocities, acceleration/warm-start/smooth buffers,
smooth force, `context.gauss` and `context.prev_cost` all match. Aggregate
differences remain: constraint force **477/1280** words, cost **52/64**, gradient
**477/1280**, gradient norm **52/64**, preconditioned gradient **1234/1280** and
hessian **4645/25600**. Whole direct-carrier equality/difference is not a claim
about fresh writes to every padding word.

Interpretation: the prior raw initialized force-bank mismatch is **not** a
paired active force mismatch in this capture. The initialized row-local subset
matches; some aggregates do not. This narrows the proposed investigation but
does not prove a CUDA cause, full solver-driver equivalence or physical contact
identity. Uncaptured parameters and instrumentation effects remain excluded
from such claims. The one-tick repetition gate is still negative and all five
qualification/training/physical flags remain false.

## Next bounded step: accumulation-order sensitivity

The earlier broad pre/post-constraint capture was a proposal, not launched.
This retained outcome supports trying a **CPU-only arithmetic side view first**.
Frozen source facts: dense constraint-force computation visits active rows in
storage order (`solver.py` 1988–2010); constraint cost uses atomic additions;
dense hessian computation uses a tiled reduction (`2923–2968`). The two arms
have different stored contact-row order. Sensitivity of finite-precision
accumulation to those orders is therefore a **hypothesis**, not a proven runtime
cause. No solver input may be sorted or rewritten on that inference.

Smallest proposed arithmetic experiment:

1. Authenticate the same complete capture and source again. Use only active
   captured dense J / initialized force terms, existing row links and original
   world/DOF offsets; reject nonfinite inputs rather than replace them.
2. Predeclare explicit round-to-nearest-even binary32 arithmetic hypotheses,
   including separate multiply/add versus fused multiply-add. Preserve signed
   zero and subnormal behavior; host NumPy output alone is not a compiled-GPU
   arithmetic oracle. Compare exact model bits with observed constraint force.
3. In a separate **analysis-only** view, compare corresponding term sets under
   the two recorded storage orders and a shared linked-row order. Never restore,
   normalize, canonicalize or mutate simulation state. Declare exactly which
   order and arithmetic model each result uses.
4. Report match counts, first differing words, nonfinite/exclusion coverage and
   hypotheses not reproduced. Even an exact modeled reproduction demonstrates
   consistency/sensitivity, not the actual compiler's instructions, atomic
   schedule, causal mechanism or training qualification. Cost and tiled hessian
   require their own protocols; do not infer them from the serial force model.

This arithmetic experiment is predeclared here at design level, **not implemented
or executed yet**. It may remove the need for another broad GPU snapshot. If its
coverage is insufficient, return to compact passive phase bracketing with a
fully specified budget. Do not start a long learner or promote a skill from this
diagnostic. No physical duck is present and no motion, perception training or
video was authorized or run.
