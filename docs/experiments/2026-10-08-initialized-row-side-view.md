# Initialized solver row side view, 2026-10-08

Status: CPU-only implementation and review; retained numerical view pending.
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

Committed-source Mac/native replay and exact outcome hashes remain pending.
