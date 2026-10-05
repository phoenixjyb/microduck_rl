# October 6 source-only forward-reduction plan

## Scope and evidence

Start at `4430dee81fed4b0e8f51251bb2bc5a9ea09e3f29` on
`feat/athletics-obstacle-curriculum`. Permit only this declaration,
`src/mjlab_microduck/stance_forward_reduction_plan.py`, and its focused test
file. Preserve all earlier source and evidence. Native 100.98 remains at the
runtime execution revision `2ecee471f7b999822ed19defd7e2d1c7ad09df07`;
no native source sync or GPU job is authorized by this planner.

The [authenticated partial runtime result](2026-10-06-crb-runtime-schedule-control.md)
does not establish CRB-only exactness: serial paired constructor bias-force
fields differ, and later serial forwards vary upstream CoM/inertia fields.
Those observations identify a useful investigation boundary, not a proven
kernel cause. Unchanged kinematics is not a complete scratch reset, and the
runtime adapter perturbs timing. Keep the original full-window rejection.

This slice produces declarative groups only: no native imports, model compile,
array allocation/readback, kernel launch, hook, integration, actor, graph,
optimizer, perception, video or physical motion. Do not change installed
dependencies, 100.100, FilmBrain or protected services. Stop new work at
2026-10-06 08:00 Asia/Shanghai.

## Distinct activity rules

Pinned MuJoCo Warp 3.8.1 `smooth.py` has whole SHA
`63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f`.
The source plan declares this expected hash; it does not freshly authenticate
an installed file or prove execution of it.

| Reduction | Symbol | Active body predicate | Root-target rule |
| --- | --- | --- | --- |
| CRB | `_crb_accumulate` | parent ID is nonzero | bodies 1 and 0 are no-ops |
| Subtree CoM | `_subtree_com_acc` | body ID is nonzero | body 1 actively writes root 0 |
| RNE backward | `_cfrc_backward` | body ID is nonzero | body 1 actively writes root 0 |

RNE calls `_rne_cfrc_backward` before `_qfrc_bias`. This source relation does
not prove that the observed bias difference originated in that reduction.

Use only the frozen canonical 64-world/16-body topology hash
`2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19`.
Reuse its existing bounded topology authenticator, not CRB-specific no-op or
nonzero-target witnesses. Independently derive each reduction's active bodies,
inactive bodies and targets. Partition each original level into stable minimum
single-writer rounds, counting target zero when active. No alternate topology
or arbitrary predicate callback is accepted.

For this topology, all three plans have nine groups: only `[2,7,11]` splits
into three singletons. CoM/RNE retain active `[1] -> [0]` followed by inactive
`[0]`; CRB retains both as no-ops. Verify every body appears once, level/order
boundaries are unchanged, active children precede their parent including root
zero, targets within a group are unique, per-level grouping attains its bounded
lower bound, and returned data is detached. This is minimal within original
level boundaries, not a global optimization or runtime determinism claim.

All execution, installed-source authentication, causal proof, original-pair
acceptance, full-window, training and physical-result flags remain false.

## Validation

The planner adds 40 CPU-only structural tests covering malformed inputs,
independently recomputed witnesses, root-zero contrasts and detached data.
The new tests plus the existing level planner passed together: `64 passed`.
The broader level-plan, full-tree, kernel-repeat, runtime-control and
runtime-probe contract check passed: `184 passed in 10.67s`. Owner review,
Ruff and formatting checks passed. The linked retained-runtime document exists.
These are source/CPU checks only; no fresh native test or execution is claimed.
A future runtime CoM/RNE control would need its own predeclared input/scratch
contract, dispatch/source/array/stream checks, timing disclosure, bounded native
supervisor, independent receiver and exact numerical gates. This planner does
not authorize that control.
