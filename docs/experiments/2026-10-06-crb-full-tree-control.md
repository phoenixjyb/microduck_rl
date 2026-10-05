# October 6 whole-tree CRB schedule investigation

## Scope and starting evidence

Continue only within the authorized window ending **2026-10-06 08:00
Asia/Shanghai (00:00 UTC)**. Native work is restricted to 100.98 via
`gw98-direct`; 100.100, packages, drivers, installed kernel source, optimizer,
raw perception, physical motion and protected-service restores are excluded.
The original full-window replay remains rejected. Preserve its exact gates and
all retained evidence; a source planner or isolated control cannot admit training.

Start from clean fork branch `feat/athletics-obstacle-curriculum` at
`f98a694033a5402e4c9afb2f87f1ff21c7112a6f`. The preceding
[serial control](2026-10-06-crb-serial-schedule-control.md) at execution source
`c9708cf68fdb812733f7a9fddd336c36a8241d26` produced 32 byte-identical outputs
matching the independently predeclared CPU candidate. The concurrent control
produced seven distinct outputs from the same derived fixture. That fixture
contains root inertia and three completed-forward child composite inertias;
other bodies are synthetic zeros. It is not actual kernel-entry capture.

The next source-only slice permits exactly three new paths: this declaration,
`src/mjlab_microduck/stance_crb_level_plan.py` and
`tests/test_stance_crb_level_plan.py`. All previous source leaves stay unchanged.
The native checkout remains at C970 until any separately declared source sync.
No new native run is authorized by this planner's source tests alone.

## Fixed topology and schedule

Use the same pinned `_crb_accumulate` kernel contract: read the body's parent,
return if that parent is zero, otherwise atomically add the body's current
ten-component composite inertia to its parent's working value. CRB input and
output alias the same working array. Children must be accumulated before parents.

The exact seven-field canonical topology JSON plus newline has SHA256
`2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19`:
64 worlds, 16 bodies, `nq=21`, `nv=20`, `nu=14`, parent vector
`[0,0,1,2,3,4,5,1,7,8,9,1,11,12,13,14]`, and these reversed levels:

| Original level | Fixed launch groups | Nonzero parent targets |
| --- | --- | --- |
| `[6,15]` | `[6,15]` | `[5,14]` |
| `[5,10,14]` | `[5,10,14]` | `[4,9,13]` |
| `[4,9,13]` | `[4,9,13]` | `[3,8,12]` |
| `[3,8,12]` | `[3,8,12]` | `[2,7,11]` |
| `[2,7,11]` | `[2]`, `[7]`, `[11]` | `[1]`, `[1]`, `[1]` |
| `[1]` | `[1]` | none; parent-zero no-op |
| `[0]` | `[0]` | none; parent-zero no-op |

Preserve every original level boundary, stable body-ID order and both no-ops.
Only level four requires splitting: its three children target the same parent.
Nine launches are minimal **within those level boundaries**, because that level
requires three distinct single-writer rounds. Do not claim global optimality
across fused/reordered levels, deterministic execution of other kernels, or
unchanged timing. Unique logical write destinations are a schedule property,
not observation of actual atomic order.

## Bounded planner and acceptance

Before hashing or traversing, require a plain dictionary with exactly seven
fields, plain integers (not booleans), exact dimensions, a 16-element parent
list and seven nonempty level lists with at most 16 IDs each. Then authenticate
the canonical whole-topology hash and reuse the frozen exact-topology validator.
No input-dependent unbounded search or alternate topology is permitted.

Return detached, owned lists and explicit witnesses for all 16 bodies occurring
once, child-before-parent order, unchanged original boundaries, stable sibling
order, unique nonzero parent targets in each group, retained no-op groups, and
the per-level lower bound. Check the exact nine-group output independently in
tests. Reject extra/missing fields, altered hashes, dimensions/parents/levels,
booleans, oversized lists and duplicate/missing IDs. Test that caller mutation
cannot change an already returned plan and output mutation cannot change input.

Actual launch execution, actual atomic-order observation, original-pair
acceptance, cause proof, full-window qualification, training authorization and
physical acceptance remain false. Source tests establish only the planner's
contract. A future whole-tree control requires a separate declaration, exact
source/pinned runtime, authenticated whole-cinert fixture provenance, CPU
prediction, cooperative lease, caps, independent checking and retained results.

## Results

The source-only planner passed **24 focused CUDA-hidden tests**. An independent
read-only worker repeated that file successfully. Owner integration with the
unchanged serial checker/supervisor passed **45 tests in 6.28 s**; Ruff lint and
format checks and `git diff --check` passed. The routing skill separated the
bounded implementation from independent review and owner integration.

No whole-tree native control, fresh compilation, array-alias witness, replay
qualification or training result is claimed by these checks. The native source
and completed serial artifacts remain unchanged at C970.
