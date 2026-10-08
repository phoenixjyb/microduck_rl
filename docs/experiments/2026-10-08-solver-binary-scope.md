# Retained solver binary scope, 2026-10-08

Status: bounded CPU verifier; committed replay and outcome pending.
Base `a0455e2b27584cbec7850b5ed19a3d67dbfe13b1`, exact feature branch
`feat/athletics-obstacle-curriculum`; three new paths only: this note,
`stance_solver_binary_scope.py` and its tests. No historical fence is expanded.

## Why this gate exists

The [qfrc arithmetic model](2026-10-08-qfrc-order-model.md) reproduced both
observed output banks under a fused hypothesis, including 477 row-order
differences. Mathematical reproduction does not identify compiled instructions.
The specific target is
`mujoco_warp._src.solver.update_constraint_init_qfrc_constraint_dense`, not the
friction/contact kernels that precede it. A sibling module's `fuse_fp` option
must not be attributed to this solver kernel.

## Predeclared CPU audit

Reauthenticate the complete historical 480-leaf / 673,600,544-byte capture at
source `e45a59c412bfe6bcf2de751f98159aa7fe0db463`, including historical Git
bindings, and reproduce its full anchored receiver before inspecting binaries.
Reuse its original raw bytes. The existing receivers and flags stay unchanged.

For all nine explicitly retained compiled leaves, authenticate whole generated
source, metadata and ELF CUBIN bytes against the original inventory and recorded
loaded-input anchors. A bounded standard-library ELF64 little-endian CUDA
executable profile reads section headers, linked string/static-symbol tables,
and defined function extents. Extended numbering, missing symbols, malformed
ranges, unknown role kernel families and inconsistent source/metadata/binary
entrypoint sets are refused. This is not a general ELF loader or disassembler.

Report every defined function and whole executable-section hash, and compare
all forward entrypoints with metadata and generated source. Retain recorded
role-specific fresh-cache and `fuse_fp` values only as role-specific records.
No symbol is promoted into a binding for another module. Absence means absence
from these retained roles, **not absence from the historical GPU context**.
Actual loaded bytes and instructions remain opaque; all five qualification,
training and physical flags remain false. No GPU initialization, subprocess
compiler, cache mutation, policy training, perception work or robot motion.

## Smallest subsequent native declaration

If the target is absent, do not disassemble unrelated friction/contact CUBINs
to claim solver FMA. Predeclare a fresh isolated child binding the literal dense
target's module, fresh `ModuleExec`/cache state, generated source, metadata,
whole CUBIN, actual sm120 device/context, explicit load and exact forward hook.
Keep the solver and row order unmodified. Bind compiler/runtime/tool bytes and
retain bounded disassembly only for that target symbol; do not infer instruction
use from module-wide string matches. Bind the artifact to the target launch in
the controlled recipe rather than a compile-only lookalike. Preserve the old
capture as a separate historical claim. Before launch, require a new bounded
protocol, focused tests, idle-GPU gate and explicit workload-preservation checks.
This declaration is preparation, not permission to run a new GPU job or learner.

## Precommit checks

The owner inspected the literal frozen target at solver.py lines 1988–2010 and
the original `_compile_modules` role selection. Parser/scope tests use synthetic
ELF carriers, independent of retained artifacts or CUDA. Tests cover header and
section ranges, malformed linked string/static-symbol tables, symbol extents,
closed role/leaf sets, whole hash mismatches, unknown target families, generated
source and metadata disagreement, duplicate metadata keys, false-admission
requirements, CUDA visibility and optimized-Python refusal. The core imports
no Warp, MuJoCo, NumPy or Torch. Full committed capture replay is the next check.

Independent worker review and primary-source web lookup could not start because
their backend returned HTTP 403. No review pass is claimed. Installed source,
whole retained artifacts, owner review and executable tests are the evidence
used here; a separate review remains required before a new native protocol.
