# Retained solver binary scope, 2026-10-08

Status: same-source Mac/WSL CPU replay complete; solver binary binding missing.
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

## Same-source retained outcome

Executed source `648037059318690ea7ee91f8d8c0ac114c8b8523` on both hosts,
CUDA hidden, numerical thread settings all one. Each passed **78 checks**
(64 new scope/parser/refusal tests plus 14 historical receiver tests), with
zero failures, errors or skips. Each authenticated all 480 original leaves
and reproduced the whole historical receiver before this scope audit.

Reports are byte-identical: 199,173 bytes, SHA256
`dbaf327295892a8f40b71c6c354eefe234ffe4c786a1a1e5b4558925ab9b9f09`.
All nine retained compiled leaves agree with the historical inventory and
loaded-input records. Metadata, generated source and defined ELF forward
symbols agree exactly across all roles:

| Historical role | ELF sections | Defined functions including helpers | Forward entrypoints | Bound solver target |
| --- | ---: | ---: | ---: | --- |
| Original friction/constraint module | 106 | 41 | 10 | no |
| Candidate ascending friction | 34 | 4 | 1 | no |
| Contact init | 34 | 2 | 1 | no |
| Total | — | **47** | **12** | **no** |

Decision **`missing-retained-solver-binary-binding`**. The target is absent
from all three retained explicit roles. This does not assert its absence from
the GPU context: the historical simulator did run the solver, but these files
do not bind that solver module's loaded input artifact or instructions. The
recorded `fuse_fp=true` and fresh-cache claims belong to the three roles above,
not to the dense solver kernel. No instructions were disassembled, no GPU
stack initialized, and every qualification/training/physical flag stays false.

### CPU owners and transfer verification

Mac owner elapsed 6.263159 s. WSL user unit
`microduck-solver-binary-scope-cpu-648037059318.service`, invocation
`6b18aa473c3c4604bd7a0a84622467db`, owner PID 4183235, finished with status
zero and MainPID zero at 12:46:37 Asia/Shanghai (start 12:46:27); owner elapsed
8.450259753 s. WSL enforced 240 s, 100% CPU, 6 GiB memory, 64 tasks and
16 MiB/file ceilings; these are not measured peaks.

| Whole proof | Bytes | SHA256 |
| --- | ---: | --- |
| Mac | 1,048 | `70b7569a0dfc853bbbf17b61a1d65c881c16530aee6c90b2396ac44a5ffd96be` |
| WSL | 1,082 | `1a86f915aec3fe1b56f80cb3364ba492088f97e831c6cea63a7bbbc40053406c` |

Each anchored proof binds exactly four leaves (JUnit, pytest log, replay stderr
and report). Both hosts reauthenticated both proofs and each leaf after mutual
transfer, including canonical serialization and whole-report equality. Original
directories are `artifacts/tools/solver-binary-scope-cpu-648037059318` on their
respective hosts; cross copies are `solver-binary-scope-native-648037059318`
on Mac and `solver-binary-scope-mac-648037059318` on WSL. Original raw capture
bytes were reused; no second raw archive was created.

FilmBrain stayed active at PIDs 521 and 298048. Protected AI-mission services
stayed inactive in both user and system scopes. The frozen `.venv` alias and
package versions were unchanged. GPU memory was already occupied by other
consumers (6,748 MiB at initial inspection, 6,021 MiB at closeout); no Duck CUDA
job was started, no existing workload was stopped, and no idle-GPU claim is made.

The read-only tool check found no `cuobjdump` or `nvdisasm` on PATH or at the
eight checked `/usr/bin` and CUDA `/usr/local` bin paths (default, 12.9, 13.2).
This is not an exhaustive installation inventory. No tooling was installed.
Resolve a byte-pinned inspection tool as part of the separately reviewed native
declaration; do not substitute unrelated binaries or guessed instructions.
