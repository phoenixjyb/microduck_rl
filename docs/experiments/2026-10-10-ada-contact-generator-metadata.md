# Retained contact-generator metadata audit

Base `203324da7f6f45c8de724cf6c78746ae169129cc`, exact branch
`feat/athletics-obstacle-curriculum`. Changed-path fence: this document,
`ada_contact_metadata_audit.py` and its focused test only. No physics, installed
package, runtime, solver, production plant or acceptance-gate changes.

## Predeclared CPU-only comparison

Follow the [collision-only replay](2026-10-09-ada-collision-only-replay.md)
using its exact JSON SHA256
`00a4e2bbd7c43a1376c97883ce274e3640b2c395a50f751561c96c5d5f8ad45b`
and complete NPZ SHA256
`c85afc7f17721c6b4d7343df587e0cb22b744261afdeb54f7623f957d93c93a1`,
source `2d0da53fd8ab4e6b9aed366a3ff925a1b7c2a0a4`. Reuse its full receiver,
including authenticated predecessor report and complete prepared/active banks.
No new compile, collision, constraints, solve, integration, or CUDA calls.

Compare Warp CPU candidates to retained Ada GPU contacts using the exact tuple
`(worldid, geom0, geom1, dim, full type, geomcollisionid)`. Require complete
integer/float32 schemas, unique full keys and exact complete key-set equality.
Keep pair ordering and original raw slots; do not join on slots, approximate
positions, geometry groups alone, or presumed physical vertex identity.
Duplicate full keys, changed metadata, missing fields and nonfinite values fail
closed. The tuple is **generator metadata**, not proof of physical point identity
or compiled generator execution identity. Native records lack its full metadata
and are not silently assigned these labels.

For each aligned generator label, report all components of Ada-minus-CPU `dist`,
`pos`, `frame` and `friction`, evaluated in float64 from authenticated float32
inputs. Keep exact byte equality and float32 bit-mismatch counts. Do not normalize
manifolds, round, invent a tolerance, or compare post-solve forces with candidates.
The full raw input arrays are already retained, not duplicated here.

Bind every actual pose field (`xpos`, `xquat`, `xmat`, `xanchor`, `xaxis`, `xipos`,
`ximat`, `geom_xpos`, `geom_xmat`) to complete-array hashes, shapes, residuals and
bit counts. Pose mismatch means this is **not** a same-pose collision-response
isolation. Record stage differences: CPU candidates are before EFC construction;
Ada records are after a complete forward/solve. CPU margin inclusion and type bit
are separate signals; Ada includemargin was not retained and is not reconstructed
from its EFC addresses. All simulator/training/physical flags remain false.

CLI: `python -m mjlab_microduck.ada_contact_metadata_audit --input <existing-core>
--replay <retained-replay-json> --output <absent-json> --source <clean-SHA>`.
Literal CUDA_VISIBLE_DEVICES empty, clean exact branch/module bytes, exclusive
bounded finite JSON. Run Mac and native CPU checks and audit. Native user
services use Type=exec, RemainAfterExit=yes, RuntimeMaxSec90 for the audit and240
for regression, MemoryMax6G, CPUQuota200%, TasksMax64, Nice10, LimitFSIZE16M,
LimitCORE0, Restart=no, KillMode=control-group, TimeoutStopSec10 and thread pools1.
Preserve Grounding DINO and inactive mission services. No GPU workload.

Independent read-only review at the base found identical eight unique full keys
but differing slot order and all-nine pose bit mismatches. This is a design input;
execution artifacts, tests, source hashes and exact observations will be appended
after the committed audit runs. It cannot qualify simulation or authorize PPO.
