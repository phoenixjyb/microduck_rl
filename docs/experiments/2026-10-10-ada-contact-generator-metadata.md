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

## Frozen execution and reception

Audit source `de0ee5e105a2b3012403ab1abf4027ff5357ea14`, module SHA256
`4a2a48f2dfa76363a24663c57cb31920603f9d309aea7e46559a4ca59f87cdf0`.
The final nine-file CPU suite passed **355 cases** on Mac191.16s and
native80.70s, with identical complete testcase multisets and zero errors,
failures or skips. The earlier354-case preflight preceded the independent
raw-slot test and is not substituted for this frozen-source result. Regression
includes existing CPU physics fixtures; the new audit itself executes no
physics or device operation. These checks do not qualify integrated simulation.

Native regression invocation `4185120af1c44452a8a2427f6cabf2e8` finished
82.194595s including startup. Native audit invocation
`ec8bbd1135db4e28b634efe4b9eb3ceb` finished0.275274s. Both used verified
Type=exec and all declared caps, then retired MainPID0/status0/Result success,
active/exited and empty ControlGroup. The receipt retains complete journals,
scope-specific inactive protected-service states and unchanged foreign GPU
owner. GPU was961MiB used/15232MiB free/45C/0%; Grounding DINO PID1592 remained
the only compute owner. No Duck CUDA workload or production-service change.

Mac and native audit JSONs are **byte-identical**, SHA256
`50875ac200c8175b624cf8662136929d00c3f1624fe17a1e8ae3b46a62916719`.
All eight full keys match uniquely: world1, ordered floor/foot pairs `[0,29]`
and `[0,79]`, dim3/type1, geomcollisionid0–3 per pair. Key-ordered CPU slots
are0–7; Ada slots are `[0,2,4,6,1,3,5,7]`. Slot order was not a matching rule.
The maximum common-field differences are depth `5.888068699277937e-09`m,
position `1.1175870895385742e-08`m, frame0 and friction0; the latter two are
byte-identical across every metadata-aligned row. Full component differences
and exact bit counts are retained, not rounded into an acceptance tolerance.

All nine actual pose arrays differ in float32 bits. Geom position maximum
residual is `1.4901161193847656e-08`m (220 components differ); geom rotation
maximum is `1.7881393432617188e-07` (934 components differ). The remaining
complete-array hashes, shapes and residuals are in the audit JSON. This is not
a same-pose collision control. All CPU candidate EFC addresses are-1, whereas
retained Ada addresses are14–42. CPU margin inclusion and the CONSTRAINT bit
cover all eight raw slots; Ada margin values remain unavailable. No post-solve
force or EFC address is treated as a common candidate field.

Artifacts are retained on both hosts in `artifacts/tools/ada-contact-metadata/`:

| Evidence | SHA256 |
| --- | --- |
| `de0ee5e1-mac-audit.json`, `de0ee5e1-linux-audit.json` | `50875ac200c8175b624cf8662136929d00c3f1624fe17a1e8ae3b46a62916719` |
| `de0ee5e1-mac-tests.xml` | `57066a4095628aa4241e58f7ac1c75b5070dbb61e46bf88538b77fdf1ab7d9ac` |
| `de0ee5e1-linux-tests.xml` | `ffe0f074ab82ff7a7f50dec4164fcfafb0e7d59c450b8138f875bc0bad365a5a` |
| `de0ee5e1-native-services.json` | `09254441984988e13b384188f97a8a1e63919938b8ed29f12cc4bc54ce24181a` |
| read-only `retain_cpu_services.py` helper | `a080d50d7ed15f7cd4007ee4300375ed99d6f25da71383ace06243f0a7c76477` |

Independent Luna source review found no remaining issue after strengthening the
slot test to decouple valid raw slots from keyed row permutation;52 focused
tests passed independently. Physics, solver, compiled generator identity,
training and physical-motion qualification all remain false.
Independent retention review also recomputed the complete audit from both
authenticated input banks and matched the saved JSON exactly, verified its
committed module hash, both retired service invocations and the receipt/helper
hashes. The owner verified both final355-case XML multisets after completion.

## Next bounded control, not launched

Separately predeclare and test a CPU-only saved-Ada-pose collision arm. Load
every supplied pose byte from the authenticated predecessor and bind the exact
collision-consumed geometry inputs, complete unchanged model and seven states.
Explicitly bypass fixture kinematics in that diagnostic arm; allocation-time
native kinematics must remain separately counted. Do not mutate the retained
default-flags kinematics/collision control or production runtime.

Read-only source tracing of installed MuJoCo-Warp3.8.1 found only `geom_xpos`
and `geom_xmat` as dynamic pose inputs in this rigid NXN/plane-mesh path
(`collision_driver.py:274-319,649-733`, `collision_primitive.py:1488-1549`,
`collision_core.py:65-156`). Retain full model-array binding: broadphase bounds,
filtered pairs, contact/pair overrides and mesh vertex/graph/polygon inputs all
matter. The contact callback at `collision_driver.py:790-791` must be explicitly
`None`; bind the no-SDF/no-flex path too. Copy all nine supplied pose fields
after allocation, with no subsequent kinematics call. `make_data` does one
allocation-time native kinematics call (`io.py:1011-1015`); count it honestly.

Require positive collision capacity and unchanged enabled flags. An early
disabled-contact return can leave `ncollision` stale, whereas normal collision
resets both counters (`collision_driver.py:774-788`). Preserve complete active
candidate records and unconstructed EFC sentinels. `flex`/`vert` are unused
rigid-output fields, not independently inferred identities. Current installed
source hashes describe source only: the old replay does not bind historical
library source or compiled CPU/GPU binaries. A new arm must bind its own current
library sources/cache/module metadata without retroactively upgrading that proof.

Stop before constraints/solve/integration, retain
complete candidates and actual supplied poses, and compare generator metadata
without claiming physical-point identity. This can separate pose computation
from collision arithmetic for the recorded fixture; it cannot resolve all
solver-response causes or admit PPO. No new solver/GPU/long training job is
predeclared by this document. Overnight work has an absolute cutoff of
2026-10-10 07:00 Asia/Shanghai; preserve unrelated owners and leave protected
services inactive unless explicitly authorized otherwise.
