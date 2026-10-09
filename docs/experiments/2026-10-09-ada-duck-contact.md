# Ada motor-friction and shallow-contact forward diagnostic

Branch `feat/athletics-obstacle-curriculum`, base `dc0439c8b8a3c48d3062a9134c34cb3b231d68b8`.
Follow-up to the [empty-constraint baseline](2026-10-09-ada-duck-forward.md).
No old numerical gate, solver-origin collector, physics integrator, observation,
policy, curriculum, checkpoint or installed package is modified. Only bounded
diagnostic code/tests/docs and a closed two-profile supervisor dispatch change.

## Predeclared fixture and reference

Two actual stance worlds, using the unchanged `WarpStanceRuntime(2)` constructor,
nominal XL330 m6 BAM (7.5 V, voltage-drop gain 0.1, firmware gain 200), no randomized
mechanics or action. World 0 retains the existing approximately 0.1 mm foot airgap;
world 1 lowers root height by float32 -0.0002 m before any measured input is frozen.
There is no trajectory or physical teleport: these are two static initial fixtures.

The constructor performs its reset forward. After setting the fixture poses,
one additional eager **fixture forward precedes the motor proposal**, so BAM
reads the contact load for the actual pose. Then one genuine zero-error nominal
position proposal for both worlds, zero velocity/effort targets, accepted mask,
no rejected rows and finite zero torque. Capture the preceding bias/constraint/
actuator forces, own-friction, zero motor history, voltage/gain and command inputs.
Commit the proposal's real friction/damping and zero ctrl, with positive dry
friction on every controlled DOF. Load-dependent fields may differ by world.

For each world separately, freshly compile the same native plant, verify its
descriptor before changing motor fields, copy **that world's exact float32
friction/damping vectors** to native double, and copy its representable state,
warmstart, ctrl and applied forces. One native forward per world; one measured
Warp batch forward. No graph, integration, Euler call, policy tick, reset after
freeze, optimizer, perception or physical motion. The runtime's unused integrator
object exists from construction; its integrate method is never called.

Retain every logical byte of all prepared model/data arrays, plus declared
native-double/Warp-float32 20-field outputs, complete active constraints,
contact tables and BAM source/proposal. Full input hashes and allocation/static
binding are rechecked immediately before the measured call. All seven state,
ctrl and applied-force fields must remain unchanged. Scratch input bytes are
retained, not promoted to semantic output or checked as solved fields.

## Coverage and comparison rules

Each world must have exactly 14 dry-friction rows: unique actual controlled DOF
IDs and unit Jacobians, not a guessed allocation order. No equality/limit rows.
World 0 has no geometric contacts. World 1 has included, penetrating dim-3
pyramidal contacts for both feet/floor and no forbidden pair. Every active
constraint row must be accounted for by friction or a solve-local contact slot
and its four-row address group. Preserve raw IDs, row ordinals, frames, forces
and all J/D/aref/force/state entries. GPU collision-type/index sidecars are
descriptive, not cross-backend identity.

Cross-backend contact keys preserve ordered geom pairs, world, dimension and
inclusion. Duplicate points or missing keys remain **unresolved correspondence**;
do not sort away orientation, invent point matching or compare raw row order.
Friction rows can be aligned by DOF. Report all dynamics max-abs/RMS/float32-bit
differences and aligned friction residuals descriptively. No new dynamics tolerance,
solver qualification or training gate is introduced. Complete array reception
recomputes every statistic and structural decision with `allow_pickle=False`
from the exact authenticated NPZ bytes. All qualification/authority flags stay false.

## Bounds, preparation findings and closeout

Use only 100.100's clean exact source, existing frozen venv, sm89 RTX 4090 Laptop
16376 MiB / driver 595.91.07 and the unchanged health guards from the
[runtime smoke](2026-10-09-ada-runtime-smoke.md). Foreign Grounding DINO and all
unrelated workloads stay untouched; AI mission services remain inactive. Same
existing advisory lease; no driver or package changes. CPU-hidden stdlib owner,
fresh CUDA-0 child with private caches, PCH off and driver cache disabled.

Same service cap 180 s, owner 150 s / child 120 s with 20 s closeout reserve,
6 GiB RAM, 200% CPU, 64 tasks, Nice 10, no core/restart, control-group kill,
10 s stop timeout and 16 MiB per-file. Shared GPU guards: at least 10240 MiB free,
aggregate growth at most 2048 MiB, below 65 C and at most 85% sampled utilization.
Contact-profile files are at most 15 MiB each; JSON at most 192 KiB (the shared
writer remains capped at 256 KiB). Baseline profile remains at its original
2 MiB payload / 64 KiB child receipt limits. Fresh absent source-specific output.

CPU preparation discovered that all 461 logical input arrays total 15011239 bytes,
mostly meshes. The first conservative raw-size reservation refused this before
writing. A 256 KiB ZIP-header reserve fits the full uncompressed NPZ within the
unchanged 15 MiB file ceiling; no service bound was widened. Independent review
also caught stale pre-pose forces: the final design refreshes the fixture solve
before BAM and uses independent per-world native models/fields.

The CPU prototype already shows **native 4 versus Warp 8 contact points**, and
30 versus 46 active rows in the shallow-contact world (14 friction plus contact
rows). Both feet are covered, but duplicate points remain unresolved. This is
not a passing native/Warp equivalence gate and is not hidden with a tolerance.
The Ada collection tests this device's actual finite forward behavior and retains
its result; it cannot by itself resolve the native contact manifold difference.

One capped GPU attempt only after focused CPU tests and independent review.
The positive decision is `ada-motor-contact-forward-collected-pending-reception-not-training`.
A separate capped CUDA-hidden receiver must verify source/report hashes, the same
retired GPU invocation/cgroup, foreign owners/services, fresh native plant,
every complete input/active/output NPZ, BAM source/proposal semantics and statistics.
The independent receiver is retained as
`artifacts/tools/ada-duck-contact/receive_ada_duck_contact.py`, with its exact
SHA recorded at closeout. Logical input dtype/shape/stride/byte consistency is
checked, including actual singleton broadcast strides; this is not runtime-origin
authentication. Both CPU regression JUnit files must contain the same passing
testcase multiset. The CPU-only regression service may run for at most 240 s;
the GPU collection and 90 s independent-reception bounds remain unchanged.
The reviewed receiver helper SHA256 is
`f69815fdfaf752cfba78f58cc84282ab80bd09504bcd9927c854c315fe139107`;
verify these exact helper bytes on native before launching it. The artifact
is outside committed source identity, so its separate hash binding is mandatory.
Retain core evidence on native and Mac; keep private compiler caches native only.
Commit/push tested source and numerical findings to the fork, then report the
next unresolved gate. Do not launch PPO, video or skill promotion from this result.
