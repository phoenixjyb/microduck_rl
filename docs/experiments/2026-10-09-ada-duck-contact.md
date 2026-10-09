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

## Retained Ada execution and independent reception

Frozen execution source **`0ce8c9d0a8203f5645f6dcd9ad02d04f18a173ac`**
passed 218 cases across six focused CPU files on each host, with zero failures,
errors or skips and identical testcase multisets. Final exact-source Mac run:
66.68 s; Linux: 80.50 s in a 240-second-capped CUDA-hidden user service,
invocation `53da7f000eca4e979b857b3d534c1973`. This is CPU regression evidence,
not CUDA simulation or training acceptance.

The single Ada collection completed in **36.37114370499967 s**, with 70 sampled
telemetry rows. All 20 output fields and active row/contact fields were finite.
All seven state/control/applied-force fields stayed unchanged; no integration,
graph, optimizer or policy tick occurred. All 461 prepared logical input arrays
were retained and reverified. Actual compiled native selected-fields SHA remained
`6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f`.

| Backend | Air-gap world | Shallow-contact world | Solver iterations |
| --- | --- | --- | --- |
| Native double | 14 friction rows, no contact | 14 friction + 16 contact rows; 4 contact points | 1, 4 |
| Ada Warp float32 | 14 friction rows, no contact | 14 friction + 32 contact rows; 8 contact points | 1, 4 |

No equality or joint-limit row was active. Both feet/floor pairs were covered.
For ordered keys `[1,0,29,3,true]` and `[1,0,79,3,true]`, native has two points
per foot and Warp four: there is **no unique cross-backend point correspondence**.
This reproduces the CPU-Warp prototype's contact counts; it does not establish
a CUDA-specific cause, nor establish physical identity for duplicate points.

| Descriptive maximum absolute difference | Air-gap world | Shallow-contact world |
| --- | ---: | ---: |
| `qacc`, mixed generalized acceleration units | `2.4740386345999442e-06` | `2.664054767702527` |
| `qfrc_constraint`, mixed generalized force units | `3.087231098745294e-09` | `0.31806942370307745` |
| `qacc_smooth`, mixed generalized acceleration units | `2.179735020035878e-06` | `2.179734565732616e-06` |
| DOF-aligned friction-row force | `3.087231098745294e-09` | `0.0051232540387700955` |

DOF-aligned friction Jacobians and `aref` differences were zero; `D` maximum
difference was `2.0454121396618063e-07` in both worlds. Across both worlds,
`qacc` had 40/40 float32 bit mismatches and `qfrc_constraint` 32/40.
These are descriptive residuals, **not a physics tolerance or numerical pass**.
The disparity is concentrated in the constrained/contact result, while smooth
acceleration differences remain small; this localizes the next investigation,
but does not prove that contact-point count alone caused the force discrepancy.

Sampled peak aggregate GPU memory was 1278 MiB, minimum free 14916 MiB and peak
temperature 52 C. Grounding DINO PID 1592 remained present throughout; all four
system/user mission-service states remained inactive. Post-exit usage returned
to 961 MiB used / 15232 MiB free. These are sampled bounds, not exclusive access
or a guarantee about unsampled instantaneous peaks.

GPU invocation `f9efb2bafa28476aa2358f1b48676965` retired successfully with
MainPID 0, exit 0, no restarts, active/exited, empty ControlGroup and absent
original cgroup. Separate CPU-only receiver invocation
`b69e50c5ee904f35a70cf177f32576a8` verified the same retired invocation,
source/report/launch/file hashes, lease identity, preserved foreign owners/services,
resource caps, every prepared input and BAM source/proposal, complete active
structure/residuals, both 218-case XMLs and a freshly compiled native plant.
It also retired successfully with MainPID 0 and empty/absent cgroup. A Mac
reception reloaded the authenticated full bytes and recomputed every residual
and structural result; it deliberately did not assert Mac/native plant bit identity.

The 11 core files total **15434257 bytes** and are retained natively under
`artifacts/evaluations/ada-duck-contact-0ce8c9d0a820/` and on Mac under
`artifacts/tools/ada-duck-contact/ada-duck-contact-0ce8c9d0a820/`. Source-specific
test XMLs, receiver helper and closeout are in `artifacts/tools/ada-duck-contact/`
on both hosts. Private compiler caches remain native only. These ignored numerical
assets are not added to Git history. All qualification/authority flags remain false.
An independent Luna read-only reception reproduced all file/XML/source bindings
and the complete CPU-hidden payload checks; the owner reviewed this before
accepting evidence retention, not simulator or training qualification.

| Evidence | SHA256 |
| --- | --- |
| GPU report | `5fa41d7edeb79cffee87093e1255367c2e01af1b7ca6c468f104aab5ae408aa9` |
| Child receipt | `b22fca232477b5818c0532ddf41ec17f37186dcd943425725c4e7444d7688655` |
| Prepared logical inputs | `3d454f4d61e1af4667bc84acfac054724f0589c13964937455d8a49a32d83f61` |
| BAM source fixture | `2d94680cbcdab49fb72c6315b4b2b0f9f8fd9fcc1fa9af24ea0b48a92b883b14` |
| Motor proposal fields | `7c5eedf0415b3b5de061dba78f80deeea8b444b6adc112f49f0f6cd750e42e1a` |
| Native-double output fields | `77b5f5fcf5e908d6a0542dd4256d6e0ea432ba4999227f80d38a9718f2b42754` |
| Ada-float32 output fields | `8b48384d371645be22f76bf316952ff76ea7c00e71fdf3c0dbb1ea6551b71c06` |
| Native active rows/contacts | `b98c88bb1ff9ec5ad9c87e4c6e2ad8f00b7595bdf134c17a2df0b36ae4d33870` |
| Ada active rows/contacts | `d493ab8a18e4e143f1cd133a37bfae35ec41f89989b6d22846d10d77ef5f0830` |
| Independent native closeout | `37d15bfbaee20e57f694b1232a5a9753c0c54e1d073dd38d342519c90abbb5f4` |
| Mac 218-case XML | `2a640842f1cf0c19f823d9c9fb7b6113830df413955feab6c5cbe19bd5e39499` |
| Linux 218-case XML | `71291c163730cab96587d74f8bc77d806f151988c9400d5d3b123cbe6cc6b1ba` |
| Independently reviewed receiver | `f69815fdfaf752cfba78f58cc84282ab80bd09504bcd9927c854c315fe139107` |

## Next unresolved gate

This collection establishes bounded actual Ada execution and retained reception
of genuine motor friction and shallow foot contact. It **does not qualify the
solver, numerical dynamics, integrated rollout or a learned Duck capability**.
The prior full-rollout numerical failure also remains unresolved.

Next localize the contact-manifold and constraint-force discrepancy using these
frozen input/row/contact banks and the installed native/Warp collision rules.
Any contact-matched reference protocol or new fixture must be separately
predeclared and tested before another bounded GPU attempt. Do not manufacture
duplicate-point correspondence, silently change the plant/solver or add a
post-hoc tolerance. Integrated rollout/PPO/video/skill promotion stay parked
until the applicable numerical and origin evidence gates are satisfied.
