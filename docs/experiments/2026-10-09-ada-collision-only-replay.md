# Frozen-pose CPU collision-only replay

Base `cf1fb7ac54721f2847c83ab5916f91ffa64f8e21`, exact branch
`feat/athletics-obstacle-curriculum`. Changed-path fence: this document,
`ada_collision_replay.py`, and its focused tests only. No packages, runtime,
compiled plant, motor, solver, training protocol or gate changes.

## Predeclared default-flags arm

Reuse the complete authenticated predecessor from
[motor/contact forward](2026-10-09-ada-duck-contact.md), through the
[arithmetic checker](2026-10-09-ada-contact-manifold-diagnosis.md)'s
`authenticated_banks`. Fixed predecessor source
`0ce8c9d0a8203f5645f6dcd9ad02d04f18a173ac` and report SHA256
`5fa41d7edeb79cffee87093e1255367c2e01af1b7ca6c468f104aab5ae408aa9`.
Decode all seven exact prepared float32 state inputs (including both saved
root heights), not a newly placed pose or newly subtracted height offset.
Native copies each world into float64; Warp CPU copies the original float32
bytes. Copy the exact retained per-world motor friction/damping fields without
running BAM again. This is a replay of the prepared plant, not new motor data.

Fresh `build_entity().compile()` must match selected compiled fields, assets,
topology, joint addresses and options, without `describe`/placement/forward.
Before and after collision, compare **every** logical Warp model array byte,
dtype, shape and stride against the predecessor's prepared bank, including all
mesh/hull inputs. Exact collision scalar defaults are enableflags0,
disableflags0, pyramidal cone0, CCD iterations35/tolerance1e-6; Warp dispatch is
NXN with the installed PLANE|SPHERE|OBB broadphase filter. Require clear world0
and nonempty shallow-contact world1, not a preselected contact multiplicity.
No flag is toggled; no MULTICCD arm or model repair is allowed.

Only native `mj_kinematics` and `mj_collision` and Warp CPU `kinematics` and
`collision` are measured. `make_data` also makes one explicitly counted native
kinematics call for initial static poses. Stop before constraint construction,
solver, actuation or integration. Verify unchanged seven-state bytes and zero
EFC/friction/equality/limit/solver counters. Verify collision/broadphase capacity.
Record all active-prefix Warp Contact fields and all native candidate fields,
including elem and solver-address sentinels. Geometric margin inclusion is
`dist < includemargin`, never inferred from addresses (which remain -1);
separately count Warp's CONSTRAINT type bit. Neither is a constructed EFC.
Native mu/H are retained **uncomputed solver-stage scratch**, not force data;
exclude is raw metadata, not an independently used eligibility signal. Preserve raw
slots, world IDs, ordered geometry pairs, type and geomcollisionid metadata.
Retain complete actual body/inertial/geometry poses immediately before each
collision call, descriptive residuals, and exact foot mesh support geometry.
No contact forces are requested because no solve has occurred.

Installed collision kernels do not expose selected vertex IDs or native
suppression traces. The broadphase context is ephemeral and not returned by
`collision`; retain its count, not invented pair scratch data. Full model-array
hashes bind the retained hull graph bank rather than duplicating that large bank.
No spatial matching, normalization, tolerance, contact injection or installed
library patch. All qualification/training/physical flags remain false.

## Execution and checks

Fresh native100.100 CPU process using frozen Torch2.9.1/Warp1.12.0/MuJoCo3.10.0/
MuJoCoWarp3.8.1/mjlab1.3.0. Require exact clean committed branch and module bytes,
literal `CUDA_VISIBLE_DEVICES=''`, all thread pools1, CPU-only enumerated Warp
devices, Torch CUDA uninitialized throughout, a fresh
private compiler cache, precompiled headers off, and exclusive outputs.

Replay user service: Type=exec, RuntimeMaxSec180, MemoryMax6G, CPUQuota200%, TasksMax64,
Nice10, LimitFSIZE16M, LimitCORE0, Restart=no, KillMode=control-group,
TimeoutStopSec10, RemainAfterExit=yes. Focused regression service uses240s with
the same remaining limits. Preserve Grounding DINO and all unrelated owners;
the four protected mission-service states must remain inactive. No GPU job.

CLI: `python -m mjlab_microduck.ada_collision_replay --input <existing-core-dir>
--output <absent-absolute-json> --cache <absent-absolute-cache> --source <SHA>`.
Retain JSON plus NPZ and the service identity/retirement. Receive NPZ by exact
file and per-array hashes, field inventories, finite values and recomputed
summaries/residuals, before recording the outcome here. Mac and native run the
focused eight-file CPU suite; only native runs the exact plant replay.

Mac preflight refused the native selected-field fingerprint. A read-only
fresh-model comparison found all347 model arrays, with differences in compiled
mesh/geometry arrays. This control does not relax that boundary or claim Mac
plant equivalence. No fixture collision was run during this preflight.

Repeating four native versus eight Warp candidates without a solver would
localize that multiplicity difference to collision generation. It would **not**
prove that it fully causes the force/acceleration residual, qualify integrated
dynamics, resolve the separate same-backend origin/repeat gate, or admit PPO.
Evidence and source hashes will be appended after the frozen run.

## Refused first execution and corrected representation guard

Initial source `9c03ae6d16007ab5affc9c89dd787734078d86ef` passed298 focused
cases on Mac and native. Native first regression used Type=oneshot; systemd
explicitly warned RuntimeMaxSec was ineffective. That service finished in80.66s
(invocation `88158298c7d7484791477fd10e5a19a7`, MainPID0/status0/empty cgroup),
but is **not** retained as proof of a time-capped execution. Retested with
Type=exec and verified RuntimeMaxUSec4min plus all declared limits;298 passed
in81.11s, invocation `ddba864ab7ed4c90a188cdeda5d5704f`, status0/empty cgroup.
Mac passed298 in53.85s; all three XML testcase multisets matched with no skips.

First replay invocation `2038c4ca71544ebf99e0997c7f3a76e4` used effective
Type=exec/3min caps but refused with `ValueError: unchanged predeclared collision
scalar options`, status1, MainPID0 and empty cgroup. No fixture kinematics,
collision, constraint or solve was reached, and no JSON/NPZ was produced.
Read-only source/live-option inspection found the guard erroneously compared
Warp's **one-element float32 array** `opt.ccd_tolerance` to a scalar. The native
value is1e-6; the installed Warp representation is its exact float32 encoding,
`9.999999974752427e-07`. Correct only this representation check: compare shape,
dtype and exact bytes, then report the actual encoding. This is not a changed
collision tolerance or a relaxed residual gate. New tests reject wrong shape,
dtype and one-ULP alterations. Complete model-array binding still covers this
option array independently.

Warp1.12 has no `config.enable_cuda` setting: assigning that name was inert.
Remove it and correct the claim. With literal CUDA_VISIBLE_DEVICES empty, Warp
does probe the driver and prints error100/no visible device, but enumerates only
CPU and creates no Duck GPU workload. Require CPU-only enumerated devices,
explicit CPU scopes, Torch CUDA uninitialized and unchanged foreign GPU owners;
do not claim no driver probe or change the installed library/driver. Preserve
the first empty private cache and failed service for diagnosis; retry at a new
tested source with fresh outputs/cache, same180s effective caps and unchanged
plant/protocol.

## Frozen replay result and reception

Corrected execution source `2d0da53fd8ab4e6b9aed366a3ff925a1b7c2a0a4`
passed **303 cases across eight focused CPU files** on Mac49.21s and
native80.79s, zero failures/errors/skips; complete testcase multisets match.
Native regression invocation `3fae219409eb4cf28b0d3ec342342fba` used verified
Type=exec/240s caps, finished82.286299s including interpreter/test startup, and
retired MainPID0/status0/Result success/empty ControlGroup.

Replay invocation `9226bbb39aff4129a32faf9366bfab5c` used verified
Type=exec/180s and every declared resource/environment limit. It finished in
**14.695247s**, MainPID0/status0/Result success, active/exited, empty ControlGroup.
The private cache compiled six modules on **CPU**; no CUDA kernel ran. Retained
service evidence includes the complete journals and all five completed service
invocations, including the ineffective first regression cap and refused replay.

Both freshly compiled selected native plants match the predecessor's fingerprint
`6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f`.
Every one of347 Warp model arrays matches the complete prepared bank's bytes,
dtype, shape and strides before/after collision. Both exact seven-state inputs
remain unchanged. All native/Warp `ne/nf/nl/nefc/solver_niter` counters are zero;
all candidate EFC addresses are -1, and the shared broadphase count is4 with no
overflow. No contact force, constraint construction, solve or integration occurs.

| Fixture | Native candidates | Warp CPU candidates |
| --- | ---: | ---: |
| Air-gap world0 | 0 | 0 |
| Shallow-contact world1, left foot | 2 | 4 |
| Shallow-contact world1, right foot | 2 | 4 |

All shallow candidates have condim3, ordered pairs `[0,29]`/`[0,79]` and distance
inside includemargin; all eight Warp candidates have the CONSTRAINT type bit.
Neither geometric margin inclusion nor that bit means an EFC was constructed.
Complete raw records, including uncomputed native mu/H scratch, are retained.
No one-to-one physical point correspondence is asserted.

The **same four-versus-eight multiplicity is already present on CPU before any
solver**. It therefore does not require a GPU solver to appear, and localizes
that count difference to the collision-generation boundary for these fixtures.
This is not evidence that all acceleration/force residuals are caused by it,
that either manifold is wrong, or that CUDA/native/integrated simulation is
qualified. Original same-backend origin/repeat and training gates remain open.

Complete actual body, joint-axis, inertial and geometry poses are retained and
compared descriptively. They are **not bit-identical**: Warp CPU versus native
double geom position maximum residual is `2.0957069468696687e-08`m, rotation
matrix maximum `2.3841857776929487e-07`; Warp CPU versus retained Ada geom
position maximum is `1.4901161193847656e-08`m, rotation maximum
`1.7881393432617188e-07`. These are results, not new acceptance tolerances.
All nine kinematic field residuals and float32 bit-mismatch counts are recorded.

The446144-byte NPZ contains84 fully hashed arrays: all candidate fields,
all nine actual pose fields on both backends, seven input states, model geometry
indices/radii/mesh addresses and both exact foot meshes. Larger hull graphs are
bound by the complete347 model-array hashes to the existing prepared bank, not
copied again. Exact whole-file/per-array hashes, field inventory, finiteness,
source/predecessor bindings, seven states, candidate summaries and both
kinematic comparisons passed the CPU-only `receive()` on Mac. These artifacts,
test XMLs, service receipt and read-only capture helper are retained on both
hosts under `artifacts/tools/ada-collision-replay/`; private compiler caches
remain only on native. Source and evidence retention are not physics admission.
Independent Luna reception reran the pure receiver and checked every declared
JSON/NPZ/helper/service/XML hash, full raw contacts and counters, both meshes,
303-test XMLs and the non-bit-exact kinematic residuals. No blocker was found;
the review performed no SSH, GPU, service operation or file edit.

Native identity, frozen packages and clean exact checkout were freshly verified.
Grounding DINO PID1592 remains the sole compute owner (946MiB); GPU snapshot
961MiB used/15232MiB free/46C/0% utilization. This is a post-run snapshot, not
a measured peak. All four system/user protected mission-service states remain
inactive. No unrelated job, driver/package or production service was changed.

| Retained corrected evidence | SHA256 |
| --- | --- |
| Replay module bytes | `3147a40562e29ddd94deb6a4d6219cfeb6b9997f6b6c9f1078a22dcd35f1d7b6` |
| `2d0da53f-linux-replay.json` | `00a4e2bbd7c43a1376c97883ce274e3640b2c395a50f751561c96c5d5f8ad45b` |
| `2d0da53f-linux-replay.npz` | `c85afc7f17721c6b4d7343df587e0cb22b744261afdeb54f7623f957d93c93a1` |
| `2d0da53f-native-services.json` | `f15fcda85b6874719f854b2c307e8eb1aedf6925e0970c682f95711800e6e1ba` |
| `2d0da53f-mac-tests.xml` | `8224fcac0cb55a05fbb9e3375e312d6bdc46ca8a51d1867c117ad3d4314ae970` |
| `2d0da53f-linux-tests.xml` | `61d3906a5bed6790f63304a811223cb4d6477fe76452b32c4009fa3a19537b10` |
| `retain_cpu_services.py`, read-only capture helper | `8011ea491669d6f394df555df767f50f26e3362163b6406355cc29fab3c2aa2f` |

## Next bounded control, not launched here

Predeclare a solver-response separation control before any new GPU/training
experiment. First determine whether CPU Warp and retained Ada generated contact
tables can be compared honestly using generator metadata and complete actual
pose inputs, without assuming duplicate ordered keys imply point identity.
If actual poses are replayed into a separate diagnostic collision arm, label that
as bypassing kinematics, bind every supplied pose byte, and do not mutate this
default-flags control or the production plant. Any later force comparison must
keep full rows, solver inputs and whole-manifold resultants; no contact-count
normalization, guessed spatial pairing, tolerance change or force injection.
This result alone authorizes no policy promotion, PPO, hardware or robot motion.
