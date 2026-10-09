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
