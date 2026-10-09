# Ada contact manifold and force arithmetic diagnosis

Base `b57a2b9562ddc385e71d7264c9fb080a7d7c3d9c`, exact feature branch
`feat/athletics-obstacle-curriculum`. Follow-up to the
[retained motor/contact forward](2026-10-09-ada-duck-contact.md).
Changed-path fence: this document, `ada_contact_diagnosis.py` and its test only.
No installed package, plant, motor, solver, physics gate or historical protocol changes.

## Source facts and hypothesis

Authenticated prepared arrays identify floor geom0 as plane (type0), feet29/79
as mesh (type7), with mesh IDs1/3. Retained float32 bounding radii are
0.03549982234835625 m and 0.03549975901842117 m.

Native MuJoCo3.10.0's dedicated plane-convex path starts from a support point,
adds qualifying mesh vertices from the hull neighborhood, caps total points at
three per pair, and suppresses extra vertices within `0.3*rbound` of the first
point. This is a contact-generation rule, not a simulation acceptance tolerance.
See official tag source [mjc_PlaneConvex / addplanemesh, lines877–1003](https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_collision_convex.c#L877-L1003).

Installed MuJoCoWarp3.8.1 routes plane/mesh through the primitive path, choosing
up to four support-region vertices and deduplicating selected vertex IDs, not
spatially merging contact rows. Its support-region criterion includes `1e-3`.
See [plane_convex, lines52–277](https://github.com/google-deepmind/mujoco_warp/blob/v3.8.1/mujoco_warp/_src/collision_primitive.py#L52-L277)
and wrapper lines810–870. The MULTICCD convex-pair path does not select these
plane/mesh points. Installed driver lines42–50, primitive dispatch lines1282–1289,
and `io.py` lines394–405 distinguish this path from box/mesh and mesh/mesh CCD.

These are primary-source explanations, not an attestation of loaded native or
GPU machine code. Read-only checks on both Mac and native100.100 matched these
installed source SHA256s:

| Installed `mujoco_warp/_src/` file | SHA256 |
| --- | --- |
| `collision_primitive.py` | `e515350dc6405494cdcf32b90ce32788fe7a5749c42439c877a426fce2bf9a72` |
| `collision_driver.py` | `dcc32d3781e3201ff3c79e714f7b6bd854f36045c16068c8ac26a101eb652b8e` |
| `collision_core.py` | `3c12b154998af28e257ff80437ca4250546947a6b3bf2af7fa3f1210fdff7d74` |
| `io.py` | `731a31f254274c6e04d9843b11bdcb0502cf9d49abf26aa1c82e52e1a13fa75a` |

Thus four native versus eight Warp points is consistent with different
manifold-generation rules; the count alone is not proof of a bug or CUDA fault.
Whether that difference fully explains the acceleration discrepancy is still
unproven. Do not require different collision algorithms to produce a bit-exact
point manifold, and do not confuse this cross-backend investigation with the
separate unresolved same-backend repeat/origin evidence gates.

## New arithmetic checker

`python -m mjlab_microduck.ada_contact_diagnosis --input <retained-core-directory>
--output <absent-json-path> --source <exact-clean-commit>` requires
`CUDA_VISIBLE_DEVICES=''`, verifies a clean exact feature branch and committed
diagnostic bytes, and exclusively
creates a finite JSON. Fixed execution source `0ce8c9d0a8203f5645f6dcd9ad02d04f18a173ac`
and independently known report SHA
`5fa41d7edeb79cffee87093e1255367c2e01af1b7ca6c468f104aab5ae408aa9`
bind the predecessor. It checks all manifest file bytes and complete payload
semantics before decoding exact authenticated in-memory NPZ bytes.

It preserves ordered world/geometry/dimension/inclusion keys and raw slot lists.
Per backend/key, count exact distinct position byte values and report minimum
pairwise separation, without assigning a spatial tolerance or matching points.
Transform each retained contact-frame force and torque using the row-axis frame;
add `position cross force` for a moment about the **common world origin**. Sum
all contact contributions, never average or normalize by point count. This is
the resultant in the retained frame convention, not net robot force after
gravity, motors and inertia, and not a new physical contact identity.

For each world, recompute complete `J.T @ efc_force` in float64, separately for
type1 motor-friction and type6 pyramidal-contact rows and for all rows. Record
every generalized component and residual against retained `qfrc_constraint`.
The GPU stored sum uses float32; the new reconstruction uses float64 arithmetic
over retained float32 values. Residuals are descriptive, not pass criteria.

No new collision, solver, integration, graph, policy or GPU call occurs. No
runtime/solver/simulator/training/skill/physical qualification flag is granted.
Tests cover a non-symmetric frame (to catch transpose errors), origin moments,
contact multiplicity sums, reversed ordered pairs, all-row reconstruction,
nonfinite arithmetic, byte corruption of each manifest file, inert imports,
CPU-hidden CLI and refusal before decoding. Full predecessor checks are reused.
The new diagnostic uses a 90-second-capped CPU-only native user service; focused
CPU regression uses 240 s and the same 6 GiB/200%/64-task limits. Neither may
initialize CUDA or overlap a Duck GPU job; foreign Grounding DINO is preserved.

## Current retained-array findings

The four Warp points per foot are distinct positions: minimum within-foot
separation is about 0.0188951 m; each native foot has two distinct points about
0.0308287 m apart. Historical references to duplicate points meant duplicate
**structural keys** and unresolved correspondence, not proven duplicate positions.
No nearest-neighbor relation or pointwise force comparison is established.

| Float64 row-force reconstruction against stored constraint force | Maximum absolute residual |
| --- | ---: |
| Native, air-gap world | 0 |
| Native, shallow-contact world | `1.3376344303039911e-17` |
| Ada, air-gap world | 0 |
| Ada, shallow-contact world | `2.942979335784912e-07` |

The raw active rows account arithmetically for the retained generalized forces
with these descriptive residuals. This is evidence against an obvious missing
active-row bookkeeping error in these banks, not solver correctness or proof
of an allocation cause.

| Total retained upward contact force | Left foot | Right foot |
| --- | ---: | ---: |
| Native | 3.1245020875313196 N | 3.1256839132651137 N |
| Ada | 3.2832817435264587 N | 3.284973382949829 N |

The summed upward-force delta is 0.3180691256798545 N, concentrated in the
shallow-contact solve. Summation does not erase the differing manifold: these
are different generated contacts, not matched force samples. The checker also
retains all horizontal forces, moments and generalized friction/contact terms.

## Next separately predeclared collision-only control

Before any new GPU or solver experiment, implement and test a CPU-hidden,
collision-only replay with the exact retained two-world float32 states and
unchanged compiled plant. Fresh native kinematics and Warp kinematics precede
their respective collision calls. Stop before make-constraint, solve, actuation
or integration; do not call forward, forward-position or the stance constructor
(which performs an initial forward). Retain complete candidate contacts and
complete actual geometry pose inputs, including any kinematic residuals.

Start with one fresh default-flags arm. Any optional MULTICCD sentinel must be
separately declared as an isolated diagnostic-model flag change, never a change
to the production plant; it is not a proposed fix for this primitive pair.
Preserve support/vertex selection and suppression evidence where obtainable.
No point matching, force injection, post-hoc tolerance or installed-code patch.

Require a clean exact tested source, existing frozen packages, CPU-only fresh
process, 180 s runtime, 6 GiB RAM, 200% CPU, 64 tasks, Nice10, no core/restart,
control-group kill, 10 s stop timeout and 16 MiB per leaf. Preserve foreign GPU
owners and mission-service states. A collision-only count repeat can localize
the generation boundary; it cannot qualify integrated dynamics, prove force
causation, repair the original repeat/origin gate or authorize PPO/hardware.
This control is preparation only here; it has not been implemented or launched.
