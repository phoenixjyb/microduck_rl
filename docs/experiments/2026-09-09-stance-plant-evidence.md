# B1-N compiled plant and recorded-state checks

This CPU-only chunk extends the [strict checkpoint bundle](2026-09-09-stance-checkpoint-bundle.md).
It launches no optimizer, full hold, GPU probe or football policy. Historical
results and numerical acceptance gates are unchanged.

## Explicit coverage

The v2 bundle requires runtime JSON with exactly `source` and `plant`. On both
publication and read-only verification, the checker freshly compiles the existing
stance entity and compares its selected native-model descriptor: XML/mesh byte
hashes, topology, ordered joint addresses/ranges, floor/foot geometry IDs, nominal
placed reset, timestep/gravity/solver settings and a canonical hash of the listed
body, joint, DOF, geometry and actuator arrays. The descriptor lists every selected
field; it is not a complete serialized MuJoCo model or live Warp-array attestation.
Source labels remain caller-supplied and require independent launcher verification.

Each recorded boundary is checked against that mapping: joint velocities, exact
hard/soft masks, unit free-root quaternion, projected gravity, body-frame angular
velocity, joint offset/speed observations and critic velocity/height/support.
Initial positions must equal the placed float32 reset exactly, with zero velocity,
counters and prior correction. Current corrections must remain bounded. Replay of
the correction slew/FIFO and complete motor-state provenance are still separate
unfinished integration, not implied by these observation checks.

First-terminal contact tables must have the declared layouts, integer world/geom
IDs in the compiled range and supported dimensions. The checker re-reduces normal
loads using the bound floor/feet and recomputes forbidden contact flags. It does
not independently solve forces, reconstruct active constraint addresses or prove
that a retained contact table originated in a live solver. Empty terminal tables
are supported; the receipt counts how many terminal records were actually checked.

Float comparisons use absolute1e-6/relative1e-5 for representation differences.
Tilt consistency compares its cosine to avoid inverse-cosine amplification near
upright. Original raw tilt/support/torque and all physical acceptance thresholds
are unchanged. Joint-limit masks and nominal reset use exact comparisons.

Bundle and launch schemas are explicitly version2; v1 artifacts are not silently
reinterpreted or overwritten. There were no retained non-test v1 bundles from this
implementation. Hash verification, exclusive output creation, manifest-last
publication, strict actor/critic restoration and deterministic action replay stay
in place. All provenance/learned-skill/admission/physical-motion flags remain false.

## Validation and next boundary

Fixtures use an untrained actor, short real CPU physics, deliberately injected
terminal tilt and synthetic malformed tables. They are not training evidence.
Local focused validation:257 CPU tests passed in14.29s, including36 new plant/
bundle cases. Markdown HTML and local-link checks passed. Exact-source Linux
confirmation follows; no GPU or optimizer was launched.

At initial implementation `be6757603ae160d2de7a68731c5cbd5175a5521e`, Linux
passed323 CPU tests in23.93s with no skips. An additional fresh-process audit
found an import-order warning hidden by test collection's earlier imports:
`flat_hold` imports `bam.mjlab` before mjlab task discovery completes. This new
module now initializes mjlab first, with a fresh-interpreter regression test;
historical probe files and installed libraries are untouched.

Exact native compilation is platform-specific. Mac/Linux assets, topology,
addresses, ranges, options and float32 nominal reset matched. Selected native
double arrays differed only in body inertia (maximum1.0843e-19), inertial
quaternions (2.8936e-15), geometry size (1.2317e-16), position (1.3878e-17) and
quaternion (1.5544e-15). These tiny differences explain distinct exact descriptor
hashes: Mac `7f60b2de94229b92f6bfa6be5009b3fcc165ee9a9de5cc01697bc08afd58069b`,
Linux `15cec59fb2aa1e0e90a53293c256a174d4912b3f345d1a382da273f109046cb5`.
Do not round away this identity difference. Linux-produced bundles must be
verified against the matching Linux compile/runtime; Mac replay fails closed
at the exact plant gate. Cross-platform bundle acceptance is not claimed.

After the import-order regression fix, local validation passed258 CPU tests
in22.15s, including37 new cases in this chunk. Updated Markdown HTML checks
passed. Linux confirmation of this exact follow-up source is next.

Next: bind the remaining action-delay/motor state and exact held-out evaluation
matrix; finish PPO transition/timeout handling, complete resumable training-state
saves and guarded launch assembly. Only then run the predeclared disposable
64-world/16-iteration smoke and use measured timing before the fresh stance pilot.
