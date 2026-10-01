# C1-S: official-walker substitution diagnostic

Predeclared October 1, 2026, starting at
`61d84991938d0181cf78c560a8a33d98d92f4526` on
`feat/athletics-obstacle-curriculum`. The user approved **predeclaring a separately
labelled official-walker diagnostic**, not a claim to reproduce Happy Hop's unpublished
walking entry/task overlay. This chunk predeclares it only: the native CPU BAM
runner and baseline adapter are not implemented, and no policy has executed.
This approval is preparation only: the declaration does not authorize policy
execution or training. A separate run go-ahead is required before either case.

The authoritative [machine-readable declaration](2026-10-01-community-hop-c1s-predeclaration.json)
pins the experiment `community-hop-c1s-velstand-substitution-v1`, artifacts,
surrogate plant, runtime, two cases, limits and required evidence. The existing
[C1 first-attempt scorer](../../src/mjlab_microduck/community_hop_rehearsal.py)
is frozen by SHA256; its measurement protocol is not this experiment's identity.
Every report must carry **both** identities, the declaration hash and the actual
runner source revision. Old H1, obstacle, stance and football gates stay unchanged.

## Question and controls

Can the pinned official `velstand` walk/stand policy stop and restart on our
explicit stock/backlash surrogate, and, **only if that control works**, can the
pinned Happy Hop policy enter from it, hop once, land/settle and hand control
back? This is an exploratory transfer question. A failed walk-only control
prevents the hop trial; a failed transfer is not automatically a hop-training
failure. No tuning or retries are included.

| Sequential case | 0–3 s | 3–4 s | 4–7 s | 7–9 s | 9–12 s |
| --- | --- | --- | --- | --- | --- |
| Walk-only control | walk at 0.20 m/s | same walker, zero | same walker, zero | same walker, zero | walk at 0.20 m/s |
| Conditional transfer | walk at 0.20 m/s | same walker, zero | Happy Hop, zero | original walker, zero | walk at 0.20 m/s |

Every other command slot remains zero. Each case starts independently from the
same fixed HOME/base pose with zero velocities and queues. Within a case the
raw-action history, 20 ms action queue and velocity-observation queue carry
through all switches. There are no random seeds or randomization in these two
deterministic nominal attempts; this is not a seed sweep or robustness test.

The control uses the frozen scorer's completion, fatal-event, approach,
stable-entry, drift, resumed-speed, current, soft-limit and derived-finiteness
indicators. Additionally, its full 8–9 s zero-command window must be stable and
it must have **zero qualified airborne episodes** during 4–9 s. Its result must
not use the hop scorer's overall decision, which correctly rejects a no-hop
trace. A separately tested baseline adapter is a launch prerequisite. The
conditional hop case must clear every unchanged C1 diagnostic indicator.

## Known declarations versus deliberately chosen surrogate settings

Actual pinned ONNX metadata was inspected without inference. Both artifacts
declare the same ordered 14 servo names, scale 1.0, observation-term order and
three-decimal HOME values. `velstand`'s run path is `None`; Happy Hop's names its
iteration-1255 export. The walk manifest names `protective_fall` but not an exact
training commit. This strengthens static interface evidence only: the metadata
does not prove normalizer parity, joint feedback semantics, delay matching or
behavioral compatibility. Rounded HOME metadata is **not** used as a new pose;
the declaration pins the existing family HOME constants instead.

For this independent experiment we deliberately choose the source-pinned
all-collisions stock/backlash scene, retaining its collision masks and contacts
(ten floor-capable robot geoms, including both soles). No body collision is
removed, contact softened, robot supported or spring mechanism substituted.
The 14 position actuators must be converted to true torque motors driven by the
pinned BAM XL330 M6 model: 7.4 V, firmware gain 200, modeled 1.75 A PWM-current
limit, zero voltage sag, output-side position feedback and motor-side back-EMF/
friction velocity. The initial servo and passive poses, solver, contacts and
motor edits must be captured from the **effective compiled plant**, not assumed
from the source XML. The existing simplified PD replay is not eligible.

Choose a one-step output-side joint-velocity observation lag for both policies,
no other observation lag/noise/bias, a carried one-step action delay, no action
filter/clipping and only the baked ONNX normalizer. These are explicit **our
surrogate settings**, not recovered Happy Hop/velstand training settings. Native
CPU MuJoCo uses 5 ms physics and 20 ms control; ONNX Runtime uses only its CPU
provider with one intra/inter thread. There is no CUDA, remote service change
or dependency upgrade. Current source/package/parameter hashes are in the JSON.

## Launch gate, budget and retained outcome

Before any policy execution, implement and test artifact/metadata checks using
the same verified ONNX bytes passed to inference; compiled joint/encoder/geom
binding; output-side BAM feedback with motor-side rotor math and current/friction
instrumentation; queue/lag parity at every switch; actual mesh-bottom clearance,
contact and substep-force capture; the baseline adapter; and bounded first-fatal,
timeout and partial-artifact handling. Fail closed on an unresolved preflight.
These implementation gates are not yet passed by this predeclaration.

Maximum budget: one setup capped at 60 wall seconds, then at most two sequential
12-simulated-second attempts capped at 60 wall seconds each; campaign cap 180
wall seconds. There is no auto-reset/retry, confirmation, speed/height sweep,
training, video or promotion. Timeout preserves a partial failure, not a smaller
success denominator. Use unique directories below
`artifacts/community/c1s-velstand-substitution-v1`; never overwrite a result.

The future retained decision is deterministic: preflight incomplete → stop;
control rejected/incomplete → stop without hop; control indicators pass → one
transfer attempt; transfer rejected/incomplete → stop and diagnose read-only;
both indicator sets pass → **nominal substituted-entry diagnostic candidate
only**. Retain both cases' traces, failures, metrics, identities and hashes,
including first-touchdown+100 ms **and** entire hop/settle force peaks. Impact
limits, thermal model, author replication, full skill acceptance, training,
deployment transition and physical-motion admission all remain unverified/false.

The [original matched C1](2026-10-01-community-hop-c1-preparation.md) remains
blocked on its author source handoff; this surrogate experiment does not resolve
or relabel that result. The proposed next code step is the tested native-CPU
runner and walker-only adapter; the baseline-first diagnostic additionally
requires a separate run go-ahead and stays within this budget.

## Predeclaration checks, not experiment results

- Declaration SHA256:
  `236d7ca25e1a0508e6a913944f822a44244fb032362bd4ee3a38e5315c9eb700`.
- 12 new declaration checks; 278 scoped tests passed with CUDA hidden, including
  existing C1, static skill binding/retention and historical hop gates.
- Both retained ONNX files were rechecked against the pinned hashes and exact
  API1 graph signatures without inference; all admission flags remain false.
- Raw scene compilation/settings were inspected without stepping physics.
  Compiled motor edits and BAM runtime binding are still future checks.
- A read-only Luna review checked controls, budget and identities and made the
  preparation-only scope and separate execution go-ahead explicit.
- `git diff --check` passed; Markdown rendered with one table and three local
  links resolved. Full repository tests, runner tests, rollouts and training
  were not run. No remote service, dependency or GPU workload was changed.
