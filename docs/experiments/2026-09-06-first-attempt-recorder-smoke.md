# First-attempt recorder runtime smoke

Date: 2026-09-06. Status: predeclared, not yet run. Diagnostic instrumentation
only; U3/U4 rejection, all numerical gates, and no-motion boundaries stand.

## Source and purpose

Recorder source is `5eb3bd25917280ff8aea0a00861f6a2ccb28c890`. Its 174 focused
tests passed with CUDA hidden locally (78.48 s) and on 100.100 (5.24 s), with
both worktrees clean at that commit. This smoke uses the subsequent exact
committed feature-branch revision containing this protocol and its independent
CPU validator, `mjlab_microduck.first_attempt_smoke`; capture that Git identity
before launch. The recorder and rollout implementation stay unchanged.

Purpose: check whether enabling passive recording changes existing reports,
and whether the sidecar's four first-attempt traces are structurally and numerically
consistent. This is not a policy comparison or a replay of historical failures.
No labels or trajectories from this smoke may enter training or acceptance.

## Frozen run

- Actor: `logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt`,
  SHA-256 `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- Supervisor: frozen rejected U4 `artifacts/checkpoints/hc4u4-bc60b2c-s42/supervisor.pt`,
  SHA-256 `29855a51df8fe885d6ffed7fedf028093a8449a68b10b4b0e8a4bde7069bcf5b`.
- Fresh physics seed **359**, four environments, one case: **0.40 m/s,
  0.90 m forward, 0.00 m lateral**, 700-step ceiling, first-terminal-attempt
  window, unchanged 12 s timeout and 0.02 s control step, exact compact geometry.
- Two separate processes, strictly sequential: recording disabled, then enabled.
  Both use identical actor, supervisor, source, configuration and seed. The only
  changed behavioral CLI option is `--record-first-attempts`; output paths are
  distinct. Never refit or retry for a pass.
- One retained user service on 100.100, total `RuntimeMaxSec=600`; each child
  has a 240 s timeout and foreground-only rollout. Service-level control-group
  termination bounds child cleanup. Fresh exclusive parent output directory:
  `artifacts/evaluations/first-attempt-recorder-s359-v1`, children `disabled`
  and `enabled`. Refuse existing output; retain commands, source, hashes,
  per-process wall times, logs and deterministic `decision.json`.

Before launch, verify clean exact branch, matching artifacts, no compute
process, GPU utilization zero, and both protected AI Mission services inactive.
Do not change those services or unrelated work. Launch no later than 06:25
Shanghai; the ten-minute cap leaves at least 25 minutes before the 07:00 stop.
Use no video, hopping, new observation, raw perception, or physical motion.

## Frozen validation and interpretation

All four first attempts must terminate. Strip **only** the enabled report's
top-level recording protocol and per-case sidecar descriptor. Then compare
canonical JSON of the complete reports with **exact equality**, including
all outcome counts, speed/motor/action metrics and the existing representative
trace. No float tolerance or field omission may be introduced after results.
This proves equality of retained outputs in this tiny case, not every internal
simulation state, RNG state, or general GPU determinism.

The independent validator checks report/case/checkpoint/protocol identity,
finite JSON, sidecar path and SHA-256, four unique ordered environment IDs,
first-attempt-only completion, exact five-step frame coverage plus immediate
pre-terminal frames, finite compact state, time and phase semantics, no
post-reset terminal state, failure-priority resolution of raw flags, and
reconciliation with per-case and pooled counts. Storage limits remain frozen.

A missing/malformed trace, inconsistent identity/count, nonfinite value, runtime
exception, timeout, or any report mismatch is a **recorder diagnostic stop**.
Report exact differences and diagnose read-only; do not assume instrumentation
causality or change the policy, tolerances or historical evidence. A matching
collision/timeout is a controller outcome, not evidence that the recorder changed
the trajectory. Report it separately. Even if reports match, any fall, hard/other
failure, nonfinite step, rated-speed exceedance or torque p99 above 0.60 closes
the next runtime gate. Those existing checks are not relaxed by instrumentation.

Only a validated recorder plus those runtime checks permits **predeclaration**
of a separate small fresh-seed trace diagnostic against U4 and its near specialist.
It does not launch that diagnostic, admit a new policy, reopen U3/U4, or explain
the uncaptured seed-347 collision. Observed process timings include startup/cache
effects and are descriptive, not a controlled overhead benchmark.

Prelaunch source checks: 200 focused smoke/recorder/rollout/U4 contract and
collection/U3 and U1 gate tests passed locally with CUDA hidden (71.06 s).
The 26 smoke validator/runner tests passed again after final runner review
(5.47 s), including dirty-source and existing-output refusal, sequential
dispatch, child error/timeout closure, tampered sidecars, nonzero-environment
collision accounting, and exact-report mismatch rejection. Remote repetition
on the exact committed source is required before launch.
