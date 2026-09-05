# First-attempt recorder runtime smoke

Date: 2026-09-06. Status: stopped at exact-report equality; not validated. Diagnostic instrumentation
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

## Retained runtime result and read-only diagnosis

Source: `f4190775abd110ffa54be275f7c862f0c747ce47`. All 200 focused tests passed
on 100.100 with CUDA hidden (5.30 s), and the source/actor/supervisor/idle-GPU/
protected-service checks passed. `microduck-rl-recorder-s359-v1.service` ran
04:51:09--04:51:42 Shanghai. Both children exited zero, taking 15.7359837843 s
(disabled) and 15.9853184209 s (enabled). The parent deliberately exited **2**
on `ValueError: legacy reports differ; exact equality required`; this is the
predeclared diagnostic stop, not a child crash or timeout. No retry was run.

Both passes had four clean first attempts and zero collisions, timeouts, falls,
NaN/nonfinite, hard/other failures, unresolved attempts, or rated-speed exceedance.
The sidecar passed independent structural, identity, hash, frame-coverage and
outcome-reconciliation checks: **four environment IDs, 324 frames, four passes**.
This does not establish instrumentation equivalence because report equality failed.

Read-only recursive comparison found 604 leaf/length differences: 15 non-trace
scalar fields and 589 representative-trace differences (including its length).
All outcome totals were identical. Selected disabled/enabled differences:

| Retained field | Disabled | Enabled |
| --- | ---: | ---: |
| Simulator steps executed | 435 | 438 |
| Mean passage time (s) | 7.944999694824219 | 7.980000019073486 |
| Mean lateral excursion (m) | 0.3644675761461258 | 0.37549589574337006 |
| Recovery route speed (m/s) | 0.28611719608306885 | 0.2838948369026184 |
| Motor torque utilization p99 | 0.5279281735420227 | 0.5256340503692627 |
| Motor thermal proxy mean | 0.03383292630314827 | 0.03381514549255371 |

The representative trace's initial state and commands match exactly. The first
retained divergence is at **0.1 s**: route speed 0.0037332605570554733 versus
0.0037333075888454914 m/s, and commanded speed 0.30400991439819336 versus
0.30400994420051575 m/s. Later traces and terminal times differ. Source review
confirms neither the rollout nor recorder changed between the recorder commit
and this smoke. The validator/runner are the only added execution paths.

These two runs cannot distinguish ordinary process-to-process simulation
variation from a recorder-induced scheduling/numerical effect. Identical initial
representative state does not prove equality of every internal state or RNG.
There was no recording-disabled/disabled repeatability control in this protocol.
The tiny initial divergence therefore is **not proof of either cause**; neither
the recorder nor the historical seed-347 collision is causally explained.

The complete stop decision was recomputed from the same absolute report paths
on CPU and matched exactly. (A first verification using relative paths differed
only in the returned sidecar path; using the original absolute inputs resolves
that verification-input mismatch, without changing any artifact or decision.)
All files are retained on 100.100 under
`artifacts/evaluations/first-attempt-recorder-s359-v1` and backed up locally under
`artifacts/overnight-20260906-u4/first-attempt-recorder-s359-v1`.

| Artifact | SHA-256 |
| --- | --- |
| `decision.json` | `292024b847ee6aa2b74c084181fd9c985055b8838d8bb68a8e15a38c124ea788` |
| Disabled report | `5cd1798d3a25bfb4882c480a57d6279ed24fc4f4184f5147eafd9b318735e70b` |
| Enabled report | `9411efd9250625c2ef2e57f834abfe9760d38092a640873d383e4ad4a12e4217` |
| Enabled `first-attempt-traces/case-000.json` | `a7fc2569c679b99cc50e9df3beaa84150500acc413f0ecc5eb94007c2fdf05fc` |
| `runtime.json` | `135a15e958d0f5c4a342e8ef808749ddef110c0ac8e124fa6598558d794b0a55` |
| `launch.json` | `ad51644747d829d38711f149f6e115a820bb22dda334cb4dfa48e7f9318ef1d7` |
| Disabled log | `4a8214718d3f812169e6764b4ab5eff5823d01f1c91331b9ad1bfb8e60ac76e3` |
| Enabled log | `cec2253941a219b52c787f609e28ee860145bcc1328dff005662a5d80c6df22f` |

The GPU returned idle (12 MiB, 0%, 48 C), with no compute process and both
protected services still inactive. Source remained clean. No policy, recorder,
comparison field, tolerance or historical decision was changed after the stop.

Next bounded work is source-only repeatability-study design: predeclare a tiny
fresh-seed recording-disabled/disabled control, with fixed source/artifacts,
counts, identity, strict comparison, bounded runtime and no retries, before
considering another GPU diagnosis. No such control is launched or authorized
by this result alone; the overnight workflow must first commit its separate
protocol and focused tests. Even if baseline variation is later observed, it
cannot retroactively pass this smoke or establish recorder non-interference.
The larger U4/near trace diagnostic, new training, and promotion remain closed.

Retention verification: all eight local-backup hashes match the remote files
and the table. The 26 focused validator/runner tests passed again with CUDA
hidden (1.19 s) before committing this unchanged-code diagnostic evidence.
