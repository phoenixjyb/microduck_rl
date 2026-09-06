# Frozen straight-line speed response: predeclaration

Parent: `94df511a6cd5818eef43ceab07d3db3807fbe2de`. Simulation only, no robot.
Protocol: **`frozen-straight-speed-s383-v1`**. This is a new no-obstacle control,
not a retry or reinterpretation of the closed seed-379 recovery-cap experiment.
The recovery-cap pilot remains blocked by its failed prerequisite.

## Source audit and question

The retained obstacle baseline requested roughly .30 m/s but averaged .20347
m/s along its route during recovery. Its representative trace was still
turning/rejoining laterally. `hierarchical_obstacle_rollout._route_state`
projects world XY velocity onto the original route. By contrast,
`scripts/evaluate_checkpoint.py` reports full body-frame root-link velocity X.
The installed mjlab `EntityData.root_link_lin_vel_b` uses the inverse root
quaternion, not a route projection. These are legitimately different quantities.
No conversion defect has been established. Body command samplers in both
evaluators disable world/heading/forward-only overrides. Command changes in the
hierarchical loop reach already-returned actor observations on the next step;
that delay is a possible transient concern, not proof of the sustained deficit.

The old general evaluator excludes startup from both speed and failure counts.
This diagnostic needs a settled speed window but cannot ignore startup failures,
so it uses a separate small loop instead of altering historical evaluation.
Question: with no obstacle, no supervisor and no commanded yaw, does this exact
actor achieve .30 m/s in body and initial-heading route frames? Can it also
meet the *unchanged* instantaneous route-speed window? This can distinguish an
open-loop speed response issue from frame/turning or instantaneous-band issues;
it cannot establish which caused all seed-379 outcomes.

## Frozen setup and numerical boundaries

- Host/worktree/branch: 100.100, `/home/converge/work/microduck_rl-athletics-obstacle-curriculum`,
  `feat/athletics-obstacle-curriculum` only; exact clean tested launch SHA.
- Actor: `logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt`,
  SHA-256 `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- Task `Mjlab-Run-MotorAware-Flat-MicroDuck`, play configuration, same **61D**
  actor observation. No obstacle entity or obstacle supervisor. Clear curriculum
  and push event; retain other play/reset settings. Head/body commands fixed zero.
- Seed **383**, torch/environment/agent seeded; **8 environments**, **400**
  control steps at **0.02 s** (8 s maximum). Initial **100** steps (2 s) are
  startup; settled window is remaining **300** steps (6 s). No seed/checkpoint search.
- Fix body velocity command to `(0.30, 0, 0)` and yaw to zero, with no world,
  heading, turn-in-place or forward-only sampler overrides. Verify each pre-step
  command. Original-route direction is each robot's initial yaw direction.
- Measure pre-step body-forward, initial-route-forward, cross-route speed and
  heading error separately. Keep all and settled summaries plus per-environment
  speed means. Never equate body speed with route speed at nonzero heading.
- Use the tested logging-only pre-reset motor stream. Capture named forces and
  joint velocities at every control step, including the terminal step. Retain
  legacy post-return motor summaries separately; do not replace their limits.
  Stop **all** environments on the first terminal event, including startup.
  No reset episode contributes another velocity sample. Nonfinite inputs,
  actions, rewards, motor samples or observations fail closed.
- All-period and settled legacy torque-utilization p99 must remain <= **.60**.
  Both legacy and pre-reset rated-speed exceedance fractions must be zero.
  Terminal events, incomplete coverage or changed commands close the control.
  Pre-reset torque/per-joint p99 and squared-utilization proxy are descriptive,
  not new hardware ratings or substitutes for the legacy torque gate.

After safety/coverage checks, classify deterministic source summaries:

1. Any environment's settled body mean outside .30 +/- .03 m/s:
   `straight-body-mean-outside-band`.
2. Body means within band but any initial-route mean outside:
   `body-route-response-diverge`.
3. Both means within band but not every environment meets the unchanged sampled
   .50-second continuous route-speed band within 2 seconds of settled-window
   entry: `mean-tracking-but-instantaneous-window-missed`.
4. All criteria met: `straight-response-within-both-criteria`.

The existing recovery observer uses phase 0 for startup and phase 2 for this
*synthetic settled-window entry*, not a claim that an obstacle was passed. Its
raw .03/.50/2.0 criteria are unchanged. Report fields explicitly identify the
straight-line protocol. None of these diagnostic classifications authorizes
training, changes any old outcome or admits a physical policy.

## Runtime, retention and next boundary

Entry: `python -m mjlab_microduck.speed_response_control --source <full SHA>`.
Exactly one retained user service, `RuntimeMaxSec=240`, `TimeoutStopSec=15`,
`KillMode=control-group`, `Restart=no`; no subprocess retries. Launch only when
the four-minute ceiling plus five-minute closeout margin fits before
**2026-09-07 07:00 Asia/Shanghai**. GPU idle/cool, both protected services
inactive and actor/runtime/dependency hashes verified before launch. Runtime
versions/dependency hashes remain those of the seed-373 motor audit.

Output: `artifacts/evaluations/frozen-straight-speed-s383-v1` on 100.100, exclusive
directory, fsynced launch/response/decision/runtime JSON, retained systemd journal.
Copy evidence locally with hashes. Source/tests and predeclaration must be
committed/pushed before launch. No actor update, optimizer, dataset, MP4 or raw
perception. If the service exceeds its cap, retain its journal/launch receipt,
classify as incomplete runtime failure and do not retry in the same directory.

After completion retain source and artifact hashes plus frame-specific findings.
Diagnose any failure read-only. Keep the seed-379 stop immutable; a straight-line
pass is not obstacle recovery or PPO admission. Any further experiment requires
a distinct predeclared question, bounded runtime and unchanged safety gates.
Preserve protected services and all unrelated work; do not use 100.98.

Precommit validation: **558 focused CPU tests passed locally** (8.92 s), including
34 new frame/configuration/classification/orchestration checks and a mocked
actual-control-loop startup-terminal test. That test verifies the pre-reset
sample is consumed, the entire loop stops and reset velocities cannot enter
the summary. Two pre-existing actuator/site-pattern warnings remain. No live
straight-control outcome was available when this protocol was predeclared.

## Retained result: speed deficit without obstacle/supervisor

Executed once at **`0679be398b29ffc79dcf003001869e4d9a146afe`**, after 558 focused
CPU checks passed on 100.100 (7.39 s). User service
`microduck-rl-straight-speed-s383-0679be3.service` ran **00:06:58–00:07:15
Shanghai September 7**, exited 0, with 7.5346382754 s inside the diagnostic.
The retained unit is `active/exited`, MainPID 0; it is not a running GPU job.

All 400 control steps completed. No terminal event (including startup), changed
command, nonfinite sample or rated-speed exposure. Actor observation remained
8 x 61 and actor hash matched. All-period / settled legacy torque p99:
**0.5166896582 / 0.5082634687**, below .60. Pre-reset pooled torque p99:
0.5166892692 / 0.5082631126; near equality here is expected with no resets and
does not invalidate the earlier terminal-sampling audit. Squared-utilization
proxy is not physical motor temperature.

Settled six-second measurements:

| Quantity | Observed |
| --- | --- |
| Body-forward mean | **0.2102451335 m/s** |
| Initial-route-forward mean | **0.1852372128 m/s** |
| Per-environment body means | **0.1976068256–0.2431169455 m/s**, all 8 below .27 |
| Per-environment route means | **0.1374899797–0.2192060368 m/s** |
| Route speed p05 / p95 | 0.0681508623 / 0.2519421816 m/s |
| Cross-route absolute speed mean | 0.1089139541 m/s |
| Maximum absolute heading drift | 1.4794538021 rad |
| Stable route-speed window | **0/8**, all window-missed, none censored |

Classification: **`straight-body-mean-outside-band`**, safety failures empty.
This is a completed diagnostic, not a successful speed-tracking controller.
Training, physical admission and reopening the recovery A/B remain false.

Inference supported by this control: navigation/rejoining is **not necessary**
for a speed deficit to appear in this runtime; it persists without an obstacle
or supervisor and even in body-forward velocity. Route projection adds another
deficit alongside substantial yaw/lateral drift under a zero-yaw command. It
does **not** establish whether the root cause is policy behavior, command/input
semantics, runtime/configuration mismatch, low-speed coverage, or a combination.
One seed/eight environments is not multi-seed policy acceptance. Do not lower
the .30 goal to .21 or change the recovery gate to make this look successful.

Four original JSON artifacts remain in
`artifacts/evaluations/frozen-straight-speed-s383-v1` on 100.100; byte-identical
copies are local at `artifacts/diagnostics/frozen-straight-speed-s383-v1`.
Systemd journal remains retained under the named unit. Exact SHA-256:

| File | SHA-256 |
| --- | --- |
| `response.json` | `4e4d94d82f90621fe4441df907236de1b76c7a0e5950b948dfdd1354d6fa3c73` |
| `decision.json` | `e47e8dbea684e7d65e968bcc9a505436576b9173c95ae90f00c95ded37d73959` |
| `launch.json` | `c3fb377ce4aa8735cc0f69d5cd52be669135577d6db63230f6383d0766fc94fc` |
| `runtime.json` | `a207b34590137bdc5d8ca908a8e3bd86f16cf0d24dcf610ac3a005bc8424ee8a` |

After completion: GPU idle (0%, 45 C, 12 MiB), no compute process; both protected
services inactive, source clean. Checkpoints unchanged. No optimizer, cap arm,
additional seed, video or physical motion was run.

**Next handoff:** close this one-shot control, no retry. Audit the frozen actor's
saved training config and normalization/command observation against the actual
installed play configuration and retained Stage-2 speed evaluations. Separate
training coverage, zero-yaw drift, command delivery and measurement contracts.
Prefer source-level counterexamples/tests before any further GPU experiment.
The recovery-speed-only PPO prerequisite remains failed; do not substitute a
new low-level locomotion training job under that authorization. Any materially
different training scope must be explicitly agreed. Continue bounded diagnostic
and code/test work within the overnight deadline while preserving all gates.

Evidence-closeout validation: **559 focused CPU tests passed locally** (8.75 s),
including hash-bound replay of both retained diagnostic outcomes. Two existing
actuator/site-pattern warnings remain; no tests were skipped on this checkout.
