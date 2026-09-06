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
