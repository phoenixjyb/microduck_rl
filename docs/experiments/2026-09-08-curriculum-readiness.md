# MicroDuck curriculum readiness: September8

Scope: the retained simulation campaign and the stricter current foundation/
retention gates, not a claim about every earlier experiment or a physical duck.
Evidence reviewed around05:20 Shanghai. No real robot is available. All3/3
overnight paired pilots are closed; no further GPU experiment is authorized
before the07:00 cutoff. This document is a readiness plan, not a launch ticket.

## What is demonstrated, and what is not

| Capability | Checked evidence | Current decision | Missing acceptance |
| --- | --- | --- | --- |
| Foundation gait at0.30m/s | F1-Y command delivery verified; final candidate misses speed/lateral and paired motor gates | Not admitted under current F protocol | A common passing checkpoint across independent training seeds and untouched confirmation |
| Stop, stable stance, restart | Action-manager and command-limit CPU checks; preserved gait weights | Not behaviorally validated by these audits | Explicit first-attempt stop/stance/restart and perturbation matrix, plus F retention |
| Near obstacle passage | Seed379 baseline has8/8 clean passages, no collisions/falls/timeouts | Demonstrated only in that retained case | Full speed/placement matrix with approach/recovery and motor gates |
| Speed recovery after passage | Same baseline:0/8 recoveries within the required window | Failed prerequisite; capped arm never ran | Nominal speed reacquisition without sacrificing stability or motor limits |
| Hop and landing | H1-T has hopping events; all six original summary evaluations rejected | H1 repair remains gated; no H2 | Original full H1 landing/drift/motor criteria and independent confirmation |
| Skill switching and sensor failures | Static/action-pipeline audits and a coverage checker exist | No integrated controller admitted | Bound mechanics/interfaces, directed transitions, interruptions and freshness-aware sensor handling |

Neither a saved checkpoint nor a green source test means a retained skill.
There is no meaningful single curriculum-completion percentage while these
acceptance gates remain open. Earlier narrow successes are preserved; they do
not waive this composed-controller protocol.

## Order of work and promotion rules

1. **F: repair the foundation first.** Keep body and route speed, lateral drift,
   heading, pooled/per-joint torque, squared load and power gates unchanged.
   Gross training guards are not held-out acceptance. Preserve all three
   rejected pilots, fixed-final selection and their development data. Before
   another training window, predeclare an evidence-backed single-axis
   hypothesis, exact parent/restoration, seeds, budget and stopping rule. Do not
   recycle development seeds as untouched confirmation or hunt a passing seed.
2. **S: stop/stance/restart, then bounded disturbances.** Use an admitted F actor
   as the reference; define measured motion/settling windows and the command
   ramp before collecting data. Rerun F on the composed controller. A zero
   desired velocity is not a zero motor action, and zero motor action is not
   proved safe standing. The dedicated numerical S protocol is still missing.
3. **O: obstacle negotiation with a frozen admitted gait.** Begin with one
   declared near, centered placement and speed, then vary one axis at a time.
   The supervisor may slow during interaction; approach and recovery must track
   nominal speed. Keep collision, timeout, first-attempt and motor accounting.
   Evaluate every placement/speed bin and worst seed, not just a pooled rate.
   Every proposed promotion must retain F/S. The failed379 baseline does not
   authorize its unexecuted cap treatment or recovery-PPO pilot.
4. **O-diversity: geometry, latency and dropout.** Only after O, extend range,
   bearing, width/height and speed separately. Independently predeclare sensor
   noise, age, dropout and reacquisition tests; do not invent real perception
   performance from exact simulated geometry. Raw camera perception remains
   outside locomotion RL.
5. **H: separate hop repair.** Preserve the rejected sprung-K3900 branch and
   its full H1 gates. A causal revision threshold is not a replacement for full
   H1 acceptance. Progressive-height work needs its own declared hypothesis and
   motor-aware evaluation; no Locked/H2/video follows from the current rejection.
6. **Integration: only after compatible admitted skills exist.** Resolve rigid
   versus sprung mechanics, ordered observations/actions, normalizers, timing
   and runtime before switching. Test all12 directed requests independently;
   some should be refused or routed through a tested landing/stance state rather
   than switched directly. Rerun every earlier skill on the composed controller.

The teacher's configured0.8m/s command ceiling is not a demonstrated operating
speed. Motor utilization/power/load in these reports are simulation metrics,
not calibrated continuous servo limits, physical thermal safety or a hardware
top-speed specification.

## Transition and interruption cases that must be predeclared

The [retention contract](2026-09-08-skill-retention-contract.md) and its
[coverage checker](../../src/mjlab_microduck/skill_retention_plan.py) define
four skill records,12 directed requests and16 cross-cutting case categories.
The initial plan still has72 unbound reference slots; this is missing binding,
not72 failed robot tests. A full reference catalog would still not admit skills.

Each relevant directed request needs a procedure and numerical evidence for:

- Fresh command and action processing before the first target application.
  Explicitly handle processed-target cache, action history, actuator delay/
  internal state, observation history and normalizer ownership. Do not assume
  an action-manager reset resets the entire environment or actuator chain.
- Stop while moving, stable stance, restart and nominal-speed reacquisition.
  Test terminal/reset/interruption cases separately; no failed attempt may be
  replaced by a later automatically reset success.
- Hop interruption before takeoff, during flight and during landing. An
  airborne interruption requires a defined abort/landing path before stance;
  instant zero actions are not a validated emergency landing policy.
- Missing, invalid and stale structured sensor data, then reacquisition.
  Define observation age and route/obstacle identity continuity; a reappearing
  packet cannot silently revive a stale phase or old action.

## Concrete sensor-contract gap: validity is not freshness

The existing [seven-channel obstacle packet](../../src/mjlab_microduck/tasks/obstacle_observation.py)
contains normalized range, bearing sine/cosine, width, height, closing rate and
validity. There is no timestamp/age/sequence field. The encoder zeros rows
flagged invalid or nonfinite; it cannot detect an old finite packet still
marked valid. Its zeroing behavior is not a freshness detector.

The [existing command limiter](../../src/mjlab_microduck/hierarchical_obstacle.py)
immediately zeros invalid-geometry commands in approach/interaction, while
recovery intentionally remains route-state driven after an observed pass.
Existing CPU tests cover that intended behavior. It is not evidence that sensor
loss in every phase produces a physical stop, and this review does not identify
it as a bug or explain the historical speed deficit. Do not patch the frozen
controller or reinterpret its old evaluations in this documentation slice.

A future external adapter must distinguish a fresh observation, a confirmed
clear/absent obstacle, transport loss, stale data and malformed input. Decide
whether acquisition timestamps, sequence/episode identity and freshness checks
live outside the actor packet, preserving trained dimensions where possible.
Define clock/age bounds, per-phase response and reacquisition semantics in a
separate tested contract before integration; no numeric age limit is selected
here. Closing this source-design gap still would not prove closed-loop safety.

## Retained evidence and immediate handoff

- [F1-Y closed pilot](2026-09-08-f1y-command-support.md): numerical stop at the
  first paired503 evaluation; candidate509/521 were not run. Five original
  reports are not a completed three-seed candidate acceptance matrix.
- [Recovery379 ledger](2026-09-06-overnight-recovery-curriculum.md): eight clean
  passages but0/8 speed recoveries. The selected gait/near/far checkpoint and
  eight-file receipt inventory is hash-verified on both hosts. The far model
  is a declared reference, not an executed far-case result.
- [H1-T ledger](2026-09-04-h0-h1-hop-campaign.md): at5999, episode pass, falls,
  drift, rated-speed exposure and near-stall still fail full H1. Its selected
  checkpoint/config/reports are mirrored; original summary decisions reproduce.
- [CPU retention audits](2026-09-08-skill-retention-contract.md): preserve the
  distinction between byte identity, synthetic action behavior, historical
  summary reconciliation, effective runtime binding and behavioral acceptance.

Remaining work this window is CPU-only review/tests, evidence organization and
the07:00 closeout. No new policy, actor-observation expansion, GPU evaluation,
training retry or physical motion is launched by this readiness plan. At the
deadline, start no new Duck work, verify checkpoints and no Duck compute process,
leave the protected AI services inactive, delete the scheduled continuation and
report the final retained state. Future training needs a separately authorized
window and fully predeclared experiment.

Validation:136 focused CPU tests passed locally, covering the existing obstacle
packet/phase behavior, retention-plan checker and retained H1/recovery audits.
The Markdown was rendered to HTML: the readiness table has seven rows with
four cells each, and all eight document/source links resolve. This was a
programmatic render/content check, not a screenshot-based UI inspection.
No historical evaluator, controller, packet layout, test threshold or policy
artifact was changed by this documentation slice.
