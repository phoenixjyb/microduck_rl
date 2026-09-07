# Bounded MicroDuck work through September8 07:00 Shanghai

User request: continue the next training round and plan the curriculum well,
working until07:00. Window ends **2026-09-07T23:00:00Z**. This is a bounded
simulation research campaign, not a promise to complete every skill overnight.

## Curriculum and anti-forgetting plan

| Order | Training objective | Required evidence before moving on |
| --- | --- | --- |
| F: stable economical gait | Correct route/speed/lateral motion without excess motor loading; one reward/control axis per experiment | Existing absolute and paired speed, heading, lateral, torque, power and load gates; then three independent training seeds and untouched confirmation seeds |
| S: stop and recovery | Ramp to rest, stand, restart, then bounded perturbations one axis at a time | First-attempt no falls, settled stop/restart windows, motor limits, and rerun the accepted F matrix on the composed controller |
| O: obstacle negotiation | Freeze admitted gait; train a low-rate supervisor on compact external geometry | Collision/timeout/route-return gates by placement/speed bin; nominal tracking before and after maneuver, slowdown permitted inside it; F/S retention |
| O-diversity | Vary range, bearing, geometry and speed separately, then structured noise/latency/dropout | Per-bin and worst-seed results, not pooled pass rate; safe stop under stale/missing input and recovery afterward |
| H: hop and landing | Separate progressive-height repair of the rejected H1 track; no claimed retained hop yet | Original H1 takeoff/landing/drift/spring/motor gates and independent-seed confirmation; no H2 on failed H1 |
| Integration | Explicit stop/bypass/hop routing with bounded transitions | Align rigid/sprung mechanics first, then test every earlier skill plus transition and interruption matrices; checkpoint preservation alone is not behavioral retention |

Keep an immutable skill library with checkpoint, training source, mechanics,
command/sensor envelope and evaluator hashes. Candidate work writes separate
directories; never overwrite the fallback gait or historical obstacle specialist.
Start with separate policies and frozen low-level actors; do not fine-tune a
single joint policy on obstacles and assume its walking/hopping survived.
Hopping repair can be planned in parallel conceptually, but no second GPU job
runs concurrently and tonight's first bottleneck is F, not higher obstacles.

## Tonight's finite experiment queue

1. Run [F1-M](2026-09-07-f1m-motor-continuation.md): identical lateral objective,
   motor weight-2 versus-4, paired seed499,500updates each. No broad gain search.
2. Close numerical evidence and compare speed/lateral versus torque/power.
   If all diagnostic gates pass, predeclare a matched independent-training-seed
   confirmation (same500updates, cadence and protocol). Do not extend iterations
   and change seeds at the same time. Candidate admission still needs the full
   three-training-seed matrix, never a single favorable result.
3. If F1-M fails, retain it and inspect failure type before another edit:
   motor improvement with lost speed calls for a single-axis objective-balance
   revision; persistent load concentration calls for a named-joint/timing
   diagnosis before modifying the reward; unsafe reference or infrastructure
   failure calls for read-only diagnosis, not a seed replacement or blind rerun.
   A new revision needs a committed hypothesis, exact parent, changed axis,
   fixed seed/budget/evaluator and focused tests **before** launch.
4. At most **three new paired500-update pilot experiments total tonight**
   including F1-M, or one F1-M plus two matched confirmation seeds. Smokes are
   ten updates per arm. Do not use leftover time for unbounded tuning. After
   this cap, continue CPU evidence audits, retention-test implementation and
   curriculum documentation until the deadline. Any GPU-backed retention
   check needs its own short predeclared protocol and a numeric entry gate.

Selection/evaluation seeds used to guide revisions become development data.
List them in the experiment ledger. Reserve a fresh, predeclared confirmation
matrix for final promotion, and include earlier development failures in the
report. Do not market adaptive development-seed improvement as generalization.
No stage advances merely because a larger reward or longer training looks good.

## Execution and handoff rules

Local `/Users/yanbo/Projects/microduckPlayground/microduck_rl`; remote
`converge@192.168.100.100:/home/converge/work/microduck_rl-athletics-obstacle-curriculum`;
exact branch `feat/athletics-obstacle-curriculum`. One write owner and one
retained GPU user service at a time. Verify clean pushed source, frozen runtime,
GPU PID/utilization/memory/temperature and both protected **system** services
inactive before launch. Preserve unrelated workloads and100.98/FilmBrain.

While a job runs, inspect progress/checkpoints, finite metrics, falls,
motor-load/soft-limit/thermal proxies and GPU state; do not edit its worktree
or service. On completion validate checkpoint counters and hashes, reconcile
the deterministic decision, mirror artifacts to the Mac and test/commit/push
concise evidence. Failures are diagnosed read-only before fixes; never erase
failed units/reports or automatically restart closed campaigns.

Scheduled thread follow-ups should be quiet on unchanged/non-actionable state
and report meaningful milestones, completion, real failure or required input.
Every service needs a hard runtime cap and a launch check proving its worst
case plus closeout fits07:00. No new GPU job after06:00; reserve the last hour
for completion/evidence and CPU work. Stop new Duck work at07:00, verify no
Duck compute process remains and checkpoints are durable, leave both protected
AI services inactive unless explicitly asked otherwise, delete the follow-up
automation and report retained state. If an unexpected job survives its cap,
diagnose it and preserve its newest durable checkpoint before stopping only
the exact Duck service; never start the protected workload concurrently.

No physical robot motion, raw perception, automatic MP4, weakened numerical
gate or claim of completed hopping is authorized by this work window.

## Live experiment ledger

- Paired pilot1/3: F1-M seed499 complete; source
  `fbc0b73ca1a61016ea8e4002f1af65e24990e892`, decision `numerical-gate-stop`.
  Two500-update pilots and two10-update smokes retained. All83 payloads mirrored
  and verified; five evaluation cases, first pair503 rejected. Treatment lowers
  load/power modestly but loses speed and redistributes load toward left knee
  and head roll. No extra509/521 candidate evaluations or seed replacement.
- Next bounded step: motor-timing instrumentation and a separately predeclared
  evaluation-only diagnosis of the already absolute-safe503 reports. Existing
  aggregate motor quantiles cannot establish the timing of the regressions.
  Do not spend paired pilot2 until this evidence is closed and a single-axis
  hypothesis is justified. Higher curriculum stages and hopping remain gated.
  Protocol and refusal rules are in
  [F1-M motor timing](2026-09-07-f1m-motor-timing.md); output
  `artifacts/experiments/f1m-motor-timing-s503-v1`,600second service cap,
  parent/control/motor seed503 only, zero optimizer updates.
- Timing diagnostic closed at its first parent case. Source
  `dbbeb8c8e708dd99968d6dde6330531baa5e2d1b`; service
  `microduck-rl-f1m-timing-dbbeb8c-s503.service`; decision
  `runtime-failure-stop` at exact historical replay identity. Parent was
  absolute-safe and internally self-consistent, but diverged first at step3;
  control/treatment timing cases were never started. All five new payloads
  mirrored and verified. See the motor-timing document for exact hashes.
- Current next step: read-only initialization/seeding/backend inspection,
  followed only if justified by a separately predeclared same-source OFF/OFF
  parent503 measurement control. No timing-directory retry, gate relaxation,
  third ON case or paired pilot2 before that measurement-validity issue is
  resolved. Continue bounded CPU evidence/retention work in the meantime.
- Read-only seed/backend inspection completed: mjlab seeds the configured
  generators but warns MuJoCo Warp is not fully deterministic; process-start
  Python hash seeding and complete historical backend fingerprints were not
  established. The separately predeclared next control is
  [same-source OFF/OFF parent503](2026-09-07-f1m-replay-control.md), module
  `mjlab_microduck.foundation_replay_control`, output
  `artifacts/experiments/f1m-replay-off-off-s503-v1`, exactly two sequential
  evaluation cases,420second cap, no backend changes and no optimizer updates.
- OFF/OFF control completed at23:05:53, source
  `79074f7b446a1a1ac854335855a9f2b10ceebce5`, service
  `microduck-rl-replay-off-off-79074f7-s503.service`. Both cases absolute-safe;
  decision `recording-disabled-same-source-divergence`,28,563 differing leaves.
  All10 payloads mirrored and the decision independently reproduced on CPU.
  Backend fingerprints match except process-start Python hash identity, which
  was observed uncontrolled. Before any third/new case, predeclare the single
  startup-hash control; no reward/precision/physics changes or pilot2 admitted.
- Next predeclared control: [startup hash503 OFF/OFF](2026-09-07-f1m-startup-hash-control.md).
  Same replay module with `--startup-hash`, new immutable output
  `artifacts/experiments/f1m-replay-hash503-off-off-v1`, two120second children,
  420second service cap, only child-start PYTHONHASHSEED changed. Zero optimizer
  updates; require effective equal hash probes; preserve all closed evidence.
