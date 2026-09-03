# MicroDuck athletics and obstacle curriculum

This document defines the training order and evidence gates for teaching the
MicroDuck to run, hop, jump, and negotiate obstacles in simulation. Perception
is a separate subsystem: locomotion consumes compact geometry and maneuver
intent, never raw images.

The curriculum is deliberately sequential. A policy advances only after a
retained, multi-seed evaluation passes. A later skill must not hide a regression
in an earlier one.

## Capability graph

1. **Foundation** — stand, recover, and track low-speed commands.
2. **Motor-aware running** — track 0.5 m/s without exceeding the rated motor
   envelope. This is the parent for obstacle locomotion.
3. **Bypass navigation** — learn a compact, route-preserving detour around a
   tall obstacle.
4. **Hop** — leave and regain the ground in place with controlled landing.
5. **Forward jump** — add forward displacement to the accepted hop primitive.
6. **Obstacle jump** — clear only low obstacles whose geometry is within the
   accepted jump envelope.
7. **Skill routing** — an external supervisor chooses stop, bypass, or jump;
   it remains outside the locomotion policy and retains safety authority.

Running, bypassing, and jumping are separate policy/evidence tracks until each
is stable. They may later share a network, but a shared network does not weaken
their individual acceptance gates.

## Bypass ladder

The failed centered-box campaign started at the graduation problem. The next
campaign starts with a visible lateral hint and changes one difficulty axis at
a time.

| Stage | Obstacle placement | Command | Sensor | New difficulty | Promotion target |
| --- | --- | --- | --- | --- | --- |
| OA0 | 1.15 m forward; signed absolute lateral offset 0.24–0.30 m | fixed 0.30 m/s | exact, zero lag | learn “move away, pass, return” | >=85% clean passes per train seed |
| OA1 | 1.15 m forward; signed absolute lateral offset 0.10–0.16 m | fixed 0.30 m/s | unchanged | smaller lateral hint | >=80% per seed |
| O1a | 1.15 m forward; centered | fixed 0.30 m/s | unchanged | remove lateral hint | >=75% per seed |
| O1b | unchanged | fixed 0.40 m/s | unchanged | speed only | >=72% per seed |
| O1 | unchanged | fixed 0.50 m/s | unchanged | speed only | existing exact O1 gates |
| O2a | forward position 0.90–1.30 m; centered | accepted O1 speed | unchanged | range only | O1 gates across range bins |
| O2b | O2a plus lateral position -0.25–0.25 m | unchanged | unchanged | bearing only | O1 gates across bearing bins |
| O2c | O2b plus measured width/height envelope | unchanged | unchanged | geometry only | gates across geometry bins |
| O3a | unchanged | unchanged | measured range noise | range noise only | no more than 5 percentage-point pass loss |
| O3b | unchanged | unchanged | add measured bearing noise | bearing noise only | same regression bound |
| O3c | unchanged | unchanged | add measured latency | latency only | same regression bound |
| O3d | unchanged | unchanged | add measured dropout | dropout only | fail-safe stop plus retained recovery |

The signed absolute offset in OA0/OA1 samples both sides but excludes the hard
center band. This supplies a bearing sign while keeping the actor interface
unchanged. O1 remains protocol `O1-centered-exact-v1`; training scaffolds must
not rename or weaken that benchmark.

### Route-preserving objective

The bypass policy must do all three actions in order:

1. commit laterally early enough to maintain the collision margin;
2. continue forward past the obstacle;
3. converge back toward the original route.

Reward alone is not accepted as evidence. Each stage reports clean passage,
collision, fall, non-finite state, pre-obstacle route speed, passage time,
maximum lateral excursion, and post-pass route-return error. A bounded attempt
horizon marks lingering as failure. Its initial value must accommodate the
stage speed and distance; it is reduced independently only after clean passage
is reliable.

## Hop and jump ladder

Obstacle jumping starts only after motor-aware running and the independent hop
track both pass.

| Stage | Task | Difficulty axis | Core evidence |
| --- | --- | --- | --- |
| H0 | symmetric in-place hop | target apex | takeoff, airborne phase, landing, no fall |
| H1 | repeated in-place hops | repetition count | height consistency and thermal estimate |
| J0 | forward jump without obstacle | forward distance | landing stability and route error |
| J1 | fixed low obstacle | obstacle height | collision-free clearance margin |
| J2 | varied low obstacle | height bins | per-bin pass rate |
| J3 | low obstacle with sensor perturbation | one sensor axis per stage | bounded degradation and fail-safe stop |

A high or uncertain obstacle is never silently assigned to the jump policy.
The supervisor checks accepted height, width, approach-speed, landing-zone, and
motor envelopes; otherwise it selects bypass or stop.

## Evidence and selection rules

- Use at least three independent training seeds and at least three held-out
  evaluation seeds per candidate.
- Compare exact checkpoint iterations shared by every training seed. Select the
  earliest passing iteration, never the final checkpoint by default.
- Keep protocol identity, source commit, parent checkpoint hash, training seed,
  evaluation seeds, and checkpoint hashes in retained manifests.
- A stage fails on any fall, NaN termination, or non-finite step in its bounded
  acceptance matrix.
- Run motor-envelope and reset-safe action-rate comparisons only after the task
  gate produces a survivor.
- Record MP4s only for a numerically accepted candidate; video is qualitative
  evidence, not a substitute for the metrics.
- Every result remains simulation-only. No report or checkpoint authorizes
  physical motion.

## Current evidence and next implementation gate

The 2026-09-03 exact O1 campaign trained seeds 42–44 for 64 iterations from the
same motor-aware warm start. Its best pooled clean-pass rate was 14.606% at
iteration 8061, against the 75% campaign target. There were no falls, NaN
terminations, or non-finite steps. Training traces showed increasing reward but
falling route speed and growing lateral displacement, consistent with a
side-step-and-linger strategy. The retained acceptance decision is rejected.

The next code slice is OA0 only: signed-offset reset sampling, a bounded attempt
horizon, post-pass route-return measurement, and an OA0-specific retained
evaluation. No new GPU campaign starts until those pieces have focused tests and
a CPU configuration/runtime smoke pass.
