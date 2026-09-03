# OA0R outcome-balanced campaign, 2026-09-03

## Outcome

Decision: **rejected, simulation-only**. No checkpoint was promoted and no
physical motion was authorized.

The campaign used source `fae0b8b`, the same retained motor-aware warm start,
training seeds 42–44, and held-out evaluation seeds 41–43. Relative to OA0, its
only task change was a time-step-normalized `+20` route-rejoined success / `-20`
collision-or-attempt-timeout outcome.

| Iteration | Pooled route-rejoined pass rate | OA0R gate |
| ---: | ---: | --- |
| 8000 | 26.134% | fail |
| 8016 | 17.701% | fail |
| 8032 | 10.047% | fail |
| 8048 | 6.477% | fail |
| 8061 | 8.025% | fail |

All candidates had zero falls, NaN terminations, and non-finite steps. The
earliest checkpoint's per-training-seed rates were 22.118%, 25.828%, and
30.693%. Its pooled pass rate improved only one percentage point over OA0, and
its per-seed pre-obstacle route speeds remained 0.204–0.210 m/s, below the
0.22 m/s gate. Later training again regressed.

## Diagnosis and next axis

Outcome scaling was necessary but not sufficient. The task still applied the
nominal linear-velocity reward and a forward route-speed reward throughout the
avoidance maneuver. That makes slowing or braking to preserve clearance fight
the ordinary locomotion objective.

OA0P retains the 0.30 m/s external command and the complete OA0R safety and
outcome contract. It applies normal speed shaping on approach, suspends only
linear-speed and route-progress shaping while the obstacle is in the
interaction zone, and restores both after the robot passes the obstacle center.
This tests one semantic axis: phase-aware speed tracking.
