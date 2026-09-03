# OA0 offset-assisted campaign, 2026-09-03

## Outcome

Decision: **rejected, simulation-only**. No checkpoint was promoted and no
physical motion was authorized.

The campaign used source `778f8c6`, the same motor-aware warm-start checkpoint,
training seeds 42–44, and held-out evaluation seeds 41–43. Each seed ran 64
iterations at 0.30 m/s with a signed absolute obstacle offset of 0.24–0.30 m.

| Iteration | Pooled route-rejoined pass rate | OA0 gate |
| ---: | ---: | --- |
| 8000 | 25.134% | fail |
| 8016 | 20.129% | fail |
| 8032 | 13.372% | fail |
| 8048 | 12.352% | fail |
| 8061 | 7.665% | fail |

At iteration 8000, per-training-seed rates were 19.136%, 28.477%, and 28.197%.
All successful routes returned within the 0.15 m corridor and all candidates
had zero falls and zero non-finite steps. Failures divided between collision
and the explicit seven-second attempt timeout. Pre-obstacle speed remained
0.18–0.21 m/s, below the 0.22 m/s gate.

## Diagnosis and next axis

Mean episode length grew toward the attempt horizon while route speed fell and
later checkpoints regressed. MJLab scales reward terms by `step_dt=0.02`; the
original one-step `+10/-10` outcomes therefore contributed only `+/-0.2`, while
one additional second of ordinary locomotion could earn roughly 12 reward
points. The task economics still favored lingering.

The next experiment is the distinct OA0R protocol. It changes only terminal
outcome scale by adding a time-step-normalized `+20` success / `-20`
collision-or-timeout impulse. Geometry, speed, sensors, horizon, and selection
gates stay fixed, so the comparison remains attributable.
