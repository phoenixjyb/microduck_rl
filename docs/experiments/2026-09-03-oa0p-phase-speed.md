# OA0P phase-aware-speed pilot, 2026-09-03

## Outcome

Decision: **rejected after one training seed, simulation-only**. Seeds 43 and
44 were not started, no checkpoint was promoted, and no physical motion was
authorized.

The pilot used source `ed7eb1f`, the original retained motor-aware warm start,
training seed 42, and held-out evaluation seeds 41–43. The external command
remained 0.30 m/s. Relative to OA0R, only linear-speed and route-progress
shaping were suspended between 0.60 m ahead of the obstacle and passage of its
center. Angular tracking and every safety/outcome term were unchanged.

| Iteration | Clean pass | Approach m/s | Interaction m/s | Recovery m/s |
| ---: | ---: | ---: | ---: | ---: |
| 8000 | 18.154% | 0.2080 | 0.2102 | 0.2114 |
| 8016 | 8.214% | 0.1953 | 0.1921 | 0.1880 |
| 8032 | 5.283% | 0.1874 | 0.1827 | 0.1829 |
| 8048 | 3.333% | 0.1788 | 0.1722 | 0.1872 |
| 8061 | 3.320% | 0.1789 | 0.1739 | 0.1909 |

Every checkpoint had zero falls and zero non-finite steps. All failed the 85%
pass gate and both the 0.22 m/s approach and recovery gates.

## Interpretation

Phase-aware speed is the correct task rule: the duck may slow or brake while
avoiding an obstacle but must track speed before and after. OA0P demonstrates
that reward gating alone cannot preserve that rule when the same PPO update
changes the complete 14-joint locomotion policy. Approach speed regressed even
where its original reward remained active, which is catastrophic interference
with the learned gait.

The next experiment must freeze the accepted motor-aware locomotion actor and
move avoidance to a bounded high-level command policy. No further OA0P seeds
should run.
