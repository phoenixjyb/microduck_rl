# HC4-E1 fixed-attempt evaluation protocol

Date: 2026-09-05

Decision: **predeclare and smoke-test the evaluation protocol before training
another obstacle-composition candidate**

## Why this protocol is needed

HC4-R2L remains rejected under its predeclared fixed-step gate. It recorded
2,588 clean passages, two collisions, and eleven timeouts among 2,601 resolved
attempts, versus 2,591 clean passages, two collisions, and eleven timeouts
among 2,604 resolved paired-source attempts. Its 99.5002% pooled clean rate
missed the fixed 99.501% continuation floor. That decision is not rerun,
rescored, or promoted post hoc.

The unequal resolved counts show that a fixed simulator-step budget is too
sensitive to episode duration for very close controller comparisons. A faster
controller can begin more automatically reset episodes before the same step
ceiling, changing the denominator as well as the outcomes. HC4-E1 fixes the
protocol for future candidates; it does not revise old evidence.

## Frozen evaluation window

Protocol identity: `first-terminal-attempt-per-environment-v1`.

- Each environment contributes exactly its first terminal attempt after the
  initial reset.
- A collision, clean passage, attempt timeout, fall, NaN termination, or other
  terminal event closes that environment's evaluation window.
- Samples after automatic reset are excluded from route, command, action,
  motor, finite-state, and outcome metrics.
- `num_envs` is the fixed denominator for every cell. `steps` is only a ceiling
  that lets every first attempt terminate.
- Reports retain expected, completed, unresolved, clean, collision, timeout,
  hard-failure, and other-terminal counts. A valid acceptance cell requires
  completed equals expected, unresolved equals zero, and no hard or other
  terminal event.
- Fixed-denominator clean, collision, and timeout rates are reported directly.
- Dataset collection is rejected in this mode, because this is an evaluation
  window rather than a training-data sampler.

All existing fixed-step reports keep the legacy
`fixed-simulator-steps-legacy` identity and their original decisions.

## Predeclared wiring smoke

The first runtime use is only a bounded wiring check of the accepted HC4-LH
controller. It does not assess or expand capability.

| Field | Frozen value |
|---|---|
| Locomotion actor | retained motor-aware `model_7998.pt` |
| Supervisor | accepted 0.02 m-gated HC4-LH checkpoint |
| Speed | 0.30 m/s |
| Obstacle | 1.15 m forward, 0.00 m lateral, unchanged box geometry |
| Seed | 197 |
| Environments | 4 |
| Step ceiling | 700 |
| Perception | exact compact structured geometry; no raw camera input |

The smoke passes only if expected attempts = completed attempts = 4,
unresolved attempts = 0, hard-failure events = 0, other-terminal events = 0,
all reported values are finite, and the report names the frozen protocol.
Collision, timeout, passage time, and motor values are retained but do not
change the already accepted HC4-LH envelope from this four-attempt smoke.

## Retained smoke result

The accepted-controller wiring smoke passed on 100.100. It ran on CPU because
the retained user service did not set a non-empty `CUDA_VISIBLE_DEVICES`; the
GPU remained idle. This is evaluator and checkpoint-wiring evidence, not CUDA
execution evidence and not a capability evaluation.

- Protocol: `first-terminal-attempt-per-environment-v1`.
- Attempts: four expected, four completed, four clean, zero unresolved.
- Failures: zero collision, timeout, fall, NaN, non-finite, hard-failure, and
  other-terminal events.
- Execution: 502 of the 700 allowed steps; mean passage time 9.4750 s.
- Motor diagnostics: speed-utilization p99 0.3567, torque-utilization p99
  0.5256, near-stall fraction 0.0038%, thermal-load proxy mean 0.0341.
- Motor-aware actor SHA-256:
  `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- HC4-LH SHA-256:
  `0b2608080671c5df85d8c9f900d68b6a6f298ec820eb1c6ba75afc948337505a`.
- Report SHA-256:
  `5052c60ac6300b20d1d5d6a306eea018f6e8b0ce327e7dfa3ddb78aead5eeed6`.

The focused obstacle suite passed 89 tests on 100.100 with CUDA hidden. The
report remains under
`artifacts/evaluations/hc4e1-6708f9a-s197-smoke/` on that host. The next
candidate's predeclared matrix must explicitly set its CUDA device; this smoke
does not substitute for that gate.

## Next candidate boundary

Only after the source tests and wiring smoke pass may a new, separately
predeclared unified supervisor candidate be prepared from accepted far/lateral
and near-range correction evidence. It must be a newly identified candidate,
not a rerun of HC4-R2L. Its paired controller matrix must use the same number
of environments per cell under this fixed-attempt protocol. No long GPU job,
MP4, raw-perception training, or physical motion is authorized by HC4-E1.
