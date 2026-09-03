# HC3-D balanced supervisor PPO

Date: 2026-09-03

Source commit: `bc958874e127ff80f82eb9a9c2d015b79e4bdb14`

Decision: **all snapshots rejected; retain HC2**

## Purpose and invariants

HC3-D tested whether the HC3 supervisor could improve passage time without
specializing to one obstacle cell. It made one design change from HC3-A/B/C:
each training batch deterministically balanced all four accepted HC2 cells.
Every iteration was retained so evaluation did not have to assume that the
final checkpoint was best.

- The Stage-2 61D motor-aware locomotion actor remained frozen and in inference
  mode.
- The 17D supervisor was initialized from HC2 and only its policy, value
  function, and exploration standard deviations were optimized.
- The execution layer continued to clamp forward speed to 0--0.8 m/s, yaw to
  +/-0.6 rad/s, and command slew to the measured HC0 envelope.
- Nominal-speed tracking remained active during approach and recovery. It was
  absent during interaction, where slowing down for collision avoidance is
  explicitly permitted.
- Inputs were exact structured simulation geometry. No raw-camera perception
  or physical motion was used.

## Training slice

HC3-D used 128 parallel environments for four PPO iterations, with 64
high-level rollout steps per iteration and five frozen low-level policy steps
per supervisor action. The four balanced cells were 0.30 x 1.15, 0.50 x 1.15,
0.50 x 1.40, and 0.80 x 1.40 (nominal speed m/s x obstacle-forward position
m). PPO used learning rate `2e-5`, HC2 anchor scale `0.5`, zero entropy bonus,
initial log standard deviations `(-2.3, -2.3)`, and collision penalty `40`.
Training seed was 97.

Training rollouts recorded 290 clean passes, one collision, and one timeout
across iterations 2--4; iteration 1 had not yet completed an attempt. They had
zero falls and non-finite events. These exploratory rollouts were diagnostic,
not acceptance evidence.

## Snapshot pre-screen

Each retained iteration first ran deterministic seed 41 at the sensitive
0.50 x 1.15 and 0.80 x 1.40 cells. Iterations 2 and 4 were rejected immediately
because they had three and two combined collisions, respectively. Iterations 1
and 3 consumed at most the one-collision HC2 budget and advanced to the full
matrix.

| Snapshot | 0.50 clean/collision/timeout | 0.80 clean/collision/timeout | Decision |
|---|---:|---:|---|
| iteration 1 | 64 / 0 / 0 | 63 / 1 / 0 | full matrix |
| iteration 2 | 64 / 1 / 0 | 63 / 2 / 0 | reject |
| iteration 3 | 64 / 0 / 0 | 63 / 1 / 0 | full matrix |
| iteration 4 | 65 / 1 / 0 | 63 / 1 / 0 | reject |

## Full retained comparison

The full matrix used the four accepted HC2 cells, seeds 41--43, 64 environments,
and 700 low-level steps per case. Promotion required no more than HC2's one
collision, a clean-pass rate no worse than HC2, no fall/non-finite/rated-speed
regression, and a measurable passage-time improvement.

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 | Max near-stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC2 accepted baseline | 759 | 1 | 8 | 768 | 98.83% | 9.251 s | 0.745 | 0.216% |
| HC3-D iteration 1 | 760 | 1 | 7 | 768 | 98.96% | 9.268 s | 0.745 | 0.215% |
| HC3-D iteration 3 | 763 | 4 | 6 | 773 | 98.71% | 9.157 s | 0.745 | 0.219% |

Both snapshots had zero falls, NaN terminations, non-finite steps, and rated
motor-speed exceedances. Iteration 1 was safety-neutral and removed one timeout,
but its weighted passage was 0.017 seconds slower than HC2. Iteration 3 improved
weighted passage by 0.094 seconds, but spent four collisions and reduced the
clean-pass rate. Neither is promoted.

The result also validates retaining per-iteration checkpoints: the final
iteration was rejected by the inexpensive pre-screen, while earlier snapshots
contained the two distinct tradeoffs worth measuring.

## Retained artifacts

Artifacts remain outside Git and are duplicated on the Mac and host `100.100`.

Checkpoint directory:

- Mac: `../artifacts/hc3d-bc95887-s97/`
- remote: `artifacts/checkpoints/hc3d-bc95887-s97/`

| Snapshot | Supervisor SHA-256 |
|---|---|
| iteration 1 | `96de9ca114166a39329a314cce5a5a7ad4708941a2ea1a103164b94bb41b95c9` |
| iteration 2 | `c5e0a728dd5e4040298d6ad1269d35bee559bf77935a2363133c5af3ae68c40c` |
| iteration 3 | `61208d6f575f51eea0606e3a3995b0c6f4b815d709db9e63819d5e7dced5dc16` |
| iteration 4 | `101b175de4d46550d3798b9da4c341d802f06a8b6650097101902f19c9f2b23c` |

Evaluation directories:

- Mac: `../artifacts/hc3d-prescreen-bc95887/` and
  `../artifacts/hc3d-full-bc95887/`
- remote: `artifacts/evaluations/hc3d-bc95887-prescreen/` and
  `artifacts/evaluations/hc3d-bc95887-full/`

Full-matrix JSON SHA-256 values for 0.30 / 0.50 / 0.80 are:

- iteration 1: `c53a979c19cd04e2f5cbc71a3de4614d4ab37635e1097c83e6404ff38965674d`
  / `26ad80722f74de8a229a3ea4dd3538e8d2a5e7b7088f3923d87e48f9020f3905`
  / `a8d4302ff5902d85d8238c0b8900db583d9b550e6be45738c4727194ffaa0293`
- iteration 3: `5460c6fd54456b9463cf43505b19c608279951ac86cdc5c094f89570ebe7c29f`
  / `08d0ea07a0e68478b8a7ebfd30805263c67d2980d6700408975974a7afabe605`
  / `7f49b3f3e1a11de6b0066aa6b4530d13077e556967e735f97a25df86aeefa8c6`

## Subsequent gate

HC3-E implemented the one-dimensional interaction-speed authority and corrected
its execution boundary so HC2 remains authoritative outside interaction. The
candidate repeated a small passage-time gain, but one of three training seeds
failed the sensitive-cell collision budget. See
`2026-09-03-hc3e-interaction-speed-only.md` for retained evidence and the next
seed-aggregation gate.
