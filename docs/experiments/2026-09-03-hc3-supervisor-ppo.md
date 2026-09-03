# HC3 bounded supervisor PPO — A/B/C

Date: 2026-09-03  
Source commits: `e38840007`, `40c468bc3`, `53a680045`  
Decision: **all candidates rejected; retain HC2**

## Invariants

- The Stage-2 61D motor-aware locomotion actor stayed frozen and in inference
  mode. It was not part of any HC3 optimizer.
- HC3 optimized only the 17D-to-2D supervisor, its value function, and two
  exploration standard deviations.
- The execution layer continued to enforce 0–0.8 m/s forward speed, +/-0.6
  rad/s yaw, and the existing command slew limits.
- Approach and recovery retain nominal-speed tracking. Interaction has no
  nominal-speed tracking term, so slowing down to avoid collision is valid.
- Obstacle input was exact structured simulation geometry. No raw-camera
  perception or physical motion was used.

## Training slices

Each candidate started independently from the retained HC2 checkpoint, used
128 parallel environments and 64 high-level steps per iteration at the
centered 0.50 m/s x 1.15 m cell, and held each high-level action for five
low-level policy steps.

| Candidate | Iterations | LR | HC2 anchor | Entropy | Initial log std | Collision penalty |
|---|---:|---:|---:|---:|---|---:|
| HC3-A | 12 | 1e-4 | 0.05 | 0.002 | -1.5, -1.5 | 12 |
| HC3-B | 6 | 3e-5 | 0.5 | 0 | -2.3, -2.3 | 40 |
| HC3-C | 3 | 1e-5 | 1.0 | 0 | -2.7, -2.7 | 80 |

Training rollouts are exploratory and were not used as acceptance evidence.
All had zero falls and non-finite events. Deterministic evaluation followed
after each checkpoint completed and the GPU became idle.

## Three-seed deterministic comparison

The retained matrix is the four accepted HC2 cells: 0.30 x 1.15, 0.50 x 1.15,
0.50 x 1.40, and 0.80 x 1.40 (speed m/s x obstacle forward position m), seeds
41–43, 64 environments, and 700 low-level steps per case. Because faster
policies complete more episodes inside the fixed step budget, compare rates and
passage time as well as raw event counts.

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 | Max near-stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC2 accepted baseline | 759 | 1 | 8 | 768 | 98.83% | 9.251 s | 0.745 | 0.216% |
| HC3-A | 854 | 24 | 1 | 879 | 97.16% | 7.672 s | 0.795 | 0.305% |
| HC3-B | 768 | 2 | 2 | 772 | 99.48% | 8.743 s | 0.751 | 0.229% |
| HC3-C | 761 | 5 | 6 | 772 | 98.58% | 9.114 s | 0.742 | 0.212% |

All candidates had zero falls, NaN terminations, non-finite steps, and rated
motor-speed exceedances. HC3-A was substantially faster but unsafe. HC3-B was
the best tradeoff, eliminating six of HC2's eight timeouts and improving
weighted passage by 0.508 s, but it recorded two collisions versus HC2's one.
HC3-C showed that simply shrinking the update was not monotonic: it regressed
to five collisions and six timeouts. None is promoted.

## Retained artifacts

Artifacts remain outside Git and are duplicated on the Mac and host `100.100`.

| Candidate | Supervisor SHA-256 | Evaluation JSON SHA-256: 0.30 / 0.50 / 0.80 |
|---|---|---|
| HC3-A | `f2cc78e716e74a87e19c672527329c03dc005c924642e8ef02922da3a628fc6e` | `c8d52ee90bbaeec9e6acee5c662af0e9276841a7545e09af3a61cc0368318efc` / `1695d0744abc8e20c2f0684d23c727be107ec1b1ca865dfe3bad982c3f9d8d45` / `4b605d84ad9a2a524c64b13108c08232469a6f999a1494f2dce2ffb92f3abd76` |
| HC3-B | `e8f042a16fea69b7c7782e0704e5c9c48ae5aef2f3095dee76588e383cf3196b` | `811edf2fe62d6220cb78db31fd2572ae3f7fcf2973a9d0d3a6d1dffadc6ebf0f` / `2d5bab11221cb879e98c6bf731695630ef8ae264c437e9d79d3bc5ec940f1a97` / `02553de8aef1d22b89793205101b35a8eb5607a244bd746a26a3768bce0caca1` |
| HC3-C | `35c4f8089a99087af7f40bf2a2ca3bc77f69e8b31230e0b798f80358ef716321` | `9b1c3f53a03dff7d4498ca19d197373b5292e86ee0d5ab56c428e5d486db7874` / `c35789b66cef0aa4bb0f57e2e0d36ab37ee2fd147202d4172f13c2d9b8b5efd9` / `98c712d5117b8b985d32b71928e4db66c10f5aa3489f7fd4653b41672c094236` |

Mac paths:

- `../artifacts/hc3a-40c468b-s73/` and
  `../artifacts/hc3a-evaluations-40c468b/`
- `../artifacts/hc3b-53a6800-s79/` and
  `../artifacts/hc3b-evaluations-53a6800/`
- `../artifacts/hc3c-53a6800-s83/` and
  `../artifacts/hc3c-evaluations-53a6800/`

Remote paths use the corresponding directories below
`/home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/`.

## Next gate

Do not continue the single-cell/final-checkpoint search. HC3-D should make one
reviewable design change: train on a balanced mixture of all four accepted HC2
cells and retain per-iteration candidates. Each candidate must pass the full
three-seed matrix; promotion requires no more than one collision, no worse
clean-pass rate than HC2, no fall/non-finite/rated-speed regression, and a
measurable passage-time improvement. Only then should placement expand or new
videos be recorded.
