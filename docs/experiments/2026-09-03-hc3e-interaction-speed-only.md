# HC3-E interaction-speed-only PPO

Date: 2026-09-03

Source commits: `af429363a080fb0d17d9782d2106c33e782866bd`,
`02e05b0ab4bfb93facab84de969a5567ecffb8c8`

Decision: **promising but not training-seed robust; retain HC2**

## Authority boundary

HC3-E reduced the learned supervisor from two outputs to one effective degree
of freedom. The Stage-2 locomotion actor remained frozen. HC2 remained
byte-exact for yaw in every phase and for forward speed during approach and
recovery. PPO could update only the final speed-output row, and that output was
used only during interaction. The execution wrapper bounded interaction speed
to 0.30 m/s through the nominal command and retained the existing slew limits.

This implements the intended speed principle: continue tracking the route
speed before and after avoidance, while allowing slower motion around the
obstacle. Exact structured simulation geometry remained the only obstacle
input. Raw-camera perception and physical motion were not used.

## Boundary correction

The first implementation at `af42936` passed nominal speed directly during
approach and recovery. That replaced HC2 behavior outside interaction and made
the first pre-screen materially faster but unsafe: every retained snapshot had
at least two combined collisions in the 0.50 x 1.15 and 0.80 x 1.40 seed-41
cells.

Commit `02e05b0` corrected the boundary by storing a byte-exact HC2 anchor in
each checkpoint. The wrapper now takes both yaw and non-interaction speed from
that anchor. The rejected first implementation and all its pre-screen evidence
remain retained; no files were overwritten.

## Corrected training

The main seed-109 run used the four balanced HC2 cells, 128 environments, 64
high-level rollout steps, five low-level steps per action, learning rate
`2e-5`, HC2 anchor scale `0.5`, zero entropy bonus, collision penalty `40`, and
initial speed log standard deviation `-2.3`. Four iterations were retained.

Exploratory training rollouts had zero falls and non-finite events. Iterations
1 and 2 survived the seed-41 sensitive-cell pre-screen; iterations 3 and 4
exceeded the one-collision budget. Only iterations 1 and 2 ran the full matrix.

## Seed-109 full matrix

The unchanged matrix used four accepted HC2 cells, seeds 41--43, 64
environments, and 700 low-level steps per case.

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 | Max near-stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC2 accepted baseline | 759 | 1 | 8 | 768 | 98.83% | 9.251 s | 0.745 | 0.216% |
| HC3-E seed 109, iteration 1 | 760 | 1 | 8 | 769 | 98.83% | 9.239 s | 0.749 | 0.229% |
| HC3-E seed 109, iteration 2 | 762 | 2 | 7 | 771 | 98.83% | 9.214 s | 0.745 | 0.218% |

Iteration 2 was rejected for two collisions. Iteration 1 matched the collision
budget and improved weighted passage by only 0.011 seconds, which is smaller
than one 0.1-second supervisor update. It therefore required fresh-seed paired
confirmation rather than immediate promotion.

## Paired confirmation

HC2 and HC3-E seed-109 iteration 1 were compared on fresh evaluation seeds
44--46 under the same four-cell protocol.

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 | Max near-stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC2, seeds 44--46 | 765 | 0 | 3 | 768 | 99.61% | 9.222 s | 0.750 | 0.242% |
| HC3-E, seeds 44--46 | 766 | 0 | 2 | 768 | 99.74% | 9.180 s | 0.754 | 0.258% |
| HC2, seeds 41--46 | 1524 | 1 | 11 | 1536 | 99.22% | 9.236 s | 0.750 | 0.242% |
| HC3-E, seeds 41--46 | 1526 | 1 | 10 | 1537 | 99.28% | 9.210 s | 0.754 | 0.258% |

The confirmation repeated the direction: one fewer timeout, the same collision
count, and a 0.027-second combined weighted-passage improvement. Motor load
rose slightly but remained inside the retained envelope. All cases had zero
falls, NaN terminations, non-finite steps, and rated motor-speed exceedances.

## Training-seed robustness gate

The exact one-iteration recipe was repeated with training seeds 113 and 127.
Each checkpoint received the same seed-41 sensitive-cell pre-screen.

| Training seed | Clean | Collision | Timeout | Resolved | Pre-screen decision |
|---:|---:|---:|---:|---:|---|
| 109 | 126 | 1 | 2 | 129 | pass |
| 113 | 125 | 3 | 1 | 129 | reject |
| 127 | 126 | 1 | 1 | 128 | pass |

Seed 113 exceeded the HC2 collision budget before the full matrix. Therefore
the training recipe is not robust across the required three independent
training seeds. The numerically promising seed-109 checkpoint is not promoted,
HC2 remains accepted, and no HC3-E MP4 is recorded.

## Retained artifacts

All material checkpoints and evaluations are outside Git and duplicated
byte-for-byte on the Mac and host `100.100`.

Mac directories:

- `../artifacts/hc3e-af42936-s103/`
- `../artifacts/hc3e-af42936-prescreen/`
- `../artifacts/hc3e2-02e05b0-s109/`
- `../artifacts/hc3e2-02e05b0-s113/`
- `../artifacts/hc3e2-02e05b0-s127/`
- `../artifacts/hc3e2-02e05b0-prescreen/`
- `../artifacts/hc3e2-02e05b0-training-seed-prescreen/`
- `../artifacts/hc3e2-02e05b0-full/`
- `../artifacts/hc3e2-02e05b0-confirm-s4446/`

Remote paths use the same leaf names below the worktree's
`artifacts/checkpoints/` or `artifacts/evaluations/` directory.

Iteration-1 supervisor SHA-256 values are:

- rejected boundary, seed 103:
  `26e14642da11f52f483e788c1330880f72b47b3c30af07886fb8f8165ae57414`
- corrected seed 109:
  `c6930cffb2d8c1e527935aeaf5c589d95fc665b808c01acde054a6e440263fdd`
- corrected seed 113:
  `4fb3711c0b5f17bba320da095a98ca99eb12ab3df84133ba3754b2b0ea7f8344`
- corrected seed 127:
  `cae05d2abac3486edd9f62dcd4b8016edd002d2a872d3907686c994090a7c369`

## Next bounded gate

Do not select a favorable single training seed or resume two-output PPO. The
next candidate should aggregate only the 65 speed-head parameters from the
three corrected iteration-1 checkpoints while preserving every HC2 anchor
parameter exactly. A deterministic seed-averaged head is a bounded way to test
whether the common update direction keeps the repeated passage benefit while
removing seed-specific collision variance.

The averaged checkpoint must pass the sensitive-cell pre-screen and then the
unchanged full matrix. Only after numeric acceptance should placement expand or
new MP4s be recorded.
