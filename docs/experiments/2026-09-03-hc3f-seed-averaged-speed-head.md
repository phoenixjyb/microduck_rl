# HC3-F seed-averaged interaction-speed head

Date: 2026-09-03

Source commits: `288000f10d33d2de58d9775127fa283110f84f2b`,
`e3793ee39005f6b9bee392c6d4314b72c62110b8`

Decision: **rejected at sensitive-cell pre-screen; retain HC2**

## Candidate contract

HC3-F deterministically averaged the 65 trainable speed-head parameters from
the corrected HC3-E iteration-1 checkpoints for training seeds 109, 113, and
127. Every hidden-layer parameter, the yaw-output row, and the HC2 anchor were
required to be byte-exact across inputs. The execution wrapper continued to
use HC2 for yaw in every phase and for speed outside interaction. Interaction
speed remained bounded to 0.30 m/s through the nominal command.

The aggregation tool fails closed on fewer than three inputs, duplicate seeds,
non-HC3-E stages, non-pending decisions, mismatched source identities, modified
frozen parameters, or mismatched PPO update and reward settings. A source run's
planned iteration horizon is retained as provenance but is not an update
invariant: all selected weights were completed iteration 1 with identical
optimization settings.

The locomotion actor remained frozen. Obstacle input remained exact structured
simulation geometry; no raw-camera perception or physical motion was used.

## Exact artifact

The aggregated checkpoint is:

`../artifacts/hc3f-e3793ee-mean-s109-113-127/supervisor.pt`

SHA-256:
`ef02e73ee7fe0e4785af3ed34c7eb4ddc345c1bf8ec262f6afd4e3836ba023e9`

Its JSON manifest SHA-256 is
`31508d448a787fba7f97e07af3da738b43720d38af84f3f71bd3f42f189dc12f`.
The three source hashes are:

- seed 109: `c6930cffb2d8c1e527935aeaf5c589d95fc665b808c01acde054a6e440263fdd`;
- seed 113: `4fb3711c0b5f17bba320da095a98ca99eb12ab3df84133ba3754b2b0ea7f8344`;
- seed 127: `cae05d2abac3486edd9f62dcd4b8016edd002d2a872d3907686c994090a7c369`.

The first build attempt intentionally failed before writing an artifact because
the seed-109 directory alias points to its final fourth iteration. The retained
build therefore names `supervisor-iter-0001.pt` explicitly for all three seeds.

## Sensitive-cell result

The unchanged pre-screen used evaluation seed 41, 64 environments, and 700
low-level steps in each of the two sensitive cells.

| Speed x forward position | Clean | Collision | Timeout | Resolved | Passage | Torque p99 | Near-stall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 m/s x 1.15 m | 66 | 0 | 0 | 66 | 8.612 s | 0.576 | 0.0139% |
| 0.80 m/s x 1.40 m | 63 | 2 | 0 | 65 | 8.546 s | 0.746 | 0.2170% |
| Total | 129 | 2 | 0 | 131 | — | 0.746 | 0.2170% |

Both cells had zero falls, NaN terminations, non-finite steps, and rated
motor-speed exceedances. The 0.80 m/s cell nevertheless exceeded the complete
one-collision pre-screen budget by itself. HC3-F therefore did not run the full
matrix or receive MP4 recording. HC2 remains the accepted simulation
controller.

The retained pre-screen reports are in
`../artifacts/hc3f-e3793ee-prescreen/` and are byte-identical to the copies on
100.100.

## Next bounded gate

HC3-G will keep only coordinates of the seed-averaged speed-head delta for
which all three training seeds agree on update direction. In the retained
inputs, 26 of 65 coordinates meet that rule; the other 39 remain exactly at
the HC2 anchor. This deterministic sign-consensus projection uses all three
seeds and reduces seed-specific authority without selecting a favorable seed.

HC3-G must pass the same two-cell pre-screen before any full matrix, placement
expansion, or video. A failure ends this speed-head optimization line rather
than triggering an unbounded parameter search.
