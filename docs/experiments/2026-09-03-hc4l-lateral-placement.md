# HC4-L lateral-placement specialist and HC4-LH hybrid

Date: 2026-09-03

Source commits:

- HC4-L stage identity: `d8cc28ec3426977079f96b3cfb41b90e6f255375`
- HC4-LH composition and runtime gate: `92b64d9e6bbcb733be01734598429d85d7a34138`
- stage-aware replay metadata: `149828e7c276c6a21f08b3b8ffa4e19feccd5c33`

Decision: **accept HC4-LH for its exact simulation envelope; do not replace HC2
with the ungated HC4-L specialist**

## Scope and authority

HC4-L changes obstacle lateral placement only. It retains the frozen Stage 2
motor-aware gait actor, the 17D compact supervisor observation, the bounded
forward-speed/yaw command interface, exact structured geometry, and the rule
that nominal speed tracking applies before and after avoidance but not during
interaction. It does not consume camera pixels and does not authorize physical
motion.

The training/evaluation envelope is the four accepted speed/range cells at
obstacle lateral positions `-0.12`, `0.00`, and `+0.12` m. Training seeds were
51--53 and held-out evaluation seeds were 61--63.

## Diagnose first: unchanged HC2

The initial seed-41 diagnostic ran unchanged HC2 at all three lateral
positions. Across 36 cells it retained 753 clean passes, nine collisions, and
12 timeouts. Seven of the nine collisions occurred at 0.80 m/s. This showed
that the centered controller could not simply be declared valid at shifted
placements.

The later exact paired HC2 run on evaluation seeds 61--63 provides the
comparison used for selection:

| Placement | Clean | Collision | Timeout | Resolved | Weighted passage |
|---|---:|---:|---:|---:|---:|
| -0.12 m | 756 | 34 | 4 | 794 | 8.865 s |
| 0.00 m | 762 | 2 | 6 | 770 | 9.276 s |
| +0.12 m | 765 | 14 | 1 | 780 | 8.822 s |
| **All** | **2,283** | **50** | **11** | **2,344** | **8.988 s** |

The paired reports are retained under
`../artifacts/hc4l-d8cc28e-hc2-paired-s61-63/`.

## Teacher data and offline fit

The deterministic teacher collected 131,438 successful-step samples from
1,688 clean episodes, with zero collisions and four timeouts among 1,692
resolved attempts. The three dataset hashes are:

- 0.30 m/s: `76da0fd8fb9efe99e332ba9d7e787f7d3c607417ee48b906bb3091a6e941f15f`
- 0.50 m/s: `9bc85efa2917c16fb007fa51647e87c1115e87dd469dd7eb267c384af3a1fad9`
- 0.80 m/s: `3d9d24355457e033e42448dfb1b71438443ee65186b4f6d644d20127b6264026`

One predeclared seed-42 BC candidate was trained for 200 epochs on the three
original centered HC1 shards plus the three new lateral shards. It selected
epoch 196 with validation speed MAE 0.003657 m/s, yaw MAE 0.017968 rad/s, and
MSE 0.000886.

The retained HC4-L checkpoint is
`../artifacts/hc4l-bc-d8cc28e-s42/supervisor.pt`, SHA-256
`8b0be10a7a89212035202a298ca8ba1081fa93971710ab0143d9fb287561af3f`.
Its manifest SHA-256 is
`a5d66080797942a187fa3ae2902c39bc8dff4ca6f03385373564f62ded5a68ad`.

## Why HC4-L is not a wholesale HC2 replacement

Across its held-out seed-61 pre-screen and seed-62/63 continuation, HC4-L
recorded 2,423 clean passes, four collisions, and six timeouts among 2,433
resolved attempts (99.589% clean). It removed every collision at `-0.12` and
`+0.12` m, but the centered bin recorded four collisions versus HC2's two.
The specialist therefore improved the new task while regressing the accepted
center safety count. It is retained but not promoted by itself.

## HC4-LH lateral-gated composition

HC4-LH packages byte-exact HC2 and HC4-L networks into one fail-closed
supervisor. A reconstructed obstacle route-lateral offset below 0.06 m, or an
invalid obstacle estimate, selects HC2; an offset at or above 0.06 m selects
HC4-L. The wrapper still produces only bounded speed and yaw commands.

The retained checkpoint is
`../artifacts/hc4lh-92b64d9-center006/supervisor.pt`, SHA-256
`89627748ef66428399a78f34a6ce57fba86037f044ab874f2cc980fae5528c69`.
Its immutable pre-rollout manifest SHA-256 is
`90d3337277c244b0351ff6cfe971792097e4ed25744d26755014e3a20e160e3b`.

The actual 36-cell held-out matrix—not a synthetic merge of prior reports—was:

| Placement | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage |
|---|---:|---:|---:|---:|---:|---:|
| -0.12 m | 840 | 1 | 1 | 842 | 99.762% | 7.761 s |
| 0.00 m | 763 | 2 | 7 | 772 | 98.834% | 9.276 s |
| +0.12 m | 805 | 0 | 1 | 806 | 99.876% | 8.044 s |
| **All** | **2,408** | **3** | **9** | **2,420** | **99.504%** | **8.336 s** |

Relative to paired HC2, shifted-position collisions fell from 48 to one while
the centered collision count stayed at two. Weighted passage improved by
0.652 s. The maximum per-cell torque-utilization p99 was 0.7543 and the maximum
near-stall fraction was 0.2439%. There were zero falls, NaN terminations,
non-finite steps, and rated motor-speed exceedances. A one-timeout difference
in the centered bin is retained rather than hidden; clean count increased by
one and centered weighted passage differed by less than one millisecond.

The three report hashes are:

- 0.30 m/s: `eed9c978d8024e6a420385428b8727298fd2a1d01350afb22a16c1bd7950f16e`
- 0.50 m/s: `2849e8ed8979f5ff7c9fa8c09793dacf7a5c7e94d4d82a3b6832ef3331a00926`
- 0.80 m/s: `d09a7cd3de26fea5b6acc8a07751133908c3fcab91d0066d2ce3910a1ba72b26`

They are retained under
`../artifacts/hc4lh-92b64d9-full-s61-63/` and are byte-identical to the copies
on 100.100.

## Visual evidence and next gate

Representative 12-second, 960 x 540 replays cover 0.30, 0.50, and 0.80 m/s,
both lateral signs, and forward positions 1.15 and 1.40 m. Each replay has a
sidecar manifest naming HC4-LH, exact checkpoint hashes, geometry, seed,
passage/collision state, and the simulation-only authority boundary. They are
retained under `../artifacts/hc4lh-149828e-heldout/`.

| Replay | MP4 SHA-256 | Sidecar SHA-256 |
|---|---|---|
| 0.30 m/s, x=1.15 m, y=-0.12 m | `2607f5ddc634fc04db85e50df492373f085d14d6cc60cc68391342c4cd74197b` | `d1d2850263a755f51b1aa0fef2517a72369cc79e4a419f65a422059a185f3fdf` |
| 0.50 m/s, x=1.15 m, y=+0.12 m | `2905c399760dfa7e17d5a41b41632c2c8891a09c66ae77e76167594deff08e62` | `9251b36b47dab15b425dc23394c946d6217d93c6ef91cbf947bf564ead581580` |
| 0.50 m/s, x=1.40 m, y=-0.12 m | `494faae1521e9d169801d25af55cf9bfd3a84304ab9b9dd42300c3c1fcb1a8ce` | `66daf3a6320f82ae9f2ec40d8f4bedb04040b2a5719a551bef2302702c91c170` |
| 0.80 m/s, x=1.40 m, y=+0.12 m | `86efb328d3754cc613b022f06db75072f0fd40044821798ecd8827ca6090559b` | `22edf6d0fd5737fb70c54b64815c4635366b2f46c4bc89e817f5c1c4dde54113` |

The next stage is diagnostic, not another long training run: sweep unseen
lateral offsets around the 0.06 m gate and just outside the trained +/-0.12 m
placements. This measures interpolation, extrapolation, and gate-boundary
behavior before expanding the accepted envelope. Sensor noise, latency,
dropout, raw perception, low-obstacle jumping, and physical motion remain
separate later gates.
