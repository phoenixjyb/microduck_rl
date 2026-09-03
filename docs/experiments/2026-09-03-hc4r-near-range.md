# HC4-R near-range obstacle specialist

Date: 2026-09-03

Source commit: `45e22f23fb131c9a35b681bc3f1b972061e8a3ac`

Decision: **reject the first HC4-R candidate for routing; retain it as
diagnostic evidence and keep HC4-LH selected**

## Scope and authority

HC4-R changes one curriculum axis: the obstacle starts 0.90 m ahead instead
of the accepted 1.15/1.40 m cells. It keeps the Stage 2 motor-aware gait actor
frozen, uses the same 17D compact supervisor observation, and can command only
bounded forward speed and yaw. It uses exact structured geometry, consumes no
camera pixels, and authorizes no physical motion.

Nominal speed tracking remains an approach and recovery goal. During obstacle
interaction the supervisor may slow down to preserve clearance; it is not
rewarded for holding nominal speed through the maneuver.

## Range diagnosis

The accepted 0.02 m-gated HC4-LH controller was first measured at the new
0.90 m boundary. At 0.50 m/s it retained 663 clean passages, one collision,
and one timeout among 665 resolved attempts. At 0.80 m/s it retained 694 clean
passages, 61 collisions, and five timeouts among 760 resolved attempts. Of the
61 collisions, 58 occurred at 0.90 m; 41 were in the centered bin.

The deterministic teacher showed the same physical timing limit at
0.80 x 0.90 m: 314 clean passages, 35 collisions, and one timeout among 350
resolved attempts. At interaction entry, the previous command is still the
0.80 m/s nominal command. The bounded 0.08 m/s command delta requires several
supervisor updates to approach the 0.30 m/s interaction floor, leaving too
little distance for a reliable maneuver.

A teacher speed sweep at 0.90 m placed the workable boundary at the slow end:

| Nominal speed | Clean | Collision | Timeout | Resolved | Clean rate |
|---|---:|---:|---:|---:|---:|
| 0.30 m/s | 208 | 0 | 0 | 208 | 100.000% |
| 0.40 m/s | 261 | 0 | 0 | 261 | 100.000% |
| 0.50 m/s | 297 | 1 | 0 | 298 | 99.664% |
| 0.60 m/s | 306 | 3 | 0 | 309 | 99.029% |
| 0.70 m/s | 311 | 10 | 1 | 322 | 96.584% |
| 0.80 m/s | 308 | 35 | 1 | 344 | 89.535% |

The accepted HC4-LH controller was then checked at 0.30--0.70 m/s. It was
collision-free only for the single seed at 0.30/0.40 m/s. Additional seeds
still produced four collisions at 0.30 m/s, three at 0.40 m/s, and six at
0.50 m/s. Therefore no speed-only cap is accepted at 0.90 m under HC4-LH.

The retained diagnostic report hashes are:

- HC4-LH range pre-screen, seed 83: `1457369bc2e714427f5adb44d9f7f425dccf85353c3137e221317aa65e8276f8`
  at 0.50 m/s and `2c536ea6fa0130c574885029ed07930e945d3c74a0ce9d593dad4330744441ed`
  at 0.80 m/s;
- 0.80 x 0.90 m teacher diagnostic: `c525db3796145c60870d2fe82a4a1765304d5586affc5100e0ff09e6ecf28d57`;
- teacher speed envelope: `f1c7f29d5554ba1940b6da5cd82a90e523658aa8980cd9b1e7eabed3e9a9e795`;
- HC4-LH speed envelope, seed 83: `f78902fb04ec0e5718a1202d8a11d1334c6bdd6aa88fb595ba6370f351d4f80a`;
- HC4-LH 0.40/0.50 m/s continuation, seeds 89/97:
  `3a0c69b803ae24843c2ffb9046d9b8605423b70941d024ffc4325bab5ed41a04`.

## Teacher corpus and offline fit

HC4-R teacher data covers 0.30/0.40 m/s, obstacle range 0.90 m, lateral
positions -0.08/0.00/+0.08 m, and seeds 101/103/107. It contains 107,665
finite samples from 1,466 successful episodes. The rollout recorded one
collision, zero timeouts, zero falls, zero NaN terminations, and zero
non-finite steps among 1,467 resolved attempts (99.932% clean). The 0.30 m/s
slice was collision-free; the sole collision was at 0.40 m/s and +0.08 m.

The dataset SHA-256 is
`69c8238505a4f60f8de9f17816993947597fbbf5920253898117e6667a961f06`.
The teacher report SHA-256 is
`1908cda57f007ed5c39c3cb9365dc1b531f7297fb2eab4008405752c75eedcf1`.

One predeclared seed-42 BC candidate was trained for 200 epochs using CPU
only. Epoch 198 minimized validation MSE. Validation speed MAE was 0.002528
m/s and yaw MAE was 0.022415 rad/s, inside the 0.025/0.050 offline gates.

The retained checkpoint is
`../artifacts/hc4r-bc-45e22f2-s42/supervisor.pt`, SHA-256
`641fb7ae5cd0a1a780b2ce8ca759e3d2ce668651b7f23834bdc79685bb88bf3f`.
Its manifest SHA-256 is
`c154597ee9848c96f8da4883ff2b64fd57c2a6eee47f73a3bbc60d60e3ac2ef4`.

## First held-out closed-loop result

At seed 109 over the six 0.30/0.40 x 0.90 x -0.08/0.00/+0.08 m cells, the
standalone HC4-R candidate retained 477 clean passages, four collisions, and
zero timeouts among 481 resolved attempts (99.168% clean). All four collisions
were in the centered bin; both shifted bins were collision-free. There were
zero falls, NaN terminations, non-finite steps, and rated motor-speed
exceedances. Maximum per-cell torque-utilization p99 was 0.5690 and maximum
near-stall fraction was 0.00399%.

The report is retained under `../artifacts/hc4r-45e22f2-prescreen-s109/`,
SHA-256
`0bd8d9c18b5280870eb0ff820e9f629a3545b0aecdd1e301a98b15a97627e964`.

The byte-exact HC4-LH paired seed-109 run recorded 455 clean passages and five
collisions among 460 resolved attempts. HC4-R's aggregate was better on that
single seed, but its two new 0.30 m/s centered collisions made the result mixed
rather than sufficient for promotion. Seeds 113 and 127 were therefore run for
both controllers under the same protocol.

## Three-seed paired decision

Across seeds 109/113/127, the paired matrix was:

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage |
|---|---:|---:|---:|---:|---:|---:|
| HC4-LH selected baseline | 1,364 | 7 | 0 | 1,371 | 99.489% | 7.530 s |
| HC4-R candidate | 1,433 | 10 | 0 | 1,443 | 99.307% | 7.353 s |

The exact cell distribution was:

| Speed | Lateral | HC4-LH collision | HC4-R collision |
|---|---:|---:|---:|
| 0.30 m/s | -0.08 m | 1 | 0 |
| 0.30 m/s | 0.00 m | 1 | 4 |
| 0.30 m/s | +0.08 m | 0 | 0 |
| 0.40 m/s | -0.08 m | 0 | 1 |
| 0.40 m/s | 0.00 m | 5 | 5 |
| 0.40 m/s | +0.08 m | 0 | 0 |

HC4-R completed more passages and improved weighted passage time by 0.177 s,
but increased collisions from seven to ten and reduced clean-pass rate by
0.182 percentage points. Its maximum per-cell torque-utilization p99 was
0.5690 versus 0.5425 for HC4-LH; both remained inside the measured motor
envelope. Both controllers recorded zero timeouts, falls, NaN terminations,
non-finite steps, and rated motor-speed exceedances.

The continuation report hashes are:

- HC4-R seeds 113/127: `8d98af0faca05ea506fe9e94a8aaad6f323bcb6f82b7badcad6deeab09bd3b5d`;
- HC4-LH seed 109: `31f41e213de5b6cfa01ff07cbc76a798555ab33f257323a0336b61c3ffe74bca`;
- HC4-LH seeds 113/127: `7fe8c95ef493e055ae1e1a197f61e0db275327b940425c6e1f0c23c56f19277d`.

The speed improvement does not compensate for the collision regression, so
HC4-R seed 42 is rejected and no MP4 is recorded. HC4-LH remains the selected
controller, and 0.90 m remains outside its accepted envelope. A later
near-range attempt should address closed-loop covariate shift explicitly—for
example, by collecting teacher corrections on states reached by the student—
instead of selecting another random BC initialization from the same fixed
dataset. Any future composition must be explicitly gated by range and valid
geometry; an invalid obstacle estimate continues to fall back to the accepted
controller or a higher-level fail-safe stop.
