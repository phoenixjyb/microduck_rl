# HC2 retained simulation assets — 2026-09-03

This manifest binds the HC2 simulation evidence to source commit
`2f7c2f38fbfa715810bd9059318fe6c57bf7d61f`. Large generated weights and media
stay outside Git history. They are retained on both the Mac workspace and GPU
host `100.100`; matching SHA-256 values below verify the important duplicate
copies.

This is simulation evidence only. It does not authorize physical motion or
claim raw-camera perception, deployment, or real-robot acceptance.

## Frozen low-level locomotion actor

- Remote path: `/home/converge/work/microduck_rl-athletics-obstacle-curriculum/logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt`
- SHA-256: `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`

## HC2 supervisor

- Mac directory: `../artifacts/hc2-bc-ef57af1-s42/`
- Remote directory: `/home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/checkpoints/hc2-bc-ef57af1-s42/`
- `supervisor.pt` (26,937 bytes): `44be1ec5734cad69e11cbfb060180c5d028761ccb267a872105cdf231767b050`
- `supervisor.json`: `3889cf45c6ab18699446eebc15aff1f5a87cac48ae97196b9060e8e0a34e75ee`

The metadata records 69,441 successful-trajectory samples from 765 episodes.
The held-out command errors were 0.0039 m/s forward-speed MAE and 0.0226 rad/s
yaw-rate MAE. Offline imitation was only a gate to closed-loop evaluation.

## Closed-loop evaluation evidence

Mac directory: `../artifacts/hc2-evaluations-21018d7/`

| File | SHA-256 | Clean | Collision | Timeout | Resolved |
|---|---|---:|---:|---:|---:|
| `smoke-030-x115-s41.json` | `46d90503a3be5362783bd90748dd1710cc1e3e2bd8d0b700be6af692a3476f87` | 63 | 0 | 1 | 64 |
| `heldout-030-x115-s4243.json` | `31231a7a21136aa2c618b20279d4c08848f8950b2b4c7dd16ec14a0e449f74fe` | 128 | 0 | 0 | 128 |
| `smoke-050-s41.json` | `2d9e8ae0234525bd6fb7947c27b8a8f47ce29886e018de28281c4937a559a7ea` | 124 | 0 | 4 | 128 |
| `heldout-050-s4243.json` | `722b45e5f723185d07975e5a60d447be131615a02601ccbcb74fa737dea9a40c` | 255 | 0 | 1 | 256 |
| `smoke-080-x140-s41.json` | `9a42763b7a3ad906be22fce45ab144357e0d73b9819c2156c7d545ae1214eb42` | 63 | 1 | 0 | 64 |
| `heldout-080-x140-s4243.json` | `f24c2a919105c363e038d176e07ae695d112fcde3b1b4c8f206a65c82efb8d2b` | 126 | 0 | 2 | 128 |

The non-overlapping three-seed total is 759 clean passes, one collision, and
eight timeouts across 768 resolved attempts: a 98.83% clean-pass rate. There
were zero falls, NaN terminations, non-finite steps, or rated motor-speed
exceedances. The highest per-case torque-utilization p99 was 0.745 and highest
near-stall exposure was 0.216%.

`hierarchical-teacher-evaluation.json` in the evaluation directory duplicates
`heldout-080-x140-s4243.json` byte-for-byte and is not counted again.

## HC2 replay media

- Mac directory: `../artifacts/hc2-replays-7edc30c/`
- Remote directory: `/home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/videos/hc2-bc-7edc30c/`
- Format: 12 seconds, 960x540, 50 fps; each retained trajectory is a successful
  simulation episode with the duck and red obstacle visible.

| File | Bytes | SHA-256 |
|---|---:|---|
| `microduck-hc2-0.30mps-x1.15m-y+0.00m-step-0.mp4` | 835,951 | `c709bf49840636d5d986853f161c3aac01755bb2553ee28d3c9623ee0a67f323` |
| `microduck-hc2-0.50mps-x1.15m-y+0.00m-step-0.mp4` | 932,515 | `74e18dd6a9af3b2ba61641d51ed0f71aa67875f18235e23249cd7a0af615f7bc` |
| `microduck-hc2-0.50mps-x1.40m-y+0.00m-step-0.mp4` | 892,548 | `f9ab0553eafc7e11994b62d1dd4ceb83ea236dab102f87497444fe8743a28c66` |
| `microduck-hc2-0.80mps-x1.40m-y+0.00m-step-0.mp4` | 1,031,649 | `c9905248f6f16faa0c96fdec9b896424d95f744c7e7cee755b66a0762704c1df` |

The matching JSON sidecars and contact sheet remain beside the MP4s. Earlier
HC1 teacher evidence is retained under `../artifacts/hc1-heldout-8af0a15/` and
`../artifacts/hc1-replays-0ef15e5/`.

## Acceptance boundary

HC2 is accepted only for centered-box simulation cells `0.30 x 1.15`,
`0.50 x 1.15`, `0.50 x 1.40`, and `0.80 x 1.40` (speed m/s x forward position
m). It is not accepted for 0.80 m/s at 1.15 m, the 0.90 m stress placement,
general obstacle distributions, degraded perception, or physical motion. Mean
passage time remains roughly 8.6–9.9 seconds, above the 4.5-second O1 goal.
