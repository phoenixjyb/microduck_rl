# HC4-R2H range/speed-gated composition

Date: 2026-09-04

Implementation parent: `df8d686a4232d16ae1184be78a50475d5ecb6b59`

Decision: **reject the stateless composition; HC4-LH remains selected**

## Purpose and authority boundary

HC4-R2 passed only at 0.30/0.40 m/s with an obstacle initially 0.90 m ahead.
HC4-LH remains the selected controller for its farther-range exact-geometry
envelope. HC4-R2H combines those retained checkpoints without retraining either
network:

- valid obstacle, reconstructed route-forward range at or below 0.95 m, and
  nominal speed at or below 0.40 m/s: use HC4-R2;
- every other observation: preserve HC4-LH exactly;
- invalid geometry: route through HC4-LH, after which the existing execution
  layer commands its immediate fail-safe stop outside recovery.

The gate reads only the existing 17D compact supervisor observation. It uses
externally supplied structured geometry and route state, not camera pixels or
simulator entity identity. It produces only normalized forward-speed and yaw
commands for the frozen locomotion actor. Physical motion remains unauthorized.

The 0.95 m threshold gives the accepted 0.90 m cell 0.05 m of exact-simulation
margin while leaving 0.96 m immediately outside the specialist's initial
authority. There is no hysteresis in this exact-geometry stage; sensor noise,
lag, dropout, and gate hysteresis remain HC6 work.

## Retained checkpoint

The composition embeds all model state needed to reconstruct HC4-LH's 0.02 m
lateral gate and the HC4-R2 specialist. Creation fails closed on mismatched
locomotion hashes, model configurations, physical-motion authority, or missing
HC4-LH composition fields.

- HC4-LH source SHA-256:
  `0b2608080671c5df85d8c9f900d68b6a6f298ec820eb1c6ba75afc948337505a`;
- HC4-R2 source SHA-256:
  `c4ba5925de7144373c94145b57b5e7a7ae3e1fc89bc7c2c3203f8724bdebf1b7`;
- composed checkpoint: `artifacts/checkpoints/hc4r2h-2962b11/supervisor.pt`,
  SHA-256
  `53808730f1558f6b05009817631dfff68863ee45131854ececf9fc9eae9ec660`;
- manifest SHA-256:
  `3379847eab3ccbcc774eac512d6e793fd1876096a589e5388824c049bece48e6`.

An actual-checkpoint CPU load check established byte-exact output selection on
four synthetic observations: near/slow selected HC4-R2, while outside-range,
outside-speed, and invalid observations each matched HC4-LH exactly. All output
was finite. This is a wiring check, not closed-loop acceptance.

## Predeclared closed-loop gate

Held-out seeds are 163, 167, and 173. The base locomotion checkpoint, obstacle
box geometry, episode timeout, command bounds, and metric definitions remain
unchanged.

The first retained run is seed 163 over 0.30/0.40 m/s, forward ranges
0.90/1.15 m, and lateral positions -0.08/0.00/+0.08 m. Continuation to seeds
167 and 173 requires zero falls, NaN terminations, non-finite steps, and rated
motor-speed exceedances, plus no obvious per-cell collision regression.

The paired baselines are HC4-R2 at 0.90 m and HC4-LH at 1.15 m. Promotion
requires, across the resulting 36 cells:

- zero hard-safety failures and rated motor-speed exceedances;
- no per-cell or aggregate collision increase against the selected baseline;
- no timeout increase and no lower pooled clean-pass rate;
- maximum per-cell torque-utilization p99 no greater than 0.60;
- retained approach and recovery speed tracking, while interaction speed is
  descriptive rather than a nominal-speed gate.

If that matrix passes, one 12-cell threshold diagnostic uses the same seeds at
0.30/0.40 m/s, centered lateral placement, and initial ranges 0.94/0.96 m. It
must add no collision or hard-safety discontinuity across the gate. Passage
time is reported but is not optimized inside interaction; attempt timeout still
fails normally.

## Closed-loop result

The seed-163 pre-screen had 864 clean passes, zero collisions, and five
timeouts among 869 resolved attempts. Its maximum per-cell torque-utilization
p99 was 0.5611, with zero hard failures and rated-speed exceedances. The paired
seed-163 baselines had zero collisions and eight timeouts, so continuation was
permitted.

Across all three seeds, the result was:

| Controller/cells | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC4-R2H, full matrix | 2,604 | 3 | 6 | 2,613 | 99.656% | 7.943 s | 0.5649 |
| HC4-R2, 0.90 m half | 1,448 | 2 | 0 | 1,450 | 99.862% | 7.360 s | 0.5656 |
| HC4-LH, 1.15 m half | 1,143 | 0 | 11 | 1,154 | 99.047% | 8.960 s | 0.5441 |
| Paired selected baselines | 2,591 | 2 | 11 | 2,604 | 99.501% | 8.066 s | 0.5656 |

HC4-R2H improved timeout count, pooled clean rate, and weighted passage time,
but it added one collision. The regression was at seed 173, 0.30 m/s,
0.90 m forward, and centered lateral placement. At seed 173 and 0.40 m/s in
that same centered near cell, both HC4-R2H and HC4-R2 recorded two collisions.
All shifted near cells and all 1.15 m cells were collision-free. All controllers
had zero falls, NaN terminations, non-finite steps, and rated motor-speed
exceedances.

The retained report hashes are:

- HC4-R2H seed 163:
  `2c376dbadaecc8a47417330751411c9f7b25326640eed2827a40a0d0f420ff63`;
- HC4-R2H seeds 167/173:
  `a4d7225368849debaa275cb3c9200e8bebaa1dca48164cbd8b90c43a923821af`;
- HC4-R2 seed 163:
  `67149fdb5cff1ee4a2fdfc97f43b364a1cf45df01a6b2e67044ccb2e2f8cc2fd`;
- HC4-R2 seeds 167/173:
  `b1869f93b49163c80c7fbf02f356c16da7d8ffa1382e72851b95c015c80c1b82`;
- HC4-LH seed 163:
  `a3c8e016c2ebd8c166bc500bf874795f245fbddcc41777b0c0d540a82f61b009`;
- HC4-LH seeds 167/173:
  `78edebfbc9d691c4820bfd573dda6aee8eb16d655c4f32c6c1702f4b10be7831`.

HC4-R2H therefore fails its predeclared per-cell and aggregate collision gates.
The threshold diagnostic was not run and no MP4 was recorded. A likely design
cause is that a stateless range gate can change specialists during one
maneuver, whereas each source controller was evaluated consistently for a
whole episode. The next bounded design is an episode-latched range/speed
selector with an explicit reset interface; it must retain the same checkpoints,
thresholds, seeds, and physical-authority boundary.
