# HC4-R2H range/speed-gated composition

Date: 2026-09-04

Implementation parent: `df8d686a4232d16ae1184be78a50475d5ecb6b59`

Decision: **composition complete, pending closed-loop rollout; HC4-LH remains
selected**

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
- composed checkpoint: `../artifacts/hc4r2h-df8d686/supervisor.pt`, SHA-256
  `ef4b1172570052f33ac145f4151eb1560decb8b9a0ad221c7c75055db8ae6cf1`;
- manifest SHA-256:
  `b754ebd82d06b0805a4ee82fe8f1b325aea4da673b07531890dbe3f1ffa43a44`.

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

HC4-R2H remains `composition-complete-pending-rollout` until both checks pass.
No MP4 is recorded before numerical acceptance, and no result expands this
exact-geometry simulation boundary to noisy perception or physical motion.
