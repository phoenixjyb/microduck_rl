# HC4-U3 phase-separated supervisor

Date: 2026-09-06. Parent source: `a1feec4ec1ae6f5e1ac19cea2f6ca1c290c2a11f`.

Status: predeclared single-candidate architecture diagnostic, simulation only.
The user authorized continued Duck development and training until 07:00
Asia/Shanghai today. O3a's running evaluation completes before this GPU job.

## Evidence and hypothesis

HC4-U1 and HC4-U2 each regressed at the far-center 0.30 m/s timeout gate.
HC4-U2 used more correction data but did not remove that local regression.
A read-only replay of its retained seed-42 validation split found speed MAE
0.003329 m/s in approach, 0.002339 in interaction, and 0.006029 in recovery.
Recovery p95 speed imitation error was 0.023411 m/s. These offline errors do
not prove timeout causality; phase interference is the hypothesis to test.

HC4-U3 replaces the single 64x64 supervisor with three independent 64x64
experts selected by the existing one-hot approach/interaction/recovery phase.
Each receives the unchanged 17D observation and outputs the same normalized
forward speed and yaw command. Phase selection uses no range threshold or new
sensor. Malformed phase codes output stop. The existing command limiter and
invalid-geometry stop still execute after selection.

The only intended experimental axis is the phase-separated architecture.
It increases parameter count; any improvement cannot distinguish additional
capacity from phase separation without a later matched-capacity control.
No such control or promotion is implied by this diagnostic.

## Frozen fit

- Ordered corpus: exactly `HC4U2_REQUIRED_DATASET_SHA256`, all 13 shards from
  the HC4-U2 manifest, including the same three correction shards and student
  identity. No O3a or evaluation observations enter training.
- Frozen locomotion actor SHA-256:
  `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- Seed 42, 200 epochs, batch size 1024, AdamW learning rate 0.0003, weight
  decay 0.00001, unchanged episode-disjoint 20% validation split.
- Use all 433,255 samples and the same 346,103/87,152 train/validation split.
  The three training phase counts must be 15,386/210,979/119,738; validation
  counts must be 4,101/52,820/30,231. Require finite observations and labels,
  valid one-hot phases, and nonempty phase coverage in both partitions.
- Select the minimum validation-MSE epoch as before. Retain the checkpoint,
  full dataset hashes, architecture, phase counts, per-phase validation errors,
  and periodic loss journal. Save an atomic progress checkpoint at epoch 1 and
  every ten epochs, with optimizer and sampler state; it is explicitly
  ineligible for rollout. Offline limits remain speed MAE <=0.025 m/s and
  yaw MAE <=0.050 rad/s; offline fitting alone never promotes the policy.

## Frozen evaluation

Before evaluation, record the trained checkpoint hash and source commit.
Fresh held-out physics seeds are 293, 307, and 311, in that order. Seed 293 is
the first gate and the remaining seeds run only after a complete pass. Each
seed uses 64 first terminal attempts per cell, ceiling 700 steps, exact compact
geometry, speeds 0.30/0.40 m/s, ranges 0.90/1.15 m, and lateral positions
-0.08/0/+0.08 m: twelve candidate cells, 768 attempts. Pair HC4-R2 at 0.90 m
and HC4-LH at 1.15 m using the existing accepted checkpoint hashes.

Apply the unchanged HC4-U1/U2 gate to the distinct HC4-U3 rollout identity:
no per-cell or pooled increase in collision or timeout counts, no decrease in
clean counts; approach/recovery speed loss <=0.03 m/s per cell and <=0.01 m/s
sample-weighted across the matrix; torque-utilization p99 <=0.60 per cell.
Require all fixed attempts resolved, finite metrics, zero falls, hard/other
failures, NaN terminations, nonfinite steps, and rated-speed exceedance.
Interaction speed remains ungated. Any failed local check stops this candidate;
do not reinterpret a pooled improvement as a pass or adjust the thresholds.

This is one training seed. Even passing all three held-out seeds establishes
only a diagnostic survivor pending independent training seeds and capacity
controls. The accepted HC4-LH/HC4-R2 specialists remain available unchanged.
No noise, raw perception, new geometry, hop/H2, video, or physical motion is
part of this experiment.

## Runtime budget

Run one retained user service on 100.100 after an idle-GPU and protected-service
check. Bound the fit to 45 minutes, and each held-out seed matrix to 20 minutes;
recheck the clock before every launch and reserve 15 minutes before 07:00 for
artifact retention. Never overlap GPU workloads. At cutoff, launch nothing
further; retain completed artifacts and leave protected services inactive.
