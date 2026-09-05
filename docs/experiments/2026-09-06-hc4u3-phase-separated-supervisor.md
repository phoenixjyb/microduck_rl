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

## Retained fit and frozen evaluation identity

Implementation/training commit: `ab1d8312865455acc77675a71ef1c14586cc2c2f`.
All 66 focused architecture, fitting, loading, rollout, and gate tests passed
locally and on 100.100 with CUDA hidden. The fit completed all 200 epochs with
finite losses and selected epoch 200; source, sample counts, partition counts,
and all thirteen dataset identities matched the predeclaration.

- Checkpoint: `artifacts/checkpoints/hc4u3-ab1d831-s42/supervisor.pt`;
  SHA-256 `6c6546448340530e4cdb7e4381247f644211029c7750146c0da6dbcf7c60aa2d`.
- Manifest SHA-256:
  `58d8327a6271a4702a10af48bcbd597837d4232c8b729ec9c3473a12f1da860c`.
- Validation MSE 0.0001491593, speed MAE 0.001879 m/s, yaw MAE 0.005441 rad/s.
- Recovery speed MAE 0.004089 m/s and p95 error 0.014493 m/s, compared with
  HC4-U2's 0.006029 and 0.023411 m/s on the identical validation partition.

Decision: `offline-imitation-pass`, pending the unchanged closed-loop gate.
The checkpoint hash above is frozen before the first seed-293 evaluation.
Fitting improvement cannot establish collision avoidance or timeout recovery.
Retained user service: `microduck-rl-hc4u3-ab1d831-s42.service` (success, exit 0).

## Seed-293 evaluator interruption and bounded accounting repair

At source `63707439c9f137357665a6595c958da80b67cb47`, service
`microduck-rl-hc4u3-ab1d831-s293-eval.service` exited 1 during the fourth
far-specialist cell (0.40 m/s, 1.15 m, -0.08 m). The exact exception was
`ValueError: resolved outcomes cannot exceed completed_attempts`. The candidate
had not run. This is incomplete evaluation evidence, not a candidate gate failure.
The completed near baseline had 381 clean / 3 collision / 0 timeout attempts;
report SHA-256 `e8b5a414f77ede009fb53a373777abd61232334f26f633de7c96ddc7c2f4c26b`.

Read-only diagnosis left policies, physics and services unchanged. An instrumented
single-cell replay and then the complete paired-source matrix did not reproduce
the exception, overlapping flags, or flags without a terminal transition. The
matrix replay retained 381/3/0 near and 379/0/5 far. The original crash therefore
has no captured raw-event proof; simultaneous outcomes are a source-supported
failure mechanism, not a confirmed explanation of that particular run.

The source independently evaluates success and the `elapsed >= 12 s` timeout.
A success on the deadline can consequently be counted twice by the old evaluator.
The bounded repair partitions first-terminal outcomes with this explicit priority:
hard failure, collision, timeout, clean pass. Hard-failure flags remain separately
retained; unknown terminal events remain failures. Nonterminal flags raise an
error. Each report retains the raw five term counts and overlap count, under
`hard-failure-collision-timeout-pass-v1`; the HC4-U3 comparator requires this
protocol in all three reports and every cell and uses prescreen protocol v2.
Thirty-two exhaustive flag combinations and deadline/inactive-mask tests cover
the arbitration. All 106 focused tests pass locally with CUDA hidden.

This changes accounting, not the policy, simulator, resets, timing, training data,
or numerical thresholds. It does not retroactively certify old evidence or alter
training-dataset collection. Retry seed 293 in the fresh directory
`artifacts/evaluations/hc4u3-ab1d831-s293-prescreen-v2`, rerunning **both** baselines
and the frozen candidate. Do not mix the original partial baseline into v2.
Proceed to seeds 307/311 only after the same unchanged numerical gates pass.

Retained diagnostic directory:
`artifacts/evaluations/hc4u3-ab1d831-s293-accounting-diagnosis`.

- Original failure journal SHA-256:
  `3a46cd7c1fe56181c62cf8f23ad91c3d000fd11d037a39b36965bc2b44578f4f`.
- Single-cell replay journal SHA-256:
  `a23ecfb8173520f651d0c3303895dbc059eb82b0c57866873c72322fa3bb1473`.
- Source-matrix replay journal SHA-256:
  `21af02f56e56d72c98425149a45f8c2eb61267317f0c90086c3f5e8cb7d7271d`.
- Replay near report SHA-256:
  `7741e04951a599e83a78fdaa05cab3fbc30688d18b504d0213f66f42869c5e68`.
- Replay far report SHA-256:
  `393f0af7a1e1db7ea551359a242122d18f2c8fbb701d7eb4d586f81a8bae14a3`.

## Seed-293 v2 result: continue to the next held-out seed

At evaluator source `a88fa90d4449ec7375ca8249aea54b600d32b439`, retained service
`microduck-rl-hc4u3-a88fa90-s293-eval.service` completed successfully (exit 0)
at 01:52 Asia/Shanghai. All 106 focused tests also passed on 100.100 before launch.
The candidate completed all 768 fixed attempts: **762 clean / 3 collisions /
3 timeouts**, versus paired sources **758 / 4 / 6**. There were no falls, NaN
terminations, nonfinite steps, unresolved attempts, other/hard failures, or rated
motor-speed exceedance. All eight numerical checks passed, including every
per-cell outcome and before/after-speed comparison. Maximum candidate cell torque
p99 was 0.555901 (limit 0.60); pooled approach/recovery speed deltas were
-0.007868 / +0.009833 m/s. This is a relative diagnostic pass, not zero-collision
acceptance, physical readiness, or a demonstrated architectural causal benefit.

The repaired evaluator captured one actual simultaneous success/timeout in the
candidate's 0.30 m/s, 1.15 m, -0.08 m cell: raw counts 63 pass + 2 timeout over
64 attempts. Failure-priority accounting correctly retained **62 clean + 2
timeout**, not an extra success. This proves the overlapping-term mechanism can
occur, but does not identify the uncaptured flags in the earlier baseline crash.

On read-only CPU recomputation, the full decision object matched exactly:
`continue_fresh_seeds`. Fresh seeds 307 and then 311 remain gated, in order.
The checkpoint and architecture stay frozen. Subsequent directories use suffix
`prescreen-v2` and the same evaluator; no partial v1 reports are admissible.

Report directory: `artifacts/evaluations/hc4u3-ab1d831-s293-prescreen-v2`.
All four JSONs are backed up locally under `artifacts/overnight-20260906-u3`.

- Candidate report SHA-256:
  `10f27d31ab42161df5b7c089072c4ab9632ad822f20dce21951011f7b628a1cd`.
- Near source report SHA-256:
  `304fe3d9f3cf0de953532c0c95b8a841090dc4810ce2158c58bcdc6c1fb052d0`.
- Far source report SHA-256:
  `280add80a541baa2dc3232b4471cd1d26db510c7c846fbac3570cb0b3a862caa`.
- Deterministic decision SHA-256:
  `f48c300bdccdb8e7bf7912775b83b30141872d99dcec476a240e12455548f225`.

## Seed-307 v2 result: continue to the last held-out seed

At source `2b87dc5279891b10a106256a9d17b76be6180d11` (unchanged evaluator),
`microduck-rl-hc4u3-a88fa90-s307-eval.service` completed successfully at 02:12
Asia/Shanghai, exit 0. Candidate **766 clean / 0 collisions / 2 timeouts** matched
the paired sources' outcome totals over 768 fixed attempts. All eight checks
passed, with no per-cell regressions, unresolved attempts, falls, NaN terminations,
nonfinite steps, hard/other failures, or rated-speed exceedance. No overlapping
terminal flags occurred. Maximum candidate cell torque p99 was 0.563957; pooled
approach/recovery speed deltas were -0.006405 / +0.009357 m/s.

The full `continue_fresh_seeds` decision was recomputed independently with CUDA
hidden and matched exactly. The next and last predeclared held-out seed is 311,
using the frozen checkpoint, both paired sources, and the same v2 protocol.
No architecture, training, sensor, geometry, or threshold change is authorized by
this intermediate result. The one-training-seed and no-physical-promotion limits
remain in force.

Directory: `artifacts/evaluations/hc4u3-ab1d831-s307-prescreen-v2`.
All four JSONs are backed up locally under `artifacts/overnight-20260906-u3`.

- Candidate report SHA-256:
  `29dd2fc18c2ab0aba7bfaf0e04cf927468bd742f0d51153d39b717947ff8fd8a`.
- Near source report SHA-256:
  `fb1ec5a111789d8013e099c82993abe771cce81626b3621c70f17217b7794834`.
- Far source report SHA-256:
  `cd7460a41e06673d2403629ee3f9dab2d07b6631ea9c6fb6c54834fc45d5ba6a`.
- Deterministic decision SHA-256:
  `fbaa2597b510cc8241261e46dd721f63ff9a1680a7d9be4fb56b4a46624b960b`.
