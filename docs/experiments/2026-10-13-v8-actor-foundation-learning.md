# V8-seeded target foundation: benchmark, baseline, first1000 updates

The [native five-update smoke](2026-10-13-v8-actor-import-smoke.md) completed
real PPO updates with finite state, a durable frozen imported transform and
regular normalized export. The initial v8 reference still fails moving tracking;
it is a candidate actor seed, not an accepted teacher. This new experiment tests
whether fresh target learning can improve that seed. It is not a matched causal
claim about a single reward, Adam reset or upstream training method.

## Locked recipe and staged budgets

Protocol `v8-actor-only-foundation-learning-v1` in
`mjlab_microduck.v8_actor_learning`. Exact clean tested source, native100.100,
same frozen runtime/UUID/DINO and protected-workload rules. Run ID
`v8-actor-foundation-<source8>-20261013`, retained under `artifacts/training`.
One sequential user service; no source change or GPU overlap while it is active.

1. **Benchmark:**256 worlds,10 updates, seed953,24 steps/update, fresh v8-seeded
   actor + fresh scalar Gaussian std0.3 + fresh critic/Adam/clocks. Retain the
   exact initial actor as the inference baseline; do not resume the benchmark.
2. **Initial held-out:** evaluate that initial checkpoint in the unchanged12-case
   matrix. It must equal the original v8 mean weights/affine and target std0.3.
   Tracking failure is allowed as the declared learning target, but an incomplete
   episode, fall, rated-speed, force/range or right-yaw-stop failure aborts the
   next stage. It is not accepted as a teacher by passing only safety subsets.
3. **Foundation diagnostic:**256 worlds,1000 fresh updates, seed967,24 steps/update
   (6,144,000 transitions). Re-import the original mean/affine into a new actor;
   critic/Adam/counters are fresh, not inherited from either earlier run. Save
   every250 iterations and final`model_999.pt`; select ONLY final999 for primary
   evaluation. No mid-checkpoint search, retries or automatic continuation.
4. **Final held-out:** same seeds839/853/857 × idle/(.3,0)/(.2,−.2)/(.2,+.2),
   eight first episodes per case,240×.02 s, startup60 for tracking only.
   Retain motor/heading/gait traces and per-world metrics. Compare fixed initial
   and final means descriptively; only the unchanged absolute gates decide
   whether the endpoint is ready for another experiment.

Both training phases inherit the existing plain flat foundation smoke recipe
unchanged except worlds, updates, seed and run names. Raw61 actor observations,
14 named XL330 BAM actions, no clipping/filtering; imported actor normalization
frozen once, critic normalization learned from target data. Initial LR1e-3,
adaptive KL0.01, five epochs/four minibatches, stock rewards/DR/noise/delay.
Twist support vx.2–.4,vy0,wz±.5; forced-forward fraction.25; no in-place pivots;
command resampling3–5 s. Zero-idle sampling is preserved and measured in actual
consumed commands, not inferred from configured support.

Stock curricula remain live, not silently collapsed or accelerated. In the first
1000 updates (24,000 steps), action-rate cost stages at0/12,000/18,000 use
−.1/−.2/−.4; idle fractions.02/.05/.10. Head ranges widen at12,000 from beginner
±.05/±.05/±.07/±.015 to±.17/±.17/±.21/±.047; CoM/head-CoM range3→5 mm.
Head bias reward opens at14,400 with weight1. Step-function stages use their
existing strict boundary behavior; record actual live manager values each
update. This budget does NOT complete the stock48,000-step head curriculum,
recovery, hopping, obstacles or football balancing. The imported actor is being
adapted to a new target recipe, not a reproduced v8 training environment.

## Exact checkpoint inference contract

`FrozenV8Checkpoint(path,sha)` authenticates bounded owned regular bytes before
one CPU weights-only load. It validates the exact14-action actor tensor layout,
pinned v8 seed digest, float32 tensors, canonical integer zero updater counter,
positive learned scalar Gaussian std and byte-equal frozen mean/denominator/
derived affine variance. The initial baseline additionally requires original
v8 MLP values. Inference constructs only `FrozenV8Actor`; no critic, Adam or
runner state is loaded. Default legacy PPO/ONNX collectors remain unchanged.
The new typed collector path is plain-task only and mutually exclusive with
ONNX/backlash overrides. Live named motor order/HOME and observation/action
contract are checked before stepping. Checkpoint hash, backend and normalization
provenance accompany every case. CPU import parity is not a claim of identical
CPU-ORT/CUDA trajectories or physical deployment acceptance.

## Retained gates, limits and failure behavior

All existing absolute held-out gates remain: each world's speed MAE≤.03idle or
≤.10moving, yaw MAE≤.10; no first-terminal or rated-speed exceedance; pooled case
force-utilization p99≤.60 (=raw.36 Nm), named motor-hinge margin≥−.005 rad and
right-hip-yaw stop proximity≤1%. No mean/smoothing substitution for failed yaw,
no torque-unit reinterpretation, no weakened acceptance. Passing numerical gates
only allows consideration of another simulation experiment; physical/teacher/
deployment acceptance remains separate. Retain failed endpoints without promotion.

Benchmark cap300s; each evaluation cap600s; learner cap2100s. Before starting
foundation, conservative projection `benchmark_elapsed ×100 ×1.5 +60s` must be
strictly below2070s. Do not remove setup overhead or extend an active cap to make
it fit. Whole sequential outer service cap4200s, memory8 GiB/CPU200%, Torch20%
GPU allocation, existing exclusive lease/occupancy/temperature guard. Pause and
restore only originally active user vLLM, with post-exit guard; preserve DINO and
every other workload. The old dated overnight windows are not reused.

Monitor finite rollout/models/losses/Adam, falls/NaN signals, actual command
exposure, raw minibatch KL/applied LR, live curricula, force utilization, power,
thermal proxy, soft/rated exposure, durable saves, GPU occupancy/temperature.
The existing stochastic-training gross guards remain abort guards, not held-out
acceptance. Any failure stops the sequence, retains evidence and restores only
the authorized original workload after proven Duck exit. Diagnose read-only
before fixes. No H2, multi-seed promotion, MP4, raw perception or physical motion.

Results pending; source/focused checks and a native affected-suite run precede
launch. The successful earlier smoke is retained separately; it is not counted
as part of this new budget.
