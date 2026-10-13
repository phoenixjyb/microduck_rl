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

The successful earlier smoke is retained separately; it is not counted as part
of this new budget.

## Fixed final999 result — forward gains, endpoint rejected

Executed source `722e7a6ac6f9cf2995cbab37b7b13aa14e7c218a`;145 focused local tests
passed (two export deprecation warnings),1409 affected native tests passed
(18 warnings). The actual saved smoke initial/final actors were also validated
CPU-only before launch. The sequential user service completed successfully and
the original user vLLM was restored with DINO preserved.

The10-update benchmark took9.517498 s (61,440 transitions), projecting1487.6247 s
against the2070-s admission bar. Its original-mean baseline passed the required
first-episode/motor/range safety subsets but rejected all nine moving tracking
cases, as anticipated. The fresh1000-update learner completed6,144,000 transitions
in567.4296 s, with finite optimizer/model state,1321 training falls and zero NaN
terminations. All17 Adam states reached20,000 steps; the actor affine stayed
byte-equal/count0 while the critic accumulated6,144,000 samples. Actual command
exposure reconciles to the complete budget. All declared checkpoints are durable.

| Command vx / wz | Initial → final actual vx | Initial → final actual wz | Speed passes /24 | Yaw passes /24 |
| --- | --- | --- | --- | --- |
| 0 / 0 | -0.000117 → 0.000365 | 0.000381 → 0.009594 | 24 → 24 | 24 → 23 |
| .3 / 0 | .124786 → .221941 | .029246 → .118079 | 0 → 17 | 0 → 0 |
| .2 / -.2 | .082334 → .148851 | -.188499 → -.070434 | 2 → 24 | 0 → 0 |
| .2 / +.2 | .081179 → .151294 | .174431 → .201844 | 2 → 24 | 0 → 0 |

These signed body-rate/speed means are descriptive, not replacement gates.
Moving speed passes improve4→65/72. Moving speed MAE means improve
.175214→.080965, .117666→.054419 and .118835→.051732. But moving yaw MAE means
worsen .255327→.303241, .261885→.282239 and .232182→.249833: yaw remains0/72.
The final endpoint rejects all nine moving cases plus idle seed853, whose world4
has yaw MAE.300762 and mean body wz.183229. Positive mean turning approaches its
target while straight/negative-turn mean authority regresses; averaging hides
the instantaneous and idle failures. The endpoint is not promoted.

All12 initial and12 final cases completed240 steps without a first terminal,
rated-speed, motor or range gate failure. Final maximum case force-utilization
p99.571236 (raw.342741 Nm) is below the unchanged.60/.36 Nm bar. Minimum named
motor-hinge margin tightens .176525→.023102 rad; the final closest joint is
**left_hip_yaw**, not the previously investigated right hip. It remains in range,
but this is a meaningful new safety/steering diagnostic, not physical acceptance.
Do not clip intentional low-kp targets or copy the old right-joint fix blindly.

GPU peak1854 MiB, minimum free14340 MiB, maximum temperature67 C. Stochastic
training's peak utilization1.067539 and soft/rated exposures are training guards,
not deterministic safety acceptance; the full retained summary reports them.
No harder skill, video or physical motion followed the failed numerical gate.

## Read-only next diagnosis and seals

The very first foundation update has raw minibatch KL mean3.788118, maximum
minibatch mean4.282735 against target.01; adaptive LR ranges1e-5–.0015 during
that update. Subsequent early means settle near.01–.02. This substantial initial
policy change is a concrete retention-risk lead, **not proof** that it caused
the final steering/idle loss. Final reward components and named-joint diagnostics
must be considered alongside it. The next useful bounded contrast is a cautious
initial learning rate from the ORIGINAL v8 actor with matched seeds/initial
actor/critic states and unchanged target recipe, rather than adopting final999
or running another blind reward/command scalar experiment. It needs its own
declaration, small smoke and early-KL admission guard before another long job.

Native full packet: `artifacts/training/v8-actor-foundation-722e7a6a-20261013`.
Create-exclusive CPU closeout verified the complete training budgets, optimizer/
normalizer clocks, original-mean initial actors, final checkpoint, all ordered
case files and recomputed unchanged gate. The inventory/summary are received
on the Mac (`-closeout`) with matching hashes; full traces/checkpoints stay native.

- Foundation result SHA256: `11c12dcc8fd31ffba3d39418a83776028d5861736fb5d63b62398985298aaaec`.
- Initial foundation PT SHA256: `29d3ea8491c84d57db12ea37ffad9d586fa8ab582be621e3028e5f27055ba974`.
- Final999 PT SHA256: `4631f81d36f7cc846cbed42efdedb9a16e2a40b2d1cadbcbc2834b43e016d7bb`.
- Final ONNX SHA256: `aab5f33d339f888977b7f2cc0a7413813d5727625b123dbf47bbc89df1870680`.
- Initial evaluation SHA256: `d28e0e866b6f38074f7f88f9f7861d9ba5626d2b1e741922a7371c24968e6b98`.
- Final evaluation SHA256: `aa083fd44993813ccd2cfd5f499075b8a562441420686c416396f60201273351`.
- Inventory SHA256: `90107a56f78ee5df936f5ef4db268d381dc97cf20c6677232f7c339796c59ec3`.
- Summary SHA256: `71f43db1064146eaee7e6bbef570c53f4288c5804bd57c0be0264d785d89017a`.
- Restoration SHA256: `dd70ae32a15553693c6459d2cfa46e5239bffad4f28d46ba780e513e24cb50f4`.
