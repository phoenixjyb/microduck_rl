# V8 steady-steering lesson — one reward change

The [measured route comparison](2026-10-13-gait-aware-route-qualification.md)
shows adequate support cycles but persistent heading drift after adaptation.
The original v8 actor steers better and walks slowly; the two learned actors
increase forward speed and lose heading accuracy. The cautious initial-rate
contrast removes the first-update KL shock but does not resolve this tradeoff.

## Prospective hypothesis and fixed recipe

Protocol `v8-steady-steering-v1`, unique run
`v8-steady-steering-<source8>-20261013`. Build on the cautious v8 recipe and add
ONLY `yaw_bias`: existing `tasks.mdp.body_yaw_bias_cost`, weight **−1.0**,
`command_name="twist"`, `tau_s=1.0`, fixed from step0. This is positive L1 cost
of a one-second EMA of `(command yaw − measured body yaw rate)`; negative weight
penalizes persistent tracking error. It is not an absolute rotation tax: a
correct nonzero turn costs zero, alternating error cancels, and command/episode
changes clear stale history. No action, observation or deployment filter is added.

The signal is privileged true body-frame rate, matching the existing body-rate
tracking objective, not projected world heading or a raw camera. Keep stock
IMU mounting/noise/delay randomization and command semantics unchanged. The
new measured world-path contract remains the qualification: a body-rate reward
improvement alone is insufficient, especially during tilted gait. Weight/tau
and all gates are fixed before replaying the prospective reward on old traces;
do not tune them to those measurements or during the run.

All other rewards, seven stock curricula, command sampler, plant/actuators,
61D actor input,14 raw actions/no clipping, noise/delays, timestep and Gaussian
settings stay fixed. Original v8 mean and frozen affine only; std.3 and fresh
critic/Adam/clocks. Never resume either rejected final999. LR1e−4 adaptive KL.01,
five epochs/four minibatches. Prove the new full cfg matches cautious after
removing only this reward, its read-only metric and protocol/run identity. Prove
initial actor/critic tensor bytes and zero state match retained stock initial
controls, whose LR is the separately declared1e−3. Do not claim the actual new
reward recipe equals the old recipe or that entire physics trajectories match.

## Bounded stages and measurements

Smoke64×5 seed947 → benchmark256×10 seed953 → fixed initial actor evaluation →
foundation256×1000 seed967 → final999 evaluation. Save0/250/500/750/999 and
select ONLY999. Controls are the sealed initial PTs listed in the
[cautious declaration](2026-10-13-v8-cautious-initial-lr.md); no duplicate control
GPU job or checkpoint search. The sealed cautious1000 round is the learning
contrast, with source/hash lineage retained separately.

Before any full budget require complete finite smoke/benchmark, regular
normalized ONNX export and conservative benchmark projection<2070s. First
update of every stage retains20 raw minibatches and must have mean KL≤.05 and
maximum minibatch mean≤.10, with evidence written BEFORE refusal. Preserve
initial and explicitly labeled aborted state on refusal, stop the sequence,
restore only the original authorized workload after proven Duck exit, diagnose
read-only and do not automatically change/retry the rate, weight or budget.

Capture raw residual, signed EMA, absolute cost and ACTUAL weighted reward
after reward/before reset. Verify finite cost≥0, reward≤0 and reward=−cost,
live term identity/weight/tau and exactly24 consumed snapshots/update; metric
reads must not advance EMA. Include terminal samples and idle/straight/negative/
positive command buckets, actual dwell and live curricula. Check signal activity
in small-run admission, not just config presence or total reward growth.

Held-out seeds839/853/857,8 worlds,240×.02s,startup60 and the same four commands.
Retain exact checkpoint bytes, complete first episodes, raw body/gait/measured
root XY traces and both old numerical and new route decisions. Original raw
yaw remains diagnostic and its legacy gate stays unchanged. Every world/cycle,
trajectory maximum, motor/hinge/rated-speed/idle gate stays fixed. Baseline may
fail tracking (the training target) but must meet old complete/motor/range
safety requirements. Final motor/no-end failures prohibit any promotion.

Smaller heading error cannot buy loss of walking/idle. Report final against
original and sealed cautious final999 for speed MAE/pass counts, actual route
error/case/world counts, continuous heading/cycle accuracy, falls, motor force
(dimensionless utilization vs Nm distinguished), power, thermal proxy and
named left/right hip margins. Require the complete new route matrix for a
route-qualified simulation claim; describe gains without promotion otherwise.

Separately fixed retention gate: moving body-speed passes must not drop below
the sealed cautious66/72; EACH moving command's mean body-speed MAE must not
exceed the sealed cautious mean by more than0.02m/s; idle must still pass all24
worlds at speed MAE≤.03 and raw yaw MAE≤.10. Authenticate the compact cautious
summary hash `5c8542012dd31a42e6482300290de20ea1a90b68735a720f94f7c7b2f20f0b32`
before a foundation launch and retain this gate alongside the new route and
legacy decisions. A better steering score with lost walking is rejected; all
gates are prospective and neither comparator accepts a physical skill.

One sequential nonenabled retained user service, caps300/300/600/2100/600s,
outer4500s,8GiB RAM/200% CPU/Torch20% GPU; exact clean source, runtime, UUID,
lease/occupancy/temperature fences. Preserve DINO and unrelated workloads;
pause/restore ONLY originally active user vLLM. Do not reuse stale September
windows, touch100.98/NX/drivers/packages, run raw perception, publish policy,
create video, start harder skill jobs or authorize physical motion.

Focused/source and exact native affected tests plus preflight precede launch.
Results pending.

Preparation checks:129 focused tests and1543 affected local tests pass (18
existing warnings in the wider set), including the installed RewardManager
compute/reset path, transition-weighted signed buckets, nonzero actual reward
activity, matched initial state receipts, first-KL refusal and speed/idle
retention. Compile, diff-whitespace and63 local documentation links pass.
Native exact-source checks and the actual GPU smoke remain separate gates.

## CPU objective rehearsal, not a learning result

The fixed prospective weight/tau above were chosen before this check. A
CUDA-hidden rehearsal reads all36 hash-bound pre-action route traces under
their sealed inventory and invokes the existing reward helper, twice per frame
to prove metric reads do not advance state. No optimizer, simulation stepping,
policy inference or workload change occurs. Because the frames are pre-action,
these are hypothetical cost values, NOT actual past post-step training rewards.

| Command vx / yaw | Original mean EMA cost | Cautious final999 | Original raw yaw MAE |
| --- | --- | --- | --- |
| .30 / 0 |.048208 |.148922 |.253558 |
| .20 / −.20 |.052674 |.154195 |.259781 |
| .20 / +.20 |.045799 |.112094 |.232953 |

This demonstrates an active signal that discounts original within-gait rate
oscillation and charges the learned persistent drift; it does not prove causal
benefit from training or exact equivalence to projected heading. No settings
were fitted to this table. Actual post-reward/pre-reset activity is still an
admission requirement in the native smoke/benchmark.

Correct rehearsal SHA256:
`4a02ee290b1863d73046c22bbc20cf2d3e61b3b8436521259697831d93822ee1`,
retained separately under the old route state directory as
`steady-steering-objective-rehearsal-v2.json`. The first diagnostic harness
incorrectly rebound an angular tensor without updating the scene-held reference;
its stale-zero input report remains retained and INVALID, hash
`43200bbd632d579b774ecb277ff5b1ac295c50d986903e58674f4e684f4e402e`.
The correction copies into the held tensor and asserts scene/input identity.
No learner or runtime source was changed to fix that harness, and neither old
raw packets nor their sealed results were overwritten.

## Initial native smoke and software-only corrections

Exact source `a6ed1bb677067721a95a2b33ce555060343a8d98` passed1543 native
affected tests. The first launcher failed CPU preflight with
`TypeError: dict() got multiple values for keyword argument 'mode'`; no Duck
packet/learner existed, original vLLM PID1123738 was never paused or restarted.
The old failed unit/receipt remain intact. The corrected unique `-preflight-v2`
launcher passed CPU preflight with no weight, rate, recipe or gate change.

That run completed its native64×5 smoke:7680 transitions,4.212s,0 falls,
0 NaNs, actual reward activity120/120 steps, mean raw EMA cost.13579684 and
weighted reward−.13579684. First-update KL mean.014301/max minibatch.026966
passes. Imported initial tensor states match the controls; actor normalizer
remains frozen and regular normalized export is durable. The benchmark did NOT
start: its prerequisite validator incorrectly looked for `negative_yaw` and
`positive_yaw` inside the eight-bin exposure map. Those four full-coverage
counters are top-level; the bins separately partition all actual transitions.
This is a schema bug, not a failed learning/skill gate.

The fix checks both actual layouts: four top-level positive counters and exact
eight integer/nonnegative bins totaling the declared transitions. It preserves
the different straight/sign thresholds (`abs(wz)<.01` versus exact-zero/any
sign), rather than forcing incompatible counters equal. A regression fixture
uses the retained native counts: top idle480/straight1680/negative2760/positive
2760; eight bins480/1560/960/480/1320/480/480/1920. The recipe, budget, signal,
first-KL and qualification gates are unchanged. Repeat the smoke from fresh
state under a new tested source identity; never restart/overwrite the old run.
Corrected-source preparation:135 focused tests and1549 affected local tests
pass,18 existing warnings; no learning recipe or numerical gate changed.

Preserved partial packet `artifacts/training/v8-steady-steering-a6ed1bb6-20261013`
contains initial/final smoke checkpoints, updates, actual activity, recipe/state
proofs, guard and export. Original user vLLM was restored active after the
prerequisite failure, with DINO unchanged; no long training budget was spent.

| Partial smoke evidence | SHA256 |
| --- | --- |
| Result | `f902c9245cc80d7f93cbc38250de8f4add84d174f59ed2c3000d7887ac1ac7fa` |
| Initial PT | `b3953680298c950979245519a3c107af19dca8e963af8aec37cc1e485faffb49` |
| Final model4 | `695cfdf0e3b50f6c63d11a934f781c31b23364d774a53d8944aff86c4eb0f1ff` |
| First-update KL | `424a93facd81fe5ae261557d7216ec564fdfe3f741f34ffee2e7829609eb6f0c` |
| Normalized ONNX | `229cbba519dc1c58e84952aea600dd0cb66f0243b27e5735a514d7195066dd56` |
| Invalid first preflight script | `90c271ca52543ca0140a3a39f9c724286083a6582b4bab3406661b5af5f7590b` |
| First no-restart receipt | `74011b1eaedcc07aeaf816675b911695f10b60660dcc7bc551d4185cb36091e3` |
| Correct CPU preflight script | `968266748fee21b953904951c9a4b83185caecf0c5caeab5465ec15753d85327` |
