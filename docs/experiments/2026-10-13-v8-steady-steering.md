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
The completed exact-source outcome is recorded below; it is not promoted.

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

An additional CPU-only audit on the retained smoke identified callable-name
instability: the script records the owned read-only metric as `__main__`, while
an imported independent verifier sees its package name. All cautious/reduced
recipe hashes were identical; only that metric's name changed the steady hash
(`c327d953…` script vs `db390c44…` import), reproduced without simulation.
Normalize ONLY this verified owned callable to its package-qualified identity
in the canonical audit copy, with recipe-proof protocolv2. Do not modify the
live function, cfg or reward, or hide any unrelated callable/parameter. A
namespace regression checks import/script equivalence and still rejects a
different metric. The `adfc48fc` corrected revision passed1549 native tests but
no learner was launched from it; the next fresh source incorporates this audit
fix so its receipts can be independently verified without rewriting old ones.

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

## Completed corrected-source round

Execution source **`7f55ded7c615e0adda4f859e6e05dde40c447e66`**, clean native
checkout, passed **1550 affected tests** (18 existing warnings,45.54s), CPU
control/retention preflight and independent imported-module smoke/benchmark
receipt validation. The namespace fix has136 focused local passes. No learning
weight, rate, budget, plant, sampler or qualification gate was changed to repair
the software-only failures. This is a fresh run, not a restart of either old unit.

Retained packet:
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/v8-steady-steering-7f55ded7-20261013`.
Its sequential user service
`recomo-duck-v8-steady-steering-7f55ded7-20261013.service` finished with
`Result=success`,exit0,inactive. Stages took4.221s smoke,7.630s benchmark,
58.755s initial evaluation,**620.539s foundation**,60.618s final evaluation.
Foundation completed **1000 updates / 6,144,000 transitions**, with892 training
fall terminations and0 NaN terminations. These exploratory training terminations
are distinct from the held-out episode results below.

First-update foundation KL mean **.01021270**, maximum minibatch mean
**.02042201**, passes the fixed .05/.10 guard. Smoke and benchmark passed their
own guards. Actual post-reward bias activity covered24000/24000 foundation
steps; mean EMA cost.16026596 and mean actual reward−.16026596. All1000 update
rows are finite with24 snapshots and20 optimizer minibatches each. The
independent CPU verifier checks exact initial state matches, empty initial Adam,
frozen actor affine/count0, final critic count6144000, all17 Adam states at
20000 steps, final environment clock24000 and checkpoint cadence0/250/500/750/999.
Regular normalized ONNX export is retained, not a deployment approval.

Maximum sampled GPU use1752MiB, minimum free14442MiB, temperature67°C. DINO
PID2029 was preserved throughout; originally active **user** vLLM was restored
active and its GPU ownership verified (engine PID1213138 at9692MiB). User
subject worker and both system service scopes remained inactive. No100.98
workload, package/driver, physical robot or perception model was touched.

### Held-out result and matched descriptive comparison

All12 final cases completed240 steps,8 worlds each, with no first-episode
terminal/fall or motor/hinge/rated-speed gate failure. Idle passes3/3 full cases
and24/24 speed/yaw worlds. Moving speed passes **66/72**, equal to the sealed
cautious control, and each command's MAE stays within its fixed+.02m/s allowance:
the independent decision is **`walking-idle-retained`**. Legacy moving yaw
passes0/72; its old rejection is unchanged.

| Moving vx / yaw | Cautious body-speed MAE | New MAE | Cautious full heading error / T | New heading error / T | Cautious full XY error / T | New XY error / T |
| --- | --- | --- | --- | --- | --- | --- |
| .30 / 0 |.074133 |.070188 |.156549 |.074679 |.121502 |.085446 |
| .20 / −.20 |.040906 |.048545 |.170334 |.081787 |.071298 |.053056 |
| .20 / +.20 |.039725 |.043694 |.120784 |.095747 |.059763 |.058954 |

Speed and XY rates are m/s; heading rates are rad/s. This table uses the
endpoint error divided by the whole settled interval, not the stricter
trajectory maximum or complete qualification. Descriptive gains: roughly52%
less full-interval heading error for straight and left commands,21% less for
right; straight/left XY error decreases roughly30%/26%. Actual mean body yaw
is now+.013690/−.163860/+.166679, versus cautious+.131807/−.043814/+.158711.
Actual body forward speed is.232875/.161649/.166122. This supports the fixed
low-frequency reward hypothesis, but one training seed and nondeterministic
GPU trajectories do not establish a statistical effect or hardware capability.
The slow original actor still has smaller full-interval heading error
(.043906/.051030/.049181); this round improves the speed/steering tradeoff
against cautious adaptation, not every measure against the original teacher.

The full **route gate still rejects all9 moving cases**, with only **3/72**
moving worlds satisfying every route component (cautious2/72). Idle passes
3/3. No actor is promoted and no harder skill/video job follows from this round.
Held-out maximum force utilization.593511 (dimensionless), raw force p99
.356107Nm, minimum named motor-hinge margin.137737rad. This passing held-out
envelope must not be confused with stochastic training maxima: utilization
1.067539, soft-limit fraction.082252, rated-speed exceed fraction.001070,
absolute mechanical power7.137665W and thermal proxy.154415. The proxy is not
measured motor temperature or a physical safety certification.

### Read-only rejection diagnosis and next lesson

Counts below are reductions of the hash-sealed route cases under the unchanged
gate, not a new acceptance definition or a checkpoint search.

| Command | Worlds passing full XY endpoint AND maximum | Worlds passing full heading endpoint AND maximum | Complete cycles | Cycle heading endpoint failures | Additional peak-only heading failures | Worlds with fewer than3 physical cycles |
| --- | --- | --- | --- | --- | --- | --- |
| .30 / 0 |17/24 |16/24 |164 |48 |51 |0 |
| .20 / −.20 |21/24 |15/24 |150 |43 |38 |2 |
| .20 / +.20 |22/24 |15/24 |153 |58 |36 |1 |

Existing completed cycles have no duration or right-touchdown-count failure.
Straight worlds have6–7 cycles; three turning worlds have insufficient physical
cycle coverage. `valid_cycle_count` also requires path/heading accuracy; its
failure is NOT automatically a contact-sensor failure. There are149 cycle
heading endpoint failures and125 additional peak-only failures across467
cycles. Therefore removing trajectory maxima would neither preserve the fixed
gate nor solve all residual steering failures.

Next prioritize a separately predeclared **projected-world heading lesson**:
the current reward targets body yaw, whereas qualification measures unwrapped
projected heading and the actual path over each gait cycle. First design/test
command/episode reset, angle unwrapping and tilted-body behavior of that reward
using existing privileged simulator state. Preserve the61D actor interface,
raw14D actions, perception separation, walking/idle retention and all existing
route/safety gates. Predeclare one matched reward change and its units/budget
before a fresh smoke; do not blindly extend this rejected actor, adjust gates,
add observation filters, or claim an obstacle/hop/football skill from this result.
That next learner has **not** been implemented or launched by this closeout.

### Durable evidence

The native packet contains100 sealed files /254823608 bytes. Full PTs and raw
traces stay native; only the32095-byte summary and15079-byte inventory are
copied to Mac `artifacts/training/v8-steady-steering-7f55ded7-20261013-closeout/`.
Independent CPU verification recomputes route, legacy, retention and paired
descriptive results from raw cases, checks every declared checkpoint/export
and authenticates original workload restoration. Its retained source is
`/home/converge/.local/state/verify-steady-steering-7f55ded7-closeout.py`.

| Completed evidence | SHA256 |
| --- | --- |
| Foundation result | `efc31f498a25f1c70c647ae2e686834936e64527a64d615b5348b81c05a661fa` |
| Foundation initial PT | `5f9f8778078dd343df7533d5b6873c58989f1656807e67fbeff3d5a8a2c141c7` |
| Final model999 | `d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f` |
| Normalized foundation ONNX | `5c7d10c007addebb9d12c0d982bf331490a98ce56ffc3a9016200646202ca1bd` |
| Initial evaluation | `0a6052d587f7a54d2c0c939e4be872988c3b4f20cc229a3a7c948a45819d38e9` |
| Final evaluation | `a361471830a58a2a02e49e9b4fe2c3b5a12e6a1844b5f759a44b90ce56cf2966` |
| Verified summary | `182e7fd99c6ec069f8e6b932ccc3008e0125fb6ba94a7e08f55e6d72edf7f155` |
| Full packet inventory | `8e0954b7853d9e810a9e13127ba0c9f86e2be1d16896da81cceffbae0b9c6de6` |
| Original-workload restoration | `3b09f2976b1a960ceec83efccdcdc761f0e5599ad6bd0df2d2d46104434162a9` |
| Independent verifier source | `dc97791c7c599a666cf8e86b69eeac98e8a05c87ecaaa468354afdcc90e7bbcd` |

Source/tests, actual training completion, numerical qualification, publication
and physical outcomes remain distinct. The later evidence-document commit is
not the executed training source.
