# Motor/range-retaining forward tracking: matched diagnostic

## Declaration before execution

Protocol `integrated-motor-range-forward-diagnostic-v1`; module
`src/mjlab_microduck/integrated_motor_range_forward.py`.
The [completed motor/range foundation](2026-10-12-fresh-motor-range-foundation.md)
passes all12 held-out load/range subsets, but fails10 combined tracking cases.
It is **not an accepted teacher**. It may be used as this explicitly exploratory
parent; no transition collector, obstacle, hop, ball or physical promotion follows.

Parent: native `artifacts/training/fresh-motor-range-d0cba9c0-20261012/foundation/model_5999.pt`,
SHA256 `811b38b530e508d08967dba0ff194cf14fddb35011b9a4878a04f71a5b804190`.
Parent counters: iteration5999, environment144000, actor/critic normalizer counts
36864000, seventeen Adam states at step120000. Both arms restore the identical
models, normalizers, moments and parent LR, then reset episode and curriculum
clocks to0. No Adam reset or learning-rate intervention is included.

## One factor and hypothesis

Both arms retain motor cost weight-1 (nominal.60Nm, soft fraction.70, overload
gain4) and bilateral named hip-yaw qpos cost weight-4, margin.15rad. The seven
stock curricula are materialized at their **actual final parent state**, not the
old1499 endpoint: action-rate-1, head-bias3, standing fraction.25, head ranges
±1.1/±1.1/±1.4/±.31rad, body xyz±.005m and rpy±.05rad, trunk CoM±.015m and
head CoM±.01m. Clear the curriculum table after materialization.

The sole difference is an existing squared body-forward command residual:
`body_forward_tracking_cost` on `twist`, weight0 in `control`, **constant-.25** in
`forward`. This cost is zero at matching nonzero velocity; it is not a motion
tax. Do not add a twist/yaw cost or change tracking standard deviations.
Stock linear/angular rewards remain2 with std sqrt(.1)/sqrt(.5).
The61D actor, raw14D unfiltered BAM targets, plant, randomization, delay/noise,
normalization, optimizer and mixed command sampler are identical between arms.

The parent achieves approximately.152/.106/.104m/s for straight.3 and two
signed-yaw.2 commands. A small forward-only penalty tests underspeed separately
from its direction-dependent yaw problem. Earlier forward escalation on another
parent, with an additional fixed twist cost, worsened yaw despite slight speed
gains. This is therefore the lowest constant dose on a different fresh-trained
parent, not repetition of that compound recipe. Improvement is a hypothesis;
one seed supports a bounded diagnostic, not robust curriculum graduation.

## Budget and numerical protocol

| Phase, each arm | Worlds × updates | Seed | Child cap |
| --- | ---: | ---: | ---: |
|Normalized ONNX smoke|64×5|1381|600s|
|Final-weight benchmark|256×10|1399|600s|
|Matched continuation|256×1000|1409|2100s|
|Both final999 matrices|8×240 per case|839/853/857|900s combined|

Both preflight arms complete before either learner. Each benchmark must satisfy
`elapsed_s *100 *1.5 <2040s`; do not widen caps after failed admission. Control
finishes before treatment; all GPU children and evaluation are sequential, with
the existing owned lease and15s termination grace. Preflight user service ceiling
2700s; learner/evaluation ceiling5700s. Retain initial plus updates0/250/500/750/999
and exact-source receipts. Each training arm executes6144000 transitions and
20000 minibatch optimizer steps; no extra duration or checkpoint selection.

Record actual weighted forward reward (including raw zero-weight control cost),
capped motor reward cost, separate named hip costs/stop exposure/range overrun,
live final curriculum, action-command dwell and adaptive KL/loss/LR/fall metrics.
Snapshots occur at the retained post-reward/pre-reset hooks, including terminal
worlds. Position and force have different simulation phases; telemetry is not
synchronized physical certification. No new observation pass or simulator
forward/sense operation is inserted.

Evaluate only each `model_999.pt` under the **unchanged full12-case protocol**:
three held-out seeds × idle/(.3,0)/(.2,-.2)/(.2,.2), eight worlds,240 first-episode
steps,60-step tracking startup exclusion. Speed MAE≤.03 idle/≤.10 moving,
yaw MAE≤.10, zero terminals/rated-speed exceedances, global pooled torque
utilization p99≤.60, all14 hard-range distances≥-.005rad, right-hip stop exposure
≤1%. Preserve the established aggregation; do not replace global torque pooling
with a per-joint gate or interpret it as every servo below.36Nm.
Keep full paired speed/yaw/load/power/thermal/range evidence, not just a mean score.
No gate weakening, reset laundering, intermediate fishing or automatic promotion.

## Host admission and exit

Only native100.100 integration checkout, exact clean pushed source and frozen
runtime. Preserve the existing DINO worker and all unrelated workloads. Recheck
its current PID before admission; a reboot invalidates the previous PID2021.
Protected AI services
remain inactive in both user/system scopes. Leave100.98/NX/FilmBrain untouched.
Require only retained DINO GPU occupancy, used≤6144MiB/free≥8192MiB/temp<75°C,
20% Torch allocation and single CPU threads. Run focused CPU regressions with
CUDA hidden before normalized64×5 smoke. Diagnose failures read-only first;
no packages/drivers/services repair or automatic retry.

After evaluation, retain hashes and paired gate decision, then stop for review.
A successful training process alone is not policy acceptance. Any new skill,
MP4, collector, publication, raw perception or physical motion remains separately
gated. Independent objective and parent-setup reviews informed this declaration;
implementation and GPU admission remain with one integration owner.

## Execution state

Prepared and source-tested:109 focused tests passed, then988 integrated/adjacent
CPU regressions passed with16 existing actuator-selector warnings (34.17s,
CUDA hidden, single-threaded). Shell syntax, verifier AST and experiment link
checks passed. Independent reviews covered the one-factor objective, exact
parent/final-curriculum restoration and operational admission checks. Initial
test-mock argument/params/sign errors were corrected and the full suite rerun;
the production objective was not altered to accommodate those fixture faults.

Historical21:13 native GPU admission was blocked (subsequently resolved):
the fresh October12 21:13 Beijing check found a host boot at20:34, protected user
`recomo-ai-mission-vllm.service` active (PID2048; engine3617 using9692MiB), and
preserved DINO now PID2029 (946MiB). Total used10656MiB/free5538MiB/temp50°C.
No Duck service is running; the native checkout remains clean at the preceding
`d0cba9c0` source. This violates the inactive-protected-service and free-memory
admission gates. Do not stop the restarted service or overlap training without
a fresh explicit GPU handoff. No learner, smoke, benchmark or held-out evaluation
for this new recipe had run at that snapshot. The subsequent explicit user
approval authorized a temporary pause of **only user vLLM**, the bounded Duck
experiment and restoration afterward; the completed evidence follows.

## Completed result: neither arm is accepted

The exact pushed source `3940e89f85c1e9ba3ef7e0b966dc7acc5ab42c86` ran on
native100.100. CPU preflight passed988 tests/16 existing selector warnings in
36.80s. Both normalized64×5 ONNX smokes passed and their exported files were
hash-authenticated. The runner's ancillary automatic-export hook emitted the
existing `Logger.logger_type` warning; the separate official normalized export
and CPU61-to14 smoke passed, so no export acceptance was inferred from that hook.
Smoke elapsed4.751/4.380s; benchmark7.754/7.640s, projecting1163.11/1146.06s
against the2040s admission limit. Both initial checkpoint hashes were identical
(`c1d8fa4f2cb1fec8367a754024ac4cb45528edec6c90204ff9160dc9e81649f6`);
native tensor checks verified exact parent models, normalizers and Adam state.

Control completed1000 updates in612.659s; forward completed1000 in615.134s.
Each consumed24000 vector steps/6144000 transitions and20000 minibatch updates.
All scheduled0/250/500/750/999 checkpoints plus initial weights are durable.
Final actor/critic normalizer counts43008000 and seventeen Adam counters140000
were verified in the actual checkpoint tensors. All2030 preflight/learner update
rows are finite with zero NaN terminations. Training falls1436/1654, first100
146/147 and last100164/166; these training resets are not hidden as held-out success.

Both final-only matrices have12/12 complete first episodes, zero held-out
terminals/rated-speed exceedances/hard-range violations/stop proximity, and
24/24 idle speed and yaw passes. Each still fails **all9 moving cases** through
combined tracking; moving yaw passes remain0/72. Neither endpoint is an accepted
teacher, adopted forward objective or new locomotion/obstacle/hop/ball capability.

| Measured quantity | Control | Forward |
| --- | ---: | ---: |
|Moving speed world passes|47/72|50/72|
|Moving yaw world passes|0/72|0/72|
|Straight speed world passes|1/24|2/24|
|Achieved forward speed, .30m/s straight|.176587|.183181|
|Straight speed MAE|.123418|.116823|
|Straight yaw MAE|.270002|.318589|
|Straight signed mean yaw, target0|+.011570|+.084639|
|Negative arc speed/yaw MAE|.074965/.276990|.067047/.311245|
|Negative arc signed mean yaw, target-.20|-.101437|-.052225|
|Positive arc speed/yaw MAE|.078202/.296206|.065951/.279215|
|Positive arc signed mean yaw, target+.20|+.121088|+.157478|
|Maximum per-case pooled motor utilization p99|.521110|.583660|
|Minimum all-joint hard-stop distance, rad|+.199689|+.234845|

The forward penalty improves mean speed error in all moving groups but worsens
straight and negative-turn yaw, and raises modeled motor p99 in every paired
moving case. Treatment's worst p99 leaves only.01634 margin below the unchanged
.60 gate. Mean modeled thermal proxy across12 cases rises.02532→.02707;
mean case power.94204→.94629W, with mixed per-command changes. Scalar per-command
power/thermal means average **three case scalars**, while per-world speed/yaw
means average24 values; never divide scalar sums by24. These are not measured
motor temperatures or continuous hardware torque certifications.

The reported p99 is pooled over each case's fourteen motor columns, eight worlds
and sampled steps. Its maximum across12 cases is not a newly pooled cross-case
p99 or a guarantee for every joint. Native verifier recomputed decisions from
all24 case records; the independent stdlib verifier and parallel JSON review
confirmed the same failure counts and tradeoff. The added raw forward penalty
averaged.024350 cost/-.0060875 weighted reward per step; actual reward activity
was confirmed rather than inferred from its configured weight. One matched seed
is a diagnostic, not a multi-seed causal generalization.

## Retention and service restoration

Native weights/evidence remain under
`artifacts/training/motor-range-forward-3940e89f-20261012` in the integration
checkout. Mac JSON-only evidence lives under
`artifacts/evaluations/motor-range-forward-3940e89f-20261012`; no duplicate weights
or ONNX binaries were pulled. The manifest authenticates59 JSON/JSONL payloads,
68736667 bytes. A separate local derived summary, local seal verification and
handoff receipts are retained without editing the sealed payloads.

| Artifact | SHA256 |
| --- | --- |
|Control final999|`936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`|
|Forward final999|`712841fe2f399ffd4273ff50679ea50bf975f1e903b7cd96f64903c53b7789a9`|
|Evaluation aggregate|`44df47f64c6dcba463066fa1c3f7e9be9be4fd1bae44a694432d708124e3d00b`|
|Native verification|`5c228d1bf3aa6ffe0cfb9139801606947adef3a4993028ce73b93db9e4ed40d0`|
|Retained manifest|`df068a2a48dbd17d776643f18bb300db7695a241df05b10ff0ce84edfdd25e49`|
|Local derived summary|`ea747b22fba9a456ffe3b3fe2bc97d9406a27154f228a9bacf3320f1880b8105`|
|Offline gait/yaw summary|`3ec1eda040dd060dcc63a2b93fa2fe6e190a50695ab7dde3b8fcb2c389a69ff6`|
|vLLM restoration receipt|`d542150ffd1f2bd60d3973fa07ceb096074257c4c789745f1e5c68b9e8bd239b`|

Preflight, matched-evaluation and handoff services all ended inactive/MainPID0/
Resultsuccess/ExecMainStatus0. They are retained, non-enabled user units, not
boot-time jobs. The handoff used a9000s outer cap and an unconditional
`ExecStopPost` restore guard; independently reviewed context/cgroup/lease checks
and three mock tests covering seven conflict cases verified refusal to restore
over a busy Duck lease, competing GPU process or invalid original state.
Original user vLLM was restored only after the GPU lease was free, no Duck unit
was running, and GPU occupancy was DINO2029 alone. At23:19:26 Beijing the normal
vLLM startup completed; its cgroup-owned engine120918 occupied9692MiB while DINO
2029 remained946MiB. Other three protected scopes stayed inactive. Peak Duck
occupancy2374MiB/temperature69°C stayed within the declared admission envelope.
No driver/package changes, unrelated service stops,100.98 changes or physical
motion occurred. The native source remains pinned at the trained revision;
later evidence-only fork commits are not substituted for that run identity.

## Next boundary

Do not stack the rejected forward treatment, relax motor/range gates or extend
this run. A small speed gain did not solve yaw and reduced load headroom.
The retained six straight yaw/gait traces have now received the bounded CPU-only
diagnosis using the existing stdlib helper, without another rollout or modifying
sealed evidence. Its29 focused tests passed; source-case hashes, per-step command
binding and per-world mean/MAE agreement were independently rechecked.

Across three straight seeds, mean absolute **per-world DC yaw** rises
.0546→.1082rad/s, while mean **within-world population standard deviation** rises
.3646→.4064rad/s. This differs from the signed cross-world mean+.0116→+.0846,
which can cancel opposing biases. Both DC bias and time variation remain relevant;
the treatment's three seed-level signed means are all positive. No-contact and
double-flight fractions are zero; one-contact fractions dominate at96.3–98.9%,
with mean horizontal foot slip about.016–.018m/s. Pooled head-yaw velocity versus
signed yaw-residual correlations are-.408/-.389/-.287 for control and
-.063/-.011/+.051 for forward. This is descriptive association, not a causal
head-motion finding; pooled correlations may include between-world covariance.
Per-world statistics and support timing caveats are retained in the47933-byte
`gait-summary.json`, outside the original59-payload seal, and copied back as a
native derived artifact. No-contact evidence from the retained sensor timing
does not prove force/position synchrony or physical support qualification.

Next, separately predeclare **one yaw-focused comparison** retaining motor and
bilateral qpos objectives and the complete unchanged held-out matrix; do not
carry forward the rejected forward cost or infer a fix from correlation alone.
Mean yaw, favorable contact statistics or slower motion cannot replace
instantaneous tracking MAE. Any new GPU job requires a fresh scoped handoff of
the now-restored vLLM workload; no automatic follow-on training or harder skill
is authorized by this completed diagnostic.
