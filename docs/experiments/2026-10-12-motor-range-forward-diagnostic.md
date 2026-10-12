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

Native GPU admission is **blocked**:
the fresh October12 21:13 Beijing check found a host boot at20:34, protected user
`recomo-ai-mission-vllm.service` active (PID2048; engine3617 using9692MiB), and
preserved DINO now PID2029 (946MiB). Total used10656MiB/free5538MiB/temp50°C.
No Duck service is running; the native checkout remains clean at the preceding
`d0cba9c0` source. This violates the inactive-protected-service and free-memory
admission gates. Do not stop the restarted service or overlap training without
a fresh explicit GPU handoff. No learner, smoke, benchmark or held-out evaluation
for this new recipe has run. No new capability is claimed by this declaration.
