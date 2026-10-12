# Fresh motor/range-aware integrated foundation: bounded declaration

The [completed fresh stock-curriculum baseline](2026-10-12-fresh-curriculum-foundation.md)
is rejected. All12 final cases survived their short first episodes but failed
motor and joint-range/stop gates; all9 moving cases failed combined tracking.
Both hip-yaw joints parked near their configured limits about97–98% of sampled
steps. This declaration adds a motor- and bilateral-qpos-aware fresh foundation,
not more duration on the rejected endpoint or a promotion to harder skills.

Protocol `integrated-fresh-motor-range-foundation-v1`; implementation
`src/mjlab_microduck/integrated_fresh_motor_range.py`. This is a combined-objective
recipe investigation, **not** a one-factor causal experiment. Matching the
earlier seed and budget allows descriptive comparisons but does not isolate
which added cost caused a change. Keep the old source, checkpoints and evidence
intact. No retrospective schedule/cap/gate adjustment is allowed.

## Frozen configuration and added objectives

Build on `integrated_fresh_foundation.prepare_config`: same plant/BAM, stock
reward stack,7 stock curricula, observation61D/action14D, normalization,
unfiltered/unclipped outputs, delays/noise, randomization, mixed forward/idle/
signed-yaw commands and PPO settings. Fresh weights, zero normalizer counts,
empty Adam and initial LR.001; no checkpoint load. No new pose, target height,
controller, raw perception, angular observation or actor interface is introduced.

Add only existing nonnegative MDP costs and negative weights:

| Objective | Initial | After250 updates | After1000 | After2000 |
| --- | ---: | ---: | ---: | ---: |
|`motor_load`|0|-.25|-.50|-1|
|`joint_limit`|-.25|-1|-2|-4|

The motor cost keeps nominal stall torque.60Nm, soft threshold.70 and squared
overload gain4. This is not a continuous physical torque rating or temperature
measurement. Qpos cost sums `.15rad` proximity penalties on exactly
`left_hip_yaw` and `right_hip_yaw`, resolved by regex
`^(left|right)_hip_yaw$`; never use numeric servo indices or clip command targets.
Small early qpos pressure discourages learning to park at a stop; motor pressure
is initially zero to leave exploration available. The hypothesis is explicitly
unproven, and standing/no-fall behavior cannot satisfy moving tracking gates.

Read the live environment clock **before every action**. New stage weights take
effect strictly when `common_step_counter > iteration*24` (not `>=`); apply to
live reward-manager terms, not copied config objects. Existing stock schedules
retain their reset-aware manager semantics and timings. After2000 the complete
new weights consolidate for about4000 updates. This is a fixed diagnostic
schedule, not evidence-gated automatic skill progression.

The five-update smoke compresses only these two added schedules to thresholds
1/2/3; the ten-update benchmark uses their final weights. The foundation uses
the table unchanged. Read-only metric snapshots clone both named qpos/cost
columns before autoreset; per-step checks reconcile the summed registered cost
and actual nonpositive weighted reward, including terminal worlds. Motor reward
is checked against the retained pre-reset MotorStepStream forces. Log raw cost
even at zero weight, separate left/right costs, stop exposure and maximum range
overrun, plus existing KL/loss/reward/fall/motor/dwell/live-curriculum telemetry.
Position and force sampling phases differ; their association is not causal proof.

## Budgets, host gates and stop rules

| Phase | Worlds x updates | Seed | Process cap |
| --- | ---: | ---: | ---: |
|Exact-source integrated/adjacent CPU regressions|no GPU|n/a|before smoke|
|Normalized finite61-to14 ONNX smoke|64x5|821|600s|
|Final-weight benchmark|256x10|823|600s|
|Fresh foundation|256x6000|1327|10800s|
|Final5999 held-out matrix|8x240 per case|839/853/857|900s total|

The prior benchmark projected8855s against9000s; additional bilateral per-step
telemetry makes that previous margin unsuitable. This recipe predeclares a
10800s cap **before its benchmark**, with measured projection
`elapsed *600 *1.5 <10740s`. Do not widen this on failed admission. Retained
preflight service cap1300s; sequential foundation/evaluation service11900s,
each child under `timeout` with15s kill grace. No automatic learner retry.

Execute only on100.100's native integration checkout with exact clean pushed
source and frozen installed pins. Smoke/benchmark must authenticate fresh
initialization, actual Adam updates, normalization, every command bucket, durable
checkpoints, finite telemetry, nonpositive applied costs and normalized ONNX.
If either fails, stop before the learner. Require an idle Duck GPU and only the
preserved DINO PID2021; lease-lock every GPU child. Preserve all unrelated work
and drivers/packages. Keep protected AI services inactive in user/system scopes;
do not touch100.98, NX or FilmBrain. Used<=6144MiB, free>=8192MiB, temperature<75C,
Torch process allocation20%, CPU thread limits1. Retain every250-update checkpoint
plus initial.pt and final5999. Existing finite/NaN and gross motor abort guards
stay unchanged; early fresh-policy falls do not waive held-out stability gates.

## Unchanged acceptance and follow-on boundary

Evaluate **only final5999**. Same held-out factory: play=True, stock delays/noise,
seeds839/853/857, idle,(.3,0),(.2,-.2),(.2,.2),8 worlds each,240 first-episode
steps and60-step startup exclusion only for tracking; no reset laundering or
intermediate-checkpoint fishing. Complete unchanged decision: per-world speed
MAE<=.03 idle/<=.10 moving, yaw MAE<=.10, zero terminals/rated-speed exceedance,
torque utilization p99<=.60, all14 joints' range distance>=-.005rad and right-hip
.05rad stop exposure<=1%. Left-hip exposure is also reported separately; the
legacy evaluator is not silently replaced with a new acceptance threshold.

Training completion or improved mean tracking is not acceptance. A full pass
permits consideration of a separately declared repeat/retention round, not
automatic obstacle/hop/football progression. Rejection closes this exact recipe;
retain and independently recompute the complete decision and trace summaries.
No MP4, publication, raw perception, physical motion, protected-service restore
or additional learner is authorized by this declaration alone.

## Source checks and native execution record

Independent read-only review preserved the fresh initialization/evaluator and
identified a telemetry mismatch for rare motor utilization above2: the stream
cost is uncapped but the reward cost is capped. The new observer now reports
the actual capped raw reward cost without changing the shared legacy observer;
a utilization3 regression distinguishes stream30.16 from reward10.76.

The affected integrated/adjacent CPU suite passed924 tests with16 existing
selector warnings in138.31s before that narrow telemetry fix. Post-fix fresh
recipe/harness checks passed25 tests with2 selector warnings in56.55s; CUDA was
hidden throughout. Documentation link targets and whitespace checks passed.
Native exact-source regressions, smoke, benchmark admission and full final
evaluation remain separate gates; record their source, unique directory,
service names and evidence below when admitted.

### Native admission and learner launch: October12

Exact source `d0cba9c0735fa49fc86b42383e756a5fbd6c70a5` was committed/pushed,
then synchronized only after an idle-GPU, clean-repository and no-Duck-service
check. Native exact-source regressions passed925/925 with16 selector warnings
in29.92s. Preflight service
`microduck-fresh-motor-range-d0cba9c0-preflight.service` completed successfully
with MainPID0/Resultsuccess/ExecMainStatus0.

Smoke completed5 updates in5.562860s,201 training falls, zero NaNs; finite
normalized61-to14 ONNX hash
`54c28564bfb311f129610ceed4e85903f8c55d994f02d26c8d2f2407f1c400c4`.
Benchmark completed10 updates in10.283680s,1658 training falls, zero NaNs.
Its conservative projection9255.312s is below the predeclared10740s gate.
All four actual command buckets, live initial/final new cost weights,
nonpositive weighted costs, bilateral named costs/exposures and unchanged
initial stock curriculum profile were authenticated from retained update rows.

The CUDA-hidden admission verifier loaded initial/final checkpoint tensors and
verified empty initial Adam, zero normalizer counts, initial LR.001,17 final
Adam parameter states with100/200 actual steps, and final normalizer counts
7680/61440 for smoke/benchmark. Checkpoint SHA-256:

|Object|SHA-256|
|---|---|
|Smoke final4|`f43721b6fe399edfd369b02745263a9b1c221444cc4ed63b7f70bc66814aedc8`|
|Benchmark final9|`9b8af5d7a0fe46b24b14a9a89ee39382f7fd8ed28e7e293794dba7dac2027ae8`|
|Admission proof|`a056f67b48cf20d88dccd83d5cef16a01f0d97c475f028993c70eff87fb2afc5`|

Retained native root:
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/fresh-motor-range-d0cba9c0-20261012`.
Admission proof is also retained on Mac in
`artifacts/evaluations/fresh-motor-range-d0cba9c0-20261012/preflight-verified.json`;
no weights/ONNX were duplicated. Native source remains detached at the exact
training commit while later documentation commits advance the fork branch.

The sequential retained service
`microduck-fresh-motor-range-d0cba9c0-foundation-eval.service` was launched:
6000 foundation updates, then only final5999's12 held-out cases, no follow-on
learner. Initial live check reached update89/6000 (2136 environment steps),
with finite optimizer telemetry20 minibatches/update, no NaN termination or
failure file, and durable initial.pt/model_0.pt. At this early stage motor
weight remains0 and qpos weight-.25 as declared; measured raw motor cost is
nonzero and bilateral weighted qpos cost is negative. This is optimizer activity,
not a new skill or a numerical pass. Early training falls and range exposure
remain expected diagnostics, not waived acceptance criteria.

At that check GPU used1717MiB/free14476MiB/59C, with only retained DINO2021
and the authorized Duck process2447653. All four protected service scopes
remained inactive. The next gate is completed training, durable final5999 and
the full unchanged numerical matrix; do not alter the active job or synchronize
its checkout while it runs. That launch snapshot claimed no result or promotion.

## Completed run: motor/range gates pass, foundation still rejected

By approximately16:35 Beijing both retained services had exited successfully
(inactive/MainPID0/Resultsuccess/ExecMainStatus0), and only preserved DINO2021
remained on the GPU. Training completed6000 updates in3769.136s (62m49s),
144,000 environment steps,36,864,000 transitions and120,000 optimizer
minibatches. All25 scheduled foundation checkpoints plus smoke/benchmark
model_0 and final checkpoints were CPU-authenticated for finite tensors,
iteration/environment counters, normalizer counts and actual Adam state/steps.
Fresh initial foundation bytes match the earlier fresh baseline's initialization;
no parent checkpoint was loaded. Final actor/critic counts are36,864,000 and
all17 Adam parameter-state clocks are120,000.

All6015 smoke/benchmark/foundation update records were finite, with zero NaN
terminations. The verifier reproduced raw FP32 KL summary reductions and
minibatch LR timing, actual command dwell, every strict-step motor/qpos stage,
cost/reward signs, bilateral named-cost sums and the final stock curriculum
endpoints. Training falls totaled73,352; first100 updates14,884 and last100
152, each interval614,400 transitions. Compared to baseline72,994/15,099/131,
this is not evidence of improved whole-episode stability or a controlled causal
result. Training torque p99 could still exceed the strict held-out.60 threshold;
the final update was.726107. No training metric replaced the held-out decision.

### Final-only matrix and baseline comparison

All12 final5999 cases completed240 first-episode steps across96 worlds with
zero terminals, falls or rated-speed exceedance. **All12 cases passed the
existing motor-envelope and joint-range/stop gates.** Minimum hard-range
distance across all joints/cases was.208368rad; every held-out hard-range
violation and.05rad stop-proximity fraction was zero. Maximum case-level global
torque utilization p99 was.578503 versus.60. This is a meaningful simulated
envelope improvement, not hardware safety or complete locomotion acceptance.

The full unchanged decision remains `foundation-not-ready`:10 tracking flags
across10 unique cases (all9 moving cases plus seed853 idle). There are12 cases,
not10 total evaluations. Per-world idle speed passes24/24 and yaw23/24;
moving speed31/72 and yaw0/72. Seed853 world2's idle yaw MAE.102167 exceeds.10;
its speed MAE.027925 passes. No threshold or evaluator was changed to rescue it.

Each row below pools three seeds/24 worlds. MAE units are m/s and rad/s;
measured responses retain their signs. The earlier baseline is the separate
[fresh stock-curriculum run](2026-10-12-fresh-curriculum-foundation.md).

|Command vx / yaw|Speed MAE baseline → new|Yaw MAE baseline → new|Achieved vx baseline → new|Signed achieved yaw baseline → new|Max global torque p99 baseline → new|
|---|---:|---:|---:|---:|---:|
|0 /0|.001150 → .002201|.008952 → .011109|.000180 → .001019|.001519 → -.000470|1.067539 → .411581|
|.30 /0|.129703 → .147940|.217597 → .271535|.170297 → .152060|.016879 → -.049079|1.067539 → .524069|
|.20 /-.20|.080758 → .093839|.255557 → .311046|.119242 → .106184|-.013998 → -.143670|1.067539 → .578503|
|.20 /+.20|.084033 → .096448|.289013 → .313767|.115972 → .103552|.010678 → .032568|1.067539 → .565574|

The new recipe trades reduced simulated load/limit exposure for worse mean
tracking, rather than adding a universally better gait. Negative-turn signed
mean response is closer to its target, but its absolute error remains high;
positive turning is still weak. Distinguish persistent bias from oscillatory
error before selecting the next tracking objective. Moving speed world passes
fell44→31, while moving yaw passes remained0. Standing/no-fall short rollouts
cannot unlock stop/start, obstacle, hopping, football or a video.

Power and thermal-load proxies are scalar means **already pooled across each
case's worlds**; pool them over three seeds, not divide again by24:

|Command|Mean absolute mechanical power W baseline → new|Mean thermal-load proxy baseline → new|
|---|---:|---:|
|Idle|.150325 → .092576|.170991 → .012219|
|Straight|1.505598 → 1.346747|.181906 → .030576|
|Negative arc|1.060346 → .929881|.178844 → .029575|
|Positive arc|1.029255 → .897988|.178820 → .028391|

The highest per-joint force p99 is.548502Nm at the left knee in seed853's
positive arc, while the existing global percentile pools all14 servo columns.
Thus passing that global gate does not mean every joint's own force percentile
is below.36Nm. No per-joint criterion was silently added/removed or physical
safety certificate inferred. Positions and forces are sampled at different
phases; retain the result's explicit nonsynchronous-position/force caveat.
Both added objectives changed together, so these differences do not establish
which cost caused the gains or regressions.

### Sealed retention and next boundary

The CPU-only native verifier authenticated all declared checkpoints, source,
pins, receipts,6000-update learner and12 standalone cases, recomputed the full
canonical decision and three straight-gait summaries, then sealed the evidence
once. Independent stdlib-only Mac reception verified33 JSON/JSONL payloads
(51,646,376 bytes), all sizes/hashes, a separately implemented numerical gate
and all three gait summaries. Mac `verified-manifest.json` is a local derived
receipt outside the original native seal. No weights/ONNX were duplicated.

|Object|SHA-256|
|---|---|
|Final5999 checkpoint|`811b38b530e508d08967dba0ff194cf14fddb35011b9a4878a04f71a5b804190`|
|Evaluation aggregate|`8be416b392f40ff3de8e5304e5bac1ee4b841b6169bfbd7436f34699d2d45b2c`|
|Straight-gait summary|`81111cd9628183ffa8dc9bbc68eea54a7a57a0139104a5f4062645c7666f7823`|
|Native verification|`4cd3a7590b02c36ab4fa96a01624c89fbaa19c776f57ebfcd9c31cc8645fc7ea`|
|Sealed33-payload manifest|`1a63e267fe25103ab611a1e0dc188302f9db5fef081e4ecb349c59446490977b`|

Observed peak GPU2073MiB/62C; post-proof969MiB/free15225MiB/49C, DINO2021 only.
Both Duck services were inactive and both protected AI services remained
inactive in user/system scopes. No new learner, collector rollout, video,
policy publication, protected-service restoration or physical motion followed.

The next separately predeclared learning question is **tracking/response with
the current motor and bilateral range objectives retained**, not weakening the
first passing envelope gates or adopting this rejected endpoint as a teacher.
Read-only signed-response/gait/reward-mass diagnosis should identify one small
tracking factor before another bounded control/treatment experiment. The
[parallel transition preparation](2026-10-12-integrated-command-transitions.md)
is source-tested but runtime-gated; its API review also records unsynchronized
metric/sensor timing and missing compiled-scene binding. No automatic follow-on
GPU job is authorized by this closeout.
