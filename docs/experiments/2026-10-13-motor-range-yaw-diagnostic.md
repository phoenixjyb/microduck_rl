# Motor/range-retaining yaw decoupling: matched declaration

Declared before GPU execution on the `integration/upstream-2026-10-10` fork.
Protocol `integrated-motor-range-yaw-decoupling-v1`; implementation
`src/mjlab_microduck/integrated_motor_range_yaw.py`.

## Parent and one factor

The [completed forward comparison](2026-10-12-motor-range-forward-diagnostic.md)
is rejected. Use its **control** final999 as an explicitly unaccepted exploratory
parent, never its rejected forward treatment or an accepted teacher:
native `artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA256 `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Parent iteration999/environment24000, actor/critic normalizers43008000,
seventeen Adam states at140000. Both arms strictly restore identical model,
normalizer, optimizer and LR state, then reset episode and learner clocks to0.

Retain motor weight-1 (nominal.60Nm, soft threshold.70, overload gain4),
bilateral named hip-yaw qpos weight-4/margin.15rad and all seven materialized
final stock curriculum values. The source curricula finished during the earlier
6000-update fresh run; do not infer their final configuration from this parent's
new24000-step continuation clock. Keep the actual control parent's zero-weight
forward term and read-only metric, not forward pressure. No twist/yaw-L1/EMA
cost, yaw-width schedule, head sampler, observation, plant or optimizer change.
Raw61D actor/14D unfiltered/unclipped BAM targets and all randomization/delays
remain unchanged.

Control keeps the **actual stock angular callable**, weight+2 and
`std=sqrt(.5)`. Treatment replaces only that reward's aggregation with the
existing `decoupled_body_angular_tracking`, yaw/roll-pitch variances.5/.5 and
weight+2 throughout. For `a=exp(-yaw_error²/.5)` and
`b=exp(-(roll_rate²+pitch_rate²)/.5)`, control is `2ab`, treatment `2(a+b-1)`.
Matching maxima and each single-error axis are preserved, but simultaneous
errors receive a stronger penalty: difference `-2(1-a)(1-b)`. Preserve signed
negative rewards; never clip them. This is **one aggregation-form factor**, not
a claim that only the yaw derivative or reward offset changes.

## Why this bounded hypothesis

The current parent passes all12 held-out motor/range/stability subsets but
moving yaw0/72; straight yaw MAE.2700rad/s despite signed mean+.0116.
Per-world absolute DC yaw.0546 and temporal std.3646 show substantial time
variation as well as bias. Contact/head correlations are descriptive, not proof
of an actuator, sensor or head-motion cause.

An earlier decoupling contrast on a different, compound continuation parent
improved yaw MAE in all three moving groups and yaw world passes0→15/72, while
retaining motor/range passes; it still failed every moving case and slightly
worsened straight speed. In contrast, earlier yaw-width tightening, instantaneous
L1 and EMA-bias comparisons worsened their yaw outcomes. This tests the reward
form on the newly fresh-trained bilateral/motor-aware basin, without inheriting
those compound forward/twist costs. It is not adoption of the old rejected
recipe, a guaranteed cure, or a scalar-dose escalation. One matched training
seed supports a diagnostic only; stricter instantaneous yaw gates stay fixed.

## Fixed budgets and evidence

| Phase, each arm | Worlds × updates | Seed | Child cap |
| --- | ---: | ---: | ---: |
|Normalized ONNX smoke|64×5|1423|600s|
|Measured benchmark|256×10|1427|600s|
|Matched continuation|256×1000|1429|2100s|
|Both final999 held-out matrices|8×240 per case|839/853/857|900s combined|

Both arms' CPU checks/smokes/benchmarks precede either learner. Benchmark
projection `elapsed *100 *1.5 <2040s`; no retrospective cap widening.
Control→decoupled→evaluation is one sequential retained service, with15s child
termination grace,45min preflight/95min matched ceilings and a150min outer
handoff safeguard. Save initial and0/250/500/750/999 checkpoints. Each learner
executes6144000 transitions/20000 minibatches; final normalizer count49152000
and Adam step160000. Evaluate final999 only, no checkpoint fishing or extra rounds.

Read-only post-reward/pre-reset snapshots capture raw squared yaw/roll-pitch
errors and actual signed reward, including terminal worlds. Per-step live
function/params/weight identity, same-step snapshot and numerical formula checks
fail closed; record negative reward sample fraction, angular components, capped
motor cost, separate bilateral costs/ranges, command dwell, adaptive KL/loss/LR,
falls and health. No extra observation computation or simulator forward/sense.

The unchanged12-case factory uses idle/(.3,0)/(.2,-.2)/(.2,.2),8 worlds each,
240 first-episode steps,60-step tracking startup exclusion. Per-world speed
MAE≤.03 idle/≤.10 moving, yaw MAE≤.10, no terminal/rated-speed exceedance,
per-case pooled torque utilization p99≤.60, all14 joint margins≥-.005rad and
right-hip stop fraction≤1%. Keep the aggregation and every gate unchanged;
pooled torque is not a per-joint physical certification. Retain paired speed,
yaw, modeled load/power/thermal and range evidence even if one metric improves.

## Shared GPU handoff and exit

Only100.100 native integration checkout, clean exact tested/pushed revision,
frozen installed runtime. Recheck retained DINO PID/exe/owner and GPU UUID before
execution. Preserve it and all unrelated workloads; leave100.98/NX/FilmBrain
untouched. Use the established same-scope temporary **user vLLM only** pause and
restoration workflow. Validate/save its original active state and other three
protected scopes inactive before pausing. While Duck runs require all protected
scopes inactive, owned GPU lease, used≤6144MiB/free≥8192MiB/temp<75°C,
Torch allocation20%, CPU threads1 and8GiB/200% service limits.

Restore only originally active user vLLM after every Duck GPU child exits and
the lease plus DINO-only GPU proof passes; verify cgroup-owned GPU occupancy.
The outer `ExecStopPost` refuses overlapping restoration if another job remains.
No driver/package repair, raw perception, physical motion, video, publication,
or automatic obstacle/hop/football promotion. Diagnose errors read-only before
fixes. After the final numerical decision, retain exact hashes, commit/push
evidence and stop for review; successful optimizer activity alone is not skill
acceptance.

## State

Prepared and source-tested:73 focused tests passed (two existing selector
warnings), then1000 integrated/adjacent tests passed in36.61s with16 existing
warnings, CUDA hidden/single-threaded. The first test fixtures had old parent
counters/callable assumptions and an incomplete manager lookup; those fixtures
were corrected and both suites rerun without changing the reward to fit them.
Independent objective and installed-runtime hook/formula reviews informed the
single-factor choice.

Native preflight has passed at exact clean trained source
`4b9de3cc258af2528a526d30e7ed81b7734c12b2`:1000 tests/16 existing warnings in
35.76s. Both official normalized ONNX exports passed finite61-to14 CPU inference
and hash checks. Both initial checkpoints have identical SHA256
`d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`;
actual parent actor/critic/normalizer/Adam tensors and counters were authenticated.
Smokes took4.422/4.385s and benchmarks7.681/7.617s, with conservative projections
1152.09/1142.58s below2040s. Smokes had0/0 falls, benchmarks15/8, and all30
preflight update rows were finite with zero NaN terminations/600 optimizer
minibatches. Treatment's actual negative angular sample fractions were40.495%
and41.144%; control0%. No signed negatives were clipped, forward pressure stayed0
and all motor/bilateral/live-final-curriculum guards passed. These short training
fall/load samples do not constitute held-out acceptance.

Native proof SHA256
`1c63ae094c6f817b01a7be3a4867428f309ca7c2df2891915625ac63537c4753`.
Closed preflight JSON-only evidence is retained on Mac under
`artifacts/evaluations/motor-range-yaw-4b9de3cc-20261013-preflight`;
models and ONNX binaries remain native only.

Historical launch: the retained sequential `microduck-yaw-4b9de3cc-matched-eval.service`
ran under `recomo-duck-yaw-4b9de3cc-handoff.service`. Run directory
`artifacts/training/motor-range-yaw-4b9de3cc-20261013`. At the23:56 Beijing
snapshot control had83/1000 updates and durable model0, with finite metrics,
zero NaN terminations, GPU1710MiB/60°C, DINO2029 retained and all four protected
scopes inactive during the authorized pause. Control→decoupled→both final999
matrices remains the declared sequence. The outer restoration guard is installed
and the initial service/GPU context is durable; vLLM restoration is **pending**
until Duck finishes or aborts safely. Do not fast-forward/change the active
native worktree or either job. Source/data/model acceptance and final gate
decision are still pending; no new capability or policy promotion is claimed.
The subsequent monitor reached287/1000 control updates/6888 vector steps with
durable model250 and model0; latest NaN terminations0, torque utilization
p99.718869 and modeled thermal proxy.062424. Training p99 is not the held-out
.60 gate; its predeclared gross abort guard is1.5. No gate was changed or waived.

## Completed result and rejection

The October13 morning check found all three Duck services inactive/MainPID0/
Resultsuccess/ExecMainStatus0, with no failure receipt. Both1000-update learners
and all24 final-only held-out cases completed. Control took617.151s, decoupled
624.817s; each24000 vector steps/6144000 transitions/20000 optimizer minibatches.
All cadence0/250/500/750/999 plus initial checkpoints were authenticated against
actual finite tensors, parent state, clocks, normalizers and Adam counters.
Final normalizer counts49152000 and seventeen Adam counters160000. All2030
preflight/learner rows are finite; zero NaN terminations. Native source remains
`4b9de3cc`, separate from later evidence-only fork commits.

Both policies are **rejected**: all9 moving cases fail combined tracking, moving
yaw passes0/72 for both. All12 cases per arm survived240 first-episode steps;
zero terminals, rated-speed exceedance, hard-range violations or stop proximity.
Idle speed/yaw passes24/24 each. No video, harder skill or accepted teacher.

| Quantity | Control | Decoupled |
| --- | ---: | ---: |
|Moving speed world passes|50/72|32/72|
|Moving yaw world passes|0/72|0/72|
|Straight achieved speed, target.30m/s|.178028|.155494|
|Straight speed/yaw MAE|.121973/.283053|.144506/.213446|
|Straight signed mean yaw, target0|+.143200|+.050822|
|Negative arc speed/yaw MAE|.068011/.297309|.089751/.237395|
|Negative arc signed mean yaw, target-.20|+.015099|-.088932|
|Positive arc speed/yaw MAE|.063466/.250642|.095910/.214901|
|Positive arc signed mean yaw, target+.20|+.227353|+.184099|
|Maximum per-case pooled torque utilization p99|.599540|.557033|
|Minimum all-joint hard-stop distance, rad|+.153735|+.130572|

Yaw MAE falls14–25% across moving groups and negative-turn mean sign becomes
correct, but speed regresses in every group; there is no yaw-world acceptance
advancement. Do not adopt/stack decoupling or infer a successful curriculum from
the partial yaw signal. Control's worst torque margin is only.0004595 below.60;
the p99 remains a per-case motor/world/time pooled statistic, not each motor's
limit or a cross-case re-pooled percentile. Gate passing is not physical safety.

Scalar mean mechanical power/thermal by command (control→decoupled, three
case scalars each): idle.067374W/.015649→.047085W/.011511;
straight1.452882W/.034945→1.504080W/.033195;
negative1.095586W/.033493→1.123255W/.031276;
positive1.120874W/.034499→1.020515W/.030809. Do not report maxima as these means,
divide by24 worlds, or call the modeled proxy measured motor temperature.

Training total falls1531/1125; first100137/133 and last100181/72. Treatment
negative angular sample fraction32.538% across6.144M samples, versus control0;
mean angular rewards.614707/.380678. Mean squared yaw errors.566609/.422007,
roll/pitch rates.820618/.630519. Reduced training falls/load/error are descriptive,
not replacements for the failed held-out tracking gates or evidence for longer
training without a new declaration.

## Retention and restored original workload

Native weights/JSON stay at the declared run path; Mac JSON-only evidence is
`artifacts/evaluations/motor-range-yaw-4b9de3cc-20261013`. No duplicate weights or
ONNX binaries were fetched. The sealed manifest authenticates59 JSON/JSONL
payloads,69497154 bytes. Independent stdlib verification recomputed case identities,
all gate flags, per-world pass counts and aggregates; a parallel numerical review
confirmed the tradeoff. Local derived summary/seal checks and restoration receipts
are retained separately without modifying sealed payloads.

| Object | SHA256 |
| --- | --- |
|Control final999|`0f30c53d4027684320d0781244664e9572016a0cf512a7bd49a623010ca53013`|
|Decoupled final999|`b994505e9b24fb68b2150e77414e7a7b12d899d0349bc2487780c1e4c0ac4c65`|
|Full aggregate|`06371ae8d756d3ea3520b1a0ffb836b12171fab03807d16c1a56a77b71077f8d`|
|Native verification|`197e8d8af459aed1d5f60c50bc778f798441cf4b44767ff20a35d8c9810f7747`|
|Sealed manifest|`41ec902f42f1e8058bb1bbc41b963288f2a17e0b59610fdfbb48482c9bd9327f`|
|Local derived summary|`d728c7eeeaa28bc9463b87aa981944ed91520189c4737cb21ad513f546bf0115`|
|Restoration receipt|`fb371920592a784de9caa5f4d89340725f6efa1cfc74f84e793b5f999389d9c7`|

Peak Duck occupancy2374MiB/68°C, below declared limits. At00:18:30 Beijing the
guard restored only originally active **user vLLM**, after free lease/no Duck
service/DINO-only GPU proof. Its new cgroup-owned engine167678 used9692MiB;
DINO2029 remained946MiB; other three protected scopes remained inactive.
The07:51 live check confirms restored vLLM and no Duck GPU process. No other
workload, driver, library,100.98, raw perception or physical robot was changed.

## Next bounded question

Do not repeat the rejected aggregation or extend it by default. Before the next
learner, inspect whether the stock angular tracking's roll/pitch factor duplicates
the explicit trunk angular-velocity/momentum regularization and discourages
forward gait. A yaw-only tracking inclusion contrast may isolate that question
while preserving those separate stability costs and every original gate. It
would need its own tested declaration, original exploratory parent rather than
either rejected new endpoint, and fresh guarded handoff. The next result must
be judged on **both** speed and yaw; quieter/slower walking is not success.
