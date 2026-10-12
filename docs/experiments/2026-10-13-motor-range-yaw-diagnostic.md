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

The retained sequential `microduck-yaw-4b9de3cc-matched-eval.service` is now
running, under `recomo-duck-yaw-4b9de3cc-handoff.service`. Run directory
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
