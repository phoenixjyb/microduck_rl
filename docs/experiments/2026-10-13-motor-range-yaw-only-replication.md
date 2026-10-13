# Yaw-only tracking: second matched seed, predeclared

Declared before GPU execution on `integration/upstream-2026-10-10`.
Protocol `integrated-motor-range-yaw-only-replication-v1`, implementation
`src/mjlab_microduck/integrated_motor_range_yaw_only_replication.py`.

## Question and fixed reference

The [first seed1447 contrast](2026-10-13-motor-range-yaw-only-diagnostic.md),
source `d71c30285ff3acbdfee12615548eaf2ea1a10975`, is sealed and rejected:
moving yaw MAE improves25–34% but yaw passes0/72 for both, speed32→29/72 and
straight0/24. No teacher/accepted skill is inherited from either endpoint.
The earlier same-recipe control at another seed1429 had speed50/72, so comparing
different treatments across these rounds is not a causal contrast.

Repeat **both** arms at the preselected second training seed1481. This measures
whether the direction-specific effect repeats, without changing the objective,
optimizer initialization, rollout budget, sampler or acceptance gates. No seed
sweep or retrospective selection: one new pair, evaluated only at final999.
Preserve the first seed regardless of what the second produces. Both paired
results, including speed/falls/load regressions, must remain visible.

Both arms strictly warm-load the same original exploratory parent, not either
new endpoint: `artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA256 `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Iteration999/env24000, normalizers43008000, seventeen Adam states140000 and LR
are restored exactly, then episode/learner clocks reset0. All seven materialized
final source curricula are unchanged; no reset-Adam or new teacher.

## Same one-callable contrast and budget

Control retains actual stock XYZ angular tracking, `2ab`; yawonly `2a`, with
`a=exp(-yaw_error²/.5)` and `b=exp(-XY_body_rate²/.5)`. Both weight+2 and
params stdsqrt(.5)/commandtwist. Positive range[0,2]; raw XY still validated,
same-step pre-reset telemetry and live function/params/weight checks unchanged.
Keep separate trunk-angular-.05 and whole-body momentum-.02, motor-1
(nominal.60Nm/soft.70/overload4), bilateral named hip-yaw qpos-4/margin.15rad,
forward weight0/read-only metric, all commands/events/DR/delays, raw61D actor,
raw14 unfiltered BAM targets and the frozen installed runtime. No MDP change.

| Phase per arm | Worlds × updates | Matched seed | Child cap |
| --- | ---: | ---: | ---: |
|Normalized ONNX smoke|64×5|1471|600s|
|Measured benchmark|256×10|1477|600s|
|Control/yawonly continuation|256×1000|1481|2100s|
|Final999 evaluation, both arms|8×240 per case|839/853/857|900s combined|

CPU tests and both smokes/benchmarks precede either learner. Projection
`elapsed*100*1.5<2040s`; no cap widening. Save initial plus0/250/500/750/999.
Each6144000 transitions/20000 minibatches, final normalizers49152000/Adam160000.
Control→yawonly→both final matrices is sequential, never overlapping GPU jobs.

The unchanged matrix has12 cases per arm, three held-out seeds ×
idle/(.3,0)/(.2,-.2)/(.2,.2), eight worlds240 first-episode steps; exclude60
startup steps from tracking only. Speed MAE≤.03 idle/≤.10 moving, yaw MAE≤.10,
no terminal/rated-speed exceedance, per-case pooled torque p99≤.60, all14
hard margins≥-.005rad and right-hip stop fraction≤1%. Original full combined
decision retained; nothing is accepted from a mean-only improvement.

## Predeclared interpretation, not a new acceptance gate

Record both seed1447 and1481 tables separately; do not pool away a failed
case or let a favorable seed erase the first round. Descriptive yaw-error
direction is repeated only if treatment MAE is strictly lower than its own
control in **each** of the three moving-command groups, in both training seeds.
Report paired speed MAE/pass counts by command, training falls (total and
first/last100), torque/power/thermal and range/stop metrics alongside it.
Call any speed/fall regression a regression; no post-hoc tolerance or benefit
label can waive it. The first seed already fails speed non-regression, so this
replication cannot establish all-round no-regression or qualify that endpoint.

Individual foundation decisions remain the unchanged full-case gate. Policy,
simulator and physical acceptance stay false; a diagnostic readiness label is
not multi-seed skill acceptance. After this pair, retain the measured decision
and choose one bounded next hypothesis on evidence, not more seed fishing,
stacking rejected endpoints, automatic longer training or harder-skill promotion.

## Shared host and restoration

Use100.100 native integration checkout at the exact clean tested/pushed revision.
Freshly validate DINO PID/exe/uid/GPU UUID and original active **user vLLM** plus
other three protected scopes inactive. Pause only that user vLLM; preserve DINO,
all unrelated workloads and100.98/NX/FilmBrain. During Duck require owned lease,
all four protected scopes inactive, used≤6144MiB/free≥8192MiB/temp<75°C,
Torch20%, CPUthreads1, memory8GiB/CPU200%. Gross training torque guard1.5
does not replace the held-out.60 gate. No driver/package changes.

Retained static/nonenabled services cap preflight45min, matched95min and outer
handoff150min, child grace15s. Existing exit guard restores only originally
active user vLLM after Duck exits, lease is free and DINO-only GPU proof holds,
then verifies cgroup GPU ownership. Unknown workloads cause refusal, not overlap.
No raw perception, actor expansion, physical motion, MP4 or publication.

## State

Prepared and source-tested:82 focused checks passed in76.30s; then1052
integrated/adjacent checks passed in36.44s with16 existing selector warnings,
CUDA hidden/single-threaded. A fixture initially marked the declared `yawonly`
mode invalid; it was corrected without changing the recipe. Full config equality
against the first-seed recipe permits only seed/experiment identity differences.
Independent source/declaration and ops-generator reviews found no blocker.
Native admission, normalized smokes, measured benchmarks and outcome pending.

## Native admission passed; matched learners active

Native source is exact clean `4693c2a6ad27ae58b78da3e314044f029871c667`, tested
and pushed before execution (code declaration7a00923c; subsequent documentation
clarification changes no recipe). Native CPU1052 checks passed in36.58s with16
existing warnings. Both official normalized ONNX exports passed finite61→14
CPU inference/hash validation; smokes0/0 falls and4.480/4.474s. Benchmarks took
7.652/7.811s with4/5 falls; conservative projections1147.779/1171.676s are below
2040s. All30 preflight updates passed positive angular reward/zero negative sample,
finite/zero-NaN, live stability/motor/bilateral/final-curriculum and command-dwell
guards. Native checks authenticate actual identical initial model, normalizer,
Adam and LR against original936a parent, initial SHA256
`d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`.
Preflight peak1710MiB/62°C. These are admission checks, not policy acceptance.

Preflight proof SHA256
`bb8b033533bd022b4e025f2103e23e3910a0e0f0b3509b69cd86394798921718`;
original service/GPU context SHA256
`9aee0a851e3e8c16c8fa62bb64758a7bbbc2fea2cad82bd88384d8d233197913`.
Closed JSON-only proof/phases/context are on Mac at
`artifacts/evaluations/motor-range-yawrep-4693c2a6-20261013-preflight`.
Local stdlib checks authenticate these hashes and all30 reward/safety rows.
Weights/ONNX stay native; no duplicate binaries fetched. Shell/Python syntax,
systemd verification,3 mock-only restore tests (including7 refusal cases), and
independent generated-ops review passed before launch.

Native run directory `artifacts/training/motor-range-yawrep-4693c2a6-20261013`.
The active sequence is `microduck-yawrep-4693c2a6-matched-eval.service`, under
`recomo-duck-yawrep-4693c2a6-handoff.service`: control→yawonly→both final999
matrices. An early89-update/2136-vector-step audit has finite rows, zero NaN
terminations/negative samples and121 training falls. Latest training torque
p99.713759/thermal proxy.063604; left/right hip stop exposure.00048828125/0.
This is below gross training guard1.5, not a pass of the held-out.60 motor gate.
CUDA-hidden CPU read authenticates actual finite control/model0 at iteration0/
env24, normalizers43014144 and seventeen Adam states140020, SHA256
`7c790b3caf92122f528f07594e11d8f70494b7dcdd328e6d29fdf512e646bab1`.

At09:22 Beijing both matched/outer units remain active, GPU1710MiB/67°C, DINO2029
retained, and all four protected scopes inactive during the authorized pause.
Native worktree stays clean at4693c2a6; do not change it or the jobs while active.
The scoped exit restoration guard is installed; original user-vLLM restoration,
final numerical decision and first/second-seed comparison remain **pending**.
No new capability, policy promotion or additional job is claimed from this early
checkpoint. Later evidence-only fork commits do not change the trained source.

## Completed pair and two-seed decision

Both1000-update learners and24 final999 held-out cases completed, control620.854s/
yawonly626.154s. Initial and0/250/500/750/999 actual checkpoints, finite tensors,
parent/source/clocks, normalizers49152000 and seventeen Adam states160000 are
authenticated; all2030 preflight/training rows finite, zero NaN terminations.
All three Duck units ended inactive/MainPID0/Resultsuccess/ExecMainStatus0.
No new policy or skill is qualified: both arms fail all9 moving tracking cases.

| Seed1481 quantity | Control | Yaw-only |
| --- | ---: | ---: |
|Moving speed/yaw world passes|41/72;0/72|21/72;0/72|
|Straight achieved vx, target.30|.173764|.146403|
|Straight speed/yaw MAE|.126236/.280615|.153597/.234292|
|Negative arc speed/yaw MAE|.080191/.282570|.102241/.224568|
|Positive arc speed/yaw MAE|.085140/.288139|.099711/.218226|
|Maximum per-case pooled torque p99|.561591|.590394|
|Minimum all-joint hard margin, rad|+.120896|+.209901|

All24 first episodes survived240 steps, idle speed/yaw24/24 each, no rated-speed
exceedance, hard-range violation or stop proximity. Training falls1539/1397;
first100141/122 and last100142/145. Yaw-only torque margin is only.009606 below
the modeled.60 gate; not a per-joint or physical safety certificate.

The predeclared **direction** criterion passes: paired yaw MAE decreases in all
three moving groups at both seeds1447 and1481 (first25–34%; second16–24%).
That is not general locomotion improvement. Speed worsens in all moving groups
at1481 (negative speed passes21→10/24, positive20→11/24); first seed also loses
positive-turn speed and total moving passes32→29/72. Every arm still has moving
yaw passes0/72 and straight speed0/24. Retain the mixed fall trend and all speed
regressions; do not pool away either seed or adopt/stack yaw-only.

Scalar mean power W/thermal proxy across three cases per command, control→yawonly:
idle.033121/.010857→.022534/.011447; straight1.447074/.031694→1.418251/.032035;
negative1.049700/.029804→1.004887/.029629; positive1.004648/.030474→.952490/.028079.
The torque table is the maximum across per-case pooled percentiles, not a new
cross-case percentile or mean. Proxies are not measured motor temperatures.

## Retention, independent checks and restoration

Native59-payload seal is69656457 bytes; portable verification recomputed failures,
world passes and group means. A separate hash-bound two-seed check authenticated
decision labels/nested flags and preserved all paired countermetrics. Independent
parallel review confirmed the full tradeoff. Mac JSON-only bundle is
`artifacts/evaluations/motor-range-yawrep-4693c2a6-20261013`; weights/ONNX stay native.

| Object | SHA256 |
| --- | --- |
|Control final999|`4b94c233e6bc72135768b51174c06a44f38a5a0ad485e5bf0f3bae51a3767f02`|
|Yaw-only final999|`1c8643194447111bf14eca0fb1266df79709962b62d5f53b71cf9919257af73b`|
|Aggregate|`c15811a2cf6cb2b3aab669d100a6a0a6a7afd13ada09e562a352bb9594b766f4`|
|Native verification|`ab34dbcd317cdeb2cc638f6622347791dc248c560408fcb149d40cc4bf92180e`|
|Manifest|`e1c95c6477d54d9830a5f7233b1cb178e532e9cf3559cbf158c3e7b8a697de81`|
|Local summary|`6b56244dff954086192bf2e93a4401b17553ce3d73b27cea5e789c18cddcf193`|
|Local verification|`cacaa5a98424b4dd4b95a4df490a45e7695c45ae8cffb6a94d335aff08254529`|
|Two-seed comparison|`0431e5f36be2ec2a302426271216ef64d6016a146a393f2af21141dbf42f43bb`|
|Episode reward-mass audit|`2e3ae23f3a19eb4c92b35c028cc3d9bf30c7c6bca2864afcdd3a985b9c964d2f`|
|Restoration receipt|`3a632f23c48159f7a89022c9f72371d85990a8b1ba2f8c6f504da4374d04b2e6`|

At09:43:53 Beijing, guard restored only original user vLLM after Duck exit/free
lease/DINO-only proof. Engine578733 uses9692MiB, DINO2029 remains946MiB and other
three protected scopes inactive. Duck peak2374MiB/68°C. No restart/reboot,
driver/library,100.98/NX/FilmBrain, physical motion, raw perception or MP4 changes.

## Next measured motion hypothesis

Stop yaw-kernel tuning. An audit of this owned Duck unit's last100 complete
episode-log blocks per arm (iterations900–999) shows action-rate weighted reward
control-1.094232/yawonly-1.168994, versus forward+1.220695/+1.182202, motor
-.058334/-.060403, bilateral range-.000561/-.001255 and head bias-.478737/-.478156.
These are means of **rounded mixed-command episode aggregates**, not per-world
or instantaneous costs; sampled exploration contributes to raw action differences.
They motivate testing, not proving, an attempt-tax mechanism. New data must
capture raw/actual weighted costs by command rather than infer behavior from
configured weights or these logger means alone.

Next is a separately bounded release-and-recovery of global action_rate_l2,
from the original936a parent and stock XYZ angular reward, never either rejected
new endpoint. Keep motor/range, other final source stages, actor/actions and
every held-out gate fixed; restore the original smoothness weight before final
evaluation. Predeclare/test/commit before the next guarded GPU handoff.
