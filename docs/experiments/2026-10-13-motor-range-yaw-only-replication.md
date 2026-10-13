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
