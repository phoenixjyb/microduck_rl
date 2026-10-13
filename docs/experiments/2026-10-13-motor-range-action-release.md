# Action-smoothness release and recovery: matched motion diagnostic

Predeclared before GPU execution on `integration/upstream-2026-10-10`.
Protocol `integrated-motor-range-action-release-v1`; implementation
`src/mjlab_microduck/integrated_motor_range_action_release.py`.

## Measured question, not a presumed cause

The [two-seed yaw-only contrast](2026-10-13-motor-range-yaw-only-replication.md)
is rejected: moving yaw passes remain0/72, with speed regression despite lower
yaw MAE. Do not carry either treatment endpoint or the yaw-only kernel forward.
In the second pair, rounded mixed-command episode action-rate reward mass was
about-1.09/-1.17, comparable to forward-tracking+1.22/+1.18 and larger than
motor/range costs. This includes sampled exploration, not just gait motion;
it does not establish that smoothness caused underspeed.

Test one bounded hypothesis: does temporary release of the raw action-difference
cost improve moving tracking **after the original smoothness coefficient is
restored**? Both arms use stock XYZ angular tracking. Repeat a fresh matched
control so source, instrumentation, initialization and seed are paired; do not
compare a new learner causally to a previous seed's control.

## Fixed parent, one-factor schedule

Both arms warm-load the original exploratory, unaccepted parent:
`artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA256 `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Restore actual actor/critic tensors, normalizers43008000, seventeen Adam
states140000 and LR, then reset learner/episode clocks0. Parent iteration999/
env24000 is checked. This is not an accepted teacher or a reset-Adam experiment.

Materialize all seven inherited curricula at their final source conditions;
then reintroduce **only** `action_rate_l2`'s coefficient schedule. Control stays-1.
Release starts-.1, becomes-.4 strictly after step250×24, and-1 strictly after
step500×24. The inherited helper uses `step > boundary`: boundary steps retain
the preceding value. Update251 and501 therefore each contain one old-weight
step and23 new-weight steps. Updates502–1000 are fully recovered at-1.
This is a clocked causal diagnostic, not a phase-qualified curriculum accepted
for harder skills. No adaptive extension, checkpoint selection or cap widening.

Stock angular reward remains `2 exp(-(yaw_error²+XY_body_rate²)/.5)` in both arms.
Keep linear tracking weight2/stdsqrt(.1), trunk-angular-.05, whole-body
momentum-.02, motor-1 (nominal.60Nm/soft.70/overload4), bilateral named hip-yaw
qpos-4/margin.15rad and forward cost weight0. Preserve all commands/events/DR,
delays, simulator, optimizer, raw61D actor and raw14 unfiltered BAM targets.
No clipping, head objective, raw perception or actor expansion is introduced.

## Instrumentation and bounded execution

Both arms add the same read-only metric: actual official positive
`sum((action-prev_action)^2)` over14 raw policy targets and its signed live
reward-manager value. Capture before autoreset, then consume each snapshot once.
No simulator forward, action filtering, reward modification, RNG or state input.
Reconcile finite/nonnegative raw cost with finite/nonpositive weighted reward,
live callable/params/weight and same-step counter. Each24-step row retains
coefficient extrema, full means and raw/signed sums and means for idle, straight,
negative-yaw and positive-yaw commands, aligned to the inherited eight-bin dwell.
The retained verifier checks exact stage timing and group/full reconciliation.

| Phase per arm | Worlds × updates | Matched seed | Child cap |
| --- | ---: | ---: | ---: |
|Normalized ONNX smoke|64×5|1501|600s|
|Measured benchmark|256×10|1511|600s|
|Control/release continuation|256×1000|1517|2100s|
|Final999 evaluation, both arms|8×240 per case|839/853/857|900s combined|

Compressed smoke release stages are-.1 initially, -.4 strictly after1×24,
and-1 strictly after3×24; control remains-1. Both benchmarks stay-1 to measure
the final recovered recipe. CPU tests, both smokes and both benchmarks precede
either learner; projection `elapsed*100*1.5<2040s` is required. Save initial plus
0/250/500/750/999; each learner6144000 transitions/20000 minibatches, final
normalizers49152000/Adam160000. Control→release→both final matrices is serial.

## Unchanged evaluation and interpretation

Twelve cases per arm: seeds839/853/857 × idle/(.3,0)/(.2,-.2)/(.2,.2), eight
worlds240 first-episode steps; exclude60 startup steps from tracking only.
Per-world speed MAE≤.03 idle/≤.10 moving, yaw MAE≤.10; no terminal/rated-speed
exceedance, per-case pooled torque p99≤.60, all14 hard margins≥-.005rad and
right-hip stop fraction≤1%. Retain the original combined decision unchanged.
Mean improvement alone is not acceptance and no failed case is discarded.

Report final paired speed/yaw MAE and pass counts separately for each command,
idle retention, training falls total/first100/last100, torque/power/thermal,
range/stop exposure and actual raw action-cost/weighted mass by stage and command.
Describe a moving-tracking benefit only if all three moving groups have lower
speed MAE without a yaw-MAE regression and no held-out motor/range/stability
regression; otherwise report the mixed or rejected result explicitly. This
descriptive rule cannot waive any combined gate or qualify a skill. Do not
select an early release-phase endpoint; evaluate only final999 under recovered
weight-1. Policy/simulator/physical acceptance remain false for this diagnostic.
No automatic longer run, obstacle/hop/football promotion, MP4 or publication.

## Shared host, bounded handoff and state

Use exact clean tested/pushed source on100.100. Freshly verify DINO PID/exe/uid/
GPU UUID and original active user vLLM with the other three protected scopes
inactive. Pause only originally active **user** vLLM, preserve DINO and unrelated
workloads; do not alter100.98/NX/FilmBrain. Own the GPU lease; while Duck runs,
all four protected scopes inactive, used≤6144MiB/free≥8192MiB/temp<75°C,
Torch20%, CPUthreads1, memory8GiB/CPU200%. Gross training torque guard1.5 does
not replace the held-out.60 gate. No driver or installed package changes.

Retain nonenabled preflight45min, matched95min and handoff150min user services
with15s child grace. The exit guard restores only originally active user vLLM
after Duck is gone, the lease free and DINO-only GPU proof holds; verify restored
cgroup ownership. Unknown workloads cause refusal, never concurrency. This
restoration is part of the existing scoped authorization, not a broader restart.

Source/testing/native admission and training outcomes are recorded separately
below as they complete. No capability claim follows from preparation.

## Source and operational preparation checks

CUDA-hidden single-thread checks passed108 focused tests in168.80s and1097
integrated/adjacent tests in458.98s with16 existing actuator-selector warnings.
Documentation links and `git diff --check` passed. Independent source review
found no semantic blocker; native measured projection remains mandatory because
per-step action-stratum reductions add synchronization overhead. The operational
generator retains the existing scoped restoration guard and inserts per-row
action validation in native preflight/final proof plus independent stdlib
portable verification. Two portable arithmetic tests passed across all modes,
strict transition boundaries and corruption refusals. Native admission and
learners remain pending at this source declaration.

## Native admission passed; matched motion training active

Exact clean native source `17d80717fd2e6ae510d375b3610296f1b99c5b53` was tested
and pushed before execution. Native1097 checks passed in36.33s with16 existing
warnings. Both official normalized61→14 ONNX smokes passed finite CPU inference
and hash checks; control/release took4.516/4.464s with0/0 falls. Benchmarks took
7.815/7.849s with5/7 falls and conservative projections1172.274/1177.291s,
both below2040s. All30 closed preflight rows passed finite/zero-NaN, stock XYZ
angular, motor/bilateral qpos and raw/signed action-stage/stratum checks.
Actual identical initial models/normalizers/Adam/LR were verified against936a;
initial SHA256 `d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`.

Native admission proof SHA256
`70be3095ffb4d5e4b6ea3aef768184b3b6ae65006055c4742e85e104605acc51`;
original service/GPU context SHA256
`6c21861de889d84cf4ab4577aa7c5cee9409f8d131459239a792191f1a3ddeeb`.
Mac JSON-only admission data is retained at
`artifacts/evaluations/motor-range-actionrelease-17d80717-20261013-preflight`;
local stdlib verification authenticates these two hashes and all30 closed rows.
Any `live-control-observation` subdirectory is explicitly an unclosed monitoring
snapshot, not part of the admission proof. Models/ONNX remain native.

Run directory `artifacts/training/motor-range-actionrelease-17d80717-20261013`.
`microduck-actionrelease-17d80717-matched-eval.service` is active under
`recomo-duck-actionrelease-17d80717-handoff.service`, serial control→release→
both final999 matrices. At10:30 Beijing control was671/1000 with durable
0/250/500 checkpoints, all finite and zero NaN terminations; training falls881,
recent100110, latest torque p99.711380/thermal.060410. This is exploratory
training evidence, not a pass of the held-out.60 motor or tracking gates.
CUDA-hidden actual model250 read verified finite tensors, iteration250/env6024,
normalizers44550144 and17 Adam states145020, SHA256
`0af04c1e09285db4134e6cf242ddad36fa4ebb41df2b4a38aa20e825aa601030`.

GPU1710MiB/67°C, DINO2029 preserved, all four protected scopes inactive during
the scoped pause. The native checkout stays17d80717 while jobs are active;
later documentation commits do not change training source. Shell/Python syntax,
systemd validation,3 mock restore tests (7 conflict refusals),2 portable row
tests and independent actual generated-ops review passed before launch.
Five additional CPU-only stage-analysis tests passed after handling the actual
nested decision schema and partial first-episode cases; analysis awaits sealed
data. Final result, checkpoint/evidence seal and original user-vLLM restoration
are still pending. No qualified new motion, harder-skill promotion or video.

## Completed pair: release schedule rejected, no skill promotion

Both1000-update learners and24 held-out cases finished successfully: control
648.281s/release657.009s, each6144000 transitions/20000 optimizer minibatches.
Actual initial and0/250/500/750/999 checkpoints were verified finite, with the
declared parent, clocks, normalizers49152000 and17 Adam states160000. All2030
preflight/training rows passed zero-NaN, fixed stock angular/motor/bilateral
guards and exact action-stage/command-stratum reconciliation, including mixed
updates251/501 and fully recovered502–1000. All three services are inactive,
MainPID0/Resultsuccess/ExecMainStatus0. Neither endpoint qualifies; all nine
moving cases fail the unchanged combined tracking decision in both arms.

| Held-out command | Speed MAE control→release | Speed passes/24 | Yaw MAE control→release | Yaw passes/24 |
| --- | ---: | ---: | ---: | ---: |
|Idle|.001425→.001155|24→24|.009590→.006695|24→24|
|Straight .3/0|.145172→.144749|0→0|.248578→.267715|0→0|
|Negative arc .2/-.2|.095936→.097678|13→14|.294186→.268246|0→0|
|Positive arc .2/.2|.097646→.091587|14→18|.268252→.285016|0→0|

Moving speed totals27→32/72 and yaw0→0/72. Straight achieved vx is nearly
unchanged, .154828→.155251m/s against.3. Negative-turn signed yaw improves
-.059533→-.139113 against-.2, but positive-turn response worsens
.160880→+.051479 against+.2. Straight signed yaw is near zero,
.003195→-.024008, despite absolute/yaw MAE.249→.268: these retained averages
suggest oscillation and directional asymmetry, not an identified gait mechanism.

All24 first episodes survive240 steps; no terminal/rated-speed exceedance,
hard-range violation or right-hip stop exposure. Maximum moving-case pooled
torque p99.575214→.513050, below.60; minimum all-case14-joint hard margin
+.134469→+.141539rad. Idle torque p99 increases.292465→.341958. Some paired
torque/range margins worsen without crossing a gate; no average hides those
regressions. Moving mechanical-power means (straight/negative/positive) are
1.325790/.928337/.896792→1.329012/.927279/.942853W; corresponding thermal
proxies.031040/.028865/.028679→.028808/.027032/.027504. Idle power/thermal
.022960/.011889→.087104/.013451. These are means of three case scalars per
command, not new pooled power/thermal acceptance gates.

Training falls1320→1424, first100120→104, last100125→147. Fully recovered
updates502–1000 raw action cost1.190550→1.585576, with signed cost nearly its
negative in both. Release early raw/signed costs are1.486820/-.148682 at1–250
and1.831139/-.732456 at252–500. Exploration contributes to these raw target
differences; do not call them deterministic gait motion or prove an attempt-tax
cause. The predeclared descriptive benefit is false: speed does not improve in
all groups, yaw regresses in two groups, and paired safety margins also regress.
Close this schedule rather than adopt it, replicate it, stack its rejected
endpoint or automatically lengthen training.

Native proof SHA256 `f9887a63df6f5b717d1c0fa4b62d17837a5b0fa058953de198f0061212ec4113`;
59-payload manifest `bfee7539632ff337e39199564bc2c78643b1a81d5f2d54b9fdd94cd6fa6f9434`;
aggregate `d3bed5c8cbdb075fb5b82d26ad818ecb1971876561a489ed18fd08e93796ed90`.
Final control checkpoint `1deb9a22671a095a4fb4e693f2bf5dbd36b6d74969354f8003aa33a3e3652113`;
release `a4380715f02ac46f652d6c1469149f3466b1b741ca1600b403cb36525106cfc9`.
Mac stdlib verification authenticated59 payloads/71834184bytes without fetching
weights/ONNX: summary `33c259500930f5432342fa5e3741fbcc1e58d3a7720b04645a317f8e7d28b56b`,
portable proof `08f6b727f04043b4c252901dc6517f8da075d91f4b5a74240b297b1384b3f8f6`.
Stage/paired analysis `472ade77548acf6d6403b55bd901ab5183924a4a030d55c67e2f992a6f98d5e4`
is hash-bound and preserves every original decision. Six stdlib tests include
end-to-end24-case/2000-row sealed fixtures and partial/missing-sample refusals.
Independent real-data review agrees on the mixed/rejected motion result.

Analysis operations are retained natively under
`/home/converge/.local/state/actionrelease-17d80717-analysis-v2`: analyzer SHA256
`a9ef444e09735958e309498a0c4c926b132f64d218655f2d076fefdf3edde6d8`, tests
`0171867373791da2d66d7a4a4fd68fbf727d5d0d4c1b8f4897d5f76cd19ce1d2`.
These are ignored operational assets, not installed training source. The earlier
unexecuted analysis backup is superseded; v2 fixes a return-hash binding and
strictly rejects changed acceptance flags before any real output is produced.

User vLLM restored at10:46:35 Beijing with GPU engine631495/9692MiB and DINO2029/
946MiB preserved; other three protected scopes remain inactive. Restoration
receipt SHA256 `74f4dcf5c98a467f91e217fe6ed78ebec7e90bdf2eff51f1abdbc0d24f4c6d68`.
Duck peak2374MiB/69°C; temporary vLLM startup heating is outside that Duck
health stream. Physical/simulator/policy acceptance remains false. Next is one
separately declared direct linear-tracking contrast from936a, with full
smoothness and stock angular tracking, not another smoothness sweep or harder
skill promotion.
