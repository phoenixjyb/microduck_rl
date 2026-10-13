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
