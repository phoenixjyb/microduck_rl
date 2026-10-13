# Linear-tracking priority: matched motion diagnostic, predeclared

Declared before GPU execution on `integration/upstream-2026-10-10`.
Protocol `integrated-motor-range-linear-boost-v1`; implementation
`src/mjlab_microduck/integrated_motor_range_linear_boost.py`.

## Question and fixed reference

The [action-release schedule](2026-10-13-motor-range-action-release.md) is sealed
and rejected: moving speed passes27→32/72, moving yaw0→0/72, and straight
achieved vx.154828→.155251 against.3. It does not establish a general motion
benefit or identify the cause of underspeed. Close that schedule rather than
repeat it, weaken safety, or inherit its endpoint as a teacher.

Test one direct priority factor: stock linear-tracking coefficient2 versus2.5.
Both arms strictly warm-load original unaccepted936a, not either new endpoint:
`artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA256 `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Restore actual actor/critic/normalizers43008000,17 Adam states140000 and LR;
verify parent iteration999/env24000, then reset learner/env clocks0. Materialize
all seven source curricula at final conditions; no new weight schedule or Adam
reset. Repeat both arms with the same fresh declared seed, not a cross-seed
comparison to a retained control.

## One coefficient, actual stock XYZ kernel

The installed official `track_linear_velocity` computes
`exp(-((vx-cx)^2+(vy-cy)^2+vz^2)/.1)` in body coordinates; commanded z is assumed
zero. It is **not** a pure forward-only reward. Keep that callable/params and
stdsqrt(.1) unchanged, raising only its coefficient2→2.5. This increases both
goal priority and the stock lateral/vertical stability contribution. Interpret
their joint result honestly, not as proof of an isolated forward-gradient cause.

Both arms retain action-rate-1 throughout, stock XYZ angular+2/stdsqrt(.5),
trunk-angular-.05, whole-body momentum-.02, motor-1 (nominal.60Nm/soft.70/
overload4), bilateral named hip-yaw qpos-4/margin.15rad, forward cost weight0,
head/body goals, commands/events/DR/delays/plant/solver and optimizer unchanged.
Actor remains raw61D and outputs raw14 unfiltered/unclipped BAM targets. No
perception training, actor expansion, curriculum hardening or sensor remapping.

## Actual terms measured; bounded execution

Both arms use the same action-rate metric already tested natively, fixed-1.
Both add one read-only official raw linear kernel/actual weighted reward capture
after reward and before autoreset. Verify finite positive raw[0,1], actual[0,w],
exact live callable/params/weight, same-step counter and `actual≈w*raw`, then
clone/consume once. No extra reward, action filtering, simulator forward or RNG.
Each24-step row retains fixed linear variance/weight, raw/weighted means and four
command strata (idle/straight/negative/positive), aligned to inherited eight-bin
dwell. Runtime and retained-output validators reconcile counts, positive bounds,
group/full means and weight. Keep separate action, angular, motor, bilateral and
forward metrics; a larger reward log by construction is not a motion gain.

| Phase per arm | Worlds × updates | Matched seed | Child cap |
| --- | ---: | ---: | ---: |
|Normalized ONNX smoke|64×5|1523|600s|
|Measured benchmark|256×10|1531|600s|
|Control/linearboost continuation|256×1000|1543|2100s|
|Both final999 matrices|8×240 per case|839/853/857|900s combined|

CPU checks, both smokes and both benchmarks precede either learner. Benchmark
uses the actual per-arm2/2.5 coefficient, with conservative projection
`elapsed*100*1.5<2040s`. No cap widening; save initial plus0/250/500/750/999.
Each6144000 transitions/20000 minibatches, final normalizers49152000/Adam160000.
Run control→linearboost→both final matrices serially, never concurrent GPU jobs.

## Fixed gates and predeclared interpretation

Unchanged twelve cases per arm: seeds839/853/857 × idle/(.3,0)/(.2,-.2)/(.2,.2),
eight worlds240 first-episode steps, omit60 startup steps from tracking only.
Per-world speed MAE≤.03 idle/≤.10 moving, yaw MAE≤.10, no terminal/rated-speed
exceedance, per-case pooled torque p99≤.60, all14 hard margins≥-.005rad and
right-hip stop fraction≤1%. Keep the original combined decision. A larger task
reward, one group's improvement or a favorable mean does not override a failure.

Report speed/yaw means and pass counts separately for all four commands, achieved
vx and signed/absolute yaw response, idle retention, falls total/first100/last100,
torque/power/thermal and range/stop metrics plus actual linear/action reward mass.
Describe a moving-tracking benefit only if all three moving-group speed MAEs are
lower without yaw-MAE regression and no paired held-out motor/range/stability
regression. All paired moving episodes must complete; retain fixed denominators,
partial failures and both endpoints. This descriptive rule waives no gate and is
not multi-seed acceptance. Evaluate only final999, not selected early checkpoints.
Policy/simulator/physical acceptance remains false for this diagnostic. No
automatic longer run, rejected-teacher adoption, harder-skill promotion or MP4.

## Shared host and separate acceptance levels

Use exact clean tested/pushed source on100.100. Recheck DINO PID2029/exe/uid/GPU
UUID and original active user vLLM with the other three protected scopes inactive.
Pause/restore only that originally active user vLLM; preserve DINO/unrelated
workloads and100.98/NX/FilmBrain. Require owned lease, all four scopes inactive
while Duck runs, used≤6144MiB/free≥8192MiB/temp<75°C, Torch20%, CPUthreads1,
memory8GiB/CPU200%. Gross training torque1.5 is not the held-out.60 gate.
No driver/package changes. Retain nonenabled preflight45min/matched95min/
handoff150min units and15s child grace. Restore only after Duck is gone, lease
free and DINO-only GPU proof, then verify original-service cgroup GPU ownership;
unknown workloads cause refusal, never overlap.

Source checks, native admission, learner activity, numerical result and physical
acceptance are separate. No physical motion or qualified skill follows from
preparation; native source is frozen while any job runs. Results are added below.

## Source preparation evidence

CUDA-hidden single-thread115 focused checks passed in231.71s, covering paired
configuration/parent/gates, fixed action-rate, positive official kernel capture,
live contract drift, reset/stale/malformed snapshots and24-step/row reconciliation.
Independent source/declaration/ops-template review found no blocker; installed
official source confirms XYZ linear error and reward→metrics→reset ordering.
Documentation links and diff whitespace checks passed. Full Mac regression was
not rerun: the changed MDP code only adds this unused read-only function; the
complete native regression set is mandatory before smokes and either learner.
Native admission/training/outputs remain pending at the source declaration.

## Native admission passed; next matched learners running

Native source is exact clean tested/pushed
`479f444d6514bd257e64e6db99322ed7a40e4e75`;1134 complete regression checks
passed in36.57s with16 existing warnings. Both normalized61→14 ONNX smokes
passed finite CPU/hash checks, took4.691/4.595s and had0/0 falls. Benchmarks
took8.044/8.091s,4/4 falls, projections1206.614/1213.588s below2040s. All30
preflight rows passed fixed action-rate-1, actual positive linear2/2.5,
stock angular, motor/bilateral qpos, command-dwell and zero-NaN guards.
Native actual initial tensors/normalizers/Adam/LR match original936a; identical
initial SHA256 `d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`.

Admission proof SHA256 `ba2eaff600df4051bacb9537ddcaaf9494c216c6663b57233ff2d66442cc7005`;
original service/GPU context `018687c4df3fc27cc1bf3f81f980f961aaa29193adc5cacd2af872e3b785a02e`.
Mac closed JSON-only phases/proof/context at
`artifacts/evaluations/motor-range-linearboost-479f444d-20261013-preflight`;
stdlib checks authenticate both hashes and all30 action/linear rows. No active
learner snapshots or model/ONNX binaries are copied into that admission packet.
Source/ops reviews, shell/Python syntax, systemd validation,3 mock restore tests
with7 refusal cases and2 portable positive-row tests passed before launch.
Review caught and corrected a stale paired-difference label **before** copying
or executing those operations; final verification expects `linearboost_minus_control`.

Run `artifacts/training/motor-range-linearboost-479f444d-20261013`, under
`microduck-linearboost-479f444d-matched-eval.service` and
`recomo-duck-linearboost-479f444d-handoff.service`: control→linearboost→both
final999 matrices. At11:32 Beijing control completed1000 with0/250/500/750/999
checkpoints,1448 training falls and zero NaNs; treatment162/1000 was finite,
203 falls, zero NaNs, actual fixed linear2.5 and action-rate-1. Its raw/weighted
linear means.699557/1.748892 demonstrate the declared coefficient, not a new
skill. GPU1710MiB/68°C, DINO2029 retained and all four protected scopes inactive
during the scoped pause. Keep native source479f444d frozen while active.
Final outcome, checkpoint seal and original user-vLLM restoration remain pending.
No physical/simulator/policy qualification or harder-skill/video promotion.

## Sealed result: complete, rejected

Both learners completed1000 updates (666.360/676.948s), each6144000
transitions/20000 minibatches. Initial models/normalizers/Adam/LR match original
936a; final normalizer counts49152000 and17 Adam counters160000 match the
declared budget. Initial and0/250/500/750/999 checkpoints are durable. All24
held-out cases completed; both endpoints are `foundation-not-ready`. Idle
speed/yaw passes24/24 per arm, but **moving yaw passes0/72 in both arms** and
all nine moving cases per arm fail the unchanged combined tracking gate.

|Held-out command|Speed MAE control→boost|Speed passes /24|Yaw MAE control→boost|Achieved vx control→boost|
|---|---:|---:|---:|---:|
|Idle|.001207→.001192|24→24|.006602→.008388|-.000072→-.000019|
|Straight .3|.108002→.104615|7→10|.298634→.276837|.191998→.195445|
|.2, yaw-.2|.056724→.060383|24→24|.328742→.259706|.144141→.139827|
|.2, yaw+.2|.057410→.069850|24→24|.281764→.275725|.142813→.130311|

Moving speed passes55→58/72 do not establish a motion benefit: both turning
speed MAEs regress, despite lower group yaw MAEs. Signed negative-turn response
-.001653→-.163691 moves toward its goal, whereas positive-turn response
.206717→+.068660 moves away. These pooled signed means do not replace individual
world tracking or heading checks. The predeclared descriptive-benefit rule fails;
do not adopt either endpoint, rerun this scalar contrast automatically or promote
hopping/obstacle/football work.

Neither arm has a held-out terminal, rated-speed exceedance, hard-range violation
or right-hip stop exposure. Worst case torque p99.523476→.514716 remains below
.60Nm, but idle torque.285027→.320172 and straight torque.507001→.514716 rise.
Minimum all-joint hard margin+.185627→+.158380rad shrinks. Straight mechanical
power1.531962→1.581090W and thermal proxy.030406→.033049 increase; other command
groups are mixed. Numerical safety passes, not physical motor qualification.

Training falls1448→1392 (first100132→121; last100137→117). Mean raw linear
kernel.687022→.709251, weighted1.374045→1.773126: the larger weighted value is
partly constructed by coefficient2→2.5. Raw action cost1.231990→1.269200 rises;
mean angular reward.629337→.598837 falls and yaw-error square.561191→.588306
rises. Exploration statistics are not deterministic held-out gait acceptance.

Native source479f444d stayed frozen. Native and portable independent verification
passed all59 retained JSON/JSONL files (73762228 bytes), including actual tensor,
clock, command-dwell and positive-linear/negative-action telemetry checks. Hashes:

- Control final `d302eb944276f95565d1639ead7a9b7981c7feee8cf7ec8e4201cef0a7f61ab5`.
- Boost final `8eeaf935596e02b88779ecf21d695adf4a8bc29507192a6843d25245a9b1ab4d`.
- Aggregate `6e37edf8420de4a7397274f6fa42d93282619dd4979e1fea61aae7fb8d8d5993`.
- Native proof `00d682d4b6606785db438cd8c684bf7f83f107c39bf99d71f5e261513f2f769e`.
- Manifest `6099536f77f62bf416940c3851b53f06800897183c30e1d918205fcbfd2c43aa`.
- Local summary `84c1db15a380703d67a21dd4228cd0edc19d520ba8a5ed1ef11bed56b49e79ae`.
- Local verification `a89d96bfc1fe2ad7951461c78a5a811b2c91c6830b676ace2ef3464fe1cbb989`.
- Restoration `4fd898412999c83b9c5ff3fd68b61a24a115ebf7e19c43423678b4c227ad1746`.

Native retained run `artifacts/training/motor-range-linearboost-479f444d-20261013`;
portable JSON-only packet `artifacts/evaluations/motor-range-linearboost-479f444d-20261013`.
All three scoped services finished successfully/inactive. Duck peak GPU2374MiB,
69°C; DINO2029 remained intact. Only original user vLLM restored with GPU engine
679947/9692MiB; other three protected scopes inactive. Fresh12:00 Beijing check
confirms no Duck GPU process, DINO946MiB plus vLLM9692MiB,48°C. No100.98,
driver/package, physical-motion, video or external publication changes.

Next hypothesis changes lesson complexity, not another reward coefficient:
temporarily keep the existing small nonzero beginner head commands while teaching
the same full locomotion command coverage, rewards and motor/range gates. This
does not prove head goals caused the failure. A separate tested predeclaration
and fresh native admission are required; do not stack this change on boost or
weaken the evaluation protocol.
