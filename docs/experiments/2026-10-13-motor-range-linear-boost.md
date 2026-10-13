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
