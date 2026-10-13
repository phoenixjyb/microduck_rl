# Motor/range-retaining yaw-only tracking: matched declaration

Declared before GPU execution on `integration/upstream-2026-10-10`.
Protocol `integrated-motor-range-yaw-only-v1`; implementation
`src/mjlab_microduck/integrated_motor_range_yaw_only.py`.

## One factor and parent

The [completed decoupling comparison](2026-10-13-motor-range-yaw-diagnostic.md)
lowered moving yaw MAE but reduced speed passes50/72→32/72; yaw passes remained
0/72. Neither endpoint is adopted or used as this experiment's parent.
Both arms restart from the original **unaccepted exploratory control**:
`artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA256 `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Restore identical actor, critic, normalizers43008000, seventeen Adam states140000
and LR; reset learner/episode clocks0, retaining all seven materialized final
stock curriculum values rather than inferring them from continuation clocks.

For `a=exp(-yaw_error²/.5)` and `b=exp(-XY_body_rate²/.5)`, stock control keeps
the actual upstream angular callable, weight+2/stdsqrt(.5), reward `2ab`.
Treatment replaces **only that callable** with positive yaw-only tracking `2a`;
its params and weight are identical. Difference `2a(1-b)` is nonnegative and
state-dependent: reward mass and gradients change, not merely a derivative.
No clipping, signed additive kernel, yaw-width change or additional cost.

Retain the separate trunk `body_ang_vel` weight-.05 and whole-body
`angular_momentum` weight-.02. These overlap in stability purpose but use
different signals/frames: trunk world-frame XY rates versus root body-frame XY
tracking, and whole-body momentum. This is an axis-inclusion diagnostic, **not
proof of duplicate penalties**. An earlier width contrast on another compound
parent also traded improved yaw for poorer speed; acceptance is unknown.

Motor weight-1 (nominal.60Nm/soft.70/overload4), bilateral named hip-yaw qpos
weight-4/margin.15rad, zero forward term/read-only metric, command distribution,
plant, events, optimizer, raw61D actor and raw14 unfiltered BAM targets remain
unchanged. No actor expansion, head change, target clipping or raw perception.

## Fixed budget and unchanged gate

| Phase per arm | Worlds × updates | Seed | Child cap |
| --- | ---: | ---: | ---: |
|Normalized ONNX smoke|64×5|1433|600s|
|Measured benchmark|256×10|1439|600s|
|Matched control/yawonly continuation|256×1000|1447|2100s|
|Both final999 held-out matrices|8×240 per case|839/853/857|900s combined|

CPU tests, both smokes and both benchmarks precede either full learner.
Conservative benchmark projection `elapsed*100*1.5<2040s`, no cap widening.
Run control→yawonly→evaluation sequentially. Retain initial and0/250/500/750/999
checkpoints; each learner6144000 transitions/20000 minibatches, expected final
normalizers49152000 and Adam160000. Evaluate final999 only, no checkpoint fishing.

Read-only post-reward/pre-reset capture validates raw yaw/XY squared errors,
same-step snapshot, live callable/params/weight, and actual positive reward[0,2].
Raw nonfinite XY still fails despite exclusion from this kernel. Log both angular
components, inclusion flag, unchanged stability weights, negative sample count0,
optimizer/command/fall/motor/range telemetry. No simulator forward/sense changes.

Unchanged12 cases per arm: three seeds × idle/(.3,0)/(.2,-.2)/(.2,.2), eight worlds,
240 first-episode steps; exclude first60 from tracking only. Per-world speed
MAE≤.03 idle/≤.10 moving, yaw MAE≤.10; no terminals/rated-speed exceedance;
per-case pooled torque utilization p99≤.60; all14 hard margins≥-.005rad and
right-hip stop fraction≤1%. Preserve every numerical gate and combined decision.
Judge both speed and yaw, load/power/modeled thermal and bilateral ranges;
slower/quieter walking alone is not success. One seed is a diagnostic, never
multi-seed policy or physical acceptance.

## Scoped handoff and exit

Use100.100 native checkout at the exact tested/pushed source only, frozen runtime.
Freshly validate DINO PID/exe/uid/GPU UUID, original active **user vLLM** and other
three protected scopes inactive before pausing only user vLLM. Preserve DINO and
all unrelated workloads; leave100.98/NX/FilmBrain untouched. During Duck require
owned GPU lease/all four protected scopes inactive, used≤6144MiB/free≥8192MiB/
temperature<75°C, Torch20%, CPUthreads1, memory8GiB/CPU200% service limits.
Gross training torque guard1.5 is distinct from the held-out.60 gate.

Retained sequential services use45min preflight/95min matched/150min outer caps,
15s child termination grace and the existing `ExecStopPost` restoration guard.
Restore only originally active user vLLM after Duck exit, free lease and DINO-only
GPU proof; verify cgroup ownership. Unknown workloads cause refusal, not overlap.
No package/driver changes, physical motion, MP4, publication, harder skill or
automatic adoption. Retain hashes and final decision, then review the result.

## State

Prepared and source-tested:76 focused checks passed in7.66s, then1046
integrated/adjacent checks passed in36.43s with16 existing selector warnings,
CUDA hidden/single-threaded. The first new fixture incorrectly required the
previous experiment's1429 seed instead of this declaration's1447; the fixture
was corrected, not the matched recipe. Independent source review confirms the
one-callable contrast, retained safety terms and observer formula. The ops builder
validates all transformed outputs before creating any and requires clean exact
HEAD; both verifier anchor substitutions were checked deterministically.
Native admission, normalized smokes, benchmarks and numerical outcome pending.

## Native admission and active continuation

The exact clean trained source is `d71c30285ff3acbdfee12615548eaf2ea1a10975`,
tested and pushed before handoff. Native CPU1046 checks passed in36.77s with16
existing warnings. Both official normalized ONNX exports passed finite61→14
inference and hash checks; both smoke learners had0 falls, actual positive reward
with0 negative samples. Control/yawonly smokes4.513/4.438s; benchmarks7.581/7.771s,
with4/6 falls and conservative projections1137.117/1165.680s below2040s.
All30 preflight updates passed finite telemetry, live formula, unchanged stability,
motor/bilateral and materialized-final-curriculum guards. Identical initial SHA256
`d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235` and native
actual parent model/normalizer/Adam comparisons passed. Preflight peak1710MiB/63°C.
These are admission checks, not held-out policy acceptance.

Native preflight proof SHA256
`774f9d4fbd364a329e60a632e9075c9dff507321a3cd0b3e50ee24813c22849b`;
original handoff context SHA256
`d1f7784e0ab8bb2bfc6e4fe1b17702a28f9d0f704f2ffd7bd769c083da6c942e`.
Closed JSON-only data/context are retained on Mac at
`artifacts/evaluations/motor-range-yawonly-d71c3028-20261013-preflight`;
local stdlib checks authenticated these hashes and all30 positive-reward/stability
rows. Weights and ONNX remain native, without duplication. Shell/Python syntax,
systemd unit verification and3 mock-only restore tests passed, including7
conflict cases that refuse service mutation; independent generated-ops review
found no launch blocker.

The retained service `microduck-yawonly-d71c3028-matched-eval.service` runs under
`recomo-duck-yawonly-d71c3028-handoff.service`; native run directory
`artifacts/training/motor-range-yawonly-d71c3028-20261013`.
At08:17 Beijing control reached182/1000 updates/4368 vector steps, with a
completed model0 checkpoint. A CUDA-hidden native read authenticated that actual
finite model0: SHA256 `6d6ea7ebe58beddd21920f0fe188aa88741a42c578e608fde6d4672589a2b2bf`,
iteration0/env24, normalizers43014144 and seventeen Adam states140020.
The preceding85-row audit was entirely finite; latest row182 NaN terminations0,
torque utilization p99.743653, thermal proxy.064968, left/right hip stop exposure
0/.00048828125, negative reward samples0. The training p99 is below gross1.5,
not evidence of passing the separate held-out.60 gate. GPU1710MiB/67°C,
DINO2029 retained and protected services inactive during the authorized pause.

Control→yawonly→both final999 matrices is active; numerical decision and original
user-vLLM restoration remain **pending**. The retained exit guard enforces scoped
restoration only after Duck exit plus free lease/DINO-only proof. Do not alter
the active worktree or services, repeat/extend the round, promote a policy or
claim a new skill from these early samples. Later evidence-only fork commits do
not change the native trained source pin.

## Completed result: neither endpoint promoted

Both1000-update learners and all24 fixed final999 cases completed. Control took
619.440s, yawonly626.194s; each6144000 transitions/20000 optimizer minibatches.
All initial and0/250/500/750/999 checkpoint tensors, source/parent pins, episode
clocks, normalizers49152000 and seventeen Adam states160000 were authenticated
on native CPU with CUDA hidden. All2030 preflight/training rows are finite, zero
NaN terminations and negative angular samples. No failed service or run receipt.

**Neither policy is qualified.** All9 moving cases fail combined tracking in
both arms. Control additionally fails one idle case's yaw gate (23/24 idle yaw
worlds), while yawonly passes idle24/24. All24 first episodes survive240 steps,
with zero rated-speed exceedance, hard-range violation or stop proximity.
The motor/range subsets pass, but that is not a complete foundation or physical
safety certificate. Do not adopt either endpoint as a teacher or new skill.

| Quantity | Control | Yaw-only |
| --- | ---: | ---: |
|Moving speed world passes|32/72|29/72|
|Moving yaw world passes|0/72|0/72|
|Straight achieved vx, target.30m/s|.155523|.157007|
|Straight speed/yaw MAE|.144477/.315643|.142993/.235391|
|Straight signed mean yaw, target0|+.070628|-.026774|
|Negative arc speed/yaw MAE|.098187/.341936|.092919/.235795|
|Negative arc signed yaw, target-.20|-.016430|-.097338|
|Positive arc speed/yaw MAE|.090072/.324469|.096380/.213880|
|Positive arc signed yaw, target+.20|+.202668|+.136324|
|Maximum per-case pooled torque p99|.571437|.555022|
|Minimum all-joint hard margin, rad|+.210422|+.240633|

Moving yaw MAE decreases25.425/31.041/34.083% for straight/negative/positive
commands, but no moving yaw world passes. Straight speed remains0/24 in both;
negative arc speed passes14→16/24, positive18→13/24. Improvement is mixed and
does not unlock turns, obstacles, hops or football. Do not compare this treatment
causally with the earlier decoupled arm: controls use different training seeds,
1429 versus1447. Their moving speed passes50→32/72 and straight MAE.121973→.144477
show material training-seed sensitivity despite the same source recipe/parent.

Scalar mean mechanical power W/thermal proxy across the three cases per command,
control→yawonly: idle.051140/.012424→.044298/.010775;
straight1.373432/.031513→1.380517/.030327;
negative.935195/.028564→.976259/.027124;
positive.965532/.028620→.944005/.026772. Torque p99 above is a maximum across
per-case pooled percentiles, not averaged or cross-case re-pooled. Thermal is a
modeled load proxy, not measured motor temperature.

Training falls1415/1510, first100129/157 and last100168/141, control/yawonly.
Mean angular reward.632242/1.266526 (kernel reward mass deliberately differs),
mean squared yaw error.546952/.483295 and XY rate.794273/1.019218. These training
statistics cannot replace the failed held-out gates; higher total reward is not
better locomotion.

## Sealed retention, independent review and restored workload

The manifest authenticates59 JSON/JSONL payloads,69654583 bytes. Local stdlib
verification recomputed gate failure arrays, per-world passes and group means;
additional checks authenticated full native/evaluation decision equality, derived
labels and nested non-acceptance flags. Parallel numerical review independently
confirmed the metrics, thresholds and aggregation. Weights/ONNX stay native.

| Object | SHA256 |
| --- | --- |
|Control final999|`e71ae2b2eae6332e9371defc72b2132b7c1074282d546250283f0fd22f571578`|
|Yaw-only final999|`b7b1f581f612f910953d490b68db376ccd2ebb7f20297feb37753f139a1b8793`|
|Aggregate|`789dbbd3c8cbb6dd971afc26b02a2fcf047020cea74c8f14b764cf69c207eabf`|
|Native verification|`55c6cdf84b2ee8ae86fb27b5437b99fc260f5269c29564c4d641198b31cd8af7`|
|Manifest|`dc7cde6f7faf8e98276b22a0598b9ed2c2a96b35086c536a1c18d420187b5715`|
|Local derived summary|`f36bfa3424c2b46710d709ee1d66dc7a0309ff11bd1496be8f86c06cafeaf879`|
|Local verification|`6a4bffed8344763467ffbe9392b01d4192bf32568beff6f72d700d4b91cbe235`|
|Restoration receipt|`68b8d591bf7db2a2c6ff763cf5f4dd79e7b3af1e6a2af9307d22d8312e6768f0`|
|Offline straight-trace diagnosis|`f0cd7d3316077ed28579c630401d2ca93d024d11cbae89025927a5eb0a5a987f`|

Mac JSON-only bundle: `artifacts/evaluations/motor-range-yawonly-d71c3028-20261013`.
Original native run stays pinned atd71c3028, independent of evidence-only fork
commits. The exit guard restored only original user vLLM at08:38:39 Beijing,
after free lease/no Duck service/DINO-only GPU proof. Its cgroup-owned engine
528984 uses9692MiB; DINO2029 remains946MiB; other three protected scopes inactive.
Duck peak2374MiB/68°C. All three Duck services now inactive/MainPID0/Resultsuccess/
ExecMainStatus0. A later SSH timeout recovered on read-only TCP/ICMP/SSH checks,
with no restart, reboot or job change. No100.98/NX/FilmBrain, driver/library,
physical motion, raw perception or video changes.

## Bounded offline diagnosis and next question

The tested body/gait summarizers passed77 CPU tests. The initial audit script
incorrectly assumed every case contained a raw trace; it failed before creating
output. Read-only inspection established that the retained evaluator traces
**only the six straight cases**, not idle/arcs. The script now explicitly checks
six traced/eighteen untraced cases and preserves that limitation; no extra GPU
rollout or acceptance gate was added. Derived trace JSON remains outside the
original manifest. Its group means across24 straight worlds, control→yawonly:
absolute per-world DC bias.114436→.054532rad/s;
within-world residual std.392729→.312587rad/s;
projected heading-rate MAE.314790→.233157rad/s. Heading-rate error remains large
and is not a gate substitute. Both have zero sampled no-contact/double-flight
fractions; one-foot contact fractions.956713/.965509. Sampled contact timing and
head correlations do not identify a causal head/actuator defect or prove physical
support. There is no turn trace from which to infer a turning mechanism.

Next: predeclare a second **matched training seed** from the same original936a
parent, identical1000-update budgets, safety objectives and held-out evaluator,
before additional reward mutation. Test repeatability of yaw-error reduction
alongside speed, falls and motor/range retention; do not retrospectively select
an endpoint or loosen gates. Replication is still an exploratory diagnostic,
not permission to promote an all-failing foundation. No further GPU job was
started during this closeout; user vLLM remains restored for the next admission.
