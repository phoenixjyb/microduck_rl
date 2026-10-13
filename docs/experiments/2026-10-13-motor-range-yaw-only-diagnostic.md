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
