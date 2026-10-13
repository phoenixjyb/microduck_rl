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
