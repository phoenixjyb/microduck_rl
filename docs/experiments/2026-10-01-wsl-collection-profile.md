# WSL eager-collection performance diagnosis

Predeclared October 1, 2026, before GPU execution. This is a short profiling
experiment, not a new training budget or reinterpretation of either rejected
[WSL qualification](2026-09-30-stance-wsl-replication.md).

## Fixed experiment

Preserve the isolated runtime checkout at
`987b452dbfdb4cba7c7fef79f419790bd602ae3a`, its frozen environment and old
receipts. Archive only the new profiling tool from its exact reviewed Git commit
into ignored artifacts; authenticate its bytes against that Git object and an
independently supplied SHA256. Bind **tool source and runtime source separately**.
No installed source, dependency, physics, observation, reward, motor, reset,
optimizer or acceptance contract changes.

Same explicit `wsl-10098-20260930` host/GPU/driver profile. Same shared FilmBrain
lock, idle/live ownership, 75 C temperature and 5 GiB aggregate memory guards;
6 GiB free reserve. FilmBrain remains active and protected AI Mission GPU services
inactive. Advisory guards do not establish hard hardware isolation.

One sequential 900-second child inside a 960-second `KillMode=control-group`
user service, `MemoryMax=6G`, `CPUQuota=200%`, `Nice=10`. Fresh absolute window
27–60 minutes, with 600-second closeout reserve. Any failed guard stops only the
owned Duck process group. No automatic retry, no full learner, no video.

Repeat the exact two-world integration/isolation cases once. Then three fresh
64-world eager plants, reseeded 523, driven by the identical frozen parent:
`baseline-before`, `python-profile`, `baseline-after`. Each repeats the existing
unchanged collection function: 24 policy ticks per update, two discarded warmup
updates and eight measured updates. Total 30 update-equivalents, zero optimizer
steps and zero checkpoint exports. Bound this short run using the retained
worst update cost with a 1.5 instrumentation reserve and 60 seconds extra; it
must fit 900 seconds. An overrun is a failed diagnostic, not license to extend.

Only the middle batch enables Python `cProfile`, preserving function calls and
all checks; it adds no CUDA synchronizations. Retain each function's call count,
self time and inclusive time as JSON, not an executable pickle. These are
Python-call wall-time measurements, **not GPU kernel attribution**; nested
inclusive times must not be added as independent fractions. Keep profiled times
out of baseline summaries and all training qualification decisions.

Capture cgroup-v2 `cpu.stat` and `cpu.max` at each collection boundary; retain
all monotone deltas including throttling. Counters cover the entire service
(supervisor, child and telemetry), not just simulator CPU. Do not infer CPU
quota causality from a counter alone or increase quota in this experiment.

## Evidence and decision

Retain launch, exact tool bytes, parent/runtime hashes, integration JSON, child
log, all three ordered timing batches, full function table, deterministic
summary and report hashes. Replay on CPU against independently recorded launch
and tool hashes. Missing batches, nonfinite times, counter regressions, changed
runtime or mismatched files cannot produce success.

Success is `profile-complete-not-training-qualification`, with all training,
timing, stance, football and motion admission flags false. Frozen-parent
collection does not measure the additional stochastic learner bookkeeping,
durable tick receipts or checkpoint writing; do not equate it to whole-training
wall time. Compare the natural `_forward`, `_sync`, contact reading, motor,
integrator, snapshot and CPU-copy boundaries to identify a narrow next experiment.
If the evidence suggests an optimization, preserve safeguards and predeclare
its equivalence checks before changing the retained learning path. Full learner
577/587/593 and full held-out WSL evaluation remain blocked.
