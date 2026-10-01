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

## Completed run and retained interpretation

Tool source `2032c54a9a2aaae56a68dbc888ee3e579511992d`; runtime source remains
`987b452dbfdb4cba7c7fef79f419790bd602ae3a`. The runtime worktree stayed clean,
and its installed environment, old receipts and weights were not rewritten.
The separate detached validation worktree is retained in ignored
`artifacts/tools/stance-profile-validation-2032c54a9a2a/`.

Before launch, **327 focused CPU tests passed on the Mac**. Linux initially
passed 326 and skipped only the pinned-parent artifact test, because the new
validation worktree lacked that ignored artifact. After copying the existing
parent with SHA256 `46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93`,
the three parent-selected tests passed, including that previously skipped test.
All 327 distinct checks therefore passed on Linux too; the full repository
suite was not run. Ruff was unavailable. A Luna read-only protocol review found
no remaining concrete defect; the owner reviewed its tests and findings.

Direct Mac-to-WSL SSH intermittently timed out during banner/key exchange. An
explicit per-command `ProxyJump=gw98` reached the same verified WSL machine
and restored reliable transfers. No SSH configuration, host service, driver
or networking configuration was changed; 100.100 was only an SSH relay, not a
Duck GPU execution host.

The retained user service `microduck-wsl-collection-profile-2032c54a9a2a.service`
finished with `Result=success`, `ExecMainStatus=0`, `inactive/dead`. Its actual
900-second child returned zero after **464.829342562 seconds**. The checked live
unit had `RuntimeMaxUSec=16min`, `MemoryMax=6442450944`, `CPUQuotaPerSecUSec=2s`,
`Nice=10` and `KillMode=control-group`. The final archive contains 12 files,
including the report and its 11-file hash inventory, retained on WSL and the Mac
at `artifacts/evaluations/stance-wsl-collection-profile-2032c54a9a2a/`.

Independent CPU-only verification on WSL rechecked live source/tool bindings;
the Mac separately replayed inventory, hashes, integration schema, ordered timing
batches and the deterministic summary. The replay API honestly reports
`live_host_rechecked=false`: that API itself is offline, even when the WSL CLI
has separately called its live binding check. Both reproduce
`profile-complete-not-training-qualification`; all admission flags remain false.

Exact retained hashes:

- Runner SHA256: `f9af2c3aa573c5729e9295dc3b3230966c987b488aac55a0aa9c02044a742b1c`.
- Launch SHA256: `662f2dda9d3900f8b914d330fcd7051133167023c21ef7177ecd36d9f2d2b025`.
- Report SHA256: `a9942d2a2abf9887e4beb28b790bd34f6cc05a5352afe9538b68eb658bee5ccc`.
- Summary SHA256: `12db38ba25455aa20b80cbf13a69d0ae98f5ecbd0ffc74fdc38b2d9f263a2ed9`.
- Function-table SHA256: `5bfe476985a6f5e484afc4b838646b472e6b3d53eae66d39c78f484bb73e2a47`.

| Batch | Measured updates | Mean seconds/update | Maximum seconds/update | Throttled periods |
| --- | ---: | ---: | ---: | ---: |
| Baseline before | 8 | 13.359384 | 13.656513 | 0 |
| Python-profiled | 8 | 18.629592 | 19.325184 | 0 |
| Baseline after | 8 | 13.460178 | 13.711136 | 0 |

The 16 uninstrumented measurements average **13.409780758 s/update**, maximum
13.711136294. Both baseline batches and the profiled batch record zero
`throttled_usec` as well as zero throttled periods. The CPU quota therefore was
not actively throttling this sampled run; this does not establish how a different
background workload would behave. Profiling adds substantial overhead, so its
slower times are not a training sizing reference.

The instrumented collection, including two warmups, accounts for 183.623344
seconds of Python-profile wall time. Selected inclusive boundaries:

| Function | Calls | Inclusive seconds | Interpretation |
| --- | ---: | ---: | --- |
| Owned runtime `_forward` | 4,801 | 123.869089 | About 67.5% of profiled time; includes solver and contact checks |
| MuJoCo Warp `forward` | 4,801 | 86.047879 | Nested inside the owned wrapper |
| Warp `launch` | 1,001,708 | 36.401533 | Kernel dispatch across the measured collection; not kernel execution attribution |
| Contact `read_contacts` | 4,801 | 24.792063 | Nested contact decoding, audits and validation |
| Owned runtime `_sync` | 9,602 | 0.954530 | Direct synchronization wrapper is not the dominant boundary here |
| CPU-owned result copying | 71,520 recursive calls | 3.469047 | `stance_attempt_trace.owned`; much smaller than the forward path |

These rows overlap and must not be summed. The million eager launches and large
forward-wrapper cost support investigating **dispatch and validation overhead**,
not more VRAM. They do not prove a specific GPU kernel is slow or authorize
removing checks. The 240 frozen-policy inference calls take only 0.103941 s
inclusive. Extra learner bookkeeping and durable training exports remain outside
this diagnostic. The earlier suspicion that direct `_sync` calls alone dominate
was not supported by this profile.

The 383 retained telemetry samples peaked at **53 C** and **1,180 MiB total GPU
memory**. FilmBrain preview and observatory remained active with zero restarts;
the protected AI Mission GPU services remained inactive. The post-run sample
showed no compute PID, 0% GPU utilization, 43 C and 677 MiB desktop memory.
Observations support successful coexistence under the sampled guards, not a
general claim of zero impact on another service.

## Next bounded chunk

Investigate an eager-path reduction in kernel-dispatch/validation overhead, with
the existing finite, capacity, contact-address, motor, clock, first-terminal and
selective-reset checks preserved. Require focused CPU contract tests and an
explicit source-bound CUDA comparison before performance qualification. Do not
cache away source integrity checks or remove synchronization merely because it
appears numerous. No speedup is asserted by this completed profiling run.

The [previous captured-forward rejection](2026-09-09-stance-forward-repeatability.md)
is still binding: do not enable that graph candidate as a shortcut. Any renewed
graph experiment needs its own numerical-equivalence protocol, not a changed
threshold chosen to pass observed discrepancies. Neither these profiling
baselines nor the earlier disposable smoke are accepted full-run calibration.
Only after an accepted optimization and a fresh timing gate may the first full
learner (577) proceed, followed by the unchanged replication/evaluation protocol.

Follow-up: the [finite-check microbenchmark](2026-10-01-wsl-finite-check-probe.md)
predeclares a standalone predicate comparison with fresh packing cost included.
It does not integrate the candidate, remove checks or qualify a learner.
