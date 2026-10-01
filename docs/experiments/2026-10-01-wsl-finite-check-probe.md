# WSL solved-field finite-check microbenchmark

Predeclared October 1, 2026, before GPU execution, within the authorized
development window ending **18:00 Asia/Shanghai (10:00 UTC)**. This follows the
[retained eager-collection profile](2026-10-01-wsl-collection-profile.md), not a
new learner or a reinterpretation of either rejected timing qualification.

## One changed axis

The runtime checks eleven solved float32 fields separately after each forward.
The candidate freshly flattens/concatenates the same current tensor values and
checks their finiteness once. On aggregate failure it replays the original field
order and exact first-field error. No casts, filters, clamps or physics arithmetic
are added. Allocation/backend failures propagate; an inconsistent aggregate
cannot pass. Inputs are quiescent, exclusively owned, same-device float32;
concurrent mutation is unsupported by both paths.

The [standalone helper](../../src/mjlab_microduck/stance_solved_field_check.py)
is **not installed into `WarpStanceRuntime`**. The runtime, dependencies and
training/evaluation/checkpoint receipts remain at
`987b452dbfdb4cba7c7fef79f419790bd602ae3a` in the isolated WSL checkout. Archive
only the new [runner](../../src/mjlab_microduck/stance_finite_check_probe.py) and
checker from their exact reviewed tool commit. Supply each SHA256 independently,
verify against Git objects and bind tool source separately from runtime source.
Do not fast-forward that runtime or rewrite its environment.

No contact/source-integrity/capacity/address/friction/motor/clock/reset/terminal
check is removed, cached or changed. Eager `mjwarp.forward` remains unchanged;
the previously rejected captured-forward candidate stays disabled.

## Fixed protocol and resource bounds

One sequential user service on the exact `wsl-10098-20260930` profile:

- 120-second child watchdog; 180-second `KillMode=control-group` service.
- `MemoryMax=6G`, `CPUQuota=200%`, `Nice=10`; 600-second closeout reserve.
- Explicit absolute job window of more than 780 and at most 3,600 seconds,
  wholly before today's 18:00 cutoff. No automatic retry or cap extension.
- Existing FilmBrain `wan-gpu.lock` lease, two idle samples and unchanged live
  ownership/temperature/memory guards. Preserve FilmBrain and unrelated services;
  both protected AI Mission GPU services must remain inactive.

Before preparing launch evidence, execute deterministic independent CPU fixtures
with the pinned checker bytes: finite predicate agreement, all 33 single-field
NaN/positive-infinity/negative-infinity errors and unchanged input bits. Retain
the exact preflight decision/hash in the launch. These are synthetic checks,
not simulator values or capability evidence. Focused source tests additionally
cover multiple faults, empty/strided views, fresh mutation, type/device/byte caps,
allocation failure, aggregate inconsistency and malformed timing/evidence.

Construct one unchanged 64-world CUDA runtime. Construction performs its normal
reset/eager forward **outside** measured blocks. `physics_steps=0` here means
zero Euler integration ticks, not zero physics computation. No policy inference,
action update, optimizer, episode, trajectory collection or checkpoint export.

Use the same current eleven solved views for all six blocks in fixed order:
`legacy, packed, packed, legacy, legacy, packed`. Each block has 100 warmup
calls and 1,000 measured calls. Synchronize CUDA before/after each timed block;
include all candidate packing/allocation/copy cost. Do not profile/instrument
these timed blocks or time only a prepacked cache.

Before timing, inject the 33 faults only into owned copies, compare exact errors
on CUDA, and never modify simulator views. Retain the original CPU tensors and
their SHA256. After timing, require every input bit unchanged, zero integrator
ticks and zero simulation time. Retain every ordered timing block separately.

## Evidence and interpretation

Retain runner/checker bytes, host/source/runtime/CPU-preflight launch, compiled
plant receipt, original input tensor payload, six blocks, all 33 CUDA error
records, summary, child log/status/telemetry and report inventory. The CPU
supervisor must rehash/replay the exact complete file set and deterministic
summary before publishing a successful report. A separate Mac CPU replay uses
the independently supplied launch/report/tool hashes and the current reviewed
checker, not arbitrary code from an evidence folder. Missing, malformed,
nonfinite, reordered or inconsistent evidence fails closed.

Success is only `finite-check-probe-complete-not-runtime-equivalence`. Report
each adjacent pair's packed/legacy timing ratio and whether all three ratios
are below one. This is a descriptive same-input microbenchmark, not statistical
proof, an end-to-end speedup or a new timing qualification. CPU replay cannot
authenticate the CUDA run or independently reproduce its measured timings.
All runtime integration, training, timing, policy, stance, football and physical
motion admission flags remain false.

Even a positive result requires a separately reviewed opt-in runtime integration,
unchanged live invariants, source-bound device checks and fresh full collection
plus optimizer timing before any learner can be launched. A negative result
closes this candidate; do not weaken checks, revise thresholds to fit measured
data or enable the rejected graph as a shortcut. No video or physical motion.

## Prelaunch validation

Implementation and focused CPU checks were complete before launch. The
October 1 read-only host check verified exact runtime source, package versions,
dependency trees, assets, driver/GPU identity and lockfile. Two idle observations
showed no compute PID, 0% utilization, 29 C and 687 MiB GPU memory; protected
system/user services were inactive. This is a point-in-time check, not a lease
or permission to overlap later FilmBrain work. The supervisor repeats the guards
under the shared lease immediately before launch.

Mac validation: **408 selected CPU tests passed** across the new inventory and
probe, unchanged community intake/compatibility, WSL qualification/profile/smoke,
stance runtime/contact/throughput and shared GPU/service guards. After final
preflight/replay integration, the 110 new-tool tests were rechecked separately.
Synthetic fixtures construct no native runtime and start no service or policy.
The broader unchanged runtime tests do use the native CPU audit backend. Full
repository tests and CUDA performance/equivalence had not run at that point. A bounded
Luna read-only review checked the candidate and source-separated service wiring;
owner integration made prelaunch CPU fixtures and pre-publication replay
mandatory, clarified initialization-forward work and hardened derived ratios.

## Retained execution result

The reviewed tool source was `80f1749bb1eb47b7ffafb1468cb1e5ee11d5d225`;
the clean runtime checkout remained at `987b452dbfdb4cba7c7fef79f419790bd602ae3a`.
The same 110 new-tool CPU tests also passed on Linux before preparation. One
`microduck-finite-check-probe-80f1749bb1eb.service` invocation
`b59952f8e7944dca9ff5df4a490b0131` completed with `Result=success`,
`ExecMainStatus=0`, inactive/dead and `MainPID=0`. The retained child exited zero
after **18.46121555 seconds**; neither watchdog was extended.

| Adjacent blocks | Legacy seconds / 1,000 checks | Packed seconds / 1,000 checks | Packed / legacy |
| --- | --- | --- | --- |
| 0, 1 | 1.52747920 | 0.15866262 | 0.10387220 |
| 2, 3 | 1.50598293 | 0.17069169 | 0.11334238 |
| 4, 5 | 1.52075444 | 0.16782242 | 0.11035471 |

Decision: **`finite-check-probe-complete-not-runtime-equivalence`**. Fresh
packing was faster in all three same-input pairs, with its copy/allocation cost
included. All 33 owned-copy CUDA fault cases agreed with the original ordered
error. Original input bits were unchanged; integrator ticks, policy inferences
and optimizer updates were zero. Construction still executed its normal eager
forward outside the measured blocks. This does not establish equivalent
trajectories or an end-to-end collection/training speedup.

Retained unchanged on WSL and copied to the Mac under:

```text
artifacts/evaluations/stance-finite-check-probe-80f1749bb1eb/
```

Fourteen files include the two tool archives, launch and plant receipt, original
tensor snapshot, six timing blocks, child log, summary and hash-inventory report.
Independent receipts:

- Launch SHA256: `0c02f566df2ff319fc7de97fc3b671bd2738c4e32e4ad12e70d4e123015d4718`.
- Report SHA256: `49dc484d2700d979bb04431fc902a981b4efc5b233b8f6012d799b77376572c6`.
- Runner SHA256: `f3d87a13a14049d9be2e55a512d967bfd07a43a13e070c4eb213e9fa3ed7ef1d`.
- Checker SHA256: `3551500208e5138a07b511bea20e2b28f1ba4cbc3094ac31cd4a58695a1d801d`.
- Input tensor SHA256: `45053013dca1226b995554271150b02c323ddd7722c2ac5c8412d4110c161d4f`.

The supervisor's pre-publication replay, separate WSL CPU verification and
separate Mac CPU replay all passed the exact inventory, hashes, strict launch
schema, all fault errors and deterministic timing-summary reconstruction. This
is consistency verification, not independent CUDA execution authentication.
Telemetry sampled only the owned child as a compute process; sampled maxima
were **33 C and 1,118 MiB**. Both post-run idle samples had no compute PID and
0% utilization. FilmBrain preview and observatory stayed active with zero
restarts; protected system/user GPU services stayed inactive. No environment,
driver, installed dependency or existing runtime file changed.

The positive checker result justifies reviewing the smallest opt-in integration
next. Runtime integration, timing qualification, replication/training, skill,
football and physical-motion admission remain **false**. Keep the fixed WSL
4,320/4,380-second training/service bounds; do not extrapolate this microbenchmark
into an accepted full-run timing receipt or retry a learner yet.
