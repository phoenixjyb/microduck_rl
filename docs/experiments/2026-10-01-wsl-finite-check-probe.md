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

## Current state

Implementation and focused CPU checks are complete; **not launched**. The
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
repository tests and CUDA performance/equivalence were not run yet. A bounded
Luna read-only review checked the candidate and source-separated service wiring;
owner integration made prelaunch CPU fixtures and pre-publication replay
mandatory, clarified initialization-forward work and hardened derived ratios.
