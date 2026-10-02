# D1: frozen stance-policy push-recovery baseline

This is a new, separate protocol after the retained
[D0 force-path fixture](2026-10-03-single-substep-force-fixture.md) passed on
WSL/CUDA with fresh CPU physics replay. D0's source was
`6d9f950254cc2127361c3384065217c8f19a1b42`, report SHA256
`33743033bf292921d09df7ac682c4dd16910af03b10318fabf9137fd72ae05c4`.
Nominal stance replication passed all 36 cases / 4,608 attempts. Neither result
establishes recovery, hopping, football balancing or physical readiness.

## Preselected policy and pulse

Before any disturbed-policy result, select **training seed 577, iteration 255**,
original source `be2d59661af293b0d67ae20d2e16db50514cce14`, file `model_255.pt`,
SHA256 `2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`.
No checkpoint search, optimizer, weight update or observation expansion. Keep
the exact **44D actor / 10 actions / 14 XL330-m6 motors**, CPU deterministic
actor inference and separate historical AVX2 archive authentication versus
portable CPU inference/replay. No raw perception or physical motion.

Protocol `football-b1d-frozen-recovery-v1`; new trace namespace
`football-b1d-frozen-recovery-trace-v1`. Five ordered independent **one-world**
cases: zero-wrench, +x, −x, +y, −y. Same nominal floor reset, fixed seed **619**,
same frozen policy, one **5-second / 250-policy-tick / 2,500-Euler-step** first
attempt. These repeats do not imply randomized generalization.

The four nonzero cases receive **2 N at world-frame trunk inertial COM**,
zero torque, for pre-Euler counters **500 through 509 inclusive**: onset
1.000 s, duration 0.020 s, impulse **0.040 N·s**. D0 qualified one 2-ms
substep; this deliberately small next lesson extends duration to ten substeps,
not magnitude. Clear both full applied-force arrays immediately after each
Euler candidate's owned integrated-state capture, before every fresh unforced
post-solve; an exception in the forced solve, phase capture or Euler also clears
the owned installed pulse in `finally`. No wrench on rejected
or already-terminal rows; a failure before onset remains a failed/undelivered
first attempt, never a reset followed by a successful push.

The nominal runtime and its permanent-fault force guard stay byte-identical.
A separately named recovery adapter uses an explicit pulse-step API, never
a generic allow-forces flag or patched nominal forward method. It reuses the
existing BAM/FIFO/Euler/transition/terminal primitives and preserves their
accepted-row masks, hard stops and immutable first-terminal state. The ordinary
nominal step/forward API does not acquire recovery semantics.

## Evidence and numerical gates

Retain every physics-boundary state/observation, policy input/action, motor
proposal/commit, full applied-force arrays before and after each Euler, and
complete solved pre/integrated/post pulse-window phases. Bind the exact selected
compiled plant and actual ordered body-name table/ID. Independent CPU checking
must authenticate retained bytes before tensor loading, replay every CPU actor
inference and BAM/FIFO accounting, reconstruct every scheduled force matrix,
and reapply the unchanged complete-first-attempt numerical scorer. Recorded
state/control scoring is **not fresh whole-trajectory physics re-simulation**;
the report must state that distinction. D0's separate full physical replay
remains pinned. No independent GPU attestation/binary equivalence claim.

Keep the original hard limits and first-failure semantics: tilt >0.35 rad,
height <0.08 m, root speed >1 m/s, applied/proposed motor torque >0.36 N·m,
joint speed >10 rad/s, hard joint limits/forbidden contacts, or unsupported
foot after the original grace period. Final baseline scoring remains unchanged:
maximum displacement ≤0.02 m, soft-limit exposure ≤0.01, final-second tilt
p95 ≤0.0873 rad, planar-speed p95 ≤0.03 m/s, minimum height ≥0.105 m, and
both-feet support fraction ≥0.99 after 0.2 s. Full delivered-pulse count and
unforced post-solve checks are additional gates, not replacements. Report
which gate fails; do not relax it to make the frozen baseline pass.

The eventual five-case baseline must also show identical resets, full matched
pre-pulse state/action prefixes, and the zero-wrench control passing. After
the push, **reactive policy actions may differ**; match weights/reset/protocol,
not action streams. Report transient tilt/speed, displacement, motor torque,
power/soft-limit and settling-window metrics without claiming a thermal model.

## Bounded timing probe first

Before a five-case GPU baseline, run only the fixed **+x one-world full-length
case** as a timing/retention probe. It cannot accept recovery or authorize a
learner regardless of its numerical diagnostic. First qualify the tested exact
source and actual CPU pulse adapter; preserve D0 hashes and all original archive
authentication gates. Source-clean WSL installation, unchanged FilmBrain states,
shared lease plus two idle samples are mandatory.

Before installing the new source into the training worktree, use an ignored,
detached exact-source snapshot for two independently capped CPU qualifications:
the integrated D0/nominal and D1 regression suites under default CPU math at
**150 seconds / 6 GiB / 200% / Nice 10 / control-group**, followed by the four
D1 suites under the actual portable profile at **120 seconds / 2 GiB / 200% /
Nice 10 / control-group**. CUDA remains hidden. This separate historical-fixture
test budget does not widen the 2-GiB probe budget. Preserve any failed service;
do not retry with larger caps after seeing its outcome. Require both successful
terminal receipts before a clean exact-source fast-forward, with no dependency
or driver changes. CPU preparation itself has a separate **180-second / 2-GiB**
service cap with the same CPU/Nice/control-group settings.

CPU preparation first runs the selected frozen actor on the real CPU adapter for
**52 policy ticks / 520 substeps**, covering approach, all ten pulse substeps
and the first ten post-pulse substeps, within **120 seconds**. Retain those
complete prefix bytes and an independently rerun exact actor/control/force
receipt; require complete pulse delivery before GPU access. This short prefix
cannot pass the full five-second recovery scorer. Every subsequent entry
reauthenticates the original seed-577 archive in isolated AVX2, checks the actual
portable profile, repeats the CPU prefix actor/control verification, and freshly
reruns all five physical D0 cases from their byte-authenticated retained CUDA
payload. The D0/nominal primitive source hashes remain pinned and unchanged.

The existing bounded timing-probe pair is **900-second child / 960-second
service**, with **2 GiB / 200% CPU / Nice 10 / control-group**. Separate
180-second closeout and 60-second deadline margin; refuse unless all **1,200
seconds** fit before the fixed **2026-10-03 08:00 Shanghai** cutoff. Use a
CUDA-hidden supervisor; only its inherited-lease child may use CUDA0. Authenticate
at most **128 MiB** per case before loading. No automatic retry, overwriting or
cap extension. A full matrix budget must be separately derived from retained
actual probe construction/collection/replay/closeout timings and committed
before launch; do not assume D0's one-step throughput predicts this episode.
A partial or early-failed capture is retained and independently scored, but
cannot receive a successful full-length timing decision or supply a matrix
budget. The fixed complete-duration gate must be true before timing admission;
failure reports retain the collection reason and numerical gates in error notes.

Stop a failed probe, preserve evidence, diagnose read-only and change only a
reviewed source/predeclaration with tests. Do not start a learner, combined-skill
promotion, video or ball support from this timing result. At 08:00 start no new
Duck work; confirm owned processes finished and evidence durable, leaving
protected services inactive unless the owner separately requests restoration.

## Source qualification before Linux results

The four new modules keep the eight pinned D0/nominal primitives byte-identical.
Owner review removed an extra constructor preparation that would have changed
the nominal reset, made pulse evidence tick-local, checked entry-time body/case
binding and graph exclusion, and added forced-phase exception cleanup tests.
The runner reuses the existing fixed 900-second stance watchdog rather than
widening the generic 120-second subprocess wrapper. A negative test caught and
closed partial-capture timing admission before any D1 GPU access.
Final read-only review additionally required the exact 52-tick / 520-substep
CPU prefix on both preparation and revalidation, and extended force cleanup to
cover exceptions during pulse installation/setup itself. Six negative prefix
tests and two after-copy failure tests cover those corrections. The initial
`f3d8466e` snapshot was transported but never qualified or executed on WSL;
only the corrected exact source may proceed to Linux qualification.

The final Mac CUDA-hidden integrated suite passed **278 tests in 27.38 s**:
123 new D1 contract/runtime/trace/runner checks and 155 unchanged D0, nominal,
forward-probe and portable-full-protocol regression checks. Python compilation
and whitespace checks passed. These are source/CPU checks, not Linux execution,
GPU recovery or skill acceptance. Exact-source Linux qualifications and the
bounded timing probe remain pending at this predeclaration commit.

## First WSL qualification and bounded test-lifetime revision

At exact source `5a0d5b5fbb543079d63aed8454dc58647f1f5cf9`, the default-math
integrated WSL suite passed **278 tests in 44.00 s** under the declared
150-second / 6-GiB budget. Service
`microduck-d1-regression-cpu-tests-5a0d5b5fbb54.service`, invocation
`1ecefaa8e8e14f2db9d3f2a8165eb183`, exited successfully; journal reports
71.507 s CPU and 5.7 GiB peak memory.

The single-process portable-profile suite did **not** qualify: service
`microduck-d1-portable-cpu-tests-5a0d5b5fbb54.service`, invocation
`484c77a5ad5546329bfdb279e214e53e`, hit its actual 2-GiB cgroup ceiling at
04:50:24 Shanghai: **`Result=oom-kill`, `ExecMainStatus=9`**, after partial test
progress; no assertion error or completed suite was reported. Its failed unit
and journal are preserved. Initial-exec profile inspection succeeded, but that
does not qualify the suite. The GPU remained idle (0%, 30 C, 663 MiB, no CUDA
PID); FilmBrain PIDs/restarts and both protected services were unchanged. No
training-worktree installation or D1 GPU execution followed this failure.

Read-only source inspection found environment-bound synthetic refresh hooks
and native fixture graphs whose cyclic Python lifetimes can outlast an
individual test. Accumulation is a plausible cause, not an independently
measured allocator attribution. The reviewed change is test-only garbage
collection after completed fixtures/monkeypatch teardown, not an active-runtime
reset or changed physics/gate. No dependency, driver or GPU cap changes.

Before new qualification outcomes, predeclare the corrected-source portable
qualification as **four sequential, fresh one-file user services** (contract,
runtime, trace, probe), each retaining **120 seconds / 2 GiB / 200% / Nice 10 /
control-group**, exact source and actual initial-exec portable profile. Preserve
all failures; no automatic retry or larger cap. All four must finish
successfully, together covering the same 123 tests, plus a fresh corrected-source
278-test default-math regression under the unchanged 150-second / 6-GiB cap,
before installation or GPU access. This bounds cross-test/suite fixture
lifetime; it does not widen the 2-GiB CPU-preparation or GPU-probe budgets.
The corrected test-lifetime source passed the same **278 Mac CPU-hidden tests
in 42.84 s**, with whitespace checks clean, before any new WSL qualification.

## Corrected-source qualification and first GPU writer refusal

Exact source `2b8ba547853f2b207c420b659a953492e46665ea` passed the integrated
WSL suite (**278 tests, 48.23 s**; service
`microduck-d1-regression-cpu-tests-2b8ba547853f.service`, invocation
`3adc59c79b1d423bb4d8457e07dad71c`). All four actual-profile, 2-GiB one-file
services also exited successfully: contract **51 / 0.04 s**, runtime **24 /
10.04 s**, trace **16 / 7.64 s**, probe **32 / 0.06 s**. Their service runtimes
were 8.562, 16.617, 14.224 and 6.615 s respectively; invocations in that order:
`a50caeaa957545bc8ecc8dad63cd1116`, `cde2794d202b4d45a55f976d37ba4262`,
`18119b15a9df4af38ebabcaa243aa86a`, `3fea338c77c140d0ad37cc8853651f51`.
Some terminal systemd memory-peak summaries were implausibly small; they are
not used as allocator measurements. The configured limits and successful
terminal results, not those peak numbers, qualify these separate test runs.

After a clean exact-source fast-forward under the shared lease and two idle
samples, CPU preparation service `microduck-d1-cpu-prepare-2b8ba547853f.service`
(invocation `7f4793620a8046c99f42dd6adbbacc42`) passed: **52 ticks / 520 steps,
ten delivered pulse and ten post-pulse substeps, exact actor replay error 0**;
qualification 7.433390758931637 s, collection 5.294208843959495 s. Prefix bytes
8,118,535; no hard failure, soft-limit exposure 0, peak modeled torque
0.11519977450370789 N m. This prefix still cannot pass the five-second scorer.

Retained directory:
`/home/yanbo/work/microduck_rl-stance-replication-20260930/artifacts/evaluations/stance-wsl-d1-frozen-recovery-probe-2b8ba547853f`.
SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| launch.json | c064f3abaeacd63a95c4b92b2155298e620c4c2ece07d435c658ebebd124607a |
| cpu-qualification.json | 1ee623fd03a0311633f6e75586d3934f662737410f2eadb651d39e45d1681524 |
| cpu-prefix.pt | f474da111f4faa1d17eb96a7f57ce897f2500c8344ac1db9ab262960da00bd05 |
| checkpoint.pt | 2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5 |
| report.json | 35a2880a99c17b2e2033042dba57049c4d604f682d5f25b37433fe02c959c5a4 |
| child.log | 14f7c794baa0dbf3fd0fa9dc4eab632f25de2a771447100f017a6a124b355354 |

The GPU service `microduck-wsl-d1-recovery-probe-2b8ba547853f.service`, invocation
`c58c519ea3484993a40fe1d490226382`, failed normally with exit 1, not OOM. Its
exact child error was **`ValueError: bounded smoke bytes`**, at
`stance_recovery_probe.child` → `stance_training_smoke.write_bytes`. Read-only
source diagnosis showed a mismatch: D1 encode/read/declaration allow **128 MiB**,
but that reused legacy writer allows only **16 MiB**. Serialization reached that
writer, but **no capture.pt or capture.json was retained**. Therefore no complete
duration, numerical recovery result or full-matrix budget is accepted from this
attempt. Its 192.22105568787083-s child elapsed time is a failed-attempt timing,
not qualified complete-case timing.

All 153 ownership samples contained no GPU PID or only child **2073789**;
maximum GPU temperature 41 C, maximum GPU memory 974 MiB, minimum free
23,188 MiB, protected services inactive throughout. Read-only inspection during
the service showed cgroup reclaim/swap pressure but `oom=0`, `oom_kill=0`;
no memory/time cap was changed. After failure the GPU was idle at 0%, 33 C,
663 MiB, no CUDA PID; FilmBrain remained at its original PIDs/restarts. The
failed unit, source, prefix, report and child log remain durable and untouched.

Before another outcome, predeclare a **new exact-source, fresh-path writer-fix
attempt**, not a retry of that failed service/directory: D1 gets its own exclusive
fsynced capture writer enforcing the already-declared 128-MiB limit. The
legacy smoke writer remains byte-identical at 16 MiB; checkpoint writes retain
that smaller helper. Tests must cover writing **16 MiB + 1 byte**, no overwrite,
invalid bytes and rejection above 128 MiB, plus the child routing to this writer.
Require the new exact-source integrated regression and all four separate actual
portable suites first (now **282 / 127 tests** respectively), with the unchanged
150-second / 6-GiB and per-file 120-second / 2-GiB caps. Then new CPU preparation
and the same **900 / 960 seconds, 2 GiB, fixed checkpoint/pulse/gates and cutoff**
may proceed. No automated retries or full matrix before a retained, CPU-scored,
full-length probe succeeds. The Mac integrated writer-fix suite passed
**282 tests in 30.32 s** before this new WSL declaration; the final focused
runner check after the explicit routing assertion passed **36 tests in 8.24 s**.
Whitespace checks passed. The unchanged legacy smoke-writer source SHA256 is
`43186854a9d20197e88e5f68000c844f45717aa459d5f3b21675393fb46aeb57`.
