# D0: single-substep external-force physics fixture

This separate predeclaration follows the completed, retained
[nominal replication](2026-10-03-portable-packed-full-evaluation.md): all 36
cases / 4,608 attempts passed, the whole-matrix CPU closeout passed again, and
all 329 files / 22,352,528,834 bytes were hashed and rechecked. Work is authorized
only through **2026-10-03 08:00 Asia/Shanghai**. This is a small force-path
qualification, **not a recovery learner, policy evaluation or skill promotion**.

Protocol: `football-b1d-single-substep-force-fixture-v1`. New contract, fixture
and runner have their own namespace. The nominal `WarpStanceRuntime` and its
permanent-fault external-force guard remain unchanged. The new fixture owns
a separately compiled full rigid Duck / floor / XL330-m6 plant; it never
instantiates or subclasses the nominal runtime or toggles its checks.

## Frozen experiment

Five ordered fresh **one-world** cases: zero-wrench, +x, −x, +y, −y. The four
nonzero vectors are **2 N** horizontal force and zero torque, applied at
`trunk_base`'s inertial center of mass in the **world frame**, ordered
`[Fx, Fy, Fz, Tx, Ty, Tz]`. Resolve the actual compiled body by name, retaining
the entire ordered body-name table and its ID; do not infer WSL IDs from a Mac
compile. Each pulse spans exactly **one 0.002-second Euler step** (0.004 N·s).
There are no actor observations, policy actions, checkpoints or optimizer steps.

Each case starts at the same floor-placed nominal reset and performs one real
zero-error BAM preparation to populate motor friction/damping. Freeze those
fields and the resulting zero control for the entire case. This matched
zero-wrench control isolates the force path; it does not demonstrate closed-loop
recovery or the capabilities of the retained stance checkpoint.

The exact order is: retain prepared unforced state → set the declared full
wrench matrix → forced forward solve → retain solved state → Euler once →
retain integrated state → **clear all xfrc/qfrc before any post-forward** →
unforced forward solve → retain fresh post-state. The integrated capture's
solver fields still describe the pre-Euler solve; only the post-forward capture
contains newly solved post-step derivatives. The physical clock advances once
and post-forward must not change integrated qpos, qvel, time or warmstart.
No reset/reuse is permitted; an input, solver, capacity or timing error faults
the owned fixture permanently. Any nonzero generalized force, off-body input,
vertical component or torque, malformed/nonfinite input, or undeclared case is
refused. The declared horizontal pulse must change solved acceleration relative
to the zero-wrench case; this is a causal-input check, not a recovery threshold.

Force semantics were checked against official
[MuJoCo 3.10.0 source](https://raw.githubusercontent.com/google-deepmind/mujoco/3.10.0/src/engine/engine_core_smooth.c)
and read-only installed MuJoCo-Warp 3.8.1 source on WSL. Warp forward reads but
does not clear applied arrays; Euler consumes the solved acceleration. Retained
installed-source SHA256:

| Module | SHA256 |
| --- | --- |
| support.py | `3ac8475d5e41318d8289601ad41493245d95d2138023d47e0462f04b6c67b94a` |
| smooth.py | `63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f` |
| forward.py | `c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3` |
| types.py | `8f0b19d7b2bc039a419fa547ba6b732837327c0735f688ad34ace5afc4843e50` |

## Replay and execution gates

Retain every phase's complete force matrices, controls, motor friction/damping,
qpos, qvel, time, warmstart, solved acceleration/forces, contacts, solver counters
and active constraint rows. CPU replay recompiles and reruns all five complete
forward/Euler/clear/post-forward cases; recorded-state scoring alone is not
replay. Exact layout, integer rows, case order, controls and external inputs are
required. Float32 solved fields have the **predeclared separate fixture**
`atol=1e-5`, `rtol=1e-4` cross-backend comparison. This does not relax any nominal
stance, recovery, motor, hopping or football gate. Refusal is retained for
read-only diagnosis; do not tune tolerances to a failed CUDA result.

First run focused contract, real CPU-fixture, runner and nominal force/integrator
regression tests on Mac, then qualify the exact source snapshot on WSL with
CUDA hidden. Commit and push the predeclared tested source to the fork feature
branch. Only after clean exact-source installation, pinned nominal closeout
receipts, compiled plant/profile qualification, FilmBrain state checks and the
existing shared lease plus two idle samples may a single CUDA child run.

User service: **180 seconds / 2 GiB / 200% CPU / Nice 10 / KillMode control-group**;
owned child watchdog **120 seconds**; separate closeout reserve **180 seconds**
and 60-second deadline margin. These are conservative limits for five single
substeps, not estimates of a recovery lesson's throughput. Refuse entry unless
the entire 420-second reservation fits before 08:00. A CPU-only supervisor
authenticates at most 16 MiB of retained tensor bytes before loading, runs the
fresh CPU physics replay, and records terminal source/hash/backend/timing/GPU
ownership evidence. Only its inherited-lease child may use CUDA0; no overlapping
compute, package/driver change, protected-service restoration or other workload
mutation. FilmBrain service state/PIDs/restarts must remain unchanged.

Preparation itself also captures all five CPU cases and runs a fresh CPU physics
replay, requiring **exact same-backend repeatability**. Retain `cpu-capture.pt`
and `cpu-qualification.json`; bind the qualification SHA256 into the launch.
Every `checked()` call authenticates those bytes before tensor loading and
re-executes the source/plant/profile-bound CPU qualification before any CUDA
capture. This enforced gate was added after owner review found that the initial
runner relied only on external test records and replayed CPU physics after GPU
capture. No GPU run preceded the fix. Failed preparation leaves its owned
partial evidence in place and never acquires the GPU lease.
CPU qualification must take less than **30 seconds**, and every repeated CPU
qualification has that same fixed bound. Immediately after child entry checks,
require at least **60 seconds for capture plus 10 seconds for child closeout**;
refuse before CUDA work if that reserve has been consumed. The outer watchdog
and service cap remain independent and unchanged. These bounds are declared
before WSL/CUDA results, not extended to rescue a slow run.

Output is new and source-unique:
`artifacts/evaluations/stance-wsl-d0-force-fixture-<source12>`.
Exclusive fsynced writes; no overwrite, reuse, automatic retry or cap extension.
The launch binds the selected compiled plant, actual body table, source, frozen
host/stack, portable CPU profile, and exact completed nominal evidence hashes.
All recovery/training/stance/football/motion/GPU-attestation/binary-equivalence
admission flags remain false. A successful decision may only be
`single-substep-force-path-replayed`.

## Following gate

Only after D0 succeeds should a **separate D1 closed-loop frozen-policy recovery
baseline** be predeclared: fix one checkpoint before observing disturbance
results, keep the 44D actor/10 actions and motor limits, declare pulse directions,
onset/duration/strength, full first-attempt/fall and post-pulse settling gates,
and measure a bounded held-out case before predicting a training/evaluation
matrix. Reactive actor actions will differ from the zero-wrench control after
perturbation; do not call them identical action streams. A future learner needs
its own training/retention objective, matched budget and checkpoint/held-out
protocol. D0 alone admits none of that work and establishes no hopping or ball
balancing capability. No video, raw perception or physical motion.

## Initial local checks

Real local CPU capture and a fresh five-case CPU solve/Euler replay passed with
maximum solved-field difference **0**. The combined contract, fixture, nominal
runtime and Euler suite passed **85 tests in 13.93 s**. The fixture tests include
retained input ownership, complete force clearing, one-use/permanent faults and
mutated state/input/binding/order rejection. These are Mac source/CPU checks,
not WSL qualification or CUDA evidence. Final owner integration checks passed
**155 tests in 18.56 s**, covering the new contract/real fixture/guarded runner
and unchanged nominal runtime, Euler, forward-evidence and full-matrix contracts.
Python compilation and diff checks passed. Luna's bounded contract/tests and
read-only integration review exposed the pre-GPU CPU-qualification ordering gap;
the owner fixed it and added the tested CPU/child entry reserves before any WSL
GPU execution. Linux qualification and actual CUDA evidence remain pending.
