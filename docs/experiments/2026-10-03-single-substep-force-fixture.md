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
GPU execution. Subsequent Linux and actual CUDA results are retained below.

### Initial Linux regression qualification memory refusal

The exact-source 155-test WSL regression qualification at
`6d9f950254cc2127361c3384065217c8f19a1b42`, service
`microduck-d0-original-cpu-tests-6d9f950254cc.service`, invocation
`be6c7e2af3334e19b0bdf15d58d9114c`, was killed by its **2-GiB cgroup cap**:
`Result=oom-kill`, `ExecMainStatus=9`, after 26.084 CPU seconds. No GPU capture
or main-worktree installation followed. The failed service and journal remain.
Read-only diagnosis found 121 completed test dots, placing the interruption in
the unchanged historical forward-probe snapshot/revalidation fixtures, not a
reported D0 assertion failure. The host had approximately 14.6 GiB available,
and the GPU remained idle (30°C / 663 MiB / no compute PIDs).

Separate qualification budget: the complete historical regression suite gets
a new **120-second / 6-GiB / 200%-CPU / Nice-10 / control-group CPU-only service**,
consistent with the earlier full-evaluator regression budget. The three new D0
suites must also pass separately under the real portable WSL profile and the
original **120-second / 2-GiB** qualification cap. No code, assertion, numerical
tolerance, GPU fixture limit or accepted result is changed by this test-service
separation. Both successful terminal qualifications remain mandatory before
the original 180-second / 2-GiB D0 GPU service may start.

### Exact-source WSL qualification

Both separate qualifications completed successfully without CUDA. The unchanged
155-test regression suite passed in **33.49 s** under the declared 6-GiB CPU
test budget, service `microduck-d0-regression-cpu-tests-6d9f950254cc.service`,
invocation `c9f6640ea56d4448bc9ee6f9bb8ae5c9`. The three D0 suites passed
**75 tests in 10.75 s** under the actual portable WSL profile and 2-GiB cap,
service `microduck-d0-portable-cpu-tests-6d9f950254cc.service`, invocation
`ae0851ca5b9f4916beeeccfca97790e2`. Both terminal results were success / exit 0.

After verifying the fork tip, clean frozen worktree and shared lease plus two
idle samples, the WSL feature worktree was fast-forwarded to exact source
`6d9f950254cc2127361c3384065217c8f19a1b42`. CPU preparation completed under
the original 180-second / 2-GiB cap, invocation
`800548e0d93d40b7af8c0beb1b09f78b`. Its five-case capture and fresh physics
replay took **3.925283435964957 s**, with maximum solved-field difference **0**
and CUDA uninitialized. Actual compiled binding: **16 bodies**, `trunk_base`
at body ID **1**; retained CPU capture **232,087 bytes**. The source-unique
launch SHA256 is
`c80cbc3ac2d63a65ebafd1b5a639cb200f597df67ff12733ccd5392a71c558f2`.

The original bounded CUDA fixture was launched as
`microduck-wsl-d0-force-fixture-6d9f950254cc.service`, invocation
`7022e182252043bdb152c7b01d7f6092`. Its terminal decision and independent
CPU replay are not inferred from these CPU qualification results.

### Terminal CUDA result and fresh CPU replay

The service completed successfully at approximately **04:03 Shanghai**, exit
0, decision **`single-substep-force-path-replayed`**. Whole service elapsed
**37.11937826592475 s**; owned child PID **2053262**, supervised elapsed
**19.234264987986535 s**, capture elapsed **16.098498686915264 s**. Actual
backend was Torch `cuda:0` / Warp CUDA. All five declared fresh cases and five
Euler steps completed, with **zero policy inferences and optimizer steps**.

After authenticating the complete 232,087-byte CUDA payload, the CPU-only
supervisor freshly compiled and reran all five physical cases. Exact layouts,
integer rows, force matrices, controls and case order passed; all floating solved
fields passed the fixed `atol=1e-5`, `rtol=1e-4` comparison. Maximum absolute
solved-field difference was **2.0503997802734375e-5** (combined absolute/relative
tolerance, not an absolute-only gate); CUDA was uninitialized during replay.
No tolerance, case, cap or numerical gate was changed after capture.

All 16 ownership samples showed only the owned CUDA child, protected services
inactive, maximum **32°C / 972 MiB**, minimum **23,190 MiB free**. FilmBrain
states/PIDs/restart counts matched their retained launch values. Two fresh
post-closeout samples confirmed no compute process, 0% utilization, **31°C /
663 MiB**. The transient service is terminal/inactive, and artifacts remain in
`artifacts/evaluations/stance-wsl-d0-force-fixture-6d9f950254cc` on WSL.

| Retained file | SHA256 |
| --- | --- |
| launch.json | `c80cbc3ac2d63a65ebafd1b5a639cb200f597df67ff12733ccd5392a71c558f2` |
| cpu-qualification.json | `eb14762d8dd25a7b54d7a2e2015abfe8fbb78695a94a71d279ebfed51e2be839` |
| cpu-capture.pt | `4b82126a4c8fcd2f18948bb266b7fa075c3862749230d44a36583c50bf7ffc5e` |
| capture.json | `348e9eda64826367f84d02783ed6dad9e062fb70290d206684ec372910468806` |
| capture.pt | `823f2902a5352c7b78ef2c62dcbd117995581644566502f33cfb256b9b7305f9` |
| child.log | `923756654d29bdd94a5b1ef697ce1964cb465ba77bd2398a94398f92a4622377` |
| report.json | `33743033bf292921d09df7ac682c4dd16910af03b10318fabf9137fd72ae05c4` |

This closes **only the separate single-substep force-path fixture**. Every
recovery, training, learned-stance, football, physical-motion, independent-GPU
attestation and complete-binary-equivalence admission flag remains false. No
policy responded to the push. The next legitimate question is how a fixed,
already replicated stance policy responds over a complete recovery interval;
that needs the separate D1 protocol described above.
