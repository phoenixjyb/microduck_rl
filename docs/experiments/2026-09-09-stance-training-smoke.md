# B1-N disposable training smoke: predeclared launch boundary

This follows the [held-out matrix](2026-09-09-stance-heldout-matrix.md) and
[CPU PPO adapter](2026-09-09-stance-ppo-adapter.md). It implements only the
already declared fresh seed-523, 64-world, 16-update integration smoke.
There is no pilot, resume, held-out evaluation, video or promotion CLI mode.

## Fixed implementation and evidence

The actor, critic, rollout storage and Adam optimizer stay on CPU. Only the
owned full-collision Warp plant runs on CUDA0. This device split is explicit;
it preserves the tested CPU learner math and exact deterministic CPU actor
replay. Observations and returned terminal data are copied to CPU, and raw
sampled actions and selected reset masks are copied to the physics device.
No reward, motor stop, policy input, action limiter or reset semantics change.
Python, NumPy and Torch are explicitly seeded before constructing CUDA physics;
the learner still owns its separate seed-523 CPU RNG stream. The old two-to-eight
world/two-update CPU fixture and its non-simulator resume codec remain bounded.

Each of 16 updates collects 24 policy ticks and performs the declared 5-epoch,
4-minibatch PPO update. A tick retains rewards and component sums, executed
substeps, failure/timeout counts, exact terminal records, applied motor torque,
joint speed, tilt, height and soft-limit sample counts. Frozen terminal siblings
are not counted repeatedly in executed physical samples. Tick records are
exclusive and fsynced, so a timeout does not erase the completed prefix.
The rigid plant has no spring-bottoming metric or calibrated motor thermal
model; these are explicitly unavailable, not asserted safe from missing data.

Initialization and all completed update weights are exported with strict smoke
identity and source/runtime/launch hashes. They are diagnostic weights, not
simulator-resume files, and the existing evaluator refuses them as pilot inputs.
Completed-update receipts retain finite optimizer metrics and checkpoint hashes.
No optimizer/simulator continuation is authorized after a timeout or failure.

## Independent supervisor and budgets

The single child has a fixed 900-second watchdog; the user service has a separate
960-second `RuntimeMaxSec` and `KillMode=control-group`. The supervisor verifies
its exact service name, main PID, active state and timeout before spawning.
The existing frozen-map 120-second child limit is unchanged: both wrappers reuse
the same tested process-group cleanup and independent watchdog implementation.
At least another 600 seconds must remain before the September 10 07:30 Shanghai
cutoff. This smoke duration is a hard cap, not a claim it will finish in time.
Insufficient measured throughput must stop progression, not silently extend it.

The parent holds the shared GPU0 advisory lock; the child verifies the inherited
inode and independently confirms that the lease is held before allocating.
Exact clean feature-branch source, host, GPU/driver, frozen package/source-tree
pins, robot assets, actuator parameters, lockfile and compiled plant are bound.
The independently supplied launch hash must match the retained plan bytes.
The parent stays CPU-only, uses allowlisted child environment variables, watches
logs for numerical/backend warnings and checks exclusive compute ownership,
temperature below 80 C and inactive protected system services throughout.
An occupied GPU fails closed; no unrelated workload is killed or restarted.

The final supervisor receipt hashes every retained file and rechecks all 17
weight exports and 384 tick records. It admits only completion of a disposable
training smoke, never nominal-stance performance or football balance. The full
matrix still requires a separately timed and reviewed fresh pilot.

## Validation and launch record

The initial focused set passed 56 tests in 42.62s. The broader local regression
passed 442 tests in 115.26s, with one known Mac process-group cleanup test
deselected; that test remains required on Linux. Markdown HTML parsing passed.
Exact-source Linux verification and retained launch remain pending at this
predeclaration. Synthetic optimizer tests do not establish live CUDA execution
or learned skill.

### First launch: closed before simulator initialization

Implementation `a89cfdf0899d7747dfdf1358aa2f605500e4c6d0` passed all 443 Linux
regression tests in 37.25s, with no skips. Its prepared launch passed CPU-only
host/runtime/plant checks. The retained service then exited unsuccessfully before
importing the child module: the `python -m` supervisor used its runtime
`__name__` (`__main__`) as the child module instead of the canonical import name.
Exact error: `Error while finding module specification for '__main__'
(ValueError: __main__.__spec__ is None)`. There were no simulator, optimizer or
checkpoint outputs, and subsequent GPU readback was idle, 12 MiB, 45 C; both
protected services remained inactive.

Retain the unchanged failed directory `artifacts/evaluations/stance-training-smoke-a89cfdf0899d`:

- Launch SHA256: `5fb5d2e477ebe88f3ca55be58545fd907ed5027c786b1a41ceec3d83f1256245`.
- Runtime SHA256: `b04b28ef35d33581c264d418656d3084218579473790bbf3a9c49754ae243a57`.
- Child log SHA256: `07b2cd129725a1b2d90e96062509f5acdd85b7c8dc2483a80d512117c9e94b8b`.
- Failure report SHA256: `9526dfb23caf903f62516200f17e85ca55a338df4f4993f41e6233d9b58584c1`.

All four files are mirrored on the Mac under the same relative path with
matching hashes. The corrected smoke suite passed 20 local tests in 15.69s.

The correction binds a literal module name. Tests now check the exact child
command under a simulated `__main__` namespace and run real CLI help in a CPU
subprocess. After exact-source Linux tests, permit one separately named attempt
under the corrected commit and the same unchanged numerical protocol, seed,
budgets and limits. This is a diagnosed launcher correction, not an automatic
retry or restart of the failed unit; do not overwrite the failed evidence.

### Corrected attempt: completed disposable CUDA-physics training smoke

Corrected implementation `d7eb23c0eb117d7bbf15b8e0ab08c6e325051d73` passed
all 444 Linux CPU tests in 44.23s, without deselection or skips. The independently
timed `microduck-stance-smoke-d7eb23c0eb11.service` completed successfully:
MainPID 0, ExecMainStatus 0, active/exited (retained receipt, not running).
Only child PID 376926 owned CUDA; protected system services remained inactive.

The run completed all 16 PPO updates and 384 policy ticks across 64 worlds.
The child took 99.858s including initialization/closeout; the collection/update/
evidence loop took 90.215s. All optimizer metrics were finite. Final metrics:
value loss 0.0666856, surrogate loss -0.0212075, entropy 2.1062493. These are
training diagnostics, not held-out scores or independent performance acceptance.

The 380 retained episode terminations all crossed the unchanged 0.35-rad tilt
stop. Mean duration among terminated episodes was 537.121 physics steps
(1.074s); this excludes unfinished episodes and is not a frozen-policy result.
No episode reached the five-second timeout. Maximum recorded tilt was
0.351771 rad, including the first failure boundary; no subsequent failed-world
motion is inferred or permitted before the explicit training reset.
Maximum applied motor torque was 0.1393695 Nm, maximum joint speed 0.700503 rad/s,
and soft-limit exposure was 0 of 3,416,504 executed joint samples. There were no
recorded torque, joint-speed, root-speed, height, hard-limit, forbidden-contact,
support-loss or proposed-torque terminal causes. This short nominal smoke does
not establish real motor safety, thermal limits or learned balance.

The independently sampled GPU temperature peaked at 57 C. After completion,
readback was 0% utilization, 12 MiB, 48 C with no compute PID. Both protected
services remained inactive and no unrelated service was changed.

Retain `artifacts/evaluations/stance-training-smoke-d7eb23c0eb11` on both hosts:
421 hashed files plus the final report (12,128,312 bytes total), including
initial weights, all 16 update weight exports, all 384 tick records, update
receipts and source/runtime/launch metadata. A separate post-run Linux CPU
verification restored every export and checked exact inventory and every hash;
CUDA stayed uninitialized. The Mac copy independently matched all file hashes
and the final report hash. Neither copy is a simulator-resume checkpoint.

| Evidence | SHA256 |
| --- | --- |
| launch.json | `1de98f53ed11034acb1855517b7e3c839cbb355495a3168e54febcb7ac92e861` |
| runtime.json | `e676cb51f373d6d7929e1968376c9339535135813e947a75893c78c0b89a5de3` |
| completed.json | `9512edea14d77b6465539082f1b880799d3413f40bb48e5786001d20fe7b4f4d` |
| report.json | `f6b1163311dff30461d672d1124304e486e3439e6e62786ca137e13ac974f46d` |
| initial.pt | `c000342aa527199d038c9f92ee967a5c10c30b1a61585d4aefa10d0ad9d8556c` |
| model_15.pt | `3450ae340826af1d0584a55e496c5131c65db38fa199b2961fabd1d41aa5b0fe` |

Decision: `disposable-training-smoke-complete-not-capability`. Do not reuse
these smoke weights as a pilot parent or admit stance, hopping, obstacle
composition or football balance from this run. Existing skill policies remain
untouched. No full pilot, held-out matrix, MP4, raw perception or physical motion
was launched by this chunk.

### Next gate: measured throughput before a fresh pilot

Measured cost was approximately 5.64s per update at 64 worlds. Even holding
that cost constant, 512 updates would take approximately 48.1 minutes, beyond
the declared 30-minute pilot service cap. This is an arithmetic scenario, not
a measured prediction at 512 worlds; larger-batch performance remains unknown.
Do not silently lengthen the cap, shorten the experiment or launch an unsafe
pilot. Next profile the collection/synchronization/evidence path and test
bounded throughput improvements while preserving every physical stop and
terminal record. Measure the actual planned batch before judging whether a
fresh pilot fits its reviewed budget. No second training job is predeclared by
this result, and no reward or numerical acceptance gate has changed.
