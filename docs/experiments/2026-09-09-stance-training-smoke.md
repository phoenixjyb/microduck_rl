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
