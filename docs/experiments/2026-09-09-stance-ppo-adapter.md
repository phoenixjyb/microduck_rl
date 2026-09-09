# B1-N CPU PPO transition and learner-state precursor

This bounded chunk follows [control-path evidence](2026-09-09-stance-control-evidence.md).
It is not the declared 64-world smoke, 512-world pilot or a new learned Duck
capability. It changes neither existing checkpoints nor the held-out gates.

## Rollout semantics

The caller-owned CPU harness uses the fresh seed523 actor44/critic50 models,
24 policy transitions per update and the declared fixed Adam/PPO settings:
learning rate3e-4, five epochs, four minibatches, gamma0.99, lambda0.95, clipping0.2,
value coefficient1, entropy0.005 and gradient norm1. Stock RSL clips actor and
critic gradients separately. Source byte hashes pin RSL5.0.1 PPO and storage in
addition to the previously pinned model sources. No installed package is edited.

Raw sampled actions and pre-action observations are retained in PPO storage;
motor clipping/slew remains inside the existing runtime. A step's observations,
reward and terminal evidence are collected before resetting only finished worlds.
The wrapper verifies that reset returns those terminal records, then reads the
next observations with only the finished rows reset.

Timeout bootstrap is computed from `V(terminal observation)` before reset.
The stored learning reward is `raw_reward + gamma * V(terminal)` for timeouts
only. Falls receive no bootstrap, and contradictory fall/timeout flags fail.
Stock RSL's `time_outs` argument is deliberately omitted because its implementation
uses the pre-action value. This bootstrap is a learning target, not an environment
reward or timeout-success bonus. Raw rollout results remain unchanged.

GAE masks continuation at every done boundary and normalizes advantages globally
over the full rollout, not per minibatch. A local subclass checks finite values
without inheriting the historical task package's `nan_to_num` return patch.
Invalid rewards, observations, advantages or gradients fault the harness.
Gradient checks run before Adam steps; updated parameters/moments are checked
after each step. A failed later minibatch can leave partial updates, but the
faulted learner cannot continue or publish a checkpoint. It must be discarded,
preserving any previously retained healthy checkpoint.

The harness is intentionally CPU-only, two-to-eight worlds, seed523 and at most
two optimizer updates. It has no service launcher or production training CLI.
Actual optimization in these tests uses explicitly synthetic stand-in dynamics;
the real two-world Duck fixture collects one policy transition with no update.

## Learner codec is not simulation resume

At an empty iteration boundary, the separate learner codec contains exact model
weights, fixed optimizer configuration, all Adam moments and step counters,
owned CPU Torch RNG state, fresh-initializer identity and caller-supplied
source/runtime/fixture-launch hashes. No partially filled rollout can be saved.
The codec returns bytes; durable publication still belongs to a guarded writer.

Loading checks the independently expected byte hash before weights-only CPU
deserialization, then strict schemas, model dimensions/finiteness, positive
Gaussian scales, exact Adam parameter order, moment shapes/finiteness/nonnegative
second moments and exactly20 Adam steps per completed PPO update. A temporary
generator validates CPU RNG state without changing the caller's RNG. Checkpoints
use a separate fixture protocol and cannot be used as pilot evaluation exports.

Crucially, this codec includes no simulator state, contact solver state, motor
history, delay queues, episode counters, or Python/NumPy/CUDA RNG. All relevant
receipts say `simulation_resume_authorized=false`. A restored harness refuses
normal simulator input and only accepts explicitly marked synthetic test inputs.
This marker is an API-use boundary, not a security sandbox for arbitrary Python.
The declared smoke and pilot must still start fresh. Do not infer full training
resume from an optimizer restore, or restart real episodes under restored weights
and call that an uninterrupted continuation.

## Tests and next step

Focused tests distinguish pre-action, terminal and reset critic values; check
failure masking, global GAE, raw actions, selective reset evidence, finite-update
gates and configuration drift. Synthetic uninterrupted versus restored
continuation compares subsequent sampled actions, losses, updated weights, Adam
moments and RNG exactly. Tampered and partial checkpoints are refused. Neither
synthetic PPO loss nor test count establishes a real stance performance result.

Local validation passed all 330 CPU regression tests in 23.26s, including 37 new
PPO/learner-codec cases. Markdown HTML and relative-link checks passed.
Exact-source Linux confirmation follows. No GPU or real-Duck optimizer run
occurred; only the explicitly synthetic optimizer fixtures updated weights.

Linux confirmation at `945609bb7e096ab8513c2a75a34ca2324928f0b4`: all 396
CPU tests passed in 33.21s, with no skips, including shared process supervision.
The same uninterrupted-versus-restored synthetic continuation test passed on
Linux. CUDA was hidden in the allowlisted test environment. Post-test source
was clean on the exact feature branch; GPU0 was 0% / 12 MiB / 45 C with no
compute PID. Both protected system services remained inactive. No GPU job,
real-Duck optimization, trained stance checkpoint or physical motion occurred.

Remaining before the declared GPU smoke: assemble the exact held-out evaluator,
runtime/launch manifest and timed GPU adapter; implement separately validated
simulation-state retention before supporting real resume. Then run the disposable
64-world/16-iteration smoke, inspect measured cost/finite metrics and predeclare
the bounded fresh pilot. No new hop, obstacle composition or ball-balance
acceptance is claimed by this CPU precursor.

The subsequent [held-out matrix and collection component](2026-09-09-stance-heldout-matrix.md)
fixes all twelve cases and their numerical aggregation, while leaving live GPU
allocation, seed initialization and independent supervision to the guarded launcher.

The subsequent [disposable training smoke](2026-09-09-stance-training-smoke.md)
completed 16 real updates with CPU PPO and CUDA physics. Its separate smoke
subclass does not expand this CPU fixture's world/update bounds or grant the
learner-only codec permission to resume a simulator. Full-pilot timing and
held-out stance acceptance remain open gates.
