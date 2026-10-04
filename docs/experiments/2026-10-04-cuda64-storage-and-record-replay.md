# CUDA64 populated storage and physical-record consistency source layer

This is a bounded continuation of the [raw-record archive source contract](2026-10-04-cuda64-record-archive-source-contract.md), not a reopened October 3 campaign, a native rollout, a training update or a new learned skill.

## Populated stock storage

`stance_recovery_cuda_storage_evidence.capture` checks the inherited shared lease before CUDA queries or storage inspection. It requires an already initialized single CUDA0 and the exact stock, nonrecurrent RSL `RolloutStorage`: 64 environments, 28 transitions and 10 raw actions. All observation, action, reward, done, value, log-probability, Gaussian-parameter, return and advantage tensors are owned CPU copies. Capture neither clears storage nor computes returns or calls an optimizer.

The CUDA-hidden `check` binds each populated slot to its retained transition, including raw Gaussian actions outside [-1, 1], rather than the separately clipped runtime actions. It checks cursor and exact shapes/dtypes, binary done bytes, finite data, zero returns/advantages and every unused tail. Empty storage is not evidence of collection. The tree has a 128 MiB tensor budget, 100,000-node limit and depth limit of 24. Every capability flag remains false.

## Physical, control and solved-phase consistency

`stance_recovery_cuda_record_replay.check` is a separate CPU-only source protocol. It does not add seeds, checkpoints or device exceptions to any historical trace protocol. It requires the independently retained **pre-action initial physical snapshot**; the first runtime boundary is already after action realization and cannot substitute for that input.

After the raw archive check, the helper validates the recorded and actual CPU math profile and independently compiles the selected CPU plant/body binding. It rehydrates physical boundaries, pre-action observation continuity, actual counters and first-terminal facts using the generic first-attempt validator, then checks plant observations/contact evidence and action slew, delay FIFO, motor command/commit masks and voltage history.

For each reached force window, all three full solved-phase payloads must validate for 64 rows: kinematics, dynamics, solver counters, ordered contacts and active constraint tables. Complete external/generalized-force arrays must match the schedule before integration and be zero afterward. Control, friction and damping must match the corresponding committed motor proposal. Physical qpos/qvel must match their recorded boundaries, pre-phase time must match actual row clocks, only accepted rows advance one 2-ms step, and unforced post kinematics cannot integrate again. Pre/integrated dynamics, contacts, solver counters and constraints must retain the same pre-Euler solve.

These are **recorded-data consistency checks**, not independent solver execution, CUDA RNG sampling, actor inference, BAM recomputation, reward recomputation from physics, model/Adam immutability or whole-trajectory physics re-simulation. Native transition, natural timeout, selective reset, training and skill admission remain false. A 28-call prefix is still only 0.56 simulated seconds; it can reach a 0.5-second pulse but cannot reach later timing cells or the natural timeout.

The helper accepts decoded CPU evidence. The calling reader must verify the independently retained whole artifact hash before decoding; a passing direct `check` does not authenticate its bytes or the capture process. Storage evidence must be combined with the archive and physical-record check rather than used alone as native admission.

## Verification and retained host boundary

The owner ran the actual CPU plant compilation on macOS with CUDA hidden: 16 bodies, `trunk_base` body 1, CUDA uninitialized. This checks compilation/selected binding only, not WSL CPU-profile replay or native physics. Synthetic source tests explicitly identify any private test-only seams; they do not create native capture evidence.

Initial storage/archive/collector integration: **130 passed** with CUDA hidden. Final populated-storage suite: **24 passed**, including two cases using installed `rsl-rl-lib==5.0.1` CPU storage populated through actual stock `add_transition`. Final record-replay suite: **27 passed**, including the actual CPU plant compile and explicitly synthetic phase/public-path cases.

Owner-reviewed **29-file integration regression: 838 passed in 95.77 seconds**, CUDA hidden, no skips. This extends the preceding 27-file/787-test archive regression with the two new test files. Ruff lint and formatting, module compilation, local documentation links and `git diff --check` passed. These checks preserve the frozen historical seed/protocol allowlists; they do not qualify native collection or add learned capabilities.

At October 4 **09:34:23 Shanghai**, the fresh read-only WSL check found the exact clean feature branch still at `1b96ccaac1326d6f0a1e00cdff8950b91a2916be`. The RTX PRO 4000 Blackwell was idle (0%, 695 MiB, 30 C), with no compute PID or running Duck user service. FilmBrain observatory and video-playground remained running; the protected AI Mission user services were inactive with PID 0. No WSL source, package, driver, service or workload was changed by this source chunk.

## Next native execution gate

Before launching a new native job, implement and test a **fresh**, bounded supervisor and whole-byte CPU reader. It must authenticate the closed preparation prerequisites, retain a freshly restored parent and caller/model/Adam snapshots, retain the pre-action initial frame, require `capture_control=True`, stop collection at the first observed terminal, retain complete populated storage and force phases, and run capture/replay serially under one inherited shared FilmBrain GPU lease with idle gates and explicit runtime/log/process caps. Use only a reachable early pulse for this short diagnostic. The old frozen sampler declaration and timings do not qualify real CUDA64 physical feedback or its budget.

No training update should be launched merely to exercise this evidence plumbing: the existing single-seed disturbance catalog found no measured deficit. The new native no-update gate, any later longer/held-out evaluation, an observed training deficit and an optimizer qualification are separate decisions. Hopping, football balance and physical motion are not admitted here.
