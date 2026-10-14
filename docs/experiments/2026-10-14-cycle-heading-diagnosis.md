# Retained cycle-heading diagnosis (CPU only)

This is a descriptive analysis of the completed `v8-motor-headroom-06c2b709-20261014`
control/headroom pair, not a new qualification gate or a policy promotion. The
ongoing steering-budget service and its detached `c340ff00` checkout remain
unchanged while this diagnostic is developed and executed separately.

## Contract

`scripts/diagnose_cycle_heading.py` binds the sealed summary and manifest by their
predeclared SHA-256 values, then authenticates each arm's result and all twelve
ordered raw cases before analyzing them. Checkpoints, seeds, commands, complete
first episodes, owned regular files, and actual raw bytes are checked. CUDA must
remain hidden and uninitialized. The optional isolated core is executed from
authenticated bytes, allowing CPU analysis without advancing an active worktree.

For every actual left-touchdown cycle, retain projected body-x heading residual
against the commanded turn, its signed endpoint, maximum excursion, and the
excursion after removing only the endpoint's linear trend. All endpoint and
maximum fields must reconcile with the existing route evaluator. Both original
gates remain in the output unchanged, including rejected results.

The earliest equal-magnitude peak has a deterministic sample index. Contact
phase and named head-yaw position/velocity at that peak are descriptive; sampled
contact timing may lag kinematics. Phase exposure is measured in actual physics
seconds, with peak-only counts also normalized by that exposure. This does not
prove that a contact phase or head motion causes the heading excursion. Group
statistics are explicitly cycle-weighted, not equal-world means. Idle has no gait
cycles and undefined cycle statistics, rather than a manufactured pass.

## Checks and execution

Focused CPU tests cover wraparound, commanded arcs, linear drift versus transient
excursion, deterministic peak ties, strict unchanged thresholds, per-world clock
origins, phase-duration closure, input immutability, no-cycle idle, invalid traces,
evidence tampering, and isolated-core authentication. Numerical execution and
retained artifact hashes will be recorded below after the diagnostic completes.

No reward, command, action, actor observation, motor model, training budget,
service, or acceptance threshold is changed by this analysis. No new GPU job,
MP4, raw-perception task, or physical motion is authorized here.
