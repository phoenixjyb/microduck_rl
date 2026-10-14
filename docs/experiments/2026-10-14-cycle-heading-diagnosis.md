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

## Executed evidence

Diagnostic implementation commit: `3183d429` on
`integration/upstream-2026-10-10`; pushed to the fork. Focused CPU checks:
**46 passed** (new core/CLI, existing route/phase diagnosis, steering-budget
contracts), with CUDA hidden. No runtime/package installation was required by
the diagnostic execution. This is not a new full native test-suite result.

Native execution used only isolated owned files in `/home/converge/.local/state`
and the existing Python runtime, with CUDA hidden and verified uninitialized.
The active checkout stayed clean and detached at
`c340ff00d8d532fa4a664a298589b7fa401d7467`; training was not interrupted.
The existing route and phase helper bytes match both the Mac checkout and the
original evaluated source `06c2b709`.

| Binding | SHA-256 |
| --- | --- |
| Diagnostic core | `612f3c7d320835e01f4d0c55c1d94631c5f9660c8e42d734057660588593b104` |
| Diagnostic CLI | `8e7224f8f478bf0a3c0063102e5560cd5cbda2f2c1a076b32f6eb72b48fc5bca` |
| Existing route helper | `119ac751ed72b8dab8fc8d4c5f0a39bb291d5f1940da4c08cf5637cca465a785` |
| Existing phase helper | `ed08e90b8f095bb84e591da5e530ca477c0642dec06fc243e94b235008717c15` |
| Sealed input summary | `61391992bf577974883c454155f01367d8faccad916734719aef06a30e574671` |
| Sealed input manifest | `f5e7d7a15b6796172d57af354c022c21ec6f7a7281ce40ac9049b8d304341bbf` |
| Diagnostic output | `97cb61a77f74f63e2c9c86d79eb76602bdc746f782d848f21c0898f555f8851f` |

Output: 1,487,020 bytes, retained natively at
`/home/converge/.local/state/microduck-cycle-heading-20261014.json`; a matching
small local copy is at
`artifacts/evaluations/cycle-heading-3183d429-20261014/diagnosis.json` (ignored).
No raw trace/checkpoint packet was duplicated. All 24 authenticated cases were
processed; idle has zero cycles, and the moving cases contain 978 cycles total.
Both original route gates remain `route-not-qualified`.

| Arm / command (speed m/s, yaw rad/s) | Cycles | Endpoint failures | Additional peak-only failures | Mean detrended excursion rad |
| --- | ---: | ---: | ---: | ---: |
| Control / 0.3, 0 | 153 | 28 | 37 | 0.03402 |
| Control / 0.2, -0.2 | 166 | 33 | 61 | 0.03909 |
| Control / 0.2, +0.2 | 160 | 12 | 62 | 0.03991 |
| Headroom / 0.3, 0 | 164 | 73 | 20 | 0.03092 |
| Headroom / 0.2, -0.2 | 167 | 49 | 18 | 0.03244 |
| Headroom / 0.2, +0.2 | 168 | 81 | 14 | 0.03373 |

The control has 73 endpoint failures plus 160 additional peak-only failures
among 479 cycles. The rejected headroom arm has 203 endpoint failures plus 52
additional peak-only failures among 499 cycles. Its smaller detrended excursions
do **not** compensate for the larger persistent bias or qualify the policy.
Control mean signed endpoint residuals are -0.00234, +0.01764, +0.00441 rad;
headroom values are -0.04147, -0.02717, -0.04502 rad, in the same command order.

Control peak-only events fall in contact-transition / left-only / right-only
intervals 54 / 9 / 97 times. Counts and exposure-normalized values are retained
as associations, not proof of a phase mechanism; head-yaw measurements likewise
do not establish causality. The ongoing one-factor budget comparison remains
predeclared and unchanged. Do not select a new reward or promote a checkpoint
until its final held-out comparison is available.
