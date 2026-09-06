# Pre-reset motor measurement audit

Date: 2026-09-06. Initial source chunk: CPU/source validation only; no GPU job
declared by that change. The subsequent separately authorized live validation
is retained in [the seed-373 smoke](2026-09-06-motor-audit-smoke.md).

## Finding and limits

The seed-367 control stopped at reported torque utilization p99
`0.6125551462173462 > 0.60`; its single retained report cannot reconstruct the
force histories. Its stop, the seed-359 recorder failure, and U3/U4/O3a rejection
remain unchanged. No threshold or historical result is relaxed or recomputed
with different semantics.

At source `bda8612203776ca0e9c71c5015b1f5490a8f34d6`, the hierarchical rollout
reads `actuator_force`, joint velocity and actions **after** `wrapped.step()`.
The installed mjlab 1.3.0 step first checks terminations and computes rewards
and metrics, then auto-resets finished environments, calls `sim.forward()`, and
returns. The OA0 rollout uses `auto_reset=True`. Its legacy sampler includes
environments that were active at step entry, including those just reset.

Thus terminal motor samples can be replacement reset-state values, and terminal
actions can be reset zeros. A CPU counterexample invokes the **actual installed
`ManagerBasedRlEnv.step` method**, with synthetic physics/reset managers: terminal
forces `[0.30, -0.48]` are replaced by either zero or `[9.0, 9.0]` before the old
sampling point. The measurement can be biased downward or upward. This proves
the sampling-order problem, **not its contribution to the seed-367 p99**. That
historical trajectory has insufficient saved motor samples to answer causality.

Audited dependency files match byte-for-byte on the Mac and 100.100:

| Installed mjlab file | SHA-256 |
| --- | --- |
| `envs/manager_based_rl_env.py` | `a381027e336d6313cd338d541230b36864657edac89708a33e7e60c0c6fb74d2` |
| `managers/metrics_manager.py` | `c78b9a58d6e5457f841847ee9e48592d2634d294d583585adeedbb030c879d85` |
| `entity/data.py` | `fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657` |

## Separate corrected measurement path

New opt-in CLI flag: `--motor-measurement-audit`. Protocol:
`motor-pre-reset-step-audit-v1`. It adds a logging-only zero metric to the existing
post-decimation metrics hook, which runs **before reset and final forward**.
The callback detaches/clones motor forces, aligned joint velocities and terminal
identity; phase and active first-attempt identity are armed before stepping.
Reset cannot overwrite these snapshots. There are no additional physics steps,
forward calls, RNG draws, rewards, commands or actor observations.

The copied force is the derived force from the **last physics substep**, with
its one-integration lag. Velocity is integrated state at that boundary. This is
not an all-substep peak measurement, a synchronized mechanical-power estimate,
or a measured temperature. The current model uses four 0.005-second physics
steps per control step. Thermal and stall references remain provisional model
assumptions until there is a physical duck to calibrate.

Force columns are bound to actuator targets by name, with checks for unique,
direct unit-gear hinge transmissions. The actual OA0 robot specification resolves
14 such joints; unsupported gears, tendon/site transmissions, missing targets,
duplicate targets and inconsistent column order fail closed. Array shape alone
is insufficient evidence of joint identity. This audit does not implement
general transmission conversion or alter the BAM actuator model.

The per-case `motor_measurement_audit` report contains:

- Explicit timing, protocol, force-column identity and model normalization.
- Pooled and per-phase force/velocity statistics, plus normalized per-joint
  distributions for approach, interaction and recovery.
- Pre-reset terminal force, post-return terminal force and their difference.
- Captured step counts, incomplete first attempts and nonfinite-sample counts.
  Nonfinite values are flagged, not silently omitted or zero-filled; affected
  distribution summaries are null. Reductions use float64 and scaled RMS.

The bound is 64 environments, 1000 control steps, 64 force columns and 12 cases.
Only first-terminal-attempt diagnostics are supported. Dataset collection and
simultaneous use of the unvalidated trajectory recorder are refused. Exactly
one capture is required per armed step; completed environments cannot reenter
or unfinished environments disappear from the sample set. Buffers are bounded
in memory; reports contain summaries rather than full motor time-series files.

The default flag is **off**. Existing motor statistics and admission thresholds
are **not replaced**, including when the new audit flag is on. Additional audit
fields are separately labelled diagnostic-only with policy/data/motion authority
false and `runtime_equivalence_validated=False`. A lower new metric cannot
retroactively pass an old result. The source integration changes executable
code, so exact GPU trajectory equivalence is not claimed even with default
settings preserved.

## Validation and next boundary

CPU tests cover the actual installed step ordering with high and low reset
forces, clone isolation, phase/joint summaries, first-attempt completion,
missing/duplicate captures, terminal mismatch, layout checks, NaN/Infinity,
finite large-value reductions, opt-out report compatibility and separate opt-in
provenance. A synthetic legacy p99 of `0.6125551462173462` remains above its old
gate even when the new audit field is zero. The existing control therefore
cannot be made to pass by reading a lower differently timed metric.

The simulated CPU step counterexample preserves mock dynamics, actions,
observations and torch RNG state when the observer is enabled. It is **not**
a MuJoCo/Warp GPU non-interference test. CPU construction of the real robot
specification verifies the force-to-joint mapping, not live policy performance.

Precommit validation: **305 focused tests passed locally** with CUDA hidden
(7.16 s), covering this audit, motor-aware rewards/config, repeatability, recorder,
rollout and U4/U3/U1 collection/gates. The real-spec construction emits the
existing broad-pattern warning about seven similarly named sites; explicit JOINT
transmission and all 14 unit-gear hinge mappings were checked independently.
Diff whitespace checks passed. Remote CPU repetition is required before any
separately authorized future rollout.

No GPU service, policy training, replacement seed, video, raw perception or
physical motion is authorized by this source change. Next, after review and a
new explicit GPU window, separately predeclare a bounded audit-validation run
with frozen source/artifacts, retention and stopping rules. It must keep the
old gate decisions and new measurement evidence separate. Only subsequent
evidence can justify a motor-aware curriculum change.
