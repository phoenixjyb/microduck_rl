# B1-N action, delay and motor-commit evidence

This bounded CPU coding chunk follows the [compiled plant checks](2026-09-09-stance-plant-evidence.md).
No PPO, full-hold evaluation, CUDA job, video, robot motion or trained checkpoint
is introduced. Existing historical numerical gates remain unchanged.

## Capture and replay contract

`WarpStanceRuntime.step(..., capture_control=True)` now returns owned control
evidence. Capture is opt-in and off by default. It changes neither actor inputs
nor command/physics ordering. Capture records:

- Initial and post-action correction, target, three-slot FIFO, motor history,
  voltage, gain, per-world friction/damping and ordered controls.
- Every attempted motor command, its physical counter and live mask, proposed
  torque, accepted/rejected masks and committed control state before physics.
- The final control state, including unchanged closed/rejected worlds.

The CPU checker requires the already validated physical trace and compiled-plant
mapping. It independently reconstructs action clipping, soft-range clipping,
0.02-radian correction slew per policy tick, nominal head positions and the
three-physics-step delay. It binds motor position/velocity inputs to the recorded
physical boundary, requires position-only commands, and rejects missing/extra
proposals. Every accepted mask must correspond to exactly one physical step;
post-step applied torque must agree with the committed ordered motor command.

The original strict `abs(torque) > 0.36 Nm` rejection rule is unchanged. Rejected
torques must match the first-terminal record without advancing that world's
FIFO, controls or motor state. Previously closed worlds cannot acquire a new
correction or motor update. The checker also reconstructs voltage sag from the
previous committed torque (7.5 V minus0.1 times summed absolute torque, floored
at6 V), requires nominal gain200 and checks that uncontrolled DOFs are untouched.

Structural identities, counters, masks, FIFO shifts and committed histories use
exact comparison. Arithmetic comparisons use absolute1e-6/relative1e-5 for
CPU/GPU representation differences, not relaxed physical thresholds. Finite,
nonnegative accepted friction/damping values are retained but not independently
recomputed. Full BAM torque/friction recomputation, live runtime/host provenance
and GPU execution of this newly instrumented path are not claimed. The receipt
explicitly preserves `bam_outputs_recomputed=false` and all existing no-admission
flags. Motor proposal counts count batched calls, not individual worlds.

## Retention and compatibility

Bundle/launch protocols are now explicitly version3. `control.pt` is required
alongside the previous seven files and shares the independently expected trace
binding. Both publication and read-only verification replay its semantics after
physical/checkpoint/plant checks. Exact bytes are hashed before weights-only CPU
deserialization; control evidence has a separate512 MiB ceiling. These size limits
are not a sandbox for hostile archives. Exclusive creation, fsync, manifest-last
publication and refusal to overwrite old artifacts remain unchanged.

Earlier v1/v2 bundle protocols and retained research evidence are not silently
promoted or reinterpreted. No non-test v3 bundle or training outcome is claimed
by the temporary pytest fixtures. The compiled-model platform identity restriction
still applies: Linux bundles need the matching Linux compile/runtime.

## Validation and remaining work

Tests use short real CPU rollouts and explicit synthetic torque/tilt failures.
Coverage includes non-mutating opt-in capture, clipped actions, both directions
of slew, queue timing, frozen siblings, all-world proposal rejection, malformed
commands/masks/history/voltage, missing steps and hash-before-load enforcement.
Local regression validation passed292 CPU checks in48.72s. After adding a final
rehashed-control tamper case, all20 bundle tests passed in19.36s. This covers293
unique local checks, with35 new cases relative to the previous chunk. The CPU
replay also passed under a caller's `meta` default device, without allocating on
CUDA. Markdown HTML and relative-link checks passed. Exact-source Linux
confirmation is next.

Exact-source Linux confirmation at
`ce560dd431560d29dc6920382715332f65a3b4ba`: all 359 CPU tests passed in
32.64s with no skips, including shared process-supervision checks. CUDA was
hidden in the allowlisted test environment. Post-test source was clean on the
exact feature branch; GPU0 remained at 0% / 12 MiB / 44 C with no compute PID.
Both protected system services remained inactive. No optimizer, GPU service,
trained checkpoint, video or physical motion occurred. The newly instrumented
CUDA path still needs its separately supervised integration/smoke gate.

Next assemble the four-checkpoint/three-seed evaluation runner, PPO transitions
and timeout bootstrap semantics, resumable optimizer/RNG saves and guarded launch
inputs. Before any optimizer pilot, run the predeclared disposable64-world/
16-iteration smoke and measure the changed runtime's cost. This control recorder
does not itself authorize multi-seed promotion, hopping or rolling-ball balance.
