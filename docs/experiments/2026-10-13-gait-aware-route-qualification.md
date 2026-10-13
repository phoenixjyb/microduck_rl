# Prospective gait-aware route qualification

The owner approved this NEW simulation qualification on October13. It does not
rescore or replace the old instantaneous-yaw gate, reopen rejected checkpoints,
change rewards/observations/plant, or authorize physical motion. Retain original
instantaneous body-z yaw MAE as a diagnostic beside the new result. A trace/test
pass is not a learned skill or physical acceptance.

## Actual path, aligned frames

The existing gait/body traces lack root XY position, so they cannot qualify a
world trajectory retrospectively. Add opt-in `gait_path_trace` capture before
policy inference/physics stepping, using the SAME cached contact/body/command
sample and measured `root_link_pos_w`; do not integrate velocity to invent a
position. Separate protocol `microduck-gait-path-trace-v1`, world positions,
wxyz body-to-world quaternions and unchanged original v1 gait/body outputs.

Use the same three held-out seeds839/853/857 and four commands: idle,
0.30 m/s straight,0.20 m/s with yaw−0.20/+0.20 rad/s. Eight worlds each,
240 steps at0.02s, startup60, full first-episode/no-reset coverage. Preserve
the checkpoint byte hash, raw traces, command/actor identity, named motor
channels, force units, sensor order, physics clock and capture timing.

Heading is the azimuth of the projected body x-axis, not body-frame angular-z.
Require unit quaternions and projected-axis squared norm≥1e-4 at every sample;
undefined heading is an invalid measurement, never silently zero-filled.
Unwrap consecutive heading changes before interval comparisons, so a full
extra revolution cannot be hidden by identical endpoint orientations.

## Prospective fixed cycle and route gates

Moving worlds need at least THREE complete left-touchdown→left-touchdown cycles
after startup. A touchdown is the first positive-contact sample following at
least two negative samples and followed by at least two positive samples
(40ms debounce each). Every consecutive left pair must span0.2–1.5s and contain
exactly one debounced right touchdown strictly inside. Retain all such pairs;
do not select convenient cycles. Stationary idle has cycle status N/A, never
“successful walking.” Contact/air timers must agree with support, with only
the initial sample allowed uninitialized zero clocks. No missing-cycle pass.

For each cycle AND the entire settled interval[60,239], anchor the reference
at measured initial XY and projected heading. For fixed body command(v,w),
the local reference arc at timeT is `(v/w sin(wT), v/w (1−cos(wT)))`, or
`(vT,0)` for w=0; rotate it by measured initial heading into world XY. Report
endpoint errors AND the maximum trajectory error over every interval sample.
Require both XY errors divided by the complete interval duration≤0.10 m/s
moving,≤0.03 idle; both continuous-heading errors divided by duration≤0.10
rad/s. This tolerates bounded within-gait oscillation without allowing route
excursions to cancel at endpoints. The full interval prevents cycle-local
reference resets from hiding accumulated route drift. No thresholds are fitted
to the as-yet-uncollected new native path packets.

Keep complete first episodes/no fall, force-utilization p99≤0.60 (raw force
p99≤0.36 Nm at the0.60-Nm reference), zero rated-speed exceedance, named motor
hinge margin≥−0.005rad and right-hip-yaw near-stop fraction≤0.01. Idle also
retains its old instantaneous speed≤0.03 and yaw≤0.10 MAE bars. All worlds and
all twelve ordered cases must pass; a case matrix cannot mix checkpoint hashes.
Output is only `route-qualified-simulation-only` or `route-not-qualified`, with
policy/physical acceptance false. Hops and football have their own contracts.

## First bounded measurement

After the active cautious learner fully exits and its evidence is durable,
verify a clean exact source and idle exclusive Duck GPU lease before any new
measurement. Replay the original v8 mean (retained stock benchmark initial PT),
the stock final999, and the cautious final999 ONLY if its declared1000-update
round completed successfully. Use only those predeclared checkpoints, unchanged
case protocol, no checkpoint search or optimizer resume. One sequential retained
service, per-checkpoint300s cap and1200s outer cap; same DINO preservation,
scoped original user-vLLM pause/restore and source/runtime/GPU fences. No training,
video, raw perception, harder-skill promotion or physical motion in this slice.

Source/capture regression checks and native affected checks precede replay.
Local affected checks:1493 tests pass,18 existing warnings. Synthetic checks
cover signed arcs, heading wrap/full-spin aliases, returned path excursions,
unsupported contact cycles/timers, hash/frame/clock mismatches, safety metrics
and default-evaluator noninterference. Results pending; no gait-aware native
qualification is claimed yet.
