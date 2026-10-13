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

## Native closeout — measured route failures, not a raw-yaw-only artifact

Exact source `c10a6a57dca64431793b16e94e6da47791129ec7`, native affected tests
1493 pass/18 existing warnings. One sequential retained
`recomo-duck-v8-gait-route-c10a6a57-20261013.service` finished inactive/success0.
The initial/stock/cautious arms took59.019/60.644/60.539s; all36 cases completed
240 frames, no terminals and zero optimizer updates. Both decisions were
independently recomputed from each retained raw trace and matched the records.
All three full matrices are `route-not-qualified`; no threshold was changed.

| Command vx / yaw | Original v8 route-passing worlds | Stock final999 | Cautious final999 |
| --- | --- | --- | --- |
| .30 / 0 |0/24 |0/24 |0/24 |
| .20 / −.20 |0/24 |0/24 |1/24 |
| .20 / +.20 |0/24 |1/24 |1/24 |

No full moving case passes. Original and cautious idle each pass3/3 cases and
all24 worlds; stock idle passes2/3 cases because its seed853 idle body-yaw bar
fails, even though all24 path-component world checks pass. This is a separate
new replay, not a claim that trajectories are byte-identical to older rollouts.
All motor/range checks pass in this replay. Cautious raw force p99 peaks
.359999Nm, very close to the unchanged .36-Nm bar: do not round away the margin.

| Full settled interval mean error | Original v8 | Stock final999 | Cautious final999 |
| --- | --- | --- | --- |
| Straight XY endpoint/T (m/s) |.178345 |.118163 |.121502 |
| Negative arc XY endpoint/T (m/s) |.118199 |.074077 |.071298 |
| Positive arc XY endpoint/T (m/s) |.119084 |.061381 |.059763 |
| Straight heading endpoint/T (rad/s) |.041639 |.145980 |.156549 |
| Negative arc heading endpoint/T (rad/s) |.056566 |.149403 |.170334 |
| Positive arc heading endpoint/T (rad/s) |.051389 |.095476 |.120784 |

Acceptance uses every world, every cycle and the maximum trajectory errors,
not these means. The new measurements confirm a real speed/route tradeoff:
the original teacher is slow but has much smaller heading bias; learned actors
make forward/arc-distance gains while worsening straight and negative-turn
heading. Cycle-heading errors dominate the learned actor failures. A low
`valid_cycle_count` means cycles failed route accuracy, not absent contact
coverage; keep physical support coverage distinct from successful route cycles.
Actual moving support-cycle count ranges are4–7 original,6–8 stock and6–7
cautious; every moving world has sufficient complete cycles and there are no
support-timer, missing-right-touchdown or cycle-duration failures.
The next learning hypothesis should price persistent steering error without
taxing unavoidable within-gait angular oscillation, and retain speed and idle.
An existing reset/command-aware `body_yaw_bias_cost` is a candidate objective;
it measures body-rate bias, not the projected-heading trajectory, so its reward
signal must be diagnosed and its recipe separately predeclared/tested before
another learner. No such new learner is launched by this measurement closeout.

Full native packet:
`artifacts/evaluations/v8-gait-route-c10a6a57-20261013`; only compact
summary/inventory counterparts are retained on the Mac. Maximum GPU1754MiB,
minimum free14440MiB, maximum65°C. DINO PID2029 remains unchanged; user vLLM
restored active and owns its GPU memory again. Other protected scopes remain
inactive. No libraries, policies, rewards, actor inputs, old gates or unrelated
workloads were changed during replay; no motion, camera perception or video.

| Sealed artifact | SHA256 |
| --- | --- |
| Initial actor replay result | `ccea297aed76500c1ac5634628206bc639ac47ab870ebc708cc49d758bd20b6a` |
| Stock actor replay result | `b6b172d70a933ee89b66de6599f18be568598459b098950e9af3fd3a9b14f31d` |
| Cautious actor replay result | `632f1656afdb0ce0a43f01beef6580d24187497fbfeaa6e877c8ad71ab1f11a2` |
| Summary | `13b6c4109cd74a50d3584f5aa99237da702eedf1a29303fac0c8053604be0c44` |
| Full packet inventory | `7a80c7721d0ff8d6f99714e49c7f0d3accea980c33ee35371c3163b1d684c853` |
| Scoped restoration | `d1dc8303ad10ffc7876daa0e8a823fa2a454933d3b8f9dadc8138d1e796ea5c5` |
