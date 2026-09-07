# F1-M startup-hash control: predeclared before execution

Status: completed, startup-hash-controlled divergence; zero optimizer updates. This is a measurement control,
not a policy promotion or a replacement for the rejected F1-M paired pilot.

## Evidence and single-axis hypothesis

The completed same-source recording-OFF/OFF pair diverged despite matching
captured numerical settings. Its public fixed-string Python hash probes
differed; PYTHONHASHSEED was unset at entry and assigned503 only after Python
startup. See [the closed replay control](2026-09-07-f1m-replay-control.md).
That observation identifies an uncontrolled factor, not a solver root cause.

Change only the fresh child process environment to PYTHONHASHSEED=503 **before
exec**. Do not change Torch determinism/precision, thread counts, policy,
runtime packages, physics, seeds, controller or recording. Parent process
environment remains unchanged. Capture effective child settings and require
the public hash probe to be identical before/after both rollouts. A mismatch
is a control failure, not an exact-repeatability result.

## Frozen protocol and stop conditions

- Protocol `f1m-replay-hash503-off-off-v1`, new immutable output
  `artifacts/experiments/f1m-replay-hash503-off-off-v1`.
- Run `python -m mjlab_microduck.foundation_replay_control --source <pushed SHA>
  --startup-hash` as one retained user service on100.100. Exact clean feature
  branch, helper/runtime hashes, idle GPU and protected system services
  inactive are mandatory before launch. Capture source SHA in every report.
- Exactly two sequential fresh children: first/second, each parent503,
  400steps,8env, heading hold enabled; raw route/command traces ON and motor
  history OFF. Parent checkpoint SHA
  `7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`.
- Each child has120seconds; service has420seconds and30second stop timeout.
  Idle check before each child;40second closeout reserve. No new work after
 06:00 or any launch whose cap cannot fit07:00 Shanghai September8.
- First unsafe child stops the pair. No retry, replacement seed, third ON
  case, tuning or training. Preserve original F1-M83 payloads, timing5 payloads
  and OFF/OFF10 payloads through pre/post hash and file-coverage verification.

## Interpretation and next gate

Compare every typed JSON report leaf and retain first trace differences and
per-column maximum deltas. All original safety/performance gates stay intact.
If reports differ with matching effective startup hashes, label
`startup-hash-controlled-divergence`: this control is insufficient; it does
not identify the specific numerical cause. If identical, label
`startup-hash-controlled-exact-match-in-this-pair`: one matching pair is not
universal determinism, attribution to Python hashing, or training admission.

Retain decision/manifest and all payload hashes; mirror and independently
recompute on CPU, then test/commit/push concise evidence. After this bounded
control, do not spend the night attempting arbitrary determinism settings.
Any further instrumentation/evaluation design must explicitly separate
within-rollout aggregate reconstruction from cross-process repeatability,
preserve historical failures and predeclare its evidence scope. No second
paired training pilot until that measurement-validity contract is reviewed.
No H2, video, raw perception, physical motion or combined-skill claim.

## Prelaunch validation

Local regression:772 passed, no skips, two existing actuator/site warnings,
18.48s. The23 replay-control tests include real fresh-process public-hash
agreement on CPU, environment-only change, no parent mutation, source/recording/
settings refusal, hash-probe refusal, closed payload/hash/coverage protection,
new output isolation, two-case/no-optimizer limits and immutable outputs.
All10 previous OFF/OFF payloads were verified locally and the original full
decision reproduced exactly on CPU. `git diff --check` passed. Live100.100
preparation check found no running Duck user service or compute PID,0% GPU,
45C,12MiB, and both protected system services inactive; repeat immediately
before launch and run the same CPU regression on the frozen remote checkout.

## Retained result: September7 23:20 Shanghai

Source `7ded54a4cad06a6aa9c46eeb76ea77b10affb252`; service
`microduck-rl-replay-hash503-7ded54a.service` started23:20:03, emitted the
completed decision23:20:42 and closed normally23:20:43. Remote regression
passed772 tests in18.92s, no skips, the same two existing warnings.

Both recording-OFF cases completed400steps/8env without absolute safety
failures. All four startup/exit snapshots have PYTHONHASHSEED503 and the same
public hash probe `-414362993647959279`. Other captured numerical settings
match. The reports nevertheless differ at28,526 typed JSON leaves; decision
`startup-hash-controlled-divergence`. First changed velocity is step3/env0/
body-forward; first position change step4/env2/route-forward. Maximum absolute
differences are body-forward0.071752m/s, route-forward0.071746m/s,
cross-route0.163936m/s, heading0.098229rad, route-position0.032411m and
cross-position0.065636m. Exact repeatability is not established; startup hash
control alone is insufficient. The specific numerical cause remains unknown.

All10 payloads (7,253,955bytes plus manifest) are mirrored locally under
`artifacts/diagnostics/f1m-replay-hash503-off-off-v1`. Both hosts independently
verified complete byte/hash/file coverage and reproduced every decision field
on CPU. Historical F1-M83, failed timing5 and OFF/OFF10 payloads and checkpoint
sentinels remain unchanged. No optimizer updates or third case ran.

- Manifest: `e90ee62d458421bdb7c69398e25d4613727dd2cac236840fadd8d0bc5451f87c`.
- Decision: `dad1bb5b099ccd12c7410655655484d3c09f57e2fa8b0f975a6f3ff08c30a4b8`.
- First report: `92afd251b26de1f0377a3812409a079c2c005a8b9657bb07d565898b24fa643a`.
- Second report: `167c6900331c188ce740724ae2a58599e0557769f6e730b745b9dfd35f4b0c2a`.

GPU returned idle0%,46C,12MiB with no compute PID; protected system services
remain inactive. This protocol is closed; do not retry it or adjust flags
until a matching result appears. The original F1-M rejection and failed
historical timing identity gate remain final, not superseded.

## Next bounded coding chunk

Define and test a **descriptive within-rollout motor measurement contract**
before collecting new timing data. It must reconstruct all current summary
metrics from that report's raw tensors at the existing1e-9 tolerance, validate
named motors, force/speed capture timing, command/route traces and unchanged
absolute safety gates. It must explicitly refuse historical identity or
causal-effect claims. Cross-run differences require separately captured
replicates and must be reported alongside within-arm variation, not explained
away by averaging or silently relaxed thresholds.

Only after CPU contract tests and a separate fixed-case, source-bound
predeclaration may a short fresh-recording dataset be collected. Suggested
scope for review is the three retained parent/control/motor checkpoints with
two fresh same-seed503 replicates each and balanced forward/reverse arm order;
no optimizer updates, no source/backend/reward changes between cases, first
unsafe case stops, and no selecting a favorable replicate. This suggestion
is not yet an executable launch protocol. Do not launch paired pilot2 until
the descriptive evidence supports one clearly predeclared reward hypothesis.
