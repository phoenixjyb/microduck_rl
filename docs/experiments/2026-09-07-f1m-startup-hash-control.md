# F1-M startup-hash control: predeclared before execution

Status: predeclared; zero optimizer updates. This is a measurement control,
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
