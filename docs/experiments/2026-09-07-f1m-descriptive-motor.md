# F1-M descriptive motor measurements

Status: six-case collection completed and retained; descriptive evidence only.
No new optimizer updates are authorized
by a measurement result.

## Why this is a separate experiment

F1-M remains rejected. The original timing experiment failed its requirement
to replay an earlier report exactly. That failure is closed and immutable.
Two later same-source OFF/OFF controls also diverged, including the pair with
identical startup hash probes. Neither control identified the numerical cause.
See [startup-hash evidence](2026-09-07-f1m-startup-hash-control.md).

The new question is narrower: **what loads occurred in each new recorded
rollout, and how do two fresh repeats vary within each retained policy?**
It does not ask to recover the exact old trajectory, prove recorder equivalence,
estimate a causal effect, or overturn an earlier numerical decision.

`motor_measurement_contract.measure_case` reconstructs that report's raw motor
summaries with the existing absolute tolerance 1e-9. It also validates source,
checkpoint, named joint order, command delivery, route summaries, capture timing,
coverage and absolute safety. Force is last-substep-derived with one integration
lag; route is pre-control. Do not call these simultaneous states or contact phases.

`describe_dataset` requires all six cases in fixed order, matching complete
entry/exit fingerprints and effective startup hash503. It reports every
replicate, within-arm min/max/spread and all four cross-arm differences for
each scalar and named-joint torque quantile. These are observed ranges, **not
confidence intervals**. It retains each replica's unchanged original paired
speed/heading/lateral/motor/power gate failures, without averaging them away.

Policy acceptance, historical replay identity, causal effect, training admission,
curriculum promotion and physical-motion authority are always false. The
contract explicitly rejects inputs asserting otherwise. Seed503 is development
data; two process repeats are not independent training seeds or generalization.

## Predeclared fresh collection

- Protocol: `f1m-descriptive-motor-replicates-v1`.
- New immutable directory:
  `artifacts/experiments/f1m-descriptive-motor-replicates-v1`.
- Intended runner: `python -m mjlab_microduck.foundation_motor_replicates
  --source <exact pushed source>`. No launch until runner tests and source push.
- Exactly six fresh child processes, in order:
  `parent-1, control-1, motor-1, motor-2, control-2, parent-2`.
- Each case: unchanged F1-M evaluator protocol, seed503, 400 control steps,
  eight environments, 0.02s step, 100 startup and 300 settled steps, heading
  hold on, nominal forward0.3m/s, raw route/command/motor recording on.
- Fix child-start PYTHONHASHSEED503; retain CUDA0, OMP1, unbuffered output and
  all captured baseline backend settings. No precision/determinism/physics/
  policy/reward changes between cases, no raw perception or actor expansion.
- Child cap120s, sequential service cap900s, stop timeout30s, 60s closeout
  reserve in the runner. No competing GPU PID. Verify protected system
  services inactive and two idle GPU samples before the service and each case.
  No start after06:00 Shanghai, and the complete cap must fit before07:00.
- Preserve and reverify the original F1-M83 payloads, failed timing5,
  OFF/OFF10 and hash-controlled10 payloads, plus all checkpoint sentinels.
  Never append to or retry a closed directory.

Fixed checkpoints (no new training):

| Arm | Checkpoint SHA-256 |
| --- | --- |
| parent | `7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f` |
| control | `cf63952d0975b120f14f92b7cb9b7d5f8b2d6f5e46b5db8f214c63b1c77df83a` |
| motor | `dcef0f75a3b3bccd0f026de34510aca793f888d10ff8aa4569e74ee4201a18fb` |

Each report is retained before validation can fail. Stop at the first unsafe,
incomplete, nonfinite, identity-mismatched or unreconstructable case; keep all
earlier/failed payloads and exact error. No replacement seed, extra repeat or
selected best case. Known performance failures remain failures: this fixed
descriptive collection may complete despite them, but cannot admit training
or promotion. Absolute safety/coverage failures stop collection immediately.

## Closure and next gate

Retain raw report, fingerprint, per-case audit, idle proof and child log, plus
launch, deterministic descriptive decision and a byte/hash manifest. Mirror
all payloads; independently recompute the complete decision on CPU and verify
historical hashes. Test, commit and push the exact evidence.

Only then consider one reward hypothesis supported by recorded joint/timing
patterns and reported repeat variation. If the direction is inconsistent or
overlaps materially, say so; do not claim statistical significance from two
repeats. A second paired pilot needs a new predeclaration and unchanged gates,
and still counts toward tonight's three-pilot cap. No harder obstacles, H2,
video, physical motion or accepted hopping is implied by this dataset.

## CPU contract validation before runner implementation

95 focused tests passed in21.54s, including29 new descriptive-contract tests.
They cover same-report reconstruction, nonmutation, source/checkpoint/seed and
command/route integrity, explicit refusal of causal/historical/admission claims,
unsafe and missing-joint inputs, fixed case order, missing replicas/settings,
observed-range arithmetic and preservation of every original failed gate.
`git diff --check` passed. These synthetic tests are not fresh GPU evidence.

## Runner validation and launch gate

The `foundation_motor_replicates` runner is implemented with fixed case order,
checkpoint/hash-history checks, identical baseline process fingerprints,
per-child idle checks,120s timeouts,900s service cap and60s remaining reserve.
It preserves returned raw reports/fingerprints before validation errors, writes
an immutable manifest on normal/error closure and never invokes training.

Final local regression passed809 tests in32.32s, no skips, two existing
actuator/site-pattern warnings. Eight runner tests cover first/later unsafe
stops, runtime/fingerprint failures, no optimizer, timeout/startup environment,
deadline refusal, manifest preservation and raw retention before validation.
The initial fingerprint test incorrectly aliased its mutable mock baseline;
the fixture now uses an independent copy matching real JSON-loaded history,
and the full final suite passed. No GPU job ran during that test correction.
Repeat the same suite on100.100, then verify clean exact pushed source and
idle GPU before launching the predeclared six-case collection once.

## Completed collection: September7 23:53–23:55 Shanghai

Source `c0ec38453ff18ee1100a1f3eb92e70ca37357521`; service
`microduck-rl-motor-replicates-c0ec384.service` started23:53:27, emitted the
complete decision23:55:15 and exited normally23:55:16. Remote prelaunch
regression passed809 tests in34.61s, no skips, the same two existing warnings.
All six cases completed400steps/8env with no absolute safety failure. All
raw motor/route/command audits passed; no optimizer ran. The observed GPU
temperature during collection reached52C; closure was idle0%,47C,12MiB,
no compute PID, both protected system services inactive.

Decision: `descriptive-replicates-only`. Both replica-specific original
performance gate readouts reject the motor policy. These are48 first episodes
at one development physics seed, not independent training-seed confirmation.

Observed settled ranges across the two repeats (not confidence intervals):

| Metric | Parent | Control | Motor |
| --- | --- | --- | --- |
| Route speed, m/s | 0.264451–0.264964 | 0.264332–0.265092 | 0.262416–0.262544 |
| Absolute lateral speed, m/s | 0.076530–0.076985 | 0.061641–0.061756 | 0.058994–0.059206 |
| Mean squared utilization | 0.039624–0.039689 | 0.042118–0.042294 | 0.040962–0.041014 |
| Mean absolute mechanical power, W/motor | 0.136972–0.137257 | 0.154569–0.154959 | 0.150968–0.151414 |
| Pooled torque-utilization p99 | 0.558068–0.558746 | 0.576717–0.579710 | 0.577809–0.580285 |

Motor versus control has non-overlapping observed ranges for lower speed,
lateral motion, squared load and power. Pooled torque p99, heading maximum
and soft-limit fraction ranges overlap: do not claim reliable improvement
for those quantities. Every gate remains unrounded in the retained decision.
Left-knee p99 is higher than control by0.033499–0.037050 across all four
comparisons (allowed nonregression margin0.02). Head-roll p99 is higher by
0.049006–0.060357; both repeats also violate the parent head-roll margin.

The head-roll load increase persists in every settled one-second bin: motor
bin mean squared utilization0.029091–0.033630 versus control0.020607–0.023790.
Head-roll share of total squared load is5.4436–5.4556% versus3.7806–3.7877%;
left-knee share11.4131–11.5837% versus9.5647–9.6432%. These are associations,
not contact-phase or causal findings.

Exploratory CPU-only inspection of the same raw joint-speed tensors, steps
100:400, gives head-roll RMS speed1.862200–1.864644rad/s for motor versus
1.577424–1.577647 for control. RMS adjacent control-sample velocity differences
(299 differences per environment, not actuator target changes) are
0.835818–0.837132rad/s versus0.695811–0.695875. This supports testing a
head-motion regularization hypothesis; it does not prove action chatter or
that reducing head activity will preserve balance or repair the knee/speed gates.

## Retention and cross-CPU reconciliation

All32 payloads (28,284,624bytes plus manifest) were mirrored locally under
`artifacts/diagnostics/f1m-descriptive-motor-replicates-v1`. Both hosts verified
the complete byte/hash/file coverage, all six report hashes, process
fingerprints, raw-summary reconstruction and the complete deterministic
decision. Original F1-M83, failed timing5, OFF/OFF10, hash-controlled10 and all
checkpoint sentinels remain unchanged.

- Manifest: `44d43f73564e73e67d4e5a456487c4129eff44cd20b128ca051ab3b211f1795e`.
- Decision: `be1b7a45a94e20d07a0ccc00984bff5745aad82b3f8b5a8aa2c6ab59b7851960`.
- parent-1: `81d835344258bec267ed90ee995de95f0a26fc2a828973af17128a0525d84547`.
- control-1: `374380ae54d14ba1b505170e3f99f0c0e8c0cc7828457c0f0c394e8b0c272e34`.
- motor-1: `e20dfd60986a3766b9604809d56845e11cc62f79d25b86a5eaad6edb1a30c998`.
- motor-2: `e59af669a66a5e3f042ed4695268be8c1275aa3e02a5dcf660b8e13803228139`.
- control-2: `419f7197996439391eb1aa7ee22f55b79a9a29fc0c0c37b7a3ed09e396c0b86e`.
- parent-2: `806a40685e91a877211d16d5f7174d40bae0ac54fc8da7a15be6f9f13e1b1925`.

The initial Mac audit's extra **bit-exact derived-analysis** assertion failed.
Read-only diagnosis found270 differing float leaves, maximum absolute
difference5.551115123125783e-17, only in derived squared means/load shares.
Per-case counts are45/39/38/49/48/51 in execution order. Every raw-summary
audit passes its original1e-9 precision and the complete decision reproduces
exactly. Linux reproduces the per-case derived analyses exactly as well.

The CPU-only `reconcile_derived_analysis` helper explicitly allows the existing
1e-9 reconstruction precision only for those named derived means/shares.
Raw reports/hashes, quantiles, peak indices, counts, source/case identity,
scope claims and decisions remain exact. It does not alter simulator outputs,
acceptance thresholds or the closed historical replay failure. Ten focused
tests cover these boundaries; combined contract/runner tests47 passed in22.06s.
Final full local regression after the reconciliation helper:819 passed in37.13s,
no skips, the same two existing warnings; `git diff --check` passed.
