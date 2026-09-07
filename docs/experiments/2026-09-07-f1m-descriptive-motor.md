# F1-M descriptive motor measurements

Status: CPU contract and runner implemented/tested; fresh collection is gated
on the pushed source and remote checks below. No new optimizer updates are authorized
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
