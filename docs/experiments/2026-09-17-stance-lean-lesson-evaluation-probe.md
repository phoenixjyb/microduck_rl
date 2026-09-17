# B1-N: lean-lesson evaluation — timing probe

Protocol `football-b1n-lean-lesson-evaluation-probe-v1`. This is a
**predeclaration**. Nothing in this document has run: no CUDA job, no service, no
probe case. It fixes what will be measured and, more importantly, the rule that
turns the measurement into the evaluation's watchdog caps — so that rule cannot
be chosen after the number is known.

It follows the [completed lean-lesson run](2026-09-16-stance-lean-lesson.md),
whose decision was `lean-lesson-complete-not-capability`. That decision is not
revisited, weakened or appealed here. It also follows the
[throughput probe](2026-09-16-stance-lean-lesson-throughput-probe.md), whose
measured caps sized the training run.

## Why the evaluation needs its own probe

The evaluation declared in the lesson is **12 cases**: four common checkpoints
(64/128/192/255) times three evaluation seeds (541/547/557), 128 environments
each, one 5 s first attempt per environment.

The parent frozen comparison is the only realized cost of a comparable job, and
it is not a usable estimate for this one:

| Parent comparison, realized | Value |
| --- | --- |
| cases | 6 (3 fresh-initializer, 3 iteration-127) |
| child wall time | **354.2161789140664 s** |
| mean per case | 59.04 s |
| fresh-initializer cases | 59 policy ticks, `all-first-attempts-complete` |
| iteration-127 cases | **250 policy ticks**, `all-first-attempts-complete` |

Two things follow, and both argue against reusing the 900 s bound by analogy.

**The lean evaluation has no short cases.** A fresh initializer dies at 59 ticks
in 1.16 s, which is why the parent's mean per case is cheap. All four lean
checkpoints are trained continuations, so all 12 cases run full length. The
parent's 59.04 s mean is therefore *contaminated*: it averages three 59-tick
cases with three 250-tick cases and is not a per-full-case figure at all.
Multiplying it by 12 to get ≈ 708 s would be exactly the kind of straight-line
estimate that the lesson's predeclaration already refuses to let size a
watchdog.

**And the existing bounds cannot be widened.** `supervised_process` caps a child
at `CELL_SECONDS = 120`; `supervised_stance_smoke` is frozen at 900 s. A 12-case
run needs a new, separately declared wrapper. That wrapper's caps must come from
a measurement, which is what this probe produces.

The probe itself needs **no new bound**: one case plus its prelude fits far
inside the existing frozen 900 s pair. So the probe runs under the existing
wrapper, and the new wrapper is declared only after the probe reports.

## What the probe measures

Exactly one case, run end-to-end through the same code path the evaluation will
use — `worker.evaluate_owned_case`, which includes strict hash-checked restore,
collection, bundle write **and** the independent replay verification. Timing a
shorter path would repeat the mistake the throughput probe already made (it
excluded `collect_one`'s bookkeeping and under-measured the real run by 7.5 %).

- checkpoint iteration **255** — the last common checkpoint, so the case is most
  likely to reach full length;
- evaluation seed **541** — the first declared seed, so the probe does not
  privilege a seed chosen after the fact;
- 128 worlds, capture device `cuda:0`, actor on CPU.

Three wall-clock quantities are measured **separately**, because they do not
repeat at the same rate:

| Quantity | Meaning | Repeats |
| --- | --- | --- |
| `prelude_seconds` | interpreter start through imports, CUDA init, `host.wait_idle()` and `inputs_check`, up to the first environment construction | once |
| `env_seconds` | one `WarpStanceRuntime(128, device='cuda:0')` construction | 12 times |
| `case_seconds` | one full `evaluate_owned_case` call | 12 times |

Plus the case's `policy_ticks` and `stop_reason`.

## What the probe deliberately does not measure

The probe records **timing and tick counts only**. It runs the real every-boundary
scorer, because that scorer is on the measured path, and it retains the resulting
bundle — but the probe's report records no gate verdict, no tilt p95, no pass
count, and no decision string.

The retained single-case bundle is declared **not** evidence for the decision
rule in the lesson predeclaration. One case out of twelve cannot support
`lean-lesson-passed-nominal` or `lean-lesson-rejected-objective-binds`, and
neither may be reported from the probe.

Its output directory is `stance-lean-eval-probe-<source12>`, structurally
distinct from the evaluation's `stance-lean-evaluation-<source12>`, so a probe
bundle can never be mistaken for one of the twelve evaluation cases.

No checkpoint is admitted, no policy is exported, no motion is authorized.

## The probe's own caps: the existing frozen pair

| Bound | Value | Source |
| --- | --- | --- |
| child watchdog | 900 s | `files.supervised_stance_smoke`, unchanged |
| whole-service timeout | 960 s | same wrapper, unchanged |
| service `RuntimeMaxUSec` | `16min` | read back with `-p KEY --value` |
| closeout reserve | 600 s | unchanged |

`16min` is **evidence, not a guess**: it is the string
`stance_eager_evaluation.check_service` asserts, and that assertion passed on a
real launch at `artifacts/evaluations/stance-eager-evaluation-bd099638f52b`. The
training run's `29min 13s` was read off the host for the same reason. No
`RuntimeMaxUSec` string in this family is ever assumed.

The 120 s `CELL_SECONDS` bound and the 900 s smoke bound are untouched. Nothing
is widened to make the probe fit.

## Declared validity condition

The probed case must reach **`policy_ticks == 250` with
`stop_reason == 'all-first-attempts-complete'`**.

A short case cannot size a full-length run. If the case ends early, the probe
records `probe-invalid-short-case` and **derives no cap**; the twelve-case
evaluation is then not launched on that basis, and a full-length cost must be
established some other way before it is.

## The cap rule, declared before any measurement

```
unit      = env_seconds + case_seconds
predicted = prelude_seconds + 12 * unit
service   = ceil(1.25 * predicted)
child     = service - 60
closeout  = 600
```

- **1.25** is the same declared safety factor the throughput probe used. It is
  applied to the measured sum, never to a mean.
- `child = service - 60` keeps the 60 s watchdog margin; the 600 s closeout is
  unchanged from the training run.
- **Window condition.** The derived service must satisfy
  `service + 600 + 60 <= 3600`, i.e. **`service <= 2940`**. If it exceeds that,
  the twelve-case evaluation is **not** launched inside one window, and no case
  is dropped silently to make it fit. A separately predeclared split is required.
  This is a hard stop, not a warning.

**A single sample is not a maximum.** The throughput probe took the *max* of 8
measured updates. This probe measures **one** case, so the 1.25 factor carries
strictly more weight here than it did there, and the window condition above is
what absorbs that. This is stated before the measurement rather than discovered
after it.

## Declared expectation, recorded to be checked

A priori, and sizing nothing: a full-length lean case is expected to land near
the full-length iteration-127 case cost inside the parent's realized 354.216 s.
This is written down **only** so it can be compared against the measurement.

The previous probe in this family declared an expectation that did not survive
contact with measurement, and it was recorded rather than quietly dropped. The
same will be done here, in whichever direction it falls.

## Launch preconditions

Unchanged from the lesson: focused tests, an exact-source Linux CPU regression,
a clean feature branch, pinned dependencies and assets, the compiled plant, and
an independently retained launch SHA. Before GPU allocation, check an idle GPU,
no competing compute owner, temperature below 80 C, and both protected AI
Mission system services inactive. Do not touch 100.98, do not stop unrelated
workloads, and do not restore protected services.

## First attempt: failed, produced no measurement

Launched on 100.100 at source `6723cd620333`, service
`microduck-lean-eval-probe-6723cd620333.service`, `RuntimeMaxSec=960`. The
service reported `Result=success` while the run had in fact failed, and the
journal-as-truth read is what caught it. The report records
`decision: failed`, `error: child exited unsuccessfully: 2`.

| What | Value |
| --- | --- |
| child wall time | 7.055826982948929 s |
| root cause | `error: unrecognized arguments: --mode probe` |
| measurement produced | **none** |

The defect was ours, in the runner: `supervise` built the child command line
with `--mode <job>` while `main` declared that argument as `--job`. Neither side
is wrong on its own, which is exactly why no unit test of either side caught it.

Everything upstream of the child was correct and is now independently confirmed:
`check_service` passed against the live unit (`RuntimeMaxUSec` read back as
`16min`, `KillMode=control-group`, `MainPID` matching), `inputs_check`
authenticated the four checkpoints, and `idle_before` shows a 46 C idle GPU with
no compute owner and both protected AI Mission services inactive.

The fix moves the command line into `child_command` and the parser into
`parser()`, and adds a test that feeds the supervisor's own child argv through
that same parser. Reintroducing `--mode` reproduces
`unrecognized arguments: --mode probe` and fails the test, so the guard is known
to bite rather than assumed to.

The failed attempt's `report.json` and `child.log` were removed before the retry,
because a fresh attempt must start from exactly `launch.json` and
`runtime.json`. No measurement was discarded: there was none.

## Second attempt: measured

Launched on 100.100 at source `cdb05a5667ca`, service
`microduck-lean-eval-probe-cdb05a5667ca.service`, `RuntimeMaxSec=960`. Child exit
0 after **109.59330233978108 s**. Decision `probe-measured`.

| Quantity | Measured |
| --- | --- |
| `prelude_seconds` (once) | 0.0008308729156851768 s |
| `env_seconds` (per case) | 0.719240453094244 s |
| `case_seconds` (per case) | **93.38529451098293 s** |
| repeating unit `env + case` | 94.10453496407717 s |
| prediction `prelude + 12 × unit` | 1,129.2552504418418 s |
| **service cap** `ceil(1.25 × prediction)` | **1,412 s** |
| **child watchdog** `service − 60` | **1,352 s** |
| service `RuntimeMaxUSec` rendering | `23min 32s` |
| window total `service + closeout + margin` | 2,072 s of the 3,600 s maximum |

The probed case reached `policy_ticks == 250` with
`stop_reason == all-first-attempts-complete`, so the declared validity condition
held and a full-length cost was in fact measured.

The GPU was idle before (46 C, 12 MiB, no compute owner), peaked at 57 C during
the case, and was back to 49 C after. Both protected AI Mission services read
`inactive` in every sample. The run admitted no checkpoint, exported no policy
and authorized no motion.

Provenance: `artifacts/evaluations/stance-lean-eval-probe-cdb05a5667ca`, launch
SHA256 `d002eba450e465c03dc86507dc8bcb7b9e87e7296c7d20ff8c90e11ae8cd26d0`,
report SHA256 `a91b5a068be96def6ae7fa18d7109ea75cb10b64f870a7d3991515589b80db58`,
measurements SHA256 `aae4de36454cc3bb8a800169de08a118d175532dfde9ef7a4ae78aae34bd6ddd`.

### The declared expectation did not survive measurement, again

The expectation above was that a full-length lean case would land "near the
full-length iteration-127 case cost inside the parent's realized 354.216 s". The
only grounded comparable number is that parent's 354.216 / 6 = 59.04 s mean per
case, and the measured unit is **94.10 s — 59 % above it**. The expectation is
recorded as failed rather than reinterpreted.

That failure is the probe's justification, and it is worth stating precisely. A
straight-line analogy from the parent — *6 cases took 354.2 s, so 12 will take
about 708 s, comfortably inside 900 s* — would have looked safe. The measured
prediction is 1,129.3 s, which exceeds the frozen 900 s child by 229 s and the
960 s service by 169 s. **Reusing the existing bound by analogy would have
guaranteed a timeout**, and the probe is what prevented that launch. The parent's
mean was cheap only because half its cases were fresh-initializer cases that died
at 59 ticks.

### The retained probe score is not evidence for the decision

The probed case's bundle retains a score, because the scorer is on the measured
path and the bundle is what that path produces. Its 128 of 128 attempts passed.
That number is stated here only so it is not discovered later as a hidden
result, and **no conclusion may be drawn from it**: one case out of twelve, at
one checkpoint and one seed, is not the declared evaluation. The predeclaration
is explicit that the probe records no gate verdict, and neither
`lean-lesson-passed-nominal` nor `lean-lesson-rejected-objective-binds` is
earned by it.

## Implementation status

| Piece | State |
| --- | --- |
| `stance_attempt_trace.LEAN_PROTOCOL` + protocol→iteration map | landed |
| `stance_evaluation_bundle.LOADERS` loader routing | landed |
| `stance_lean_evaluation` module (probe and evaluate modes) | landed, tested |
| probe case executed | **measured**, 2026-09-17 |
| twelve-case caps | **1,352 s child / 1,412 s service**, transcribed from the probe |
| `supervised_lean_evaluation` child bound | landed |
| twelve-case evaluation | **not run** |

The transcription is checked, not trusted. `measured_caps` re-derives the caps
from the retained probe report on every launch and refuses if the constants
disagree, so a hand-edited number, or a probe re-run that produced different
ones, cannot size a watchdog. A test additionally pins the transcription against
the wrapper the supervisor actually applies, so the bound in the code and the
bound in the constants cannot drift.

The service's `RuntimeMaxUSec` string is computed by `systemd_runtime_max` from
the seconds instead of being typed in. It reproduces both renderings read off the
host (960 s → `16min`, 1,753 s → `29min 13s`), which is the mistake the
throughput probe's first launch made and this one does not.

Two changes were needed in existing modules and both are additive: the trace
gained a third protocol entry and the bundle gained a third loader entry, with no
existing entry altered. The bundle's loader routing also changed from a two-way
conditional whose `else` sent any unrecognised protocol to the pilot loader, to an
explicit map that refuses an unlisted protocol. That is a real hole closed, not a
tidy-up: under the old form a lean-protocol binding could restore a
pilot-purpose export at iteration 128, which is a valid iteration under both
protocols. A test pins that refusal, and reverting the map makes it fail.

### One defect that only a host run could find

`derive_caps` was written to take a probe report with a top-level `measurements`
key, but `supervise` nests it under `probe`. The unit fixture had been written
with the same flat shape, so it agreed with the bug and 48 tests passed while the
function could not read a real report. It failed the first time it was pointed at
the retained artifact. The rule now takes the measurements block itself, a
separate `probe_measurements_of` owns the report shape, and the fixture writes
the layout `supervise` actually writes. The general lesson is recorded because it
will recur: a fixture that invents a document's shape tests the fixture.

## What this does not authorize

No physical motion. No claim of learned stance, football balance, disturbance
recovery or motor safety. `checkpoint_admitted`, `learned_stance_accepted`,
`football_balance_accepted` and `physical_motion_authorized` remain false
throughout, including after a successful probe. No MP4 is recorded. No gait, hop
or obstacle policy or its evidence is modified or re-adjudicated. The 0.0873 rad
tilt gate is not touched, relaxed or re-weighted.

This document is simulation-only, and it is a declaration, not a result.
