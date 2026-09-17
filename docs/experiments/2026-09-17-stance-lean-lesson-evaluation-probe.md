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

## Implementation status

Written down before the probe runs, so that "the runner exists" and "the runner
has measured something" cannot be confused.

| Piece | State |
| --- | --- |
| `stance_attempt_trace.LEAN_PROTOCOL` + protocol→iteration map | landed |
| `stance_evaluation_bundle.LOADERS` loader routing | landed |
| `stance_lean_evaluation` module (probe and evaluate modes) | landed, tested |
| probe case executed | **not run** |
| twelve-case caps | **not measured** — `None`, and the evaluate mode refuses to plan |

The evaluate mode fails closed on purpose. `CHILD_SECONDS`/`SERVICE_SECONDS` are
`None` until the probe has run, and `service_caps('evaluate')` refuses with
`twelve-case caps are not measured yet` rather than borrowing the 900 s pair.
When they are filled in they are a *transcription*, and `measured_caps`
re-derives them from the retained probe report on every launch — so a hand-edited
constant, or a probe re-run that produced different numbers, is refused rather
than launched.

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

## What this does not authorize

No physical motion. No claim of learned stance, football balance, disturbance
recovery or motor safety. `checkpoint_admitted`, `learned_stance_accepted`,
`football_balance_accepted` and `physical_motion_authorized` remain false
throughout, including after a successful probe. No MP4 is recorded. No gait, hop
or obstacle policy or its evidence is modified or re-adjudicated. The 0.0873 rad
tilt gate is not touched, relaxed or re-weighted.

This document is simulation-only, and it is a declaration, not a result.
