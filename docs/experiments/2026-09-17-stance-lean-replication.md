# B1-N: fresh-seed training replication of the residual-lean lesson

Protocol `football-b1n-lean-replication-v1`. This is a **predeclaration**.
Nothing in this document has run: no optimizer, no CUDA job, no service, no
video. It fixes one bounded replication and the gates that judge it, so that the
seeds, the thresholds and the decision rule cannot be renegotiated after the
numbers are known.

It follows the [lean-lesson evaluation](2026-09-16-stance-lean-lesson.md), whose
declared decision was `lean-lesson-passed-nominal`. That decision is not
revisited, weakened or appealed here. Its decision rule named this gate
explicitly:

> That admits only a **nominal stance candidate**. Fresh-seed training
> replication and small-disturbance recovery still precede B1 acceptance.

This document is that replication. It is not the disturbance experiment, and it
does not substitute for it.

## Evidence this replication answers

The lean lesson trained one 256-update weight-initialized continuation at
learner seed **571** and evaluated it at four common checkpoints across
evaluation seeds 541/547/557, 128 environments each. It passed everything:

| Checkpoint | Worst per-attempt tilt p95 | Gate | Passes per seed |
| --- | --- | --- | --- |
| 64 | 2.25299 deg | 5.00000 deg | 128/128 |
| 128 | 2.40004 deg | 5.00000 deg | 128/128 |
| 192 | 1.55985 deg | 5.00000 deg | 128/128 |
| 255 | 1.02774 deg | 5.00000 deg | 128/128 |

**1,536 of 1,536 attempts passed, zero hard failures**, with the 0.0873 rad tilt
gate untouched. Recorded at source `602ac07aecd3`, launch
`e9eedf85653c3e1b25c90e239c2ef0e559d1bdaf462dcac04d2526ee83a15f50`.

That is **one training run**. A single run cannot distinguish a property of the
recipe from a property of one learner seed. Everything downstream — B1
acceptance, and any disturbance-driven retraining — would rest on that one run.
The question this replication answers is therefore narrow and specific:

> Does the lean-lesson recipe produce a gate-passing stance at learner seeds
> other than 571?

## What the learner seed actually perturbs

This matters for what the replication can and cannot conclude, so it is stated
before the result rather than inferred from it.

`LeanStanceLearner.install_parent` overwrites the actor and critic **in place**
after `_initialize`, so the fresh initializer built from the seed is discarded
before training begins. The starting weights are the pinned `model_127.pt`
export at SHA256 `46cd52b5…` in every replication, exactly as in the lesson.

The seed's remaining effects are therefore:

- **rollout sampling** — which worlds and which stochastic action draws the
  collection step sees;
- **the learner's own RNG stream**, which drives PPO's sampling noise.

It does **not** change the starting weights, the reward, the reset, the plant,
the correction authority, or the episode length. A replication is a test of
whether the recipe's *optimization path* is robust to sampling noise, not of
whether it is robust to initialization. That is a real and useful question — it
is the question the lesson's own evidence leaves open — but it is a narrower
claim than "the recipe is seed-robust", and the record must not widen it.

## The one changed axis

Changed: **the learner seed.**

Unchanged, deliberately and explicitly:

- the 44D actor / 50D critic layout, fixed scalings and ELU (128,128,64) MLPs;
- the pinned parent export, its SHA256, and the weight-initialization identity;
- the B1-N reward sum and every coefficient, including the tilt term;
- the reset pose, the +/-0.2 rad correction bound, the 1 rad/s target slew,
  the central-90% soft range intersection and the three-physics-step delay;
- the BAM XL330 m6 motor model, action slew, and every failure stop;
- **64 worlds, 256 updates, 24 policy ticks per update**;
- the common checkpoints **64, 128, 192, 255** and the evaluation seeds
  **541, 547, 557** at 128 environments;
- the evaluation scorer, every threshold, the 122/128 per-seed requirement, and
  the **0.0873 rad tilt gate**.

No reward coefficient is touched. No threshold is touched. No gate is relaxed
in any branch of the decision rule below.

## Declared replication seeds

Three new learner seeds, as bounded literals:

**577, 587, 593**

Verified unused: none of the three appears anywhere in `src/`, `tests/` or
`docs/` before this document. Seeds already consumed are 521, 523, 563 and 571
(fresh initialization) and 541, 547, 557 (evaluation).

The `fresh_models` allowlist currently admits `(521, 523, 563, 571)` and is a
bounded literal tuple by design. It is extended by exactly these three literals.
**It must not be widened to a range, a predicate or a caller-supplied value.**

Three seeds is the declared count. If one of the three fails to launch for an
infrastructure reason, the replication is reported as **incomplete** and no
verdict is recorded — a missing seed is never dropped from the denominator to
make the remaining two look sufficient.

## Why this is a code change, not a configuration change

Recorded up front because it is the reason this experiment is larger than its
one-axis description suggests.

`checkpoint.LEAN_SEED = 571` is a module constant, not a parameter. It feeds:

- `fresh_models`' bounded allowlist;
- `SCOPE[LEAN_PURPOSE] = (LEAN_SEED, LEAN_WORLDS, LEAN_UPDATES)`;
- `stance_lean_lesson.plan`'s `seed=` and `identity`'s `training_seed=`;
- `stance_lean_lesson.output_path` and `service_name`, which are keyed by the
  **source hash alone**.

The last point is not cosmetic. All three replications run from the same commit,
so a source-keyed directory would make them collide on one path and overwrite
each other. The replication path and service name must therefore carry the seed.

Replication also gets its **own purpose, trace protocol and evaluation
protocol**, on the established principle that a purpose must never borrow
another's iteration labels or loader. The lesson's evaluation must keep
rejecting replication exports, and the replication's evaluation must keep
rejecting lesson exports, in both directions.

| Concern | Lesson | Replication |
| --- | --- | --- |
| purpose | `lean-lesson` | `lean-replication` |
| trace protocol | `football-b1n-lean-first-attempt-trace-v1` | `football-b1n-lean-replication-first-attempt-trace-v1` |
| evaluation protocol | `football-b1n-lean-lesson-evaluation-v1` | `football-b1n-lean-replication-evaluation-v1` |
| learner seed | 571 (frozen) | 577, 587, 593 (declared) |
| output path | `stance-lean-lesson-<source12>` | `stance-lean-replication-<source12>-seed-<seed>` |

The evaluable-iteration set is identical — the same four common checkpoints —
but it is reached through a distinct map entry, so the equality is enforced
rather than assumed.

## Fixed training experiment

Per declared seed, one run:

- **64 worlds, 256 updates, 24 policy ticks per update.** Identical to the
  lesson. The budget is not re-tuned.
- CPU actor/critic/Adam, unchanged PPO mathematics. CUDA0 eager full-collision
  physics. Forward graphs remain disabled.
- Common checkpoints at iterations **64, 128, 192, 255**, plus the retained
  initializer export. All four are evaluated; none substitutes for another and
  **no best-checkpoint search is permitted**.
- Retain every completed update, all tick records, optimizer receipts,
  source/runtime/launch hashes and a final supervisor file-hash inventory.
  Exclusive fsynced writes; no overwrite; the parent archive is read-only input
  and must be byte-identical after every run.
- Purpose `lean-replication`, so the lesson loader continues to reject these
  exports and vice versa.

## Caps: reused, and re-derived on every launch

The replication's training and evaluation configurations are **byte-identical**
to the lesson's except for the seed. Worlds, ticks per update, update budget,
episode length, checkpoints and evaluation seeds are all unchanged, and the
throughput probe measured collection with the *frozen parent policy* — which is
also the starting point of every replication.

The lesson's measured caps are therefore the correct basis, and they are not
re-estimated here:

| Bound | Value | Basis |
| --- | --- | --- |
| training child watchdog | 1,693 s | measured throughput probe, `4f0e61b0004a` |
| training service cap | 1,753 s | `ceil(1.25 * 1,402 s)` |
| evaluation child watchdog | 1,352 s | measured evaluation probe, `cdb05a5667ca` |
| evaluation service cap | 1,412 s | `ceil(1.25 * 1,129.2552504418418 s)` |

Both remain a **transcription**, and both paths re-derive them from their
retained probe reports on every launch, refusing if the constants disagree. A
hand-edited constant still cannot size a watchdog. **If a replication run
exceeds its measured cap it is killed and reported as killed** — the stop is not
weakened to fit, and the run is not retried or extended.

This reuse is a deliberate, narrow exception to the "measure it, do not
estimate it" rule, and it is justified by the configurations being identical
rather than analogous. It is recorded here so that it is a stated decision
rather than an oversight. **If any axis changes — worlds, ticks, budget,
checkpoints or episode length — the caps must be re-measured before launch.**

Honest cost, stated before the fact rather than after: three runs at roughly
1,693 s of training plus roughly 1,135 s of evaluation each is about **2.4 hours
of exclusive GPU**, run sequentially. A GPU job needs an idle GPU, a
temperature check, and the service timeout, watchdog and closeout reserve.

## Evaluation and decision rule

Evaluation reuses the unchanged every-boundary stance scorer and the
HC4-E1-style fixed first-attempt denominator. **No new tolerances.**

Per replication seed, per common checkpoint: evaluation seeds 541/547/557, 128
environments each, one 5 s first attempt per environment from the nominal reset.
No auto-reset averaging, no smoothing, no post-reset samples.

Per-attempt pass requires **all** of: no failure; final-1 s tilt p95
`<= 0.0873 rad`; planar speed p95 `<= 0.03 m/s`; planar displacement
`<= 0.02 m` throughout; final-1 s height minimum `>= 0.105 m`; supporting feet
on `>= 99%` of physics samples after 0.2 s; soft-limit exposure `<= 1%` of
joint-time samples.

**Declared decision rule, fixed before launch:**

- A replication seed **passes** if at least one common checkpoint passes every
  gate on **all three** evaluation seeds, with at least **122/128** attempts
  passing per seed.
- Record **`lean-replication-passed`** if **all three** declared seeds pass.
- Record **`lean-replication-seed-dependent`** if some but not all pass. This is
  the outcome that matters most: it means the recipe's success depends on the
  learner seed, and the lesson's pass cannot be treated as a property of the
  recipe.
- Record **`lean-replication-rejected`** if none pass.
- Record **`lean-replication-incomplete`** if any declared seed did not produce
  a complete, verifiable run. No other verdict is then recorded, and the missing
  seed is not silently dropped.

**The 0.0873 rad gate is not to be relaxed, and the tilt term is not to be
re-weighted, in any branch. A near-miss is a rejection.**

## What a verdict would and would not mean

- `lean-replication-passed` would mean the recipe produced a gate-passing stance
  at all three declared learner seeds. It would still admit only a **nominal
  stance candidate**. It would **not** establish disturbance robustness, motor
  safety, or physical readiness, and it would not authorize B1 acceptance on its
  own — small-disturbance recovery still precedes that.
- `lean-replication-seed-dependent` would mean the lesson's pass is not a
  property of the recipe. The next experiment would then be a
  **separately predeclared** investigation of what the seed changes, not a
  widened budget and not a re-weighting.
- `lean-replication-rejected` would disfavour the lesson's result as
  seed-specific and would reopen the budget-versus-objective question the lesson
  was built to settle.

In every branch, `checkpoint_admitted`, `learned_stance_accepted`,
`football_balance_accepted` and `physical_motion_authorized` stay false.

## Implementation prerequisites

These are real gaps found by reading the current source. None is optional.

1. **Seed parameterization.** Thread the learner seed through
   `stance_lean_lesson` so a replication declares its own seed without touching
   the lesson's frozen 571, and so the lesson path is bit-for-bit unchanged.
2. **Seed-bearing output paths.** `output_path` and `service_name` must include
   the seed, or three replications from one commit collide.
3. **A distinct purpose** `lean-replication`, with `SCOPE`, `EVALUABLE` and
   `FRESH_PURPOSES`/`PURPOSES` entries, and a loader the lesson path cannot use.
4. **A distinct trace protocol** with its own `ITERATIONS` entry, so the two
   purposes cannot borrow each other's iteration labels.
5. **A bounded allowlist extension** for 577, 587, 593 — three literals.
6. **A replication evaluation path** that takes its training source, report hash
   and purpose from a declared replication identity rather than the lesson's
   hard-coded constants, reusing the scorer and the 0.0873 rad gate **by
   reference**.
7. **Tests** proving the two directions of loader rejection, the bounded seed
   literals, the all-seeds decision rule, the fail-closed caps, and that no
   existing frozen bound or gate moved. Each new guard is mutation-tested:
   reverting it must make its test fail.
8. **Exact-source Linux CPU regression** remains the launch gate.

## What this does not authorize

No physical motion. No claim of learned stance, football balance, disturbance
recovery, or motor safety. No MP4 is recorded. No gait, hop or obstacle policy
or its evidence is modified or re-adjudicated. The 0.90 m and 1.15 m obstacle
specialists and the rejected direct-joint O1 line are untouched. The lean-lesson
result at seed 571 is neither revisited nor weakened.

This document is simulation-only, and it is a declaration, not a result.

## Implementation status

Recorded separately from the declaration above, and it changes no seed,
threshold, gate or decision rule. Nothing has been launched.

**Landed.**

- **Seed parameterization** (prerequisite 1). `LeanStanceLearner` takes a
  `seed` keyword, and `identity`, `plan`, `prepare`, `inputs_check`,
  `check_service`, `run_updates`, `child`, `verify_completed`, `supervise` and
  the child command line all take an explicit `declaration` and `seed`, both
  defaulting to the lesson's. The module now declares two continuations,
  `LESSON` and `REPLICATION`, that share one code path.
- **Seed-bearing paths** (prerequisite 2). `output_path` and `service_name`
  carry the seed when the declaration says so, giving three distinct
  directories and three distinct units for three runs from one commit.
- **Distinct purpose and allowlist** (prerequisites 3, 5). `lean-replication`
  joins `PURPOSES`, `SCOPE` and `EVALUABLE`; `fresh_models` admits exactly
  `(521, 523, 563, 571, 577, 587, 593)`, composed from the per-purpose literals
  rather than widened to a range.
- **Distinct trace protocol** (prerequisite 4).
  `football-b1n-lean-replication-first-attempt-trace-v1` has its own `ITERATIONS`
  entry and its own loader in `bundle.LOADERS`. The two lean protocols admit the
  same four iterations, which is a property of the declared configurations rather
  than a shared name.
- **A latent bug fixed.** `install_parent` compared the installed weights against
  the module-level `SEED`'s initializer rather than the learner's own. That is
  invisible at 571 and wrong for any other seed, so it would have made every
  replication check the wrong thing. It now reads `self.seed`.

**Verified.** The lesson's evidence path, service name, plan and child command
line are asserted unchanged, so the frozen 571 result and its retained directory
are untouched. 136 tests pass across the directly affected modules, and the full
stance regression gives **712 passed**. Two new guards were mutation-tested:
reintroducing the `--mode`/`--job` mismatch reproduces
`unrecognized arguments: --mode` and fails its test, and routing the replication
through the lesson's loader fails two tests with `DID NOT RAISE`.

The stance regression also shows 1 failure and 2 errors in
`tests/test_stance_eager_learning.py`, which is **pre-existing and not caused by
this change**. That file is untouched here, and the same three tests fail with
the same signature on a stashed pristine tree: its `--help` subprocess is given a
hardcoded 20 s timeout against an 11.2 s standalone cost, and its 120 s
collection deadline expires under suite load. Recorded rather than waved off,
because a failing test that is assumed rather than demonstrated to be unrelated
is how a real regression gets missed.

**Not yet landed — this is the remaining blocker.** The evaluation path
(prerequisite 6) is still bound to the lesson's `TRAINING_SOURCE`,
`TRAINING_REPORT`, purpose, trace protocol and decision strings. Until it takes a
declared replication identity, a replication can be *trained* but not *judged*,
so **no replication may be launched**: a run whose only available scorer is the
lesson's would either fail closed or, worse, score replication exports against
the lesson's pinned archive. Prerequisite 7's tests for the all-seeds decision
rule and prerequisite 8's exact-source Linux regression remain open with it.

