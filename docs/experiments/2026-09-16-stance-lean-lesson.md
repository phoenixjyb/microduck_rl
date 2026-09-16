# B1-N: residual-lean lesson — weight-initialized continuation

Protocol `football-b1n-lean-lesson-v1`. This is a **predeclaration**. Nothing in
this document has run: no optimizer, no CUDA job, no service, no video. It
declares one bounded training lesson and the evaluation that would judge it, so
that the budget cannot be renegotiated after the numbers are known.

It follows the [completed frozen initializer/final comparison](2026-09-09-stance-eager-evaluation.md),
whose decision was `final-numerical-rejected`. That decision is not revisited,
weakened or appealed here.

## Evidence this lesson answers

The frozen evaluation scored 768 first attempts over seeds 541/547/557. The
fresh initializer failed 128/128 attempts per seed at a mean 1.1645 s. The
final iteration-127 export reached the full five-second endpoint on every one
of 384 attempts with **zero hard failures** and every gate clean except one:

| Final-second quantity | Observed (384 attempts) | Unchanged limit | Result |
| --- | --- | --- | --- |
| tilt p95 | 7.95283–8.13340 deg | 0.0873 rad (~5 deg) | **fail** |
| planar displacement | max 0.01224868 m | <= 0.02 m | pass |
| planar speed p95 | max 0.00161868 m/s | <= 0.03 m/s | pass |
| minimum height | >= 0.11518921 m | >= 0.105 m | pass |
| two-foot support | 100% | >= 99% | pass |
| soft-limit exposure | 0% | <= 1% | pass |

So the policy holds a **stable, almost drift-free, but persistently tilted**
stance. Tilt is the only failing gate in `comparison.json`, and it fails on
every attempt rather than marginally on some. That is a qualitatively different
failure from instability: the policy is not falling over, it is settling.

Two competing explanations are live, and this lesson is designed to separate
them:

1. **Budget.** 128 updates is a very short run. The tilt may still be
   descending and would cross the gate with more optimization from the same
   point.
2. **Objective or mechanical feasibility.** The reward already pays
   `2 exp(-(tilt/0.10)^2)` at every executed physics boundary, which is worth
   about 0.9333 at the 0.0873 rad gate and about 0.2787 at the observed band
   midpoint (8.043 deg = 0.140379 rad). The policy is declining roughly 0.655
   per boundary over up to 2,500 boundaries per episode to hold that lean. A
   policy that surrenders that much shaping while remaining stable may be
   finding the only equilibrium its available authority admits — for example
   if the nominal HOME pose plus +/-0.2 rad of leg correction does not contain
   a statically balanced stance, leaving a small forward lean as the reachable
   solution.

Explanation 1 yields to more training. Explanation 2 does not, and would move
the next experiment to reset pose or correction authority rather than to the
reward. Distinguishing them is the entire purpose of this lesson.

## The one changed axis

Changed: **optimizer budget and starting weights.**

Unchanged, deliberately and explicitly:

- the 44D actor / 50D critic layout, fixed scalings and ELU (128,128,64) MLPs;
- the B1-N reward sum and every coefficient, including the tilt term;
- the reset pose, the +/-0.2 rad correction bound, the 1 rad/s target slew,
  the central-90% soft range intersection and the three-physics-step delay;
- the BAM XL330 m6 motor model, action slew, and every failure stop;
- the evaluation scorer, its thresholds, and the 0.0873 rad tilt gate;
- all existing gait, hop, roller and obstacle policies and their evidence.

No reward coefficient is touched. This lesson exists precisely to earn or
refuse the right to change one, on evidence rather than on a hunch.

## Initialization identity: weight-initialized, not a resume

This is the honesty-critical section, and it is why the lesson is named
separately from any continuation.

`model_127.pt` is a **weight export**, not an optimizer or simulator
checkpoint. Inspection of the retained archive confirms this: the file
contains only `identity` and `states`, and `states` contains only `actor` and
`critic` tensors. There is no Adam moment state, no rollout storage, no
simulator state, no RNG state.

Therefore the new run must be declared, and must record itself, as follows:

- **Weights:** actor and critic are initialized from the frozen `model_127.pt`
  export, SHA256 `46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93`,
  loaded with the strict loader that refuses missing, extra, legacy or
  normalizer tensors.
- **Optimizer:** Adam state starts **fresh**. This is the single largest
  difference from a resume and must be stated in the run identity.
- **Learner RNG:** fresh, new declared seed. Not inherited.
- **Rollout storage:** empty. No replay of any prior transition.
- **Simulator state:** fresh, from the declared nominal reset. No restored
  physics, contact or delay state.
- **Identity:** the run identity must name the parent export path and hash
  explicitly, and must record the loaded logical model-state hash as its
  starting `initial_state_sha256`. It must not claim to continue the parent's
  optimization trajectory.

A consequence that must not be papered over: because Adam restarts, **new-run
iteration N is not comparable to old-run iteration N**. The two runs share a
starting point in parameter space, not an optimization history. Any comparison
is between evaluated final states under the unchanged scorer, never between
iteration labels.

The existing `CpuStanceLearner` deliberately refuses to let a restored learner
drive the real simulator: `collect_one` requires
`not self.restored_fixture_only or env.synthetic_ppo_fixture is True`. That
guard is correct and must not be relaxed. A weight-initialized real run needs
its own declared path that loads weights while leaving
`restored_fixture_only` false, and that path needs its own tests (see
prerequisites).

## Fixed training experiment

- Training seed **571** (new, not previously used). Fresh initialization seed
  **571** must be added to the `fresh_models` allowlist, which currently
  admits only 521, 523 and 563; the allowlist must not be widened to an
  unbounded range.
- **64 worlds**, **256 updates**, **24 policy ticks per update** — the same
  worlds and steps as the parent, with the budget doubled as the single axis.
- CPU actor/critic/Adam, unchanged PPO mathematics. CUDA0 eager
  full-collision physics. Forward graphs remain disabled.
- Common checkpoints at iterations **64, 128, 192, 255**, plus the retained
  initializer export. All four are evaluated; none is a substitute for
  another and no best-checkpoint search is permitted.
- Retain every completed update, all tick records, optimizer receipts,
  source/runtime/launch hashes and a final supervisor file-hash inventory.
  Exclusive fsynced writes; no overwrite; the parent archive is read-only
  input and must be byte-identical after the run.
- Distinct checkpoint purpose, e.g. `lean-lesson`, so the existing
  `eager-learning` and evaluation loaders continue to reject these exports.
  Do not alias iteration labels across purposes.

## Evaluation, promotion and launch gates

Evaluation reuses the unchanged every-boundary stance scorer and the
HC4-E1-style fixed first-attempt denominator. No new tolerances.

- All four common checkpoints, evaluation seeds **541, 547, 557**, 128
  environments each, one 5 s first attempt per environment from the nominal
  reset. No auto-reset averaging, no smoothing, no post-reset samples.
- Per-attempt pass requires **all** of: no failure; final-1 s tilt p95
  `<= 0.0873 rad`; planar speed p95 `<= 0.03 m/s`; planar displacement
  `<= 0.02 m` throughout; final-1 s height minimum `>= 0.105 m`; supporting
  feet on `>= 99%` of physics samples after 0.2 s; soft-limit exposure
  `<= 1%` of joint-time samples.
- A checkpoint advances only if it passes on **all three** evaluation seeds.
  At least 122/128 attempts per seed must pass, matching the original per-seed
  numerical threshold.
- Report tilt p95 per seed for every checkpoint, not only the survivors. The
  trend across 64/128/192/255 is the primary evidence for budget-versus-objective,
  so a single final number is not sufficient.

**Declared decision rule, fixed before launch:**

- If any common checkpoint passes every gate on all three seeds, record
  `lean-lesson-passed-nominal`. That admits only a **nominal stance candidate**.
  Fresh-seed training replication and small-disturbance recovery still precede
  B1 acceptance, and no fixed-ball or rolling-football training follows
  directly.
- If no checkpoint passes, record `lean-lesson-rejected-objective-binds` and
  treat explanation 1 as disfavoured. The next experiment is then a separately
  predeclared single-axis change to **reset pose or correction authority**,
  tested against the same tilt gate.
- **The 0.0873 rad gate is not to be relaxed, and the tilt term is not to be
  re-weighted, in either branch.** A near-miss is a rejection.

## Timing, ownership and stop boundary

The parent run completed 128 updates in 735.963 s of child time (5.7497 s per
update) at 64 worlds. Straight-line scaling to 256 updates gives about
**1,471.9 s (24.53 min)**. That estimate is unreliable in the pessimistic
direction and must not be used to size the watchdog: the parent's average
includes early updates whose episodes terminated at roughly 1.16 s, whereas a
weight-initialized policy will run near-full 250-tick episodes from update 0.
The real cost is likely higher.

Therefore: **a separately declared measured throughput probe must run first**
and its measurement, not this estimate, must set the service cap. That probe is
now itself predeclared as the
[lean-lesson throughput probe](2026-09-16-stance-lean-lesson-throughput-probe.md)
(protocol `football-b1n-lean-lesson-throughput-v1`), which measures collection on
CUDA0 under the frozen parent policy and the CPU optimizer term on the host, and
derives the caps from the measured worst case with a declared 1.25 safety factor.
If the probe does not leave a safe margin inside the authorization window, the
budget is reduced or the run is not launched — the budget is not to be exceeded
by weakening the stop.

The existing 900 s child watchdog and 960 s service cap are **too short for
this budget** and must be re-declared for the measured duration, keeping the
control-group kill, the independent watchdog, and a 600 s closeout reserve. A
new explicit absolute launch window is required; authority must not be derived
from any expired cutoff.

**Measured, not estimated.** The probe ran on 100.100 at source `4f0e61b0004a`
on 2026-09-16. Its measurement is the only authority for the two numbers below.

| Quantity | Measured | How |
| --- | --- | --- |
| collection, worst of 8 updates | 5.395445651840419 s | CUDA0, 64 worlds, 24 ticks, frozen parent policy |
| optimizer, worst of 8 updates | 0.07417336199432611 s | host CPU, 64x24 samples, CUDA hidden |
| per-update worst case | 5.469619013834745 s | sum of the two maxima |
| setup (one reset) | 0.03163699014112353 s | measured, not assumed |
| predicted child | 1,402 s | `ceil(256 * 5.4696…) + ceil(0.0316…)` |
| **service cap** | **1,753 s** | `ceil(1.25 * 1,402)` |
| **child watchdog** | **1,693 s** | `service - 60` |
| parent estimate, superseded | 1,472 s | recorded only for comparison; never used |

The declared 1.25 factor is applied to the **sum of the two maxima**, never to a
mean; the 60 s watchdog margin and the 600 s closeout are unchanged. Provenance:
`artifacts/evaluations/stance-lean-throughput-4f0e61b0004a`, launch SHA256
`71a9bb69b7f419156b81577d6a6b758764cb6c13f71c5ae902f85cdf4316f150`, parent
archive byte-identical after the probe at `46cd52b5…`. The probe admitted no
checkpoint, exported none, and authorized no motion.

One declared expectation above did **not** survive contact with measurement, and
it is recorded rather than quietly dropped. The claim was that the 1,471.9 s
estimate was "unreliable in the pessimistic direction" and that "the real cost is
likely higher". It was not: the measured worst per-update cost (5.4696 s) is
*below* the parent's 5.7497 s average. The measured cap is nevertheless larger
than the estimate (1,753 s vs 1,472 s) because the safety factor is applied to a
measured worst case rather than to an average. That conservatism is the intended
effect, and it is not grounds to reduce the factor.

Before GPU allocation: focused tests, exact-source Linux CPU regression, a
clean feature branch, pinned dependencies and assets, the compiled plant, and
an independently retained launch SHA must all pass. Check an idle GPU, no
competing compute owner, temperature below 80 C, and both protected AI Mission
system services inactive. Do not touch 100.98, do not stop unrelated
workloads, and do not restore protected services.

On timeout or failure, preserve the durable completed prefix and diagnose
read-only. Do not resume the partial run, do not retry, and do not extend.
On success, verify every export, update receipt and tick inventory, mirror
hashes, and report only the declared decision string. Never report a capability.

## Implementation prerequisites

These are real gaps found by reading the current source. None is optional.

1. **A weight-initialization path distinct from restore.** Load actor and
   critic tensors into the learner while leaving `restored_fixture_only` false
   and without touching the existing fixture-only restore guard.
2. **Seed allowlist extension** for the fresh initialization seed 571, bounded
   to that literal value.
3. **A checkpoint purpose** distinct from `eager-learning`, with the evaluation
   loader still refusing cross-purpose loads.
4. **A measured throughput probe** to size the watchdog and service cap.
5. **A preflight test** proving that the parent archive is unchanged after a
   load, that the recorded starting state hash equals the parent's
   `initial_state_sha256`-equivalent, and that a corrupted or substituted
   parent export is refused before deserialization.
6. **Focused tests** for the new path, plus the unchanged existing stance
   regression set. Linux exact-source confirmation remains the launch gate.

## Prerequisite status

This section records implementation and validation only. It is not an
experiment result, and it changes no seed, threshold, gate or decision rule
above.

Implemented at commit `3d7e286b`:

- the weight-initialization path (`stance_lean_lesson.LeanStanceLearner`) with
  the seed-571 allowlist, the `lean-lesson` checkpoint purpose, and the three
  loaders;
- the parent byte hash is checked *before* deserialization, so a corrupted or
  substituted export is refused rather than loaded, and the archive is
  byte-identical after a load;
- evaluation admission is purpose-exact, so the retained pilot evaluation path
  still refuses `lean-lesson` exports and the lean path refuses pilot ones.

Validation:

- Local focused CPU regression: 192 passed, 2 errors across the eleven
  checkpoint/evaluation files. Both errors are the pre-existing
  `eager collection deadline` wall-clock budget in `test_stance_eager_learning.py`
  and reproduce unchanged on stashed, unmodified source — they are not caused by
  this work. Ten new tests in `tests/test_stance_lean_lesson.py` pass.
- **Exact-source Linux CPU regression: 610 passed in 92.44 s** at `3d7e286b`
  on 100.100 with `CUDA_VISIBLE_DEVICES` empty and CUDA never initialized.

Still outstanding before any launch, and deliberately not fabricated here:

1. **The measured throughput probe — done.** Predeclared and implemented as the
   [lean-lesson throughput probe](2026-09-16-stance-lean-lesson-throughput-probe.md),
   then run on 100.100 at source `4f0e61b0004a`. It produced
   `child_seconds = 1693` and `service_seconds = 1753` from the measured worst
   case, as tabulated above. It admitted nothing and exported no checkpoint. The
   caps are now declared numbers; they are deliberately not yet written into any
   supervisor source, because that source does not exist yet (item 2).
2. **The CUDA supervisor and child for the lean-lesson run itself — implemented.**
   `stance_lean_lesson` now carries `prepare` / `supervise` / `child` alongside the
   initialization path, using **1,693 / 1,753** and not the superseded 1,472 s
   estimate. The child bound is enforced by a new `supervised_lean_lesson`
   wrapper (`LEAN_LESSON_CHILD_SECONDS = 1693`), added as its own function so
   that no existing frozen bound moves: `supervised_process` still caps at
   `CELL_SECONDS` (120 s) and `supervised_stance_smoke` still holds 900 s.
   The window floor is 1,753 + 600 + 60 = **2,413 s**, so this run needs a fresh
   **41-to-60-minute** window, and `RuntimeMaxUSec` must read back as
   `29min 13s`.
3. A clean feature branch, pinned dependencies and assets, the compiled plant,
   and an independently retained launch SHA.

One measured-scope caveat is recorded here rather than left to be discovered at
launch. The probe's collection timer covers the observation build, the policy
forward pass and the physics step, but not `collect_one`'s own bookkeeping or the
per-tick evidence write. The parent's realized 5.7497 s/update puts that gap at
roughly 0.35 s per update, so a 256-update run is expected near 1,490 s against
the 1,663 s internal child deadline (`CHILD_SECONDS - 30`) and the 1,693 s
watchdog — about 11% headroom. The declared 1.25 factor is what covers the
difference. This is not grounds to change any declared number: if the run
overruns, the predeclared failure path applies, the durable completed prefix is
preserved, and the run is diagnosed read-only rather than resumed or extended.

## What this does not authorize

No physical motion. No claim of learned stance, football balance, disturbance
recovery, or motor safety. `checkpoint_admitted`, `learned_stance_accepted`,
`football_balance_accepted` and `physical_motion_authorized` remain false
throughout, including on a pass. No MP4 is recorded. No gait, hop or obstacle
policy or its evidence is modified or re-adjudicated. The 0.90 m and 1.15 m
obstacle specialists and the rejected direct-joint O1 line are untouched.

This document is simulation-only, and it is a declaration, not a result.
